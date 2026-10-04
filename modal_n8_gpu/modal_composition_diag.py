"""Targeted paired edge/full-policy diagnostic on the 20 unused TRAIN prompts."""
from __future__ import annotations

import json
from pathlib import Path
import time

import modal

from modal_n8_gpu.modal_app import image, cache

app = modal.App("static-bfs-n8-composition-diagnostic-v1")
result = modal.Volume.from_name("static-bfs-n8-composition-diagnostic-v1-results",
                                create_if_missing=True)
ROOT = Path("/result/n8-composition-diagnostic-v1")
REMOTE = Path("/opt/repro")
PLAN_SHA256 = "2e4a60c37a428397f08b034e8061a2cffb8672aad8088aa9c15f2a5e11a631ce"
RATE_USD_PER_SECOND = 0.000542 + 8 * 0.0000131 + 24 * 0.00000222


def _plan():
    from modal_n8_gpu.composition_plan import digest, read
    path = ROOT / "plan.json"
    if digest(path) != PLAN_SHA256:
        raise RuntimeError("frozen plan hash changed")
    return read(path)


def _context(seed: int):
    from modal_n8_gpu.gpu_runtime import Runtime
    from scripts.static_bfs_graph.n8_plan import n8_config
    from scripts.static_bfs_graph.runner import read
    pool = read(REMOTE / "modal_n8_gpu/train_pool.json")
    old = read(REMOTE / "configs/diffusion_classical_search_t2i_table12.json")
    prompts = [{"prompt": pool["prompts"].get(str(i))} for i in range(100)]
    return Runtime(ROOT, Path("/opt/official"), seed, n8_config(), old, prompts)


@app.function(image=image, volumes={"/result": result}, timeout=300, cpu=1, memory=2048)
def remote_preflight():
    from modal_n8_gpu.composition_plan import digest, read, write_once
    source = REMOTE / "modal_n8_gpu/composition_diagnostic_plan.json"
    if digest(source) != PLAN_SHA256:
        raise RuntimeError("local frozen plan changed before upload")
    plan = read(source)
    if (len(plan["trajectories"]) != 20 or len(plan["edge_union"]) != 12 or
        len(plan["policies"]) != 4 or plan["PSP_in_scope"] or
        plan["VALIDATION_touched"] or plan["TEST_touched"]):
        raise RuntimeError("preflight scope invalid")
    write_once(ROOT / "plan.json", plan)
    if digest(ROOT / "plan.json") != PLAN_SHA256:
        raise RuntimeError("remote frozen plan differs")
    result.commit()
    return {"plan_sha256": PLAN_SHA256, "trajectories": 20, "edge_union": 12,
            "policies": 4, "max_estimated_gpu_usd": plan["max_estimated_gpu_usd"]}


@app.local_entrypoint()
def preflight():
    print("COMPOSITION_PREFLIGHT", json.dumps(remote_preflight.remote(), sort_keys=True))


@app.function(image=image, gpu="L40S", volumes={"/vol": cache, "/result": result},
              timeout=3600, cpu=8, memory=24576, max_containers=4)
def remote_trajectory(rank: int):
    import torch
    from modal_n8_gpu.prefix_replay import reference_with_snapshots
    from scripts.static_bfs_graph.graph_builder import (
        all_edges, dense_reference, edge_intervention)
    from scripts.static_bfs_graph.offline_profiler.profiler import profile_trajectory
    from scripts.static_bfs_graph.policy import StaticBFSPolicy
    from scripts.static_bfs_graph.runner import read, static_record, write_atomic

    result.reload()
    cache.reload()
    plan = _plan()
    if rank not in range(20):
        raise ValueError("trajectory rank outside frozen plan")
    if rank > 0:
        gate = read(ROOT / "cost_gate.json")
        if not gate["allowed"] or gate["max_estimated_gpu_usd"] != plan["max_estimated_gpu_usd"]:
            raise RuntimeError("cost gate missing or changed")
    trajectory = plan["trajectories"][rank]
    prompt_id, seed = trajectory["prompt_id"], trajectory["seed"]
    output = ROOT / f"trajectories/rank_{rank:02d}"
    timing_path = output / "timing.json"
    if timing_path.exists():
        old = read(timing_path)
        if (old["prompt_id"] == prompt_id and old["seed"] == seed and
            old["edge_count"] == 12 and old["full_policy_count"] == 4):
            return old
        raise RuntimeError("completed trajectory metadata inconsistent")
    started = time.perf_counter()
    runtime = _context(seed)
    reference = dense_reference((10, 20, 30, 40, 60, 80, 90), 8.0, 8)
    snapshot_path = output / "reference_snapshots.pt"
    reference_path = output / "reference.json"
    if snapshot_path.exists() != reference_path.exists():
        (snapshot_path if snapshot_path.exists() else reference_path).unlink()
    full_reference = reference_with_snapshots(
        runtime, prompt_index=prompt_id, trial_seed=seed,
        snapshot_path=snapshot_path, record_path=reference_path,
        protocol=plan["protocol"])
    checkpoint = torch.load(snapshot_path, map_location="cpu", weights_only=False)
    if (checkpoint["prompt_index"] != prompt_id or checkpoint["trial_seed"] != seed or
        full_reference["selected_final_score"] != max(full_reference["final_particle_scores"])):
        raise RuntimeError("reference pairing failed")
    result.commit()

    edges_path = output / "edges"
    expected_edges = set(plan["edge_union"])
    edge_started = time.perf_counter()
    if ({p.stem for p in edges_path.glob("*.json")} != expected_edges or
        not (output / "edge_profile_timing.json").exists()):
        profile = profile_trajectory(runtime, checkpoint, full_reference,
                                     edges_path, expected_edges)
    else:
        profile = read(output / "edge_profile_timing.json")
    edge_seconds = time.perf_counter() - edge_started
    if profile["logical_edges"] != 12 or {p.stem for p in edges_path.glob("*.json")} != expected_edges:
        raise RuntimeError("targeted edge profile incomplete")
    write_atomic(output / "edge_profile_timing.json", profile)
    result.commit()

    gate_seconds = 0.0
    if rank == 0:
        gate_started = time.perf_counter()
        naive_ref = runtime.run("composition_reference_gate", static_record(reference),
                                prompt_id, reference)
        if (naive_ref["resampling_events"] != full_reference["resampling_events"] or
            naive_ref["final_particle_scores"] != full_reference["final_particle_scores"] or
            naive_ref["selected_image_rgb_sha256"] != full_reference["selected_image_rgb_sha256"]):
            raise RuntimeError("full-run dense reference disagrees with snapshot reference")
        edge = next(e for e in all_edges((10, 20, 30, 40, 60, 80, 90),
                                         (2.0, 8.0, 32.0)) if e.key == "30_to_80_tau_8")
        naive_edge = runtime.run("composition_edge_gate",
                                 static_record(edge_intervention(edge, reference)),
                                 prompt_id, reference, edge)
        shared_edge = read(edges_path / f"{edge.key}.json")
        if (naive_edge["resampling_events"] != shared_edge["resampling_events"] or
            naive_edge["final_particle_scores"] != shared_edge["final_particle_scores"] or
            naive_edge["selected_image_rgb_sha256"] != shared_edge["selected_image_rgb_sha256"]):
            raise RuntimeError("new-prompt targeted shared edge disagrees with full rollout")
        gate_seconds = time.perf_counter() - gate_started
        write_atomic(ROOT / "correctness_gate.json", {
            "passed": True, "rank": 0, "prompt_id": prompt_id, "seed": seed,
            "reference_exact": True, "target_edge": edge.key, "edge_exact": True,
            "gate_wall_seconds": gate_seconds})
        result.commit()

    full_seconds = 0.0
    comparisons = []
    for item in plan["policies"]:
        policy = StaticBFSPolicy(tuple(item["steps"]),
                                 tuple(zip(item["steps"], item["taus"])), 8)
        record = static_record(policy)
        if record["id"] != item["policy_id"]:
            raise RuntimeError("frozen policy ID changed")
        row = runtime.run("composition_full_policy", record, prompt_id, reference)
        full_seconds += row["elapsed_seconds"]
        edge_rows = [read(edges_path / f"{key}.json") for key in item["edge_keys"]]
        predicted = sum(edge_row["delta"] for edge_row in edge_rows)
        actual = full_reference["selected_final_score"] - row["selected_final_score"]
        comparison = {"rank": rank, "prompt_id": prompt_id, "seed": seed,
                      "policy": item["name"], "policy_id": item["policy_id"],
                      "reference_reward": full_reference["selected_final_score"],
                      "full_policy_reward": row["selected_final_score"],
                      "predicted_regret": predicted, "actual_regret": actual,
                      "composition_residual": actual - predicted,
                      "edge_keys": item["edge_keys"],
                      "full_policy_image_sha256": row["selected_image_rgb_sha256"],
                      "full_policy_seconds": row["elapsed_seconds"]}
        write_atomic(output / "full_policies" / f"{item['name']}.json", comparison)
        comparisons.append(comparison)
        result.commit()
    timing = {"rank": rank, "prompt_id": prompt_id, "seed": seed,
              "GPU_type": torch.cuda.get_device_name(0),
              "reference_seconds": full_reference["reference_wall_seconds"],
              "edge_profile_seconds": edge_seconds, "full_policy_seconds": full_seconds,
              "correctness_gate_seconds": gate_seconds,
              "pipeline_load_seconds": runtime.load_seconds,
              "trajectory_wall_seconds": time.perf_counter() - started,
              "edge_count": len(expected_edges), "full_policy_count": len(comparisons),
              "snapshot_bytes": snapshot_path.stat().st_size,
              "shared_destination_verifier_evaluations":
                  profile["shared_destination_verifier_evaluations"],
              "tau_suffix_branches": profile["tau_suffix_branches"]}
    write_atomic(timing_path, timing)
    result.commit()
    return timing


@app.local_entrypoint()
def pilot():
    print("COMPOSITION_PILOT", json.dumps(remote_trajectory.remote(0), sort_keys=True))


@app.function(image=image, volumes={"/result": result}, timeout=300, cpu=1, memory=2048)
def remote_cost_gate(measured_n: int):
    from scripts.static_bfs_graph.runner import read, write_atomic
    result.reload()
    plan = _plan()
    if measured_n not in (1, 5):
        raise ValueError("cost gate must be after rank 0 or ranks 0..4")
    if not read(ROOT / "correctness_gate.json")["passed"]:
        raise RuntimeError("correctness gate failed")
    timing = [read(ROOT / f"trajectories/rank_{rank:02d}/timing.json")
              for rank in range(measured_n)]
    consumed = sum(row["trajectory_wall_seconds"] for row in timing)
    normal = [row["trajectory_wall_seconds"] - row["correctness_gate_seconds"]
              for row in timing]
    projected_seconds = consumed + (20 - measured_n) * max(normal) * 1.25
    projected_usd = projected_seconds * RATE_USD_PER_SECOND
    allowed = projected_usd <= plan["max_estimated_gpu_usd"]
    output = {"measured_n": measured_n, "consumed_gpu_seconds": consumed,
              "projected_total_gpu_seconds": projected_seconds,
              "projected_total_estimated_usd": projected_usd,
              "max_estimated_gpu_usd": plan["max_estimated_gpu_usd"],
              "rate_usd_per_second": RATE_USD_PER_SECOND, "allowed": allowed,
              "correctness_gate_passed": True}
    write_atomic(ROOT / "cost_gate.json", output)
    result.commit()
    if not allowed:
        raise RuntimeError("projected cost exceeds frozen cap")
    return output


@app.local_entrypoint()
def cost_gate(measured_n: int):
    print("COMPOSITION_COST_GATE", json.dumps(remote_cost_gate.remote(measured_n), sort_keys=True))


@app.function(image=image, volumes={"/result": result}, timeout=7200, cpu=1, memory=2048)
def remote_stage(first_rank: int, stop_rank: int):
    result.reload()
    if (first_rank, stop_rank) not in ((1, 5), (5, 20)):
        raise ValueError("only frozen stage boundaries supported")
    if _plan()["policy_selection_from_new_rewards"] is not False:
        raise RuntimeError("policy-selection scope changed")
    timings = list(remote_trajectory.map(range(first_rank, stop_rank)))
    if (len(timings) != stop_rank - first_rank or
        any(row["edge_count"] != 12 or row["full_policy_count"] != 4
            for row in timings)):
        raise RuntimeError("stage incomplete")
    return {"first_rank": first_rank, "stop_rank": stop_rank,
            "completed": len(timings),
            "summed_gpu_seconds": sum(row["trajectory_wall_seconds"] for row in timings)}


@app.local_entrypoint()
def stage(first_rank: int, stop_rank: int):
    print("COMPOSITION_STAGE", json.dumps(remote_stage.remote(first_rank, stop_rank), sort_keys=True))


@app.function(image=image, volumes={"/result": result}, timeout=120, cpu=1, memory=1024)
def remote_status():
    from scripts.static_bfs_graph.runner import read
    result.reload()
    plan = _plan()
    completed = []
    for rank in range(20):
        folder = ROOT / f"trajectories/rank_{rank:02d}"
        if (folder / "timing.json").exists():
            completed.append(rank)
    return {"plan_sha256": PLAN_SHA256, "completed_ranks": completed,
            "completed": len(completed), "target": len(plan["trajectories"]),
            "correctness_gate": read(ROOT / "correctness_gate.json") if
            (ROOT / "correctness_gate.json").exists() else None,
            "cost_gate": read(ROOT / "cost_gate.json") if
            (ROOT / "cost_gate.json").exists() else None}


@app.local_entrypoint()
def status():
    print("COMPOSITION_STATUS", json.dumps(remote_status.remote(), sort_keys=True))
