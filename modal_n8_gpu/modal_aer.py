"""Modal L40S execution of frozen N=8 adaptive edge racing and independent audit."""
from __future__ import annotations

import json
from pathlib import Path
import time

import modal

from modal_n8_gpu.modal_app import image, cache

app = modal.App("static-bfs-n8-adaptive-edge-racing-v1")
result = modal.Volume.from_name("static-bfs-n8-adaptive-edge-racing-v1-results",
                                create_if_missing=True)
ROOT = Path("/result/adaptive-edge-racing-v1")
REMOTE = Path("/opt/repro")
EXPECTED = {
    "stage_a/candidate_set.json": "ba62b5e93e6bbbd21d83eec2bbdf9ac893c0ba6d3d5166d80d9350d786e3a8a3",
    "racing/discriminating_edges.json": "2d68e18114ec095da86f1fad4d54793c05f38dd0b1499f07c2fc383a208934a3",
    "racing/racing_trajectory_plan.json": "b73cd5ff904a4188d93cd358f99352cbd0385b111ea65342e0136f3e201b5715",
    "audit/audit_plan.json": "bedb37215974aa38c947a94daf7a6e52fb0fc3dcddbda774e399ef397042faa0",
    "experiment_manifest.json": "a05b8f8a00572428b754d46d20e0e12b0fa8dd53f7e6876d37bfaa61b2d8938b",
}


def _context(seed: int):
    from modal_n8_gpu.gpu_runtime import Runtime
    from scripts.static_bfs_graph.n8_plan import n8_config
    from scripts.static_bfs_graph.runner import read
    pool = read(REMOTE / "modal_n8_gpu/train_pool.json")
    old = read(REMOTE / "configs/diffusion_classical_search_t2i_table12.json")
    prompts = [{"prompt": pool["prompts"].get(str(i))} for i in range(100)]
    return Runtime(ROOT, Path("/opt/official"), seed, n8_config(), old, prompts)


def _check_plan():
    from modal_n8_gpu.aer_plan import digest
    for name, expected in EXPECTED.items():
        if digest(ROOT / name) != expected:
            raise RuntimeError(f"frozen AER file hash changed: {name}")


@app.function(image=image, volumes={"/result": result}, timeout=300, cpu=1, memory=2048)
def remote_preflight(files: dict[str, str]):
    from hashlib import sha256
    from modal_n8_gpu.aer_plan import digest
    if set(files) != set(EXPECTED):
        raise RuntimeError("frozen file set changed")
    for name, raw in files.items():
        if sha256(raw.encode()).hexdigest() != EXPECTED[name]:
            raise RuntimeError(f"local input hash changed: {name}")
        path = ROOT / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.read_text() != raw:
            raise RuntimeError(f"remote frozen file changed: {name}")
        path.write_text(raw)
    _check_plan()
    result.commit()
    return {"passed": True, "candidate_set_sha256": digest(ROOT / "stage_a/candidate_set.json"),
            "racing_plan_sha256": digest(ROOT / "racing/racing_trajectory_plan.json")}


@app.local_entrypoint()
def preflight():
    root = Path(__file__).resolve().parents[1] / "results/static-bfs-graph/adaptive-edge-racing-v1"
    files = {name: (root / name).read_text() for name in EXPECTED}
    print("AER_PREFLIGHT", json.dumps(remote_preflight.remote(files), sort_keys=True))


@app.function(image=image, gpu="L40S", volumes={"/vol": cache, "/result": result},
              timeout=3600, cpu=8, memory=24576, max_containers=4)
def remote_racing_trajectory(rank: int):
    import torch
    from modal_n8_gpu.aer_plan import read
    from modal_n8_gpu.prefix_replay import reference_with_snapshots
    from scripts.static_bfs_graph.graph_builder import (
        all_edges, dense_reference, edge_intervention)
    from scripts.static_bfs_graph.offline_profiler.profiler import profile_trajectory
    from scripts.static_bfs_graph.runner import static_record, write_atomic

    result.reload()
    cache.reload()
    _check_plan()
    if rank not in range(20):
        raise ValueError("rank out of frozen racing range")
    if rank > 0:
        gate = read(ROOT / "racing/cost_gate.json")
        if not gate["allowed"]:
            raise RuntimeError("racing cost gate not passed")
    item = read(ROOT / "racing/racing_trajectory_plan.json")["trajectories"][rank]
    prompt_id, seed = item["prompt_id"], item["seed"]
    pool = read(REMOTE / "modal_n8_gpu/train_pool.json")
    from hashlib import sha256
    if sha256(pool["prompts"][str(prompt_id)].encode()).hexdigest() != item["prompt_sha256"]:
        raise RuntimeError("prompt changed")
    folder = ROOT / f"racing/raw/trajectory_{rank:02d}"
    timing_path = folder / "timing.json"
    keys = set(read(ROOT / "racing/discriminating_edges.json")["discriminating_edges"])
    if timing_path.exists():
        timing = read(timing_path)
        if (timing["prompt_id"] != prompt_id or timing["seed"] != seed or
            timing["edge_count"] != len(keys)):
            raise RuntimeError("completed racing trajectory inconsistent")
        return timing
    start_unix = time.time()
    started = time.perf_counter()
    runtime = _context(seed)
    reference = dense_reference((10, 20, 30, 40, 60, 80, 90), 8.0, 8)
    snap = folder / "reference_snapshots.pt"
    ref_file = folder / "reference.json"
    if snap.exists() != ref_file.exists():
        raise RuntimeError("partial reference/snapshot requires explicit repair")
    ref = reference_with_snapshots(runtime, prompt_index=prompt_id, trial_seed=seed,
                                   snapshot_path=snap, record_path=ref_file,
                                   protocol="STATIC_BFS_N8_ADAPTIVE_EDGE_RACING_V1")
    checkpoint = torch.load(snap, map_location="cpu", weights_only=False)
    result.commit()
    edge_started = time.perf_counter()
    edges_dir = folder / "edges"
    if ({path.stem for path in edges_dir.glob("*.json")} != keys or
        not (folder / "edge_profile_timing.json").exists()):
        profile = profile_trajectory(runtime, checkpoint, ref, edges_dir, keys)
    else:
        profile = read(folder / "edge_profile_timing.json")
    if profile["logical_edges"] != len(keys):
        raise RuntimeError("targeted edge profile incomplete")
    edge_seconds = time.perf_counter() - edge_started
    write_atomic(folder / "edge_profile_timing.json", profile)
    result.commit()
    gate_seconds = 0.0
    if rank == 0:
        gate_started = time.perf_counter()
        naive_ref = runtime.run("aer_reference_gate", static_record(reference),
                                prompt_id, reference)
        if (naive_ref["resampling_events"] != ref["resampling_events"] or
            naive_ref["final_particle_scores"] != ref["final_particle_scores"] or
            naive_ref["selected_image_rgb_sha256"] != ref["selected_image_rgb_sha256"]):
            raise RuntimeError("new-seed dense reference replay failed")
        target_key = "30_to_80_tau_8"
        if target_key not in keys:
            raise RuntimeError("frozen correctness edge absent")
        edge = next(e for e in all_edges((10, 20, 30, 40, 60, 80, 90),
                                         (2.0, 8.0, 32.0)) if e.key == target_key)
        naive = runtime.run("aer_edge_gate", static_record(edge_intervention(edge, reference)),
                            prompt_id, reference, edge)
        shared = read(edges_dir / f"{target_key}.json")
        if (naive["resampling_events"] != shared["resampling_events"] or
            naive["final_particle_scores"] != shared["final_particle_scores"] or
            naive["selected_image_rgb_sha256"] != shared["selected_image_rgb_sha256"]):
            raise RuntimeError("new-seed targeted shared edge replay failed")
        gate_seconds = time.perf_counter() - gate_started
        write_atomic(ROOT / "racing/correctness_gate.json", {
            "passed": True, "rank": rank, "prompt_id": prompt_id, "seed": seed,
            "reference_exact": True, "edge_exact": True,
            "target_edge": target_key, "gate_wall_seconds": gate_seconds})
        result.commit()
    timing = {"rank": rank, "prompt_id": prompt_id, "seed": seed,
              "GPU_type": torch.cuda.get_device_name(0),
              "pipeline_load_seconds": runtime.load_seconds,
              "reference_seconds": ref["reference_wall_seconds"],
              "edge_profile_seconds": edge_seconds,
              "correctness_gate_seconds": gate_seconds,
              "trajectory_wall_seconds": time.perf_counter() - started,
              "wall_start_unix": start_unix, "wall_end_unix": time.time(),
              "edge_count": len(keys),
              "shared_destination_verifier_evaluations": profile["shared_destination_verifier_evaluations"],
              "snapshot_bytes": snap.stat().st_size}
    write_atomic(timing_path, timing)
    result.commit()
    return timing


@app.local_entrypoint()
def racing_pilot():
    print("AER_RACING_PILOT", json.dumps(remote_racing_trajectory.remote(0), sort_keys=True))


@app.function(image=image, volumes={"/result": result}, timeout=300, cpu=1, memory=2048)
def remote_racing_cost_gate(measured_n: int):
    from modal_n8_gpu.aer_plan import RATE_USD_PER_SECOND, RACING_COST_CAP_USD, read
    from scripts.static_bfs_graph.runner import write_atomic
    result.reload()
    _check_plan()
    if measured_n not in (1, 5, 10, 15):
        raise ValueError("cost gate only at frozen boundary")
    if not read(ROOT / "racing/correctness_gate.json")["passed"]:
        raise RuntimeError("correctness gate failed")
    rows = [read(ROOT / f"racing/raw/trajectory_{i:02d}/timing.json")
            for i in range(measured_n)]
    consumed = sum(x["trajectory_wall_seconds"] for x in rows)
    normal = max(x["trajectory_wall_seconds"] - x["correctness_gate_seconds"]
                 for x in rows)
    projected = consumed + (20-measured_n) * normal * 1.25
    estimate = projected * RATE_USD_PER_SECOND
    out = {"measured_n": measured_n, "consumed_GPU_seconds": consumed,
           "projected_20_GPU_seconds": projected,
           "projected_20_estimated_USD": estimate,
           "cap_estimated_USD": RACING_COST_CAP_USD,
           "allowed": estimate <= RACING_COST_CAP_USD}
    write_atomic(ROOT / "racing/cost_gate.json", out)
    result.commit()
    if not out["allowed"]:
        raise RuntimeError("racing projected cost exceeds frozen cap")
    return out


@app.local_entrypoint()
def racing_cost_gate(measured_n: int):
    print("AER_RACING_COST_GATE", json.dumps(remote_racing_cost_gate.remote(measured_n), sort_keys=True))


@app.function(image=image, volumes={"/result": result}, timeout=7200, cpu=1, memory=2048)
def remote_racing_stage(round_number: int):
    from modal_n8_gpu.aer_plan import read
    result.reload()
    _check_plan()
    if round_number not in (1, 2, 3, 4):
        raise ValueError("round outside frozen range")
    if round_number > 1:
        prior = read(ROOT / f"racing/round_{round_number-1}_state.json")
        if prior["stop"]:
            raise RuntimeError("racing already stopped")
    ranks = range(5 * (round_number-1), 5 * round_number)
    rows = list(remote_racing_trajectory.map(ranks))
    if len(rows) != 5:
        raise RuntimeError("racing batch incomplete")
    return {"round": round_number, "completed": 5,
            "summed_GPU_seconds": sum(x["trajectory_wall_seconds"] for x in rows)}


@app.local_entrypoint()
def racing_stage(round_number: int):
    print("AER_RACING_STAGE", json.dumps(remote_racing_stage.remote(round_number), sort_keys=True))


@app.function(image=image, volumes={"/result": result}, timeout=900, cpu=2, memory=4096)
def remote_analyze_round(round_number: int):
    from modal_n8_gpu.aer_racing import analyze_round
    result.reload()
    _check_plan()
    out = analyze_round(ROOT, round_number)
    result.commit()
    return out


@app.local_entrypoint()
def analyze_round(round_number: int):
    print("AER_ROUND", json.dumps(remote_analyze_round.remote(round_number), sort_keys=True))


@app.function(image=image, volumes={"/result": result}, timeout=300, cpu=1, memory=2048)
def remote_freeze_audit():
    from modal_n8_gpu.aer_plan import digest, read, write_once
    from scripts.static_bfs_graph.n8_run import manual8_record
    from scripts.static_bfs_graph.policy import StaticBFSPolicy
    from scripts.static_bfs_graph.runner import static_record
    result.reload()
    _check_plan()
    compiled = read(ROOT / "racing/aer_compiled_policy.json")
    candidates = read(ROOT / "stage_a/candidate_set.json")
    audit = read(ROOT / "audit/audit_plan.json")
    if compiled["audit_scoring_used"] or not read(ROOT / f"racing/round_{compiled['racing_trajectories_used']//5}_state.json")["stop"]:
        raise RuntimeError("AER winner not frozen")
    old_p20 = StaticBFSPolicy((30, 80, 90), ((30, 8.0), (80, 8.0), (90, 8.0)), 8)
    transfer = StaticBFSPolicy((40, 60, 90), ((40, 8.0), (60, 8.0), (90, 32.0)), 8)
    all_items = {item["policy_id"]: item for item in candidates["candidates"]}
    if (old_p20.id != "static_669da784337a0b16" or
        transfer.id != candidates["incumbent_policy_id"]):
        raise RuntimeError("historical baselines changed")
    aliases = {
        "AER_WINNER": compiled["selected_policy"]["policy_id"],
        "COARSE_TOP1": candidates["coarse_top1_policy_id"],
        "G20_OLD_TOP1": old_p20.id,
        "TRANSFER4_TO_8": transfer.id,
        "MANUAL8": manual8_record()["id"],
    }
    if list(aliases) != audit["policy_aliases"] or len(audit["pairs"]) != 20:
        raise RuntimeError("frozen audit aliases or pairs changed")
    unique = {}
    for alias, pid in aliases.items():
        if alias == "MANUAL8":
            record = manual8_record()
        else:
            item = (all_items.get(pid) or
                    {"steps": [30, 80, 90], "taus": [8.0, 8.0, 8.0], "policy_id": old_p20.id})
            policy = StaticBFSPolicy(tuple(item["steps"]),
                                     tuple(zip(item["steps"], item["taus"])), 8)
            record = static_record(policy)
        if record["id"] != pid:
            raise RuntimeError(f"audit alias policy changed: {alias}")
        unique.setdefault(pid, {"policy_id": pid, "record": record,
                                "first_alias": alias})
    frozen = {"protocol": compiled["protocol"],
              "aer_compiled_policy_sha256": digest(ROOT / "racing/aer_compiled_policy.json"),
              "audit_plan_sha256": digest(ROOT / "audit/audit_plan.json"),
              "aliases": aliases, "unique_policies": list(unique.values()),
              "pairs": audit["pairs"], "freeze_before_audit_scoring": True}
    write_once(ROOT / "audit/frozen_policy_plan.json", frozen)
    result.commit()
    return {"frozen_policy_plan_sha256": digest(ROOT / "audit/frozen_policy_plan.json"),
            "unique_policies": len(unique), "aliases": aliases}


@app.local_entrypoint()
def freeze_audit():
    print("AER_AUDIT_FREEZE", json.dumps(remote_freeze_audit.remote(), sort_keys=True))


@app.function(image=image, gpu="L40S", volumes={"/vol": cache, "/result": result},
              timeout=3600, cpu=8, memory=24576, max_containers=4)
def remote_audit_trajectory(rank: int):
    import torch
    from modal_n8_gpu.aer_plan import digest, read
    from scripts.static_bfs_graph.graph_builder import dense_reference
    from scripts.static_bfs_graph.runner import write_atomic
    result.reload()
    cache.reload()
    _check_plan()
    if rank not in range(20):
        raise ValueError("audit rank outside frozen set")
    if rank > 0:
        gate = read(ROOT / "audit/cost_gate.json")
        if not gate["allowed"]:
            raise RuntimeError("audit cost gate not passed")
    frozen = read(ROOT / "audit/frozen_policy_plan.json")
    if frozen["aer_compiled_policy_sha256"] != digest(ROOT / "racing/aer_compiled_policy.json"):
        raise RuntimeError("AER winner changed after freeze")
    item = frozen["pairs"][rank]
    prompt_id, seed = item["prompt_id"], item["seed"]
    from hashlib import sha256
    pool = read(REMOTE / "modal_n8_gpu/train_pool.json")
    if sha256(pool["prompts"][str(prompt_id)].encode()).hexdigest() != item["prompt_sha256"]:
        raise RuntimeError("audit prompt changed")
    timing_path = ROOT / f"audit/timing/trajectory_{rank:02d}.json"
    if timing_path.exists():
        timing = read(timing_path)
        if (timing["prompt_id"] != prompt_id or timing["seed"] != seed or
            timing["unique_policy_count"] != len(frozen["unique_policies"])):
            raise RuntimeError("completed audit trajectory inconsistent")
        return timing
    start_unix = time.time()
    started = time.perf_counter()
    runtime = _context(seed)
    reference = dense_reference((10, 20, 30, 40, 60, 80, 90), 8.0, 8)
    policy_seconds = 0.0
    for policy in frozen["unique_policies"]:
        row = runtime.run("aer_independent_audit", policy["record"], prompt_id, reference)
        if row["policy_id"] != policy["policy_id"] or row["trial_seed"] != seed:
            raise RuntimeError("audit policy rollout mismatch")
        policy_seconds += row["elapsed_seconds"]
        write_atomic(ROOT / "audit/raw" / policy["policy_id"] /
                     f"p{prompt_id}_s{seed}.json", row)
        result.commit()
    timing = {"rank": rank, "prompt_id": prompt_id, "seed": seed,
              "GPU_type": torch.cuda.get_device_name(0),
              "pipeline_load_seconds": runtime.load_seconds,
              "policy_seconds": policy_seconds,
              "unique_policy_count": len(frozen["unique_policies"]),
              "wall_start_unix": start_unix, "wall_end_unix": time.time(),
              "trajectory_wall_seconds": time.perf_counter() - started}
    write_atomic(timing_path, timing)
    result.commit()
    return timing


@app.local_entrypoint()
def audit_pilot():
    print("AER_AUDIT_PILOT", json.dumps(remote_audit_trajectory.remote(0), sort_keys=True))


@app.function(image=image, volumes={"/result": result}, timeout=300, cpu=1, memory=2048)
def remote_audit_cost_gate():
    from modal_n8_gpu.aer_plan import AUDIT_COST_CAP_USD, RATE_USD_PER_SECOND, read
    from scripts.static_bfs_graph.runner import write_atomic
    result.reload()
    first = read(ROOT / "audit/timing/trajectory_00.json")
    projected = first["trajectory_wall_seconds"] * 20 * 1.25
    estimate = projected * RATE_USD_PER_SECOND
    out = {"pilot_GPU_seconds": first["trajectory_wall_seconds"],
           "projected_20_GPU_seconds": projected,
           "projected_20_estimated_USD": estimate,
           "cap_estimated_USD": AUDIT_COST_CAP_USD,
           "allowed": estimate <= AUDIT_COST_CAP_USD}
    write_atomic(ROOT / "audit/cost_gate.json", out)
    result.commit()
    if not out["allowed"]:
        raise RuntimeError("audit projected cost exceeds frozen cap")
    return out


@app.local_entrypoint()
def audit_cost_gate():
    print("AER_AUDIT_COST_GATE", json.dumps(remote_audit_cost_gate.remote(), sort_keys=True))


@app.function(image=image, volumes={"/result": result}, timeout=7200, cpu=1, memory=2048)
def remote_audit_stage():
    result.reload()
    rows = list(remote_audit_trajectory.map(range(1, 20)))
    if len(rows) != 19:
        raise RuntimeError("audit incomplete")
    return {"completed": 19,
            "summed_GPU_seconds": sum(row["trajectory_wall_seconds"] for row in rows)}


@app.local_entrypoint()
def audit_stage():
    print("AER_AUDIT_STAGE", json.dumps(remote_audit_stage.remote(), sort_keys=True))


@app.function(image=image, volumes={"/result": result}, timeout=120, cpu=1, memory=1024)
def remote_status():
    from modal_n8_gpu.aer_plan import read
    result.reload()
    racing_done = [i for i in range(20) if
                   (ROOT / f"racing/raw/trajectory_{i:02d}/timing.json").exists()]
    audit_done = [i for i in range(20) if
                  (ROOT / f"audit/timing/trajectory_{i:02d}.json").exists()]
    compiled = ROOT / "racing/aer_compiled_policy.json"
    return {"racing_done": racing_done, "audit_done": audit_done,
            "winner": read(compiled)["selected_policy"]["policy_id"] if compiled.exists() else None}


@app.local_entrypoint()
def status():
    print("AER_STATUS", json.dumps(remote_status.remote(), sort_keys=True))
