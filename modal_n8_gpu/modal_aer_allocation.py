"""Modal L40S collection for frozen AER budget-allocation comparison."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import time

import modal

from modal_n8_gpu.modal_app import cache, image

app = modal.App("static-bfs-n8-aer-budget-allocation-v1")
result = modal.Volume.from_name("static-bfs-n8-aer-budget-allocation-v1-results", create_if_missing=True)
ROOT = Path("/result/aer-budget-allocation-v1")
REMOTE = Path("/opt/repro")
FROZEN = (
    "experiment_manifest.json", "proposal/g5_candidates.json", "proposal/g10_candidates.json",
    "proposal/candidate_overlap.json", "edges/d5.json", "edges/d10.json",
    "edges/d_union.json", "racing/common_racing_pool.json", "audit/audit_pool.json",
    "analysis/difficulty_design.json",
)


def _check():
    from modal_n8_gpu.aer_plan import digest, read
    manifest = read(ROOT / "experiment_manifest.json")
    names = {"proposal/g5_candidates.json": "g5_candidates_sha256",
             "proposal/g10_candidates.json": "g10_candidates_sha256",
             "edges/d_union.json": "d_union_sha256",
             "racing/common_racing_pool.json": "racing_pool_sha256",
             "audit/audit_pool.json": "audit_pool_sha256"}
    for path, key in names.items():
        if digest(ROOT / path) != manifest[key]:
            raise RuntimeError(f"frozen plan changed: {path}")
    for arm in (5, 10):
        d = read(ROOT / f"edges/d{arm}.json")
        if d["candidate_file_sha256"] != digest(ROOT / f"proposal/g{arm}_candidates.json"):
            raise RuntimeError("arm candidate/edge hash changed")
    u = read(ROOT / "edges/d_union.json")
    if u["d5_sha256"] != digest(ROOT / "edges/d5.json") or u["d10_sha256"] != digest(ROOT / "edges/d10.json"):
        raise RuntimeError("union edge hash changed")
    design = ROOT / "analysis/difficulty_design.json"
    if design.exists():
        value = read(design)
        if value["protocol"] != manifest["protocol"] or value["epsilon"] != 1e-6:
            raise RuntimeError("frozen difficulty diagnostic changed")


def _context(seed: int):
    from modal_n8_gpu.gpu_runtime import Runtime
    from scripts.static_bfs_graph.n8_plan import n8_config
    from scripts.static_bfs_graph.runner import read
    pool = read(REMOTE / "modal_n8_gpu/train_pool.json")
    old = read(REMOTE / "configs/diffusion_classical_search_t2i_table12.json")
    prompts = [{"prompt": pool["prompts"].get(str(i))} for i in range(100)]
    return Runtime(ROOT, Path("/opt/official"), seed, n8_config(), old, prompts)


@app.function(image=image, volumes={"/result": result}, timeout=300, cpu=1, memory=2048)
def remote_preflight(files: dict[str, str]):
    if set(files) != set(FROZEN):
        raise RuntimeError("frozen input file list changed")
    for name, raw in files.items():
        path = ROOT / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.read_text() != raw:
            raise RuntimeError(f"remote frozen file changed: {name}")
        path.write_text(raw)
    _check()
    result.commit()
    return {"passed": True, "frozen_files": len(files)}


@app.local_entrypoint()
def preflight():
    root = Path(__file__).resolve().parents[1] / "results/static-bfs-graph/aer-budget-allocation-v1"
    print(json.dumps(remote_preflight.remote({name: (root/name).read_text() for name in FROZEN})))


@app.function(image=image, gpu="L40S", volumes={"/vol": cache, "/result": result},
              timeout=3600, cpu=8, memory=24576, max_containers=4)
def remote_racing_trajectory(rank: int):
    import torch
    from modal_n8_gpu.aer_plan import read
    from modal_n8_gpu.aer_allocation_plan import PROTOCOL
    from modal_n8_gpu.prefix_replay import reference_with_snapshots
    from scripts.static_bfs_graph.graph_builder import all_edges, dense_reference, edge_intervention
    from scripts.static_bfs_graph.offline_profiler.profiler import profile_trajectory
    from scripts.static_bfs_graph.runner import static_record, write_atomic
    result.reload()
    cache.reload()
    _check()
    if rank not in range(30):
        raise ValueError("rank outside frozen pool")
    if rank >= 5 and not read(ROOT / "racing/cost_projection.json")["allowed"]:
        raise RuntimeError("cost gate has not passed")
    plan = read(ROOT / "racing/common_racing_pool.json")["trajectories"][rank]
    prompt_id, seed = plan["prompt_id"], plan["seed"]
    pool = read(REMOTE / "modal_n8_gpu/train_pool.json")
    if sha256(pool["prompts"][str(prompt_id)].encode()).hexdigest() != plan["prompt_sha256"]:
        raise RuntimeError("prompt hash changed")
    folder = ROOT / f"racing/raw/trajectory_{rank:02d}"
    timing_path = folder / "timing.json"
    keys = set(read(ROOT / "edges/d_union.json")["discriminating_edges"])
    if timing_path.exists():
        timing = read(timing_path)
        if (timing["prompt_id"] != prompt_id or timing["seed"] != seed or
            timing["edge_count"] != len(keys)):
            raise RuntimeError("completed racing trajectory inconsistent")
        return timing
    started = time.perf_counter()
    start_unix = time.time()
    runtime = _context(seed)
    reference = dense_reference((10, 20, 30, 40, 60, 80, 90), 8.0, 8)
    snap = folder / "reference_snapshots.pt"
    ref_file = folder / "reference.json"
    if snap.exists() != ref_file.exists():
        raise RuntimeError("partial reference/snapshot pair")
    ref = reference_with_snapshots(runtime, prompt_index=prompt_id, trial_seed=seed,
                                   snapshot_path=snap, record_path=ref_file,
                                   protocol=PROTOCOL)
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
        raise RuntimeError("targeted union edge profile incomplete")
    edge_seconds = time.perf_counter() - edge_started
    write_atomic(folder / "edge_profile_timing.json", profile)
    result.commit()
    correctness_seconds = 0.0
    if rank == 0:
        gate_start = time.perf_counter()
        naive_ref = runtime.run("allocation_reference_gate", static_record(reference),
                                prompt_id, reference)
        if (naive_ref["resampling_events"] != ref["resampling_events"] or
            naive_ref["final_particle_scores"] != ref["final_particle_scores"] or
            naive_ref["selected_image_rgb_sha256"] != ref["selected_image_rgb_sha256"]):
            raise RuntimeError("new-seed reference replay failed")
        key = "30_to_80_tau_8" if "30_to_80_tau_8" in keys else sorted(keys)[0]
        edge = next(x for x in all_edges((10, 20, 30, 40, 60, 80, 90), (2.0, 8.0, 32.0)) if x.key == key)
        naive = runtime.run("allocation_edge_gate", static_record(edge_intervention(edge, reference)),
                            prompt_id, reference, edge)
        shared = read(edges_dir / f"{key}.json")
        if (naive["resampling_events"] != shared["resampling_events"] or
            naive["final_particle_scores"] != shared["final_particle_scores"] or
            naive["selected_image_rgb_sha256"] != shared["selected_image_rgb_sha256"]):
            raise RuntimeError("new-seed targeted edge replay failed")
        correctness_seconds = time.perf_counter()-gate_start
        write_atomic(ROOT / "racing/correctness_gate.json", {
            "passed": True, "rank": rank, "reference_exact": True,
            "edge_exact": True, "target_edge": key, "GPU_seconds": correctness_seconds})
        result.commit()
    timing = {"rank": rank, "prompt_id": prompt_id, "seed": seed,
              "GPU_type": torch.cuda.get_device_name(0),
              "pipeline_load_seconds": runtime.load_seconds,
              "reference_seconds": ref["reference_wall_seconds"],
              "edge_profile_seconds": edge_seconds,
              "correctness_gate_seconds": correctness_seconds,
              "trajectory_wall_seconds": time.perf_counter()-started,
              "wall_start_unix": start_unix, "wall_end_unix": time.time(),
              "edge_count": len(keys), "snapshot_bytes": snap.stat().st_size,
              "shared_destination_verifier_evaluations": profile["shared_destination_verifier_evaluations"]}
    write_atomic(timing_path, timing)
    result.commit()
    return timing


@app.function(image=image, volumes={"/result": result}, timeout=7200, cpu=1, memory=2048)
def remote_racing_batch(first: int, last: int):
    if (first, last) not in [(i, i+5) for i in range(0, 30, 5)]:
        raise ValueError("invalid batch")
    rows = list(remote_racing_trajectory.map(range(first, last)))
    return {"first": first, "last": last, "completed": len(rows),
            "summed_GPU_seconds": sum(x["trajectory_wall_seconds"] for x in rows)}


@app.local_entrypoint()
def racing_batch(first: int, last: int):
    print(json.dumps(remote_racing_batch.remote(first, last), sort_keys=True))


@app.function(image=image, volumes={"/result": result}, timeout=300, cpu=1, memory=2048)
def remote_racing_cost_gate(measured_n: int, audit_projection_usd: float):
    from modal_n8_gpu.aer_allocation_plan import RATE_USD_PER_SECOND
    from modal_n8_gpu.aer_plan import read
    from scripts.static_bfs_graph.runner import write_atomic
    result.reload()
    _check()
    if measured_n not in (5, 10, 15, 20, 25, 30):
        raise ValueError("invalid cost checkpoint")
    if not read(ROOT / "racing/correctness_gate.json")["passed"]:
        raise RuntimeError("correctness gate failed")
    rows = [read(ROOT / f"racing/raw/trajectory_{i:02d}/timing.json") for i in range(measured_n)]
    max_new_usd = float(read(ROOT / "experiment_manifest.json")["max_new_gpu_usd"])
    used = sum(x["trajectory_wall_seconds"] for x in rows)
    normal = max(x["trajectory_wall_seconds"]-x["correctness_gate_seconds"] for x in rows)
    projected = used + (30-measured_n)*normal*1.25
    estimate = projected*RATE_USD_PER_SECOND + audit_projection_usd
    out = {"measured_n": measured_n, "used_racing_GPU_seconds": used,
           "projected_30_racing_GPU_seconds": projected,
           "projected_30_racing_USD": projected*RATE_USD_PER_SECOND,
           "audit_projection_USD": audit_projection_usd,
           "projected_total_new_USD": estimate,
           "cap_new_USD": max_new_usd, "allowed": estimate <= max_new_usd}
    write_atomic(ROOT / "racing/cost_projection.json", out)
    result.commit()
    if not out["allowed"]:
        raise RuntimeError("new GPU cost projection exceeds frozen cap")
    return out


@app.local_entrypoint()
def racing_cost_gate(measured_n: int, audit_projection_usd: float):
    print(json.dumps(remote_racing_cost_gate.remote(measured_n, audit_projection_usd), sort_keys=True))


@app.function(image=image, volumes={"/result": result}, timeout=1800, cpu=4, memory=8192)
def remote_analyze_round(round_number: int):
    from modal_n8_gpu.aer_allocation_racing import analyze_round
    from modal_n8_gpu.aer_plan import read
    result.reload()
    _check()
    output = {str(arm): analyze_round(ROOT, arm, round_number)
              for arm in (5, 10)
              if not (round_number > 1 and
                      read(ROOT / f"racing/arm{arm}_round_{round_number-1}_state.json")["stop"])}
    result.commit()
    return output


@app.local_entrypoint()
def analyze_round(round_number: int):
    print(json.dumps(remote_analyze_round.remote(round_number), sort_keys=True))


@app.function(image=image, volumes={"/result": result}, timeout=300, cpu=1, memory=2048)
def remote_status():
    result.reload()
    return {"racing_done": [i for i in range(30) if
            (ROOT / f"racing/raw/trajectory_{i:02d}/timing.json").exists()],
            "audit_done": [i for i in range(20) if
            (ROOT / f"audit/timing/trajectory_{i:02d}.json").exists()]}


@app.local_entrypoint()
def status():
    print(json.dumps(remote_status.remote(), sort_keys=True))


@app.function(image=image, volumes={"/result": result}, timeout=300, cpu=1, memory=2048)
def remote_freeze_audit(files: dict[str, str], hashes: dict[str, str]):
    from modal_n8_gpu.aer_plan import digest, read
    if set(files) != {"frozen/frozen_compiler_outputs.json", "frozen/frozen_audit_plan.json"} or set(files) != set(hashes):
        raise RuntimeError("frozen audit file list invalid")
    result.reload()
    _check()
    for name, raw in files.items():
        if sha256(raw.encode()).hexdigest() != hashes[name]:
            raise RuntimeError("local freeze hash mismatch")
        path = ROOT / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.read_text() != raw:
            raise RuntimeError("remote audit freeze changed")
        path.write_text(raw)
    frozen = read(ROOT / "frozen/frozen_audit_plan.json")
    if (frozen["compiler_outputs_sha256"] != digest(ROOT / "frozen/frozen_compiler_outputs.json") or
        frozen["audit_pool_sha256"] != digest(ROOT / "audit/audit_pool.json") or
        len(frozen["pairs"]) != 20):
        raise RuntimeError("audit freeze references changed")
    result.commit()
    return {"passed": True, "frozen_audit_plan_sha256": digest(ROOT / "frozen/frozen_audit_plan.json"),
            "unique_policies": len(frozen["unique_policies"])}


@app.local_entrypoint()
def freeze_audit():
    root = Path(__file__).resolve().parents[1] / "results/static-bfs-graph/aer-budget-allocation-v1"
    files = {name: (root/name).read_text() for name in
             ("frozen/frozen_compiler_outputs.json", "frozen/frozen_audit_plan.json")}
    print(json.dumps(remote_freeze_audit.remote(files, {
        name: sha256(raw.encode()).hexdigest() for name, raw in files.items()}), sort_keys=True))


@app.function(image=image, volumes={"/result": result}, timeout=1800, cpu=4, memory=8192)
def remote_freeze_compiler():
    from modal_n8_gpu.aer_allocation_analysis import freeze
    from modal_n8_gpu.aer_allocation_cost import build, difficulty
    result.reload()
    _check()
    cost = build(ROOT)
    diagnostic = difficulty(ROOT)
    frozen = freeze(ROOT)
    result.commit()
    return {"cost": cost, "difficulty_rows": diagnostic["difficulty_rows"], "frozen": frozen}


@app.local_entrypoint()
def freeze_compiler():
    print(json.dumps(remote_freeze_compiler.remote(), sort_keys=True))


@app.function(image=image, gpu="L40S", volumes={"/vol": cache, "/result": result},
              timeout=3600, cpu=8, memory=24576, max_containers=4)
def remote_audit_trajectory(rank: int):
    import torch
    from modal_n8_gpu.aer_plan import digest, read
    from scripts.static_bfs_graph.graph_builder import dense_reference
    from scripts.static_bfs_graph.runner import write_atomic
    result.reload()
    cache.reload()
    _check()
    if rank not in range(20):
        raise ValueError("audit rank invalid")
    if rank >= 1 and not read(ROOT / "audit/cost_projection.json")["allowed"]:
        raise RuntimeError("audit cost gate has not passed")
    frozen = read(ROOT / "frozen/frozen_audit_plan.json")
    if frozen["compiler_outputs_sha256"] != digest(ROOT / "frozen/frozen_compiler_outputs.json"):
        raise RuntimeError("compiler selection changed after freeze")
    pair = frozen["pairs"][rank]
    prompt_id, seed = pair["prompt_id"], pair["seed"]
    pool = read(REMOTE / "modal_n8_gpu/train_pool.json")
    if sha256(pool["prompts"][str(prompt_id)].encode()).hexdigest() != pair["prompt_sha256"]:
        raise RuntimeError("audit prompt hash changed")
    timing_path = ROOT / f"audit/timing/trajectory_{rank:02d}.json"
    if timing_path.exists():
        timing = read(timing_path)
        if (timing["prompt_id"] != prompt_id or timing["seed"] != seed or
            timing["unique_policy_count"] != len(frozen["unique_policies"])):
            raise RuntimeError("completed audit trajectory inconsistent")
        return timing
    started = time.perf_counter()
    start_unix = time.time()
    runtime = _context(seed)
    reference = dense_reference((10, 20, 30, 40, 60, 80, 90), 8.0, 8)
    policy_seconds = 0.0
    for item in frozen["unique_policies"]:
        row = runtime.run("allocation_independent_audit", item["record"], prompt_id, reference)
        if row["policy_id"] != item["policy_id"] or row["trial_seed"] != seed:
            raise RuntimeError("audit policy rollout mismatch")
        policy_seconds += row["elapsed_seconds"]
        write_atomic(ROOT / "audit/raw" / item["policy_id"] / f"p{prompt_id}_s{seed}.json", row)
        result.commit()
    timing = {"rank": rank, "prompt_id": prompt_id, "seed": seed,
              "GPU_type": torch.cuda.get_device_name(0),
              "pipeline_load_seconds": runtime.load_seconds,
              "policy_seconds": policy_seconds,
              "unique_policy_count": len(frozen["unique_policies"]),
              "wall_start_unix": start_unix, "wall_end_unix": time.time(),
              "trajectory_wall_seconds": time.perf_counter()-started}
    write_atomic(timing_path, timing)
    result.commit()
    return timing


@app.local_entrypoint()
def audit_pilot():
    print(json.dumps(remote_audit_trajectory.remote(0), sort_keys=True))


@app.function(image=image, volumes={"/result": result}, timeout=300, cpu=1, memory=2048)
def remote_audit_cost_gate():
    from modal_n8_gpu.aer_allocation_plan import RATE_USD_PER_SECOND
    from modal_n8_gpu.aer_plan import read
    from scripts.static_bfs_graph.runner import write_atomic
    result.reload()
    _check()
    racing = [read(ROOT / f"racing/raw/trajectory_{i:02d}/timing.json")
              for i in range(30) if (ROOT / f"racing/raw/trajectory_{i:02d}/timing.json").exists()]
    pilot = read(ROOT / "audit/timing/trajectory_00.json")
    max_new_usd = float(read(ROOT / "experiment_manifest.json")["max_new_gpu_usd"])
    racing_usd = sum(x["trajectory_wall_seconds"] for x in racing)*RATE_USD_PER_SECOND
    projected_audit = pilot["trajectory_wall_seconds"]*20*1.25*RATE_USD_PER_SECOND
    estimated = racing_usd + projected_audit
    out = {"measured_racing_trajectories": len(racing), "racing_estimated_USD": racing_usd,
           "audit_pilot_GPU_seconds": pilot["trajectory_wall_seconds"],
           "projected_audit_USD": projected_audit,
           "projected_total_new_USD": estimated,
           "cap_new_USD": max_new_usd, "allowed": estimated <= max_new_usd}
    write_atomic(ROOT / "audit/cost_projection.json", out)
    result.commit()
    if not out["allowed"]:
        raise RuntimeError("projected new GPU cost exceeds cap")
    return out


@app.local_entrypoint()
def audit_cost_gate():
    print(json.dumps(remote_audit_cost_gate.remote(), sort_keys=True))


@app.function(image=image, volumes={"/result": result}, timeout=7200, cpu=1, memory=2048)
def remote_audit_stage():
    rows = list(remote_audit_trajectory.map(range(1, 20)))
    if len(rows) != 19:
        raise RuntimeError("audit incomplete")
    return {"completed": 19, "summed_GPU_seconds": sum(x["trajectory_wall_seconds"] for x in rows)}


@app.local_entrypoint()
def audit_stage():
    print(json.dumps(remote_audit_stage.remote(), sort_keys=True))


@app.function(image=image, volumes={"/result": result}, timeout=1200, cpu=2, memory=4096)
def remote_audit_aggregate():
    from modal_n8_gpu.aer_allocation_analysis import audit_aggregate
    result.reload()
    _check()
    out = audit_aggregate(ROOT)
    result.commit()
    return out


@app.local_entrypoint()
def audit_aggregate():
    print(json.dumps(remote_audit_aggregate.remote(), sort_keys=True))
