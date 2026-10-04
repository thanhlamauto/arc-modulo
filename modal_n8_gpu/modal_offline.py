"""Modal GPU execution for the offline shared-tree N=8 BFS compiler."""
from __future__ import annotations

import json
import os
from pathlib import Path
import time

import modal

from modal_n8_gpu.modal_app import image, cache

app = modal.App("static-bfs-n8-offline-compiler-v1")
result = modal.Volume.from_name("static-bfs-n8-offline-compiler-v1-results", create_if_missing=True)
ROOT = Path("/result/n8-offline-compiler-v1")
REMOTE = Path("/opt/repro")
PROTOCOL = "STATIC_BFS_N8_OFFLINE_COMPILER_V1"


def _context(seed: int):
    from modal_n8_gpu.gpu_runtime import Runtime
    from scripts.static_bfs_graph.n8_plan import n8_config
    from scripts.static_bfs_graph.runner import read
    pool = read(REMOTE / "modal_n8_gpu/train_pool.json")
    old = read(REMOTE / "configs/diffusion_classical_search_t2i_table12.json")
    prompts = [{"prompt": pool["prompts"].get(str(i))} for i in range(100)]
    return Runtime(ROOT, Path("/opt/official"), seed, n8_config(), old, prompts)


@app.function(image=image, volumes={"/result": result}, timeout=300, cpu=1, memory=2048)
def offline_preflight():
    from modal_n8_gpu.lightweight_plan import preflight
    out = preflight(ROOT, REMOTE / "modal_n8_gpu/train_pool.json", PROTOCOL)
    result.commit()
    return out


@app.local_entrypoint()
def preflight():
    print("OFFLINE_PREFLIGHT", json.dumps(offline_preflight.remote(), sort_keys=True))


@app.function(image=image, volumes={"/result": result}, timeout=120, cpu=1, memory=1024)
def offline_lock_stop_rule():
    from modal_n8_gpu.lightweight_plan import digest, write_once
    result.reload()
    path = ROOT / "engineering_stop_rule.json"
    write_once(path, {"preregistered_before_graph_analysis": True,
                      "skip_n20_only_if_all": ["P5_equals_P10",
                                                "n10_bootstrap_exact_path_frequency_at_least_0.80",
                                                "Spearman_G5_G10_at_least_0.95",
                                                "projected_n20_compiler_cost_at_least_USD_5"],
                      "otherwise": "run n20 if compiler cost gate permits",
                      "n20_if_skipped": "unmeasured; audit P10"})
    result.commit()
    return {"sha256": digest(path)}


@app.local_entrypoint()
def lock_stop_rule():
    print("OFFLINE_STOP_RULE", json.dumps(offline_lock_stop_rule.remote(), sort_keys=True))


@app.function(image=image, gpu="L40S", volumes={"/vol": cache, "/result": result},
              timeout=3600, cpu=8, memory=24576)
def offline_replay_equivalence():
    import math
    import torch
    from modal_n8_gpu.prefix_replay import reference_with_snapshots, replay_suffix
    from scripts.static_bfs_graph.graph_builder import all_edges, dense_reference, edge_intervention
    from scripts.static_bfs_graph.runner import read, static_record, write_atomic
    from scripts.static_bfs_graph.offline_profiler.profiler import profile_trajectory

    result.reload()
    cache.reload()
    manifest = read(ROOT / "experiment_manifest.json")
    if manifest["protocol"] != PROTOCOL or manifest["PSP_in_scope"]:
        raise RuntimeError("preflight missing or scope changed")
    trajectories = read(ROOT / "calibration_trajectories.json")
    reference = dense_reference((10, 20, 30, 40, 60, 80, 90), 8.0, 8)
    checks = []
    naive_benchmarks = []
    first_runtime = None
    for rank in (0, 1):
        p, seed = trajectories[rank]["prompt_id"], trajectories[rank]["seed"]
        runtime = _context(seed)
        snap_path = ROOT / f"replay/probe_{rank}.pt"
        row_path = ROOT / f"replay/reference_{rank}.json"
        full = reference_with_snapshots(runtime, prompt_index=p, trial_seed=seed,
                                        snapshot_path=snap_path, record_path=row_path,
                                        protocol=PROTOCOL)
        checkpoint = torch.load(snap_path, map_location="cpu", weights_only=False)
        for src in (10, 40, 80):
            replay = replay_suffix(runtime, checkpoint=checkpoint, src=src, policy=reference)
            same_events = replay["resampling_events"] == full["resampling_events"]
            passed = (same_events and replay["final_particle_scores"] == full["final_particle_scores"]
                      and replay["selected_particle_index"] == full["selected_particle_index"]
                      and replay["final_latent_sha256"] == full["final_latent_sha256"]
                      and replay["selected_image_rgb_sha256"] == full["selected_image_rgb_sha256"]
                      and abs(replay["selected_final_score"] - full["selected_final_score"]) <= 1e-6)
            checks.append({"rank": rank, "src": src, "pass": passed,
                           "event_schedule_raw_scores_weights_parents_equal": same_events,
                           "selected_reward_error": abs(replay["selected_final_score"] - full["selected_final_score"]),
                           "final_latent_equal": replay["final_latent_sha256"] == full["final_latent_sha256"]})
            if not passed:
                break
        if not all(x["pass"] for x in checks):
            break
        if rank == 0:
            edges = all_edges((10, 20, 30, 40, 60, 80, 90), (2.0, 8.0, 32.0))
            keys = {"-1_to_10_tau_2", "10_to_20_tau_8", "20_to_60_tau_32",
                    "40_to_90_tau_2", "90_to_100_tau_end"}
            chosen = [e for e in edges if e.key in keys]
            if len(chosen) != 5:
                raise RuntimeError("benchmark edge keys changed")
            tree = profile_trajectory(runtime, checkpoint, full, ROOT / "replay/tree_probe", keys)
            for edge in chosen:
                policy = edge_intervention(edge, reference)
                naive = runtime.run("offline_naive_gate", static_record(policy), p, reference, edge)
                shared = read(ROOT / "replay/tree_probe" / f"{edge.key}.json")
                same_events = shared["resampling_events"] == naive["resampling_events"]
                passed = (same_events and shared["final_particle_scores"] == naive["final_particle_scores"]
                          and shared["selected_particle_index"] == naive["selected_particle_index"]
                          and shared["selected_image_rgb_sha256"] == naive["selected_image_rgb_sha256"]
                          and abs(shared["selected_final_score"] - naive["selected_final_score"]) <= 1e-6)
                naive_benchmarks.append({"edge_key": edge.key, "pass": passed,
                                         "naive_seconds": naive["elapsed_seconds"],
                                         "tree_branch_suffix_seconds": shared["branch_suffix_wall_seconds"],
                                         "full_event_records_equal": same_events})
                if not passed:
                    break
            write_atomic(ROOT / "replay/benchmark.json", {"tree": tree, "naive": naive_benchmarks,
                         "five_edge_naive_seconds": sum(x["naive_seconds"] for x in naive_benchmarks),
                         "five_edge_shared_seconds": tree["source_trunk_seconds"] +
                             tree["shared_verifier_seconds"] + tree["tau_suffix_seconds"]})
            if not all(x["pass"] for x in naive_benchmarks):
                break
    out = {"passed": len(checks) == 6 and len(naive_benchmarks) == 5 and
            all(x["pass"] for x in checks) and all(x["pass"] for x in naive_benchmarks),
           "reference_replays": checks, "shared_tree_vs_naive_edges": naive_benchmarks,
           "terminal_reward_tolerance": 1e-6, "scientific_profiling_started": False}
    write_atomic(ROOT / "replay/equivalence.json", out)
    result.commit()
    if not out["passed"]:
        raise RuntimeError("shared-tree equivalence gate failed; graph profiling prohibited")
    return out


@app.local_entrypoint()
def replay_test():
    print("OFFLINE_REPLAY_GATE", json.dumps(offline_replay_equivalence.remote(), sort_keys=True))


@app.function(image=image, gpu="L40S", volumes={"/vol": cache, "/result": result},
              timeout=1800, cpu=8, memory=24576)
def offline_tau_fork_equivalence():
    import torch
    from scripts.static_bfs_graph.graph_builder import all_edges, dense_reference, edge_intervention
    from scripts.static_bfs_graph.offline_profiler.profiler import profile_trajectory
    from scripts.static_bfs_graph.runner import read, static_record, write_atomic
    result.reload()
    cache.reload()
    if not read(ROOT / "replay/equivalence.json")["passed"]:
        raise RuntimeError("primary replay gate failed")
    checkpoint = torch.load(ROOT / "replay/probe_0.pt", map_location="cpu", weights_only=False)
    full = read(ROOT / "replay/reference_0.json")
    runtime = _context(checkpoint["trial_seed"])
    edges = [e for e in all_edges((10, 20, 30, 40, 60, 80, 90), (2.0, 8.0, 32.0))
             if e.src == 20 and e.dst == 60]
    tree = profile_trajectory(runtime, checkpoint, full, ROOT / "replay/tau_fork_tree",
                              {e.key for e in edges})
    reference = dense_reference((10, 20, 30, 40, 60, 80, 90), 8.0, 8)
    checks = []
    for edge in edges:
        policy = edge_intervention(edge, reference)
        naive = runtime.run("offline_tau_fork_naive", static_record(policy),
                            checkpoint["prompt_index"], reference, edge)
        branch = read(ROOT / "replay/tau_fork_tree" / f"{edge.key}.json")
        passed = (branch["resampling_events"] == naive["resampling_events"] and
                  branch["final_particle_scores"] == naive["final_particle_scores"] and
                  branch["selected_image_rgb_sha256"] == naive["selected_image_rgb_sha256"])
        checks.append({"edge_key": edge.key, "passed": passed,
                       "same_full_events": branch["resampling_events"] == naive["resampling_events"]})
    out = {"passed": len(checks) == 3 and all(x["passed"] for x in checks),
           "shared_verifier_count": tree["shared_destination_verifier_evaluations"],
           "tau_suffix_branches": tree["tau_suffix_branches"], "checks": checks}
    write_atomic(ROOT / "replay/tau_fork_equivalence.json", out)
    result.commit()
    if not out["passed"] or out["shared_verifier_count"] != 1 or out["tau_suffix_branches"] != 3:
        raise RuntimeError("tau-fork equivalence failed")
    return out


@app.local_entrypoint()
def tau_fork_test():
    print("OFFLINE_TAU_FORK_GATE", json.dumps(offline_tau_fork_equivalence.remote(), sort_keys=True))


@app.function(image=image, gpu="L40S", volumes={"/vol": cache, "/result": result},
              timeout=7200, cpu=8, memory=24576, max_containers=4)
def offline_profile_trajectory(rank: int):
    import torch
    from modal_n8_gpu.prefix_replay import reference_with_snapshots
    from scripts.static_bfs_graph.offline_profiler.profiler import profile_trajectory
    from scripts.static_bfs_graph.runner import read, write_atomic

    if rank not in range(20):
        raise ValueError("rank out of range")
    result.reload()
    cache.reload()
    gate = read(ROOT / "replay/equivalence.json")
    if not gate["passed"]:
        raise RuntimeError("replay equivalence gate failed")
    if rank > 0:
        cost = read(ROOT / "profiler/cost_projection.json")
        if not cost["allowed"]:
            raise RuntimeError("compiler cost cap exceeded")
    traj = read(ROOT / "calibration_trajectories.json")[rank]
    p, seed = traj["prompt_id"], traj["seed"]
    timing_path = ROOT / f"profiler/timing/trajectory_{rank:02d}.json"
    if timing_path.exists():
        row = read(timing_path)
        if row["prompt_index"] == p and row["trial_seed"] == seed and row["logical_edges"] == 92:
            return row
        raise RuntimeError("invalid prior trajectory timing")
    runtime = _context(seed)
    started = time.perf_counter()
    snap_path = ROOT / f"profiler/trajectory_{rank:02d}/reference_snapshots.pt"
    ref_path = ROOT / f"profiler/trajectory_{rank:02d}/reference.json"
    if snap_path.exists() != ref_path.exists():
        partial = snap_path if snap_path.exists() else ref_path
        partial.unlink()
    full = reference_with_snapshots(runtime, prompt_index=p, trial_seed=seed,
                                    snapshot_path=snap_path, record_path=ref_path,
                                    protocol=PROTOCOL)
    checkpoint = torch.load(snap_path, map_location="cpu", weights_only=False)
    profile = profile_trajectory(runtime, checkpoint, full,
                                 ROOT / f"profiler/trajectory_{rank:02d}/edges")
    summary = {"rank": rank, "prompt_index": p, "trial_seed": seed,
               "GPU_type": torch.cuda.get_device_name(0),
               "reference_wall_seconds": full["reference_wall_seconds"],
               "snapshot_creation_seconds": full["snapshot_creation_seconds"],
               "snapshot_storage_bytes": snap_path.stat().st_size,
               "pipeline_load_seconds": runtime.load_seconds,
               "trajectory_wall_seconds": time.perf_counter() - started,
               **profile}
    write_atomic(timing_path, summary)
    result.commit()
    return summary


@app.local_entrypoint()
def profile_one():
    print("OFFLINE_PROFILE_1", json.dumps(offline_profile_trajectory.remote(0), sort_keys=True))


@app.function(image=image, volumes={"/result": result}, timeout=7200, cpu=1, memory=2048)
def offline_profile_stage(first_rank: int, stop_rank: int):
    from scripts.static_bfs_graph.runner import read
    if (first_rank, stop_rank) not in ((1, 5), (5, 10), (10, 20)):
        raise ValueError("profiling stage must be (1,5), (5,10), or (10,20)")
    result.reload()
    if not read(ROOT / "profiler/cost_projection.json")["allowed"]:
        raise RuntimeError("compiler cost gate failed")
    rows = list(offline_profile_trajectory.map(range(first_rank, stop_rank)))
    if len(rows) != stop_rank - first_rank or any(row["logical_edges"] != 92 for row in rows):
        raise RuntimeError("profiling stage incomplete")
    return {"first_rank": first_rank, "stop_rank": stop_rank,
            "completed_trajectories": len(rows),
            "summed_gpu_seconds": sum(row["trajectory_wall_seconds"] for row in rows)}


@app.local_entrypoint()
def profile_stage(first_rank: int, stop_rank: int):
    print("OFFLINE_PROFILE_STAGE", json.dumps(offline_profile_stage.remote(first_rank, stop_rank), sort_keys=True))


@app.function(image=image, volumes={"/result": result}, timeout=300, cpu=1, memory=2048)
def offline_cost_gate(max_usd: float = 25.0, measured_n: int = 1):
    from scripts.static_bfs_graph.runner import read, write_atomic
    result.reload()
    if measured_n not in (1, 5, 10):
        raise ValueError("cost gate runs after n=1,5,10")
    timings = [read(ROOT / f"profiler/timing/trajectory_{i:02d}.json") for i in range(measured_n)]
    measured_seconds = sum(row["trajectory_wall_seconds"] for row in timings)
    rate = 0.000542 + 8 * 0.0000131 + 24 * 0.00000222
    projected = 20 * measured_seconds / measured_n * rate
    out = {"max_usd": max_usd, "measured_n": measured_n,
           "measured_gpu_seconds": measured_seconds,
           "measured_estimated_usd": measured_seconds * rate,
           "projected_n20_compiler_usd": projected,
           "pricing_source": "https://modal.com/pricing", "audit_excluded": True,
           "allowed": projected <= max_usd}
    write_atomic(ROOT / "profiler/cost_projection.json", out)
    result.commit()
    if not out["allowed"]:
        raise RuntimeError(f"Projected compiler ${projected:.2f} exceeds cap ${max_usd:.2f}")
    return out


@app.local_entrypoint()
def cost_gate(measured_n: int = 1):
    cap = float(os.environ.get("STATIC_BFS_COMPILER_MAX_USD", "25"))
    print("OFFLINE_COMPILER_COST_GATE", json.dumps(offline_cost_gate.remote(cap, measured_n), sort_keys=True))


@app.function(image=image, volumes={"/result": result}, timeout=900, cpu=2, memory=4096)
def offline_graph_analysis(max_n: int):
    from modal_n8_gpu.offline_analysis import analyze
    result.reload()
    output = analyze(ROOT, max_n)
    result.commit()
    return output


@app.local_entrypoint()
def graph_analysis(max_n: int):
    print("OFFLINE_GRAPH_ANALYSIS", json.dumps(offline_graph_analysis.remote(max_n), sort_keys=True))


@app.function(image=image, volumes={"/result": result}, timeout=300, cpu=1, memory=2048)
def offline_freeze_audit(measured_n: int):
    from modal_n8_gpu.lightweight_audit import freeze_audit
    from modal_n8_gpu.lightweight_plan import digest, write_once
    from scripts.static_bfs_graph.runner import read
    result.reload()
    plan = freeze_audit(ROOT, measured_n)
    policies = read(ROOT / "graphs/policies_by_n.json")
    final = next(row for row in policies if row["n"] == measured_n)
    source = ROOT / f"graphs/edge_values_n{measured_n}.csv"
    cfg = ROOT / "experiment_manifest.json"
    compiled = {"schema_version": 1, "model": "sd15",
                "particle_budget_used_for_compilation": 8, "K": 3,
                "events": [{"sampling_idx": s, "tau": t}
                           for s, t in zip(final["steps"], final["taus"])],
                "scoring": "max", "resampling": "ssp",
                "source_edge_table_sha256": digest(source),
                "compiler_config_sha256": digest(cfg),
                "compiled_from_n": measured_n, "policy_id": final["policy_id"]}
    path = ROOT / "compiled/compiled_policy.json"
    write_once(path, compiled)
    write_once(ROOT / "compiled/compilation_manifest.json", {
        "compiled_policy_sha256": digest(path), "audit_plan_sha256": plan["audit_policy_plan_sha256"],
        "graph_policy_source_sha256": digest(ROOT / "graphs/policies_by_n.json"),
        "deployment_requires_graph": False})
    result.commit()
    return {**plan, "compiled_policy_sha256": digest(path), "compiled_policy": compiled}


@app.local_entrypoint()
def freeze_audit(measured_n: int):
    print("OFFLINE_FREEZE_AUDIT", json.dumps(offline_freeze_audit.remote(measured_n), sort_keys=True))


@app.function(image=image, volumes={"/result": result}, timeout=300, cpu=1, memory=2048)
def offline_audit_cost_gate(max_usd: float = 25.0):
    from scripts.static_bfs_graph.runner import read, write_atomic
    result.reload()
    plan = read(ROOT / "audit/audit_policy_plan.json")
    benchmark = read(ROOT / "replay/benchmark.json")
    edge_times = [x["naive_seconds"] for x in benchmark["naive"]]
    rate = 0.000542 + 8 * 0.0000131 + 24 * 0.00000222
    projected = len(plan["policies"]) * 80 * max(edge_times) * rate
    out = {"max_usd": max_usd, "projected_audit_usd": projected,
           "unique_policies": len(plan["policies"]), "audit_units": len(plan["policies"]) * 80,
           "projection_seconds_per_unit": max(edge_times),
           "pricing_source": "https://modal.com/pricing", "compiler_cost_excluded": True,
           "allowed": projected <= max_usd}
    write_atomic(ROOT / "audit/cost_projection.json", out)
    result.commit()
    if not out["allowed"]:
        raise RuntimeError(f"Projected audit ${projected:.2f} exceeds cap ${max_usd:.2f}")
    return out


@app.local_entrypoint()
def audit_cost_gate():
    cap = float(os.environ.get("STATIC_BFS_AUDIT_MAX_USD", "25"))
    print("OFFLINE_AUDIT_COST_GATE", json.dumps(offline_audit_cost_gate.remote(cap), sort_keys=True))


@app.function(image=image, gpu="L40S", volumes={"/vol": cache, "/result": result},
              timeout=7200, cpu=8, memory=24576, max_containers=4)
def offline_audit_seed(seed: int):
    from modal_n8_gpu.lightweight_audit import records_for_plan
    from scripts.static_bfs_graph.graph_builder import dense_reference
    from scripts.static_bfs_graph.runner import read, write_atomic
    if seed not in (42, 43, 44, 45):
        raise ValueError("audit seed invalid")
    result.reload()
    cache.reload()
    plan = read(ROOT / "audit/audit_policy_plan.json")
    cost = read(ROOT / "audit/cost_projection.json")
    if not cost["allowed"] or plan["PSP_in_scope"]:
        raise RuntimeError("audit cost or scope gate failed")
    records = records_for_plan(plan)
    prompts = read(ROOT / "audit_prompts.json")
    reference = dense_reference((10, 20, 30, 40, 60, 80, 90), 8.0, 8)
    runtime = _context(seed)
    completed = 0
    for prompt in prompts:
        p = prompt["prompt_id"]
        for record in records.values():
            pid = record["id"]
            target = ROOT / f"audit/raw/{pid}/p{p}_s{seed}.json"
            if target.exists():
                old = read(target)
                if old["policy_id"] != pid or old["prompt_index"] != p or old["trial_seed"] != seed:
                    raise RuntimeError("invalid resume row")
            else:
                row = runtime.run("offline_heldout_train_audit", record, p, reference)
                write_atomic(target, row)
            completed += 1
        result.commit()
    return {"seed": seed, "completed_rows": completed, "unique_policies": len(records),
            "prompt_count": len(prompts)}


@app.local_entrypoint()
def audit_one_seed(seed: int):
    print("OFFLINE_AUDIT_SEED", json.dumps(offline_audit_seed.remote(seed), sort_keys=True))


@app.function(image=image, volumes={"/result": result}, timeout=7200, cpu=1, memory=2048)
def offline_audit_stage():
    from scripts.static_bfs_graph.runner import read
    result.reload()
    if not read(ROOT / "audit/cost_projection.json")["allowed"]:
        raise RuntimeError("audit cost gate failed")
    rows = list(offline_audit_seed.map((42, 43, 44, 45)))
    if sorted(x["seed"] for x in rows) != [42, 43, 44, 45]:
        raise RuntimeError("audit shards incomplete")
    return {"seeds_complete": 4, "raw_rows": sum(x["completed_rows"] for x in rows)}


@app.local_entrypoint()
def audit_stage():
    print("OFFLINE_AUDIT_STAGE", json.dumps(offline_audit_stage.remote(), sort_keys=True))


@app.function(image=image, volumes={"/result": result}, timeout=900, cpu=2, memory=4096)
def offline_aggregate():
    from modal_n8_gpu.lightweight_audit import aggregate
    result.reload()
    out = aggregate(ROOT)
    result.commit()
    return out


@app.local_entrypoint()
def aggregate():
    print("OFFLINE_AGGREGATE", json.dumps(offline_aggregate.remote(), sort_keys=True))
