"""TRAIN-only N8 static-BFS lightweight compilation on Modal CUDA."""
from __future__ import annotations

import json
import os
from pathlib import Path

import modal

from modal_n8_gpu.modal_app import image, cache

app = modal.App("static-bfs-n8-lightweight-v1")
result = modal.Volume.from_name("static-bfs-n8-lightweight-v1-results", create_if_missing=True)
RUN_ROOT = Path("/result/n8-lightweight-v1")
REMOTE = Path("/opt/repro")


def _context():
    from modal_n8_gpu.gpu_runtime import Runtime
    from scripts.static_bfs_graph.n8_plan import n8_config
    from scripts.static_bfs_graph.runner import read
    pool = read(REMOTE / "modal_n8_gpu/train_pool.json")
    old = read(REMOTE / "configs/diffusion_classical_search_t2i_table12.json")
    prompts = [{"prompt": pool["prompts"].get(str(i))} for i in range(100)]
    return Runtime, n8_config(), old, prompts


@app.function(image=image, volumes={"/result": result}, timeout=300, cpu=2, memory=4096)
def lw_preflight():
    from modal_n8_gpu.lightweight_plan import preflight
    data = preflight(RUN_ROOT, REMOTE / "modal_n8_gpu/train_pool.json")
    result.commit()
    return data


@app.function(image=image, gpu="L40S", volumes={"/vol": cache, "/result": result},
              timeout=3600, cpu=8, memory=24576)
def lw_replay_test():
    import math
    import torch
    from modal_n8_gpu.prefix_replay import reference_with_snapshots, replay_suffix
    from scripts.static_bfs_graph.graph_builder import all_edges, dense_reference, edge_intervention
    from scripts.static_bfs_graph.runner import read, static_record, write_atomic

    cache.reload()
    result.reload()
    manifest = read(RUN_ROOT / "experiment_manifest.json")
    if manifest["protocol"] != "STATIC_BFS_N8_LIGHTWEIGHT_V1":
        raise RuntimeError("preflight missing")
    trajectory = read(RUN_ROOT / "calibration_trajectories.json")[0]
    p, seed = trajectory["prompt_id"], trajectory["seed"]
    Runtime, cfg, old, prompts = _context()
    runtime = Runtime(RUN_ROOT, Path("/opt/official"), seed, cfg, old, prompts)
    snapshot_path = RUN_ROOT / "replay/snapshots/probe0.pt"
    record_path = RUN_ROOT / "replay/reference_probe.json"
    refrow = reference_with_snapshots(
        runtime, prompt_index=p, trial_seed=seed,
        snapshot_path=snapshot_path, record_path=record_path,
    )
    checkpoint = torch.load(snapshot_path, map_location="cpu", weights_only=False)
    ref = dense_reference((10, 20, 30, 40, 60, 80, 90), 8.0, 8)
    checks = []
    for src in (-1, 10, 40, 90):
        rr = replay_suffix(runtime, checkpoint=checkpoint, src=src, policy=ref)
        score_error = max(abs(a - b) for a, b in zip(refrow["final_particle_scores"], rr["final_particle_scores"]))
        parents_equal = [e["parent_indices"] for e in rr["resampling_events"]] == [
            e["parent_indices"] for e in refrow["resampling_events"]]
        passed = (score_error <= 1e-6 and
                  rr["selected_particle_index"] == refrow["selected_particle_index"] and
                  rr["final_latent_sha256"] == refrow["final_latent_sha256"] and
                  rr["selected_image_rgb_sha256"] == refrow["selected_image_rgb_sha256"] and
                  parents_equal)
        checks.append({"kind": "reference", "src": src, "passed": passed,
                       "max_particle_score_error": score_error,
                       "latent_equal": rr["final_latent_sha256"] == refrow["final_latent_sha256"],
                       "image_equal": rr["selected_image_rgb_sha256"] == refrow["selected_image_rgb_sha256"],
                       "parents_equal": parents_equal,
                       "suffix_seconds": rr["edge_replay_wall_seconds"]})
        if not passed:
            break
    if all(x["passed"] for x in checks):
        edges = all_edges((10, 20, 30, 40, 60, 80, 90), (2.0, 8.0, 32.0))
        for edge in (edges[9], edges[60], edges[91]):
            policy = edge_intervention(edge, ref)
            full = runtime.run("lw_replay_full_edge", static_record(policy), p, ref, edge)
            replay = replay_suffix(runtime, checkpoint=checkpoint, src=edge.src, policy=policy)
            score_error = max(abs(a - b) for a, b in zip(full["final_particle_scores"], replay["final_particle_scores"]))
            parents_equal = [e["parent_indices"] for e in full["resampling_events"]] == [
                e["parent_indices"] for e in replay["resampling_events"]]
            passed = (score_error <= 1e-6 and
                      replay["selected_particle_index"] == full["selected_particle_index"] and
                      replay["selected_image_rgb_sha256"] == full["selected_image_rgb_sha256"] and
                      parents_equal)
            checks.append({"kind": "edge", "edge": edge.key, "src": edge.src,
                           "passed": passed, "max_particle_score_error": score_error,
                           "image_equal": replay["selected_image_rgb_sha256"] == full["selected_image_rgb_sha256"],
                           "parents_equal": parents_equal,
                           "full_seconds": full["elapsed_seconds"],
                           "suffix_seconds": replay["edge_replay_wall_seconds"]})
            if not passed:
                break
    passed = all(x["passed"] for x in checks) and len(checks) == 7
    out = {"passed": passed, "prompt_index": p, "trial_seed": seed,
           "score_tolerance": 1e-6, "checks": checks,
           "GPU_type": torch.cuda.get_device_name(0),
           "no_graph_profiling_before_pass": True}
    write_atomic(RUN_ROOT / "replay/equivalence_results.json", out)
    result.commit()
    if not passed:
        raise RuntimeError("prefix replay equivalence failed; STOP before graph profiling")
    return out


@app.local_entrypoint()
def preflight():
    print("LW_PREFLIGHT", json.dumps(lw_preflight.remote(), sort_keys=True))


@app.local_entrypoint()
def replay_test():
    print("LW_REPLAY_TEST", json.dumps(lw_replay_test.remote(), sort_keys=True))


@app.function(image=image, volumes={"/result": result}, timeout=300, cpu=1, memory=2048)
def lw_cost_canary(max_usd: float = 50.0):
    from scripts.static_bfs_graph.graph_builder import all_edges, dense_reference, edge_intervention
    from scripts.static_bfs_graph.runner import read, write_atomic
    result.reload()
    test = read(RUN_ROOT / "replay/equivalence_results.json")
    if not test["passed"]:
        raise RuntimeError("replay gate failed")
    ref = dense_reference((10, 20, 30, 40, 60, 80, 90), 8.0, 8)
    edges = all_edges((10, 20, 30, 40, 60, 80, 90), (2.0, 8.0, 32.0))
    replay_reference = [x for x in test["checks"] if x["kind"] == "reference"]
    max_seconds_per_step = max(x["suffix_seconds"] / (100 - x["src"] - 1)
                               for x in replay_reference)
    suffix_steps = sum(100 - e.src - 1 for e in edges if edge_intervention(e, ref) != ref)
    reference_seconds = read(RUN_ROOT / "replay/reference_probe.json")["reference_wall_seconds"]
    # Modal published on-demand rates, 2026-10-02; CPU/memory requests match GPU worker.
    rate = 0.000542 + 8 * 0.0000131 + 24 * 0.00000222
    projected_seconds_per_trajectory = suffix_steps * max_seconds_per_step + reference_seconds + 30
    projected_profile_20_usd = 20 * projected_seconds_per_trajectory * rate
    out = {"stage": "lw_cost_canary", "max_usd": max_usd,
           "rate_usd_per_second_L40S_cpu8_ram24GiB": rate,
           "pricing_source": "https://modal.com/pricing",
           "max_observed_reference_suffix_seconds_per_step": max_seconds_per_step,
           "additional_suffix_steps_per_trajectory": suffix_steps,
           "projected_seconds_per_trajectory": projected_seconds_per_trajectory,
           "projected_profile_20_usd": projected_profile_20_usd,
           "audit_cost_excluded_and_reported_separately": True,
           "allowed": projected_profile_20_usd <= max_usd,
           "projection_type": "conservative canary extrapolation, not actual invoice"}
    write_atomic(RUN_ROOT / "profiling/cost_projection.json", out)
    result.commit()
    if not out["allowed"]:
        raise RuntimeError(f"projected profiling ${projected_profile_20_usd:.2f} exceeds cap ${max_usd:.2f}; override STATIC_BFS_MODAL_MAX_USD explicitly")
    return out


@app.local_entrypoint()
def cost_canary():
    cap = float(os.environ.get("STATIC_BFS_MODAL_MAX_USD", "50"))
    print("LW_COST_CANARY", json.dumps(lw_cost_canary.remote(cap), sort_keys=True))


@app.function(image=image, gpu="L40S", volumes={"/vol": cache, "/result": result},
              timeout=7200, cpu=8, memory=24576, max_containers=8)
def lw_profile_trajectory(rank: int):
    import time
    import torch
    from modal_n8_gpu.prefix_replay import reference_with_snapshots, replay_suffix
    from scripts.static_bfs_graph.graph_builder import all_edges, dense_reference, edge_intervention
    from scripts.static_bfs_graph.runner import read, write_atomic

    if rank not in range(20):
        raise ValueError("calibration rank must be 0..19")
    cache.reload()
    result.reload()
    test = read(RUN_ROOT / "replay/equivalence_results.json")
    if not test["passed"] or len(test["checks"]) != 7:
        raise RuntimeError("replay gate did not pass")
    projection = read(RUN_ROOT / "profiling/cost_projection.json")
    if not projection["allowed"]:
        raise RuntimeError("cost gate did not pass")
    trajectory = read(RUN_ROOT / "calibration_trajectories.json")[rank]
    p, seed = trajectory["prompt_id"], trajectory["seed"]
    Runtime, cfg, old, prompts = _context()
    start = time.perf_counter()
    runtime = Runtime(RUN_ROOT, Path("/opt/official"), seed, cfg, old, prompts)
    snapshot_path = RUN_ROOT / f"profiling/snapshots/trajectory_{rank:02d}.pt"
    reference_path = RUN_ROOT / f"profiling/raw/trajectory_{rank:02d}/reference.json"
    if snapshot_path.exists() != reference_path.exists():
        partial = snapshot_path if snapshot_path.exists() else reference_path
        partial.unlink()
        print("RECOVERED_PARTIAL_REFERENCE", rank, flush=True)
    refrow = reference_with_snapshots(
        runtime, prompt_index=p, trial_seed=seed,
        snapshot_path=snapshot_path, record_path=reference_path,
    )
    checkpoint = torch.load(snapshot_path, map_location="cpu", weights_only=False)
    ref = dense_reference((10, 20, 30, 40, 60, 80, 90), 8.0, 8)
    edges = all_edges((10, 20, 30, 40, 60, 80, 90), (2.0, 8.0, 32.0))
    if len(edges) != 92:
        raise RuntimeError("edge grid changed")
    done = new = copied = 0
    suffix_seconds = 0.0
    # all_edges is sorted by (src,dst,tau), so prefix states stay grouped.
    for edge in edges:
        dest = RUN_ROOT / f"profiling/raw/trajectory_{rank:02d}/{edge.key}.json"
        policy = edge_intervention(edge, ref)
        if dest.exists():
            oldrow = read(dest)
            if (oldrow["edge_key"] != edge.key or oldrow["policy_id"] != policy.id or
                    oldrow["prompt_index"] != p or oldrow["trial_seed"] != seed or
                    oldrow["backend"] != "CUDA_Modal"):
                raise RuntimeError(f"invalid resumable edge {dest}")
            done += 1
            continue
        if policy == ref:
            row = {
                **refrow, "stage": "edge_replay", "edge_key": edge.key,
                "edge": edge.record(), "policy_id": policy.id,
                "src": edge.src, "reference_score": refrow["selected_final_score"],
                "delta": 0.0, "reused_identical_reference": True,
                "suffix_diffusion_steps": 0, "additional_NFE": 0,
                "edge_replay_wall_seconds": 0.0,
            }
            copied += 1
        else:
            row = replay_suffix(runtime, checkpoint=checkpoint, src=edge.src, policy=policy)
            row.update({"edge_key": edge.key, "edge": edge.record(),
                        "reference_score": refrow["selected_final_score"],
                        "delta": refrow["selected_final_score"] - row["selected_final_score"],
                        "reused_identical_reference": False,
                        "additional_NFE": row["diffusion_NFE"]})
            suffix_seconds += row["edge_replay_wall_seconds"]
            new += 1
        write_atomic(dest, row)
        done += 1
        if done % 5 == 0:
            result.commit()
    wall = time.perf_counter() - start
    summary = {"rank": rank, "prompt_index": p, "trial_seed": seed,
               "GPU_type": torch.cuda.get_device_name(0), "edges": done,
               "new_edge_rollouts": new, "exact_reference_copies": copied,
               "reference_wall_seconds": refrow["reference_wall_seconds"],
               "snapshot_creation_seconds": refrow["snapshot_creation_seconds"],
               "total_edge_replay_wall_seconds": suffix_seconds,
               "mean_edge_replay_seconds": suffix_seconds / new if new else 0,
               "pipeline_load_seconds": runtime.load_seconds,
               "trajectory_wall_seconds": wall,
               "calibration_selection_hash": trajectory["selection_hash"]}
    write_atomic(RUN_ROOT / f"profiling/timing/trajectory_{rank:02d}.json", summary)
    result.commit()
    print("LW_PROFILE_TRAJECTORY_DONE", json.dumps(summary, sort_keys=True), flush=True)
    return summary


@app.local_entrypoint()
def profile_one():
    print("LW_PROFILE_1", json.dumps(lw_profile_trajectory.remote(0), sort_keys=True))


@app.function(image=image, volumes={"/result": result}, timeout=120, cpu=1, memory=1024)
def freeze_stop_rule():
    from modal_n8_gpu.lightweight_plan import write_once, digest
    result.reload()
    path = RUN_ROOT / "engineering_stop_rule.json"
    write_once(path, {
        "protocol": "STATIC_BFS_N8_LIGHTWEIGHT_V1",
        "skip_n20_only_if_all": [
            "P10_policy_id_equals_P5_policy_id",
            "n10_prompt_bootstrap_exact_path_frequency_at_least_0.80",
            "Spearman_edge_rank_G10_vs_G5_at_least_0.95",
        ],
        "if_fired": "record n20 unmeasured and audit frozen P10; do not claim G20",
        "otherwise": "continue to n20 if cost gate permits",
        "preregistered_before_graph_analysis": True,
    })
    result.commit()
    return {"stop_rule_sha256": digest(path)}


@app.local_entrypoint()
def lock_stop_rule():
    print("STOP_RULE", json.dumps(freeze_stop_rule.remote(), sort_keys=True))


@app.function(image=image, volumes={"/result": result}, timeout=900, cpu=2, memory=4096)
def lw_graph_analysis(max_n: int):
    from modal_n8_gpu.lightweight_analysis import analyze
    result.reload()
    output = analyze(RUN_ROOT, max_n)
    result.commit()
    return output


@app.local_entrypoint()
def graph_one(max_n: int):
    print("LW_GRAPH_ANALYSIS", json.dumps(lw_graph_analysis.remote(max_n), sort_keys=True))


@app.function(image=image, volumes={"/result": result}, timeout=300, cpu=1, memory=2048)
def lw_actual_cost_gate(max_n: int, max_usd: float = 50.0):
    from scripts.static_bfs_graph.runner import read, write_atomic
    result.reload()
    if max_n not in (1, 5, 10):
        raise ValueError("gate runs after n=1, 5, or 10")
    timings = [read(RUN_ROOT / f"profiling/timing/trajectory_{i:02d}.json") for i in range(max_n)]
    rate = 0.000542 + 8 * 0.0000131 + 24 * 0.00000222
    actual_seconds = sum(row["trajectory_wall_seconds"] for row in timings)
    predicted_20 = actual_seconds / max_n * 20 * rate
    prior_projection = read(RUN_ROOT / "profiling/cost_projection.json")
    out = {"stage": f"actual_cost_gate_after_n{max_n}", "max_usd": max_usd,
           "measured_n": max_n, "measured_gpu_seconds": actual_seconds,
           "measured_estimated_usd": actual_seconds * rate,
           "projected_profile_20_usd_from_actual_mean": predicted_20,
           "prior_canary_projection_usd": prior_projection["projected_profile_20_usd"],
           "rate_usd_per_second": rate, "pricing_source": "https://modal.com/pricing",
           "audit_cost_excluded": True, "allowed": predicted_20 <= max_usd}
    write_atomic(RUN_ROOT / f"profiling/cost_gate_after_n{max_n}.json", out)
    result.commit()
    if not out["allowed"]:
        raise RuntimeError(f"Projected profiling ${predicted_20:.2f} exceeds cap ${max_usd:.2f}")
    return out


@app.local_entrypoint()
def cost_update(max_n: int):
    cap = float(os.environ.get("STATIC_BFS_MODAL_MAX_USD", "50"))
    print("LW_ACTUAL_COST_GATE", json.dumps(lw_actual_cost_gate.remote(max_n, cap), sort_keys=True))
