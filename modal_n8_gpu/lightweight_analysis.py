"""Offline first-order graph search for nested 1/5/10/20 trajectory sets."""
from __future__ import annotations

from collections import Counter
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

from scripts.static_bfs_graph.graph_builder import all_edges, all_paths, dense_reference, edge_intervention, policy_from_path
from scripts.static_bfs_graph.runner import read, write_atomic

from .lightweight_plan import STEPS, TAUS, SIZES


def _csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _complete_matrix(root: Path, max_n: int):
    calibration = read(root / "calibration_trajectories.json")
    edges = all_edges(STEPS, TAUS)
    ref = dense_reference(STEPS, 8.0, 8)
    scores = np.empty((max_n, len(edges)), dtype=float)
    timing = []
    for rank, traj in enumerate(calibration[:max_n]):
        p, seed = traj["prompt_id"], traj["seed"]
        folder = root / f"profiling/raw/trajectory_{rank:02d}"
        actual = list(folder.glob("*.json"))
        if len(actual) != 93:
            raise RuntimeError(f"trajectory {rank} has {len(actual)}/93 raw records")
        rr = read(folder / "reference.json")
        if (rr["backend"] != "CUDA_Modal" or rr["prompt_index"] != p or
                rr["trial_seed"] != seed or rr["policy_id"] != ref.id or
                rr["diffusion_NFE"] != 800 or len(rr["final_particle_scores"]) != 8):
            raise RuntimeError(f"bad reference trajectory {rank}")
        for ei, edge in enumerate(edges):
            row = read(folder / f"{edge.key}.json")
            policy = edge_intervention(edge, ref)
            if (row["backend"] != "CUDA_Modal" or row["prompt_index"] != p or
                    row["trial_seed"] != seed or row["edge_key"] != edge.key or
                    row["policy_id"] != policy.id or len(row["final_particle_scores"]) != 8 or
                    not math.isclose(row["reference_score"], rr["selected_final_score"], abs_tol=1e-10) or
                    not math.isclose(row["delta"], rr["selected_final_score"] - row["selected_final_score"], abs_tol=1e-10) or
                    [e["sampling_index"] for e in row["resampling_events"]] != list(policy.resampling_steps)):
                raise RuntimeError(f"bad edge trajectory {rank} {edge.key}")
            scores[rank, ei] = row["delta"]
        tm = read(root / f"profiling/timing/trajectory_{rank:02d}.json")
        if tm["edges"] != 92 or tm["prompt_index"] != p or tm["trial_seed"] != seed:
            raise RuntimeError(f"bad timing trajectory {rank}")
        timing.append(tm)
    if not np.isfinite(scores).all():
        raise RuntimeError("nonfinite edge delta")
    return edges, scores, timing


def _best_tau(values: np.ndarray, edges) -> dict:
    groups: dict[tuple[int, int], list[int]] = {}
    for i, edge in enumerate(edges):
        if edge.tau_dst is not None:
            groups.setdefault((edge.src, edge.dst), []).append(i)
    return {key: edges[min(indices, key=lambda i: (values[i], edges[i].tau_dst))].tau_dst
            for key, indices in groups.items()}


def analyze(root: Path, max_n: int):
    if max_n not in SIZES:
        raise ValueError("max_n must be 1,5,10,20")
    manifest = read(root / "experiment_manifest.json")
    replay = read(root / "replay/equivalence_results.json")
    if manifest["protocol"] != "STATIC_BFS_N8_LIGHTWEIGHT_V1" or not replay["passed"]:
        raise RuntimeError("preflight/replay gate invalid")
    if manifest["PSP_in_scope"] or manifest["old_TPU_scientific_results_mixed"]:
        raise RuntimeError("mixed science inputs")
    edges, matrix, timings = _complete_matrix(root, max_n)
    paths = sorted(all_paths(STEPS, TAUS, 3), key=lambda path: policy_from_path(path, 8).id)
    policies = [policy_from_path(path, 8) for path in paths]
    edge_index = {e.key: i for i, e in enumerate(edges)}
    path_indices = np.asarray([[edge_index[e.key] for e in path] for path in paths], dtype=int)
    if len(paths) != 945:
        raise RuntimeError("K=3 path count changed")
    out = []
    stability = []
    means = {}
    for n in (x for x in SIZES if x <= max_n):
        v = matrix[:n].mean(axis=0)
        means[n] = v
        selected = int(np.argmin(v[path_indices].sum(axis=1)))
        policy = policies[selected]
        chosen_path = paths[selected]
        out.append({"n": n, "policy_id": policy.id,
                    "steps": list(policy.resampling_steps),
                    "taus": [t for _, t in policy.temperature_by_step],
                    "edge_ids": [e.key for e in chosen_path],
                    "predicted_regret": float(v[path_indices[selected]].sum())})
        _csv(root / f"graphs/edge_values_n{n}.csv", [
            {"edge_key": e.key, "src": e.src, "dst": e.dst,
             "tau_dst": e.tau_dst, "mean_delta": float(v[i]),
             "sample_std": float(matrix[:n, i].std(ddof=1)) if n > 1 else math.nan}
            for i, e in enumerate(edges)
        ])
        if n == 1:
            stability.append({"n": n, "policy_id": policy.id,
                              "exact_path_frequency": math.nan,
                              "top2_cumulative_frequency": math.nan,
                              "bootstrap_status": "degenerate_n1",
                              "timestep_marginals": json.dumps({}),
                              "tau_marginals": json.dumps({})})
            continue
        rng = np.random.default_rng(20261022 + n)
        picks = []
        for sample in rng.integers(0, n, size=(2000, n)):
            weight = np.bincount(sample, minlength=n) / n
            vv = weight @ matrix[:n]
            picks.append(int(np.argmin(vv[path_indices].sum(axis=1))))
        counts = Counter(picks)
        timestep_marginals = {str(step): sum(step in policies[i].resampling_steps for i in picks) / 2000
                              for step in STEPS}
        tau_marginals = {str(step): {str(tau): sum(policies[i].temperature_map.get(step) == tau
                                                   for i in picks) / 2000 for tau in TAUS}
                         for step in STEPS}
        stability.append({"n": n, "policy_id": policy.id,
                          "exact_path_frequency": counts[selected] / 2000,
                          "top2_cumulative_frequency": sum(c for _, c in counts.most_common(2)) / 2000,
                          "bootstrap_status": "2000_prompt_seed_trajectory_resamples",
                          "timestep_marginals": json.dumps(timestep_marginals, sort_keys=True),
                          "tau_marginals": json.dumps(tau_marginals, sort_keys=True)})
    _csv(root / "graphs/bootstrap_stability.csv", stability)
    write_atomic(root / "graphs/policies_by_n.json", out)
    comparisons = []
    for n in (x for x in SIZES if x <= max_n and x != max_n):
        a, b = means[n], means[max_n]
        best_a, best_b = _best_tau(a, edges), _best_tau(b, edges)
        comparisons.append({"n": n, "reference_n": max_n,
                            "spearman_edge_rank": float(spearmanr(a, b).statistic),
                            "pearson_edge_weight": float(pearsonr(a, b).statistic),
                            "mae": float(np.mean(np.abs(a - b))),
                            "sign_agreement": float(np.mean(np.sign(a) == np.sign(b))),
                            "best_tau_agreement": float(np.mean([best_a[k] == best_b[k] for k in best_a]))})
    if comparisons:
        _csv(root / "analysis/edge_convergence_interim.csv", comparisons)
    policy_comparisons = []
    last = out[-1]
    for row in out:
        a = dict(zip(row["steps"], row["taus"]))
        b = dict(zip(last["steps"], last["taus"]))
        union = set(a) | set(b)
        overlap = set(a) & set(b)
        policy_comparisons.append({"n": row["n"], "reference_n": max_n,
                                   "exact_policy_match": row["policy_id"] == last["policy_id"],
                                   "timestep_jaccard": len(overlap) / len(union),
                                   "tau_agreement_at_shared_steps":
                                       sum(a[s] == b[s] for s in overlap) / len(overlap) if overlap else math.nan})
    _csv(root / "analysis/policy_convergence_interim.csv", policy_comparisons)
    rate = 0.000542 + 8 * 0.0000131 + 24 * 0.00000222
    cost_rows = []
    for n in (x for x in SIZES if x <= max_n):
        seconds = sum(x["trajectory_wall_seconds"] for x in timings[:n])
        cost_rows.append({"n": n, "summed_gpu_seconds": seconds,
                          "gpu_hours": seconds / 3600,
                          "estimated_modal_usd": seconds * rate,
                          "rate_source": "https://modal.com/pricing",
                          "audit_excluded": True})
    _csv(root / "analysis/compilation_cost.csv", cost_rows)
    stop_fired = False
    if max_n >= 10:
        p5 = next(x for x in out if x["n"] == 5)
        p10 = next(x for x in out if x["n"] == 10)
        b10 = next(x for x in stability if x["n"] == 10)
        rho = float(spearmanr(means[5], means[10]).statistic)
        stop_fired = (p5["policy_id"] == p10["policy_id"] and
                      b10["exact_path_frequency"] >= 0.80 and rho >= 0.95)
        write_atomic(root / "analysis/progressive_stop_decision.json", {
            "evaluated_after_n": 10, "stop_rule_fired": stop_fired,
            "P10_equals_P5": p5["policy_id"] == p10["policy_id"],
            "n10_exact_path_frequency": b10["exact_path_frequency"],
            "Spearman_G10_G5": rho,
            "n20_measured": max_n >= 20,
        })
    return {"max_n": max_n, "policies": out, "stability": stability,
            "cost": cost_rows, "edge_comparisons": comparisons,
            "stop_rule_fired": stop_fired}
