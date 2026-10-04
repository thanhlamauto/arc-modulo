"""Strict GPU-only integrity and prompt-bootstrap analysis for N8 graph calibration."""
from __future__ import annotations

from collections import Counter
import csv
import json
import math
from pathlib import Path

import numpy as np

from scripts.static_bfs_graph.adapter import RNG_PROTOCOL, event_seed
from scripts.static_bfs_graph.graph_builder import all_edges, all_paths, dense_reference, edge_intervention, policy_from_path
from scripts.static_bfs_graph.n8_plan import n8_config
from scripts.static_bfs_graph.n8_run import dest_for
from scripts.static_bfs_graph.runner import file_sha, read, static_record, write_atomic


def _csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(root: Path):
    plan = read(root / "graph_plan.json")
    manifest = read(root / "experiment_manifest.json")
    if file_sha(root / "graph_plan.json") != manifest["graph_plan_sha256"]:
        raise RuntimeError("graph plan SHA mismatch")
    if plan["backend"] != "CUDA_Modal" or not plan["GPU_data_only"]:
        raise RuntimeError("backend/provenance mismatch")
    cfg = n8_config()
    grid = cfg["pilot"]
    steps = tuple(grid["candidate_steps"])
    taus = tuple(map(float, grid["taus"]))
    edges = all_edges(steps, taus)
    ref = dense_reference(steps, float(grid["reference_tau"]), 8)
    if [e.key for e in edges] != plan["edge_keys"] or len(edges) != 92:
        raise RuntimeError("edge catalogue changed")
    fit = plan["fit_order_60"]
    seeds = plan["seeds"]
    if len(fit) != 60 or len(set(fit)) != 60 or seeds != [42, 43, 44, 45]:
        raise RuntimeError("TRAIN units changed")
    for stage, expected in (("n8_reference", 240), ("n8_edges", 22080)):
        actual = len(list((root / "raw" / stage / "n8").glob("seed*/*.json")))
        if actual != expected:
            raise RuntimeError(f"{stage}: {actual} physical records, expected {expected}")
    scores = np.empty((60, 4, 92), dtype=float)
    references = np.empty((60, 4), dtype=float)
    refrec = static_record(ref)
    copied = 0
    for pi, p in enumerate(fit):
        for si, seed in enumerate(seeds):
            rp, _, rh = dest_for(root, "n8_reference", cfg, ref, refrec, p, seed)
            rr = read(rp)
            if (rr["unit_hash"] != rh or rr["policy_id"] != ref.id or
                    rr["backend"] != "CUDA_Modal" or rr["diffusion_NFE"] != 800 or
                    rr["rng_protocol"] != RNG_PROTOCOL or len(rr["final_particle_scores"]) != 8 or
                    [x["sampling_index"] for x in rr["resampling_events"]] != list(ref.resampling_steps)):
                raise RuntimeError(f"invalid GPU reference p={p} seed={seed}")
            references[pi, si] = rr["selected_final_score"]
            for ei, edge in enumerate(edges):
                policy = edge_intervention(edge, ref)
                rec = static_record(policy)
                ep, _, eh = dest_for(root, "n8_edges", cfg, ref, rec, p, seed, edge)
                er = read(ep)
                if (er["unit_hash"] != eh or er["policy_id"] != policy.id or
                        er["backend"] != "CUDA_Modal" or er["diffusion_NFE"] != 800 or
                        er["initial_generator_state_sha256"] != rr["initial_generator_state_sha256"] or
                        er["rng_protocol"] != RNG_PROTOCOL or len(er["final_particle_scores"]) != 8 or
                        [x["sampling_index"] for x in er["resampling_events"]] != list(policy.resampling_steps) or
                        any(x["event_seed"] != event_seed(seed, p, x["sampling_index"])
                            for x in er["resampling_events"])):
                    raise RuntimeError(f"invalid GPU edge {edge.key} p={p} seed={seed}")
                if er.get("reused_identical_reference"):
                    if policy != ref or er["selected_final_score"] != rr["selected_final_score"]:
                        raise RuntimeError("invalid reference reuse")
                    copied += 1
                scores[pi, si, ei] = er["selected_final_score"]
    if not np.isfinite(scores).all() or not np.isfinite(references).all():
        raise RuntimeError("nonfinite reward")
    if copied != 1920:
        raise RuntimeError(f"expected 1920 exact-reference edge copies, got {copied}")
    failures = list((root / "failures").rglob("*.json")) if (root / "failures").exists() else []
    if failures:
        raise RuntimeError(f"failure records exist: {len(failures)}")
    matrix = (references[:, :, None] - scores).mean(axis=1)
    paths = sorted(all_paths(steps, taus, 3), key=lambda x: policy_from_path(x, 8).id)
    edge_idx = {e.key: i for i, e in enumerate(edges)}
    path_idx = np.asarray([[edge_idx[e.key] for e in path] for path in paths], dtype=int)
    policies = [policy_from_path(path, 8) for path in paths]
    if len(paths) != 945:
        raise RuntimeError("K3 path count changed")
    rows, boot_rows, edge_rows = [], [], []
    for n in plan["sizes"]:
        sub = matrix[:n]
        values = sub.mean(axis=0)
        choice = int(np.argmin(values[path_idx].sum(axis=1)))
        policy = policies[choice]
        seed = 20261022 if n == 12 else 20261022 + n
        rng = np.random.default_rng(seed)
        picks = []
        for sample in rng.integers(0, n, size=(2000, n)):
            weights = np.bincount(sample, minlength=n) / n
            v = weights @ sub
            picks.append(int(np.argmin(v[path_idx].sum(axis=1))))
        counts = Counter(picks)
        exact = counts[choice] / 2000
        p2 = sum(policies[i].temperature_map.get(40) == 2.0 for i in picks) / 2000
        p8 = sum(policies[i].temperature_map.get(40) == 8.0 for i in picks) / 2000
        p32 = sum(policies[i].temperature_map.get(40) == 32.0 for i in picks) / 2000
        include = p2 + p8 + p32
        rows.append({
            "n_fit": n, "policy_id": policy.id,
            "steps": json.dumps(list(policy.resampling_steps)),
            "taus": json.dumps([t for _, t in policy.temperature_by_step]),
            "predicted_additive_regret": float(values[path_idx[choice]].sum()),
            "exact_path_frequency": exact,
            "top2_cumulative_frequency": sum(c for _, c in counts.most_common(2)) / 2000,
            "unique_paths": len(counts), "P_tau40_2": p2, "P_tau40_8": p8,
            "P_tau40_32": p32, "P_step40_included": include,
            "P_tau40_2_given_included": p2 / include if include else math.nan,
            "P_tau40_8_given_included": p8 / include if include else math.nan,
            "bootstrap_seed": seed,
        })
        for i, count in counts.most_common():
            pol = policies[i]
            boot_rows.append({"n_fit": n, "policy_id": pol.id,
                              "steps": json.dumps(pol.resampling_steps),
                              "taus": json.dumps([t for _, t in pol.temperature_by_step]),
                              "count": count, "frequency": count / 2000,
                              "is_full_data_policy": i == choice})
        for ei, edge in enumerate(edges):
            edge_rows.append({"n_fit": n, "edge_key": edge.key, "src": edge.src,
                              "dst": edge.dst, "tau_dst": edge.tau_dst,
                              "mean_regret": float(values[ei]),
                              "prompt_std": float(sub[:, ei].std(ddof=1))})
    _csv(root / "analysis/policy_by_fit_size.csv", rows)
    _csv(root / "analysis/bootstrap_paths.csv", boot_rows)
    _csv(root / "analysis/edge_values_by_fit_size.csv", edge_rows)
    summary = {"protocol": plan["protocol"], "status": "complete", "backend": "CUDA_Modal",
               "graph_plan_sha256": manifest["graph_plan_sha256"], "N": 8, "K": 3,
               "reference_records": 240, "edge_records": 22080,
               "exact_reference_copies": copied, "new_edge_rollouts": 20160,
               "TPU_raw_reused": False, "failed_rollouts": 0,
               "validation_touched": False, "test_touched": False,
               "external_100_prompt_touched": False, "no_new_policy_reward_evaluation": True,
               "n12_reproduced_TPU": False, "rows": rows}
    write_atomic(root / "integrity.json", summary)
    lines = ["# N=8 graph sample efficiency — Modal GPU (TRAIN only)", "",
             "All 60 TRAIN prompts and four seeds were rerun on CUDA/BF16. No TPU raw record was mixed in. These results are exploratory because the 48 additional prompts were previously used for a policy diagnostic.", "",
             "| Fit prompts | Selected steps | Selected taus | Exact-path bootstrap | P(tau40=2) | P(tau40=8) |",
             "|---:|---|---|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['n_fit']} | {row['steps']} | {row['taus']} | {row['exact_path_frequency']:.1%} | {row['P_tau40_2']:.1%} | {row['P_tau40_8']:.1%} |")
    lines += ["", "Bootstrap unit: one TRAIN prompt with all four seeds grouped, 2,000 replicates per fit size.",
              "All 240 dense references and 22,080 edge records passed strict integrity checks; 1,920 edge records reuse identical GPU references.",
              "No VALIDATION, TEST, external 100-prompt pool, or new policy reward evaluation was used.", ""]
    (root / "report.md").write_text("\n".join(lines))
    return summary
