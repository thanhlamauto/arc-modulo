"""Offline AER audit, ablations, cost accounting, and integrity checks."""
from __future__ import annotations

from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np

from scripts.static_bfs_graph.graph_builder import all_paths, path_edges, policy_from_path

from .aer_plan import (AUDIT_BOOTSTRAPS, COARSE_N, MAX_CANDIDATES,
                       PROPOSAL_BOOTSTRAPS, PROTOCOL, RATE_USD_PER_SECOND,
                       STEPS, TAUS, digest, read, select_candidates)


def _csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def audit_aggregate(root: Path):
    manifest = read(root / "experiment_manifest.json")
    compiled_path = root / "racing/aer_compiled_policy.json"
    compiled = read(compiled_path)
    audit_plan = read(root / "audit/audit_plan.json")
    frozen = read(root / "audit/frozen_policy_plan.json")
    race = read(root / "racing/racing_trajectory_plan.json")
    stage_a = read(root / "stage_a/coarse_trajectories.json")
    if (manifest["protocol"] != PROTOCOL or compiled["protocol"] != PROTOCOL or
        manifest["audit_plan_sha256"] != digest(root / "audit/audit_plan.json") or
        frozen["aer_compiled_policy_sha256"] != digest(compiled_path) or
        frozen["audit_plan_sha256"] != digest(root / "audit/audit_plan.json") or
        len(frozen["pairs"]) != 20 or len(race["trajectories"]) != 20 or
        len(stage_a) != COARSE_N or
        len({x["prompt_id"] for x in stage_a + race["trajectories"] + frozen["pairs"]}) != 45):
        raise RuntimeError("frozen plan or prompt disjointness invalid")
    aliases = frozen["aliases"]
    if set(aliases) != set(audit_plan["policy_aliases"]) or len(aliases) != len(audit_plan["policy_aliases"]):
        raise RuntimeError("audit alias set changed")
    values = {}
    raw_rows = []
    audit_timings = []
    for rank, pair in enumerate(frozen["pairs"]):
        timing = read(root / f"audit/timing/trajectory_{rank:02d}.json")
        if (timing["prompt_id"] != pair["prompt_id"] or timing["seed"] != pair["seed"] or
            timing["unique_policy_count"] != len(frozen["unique_policies"])):
            raise RuntimeError(f"audit timing mismatch {rank}")
        audit_timings.append(timing)
    for policy in frozen["unique_policies"]:
        pid = policy["policy_id"]
        scores = []
        for rank, pair in enumerate(frozen["pairs"]):
            path = root / "audit/raw" / pid / f"p{pair['prompt_id']}_s{pair['seed']}.json"
            row = read(path)
            if (row["policy_id"] != pid or row["prompt_index"] != pair["prompt_id"] or
                row["trial_seed"] != pair["seed"] or row["backend"] != "CUDA_Modal" or
                row["diffusion_NFE"] != 800 or
                len(row["final_particle_scores"]) != 8 or
                not np.isfinite(row["selected_final_score"])):
                raise RuntimeError(f"audit raw record invalid: {rank}:{pid}")
            scores.append(float(row["selected_final_score"]))
            raw_rows.append({"rank": rank, "prompt_id": pair["prompt_id"],
                             "seed": pair["seed"], "policy_id": pid,
                             "ImageReward": row["selected_final_score"],
                             "elapsed_seconds": row["elapsed_seconds"]})
        values[pid] = np.asarray(scores)
    if len(raw_rows) != 20 * len(frozen["unique_policies"]):
        raise RuntimeError("audit row count invalid")
    policy_rows = [{"alias": alias, "policy_id": pid,
                    "mean_ImageReward": float(values[pid].mean()),
                    "std_ImageReward": float(values[pid].std(ddof=1)),
                    "prompt_count": 20, "seed_per_prompt": 1,
                    "is_distinct_policy": alias == next(x["first_alias"] for x in
                                                      frozen["unique_policies"] if x["policy_id"] == pid)}
                   for alias, pid in aliases.items()]
    rng = np.random.default_rng(20261003)
    picks = rng.integers(0, 20, size=(AUDIT_BOOTSTRAPS, 20))
    comparisons = [("AER_WINNER", other) for other in
                   ("COARSE_TOP1", "G20_OLD_TOP1", "TRANSFER4_TO_8", "MANUAL8")]
    comparisons += [("COARSE_TOP1", "TRANSFER4_TO_8"),
                    ("G20_OLD_TOP1", "TRANSFER4_TO_8")]
    paired = []
    for a, b in comparisons:
        diffs = values[aliases[a]] - values[aliases[b]]
        boot = diffs[picks].mean(axis=1)
        lo, hi = np.quantile(boot, [.025, .975])
        paired.append({"method": a, "baseline": b,
                       "mean_delta_ImageReward": float(diffs.mean()),
                       "ci95_low": float(lo), "ci95_high": float(hi),
                       "prompt_wins": int((diffs > 1e-12).sum()),
                       "prompt_ties": int((np.abs(diffs) <= 1e-12).sum()),
                       "prompt_losses": int((diffs < -1e-12).sum()),
                       "bootstrap_resamples": AUDIT_BOOTSTRAPS,
                       "statistical_unit": "TRAIN prompt; one shared seed per prompt"})
    _csv(root / "audit/policy_results.csv", policy_rows)
    _csv(root / "audit/paired_results.csv", paired)
    _csv(root / "audit/prompt_policy_rows.csv", raw_rows)
    summary = {"protocol": PROTOCOL,
               "frozen_policy_plan_sha256": digest(root / "audit/frozen_policy_plan.json"),
               "AER_compiled_policy_sha256": digest(compiled_path),
               "unique_policy_count": len(frozen["unique_policies"]),
               "raw_policy_rows": len(raw_rows), "audit_prompt_count": 20,
               "policy_results": policy_rows, "paired_results": paired,
               "summed_GPU_seconds": sum(row["trajectory_wall_seconds"] for row in audit_timings),
               "estimated_Modal_USD": sum(row["trajectory_wall_seconds"] for row in audit_timings)*RATE_USD_PER_SECOND,
               "VALIDATION_touched": False, "TEST_touched": False,
               "audit_not_used_to_select_AER": True}
    (root / "audit/summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True)+"\n")
    return summary


def analyze_ablations_cost(root: Path, prior: Path):
    compiled = read(root / "racing/aer_compiled_policy.json")
    n = compiled["racing_trajectories_used"]
    candidate = read(root / "stage_a/candidate_set.json")
    incumbent = candidate["incumbent_policy_id"]
    budget_rows = []
    for budget in (5, 10, 15, 20):
        if budget <= n:
            state = read(root / f"racing/round_{budget//5}_state.json")
            if state["stop"]:
                selected = state["selected_policy_id"]
                status = state["selection_status"]
            else:
                survivors = state["survivor_policy_ids"]
                selected = incumbent if incumbent in survivors else min(
                    survivors, key=lambda pid: (
                        state["survivor_mean_discriminating_cost"][pid], pid))
                status = "underidentified"
            budget_rows.append({"max_budget": budget, "observed_trajectories": budget,
                                "survivors": json.dumps(state["survivor_policy_ids"]),
                                "survivor_count": len(state["survivor_policy_ids"]),
                                "selected_policy_id": selected, "selection_status": status,
                                "data_status": "observed_prefix"})
        else:
            budget_rows.append({"max_budget": budget, "observed_trajectories": n,
                                "survivors": json.dumps(compiled["final_survivors"]),
                                "survivor_count": len(compiled["final_survivors"]),
                                "selected_policy_id": compiled["selected_policy"]["policy_id"],
                                "selection_status": compiled["selection_status"],
                                "data_status": "not_observed_after_early_stop"})
    _csv(root / "analysis/budget_ablation.csv", budget_rows)

    paths = sorted(all_paths(STEPS, TAUS, 3), key=lambda p: policy_from_path(p, 8).id)
    policies = [policy_from_path(path, 8) for path in paths]
    index = {policy.id: i for i, policy in enumerate(policies)}
    counts = Counter(index[row["policy_id"]] for row in
                     _read_csv(root / "stage_a/bootstrap_paths.csv"))
    top1_idx = index[candidate["coarse_top1_policy_id"]]
    racing_times = [read(root / f"racing/raw/trajectory_{i:02d}/timing.json")
                    for i in range(n)]
    primary_d = read(root / "racing/discriminating_edges.json")["discriminating_edge_count"]
    fixed = float(np.mean([row["trajectory_wall_seconds"] - row["edge_profile_seconds"]
                           for row in racing_times]))
    edge_cost = float(np.mean([row["edge_profile_seconds"] for row in racing_times]))
    stage_a_cost = float(read(root / "experiment_manifest.json")["stage_a_estimated_usd"])
    mass_rows = []
    from scripts.static_bfs_graph.policy import StaticBFSPolicy
    transfer = StaticBFSPolicy((40, 60, 90), ((40, 8.0), (60, 8.0), (90, 32.0)), 8)
    transfer_edges = set(e.key for e in path_edges((40, 60, 90), (8.0, 8.0, 32.0)))
    for mass in (.70, .80, .90, .95):
        chosen, covered = select_candidates(counts, paths, policies, top1_idx, mass)
        edge_sets = [set(edge.key for edge in paths[i]) for i in chosen]
        if transfer.id not in {policies[i].id for i in chosen}:
            edge_sets.append(transfer_edges)
        d_count = len(set.union(*edge_sets) - set.intersection(*edge_sets))
        projected_racing_seconds = 20 * (fixed + edge_cost * d_count / primary_d)
        mass_rows.append({"mass_target": mass, "actual_mass_covered": covered,
                          "graph_candidates": len(chosen),
                          "candidates_with_incumbent": len(edge_sets),
                          "discriminating_edges": d_count,
                          "proposal_uncertainty_high": covered < mass,
                          "estimated_max20_racing_GPU_seconds": projected_racing_seconds,
                          "estimated_max20_total_compiler_USD": stage_a_cost +
                              projected_racing_seconds * RATE_USD_PER_SECOND,
                          "estimate_note": "fixed reference/load plus edge-profile time scaled by |D|; shared-tree geometry not modeled"})
    _csv(root / "analysis/candidate_mass_ablation.csv", mass_rows)
    uniform = next(row for row in _read_csv(prior / "analysis/compilation_cost.csv")
                   if int(row["n"]) == 20)
    stage_b_seconds = sum(row["trajectory_wall_seconds"] for row in racing_times)
    stage_a_seconds = float(next(row["summed_gpu_seconds"]
                                 for row in _read_csv(prior / "analysis/compilation_cost.csv")
                                 if int(row["n"]) == 5))
    total_seconds = stage_a_seconds + stage_b_seconds
    uniform_seconds = float(uniform["summed_gpu_seconds"])
    cost = {"stage_a_reused_historical_GPU_seconds": stage_a_seconds,
            "stage_a_reused_historical_GPU_hours": stage_a_seconds/3600,
            "stage_a_reused_estimated_Modal_USD": stage_a_seconds*RATE_USD_PER_SECOND,
            "stage_b_new_GPU_seconds": stage_b_seconds,
            "stage_b_new_GPU_hours": stage_b_seconds/3600,
            "stage_b_new_estimated_Modal_USD": stage_b_seconds*RATE_USD_PER_SECOND,
            "AER_total_GPU_seconds": total_seconds,
            "AER_total_GPU_hours": total_seconds/3600,
            "AER_total_estimated_Modal_USD": total_seconds*RATE_USD_PER_SECOND,
            "uniform_G20_GPU_seconds": uniform_seconds,
            "uniform_G20_GPU_hours": uniform_seconds/3600,
            "uniform_G20_estimated_Modal_USD": uniform_seconds*RATE_USD_PER_SECOND,
            "cost_ratio_AER_over_G20": total_seconds/uniform_seconds,
            "incremental_new_AER_spend_excludes_reused_stage_a": True,
            "stage_a_logical_edge_observations": 5*92,
            "stage_b_logical_edge_observations": n*primary_d,
            "AER_total_logical_edge_observations": 5*92+n*primary_d,
            "uniform_G20_logical_edge_observations": 20*92,
            "logical_edge_ratio_AER_over_G20": (5*92+n*primary_d)/(20*92),
            "racing_trajectory_count": n,
            "racing_discriminating_edges": primary_d,
            "racing_GPU_wall_span_seconds": None if any(
                "wall_start_unix" not in row for row in racing_times) else
                max(row["wall_end_unix"] for row in racing_times)-
                min(row["wall_start_unix"] for row in racing_times),
            "cost_estimate_not_invoice": True,
            "audit_cost_separate": True}
    (root / "analysis/cost_summary.json").write_text(json.dumps(cost, indent=2, sort_keys=True)+"\n")
    return cost


def integrity(root: Path):
    manifest = read(root / "experiment_manifest.json")
    candidate = read(root / "stage_a/candidate_set.json")
    compiled = read(root / "racing/aer_compiled_policy.json")
    frozen = read(root / "audit/frozen_policy_plan.json")
    audit = read(root / "audit/summary.json")
    cost = read(root / "analysis/cost_summary.json")
    n = compiled["racing_trajectories_used"]
    d = read(root / "racing/discriminating_edges.json")["discriminating_edge_count"]
    passed = (manifest["protocol"] == PROTOCOL and
              candidate["coarse_n"] == COARSE_N and
              candidate["proposal_bootstraps"] == PROPOSAL_BOOTSTRAPS and
              len(candidate["candidates"]) <= MAX_CANDIDATES+1 and
              compiled["candidate_set_sha256"] == digest(root / "stage_a/candidate_set.json") and
              frozen["aer_compiled_policy_sha256"] == digest(root / "racing/aer_compiled_policy.json") and
              audit["raw_policy_rows"] == 20*len(frozen["unique_policies"]) and
              cost["stage_b_logical_edge_observations"] == n*d and
              not manifest["VALIDATION_touched"] and not manifest["TEST_touched"] and
              not manifest["external_100_prompt_pool_touched"] and
              not compiled["audit_scoring_used"])
    if not passed:
        raise RuntimeError("AER final integrity failed")
    output = {"passed": True, "protocol": PROTOCOL,
              "experiment_manifest_sha256": digest(root / "experiment_manifest.json"),
              "candidate_set_sha256": digest(root / "stage_a/candidate_set.json"),
              "discriminating_edges_sha256": digest(root / "racing/discriminating_edges.json"),
              "racing_plan_sha256": digest(root / "racing/racing_trajectory_plan.json"),
              "aer_compiled_policy_sha256": digest(root / "racing/aer_compiled_policy.json"),
              "audit_plan_sha256": digest(root / "audit/audit_plan.json"),
              "frozen_audit_policy_plan_sha256": digest(root / "audit/frozen_policy_plan.json"),
              "stage_a_trajectories": 5, "stage_a_edges": 460,
              "racing_trajectories": n, "racing_edges": n*d,
              "audit_prompts": 20, "audit_unique_policies": len(frozen["unique_policies"]),
              "audit_raw_rows": audit["raw_policy_rows"],
              "TRAIN_only": True, "VALIDATION_touched": False, "TEST_touched": False,
              "no_policy_search_on_audit": True}
    (root / "integrity.json").write_text(json.dumps(output, indent=2, sort_keys=True)+"\n")
    return output


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("prior", type=Path)
    args = parser.parse_args()
    a = audit_aggregate(args.run)
    c = analyze_ablations_cost(args.run, args.prior)
    i = integrity(args.run)
    print(json.dumps({"audit": a, "cost": c, "integrity": i}, indent=2,
                     sort_keys=True))
