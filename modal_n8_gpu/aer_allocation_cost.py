"""Measured union GPU expense and conservative arm-specific compilation curves."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .aer_allocation_plan import RATE_USD_PER_SECOND
from .aer_allocation_racing import costs
from .aer_plan import read


def _csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build(root: Path):
    manifest = read(root / "experiment_manifest.json")
    arms = {}
    for arm in (5, 10):
        states = []
        for round_number in range(1, 7):
            path = root / f"racing/arm{arm}_round_{round_number}_state.json"
            if not path.exists():
                break
            states.append(read(path))
        if not states or not states[-1]["stop"]:
            raise RuntimeError(f"arm {arm} incomplete")
        arms[arm] = states
    measured_n = max(states[-1]["trajectories"] for states in arms.values())
    union = read(root / "edges/d_union.json")["discriminating_edges"]
    arm_keys = {arm: set(read(root / f"edges/d{arm}.json")["discriminating_edges"])
                for arm in (5, 10)}
    times = []
    for rank in range(measured_n):
        folder = root / f"racing/raw/trajectory_{rank:02d}"
        timing = read(folder / "timing.json")
        profile = read(folder / "edge_profile_timing.json")
        if timing["edge_count"] != len(union) or profile["logical_edges"] != len(union):
            raise RuntimeError("union profile incomplete")
        branch = {key: read(folder / "edges" / f"{key}.json")["branch_suffix_wall_seconds"]
                  for key in union}
        shared_overhead = max(0., timing["edge_profile_seconds"] - sum(branch.values()))
        fixed = timing["trajectory_wall_seconds"] - timing["edge_profile_seconds"] - timing["correctness_gate_seconds"]
        if fixed <= 0:
            raise RuntimeError("invalid racing GPU timing")
        times.append({"actual_union_seconds": timing["trajectory_wall_seconds"],
                      "fixed_seconds": fixed, "shared_overhead_union_seconds": shared_overhead,
                      "lower_arms": {arm: fixed + sum(branch[key] for key in arm_keys[arm])
                                     for arm in (5, 10)},
                      "arms": {arm: fixed + shared_overhead + sum(branch[key] for key in arm_keys[arm])
                               for arm in (5, 10)}})
    rows = []
    for arm in (5, 10):
        coarse_sec = float(manifest[f"g{arm}_historical_gpu_seconds"])
        coarse_usd = float(manifest[f"g{arm}_historical_estimated_usd"])
        d_count = len(arm_keys[arm])
        for state in arms[arm]:
            n = state["trajectories"]
            race_sec = sum(row["arms"][arm] for row in times[:n])
            race_lower_sec = sum(row["lower_arms"][arm] for row in times[:n])
            rows.append({"arm": arm, "coarse_trajectories": arm,
                         "coarse_candidates": len(read(root / f"proposal/g{arm}_candidates.json")["candidates"]),
                         "coarse_bootstrap_mass": read(root / f"proposal/g{arm}_candidates.json")["bootstrap_mass_covered"],
                         "discriminating_edges": d_count, "racing_trajectories": n,
                         "survivor_count": len(state["survivor_policy_ids"]),
                         "selection_status": state["selection_status"],
                         "selected_or_fallback_policy_id": state["selected_or_fallback_policy_id"],
                         "total_logical_edge_observations": arm*92+n*d_count,
                         "coarse_GPU_seconds": coarse_sec,
                         "targeted_GPU_seconds_upper_estimate": race_sec,
                         "targeted_GPU_seconds_lower_estimate": race_lower_sec,
                         "total_compiler_GPU_seconds_upper_estimate": coarse_sec+race_sec,
                         "total_compiler_GPU_seconds_lower_estimate": coarse_sec+race_lower_sec,
                         "coarse_USD": coarse_usd,
                         "targeted_USD_upper_estimate": race_sec*RATE_USD_PER_SECOND,
                         "targeted_USD_lower_estimate": race_lower_sec*RATE_USD_PER_SECOND,
                         "total_compiler_USD_upper_estimate": coarse_usd+race_sec*RATE_USD_PER_SECOND,
                         "total_compiler_USD_lower_estimate": coarse_usd+race_lower_sec*RATE_USD_PER_SECOND,
                         "arm_subset_timing": "conservative upper estimate using measured union shared overhead plus measured arm branch times"})
    _csv(root / "budget/compiler_cost_curve.csv", rows)
    chosen_budgets = sorted(set([4., 5., 6., 7., 8., 10.] + [
        float(row["total_compiler_USD_upper_estimate"]) for row in rows]))
    equal = []
    for budget in chosen_budgets:
        info = {}
        for arm in (5, 10):
            feasible = [row for row in rows if row["arm"] == arm and
                        row["total_compiler_USD_upper_estimate"] <= budget+1e-10]
            if feasible:
                item = feasible[-1]
                info[arm] = item
                entry = {"budget_USD": budget, "arm": arm, "feasible": True,
                         "racing_trajectories": item["racing_trajectories"],
                         "survivor_count": item["survivor_count"],
                         "selection_status": item["selection_status"],
                         "selected_or_fallback_policy_id": item["selected_or_fallback_policy_id"],
                         "compiler_cost_USD_upper_estimate": item["total_compiler_USD_upper_estimate"]}
            else:
                entry = {"budget_USD": budget, "arm": arm, "feasible": False,
                         "racing_trajectories": 0, "survivor_count": "",
                         "selection_status": "no_racing_checkpoint_affordable",
                         "selected_or_fallback_policy_id": "",
                         "compiler_cost_USD_upper_estimate": ""}
            equal.append(entry)
    _csv(root / "budget/equal_budget_comparison.csv", equal)
    actual_union_sec = sum(item["actual_union_seconds"] for item in times)
    summary = {"historical_coarse_5_USD": manifest["g5_historical_estimated_usd"],
               "historical_coarse_10_USD": manifest["g10_historical_estimated_usd"],
               "racing_measured_trajectories": measured_n,
               "racing_actual_union_GPU_seconds": actual_union_sec,
               "racing_actual_union_estimated_USD": actual_union_sec*RATE_USD_PER_SECOND,
               "arm_timing_limitation": "D5 and D10 were not separately profiled; arm cost is bounded by measured fixed+branch time and by adding all measured union shared overhead",
               "g10_top1_historical_compiler_USD": manifest["g10_historical_estimated_usd"],
               "new_GPU_cost_cap_USD": manifest["max_new_gpu_usd"]}
    (root / "analysis").mkdir(parents=True, exist_ok=True)
    (root / "analysis/cost_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True)+"\n")
    return summary


def difficulty(root: Path):
    eps = read(root / "analysis/difficulty_design.json")["epsilon"]
    rows = []
    for arm in (5, 10):
        candidates = read(root / f"proposal/g{arm}_candidates.json")["candidates"]
        ids = [x["policy_id"] for x in candidates]
        for round_number in range(1, 7):
            state_path = root / f"racing/arm{arm}_round_{round_number}_state.json"
            if not state_path.exists():
                break
            state = read(state_path)
            n = state["trajectories"]
            union, matrix = costs(root, n)
            idx = {key: i for i, key in enumerate(union)}
            d = set(read(root / f"edges/d{arm}.json")["discriminating_edges"])
            path_cost = {item["policy_id"]: matrix[:, [idx[k] for k in item["edge_keys"] if k in d]].sum(axis=1)
                         for item in candidates}
            active = state["start_survivors"]
            best = min(active, key=lambda pid: (float(path_cost[pid].mean()), pid))
            for pid in active:
                if pid == best:
                    continue
                diff = path_cost[pid]-path_cost[best]
                gap = float(diff.mean())
                variance = float(diff.var(ddof=1))
                rows.append({"arm": arm, "round": round_number, "trajectories": n,
                             "candidate_policy_id": pid, "current_best_policy_id": best,
                             "empirical_gap": gap, "paired_variance": variance,
                             "difficulty_score": variance/max(abs(gap), eps)**2,
                             "eliminated_this_round": pid in {x["policy_id"] for x in state["eliminations"]},
                             "survived_round": pid in state["survivor_policy_ids"],
                             "epsilon": eps, "descriptive_not_theorem": True})
    _csv(root / "analysis/difficulty_scores.csv", rows)
    entropy_rows = []
    for arm in (5, 10):
        proposal = read(root / f"proposal/g{arm}_candidates.json")
        entropy_rows.append({"coarse_n": arm,
                             "bootstrap_entropy_bits": proposal["bootstrap_entropy_bits"],
                             "bootstrap_effective_paths": proposal["bootstrap_effective_paths"],
                             "graph_candidates": len(proposal["graph_candidates"]),
                             "bootstrap_mass_covered": proposal["bootstrap_mass_covered"],
                             "proposal_undercoverage": proposal["proposal_undercoverage"]})
    _csv(root / "analysis/proposal_entropy.csv", entropy_rows)
    return {"difficulty_rows": len(rows), "proposal_entropy": entropy_rows}


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1] / "results/static-bfs-graph/aer-budget-allocation-v1"
    print(json.dumps({"cost": build(root), "difficulty": difficulty(root)}, indent=2, sort_keys=True))
