"""Fixed-round paired edge racing with Bonferroni-adjusted bootstrap intervals."""
from __future__ import annotations

import csv
import json
from math import comb
from pathlib import Path

import numpy as np

from .aer_plan import (PAIR_BOOTSTRAPS, PATH_EQUIV_EPS, PROTOCOL, TOTAL_DELTA,
                       digest, read, write_once)


def _csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def costs_for_completed(root: Path, n: int):
    candidate_set = read(root / "stage_a/candidate_set.json")
    discriminating = read(root / "racing/discriminating_edges.json")
    plan = read(root / "racing/racing_trajectory_plan.json")
    if (candidate_set["protocol"] != PROTOCOL or plan["protocol"] != PROTOCOL or
        plan["candidate_set_sha256"] != digest(root / "stage_a/candidate_set.json") or
        plan["discriminating_edges_sha256"] != digest(root / "racing/discriminating_edges.json") or
        plan["paired_bootstraps"] != PAIR_BOOTSTRAPS or
        plan["total_delta"] != TOTAL_DELTA or
        plan["equivalence_margin"] != PATH_EQUIV_EPS):
        raise RuntimeError("frozen racing inputs changed")
    candidates = candidate_set["candidates"]
    keys = discriminating["discriminating_edges"]
    index = {key: i for i, key in enumerate(keys)}
    if len(candidates) != discriminating["candidate_count"] or n not in (5, 10, 15, 20):
        raise RuntimeError("candidate count or round boundary invalid")
    matrix = np.full((n, len(keys)), np.nan)
    for rank in range(n):
        traj = plan["trajectories"][rank]
        folder = root / f"racing/raw/trajectory_{rank:02d}"
        reference = read(folder / "reference.json")
        timing = read(folder / "timing.json")
        if (reference["prompt_index"] != traj["prompt_id"] or
            reference["trial_seed"] != traj["seed"] or
            timing["prompt_id"] != traj["prompt_id"] or
            timing["seed"] != traj["seed"] or timing["edge_count"] != len(keys)):
            raise RuntimeError(f"racing trajectory {rank} metadata changed")
        actual = {p.stem for p in (folder / "edges").glob("*.json")
                  if not p.name.startswith("._")}
        if actual != set(keys):
            raise RuntimeError(f"racing trajectory {rank} edge set incomplete")
        for key, ei in index.items():
            edge = read(folder / "edges" / f"{key}.json")
            if (edge["prompt_index"] != traj["prompt_id"] or
                edge["trial_seed"] != traj["seed"] or edge["edge_key"] != key or
                edge["backend"] != "CUDA_Modal" or
                not np.isclose(edge["delta"], reference["selected_final_score"] -
                               edge["selected_final_score"], atol=1e-10, rtol=0)):
                raise RuntimeError(f"racing edge {rank}:{key} invalid")
            matrix[rank, ei] = edge["delta"]
    if not np.isfinite(matrix).all():
        raise RuntimeError("nonfinite racing edge costs")
    cost = np.stack([matrix[:, [index[key] for key in item["edge_keys"]
                                if key in index]].sum(axis=1)
                     for item in candidates], axis=1)
    return candidates, cost


def analyze_round(root: Path, round_number: int):
    if round_number not in (1, 2, 3, 4):
        raise ValueError("round must be 1..4")
    n = 5 * round_number
    candidates, cost = costs_for_completed(root, n)
    m = len(candidates)
    ids = [item["policy_id"] for item in candidates]
    if len(set(ids)) != m or m < 2:
        raise RuntimeError("candidate IDs not unique")
    prior = (read(root / f"racing/round_{round_number-1}_state.json")
             if round_number > 1 else None)
    if prior and prior["stop"]:
        raise RuntimeError("cannot race after frozen stop")
    survivors = set(prior["survivor_policy_ids"] if prior else ids)
    active = [i for i, pid in enumerate(ids) if pid in survivors]
    alpha = TOTAL_DELTA / (4 * comb(m, 2))
    picks = np.random.default_rng(20261003 + round_number).integers(
        0, n, size=(PAIR_BOOTSTRAPS, n))
    pair_rows = []
    eliminated = {}
    for ai, i in enumerate(active):
        for j in active[ai + 1:]:
            diff = cost[:, i] - cost[:, j]
            resampled = diff[picks].mean(axis=1)
            lower, upper = np.quantile(resampled, [alpha / 2, 1 - alpha / 2])
            mean = float(diff.mean())
            pair_rows.append({"round": round_number, "trajectories": n,
                              "policy_p": ids[i], "policy_q": ids[j],
                              "mean_d_p_minus_q": mean,
                              "ci_lower": float(lower), "ci_upper": float(upper),
                              "pair_round_alpha": alpha,
                              "bootstrap_resamples": PAIR_BOOTSTRAPS,
                              "paired": True})
            if lower > 0:
                eliminated.setdefault(ids[i], {"dominated_by": ids[j],
                                          "mean_d": mean, "ci": [float(lower), float(upper)]})
            if upper < 0:
                eliminated.setdefault(ids[j], {"dominated_by": ids[i],
                                          "mean_d": -mean,
                                          "ci": [float(-upper), float(-lower)]})
    remaining = [pid for pid in ids if pid in survivors and pid not in eliminated]
    if not remaining:
        raise RuntimeError("sequential elimination removed all candidates")
    all_inside_equivalence = all(
        row["ci_lower"] >= -PATH_EQUIV_EPS and row["ci_upper"] <= PATH_EQUIV_EPS
        for row in pair_rows
        if row["policy_p"] in remaining and row["policy_q"] in remaining)
    incumbent = read(root / "stage_a/candidate_set.json")["incumbent_policy_id"]
    stop = len(remaining) == 1 or all_inside_equivalence or round_number == 4
    selection_status = ("certified_unique" if len(remaining) == 1 else
                        "practically_equivalent" if all_inside_equivalence else
                        "underidentified" if round_number == 4 else None)
    chosen = None
    if stop:
        if incumbent in remaining:
            chosen = incumbent
        elif len(remaining) == 1:
            chosen = remaining[0]
        else:
            # Deterministic lowest fresh mean, then higher bootstrap frequency,
            # then stable policy ID. No full-policy outcome enters this rule.
            frequencies = {item["policy_id"]: item["bootstrap_frequency"]
                           for item in candidates}
            chosen = min(remaining, key=lambda pid: (
                float(cost[:, ids.index(pid)].mean()), -frequencies[pid], pid))
    state = {"round": round_number, "trajectories": n,
             "frozen_candidate_count": m, "start_survivors": [ids[i] for i in active],
             "eliminations": [{"policy_id": pid, **reason}
                              for pid, reason in eliminated.items()],
             "survivor_policy_ids": remaining,
             "survivor_mean_discriminating_cost": {
                 pid: float(cost[:, ids.index(pid)].mean()) for pid in remaining},
             "all_surviving_pairs_equivalent": bool(all_inside_equivalence),
             "stop": stop, "selection_status": selection_status,
             "selected_policy_id": chosen,
             "confidence_procedure": "paired percentile bootstrap; two-sided Bonferroni over all frozen pairs and four rounds",
             "pair_round_alpha": alpha,
             "candidate_set_sha256": digest(root / "stage_a/candidate_set.json"),
             "racing_plan_sha256": digest(root / "racing/racing_trajectory_plan.json")}
    write_once(root / f"racing/round_{round_number}_state.json", state)
    write_once(root / f"racing/round_{round_number}_pairs.json", pair_rows)
    states = [read(root / f"racing/round_{r}_state.json")
              for r in range(1, round_number + 1)]
    _csv(root / "racing/round_summary.csv", [{
        "round": item["round"], "trajectories": item["trajectories"],
        "start_candidate_count": len(item["start_survivors"]),
        "eliminated_count": len(item["eliminations"]),
        "survivor_count": len(item["survivor_policy_ids"]),
        "eliminations_json": json.dumps(item["eliminations"], sort_keys=True),
        "survivors_json": json.dumps(item["survivor_policy_ids"]),
        "stop": item["stop"], "selection_status": item["selection_status"]}
        for item in states])
    _csv(root / "racing/pairwise_intervals.csv", [row
        for r in range(1, round_number + 1)
        for row in read(root / f"racing/round_{r}_pairs.json")])
    if stop:
        chosen_item = next(item for item in candidates if item["policy_id"] == chosen)
        compiled = {"protocol": PROTOCOL,
                    "candidate_set_sha256": digest(root / "stage_a/candidate_set.json"),
                    "racing_plan_sha256": digest(root / "racing/racing_trajectory_plan.json"),
                    "candidate_set": candidates,
                    "candidate_bootstrap_frequencies": {
                        item["policy_id"]: item["bootstrap_frequency"] for item in candidates},
                    "discriminating_edges": read(root / "racing/discriminating_edges.json")["discriminating_edges"],
                    "racing_trajectories_used": n,
                    "eliminations": [item for s in states for item in s["eliminations"]],
                    "final_survivors": remaining,
                    "selected_policy": chosen_item,
                    "selection_status": selection_status,
                    "confidence_procedure": state["confidence_procedure"],
                    "pair_round_alpha": alpha,
                    "bootstrap_not_finite_sample_certification": True,
                    "audit_scoring_used": False}
        write_once(root / "racing/aer_compiled_policy.json", compiled)
    return state
