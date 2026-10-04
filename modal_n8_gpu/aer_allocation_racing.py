"""Frozen paired-bootstrap racing for the two nested coarse-graph arms."""
from __future__ import annotations

import csv
import json
from math import comb
from pathlib import Path

import numpy as np

from .aer_allocation_plan import (BATCH_SIZE, MAX_ROUNDS, PAIR_BOOTSTRAPS,
                                  PATH_EQUIV_EPS, PROTOCOL, TOTAL_DELTA)
from .aer_plan import digest, read, write_once


def _csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def costs(root: Path, n: int):
    if n not in range(5, 31, 5):
        raise ValueError("invalid frozen checkpoint")
    pool = read(root / "racing/common_racing_pool.json")
    union = read(root / "edges/d_union.json")["discriminating_edges"]
    if (pool["protocol"] != PROTOCOL or pool["d_union_sha256"] !=
        digest(root / "edges/d_union.json") or len(pool["trajectories"]) != 30):
        raise RuntimeError("frozen racing pool changed")
    matrix = np.full((n, len(union)), np.nan)
    for rank in range(n):
        item = pool["trajectories"][rank]
        folder = root / f"racing/raw/trajectory_{rank:02d}"
        ref = read(folder / "reference.json")
        timing = read(folder / "timing.json")
        if (ref["prompt_index"] != item["prompt_id"] or
            ref["trial_seed"] != item["seed"] or
            timing["prompt_id"] != item["prompt_id"] or
            timing["seed"] != item["seed"] or timing["edge_count"] != len(union)):
            raise RuntimeError(f"trajectory metadata changed: {rank}")
        edge_dir = folder / "edges"
        if {p.stem for p in edge_dir.glob("*.json") if not p.name.startswith("._")} != set(union):
            raise RuntimeError(f"incomplete union edge set: {rank}")
        for j, key in enumerate(union):
            row = read(edge_dir / f"{key}.json")
            if (row["prompt_index"] != item["prompt_id"] or
                row["trial_seed"] != item["seed"] or
                row["edge_key"] != key or row["backend"] != "CUDA_Modal" or
                not np.isclose(row["delta"], ref["selected_final_score"]-
                               row["selected_final_score"], atol=1e-10, rtol=0)):
                raise RuntimeError(f"invalid edge record: {rank}:{key}")
            matrix[rank, j] = row["delta"]
    if not np.isfinite(matrix).all():
        raise RuntimeError("nonfinite edge costs")
    return union, matrix


def analyze_round(root: Path, arm: int, round_number: int):
    if arm not in (5, 10) or round_number not in range(1, MAX_ROUNDS+1):
        raise ValueError("invalid arm or round")
    n = BATCH_SIZE * round_number
    proposal_path = root / f"proposal/g{arm}_candidates.json"
    proposal = read(proposal_path)
    d = read(root / f"edges/d{arm}.json")
    pool = read(root / "racing/common_racing_pool.json")
    if (d["candidate_file_sha256"] != digest(proposal_path) or
        pool["paired_bootstraps"] != PAIR_BOOTSTRAPS or
        pool["max_rounds"] != MAX_ROUNDS or
        pool["total_delta"] != TOTAL_DELTA or
        pool["equivalence_margin"] != PATH_EQUIV_EPS):
        raise RuntimeError("frozen racing design changed")
    candidates = proposal["candidates"]
    ids = [x["policy_id"] for x in candidates]
    if len(ids) != len(set(ids)) or len(ids) < 2:
        raise RuntimeError("invalid candidate list")
    union, matrix = costs(root, n)
    if not set(d["discriminating_edges"]) <= set(union):
        raise RuntimeError("arm edges missing from union")
    index = {key: i for i, key in enumerate(union)}
    arm_keys = set(d["discriminating_edges"])
    cost = np.stack([matrix[:, [index[key] for key in candidate["edge_keys"]
                                 if key in arm_keys]].sum(axis=1)
                     for candidate in candidates], axis=1)
    prior = read(root / f"racing/arm{arm}_round_{round_number-1}_state.json") if round_number > 1 else None
    if prior and prior["stop"]:
        raise RuntimeError("cannot continue stopped arm")
    survivors = set(prior["survivor_policy_ids"] if prior else ids)
    active = [i for i, pid in enumerate(ids) if pid in survivors]
    alpha = TOTAL_DELTA / (MAX_ROUNDS * comb(len(ids), 2))
    picks = np.random.default_rng(20261003+round_number).integers(0, n, size=(PAIR_BOOTSTRAPS, n))
    intervals = []
    eliminated = {}
    for ai, i in enumerate(active):
        for j in active[ai+1:]:
            diff = cost[:, i] - cost[:, j]
            boot = diff[picks].mean(axis=1)
            low, high = np.quantile(boot, [alpha/2, 1-alpha/2])
            mean = float(diff.mean())
            intervals.append({"arm": arm, "round": round_number, "trajectories": n,
                              "policy_p": ids[i], "policy_q": ids[j],
                              "mean_d_p_minus_q": mean, "ci_lower": float(low),
                              "ci_upper": float(high), "pair_round_alpha": alpha,
                              "bootstrap_resamples": PAIR_BOOTSTRAPS, "paired": True})
            if low > 0:
                eliminated.setdefault(ids[i], {"dominated_by": ids[j], "mean_d": mean,
                                               "ci": [float(low), float(high)]})
            if high < 0:
                eliminated.setdefault(ids[j], {"dominated_by": ids[i], "mean_d": -mean,
                                               "ci": [float(-high), float(-low)]})
    remaining = [pid for pid in ids if pid in survivors and pid not in eliminated]
    if not remaining:
        raise RuntimeError("all candidates eliminated")
    equivalent = all(row["ci_lower"] >= -PATH_EQUIV_EPS and row["ci_upper"] <= PATH_EQUIV_EPS
                     for row in intervals if row["policy_p"] in remaining and row["policy_q"] in remaining)
    status = ("unique" if len(remaining) == 1 else
              "practically_equivalent" if equivalent else "underidentified")
    incumbent = proposal["incumbent_policy_id"]
    frequencies = {x["policy_id"]: x["bootstrap_frequency"] for x in candidates}
    selected = (incumbent if incumbent in remaining else min(remaining, key=lambda pid: (
        float(cost[:, ids.index(pid)].mean()), -frequencies[pid], pid)))
    stop = status != "underidentified" or round_number == MAX_ROUNDS
    state = {"arm": arm, "round": round_number, "trajectories": n,
             "frozen_candidate_count": len(ids), "start_survivors": [ids[i] for i in active],
             "eliminations": [{"policy_id": pid, **reason} for pid, reason in eliminated.items()],
             "survivor_policy_ids": remaining, "survivor_mean_discriminating_cost": {
                 pid: float(cost[:, ids.index(pid)].mean()) for pid in remaining},
             "all_surviving_pairs_equivalent": bool(equivalent), "stop": stop,
             "selection_status": status, "selected_or_fallback_policy_id": selected,
             "confidence_procedure": pool["confidence_procedure"],
             "pair_round_alpha": alpha, "candidate_set_sha256": digest(proposal_path),
             "racing_pool_sha256": digest(root / "racing/common_racing_pool.json")}
    write_once(root / f"racing/arm{arm}_round_{round_number}_state.json", state)
    write_once(root / f"racing/arm{arm}_round_{round_number}_pairs.json", intervals)
    states = [read(root / f"racing/arm{arm}_round_{r}_state.json") for r in range(1, round_number+1)]
    _csv(root / f"racing/arm{arm}_rounds.csv", [{
        "arm": arm, "round": s["round"], "trajectories": s["trajectories"],
        "start_candidate_count": len(s["start_survivors"]),
        "eliminated_count": len(s["eliminations"]), "survivor_count": len(s["survivor_policy_ids"]),
        "survivors_json": json.dumps(s["survivor_policy_ids"]),
        "eliminations_json": json.dumps(s["eliminations"], sort_keys=True),
        "stop": s["stop"], "selection_status": s["selection_status"],
        "selected_or_fallback_policy_id": s["selected_or_fallback_policy_id"]} for s in states])
    _csv(root / f"racing/pairwise_intervals_arm{arm}.csv", [row for r in range(1, round_number+1)
         for row in read(root / f"racing/arm{arm}_round_{r}_pairs.json")])
    return state
