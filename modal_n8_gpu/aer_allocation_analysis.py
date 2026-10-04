"""Offline accounting, frozen audit, integrity, and analysis of AER allocation."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from scripts.static_bfs_graph.n8_run import manual8_record
from scripts.static_bfs_graph.policy import StaticBFSPolicy
from scripts.static_bfs_graph.runner import static_record

from .aer_allocation_plan import (AUDIT_BOOTSTRAPS, MAX_NEW_USD, PROTOCOL,
                                  RATE_USD_PER_SECOND)
from .aer_allocation_racing import costs
from .aer_plan import digest, read, write_once


def _csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _existing_states(root: Path, arm: int):
    out = []
    for r in range(1, 7):
        path = root / f"racing/arm{arm}_round_{r}_state.json"
        if not path.exists():
            break
        out.append(read(path))
    if not out or not out[-1]["stop"]:
        raise RuntimeError(f"arm {arm} racing not finished")
    return out


def freeze(root: Path):
    """Freeze every compiler output and deduplicated audit policy before inference."""
    states = {arm: _existing_states(root, arm) for arm in (5, 10)}
    proposals = {arm: read(root / f"proposal/g{arm}_candidates.json") for arm in (5, 10)}
    outputs = {"protocol": PROTOCOL, "audit_scoring_used": False,
               "racing_pool_sha256": digest(root / "racing/common_racing_pool.json"),
               "arms": {str(arm): {
                   "proposal_sha256": digest(root / f"proposal/g{arm}_candidates.json"),
                   "final_state_sha256": digest(root / f"racing/arm{arm}_round_{len(states[arm])}_state.json"),
                   "racing_trajectories_used": states[arm][-1]["trajectories"],
                   "selected_or_fallback_policy_id": states[arm][-1]["selected_or_fallback_policy_id"],
                   "selection_status": states[arm][-1]["selection_status"],
                   "survivors": states[arm][-1]["survivor_policy_ids"]}
                        for arm in (5, 10)}}
    write_once(root / "frozen/frozen_compiler_outputs.json", outputs)
    aliases = {"AER5_FINAL": outputs["arms"]["5"]["selected_or_fallback_policy_id"],
               "AER10_FINAL": outputs["arms"]["10"]["selected_or_fallback_policy_id"],
               "G10_TOP1": proposals[10]["coarse_top1_policy_id"],
               "TRANSFER4_TO_8": proposals[10]["incumbent_policy_id"],
               "MANUAL8": manual8_record()["id"]}
    rows = {item["policy_id"]: item for arm in (5, 10) for item in proposals[arm]["candidates"]}
    unique = {}
    for alias, pid in aliases.items():
        if alias == "MANUAL8":
            record = manual8_record()
        else:
            item = rows[pid]
            policy = StaticBFSPolicy(tuple(item["steps"]), tuple(zip(item["steps"], item["taus"])), 8)
            record = static_record(policy)
        if record["id"] != pid:
            raise RuntimeError("frozen audit policy ID mismatch")
        unique.setdefault(pid, {"policy_id": pid, "record": record, "first_alias": alias})
    pool = read(root / "audit/audit_pool.json")
    frozen = {"protocol": PROTOCOL,
              "compiler_outputs_sha256": digest(root / "frozen/frozen_compiler_outputs.json"),
              "audit_pool_sha256": digest(root / "audit/audit_pool.json"),
              "aliases": aliases, "unique_policies": list(unique.values()),
              "pairs": pool["pairs"], "freeze_before_audit_scoring": True}
    write_once(root / "frozen/frozen_audit_plan.json", frozen)
    return {"aliases": aliases, "unique_policy_count": len(unique),
            "frozen_audit_plan_sha256": digest(root / "frozen/frozen_audit_plan.json")}


def audit_aggregate(root: Path):
    frozen = read(root / "frozen/frozen_audit_plan.json")
    outputs = read(root / "frozen/frozen_compiler_outputs.json")
    if (frozen["compiler_outputs_sha256"] != digest(root / "frozen/frozen_compiler_outputs.json") or
        frozen["audit_pool_sha256"] != digest(root / "audit/audit_pool.json") or
        outputs["audit_scoring_used"] or len(frozen["pairs"]) != 20):
        raise RuntimeError("audit freeze changed")
    timing = [read(root / f"audit/timing/trajectory_{i:02d}.json") for i in range(20)]
    values = {}
    prompt_rows = []
    for item in frozen["unique_policies"]:
        pid = item["policy_id"]
        scores = []
        for i, pair in enumerate(frozen["pairs"]):
            if (timing[i]["prompt_id"] != pair["prompt_id"] or
                timing[i]["seed"] != pair["seed"] or
                timing[i]["unique_policy_count"] != len(frozen["unique_policies"])):
                raise RuntimeError("audit timing/pool mismatch")
            row = read(root / "audit/raw" / pid / f"p{pair['prompt_id']}_s{pair['seed']}.json")
            if (row["policy_id"] != pid or row["prompt_index"] != pair["prompt_id"] or
                row["trial_seed"] != pair["seed"] or row["backend"] != "CUDA_Modal" or
                row["diffusion_NFE"] != 800 or len(row["final_particle_scores"]) != 8 or
                not np.isfinite(row["selected_final_score"])):
                raise RuntimeError(f"invalid audit row: {i}:{pid}")
            score = float(row["selected_final_score"])
            scores.append(score)
            prompt_rows.append({"rank": i, "prompt_id": pair["prompt_id"],
                                "seed": pair["seed"], "policy_id": pid,
                                "ImageReward": score, "elapsed_seconds": row["elapsed_seconds"]})
        values[pid] = np.asarray(scores)
    aliases = frozen["aliases"]
    policy_rows = [{"alias": alias, "policy_id": pid,
                    "mean_ImageReward": float(values[pid].mean()),
                    "std_ImageReward": float(values[pid].std(ddof=1)),
                    "prompt_count": 20, "seed_per_prompt": 1,
                    "is_distinct_policy": alias == next(item["first_alias"] for item in
                                                       frozen["unique_policies"] if item["policy_id"] == pid)}
                   for alias, pid in aliases.items()]
    comparisons = [(a, b) for a, b in (
        ("AER5_FINAL", "G10_TOP1"), ("AER10_FINAL", "G10_TOP1"),
        ("AER5_FINAL", "AER10_FINAL"), ("AER5_FINAL", "TRANSFER4_TO_8"),
        ("AER10_FINAL", "TRANSFER4_TO_8"))]
    picks = np.random.default_rng(20261003).integers(0, 20, size=(AUDIT_BOOTSTRAPS, 20))
    paired = []
    for a, b in comparisons:
        d = values[aliases[a]] - values[aliases[b]]
        low, high = np.quantile(d[picks].mean(axis=1), [.025, .975])
        paired.append({"method": a, "baseline": b,
                       "mean_delta_ImageReward": float(d.mean()),
                       "ci95_low": float(low), "ci95_high": float(high),
                       "prompt_wins": int((d > 1e-12).sum()),
                       "prompt_ties": int((np.abs(d) <= 1e-12).sum()),
                       "prompt_losses": int((d < -1e-12).sum()),
                       "bootstrap_resamples": AUDIT_BOOTSTRAPS,
                       "statistical_unit": "TRAIN prompt; one shared seed per prompt"})
    _csv(root / "audit/policy_results.csv", policy_rows)
    _csv(root / "audit/paired_results.csv", paired)
    _csv(root / "audit/prompt_policy_rows.csv", prompt_rows)
    summary = {"policy_results": policy_rows, "paired_results": paired,
               "raw_policy_rows": len(prompt_rows),
               "audit_GPU_seconds": sum(x["trajectory_wall_seconds"] for x in timing),
               "audit_estimated_USD": sum(x["trajectory_wall_seconds"] for x in timing)*RATE_USD_PER_SECOND,
               "frozen_audit_plan_sha256": digest(root / "frozen/frozen_audit_plan.json")}
    write_once(root / "audit/summary.json", summary)
    return summary


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1] / "results/static-bfs-graph/aer-budget-allocation-v1"
    import sys
    if sys.argv[1] == "freeze":
        print(json.dumps(freeze(root), indent=2, sort_keys=True))
    elif sys.argv[1] == "audit":
        print(json.dumps(audit_aggregate(root), indent=2, sort_keys=True))
    else:
        raise ValueError("usage: freeze|audit")
