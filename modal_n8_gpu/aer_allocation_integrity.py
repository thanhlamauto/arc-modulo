"""Final artifact and split audit for AER budget allocation."""
from __future__ import annotations

from pathlib import Path
import json

from .aer_allocation_plan import MAX_NEW_USD, PROTOCOL, RATE_USD_PER_SECOND
from .aer_plan import digest, read, write_once


def validate(repo: Path, root: Path):
    manifest = read(root / "experiment_manifest.json")
    if manifest["protocol"] != PROTOCOL:
        raise RuntimeError("protocol changed")
    frozen = read(root / "frozen/frozen_audit_plan.json")
    compiler = read(root / "frozen/frozen_compiler_outputs.json")
    if (frozen["compiler_outputs_sha256"] != digest(root / "frozen/frozen_compiler_outputs.json") or
        frozen["audit_pool_sha256"] != digest(root / "audit/audit_pool.json") or
        compiler["audit_scoring_used"]):
        raise RuntimeError("audit/compile separation failed")
    for path, field in (("proposal/g5_candidates.json", "g5_candidates_sha256"),
                        ("proposal/g10_candidates.json", "g10_candidates_sha256"),
                        ("edges/d_union.json", "d_union_sha256"),
                        ("racing/common_racing_pool.json", "racing_pool_sha256"),
                        ("audit/audit_pool.json", "audit_pool_sha256")):
        if digest(root/path) != manifest[field]:
            raise RuntimeError(f"frozen hash mismatch: {path}")
    historical = repo / "results/static-bfs-graph/n8-offline-compiler-v1"
    for path, field in (("integrity.json", "source_integrity_sha256"),
                        ("profiler/edge_observations.csv", "source_edge_observations_sha256"),
                        ("calibration_trajectories.json", "source_calibration_trajectories_sha256"),
                        ("analysis/compilation_cost.csv", "source_compilation_cost_sha256")):
        if digest(historical/path) != manifest[field]:
            raise RuntimeError(f"historical source changed: {path}")
    pool = read(repo / "modal_n8_gpu/train_pool.json")
    if digest(repo / "modal_n8_gpu/train_pool.json") != manifest["source_train_pool_sha256"]:
        raise RuntimeError("TRAIN pool changed")
    race = read(root / "racing/common_racing_pool.json")["trajectories"]
    audit = read(root / "audit/audit_pool.json")["pairs"]
    historical_pairs = read(historical / "calibration_trajectories.json")[:10]
    old_aer_root = repo / "results/static-bfs-graph/adaptive-edge-racing-v1"
    old_race = read(old_aer_root / "racing/racing_trajectory_plan.json")["trajectories"]
    old_audit = read(old_aer_root / "audit/audit_plan.json")["pairs"]
    prior_pairs = {(x["prompt_id"], x["seed"]) for x in historical_pairs+old_race+old_audit}
    fresh_pairs = {(x["prompt_id"], x["seed"]) for x in race+audit}
    if (len(race) != 30 or len(audit) != 20 or len(fresh_pairs) != 50 or
        prior_pairs & fresh_pairs or
        not {x["prompt_id"] for x in race+audit} <= set(map(int, pool["train_indices"])) or
        not {x["seed"] for x in race+audit}.isdisjoint({x["seed"] for x in historical_pairs+old_race+old_audit})):
        raise RuntimeError("fresh TRAIN pool disjointness failed")
    measured_n = max(x["racing_trajectories_used"] for x in compiler["arms"].values())
    union = read(root / "edges/d_union.json")["discriminating_edges"]
    race_seconds = 0.
    for rank in range(measured_n):
        folder = root / f"racing/raw/trajectory_{rank:02d}"
        item = race[rank]
        timing = read(folder / "timing.json")
        ref = read(folder / "reference.json")
        if (timing["prompt_id"] != item["prompt_id"] or timing["seed"] != item["seed"] or
            timing["edge_count"] != len(union) or
            ref["prompt_index"] != item["prompt_id"] or ref["trial_seed"] != item["seed"] or
            {p.stem for p in (folder / "edges").glob("*.json") if not p.name.startswith("._")} != set(union)):
            raise RuntimeError(f"racing trajectory invalid: {rank}")
        race_seconds += timing["trajectory_wall_seconds"]
    if not read(root / "racing/correctness_gate.json")["passed"]:
        raise RuntimeError("correctness gate failed")
    unique = frozen["unique_policies"]
    audit_seconds = 0.
    for rank, pair in enumerate(audit):
        timing = read(root / f"audit/timing/trajectory_{rank:02d}.json")
        if (timing["prompt_id"] != pair["prompt_id"] or timing["seed"] != pair["seed"] or
            timing["unique_policy_count"] != len(unique)):
            raise RuntimeError(f"audit timing invalid: {rank}")
        audit_seconds += timing["trajectory_wall_seconds"]
        for item in unique:
            path = root / "audit/raw" / item["policy_id"] / f"p{pair['prompt_id']}_s{pair['seed']}.json"
            row = read(path)
            if row["policy_id"] != item["policy_id"] or row["trial_seed"] != pair["seed"]:
                raise RuntimeError("audit policy row invalid")
    new_cost = (race_seconds+audit_seconds)*RATE_USD_PER_SECOND
    if new_cost > float(manifest["max_new_gpu_usd"]):
        raise RuntimeError("new GPU cost cap exceeded")
    required = ["budget/compiler_cost_curve.csv", "budget/equal_budget_comparison.csv",
                "audit/policy_results.csv", "audit/paired_results.csv",
                "analysis/difficulty_scores.csv", "analysis/proposal_entropy.csv",
                "analysis/cost_summary.json", "THEORY_ALLOCATION.md", "report.md",
                "terminal_summary.txt"]
    required += [f"plots/{i:02d}_{name}.png" for i, name in enumerate((
        "bootstrap_mass", "survivors_vs_trajectories", "survivors_vs_total_compiler_cost",
        "compiler_cost_decomposition", "audit_reward", "difficulty_vs_elimination"), 1)]
    for path in required:
        if not (root/path).exists():
            raise RuntimeError(f"required output missing: {path}")
    out = {"passed": True, "protocol": PROTOCOL,
           "racing_trajectories": measured_n, "union_edge_count": len(union),
           "racing_edge_records": measured_n*len(union),
           "audit_pairs": len(audit), "unique_audit_policies": len(unique),
           "audit_policy_records": len(audit)*len(unique),
           "racing_GPU_seconds": race_seconds, "audit_GPU_seconds": audit_seconds,
           "new_GPU_seconds": race_seconds+audit_seconds,
           "new_GPU_estimated_USD": new_cost,
           "cost_cap_USD": manifest["max_new_gpu_usd"],
           "frozen_compiler_outputs_sha256": digest(root / "frozen/frozen_compiler_outputs.json"),
           "frozen_audit_plan_sha256": digest(root / "frozen/frozen_audit_plan.json"),
           "VALIDATION_touched": False, "TEST_touched": False,
           "external_100_prompt_pool_touched": False,
           "historical_graph_GPU_rerun": False}
    write_once(root / "integrity.json", out)
    return out


if __name__ == "__main__":
    repo = Path(__file__).resolve().parents[1]
    root = repo / "results/static-bfs-graph/aer-budget-allocation-v1"
    print(json.dumps(validate(repo, root), indent=2, sort_keys=True))
