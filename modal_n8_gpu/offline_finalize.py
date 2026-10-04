"""Validate and package a completed N=8 offline-compiler run locally."""
from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
from shutil import copyfile


def read(path: Path):
    return json.loads(path.read_text())


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def finish(root: Path) -> dict:
    manifest = read(root / "experiment_manifest.json")
    calibration = read(root / "calibration_trajectories.json")
    audit_prompts = read(root / "audit_prompts.json")
    replay = read(root / "replay/equivalence.json")
    tau_fork = read(root / "replay/tau_fork_equivalence.json")
    graph = read(root / "graphs/policies_by_n.json")
    compiled_path = root / "compiled/compiled_policy.json"
    compiled = read(compiled_path)
    compilation = read(root / "compiled/compilation_manifest.json")
    plan_path = root / "audit/audit_policy_plan.json"
    plan = read(plan_path)
    audit = read(root / "audit/summary.json")
    if (manifest["protocol"] != "STATIC_BFS_N8_OFFLINE_COMPILER_V1" or
        manifest["N"] != 8 or manifest["K"] != 3 or
        manifest["PSP_in_scope"] or manifest["validation_touched"] or
        manifest["test_touched"] or manifest["external_100_prompt_pool_touched"] or
        manifest["old_TPU_scientific_results_mixed"] or
        manifest["old_Modal_canary_scientific_results_mixed"]):
        raise RuntimeError("experiment manifest is outside the frozen scope")
    if (len(calibration) != 20 or len(audit_prompts) != 20 or
        len({x["prompt_id"] for x in calibration}) != 20 or
        len({x["prompt_id"] for x in audit_prompts}) != 20 or
        {x["prompt_id"] for x in calibration} & {x["prompt_id"] for x in audit_prompts}):
        raise RuntimeError("calibration/audit split invalid")
    if (not replay["passed"] or len(replay["reference_replays"]) != 6 or
        not all(x["pass"] for x in replay["reference_replays"]) or
        not all(x["pass"] for x in replay["shared_tree_vs_naive_edges"]) or
        not tau_fork["passed"] or len(tau_fork["checks"]) != 3):
        raise RuntimeError("replay or shared-tree gate failed")
    if ([x["n"] for x in graph] != [1, 5, 10, 20] or
        compiled["policy_id"] != graph[-1]["policy_id"] or
        compilation["compiled_policy_sha256"] != digest(compiled_path) or
        compilation["audit_plan_sha256"] != digest(plan_path) or
        audit["audit_policy_plan_sha256"] != digest(plan_path)):
        raise RuntimeError("frozen policy or audit hashes changed")

    edge_keys = set(manifest["edge_keys"])
    if len(edge_keys) != 92:
        raise RuntimeError("logical edge catalogue changed")
    observations = []
    timing = []
    for rank, traj in enumerate(calibration):
        prefix = root / f"profiler/trajectory_{rank:02d}"
        reference = read(prefix / "reference.json")
        tm = read(root / f"profiler/timing/trajectory_{rank:02d}.json")
        if (reference["prompt_index"] != traj["prompt_id"] or
            reference["trial_seed"] != traj["seed"] or
            tm["rank"] != rank or tm["prompt_index"] != traj["prompt_id"] or
            tm["trial_seed"] != traj["seed"] or tm["logical_edges"] != 92 or
            tm["reference_passes"] != 1 or tm["source_trunks"] != 8 or
            tm["shared_destination_verifier_evaluations"] != 28 or
            tm["tau_suffix_branches"] != 84 or
            tm["snapshot_storage_bytes"] != (prefix / "reference_snapshots.pt").stat().st_size):
            raise RuntimeError(f"invalid profiler reference/timing rank {rank}")
        actual = {p.stem for p in (prefix / "edges").glob("*.json")}
        if actual != edge_keys:
            raise RuntimeError(f"incomplete edge set at rank {rank}")
        for key in sorted(edge_keys):
            row = read(prefix / "edges" / f"{key}.json")
            if (row["edge_key"] != key or row["prompt_index"] != traj["prompt_id"] or
                row["trial_seed"] != traj["seed"] or row["backend"] != "CUDA_Modal" or
                row["diffusion_NFE"] != 800 or
                abs(row["delta"] - (reference["selected_final_score"] -
                                    row["selected_final_score"])) > 1e-9):
                raise RuntimeError(f"invalid edge {rank}: {key}")
            observations.append({"rank": rank, "prompt_id": traj["prompt_id"],
                                 "seed": traj["seed"], "edge_key": key,
                                 "reference_reward": row["reference_score"],
                                 "edge_reward": row["selected_final_score"],
                                 "delta": row["delta"]})
        timing.append(tm)
    edge_csv = root / "profiler/edge_observations.csv"
    with edge_csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(observations[0]))
        writer.writeheader()
        writer.writerows(observations)
    copyfile(root / "analysis/compilation_cost.csv", root / "profiler/compiler_cost.csv")
    reuse_keys = ("logical_edges", "reference_passes", "source_trunks",
                  "shared_destination_verifier_evaluations", "tau_suffix_branches",
                  "verifier_evaluations_naive_at_destinations",
                  "verifier_evaluations_shared_at_destinations",
                  "diffusion_steps_naive", "diffusion_steps_shared_profiler",
                  "snapshot_storage_bytes", "trajectory_wall_seconds")
    reuse = {key: sum(row[key] for row in timing) for key in reuse_keys}
    reuse["diffusion_work_avoided_fraction"] = 1 - reuse["diffusion_steps_shared_profiler"] / reuse["diffusion_steps_naive"]
    reuse["destination_verifier_work_avoided_fraction"] = 1 - reuse["verifier_evaluations_shared_at_destinations"] / reuse["verifier_evaluations_naive_at_destinations"]
    (root / "profiler/reuse_statistics.json").write_text(json.dumps(reuse, indent=2, sort_keys=True) + "\n")

    if (plan["PSP_in_scope"] or len(plan["policies"]) != 5 or
        plan["seeds"] != [42, 43, 44, 45] or
        plan["prompt_ids"] != [x["prompt_id"] for x in audit_prompts] or
        audit["raw_rows"] != 400 or audit["PSP_scored"] or
        audit["validation_touched"] or audit["test_touched"] or
        audit["external_100_prompt_pool_touched"]):
        raise RuntimeError("audit scope or count invalid")
    for policy in plan["policies"]:
        pid = policy["policy_id"]
        for prompt in audit_prompts:
            for seed in plan["seeds"]:
                row = read(root / f"audit/raw/{pid}/p{prompt['prompt_id']}_s{seed}.json")
                if (row["policy_id"] != pid or row["prompt_index"] != prompt["prompt_id"] or
                    row["trial_seed"] != seed or row["backend"] != "CUDA_Modal" or
                    row["diffusion_NFE"] != 800 or len(row["final_particle_scores"]) != 8):
                    raise RuntimeError(f"invalid audit row: {pid}/{prompt['prompt_id']}/{seed}")
    if len(list((root / "plots").glob("*.png"))) != 6:
        raise RuntimeError("six plots required")
    copyfile(root / "audit/policy_results.csv", root / "audit/results.csv")
    integrity = {
        "passed": True, "protocol": manifest["protocol"],
        "calibration_trajectories": len(calibration), "audit_train_prompts": len(audit_prompts),
        "audit_seed_count": 4, "reference_records": len(timing),
        "logical_edges_per_trajectory": 92, "edge_observation_records": len(observations),
        "unique_audit_policies": len(plan["policies"]), "audit_raw_records": 400,
        "frozen_audit_plan_sha256": digest(plan_path),
        "compiled_policy_sha256": digest(compiled_path),
        "calibration_trajectories_sha256": digest(root / "calibration_trajectories.json"),
        "audit_prompts_sha256": digest(root / "audit_prompts.json"),
        "replay_checks": 6, "shared_tree_naive_checks": 5, "tau_fork_checks": 3,
        "validation_touched": False, "test_touched": False,
        "external_100_prompt_pool_touched": False, "PSP_scored": False,
        "old_TPU_edge_values_mixed": False, "plot_count": 6,
    }
    (root / "integrity.json").write_text(json.dumps(integrity, indent=2, sort_keys=True) + "\n")
    return {**integrity, "reuse": reuse}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    print(json.dumps(finish(parser.parse_args().root), indent=2, sort_keys=True))
