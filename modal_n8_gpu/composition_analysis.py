"""TRAIN-only paired edge-estimation versus composition diagnostic."""
from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr


BOOTSTRAPS = 20_000
BOOT_SEED = 20261003
RATE_USD_PER_SECOND = 0.000542 + 8 * 0.0000131 + 24 * 0.00000222


def read(path: Path):
    return json.loads(path.read_text())


def digest(path: Path):
    return sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, records: list[dict]):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def bootstrap_summary(values: np.ndarray, picks: np.ndarray):
    means = values[picks].mean(axis=1)
    return {"mean": float(values.mean()),
            "ci95": [float(x) for x in np.quantile(means, [.025, .975])],
            "SEM": float(values.std(ddof=1) / np.sqrt(len(values))),
            "bootstrap_probability_mean_gt_zero": float(np.mean(means > 0))}


def analyze(root: Path, prior: Path):
    plan_path = root / "plan.json"
    plan = read(plan_path)
    source_plan = prior / "audit/audit_policy_plan.json"
    source_compiled = prior / "compiled/compiled_policy.json"
    source_edges = prior / "profiler/edge_observations.csv"
    source_manifest = prior / "experiment_manifest.json"
    source_pool = Path(__file__).resolve().parent / "train_pool.json"
    if (plan["protocol"] != "STATIC_BFS_N8_COMPOSITION_DIAGNOSTIC_V1" or
        plan["N"] != 8 or plan["K"] != 3 or len(plan["trajectories"]) != 20 or
        len(plan["edge_union"]) != 12 or len(plan["policies"]) != 4 or
        plan["PSP_in_scope"] or plan["VALIDATION_touched"] or plan["TEST_touched"] or
        plan["external_100_prompt_pool_touched"] or
        digest(source_plan) != plan["source_audit_plan_sha256"] or
        digest(source_compiled) != plan["source_compiled_policy_sha256"] or
        digest(source_edges) != plan["source_edge_observations_sha256"] or
        digest(source_manifest) != plan["source_manifest_sha256"] or
        digest(source_pool) != plan["source_train_pool_sha256"]):
        raise RuntimeError("source or frozen diagnostic plan changed")
    fit = read(prior / "calibration_trajectories.json")
    audit = read(prior / "audit_prompts.json")
    used = {x["prompt_id"] for x in fit + audit}
    new_ids = [x["prompt_id"] for x in plan["trajectories"]]
    pool = read(source_pool)
    if (len(set(new_ids)) != 20 or set(new_ids) & used or
        any(sum(item["seed"] == seed for item in plan["trajectories"]) != 5
            for seed in (42, 43, 44, 45)) or
        any(sha256(pool["prompts"][str(item["prompt_id"])].encode()).hexdigest()
            != item["prompt_sha256"] for item in plan["trajectories"])):
        raise RuntimeError("new diagnostic prompts overlap prior use")

    keys = plan["edge_union"]
    names = [item["name"] for item in plan["policies"]]
    key_index = {key: i for i, key in enumerate(keys)}
    old = np.full((20, len(keys)), np.nan)
    for row in read_csv(source_edges):
        if row["edge_key"] in key_index:
            old[int(row["rank"]), key_index[row["edge_key"]]] = float(row["delta"])
    if not np.isfinite(old).all():
        raise RuntimeError("old edge matrix incomplete")
    new = np.full((20, len(keys)), np.nan)
    predicted = np.full((20, 4), np.nan)
    actual = np.full((20, 4), np.nan)
    residual = np.full((20, 4), np.nan)
    full_reward = np.full((20, 4), np.nan)
    reference_rewards = []
    timings = []
    paired_rows = []
    for rank, trajectory in enumerate(plan["trajectories"]):
        folder = root / f"trajectories/rank_{rank:02d}"
        p, seed = trajectory["prompt_id"], trajectory["seed"]
        reference = read(folder / "reference.json")
        timing = read(folder / "timing.json")
        if (reference["prompt_index"] != p or reference["trial_seed"] != seed or
            reference["backend"] != "CUDA_Modal" or reference["diffusion_NFE"] != 800 or
            timing["prompt_id"] != p or timing["seed"] != seed or
            timing["edge_count"] != 12 or timing["full_policy_count"] != 4):
            raise RuntimeError(f"invalid paired trajectory {rank}")
        if {path.stem for path in (folder / "edges").glob("*.json")
            if not path.name.startswith("._")} != set(keys):
            raise RuntimeError(f"targeted edge set incomplete at rank {rank}")
        for key in keys:
            edge = read(folder / "edges" / f"{key}.json")
            if (edge["edge_key"] != key or edge["prompt_index"] != p or
                edge["trial_seed"] != seed or edge["backend"] != "CUDA_Modal" or
                edge["diffusion_NFE"] != 800 or
                not np.isclose(edge["delta"], reference["selected_final_score"] -
                               edge["selected_final_score"], atol=1e-10, rtol=0)):
                raise RuntimeError(f"invalid edge record {rank}:{key}")
            new[rank, key_index[key]] = float(edge["delta"])
        for pi, item in enumerate(plan["policies"]):
            row = read(folder / "full_policies" / f"{item['name']}.json")
            if (row["policy_id"] != item["policy_id"] or row["prompt_id"] != p or
                row["seed"] != seed or row["edge_keys"] != item["edge_keys"] or
                not np.isclose(row["predicted_regret"],
                               sum(new[rank, key_index[key]] for key in item["edge_keys"]),
                               atol=1e-10, rtol=0) or
                not np.isclose(row["actual_regret"],
                               reference["selected_final_score"] - row["full_policy_reward"],
                               atol=1e-10, rtol=0) or
                not np.isclose(row["composition_residual"],
                               row["actual_regret"] - row["predicted_regret"],
                               atol=1e-10, rtol=0)):
                raise RuntimeError(f"invalid full-policy pairing {rank}:{item['name']}")
            predicted[rank, pi] = float(row["predicted_regret"])
            actual[rank, pi] = float(row["actual_regret"])
            residual[rank, pi] = float(row["composition_residual"])
            full_reward[rank, pi] = float(row["full_policy_reward"])
            paired_rows.append({"rank": rank, "prompt_id": p, "seed": seed,
                                "policy": item["name"],
                                "reference_reward": reference["selected_final_score"],
                                "full_policy_reward": row["full_policy_reward"],
                                "predicted_regret": row["predicted_regret"],
                                "actual_regret": row["actual_regret"],
                                "composition_residual": row["composition_residual"]})
        reference_rewards.append(reference["selected_final_score"])
        timings.append(timing)
    if not all(np.isfinite(x).all() for x in (new, predicted, actual, residual, full_reward)):
        raise RuntimeError("missing or nonfinite paired measurements")
    gate = read(root / "correctness_gate.json")
    cost = read(root / "cost_gate.json")
    if not gate["passed"] or not gate["reference_exact"] or not gate["edge_exact"] or not cost["allowed"]:
        raise RuntimeError("correctness/cost gate invalid")

    rng = np.random.default_rng(BOOT_SEED)
    old_picks = rng.integers(0, 20, size=(BOOTSTRAPS, 20))
    new_picks = rng.integers(0, 20, size=(BOOTSTRAPS, 20))
    old_boot = old[old_picks].mean(axis=1)
    new_boot = new[new_picks].mean(axis=1)
    edge_rows = []
    for ki, key in enumerate(keys):
        difference = new_boot[:, ki] - old_boot[:, ki]
        edge_rows.append({"edge_key": key,
                          "old_mean_delta": float(old[:, ki].mean()),
                          "new_mean_delta": float(new[:, ki].mean()),
                          "new_minus_old": float(new[:, ki].mean() - old[:, ki].mean()),
                          "new_minus_old_ci95_low": float(np.quantile(difference, .025)),
                          "new_minus_old_ci95_high": float(np.quantile(difference, .975)),
                          "old_SEM": float(old[:, ki].std(ddof=1) / np.sqrt(20)),
                          "new_SEM": float(new[:, ki].std(ddof=1) / np.sqrt(20))})
    old_means, new_means = old.mean(axis=0), new.mean(axis=0)
    nontrivial = np.array([not (np.all(old[:, k] == 0) and np.all(new[:, k] == 0))
                           for k in range(len(keys))])
    edge_comparison = {
        "all_12_spearman": float(spearmanr(old_means, new_means).statistic),
        "nontrivial_edge_count": int(nontrivial.sum()),
        "nontrivial_spearman": float(spearmanr(old_means[nontrivial],
                                                new_means[nontrivial]).statistic),
        "nontrivial_pearson": float(pearsonr(old_means[nontrivial],
                                              new_means[nontrivial]).statistic),
        "nontrivial_mean_absolute_change": float(np.abs(old_means[nontrivial] -
                                                          new_means[nontrivial]).mean()),
        "nontrivial_sign_agreement": float(np.mean(
            np.sign(old_means[nontrivial]) == np.sign(new_means[nontrivial]))),
        "edge_changes_CI_excludes_zero": [row["edge_key"] for row in edge_rows
            if row["new_minus_old_ci95_low"] > 0 or row["new_minus_old_ci95_high"] < 0],
    }

    picks = rng.integers(0, 20, size=(BOOTSTRAPS, 20))
    policy_rows = []
    for pi, item in enumerate(plan["policies"]):
        pkeys = [key_index[key] for key in item["edge_keys"]]
        pred_old = old[:, pkeys].sum(axis=1)
        summary = bootstrap_summary(residual[:, pi], picks)
        corr = float(pearsonr(predicted[:, pi], actual[:, pi]).statistic)
        policy_rows.append({"policy": item["name"],
                            "old_mean_predicted_regret": float(pred_old.mean()),
                            "new_mean_predicted_regret": float(predicted[:, pi].mean()),
                            "new_mean_actual_regret": float(actual[:, pi].mean()),
                            "new_mean_full_policy_reward": float(full_reward[:, pi].mean()),
                            "residual_mean": summary["mean"],
                            "residual_ci95_low": summary["ci95"][0],
                            "residual_ci95_high": summary["ci95"][1],
                            "residual_SEM": summary["SEM"],
                            "residual_MAE": float(np.abs(residual[:, pi]).mean()),
                            "predicted_actual_prompt_pearson": corr,
                            "residual_positive_prompts": int((residual[:, pi] > 0).sum())})
    transfer_i = names.index("TRANSFER4_TO_8")
    p20_i = names.index("P20")
    p20_keys = [key_index[key] for key in plan["policies"][p20_i]["edge_keys"]]
    transfer_keys = [key_index[key] for key in plan["policies"][transfer_i]["edge_keys"]]
    old_margin = old[:, p20_keys].sum(axis=1) - old[:, transfer_keys].sum(axis=1)
    new_margin = predicted[:, p20_i] - predicted[:, transfer_i]
    margin_shift_boot = (new_margin[new_picks].mean(axis=1) -
                         old_margin[old_picks].mean(axis=1))
    contrast = {
        "definition": "(actual−predicted regret)_P20 − (actual−predicted regret)_TRANSFER4_TO_8",
        **bootstrap_summary(residual[:, p20_i] - residual[:, transfer_i], picks),
    }
    pair = {
        "old_predicted_P20_minus_transfer": bootstrap_summary(old_margin, old_picks),
        "predicted_margin_shift_new_minus_old": {
            "mean": float(new_margin.mean() - old_margin.mean()),
            "ci95": [float(x) for x in np.quantile(margin_shift_boot, [.025, .975])],
            "bootstrap_probability_mean_gt_zero": float(np.mean(margin_shift_boot > 0))},
        "new_predicted_P20_minus_transfer": bootstrap_summary(
            new_margin, picks),
        "new_actual_regret_P20_minus_transfer": bootstrap_summary(
            actual[:, p20_i] - actual[:, transfer_i], picks),
        "new_full_reward_P20_minus_transfer": bootstrap_summary(
            full_reward[:, p20_i] - full_reward[:, transfer_i], picks),
        "composition_residual_contrast": contrast,
    }
    old_path_means = np.array([row["old_mean_predicted_regret"] for row in policy_rows])
    new_path_means = predicted.mean(axis=0)
    actual_means = actual.mean(axis=0)
    rankings = {
        "old_graph_predicted_best_to_worst": [names[i] for i in np.argsort(old_path_means)],
        "new_graph_predicted_best_to_worst": [names[i] for i in np.argsort(new_path_means)],
        "new_full_policy_actual_best_to_worst": [names[i] for i in np.argsort(actual_means)],
        "old_predicted_vs_new_actual_spearman": float(spearmanr(old_path_means,
                                                                 actual_means).statistic),
        "new_predicted_vs_new_actual_spearman": float(spearmanr(new_path_means,
                                                                 actual_means).statistic),
    }
    seconds = sum(row["trajectory_wall_seconds"] for row in timings)
    output = root / "analysis"
    output.mkdir(exist_ok=True)
    write_csv(output / "paired_prompt_policy_rows.csv", paired_rows)
    write_csv(output / "edge_old_vs_new.csv", edge_rows)
    write_csv(output / "policy_residuals.csv", policy_rows)
    summary = {
        "protocol": plan["protocol"], "plan_sha256": digest(plan_path),
        "source_edge_observations_sha256": digest(source_edges),
        "correctness_gate": gate, "cost_gate": cost,
        "new_trajectories": 20, "new_unique_train_prompts": 20,
        "logical_edge_records": 20 * len(keys), "full_policy_records": 20 * 4,
        "reference_records": 20, "paired_statistical_unit": "one TRAIN prompt with one frozen seed",
        "bootstrap_resamples": BOOTSTRAPS, "bootstrap_seed": BOOT_SEED,
        "edge_comparison": edge_comparison, "policy_residuals": policy_rows,
        "p20_vs_transfer": pair, "rankings": rankings,
        "summed_GPU_seconds": seconds, "summed_GPU_hours": seconds / 3600,
        "estimated_GPU_USD_at_frozen_rate": seconds * RATE_USD_PER_SECOND,
        "snapshot_storage_bytes": sum(row["snapshot_bytes"] for row in timings),
        "VALIDATION_touched": False, "TEST_touched": False, "PSP_scored": False,
        "no_new_policy_search": True,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (root / "integrity.json").write_text(json.dumps({
        "passed": True, "plan_sha256": digest(plan_path),
        "source_train_pool_sha256": digest(source_pool),
        "source_manifest_sha256": digest(source_manifest),
        "source_audit_plan_sha256": digest(source_plan),
        "source_compiled_policy_sha256": digest(source_compiled),
        "source_edge_observations_sha256": digest(source_edges),
        "reference_records": 20, "edge_records": 240, "full_policy_records": 80,
        "new_train_prompts": 20, "old_fit_and_audit_disjoint": True,
        "seed_counts": {str(seed): 5 for seed in (42, 43, 44, 45)},
        "same_prompt_seed_paired": True, "correctness_gate_passed": True,
        "VALIDATION_touched": False, "TEST_touched": False,
        "external_100_prompt_pool_touched": False,
    }, indent=2, sort_keys=True) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("new_run", type=Path)
    parser.add_argument("prior_run", type=Path)
    print(json.dumps(analyze(parser.parse_args().new_run, parser.parse_args().prior_run),
                     indent=2, sort_keys=True))
