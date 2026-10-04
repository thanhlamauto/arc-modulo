"""Post-audit diagnostics of the frozen offline N=8 graph (no GPU calls).

This script reads only the existing TRAIN calibration edge rows and the
pre-frozen TRAIN audit rows. Robust weights are sensitivity analyses; they do
not overwrite the compiled policy or select a deployment method.
"""
from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from scripts.static_bfs_graph.compiler.graph_search import compile_from_weights
from scripts.static_bfs_graph.graph_builder import path_edges


NAMES = ("P1", "P5=P10", "P20", "TRANSFER4_TO_8")
BOOTSTRAPS = 20_000
BOOT_SEED = 20261003
ROBUST_LAMBDAS = (0.0, 0.5, 1.0, 2.0, 4.0)
STEPS = (10, 20, 30, 40, 60, 80, 90)
TAUS = (2.0, 8.0, 32.0)


def read(path: Path):
    return json.loads(path.read_text())


def rows(path: Path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def digest(path: Path):
    return sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, records: list[dict]):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def rho_from_ranks(a: np.ndarray, b: np.ndarray):
    count = a.shape[-1]
    return 1.0 - 6.0 * np.square(a - b).sum(axis=-1) / (count * (count**2 - 1))


def ranks(values: np.ndarray, higher_better: bool):
    ordered = np.argsort(-values if higher_better else values, axis=-1)
    return np.argsort(ordered, axis=-1) + 1


def path_keys(policy: dict):
    return tuple(edge.key for edge in path_edges(tuple(policy["steps"]),
                                                  tuple(policy["taus"])))


def analyze(root: Path):
    manifest = read(root / "experiment_manifest.json")
    integrity = read(root / "integrity.json")
    plan = read(root / "audit/audit_policy_plan.json")
    if (not integrity["passed"] or manifest["N"] != 8 or
        manifest["PSP_in_scope"] or manifest["validation_touched"] or
        manifest["test_touched"] or plan["PSP_in_scope"]):
        raise RuntimeError("frozen TRAIN-only scope not verified")
    graphs = read(root / "graphs/policies_by_n.json")
    if [g["n"] for g in graphs] != [1, 5, 10, 20]:
        raise RuntimeError("graph sizes changed")
    by_n = {g["n"]: g for g in graphs}
    policies = {
        "P1": by_n[1], "P5=P10": by_n[5], "P20": by_n[20],
        "TRANSFER4_TO_8": {"steps": [40, 60, 90], "taus": [8.0, 8.0, 32.0]},
    }
    if by_n[5]["policy_id"] != by_n[10]["policy_id"]:
        raise RuntimeError("P5 and P10 are not aliases")
    paths = {name: path_keys(policies[name]) for name in NAMES}
    if any(len(path) != 4 for path in paths.values()):
        raise RuntimeError("K=3 path invalid")

    observations = rows(root / "profiler/edge_observations.csv")
    if len(observations) != 1840:
        raise RuntimeError("calibration edge matrix incomplete")
    edge_keys = tuple(manifest["edge_keys"])
    edge_index = {key: i for i, key in enumerate(edge_keys)}
    matrix = np.full((20, 92), np.nan)
    for row in observations:
        rank, key = int(row["rank"]), row["edge_key"]
        if not (0 <= rank < 20) or key not in edge_index or np.isfinite(matrix[rank, edge_index[key]]):
            raise RuntimeError("duplicate or unknown calibration edge")
        matrix[rank, edge_index[key]] = float(row["delta"])
    if not np.isfinite(matrix).all():
        raise RuntimeError("missing or nonfinite calibration edge")

    graph_tables = {}
    for n in (1, 5, 10, 20):
        table = {row["edge_key"]: row for row in rows(root / f"graphs/edge_values_n{n}.csv")}
        if set(table) != set(edge_keys):
            raise RuntimeError(f"G{n} edge set incomplete")
        for key in edge_keys:
            if not np.isclose(float(table[key]["mean_delta"]),
                              matrix[:n, edge_index[key]].mean(), atol=1e-10, rtol=0):
                raise RuntimeError(f"G{n} value does not reproduce raw edge {key}")
        graph_tables[n] = table

    audit_prompts = read(root / "audit_prompts.json")
    if len(audit_prompts) != 20 or plan["prompt_ids"] != [p["prompt_id"] for p in audit_prompts]:
        raise RuntimeError("audit prompt order changed")
    prompt_scores = np.empty((20, len(NAMES)))
    for pi, prompt in enumerate(audit_prompts):
        for mi, name in enumerate(NAMES):
            alias = "P5" if name == "P5=P10" else name
            policy_id = plan["policy_aliases"][alias]
            scores = []
            for seed in plan["seeds"]:
                row = read(root / f"audit/raw/{policy_id}/p{prompt['prompt_id']}_s{seed}.json")
                if (row["policy_id"] != policy_id or row["prompt_index"] != prompt["prompt_id"] or
                    row["trial_seed"] != seed):
                    raise RuntimeError("audit row does not match frozen plan")
                scores.append(float(row["selected_final_score"]))
            prompt_scores[pi, mi] = np.mean(scores)
    actual_means = prompt_scores.mean(axis=0)
    actual_rank = ranks(actual_means, higher_better=True)
    rng = np.random.default_rng(BOOT_SEED)
    audit_samples = rng.integers(0, 20, size=(BOOTSTRAPS, 20))
    boot_means = prompt_scores[audit_samples].mean(axis=1)
    boot_ranks = ranks(boot_means, higher_better=True)

    ranking_rows = []
    ranking_summary = []
    for n in (1, 5, 10, 20):
        table = graph_tables[n]
        costs = np.array([sum(float(table[key]["mean_delta"]) for key in paths[name])
                          for name in NAMES])
        predicted_rank = ranks(costs, higher_better=False)
        rho_boot = rho_from_ranks(boot_ranks, predicted_rank)
        rho = float(rho_from_ranks(actual_rank, predicted_rank))
        ranking_summary.append({
            "n": n, "spearman_predicted_vs_audit": rho,
            "audit_bootstrap_spearman_ci95": [float(x) for x in np.quantile(rho_boot, [.025, .975])],
            "predicted_top1": NAMES[int(np.argmin(costs))],
            "audit_top1": NAMES[int(np.argmax(actual_means))],
            "audit_bootstrap_top1_agreement_fraction": float(
                np.mean(np.argmax(boot_means, axis=1) == np.argmin(costs))),
        })
        for i, name in enumerate(NAMES):
            ranking_rows.append({"n": n, "policy": name, "predicted_path_cost": float(costs[i]),
                                 "predicted_rank": int(predicted_rank[i]),
                                 "audit_mean_ImageReward": float(actual_means[i]),
                                 "audit_rank": int(actual_rank[i])})

    # Keep the direct comparison path-paired across the same 20 calibration
    # trajectories. Summing independent edge SEMs would ignore covariance.
    p20, transfer = paths["P20"], paths["TRANSFER4_TO_8"]
    edge_rows = []
    for label, path in (("P20", p20), ("TRANSFER4_TO_8", transfer)):
        for key in path:
            values = matrix[:, edge_index[key]]
            edge_rows.append({"policy": label, "edge_key": key,
                              "mean_delta": float(values.mean()),
                              "SEM": float(values.std(ddof=1) / np.sqrt(20))})
    p20_values = matrix[:, [edge_index[key] for key in p20]].sum(axis=1)
    transfer_values = matrix[:, [edge_index[key] for key in transfer]].sum(axis=1)
    edge_difference_rows = []
    for position, (p20_edge, transfer_edge) in enumerate(zip(p20, transfer), start=1):
        differences = matrix[:, edge_index[p20_edge]] - matrix[:, edge_index[transfer_edge]]
        edge_difference_rows.append({"path_position": position, "P20_edge": p20_edge,
                                     "TRANSFER4_TO_8_edge": transfer_edge,
                                     "mean_cost_difference": float(differences.mean()),
                                     "paired_trajectory_SEM": float(
                                         differences.std(ddof=1) / np.sqrt(20))})
    paired_cost_diff = p20_values - transfer_values
    graph_samples = rng.integers(0, 20, size=(BOOTSTRAPS, 20))
    boot_margin = paired_cost_diff[graph_samples].mean(axis=1)
    margin = {
        "comparison": "P20 minus TRANSFER4_TO_8 additive path cost; negative favors P20",
        "n": 20, "mean_difference": float(paired_cost_diff.mean()),
        "paired_trajectory_SEM": float(paired_cost_diff.std(ddof=1) / np.sqrt(20)),
        "calibration_bootstrap_ci95": [float(x) for x in np.quantile(boot_margin, [.025, .975])],
        "calibration_bootstrap_fraction_favoring_P20": float(np.mean(boot_margin < 0)),
    }

    audit_pairwise = []
    for left, right in (("P20", "P1"), ("P20", "P5=P10"),
                        ("P20", "TRANSFER4_TO_8"), ("P1", "TRANSFER4_TO_8")):
        d = prompt_scores[:, NAMES.index(left)] - prompt_scores[:, NAMES.index(right)]
        boot = d[audit_samples].mean(axis=1)
        audit_pairwise.append({"left": left, "right": right, "mean_reward_difference": float(d.mean()),
                               "audit_prompt_bootstrap_ci95": [float(x) for x in np.quantile(boot, [.025, .975])],
                               "prompt_wins": int((d > 1e-12).sum()),
                               "prompt_ties": int((np.abs(d) <= 1e-12).sum()),
                               "prompt_losses": int((d < -1e-12).sum())})

    robust_rows = []
    for n in (5, 10, 20):
        table = graph_tables[n]
        for lam in ROBUST_LAMBDAS:
            weights = {key: float(row["mean_delta"]) +
                       lam * float(row["sample_std"]) / np.sqrt(n)
                       for key, row in table.items()}
            winner = compile_from_weights(weights, STEPS, TAUS, particles=8, k=3)
            if lam == 0 and winner["policy_id"] != by_n[n]["policy_id"]:
                raise RuntimeError(f"robust diagnostic λ=0 does not reproduce P{n}")
            robust_rows.append({"n": n, "lambda": lam,
                                "selected_policy_id": winner["policy_id"],
                                "steps": winner["steps"], "taus": winner["taus"],
                                "robust_path_cost": winner["predicted_regret"],
                                "among_frozen_audit_policies": winner["policy_id"] in
                                set(plan["policy_aliases"].values())})
    p20_sem_sum = sum(float(graph_tables[20][key]["sample_std"]) / np.sqrt(20) for key in p20)
    transfer_sem_sum = sum(float(graph_tables[20][key]["sample_std"]) / np.sqrt(20)
                           for key in transfer)
    pair_crossover = ((-margin["mean_difference"]) / (p20_sem_sum - transfer_sem_sum)
                      if p20_sem_sum > transfer_sem_sum else None)

    output_dir = root / "diagnostics"
    output_dir.mkdir(exist_ok=True)
    write_csv(output_dir / "predicted_vs_audit_rank.csv", ranking_rows)
    write_csv(output_dir / "p20_vs_transfer_edge_terms.csv", edge_rows)
    write_csv(output_dir / "p20_vs_transfer_edge_differences.csv", edge_difference_rows)
    write_csv(output_dir / "robust_sensitivity.csv", robust_rows)
    result = {
        "scope": "post-audit TRAIN-only descriptive diagnostic; frozen method unchanged",
        "calibration_trajectories": 20, "audit_prompts": 20, "audit_seeds": 4,
        "static_policies_compared": list(NAMES), "MANUAL8_excluded_reason":
        "gamma-increase temperatures are outside the graph's fixed tau grid",
        "audit_policy_plan_sha256": digest(root / "audit/audit_policy_plan.json"),
        "audit_summary_sha256": digest(root / "audit/summary.json"),
        "compiled_policy_sha256": digest(root / "compiled/compiled_policy.json"),
        "edge_observations_sha256": digest(root / "profiler/edge_observations.csv"),
        "graph_tables_sha256": {
            str(n): digest(root / f"graphs/edge_values_n{n}.csv") for n in (1, 5, 10, 20)},
        "ranking": ranking_summary, "p20_vs_transfer_calibration_margin": margin,
        "p20_sum_edge_SEM": p20_sem_sum, "transfer_sum_edge_SEM": transfer_sem_sum,
        "pairwise_robust_lambda_crossover": pair_crossover,
        "audit_pairwise_differences": audit_pairwise,
        "audit_bootstrap_top1_frequency": {
            name: float(np.mean(np.argmax(boot_means, axis=1) == i))
            for i, name in enumerate(NAMES)},
        "audit_bootstrap_fraction_P1_gt_P5_gt_P20": float(np.mean(
            (boot_means[:, NAMES.index("P1")] > boot_means[:, NAMES.index("P5=P10")]) &
            (boot_means[:, NAMES.index("P5=P10")] > boot_means[:, NAMES.index("P20")]))),
        "bootstrap_resamples": BOOTSTRAPS, "bootstrap_seed": BOOT_SEED,
        "robust_lambda_grid": list(ROBUST_LAMBDAS),
        "robust_is_method_change": False,
    }
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    print(json.dumps(analyze(parser.parse_args().root), indent=2, sort_keys=True))
