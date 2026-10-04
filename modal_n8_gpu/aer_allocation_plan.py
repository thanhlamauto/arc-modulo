"""Freeze nested G5/G10 proposals and disjoint TRAIN pools before new GPU work."""
from __future__ import annotations

from collections import Counter
import csv
from hashlib import sha256
import json
import os
from pathlib import Path

import numpy as np

from scripts.static_bfs_graph.graph_builder import all_edges, all_paths, path_edges, policy_from_path
from scripts.static_bfs_graph.policy import StaticBFSPolicy

from .aer_plan import digest, read, write_csv_once, write_once

PROTOCOL = "STATIC_BFS_N8_AER_BUDGET_ALLOCATION_V1"
STEPS = (10, 20, 30, 40, 60, 80, 90)
TAUS = (2.0, 8.0, 32.0)
N_BOOT = 5000
BOOT_SEED = 20261003
MASS_TARGET = .90
MAX_GRAPH_CANDIDATES = 16
BATCH_SIZE = 5
MAX_ROUNDS = 6
TOTAL_DELTA = .05
PATH_EQUIV_EPS = .01
PAIR_BOOTSTRAPS = 100_000
AUDIT_BOOTSTRAPS = 20_000
RACING_SEEDS = tuple(range(3000, 3030))
AUDIT_SEEDS = tuple(range(4000, 4020))
RATE_USD_PER_SECOND = .000542 + 8 * .0000131 + 24 * .00000222
MAX_NEW_USD = float(os.environ.get("AER_ALLOCATION_MAX_NEW_USD", "12"))


def policy_row(path, count: int, source: str) -> dict:
    policy = policy_from_path(path, 8)
    return {"policy_id": policy.id, "steps": list(policy.resampling_steps),
            "taus": [tau for _, tau in policy.temperature_by_step],
            "edge_keys": [e.key for e in path], "bootstrap_count": count,
            "bootstrap_frequency": count / N_BOOT, "source": source}


def choose(counts: Counter, paths, policies, top1: int, target: float) -> tuple[list[int], float, bool]:
    ranked = sorted(counts, key=lambda i: (-counts[i], policies[i].id))
    selected = []
    for i in ranked:
        if len(selected) >= MAX_GRAPH_CANDIDATES:
            break
        selected.append(i)
        if len(selected) >= 2 and sum(counts[j] for j in selected) / N_BOOT >= target:
            break
    if len(selected) < 2:
        for i in range(len(paths)):
            if i not in selected:
                selected.append(i)
            if len(selected) == 2:
                break
    if top1 not in selected:
        if len(selected) == MAX_GRAPH_CANDIDATES:
            selected.pop()
        selected.append(top1)
    selected.sort(key=lambda i: (-counts[i], policies[i].id))
    mass = sum(counts[i] for i in selected) / N_BOOT
    return selected, mass, mass < target


def _rank_average(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    out = np.empty(len(values), dtype=float)
    k = 0
    while k < len(order):
        j = k + 1
        while j < len(order) and values[order[j]] == values[order[k]]:
            j += 1
        out[order[k:j]] = (k + j - 1) / 2
        k = j
    return out


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(_rank_average(a), _rank_average(b))[0, 1])


def _entropy(counts: Counter) -> float:
    probs = np.asarray(list(counts.values()), dtype=float) / N_BOOT
    return float(-(probs * np.log2(probs)).sum())


def build(repo: Path, root: Path) -> dict:
    prior = repo / "results/static-bfs-graph/n8-offline-compiler-v1"
    integrity_path = prior / "integrity.json"
    integrity = read(integrity_path)
    observations = prior / "profiler/edge_observations.csv"
    calibrations_path = prior / "calibration_trajectories.json"
    calibration = read(calibrations_path)
    pool_path = repo / "modal_n8_gpu/train_pool.json"
    pool = read(pool_path)
    paths = sorted(all_paths(STEPS, TAUS, 3), key=lambda p: policy_from_path(p, 8).id)
    policies = [policy_from_path(p, 8) for p in paths]
    edges = all_edges(STEPS, TAUS)
    edge_index = {e.key: i for i, e in enumerate(edges)}
    path_index = np.asarray([[edge_index[e.key] for e in p] for p in paths], dtype=int)
    if (not integrity["passed"] or integrity["edge_observation_records"] != 1840 or
        digest(calibrations_path) != integrity["calibration_trajectories_sha256"] or
        len(calibration) != 20 or len(edges) != 92 or len(paths) != 945 or
        len(pool["train_indices"]) != 60):
        raise RuntimeError("historical graph source integrity or geometry failed")
    matrix = np.full((10, 92), np.nan)
    seen = set()
    with observations.open(newline="") as stream:
        for row in csv.DictReader(stream):
            rank = int(row["rank"])
            if rank >= 10:
                continue
            item = calibration[rank]
            key = (rank, row["edge_key"])
            if (key in seen or row["edge_key"] not in edge_index or
                int(row["prompt_id"]) != item["prompt_id"] or
                int(row["seed"]) != item["seed"]):
                raise RuntimeError("duplicate or mismatched historical edge")
            seen.add(key)
            matrix[rank, edge_index[row["edge_key"]]] = float(row["delta"])
    if len(seen) != 920 or not np.isfinite(matrix).all():
        raise RuntimeError("G10 historical edge observations incomplete")

    historical_policies = {row["n"]: row["policy_id"]
                           for row in read(prior / "graphs/policies_by_n.json")}
    cost_rows = {int(row["n"]): row for row in csv.DictReader(
        (prior / "analysis/compilation_cost.csv").open(newline=""))}
    if not {5, 10} <= set(cost_rows):
        raise RuntimeError("historical compiler timing absent")

    transfer = StaticBFSPolicy((40, 60, 90),
                               ((40, 8.0), (60, 8.0), (90, 32.0)), 8)
    transfer_path = path_edges((40, 60, 90), (8.0, 8.0, 32.0))
    proposals = {}
    counts_by_arm = {}
    mass_rows = []
    means_by_arm = {}
    for n in (5, 10):
        means = matrix[:n].mean(axis=0)
        means_by_arm[n] = means
        top1 = int(np.argmin(means[path_index].sum(axis=1)))
        if policies[top1].id != historical_policies[n]:
            raise RuntimeError(f"G{n} top path does not reproduce historical compiler")
        rng = np.random.default_rng(BOOT_SEED)
        samples = rng.integers(0, n, size=(N_BOOT, n))
        winners = []
        for sample in samples:
            weights = np.bincount(sample, minlength=n) / n
            winners.append(int(np.argmin((weights @ matrix[:n])[path_index].sum(axis=1))))
        counts = Counter(winners)
        counts_by_arm[n] = counts
        selected, mass, undercoverage = choose(counts, paths, policies, top1, MASS_TARGET)
        graph = [policy_row(paths[i], counts[i], "coarse_bootstrap") for i in selected]
        final = list(graph)
        if transfer.id not in {x["policy_id"] for x in final}:
            final.append(policy_row(transfer_path, 0, "fixed_incumbent"))
        entropy = _entropy(counts)
        proposal = {"protocol": PROTOCOL, "coarse_n": n,
                    "source_edge_observations_sha256": digest(observations),
                    "source_calibration_trajectories_sha256": digest(calibrations_path),
                    "proposal_bootstraps": N_BOOT, "bootstrap_seed": BOOT_SEED,
                    "mass_target": MASS_TARGET, "max_graph_candidates": MAX_GRAPH_CANDIDATES,
                    "bootstrap_mass_covered": mass, "proposal_undercoverage": undercoverage,
                    "bootstrap_entropy_bits": entropy,
                    "bootstrap_effective_paths": float(2**entropy),
                    "coarse_top1_policy_id": policies[top1].id,
                    "incumbent_policy_id": transfer.id,
                    "graph_candidates": graph, "candidates": final,
                    "all_bootstrap_winner_counts": {policies[i].id: count
                                                    for i, count in counts.items()}}
        write_once(root / f"proposal/g{n}_candidates.json", proposal)
        proposals[n] = proposal
        for target in (.70, .80, .90, .95):
            indices, covered, capped = choose(counts, paths, policies, top1, target)
            mass_rows.append({"coarse_n": n, "mass_target": target,
                              "graph_candidates": len(indices),
                              "actual_mass": covered, "undercoverage": capped,
                              "bootstrap_entropy_bits": entropy})
    write_csv_once(root / "proposal/bootstrap_mass_summary.csv", mass_rows)

    graph5 = {x["policy_id"] for x in proposals[5]["graph_candidates"]}
    graph10 = {x["policy_id"] for x in proposals[10]["graph_candidates"]}
    overlap = {"g5_top1": proposals[5]["coarse_top1_policy_id"],
               "g10_top1": proposals[10]["coarse_top1_policy_id"],
               "top1_equal": proposals[5]["coarse_top1_policy_id"] ==
                             proposals[10]["coarse_top1_policy_id"],
               "graph_candidate_intersection": sorted(graph5 & graph10),
               "graph_candidate_union_count": len(graph5 | graph10),
               "graph_candidate_jaccard": len(graph5 & graph10) / len(graph5 | graph10),
               "edge_rank_spearman_g5_g10": _spearman(means_by_arm[5], means_by_arm[10])}
    write_once(root / "proposal/candidate_overlap.json", overlap)
    for n in (5, 10):
        final = proposals[n]["candidates"]
        sets = [set(item["edge_keys"]) for item in final]
        common = set.intersection(*sets)
        difference = sorted(set.union(*sets) - common)
        write_once(root / f"edges/d{n}.json", {
            "protocol": PROTOCOL, "coarse_n": n,
            "candidate_file_sha256": digest(root / f"proposal/g{n}_candidates.json"),
            "candidate_count": len(final), "common_edges": sorted(common),
            "discriminating_edges": difference, "discriminating_edge_count": len(difference)})
    d5 = read(root / "edges/d5.json")["discriminating_edges"]
    d10 = read(root / "edges/d10.json")["discriminating_edges"]
    union = sorted(set(d5) | set(d10))
    if not union or not set(union) <= set(edge_index):
        raise RuntimeError("invalid discriminating edge union")
    write_once(root / "edges/d_union.json", {
        "protocol": PROTOCOL, "d5_sha256": digest(root / "edges/d5.json"),
        "d10_sha256": digest(root / "edges/d10.json"),
        "discriminating_edges": union, "discriminating_edge_count": len(union)})

    old_ids = {item["prompt_id"] for item in calibration[:10]}
    eligible = [int(i) for i in pool["train_indices"] if int(i) not in old_ids]
    eligible.sort(key=lambda i: sha256(
        f"{PROTOCOL}|prompt|{i}|{pool['prompts'][str(i)]}".encode()).hexdigest())
    if len(eligible) != 50:
        raise RuntimeError("fresh prompt pool not exactly 50 TRAIN prompts")
    def pair_rows(ids, seeds):
        return [{"rank": rank, "prompt_id": pid, "seed": seed,
                 "prompt_sha256": sha256(pool["prompts"][str(pid)].encode()).hexdigest()}
                for rank, (pid, seed) in enumerate(zip(ids, seeds))]
    racing = pair_rows(eligible[:30], RACING_SEEDS)
    audit = pair_rows(eligible[30:], AUDIT_SEEDS)
    if (len({(x["prompt_id"], x["seed"]) for x in racing + audit}) != 50 or
        len({x["prompt_id"] for x in racing + audit + calibration[:10]}) != 60):
        raise RuntimeError("fresh racing/audit pool overlaps historical coarse pair")
    write_once(root / "racing/common_racing_pool.json", {
        "protocol": PROTOCOL, "trajectories": racing,
        "batch_size": BATCH_SIZE, "checkpoints": [5, 10, 15, 20, 25, 30],
        "d_union_sha256": digest(root / "edges/d_union.json"),
        "max_rounds": MAX_ROUNDS, "total_delta": TOTAL_DELTA,
        "equivalence_margin": PATH_EQUIV_EPS,
        "paired_bootstraps": PAIR_BOOTSTRAPS,
        "confidence_procedure": "paired percentile bootstrap; two-sided Bonferroni over all frozen pairs and six rounds"})
    write_once(root / "audit/audit_pool.json", {
        "protocol": PROTOCOL, "pairs": audit,
        "bootstrap_resamples": AUDIT_BOOTSTRAPS,
        "freeze_before_racing": True})
    write_once(root / "analysis/difficulty_design.json", {
        "protocol": PROTOCOL, "epsilon": 1e-6,
        "definition": "sample variance of paired path-cost difference divided by max(abs(mean gap), epsilon)^2",
        "interpretation": "descriptive only; not a theorem or decision criterion"})
    write_once(root / "experiment_manifest.json", {
        "protocol": PROTOCOL, "branch": "ngocminh", "base_commit": "34f17bd984921aeba046e8a5b7936e4a7a8346c0",
        "backend": "CUDA_Modal", "GPU": "L40S", "model": "SD1.5", "dtype": "bfloat16",
        "N": 8, "K": 3, "candidate_steps": list(STEPS), "taus": list(TAUS),
        "DDIM_steps": 100, "eta": 1.0, "scoring": "max", "resampling": "ssp",
        "verifier": "ImageReward-v1.0", "additive_path_cost": True,
        "source_integrity_sha256": digest(integrity_path),
        "source_edge_observations_sha256": digest(observations),
        "source_calibration_trajectories_sha256": digest(calibrations_path),
        "source_compilation_cost_sha256": digest(prior / "analysis/compilation_cost.csv"),
        "source_train_pool_sha256": digest(pool_path),
        "g5_historical_gpu_seconds": float(cost_rows[5]["summed_gpu_seconds"]),
        "g10_historical_gpu_seconds": float(cost_rows[10]["summed_gpu_seconds"]),
        "g5_historical_estimated_usd": float(cost_rows[5]["estimated_modal_usd"]),
        "g10_historical_estimated_usd": float(cost_rows[10]["estimated_modal_usd"]),
        "max_new_gpu_usd": MAX_NEW_USD,
        "new_gpu_cost_includes": ["union_edge_racing", "independent_audit"],
        "g5_candidates_sha256": digest(root / "proposal/g5_candidates.json"),
        "g10_candidates_sha256": digest(root / "proposal/g10_candidates.json"),
        "d_union_sha256": digest(root / "edges/d_union.json"),
        "racing_pool_sha256": digest(root / "racing/common_racing_pool.json"),
        "audit_pool_sha256": digest(root / "audit/audit_pool.json"),
        "VALIDATION_touched": False, "TEST_touched": False,
        "external_100_prompt_pool_touched": False,
        "historical_G5_G10_reused_without_GPU": True})
    return {"g5_graph_candidates": len(proposals[5]["graph_candidates"]),
            "g5_mass": proposals[5]["bootstrap_mass_covered"],
            "g10_graph_candidates": len(proposals[10]["graph_candidates"]),
            "g10_mass": proposals[10]["bootstrap_mass_covered"],
            "d5": len(d5), "d10": len(d10), "d_union": len(union),
            "edge_rank_spearman": overlap["edge_rank_spearman_g5_g10"],
            "candidate_jaccard": overlap["graph_candidate_jaccard"],
            "racing_pool_sha256": digest(root / "racing/common_racing_pool.json"),
            "audit_pool_sha256": digest(root / "audit/audit_pool.json")}


if __name__ == "__main__":
    repo = Path(__file__).resolve().parents[1]
    root = repo / "results/static-bfs-graph/aer-budget-allocation-v1"
    print(json.dumps(build(repo, root), indent=2, sort_keys=True))
