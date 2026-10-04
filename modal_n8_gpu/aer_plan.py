"""Freeze AER proposals and disjoint TRAIN prompt/seed pairs before new scoring."""
from __future__ import annotations

from collections import Counter
import csv
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from scripts.static_bfs_graph.graph_builder import all_edges, all_paths, policy_from_path, path_edges
from scripts.static_bfs_graph.policy import StaticBFSPolicy


PROTOCOL = "STATIC_BFS_N8_ADAPTIVE_EDGE_RACING_V1"
STEPS = (10, 20, 30, 40, 60, 80, 90)
TAUS = (2.0, 8.0, 32.0)
COARSE_N = 5
PROPOSAL_BOOTSTRAPS = 5000
PROPOSAL_SEED = 20261003
BOOTSTRAP_MASS = .90
MIN_CANDIDATES = 2
MAX_CANDIDATES = 8
RACING_SEEDS = tuple(range(1000, 1020))
AUDIT_SEEDS = tuple(range(2000, 2020))
RACING_BATCH_SIZE = 5
MAX_ROUNDS = 4
TOTAL_DELTA = .05
PATH_EQUIV_EPS = .01
PAIR_BOOTSTRAPS = 100_000
AUDIT_BOOTSTRAPS = 20_000
RATE_USD_PER_SECOND = .000542 + 8 * .0000131 + 24 * .00000222
RACING_COST_CAP_USD = 10.0
AUDIT_COST_CAP_USD = 10.0


def read(path: Path):
    return json.loads(path.read_text())


def digest(path: Path):
    return sha256(path.read_bytes()).hexdigest()


def write_once(path: Path, data):
    raw = (json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != raw:
            raise RuntimeError(f"frozen artifact changed: {path}")
    else:
        path.write_bytes(raw)


def write_csv_once(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    from io import StringIO
    stream = StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    raw = stream.getvalue().encode()
    if path.exists():
        if path.read_bytes() != raw:
            raise RuntimeError(f"frozen CSV changed: {path}")
    else:
        path.write_bytes(raw)


def policy_row(path, frequency: int, source: str):
    policy = policy_from_path(path, 8)
    return {"policy_id": policy.id, "steps": list(policy.resampling_steps),
            "taus": [tau for _, tau in policy.temperature_by_step],
            "edge_keys": [edge.key for edge in path], "bootstrap_count": frequency,
            "bootstrap_frequency": frequency / PROPOSAL_BOOTSTRAPS, "source": source}


def select_candidates(counts: Counter, paths, policies, top1_index: int, mass: float):
    ranking = sorted(counts, key=lambda i: (-counts[i], policies[i].id))
    chosen = []
    total = 0
    for i in ranking:
        if len(chosen) >= MAX_CANDIDATES:
            break
        chosen.append(i)
        total += counts[i]
        if total / PROPOSAL_BOOTSTRAPS >= mass and len(chosen) >= MIN_CANDIDATES:
            break
    if len(chosen) < MIN_CANDIDATES:
        for i in range(len(paths)):
            if i not in chosen:
                chosen.append(i)
                if len(chosen) == MIN_CANDIDATES:
                    break
    if top1_index not in chosen:
        if len(chosen) == MAX_CANDIDATES:
            chosen.pop()
        chosen.append(top1_index)
    chosen.sort(key=lambda i: (-counts[i], policies[i].id))
    covered = sum(counts[i] for i in chosen) / PROPOSAL_BOOTSTRAPS
    return chosen, covered


def build(repo: Path, root: Path):
    old = repo / "results/static-bfs-graph/n8-offline-compiler-v1"
    previous_integrity = read(old / "integrity.json")
    if not previous_integrity["passed"]:
        raise RuntimeError("source compiler integrity failed")
    source_edges = old / "profiler/edge_observations.csv"
    prior_fit = read(old / "calibration_trajectories.json")
    pool_path = repo / "modal_n8_gpu/train_pool.json"
    pool = read(pool_path)
    edges = all_edges(STEPS, TAUS)
    paths = sorted(all_paths(STEPS, TAUS, 3),
                   key=lambda p: policy_from_path(p, 8).id)
    policies = [policy_from_path(p, 8) for p in paths]
    if len(edges) != 92 or len(paths) != 945 or len(prior_fit) != 20:
        raise RuntimeError("frozen graph size changed")
    edge_index = {e.key: i for i, e in enumerate(edges)}
    path_indices = np.asarray([[edge_index[e.key] for e in p] for p in paths], dtype=int)
    matrix = np.full((COARSE_N, len(edges)), np.nan)
    with source_edges.open(newline="") as stream:
        for row in csv.DictReader(stream):
            rank = int(row["rank"])
            if rank < COARSE_N:
                traj = prior_fit[rank]
                if int(row["prompt_id"]) != traj["prompt_id"] or int(row["seed"]) != traj["seed"]:
                    raise RuntimeError("Stage A source trajectory mismatch")
                matrix[rank, edge_index[row["edge_key"]]] = float(row["delta"])
    if not np.isfinite(matrix).all() or len({t["prompt_id"] for t in prior_fit[:5]}) != 5:
        raise RuntimeError("incomplete or overlapping Stage A")
    means = matrix.mean(axis=0)
    coarse_path_costs = means[path_indices].sum(axis=1)
    top1_index = int(np.argmin(coarse_path_costs))
    old_p5 = next(row for row in read(old / "graphs/policies_by_n.json") if row["n"] == 5)
    if policies[top1_index].id != old_p5["policy_id"]:
        raise RuntimeError("Stage A top-1 does not reproduce prior G5")

    rng = np.random.default_rng(PROPOSAL_SEED)
    resamples = rng.integers(0, COARSE_N, size=(PROPOSAL_BOOTSTRAPS, COARSE_N))
    winners = []
    for sample in resamples:
        weights = np.bincount(sample, minlength=COARSE_N) / COARSE_N
        costs = (weights @ matrix)[path_indices].sum(axis=1)
        winners.append(int(np.argmin(costs)))
    counts = Counter(winners)
    chosen, covered = select_candidates(counts, paths, policies, top1_index,
                                         BOOTSTRAP_MASS)
    graph_candidates = [policy_row(paths[i], counts[i], "StageA_bootstrap")
                        for i in chosen]
    transfer = StaticBFSPolicy((40, 60, 90),
                               ((40, 8.0), (60, 8.0), (90, 32.0)), 8)
    transfer_path = path_edges((40, 60, 90), (8.0, 8.0, 32.0))
    candidates = list(graph_candidates)
    if transfer.id not in {item["policy_id"] for item in candidates}:
        candidates.append(policy_row(transfer_path, 0, "fixed_incumbent"))
    sets = [set(item["edge_keys"]) for item in candidates]
    common = set.intersection(*sets)
    discriminating = sorted(set.union(*sets) - common)
    if not discriminating or not set(discriminating) <= set(edge_index):
        raise RuntimeError("invalid discriminating-edge set")

    stage_a = [{"rank": i, "prompt_id": item["prompt_id"],
                "seed": item["seed"],
                "prompt_sha256": sha256(pool["prompts"][str(item["prompt_id"])].encode()).hexdigest()}
               for i, item in enumerate(prior_fit[:COARSE_N])]
    if len(pool["train_indices"]) != 60:
        raise RuntimeError("TRAIN pool size changed")
    eligible = [int(i) for i in pool["train_indices"]
                if int(i) not in {item["prompt_id"] for item in stage_a}]
    ordered = sorted(eligible, key=lambda i: sha256(
        f"{PROTOCOL}|prompt|{i}|{pool['prompts'][str(i)]}".encode()).hexdigest())
    if len(ordered) < 40:
        raise RuntimeError("insufficient TRAIN prompt IDs for disjoint racing/audit")
    def plan_rows(ids, seeds):
        return [{"rank": rank, "prompt_id": prompt_id, "seed": seed,
                 "prompt_sha256": sha256(pool["prompts"][str(prompt_id)].encode()).hexdigest()}
                for rank, (prompt_id, seed) in enumerate(zip(ids, seeds))]
    racing_pairs = plan_rows(ordered[:20], RACING_SEEDS)
    audit_pairs = plan_rows(ordered[20:40], AUDIT_SEEDS)
    if len({x["prompt_id"] for x in stage_a + racing_pairs + audit_pairs}) != 45:
        raise RuntimeError("prompt sets not disjoint")

    candidate_file = {
        "protocol": PROTOCOL, "source_edge_observations_sha256": digest(source_edges),
        "source_stage_a_trajectories_sha256": digest(old / "calibration_trajectories.json"),
        "coarse_n": COARSE_N, "proposal_bootstraps": PROPOSAL_BOOTSTRAPS,
        "proposal_seed": PROPOSAL_SEED, "mass_target": BOOTSTRAP_MASS,
        "min_candidates": MIN_CANDIDATES, "max_candidates": MAX_CANDIDATES,
        "coarse_top1_policy_id": policies[top1_index].id,
        "coarse_top1_predicted_regret": float(coarse_path_costs[top1_index]),
        "graph_candidates": graph_candidates, "candidates": candidates,
        "actual_bootstrap_mass_covered": covered,
        "proposal_uncertainty_high": covered < BOOTSTRAP_MASS,
        "incumbent_policy_id": transfer.id,
    }
    write_once(root / "stage_a/candidate_set.json", candidate_file)
    write_csv_once(root / "stage_a/coarse_trajectories.csv", stage_a)
    write_once(root / "stage_a/coarse_trajectories.json", stage_a)
    write_csv_once(root / "stage_a/edge_values.csv", [
        {"edge_key": edge.key, "mean_delta": float(means[i]),
         "sample_std": float(matrix[:, i].std(ddof=1))}
        for i, edge in enumerate(edges)])
    write_csv_once(root / "stage_a/bootstrap_paths.csv", [
        {"bootstrap": j, "policy_id": policies[i].id,
         "steps": json.dumps(list(policies[i].resampling_steps)),
         "taus": json.dumps([t for _, t in policies[i].temperature_by_step])}
        for j, i in enumerate(winners)])
    write_once(root / "racing/discriminating_edges.json", {
        "candidate_set_sha256": digest(root / "stage_a/candidate_set.json"),
        "candidate_count": len(candidates), "all_graph_edges": 92,
        "common_edges": sorted(common), "discriminating_edges": discriminating,
        "discriminating_edge_count": len(discriminating),
        "union_edges": sorted(set.union(*sets)),
    })
    write_once(root / "racing/racing_trajectory_plan.json", {
        "protocol": PROTOCOL, "candidate_set_sha256": digest(root / "stage_a/candidate_set.json"),
        "discriminating_edges_sha256": digest(root / "racing/discriminating_edges.json"),
        "trajectories": racing_pairs, "batch_size": RACING_BATCH_SIZE,
        "max_rounds": MAX_ROUNDS, "max_trajectories": 20,
        "total_delta": TOTAL_DELTA, "equivalence_margin": PATH_EQUIV_EPS,
        "confidence_procedure": "paired percentile bootstrap, Bonferroni across frozen pairs and all four rounds",
        "paired_bootstraps": PAIR_BOOTSTRAPS,
        "racing_cost_cap_estimated_usd": RACING_COST_CAP_USD,
    })
    write_once(root / "audit/audit_plan.json", {
        "protocol": PROTOCOL, "pairs": audit_pairs,
        "policy_aliases": ["AER_WINNER", "COARSE_TOP1", "G20_OLD_TOP1",
                           "TRANSFER4_TO_8", "MANUAL8"],
        "audit_bootstraps": AUDIT_BOOTSTRAPS,
        "audit_cost_cap_estimated_usd": AUDIT_COST_CAP_USD,
        "no_audit_scoring_before_aer_freeze": True,
    })
    manifest = {
        "protocol": PROTOCOL, "repository_branch": "ngocminh",
        "backend": "CUDA_Modal", "GPU": "L40S", "model": "SD1.5",
        "dtype": "bfloat16", "N": 8, "K": 3, "candidate_steps": list(STEPS),
        "taus": list(TAUS), "reference_tau": 8.0, "DDIM_steps": 100,
        "eta": 1.0, "scoring": "max", "resampling": "ssp",
        "verifier": "ImageReward-v1.0", "additive_path_cost": True,
        "stage_a_source": "existing G5 first five complete shared-tree trajectories",
        "stage_a_source_integrity_sha256": digest(old / "integrity.json"),
        "source_train_pool_sha256": digest(pool_path),
        "stage_a_cost_counted_not_new_spend": True,
        "stage_a_estimated_usd": 3.8644423597751207,
        "uniform_G20_estimated_usd": 15.675077735141828,
        "racing_plan_sha256": digest(root / "racing/racing_trajectory_plan.json"),
        "audit_plan_sha256": digest(root / "audit/audit_plan.json"),
        "VALIDATION_touched": False, "TEST_touched": False,
        "external_100_prompt_pool_touched": False, "PSP_in_scope": False,
        "policy_selection_from_audit": False,
    }
    write_once(root / "experiment_manifest.json", manifest)
    return {"coarse_top1": policies[top1_index].id,
            "graph_candidate_count": len(graph_candidates),
            "candidate_count_with_incumbent": len(candidates),
            "discriminating_edges": len(discriminating),
            "proposal_mass_covered": covered,
            "proposal_uncertainty_high": covered < BOOTSTRAP_MASS,
            "candidate_set_sha256": digest(root / "stage_a/candidate_set.json"),
            "racing_plan_sha256": digest(root / "racing/racing_trajectory_plan.json"),
            "audit_plan_sha256": digest(root / "audit/audit_plan.json")}


if __name__ == "__main__":
    repo = Path(__file__).resolve().parents[1]
    root = repo / "results/static-bfs-graph/adaptive-edge-racing-v1"
    print(json.dumps(build(repo, root), indent=2, sort_keys=True))
