"""Deterministic additive K=3 shortest path over the frozen DAG."""
from __future__ import annotations

from scripts.static_bfs_graph.graph_builder import all_paths, policy_from_path


def compile_from_weights(weights: dict[str, float], steps: tuple[int, ...],
                         taus: tuple[float, ...], particles: int = 8, k: int = 3):
    candidates = []
    for path in all_paths(steps, taus, k):
        policy = policy_from_path(path, particles)
        candidates.append((sum(weights[edge.key] for edge in path), policy.id, path, policy))
    if len(candidates) != 945:
        raise RuntimeError("frozen K=3 path count changed")
    score, _, path, policy = min(candidates, key=lambda x: (x[0], x[1]))
    return {"policy_id": policy.id, "steps": list(policy.resampling_steps),
            "taus": [tau for _, tau in policy.temperature_by_step],
            "edge_ids": [edge.key for edge in path], "predicted_regret": score}
