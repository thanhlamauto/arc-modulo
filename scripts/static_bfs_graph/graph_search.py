"""Exact small-DAG path search for additive and lexicographic minimax objectives."""
from __future__ import annotations

from .graph_builder import Edge, all_paths, policy_from_path


def predicted_cost(edges: tuple[Edge, ...], values: dict[str, float]) -> float:
    return sum(values[e.key] for e in edges)


def select_paths(candidate_steps: tuple[int, ...], taus: tuple[float, ...],
                 values: dict[str, float], particles: int, k: int) -> dict:
    paths = list(all_paths(candidate_steps, taus, k))
    assert paths
    assert all(e.key in values for path in paths for e in path)
    additive = min(paths, key=lambda p: (predicted_cost(p, values),
                                         policy_from_path(p, particles).id))
    lexicographic = min(paths, key=lambda p: (tuple(sorted((values[e.key] for e in p), reverse=True)),
                                              policy_from_path(p, particles).id))
    return {"n_paths": len(paths), "additive": additive, "lexicographic": lexicographic}
