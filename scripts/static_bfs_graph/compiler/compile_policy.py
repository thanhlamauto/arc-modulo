"""Compile a deployable BFS schedule from an edge-value CSV alone."""
from __future__ import annotations

from pathlib import Path

from .edge_table import load_edge_table
from .graph_search import compile_from_weights


def compile_edge_table(path: Path, *, steps=(10, 20, 30, 40, 60, 80, 90),
                       taus=(2.0, 8.0, 32.0), particles=8, k=3):
    weights = load_edge_table(path, steps, taus)
    return compile_from_weights(weights, steps, taus, particles, k)
