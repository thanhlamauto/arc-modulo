"""Read profiler-produced edge weights with strict catalogue integrity."""
from __future__ import annotations

import csv
from pathlib import Path

from scripts.static_bfs_graph.graph_builder import all_edges


def load_edge_table(path: Path, steps: tuple[int, ...], taus: tuple[float, ...]):
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    edges = all_edges(steps, taus)
    if len(rows) != len(edges) or [row["edge_key"] for row in rows] != [e.key for e in edges]:
        raise RuntimeError("edge-table catalogue mismatch")
    return {row["edge_key"]: float(row["mean_delta"]) for row in rows}
