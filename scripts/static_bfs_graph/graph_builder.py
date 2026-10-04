"""DAG actions and full-rollout edge interventions."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from .policy import StaticBFSPolicy

START = -1
END = 100


@dataclass(frozen=True, order=True)
class Edge:
    src: int
    dst: int
    tau_dst: float | None

    def __post_init__(self):
        assert START <= self.src < self.dst <= END
        assert (self.dst == END) == (self.tau_dst is None)

    @property
    def key(self) -> str:
        tau = "end" if self.tau_dst is None else f"{self.tau_dst:g}"
        return f"{self.src}_to_{self.dst}_tau_{tau}"

    def record(self) -> dict:
        return {"src": self.src, "dst": self.dst, "tau_dst": self.tau_dst,
                "key": self.key}


def all_edges(candidate_steps: tuple[int, ...], taus: tuple[float, ...]) -> tuple[Edge, ...]:
    assert candidate_steps == tuple(sorted(set(candidate_steps)))
    assert all(0 <= x < END for x in candidate_steps)
    assert len(taus) == len(set(taus)) and all(x > 0 for x in taus)
    nodes = (START,) + candidate_steps
    out = [Edge(src, dst, tau) for src in nodes for dst in candidate_steps
           if src < dst for tau in taus]
    out.extend(Edge(src, END, None) for src in nodes)
    return tuple(sorted(out))


def dense_reference(candidate_steps: tuple[int, ...], tau: float, particles: int) -> StaticBFSPolicy:
    return StaticBFSPolicy(candidate_steps, tuple((s, tau) for s in candidate_steps), particles)


def edge_intervention(edge: Edge, reference: StaticBFSPolicy) -> StaticBFSPolicy:
    steps = reference.resampling_steps
    assert edge.src == START or edge.src in steps
    assert edge.dst == END or edge.dst in steps
    assert edge.src < edge.dst
    kept = [s for s in steps if s <= edge.src or s > edge.dst]
    tau = reference.temperature_map
    if edge.dst != END:
        kept.append(edge.dst)
        tau[edge.dst] = float(edge.tau_dst)
    kept.sort()
    return StaticBFSPolicy(tuple(kept), tuple((s, tau[s]) for s in kept), reference.particles)


def policy_from_path(edges: tuple[Edge, ...], particles: int) -> StaticBFSPolicy:
    assert edges and edges[0].src == START and edges[-1].dst == END
    assert all(a.dst == b.src for a, b in zip(edges, edges[1:]))
    items = [(e.dst, float(e.tau_dst)) for e in edges if e.dst != END]
    return StaticBFSPolicy(tuple(s for s, _ in items), tuple(items), particles)


def path_edges(steps: tuple[int, ...], taus: tuple[float, ...]) -> tuple[Edge, ...]:
    assert len(steps) == len(taus)
    prev = START
    out = []
    for step, tau in zip(steps, taus):
        out.append(Edge(prev, step, tau))
        prev = step
    out.append(Edge(prev, END, None))
    return tuple(out)


def all_paths(candidate_steps: tuple[int, ...], taus: tuple[float, ...], k: int):
    from itertools import product
    for selected in combinations(candidate_steps, k):
        for chosen_taus in product(taus, repeat=k):
            yield path_edges(selected, chosen_taus)
