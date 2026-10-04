"""Explicit host-side BFS event policy; no static particle ancestry."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math


@dataclass(frozen=True)
class StaticBFSPolicy:
    resampling_steps: tuple[int, ...]
    temperature_by_step: tuple[tuple[int, float], ...]
    particles: int
    selection_mode: str = "raw_tau"

    def __post_init__(self) -> None:
        steps = self.resampling_steps
        assert self.selection_mode == "raw_tau"
        assert self.particles > 0
        assert steps == tuple(sorted(set(steps)))
        assert all(0 <= x < 100 for x in steps)
        assert tuple(k for k, _ in self.temperature_by_step) == steps
        assert all(math.isfinite(v) and v > 0 for _, v in self.temperature_by_step)

    @property
    def temperature_map(self) -> dict[int, float]:
        return dict(self.temperature_by_step)

    def record(self) -> dict:
        return {"resampling_steps": list(self.resampling_steps),
                "temperature_by_step": [[i, t] for i, t in self.temperature_by_step],
                "particles": self.particles, "selection_mode": self.selection_mode}

    @property
    def id(self) -> str:
        payload = json.dumps(self.record(), sort_keys=True, separators=(",", ":"))
        return "static_" + sha256(payload.encode()).hexdigest()[:16]


def explicit_policy(steps: tuple[int, ...], taus: tuple[float, ...], particles: int) -> StaticBFSPolicy:
    return StaticBFSPolicy(steps, tuple(zip(steps, taus)), particles)
