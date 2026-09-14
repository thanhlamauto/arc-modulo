"""Small, config-driven orchestration layer for paired T2I search runs."""

from .core import ExperimentSpec, PairKey, Shard

__all__ = ["ExperimentSpec", "PairKey", "Shard"]
