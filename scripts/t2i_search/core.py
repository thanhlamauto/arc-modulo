"""Accelerator-free configuration, task-grid, sharding, and artifact helpers."""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


METHODS = ("bfs", "vasr")


class ConfigError(ValueError):
    """Raised before accelerator initialization when a run is ill-specified."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"cannot read JSON {path}: {exc}") from exc
    require(isinstance(value, dict), f"expected a JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, Mapping):
        return all(_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite(item) for item in value)
    return True


def write_json(path: Path, value: Any, *, immutable: bool = False) -> None:
    """Atomically write strict JSON; immutable files may only be resumed identically."""
    require(_finite(value), f"NaN/Inf in artifact: {path}")
    if immutable and path.exists():
        require(read_json(path) == value, f"immutable artifact changed: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _unique_ints(values: Any, name: str, *, minimum: int | None = None) -> tuple[int, ...]:
    require(isinstance(values, list) and values, f"{name} must be a non-empty list")
    require(all(isinstance(value, int) for value in values), f"{name} must contain integers")
    converted = tuple(values)
    require(len(set(converted)) == len(converted), f"{name} contains duplicates")
    if minimum is not None:
        require(all(value >= minimum for value in converted), f"{name} values must be >= {minimum}")
    return converted


def _runtime_path(value: Any, name: str) -> Path:
    require(isinstance(value, str) and value.strip(), f"missing path: {name}")
    return Path(os.path.expandvars(os.path.expanduser(value)))


@dataclass(frozen=True, order=True)
class PairKey:
    """Atomic scheduling unit; both methods stay together to preserve pairing."""

    particles: int
    prompt_index: int
    seed: int

    @property
    def id(self) -> str:
        return f"n{self.particles}/seed{self.seed}/prompt_{self.prompt_index:03d}"


@dataclass(frozen=True)
class Shard:
    index: int = 0
    count: int = 1

    def __post_init__(self) -> None:
        require(self.count >= 1, "shard count must be positive")
        require(0 <= self.index < self.count, "shard index must be in [0, count)")

    @classmethod
    def parse(cls, value: str) -> "Shard":
        try:
            index, count = (int(part) for part in value.split("/", maxsplit=1))
        except (TypeError, ValueError) as exc:
            raise ConfigError("shard must use INDEX/COUNT, for example 0/4") from exc
        return cls(index, count)

    def select(self, keys: Iterable[PairKey]) -> list[PairKey]:
        return [key for ordinal, key in enumerate(keys) if ordinal % self.count == self.index]

    @property
    def label(self) -> str:
        return f"{self.index}-of-{self.count}"


@dataclass(frozen=True)
class ExperimentSpec:
    config_path: Path
    protocol_path: Path
    backend: str
    run_root: Path
    official_root: Path
    baseline_results: Path
    cache_root: Path
    tpu_chip: str
    torch_threads: int
    offline: bool
    experiment_id: str
    particles: tuple[int, ...]
    seeds: tuple[int, ...]
    prompt_count: int
    resampling_steps: tuple[int, ...]
    smoke: PairKey
    protocol: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentSpec":
        config_path = Path(path).expanduser().resolve()
        config = read_json(config_path)
        require(config.get("schema_version") == 1, "unsupported config schema_version")
        require(config.get("backend") == "soft_vasr_v1", "unsupported backend")

        protocol_name = config.get("protocol")
        require(isinstance(protocol_name, str) and protocol_name, "missing protocol path")
        protocol_path = (config_path.parent / protocol_name).resolve()
        protocol = read_json(protocol_path)

        particles = _unique_ints(protocol.get("particles"), "protocol.particles", minimum=2)
        seeds = _unique_ints(protocol.get("seeds"), "protocol.seeds", minimum=0)
        prompt_count = protocol.get("prompt_count")
        require(isinstance(prompt_count, int) and prompt_count > 0, "invalid protocol.prompt_count")
        steps = _unique_ints(protocol.get("resampling_steps"), "protocol.resampling_steps", minimum=0)
        require(tuple(sorted(steps)) == steps, "resampling steps must be strictly increasing")
        require(steps[-1] < protocol.get("num_inference_steps", 0), "resampling step outside denoising grid")

        execution = protocol.get("execution", {})
        smoke = execution.get("smoke", {})
        smoke_values = (smoke.get("particles"), smoke.get("prompt_index"), smoke.get("seed"))
        require(all(isinstance(value, int) for value in smoke_values), "invalid execution.smoke tuple")
        smoke_key = PairKey(*smoke_values)
        require(smoke_key.particles in particles, "smoke particles are outside the grid")
        require(smoke_key.seed in seeds, "smoke seed is outside the grid")
        require(0 <= smoke_key.prompt_index < prompt_count, "smoke prompt is outside the grid")

        paths = config.get("paths", {})
        runtime = config.get("runtime", {})
        threads = runtime.get("torch_threads", 16)
        require(isinstance(threads, int) and threads > 0, "runtime.torch_threads must be positive")
        chip = runtime.get("tpu_chip", "")
        require(isinstance(chip, str) and chip, "runtime.tpu_chip must be a string")
        experiment_id = protocol.get("experiment_id")
        require(isinstance(experiment_id, str) and experiment_id, "missing protocol.experiment_id")

        return cls(
            config_path=config_path,
            protocol_path=protocol_path,
            backend=config["backend"],
            run_root=_runtime_path(paths.get("run_root"), "paths.run_root"),
            official_root=_runtime_path(paths.get("official_root"), "paths.official_root"),
            baseline_results=_runtime_path(paths.get("baseline_results"), "paths.baseline_results"),
            cache_root=_runtime_path(paths.get("cache_root"), "paths.cache_root"),
            tpu_chip=chip,
            torch_threads=threads,
            offline=bool(runtime.get("offline", True)),
            experiment_id=experiment_id,
            particles=particles,
            seeds=seeds,
            prompt_count=prompt_count,
            resampling_steps=steps,
            smoke=smoke_key,
            protocol=protocol,
        )

    @property
    def output(self) -> Path:
        return self.run_root / "results"

    def override_paths(
        self,
        *,
        run_root: Path | None = None,
        official_root: Path | None = None,
        baseline_results: Path | None = None,
        cache_root: Path | None = None,
        tpu_chip: str | None = None,
    ) -> "ExperimentSpec":
        return replace(
            self,
            run_root=run_root or self.run_root,
            official_root=official_root or self.official_root,
            baseline_results=baseline_results or self.baseline_results,
            cache_root=cache_root or self.cache_root,
            tpu_chip=tpu_chip or self.tpu_chip,
        )

    def pairs(
        self,
        *,
        particles: Sequence[int] | None = None,
        seeds: Sequence[int] | None = None,
        prompt_range: range | None = None,
    ) -> Iterator[PairKey]:
        selected_particles = tuple(particles) if particles is not None else self.particles
        selected_seeds = tuple(seeds) if seeds is not None else self.seeds
        selected_prompts = prompt_range if prompt_range is not None else range(self.prompt_count)
        require(set(selected_particles) <= set(self.particles), "requested particles are outside the protocol")
        require(set(selected_seeds) <= set(self.seeds), "requested seeds are outside the protocol")
        require(
            all(0 <= index < self.prompt_count for index in selected_prompts),
            "requested prompt range is outside the protocol",
        )
        for particle_count in self.particles:
            if particle_count not in selected_particles:
                continue
            for prompt_index in selected_prompts:
                for seed in self.seeds:
                    if seed in selected_seeds:
                        yield PairKey(particle_count, prompt_index, seed)

    def method_order(self, key: PairKey) -> tuple[str, str]:
        require(key.seed in self.seeds, "unknown seed")
        parity = key.prompt_index + self.seeds.index(key.seed)
        return METHODS if parity % 2 == 0 else tuple(reversed(METHODS))


def parse_int_list(value: str | None) -> tuple[int, ...] | None:
    if value is None:
        return None
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise ConfigError("expected a comma-separated integer list") from exc
    require(result and len(set(result)) == len(result), "integer list is empty or contains duplicates")
    return result


def parse_prompt_range(value: str | None, prompt_count: int) -> range | None:
    if value is None:
        return None
    try:
        start, stop = (int(part) for part in value.split(":", maxsplit=1))
    except (TypeError, ValueError) as exc:
        raise ConfigError("prompt range must use START:STOP, for example 0:25") from exc
    require(0 <= start < stop <= prompt_count, f"prompt range must be within 0:{prompt_count}")
    return range(start, stop)


def pair_path(output: Path, key: PairKey) -> Path:
    return output / "pairs" / f"n{key.particles}" / f"seed{key.seed}" / f"prompt_{key.prompt_index:03d}.json"


def raw_path(output: Path, key: PairKey, method: str) -> Path:
    require(method in METHODS, f"unknown method: {method}")
    return output / "raw" / f"n{key.particles}" / method / f"seed{key.seed}" / f"prompt_{key.prompt_index:03d}.json"


def status_summary(spec: ExperimentSpec) -> dict[str, Any]:
    keys = list(spec.pairs())
    completed_pairs = sum(pair_path(spec.output, key).is_file() for key in keys)
    method_records = {
        method: sum(raw_path(spec.output, key, method).is_file() for key in keys) for method in METHODS
    }
    by_particles = {
        str(k): {
            "expected_pairs": spec.prompt_count * len(spec.seeds),
            "completed_pairs": sum(pair_path(spec.output, key).is_file() for key in keys if key.particles == k),
        }
        for k in spec.particles
    }
    return {
        "experiment_id": spec.experiment_id,
        "output": str(spec.output),
        "expected_pairs": len(keys),
        "completed_pairs": completed_pairs,
        "remaining_pairs": len(keys) - completed_pairs,
        "raw_method_records": method_records,
        "complete": completed_pairs == len(keys) and all(count == len(keys) for count in method_records.values()),
        "by_particles": by_particles,
    }


def plan_summary(spec: ExperimentSpec, keys: Sequence[PairKey], shard: Shard) -> dict[str, Any]:
    return {
        "experiment_id": spec.experiment_id,
        "backend": spec.backend,
        "protocol": str(spec.protocol_path),
        "shard": shard.label,
        "paired_units": len(keys),
        "method_runs": len(keys) * len(METHODS),
        "particles": sorted({key.particles for key in keys}),
        "seeds": sorted({key.seed for key in keys}),
        "prompt_indices": (
            {"first": min(key.prompt_index for key in keys), "last": max(key.prompt_index for key in keys)}
            if keys
            else None
        ),
        "first_units": [key.id for key in keys[:5]],
    }
