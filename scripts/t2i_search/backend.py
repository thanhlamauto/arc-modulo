"""Thin adapter around the frozen, already-audited Soft-VASR scientific code."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import time
from types import ModuleType, SimpleNamespace
from typing import Any
from contextlib import contextmanager

try:
    from .core import ExperimentSpec, PairKey, require, sha256, write_json
except ImportError:  # Direct execution through cli.py.
    from core import ExperimentSpec, PairKey, require, sha256, write_json


class SoftVasrBackend:
    """Preserve scientific semantics while exposing a small orchestration API."""

    def __init__(self, spec: ExperimentSpec, *, shard_label: str) -> None:
        self.spec = spec
        self.shard_label = shard_label
        self.harness_root = Path(__file__).resolve().parents[1]
        self.soft_dir = self.harness_root / "soft_vasr_t2i"
        self.baseline_dir = self.harness_root / "diffusion_classical_search_t2i"
        self.args = SimpleNamespace(
            root=spec.run_root,
            output=spec.output,
            protocol=spec.protocol_path,
            official_root=spec.official_root,
            baseline_results=spec.baseline_results,
            cache_root=spec.cache_root,
        )
        self.legacy: ModuleType | None = None
        self.analysis: ModuleType | None = None
        self.old_manifest: dict[str, Any] | None = None
        self.prompts: list[dict[str, Any]] | None = None
        self.frozen: dict[str, Any] | None = None
        self.pipeline: Any = None
        self.reward: Any = None

    def _configure_environment(self) -> None:
        cache = self.spec.cache_root
        work_root = cache.parent
        for directory in (cache, work_root / "logs", work_root / "tmp"):
            directory.mkdir(parents=True, exist_ok=True)
        defaults = {
            "PJRT_DEVICE": "TPU",
            "TPU_VISIBLE_CHIPS": self.spec.tpu_chip,
            "HF_HOME": str(cache / "huggingface"),
            "TRANSFORMERS_CACHE": str(cache / "huggingface" / "transformers"),
            "HF_HUB_CACHE": str(cache / "huggingface" / "hub"),
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONDONTWRITEBYTECODE": "1",
            "GOOGLE_LOG_DIR": str(work_root / "logs"),
            "TPU_LOG_DIR": str(work_root / "logs"),
            "TMPDIR": str(work_root / "tmp"),
        }
        if self.spec.offline:
            defaults.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        for name, value in defaults.items():
            os.environ.setdefault(name, value)
        require(
            os.environ.get("TPU_VISIBLE_CHIPS") == self.spec.tpu_chip,
            f"TPU_VISIBLE_CHIPS must be {self.spec.tpu_chip} for this worker",
        )

    @staticmethod
    @contextmanager
    def _prepare_lock(path: Path):
        """Serialize shared preregistration writes while allowing parallel run shards."""
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = path.open("a+b")
        acquired = False
        try:
            if os.name == "nt":
                import msvcrt

                if path.stat().st_size == 0:
                    handle.write(b"\0")
                    handle.flush()
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            acquired = True
            yield
        finally:
            if acquired and os.name == "nt":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            elif acquired:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()

    @staticmethod
    def _load_file(name: str, path: Path) -> ModuleType:
        module_spec = importlib.util.spec_from_file_location(name, path)
        require(module_spec is not None and module_spec.loader is not None, f"cannot import {path}")
        module = importlib.util.module_from_spec(module_spec)
        sys.modules[name] = module
        module_spec.loader.exec_module(module)
        return module

    def _load_legacy(self) -> None:
        if self.legacy is not None:
            return
        fkd_dir = self.spec.official_root / "text_to_image/fkd_diffusers"
        for path in (self.soft_dir, self.baseline_dir, fkd_dir):
            value = str(path)
            if value not in sys.path:
                sys.path.insert(0, value)
        self.legacy = self._load_file("frozen_soft_vasr_run", self.soft_dir / "run.py")
        self.analysis = self._load_file("frozen_soft_vasr_analysis", self.soft_dir / "analyze.py")
        require(
            Path(self.legacy.baseline.__file__).resolve() == (self.baseline_dir / "tpu_runner.py").resolve(),
            "resolved the wrong Classical Search adapter",
        )

    def _orchestration_manifest(self) -> dict[str, Any]:
        runner_dir = Path(__file__).resolve().parent
        return {
            "schema_version": 1,
            "backend": self.spec.backend,
            "experiment_id": self.spec.experiment_id,
            "config_sha256": sha256(self.spec.config_path),
            "protocol_sha256": sha256(self.spec.protocol_path),
            "grid": {
                "particles": list(self.spec.particles),
                "seeds": list(self.spec.seeds),
                "prompt_count": self.spec.prompt_count,
                "methods": ["bfs", "vasr"],
                "pair_is_atomic_scheduling_unit": True,
            },
            "runner_sha256": {
                path.name: sha256(path)
                for path in sorted(runner_dir.glob("*.py"))
                if not path.name.startswith("test_")
            },
            "scientific_adapter": "frozen scripts/soft_vasr_t2i plus diffusion_classical_search_t2i",
        }

    def _run_structural_tests(self) -> None:
        assert self.legacy is not None and self.frozen is not None
        environment = dict(os.environ, OFFICIAL_ROOT=str(self.spec.official_root))
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "unittest",
                "discover",
                "-s",
                str(self.soft_dir),
                "-p",
                "test_*.py",
                "-v",
            ],
            env=environment,
            capture_output=True,
            text=True,
        )
        self.legacy.save(
            self.spec.output / "structural_tests.json",
            {
                "passed": result.returncode == 0,
                "output": result.stdout + result.stderr,
                "code_hashes": self.frozen["code_hashes"],
            },
        )
        require(result.returncode == 0, "structural unit tests failed; science prohibited")

    def prepare(self) -> None:
        self._configure_environment()
        self._load_legacy()
        assert self.legacy is not None
        with self._prepare_lock(self.spec.output / "orchestration.lock"):
            write_json(
                self.spec.output / "orchestration_manifest.json",
                self._orchestration_manifest(),
                immutable=True,
            )
            self.old_manifest, self.prompts, self.frozen = self.legacy.prepare(
                self.args, self.spec.protocol
            )
            self._run_structural_tests()

    def start_accelerator(self) -> None:
        require(self.frozen is not None and self.old_manifest is not None, "prepare must run first")
        assert self.legacy is not None
        import torch
        import torch_xla.core.xla_model as xm
        import torch_xla.runtime as xr

        (self.spec.cache_root / "xla").mkdir(parents=True, exist_ok=True)
        xr.initialize_cache(str(self.spec.cache_root / "xla"), readonly=False)
        device = xm.xla_device()
        require(device.type == "xla", "TPU/XLA device unavailable")
        require(xr.addressable_runtime_device_count() == 1, "one TPU device per worker is required")
        torch.set_num_threads(self.spec.torch_threads)
        self.reward = self.legacy.baseline._load_reward(
            self.spec.official_root, self.spec.cache_root
        )
        self.pipeline = self.legacy.baseline._load_pipeline(
            "sd15",
            self.old_manifest,
            self.spec.official_root,
            self.spec.cache_root,
            device,
        )

    def status(self, state: str, **details: Any) -> None:
        write_json(
            self.spec.output / "workers" / self.shard_label / "status.json",
            {
                "state": state,
                "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "tpu_chip": self.spec.tpu_chip,
                **details,
            },
        )

    def run_pair(self, key: PairKey) -> None:
        require(self.pipeline is not None and self.reward is not None, "accelerator is not initialized")
        require(self.prompts is not None and self.frozen is not None, "prepare must run first")
        assert self.legacy is not None
        prompt_record = self.prompts[key.prompt_index]
        prompt = prompt_record.get("prompt", prompt_record.get("text"))
        records = {}
        for method in self.spec.method_order(key):
            self.status(
                "RUNNING",
                particles=key.particles,
                seed=key.seed,
                prompt_index=key.prompt_index,
                method=method,
            )
            records[method] = self.legacy.run_one(
                self.args,
                self.spec.protocol,
                self.frozen,
                self.pipeline,
                self.reward,
                prompt,
                key.prompt_index,
                key.seed,
                key.particles,
                method,
            )
        evidence = self.legacy.validate_pair(records["bfs"], records["vasr"], self.spec.protocol)
        self.legacy.save(
            self.spec.output
            / "pairs"
            / f"n{key.particles}"
            / f"seed{key.seed}"
            / f"prompt_{key.prompt_index:03d}.json",
            evidence,
            immutable=True,
        )

    def analyze(self) -> None:
        require(self.analysis is not None, "prepare must run first")
        # The frozen analysis owns the exact 20k paired bootstrap semantics.
        for particles in self.spec.particles:
            self.analysis.analyze_n(self.spec.output, self.spec.protocol, particles)
        self.analysis.combined_report(self.spec.output, self.spec.protocol)


def build_backend(spec: ExperimentSpec, *, shard_label: str) -> SoftVasrBackend:
    require(spec.backend == "soft_vasr_v1", f"unsupported backend: {spec.backend}")
    return SoftVasrBackend(spec, shard_label=shard_label)
