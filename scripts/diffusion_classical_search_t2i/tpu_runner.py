#!/usr/bin/env python3
"""TPU runner for Tables 1 and 2 of arXiv:2505.23614v2.

The upstream repository is imported, never modified.  This file contains the
paper-faithful Algorithm 3 adapter needed because the committed upstream class
does not expose gamma and unconditionally adds a final resampling step.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import random
import shutil
import socket
import subprocess
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numba import njit


_ACTIVE_REWARD = None
_LAST_ADAPTER = None


@njit
def _seed_numba(seed: int) -> None:
    np.random.seed(seed)


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _slug(value: str) -> str:
    return value.replace(".", "p").replace("/", "--")


def _prompt_seed(trial_seed: int, prompt_index: int) -> int:
    sequence = np.random.SeedSequence([trial_seed, prompt_index])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _resampling_seed(trial_seed: int, prompt_index: int, config_id: str) -> int:
    tag = int.from_bytes(hashlib.sha256(config_id.encode("utf-8")).digest()[:4], "little")
    sequence = np.random.SeedSequence([trial_seed, prompt_index, tag])
    return int(sequence.generate_state(1, dtype=np.uint32)[0] % (2**31 - 1))


def _install_reward_stub() -> None:
    """Avoid importing unused CUDA-only verifier modules from upstream rewards.py."""

    module = types.ModuleType("rewards")

    def get_reward_function(
        reward_name: str,
        *,
        images: list[Any],
        prompts: list[str] | str,
        metric_to_chase: str | None = None,
        **_: Any,
    ) -> list[float]:
        del metric_to_chase
        if reward_name.lower() not in {"imagereward", "image_reward"}:
            raise ValueError(f"Only ImageReward is in scope, got {reward_name!r}")
        if _ACTIVE_REWARD is None:
            raise RuntimeError("ImageReward model has not been initialized")
        if isinstance(prompts, str):
            prompts = [prompts] * len(images)
        return [float(x) for x in _ACTIVE_REWARD.score_batched(list(prompts), list(images))]

    module.get_reward_function = get_reward_function
    sys.modules["rewards"] = module


@dataclass(frozen=True)
class Task:
    config_id: str
    model: str
    particles: int
    method: str
    use_smc: bool
    tempering: str = "constant"
    scoring: str = "max"
    resampling: str = "ssp"
    gamma: float | None = None
    branch_out: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


class PaperBFSAdapter:
    """Algorithm 3 with explicit sparse resampling and CPU offspring allocation."""

    def __init__(
        self,
        *,
        potential_type: str,
        lmbda: float,
        num_particles: int,
        time_steps: int,
        reward_fn: Any,
        latent_to_decode_fn: Any,
        device: torch.device,
        resampling: str = "ssp",
        tempering_schedule: str = "constant",
        gamma: float | None = None,
        resampling_steps: list[int] | tuple[int, ...] = (20, 40, 80),
        **_: Any,
    ) -> None:
        global _LAST_ADAPTER
        self.scoring = potential_type
        self.base_temperature = float(lmbda)
        self.num_particles = int(num_particles)
        self.time_steps = int(time_steps)
        self.reward_fn = reward_fn
        self.latent_to_decode_fn = latent_to_decode_fn
        self.device = device
        self.resampling = resampling
        self.tempering = tempering_schedule
        self.gamma = gamma
        self.resampling_steps = frozenset(int(x) for x in resampling_steps)
        self.previous = torch.zeros(self.num_particles, device=device, dtype=torch.float32)
        self.events: list[dict[str, Any]] = []
        _LAST_ADAPTER = self

    def _temperature(self, sampling_idx: int) -> float:
        if self.tempering == "constant":
            return self.base_temperature
        if self.tempering == "increase":
            if self.gamma is None:
                raise ValueError("Increasing tempering requires gamma")
            return self.base_temperature * ((1.0 + float(self.gamma)) ** sampling_idx - 1.0)
        if self.tempering == "inf":
            return float("inf")
        raise ValueError(f"Unknown tempering schedule: {self.tempering}")

    def resample(
        self, *, sampling_idx: int, latents: torch.Tensor, x0_preds: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if sampling_idx not in self.resampling_steps:
            return latents, None

        population_images = self.latent_to_decode_fn(x0_preds)
        raw = self.reward_fn(population_images).to(dtype=torch.float32)
        temperature = self._temperature(sampling_idx)

        if self.tempering == "inf":
            if self.scoring == "rt":
                scores = raw
            elif self.scoring == "max":
                # The common infinite scale cancels in argmax; retain the
                # unscaled path maximum so Inf+Max is not reduced to Current.
                scores = torch.maximum(raw, self.previous)
            elif self.scoring == "diff":
                scores = raw - self.previous
            else:
                raise ValueError(f"Unknown scoring function: {self.scoring}")
            scores_cpu = scores.detach().cpu().numpy()
            parent = int(np.argmax(scores_cpu))
            indices_np = np.repeat(parent, self.num_particles).astype(np.int64)
            weights_np = np.zeros(self.num_particles, dtype=np.float64)
            weights_np[parent] = 1.0
        else:
            tempered = raw * temperature
            if self.scoring == "rt":
                scores = tempered
            elif self.scoring == "diff":
                scores = tempered - self.previous
            elif self.scoring == "max":
                scores = torch.maximum(tempered, self.previous)
            else:
                raise ValueError(f"Unknown scoring function: {self.scoring}")

            # Algorithm 3 specifies softmax. Offspring allocation remains the
            # exact implementation imported from upstream smc_utils.
            weights_np = torch.softmax(scores, dim=0).detach().cpu().numpy().astype(np.float64)
            import smc_utils

            if self.resampling == "deterministic_top":
                indices_np = np.repeat(int(np.argmax(weights_np)), self.num_particles).astype(np.int64)
            else:
                indices_np = np.asarray(
                    smc_utils.Resample_dict[self.resampling](weights_np, self.num_particles),
                    dtype=np.int64,
                )

        indices = torch.tensor(indices_np, device=latents.device, dtype=torch.long)
        resampled_latents = torch.index_select(latents, 0, indices)
        resampled_images = torch.index_select(population_images, 0, indices)

        if self.tempering != "inf":
            if self.scoring == "diff":
                buffer = raw * temperature
            elif self.scoring == "max":
                buffer = scores
            else:
                buffer = raw * temperature
            self.previous = torch.index_select(buffer, 0, indices)
        else:
            if self.scoring == "max":
                buffer = scores
            else:
                buffer = raw
            self.previous = torch.index_select(buffer, 0, indices)

        self.events.append(
            {
                "sampling_index": int(sampling_idx),
                "temperature": "inf" if np.isinf(temperature) else float(temperature),
                "raw_rewards": [float(x) for x in raw.detach().cpu().tolist()],
                "particle_scores": [float(x) for x in scores.detach().cpu().tolist()],
                "normalized_weights": [float(x) for x in weights_np.tolist()],
                "parent_indices": [int(x) for x in indices_np.tolist()],
                "xla_cpu_sync": True,
            }
        )
        return resampled_latents, resampled_images


def _task_id(method: str, particles: int, gamma: float | None = None) -> str:
    suffix = f"_g{_slug(f'{gamma:.3f}')}" if gamma is not None else ""
    return f"{method}{suffix}_n{particles}"


def _method_task(model: str, particles: int, method: str, gamma: float | None = None) -> Task:
    definitions: dict[str, dict[str, Any]] = {
        "bon": {"use_smc": False},
        "fk": {"use_smc": True, "tempering": "constant", "scoring": "max", "resampling": "multinomial"},
        "das": {"use_smc": True, "tempering": "increase", "scoring": "diff", "resampling": "ssp"},
        "treeg": {"use_smc": True, "tempering": "constant", "scoring": "rt", "resampling": "treeg", "branch_out": 2},
        "svdd": {"use_smc": True, "tempering": "inf", "scoring": "rt", "resampling": "deterministic_top"},
        "bfs": {"use_smc": True, "tempering": "increase", "scoring": "max", "resampling": "ssp"},
        "constant_max_ssp": {"use_smc": True, "tempering": "constant", "scoring": "max", "resampling": "ssp"},
        "constant_current_ssp": {"use_smc": True, "tempering": "constant", "scoring": "rt", "resampling": "ssp"},
        "constant_difference_ssp": {"use_smc": True, "tempering": "constant", "scoring": "diff", "resampling": "ssp"},
        "inf_max": {"use_smc": True, "tempering": "inf", "scoring": "max", "resampling": "deterministic_top"},
    }
    data = definitions[method]
    return Task(
        config_id=_task_id(method, particles, gamma),
        model=model,
        particles=particles,
        method=method,
        gamma=gamma,
        **data,
    )


def _tasks_for_stage(stage: str) -> list[Task]:
    gamma_values = (0.008, 0.024)
    tasks: list[Task] = []
    if stage == "canary":
        return [_method_task("sd15", 4, "bfs", 0.008)]
    if stage == "table1":
        for particles in (4, 8):
            for method in (
                "bon",
                "fk",
                "constant_max_ssp",
                "constant_current_ssp",
                "constant_difference_ssp",
            ):
                tasks.append(_method_task("sd15", particles, method))
            tasks.extend(_method_task("sd15", particles, "bfs", gamma) for gamma in gamma_values)
            tasks.append(_method_task("sd15", particles, "inf_max"))
        return tasks
    if stage == "table2_sd15":
        for particles in (4, 8):
            tasks.extend(_method_task("sd15", particles, "das", gamma) for gamma in gamma_values)
            tasks.append(_method_task("sd15", particles, "treeg"))
            tasks.append(_method_task("sd15", particles, "svdd"))
        return tasks
    if stage == "table2_sdxl":
        for particles in (4, 8):
            tasks.append(_method_task("sdxl", particles, "bon"))
            tasks.append(_method_task("sdxl", particles, "fk"))
            tasks.extend(_method_task("sdxl", particles, "das", gamma) for gamma in gamma_values)
            tasks.append(_method_task("sdxl", particles, "treeg"))
            tasks.append(_method_task("sdxl", particles, "svdd"))
            tasks.extend(_method_task("sdxl", particles, "bfs", gamma) for gamma in gamma_values)
        return tasks
    raise ValueError(f"Unknown stage {stage!r}")


def _load_prompts(manifest: dict[str, Any], official_root: Path) -> list[dict[str, Any]]:
    path = official_root / manifest["benchmark"]["relative_path"]
    if _sha256(path) != manifest["benchmark"]["sha256"]:
        raise RuntimeError(f"Prompt benchmark hash mismatch: {path}")
    with path.open(encoding="utf-8") as handle:
        prompts = json.load(handle)
    if len(prompts) != manifest["benchmark"]["count"]:
        raise RuntimeError(f"Expected 100 prompts, found {len(prompts)}")
    return prompts


def _load_reward(official_root: Path, cache_root: Path) -> Any:
    global _ACTIVE_REWARD
    fkd_root = official_root / "text_to_image" / "fkd_diffusers"
    sys.path.insert(0, str(fkd_root))
    from image_reward_utils import rm_load

    _ACTIVE_REWARD = rm_load(
        "ImageReward-v1.0",
        device="cpu",
        download_root=str(cache_root / "ImageReward"),
    )
    _ACTIVE_REWARD.eval()
    return _ACTIVE_REWARD


def _load_pipeline(
    model_key: str,
    manifest: dict[str, Any],
    official_root: Path,
    cache_root: Path,
    device: torch.device,
) -> Any:
    _install_reward_stub()
    fkd_root = official_root / "text_to_image" / "fkd_diffusers"
    sys.path.insert(0, str(fkd_root))
    import fkd_pipeline_sd
    import fkd_pipeline_sdxl
    from diffusers import DDIMScheduler

    fkd_pipeline_sd.FKD = PaperBFSAdapter
    fkd_pipeline_sdxl.FKD = PaperBFSAdapter
    spec = manifest["models"][model_key]
    pipeline_class = (
        fkd_pipeline_sd.FKDStableDiffusion
        if model_key == "sd15"
        else fkd_pipeline_sdxl.FKDStableDiffusionXL
    )
    execution_dtype = getattr(torch, spec["execution_dtype"])
    load_kwargs = {
        "revision": spec["revision"],
        "torch_dtype": execution_dtype,
        "cache_dir": str(cache_root / "huggingface"),
        "local_files_only": True,
    }
    if "variant" in spec:
        load_kwargs["variant"] = spec["variant"]
    if "use_safetensors" in spec:
        load_kwargs["use_safetensors"] = spec["use_safetensors"]
    pipe = pipeline_class.from_pretrained(spec["id"], **load_kwargs)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    return pipe.to(device)


def _xla_callback(pipe: Any, step: int, timestep: Any, callback_kwargs: dict[str, Any]) -> dict[str, Any]:
    del pipe, step, timestep
    import torch_xla.core.xla_model as xm

    xm.mark_step()
    return callback_kwargs


def _completed_indices(path: Path) -> set[int]:
    if not path.exists():
        return set()
    completed: set[int] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                completed.add(int(json.loads(line)["prompt_index"]))
    return completed


def _run_task(
    *,
    task: Task,
    trial_seed: int,
    pipe: Any,
    reward_model: Any,
    prompts: list[dict[str, Any]],
    manifest: dict[str, Any],
    run_root: Path,
    stage: str,
    max_prompts: int | None,
) -> None:
    import torch_xla.core.xla_model as xm

    data_root = run_root / "diagnostics" / "canary" if stage == "canary" else run_root
    model_root = data_root / "raw" / task.model / task.config_id
    result_path = model_root / f"trial_seed{trial_seed}.jsonl"
    completed = _completed_indices(result_path)
    selected_root = model_root / "selected" / f"trial_seed{trial_seed}"
    selected_root.mkdir(parents=True, exist_ok=True)

    runtime = manifest["runtime"]
    num_steps = int(runtime["num_inference_steps"])
    resampling_steps = list(runtime["resampling_steps"])
    prompt_limit = len(prompts) if max_prompts is None else min(max_prompts, len(prompts))
    if stage == "canary":
        prompt_limit = 1

    for prompt_index, item in enumerate(prompts[:prompt_limit]):
        if prompt_index in completed:
            continue
        prompt = item.get("prompt", item.get("text"))
        if not isinstance(prompt, str):
            raise TypeError(f"Prompt {prompt_index} is not a string")

        pseed = _prompt_seed(trial_seed, prompt_index)
        rseed = _resampling_seed(trial_seed, prompt_index, task.config_id)
        random.seed(pseed)
        np.random.seed(rseed)
        _seed_numba(rseed)
        torch.manual_seed(pseed)
        xm.set_rng_state(pseed, pipe.device)
        generator = torch.Generator(device="cpu").manual_seed(pseed)

        fkd_args = {
            "lmbda": float(runtime["base_temperature"]),
            "num_particles": task.particles,
            "use_smc": task.use_smc,
            "adaptive_resampling": False,
            "resample_frequency": 0,
            "time_steps": num_steps,
            "resampling_t_start": -1,
            "resampling_t_end": -1,
            "guidance_reward_fn": "ImageReward",
            "potential_type": task.scoring,
            "tempering_schedule": task.tempering,
            "resampling": task.resampling,
            "gamma": task.gamma,
            "resampling_steps": resampling_steps,
        }

        started = time.perf_counter()
        output = pipe(
            [prompt] * task.particles,
            num_inference_steps=num_steps,
            eta=float(runtime["eta"]),
            num_images_per_prompt=1,
            generator=generator,
            fkd_args=fkd_args,
            output_type="pil",
            callback_on_step_end=_xla_callback,
            callback_on_step_end_tensor_inputs=["latents"],
        )
        xm.mark_step()
        xm.wait_device_ops()
        images = list(output.images)
        final_scores = [
            float(x)
            for x in reward_model.score_batched([prompt] * task.particles, images)
        ]
        selected_particle = int(np.argmax(np.asarray(final_scores, dtype=np.float64)))
        image_path = selected_root / f"prompt_{prompt_index:05d}_particle_{selected_particle:02d}.jpg"
        # The verifier is evaluated on the in-memory, lossless PIL result above.
        # JPEG is artifact storage only and keeps the full experiment within the
        # TPU VM's constrained root disk.
        images[selected_particle].save(image_path, format="JPEG", quality=90)
        selected_rgb_std = float(np.asarray(images[selected_particle].convert("RGB"), dtype=np.float32).std())
        elapsed = time.perf_counter() - started

        adapter_events = []
        if task.use_smc:
            if _LAST_ADAPTER is None:
                raise RuntimeError("SMC run completed without constructing the adapter")
            adapter_events = list(_LAST_ADAPTER.events)
            observed = [event["sampling_index"] for event in adapter_events]
            if observed != resampling_steps:
                raise RuntimeError(f"Observed resampling schedule {observed}, expected {resampling_steps}")

        reconstruction_components = []
        if task.tempering == "increase":
            reconstruction_components.append("increase_gamma_tempering")
        if task.tempering == "inf":
            reconstruction_components.append("infinite_tempering")
        if task.use_smc:
            reconstruction_components.append("explicit_20_40_80_schedule_without_final_resampling")
        model_spec = manifest["models"][task.model]
        record = {
            "schema_version": 1,
            "paper": {"arxiv": manifest["paper"]["arxiv"], "table_stage": stage},
            "source": {
                "git_sha": manifest["official_source"]["commit"],
                "repository": manifest["official_source"]["repository"],
            },
            "model_asset": {
                "id": model_spec["id"],
                "revision": model_spec["revision"],
                "source_dtype": model_spec["source_dtype"],
                "execution_dtype": model_spec["execution_dtype"],
                "asset_manifest": "asset_manifest.json",
            },
            "verifier": {
                "id": "ImageReward-v1.0",
                "source_commit": manifest["runtime"]["image_reward_commit"],
                "device": "cpu",
                "asset_manifest": "asset_manifest.json",
            },
            "prompt_benchmark": dict(manifest["benchmark"]),
            "stage": stage,
            "model": task.model,
            "config": task.as_dict(),
            "trial_seed": trial_seed,
            "prompt_seed": pseed,
            "resampling_seed": rseed,
            "prompt_index": prompt_index,
            "prompt": prompt,
            "num_inference_steps": num_steps,
            "temperature": float(runtime["base_temperature"]),
            "eta": float(runtime["eta"]),
            "resampling_steps": resampling_steps if task.use_smc else [],
            "final_particle_ids": [f"p{prompt_index:05d}-k{k:02d}" for k in range(task.particles)],
            "final_particle_scores": final_scores,
            "selected_particle_index": selected_particle,
            "selected_particle_id": f"p{prompt_index:05d}-k{selected_particle:02d}",
            "selected_final_score": final_scores[selected_particle],
            "selected_image_path": str(image_path.relative_to(run_root)),
            "selected_image_sha256": _sha256(image_path),
            "selected_image_rgb_std": selected_rgb_std,
            "resampling_events": adapter_events,
            "elapsed_seconds": elapsed,
            "PAPER_RECONSTRUCTION": bool(reconstruction_components),
            "RECONSTRUCTION_COMPONENT": reconstruction_components,
            "backend": {
                "tpu_type": runtime["tpu_type"],
                "diffusion_device": str(pipe.device),
                "xla_global_ordinal": int(xm.get_ordinal()),
                "execution_dtype": str(pipe.unet.dtype),
                "verifier_device": "cpu",
                "cpu_fallbacks": [
                    "ImageReward exact model",
                    "official NumPy/Numba offspring allocation",
                ],
                "xla_cpu_syncs_per_prompt": len(adapter_events) + 1,
                "compile_time_separately_measurable": False,
            },
        }
        with result_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        print(
            json.dumps(
                {
                    "event": "prompt_complete",
                    "stage": stage,
                    "model": task.model,
                    "config": task.config_id,
                    "seed": trial_seed,
                    "prompt": prompt_index,
                    "score": final_scores[selected_particle],
                    "seconds": elapsed,
                },
                sort_keys=True,
            ),
            flush=True,
        )


def _worker(index: int, payload: dict[str, Any]) -> None:
    import torch_xla.core.xla_model as xm
    import torch_xla.runtime as xr

    manifest = payload["manifest"]
    seeds = manifest["trials"]["seeds"]
    trial_seed = int(seeds[index])
    cache_root = Path(payload["cache_root"])
    # One writer prevents torch_xla 2.4's concurrent-write corruption. Other
    # workers consume completed entries and compile cache misses in memory.
    cache_path = cache_root / "xla" / "persistent"
    cache_path.mkdir(parents=True, exist_ok=True)
    # SDXL executable blobs would exhaust the remaining root disk. It still
    # reuses compilation in memory across all SDXL configs within the stage.
    cache_readonly = index != 0 or payload["stage"] == "table2_sdxl"
    xr.initialize_cache(str(cache_path), readonly=cache_readonly)
    device = xm.xla_device()
    torch.set_num_threads(int(payload["cpu_threads_per_process"]))

    official_root = Path(payload["official_root"])
    run_root = Path(payload["run_root"])
    tasks = [Task(**item) for item in payload["tasks"]]
    prompts = _load_prompts(manifest, official_root)
    reward_model = _load_reward(official_root, cache_root)
    pipeline = _load_pipeline(tasks[0].model, manifest, official_root, cache_root, device)

    status_path = run_root / "status" / f"{payload['stage']}_seed{trial_seed}.json"
    _json_dump(
        status_path,
        {
            "state": "running",
            "stage": payload["stage"],
            "trial_seed": trial_seed,
            "ordinal": int(xm.get_ordinal()),
            "device": str(device),
            "global_runtime_device_count": int(xr.global_runtime_device_count()),
            "addressable_runtime_device_count": int(xr.addressable_runtime_device_count()),
            "started_unix": time.time(),
        },
    )
    try:
        task_failures = []
        for task in tasks:
            try:
                _run_task(
                    task=task,
                    trial_seed=trial_seed,
                    pipe=pipeline,
                    reward_model=reward_model,
                    prompts=prompts,
                    manifest=manifest,
                    run_root=run_root,
                    stage=payload["stage"],
                    max_prompts=payload["max_prompts"],
                )
            except BaseException as task_exc:
                failure = {
                    "config": task.as_dict(),
                    "trial_seed": trial_seed,
                    "error_type": type(task_exc).__name__,
                    "error": str(task_exc),
                    "failed_unix": time.time(),
                }
                task_failures.append(failure)
                _json_dump(
                    run_root
                    / "failures"
                    / payload["stage"]
                    / task.model
                    / task.config_id
                    / f"trial_seed{trial_seed}.json",
                    failure,
                )
                if payload["stage"] == "canary":
                    raise
                print(json.dumps({"event": "task_failed", **failure}, sort_keys=True), flush=True)
        xm.wait_device_ops()
        _json_dump(
            status_path,
            {
                "state": "complete_with_failures" if task_failures else "complete",
                "stage": payload["stage"],
                "trial_seed": trial_seed,
                "ordinal": int(xm.get_ordinal()),
                "device": str(device),
                "completed_unix": time.time(),
                "task_failures": task_failures,
            },
        )
    except BaseException as exc:
        _json_dump(
            status_path,
            {
                "state": "failed",
                "stage": payload["stage"],
                "trial_seed": trial_seed,
                "ordinal": int(xm.get_ordinal()),
                "device": str(device),
                "failed_unix": time.time(),
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise
    finally:
        del pipeline
        del reward_model
        gc.collect()


def _prefetch(args: argparse.Namespace, manifest: dict[str, Any]) -> None:
    from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline

    cache_root = args.cache_root
    hf_cache = cache_root / "huggingface"
    hf_cache.mkdir(parents=True, exist_ok=True)
    classes = {"sd15": StableDiffusionPipeline, "sdxl": StableDiffusionXLPipeline}
    snapshots: dict[str, Any] = {}
    for model_key, spec in manifest["models"].items():
        before = time.time()
        load_kwargs = {
            "revision": spec["revision"],
            "torch_dtype": torch.float16,
            "cache_dir": str(hf_cache),
        }
        if "variant" in spec:
            load_kwargs["variant"] = spec["variant"]
        if "use_safetensors" in spec:
            load_kwargs["use_safetensors"] = spec["use_safetensors"]
        pipe = classes[model_key].from_pretrained(spec["id"], **load_kwargs)
        snapshots[model_key] = {
            "id": spec["id"],
            "revision": spec["revision"],
            "variant": spec.get("variant"),
            "use_safetensors": spec.get("use_safetensors"),
            "load_seconds": time.time() - before,
        }
        del pipe
        gc.collect()

    _load_reward(args.official_root, cache_root)
    global _ACTIVE_REWARD
    del _ACTIVE_REWARD
    _ACTIVE_REWARD = None
    gc.collect()

    files = []
    roots = [hf_cache, cache_root / "ImageReward"]
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix.lower() in {
                ".safetensors",
                ".bin",
                ".json",
                ".pt",
                ".txt",
                ".model",
            }:
                files.append(
                    {
                        "path": str(path.relative_to(cache_root)),
                        "bytes": path.stat().st_size,
                        "sha256": _sha256(path),
                    }
                )
    prompt_path = args.official_root / manifest["benchmark"]["relative_path"]
    package_freeze = subprocess.check_output(
        [sys.executable, "-m", "pip", "freeze", "--all"], text=True
    ).splitlines()
    asset_manifest = {
        "created_unix": time.time(),
        "host": socket.gethostname(),
        "snapshots": snapshots,
        "prompt": {
            "path": str(prompt_path),
            "count": manifest["benchmark"]["count"],
            "sha256": _sha256(prompt_path),
        },
        "files": files,
        "package_freeze": package_freeze,
    }
    _json_dump(args.run_root / "asset_manifest.json", asset_manifest)
    print(json.dumps({"event": "prefetch_complete", "hashed_files": len(files)}), flush=True)


def _adapter_self_test() -> None:
    import tempfile

    upstream = Path(tempfile.mkdtemp())
    try:
        (upstream / "smc_utils.py").write_text(
            "import numpy as np\n"
            "Resample_dict={'ssp': lambda w,n: np.arange(n), "
            "'multinomial': lambda w,n: np.arange(n), "
            "'treeg': lambda w,n: np.repeat(np.argsort(w)[-n//2:],2)}\n",
            encoding="utf-8",
        )
        sys.path.insert(0, str(upstream))
        adapter = PaperBFSAdapter(
            potential_type="max",
            lmbda=10.0,
            num_particles=4,
            time_steps=100,
            reward_fn=lambda x: torch.tensor([0.1, 0.5, 0.2, -0.1]),
            latent_to_decode_fn=lambda x: x,
            device=torch.device("cpu"),
            resampling="ssp",
            tempering_schedule="increase",
            gamma=0.008,
            resampling_steps=[20, 40, 80],
        )
        latents = torch.arange(4.0).reshape(4, 1)
        untouched, decoded = adapter.resample(sampling_idx=19, latents=latents, x0_preds=latents)
        assert decoded is None and torch.equal(untouched, latents)
        adapter.resample(sampling_idx=20, latents=latents, x0_preds=latents)
        assert len(adapter.events) == 1
        expected = 10.0 * ((1.008**20) - 1.0)
        assert abs(adapter.events[0]["temperature"] - expected) < 1e-9
        rewards = iter(
            [
                torch.tensor([0.9, 0.8, 0.7, 0.6]),
                torch.tensor([0.1, 0.2, 0.3, 0.4]),
            ]
        )
        inf_adapter = PaperBFSAdapter(
            potential_type="max",
            lmbda=10.0,
            num_particles=4,
            time_steps=100,
            reward_fn=lambda x: next(rewards),
            latent_to_decode_fn=lambda x: x,
            device=torch.device("cpu"),
            resampling="deterministic_top",
            tempering_schedule="inf",
            resampling_steps=[20, 40],
        )
        inf_adapter.resample(sampling_idx=20, latents=latents, x0_preds=latents)
        inf_adapter.resample(sampling_idx=40, latents=latents, x0_preds=latents)
        assert inf_adapter.events[1]["parent_indices"] == [0, 0, 0, 0]
        print(json.dumps({"event": "adapter_self_test_passed", "temperature_at_20": expected}))
    finally:
        shutil.rmtree(upstream)


def _preflight_record(args: argparse.Namespace, manifest: dict[str, Any], tasks: list[Task]) -> None:
    import torch_xla

    args.run_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.manifest, args.run_root / "experiment_manifest.json")
    record = {
        "created_unix": time.time(),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "torch_xla": torch_xla.__version__,
        "pjrt_device": os.environ.get("PJRT_DEVICE"),
        "topology_query_deferred_to_workers": True,
        "visible_vfio_nodes": sorted(path.name for path in Path("/dev/vfio").glob("[0-9]*")),
        "official_commit": subprocess.check_output(
            ["git", "-C", str(args.official_root), "rev-parse", "HEAD"], text=True
        ).strip(),
        "official_status": subprocess.check_output(
            ["git", "-C", str(args.official_root), "status", "--porcelain"], text=True
        ).splitlines(),
        "official_root": str(args.official_root),
        "stage": args.stage,
        "tasks": [task.as_dict() for task in tasks],
    }
    if record["official_commit"] != manifest["official_source"]["commit"]:
        raise RuntimeError("Official source commit does not match the experiment manifest")
    if record["official_status"]:
        raise RuntimeError("Official source checkout is not clean")
    _json_dump(args.run_root / "preflight" / f"{args.stage}.json", record)


def _validate_canary(run_root: Path, manifest: dict[str, Any]) -> None:
    config_id = _task_id("bfs", 4, 0.008)
    records = []
    for seed in manifest["trials"]["seeds"]:
        path = (
            run_root
            / "diagnostics"
            / "canary"
            / "raw"
            / "sd15"
            / config_id
            / f"trial_seed{seed}.jsonl"
        )
        if not path.exists():
            raise RuntimeError(f"Canary record missing: {path}")
        with path.open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        if len(rows) != 1:
            raise RuntimeError(f"Expected one canary record in {path}, found {len(rows)}")
        records.append(rows[0])
    failures = []
    for record in records:
        if not all(np.isfinite(record["final_particle_scores"])):
            failures.append(f"seed {record['trial_seed']}: non-finite ImageReward")
        if record["selected_image_rgb_std"] <= 1.0:
            failures.append(f"seed {record['trial_seed']}: near-uniform image")
    unique_images = len({record["selected_image_sha256"] for record in records})
    if unique_images < 2:
        failures.append("all trial seeds produced an identical selected image")
    report = {
        "passed": not failures,
        "failures": failures,
        "unique_selected_images": unique_images,
        "rgb_standard_deviations": [record["selected_image_rgb_std"] for record in records],
        "scores": [record["selected_final_score"] for record in records],
    }
    _json_dump(run_root / "diagnostics" / "canary" / "validation.json", report)
    if failures:
        raise RuntimeError("Canary validation failed: " + "; ".join(failures))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--stage",
        choices=("prefetch", "self_test", "canary", "table1", "table2_sd15", "table2_sdxl"),
        required=True,
    )
    parser.add_argument("--max-prompts", type=int)
    parser.add_argument("--cpu-threads-per-process", type=int, default=24)
    return parser.parse_args()


def _clear_sd15_xla_cache_before_sdxl(args: argparse.Namespace) -> None:
    """Reclaim compiled SD1.5 graphs before the disk-heavier SDXL stage."""
    xla_root = (args.cache_root / "xla").resolve()
    persistent = (xla_root / "persistent").resolve()
    if persistent.parent != xla_root or xla_root.parent != args.cache_root:
        raise RuntimeError(f"Refusing to clear unexpected XLA cache path: {persistent}")

    bytes_removed = 0
    files_removed = 0
    if persistent.exists():
        for path in persistent.rglob("*"):
            if path.is_file():
                files_removed += 1
                bytes_removed += path.stat().st_size
        shutil.rmtree(persistent)
    persistent.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(args.cache_root).free
    _json_dump(
        args.run_root / "status" / "table2_sdxl_cache_cleanup.json",
        {
            "cache_path": str(persistent),
            "files_removed": files_removed,
            "bytes_removed": bytes_removed,
            "free_bytes_after": free_bytes,
            "reason": "Reclaim completed SD1.5 XLA graphs before SDXL; all SDXL workers compile cache misses in memory.",
        },
    )


def main() -> None:
    args = parse_args()
    args.manifest = args.manifest.resolve()
    args.official_root = args.official_root.resolve()
    args.cache_root = args.cache_root.resolve()
    args.output_root = args.output_root.resolve()
    args.run_root = args.output_root / args.run_id
    with args.manifest.open(encoding="utf-8") as handle:
        manifest = json.load(handle)

    if args.stage == "self_test":
        _adapter_self_test()
        return
    if args.stage == "prefetch":
        args.run_root.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.manifest, args.run_root / "experiment_manifest.json")
        _prefetch(args, manifest)
        return

    if args.stage == "table2_sdxl":
        _clear_sd15_xla_cache_before_sdxl(args)

    tasks = _tasks_for_stage(args.stage)
    _preflight_record(args, manifest, tasks)
    payload = {
        "manifest": manifest,
        "official_root": str(args.official_root),
        "cache_root": str(args.cache_root),
        "run_root": str(args.run_root),
        "tasks": [task.as_dict() for task in tasks],
        "stage": args.stage,
        "max_prompts": args.max_prompts,
        "cpu_threads_per_process": args.cpu_threads_per_process,
    }
    import torch_xla.distributed.xla_multiprocessing as xmp

    # PJRT selects all addressable devices; specifying nprocs is unsupported in
    # torch_xla 2.4 and is intentionally left to the runtime.
    xmp.spawn(_worker, args=(payload,), nprocs=None, start_method="spawn")
    if args.stage == "canary":
        _validate_canary(args.run_root, manifest)
    _json_dump(
        args.run_root / "status" / f"{args.stage}.json",
        {"state": "complete", "stage": args.stage, "completed_unix": time.time()},
    )


if __name__ == "__main__":
    main()
