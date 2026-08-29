#!/usr/bin/env python3
"""External, provenance-first runner for the unmodified official Locomotion code."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import shutil
import statistics
import subprocess
import sys
import time
import traceback
import warnings

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("D4RL_SUPPRESS_IMPORT_ERROR", "1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--run-kind", choices=("checkpoint", "smoke", "seed"), required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--harness-sha", required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def package_versions(names: list[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def command_text(command: list[str]) -> dict[str, object]:
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout[-12000:],
        "stderr": completed.stderr[-12000:],
    }


def checkpoint_summary(path: Path, torch_module: object) -> dict[str, object]:
    state = torch_module.load(path, map_location="cpu")
    if not isinstance(state, dict):
        raise TypeError(f"checkpoint is not a state-dict mapping: {path}")
    tensor_items = [(key, value) for key, value in state.items() if hasattr(value, "shape")]
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "mapping_type": type(state).__name__,
        "key_count": len(state),
        "tensor_count": len(tensor_items),
        "first_keys": list(state.keys())[:20],
        "first_tensor_shapes": {key: list(value.shape) for key, value in tensor_items[:20]},
    }


def install_instrumentation(sampler: object, model: object, np_module: object) -> dict[str, object]:
    trace: dict[str, object] = {
        "guide_step_calls": 0,
        "solver_update_calls": 0,
        "predict_xt_calls": 0,
        "tilde_guidance_calls": 0,
        "q_verifier_calls": 0,
        "q_candidate_shapes": [],
        "policy_call_guide_steps": [],
        "first_policy_step_trace": [],
        "final_selection_checks": [],
    }

    original_solver = sampler.solver_update
    original_predict = sampler._predict_xt
    original_tilde = sampler.tilde_get_guidance
    original_guide = sampler.guide_step
    original_q = model.calculateQ
    original_get_guidance = sampler.get_guidance
    original_tensor_to_obj = sampler.tensor_to_obj
    capture: dict[str, object] = {"enabled": False, "x": None, "q": None}

    def counted_solver(*args: object, **kwargs: object) -> object:
        trace["solver_update_calls"] += 1
        return original_solver(*args, **kwargs)

    def counted_predict(*args: object, **kwargs: object) -> object:
        trace["predict_xt_calls"] += 1
        return original_predict(*args, **kwargs)

    def counted_tilde(*args: object, **kwargs: object) -> object:
        trace["tilde_guidance_calls"] += 1
        return original_tilde(*args, **kwargs)

    def counted_q(*args: object, **kwargs: object) -> object:
        result = original_q(*args, **kwargs)
        trace["q_verifier_calls"] += 1
        action = kwargs.get("a")
        if action is not None and len(trace["q_candidate_shapes"]) < 32:
            trace["q_candidate_shapes"].append(list(action.shape))
        return result

    def captured_get_guidance(*args: object, **kwargs: object) -> object:
        result = original_get_guidance(*args, **kwargs)
        if capture["enabled"] and kwargs.get("return_logp", True):
            capture["x"] = args[0].detach().cpu()
            capture["q"] = result.detach().cpu()
        return result

    def counted_guide(x: object, i: int, **kwargs: object) -> object:
        before = {
            "solver": trace["solver_update_calls"],
            "predict": trace["predict_xt_calls"],
            "tilde": trace["tilde_guidance_calls"],
        }
        result = original_guide(x, i, **kwargs)
        trace["guide_step_calls"] += 1
        if len(trace["first_policy_step_trace"]) < sampler.args.inference_steps:
            trace["first_policy_step_trace"].append(
                {
                    "denoise_index": i,
                    "solver_calls": trace["solver_update_calls"] - before["solver"],
                    "predict_xt_calls": trace["predict_xt_calls"] - before["predict"],
                    "tilde_guidance_calls": trace["tilde_guidance_calls"] - before["tilde"],
                }
            )
        return result

    def checked_tensor_to_obj(x: object, num_states: int, multiple_input: bool) -> object:
        capture["enabled"] = True
        capture["x"] = None
        capture["q"] = None
        try:
            result = original_tensor_to_obj(x, num_states, multiple_input)
        finally:
            capture["enabled"] = False
        captured_x = capture["x"]
        captured_q = capture["q"]
        if captured_x is None or captured_q is None:
            raise AssertionError("final Q values were not captured")
        particle_count = sampler.args.per_sample_batch_size
        state_count = captured_x.shape[0] // particle_count
        expected = []
        selected_indices = []
        for x_chunk, q_chunk in zip(captured_x.chunk(state_count, dim=0), captured_q.chunk(state_count, dim=0)):
            selected_index = int(q_chunk.reshape(-1).argmax().item())
            selected_indices.append(selected_index)
            expected.append(x_chunk[selected_index].numpy())
        observed = np_module.asarray(result)
        expected_array = np_module.asarray(expected)
        verified = bool(np_module.allclose(observed, expected_array, rtol=0.0, atol=0.0))
        trace["final_selection_checks"].append(
            {
                "state_count": state_count,
                "candidate_count": int(captured_x.shape[0]),
                "particles_per_state": particle_count,
                "selected_indices": selected_indices,
                "exact_action_match": verified,
            }
        )
        if not verified:
            raise AssertionError("official tensor_to_obj output is not the exact max-Q candidate")
        return result

    sampler.solver_update = counted_solver
    sampler._predict_xt = counted_predict
    sampler.tilde_get_guidance = counted_tilde
    sampler.guide_step = counted_guide
    model.calculateQ = counted_q
    sampler.get_guidance = captured_get_guidance
    sampler.tensor_to_obj = checked_tensor_to_obj
    return trace


def main() -> int:
    cli = parse_args()
    source_root = cli.source_root.resolve()
    asset_root = cli.asset_root.resolve()
    manifest_path = cli.manifest.resolve()
    output_dir = cli.output_dir.resolve()
    work_dir = cli.work_dir.resolve()
    official_locomotion = source_root / "locomotion"
    if not official_locomotion.is_dir():
        raise FileNotFoundError(official_locomotion)
    if output_dir.exists():
        raise FileExistsError(f"immutable output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if cli.source_sha != manifest["official_source"]["commit"]:
        raise ValueError("source SHA differs from canonical manifest")
    if cli.task not in manifest["tasks"]:
        raise KeyError(cli.task)
    if cli.seed not in manifest["outer_seeds"]:
        raise ValueError(f"non-canonical seed: {cli.seed}")
    expected_episodes = 1 if cli.run_kind == "smoke" else manifest["episodes_per_seed"]
    if cli.run_kind != "checkpoint" and cli.episodes != expected_episodes:
        raise ValueError(f"{cli.run_kind} requires {expected_episodes} episodes")

    task_config = manifest["tasks"][cli.task]
    fixed = manifest["fixed_search"]
    checkpoint_dir = asset_root / "models_rl" / f"{cli.task}0large_actor"
    actor_path = checkpoint_dir / "behavior_ckpt600.pth"
    critic_path = checkpoint_dir / "critic_ckpt100.pth"
    for required in (actor_path, critic_path):
        if not required.is_file():
            raise FileNotFoundError(required)

    if work_dir.exists():
        raise FileExistsError(f"work directory already exists: {work_dir}")
    work_dir.mkdir(parents=True)
    models_link = work_dir / "models_rl"
    try:
        models_link.symlink_to(asset_root / "models_rl", target_is_directory=True)
    except OSError:
        shutil.copytree(asset_root / "models_rl", models_link)

    os.chdir(work_dir)
    sys.path.insert(0, str(official_locomotion))

    import gym
    import numpy as np
    import torch
    import d4rl  # noqa: F401 - registers the legacy environments
    from pipeline import Pipeline
    from search.configs import Arguments

    random.seed(cli.seed)
    np.random.seed(cli.seed)
    torch.manual_seed(cli.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cli.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.set_device(cli.device)
        torch.cuda.reset_peak_memory_stats(cli.device)

    actor_summary = checkpoint_summary(actor_path, torch)
    critic_summary = checkpoint_summary(critic_path, torch)

    scientific_args = Arguments()
    scientific_args.dataset = cli.task
    scientific_args.device = cli.device
    scientific_args.seed = cli.seed
    scientific_args.per_sample_batch_size = task_config["particles"]
    scientific_args.recur_steps = task_config["n_recur"]
    scientific_args.iter_steps = task_config["n_iter"]
    scientific_args.rho = task_config["rho"]
    scientific_args.mu = task_config["mu"]
    scientific_args.sigma = fixed["sigma"]
    scientific_args.eps_bsz = fixed["eps_bsz"]
    scientific_args.rho_schedule = fixed["rho_schedule"]
    scientific_args.mu_schedule = fixed["mu_schedule"]
    scientific_args.sigma_schedule = fixed["sigma_schedule"]
    scientific_args.guidance_strength = fixed["guidance_strength"]
    scientific_args.inference_steps = manifest["sampling_steps"]

    captured_warnings: list[dict[str, str]] = []
    load_started = time.perf_counter()
    with warnings.catch_warnings(record=True) as warning_records:
        warnings.simplefilter("always")
        pipeline = Pipeline(scientific_args)
        load_elapsed = time.perf_counter() - load_started
        trace = install_instrumentation(pipeline.sampler, pipeline.model, np)
        load_validation = {
            "actor_model_load": "PASS",
            "critic_model_load": "PASS",
            "model_class": type(pipeline.model).__name__,
            "sampler_class": type(pipeline.sampler).__name__,
        }

        if cli.run_kind == "checkpoint":
            episodes: list[dict[str, object]] = []
            evaluation_elapsed = 0.0
        else:
            envs = []
            active = []
            for episode_index in range(cli.episodes):
                env = gym.make(cli.task)
                env_seed = cli.seed + 1001 + episode_index
                env.seed(env_seed)
                env.buffer_state = env.reset()
                env.buffer_raw_return = 0.0
                env.buffer_steps = 0
                env.buffer_episode_index = episode_index
                env.buffer_env_seed = env_seed
                envs.append(env)
                active.append(env)

            episodes_path = output_dir / "episodes.jsonl"
            evaluation_started = time.perf_counter()
            with episodes_path.open("x", encoding="utf-8", buffering=1) as episode_file:
                while active:
                    states = np.stack([env.buffer_state for env in active])
                    before_steps = trace["guide_step_calls"]
                    actions = pipeline.sampler(states, diffusion_steps=manifest["sampling_steps"])
                    guide_steps = trace["guide_step_calls"] - before_steps
                    trace["policy_call_guide_steps"].append(guide_steps)
                    if guide_steps != manifest["sampling_steps"]:
                        raise AssertionError(f"expected 15 guide steps, saw {guide_steps}")
                    survivors = []
                    for active_index, env in enumerate(active):
                        state, reward, done, _info = env.step(actions[active_index])
                        env.buffer_raw_return += float(reward)
                        env.buffer_steps += 1
                        env.buffer_state = state
                        if done:
                            native = float(env.get_normalized_score(env.buffer_raw_return))
                            record = {
                                "schema_version": 1,
                                "task": cli.task,
                                "outer_seed": cli.seed,
                                "episode_index": env.buffer_episode_index,
                                "environment_seed": env.buffer_env_seed,
                                "steps": env.buffer_steps,
                                "raw_return": env.buffer_raw_return,
                                "d4rl_normalized_fraction": native,
                                "paper_normalized_score": 100.0 * native,
                            }
                            episode_file.write(json.dumps(record, sort_keys=True) + "\n")
                            episode_file.flush()
                        else:
                            survivors.append(env)
                    active = survivors
            evaluation_elapsed = time.perf_counter() - evaluation_started
            episodes = [
                json.loads(line)
                for line in episodes_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            for env in envs:
                env.close()

        for warning in warning_records:
            item = {
                "category": warning.category.__name__,
                "message": str(warning.message),
                "filename": str(warning.filename),
                "lineno": str(warning.lineno),
            }
            if item not in captured_warnings:
                captured_warnings.append(item)

    scores = [float(item["paper_normalized_score"]) for item in episodes]
    raw_returns = [float(item["raw_return"]) for item in episodes]
    first_trace = trace["first_policy_step_trace"]
    expected_tilde = (1 if task_config["rho"] != 0.0 else 0) + (
        task_config["n_iter"] if task_config["mu"] != 0.0 else 0
    )
    scientific_invariants = {
        "sampling_steps_per_policy_call": bool(
            not trace["policy_call_guide_steps"]
            or all(value == manifest["sampling_steps"] for value in trace["policy_call_guide_steps"])
        ),
        "particles": scientific_args.per_sample_batch_size == task_config["particles"],
        "n_recur": scientific_args.recur_steps == task_config["n_recur"],
        "n_iter": scientific_args.iter_steps == task_config["n_iter"],
        "rho": scientific_args.rho == task_config["rho"],
        "mu": scientific_args.mu == task_config["mu"],
        "first_policy_solver_calls_match_recurrence": bool(
            not first_trace or all(item["solver_calls"] == task_config["n_recur"] for item in first_trace)
        ),
        "first_policy_local_guidance_calls_match": bool(
            not first_trace or all(item["tilde_guidance_calls"] == expected_tilde * task_config["n_recur"] for item in first_trace)
        ),
        "q_verifier_called": trace["q_verifier_calls"] > 0 if cli.run_kind != "checkpoint" else True,
        "max_q_selection_exact": bool(
            not trace["final_selection_checks"]
            or all(item["exact_action_match"] for item in trace["final_selection_checks"])
        ),
    }
    status = "PASS" if all(scientific_invariants.values()) else "FAIL"
    if len(episodes) != (0 if cli.run_kind == "checkpoint" else cli.episodes):
        status = "FAIL"

    environment = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "packages": package_versions(
            ["torch", "numpy", "scipy", "gym", "D4RL", "mujoco-py", "diffusers", "tensorboard"]
        ),
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count(),
        "gpu_names": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "nvidia_smi": command_text(["nvidia-smi", "-L"]) if shutil.which("nvidia-smi") else None,
    }
    memory = {
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated(cli.device)) if torch.cuda.is_available() else 0,
        "max_reserved_bytes": int(torch.cuda.max_memory_reserved(cli.device)) if torch.cuda.is_available() else 0,
    }
    summary = {
        "schema_version": 1,
        "status": status,
        "run_kind": cli.run_kind,
        "task": cli.task,
        "outer_seed": cli.seed,
        "episodes_requested": cli.episodes,
        "episodes_completed": len(episodes),
        "scientific_config": {
            "particles": scientific_args.per_sample_batch_size,
            "n_recur": scientific_args.recur_steps,
            "n_iter": scientific_args.iter_steps,
            "rho": scientific_args.rho,
            "mu": scientific_args.mu,
            "sigma": scientific_args.sigma,
            "sampling_steps": scientific_args.inference_steps,
        },
        "scientific_invariants": scientific_invariants,
        "instrumentation": trace,
        "checkpoint_load": load_validation,
        "checkpoints": {"actor": actor_summary, "critic": critic_summary},
        "source_git_sha": cli.source_sha,
        "harness_git_sha": cli.harness_sha,
        "manifest_sha256": sha256(manifest_path),
        "load_runtime_seconds": load_elapsed,
        "evaluation_runtime_seconds": evaluation_elapsed,
        "total_runtime_seconds": load_elapsed + evaluation_elapsed,
        "cuda_memory": memory,
        "raw_returns": raw_returns,
        "paper_normalized_scores": scores,
        "paper_score_mean": statistics.fmean(scores) if scores else None,
        "episode_population_std": statistics.pstdev(scores) if len(scores) > 1 else 0.0 if scores else None,
        "environment": environment,
        "warnings": captured_warnings,
    }
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "environment.json", environment)
    print("DCS_LOCOMOTION_SUMMARY " + json.dumps(summary, sort_keys=True))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise
