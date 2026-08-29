#!/usr/bin/env python3
"""Validate the exact legacy Gym/D4RL/MuJoCo API used by the official source."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import platform

import numpy as np


def version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="halfcheetah-medium-expert-v2")
    parser.add_argument("--output", type=Path, required=True)
    cli = parser.parse_args()

    import gym
    import d4rl  # noqa: F401 - environment registration
    import mujoco_py

    env = gym.make(cli.task)
    env.seed(1001)
    env.action_space.seed(1001)
    observation = env.reset()
    action = env.action_space.sample()
    transition = env.step(action)
    if not isinstance(transition, tuple) or len(transition) != 4:
        raise AssertionError(f"legacy step API expected four values, got {type(transition)} / {len(transition)}")
    next_observation, reward, done, info = transition
    native = float(env.get_normalized_score(float(reward)))
    ref_min = float(env.ref_min_score)
    ref_max = float(env.ref_max_score)
    independently_computed = (float(reward) - ref_min) / (ref_max - ref_min)
    if not np.isclose(native, independently_computed, rtol=0.0, atol=1e-12):
        raise AssertionError("D4RL normalization semantics differ from source formula")
    env.close()
    payload = {
        "schema_version": 1,
        "status": "PASS",
        "task": cli.task,
        "python": platform.python_version(),
        "packages": {
            "gym": version("gym"),
            "D4RL": version("D4RL"),
            "mujoco-py": version("mujoco-py"),
            "numpy": version("numpy"),
        },
        "mujoco_py_version": getattr(mujoco_py, "__version__", None),
        "reset_observation_shape": list(np.asarray(observation).shape),
        "step_observation_shape": list(np.asarray(next_observation).shape),
        "action_shape": list(np.asarray(action).shape),
        "step_api_tuple_length": len(transition),
        "one_step_reward": float(reward),
        "one_step_done": bool(done),
        "one_step_info_keys": sorted(str(key) for key in info),
        "ref_min_score": ref_min,
        "ref_max_score": ref_max,
        "d4rl_native_normalized_fraction": native,
        "independent_normalized_fraction": independently_computed,
        "paper_scale_definition": "100 * d4rl_native_normalized_fraction",
    }
    cli.output.parent.mkdir(parents=True, exist_ok=True)
    cli.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("DCS_LEGACY_ENV_PREFLIGHT " + json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
