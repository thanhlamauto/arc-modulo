#!/usr/bin/env python3
"""Load the first official checkpoint pair into the original CPU model definitions."""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import torch


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--task", default="halfcheetah-medium-expert-v2")
    parser.add_argument("--output", type=Path)
    cli = parser.parse_args()

    locomotion = cli.source_root.resolve() / "locomotion"
    sys.path.insert(0, str(locomotion))
    from diffusion_SDE.model import ScoreNet
    from diffusion_SDE.schedule import marginal_prob_std

    dimensions = {
        "halfcheetah": (17, 6),
        "hopper": (11, 3),
        "walker2d": (17, 6),
    }
    family = cli.task.split("-", 1)[0]
    state_dim, action_dim = dimensions[family]
    args = SimpleNamespace(device="cpu", alpha=3.0, q_alpha=1.0, method="CEP")
    marginal = functools.partial(marginal_prob_std, device="cpu")
    model = ScoreNet(
        input_dim=state_dim + action_dim,
        output_dim=action_dim,
        marginal_prob_std=marginal,
        args=args,
    )
    checkpoint_dir = cli.asset_root.resolve() / "models_rl" / f"{cli.task}0large_actor"
    actor_path = checkpoint_dir / "behavior_ckpt600.pth"
    critic_path = checkpoint_dir / "critic_ckpt100.pth"
    actor = torch.load(actor_path, map_location="cpu", weights_only=False)
    critic = torch.load(critic_path, map_location="cpu", weights_only=False)
    model.load_state_dict(actor, strict=True)
    critic_compatibility = model.q[0].load_state_dict(critic, strict=False)
    payload = {
        "schema_version": 1,
        "status": "PASS",
        "task": cli.task,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "model_class": type(model).__name__,
        "actor": {
            "bytes": actor_path.stat().st_size,
            "sha256": digest(actor_path),
            "key_count": len(actor),
            "strict_load": "PASS",
        },
        "critic": {
            "bytes": critic_path.stat().st_size,
            "sha256": digest(critic_path),
            "key_count": len(critic),
            "missing_keys": list(critic_compatibility.missing_keys),
            "unexpected_keys": list(critic_compatibility.unexpected_keys),
        },
    }
    if cli.output:
        cli.output.parent.mkdir(parents=True, exist_ok=True)
        cli.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
