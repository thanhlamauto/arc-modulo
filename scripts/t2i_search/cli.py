#!/usr/bin/env python3
"""One entrypoint for planning, running, resuming, and analyzing paired T2I search."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

try:
    from .backend import build_backend
    from .core import (
        ConfigError,
        ExperimentSpec,
        Shard,
        parse_int_list,
        parse_prompt_range,
        plan_summary,
        status_summary,
    )
except ImportError:  # Support `python scripts/t2i_search/cli.py ...`.
    from backend import build_backend
    from core import (
        ConfigError,
        ExperimentSpec,
        Shard,
        parse_int_list,
        parse_prompt_range,
        plan_summary,
        status_summary,
    )


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "t2i_search.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("plan", "status", "prepare", "smoke", "run", "analyze", "full"),
        help="full = prepare + complete unsharded grid + analyze",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--official-root", type=Path)
    parser.add_argument("--baseline-results", type=Path)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--tpu-chip", help="one visible TPU chip for this worker")
    parser.add_argument("--shard", default="0/1", help="paired-unit shard as INDEX/COUNT")
    parser.add_argument("--prompts", dest="prompt_range", help="half-open START:STOP")
    parser.add_argument("--particles", help="comma-separated subset, e.g. 4,8")
    parser.add_argument("--seeds", help="comma-separated subset")
    return parser


def _print(value: object) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), flush=True)


def _is_filtered(args: argparse.Namespace) -> bool:
    return any(value is not None for value in (args.prompt_range, args.particles, args.seeds))


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    backend = None
    try:
        spec = ExperimentSpec.load(args.config).override_paths(
            run_root=args.run_root,
            official_root=args.official_root,
            baseline_results=args.baseline_results,
            cache_root=args.cache_root,
            tpu_chip=args.tpu_chip,
        )
        shard = Shard.parse(args.shard)
        prompt_range = parse_prompt_range(args.prompt_range, spec.prompt_count)
        particles = parse_int_list(args.particles)
        seeds = parse_int_list(args.seeds)
        selected = shard.select(
            spec.pairs(particles=particles, seeds=seeds, prompt_range=prompt_range)
        )

        if args.command == "plan":
            _print(plan_summary(spec, selected, shard))
            return 0
        if args.command == "status":
            _print(status_summary(spec))
            return 0
        if args.command in ("analyze", "full"):
            if shard != Shard() or _is_filtered(args):
                raise ConfigError(f"{args.command} requires the complete unsharded protocol grid")
        if args.command in ("smoke", "run", "full") and not selected:
            raise ConfigError("selection produced no paired units")

        backend = build_backend(spec, shard_label=shard.label)
        backend.status("PREPARING")
        backend.prepare()
        if args.command == "prepare":
            backend.status("PREPARED_UNIT_TESTS_PASSED")
            return 0
        if args.command == "analyze":
            summary = status_summary(spec)
            if not summary["complete"]:
                raise ConfigError(
                    f"analysis requires a complete grid; {summary['remaining_pairs']} pairs remain"
                )
            backend.status("ANALYZING")
            backend.analyze()
            backend.status("COMPLETE")
            return 0

        backend.status("INITIALIZING_ACCELERATOR")
        backend.start_accelerator()
        keys = [spec.smoke] if args.command == "smoke" else selected
        for ordinal, key in enumerate(keys, start=1):
            backend.run_pair(key)
            backend.status("RUNNING", completed_by_worker=ordinal, assigned_to_worker=len(keys))
        if args.command == "smoke":
            backend.status("SMOKE_VALID")
            return 0
        backend.status("SHARD_COMPLETE", completed_by_worker=len(keys))
        if args.command == "full":
            summary = status_summary(spec)
            if not summary["complete"]:
                raise ConfigError("full run ended with an incomplete artifact grid")
            backend.status("ANALYZING")
            backend.analyze()
            backend.status("COMPLETE")
        return 0
    except (ConfigError, ValueError) as exc:
        if backend is not None:
            try:
                backend.status("STOPPED_FOR_REVIEW", error_type=type(exc).__name__, detail=str(exc))
            except Exception:
                pass
        print(f"error: {exc}", file=sys.stderr, flush=True)
        return 2
    except Exception as exc:
        if backend is not None:
            try:
                backend.status("STOPPED_FOR_REVIEW", error_type=type(exc).__name__, detail=str(exc))
            except Exception:
                pass
        # Preserve the traceback for runtime/scientific failures.
        print(f"fatal: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
