#!/usr/bin/env python3
"""Fail-closed aggregation for isolated Locomotion seed directories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task")
    return parser.parse_args()


def main() -> int:
    cli = parse_args()
    manifest = json.loads(cli.manifest.read_text(encoding="utf-8"))
    task_names = [cli.task] if cli.task else list(manifest["tasks"])
    task_results = []
    for task in task_names:
        summaries = []
        for path in sorted(cli.runs_root.glob(f"**/{task}/seed-*/summary.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("run_kind") == "seed":
                summaries.append(payload)
        by_seed = {int(item["outer_seed"]): item for item in summaries}
        expected_seeds = set(manifest["outer_seeds"])
        completed_seeds = {
            seed
            for seed, item in by_seed.items()
            if item["status"] == "PASS" and item["episodes_completed"] == manifest["episodes_per_seed"]
        }
        seed_means = [float(by_seed[seed]["paper_score_mean"]) for seed in sorted(completed_seeds)]
        episode_scores = [
            float(score)
            for seed in sorted(completed_seeds)
            for score in by_seed[seed]["paper_normalized_scores"]
        ]
        raw_returns = [
            float(score)
            for seed in sorted(completed_seeds)
            for score in by_seed[seed]["raw_returns"]
        ]
        complete = completed_seeds == expected_seeds and len(episode_scores) == 50
        reproduced_mean = statistics.fmean(seed_means) if seed_means else None
        seed_population_std = statistics.pstdev(seed_means) if len(seed_means) > 1 else None
        episode_mean = statistics.fmean(episode_scores) if episode_scores else None
        episode_population_std = statistics.pstdev(episode_scores) if len(episode_scores) > 1 else None
        target = manifest["tasks"][task]
        delta = reproduced_mean - target["target_mean"] if reproduced_mean is not None else None
        if not complete or delta is None:
            status = "FAIL"
        elif abs(delta) <= manifest["gate_policy"]["reproduced_max_absolute_delta"]:
            status = "REPRODUCED"
        elif abs(delta) <= manifest["gate_policy"]["close_max_absolute_delta"]:
            status = "CLOSE"
        else:
            status = "FAIL"
        combined_spread = None
        standardized_discrepancy = None
        if seed_population_std is not None:
            combined_spread = (target["target_spread"] ** 2 + seed_population_std ** 2) ** 0.5
            if combined_spread > 0:
                standardized_discrepancy = delta / combined_spread
        task_results.append(
            {
                "task": task,
                "paper_mean": target["target_mean"],
                "paper_spread": target["target_spread"],
                "reproduced_seed_mean": reproduced_mean,
                "reproduced_seed_population_std": seed_population_std,
                "delta_mean": delta,
                "combined_spread": combined_spread,
                "standardized_discrepancy": standardized_discrepancy,
                "episode_mean": episode_mean,
                "episode_population_std": episode_population_std,
                "seed_means": seed_means,
                "all_50_normalized_scores": episode_scores,
                "all_50_raw_returns": raw_returns,
                "seeds_completed": sorted(completed_seeds),
                "episodes_completed": len(episode_scores),
                "runtime_seconds": sum(float(by_seed[seed]["total_runtime_seconds"]) for seed in completed_seeds),
                "peak_allocated_bytes": max(
                    (int(by_seed[seed]["cuda_memory"]["max_allocated_bytes"]) for seed in completed_seeds),
                    default=None,
                ),
                "peak_reserved_bytes": max(
                    (int(by_seed[seed]["cuda_memory"]["max_reserved_bytes"]) for seed in completed_seeds),
                    default=None,
                ),
                "status": status,
            }
        )

    successful_means = [item["reproduced_seed_mean"] for item in task_results if item["reproduced_seed_mean"] is not None]
    result = {
        "schema_version": 1,
        "paper": manifest["paper"],
        "source_git_sha": manifest["official_source"]["commit"],
        "paper_locomotion_average": manifest["paper_locomotion_average"],
        "reproduced_locomotion_average": statistics.fmean(successful_means) if len(successful_means) == 9 else None,
        "tasks": task_results,
    }
    cli.output.parent.mkdir(parents=True, exist_ok=True)
    cli.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if all(item["status"] != "FAIL" for item in task_results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
