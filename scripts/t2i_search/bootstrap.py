#!/usr/bin/env python3
"""Prepare portable provenance inputs by reusing a completed baseline cache."""
from __future__ import annotations

import argparse
import importlib.metadata
from pathlib import Path
import shutil
import sys
from typing import Sequence

try:
    from .core import ExperimentSpec, read_json, require, sha256, write_json
except ImportError:  # Support direct execution.
    from core import ExperimentSpec, read_json, require, sha256, write_json


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "t2i_search.json"


def _gamma_evidence(baseline_results: Path, explicit: Path | None) -> Path:
    candidates = [
        explicit,
        baseline_results
        / "export/results/diffusion-classical-search/t2i/table1/table1_reproduced.csv",
        baseline_results / "table1/table1_reproduced.csv",
        REPO_ROOT / "results/t2i-search/classical-search/table1_reproduced.csv",
    ]
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return candidate
    raise ValueError("table1_reproduced.csv was not found; run the Classical Search baseline first")


def _environment_freeze() -> str:
    rows = {
        f"{distribution.metadata.get('Name', 'unknown')}=={distribution.version}"
        for distribution in importlib.metadata.distributions()
    }
    return "\n".join(sorted(rows, key=str.lower)) + "\n"


def bootstrap_artifacts(
    spec: ExperimentSpec,
    *,
    gamma_evidence: Path | None = None,
) -> dict[str, object]:
    """Verify reused model files and create the inputs required by frozen preflight."""
    asset_manifest_path = spec.baseline_results / "asset_manifest.json"
    manifest = read_json(asset_manifest_path)
    entries = manifest.get("files")
    require(isinstance(entries, list) and entries, "baseline asset manifest has no files")
    selected = [
        entry
        for entry in entries
        if "models--runwayml--stable-diffusion-v1-5/snapshots/" in entry.get("path", "")
        or "models--bert-base-uncased/snapshots/" in entry.get("path", "")
        or entry.get("path", "").startswith("ImageReward/")
    ]
    require(selected, "baseline manifest contains no SD1.5/ImageReward assets")

    for entry in selected:
        path = spec.cache_root / entry["path"]
        require(path.is_file(), f"missing cached asset: {path}")
        require(path.stat().st_size == entry["bytes"], f"asset size changed: {path}")
        require(sha256(path) == entry["sha256"], f"asset hash changed: {path}")

    evidence = _gamma_evidence(spec.baseline_results, gamma_evidence)
    expected_gamma_hash = spec.protocol["bfs"]["gamma_source_sha256"]
    require(sha256(evidence) == expected_gamma_hash, "gamma-selection evidence hash changed")
    gamma_target = spec.run_root / "harness/provenance/table1_reproduced.csv"
    gamma_target.parent.mkdir(parents=True, exist_ok=True)
    if evidence.resolve() != gamma_target.resolve():
        shutil.copy2(evidence, gamma_target)

    spec.output.mkdir(parents=True, exist_ok=True)
    restored_assets = {
        "all_match_historical_hashes": True,
        "source_manifest": str(asset_manifest_path),
        "files": selected,
    }
    write_json(spec.output / "restored_assets.json", restored_assets)
    environment_path = spec.output / "restored_environment.txt"
    temporary = environment_path.with_suffix(".txt.tmp")
    temporary.write_text(_environment_freeze(), encoding="utf-8")
    temporary.replace(environment_path)

    status = {
        "ready": True,
        "verified_assets": len(selected),
        "asset_manifest_sha256": sha256(asset_manifest_path),
        "gamma_evidence": str(gamma_target),
        "gamma_evidence_sha256": sha256(gamma_target),
        "environment_sha256": sha256(environment_path),
        "cache_root": str(spec.cache_root),
    }
    write_json(spec.output / "bootstrap_status.json", status)
    return status


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--baseline-results", type=Path)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--gamma-evidence", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        spec = ExperimentSpec.load(args.config).override_paths(
            run_root=args.run_root,
            baseline_results=args.baseline_results,
            cache_root=args.cache_root,
        )
        status = bootstrap_artifacts(spec, gamma_evidence=args.gamma_evidence)
        print(f"READY: verified {status['verified_assets']} cached assets", flush=True)
        return 0
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
