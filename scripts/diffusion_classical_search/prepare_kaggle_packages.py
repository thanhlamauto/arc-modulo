#!/usr/bin/env python3
"""Build credential-safe Kaggle source and asset packages from strict allowlists."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import subprocess
import zipfile


EXPECTED_SOURCE_SHA = "c158ab9eb058adb24ad0f8d48506ab837d7fae91"
OFFICIAL_ASSET_URL = "https://drive.google.com/drive/folders/1snFcmcJaalcCWW9roBjeCjpWjpCeDM_P?usp=drive_link"
TASKS = (
    "halfcheetah-medium-expert-v2",
    "hopper-medium-expert-v2",
    "walker2d-medium-expert-v2",
    "halfcheetah-medium-v2",
    "hopper-medium-v2",
    "walker2d-medium-v2",
    "halfcheetah-medium-replay-v2",
    "hopper-medium-replay-v2",
    "walker2d-medium-replay-v2",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--harness-repo", type=Path, required=True)
    parser.add_argument("--asset-archive", type=Path, required=True)
    parser.add_argument("--extracted-asset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def git(repo: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr)
    return completed.stdout.strip()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def archive_bytes(repo: Path, revision: str) -> bytes:
    completed = subprocess.run(
        ["git", "-C", str(repo), "archive", "--format=zip", revision],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.decode("utf-8", errors="replace"))
    return completed.stdout


def safe_members(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    members = []
    for info in archive.infolist():
        pure = PurePosixPath(info.filename)
        if pure.is_absolute() or ".." in pure.parts:
            raise ValueError(f"unsafe archive member: {info.filename}")
        members.append(info)
    return members


def add_archive(target: zipfile.ZipFile, raw_archive: bytes, prefix: str, hashes: dict[str, str]) -> None:
    with zipfile.ZipFile(io.BytesIO(raw_archive), "r") as source:
        for info in safe_members(source):
            if info.is_dir():
                continue
            data = source.read(info)
            name = f"{prefix}/{info.filename}"
            target.writestr(name, data)
            hashes[name] = sha256_bytes(data)


def main() -> int:
    cli = parse_args()
    official = cli.official_repo.resolve()
    harness = cli.harness_repo.resolve()
    asset_archive = cli.asset_archive.resolve()
    asset_root = cli.extracted_asset_root.resolve()
    output_root = cli.output_root.resolve()
    source_dir = output_root / "source-dataset"
    asset_dir = output_root / "asset-dataset"
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"package output must be new or empty: {output_root}")
    source_dir.mkdir(parents=True)
    asset_dir.mkdir(parents=True)

    official_sha = git(official, "rev-parse", "HEAD")
    harness_sha = git(harness, "rev-parse", "HEAD")
    if official_sha != EXPECTED_SOURCE_SHA:
        raise ValueError(f"official HEAD mismatch: {official_sha}")
    if git(official, "status", "--short"):
        raise ValueError("official repository is dirty")
    if git(harness, "status", "--short"):
        raise ValueError("harness repository is dirty; commit the exact package source first")

    official_archive = archive_bytes(official, official_sha)
    harness_archive = archive_bytes(harness, harness_sha)
    lock_path = harness / "envs" / "diffusion-classical-search-locomotion" / "uv.lock"
    canonical_manifest = harness / "configs" / "diffusion_classical_search_table3.json"
    bundle_manifest = {
        "schema_version": 1,
        "paper": "arXiv:2505.23614v2",
        "official_source": {
            "url": "https://github.com/XiangchengZhang/Diffusion-inference-scaling.git",
            "git_sha": official_sha,
            "git_archive_sha256": sha256_bytes(official_archive),
        },
        "harness": {
            "git_sha": harness_sha,
            "git_archive_sha256": sha256_bytes(harness_archive),
        },
        "environment_lock_sha256": sha256_file(lock_path),
        "canonical_manifest_sha256": sha256_file(canonical_manifest),
        "credential_packaging_policy": "git-archive allowlist only; kaggle.json impossible to include",
    }
    bundle_path = source_dir / "source_bundle.zip"
    hashes: dict[str, str] = {}
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as target:
        add_archive(target, official_archive, "official", hashes)
        add_archive(target, harness_archive, "harness", hashes)
        manifest_bytes = (json.dumps(bundle_manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
        target.writestr("bundle_manifest.json", manifest_bytes)
        hashes["bundle_manifest.json"] = sha256_bytes(manifest_bytes)
        sums = "".join(f"{digest}  {name}\n" for name, digest in sorted(hashes.items())).encode("utf-8")
        target.writestr("SHA256SUMS", sums)
    (source_dir / "bundle_manifest.json").write_text(
        json.dumps(bundle_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    archive_hash = sha256_file(asset_archive)
    checkpoint_records = []
    models_root = asset_root / "models_rl"
    for task in TASKS:
        for filename in ("behavior_ckpt600.pth", "critic_ckpt100.pth"):
            checkpoint = models_root / f"{task}0large_actor" / filename
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            checkpoint_records.append(
                {
                    "task": task,
                    "path": str(checkpoint.relative_to(asset_root)).replace("\\", "/"),
                    "bytes": checkpoint.stat().st_size,
                    "sha256": sha256_file(checkpoint),
                }
            )
    asset_manifest = {
        "schema_version": 1,
        "source_url": OFFICIAL_ASSET_URL,
        "archive_name": asset_archive.name,
        "archive_bytes": asset_archive.stat().st_size,
        "archive_sha256": archive_hash,
        "expected_checkpoint_count": 18,
        "checkpoints": checkpoint_records,
    }
    (asset_dir / "assets_manifest.json").write_text(
        json.dumps(asset_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    target_archive = asset_dir / asset_archive.name
    target_archive.write_bytes(asset_archive.read_bytes())
    package_summary = {
        "schema_version": 1,
        "source_dataset_directory": str(source_dir),
        "source_bundle_bytes": bundle_path.stat().st_size,
        "source_bundle_sha256": sha256_file(bundle_path),
        "asset_dataset_directory": str(asset_dir),
        "asset_archive_bytes": target_archive.stat().st_size,
        "asset_archive_sha256": sha256_file(target_archive),
        "official_source_status": "CLEAN",
        "credential_included": False,
        "bundle_manifest": bundle_manifest,
    }
    (output_root / "package_summary.json").write_text(
        json.dumps(package_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(package_summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
