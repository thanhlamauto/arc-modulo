#!/usr/bin/env python3
"""Kaggle entry point for P0 canary and isolated Locomotion run units."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
import zipfile


WORKING = Path("/kaggle/working")
INPUT = Path("/kaggle/input")
RESULTS = WORKING / "dcs-results"
CONFIG = Path(__file__).with_name("job_config.json")
MUJOCO_URL = "https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz"
UV_VERSION = "0.10.9"


def run(
    command: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    timeout: int = 3600,
    check: bool = True,
) -> dict[str, object]:
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )
    payload = {
        "command": command,
        "returncode": completed.returncode,
        "runtime_seconds": time.perf_counter() - started,
        "stdout_tail": completed.stdout[-16000:],
        "stderr_tail": completed.stderr[-16000:],
    }
    print("DCS_COMMAND " + json.dumps(payload, sort_keys=True))
    if check and completed.returncode != 0:
        raise RuntimeError(f"command failed ({completed.returncode}): {command}")
    return payload


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def find_input(filename: str) -> Path:
    candidates = [path for path in INPUT.rglob(filename) if path.is_file()]
    if len(candidates) != 1:
        raise FileNotFoundError(f"expected one {filename} under Kaggle input, found {candidates}")
    return candidates[0]


def safe_extract_zip(archive_path: Path, target: Path) -> None:
    with zipfile.ZipFile(archive_path, "r") as archive:
        for info in archive.infolist():
            pure = PurePosixPath(info.filename)
            if pure.is_absolute() or ".." in pure.parts:
                raise ValueError(f"unsafe ZIP member: {info.filename}")
        archive.extractall(target)


def verify_bundle(root: Path) -> None:
    sums_path = root / "SHA256SUMS"
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        observed = sha256(root / relative)
        if observed != expected:
            raise ValueError(f"bundle hash mismatch: {relative}")


def prepare_bundle() -> tuple[Path, Path, dict[str, object], dict[str, object]]:
    bundle_archive = find_input("source_bundle.zip")
    asset_archive = find_input("models_rl.zip")
    asset_manifest_path = find_input("assets_manifest.json")
    root = WORKING / "dcs-bundle"
    assets = WORKING / "dcs-assets"
    root.mkdir(parents=True, exist_ok=False)
    assets.mkdir(parents=True, exist_ok=False)
    safe_extract_zip(bundle_archive, root)
    verify_bundle(root)
    bundle_manifest = json.loads((root / "bundle_manifest.json").read_text(encoding="utf-8"))
    asset_manifest = json.loads(asset_manifest_path.read_text(encoding="utf-8"))
    if sha256(asset_archive) != asset_manifest["archive_sha256"]:
        raise ValueError("official asset archive hash mismatch")
    safe_extract_zip(asset_archive, assets)
    for record in asset_manifest["checkpoints"]:
        checkpoint = assets / record["path"]
        if checkpoint.stat().st_size != record["bytes"] or sha256(checkpoint) != record["sha256"]:
            raise ValueError(f"checkpoint verification failed: {record['path']}")
    return root, assets, bundle_manifest, asset_manifest


def install_system_and_mujoco() -> dict[str, object]:
    apt = run(
        [
            "apt-get",
            "update",
            "-qq",
        ],
        timeout=900,
    )
    apt_install = run(
        [
            "apt-get",
            "install",
            "-y",
            "-qq",
            "build-essential",
            "libgl1",
            "libgl1-mesa-dev",
            "libglew-dev",
            "libglfw3",
            "libglfw3-dev",
            "libosmesa6-dev",
            "patchelf",
        ],
        timeout=1200,
    )
    mujoco_archive = WORKING / "mujoco210-linux-x86_64.tar.gz"
    urllib.request.urlretrieve(MUJOCO_URL, mujoco_archive)
    mujoco_sha = sha256(mujoco_archive)
    mujoco_home = Path.home() / ".mujoco"
    mujoco_home.mkdir(parents=True, exist_ok=True)
    with tarfile.open(mujoco_archive, "r:gz") as archive:
        for member in archive.getmembers():
            pure = PurePosixPath(member.name)
            if pure.is_absolute() or ".." in pure.parts:
                raise ValueError(f"unsafe MuJoCo archive member: {member.name}")
        archive.extractall(mujoco_home)
    return {
        "url": MUJOCO_URL,
        "archive_bytes": mujoco_archive.stat().st_size,
        "archive_sha256": mujoco_sha,
        "mujoco_path": str(mujoco_home / "mujoco210"),
        "apt_update": apt,
        "apt_install": apt_install,
    }


def prepare_environment(harness: Path) -> tuple[Path, dict[str, object], dict[str, str]]:
    system = install_system_and_mujoco()
    uv = shutil.which("uv")
    install = None
    if uv is None:
        install = run(
            [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", f"uv=={UV_VERSION}"],
            timeout=600,
        )
        uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv unavailable after installation")
    project = harness / "envs" / "diffusion-classical-search-locomotion"
    venv = WORKING / "dcs-venv"
    env = os.environ.copy()
    mujoco_bin = str(Path.home() / ".mujoco" / "mujoco210" / "bin")
    env.update(
        {
            "UV_PROJECT_ENVIRONMENT": str(venv),
            "UV_CACHE_DIR": str(WORKING / ".uv-cache"),
            "UV_LINK_MODE": "copy",
            "UV_PYTHON_DOWNLOADS": "automatic",
            "MUJOCO_PY_MUJOCO_PATH": str(Path.home() / ".mujoco" / "mujoco210"),
            "LD_LIBRARY_PATH": ":".join(
                item for item in [mujoco_bin, "/usr/lib/nvidia", env.get("LD_LIBRARY_PATH", "")] if item
            ),
            "MUJOCO_GL": "osmesa",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "D4RL_SUPPRESS_IMPORT_ERROR": "1",
        }
    )
    sync = run([uv, "sync", "--frozen", "--no-install-project"], cwd=project, env=env, timeout=3600)
    python = venv / "bin" / "python"
    if not python.is_file():
        raise FileNotFoundError(python)
    summary = {
        "uv_version": run([uv, "--version"], env=env, timeout=60),
        "uv_install": install,
        "uv_sync": sync,
        "uv_lock_sha256": sha256(project / "uv.lock"),
        "system": system,
    }
    return python, summary, env


def runner_command(
    python: Path,
    harness: Path,
    official: Path,
    assets: Path,
    bundle_manifest: dict[str, object],
    task: str,
    seed: int,
    episodes: int,
    run_kind: str,
    output: Path,
    work: Path,
) -> list[str]:
    return [
        str(python),
        str(harness / "scripts" / "diffusion_classical_search" / "run_locomotion.py"),
        "--source-root",
        str(official),
        "--asset-root",
        str(assets),
        "--manifest",
        str(harness / "configs" / "diffusion_classical_search_table3.json"),
        "--output-dir",
        str(output),
        "--work-dir",
        str(work),
        "--task",
        task,
        "--seed",
        str(seed),
        "--episodes",
        str(episodes),
        "--device",
        "cuda:0",
        "--run-kind",
        run_kind,
        "--source-sha",
        str(bundle_manifest["official_source"]["git_sha"]),
        "--harness-sha",
        str(bundle_manifest["harness"]["git_sha"]),
    ]


def run_parallel_seeds(
    python: Path,
    harness: Path,
    official: Path,
    assets: Path,
    bundle_manifest: dict[str, object],
    base_env: dict[str, str],
    task: str,
    seeds: list[int],
) -> list[dict[str, object]]:
    import torch

    gpu_count = torch.cuda.device_count()
    if gpu_count < 1:
        raise RuntimeError("scientific run requires CUDA")
    pending = list(seeds)
    active: list[dict[str, object]] = []
    completed = []
    runs_root = RESULTS / "runs" / task
    works_root = WORKING / "dcs-run-work" / task
    runs_root.mkdir(parents=True, exist_ok=True)
    works_root.mkdir(parents=True, exist_ok=True)
    while pending or active:
        used_devices = {item["device"] for item in active}
        free_devices = [index for index in range(gpu_count) if index not in used_devices]
        while pending and free_devices:
            seed = pending.pop(0)
            device = free_devices.pop(0)
            output = runs_root / f"seed-{seed}"
            work = works_root / f"seed-{seed}"
            command = runner_command(
                python, harness, official, assets, bundle_manifest, task, seed, 10, "seed", output, work
            )
            env = base_env.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(device)
            stdout_path = runs_root / f"seed-{seed}.stdout.log"
            stderr_path = runs_root / f"seed-{seed}.stderr.log"
            stdout_handle = stdout_path.open("x", encoding="utf-8")
            stderr_handle = stderr_path.open("x", encoding="utf-8")
            process = subprocess.Popen(command, env=env, stdout=stdout_handle, stderr=stderr_handle, text=True)
            active.append(
                {
                    "seed": seed,
                    "device": device,
                    "process": process,
                    "stdout_handle": stdout_handle,
                    "stderr_handle": stderr_handle,
                    "started": time.perf_counter(),
                }
            )
        time.sleep(2)
        survivors = []
        for item in active:
            returncode = item["process"].poll()
            if returncode is None:
                survivors.append(item)
                continue
            item["stdout_handle"].close()
            item["stderr_handle"].close()
            record = {
                "seed": item["seed"],
                "physical_gpu": item["device"],
                "returncode": returncode,
                "runtime_seconds": time.perf_counter() - item["started"],
            }
            completed.append(record)
            print("DCS_SEED_PROCESS " + json.dumps(record, sort_keys=True))
            if returncode != 0:
                raise RuntimeError(f"seed process failed: {record}")
        active = survivors
    return completed


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    mode = config["mode"]
    RESULTS.mkdir(parents=True, exist_ok=True)
    if mode == "canary":
        return run([sys.executable, str(Path(__file__).with_name("t4_canary.py"))], timeout=300)["returncode"]

    root, assets, bundle_manifest, asset_manifest = prepare_bundle()
    harness = root / "harness"
    official = root / "official"
    python, environment_setup, env = prepare_environment(harness)
    write_json(RESULTS / "environment_setup.json", environment_setup)
    write_json(RESULTS / "bundle_manifest.json", bundle_manifest)
    write_json(RESULTS / "assets_manifest.json", asset_manifest)
    preflight = run(
        [
            str(python),
            str(harness / "scripts" / "diffusion_classical_search" / "preflight_legacy_env.py"),
            "--task",
            config.get("task", "halfcheetah-medium-expert-v2"),
            "--output",
            str(RESULTS / "legacy_environment_preflight.json"),
        ],
        env=env,
        timeout=900,
    )
    write_json(RESULTS / "legacy_environment_preflight_command.json", preflight)

    task = config.get("task", "halfcheetah-medium-expert-v2")
    if mode == "checkpoint":
        command = runner_command(
            python,
            harness,
            official,
            assets,
            bundle_manifest,
            task,
            0,
            0,
            "checkpoint",
            RESULTS / "checkpoint",
            WORKING / "dcs-run-work" / "checkpoint",
        )
        run(command, env=env, timeout=900)
    elif mode == "smoke":
        command = runner_command(
            python,
            harness,
            official,
            assets,
            bundle_manifest,
            task,
            0,
            1,
            "smoke",
            RESULTS / "smoke" / task / "seed-0",
            WORKING / "dcs-run-work" / "smoke-seed-0",
        )
        run(command, env=env, timeout=3600)
    elif mode == "seed_replay":
        replay_summaries = []
        for repeat in (1, 2):
            command = runner_command(
                python,
                harness,
                official,
                assets,
                bundle_manifest,
                task,
                0,
                10,
                "seed",
                RESULTS / "replay" / f"repeat-{repeat}" / task / "seed-0",
                WORKING / "dcs-run-work" / f"replay-{repeat}-seed-0",
            )
            run(command, env=env, timeout=14400)
            replay_summaries.append(
                json.loads(
                    (RESULTS / "replay" / f"repeat-{repeat}" / task / "seed-0" / "summary.json").read_text(
                        encoding="utf-8"
                    )
                )
            )
        first = replay_summaries[0]["paper_normalized_scores"]
        second = replay_summaries[1]["paper_normalized_scores"]
        exact = first == second and replay_summaries[0]["raw_returns"] == replay_summaries[1]["raw_returns"]
        max_delta = max((abs(float(a) - float(b)) for a, b in zip(first, second)), default=0.0)
        replay = {
            "status": "EXACT" if exact else "NUMERICALLY_CLOSE" if max_delta <= 1e-6 else "STOCHASTIC",
            "exact": exact,
            "maximum_episode_score_delta": max_delta,
            "repeat_1_scores": first,
            "repeat_2_scores": second,
        }
        write_json(RESULTS / "replay" / "replay_comparison.json", replay)
        if replay["status"] == "STOCHASTIC":
            raise RuntimeError(f"unexplained seed replay divergence: {replay}")
    elif mode in ("first_row", "table_task"):
        process_results = run_parallel_seeds(
            python,
            harness,
            official,
            assets,
            bundle_manifest,
            env,
            task,
            [0, 1, 2, 3, 4],
        )
        write_json(RESULTS / "runs" / task / "process_results.json", process_results)
        table_path = RESULTS / "runs" / task / "table3_reproduction.json"
        aggregate = run(
            [
                str(python),
                str(harness / "scripts" / "diffusion_classical_search" / "aggregate_table3.py"),
                "--manifest",
                str(harness / "configs" / "diffusion_classical_search_table3.json"),
                "--runs-root",
                str(RESULTS / "runs"),
                "--output",
                str(table_path),
                "--task",
                task,
            ],
            env=env,
            timeout=300,
            check=False,
        )
        write_json(RESULTS / "runs" / task / "aggregate_command.json", aggregate)
        table = json.loads(table_path.read_text(encoding="utf-8"))
        status = table["tasks"][0]["status"]
        if status == "FAIL":
            raise RuntimeError(f"scientific gate failed for {task}")
    else:
        raise ValueError(f"unsupported mode: {mode}")

    write_json(
        RESULTS / "job_summary.json",
        {
            "schema_version": 1,
            "status": "PASS",
            "mode": mode,
            "task": task,
            "source_git_sha": bundle_manifest["official_source"]["git_sha"],
            "harness_git_sha": bundle_manifest["harness"]["git_sha"],
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
