#!/usr/bin/env python3
"""Aggregate complete four-trial Table 1/Table 2 ImageReward results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def gamma_id(method: str, particles: int, gamma: float) -> str:
    value = f"{gamma:.3f}".replace(".", "p")
    return f"{method}_g{value}_n{particles}"


def plain_id(method: str, particles: int) -> str:
    return f"{method}_n{particles}"


def load_trials(run_root: Path) -> tuple[dict[tuple[str, str, int], dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[tuple[str, str, int], dict[str, Any]] = {}
    all_records: list[dict[str, Any]] = []
    for path in sorted((run_root / "raw").glob("*/*/trial_seed*.jsonl")):
        records = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    records.append(json.loads(line))
        if not records:
            continue
        first = records[0]
        key = (first["model"], first["config"]["config_id"], int(first["trial_seed"]))
        indices = [int(record["prompt_index"]) for record in records]
        if len(indices) != len(set(indices)):
            raise RuntimeError(f"Duplicate prompt indices in {path}")
        groups[key] = {
            "model": key[0],
            "config_id": key[1],
            "trial_seed": key[2],
            "prompt_count": len(records),
            "complete": sorted(indices) == list(range(100)),
            "trial_mean": statistics.fmean(float(record["selected_final_score"]) for record in records),
            "elapsed_seconds": sum(float(record["elapsed_seconds"]) for record in records),
            "path": str(path.relative_to(run_root)),
        }
        all_records.extend(records)
    return groups, all_records


def summarize_config(
    groups: dict[tuple[str, str, int], dict[str, Any]],
    *,
    model: str,
    config_id: str,
    seeds: list[int],
    target: list[float],
) -> dict[str, Any]:
    trials = [groups.get((model, config_id, seed)) for seed in seeds]
    present = [trial for trial in trials if trial is not None]
    complete = len(present) == len(seeds) and all(trial["complete"] for trial in present)
    values = [float(trial["trial_mean"]) for trial in present if trial["complete"]]
    mean = statistics.fmean(values) if values else None
    std = statistics.stdev(values) if len(values) > 1 else (0.0 if len(values) == 1 else None)
    return {
        "config_id": config_id,
        "complete": complete,
        "completed_trials": len(values),
        "expected_trials": len(seeds),
        "trial_means": values,
        "mean": mean,
        "std": std,
        "paper_mean": float(target[0]),
        "paper_std": float(target[1]),
        "mean_delta": None if mean is None else mean - float(target[0]),
        "std_delta": None if std is None else std - float(target[1]),
    }


def select_gamma(
    groups: dict[tuple[str, str, int], dict[str, Any]],
    *,
    model: str,
    method: str,
    particles: int,
    seeds: list[int],
    gammas: list[float],
) -> tuple[str, dict[str, Any]]:
    candidates: dict[str, Any] = {}
    for gamma in gammas:
        config_id = gamma_id(method, particles, gamma)
        trials = [groups.get((model, config_id, seed)) for seed in seeds]
        values = [trial["trial_mean"] for trial in trials if trial and trial["complete"]]
        candidates[str(gamma)] = {
            "config_id": config_id,
            "complete_trials": len(values),
            "mean": statistics.fmean(values) if values else None,
        }
    eligible = [item for item in candidates.values() if item["complete_trials"] == len(seeds)]
    if eligible:
        selected = max(eligible, key=lambda item: item["mean"])["config_id"]
    else:
        partial = [item for item in candidates.values() if item["mean"] is not None]
        selected = max(partial, key=lambda item: item["mean"])["config_id"] if partial else gamma_id(method, particles, gammas[0])
    return selected, candidates


def build_table1(manifest: dict[str, Any], groups: dict[tuple[str, str, int], dict[str, Any]]) -> dict[str, Any]:
    seeds = [int(x) for x in manifest["trials"]["seeds"]]
    gammas = [float(x) for x in manifest["gamma_search"]]
    config_patterns = {
        "BoN": "bon",
        "Multinomial": "fk",
        "SSP": "constant_max_ssp",
        "Current": "constant_current_ssp",
        "Difference": "constant_difference_ssp",
        "Max": "constant_max_ssp",
        "Inf": "inf_max",
        "Constant": "constant_max_ssp",
    }
    rows: dict[str, Any] = {}
    selections: dict[str, Any] = {}
    for particles in manifest["table1"]["budgets"]:
        particles = int(particles)
        row: dict[str, Any] = {}
        targets = manifest["table1"]["targets"][str(particles)]
        selected_increase, candidates = select_gamma(
            groups,
            model="sd15",
            method="bfs",
            particles=particles,
            seeds=seeds,
            gammas=gammas,
        )
        selections[str(particles)] = {"selected": selected_increase, "candidates": candidates}
        for cell, method in config_patterns.items():
            row[cell] = summarize_config(
                groups,
                model="sd15",
                config_id=plain_id(method, particles),
                seeds=seeds,
                target=targets[cell],
            )
        row["Increase"] = summarize_config(
            groups,
            model="sd15",
            config_id=selected_increase,
            seeds=seeds,
            target=targets["Increase"],
        )
        rows[str(particles)] = row
    return {
        "table": 1,
        "metric": "ImageReward",
        "dispersion": "sample standard deviation across four trial means",
        "gamma_selection": selections,
        "rows": rows,
        "complete": all(cell["complete"] for row in rows.values() for cell in row.values()),
    }


def build_table2(manifest: dict[str, Any], groups: dict[tuple[str, str, int], dict[str, Any]]) -> dict[str, Any]:
    seeds = [int(x) for x in manifest["trials"]["seeds"]]
    gammas = [float(x) for x in manifest["gamma_search"]]
    rows: dict[str, Any] = {}
    selections: dict[str, Any] = {}
    for model in manifest["table2"]["models"]:
        rows[model] = {}
        selections[model] = {}
        for particles in manifest["table2"]["budgets"]:
            particles = int(particles)
            row: dict[str, Any] = {}
            targets = manifest["table2"]["targets"][model][str(particles)]
            for method in manifest["table2"]["methods"]:
                if method in {"das", "bfs"}:
                    config_id, candidates = select_gamma(
                        groups,
                        model=model,
                        method=method,
                        particles=particles,
                        seeds=seeds,
                        gammas=gammas,
                    )
                    selections[model][f"{method}_n{particles}"] = {
                        "selected": config_id,
                        "candidates": candidates,
                    }
                else:
                    config_id = plain_id(method, particles)
                row[method] = summarize_config(
                    groups,
                    model=model,
                    config_id=config_id,
                    seeds=seeds,
                    target=targets[method],
                )
            rows[model][str(particles)] = row
    return {
        "table": 2,
        "metric": "ImageReward",
        "dispersion": "sample standard deviation across four trial means",
        "gamma_selection": selections,
        "rows": rows,
        "complete": all(
            cell["complete"]
            for model_rows in rows.values()
            for row in model_rows.values()
            for cell in row.values()
        ),
    }


def value_text(cell: dict[str, Any]) -> str:
    if cell["mean"] is None:
        return "missing"
    marker = "" if cell["complete"] else " (partial)"
    return f"{cell['mean']:.3f} ± {cell['std']:.3f}{marker}"


def render_markdown(table1: dict[str, Any], table2: dict[str, Any], compute: dict[str, Any]) -> str:
    lines = ["# Table 1 Reproduction", ""]
    columns1 = ["BoN", "Multinomial", "SSP", "Current", "Difference", "Max", "Increase", "Inf", "Constant"]
    lines.append("| N | " + " | ".join(columns1) + " |")
    lines.append("|---:|" + "|".join(["---:"] * len(columns1)) + "|")
    for particles, row in table1["rows"].items():
        lines.append(f"| {particles} | " + " | ".join(value_text(row[column]) for column in columns1) + " |")
    lines.extend(["", "# Table 2 Reproduction", ""])
    columns2 = ["bon", "fk", "das", "treeg", "svdd", "bfs"]
    lines.append("| Model | N | BoN | FK | DAS | TreeG | SVDD | BFS |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for model, model_rows in table2["rows"].items():
        for particles, row in model_rows.items():
            lines.append(
                f"| {model} | {particles} | "
                + " | ".join(value_text(row[column]) for column in columns2)
                + " |"
            )
    lines.extend(["", "# Main Finding", ""])
    if table2["complete"]:
        wins = []
        for model, model_rows in table2["rows"].items():
            for particles, row in model_rows.items():
                bfs = row["bfs"]["mean"]
                prior = max(row[name]["mean"] for name in ("bon", "fk", "das", "treeg", "svdd"))
                wins.append((model, particles, bfs - prior))
        count = sum(delta > 0 for _, _, delta in wins)
        lines.append(f"BFS beats the strongest reproduced prior in {count}/{len(wins)} model-budget settings.")
    else:
        lines.append("The full finding is pending because one or more required trials are incomplete.")
    lines.extend(["", "# Compute", ""])
    lines.append(f"Recorded TPU chip-hours: {compute['tpu_chip_hours']:.3f}.")
    lines.append(f"Recorded prompt-runs: {compute['prompt_runs']}.")
    return "\n".join(lines) + "\n"


def decorate_cell(cell: dict[str, Any]) -> None:
    config_id = cell["config_id"]
    if cell["mean_delta"] is None:
        cell["absolute_delta"] = None
        cell["relative_delta"] = None
        cell["numeric_status"] = "INCOMPLETE"
        cell["status"] = "INCOMPLETE"
    else:
        cell["absolute_delta"] = abs(cell["mean_delta"])
        cell["relative_delta"] = cell["absolute_delta"] / abs(cell["paper_mean"])
        tolerance = max(0.10, 2.0 * cell["paper_std"])
        cell["numeric_status"] = (
            "REPRODUCED_CLOSE" if cell["absolute_delta"] <= tolerance else "REPRODUCED_DIFFERENT"
        )
        cell["status"] = cell["numeric_status"] if cell["complete"] else "INCOMPLETE"

    if config_id.startswith("bon_"):
        cell["origin"] = "OFFICIAL_CODE"
        cell["reconstruction_components"] = []
    elif config_id.startswith(("inf_max_", "svdd_")):
        cell["origin"] = "PAPER_RECONSTRUCTION"
        cell["reconstruction_components"] = [
            "infinite_tempering",
            "explicit_20_40_80_schedule_without_final_resampling",
        ]
        if cell["complete"]:
            cell["status"] = "PAPER_RECONSTRUCTION"
    elif "_g0p" in config_id:
        cell["origin"] = "PAPER_RECONSTRUCTION"
        cell["reconstruction_components"] = [
            "increase_gamma_tempering",
            "explicit_20_40_80_schedule_without_final_resampling",
        ]
        if cell["complete"]:
            cell["status"] = "PAPER_RECONSTRUCTION"
    else:
        cell["origin"] = "OFFICIAL_PIPELINE_WITH_PAPER_SCHEDULE_ADAPTER"
        cell["reconstruction_components"] = [
            "explicit_20_40_80_schedule_without_final_resampling"
        ]


def decorate_tables(table1: dict[str, Any], table2: dict[str, Any]) -> None:
    for row in table1["rows"].values():
        for cell in row.values():
            decorate_cell(cell)
    for model_rows in table2["rows"].values():
        for row in model_rows.values():
            for cell in row.values():
                decorate_cell(cell)


def comparison_markdown(table: dict[str, Any]) -> str:
    lines = []
    if table["table"] == 1:
        lines.extend(
            [
                "# Table 1 comparison",
                "",
                "| N | Cell | Paper | Reproduction | Absolute delta | Relative delta | Trials | Status | Origin |",
                "|---:|---|---:|---:|---:|---:|---:|---|---|",
            ]
        )
        iterator = (
            ("SD v1.5", particles, name, cell)
            for particles, row in table["rows"].items()
            for name, cell in row.items()
        )
    else:
        lines.extend(
            [
                "# Table 2 comparison",
                "",
                "| Model | N | Method | Paper | Reproduction | Absolute delta | Relative delta | Trials | Status | Origin |",
                "|---|---:|---|---:|---:|---:|---:|---:|---|---|",
            ]
        )
        iterator = (
            (model, particles, name, cell)
            for model, model_rows in table["rows"].items()
            for particles, row in model_rows.items()
            for name, cell in row.items()
        )
    for model, particles, name, cell in iterator:
        paper = f"{cell['paper_mean']:.3f} ± {cell['paper_std']:.3f}"
        reproduced = value_text(cell)
        absolute = "—" if cell["absolute_delta"] is None else f"{cell['absolute_delta']:.3f}"
        relative = "—" if cell["relative_delta"] is None else f"{100.0 * cell['relative_delta']:.1f}%"
        prefix = f"| {particles} | {name}" if table["table"] == 1 else f"| {model} | {particles} | {name}"
        lines.append(
            f"{prefix} | {paper} | {reproduced} | {absolute} | {relative} | "
            f"{cell['completed_trials']}/4 | {cell['status']} | {cell['origin']} |"
        )
    if table["table"] == 2:
        lines.extend(["", "## BFS improvements", ""])
        lines.append("| Model | N | vs BoN | vs nearest reproduced competitor |")
        lines.append("|---|---:|---:|---:|")
        for model, model_rows in table["rows"].items():
            for particles, row in model_rows.items():
                if any(row[name]["mean"] is None for name in row):
                    lines.append(f"| {model} | {particles} | incomplete | incomplete |")
                    continue
                bfs = row["bfs"]["mean"]
                bon = row["bon"]["mean"]
                nearest = max(row[name]["mean"] for name in ("fk", "das", "treeg", "svdd", "bon"))
                lines.append(f"| {model} | {particles} | {bfs - bon:+.3f} | {bfs - nearest:+.3f} |")
    return "\n".join(lines) + "\n"


def table_cell_rows(table: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    if table["table"] == 1:
        for particles, row in table["rows"].items():
            for name, cell in row.items():
                output.append({"model": "sd15", "N": particles, "cell": name, **cell})
    else:
        for model, model_rows in table["rows"].items():
            for particles, row in model_rows.items():
                for method, cell in row.items():
                    output.append({"model": model, "N": particles, "method": method, **cell})
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    scalar_rows = []
    for row in rows:
        scalar_rows.append(
            {
                key: json.dumps(value, sort_keys=True) if isinstance(value, (list, dict)) else value
                for key, value in row.items()
            }
        )
    fields = []
    for row in scalar_rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(scalar_rows)


def hardlink_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if os.path.samefile(source, destination):
            return
        destination.unlink()
    os.link(source, destination)


def hardlink_tree(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    for path in source.rglob("*"):
        if path.is_file():
            hardlink_file(path, destination / path.relative_to(source))


def config_ids_for_table1() -> set[str]:
    ids = set()
    for particles in (4, 8):
        for method in (
            "bon",
            "fk",
            "constant_max_ssp",
            "constant_current_ssp",
            "constant_difference_ssp",
            "inf_max",
        ):
            ids.add(plain_id(method, particles))
        for gamma in (0.008, 0.024):
            ids.add(gamma_id("bfs", particles, gamma))
    return ids


def config_ids_for_table2() -> set[str]:
    ids = set()
    for particles in (4, 8):
        for method in ("bon", "fk", "treeg", "svdd"):
            ids.add(plain_id(method, particles))
        for method in ("das", "bfs"):
            for gamma in (0.008, 0.024):
                ids.add(gamma_id(method, particles, gamma))
    return ids


def normalize_record_metadata(record: dict[str, Any], manifest: dict[str, Any]) -> None:
    """Complete legacy run metadata without changing any scientific output."""
    config = record["config"]
    model_spec = manifest["models"][config["model"]]
    components = []
    if config["tempering"] == "increase":
        components.append("increase_gamma_tempering")
    if config["tempering"] == "inf":
        components.append("infinite_tempering")
    if config["use_smc"]:
        components.append("explicit_20_40_80_schedule_without_final_resampling")

    defaults = {
        "paper": {"arxiv": manifest["paper"]["arxiv"], "table_stage": record["stage"]},
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
        "temperature": float(manifest["runtime"]["base_temperature"]),
        "eta": float(manifest["runtime"]["eta"]),
        "PAPER_RECONSTRUCTION": bool(components),
        "RECONSTRUCTION_COMPONENT": components,
    }
    added = []
    for key, value in defaults.items():
        if key not in record or record[key] is None:
            record[key] = value
            added.append(key)

    backend = record.setdefault("backend", {})
    ordinal = manifest["trials"]["seeds"].index(int(record["trial_seed"]))
    backend_defaults = {
        "tpu_type": manifest["runtime"]["tpu_type"],
        "execution_dtype": f"torch.{model_spec['execution_dtype']}",
        "xla_global_ordinal": ordinal,
        "cpu_fallbacks": [
            "ImageReward exact model",
            "official NumPy/Numba offspring allocation",
        ],
        "compile_time_separately_measurable": False,
    }
    for key, value in backend_defaults.items():
        if key not in backend or backend[key] is None:
            backend[key] = value
            added.append(f"backend.{key}")

    if added:
        record["metadata_normalization"] = {
            "added_fields": added,
            "source": "pinned experiment manifest and deterministic trial-to-ordinal mapping",
            "scientific_outputs_modified": False,
        }


def export_table_runs(
    run_root: Path,
    table_root: Path,
    allowed_ids: set[str],
    manifest: dict[str, Any],
    allowed_models: set[str] | None = None,
) -> list[Path]:
    record_paths = []
    for source in sorted((run_root / "raw").glob("*/*/trial_seed*.jsonl")):
        config_id = source.parent.name
        if config_id not in allowed_ids:
            continue
        with source.open(encoding="utf-8") as handle:
            first = json.loads(next(line for line in handle if line.strip()))
        config = first["config"]
        if allowed_models is not None and config["model"] not in allowed_models:
            continue
        gamma = "gamma-none" if config["gamma"] is None else f"gamma-{config['gamma']:.3f}".replace(".", "p")
        seed = int(first["trial_seed"])
        destination = (
            table_root
            / "runs"
            / config["model"]
            / f"N{config['particles']}"
            / config["method"]
            / gamma
            / f"seed{seed}"
        )
        selected = source.parent / "selected" / f"trial_seed{seed}"
        hardlink_tree(selected, destination / "selected")
        exported_records = destination / "records.jsonl"
        temporary = exported_records.with_suffix(".jsonl.tmp")
        with source.open(encoding="utf-8") as input_handle, temporary.open("w", encoding="utf-8") as output_handle:
            for line in input_handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                normalize_record_metadata(record, manifest)
                original_path = record["selected_image_path"]
                image_name = Path(original_path).name
                record["source_selected_image_path"] = original_path
                record["selected_image_path"] = str(
                    (destination / "selected" / image_name).relative_to(table_root.parent)
                )
                output_handle.write(json.dumps(record, sort_keys=True) + "\n")
        temporary.replace(exported_records)
        record_paths.append(exported_records)
    return record_paths


def concatenate_jsonl(paths: list[Path], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("wb") as output:
        for path in sorted(paths):
            with path.open("rb") as source:
                shutil.copyfileobj(source, output)
    temporary.replace(destination)


def build_export(
    run_root: Path,
    manifest: dict[str, Any],
    groups: dict[tuple[str, str, int], dict[str, Any]],
    table1: dict[str, Any],
    table2: dict[str, Any],
) -> Path:
    t2i_root = run_root / "export" / "results" / "diffusion-classical-search" / "t2i"
    table1_root = t2i_root / "table1"
    table2_root = t2i_root / "table2"
    table1_root.mkdir(parents=True, exist_ok=True)
    table2_root.mkdir(parents=True, exist_ok=True)
    for table_root in (table1_root, table2_root):
        runs_root = table_root / "runs"
        if runs_root.exists():
            shutil.rmtree(runs_root)

    paths1 = export_table_runs(
        run_root,
        table1_root,
        config_ids_for_table1(),
        manifest,
        allowed_models={"sd15"},
    )
    paths2 = export_table_runs(run_root, table2_root, config_ids_for_table2(), manifest)
    concatenate_jsonl(paths1, table1_root / "table1_raw.jsonl")
    concatenate_jsonl(paths2, table2_root / "table2_raw.jsonl")
    write_json(
        table1_root / "table1_trials.json",
        [
            row
            for row in groups.values()
            if row["model"] == "sd15" and row["config_id"] in config_ids_for_table1()
        ],
    )
    write_json(table2_root / "table2_trials.json", [row for row in groups.values() if row["config_id"] in config_ids_for_table2()])
    write_csv(table1_root / "table1_reproduced.csv", table_cell_rows(table1))
    write_csv(table2_root / "table2_reproduced.csv", table_cell_rows(table2))
    (table1_root / "table1_comparison.md").write_text(comparison_markdown(table1), encoding="utf-8")
    (table2_root / "table2_comparison.md").write_text(comparison_markdown(table2), encoding="utf-8")

    asset = read_json(run_root / "asset_manifest.json")
    preflights = {path.stem: read_json(path) for path in sorted((run_root / "preflight").glob("*.json"))}
    statuses = {path.stem: read_json(path) for path in sorted((run_root / "status").glob("*.json"))}
    environment = {
        "host": asset["host"],
        "package_freeze": asset["package_freeze"],
        "paper_execution_dtype": manifest["runtime"]["paper_execution_dtype"],
        "tpu_execution_dtype": manifest["runtime"]["tpu_execution_dtype"],
        "cpu_fallbacks": ["ImageReward exact model", "official NumPy/Numba offspring allocation"],
        "xla_cpu_synchronization": "three intermediate score transfers plus final scoring for SMC; final scoring only for BoN",
    }
    official_roots = [value.get("official_root") for value in preflights.values() if value.get("official_root")]
    official_root = (
        Path(official_roots[0])
        if official_roots
        else Path(asset["prompt"]["path"]).parents[2]
    )
    final_commit = subprocess.check_output(
        ["git", "-C", str(official_root), "rev-parse", "HEAD"], text=True
    ).strip()
    final_status = subprocess.check_output(
        ["git", "-C", str(official_root), "status", "--porcelain"], text=True
    ).splitlines()
    if final_commit != manifest["official_source"]["commit"] or final_status:
        raise RuntimeError(
            f"Official source integrity failure: commit={final_commit}, status={final_status}"
        )
    source_manifest = {
        "repository": manifest["official_source"]["repository"],
        "commit": manifest["official_source"]["commit"],
        "immutable_required": True,
        "preflight_statuses": {name: value.get("official_status", []) for name, value in preflights.items()},
        "final_commit": final_commit,
        "final_status": final_status,
        "clean": not final_status,
    }
    prompt_manifest = dict(manifest["benchmark"])
    prompt_manifest["complete_prompt_set_required"] = True
    tpu_manifest = {
        "tpu_type": manifest["runtime"]["tpu_type"],
        "topology": manifest["runtime"]["tpu_topology"],
        "preflights": preflights,
        "worker_statuses": statuses,
        "parallelization": "one independent trial per PJRT process/global ordinal",
        "persistent_cache": "ordinal 0 writer; ordinals 1-3 read-only",
        "sdxl_cache_policy": "all ordinals read-only on disk due root capacity; in-memory compilation reused within the SDXL stage",
    }
    write_json(t2i_root / "environment.json", environment)
    write_json(t2i_root / "source_manifest.json", source_manifest)
    hardlink_file(run_root / "asset_manifest.json", t2i_root / "asset_manifest.json")
    write_json(t2i_root / "prompt_manifest.json", prompt_manifest)
    write_json(t2i_root / "tpu_manifest.json", tpu_manifest)
    write_json(t2i_root / "experiment_manifest.json", manifest)

    log_root = run_root.parents[1] / "logs"
    for path in log_root.glob(f"{run_root.name}-*.log"):
        hardlink_file(path, t2i_root / "logs" / path.name)
    hardlink_tree(run_root / "failures", t2i_root / "failures")
    hardlink_tree(run_root / "diagnostics", t2i_root / "diagnostics")
    return t2i_root


def table1_gate(table1: dict[str, Any]) -> list[str]:
    failures = []
    if not table1["complete"]:
        failures.append("one or more Table 1 cells lack 4 complete 100-prompt trials")
    for particles, row in table1["rows"].items():
        for name, cell in row.items():
            if cell["mean"] is not None and not math.isfinite(cell["mean"]):
                failures.append(f"N={particles} {name}: non-finite mean")
            if cell["mean"] is not None and not (-5.0 <= cell["mean"] <= 5.0):
                failures.append(f"N={particles} {name}: implausible ImageReward scale {cell['mean']}")
        if row["Increase"]["mean"] is not None and row["BoN"]["mean"] is not None:
            if row["Increase"]["mean"] < row["BoN"]["mean"] - 0.5:
                failures.append(f"N={particles}: BFS is catastrophically below BoN")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--gate", choices=("none", "table1"), default="none")
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    manifest = read_json(run_root / "experiment_manifest.json")
    groups, records = load_trials(run_root)
    table1 = build_table1(manifest, groups)
    table2 = build_table2(manifest, groups)
    decorate_tables(table1, table2)
    aggregate_root = run_root / "aggregate"
    aggregate_root.mkdir(parents=True, exist_ok=True)

    trial_rows = sorted(groups.values(), key=lambda x: (x["model"], x["config_id"], x["trial_seed"]))
    with (aggregate_root / "trial_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(trial_rows[0].keys()) if trial_rows else ["model"])
        writer.writeheader()
        writer.writerows(trial_rows)

    elapsed = sum(float(record["elapsed_seconds"]) for record in records)
    compute = {
        "prompt_runs": len(records),
        "sum_prompt_elapsed_seconds": elapsed,
        "tpu_chip_hours": elapsed / 3600.0,
        "note": "Each prompt-run occupied one TPU process/device; sum of prompt elapsed times is chip-time, not host wall-time.",
    }
    write_json(aggregate_root / "table1.json", table1)
    write_json(aggregate_root / "table2.json", table2)
    write_json(aggregate_root / "compute.json", compute)
    (aggregate_root / "report.md").write_text(render_markdown(table1, table2, compute), encoding="utf-8")
    export_root = build_export(run_root, manifest, groups, table1, table2)

    artifact_index = []
    for path in sorted(run_root.rglob("*")):
        if path.is_file() and ".tmp" not in path.name and path.name != "artifact_index.json":
            artifact_index.append(
                {
                    "path": str(path.relative_to(run_root)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    write_json(aggregate_root / "artifact_index.json", artifact_index)
    print(
        json.dumps(
            {
                "event": "aggregation_complete",
                "table1_complete": table1["complete"],
                "table2_complete": table2["complete"],
                "prompt_runs": len(records),
                "export_root": str(export_root),
            },
            sort_keys=True,
        )
    )
    if args.gate == "table1":
        print(comparison_markdown(table1), flush=True)
        failures = table1_gate(table1)
        write_json(aggregate_root / "table1_gate.json", {"passed": not failures, "failures": failures})
        if failures:
            raise RuntimeError("Table 1 gate failed: " + "; ".join(failures))


if __name__ == "__main__":
    main()
