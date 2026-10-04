"""Freeze and summarize TRAIN-only held-out audit for lightweight N=8 BFS."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from scripts.static_bfs_graph.graph_builder import dense_reference
from scripts.static_bfs_graph.n8_run import manual8_record
from scripts.static_bfs_graph.policy import StaticBFSPolicy
from scripts.static_bfs_graph.runner import read, static_record, write_atomic

from .lightweight_plan import SEEDS, STEPS, digest, write_once


def records_for_plan(plan: dict) -> dict[str, dict]:
    out = {}
    for item in plan["policies"]:
        if item["method"] == "MANUAL8":
            record = manual8_record()
        else:
            policy = StaticBFSPolicy(tuple(item["steps"]),
                                     tuple(zip(item["steps"], item["taus"])), 8)
            record = static_record(policy)
        if record["id"] != item["policy_id"]:
            raise RuntimeError(f"frozen policy mismatch: {item['method']}")
        out[item["method"]] = record
    return out


def freeze_audit(root: Path, measured_n: int):
    if measured_n not in (10, 20):
        raise ValueError("freeze only after graph analysis at n=10 or n=20")
    manifest = read(root / "experiment_manifest.json")
    if manifest["PSP_in_scope"] or manifest["validation_touched"] or manifest["test_touched"]:
        raise RuntimeError("audit scope changed")
    calibration = read(root / "calibration_trajectories.json")
    audit = read(root / "audit_prompts.json")
    if len(audit) != 20 or {x["prompt_id"] for x in audit} & {x["prompt_id"] for x in calibration}:
        raise RuntimeError("audit membership invalid")
    graph = read(root / "graphs/policies_by_n.json")
    included = [row for row in graph if row["n"] <= measured_n]
    if [row["n"] for row in included] != ([1, 5, 10] if measured_n == 10 else [1, 5, 10, 20]):
        raise RuntimeError("graph stages incomplete")
    if measured_n == 10:
        stop = read(root / "analysis/progressive_stop_decision.json")
        if not stop["stop_rule_fired"]:
            raise RuntimeError("cannot freeze at 10 without preregistered stop")
    unique = {}
    aliases = {}
    for row in included:
        name = f"P{row['n']}"
        aliases[name] = row["policy_id"]
        unique.setdefault(row["policy_id"], {"method": name, "policy_id": row["policy_id"],
                                               "steps": row["steps"], "taus": row["taus"]})
    manual = manual8_record()
    transfer = StaticBFSPolicy((40, 60, 90), ((40, 8.0), (60, 8.0), (90, 32.0)), 8)
    unique.setdefault(manual["id"], {"method": "MANUAL8", "policy_id": manual["id"],
                                         "steps": [20, 40, 80], "taus": None})
    unique.setdefault(transfer.id, {"method": "TRANSFER4_TO_8", "policy_id": transfer.id,
                                          "steps": [40, 60, 90], "taus": [8.0, 8.0, 32.0]})
    aliases["MANUAL8"] = manual["id"]
    aliases["TRANSFER4_TO_8"] = transfer.id
    plan = {"protocol": manifest["protocol"], "measured_max_n": measured_n,
            "audit_prompts_sha256": digest(root / "audit_prompts.json"),
            "calibration_trajectories_sha256": digest(root / "calibration_trajectories.json"),
            "graph_policies_sha256": digest(root / "graphs/policies_by_n.json"),
            "prompt_ids": [row["prompt_id"] for row in audit], "seeds": list(SEEDS),
            "policy_aliases": aliases, "policies": list(unique.values()),
            "comparison_baselines": ["MANUAL8", "TRANSFER4_TO_8"],
            "PSP_in_scope": False, "freeze_before_audit_scoring": True}
    path = root / "audit/audit_policy_plan.json"
    write_once(path, plan)
    return {"audit_policy_plan_sha256": digest(path), "unique_policy_count": len(unique),
            "aliases": aliases, "audit_prompt_count": len(audit)}


def aggregate(root: Path):
    plan_path = root / "audit/audit_policy_plan.json"
    plan = read(plan_path)
    records = records_for_plan(plan)
    audit = read(root / "audit_prompts.json")
    by_id = {}
    audit_gpu_seconds = 0.0
    for item in plan["policies"]:
        policy_id = item["policy_id"]
        values = np.empty((20, 4), dtype=float)
        for pi, prompt in enumerate(audit):
            for si, seed in enumerate(SEEDS):
                row = read(root / f"audit/raw/{policy_id}/p{prompt['prompt_id']}_s{seed}.json")
                if (row["policy_id"] != policy_id or row["prompt_index"] != prompt["prompt_id"] or
                    row["trial_seed"] != seed or row["backend"] != "CUDA_Modal" or
                    row["diffusion_NFE"] != 800 or len(row["final_particle_scores"]) != 8 or
                    not np.isfinite(row["selected_final_score"])):
                    raise RuntimeError(f"invalid audit row {policy_id} {prompt['prompt_id']} {seed}")
                values[pi, si] = row["selected_final_score"]
                audit_gpu_seconds += float(row["elapsed_seconds"])
        by_id[policy_id] = values
    policy_rows = []
    for item in plan["policies"]:
        v = by_id[item["policy_id"]]
        policy_rows.append({"method": item["method"], "policy_id": item["policy_id"],
                            "mean_ImageReward": float(v.mean()), "std_ImageReward": float(v.std(ddof=1)),
                            "prompt_count": 20, "prompt_seed_units": 80})
    _csv(root / "audit/policy_results.csv", policy_rows)
    paired = []
    rng = np.random.default_rng(20261002)
    samples = rng.integers(0, 20, size=(20000, 20))
    for alias, pid in plan["policy_aliases"].items():
        for baseline in plan["comparison_baselines"]:
            if alias == baseline:
                continue
            bpid = plan["policy_aliases"][baseline]
            difference = (by_id[pid] - by_id[bpid]).mean(axis=1)
            lo, hi = np.quantile(difference[samples].mean(axis=1), [0.025, 0.975])
            paired.append({"method": alias, "baseline": baseline,
                           "mean_paired_delta": float(difference.mean()),
                           "ci95_low": float(lo), "ci95_high": float(hi),
                           "prompt_wins": int((difference > 1e-12).sum()),
                           "prompt_ties": int((np.abs(difference) <= 1e-12).sum()),
                           "prompt_losses": int((difference < -1e-12).sum()),
                           "prompt_bootstraps": 20000})
    _csv(root / "audit/paired_comparisons.csv", paired)
    rate = 0.000542 + 8 * 0.0000131 + 24 * 0.00000222
    output = {"audit_policy_plan_sha256": digest(plan_path), "unique_policies": len(records),
              "raw_rows": len(records) * 80, "policy_results": policy_rows,
              "paired_comparisons": paired, "PSP_scored": False,
              "audit_inference_gpu_seconds_excluding_model_load": audit_gpu_seconds,
              "audit_inference_estimated_modal_usd_excluding_model_load": audit_gpu_seconds * rate,
              "audit_cost_separate_from_offline_compiler": True,
              "validation_touched": False, "test_touched": False,
              "external_100_prompt_pool_touched": False}
    write_atomic(root / "audit/summary.json", output)
    return output


def _csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
