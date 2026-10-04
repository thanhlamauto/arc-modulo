"""Freeze the remaining TRAIN prompts and four policies before GPU scoring."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

from scripts.static_bfs_graph.graph_builder import path_edges
from scripts.static_bfs_graph.policy import StaticBFSPolicy

PROTOCOL = "STATIC_BFS_N8_COMPOSITION_DIAGNOSTIC_V1"
SEEDS = (42, 43, 44, 45)
MAX_USD = 10.0


def read(path: Path):
    return json.loads(path.read_text())


def digest(path: Path):
    return sha256(path.read_bytes()).hexdigest()


def write_once(path: Path, value):
    raw = (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != raw:
            raise RuntimeError(f"frozen plan differs: {path}")
    else:
        path.write_bytes(raw)


def build(repo: Path):
    old = repo / "results/static-bfs-graph/n8-offline-compiler-v1"
    pool_path = repo / "modal_n8_gpu/train_pool.json"
    pool = read(pool_path)
    manifest = read(old / "experiment_manifest.json")
    integrity = read(old / "integrity.json")
    graph = {row["n"]: row for row in read(old / "graphs/policies_by_n.json")}
    audit_plan_path = old / "audit/audit_policy_plan.json"
    audit_plan = read(audit_plan_path)
    compiled_path = old / "compiled/compiled_policy.json"
    fit = read(old / "calibration_trajectories.json")
    audit = read(old / "audit_prompts.json")
    if (not integrity["passed"] or manifest["PSP_in_scope"] or
        manifest["validation_touched"] or manifest["test_touched"] or
        len(pool["train_indices"]) != 60 or len(fit) != 20 or len(audit) != 20 or
        graph[5]["policy_id"] != graph[10]["policy_id"] or
        digest(audit_plan_path) != integrity["frozen_audit_plan_sha256"] or
        digest(compiled_path) != integrity["compiled_policy_sha256"]):
        raise RuntimeError("source run or split invalid")
    used = {item["prompt_id"] for item in fit + audit}
    remaining = set(pool["train_indices"]) - used
    if len(remaining) != 20 or len(used) != 40 or used - set(pool["train_indices"]):
        raise RuntimeError("expected exactly 20 untouched TRAIN prompts")
    ordered = sorted(remaining, key=lambda i: sha256(
        f"{PROTOCOL}|{i}|{pool['prompts'][str(i)]}".encode()).hexdigest())
    trajectories = [{"rank": rank, "prompt_id": prompt_id,
                     "prompt_sha256": sha256(pool["prompts"][str(prompt_id)].encode()).hexdigest(),
                     "seed": SEEDS[rank % 4]} for rank, prompt_id in enumerate(ordered)]
    policies = []
    for name, item in (("P1", graph[1]), ("P5", graph[5]), ("P20", graph[20]),
                       ("TRANSFER4_TO_8", {"steps": [40, 60, 90], "taus": [8.0, 8.0, 32.0]})):
        steps, taus = tuple(item["steps"]), tuple(item["taus"])
        policy = StaticBFSPolicy(steps, tuple(zip(steps, taus)), 8)
        alias = audit_plan["policy_aliases"][name]
        if policy.id != alias:
            raise RuntimeError(f"frozen {name} policy changed")
        policies.append({"name": name, "policy_id": policy.id, "steps": list(steps),
                         "taus": list(taus),
                         "edge_keys": [edge.key for edge in path_edges(steps, taus)]})
    union = sorted({key for item in policies for key in item["edge_keys"]})
    if len(union) != 12 or not set(union) <= set(manifest["edge_keys"]):
        raise RuntimeError("targeted edge union changed")
    return {
        "protocol": PROTOCOL, "backend": "CUDA_Modal", "workspace": "thanhlamtba",
        "GPU": "L40S", "model": "SD1.5", "dtype": "bfloat16", "N": 8, "K": 3,
        "DDIM_steps": 100, "eta": 1.0, "scoring": "max", "resampling": "ssp",
        "verifier": "ImageReward-v1.0", "reference_tau": 8.0,
        "trajectories": trajectories, "policies": policies, "edge_union": union,
        "old_calibration_prompt_ids": sorted(item["prompt_id"] for item in fit),
        "old_audit_prompt_ids": sorted(item["prompt_id"] for item in audit),
        "source_train_pool_sha256": digest(pool_path),
        "source_manifest_sha256": digest(old / "experiment_manifest.json"),
        "source_audit_plan_sha256": digest(audit_plan_path),
        "source_compiled_policy_sha256": digest(compiled_path),
        "source_edge_observations_sha256": digest(old / "profiler/edge_observations.csv"),
        "max_estimated_gpu_usd": MAX_USD,
        "cost_gate": "run rank 0 then project 20 trajectories at 1.25x its measured duration",
        "purpose": "separate edge-estimation noise from additive composition error",
        "policy_selection_from_new_rewards": False,
        "PSP_in_scope": False, "VALIDATION_touched": False, "TEST_touched": False,
        "external_100_prompt_pool_touched": False,
    }


if __name__ == "__main__":
    repo = Path(__file__).resolve().parents[1]
    plan = build(repo)
    path = repo / "modal_n8_gpu/composition_diagnostic_plan.json"
    write_once(path, plan)
    print(json.dumps({"plan": str(path), "sha256": digest(path),
                      "trajectories": len(plan["trajectories"]),
                      "edges": len(plan["edge_union"]), "policies": len(plan["policies"])}))
