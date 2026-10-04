"""Freeze TRAIN-only lightweight calibration and audit membership before rewards."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

from scripts.static_bfs_graph.graph_builder import all_edges

PROTOCOL = "STATIC_BFS_N8_LIGHTWEIGHT_V1"
SEEDS = (42, 43, 44, 45)
SIZES = (1, 5, 10, 20)
STEPS = (10, 20, 30, 40, 60, 80, 90)
TAUS = (2.0, 8.0, 32.0)


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def write_once(path: Path, obj) -> None:
    raw = (json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != raw:
            raise RuntimeError(f"frozen file changed: {path}")
    else:
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_bytes(raw)
        tmp.replace(path)


def _key(kind: str, index: int, prompt: str) -> str:
    return sha256(f"{PROTOCOL}|{kind}|{index}|{prompt}".encode()).hexdigest()


def preflight(run_root: Path, train_pool_path: Path, protocol: str = PROTOCOL):
    pool = json.loads(train_pool_path.read_text())
    train = pool["train_indices"]
    texts = pool["prompts"]
    if len(train) != 60 or len(set(train)) != 60 or sorted(map(int, texts)) != train:
        raise RuntimeError("frozen TRAIN pool changed")
    choices = sorted(((_key("calibration", i, texts[str(i)]), i) for i in train))
    selected = choices[:20]
    calibration = [{"prompt_id": i, "prompt": texts[str(i)], "seed": SEEDS[rank % 4],
                    "selection_hash": key, "seed_assignment_rule": "calibration_hash_rank_mod_4"}
                   for rank, (key, i) in enumerate(selected)]
    excluded = {row["prompt_id"] for row in calibration}
    audit_candidates = sorted(((_key("audit", i, texts[str(i)]), i) for i in train if i not in excluded))
    audit = [{"prompt_id": i, "prompt": texts[str(i)], "selection_hash": key,
              "seeds": list(SEEDS)} for key, i in audit_candidates[:20]]
    if len(calibration) != 20 or len(audit) != 20 or excluded & {r["prompt_id"] for r in audit}:
        raise RuntimeError("calibration/audit split invalid")
    if sorted(r["seed"] for r in calibration).count(42) != 5 or any(
        sum(r["seed"] == seed for r in calibration) != 5 for seed in SEEDS
    ):
        raise RuntimeError("calibration seeds not balanced")
    edges = all_edges(STEPS, TAUS)
    if len(edges) != 92:
        raise RuntimeError("logical edge count changed")
    write_once(run_root / "calibration_trajectories.json", calibration)
    write_once(run_root / "audit_prompts.json", audit)
    write_once(run_root / "experiment_manifest.json", {
        "protocol": protocol, "branch": "ngocminh",
        "source_commit": "34f17bd984921aeba046e8a5b7936e4a7a8346c0",
        "modal_workspace": "thanhlamtba", "backend": "CUDA_Modal",
        "model": "SD1.5", "N": 8, "K": 3, "candidate_steps": list(STEPS),
        "taus": list(TAUS), "reference_tau": 8.0, "DDIM_steps": 100, "eta": 1.0,
        "dtype": "bfloat16", "scoring": "max", "resampling": "ssp",
        "verifier": "ImageReward-v1.0", "sizes": list(SIZES),
        "edge_keys": [e.key for e in edges], "logical_edge_count": 92,
        "train_pool_sha256": digest(train_pool_path),
        "calibration_trajectories_sha256": digest(run_root / "calibration_trajectories.json"),
        "audit_prompts_sha256": digest(run_root / "audit_prompts.json"),
        "old_TPU_scientific_results_mixed": False,
        "old_Modal_canary_scientific_results_mixed": False,
        "validation_touched": False, "test_touched": False,
        "external_100_prompt_pool_touched": False,
        "audit_baselines": ["MANUAL8", "TRANSFER4_TO_8"],
        "PSP_in_scope": False,
        "max_usd_default": 50.0,
    })
    return {"calibration": len(calibration), "audit_prompts": len(audit),
            "seeds_per_audit_prompt": 4, "edge_count": len(edges),
            "calibration_sha256": digest(run_root / "calibration_trajectories.json"),
            "audit_sha256": digest(run_root / "audit_prompts.json")}
