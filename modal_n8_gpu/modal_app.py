"""Modal CUDA continuation of the TRAIN-only N8 sample-efficiency graph.

Launch with MODAL_PROFILE=thanhlamtba. The canary is mandatory before rollout.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import random

import modal

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = Path(os.environ.get(
    "STATIC_BFS_OFFICIAL_ROOT",
    str(ROOT.parent.parent / "upstream_diffusion_inference_scaling"),
))
REMOTE = Path("/opt/repro")
RUN_ID = "n8-graph-sample-efficiency-gpu-v1"
RUN_ROOT = Path("/result") / RUN_ID

app = modal.App("static-bfs-n8-sample-efficiency-gpu")
cache = modal.Volume.from_name("static-bfs-n8-gpu-cache", create_if_missing=True)
result = modal.Volume.from_name("static-bfs-n8-gpu-results", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .uv_pip_install(
        "torch==2.4.0", "torchvision==0.19.0", "numpy==1.26.3",
        "scipy==1.13.1", "numba==0.60.0", "transformers==4.38.2",
        "tokenizers==0.15.2", "accelerate==1.2.1", "huggingface-hub==0.27.1",
        "safetensors==0.5.2", "sentencepiece==0.2.0", "fairscale==0.4.13",
        "timm==1.0.12", "pillow==11.1.0", "protobuf==3.20.3",
        "ftfy==6.3.1", "einops==0.8.0", "tqdm==4.66.4",
    )
    .uv_pip_install("setuptools==80.9.0", "wheel")
    .uv_pip_install("diffusers @ git+https://github.com/huggingface/diffusers.git@af28ae2d5ba0ef80d99fff7859ebea730e1cf3f8")
    .run_commands("/.uv/uv pip install --python $(command -v python) --no-build-isolation 'image-reward @ git+https://github.com/THUDM/ImageReward.git@2ca71bac4ed86b922fe53ddaec3109fe94d45fd3'")
    .add_local_dir(ROOT / "scripts" / "static_bfs_graph", str(REMOTE / "scripts/static_bfs_graph"), copy=True)
    .add_local_dir(ROOT / "scripts" / "diffusion_classical_search_t2i", str(REMOTE / "scripts/diffusion_classical_search_t2i"), copy=True)
    .add_local_dir(ROOT / "modal_n8_gpu", str(REMOTE / "modal_n8_gpu"), copy=True)
    .add_local_file(ROOT / "configs/static_bfs_graph_t2i.json", str(REMOTE / "configs/static_bfs_graph_t2i.json"), copy=True)
    .add_local_file(ROOT / "configs/diffusion_classical_search_t2i_table12.json", str(REMOTE / "configs/diffusion_classical_search_t2i_table12.json"), copy=True)
    .add_local_dir(OFFICIAL / "text_to_image/fkd_diffusers", "/opt/official/text_to_image/fkd_diffusers", copy=True,
                   ignore=["**/__pycache__/**"])
    .add_local_file(OFFICIAL / "text_to_image/prompt_files/benchmark_ir.json",
                    "/opt/official/text_to_image/prompt_files/benchmark_ir.json", copy=True)
    .env({"PYTHONPATH": "/opt/repro", "HF_HOME": "/vol/cache/huggingface"})
)


@app.function(image=image, volumes={"/vol": cache, "/result": result}, timeout=3600, cpu=4, memory=16384)
def prepare():
    import hashlib
    from huggingface_hub import snapshot_download
    import ImageReward as RM
    from scripts.static_bfs_graph.graph_builder import all_edges
    from scripts.static_bfs_graph.n8_plan import n8_config
    from scripts.static_bfs_graph.runner import config_and_prompts, write_once, file_sha

    cfg, old, prompts = config_and_prompts(Path("/opt"))
    keys = []
    for i, item in enumerate(prompts):
        prompt = item.get("prompt", item.get("text"))
        keys.append((hashlib.sha256(f"{cfg['split_seed']}|{i}|{prompt}".encode()).hexdigest(), i))
    train = sorted(i for _, i in sorted(keys)[:60])
    fit12 = train[:12]
    if fit12 != [0, 2, 3, 4, 5, 11, 14, 15, 16, 18, 19, 20]:
        raise RuntimeError(f"original graph-fit prompts changed: {fit12}")
    remaining = train[12:]
    random.Random(20261001).shuffle(remaining)
    order = fit12 + remaining
    grid = n8_config()["pilot"]
    edges = all_edges(tuple(grid["candidate_steps"]), tuple(grid["taus"]))
    if len(edges) != 92:
        raise RuntimeError("graph edge count changed")
    plan = {
        "protocol": "STATIC_BFS_N8_GRAPH_SAMPLE_EFFICIENCY_GPU_V1",
        "source_commit": "34f17bd984921aeba046e8a5b7936e4a7a8346c0",
        "official_commit": old["official_source"]["commit"],
        "benchmark_sha256": old["benchmark"]["sha256"],
        "backend": "CUDA_Modal", "GPU_data_only": True, "N": 8, "K": 3,
        "seeds": [42, 43, 44, 45], "fit_order_60": order,
        "original_graph_fit_12": fit12, "additional_train_48": remaining,
        "sizes": [12, 24, 36, 48, 60], "order_seed": 20261001,
        "bootstrap_seed_base": 20261022, "bootstraps_per_size": 2000,
        "candidate_steps": grid["candidate_steps"], "taus": grid["taus"],
        "reference_tau": grid["reference_tau"], "edge_keys": [e.key for e in edges],
        "DDIM_steps": 100, "eta": 1.0, "dtype": "bfloat16",
        "verifier": "ImageReward-v1.0", "scoring": "Max", "resampling": "SSP",
        "validation_test_external_pool_touched": False,
        "no_new_policy_reward_evaluation": True,
        "old_48_were_previously_used_for_policy_diagnostic": True,
    }
    write_once(RUN_ROOT / "graph_plan.json", plan)
    write_once(RUN_ROOT / "experiment_manifest.json", {
        "graph_plan_sha256": file_sha(RUN_ROOT / "graph_plan.json"),
        "config_sha256": file_sha(REMOTE / "configs/static_bfs_graph_t2i.json"),
        "baseline_config_sha256": file_sha(REMOTE / "configs/diffusion_classical_search_t2i_table12.json"),
        "backend": "CUDA_Modal", "expected_reference_records": 240,
        "expected_edge_records": 22080, "TPU_raw_reused": False,
    })
    snapshot_download(
        repo_id=old["models"]["sd15"]["id"], revision=old["models"]["sd15"]["revision"],
        cache_dir="/vol/cache/huggingface", allow_patterns=["*.json", "*.bin", "*.safetensors", "*.txt", "*.model", "*.vocab", "*.merges"],
    )
    ir_root = "/vol/cache/ImageReward"
    RM.ImageReward_download(RM.utils._MODELS["ImageReward-v1.0"], ir_root)
    RM.ImageReward_download("https://huggingface.co/THUDM/ImageReward/blob/main/med_config.json", ir_root)
    cache.commit()
    result.commit()
    print("PREPARED", json.dumps({"fit12": fit12, "train_count": len(order), "edge_count": len(edges)}), flush=True)


@app.function(image=image, gpu="L40S", volumes={"/vol": cache, "/result": result},
              timeout=3600, cpu=8, memory=24576)
def canary():
    import sys
    sys.path.insert(0, str(REMOTE))
    from modal_n8_gpu.gpu_runtime import Runtime
    from scripts.static_bfs_graph.graph_builder import all_edges, dense_reference, edge_intervention
    from scripts.static_bfs_graph.n8_plan import n8_config
    from scripts.static_bfs_graph.runner import config_and_prompts, static_record, read

    cache.reload()
    result.reload()
    cfg = n8_config()
    _, old, prompts = config_and_prompts(Path("/opt"))
    plan = read(RUN_ROOT / "graph_plan.json")
    p = plan["fit_order_60"][0]
    grid = cfg["pilot"]
    steps = tuple(grid["candidate_steps"])
    ref = dense_reference(steps, float(grid["reference_tau"]), 8)
    edge = all_edges(steps, tuple(map(float, grid["taus"])))[0]
    policy = edge_intervention(edge, ref)
    runtime = Runtime(RUN_ROOT, Path("/opt/official"), 42, cfg, old, prompts)
    reference_row = runtime.run("n8_reference", static_record(ref), p, ref)
    edge_row = runtime.run("n8_edges", static_record(policy), p, ref, edge)
    result.commit()
    return {"reference_seconds": reference_row["elapsed_seconds"],
            "edge_seconds": edge_row["elapsed_seconds"],
            "load_seconds": runtime.load_seconds,
            "reference_score": reference_row["selected_final_score"],
            "edge_score": edge_row["selected_final_score"],
            "reference_events": len(reference_row["resampling_events"]),
            "edge_events": len(edge_row["resampling_events"]),
            "gpu": runtime.device.type}


@app.local_entrypoint()
def smoke():
    prepare.remote()
    print("CANARY", json.dumps(canary.remote(), sort_keys=True))


@app.function(image=image, gpu="L40S", volumes={"/vol": cache, "/result": result},
              timeout=7200, cpu=8, memory=24576, max_containers=8)
def run_shard(stage: str, seed: int, first_edge: int = 0, last_edge: int = 0):
    """One seed, either all references or two edge indices; each unit is resumable."""
    import sys
    sys.path.insert(0, str(REMOTE))
    from modal_n8_gpu.gpu_runtime import Runtime
    from scripts.static_bfs_graph.graph_builder import all_edges, dense_reference, edge_intervention
    from scripts.static_bfs_graph.n8_plan import n8_config
    from scripts.static_bfs_graph.n8_run import dest_for
    from scripts.static_bfs_graph.runner import config_and_prompts, file_sha, read, static_record, write_atomic

    if seed not in (42, 43, 44, 45) or stage not in ("reference", "edges"):
        raise ValueError("invalid shard")
    if stage == "edges" and not (0 <= first_edge < last_edge <= 92 and last_edge - first_edge <= 2):
        raise ValueError("edge shard must cover 1–2 indices")
    cache.reload()
    result.reload()
    cfg = n8_config()
    _, old, prompts = config_and_prompts(Path("/opt"))
    plan = read(RUN_ROOT / "graph_plan.json")
    manifest = read(RUN_ROOT / "experiment_manifest.json")
    if file_sha(RUN_ROOT / "graph_plan.json") != manifest["graph_plan_sha256"]:
        raise RuntimeError("frozen plan SHA changed")
    if plan["backend"] != "CUDA_Modal" or len(plan["fit_order_60"]) != 60:
        raise RuntimeError("wrong GPU plan")
    grid = cfg["pilot"]
    steps = tuple(grid["candidate_steps"])
    edges = all_edges(steps, tuple(map(float, grid["taus"])))
    if [e.key for e in edges] != plan["edge_keys"]:
        raise RuntimeError("edge catalogue changed")
    ref = dense_reference(steps, float(grid["reference_tau"]), 8)
    refrec = static_record(ref)
    indices = range(first_edge, last_edge) if stage == "edges" else ()
    work = []
    if stage == "reference":
        for p in plan["fit_order_60"]:
            dest, _, unit_hash = dest_for(RUN_ROOT, "n8_reference", cfg, ref, refrec, p, seed)
            if not dest.exists():
                work.append((p, None, refrec))
            else:
                row = read(dest)
                if row.get("unit_hash") != unit_hash or row.get("backend") != "CUDA_Modal":
                    raise RuntimeError(f"invalid resumed reference {dest}")
    else:
        for ei in indices:
            e = edges[ei]
            pol = edge_intervention(e, ref)
            rec = static_record(pol)
            for p in plan["fit_order_60"]:
                dest, _, unit_hash = dest_for(RUN_ROOT, "n8_edges", cfg, ref, rec, p, seed, e)
                if not dest.exists():
                    work.append((p, e, rec))
                else:
                    row = read(dest)
                    if row.get("unit_hash") != unit_hash or row.get("backend") != "CUDA_Modal":
                        raise RuntimeError(f"invalid resumed edge {dest}")
    if not work:
        return {"stage": stage, "seed": seed, "first_edge": first_edge,
                "last_edge": last_edge, "new": 0, "copied": 0}
    runtime = Runtime(RUN_ROOT, Path("/opt/official"), seed, cfg, old, prompts)
    new = copied = 0
    for p, e, rec in work:
        try:
            if e is None:
                runtime.run("n8_reference", rec, p, ref)
                new += 1
            elif edge_intervention(e, ref) == ref:
                src, _, _ = dest_for(RUN_ROOT, "n8_reference", cfg, ref, refrec, p, seed)
                if not src.exists():
                    raise RuntimeError(f"reference missing for {p}|{seed}")
                row = read(src)
                dst, key, unit_hash = dest_for(RUN_ROOT, "n8_edges", cfg, ref, rec, p, seed, e)
                if row["policy_id"] != rec["id"] or row["diffusion_NFE"] != 800:
                    raise RuntimeError("invalid exact-reference copy")
                write_atomic(dst, {**row, "unit_key": key, "unit_hash": unit_hash,
                                   "edge": e.record(), "reused_identical_reference": True})
                copied += 1
            else:
                runtime.run("n8_edges", rec, p, ref, e)
                new += 1
            if (new + copied) % 5 == 0:
                result.commit()
        except BaseException as exc:
            write_atomic(RUN_ROOT / "failures" / stage / f"seed{seed}_p{p}_edge{e.key if e else 'ref'}.json",
                         {"seed": seed, "prompt_index": p, "edge": e.key if e else None,
                          "error": repr(exc), "backend": "CUDA_Modal"})
            result.commit()
            raise
    result.commit()
    print("SHARD_DONE", stage, seed, first_edge, last_edge, new, copied, flush=True)
    return {"stage": stage, "seed": seed, "first_edge": first_edge,
            "last_edge": last_edge, "new": new, "copied": copied}


@app.local_entrypoint()
def launch_references():
    raise RuntimeError("CANCELLED by n8-lightweight-v1 protocol: do not launch 60x4x92 profiling")


@app.local_entrypoint()
def launch_edges():
    raise RuntimeError("CANCELLED by n8-lightweight-v1 protocol: do not launch 60x4x92 profiling")


@app.function(image=image, timeout=600, cpu=2, memory=4096)
def self_test():
    import os
    import subprocess
    env = dict(os.environ, OFFICIAL_ROOT="/opt/official", PYTHONPATH="/opt/repro")
    proc = subprocess.run(
        ["python", "-m", "unittest", "scripts.static_bfs_graph.test_graph", "-v"],
        cwd="/opt/repro", env=env, text=True, capture_output=True,
    )
    print(proc.stdout, proc.stderr, flush=True)
    if proc.returncode:
        raise RuntimeError("graph structural tests failed")
    return {"passed": True, "test_count": proc.stdout.count(" ... ok") + proc.stderr.count(" ... ok")}


@app.local_entrypoint()
def check_structure():
    print("STRUCTURAL", json.dumps(self_test.remote(), sort_keys=True))


@app.function(image=image, volumes={"/result": result}, timeout=1800, cpu=4, memory=8192)
def analyze_results():
    import sys
    sys.path.insert(0, str(REMOTE))
    from modal_n8_gpu.analyze_gpu import analyze
    result.reload()
    summary = analyze(RUN_ROOT)
    result.commit()
    return {"status": summary["status"], "reference_records": summary["reference_records"],
            "edge_records": summary["edge_records"], "rows": summary["rows"]}


@app.local_entrypoint()
def finish_analysis():
    print("ANALYSIS", json.dumps(analyze_results.remote(), sort_keys=True))


@app.function(image=modal.Image.debian_slim(python_version="3.10"),
              volumes={"/result": result}, timeout=120, cpu=1, memory=512)
def status():
    result.reload()
    counts = {}
    for stage in ("n8_reference", "n8_edges"):
        counts[stage] = {
            str(seed): len(list((RUN_ROOT / "raw" / stage / "n8" / f"seed{seed}").glob("*.json")))
            for seed in (42, 43, 44, 45)
        }
    failures = list((RUN_ROOT / "failures").rglob("*.json")) if (RUN_ROOT / "failures").exists() else []
    return {"counts": counts, "failures": len(failures),
            "integrity_ready": (RUN_ROOT / "integrity.json").exists()}


@app.local_entrypoint()
def check_status():
    print("STATUS", json.dumps(status.remote(), sort_keys=True))
