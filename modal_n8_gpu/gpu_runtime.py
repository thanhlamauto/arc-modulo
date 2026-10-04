"""CUDA execution of the frozen static-BFS N=8 graph intervention protocol."""
from __future__ import annotations

from hashlib import sha256
import math
from pathlib import Path
import random
import time
from uuid import uuid4

import numpy as np
import torch

from scripts.diffusion_classical_search_t2i import tpu_runner as baseline
from scripts.static_bfs_graph.adapter import RNG_PROTOCOL, event_seed, install_adapter
from scripts.static_bfs_graph.runner import _unit_key, digest, read, write_atomic


class Runtime:
    def __init__(self, root: Path, official: Path, seed: int, cfg: dict, old: dict, prompts: list):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required")
        self.root, self.seed, self.cfg, self.prompts = root, seed, cfg, prompts
        self.device = torch.device("cuda:0")
        torch.set_num_threads(8)
        cache = Path("/vol/cache")
        install_adapter()
        start = time.perf_counter()
        self.reward = baseline._load_reward(official, cache)
        self.pipe = baseline._load_pipeline("sd15", old, official, cache, self.device)
        self.load_seconds = time.perf_counter() - start
        if self.pipe.unet.dtype != torch.bfloat16:
            raise RuntimeError("BF16 dtype changed")
        write_atomic(root / "environment" / f"seed{seed}_{uuid4().hex}.json", {
            "device": str(self.device), "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__, "torch_cuda": torch.version.cuda,
            "execution_dtype": str(self.pipe.unet.dtype), "cache": str(cache),
            "pipeline_load_seconds": self.load_seconds, "rng_protocol": RNG_PROTOCOL,
            "backend": "CUDA_Modal", "baseline_source": "frozen PaperBFSAdapter",
        })

    def run(self, stage: str, policy_record: dict, prompt_index: int, reference, edge=None):
        key = _unit_key(self.cfg, reference, policy_record, prompt_index, self.seed, stage, edge)
        key_hash = digest(key)
        dest = self.root / "raw" / stage / "n8" / f"seed{self.seed}" / f"{key_hash}.json"
        if dest.exists():
            row = read(dest)
            if row.get("unit_hash") != key_hash or not math.isfinite(row["selected_final_score"]):
                raise RuntimeError(f"invalid resumed record {dest}")
            return row
        prompt = self.prompts[prompt_index].get("prompt", self.prompts[prompt_index].get("text"))
        pseed = baseline._prompt_seed(self.seed, prompt_index)
        random.seed(pseed)
        np.random.seed(pseed)
        baseline._seed_numba(pseed)
        torch.manual_seed(pseed)
        torch.cuda.manual_seed_all(pseed)
        generator = torch.Generator(device="cpu").manual_seed(pseed)
        generator_hash = sha256(generator.get_state().numpy().tobytes()).hexdigest()
        steps = policy_record["resampling_steps"]
        manual = policy_record["kind"] == "manual"
        fkd_args = {
            "lmbda": float(policy_record.get("base_temperature", 10.0)),
            "num_particles": 8, "use_smc": True, "adaptive_resampling": False,
            "resample_frequency": 0, "time_steps": 100,
            "resampling_t_start": -1, "resampling_t_end": -1,
            "guidance_reward_fn": "ImageReward", "potential_type": "max",
            "tempering_schedule": policy_record.get("tempering", "constant"),
            "resampling": "ssp", "gamma": policy_record.get("gamma"),
            "resampling_steps": list(steps), "selection_mode": "manual" if manual else "raw_tau",
            "temperature_by_step": policy_record.get("temperature_by_step", []),
            "graph_trial_seed": self.seed, "graph_prompt_index": prompt_index,
        }
        start = time.perf_counter()
        output = self.pipe(
            [prompt] * 8, num_inference_steps=100, eta=1.0,
            num_images_per_prompt=1, generator=generator, fkd_args=fkd_args,
            output_type="pil",
        )
        torch.cuda.synchronize()
        images = list(output.images)
        final_scores = [float(x) for x in self.reward.score_batched([prompt] * 8, images)]
        if len(final_scores) != 8 or not all(math.isfinite(x) for x in final_scores):
            raise RuntimeError("incomplete or nonfinite final ImageReward")
        selected = int(np.argmax(np.asarray(final_scores, dtype=np.float64)))
        rgb = np.asarray(images[selected].convert("RGB"), dtype=np.uint8)
        rgb_std = float(rgb.astype(np.float32).std())
        if rgb_std <= 1.0:
            raise RuntimeError(f"near-uniform selected image std={rgb_std}")
        adapter = baseline._LAST_ADAPTER
        if adapter is None:
            raise RuntimeError("adapter missing")
        events = list(adapter.events)
        if [e["sampling_index"] for e in events] != steps:
            raise RuntimeError("event count or order mismatch")
        for e in events:
            idx = e["sampling_index"]
            if manual and policy_record["tempering"] == "increase":
                expected = float(policy_record["base_temperature"]) * (
                    (1.0 + float(policy_record["gamma"])) ** idx - 1.0)
            elif manual and policy_record["tempering"] == "constant":
                expected = float(policy_record["base_temperature"])
            else:
                expected = dict(policy_record["temperature_by_step"])[idx]
            if not math.isclose(e["temperature"], expected, rel_tol=1e-10, abs_tol=1e-10):
                raise RuntimeError("event temperature mismatch")
            if e["event_seed"] != event_seed(self.seed, prompt_index, idx):
                raise RuntimeError("event RNG mismatch")
            if len(e["parent_indices"]) != 8 or any(not 0 <= int(x) < 8 for x in e["parent_indices"]):
                raise RuntimeError("invalid SSP parent indices")
        row = {
            "unit_hash": key_hash, "unit_key": key, "rng_protocol": RNG_PROTOCOL,
            "prompt_seed": pseed, "initial_generator_state_sha256": generator_hash,
            "policy_id": policy_record["id"], "prompt_index": prompt_index,
            "trial_seed": self.seed, "final_particle_scores": final_scores,
            "selected_particle_index": selected, "selected_final_score": final_scores[selected],
            "selected_image_rgb_std": rgb_std, "selected_image_rgb_sha256": sha256(rgb.tobytes()).hexdigest(),
            "resampling_events": events, "verifier_calls": len(events) * 8 + 8,
            "diffusion_NFE": 800, "elapsed_seconds": time.perf_counter() - start,
            "pipeline_load_seconds": self.load_seconds, "execution_dtype": str(self.pipe.unet.dtype),
            "backend": "CUDA_Modal",
        }
        write_atomic(dest, row)
        print({"stage": stage, "seed": self.seed, "prompt": prompt_index,
               "policy": policy_record["id"], "score": row["selected_final_score"],
               "seconds": row["elapsed_seconds"]}, flush=True)
        return row
