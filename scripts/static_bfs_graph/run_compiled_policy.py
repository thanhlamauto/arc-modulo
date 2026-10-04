"""Run a frozen static BFS policy; no graph, profiler, or compiler is loaded."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch

from scripts.diffusion_classical_search_t2i import tpu_runner as baseline
from scripts.static_bfs_graph.adapter import install_adapter


def run(policy_path: Path, prompt: str, output: Path, seed: int,
        prompt_index: int, cache_root: Path, official_root: Path):
    policy = json.loads(policy_path.read_text())
    if (policy.get("schema_version") != 1 or policy.get("model") != "sd15" or
        policy.get("particle_budget_used_for_compilation") != 8 or
        policy.get("K") != 3 or policy.get("scoring") != "max" or
        policy.get("resampling") != "ssp" or len(policy.get("events", [])) != 3):
        raise ValueError("unsupported compiled policy schema")
    steps = [int(x["sampling_idx"]) for x in policy["events"]]
    taus = [[int(x["sampling_idx"]), float(x["tau"])] for x in policy["events"]]
    if steps != sorted(set(steps)) or any(not 0 <= x < 100 for x in steps):
        raise ValueError("invalid compiled schedule")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for SD1.5 deployment")
    install_adapter()
    config = json.loads((Path(__file__).resolve().parents[2] /
                         "configs/diffusion_classical_search_t2i_table12.json").read_text())
    reward = baseline._load_reward(official_root, cache_root)
    pipe = baseline._load_pipeline("sd15", config, official_root, cache_root,
                                   torch.device("cuda:0"))
    prompt_seed = baseline._prompt_seed(seed, prompt_index)
    random.seed(prompt_seed)
    np.random.seed(prompt_seed)
    baseline._seed_numba(prompt_seed)
    torch.manual_seed(prompt_seed)
    torch.cuda.manual_seed_all(prompt_seed)
    generator = torch.Generator(device="cpu").manual_seed(prompt_seed)
    out = pipe([prompt] * 8, num_inference_steps=100, eta=1.0,
               num_images_per_prompt=1, generator=generator, output_type="pil",
               fkd_args={"lmbda": 10.0, "num_particles": 8, "use_smc": True,
                         "adaptive_resampling": False, "resample_frequency": 0,
                         "time_steps": 100, "resampling_t_start": -1,
                         "resampling_t_end": -1, "guidance_reward_fn": "ImageReward",
                         "potential_type": "max", "tempering_schedule": "constant",
                         "resampling": "ssp", "gamma": None,
                         "resampling_steps": steps, "selection_mode": "raw_tau",
                         "temperature_by_step": taus,
                         "graph_trial_seed": seed, "graph_prompt_index": prompt_index})
    images = list(out.images)
    scores = [float(x) for x in reward.score_batched([prompt] * 8, images)]
    best = int(np.argmax(np.asarray(scores, dtype=np.float64)))
    output.parent.mkdir(parents=True, exist_ok=True)
    images[best].save(output)
    return {"image": str(output), "selected_particle": best,
            "selected_ImageReward": scores[best], "policy_sha256": policy["source_edge_table_sha256"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", type=Path, default=Path("compiled_bfs_output.png"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt-index", type=int, default=0)
    parser.add_argument("--cache-root", type=Path, default=Path("/vol/cache"))
    parser.add_argument("--official-root", type=Path, default=Path("/opt/official"))
    args = parser.parse_args()
    print(json.dumps(run(args.policy, args.prompt, args.output, args.seed,
                         args.prompt_index, args.cache_root, args.official_root)))


if __name__ == "__main__":
    main()
