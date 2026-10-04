"""Exact DDIM100/BFS suffix replay from dense-reference checkpoints on CUDA.

Snapshots are taken after scheduler.step and the reference resampling event at
zero-based sampling_idx=src. A suffix starts at src+1. START is before step 0.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from scripts.diffusion_classical_search_t2i import tpu_runner as baseline
from scripts.static_bfs_graph.adapter import StaticGraphBFSAdapter, RNG_PROTOCOL, event_seed
from scripts.static_bfs_graph.graph_builder import dense_reference
from scripts.static_bfs_graph.policy import StaticBFSPolicy
from scripts.static_bfs_graph.runner import write_atomic

STEPS = (10, 20, 30, 40, 60, 80, 90)
START = -1


def _tensor_sha(t: torch.Tensor) -> str:
    return sha256(t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()


def _seed(trial_seed: int, prompt_index: int):
    pseed = baseline._prompt_seed(trial_seed, prompt_index)
    random.seed(pseed)
    np.random.seed(pseed)
    baseline._seed_numba(pseed)
    torch.manual_seed(pseed)
    torch.cuda.manual_seed_all(pseed)
    return pseed


def _fkd_args(policy: StaticBFSPolicy, trial_seed: int, prompt_index: int):
    return {
        "lmbda": 10.0, "num_particles": 8, "use_smc": True,
        "adaptive_resampling": False, "resample_frequency": 0,
        "time_steps": 100, "resampling_t_start": -1, "resampling_t_end": -1,
        "guidance_reward_fn": "ImageReward", "potential_type": "max",
        "tempering_schedule": "constant", "resampling": "ssp", "gamma": None,
        "resampling_steps": list(policy.resampling_steps), "selection_mode": "raw_tau",
        "temperature_by_step": [[i, t] for i, t in policy.temperature_by_step],
        "graph_trial_seed": trial_seed, "graph_prompt_index": prompt_index,
    }


def _snapshot(idx: int, latents: torch.Tensor, embeddings: torch.Tensor,
              generator: torch.Generator, adapter, scheduler) -> dict:
    return {
        "sampling_index": idx,
        "latents": latents.detach().clone().cpu(),
        "prompt_embeds": embeddings.detach().clone().cpu(),
        "previous": adapter.previous.detach().clone().cpu(),
        "events": deepcopy(adapter.events),
        "generator_state": generator.get_state().clone(),
        "torch_rng_state": torch.get_rng_state().clone(),
        "cuda_rng_states": [x.clone() for x in torch.cuda.get_rng_state_all()],
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
        "scheduler_timesteps": scheduler.timesteps.detach().clone().cpu(),
        "scheduler_num_inference_steps": scheduler.num_inference_steps,
        "event_rng": "NumPy and Numba reseeded by event_seed(seed,prompt,index) before each SSP event",
    }


def save_snapshots(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    torch.save(data, temp)
    os.replace(temp, path)


@torch.no_grad()
def reference_with_snapshots(runtime, *, prompt_index: int, trial_seed: int,
                             snapshot_path: Path, record_path: Path,
                             protocol: str = "STATIC_BFS_N8_LIGHTWEIGHT_V1") -> dict:
    if snapshot_path.exists() or record_path.exists():
        if not snapshot_path.exists() or not record_path.exists():
            raise RuntimeError("partial reference/snapshot pair requires integrity recovery")
        import json
        return json.loads(record_path.read_text())
    pipe, reward = runtime.pipe, runtime.reward
    prompt = runtime.prompts[prompt_index].get("prompt", runtime.prompts[prompt_index].get("text"))
    pseed = _seed(trial_seed, prompt_index)
    generator = torch.Generator(device="cpu").manual_seed(pseed)
    initial_latents = pipe.prepare_latents(
        8, pipe.unet.config.in_channels,
        pipe.unet.config.sample_size * pipe.vae_scale_factor,
        pipe.unet.config.sample_size * pipe.vae_scale_factor,
        pipe.unet.dtype, pipe.device, generator,
    )
    initial_generator_state = generator.get_state().clone()
    initial_torch_rng_state = torch.get_rng_state().clone()
    initial_cuda_rng_states = [x.clone() for x in torch.cuda.get_rng_state_all()]
    initial_numpy_rng_state = np.random.get_state()
    initial_python_rng_state = random.getstate()
    reference = dense_reference(STEPS, 8.0, 8)
    captures: dict[int, dict] = {}
    final_latent = None
    initial_embeds = None
    snapshot_seconds = 0.0

    def callback(_, idx, timestep, kwargs):
        nonlocal final_latent, initial_embeds, snapshot_seconds
        del timestep
        if idx == 0:
            initial_embeds = kwargs["prompt_embeds"].detach().clone().cpu()
        if idx in STEPS:
            t0 = time.perf_counter()
            adapter = baseline._LAST_ADAPTER
            captures[idx] = _snapshot(idx, kwargs["latents"], kwargs["prompt_embeds"],
                                      generator, adapter, pipe.scheduler)
            snapshot_seconds += time.perf_counter() - t0
        if idx == 99:
            final_latent = kwargs["latents"].detach().clone().cpu()
        return kwargs

    start = time.perf_counter()
    output = pipe(
        [prompt] * 8, num_inference_steps=100, eta=1.0,
        num_images_per_prompt=1, generator=generator,
        latents=initial_latents.clone(), fkd_args=_fkd_args(reference, trial_seed, prompt_index),
        output_type="pil", callback_on_step_end=callback,
        callback_on_step_end_tensor_inputs=["latents", "prompt_embeds"],
    )
    torch.cuda.synchronize()
    images = list(output.images)
    final_scores = [float(x) for x in reward.score_batched([prompt] * 8, images)]
    elapsed = time.perf_counter() - start
    if initial_embeds is None or final_latent is None or set(captures) != set(STEPS):
        raise RuntimeError("incomplete reference snapshots")
    first = captures[STEPS[0]]
    captures[START] = {
        **first,
        "sampling_index": START, "latents": initial_latents.detach().clone().cpu(),
        "prompt_embeds": initial_embeds, "previous": torch.zeros(8, dtype=torch.float32),
        "events": [], "generator_state": initial_generator_state,
        "torch_rng_state": initial_torch_rng_state,
        "cuda_rng_states": initial_cuda_rng_states,
        "numpy_rng_state": initial_numpy_rng_state,
        "python_rng_state": initial_python_rng_state,
    }
    selected = int(np.argmax(np.asarray(final_scores, dtype=np.float64)))
    rgb = np.asarray(images[selected].convert("RGB"), dtype=np.uint8)
    adapter = baseline._LAST_ADAPTER
    row = {
        "backend": "CUDA_Modal", "protocol": protocol,
        "stage": "reference", "prompt_index": prompt_index,
        "trial_seed": trial_seed, "prompt_seed": pseed,
        "policy_id": reference.id, "rng_protocol": RNG_PROTOCOL,
        "initial_latent_sha256": _tensor_sha(initial_latents),
        "initial_generator_state_sha256": _tensor_sha(initial_generator_state),
        "final_latent_sha256": _tensor_sha(final_latent),
        "final_particle_scores": final_scores,
        "selected_particle_index": selected,
        "selected_final_score": final_scores[selected],
        "selected_image_rgb_sha256": sha256(rgb.tobytes()).hexdigest(),
        "selected_image_rgb_std": float(rgb.astype(np.float32).std()),
        "resampling_events": deepcopy(adapter.events),
        "reference_wall_seconds": elapsed, "snapshot_creation_seconds": snapshot_seconds,
        "diffusion_NFE": 800, "verifier_calls": 7 * 8 + 8,
    }
    if not all(math.isfinite(x) for x in final_scores) or row["selected_image_rgb_std"] <= 1:
        raise RuntimeError("invalid reference output")
    save_snapshots(snapshot_path, {
        "protocol": row["protocol"], "prompt_index": prompt_index,
        "trial_seed": trial_seed, "prompt": prompt,
        "reference_policy_id": reference.id, "snapshots": captures,
        "final_latent": final_latent,
        "final_scores": final_scores,
        "selected_image_rgb_sha256": row["selected_image_rgb_sha256"],
    })
    write_atomic(record_path, row)
    return row


@torch.no_grad()
def replay_suffix(runtime, *, checkpoint: dict, src: int,
                  policy: StaticBFSPolicy, record_events: bool = True) -> dict:
    """Run original DDIM equation and adapter semantics from a saved prefix."""
    from fkd_pipeline_sd import latent_to_decode
    pipe, reward = runtime.pipe, runtime.reward
    prompt_index, trial_seed = checkpoint["prompt_index"], checkpoint["trial_seed"]
    prompt = checkpoint["prompt"]
    snap = checkpoint["snapshots"][src]
    if snap["sampling_index"] != src or checkpoint["reference_policy_id"] != dense_reference(STEPS, 8.0, 8).id:
        raise RuntimeError("snapshot/policy mismatch")
    torch.set_rng_state(snap["torch_rng_state"])
    torch.cuda.set_rng_state_all(snap["cuda_rng_states"])
    np.random.set_state(snap["numpy_rng_state"])
    random.setstate(snap["python_rng_state"])
    generator = torch.Generator(device="cpu")
    generator.set_state(snap["generator_state"])
    pipe.scheduler.set_timesteps(100, device=pipe.device)
    if pipe.scheduler.num_inference_steps != 100 or not torch.equal(
        pipe.scheduler.timesteps.cpu(), snap["scheduler_timesteps"]
    ):
        raise RuntimeError("DDIM scheduler mismatch")
    if pipe.unet.config.time_cond_proj_dim is not None or pipe.guidance_rescale != 0 or pipe.cross_attention_kwargs is not None:
        raise RuntimeError("unsupported conditioning/guidance state")
    if not pipe.do_classifier_free_guidance or pipe.guidance_scale != 7.5:
        raise RuntimeError("guidance changed")
    latents = snap["latents"].to(pipe.device)
    prompt_embeds = snap["prompt_embeds"].to(pipe.device)
    args = _fkd_args(policy, trial_seed, prompt_index)

    def postprocess_and_apply_reward_fn(x):
        images = pipe.image_processor.postprocess(x, output_type="pil")
        rewards = [float(v) for v in reward.score_batched([prompt] * len(images), list(images))]
        return torch.tensor(rewards).to(x.device)

    adapter = StaticGraphBFSAdapter(
        latent_to_decode_fn=lambda x: latent_to_decode(model=pipe, output_type="pil", latents=x),
        reward_fn=postprocess_and_apply_reward_fn, device=pipe.device, **args,
    )
    adapter.previous = snap["previous"].to(pipe.device)
    adapter.events = deepcopy(snap["events"])
    extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator, 1.0)
    start = time.perf_counter()
    for i in range(src + 1, 100):
        t = pipe.scheduler.timesteps[i]
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
        noise_pred = pipe.unet(
            latent_model_input, t, encoder_hidden_states=prompt_embeds,
            timestep_cond=None, cross_attention_kwargs=pipe.cross_attention_kwargs,
            added_cond_kwargs=None, return_dict=False,
        )[0]
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + pipe.guidance_scale * (noise_pred_text - noise_pred_uncond)
        step_dict = pipe.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=True)
        latents = step_dict["prev_sample"]
        x0_preds = step_dict["pred_original_sample"]
        latents, _ = adapter.resample(sampling_idx=i, latents=latents, x0_preds=x0_preds)
    final_latent = latents.detach().clone().cpu()
    image_tensor = pipe.vae.decode(
        latents / pipe.vae.config.scaling_factor, return_dict=False, generator=generator,
    )[0]
    images = list(pipe.image_processor.postprocess(
        image_tensor, output_type="pil", do_denormalize=[True] * 8,
    ))
    torch.cuda.synchronize()
    final_scores = [float(x) for x in reward.score_batched([prompt] * 8, images)]
    elapsed = time.perf_counter() - start
    selected = int(np.argmax(np.asarray(final_scores, dtype=np.float64)))
    rgb = np.asarray(images[selected].convert("RGB"), dtype=np.uint8)
    if len(final_scores) != 8 or not all(math.isfinite(x) for x in final_scores):
        raise RuntimeError("invalid replay terminal reward")
    expected_steps = list(policy.resampling_steps)
    if [e["sampling_index"] for e in adapter.events] != expected_steps:
        raise RuntimeError("replayed event schedule mismatch")
    if any(e["event_seed"] != event_seed(trial_seed, prompt_index, e["sampling_index"])
           for e in adapter.events):
        raise RuntimeError("replayed event RNG mismatch")
    return {
        "backend": "CUDA_Modal", "protocol": checkpoint["protocol"],
        "stage": "edge_replay", "prompt_index": prompt_index,
        "trial_seed": trial_seed, "src": src, "policy_id": policy.id,
        "rng_protocol": RNG_PROTOCOL,
        "initial_generator_state_sha256": _tensor_sha(checkpoint["snapshots"][START]["generator_state"]),
        "final_latent_sha256": _tensor_sha(final_latent),
        "final_particle_scores": final_scores,
        "selected_particle_index": selected, "selected_final_score": final_scores[selected],
        "selected_image_rgb_sha256": sha256(rgb.tobytes()).hexdigest(),
        "selected_image_rgb_std": float(rgb.astype(np.float32).std()),
        "resampling_events": deepcopy(adapter.events) if record_events else [],
        "suffix_diffusion_steps": 100 - (src + 1),
        "edge_replay_wall_seconds": elapsed,
        "diffusion_NFE": 8 * (100 - (src + 1)),
        "verifier_calls": 8 * (len(expected_steps) - len(snap["events"])) + 8,
    }
