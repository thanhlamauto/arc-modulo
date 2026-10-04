"""Shared exact DDIM step and adapter-state restoration."""
from __future__ import annotations

from copy import deepcopy
import random

import numpy as np
import torch

from scripts.static_bfs_graph.adapter import StaticGraphBFSAdapter
from modal_n8_gpu.prefix_replay import _fkd_args
from .snapshot import ReplaySnapshot


def restore(runtime, snap: ReplaySnapshot, policy, prompt_index: int, trial_seed: int):
    from fkd_pipeline_sd import latent_to_decode

    pipe, reward = runtime.pipe, runtime.reward
    torch.set_rng_state(snap.torch_rng_state)
    torch.cuda.set_rng_state_all(snap.cuda_rng_states)
    np.random.set_state(snap.numpy_rng_state)
    random.setstate(snap.python_rng_state)
    generator = torch.Generator(device="cpu")
    generator.set_state(snap.generator_state)
    # A fresh pipeline has not run __call__, where these fixed inference
    # properties are normally initialized. They are part of the frozen setup.
    if not hasattr(pipe, "_guidance_scale"):
        pipe._guidance_scale = 7.5
    if not hasattr(pipe, "_guidance_rescale"):
        pipe._guidance_rescale = 0.0
    if not hasattr(pipe, "_cross_attention_kwargs"):
        pipe._cross_attention_kwargs = None
    pipe.scheduler.set_timesteps(100, device=pipe.device)
    if pipe.scheduler.num_inference_steps != snap.scheduler_num_inference_steps or not torch.equal(
        pipe.scheduler.timesteps.cpu(), snap.scheduler_timesteps
    ):
        raise RuntimeError("DDIM scheduler mismatch")
    if pipe.unet.config.time_cond_proj_dim is not None or pipe.guidance_rescale != 0 or pipe.cross_attention_kwargs is not None:
        raise RuntimeError("unsupported conditioning/guidance state")
    if not pipe.do_classifier_free_guidance or pipe.guidance_scale != 7.5:
        raise RuntimeError("guidance changed")
    prompt = runtime.prompts[prompt_index]["prompt"]

    def reward_fn(images):
        pil = pipe.image_processor.postprocess(images, output_type="pil")
        scores = reward.score_batched([prompt] * len(pil), list(pil))
        return torch.tensor([float(v) for v in scores], device=images.device)

    adapter = StaticGraphBFSAdapter(
        latent_to_decode_fn=lambda x: latent_to_decode(model=pipe, output_type="pil", latents=x),
        reward_fn=reward_fn, device=pipe.device,
        **_fkd_args(policy, trial_seed, prompt_index),
    )
    adapter.previous = snap.previous_scores.to(pipe.device)
    adapter.events = deepcopy(snap.events)
    return snap.latents.to(pipe.device), snap.prompt_embeds.to(pipe.device), generator, adapter


def denoise_step(pipe, latents, prompt_embeds, generator, idx):
    t = pipe.scheduler.timesteps[idx]
    x = pipe.scheduler.scale_model_input(torch.cat([latents] * 2), t)
    noise = pipe.unet(x, t, encoder_hidden_states=prompt_embeds,
                      timestep_cond=None, cross_attention_kwargs=pipe.cross_attention_kwargs,
                      added_cond_kwargs=None, return_dict=False)[0]
    unconditional, conditional = noise.chunk(2)
    noise = unconditional + pipe.guidance_scale * (conditional - unconditional)
    kwargs = pipe.prepare_extra_step_kwargs(generator, 1.0)
    result = pipe.scheduler.step(noise, t, latents, **kwargs, return_dict=True)
    return result["prev_sample"], result["pred_original_sample"]


def capture(idx, latents, embeds, generator, adapter, scheduler):
    return ReplaySnapshot(
        sampling_idx=idx, latents=latents.detach().clone().cpu(),
        prompt_embeds=embeds.detach().clone().cpu(),
        previous_scores=adapter.previous.detach().clone().cpu(),
        events=deepcopy(adapter.events), generator_state=generator.get_state().clone(),
        torch_rng_state=torch.get_rng_state().clone(),
        cuda_rng_states=[x.clone() for x in torch.cuda.get_rng_state_all()],
        numpy_rng_state=np.random.get_state(), python_rng_state=random.getstate(),
        scheduler_timesteps=scheduler.timesteps.detach().clone().cpu(),
        scheduler_num_inference_steps=scheduler.num_inference_steps,
    )
