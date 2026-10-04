"""Share x0 decode and ImageReward across tau forks at one destination."""
from __future__ import annotations

import torch

from .replay import restore
from .snapshot import ReplaySnapshot


@torch.no_grad()
def shared_verifier(runtime, pre_event: ReplaySnapshot, x0_cpu: torch.Tensor,
                    policy, prompt_index: int, trial_seed: int):
    _, _, _, adapter = restore(runtime, pre_event, policy, prompt_index, trial_seed)
    x0 = x0_cpu.to(runtime.pipe.device)
    decoded = adapter.latent_to_decode_fn(x0)
    raw = adapter.reward_fn(decoded).to(torch.float32)
    if raw.numel() != 8:
        raise RuntimeError("shared verifier did not produce 8 rewards")
    return decoded, raw


@torch.no_grad()
def apply_cached_verifier(runtime, pre_event: ReplaySnapshot, policy,
                          decoded: torch.Tensor, raw: torch.Tensor,
                          prompt_index: int, trial_seed: int):
    latents, embeds, generator, adapter = restore(runtime, pre_event, policy, prompt_index, trial_seed)
    original_decode, original_reward = adapter.latent_to_decode_fn, adapter.reward_fn
    adapter.latent_to_decode_fn = lambda ignored: decoded
    adapter.reward_fn = lambda ignored: raw
    try:
        latents, _ = adapter.resample(sampling_idx=pre_event.sampling_idx,
                                      latents=latents, x0_preds=torch.empty(0, device=latents.device))
    finally:
        adapter.latent_to_decode_fn = original_decode
        adapter.reward_fn = original_reward
    return latents, embeds, generator, adapter
