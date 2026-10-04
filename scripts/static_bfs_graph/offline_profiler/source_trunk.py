"""One no-search denoising trunk per reference source node."""
from __future__ import annotations

import torch

from scripts.static_bfs_graph.policy import StaticBFSPolicy
from .replay import capture, denoise_step, restore
from .snapshot import ReplaySnapshot


@torch.no_grad()
def build_source_trunk(runtime, source: ReplaySnapshot, destinations: tuple[int, ...],
                       prompt_index: int, trial_seed: int):
    prefix = tuple(e["sampling_index"] for e in source.events)
    taus = tuple(float(e["temperature"]) for e in source.events)
    sparse = StaticBFSPolicy(prefix, tuple(zip(prefix, taus)), 8)
    latents, embeds, generator, adapter = restore(runtime, source, sparse, prompt_index, trial_seed)
    initial_previous = adapter.previous.detach().clone()
    states = {}
    steps = 0
    terminal = 100 in destinations
    for i in range(source.sampling_idx + 1, min(max(destinations), 99) + 1):
        latents, x0 = denoise_step(runtime.pipe, latents, embeds, generator, i)
        steps += 1
        if not torch.equal(adapter.previous, initial_previous):
            raise RuntimeError("Max buffer changed on no-search trunk")
        if i in destinations:
            states[i] = (capture(i, latents, embeds, generator, adapter, runtime.pipe.scheduler),
                         x0.detach().clone().cpu())
        if i == 99 and terminal:
            states[100] = (capture(i, latents, embeds, generator, adapter, runtime.pipe.scheduler), None)
    if len(states) != len(destinations):
        raise RuntimeError("destination states missing")
    return states, steps
