"""Explicit replay state for a post-event or pre-event DDIM position."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class ReplaySnapshot:
    sampling_idx: int
    latents: torch.Tensor
    prompt_embeds: torch.Tensor
    previous_scores: torch.Tensor
    events: list[dict[str, Any]]
    generator_state: torch.Tensor
    torch_rng_state: torch.Tensor
    cuda_rng_states: list[torch.Tensor]
    numpy_rng_state: Any
    python_rng_state: Any
    scheduler_timesteps: torch.Tensor
    scheduler_num_inference_steps: int

    @classmethod
    def from_reference(cls, data: dict) -> "ReplaySnapshot":
        return cls(
            sampling_idx=data["sampling_index"], latents=data["latents"],
            prompt_embeds=data["prompt_embeds"], previous_scores=data["previous"],
            events=data["events"], generator_state=data["generator_state"],
            torch_rng_state=data["torch_rng_state"],
            cuda_rng_states=data["cuda_rng_states"],
            numpy_rng_state=data["numpy_rng_state"],
            python_rng_state=data["python_rng_state"],
            scheduler_timesteps=data["scheduler_timesteps"],
            scheduler_num_inference_steps=data["scheduler_num_inference_steps"],
        )
