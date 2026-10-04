"""Host-side explicit-temperature adapter around frozen PaperBFSAdapter."""
from __future__ import annotations

import numpy as np

from scripts.diffusion_classical_search_t2i import tpu_runner as baseline

FROZEN_BASE_CLASS = baseline.PaperBFSAdapter
RNG_PROTOCOL = "GRAPH_EXPERIMENT_RNG_PROTOCOL"
_EVENT_TAG = 0x47524150  # ASCII "GRAP"; fixed independently of policy/config ID.


def event_seed(trial_seed: int, prompt_index: int, sampling_idx: int) -> int:
    sequence = np.random.SeedSequence(
        [int(trial_seed), int(prompt_index), int(sampling_idx), _EVENT_TAG]
    )
    return int(sequence.generate_state(1, dtype=np.uint32)[0] % (2**31 - 1))


class StaticGraphBFSAdapter(FROZEN_BASE_CLASS):
    """Only event temperature and RNG coupling differ from the frozen adapter.

    Particle scoring, Max history, softmax, upstream SSP allocation, and latent
    selection are inherited unchanged from PaperBFSAdapter.
    """

    def __init__(self, *, selection_mode: str = "manual",
                 temperature_by_step=(), graph_trial_seed: int | None = None,
                 graph_prompt_index: int | None = None, **kwargs):
        self.selection_mode = selection_mode
        if selection_mode not in {"manual", "raw_tau"}:
            raise ValueError(f"unsupported selection mode {selection_mode}")
        self._explicit_tau = {int(k): float(v) for k, v in temperature_by_step}
        self.graph_trial_seed = graph_trial_seed
        self.graph_prompt_index = graph_prompt_index
        super().__init__(**kwargs)
        if self.selection_mode == "raw_tau" and set(self._explicit_tau) != self.resampling_steps:
            raise ValueError("explicit temperatures must cover exactly the event schedule")
        if graph_trial_seed is None or graph_prompt_index is None:
            raise ValueError("graph experiment requires event RNG context")

    def _temperature(self, sampling_idx: int) -> float:
        if self.selection_mode == "raw_tau":
            return self._explicit_tau[sampling_idx]
        return super()._temperature(sampling_idx)

    def resample(self, *, sampling_idx, latents, x0_preds):
        if sampling_idx not in self.resampling_steps:
            return latents, None
        seed = event_seed(self.graph_trial_seed, self.graph_prompt_index, sampling_idx)
        np.random.seed(seed)
        baseline._seed_numba(seed)
        result = super().resample(sampling_idx=sampling_idx, latents=latents, x0_preds=x0_preds)
        event = self.events[-1]
        event["event_seed"] = seed
        event["rng_protocol"] = RNG_PROTOCOL
        assert event["sampling_index"] == sampling_idx
        assert len(event["parent_indices"]) == self.num_particles
        assert all(0 <= int(i) < self.num_particles for i in event["parent_indices"])
        return result


def install_adapter() -> None:
    # Frozen source file is untouched. _load_pipeline reads this module global.
    baseline.PaperBFSAdapter = StaticGraphBFSAdapter
