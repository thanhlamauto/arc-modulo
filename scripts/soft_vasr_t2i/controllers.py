"""External soft VASR (2604.06779v2, Algorithm 1/Eqs. 6,8) and BFS diagnostics.

No model code, transition, reward transform, clipping, or FKD uniform-weight
wrapper is used here. Float64 log weights are persistent between events.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'diffusion_classical_search_t2i'))
from tpu_runner import PaperBFSAdapter


def require(condition, message):
    if not condition:
        raise ValueError(message)


def vector(values, k, name):
    values = np.asarray(values, dtype=np.float64)
    require(values.shape == (k,) and np.isfinite(values).all(), f'invalid {name}')
    return values


def event_summary(rewards, probabilities, parents, lineage_before):
    k = len(rewards)
    parents = np.asarray(parents, dtype=np.int64)
    require(parents.shape == (k,) and ((parents >= 0) & (parents < k)).all(), 'invalid parents')
    counts = np.bincount(parents, minlength=k)
    require(int(counts.sum()) == k, 'population changed')
    probability = np.asarray(probabilities, dtype=np.float64)
    positive = probability[probability > 0]
    lineage_after = np.asarray(lineage_before, dtype=np.int64)[parents]
    return {
        'reward_mean': float(np.mean(rewards)), 'reward_std': float(np.std(rewards)),
        'reward_max': float(np.max(rewards)), 'unique_parents': int(np.count_nonzero(counts)),
        'distinct_original_lineages_before': int(len(set(lineage_before))),
        'distinct_original_lineages': int(len(set(lineage_after))),
        'lineage_ids_before': np.asarray(lineage_before).tolist(),
        'lineage_ids_after': lineage_after.tolist(), 'offspring_counts': counts.tolist(),
        # Cross-parent realized dispersion, NOT Monte Carlo conditional variance.
        'offspring_count_variance': float(np.var(counts)),
        'offspring_variance_definition': 'population variance across parent counts in one realized event',
        'max_offspring_count': int(counts.max()),
        'selection_entropy': float(-np.sum(positive * np.log(positive))),
        'selection_ess': float(1.0 / np.sum(probability ** 2)),
    }


def systematic_with_u(probabilities, k, u):
    """Call the unchanged official LOW-LEVEL resampler with an explicit grid U.

    Only its module-local RNG binding is temporarily replaced (not NumPy's
    global RNG). Workers are single-threaded at the controller level. This lets
    the actual official code be tested against Eq. 6 with a known U.
    """
    import smc_utils
    require(0.0 < u < 1.0 and math.isfinite(u), 'systematic U must be in (0,1)')
    probabilities = vector(probabilities, k, 'systematic probabilities')
    require((probabilities >= 0).all() and abs(probabilities.sum() - 1) < 1e-12,
            'systematic probabilities do not sum to one')
    def rand(*shape):
        require(shape == (1,), 'official systematic RNG signature changed')
        return np.asarray([u], dtype=np.float64)
    with patch.object(smc_utils, 'random', SimpleNamespace(rand=rand)):
        parents = smc_utils.systematic(W=probabilities, M=k)
    return np.asarray(parents, dtype=np.int64)


class SoftVASRState:
    """Persistent soft allocation; never VASR-MAX and never uniform reset."""
    def __init__(self, k, seed=0, lmbda=1.0, zscore=False, ordering=True):
        require(k >= 2, 'at least two particles required')
        require(lmbda == 1.0 and zscore is False and ordering is True,
                'primary protocol is frozen: lambda=1, zscore=False, ordering=True')
        self.k = int(k)
        self.log_weights = np.zeros(k, dtype=np.float64)
        self.lineages = np.arange(k, dtype=np.int64)
        self.rng = np.random.RandomState(seed)

    @property
    def weights(self):
        with np.errstate(over='raise', invalid='raise', under='ignore'):
            weights = np.exp(self.log_weights)
        require(np.isfinite(weights).all() and (weights > 0).all(), 'unrepresentable persistent weights')
        return weights

    def select(self, rewards, u=None):
        rewards = vector(rewards, self.k, 'rewards')
        old_logs = self.log_weights.copy()
        old_weights = self.weights.copy()
        old_lineages = self.lineages.copy()
        log_allocation = old_logs + rewards
        # Center first, avoiding exp(raw reward), overflow and loss from a large
        # common reward offset. No reward clipping or mass floor is applied.
        center = float(np.max(log_allocation))
        shifted = log_allocation - center
        log_denominator_centered = float(np.log(np.exp(shifted).sum()))
        log_probabilities = shifted - log_denominator_centered
        log_masses = math.log(self.k) + log_probabilities
        masses = np.exp(log_masses)
        require(np.isfinite(masses).all() and (masses > 0).all(),
                'allocation exceeds float64 dynamic range; fail closed, do not clip')
        require(abs(float(masses.sum()) - self.k) <= 1e-12 * self.k, 'mass total differs from K')
        log_offspring_weights = old_logs - log_masses
        with np.errstate(over='raise', invalid='raise', under='ignore'):
            offspring_weights = np.exp(log_offspring_weights)
        require(np.isfinite(offspring_weights).all() and (offspring_weights > 0).all(),
                'offspring weights exceed float64 dynamic range; fail closed')

        # Algebraically w_i*exp(r_i)/m_i == Z/K for EVERY i when lambda=1.
        # Evaluate the equivalent constant directly: floating cancellation must
        # not create a spurious reward-dependent tie break. Stable descending
        # sort therefore preserves original particle order, with ordering ON.
        log_ordering_scores = np.full(self.k, center + log_denominator_centered - math.log(self.k))
        permutation = np.argsort(-log_ordering_scores, kind='stable')
        if u is None:
            u = float(self.rng.random_sample())
            while u == 0.0:  # measure-zero endpoint; never shift/clip masses
                u = float(self.rng.random_sample())
        ordered_parents = systematic_with_u(masses[permutation] / self.k, self.k, float(u))
        parents = permutation[ordered_parents]
        summary = event_summary(rewards, masses / self.k, parents, old_lineages)
        self.log_weights = log_offspring_weights[parents].copy()
        self.lineages = old_lineages[parents].copy()
        normalized_old = old_weights / old_weights.sum()
        result = {
            'raw_rewards': rewards.tolist(), 'particle_weights_before': old_weights.tolist(),
            'log_particle_weights_before': old_logs.tolist(), 'masses': masses.tolist(),
            'normalized_weights': (masses / self.k).tolist(),
            'offspring_weights_by_parent': offspring_weights.tolist(),
            'log_offspring_weights_by_parent': log_offspring_weights.tolist(),
            'particle_weights_after': self.weights.tolist(),
            'log_particle_weights_after': self.log_weights.tolist(),
            'ordering_enabled': True, 'ordering_permutation': permutation.tolist(),
            'log_ordering_scores': log_ordering_scores.tolist(),
            'ordering_ties': 'exact algebraic ties; stable original-index order',
            'ordered_parent_indices': ordered_parents.tolist(), 'parent_indices': parents.tolist(),
            'systematic_u': float(u), 'lambda': 1.0, 'zscore': False,
            'persistent_weight_ess_before': float(1.0 / np.sum(normalized_old ** 2)),
            **summary,
        }
        return parents, result


class SoftVASRController:
    """Same keyword-only resample interface and supplied decode/reward as FKD."""
    def __init__(self, *, num_particles, time_steps, reward_fn, latent_to_decode_fn,
                 device, resampling_steps, resampling_seed, lmbda=1.0, **kwargs):
        require(time_steps == 100, 'denoising steps changed')
        require(tuple(resampling_steps) == (20, 40, 80), 'frozen event list changed')
        self.num_particles = int(num_particles)
        self.reward_fn, self.latent_to_decode_fn = reward_fn, latent_to_decode_fn
        self.events = []
        self.state = SoftVASRState(num_particles, seed=resampling_seed, lmbda=lmbda)
        self.resampling_steps = tuple(resampling_steps)

    def resample(self, *, sampling_idx, latents, x0_preds):
        require(len(latents) == self.num_particles, 'population changed before resampling')
        if sampling_idx not in self.resampling_steps:
            return latents, None
        require(sampling_idx == self.resampling_steps[len(self.events)], 'duplicate/out-of-order event')
        images = self.latent_to_decode_fn(x0_preds)
        rewards = self.reward_fn(images).to(dtype=torch.float32).detach().cpu().numpy()
        parents, event = self.state.select(rewards)
        indices = torch.tensor(parents, device=latents.device, dtype=torch.long)
        self.events.append({'sampling_index': int(sampling_idx), **event})
        return torch.index_select(latents, 0, indices), torch.index_select(images, 0, indices)


class AuditedBFSController(PaperBFSAdapter):
    """Instrumentation-only subclass: inherited BFS allocation/RNG unchanged."""
    def __init__(self, **kwargs):
        require(kwargs['tempering_schedule'] == 'increase' and kwargs['potential_type'] == 'max'
                and kwargs['resampling'] == 'ssp' and kwargs['lmbda'] == 10.0,
                'BFS protocol changed')
        require(tuple(kwargs['resampling_steps']) == (20, 40, 80), 'BFS event list changed')
        super().__init__(**kwargs)
        self.lineages = np.arange(self.num_particles, dtype=np.int64)

    def resample(self, *, sampling_idx, latents, x0_preds):
        before = len(self.events)
        result = super().resample(sampling_idx=sampling_idx, latents=latents, x0_preds=x0_preds)
        if len(self.events) > before:
            event = self.events[-1]
            event.update(event_summary(event['raw_rewards'], event['normalized_weights'],
                                       event['parent_indices'], self.lineages))
            event['masses'] = (self.num_particles * np.asarray(event['normalized_weights'])).tolist()
            event['particle_weights_before'] = [1.0] * self.num_particles
            event['particle_weights_after'] = [1.0] * self.num_particles
            event['weight_semantics'] = 'BFS selection probabilities, not persistent importance weights'
            event['ordering_permutation'] = list(range(self.num_particles))
            self.lineages = np.asarray(event['lineage_ids_after'], dtype=np.int64)
        return result
