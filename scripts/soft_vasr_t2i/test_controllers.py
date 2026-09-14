"""CPU-only structural tests. Set OFFICIAL_ROOT; never initializes TPU/models."""
import os
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

official = Path(os.environ.get('OFFICIAL_ROOT', Path(__file__).resolve().parents[3] / 'Diffusion-inference-scaling'))
sys.path.insert(0, str(official / 'text_to_image/fkd_diffusers'))
from controllers import SoftVASRState, SoftVASRController, AuditedBFSController, systematic_with_u
from tpu_runner import PaperBFSAdapter, _seed_numba


def reference(weights, rewards, u):
    """Literal scalar Algorithm 1 + Eq.6, independent of production log-space."""
    import math
    k = len(weights)
    numerators = [w * math.exp(r) for w, r in zip(weights, rewards)]
    masses = [k * a / sum(numerators) for a in numerators]
    child_weights = [w / m for w, m in zip(weights, masses)]
    # All ordering scores are exactly sum(numerators)/K. Stable ties.
    cumulative = np.cumsum(masses)
    parents = [next(i for i, mass in enumerate(cumulative) if u+j <= mass) for j in range(k)]
    return np.asarray(masses), np.asarray(parents), np.asarray(child_weights)[parents]


class VASRTests(unittest.TestCase):
    def test_masses_counts_indices(self):
        state = SoftVASRState(4)
        parents, d = state.select([-.9, .2, .8, 1.3], u=.31)
        self.assertAlmostEqual(sum(d['masses']), 4, places=12)
        self.assertEqual(sum(d['offspring_counts']), 4)
        self.assertTrue(((parents >= 0) & (parents < 4)).all())

    def test_persistent_weights_and_lineages_two_events(self):
        state = SoftVASRState(4)
        p1, d1 = state.select([-1, -.2, .5, 1], u=.4)
        np.testing.assert_allclose(state.weights, (1/np.asarray(d1['masses']))[p1])
        np.testing.assert_array_equal(state.lineages, p1)
        old = state.weights.copy()
        p2, d2 = state.select([.5, -.4, .7, .1], u=.7)
        np.testing.assert_allclose(d2['particle_weights_before'], old)
        np.testing.assert_allclose(state.weights, (old/np.asarray(d2['masses']))[p2])
        np.testing.assert_array_equal(state.lineages, p1[p2])
        self.assertFalse(np.allclose(state.weights, 1))

    def test_ordering_preserves_pairs_and_is_stable(self):
        state = SoftVASRState(4)
        state.log_weights = np.log([.2, 2, 1, 3])
        p, d = state.select([.7, -.2, 1.2, .1], u=.9)
        permutation = np.asarray(d['ordering_permutation'])
        self.assertTrue(d['ordering_enabled'])
        np.testing.assert_array_equal(permutation, np.arange(4))
        self.assertEqual(len(set(d['log_ordering_scores'])), 1)
        np.testing.assert_array_equal(p, permutation[d['ordered_parent_indices']])
        for key in ['raw_rewards','particle_weights_before','masses']:
            np.testing.assert_array_equal(np.asarray(d[key])[permutation][np.argsort(permutation)], d[key])

    def test_fixed_u_deterministic_actual_official_resampler(self):
        weights = np.asarray([.05, .15, .3, .5])
        a = systematic_with_u(weights, 4, .6)
        b = systematic_with_u(weights, 4, .6)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(a, np.searchsorted(np.cumsum(weights), (.6+np.arange(4))/4))

    def test_uniform_case_one_child_each(self):
        state = SoftVASRState(8)
        p, d = state.select([2]*8, u=.37)
        np.testing.assert_array_equal(p, np.arange(8))
        np.testing.assert_allclose(d['masses'], 1)
        np.testing.assert_allclose(state.weights, 1)

    def test_extreme_rewards_finite_without_clipping(self):
        for rewards in [[-300, -100, 100, 300], [10000, 10000.1, 9999.5, 10001]]:
            state = SoftVASRState(4)
            _, d = state.select(rewards, u=.5)
            for key in ['masses','offspring_weights_by_parent','particle_weights_after']:
                self.assertTrue(np.isfinite(d[key]).all())
            self.assertAlmostEqual(sum(d['masses']), 4, places=12)

    def test_nonfinite_and_unrepresentable_inputs_fail_closed(self):
        for rewards in [[0,0,0,float('nan')], [0,0,0,float('inf')], [-10000,0,0,10000]]:
            with self.assertRaises((ValueError, FloatingPointError)):
                SoftVASRState(4).select(rewards, u=.5)

    def test_reference_algorithm_multiple_events(self):
        state = SoftVASRState(4)
        weights = np.ones(4)
        lineages = np.arange(4)
        for rewards, u in [([-.2,.3,-.7,1],.17), ([.8,-.1,.4,.2],.8), ([.1,1.2,-.4,.6],.36)]:
            masses, ref_parents, weights = reference(weights, rewards, u)
            parents, d = state.select(rewards, u=u)
            lineages = lineages[ref_parents]
            np.testing.assert_allclose(d['masses'], masses, rtol=1e-13)
            np.testing.assert_array_equal(parents, ref_parents)
            np.testing.assert_allclose(state.weights, weights, rtol=1e-13)
            np.testing.assert_array_equal(state.lineages, lineages)

    def test_controller_schedule_and_exact_latent_mapping(self):
        calls = []
        controller = SoftVASRController(num_particles=4,time_steps=100,device=torch.device('cpu'),
            resampling_steps=[20,40,80],resampling_seed=3,
            latent_to_decode_fn=lambda x:x, reward_fn=lambda x:(calls.append(len(x)) or x[:,0]))
        latents = torch.arange(4, dtype=torch.float32)[:,None]
        for i in range(100):
            before = latents.clone()
            latents, images = controller.resample(sampling_idx=i,latents=latents,x0_preds=before)
            if i in [20,40,80]:
                p = controller.events[-1]['parent_indices']
                torch.testing.assert_close(latents,before[p])
                torch.testing.assert_close(images,before[p])
        self.assertEqual(calls,[4,4,4])
        self.assertEqual([d['sampling_index'] for d in controller.events],[20,40,80])

    def test_primary_hyperparameters_cannot_be_tuned(self):
        for kwargs in [{'lmbda':2},{'zscore':True},{'ordering':False}]:
            with self.assertRaises(ValueError): SoftVASRState(4,**kwargs)


class BFSInstrumentationTests(unittest.TestCase):
    def test_subclass_preserves_existing_bfs_exactly(self):
        def run(cls):
            _seed_numba(123)
            controller = cls(potential_type='max',lmbda=10.,num_particles=4,time_steps=100,
                reward_fn=lambda x:x[:,0],latent_to_decode_fn=lambda x:x,device=torch.device('cpu'),
                resampling='ssp',tempering_schedule='increase',gamma=.008,resampling_steps=[20,40,80])
            latents = torch.tensor([[-.3],[.2],[.5],[.8]])
            for i in [20,40,80]:
                latents,_ = controller.resample(sampling_idx=i,latents=latents,x0_preds=latents)
            return controller,latents
        base,x = run(PaperBFSAdapter)
        audited,y = run(AuditedBFSController)
        torch.testing.assert_close(x,y,rtol=0,atol=0)
        for b,a in zip(base.events,audited.events):
            for key in ['parent_indices','raw_rewards','particle_scores','normalized_weights','temperature']:
                self.assertEqual(b[key],a[key])


if __name__ == '__main__':
    unittest.main(verbosity=2)
