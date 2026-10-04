from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import numpy as np

from scripts.static_bfs_graph.adapter import StaticGraphBFSAdapter, event_seed
from scripts.static_bfs_graph.graph_builder import (
    START, END, Edge, all_edges, all_paths, dense_reference,
    edge_intervention, policy_from_path,
)
from scripts.static_bfs_graph.graph_search import select_paths
from scripts.static_bfs_graph.policy import StaticBFSPolicy


class GraphTests(unittest.TestCase):
    def test_policy_and_edge_semantics(self):
        steps=(20,40,60,80);taus=(2.0,8.0,32.0)
        reference=dense_reference(steps,8.0,4)
        self.assertEqual(len(all_edges(steps,taus)),35)
        self.assertEqual(len(list(all_paths(steps,taus,2))),54)
        self.assertEqual(len(list(all_paths(steps,taus,3))),108)
        chosen=edge_intervention(Edge(20,60,32.0),reference)
        self.assertEqual(chosen.resampling_steps,(20,60,80))
        self.assertEqual(chosen.temperature_by_step,((20,8.0),(60,32.0),(80,8.0)))
        self.assertEqual(edge_intervention(Edge(START,60,8.0),reference).resampling_steps,(60,80))
        self.assertEqual(edge_intervention(Edge(40,END,None),reference).resampling_steps,(20,40))
        self.assertEqual(edge_intervention(Edge(80,END,None),reference),reference)
        self.assertEqual(policy_from_path((Edge(START,20,2.0),Edge(20,40,8.0),Edge(40,END,None)),4).temperature_map,{20:2.0,40:8.0})

    def test_search_and_negative_edges(self):
        steps=(20,40,60,80);taus=(2.0,8.0,32.0)
        values={e.key:0.01 for e in all_edges(steps,taus)}
        values[Edge(START,20,2.0).key]=-0.5
        result=select_paths(steps,taus,values,4,2)
        self.assertEqual(result['n_paths'],54)
        self.assertEqual(result['additive'][0],Edge(START,20,2.0))

    def test_rng_is_event_not_policy_keyed(self):
        self.assertEqual(event_seed(42,7,40),event_seed(42,7,40))
        self.assertNotEqual(event_seed(42,7,40),event_seed(43,7,40))
        self.assertNotEqual(event_seed(42,7,40),event_seed(42,7,80))

    def test_bootstrap_search_matches_exact_path_search(self):
        from scripts.static_bfs_graph.pilot_analysis import path_tables, select_from_values
        steps=(10,20,30,40,60,80,90);taus=(2.0,8.0,32.0)
        edges=all_edges(steps,taus)
        for random_seed in (0,1,2):
            values=np.random.default_rng(random_seed).normal(0,0.1,len(edges))
            by_key={e.key:float(values[i]) for i,e in enumerate(edges)}
            for k in (2,3,4):
                ids,indices,_,_=path_tables(steps,taus,edges,k)
                exact=select_paths(steps,taus,by_key,4,k)
                for objective in ('additive','lexicographic'):
                    selected=ids[select_from_values(values,indices,objective)]
                    self.assertEqual(selected,policy_from_path(exact[objective],4).id)

    def test_explicit_temperature_and_manual_family(self):
        import torch
        common={"potential_type":"max","lmbda":10.0,"num_particles":4,"time_steps":100,
                "reward_fn":lambda x:torch.tensor([0.1,0.2,0.3,0.4]),
                "latent_to_decode_fn":lambda x:x,"device":torch.device('cpu'),
                "resampling":"ssp","resampling_steps":[20,40,80],
                "graph_trial_seed":42,"graph_prompt_index":0}
        explicit=StaticGraphBFSAdapter(**common,selection_mode='raw_tau',
                 temperature_by_step=((20,10.0),(40,10.0),(80,10.0)),
                 tempering_schedule='constant')
        manual_constant=StaticGraphBFSAdapter(**common,selection_mode='manual',
                 tempering_schedule='constant')
        self.assertEqual([explicit._temperature(i) for i in (20,40,80)],[10.0]*3)
        self.assertEqual([manual_constant._temperature(i) for i in (20,40,80)],[10.0]*3)
        manual_increase=StaticGraphBFSAdapter(**common,selection_mode='manual',
                 tempering_schedule='increase',gamma=.008)
        self.assertAlmostEqual(manual_increase._temperature(40),10*((1+.008)**40-1))
        fkd_root=os.environ.get('OFFICIAL_ROOT')
        if fkd_root:
            sys.path.insert(0,str(Path(fkd_root)/'text_to_image/fkd_diffusers'))
            latents=torch.arange(16,dtype=torch.float32).reshape(4,1,2,2)
            x0=latents.clone()
            for i in (20,40,80):
                a,_=explicit.resample(sampling_idx=i,latents=latents,x0_preds=x0)
                b,_=manual_constant.resample(sampling_idx=i,latents=latents,x0_preds=x0)
                self.assertTrue(torch.equal(a,b))
                self.assertEqual(explicit.events[-1]['parent_indices'],manual_constant.events[-1]['parent_indices'])
                self.assertEqual(len(explicit.events[-1]['parent_indices']),4)
                self.assertEqual(explicit.events[-1]['temperature'],10.0)


if __name__=='__main__':unittest.main()
