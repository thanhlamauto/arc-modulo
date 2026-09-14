import copy
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from analyze import paired_statistics
from controllers import SoftVASRState
from run import ForwardCounter, RewardCounter
from validation import validate_record, validate_pair


PROTOCOL=json.loads((Path(__file__).resolve().parents[2]/'configs/soft_vasr_t2i.json').read_text())


def fixtures():
    k=4
    events=[]
    state=SoftVASRState(k)
    for step in [20,40,80]:
        _,event=state.select([0]*k,u=.5)
        events.append({'sampling_index':step,**event})
    effective={key:PROTOCOL[key] for key in ('height','width','guidance_scale','eta','precision','scheduler','num_inference_steps','model','revision')}
    effective.update(backend='TPU',verifier_device='cpu')
    record={'particles':4,'trial_seed':42,'prompt_index':0,'prompt':'test','prompt_seed':12,
            'method':'vasr','gamma':None,'resampling_events':events,'final_particle_rewards':[0,1,2,3],
            'selected_max_reward':3,'initial_latents_sha256':'matched','transition_rng_trace_sha256':'matched',
            'protocol_sha256':'frozen','code_hashes':{},'effective_protocol':effective,
            'counts':{'unet_invocations':100,'unet_sample_forwards':800,'nfe':400,'scheduler_steps':100,
                      'intermediate_reward_calls':12,'final_reward_calls':4,'reward_batch_sizes':[4]*4,
                      'vae_invocations':4,'vae_decoded_particles':16,'callback_steps':list(range(100)),
                      'population_sizes':[4]*100}}
    bfs=copy.deepcopy(record)
    bfs.update(method='bfs',gamma=.008)
    return bfs,record


class FairnessTests(unittest.TestCase):
    def test_valid_pair(self):
        bfs,vasr=fixtures()
        self.assertTrue(validate_pair(bfs,vasr,PROTOCOL)['valid'])

    def test_counts_fail_closed(self):
        for name in ['unet_invocations','unet_sample_forwards','nfe','scheduler_steps','intermediate_reward_calls',
                     'final_reward_calls','vae_invocations','vae_decoded_particles']:
            with self.subTest(name=name):
                bfs,_=fixtures()
                bfs['counts'][name]+=1
                with self.assertRaises(ValueError): validate_record(bfs,PROTOCOL)

    def test_paired_noise_and_protocol_mismatch_fail_closed(self):
        for key in ['initial_latents_sha256','transition_rng_trace_sha256','prompt','prompt_seed','code_hashes']:
            with self.subTest(key=key):
                bfs,vasr=fixtures()
                vasr[key]='different'
                with self.assertRaises(ValueError): validate_pair(bfs,vasr,PROTOCOL)

    def test_step60_final_event_and_weight_reset_rejected(self):
        _,vasr=fixtures()
        for value in [60,99]:
            bad=copy.deepcopy(vasr)
            bad['resampling_events'][-1]['sampling_index']=value
            with self.assertRaises(ValueError): validate_record(bad,PROTOCOL)
        vasr['resampling_events'][1]['particle_weights_before'][0]=2.
        with self.assertRaises(ValueError): validate_record(vasr,PROTOCOL)

    def test_paired_bootstrap_uses_prompt_averages(self):
        b=np.zeros((5,4))
        v=np.tile([1.,-1.,1.,-1.],(5,1))
        result=paired_statistics(b,v)
        self.assertEqual(result['bootstrap_ci95'],[0.,0.])
        self.assertEqual(result['win_fraction'],0)
        shifted=paired_statistics(b,v+.25)
        self.assertEqual(shifted['bootstrap_ci95'],[.25,.25])
        self.assertEqual(shifted['mean_delta'],.25)
        self.assertEqual(shifted['bootstrap_samples'],20000)

    def test_forward_instrumentation_preserves_scheduler_signature(self):
        class Scheduler:
            def step(self,value,timestep,latents,eta=0.0,generator=None):
                torch.randn((4,),generator=generator)
                return latents
        class UNet(torch.nn.Module):
            def forward(self,x): return x
        pipe=SimpleNamespace(unet=UNet(),scheduler=Scheduler(),vae=SimpleNamespace(decode=lambda x:x),
                             prepare_latents=lambda:torch.zeros((4,4,8,8)))
        original=pipe.scheduler.step
        generator=torch.Generator().manual_seed(42)
        with ForwardCounter(pipe,4) as c, patch('tpu_runner._xla_callback',side_effect=lambda p,s,t,k:k):
            self.assertIn('eta',inspect.signature(pipe.scheduler.step).parameters)
            self.assertIn('generator',inspect.signature(pipe.scheduler.step).parameters)
            latents=pipe.prepare_latents()
            for i in range(100):
                pipe.unet(torch.zeros((8,4,8,8)))
                pipe.scheduler.step(None,i,latents,eta=1.,generator=generator)
                c.callback(pipe,i,i,{'latents':latents})
                if i in [20,40,80,99]: pipe.vae.decode(latents)
            reward=RewardCounter(SimpleNamespace(score_batched=lambda p,x:[0]*4),'test',4)
            for i in range(4):
                if i==3: reward.phase='final'
                reward.score_batched(['test']*4,[None]*4)
        self.assertEqual(pipe.scheduler.step,original)
        expected=fixtures()[0]['counts']
        self.assertEqual(c.counts(reward),expected)


if __name__=='__main__':
    unittest.main(verbosity=2)
