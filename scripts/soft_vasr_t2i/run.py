#!/usr/bin/env python3
"""Matched SD1.5 BFS/soft-VASR study, using the unmodified Classical Search bed."""
from __future__ import annotations

import argparse
import csv
import functools
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import socket
import subprocess
import sys
import time

import numpy as np
import torch

from controllers import AuditedBFSController, SoftVASRController
import tpu_runner as baseline
from validation import check, save, sha, validate_record, validate_pair


def tensor_sha(value):
    cpu = value.detach().cpu().contiguous()
    raw = cpu.view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(str((tuple(cpu.shape),str(cpu.dtype))).encode()+raw).hexdigest()


class RewardCounter:
    def __init__(self, model, prompt, k):
        self.model, self.prompt, self.k = model, prompt, k
        self.phase = 'intermediate'
        self.intermediate = 0
        self.final = 0
        self.batch_sizes = []

    def score_batched(self, prompts, images):
        check(prompts == [self.prompt]*self.k and len(images)==self.k, 'reward prompt/population changed')
        scores = self.model.score_batched(prompts,images)
        check(len(scores)==self.k and np.isfinite(scores).all(), 'invalid ImageReward output')
        self.batch_sizes.append(len(images))
        if self.phase=='intermediate': self.intermediate += len(images)
        else: self.final += len(images)
        return scores


class ForwardCounter:
    def __init__(self, pipe, k):
        self.pipe, self.k = pipe,k
        self.unet_calls = self.sample_forwards = self.scheduler_calls = 0
        self.vae_calls = self.decoded = 0
        self.initial_hash = None
        self.rng_trace = hashlib.sha256()
        self.callback_steps, self.population_sizes = [],[]
        self.original_prepare = pipe.prepare_latents
        self.original_step = pipe.scheduler.step

    def __enter__(self):
        @functools.wraps(self.original_prepare)
        def prepare(*args,**kwargs):
            value=self.original_prepare(*args,**kwargs)
            check(self.initial_hash is None, 'multiple initial particle allocations')
            self.initial_hash=tensor_sha(value)
            return value
        @functools.wraps(self.original_step)
        def step(*args,**kwargs):
            check(kwargs.get('eta')==1.0, 'DDIM eta missing or changed')
            generator=kwargs.get('generator')
            check(isinstance(generator,torch.Generator),'DDIM generator missing')
            self.rng_trace.update(generator.get_state().numpy().tobytes())
            value=self.original_step(*args,**kwargs)
            self.rng_trace.update(generator.get_state().numpy().tobytes())
            self.scheduler_calls+=1
            return value
        def unet_hook(module,args):
            check(args[0].shape[0]==2*self.k,'unexpected classifier-free guidance batch')
            self.unet_calls+=1
            self.sample_forwards+=args[0].shape[0]
        self.original_decode=self.pipe.vae.decode
        @functools.wraps(self.original_decode)
        def decode(*args,**kwargs):
            check(args[0].shape[0]==self.k,'VAE population changed')
            self.vae_calls+=1
            self.decoded+=args[0].shape[0]
            return self.original_decode(*args,**kwargs)
        self.pipe.prepare_latents=prepare
        self.pipe.scheduler.step=step
        self.pipe.vae.decode=decode
        self.unet_handle=self.pipe.unet.register_forward_pre_hook(unet_hook)
        return self

    def callback(self,pipe,step,timestep,kwargs):
        self.callback_steps.append(int(step))
        self.population_sizes.append(int(kwargs['latents'].shape[0]))
        return baseline._xla_callback(pipe,step,timestep,kwargs)

    def __exit__(self,*unused):
        self.pipe.prepare_latents=self.original_prepare
        self.pipe.scheduler.step=self.original_step
        self.pipe.vae.decode=self.original_decode
        self.unet_handle.remove()

    def counts(self,reward):
        return {'unet_invocations':self.unet_calls,'unet_sample_forwards':self.sample_forwards,
                'nfe':self.sample_forwards//2,'scheduler_steps':self.scheduler_calls,
                'intermediate_reward_calls':reward.intermediate,'final_reward_calls':reward.final,
                'reward_batch_sizes':reward.batch_sizes,'vae_invocations':self.vae_calls,
                'vae_decoded_particles':self.decoded,'callback_steps':self.callback_steps,
                'population_sizes':self.population_sizes}


def code_hashes(official):
    here=Path(__file__).resolve().parent
    files={f.name:sha(f) for f in here.glob('*.py')}
    files['baseline_tpu_runner.py']=sha(Path(baseline.__file__))
    for name in ['fkd_pipeline_sd.py','smc_utils.py','image_reward_utils.py']:
        files['official/'+name]=sha(official/'text_to_image/fkd_diffusers'/name)
    return files


def prepare(args,protocol):
    source=args.baseline_results
    old=json.loads((source/'experiment_manifest.json').read_text())
    check(old['runtime']['resampling_steps']==protocol['resampling_steps']==[20,40,80], 'schedule mismatch')
    check(old['trials']['seeds']==protocol['seeds']==[42,43,44,45], 'seed mismatch')
    check(old['models']['sd15']['id']==protocol['model'] and old['models']['sd15']['revision']==protocol['revision'], 'model mismatch')
    check(sha(Path(baseline.__file__))==protocol['baseline_adapter_sha256'], 'baseline implementation changed')
    git=lambda *x:subprocess.check_output(['git','-C',str(args.official_root),*x],text=True).strip()
    check(git('rev-parse','HEAD')==protocol['official_commit'] and not git('status','--porcelain'), 'official checkout changed')
    gamma_evidence=args.root/'harness/provenance/table1_reproduced.csv'
    with gamma_evidence.open() as handle:
        selected={int(r['N']):r['config_id'] for r in csv.DictReader(handle) if r['cell']=='Increase' and r['model']=='sd15'}
    check(selected=={4:'bfs_g0p008_n4',8:'bfs_g0p024_n8'}, 'historical gamma selection differs')
    # Historical CSV can be CRLF or LF; retain its actual hash as provenance.
    check(sha(gamma_evidence)==protocol['bfs']['gamma_source_sha256'], 'gamma evidence changed')
    assets=json.loads((args.output/'restored_assets.json').read_text())
    check(assets['all_match_historical_hashes'],'assets not restored')
    for entry in assets['files']:
        check(sha(args.cache_root/entry['path'])==entry['sha256'], 'asset content changed')
    prompts=baseline._load_prompts(old,args.official_root)
    check(len(prompts)==100,'benchmark incomplete')
    frozen={'protocol':protocol,'protocol_sha256':sha(args.protocol),'code_hashes':code_hashes(args.official_root),
            'baseline_manifest_sha256':sha(source/'experiment_manifest.json'),
            'baseline_selected_gamma_csv_sha256':sha(gamma_evidence), 'prompts':prompts,
            'benchmark_sha256':old['benchmark']['sha256'],'ordered_prompts_sha256':hashlib.sha256(
                json.dumps(prompts,sort_keys=True,ensure_ascii=False).encode()).hexdigest(),
            'model_assets':assets,'historical_bfs_not_reused':True}
    save(args.output/'preregistration.json',frozen,immutable=True)
    print(json.dumps({'event':'FROZEN_PROTOCOL','resampling_steps':protocol['resampling_steps'],
                      'seeds':protocol['seeds'],'gamma':protocol['bfs']['gamma_by_n'],
                      'protocol_sha256':frozen['protocol_sha256']}),flush=True)
    return old,prompts,frozen


def run_one(args,protocol,frozen,pipe,reward_model,prompt,index,seed,k,method):
    import torch_xla.core.xla_model as xm
    import fkd_pipeline_sd
    destination=args.output/'raw'/f'n{k}'/method/f'seed{seed}'/f'prompt_{index:03d}.json'
    if destination.exists():
        record=json.loads(destination.read_text())
        validate_record(record,protocol)
        check(record['protocol_sha256']==frozen['protocol_sha256'] and record['code_hashes']==frozen['code_hashes'],'resume mismatch')
        check(sha(args.output/record['selected_image_path'])==record['selected_image_sha256'],'selected image changed')
        return record
    pseed=baseline._prompt_seed(seed,index)
    gamma=protocol['bfs']['gamma_by_n'][str(k)]
    task_id=baseline._task_id('bfs',k,gamma) if method=='bfs' else f'soft_vasr_n{k}'
    rseed=baseline._resampling_seed(seed,index,task_id)
    random.seed(pseed)
    np.random.seed(rseed)
    baseline._seed_numba(rseed)
    torch.manual_seed(pseed)
    xm.set_rng_state(pseed,pipe.device)
    generator=torch.Generator(device='cpu').manual_seed(pseed)
    reward=RewardCounter(reward_model,prompt,k)
    baseline._ACTIVE_REWARD=reward
    controller_holder=[]
    def factory(**kwargs):
        controller=(AuditedBFSController if method=='bfs' else SoftVASRController)(**kwargs)
        controller_holder.append(controller)
        return controller
    fkd_pipeline_sd.FKD=factory
    fkd_args={'lmbda':10.0 if method=='bfs' else 1.0,'num_particles':k,'use_smc':True,
              'adaptive_resampling':False,'resample_frequency':0,'time_steps':100,
              'resampling_t_start':-1,'resampling_t_end':-1,'guidance_reward_fn':'ImageReward',
              'potential_type':'max' if method=='bfs' else 'soft_vasr',
              'tempering_schedule':'increase' if method=='bfs' else 'none',
              'resampling':'ssp' if method=='bfs' else 'systematic','gamma':gamma if method=='bfs' else None,
              'resampling_steps':protocol['resampling_steps'],'resampling_seed':rseed}
    started=time.perf_counter()
    with ForwardCounter(pipe,k) as count, torch.no_grad():
        output=pipe([prompt]*k,height=protocol['height'],width=protocol['width'],
                    guidance_scale=protocol['guidance_scale'],num_inference_steps=100,eta=1.0,
                    num_images_per_prompt=1,generator=generator,fkd_args=fkd_args,output_type='pil',
                    callback_on_step_end=count.callback,callback_on_step_end_tensor_inputs=['latents'])
        xm.mark_step()
        xm.wait_device_ops()
        images=list(output.images)
        check(len(images)==k,'final population differs')
        reward.phase='final'
        final=[float(x) for x in reward.score_batched([prompt]*k,images)]
    elapsed=time.perf_counter()-started
    check(len(controller_holder)==1,'unexpected controller count')
    selected=int(np.argmax(final))
    image_path=destination.parent/f'prompt_{index:03d}_selected_{selected}.jpg'
    image_path.parent.mkdir(parents=True,exist_ok=True)
    # Identical to the historical artifact policy: score lossless in-memory PIL,
    # then save JPEG for storage only. Never score recompressed artifacts.
    images[selected].save(image_path,format='JPEG',quality=90)
    rgb_std=[float(np.asarray(im.convert('RGB'),dtype=np.float32).std()) for im in images]
    check(min(rgb_std)>1e-3,'degenerate/black final particle')
    effective={key:protocol[key] for key in ('height','width','guidance_scale','eta','precision','scheduler','num_inference_steps','model','revision')}
    check(pipe.scheduler.__class__.__name__==protocol['scheduler'],'scheduler class changed')
    check(all(module.dtype==torch.bfloat16 for module in (pipe.unet,pipe.vae,pipe.text_encoder)), 'module precision differs')
    effective.update({'scheduler_config':dict(pipe.scheduler.config),'scheduler_timesteps':pipe.scheduler.timesteps.detach().cpu().tolist(),
                      'backend':'TPU','host':socket.gethostname(),'device':str(pipe.device),
                      'visible_chip':os.environ.get('TPU_VISIBLE_CHIPS'),'verifier_device':'cpu',
                      'torch':torch.__version__,'torch_xla':importlib.metadata.version('torch-xla'),
                      'numpy':np.__version__,'environment_sha256':sha(args.output/'restored_environment.txt'),
                      'common_pipeline_sha256':frozen['code_hashes']['official/fkd_pipeline_sd.py'],
                      'verifier_source_sha256':frozen['code_hashes']['official/image_reward_utils.py']})
    record={'method':method,'particles':k,'trial_seed':seed,'prompt_index':index,'prompt':prompt,
            'prompt_seed':pseed,'resampling_seed':rseed,'gamma':gamma if method=='bfs' else None,
            'initial_latents_sha256':count.initial_hash,'transition_rng_trace_sha256':count.rng_trace.hexdigest(),
            'final_particle_rewards':final,'selected_max_reward':final[selected],
            'selected_particle_index':selected,'selected_image_path':str(image_path.relative_to(args.output)),
            'selected_image_sha256':sha(image_path),'final_particle_rgb_std':rgb_std,
            'final_lossless_rgb_sha256':[hashlib.sha256(np.asarray(im.convert('RGB')).tobytes()).hexdigest() for im in images],
            'resampling_events':controller_holder[0].events,'runtime_seconds':elapsed,
            'runtime_scope':'pipeline plus final ImageReward; includes initial hash synchronization; excludes loading and JPEG storage; compilation not separated',
            'counts':count.counts(reward),'effective_protocol':effective,
            'protocol_sha256':frozen['protocol_sha256'],'code_hashes':frozen['code_hashes']}
    validate_record(record,protocol)
    save(destination,record,immutable=True)
    print(json.dumps({'event':'PROMPT_COMPLETE','n':k,'method':method,'seed':seed,'prompt_index':index,
                      'seconds':elapsed,'nfe':record['counts']['nfe'],'verifier_calls':4*k}),flush=True)
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('/home/nguyenngocminh2005bn/soft-vasr-t2i-20260905'))
    parser.add_argument('--official-root',type=Path,default=Path('/home/nguyenngocminh2005bn/dcs-table12-repro-20260903/official'))
    parser.add_argument('--baseline-results',type=Path,default=Path('/home/nguyenngocminh2005bn/dcs-table12-repro-20260903/results/table12-tpu-20260903'))
    parser.add_argument('--cache-root',type=Path,default=Path('/dev/shm/soft-vasr-t2i-20260905/cache'))
    parser.add_argument('--stage',choices=['prepare','smoke','full'],default='full')
    args=parser.parse_args()
    args.output=args.root/'results'
    args.protocol=args.root/'harness/configs/soft_vasr_t2i.json'
    protocol=json.loads(args.protocol.read_text())
    def status(state,**details):
        save(args.output/'status.json',{'state':state,'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**details})
    try:
        old,prompts,frozen=prepare(args,protocol)
        test_env=dict(os.environ,OFFICIAL_ROOT=str(args.official_root))
        tests=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(Path(__file__).parent),'-p','test_*.py','-v'],
                             env=test_env,capture_output=True,text=True)
        save(args.output/'structural_tests.json',{'passed':tests.returncode==0,'output':tests.stdout+tests.stderr,
                                                 'code_hashes':frozen['code_hashes']})
        check(tests.returncode==0,'structural unit tests failed; science prohibited')
        if args.stage=='prepare':
            status('PREPARED_UNIT_TESTS_PASSED')
            return
        import fcntl
        lock=(args.output/'run.lock').open('a')
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        check(os.environ.get('TPU_VISIBLE_CHIPS')=='2','only the free assigned TPU chip 2 is authorized')
        status('INITIALIZING_TPU_CHIP_2')
        import torch_xla.core.xla_model as xm
        import torch_xla.runtime as xr
        xr.initialize_cache(str(args.cache_root/'xla'),readonly=False)
        device=xm.xla_device()
        check(device.type=='xla' and xr.addressable_runtime_device_count()==1,'unexpected TPU topology')
        torch.set_num_threads(16)
        reward=baseline._load_reward(args.official_root,args.cache_root)
        pipeline=baseline._load_pipeline('sd15',old,args.official_root,args.cache_root,device)
        def pair(k,seed,index):
            prompt=prompts[index].get('prompt',prompts[index].get('text'))
            order=['bfs','vasr'] if (index+protocol['seeds'].index(seed))%2==0 else ['vasr','bfs']
            records={}
            for method in order:
                status('RUNNING',n=k,seed=seed,prompt_index=index,method=method)
                records[method]=run_one(args,protocol,frozen,pipeline,reward,prompt,index,seed,k,method)
            evidence=validate_pair(records['bfs'],records['vasr'],protocol)
            save(args.output/'pairs'/f'n{k}'/f'seed{seed}'/f'prompt_{index:03d}.json',evidence,immutable=True)
            return evidence
        smoke=pair(4,42,0)
        save(args.output/'smoke_validation.json',smoke,immutable=True)
        if args.stage=='smoke':
            status('SMOKE_VALID')
            return
        from analyze import analyze_n, combined_report
        for k in [4,8]:
            for index in range(100):
                for seed in protocol['seeds']:
                    pair(k,seed,index)
            status('ANALYZING',n=k)
            analyze_n(args.output,protocol,k)
            # No win/significance gate: N8 follows a valid N4 automatically.
        combined_report(args.output,protocol)
        status('COMPLETE',n4_valid=True,n8_valid=True,retuning=False)
    except Exception as exc:
        status('STOPPED_FOR_REVIEW',error_type=type(exc).__name__,detail=str(exc))
        raise


if __name__=='__main__':
    main()
