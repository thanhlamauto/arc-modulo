"""TPU execution for the frozen tiny static-BFS graph experiment.

Each invocation owns one TPU chip and one trial seed. Policy/prompt records are
atomic, and the existing SD1.5 pipeline plus CPU ImageReward load only once.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback

import numpy as np
import torch

from scripts.diffusion_classical_search_t2i import tpu_runner as baseline
from scripts.static_bfs_graph.adapter import RNG_PROTOCOL, install_adapter, event_seed
from scripts.static_bfs_graph.graph_builder import all_edges, dense_reference, edge_intervention
from scripts.static_bfs_graph.policy import StaticBFSPolicy

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs/static_bfs_graph_t2i.json"
BASELINE_CONFIG = REPO / "configs/diffusion_classical_search_t2i_table12.json"


def digest(value) -> str:
    raw=json.dumps(value,sort_keys=True,separators=(',',':')).encode()
    return sha256(raw).hexdigest()


def file_sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def write_once(path: Path, obj) -> None:
    raw=(json.dumps(obj,indent=2,sort_keys=True)+'\n').encode()
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if path.read_bytes()!=raw: raise RuntimeError(f'frozen file differs: {path}')
    else:
        tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
        tmp.write_bytes(raw); os.replace(tmp,path)


def write_atomic(path: Path, obj) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')
    os.replace(tmp,path)


def read(path: Path):
    return json.loads(path.read_text())


def config_and_prompts(dcs_root: Path):
    cfg=read(CONFIG);old=read(BASELINE_CONFIG)
    prompts=baseline._load_prompts(old,dcs_root/'official')
    assert cfg['steps']==old['runtime']['num_inference_steps']==100
    assert cfg['eta']==old['runtime']['eta']==1.0
    assert cfg['execution_dtype']==old['models']['sd15']['execution_dtype']=='bfloat16'
    assert cfg['scoring']=='max' and cfg['resampling']=='ssp'
    assert len(prompts)==100
    return cfg,old,prompts


def prepare(run_root: Path,dcs_root: Path):
    cfg,old,prompts=config_and_prompts(dcs_root)
    import subprocess
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip()
    status=subprocess.check_output(['git','status','--short'],cwd=REPO,text=True)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    if branch!='ngocminh':raise RuntimeError(f'requires branch ngocminh, got {branch!r}')
    protected=['scripts/diffusion_classical_search_t2i/tpu_runner.py',
               'configs/diffusion_classical_search_t2i_table12.json','T2I_RESULTS.md']
    if any(line[3:].startswith(tuple(protected)) for line in status.splitlines()):
        raise RuntimeError('a protected baseline file is modified')
    keys=[]
    for index,p in enumerate(prompts):
        prompt=p.get('prompt',p.get('text'))
        keys.append((sha256(f"{cfg['split_seed']}|{index}|{prompt}".encode()).hexdigest(),index))
    ordered=[i for _,i in sorted(keys)]
    train,val,test=ordered[:60],ordered[60:80],ordered[80:]
    assert len(set(train+val+test))==100
    split={'split_seed':cfg['split_seed'],'method':'SHA256(prompt_index + prompt + split_seed)',
           'train_indices':sorted(train),'validation_indices':sorted(val),'test_indices':sorted(test),
           'tiny_train_indices':sorted(train)[:cfg['tiny']['train_prompts']]}
    source_paths=[CONFIG,BASELINE_CONFIG,REPO/'scripts/diffusion_classical_search_t2i/tpu_runner.py']
    manifest={'protocol_version':cfg['protocol_version'],'rng_protocol':RNG_PROTOCOL,
              'repository':'NguyenNgocMinh30012005/arc-modulo','branch':branch,'git_commit':commit,
              'existing_reproduction_untouched':True,'baseline_source_sha256':file_sha(source_paths[-1]),
              'config_sha256':file_sha(CONFIG),'baseline_config_sha256':file_sha(BASELINE_CONFIG),
              'benchmark_sha256':old['benchmark']['sha256'],'official_source_commit':old['official_source']['commit'],
              'model':cfg['model'],'execution_dtype':cfg['execution_dtype'],'eta':cfg['eta'],
              'diffusion_steps':cfg['steps'],'tiny':cfg['tiny'],'pilot_locked_but_not_launched':cfg['pilot'],
              'graph_semantics':'START=-1 -> increasing zero-based sampling_idx -> END=100',
              'edge_target':'dense-reference selected terminal IR minus intervention selected terminal IR',
              'manual_n4':cfg['manual_n4']}
    write_once(run_root/'experiment_manifest.json',manifest)
    write_once(run_root/'prompt_split.json',split)
    return cfg,old,prompts,split


def manual_policy_record(cfg):
    m=cfg['manual_n4']
    return {'kind':'manual','id':'manual_bfs_g0p008_n4','resampling_steps':m['resampling_steps'],
            'base_temperature':m['base_temperature'],'tempering':m['tempering'],'gamma':m['gamma'],
            'particles':cfg['particles'],'selection_mode':'manual'}


def static_record(policy:StaticBFSPolicy):
    return {'kind':'static','id':policy.id,**policy.record()}


def _unit_key(cfg, reference, policy_record, prompt_index, trial_seed, stage, edge=None):
    return {'N':cfg['particles'],'reference_schedule':reference.record(),
            'policy':policy_record,'edge':edge.record() if edge else None,
            'prompt_index':prompt_index,'trial_seed':trial_seed,
            'protocol_version':cfg['protocol_version'],'stage':stage}


class Runtime:
    def __init__(self,run_root:Path,dcs_root:Path,seed:int,chip:int,cfg,old,prompts):
        import torch_xla.core.xla_model as xm
        import torch_xla.runtime as xr
        if os.environ.get('TPU_VISIBLE_CHIPS')!=str(chip):
            raise RuntimeError(f'TPU_VISIBLE_CHIPS must be {chip}')
        self.run_root=run_root;self.dcs_root=dcs_root;self.seed=seed;self.chip=chip
        self.cfg=cfg;self.old=old;self.prompts=prompts
        cache=Path(os.environ.get('STATIC_BFS_CACHE_ROOT',f'/dev/shm/static-bfs-{run_root.name}/chip{chip}/cache'))
        cache.mkdir(parents=True,exist_ok=True)
        for name in ('huggingface','ImageReward'):
            link=cache/name;target=dcs_root/'cache'/name
            if not link.exists():link.symlink_to(target,target_is_directory=True)
            if link.resolve()!=target.resolve():raise RuntimeError(f'wrong cache link {link}')
        (cache/'xla').mkdir(exist_ok=True)
        xr.initialize_cache(str(cache/'xla'),readonly=False)
        self.cache=cache
        self.device=xm.xla_device()
        if self.device.type!='xla' or xr.addressable_runtime_device_count()!=1:
            raise RuntimeError('expected one visible TPU/XLA device')
        torch.set_num_threads(16)
        install_adapter()
        start=time.perf_counter()
        self.reward=baseline._load_reward(dcs_root/'official',cache)
        self.pipe=baseline._load_pipeline('sd15',old,dcs_root/'official',cache,self.device)
        self.load_seconds=time.perf_counter()-start
        if str(self.pipe.unet.dtype)!='torch.bfloat16':raise RuntimeError('execution dtype changed')
        write_atomic(run_root/'environment'/f'chip{chip}_seed{seed}.json',
                     {'tpu_device':str(self.device),'addressable_devices':xr.addressable_runtime_device_count(),
                      'torch':torch.__version__,'torch_xla':__import__('torch_xla').__version__,
                      'execution_dtype':str(self.pipe.unet.dtype),'cache':str(cache),
                      'pipeline_load_seconds':self.load_seconds,'rng_protocol':RNG_PROTOCOL})

    def run(self,stage,policy_record,prompt_index,reference,edge=None):
        import torch_xla.core.xla_model as xm
        cfg=self.cfg;seed=self.seed
        key=_unit_key(cfg,reference,policy_record,prompt_index,seed,stage,edge)
        key_hash=digest(key)
        dest=self.run_root/'raw'/stage/f'n{cfg["particles"]}'/f'seed{seed}'/f'{key_hash}.json'
        if dest.exists():
            old=read(dest)
            if old.get('unit_hash')!=key_hash or not math.isfinite(old['selected_final_score']):
                raise RuntimeError(f'invalid resumable record {dest}')
            return old
        prompt=self.prompts[prompt_index].get('prompt',self.prompts[prompt_index].get('text'))
        pseed=baseline._prompt_seed(seed,prompt_index)
        random.seed(pseed);np.random.seed(pseed);baseline._seed_numba(pseed)
        torch.manual_seed(pseed);xm.set_rng_state(pseed,self.pipe.device)
        generator=torch.Generator(device='cpu').manual_seed(pseed)
        generator_hash=sha256(generator.get_state().numpy().tobytes()).hexdigest()
        steps=policy_record['resampling_steps']
        manual=policy_record['kind']=='manual'
        fkd_args={'lmbda':float(policy_record.get('base_temperature',10.0)),
                  'num_particles':cfg['particles'],'use_smc':True,'adaptive_resampling':False,
                  'resample_frequency':0,'time_steps':100,'resampling_t_start':-1,
                  'resampling_t_end':-1,'guidance_reward_fn':'ImageReward',
                  'potential_type':'max','tempering_schedule':policy_record.get('tempering','constant'),
                  'resampling':'ssp','gamma':policy_record.get('gamma'),
                  'resampling_steps':list(steps),'selection_mode':'manual' if manual else 'raw_tau',
                  'temperature_by_step':policy_record.get('temperature_by_step',[]),
                  'graph_trial_seed':seed,'graph_prompt_index':prompt_index}
        start=time.perf_counter()
        try:
            output=self.pipe([prompt]*cfg['particles'],num_inference_steps=100,eta=1.0,
                        num_images_per_prompt=1,generator=generator,fkd_args=fkd_args,
                        output_type='pil',callback_on_step_end=baseline._xla_callback,
                        callback_on_step_end_tensor_inputs=['latents'])
            xm.mark_step();xm.wait_device_ops()
            images=list(output.images)
            final_scores=[float(x) for x in self.reward.score_batched([prompt]*cfg['particles'],images)]
            if len(final_scores)!=cfg['particles'] or not all(math.isfinite(x) for x in final_scores):
                raise RuntimeError('non-finite or incomplete final ImageReward')
            selected=int(np.argmax(np.asarray(final_scores,dtype=np.float64)))
            rgb=np.asarray(images[selected].convert('RGB'),dtype=np.uint8)
            rgb_std=float(rgb.astype(np.float32).std())
            if rgb_std<=1.0:raise RuntimeError(f'near-uniform selected image std={rgb_std}')
            adapter=baseline._LAST_ADAPTER
            if adapter is None:raise RuntimeError('adapter missing')
            events=list(adapter.events)
            if [e['sampling_index'] for e in events]!=steps:
                raise RuntimeError('event count or order mismatch')
            for event in events:
                idx=event['sampling_index']
                if manual and policy_record['tempering']=='increase':
                    expected=float(policy_record['base_temperature'])*((1+float(policy_record['gamma']))**idx-1)
                elif manual and policy_record['tempering']=='constant':
                    expected=float(policy_record['base_temperature'])
                else:
                    expected=dict(policy_record['temperature_by_step'])[idx]
                if not math.isclose(event['temperature'],expected,rel_tol=1e-10,abs_tol=1e-10):
                    raise RuntimeError('event temperature mismatch')
                if event['event_seed']!=event_seed(seed,prompt_index,idx):
                    raise RuntimeError('event RNG mismatch')
                if len(event['parent_indices'])!=cfg['particles'] or any(
                    not 0<=int(x)<cfg['particles'] for x in event['parent_indices']):
                    raise RuntimeError('invalid SSP parent indices')
            record={'unit_hash':key_hash,'unit_key':key,'rng_protocol':RNG_PROTOCOL,
                    'prompt_seed':pseed,'initial_generator_state_sha256':generator_hash,
                    'policy_id':policy_record['id'],'prompt_index':prompt_index,'trial_seed':seed,
                    'final_particle_scores':final_scores,'selected_particle_index':selected,
                    'selected_final_score':final_scores[selected],'selected_image_rgb_std':rgb_std,
                    'selected_image_rgb_sha256':sha256(rgb.tobytes()).hexdigest(),
                    'resampling_events':events,'verifier_calls':len(events)*cfg['particles']+cfg['particles'],
                    'diffusion_NFE':100*cfg['particles'],'elapsed_seconds':time.perf_counter()-start,
                    'pipeline_load_seconds':self.load_seconds,'execution_dtype':str(self.pipe.unet.dtype)}
            write_atomic(dest,record)
            print(json.dumps({'stage':stage,'seed':seed,'prompt':prompt_index,
                  'policy':policy_record['id'],'score':record['selected_final_score'],
                  'seconds':record['elapsed_seconds']}),flush=True)
            return record
        except BaseException as exc:
            fail=self.run_root/'failures'/stage/f'{key_hash}.json'
            write_atomic(fail,{'unit_key':key,'unit_hash':key_hash,'exception':repr(exc),
                               'traceback':traceback.format_exc(),'stage':stage,
                               'rng_protocol':RNG_PROTOCOL})
            raise


def run_stage(args):
    run_root=REPO/'results/static-bfs-graph'/args.run_id
    dcs_root=Path(args.dcs_root)
    cfg,old,prompts,split=prepare(run_root,dcs_root)
    if args.stage=='prepare':return
    if args.stage=='self_test':
        import subprocess
        env=dict(os.environ,OFFICIAL_ROOT=str(dcs_root/'official'))
        proc=subprocess.run([sys.executable,'-m','unittest','scripts.static_bfs_graph.test_graph','-v'],
                            cwd=REPO,env=env,text=True,capture_output=True)
        write_atomic(run_root/'structural_tests.json',{'passed':proc.returncode==0,
                       'stdout':proc.stdout,'stderr':proc.stderr,'rng_protocol':RNG_PROTOCOL})
        if proc.returncode:raise RuntimeError(proc.stdout+proc.stderr)
        print('SELF_TEST_PASS',flush=True);return
    if args.stage in {'tiny_search','tiny_aggregate','pilot_search','pilot_aggregate'}:
        if args.stage.endswith('_search'):
            from scripts.static_bfs_graph.evaluate_paths import search_graph
            search_graph(run_root,cfg,split,args.stage.split('_')[0])
        elif args.stage=='tiny_aggregate':
            from scripts.static_bfs_graph.aggregate import aggregate_tiny
            aggregate_tiny(run_root,cfg,split)
        else:
            from scripts.static_bfs_graph.aggregate import aggregate_pilot
            aggregate_pilot(run_root,cfg,split)
        return
    if args.seed not in cfg['tiny']['seeds'] or args.chip is None:
        raise RuntimeError('TPU stage needs registered --seed and --chip')
    runtime=Runtime(run_root,dcs_root,args.seed,args.chip,cfg,old,prompts)
    scope='pilot' if args.stage.startswith('pilot_') or args.stage=='validation' else 'tiny'
    grid=cfg[scope]
    indices=split['tiny_train_indices'] if scope=='tiny' else sorted(split['train_indices'])[:grid['train_prompts']]
    steps=tuple(grid['candidate_steps'])
    taus=tuple(map(float,grid['taus']))
    reference=dense_reference(steps,float(grid['reference_tau']),cfg['particles'])
    refrec=static_record(reference)
    if args.stage=='canary':
        idx=indices[0]
        constant_steps=(20,40,80)
        constant=StaticBFSPolicy(constant_steps,tuple((s,10.0) for s in constant_steps),4)
        explicit=runtime.run('canary_explicit_constant',static_record(constant),idx,reference)
        old_constant={'kind':'manual','id':'manual_constant_max_ssp_n4','resampling_steps':list(constant_steps),
                      'base_temperature':10.0,'tempering':'constant','gamma':None,'particles':4,
                      'selection_mode':'manual'}
        manual_constant=runtime.run('canary_manual_constant',old_constant,idx,reference)
        manual_bfs=runtime.run('canary_manual_bfs',manual_policy_record(cfg),idx,reference)
        equal=(explicit['final_particle_scores']==manual_constant['final_particle_scores'] and
               [e['parent_indices'] for e in explicit['resampling_events']]==
               [e['parent_indices'] for e in manual_constant['resampling_events']])
        result={'passed':equal,'prompt_index':idx,'seed':args.seed,'manual_bfs_reward':manual_bfs['selected_final_score'],
                'explicit_constant_reward':explicit['selected_final_score'],
                'manual_constant_reward':manual_constant['selected_final_score'],
                'manual_bfs_temperatures':[e['temperature'] for e in manual_bfs['resampling_events']],
                'rng_protocol':RNG_PROTOCOL,'execution_dtype':'bfloat16'}
        write_atomic(run_root/'canary.json',result)
        if not equal:raise RuntimeError('explicit tau10 != old constant-Max-SSP semantics')
        print('CANARY_PASS',flush=True);return
    if args.stage in {'tiny_reference','pilot_reference'}:
        for idx in indices:runtime.run(args.stage,refrec,idx,reference)
        return
    if args.stage in {'tiny_edges','pilot_edges'}:
        for edge in all_edges(steps,taus):
            policy=edge_intervention(edge,reference);rec=static_record(policy)
            for idx in indices:
                if policy==reference:
                    original=runtime.run(f'{scope}_reference',refrec,idx,reference)
                    key=_unit_key(cfg,reference,rec,idx,args.seed,args.stage,edge)
                    h=digest(key);dest=run_root/'raw'/args.stage/f'n{cfg["particles"]}'/f'seed{args.seed}'/f'{h}.json'
                    row={**original,'unit_hash':h,'unit_key':key,'edge':edge.record(),
                         'reused_identical_reference':True}
                    write_once(dest,row)
                else:runtime.run(args.stage,rec,idx,reference,edge)
        return
    if args.stage in {'tiny_validate','pilot_validate'}:
        paths=read(run_root/f'paths/{scope}_selected.json')
        for entry in paths['evaluation_policies']:
            if entry.get('kind')=='manual': rec=manual_policy_record(cfg)
            else:
                p=entry['policy'];policy=StaticBFSPolicy(tuple(p['resampling_steps']),
                         tuple((int(i),float(t)) for i,t in p['temperature_by_step']),4)
                rec=static_record(policy)
            for idx in indices:runtime.run(args.stage,rec,idx,reference)
        return
    if args.stage=='validation':
        frozen_path=run_root/'frozen_policies.json'
        expected=(run_root/'frozen_policies.sha256').read_text().strip().split()[0]
        if file_sha(frozen_path)!=expected:
            raise RuntimeError('frozen policy hash changed before validation')
        frozen=read(frozen_path)
        if frozen['prompt_split_sha256']!=file_sha(run_root/'prompt_split.json'):
            raise RuntimeError('prompt split differs from frozen checkpoint')
        if frozen['edge_table_sha256']!=file_sha(run_root/'edges/pilot_edge_values.csv'):
            raise RuntimeError('graph edge table differs from frozen checkpoint')
        if frozen['config_sha256']!=file_sha(CONFIG):
            raise RuntimeError('experiment config differs from frozen checkpoint')
        for relative,expected_code_hash in frozen['code_sha256'].items():
            if file_sha(REPO/relative)!=expected_code_hash:
                raise RuntimeError(f'code differs from frozen checkpoint: {relative}')
        for entry in frozen['policies']:
            if entry['kind']=='manual':rec=manual_policy_record(cfg)
            else:
                p=entry['policy'];policy=StaticBFSPolicy(tuple(p['resampling_steps']),
                    tuple((int(i),float(t)) for i,t in p['temperature_by_step']),4)
                if policy.id!=entry['policy_id']:raise RuntimeError('frozen policy ID mismatch')
                rec=static_record(policy)
            for idx in split['validation_indices']:
                runtime.run('validation',rec,idx,reference)
        return
    raise ValueError(args.stage)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['prepare','self_test','canary','tiny_reference','tiny_edges',
                                          'tiny_search','tiny_validate','tiny_aggregate',
                                          'pilot_reference','pilot_edges','pilot_search',
                                          'pilot_validate','pilot_aggregate','validation'])
    parser.add_argument('--run-id',default=os.environ.get('STATIC_BFS_RUN_ID','tiny-v1'))
    parser.add_argument('--dcs-root',default=os.environ.get('DCS_ROOT','/home/thanhlamtba31/sd15-tpu-test'))
    parser.add_argument('--seed',type=int)
    parser.add_argument('--chip',type=int)
    args=parser.parse_args()
    run_stage(args)


if __name__=='__main__':main()
