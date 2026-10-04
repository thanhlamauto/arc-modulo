"""Resume-safe N=8 reference, edge, and frozen-policy TPU workers."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

from .adapter import RNG_PROTOCOL, event_seed
from .graph_builder import all_edges, dense_reference, edge_intervention
from .n8_plan import PILOT, PILOT_EDGE_SHA, PILOT_FROZEN_SHA, PILOT_SPLIT_SHA, SEEDS, n8_config
from .policy import StaticBFSPolicy
from .runner import (REPO, Runtime, _unit_key, config_and_prompts,
                     digest, file_sha, read, static_record, write_atomic)


def context(root: Path, dcs_root: Path):
    _, old, prompts = config_and_prompts(dcs_root)
    cfg = n8_config()
    manifest = read(root/'experiment_manifest.json')
    plan = read(root/'graph_plan.json')
    if (file_sha(root/'graph_plan.json') != manifest['graph_plan_sha256'] or
            file_sha(root/'graph_plan.json') != (root/'graph_plan.sha256').read_text().split()[0] or
            file_sha(PILOT/'frozen_policies.json') != PILOT_FROZEN_SHA or
            file_sha(PILOT/'edges/pilot_edge_values.csv') != PILOT_EDGE_SHA or
            file_sha(PILOT/'prompt_split.json') != PILOT_SPLIT_SHA or
            cfg['particles'] != 8 or plan['K'] != 3):
        raise RuntimeError('frozen N=8 source changed')
    return cfg, old, prompts, plan


def manual8_record():
    return {'kind':'manual','id':'manual_bfs_g0p024_n8',
            'resampling_steps':[20,40,80],'base_temperature':10.0,
            'tempering':'increase','gamma':0.024,'particles':8,
            'selection_mode':'manual'}


def static_item(item):
    policy = StaticBFSPolicy(tuple(item['steps']),
                             tuple((int(s),float(t)) for s,t in item['taus']),8)
    if 'policy_id' in item and policy.id != item['policy_id']:
        raise RuntimeError('frozen policy ID differs')
    return static_record(policy)


def dest_for(root, stage, cfg, ref, rec, prompt, seed, edge=None):
    key = _unit_key(cfg,ref,rec,prompt,seed,stage,edge)
    h = digest(key)
    return root/'raw'/stage/f'n{cfg["particles"]}'/f'seed{seed}'/f'{h}.json',key,h


def worker(root:Path,dcs_root:Path,stage:str,seed:int,chip:int,max_new_units:int|None):
    if chip not in range(4) or seed != SEEDS[chip]:
        raise RuntimeError('one paired trial seed per physical TPU chip')
    cfg,old,prompts,plan = context(root,dcs_root)
    grid=cfg['pilot']
    steps=tuple(grid['candidate_steps']);taus=tuple(map(float,grid['taus']))
    ref=dense_reference(steps,float(grid['reference_tau']),8)
    refrec=static_record(ref)
    if stage == 'reference':
        runtime=Runtime(root,dcs_root,seed,chip,cfg,old,prompts)
        new=0
        for p in plan['graph_fit_12']:
            dest,_,_=dest_for(root,'n8_reference',cfg,ref,refrec,p,seed)
            if not dest.exists():new+=1
            runtime.run('n8_reference',refrec,p,ref)
            if max_new_units and new>=max_new_units:
                print('N8_SMOKE_DONE',stage,seed,new,flush=True);return
        print('N8_SEED_DONE',stage,seed,'new',new,flush=True);return
    if stage == 'edges':
        runtime=Runtime(root,dcs_root,seed,chip,cfg,old,prompts)
        new=copied=0
        edges=all_edges(steps,taus)
        if len(edges)!=92 or [e.key for e in edges]!=plan['edge_keys']:
            raise RuntimeError('edge catalogue changed')
        for edge in edges:
            policy=edge_intervention(edge,ref)
            rec=static_record(policy)
            for p in plan['graph_fit_12']:
                dest,key,h=dest_for(root,'n8_edges',cfg,ref,rec,p,seed,edge)
                if dest.exists():
                    row=read(dest)
                    if row['unit_hash']!=h or row['policy_id']!=rec['id']:
                        raise RuntimeError(f'invalid resumable edge: {dest}')
                    continue
                if policy==ref:
                    original,_,_=dest_for(root,'n8_reference',cfg,ref,refrec,p,seed)
                    if not original.exists():
                        raise RuntimeError(f'missing N8 reference for exact copy: {p},{seed}')
                    row=read(original)
                    if (row['policy_id']!=rec['id'] or row['prompt_index']!=p or
                            row['trial_seed']!=seed or row['diffusion_NFE']!=800 or
                            row['rng_protocol']!=RNG_PROTOCOL):
                        raise RuntimeError('reference not eligible for exact reuse')
                    write_atomic(dest,{**row,'unit_key':key,'unit_hash':h,
                                       'edge':edge.record(),
                                       'reused_identical_reference':True})
                    copied+=1
                else:
                    runtime.run('n8_edges',rec,p,ref,edge)
                    new+=1
                    if max_new_units and new>=max_new_units:
                        print('N8_SMOKE_DONE',stage,seed,new,'copied',copied,flush=True);return
        print('N8_SEED_DONE',stage,seed,'new',new,'copied',copied,flush=True);return
    if stage == 'evaluate':
        policy_path=root/'graph/retuned_n8_policy.json'
        plan_path=root/'evaluation/n8_evaluation_plan.json'
        frozen=(root/'graph/retuned_n8_policy.sha256').read_text().split()[0]
        plan_sha=(root/'evaluation/n8_evaluation_plan.sha256').read_text().split()[0]
        if file_sha(policy_path)!=frozen or file_sha(plan_path)!=plan_sha:
            raise RuntimeError('RETUNED8 or held-out plan changed')
        evaluation=read(plan_path)
        if (evaluation['prompts']!=plan['heldout_train_48'] or
                evaluation['seeds']!=SEEDS or len(evaluation['policies'])!=3):
            raise RuntimeError('held-out evaluation plan mismatch')
        records=[]
        for item in evaluation['policies']:
            rec=manual8_record() if item['method']=='MANUAL8' else static_item(item)
            if rec['id']!=item['policy_id']:
                raise RuntimeError('silent evaluation policy substitution')
            records.append(rec)
        runtime=Runtime(root/'evaluation',dcs_root,seed,chip,cfg,old,prompts)
        new=0
        for rec in records:
            for p in evaluation['prompts']:
                dest,_,_=dest_for(root/'evaluation','n8_evaluate',cfg,ref,rec,p,seed)
                if not dest.exists():new+=1
                runtime.run('n8_evaluate',rec,p,ref)
                if max_new_units and new>=max_new_units:
                    print('N8_SMOKE_DONE',stage,seed,new,flush=True);return
        print('N8_SEED_DONE',stage,seed,'new',new,flush=True);return
    raise ValueError(stage)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('stage',choices=('reference','edges','evaluate'))
    ap.add_argument('--run-id',default='n8-retuning-v1')
    ap.add_argument('--dcs-root',default='/home/thanhlamtba31/sd15-tpu-test')
    ap.add_argument('--seed',type=int,required=True)
    ap.add_argument('--chip',type=int,required=True)
    ap.add_argument('--max-new-units',type=int)
    args=ap.parse_args()
    worker(REPO/'results/static-bfs-graph'/args.run_id,Path(args.dcs_root),
           args.stage,args.seed,args.chip,args.max_new_units)


if __name__=='__main__':main()
