"""TRAIN-only N=8 graph sample-efficiency study; no policy evaluation or TEST."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
import os
from pathlib import Path
import random
import shutil
from statistics import mean

import numpy as np

from .adapter import RNG_PROTOCOL, event_seed
from .graph_builder import all_edges, all_paths, dense_reference, edge_intervention, policy_from_path
from .n8_plan import PILOT, PILOT_SPLIT_SHA, SEEDS, n8_config
from .n8_run import dest_for
from .runner import REPO, CONFIG, BASELINE_CONFIG, Runtime, config_and_prompts, file_sha, read, static_record, write_atomic, write_once

SOURCE = REPO/'results/static-bfs-graph/n8-retuning-v1'
RUN_ID = 'n8-graph-sample-efficiency-v1'
SIZES = (12, 24, 36, 48, 60)
ORDER_SEED = 20261001
BOOTSTRAP_SEED = 20261022
BOOTSTRAPS = 2000


def sha_guard(path: Path, expected: str):
    if file_sha(path) != expected:
        raise RuntimeError(f'SHA mismatch: {path}')


def preflight(root: Path):
    integrity = read(SOURCE/'integrity.json')
    if not integrity['passed'] or integrity['N'] != 8 or integrity['edge_records'] != 4416 or integrity['failed_rollouts']:
        raise RuntimeError('source N=8 graph has invalid integrity')
    for rel, key in (('graph_plan.json','graph_plan_sha256'),
                     ('graph/n8_edge_values.csv','edge_table_sha256'),
                     ('graph/retuned_n8_policy.json','retuned_policy_sha256')):
        sha_guard(SOURCE/rel, integrity[key])
    sha_guard(PILOT/'prompt_split.json', PILOT_SPLIT_SHA)
    original = read(SOURCE/'graph_plan.json')
    split = read(PILOT/'prompt_split.json')
    train = sorted(split['train_indices'])
    fit12 = original['graph_fit_12']
    remaining = original['heldout_train_48'][:]
    if len(train) != 60 or fit12 != train[:12] or remaining != train[12:] or set(train)&set(split['validation_indices']+split['test_indices']):
        raise RuntimeError('TRAIN membership changed')
    random.Random(ORDER_SEED).shuffle(remaining)
    order = fit12 + remaining
    cfg = n8_config()
    grid = cfg['pilot']
    edges = all_edges(tuple(grid['candidate_steps']), tuple(map(float,grid['taus'])))
    if len(edges) != 92 or [e.key for e in edges] != original['edge_keys']:
        raise RuntimeError('N=8 graph action space changed')
    plan = {'protocol':'STATIC_BFS_N8_GRAPH_SAMPLE_EFFICIENCY_V1',
            'source_run':str(SOURCE), 'source_graph_plan_sha256':integrity['graph_plan_sha256'],
            'source_edge_table_sha256':integrity['edge_table_sha256'],
            'N':8, 'K':3, 'seeds':SEEDS, 'fit_order_60':order,
            'original_graph_fit_12':fit12, 'additional_train_48':remaining,
            'sizes':list(SIZES), 'order_seed':ORDER_SEED,
            'bootstrap_seed_base':BOOTSTRAP_SEED, 'bootstraps_per_size':BOOTSTRAPS,
            'candidate_steps':grid['candidate_steps'], 'taus':grid['taus'],
            'reference_tau':grid['reference_tau'], 'edge_keys':[e.key for e in edges],
            'DDIM_steps':100, 'eta':1.0, 'dtype':'bfloat16',
            'verifier':'ImageReward-v1.0', 'scoring':'Max', 'resampling':'SSP',
            'rng_protocol':RNG_PROTOCOL, 'old_48_were_previously_used_for_policy_diagnostic':True,
            'no_new_policy_reward_evaluation':True,
            'validation_test_external_pool_touched':False}
    write_once(root/'graph_plan.json', plan)
    plan_sha = file_sha(root/'graph_plan.json')
    write_once(root/'experiment_manifest.json',
               {'protocol':plan['protocol'], 'source_run':str(SOURCE),
                'source_graph_plan_sha256':integrity['graph_plan_sha256'],
                'graph_plan_sha256':plan_sha, 'N4_split_sha256':PILOT_SPLIT_SHA,
                'config_sha256':file_sha(CONFIG),
                'baseline_config_sha256':file_sha(BASELINE_CONFIG),
                'branch':integrity['branch'], 'commit':integrity['commit'],
                'physical_tpu_chips':4, 'seeds':SEEDS,
                'reused_prompt_count':12, 'new_prompt_count':48,
                'expected_reference_records':240, 'expected_edge_records':22080,
                'validation_touched':False, 'test_touched':False,
                'external_100_prompt_touched':False})
    copied = 0
    for stage in ('n8_reference','n8_edges'):
        for seed in SEEDS:
            src = SOURCE/'raw'/stage/'n8'/f'seed{seed}'
            dest = root/'raw'/stage/'n8'/f'seed{seed}'
            dest.mkdir(parents=True,exist_ok=True)
            for path in src.glob('*.json'):
                row = read(path)
                if int(row['prompt_index']) not in fit12 or int(row['trial_seed']) != seed:
                    raise RuntimeError(f'bad source raw record {path}')
                target = dest/path.name
                if target.exists():
                    if file_sha(target) != file_sha(path):
                        raise RuntimeError(f'copied source changed {target}')
                else:
                    shutil.copy2(path,target)
                copied += 1
    if copied != 48+4416:
        raise RuntimeError(f'wrong reused record count: {copied}')
    write_once(root/'source_reuse.json',
               {'source_run':str(SOURCE), 'source_integrity_sha256':file_sha(SOURCE/'integrity.json'),
                'reused_reference_records':48, 'reused_edge_records':4416,
                'copied_record_count':copied, 'new_edge_records_planned':17664})
    print('N8_SAMPLE_EFFICIENCY_PREFLIGHT_PASS', plan_sha, 'reused',copied,
          'new_edge_units',17664,flush=True)


def worker(root:Path, dcs_root:Path, stage:str, seed:int, chip:int, max_new_units:int|None):
    if chip not in range(4) or seed != SEEDS[chip]:
        raise RuntimeError('seed/chip pairing differs')
    plan = read(root/'graph_plan.json')
    manifest = read(root/'experiment_manifest.json')
    sha_guard(root/'graph_plan.json', manifest['graph_plan_sha256'])
    sha_guard(SOURCE/'graph_plan.json', plan['source_graph_plan_sha256'])
    if plan['fit_order_60'][:12] != plan['original_graph_fit_12'] or len(plan['fit_order_60']) != 60:
        raise RuntimeError('fit order changed')
    cfg = n8_config()
    _,old,prompts = config_and_prompts(dcs_root)
    grid = cfg['pilot']
    steps = tuple(grid['candidate_steps'])
    taus = tuple(map(float,grid['taus']))
    edges = all_edges(steps,taus)
    if [e.key for e in edges] != plan['edge_keys']:
        raise RuntimeError('edge catalogue changed')
    ref = dense_reference(steps,float(grid['reference_tau']),8)
    refrec = static_record(ref)
    additional = plan['additional_train_48']
    runtime = Runtime(root,dcs_root,seed,chip,cfg,old,prompts)
    new=copied=0
    if stage == 'reference':
        for p in additional:
            dest,_,_=dest_for(root,'n8_reference',cfg,ref,refrec,p,seed)
            if not dest.exists(): new += 1
            runtime.run('n8_reference',refrec,p,ref)
            if max_new_units and new >= max_new_units: break
    elif stage == 'edges':
        for edge in edges:
            policy = edge_intervention(edge,ref)
            rec = static_record(policy)
            for p in additional:
                dest,key,h=dest_for(root,'n8_edges',cfg,ref,rec,p,seed,edge)
                if dest.exists():
                    row=read(dest)
                    if row['unit_hash'] != h or row['policy_id'] != rec['id']:
                        raise RuntimeError(f'invalid resumed edge {dest}')
                    continue
                if policy == ref:
                    src,_,_=dest_for(root,'n8_reference',cfg,ref,refrec,p,seed)
                    row=read(src)
                    if row['policy_id'] != rec['id'] or row['diffusion_NFE'] != 800:
                        raise RuntimeError('invalid exact-reference reuse')
                    write_atomic(dest,{**row,'unit_key':key,'unit_hash':h,
                                       'edge':edge.record(), 'reused_identical_reference':True})
                    copied += 1
                else:
                    runtime.run('n8_edges',rec,p,ref,edge)
                    new += 1
                    if max_new_units and new >= max_new_units: break
            if max_new_units and new >= max_new_units: break
    else:
        raise ValueError(stage)
    print('N8_SAMPLE_EFFICIENCY_SEED_DONE',stage,seed,'new',new,'copied',copied,flush=True)


def _csv(path:Path, rows:list[dict]):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def analyze(root:Path):
    plan=read(root/'graph_plan.json');manifest=read(root/'experiment_manifest.json')
    sha_guard(root/'graph_plan.json',manifest['graph_plan_sha256'])
    if any(read(root/'status'/f'{s}.json')['state']!='complete' for s in ('preflight','reference','edges')):
        raise RuntimeError('rollout stages incomplete')
    cfg=n8_config();grid=cfg['pilot'];steps=tuple(grid['candidate_steps']);taus=tuple(map(float,grid['taus']))
    edges=all_edges(steps,taus);ref=dense_reference(steps,float(grid['reference_tau']),8)
    fit=plan['fit_order_60'];seeds=plan['seeds']
    for stage,expected in (('n8_reference',240),('n8_edges',22080)):
        actual=len(list((root/'raw'/stage/'n8').glob('seed*/*.json')))
        if actual!=expected:raise RuntimeError(f'{stage} physical count {actual} != {expected}')
    scores=np.empty((60,4,92),dtype=float)
    ref_scores=np.empty((60,4),dtype=float)
    for pi,p in enumerate(fit):
        for si,s in enumerate(seeds):
            refrec=static_record(ref)
            rp,_,rh=dest_for(root,'n8_reference',cfg,ref,refrec,p,s)
            rr=read(rp)
            if (rr['unit_hash']!=rh or rr['policy_id']!=ref.id or rr['diffusion_NFE']!=800 or
                rr['rng_protocol']!=RNG_PROTOCOL or len(rr['final_particle_scores'])!=8 or
                [ev['sampling_index'] for ev in rr['resampling_events']]!=list(ref.resampling_steps)):
                raise RuntimeError(f'invalid reference {p}|{s}')
            ref_scores[pi,si]=rr['selected_final_score']
            for ei,e in enumerate(edges):
                policy=edge_intervention(e,ref);rec=static_record(policy)
                ep,_,eh=dest_for(root,'n8_edges',cfg,ref,rec,p,s,e)
                er=read(ep)
                if (er['unit_hash']!=eh or er['policy_id']!=policy.id or
                    er['initial_generator_state_sha256']!=rr['initial_generator_state_sha256'] or
                    er['diffusion_NFE']!=800 or er['rng_protocol']!=RNG_PROTOCOL or
                    len(er['final_particle_scores'])!=8 or
                    [ev['sampling_index'] for ev in er['resampling_events']]!=list(policy.resampling_steps) or
                    any(ev['event_seed']!=event_seed(s,p,ev['sampling_index']) for ev in er['resampling_events'])):
                    raise RuntimeError(f'invalid edge {e.key}|{p}|{s}')
                scores[pi,si,ei]=er['selected_final_score']
    if not np.isfinite(scores).all() or not np.isfinite(ref_scores).all():
        raise RuntimeError('nonfinite score')
    matrix=(ref_scores[:,:,None]-scores).mean(axis=1)
    paths=list(all_paths(steps,taus,3))
    paths.sort(key=lambda x:policy_from_path(x,8).id)
    edge_idx={e.key:i for i,e in enumerate(edges)}
    path_idx=np.asarray([[edge_idx[e.key] for e in path] for path in paths],dtype=int)
    path_policies=[policy_from_path(path,8) for path in paths]
    if len(paths)!=945:raise RuntimeError('wrong K=3 path count')
    frozen=read(SOURCE/'graph/retuned_n8_policy.json')
    out=[];boot_rows=[];edge_rows=[]
    for n in SIZES:
        sub=matrix[:n]
        values=sub.mean(axis=0)
        choice=int(np.argmin(values[path_idx].sum(axis=1)))
        policy=path_policies[choice]
        if n==12:
            if policy.id!=frozen['policy_id']:
                raise RuntimeError('reused n12 graph does not reproduce frozen policy')
            with (SOURCE/'graph/n8_edge_values.csv').open() as f:
                old={r['edge_key']:float(r['mean_delta']) for r in csv.DictReader(f)}
            if max(abs(values[edge_idx[k]]-v) for k,v in old.items())>1e-12:
                raise RuntimeError('n12 edge values do not reproduce')
        rng=np.random.default_rng(BOOTSTRAP_SEED if n==12 else BOOTSTRAP_SEED+n)
        samples=rng.integers(0,n,size=(BOOTSTRAPS,n))
        picks=[]
        for sample in samples:
            weights=np.bincount(sample,minlength=n)/n
            v=weights@sub
            picks.append(int(np.argmin(v[path_idx].sum(axis=1))))
        counts=Counter(picks)
        exact=counts[choice]/BOOTSTRAPS
        if n==12 and abs(exact-read(SOURCE/'graph/bootstrap_summary.json')['frozen_P8_exact_path_frequency'])>1e-12:
            raise RuntimeError('n12 bootstrap does not reproduce')
        tau40_2=sum(path_policies[i].temperature_map.get(40)==2.0 for i in picks)/BOOTSTRAPS
        tau40_8=sum(path_policies[i].temperature_map.get(40)==8.0 for i in picks)/BOOTSTRAPS
        tau40_32=sum(path_policies[i].temperature_map.get(40)==32.0 for i in picks)/BOOTSTRAPS
        tau40_inclusion=tau40_2+tau40_8+tau40_32
        steps_out=list(policy.resampling_steps);taus_out=[t for _,t in policy.temperature_by_step]
        out.append({'n_fit':n,'policy_id':policy.id,'steps':json.dumps(steps_out),
                    'taus':json.dumps(taus_out),'predicted_additive_regret':float(values[path_idx[choice]].sum()),
                    'exact_path_frequency':exact,
                    'top2_cumulative_frequency':sum(c for _,c in counts.most_common(2))/BOOTSTRAPS,
                    'unique_paths':len(counts),'P_tau40_2':tau40_2,'P_tau40_8':tau40_8,
                    'P_tau40_32':tau40_32,'P_step40_included':tau40_inclusion,
                    'P_tau40_2_given_included':tau40_2/tau40_inclusion if tau40_inclusion else math.nan,
                    'P_tau40_8_given_included':tau40_8/tau40_inclusion if tau40_inclusion else math.nan,
                    'bootstrap_seed':BOOTSTRAP_SEED if n==12 else BOOTSTRAP_SEED+n})
        for i,c in counts.most_common():
            pol=path_policies[i]
            boot_rows.append({'n_fit':n,'policy_id':pol.id,'steps':json.dumps(pol.resampling_steps),
                              'taus':json.dumps([t for _,t in pol.temperature_by_step]),
                              'count':c,'frequency':c/BOOTSTRAPS,'is_full_data_policy':i==choice})
        for ei,e in enumerate(edges):
            edge_rows.append({'n_fit':n,'edge_key':e.key,'src':e.src,'dst':e.dst,
                              'tau_dst':e.tau_dst,'mean_regret':float(values[ei]),
                              'prompt_std':float(sub[:,ei].std(ddof=1))})
    _csv(root/'analysis/policy_by_fit_size.csv',out)
    _csv(root/'analysis/bootstrap_paths.csv',boot_rows)
    _csv(root/'analysis/edge_values_by_fit_size.csv',edge_rows)
    summary={'protocol':plan['protocol'],'status':'complete','source_graph_plan_sha256':plan['source_graph_plan_sha256'],
             'graph_plan_sha256':manifest['graph_plan_sha256'],'N':8,'K':3,'edge_count':92,
             'fit_sizes':list(SIZES),'reference_records':240,'edge_records':22080,
             'reused_reference_records':48,'reused_edge_records':4416,
             'new_reference_records':192,'new_edge_records':17664,
             'failed_rollouts':len(list((root/'failures').rglob('*.json'))) if (root/'failures').exists() else 0,
             'validation_touched':False,'test_touched':False,'external_100_prompt_touched':False,
             'no_new_policy_reward_evaluation':True,'n12_reproduced':True,
             'rows':out}
    if summary['failed_rollouts']:
        raise RuntimeError('failure records exist')
    write_atomic(root/'integrity.json',summary)
    lines=['# N=8 graph sample efficiency (TRAIN only)','',
           'The old 12-prompt graph is reused exactly. The other 48 TRAIN prompts were previously used in a policy diagnostic, so all 60-prompt findings are exploratory. No new policy reward evaluation was run.','',
           '| Fit prompts | Selected steps | Selected taus | Exact-path bootstrap | P(tau40=2) | P(tau40=8) |',
           '|---:|---|---|---:|---:|---:|']
    for row in out:
        lines.append(f"| {row['n_fit']} | {row['steps']} | {row['taus']} | {row['exact_path_frequency']:.1%} | {row['P_tau40_2']:.1%} | {row['P_tau40_8']:.1%} |")
    lines += ['',f"All {summary['edge_records']} edge records and {summary['reference_records']} dense-reference records passed integrity checks; zero failures.",
              'Bootstrap unit: prompt with all four seeds grouped; 2,000 replicates per fit size.',
              'The plotted tau40 probabilities are unconditional: policies without a step-40 event count toward neither tau.',
              'No VALIDATION, TEST, external 100-prompt pool, or final policy selection was used.','']
    (root/'report.md').write_text('\n'.join(lines))
    print('N8_SAMPLE_EFFICIENCY_ANALYSIS_COMPLETE',json.dumps(out),flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('stage',choices=('preflight','reference','edges','analyze'))
    ap.add_argument('--run-id',default=RUN_ID)
    ap.add_argument('--dcs-root',default='/home/thanhlamtba31/sd15-tpu-test')
    ap.add_argument('--seed',type=int)
    ap.add_argument('--chip',type=int)
    ap.add_argument('--max-new-units',type=int)
    args=ap.parse_args()
    root=REPO/'results/static-bfs-graph'/args.run_id
    if args.stage=='preflight':preflight(root)
    elif args.stage=='analyze':analyze(root)
    else:
        if args.seed is None or args.chip is None:raise RuntimeError('seed/chip required')
        worker(root,Path(args.dcs_root),args.stage,args.seed,args.chip,args.max_new_units)


if __name__=='__main__':main()
