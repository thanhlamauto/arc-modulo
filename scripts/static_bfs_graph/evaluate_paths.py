"""Train-only edge targets and frozen tiny path selection."""
from __future__ import annotations

import csv
from itertools import combinations
from pathlib import Path
import json
import math
import random
import statistics

from .graph_builder import all_edges, all_paths, dense_reference, path_edges, policy_from_path
from .graph_search import predicted_cost, select_paths
from .policy import StaticBFSPolicy


def read(path):return json.loads(path.read_text())


def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    raw=json.dumps(obj,indent=2,sort_keys=True)+'\n'
    if path.exists() and path.read_text()!=raw:raise RuntimeError(f'frozen output differs: {path}')
    if not path.exists():path.write_text(raw)


def write_csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def stage_records(root:Path,stage:str):
    records={}
    for path in sorted((root/'raw'/stage).glob('n4/seed*/*.json')):
        row=read(path)
        key=row['unit_key'];pair=(int(key['prompt_index']),int(key['trial_seed']))
        if stage.endswith('_edges'):
            edge_key=key['edge']['key']
            record_key=(edge_key,*pair)
        else:
            record_key=(row['policy_id'],*pair)
        if record_key in records:raise RuntimeError(f'duplicate unit: {record_key}')
        if not math.isfinite(row['selected_final_score']):raise RuntimeError('non-finite reward')
        records[record_key]=row
    return records


def uniform_steps(candidate:tuple[int,...],k:int)->tuple[int,...]:
    targets=[100*(i+1)/(k+1) for i in range(k)]
    return min(combinations(candidate,k),key=lambda x:(sum((a-b)**2 for a,b in zip(x,targets)),x))


def search_graph(run_root:Path,cfg:dict,split:dict,scope:str):
    grid=cfg[scope];steps=tuple(grid['candidate_steps']);taus=tuple(map(float,grid['taus']))
    reference=dense_reference(steps,float(grid['reference_tau']),4)
    indices=split['tiny_train_indices'] if scope=='tiny' else sorted(split['train_indices'])[:grid['train_prompts']]
    pairs={(p,s) for p in indices for s in grid.get('seeds',cfg['tiny']['seeds'])}
    ref=stage_records(run_root,f'{scope}_reference');edges=stage_records(run_root,f'{scope}_edges')
    if (run_root/f'failures/{scope}_edges').exists() and list((run_root/f'failures/{scope}_edges').glob('*.json')):
        raise RuntimeError('edge failures exist')
    edge_rows=[];values={}
    for edge in all_edges(steps,taus):
        deltas=[]
        for p,s in sorted(pairs):
            a=ref.get((reference.id,p,s));b=edges.get((edge.key,p,s))
            if a is None or b is None:raise RuntimeError(f'missing edge/reference {edge.key} {p} {s}')
            deltas.append(float(a['selected_final_score'])-float(b['selected_final_score']))
        mean=statistics.mean(deltas)
        values[edge.key]=mean
        edge_rows.append({'N':4,'src':edge.src,'dst':edge.dst,'tau_dst':edge.tau_dst,
                          'edge_key':edge.key,'mean_delta':mean,
                          'std_delta':statistics.stdev(deltas),
                          'sem_delta':statistics.stdev(deltas)/len(deltas)**.5,
                          'median_delta':statistics.median(deltas),
                          'n_prompt_seed_units':len(deltas)})
    write_csv(run_root/f'edges/{scope}_edge_values.csv',edge_rows)
    if scope=='tiny':write_csv(run_root/'edges/edge_values.csv',edge_rows)
    chosen=[]
    selected_ids=set()
    for k in grid['k_values']:
        result=select_paths(steps,taus,values,4,k)
        for method in ('additive','lexicographic'):
            path=result[method];policy=policy_from_path(path,4)
            chosen.append({'method':method,'K':k,'policy':policy.record(),
                           'policy_id':policy.id,'predicted_delta':predicted_cost(path,values),
                           'path':[e.record() for e in path],
                           'n_candidates_at_K':result['n_paths']})
            selected_ids.add(policy.id)
    rng=random.Random(grid.get('random_seed',cfg['tiny']['random_seed']))
    if scope=='tiny':
        pool=[p for k in grid['k_values'] for p in all_paths(steps,taus,k)
              if policy_from_path(p,4).id not in selected_ids]
        sampled=rng.sample(pool,grid['random_path_count'])
        for i,path in enumerate(sampled):
            policy=policy_from_path(path,4)
            chosen.append({'method':f'random_{i:02d}','K':len(policy.resampling_steps),
                           'policy':policy.record(),'policy_id':policy.id,
                           'predicted_delta':predicted_cost(path,values),
                           'path':[e.record() for e in path]})
    for k in grid['k_values']:
        picked=uniform_steps(steps,k)
        path=path_edges(picked,(float(cfg['uniform_explicit_tau']),)*k)
        policy=policy_from_path(path,4)
        chosen.append({'method':'uniform' if scope=='tiny' else 'uniform_tau8','K':k,
                       'policy':policy.record(),
                       'policy_id':policy.id,'predicted_delta':predicted_cost(path,values),
                       'path':[e.record() for e in path]})
        selected_ids.add(policy.id)
    if scope=='pilot':
        purposeful={
            2:((10,20),(30,60),(80,90)),
            3:((10,20,30),(30,40,60),(60,80,90)),
            4:((10,20,30,40),(20,40,60,80),(40,60,80,90)),
        }
        for k in grid['k_values']:
            for j,(picked,tau) in enumerate(zip(purposeful[k],taus)):
                path=path_edges(picked,(tau,)*k)
                policy=policy_from_path(path,4)
                if policy.id in selected_ids:continue
                chosen.append({'method':f'purposeful_{k}_{j}','K':k,
                               'policy':policy.record(),'policy_id':policy.id,
                               'predicted_delta':predicted_cost(path,values),
                               'path':[e.record() for e in path]})
                selected_ids.add(policy.id)
    if scope=='pilot':
        per_k=10
        for k in grid['k_values']:
            pool=[p for p in all_paths(steps,taus,k)
                  if policy_from_path(p,4).id not in selected_ids]
            for i,path in enumerate(rng.sample(pool,per_k)):
                policy=policy_from_path(path,4)
                chosen.append({'method':f'random_k{k}_{i:02d}','K':k,
                               'policy':policy.record(),'policy_id':policy.id,
                               'predicted_delta':predicted_cost(path,values),
                               'path':[e.record() for e in path]})
                selected_ids.add(policy.id)
    chosen.append({'method':'manual_bfs','K':3,'kind':'manual','policy_id':'manual_bfs_g0p008_n4',
                   'predicted_delta':None,'policy':None,'path':None})
    payload={'protocol_version':cfg['protocol_version'],'rng_protocol':cfg['rng_protocol'],
             'train_pairs':len(pairs),'edge_count':len(edge_rows),'candidate_steps':list(steps),
             'taus':list(taus),'selected':[x for x in chosen if x['method'] in ('additive','lexicographic')],
             'evaluation_policies':chosen}
    write(run_root/f'paths/{scope}_selected.json',payload)
    write(run_root/f'paths/{scope}_additive.json',{'selected':[x for x in chosen if x['method']=='additive']})
    write(run_root/f'paths/{scope}_lexicographic.json',{'selected':[x for x in chosen if x['method']=='lexicographic']})
    write(run_root/f'paths/{scope}_random_paths.json',{'selected':[x for x in chosen if x['method'].startswith('random_')]})
    if scope=='tiny':
        write(run_root/'paths/additive.json',{'selected':[x for x in chosen if x['method']=='additive']})
        write(run_root/'paths/lexicographic.json',{'selected':[x for x in chosen if x['method']=='lexicographic']})
        write(run_root/'paths/random_paths.json',{'selected':[x for x in chosen if x['method'].startswith('random_')]})
    print(f'{scope.upper()}_SEARCH_COMPLETE',len(edge_rows),len(chosen),flush=True)


def search_tiny(run_root:Path,cfg:dict,split:dict):
    return search_graph(run_root,cfg,split,'tiny')
