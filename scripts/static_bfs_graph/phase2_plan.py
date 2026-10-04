"""Freeze TRAIN-only phase-2 graph shortlist before any new TPU scoring."""
from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
import math
from pathlib import Path
import random
import subprocess

from .adapter import RNG_PROTOCOL
from .graph_builder import all_edges, all_paths, policy_from_path
from .policy import StaticBFSPolicy
from .runner import REPO, CONFIG, BASELINE_CONFIG, file_sha, read, static_record, write_once

PILOT = REPO / 'results/static-bfs-graph/pilot-v1'
STABILITY = REPO / 'results/static-bfs-graph/stability-v1'
BASE_COMMIT = '34f17bd984921aeba046e8a5b7936e4a7a8346c0'
PILOT_FROZEN_SHA = 'ffc61806390cee8502183aba87d8e33e0ee86eb046985b46cb7927cc46ad2408'
PILOT_EDGE_SHA = '451619d6a3115a7931da51fef5e55c8836e210618f2256cea15977e5eb35a435'
PILOT_SPLIT_SHA = 'cd1ced923406b6fa5fce5ffd199880e1a4998830641a22830356c3bc285e733b'
COMP_SEED = 20261001
BOOTSTRAP_SEED = 20261002


def canonical(obj):
    return json.dumps(obj, indent=2, sort_keys=True) + '\n'


def freeze_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text() != text:
            raise RuntimeError(f'frozen output differs: {path}')
    else:
        path.write_text(text)


def policy_record(policy: StaticBFSPolicy, graph_rank=None, graph_score=None):
    result = {'policy_id': policy.id, 'K': len(policy.resampling_steps),
              'steps': list(policy.resampling_steps),
              'taus': [[s, t] for s, t in policy.temperature_by_step]}
    if graph_rank is not None:
        result['graph_rank'] = graph_rank
    if graph_score is not None:
        result['additive_graph_score'] = graph_score
    return result


def validate_sources():
    if file_sha(PILOT / 'frozen_policies.json') != PILOT_FROZEN_SHA:
        raise RuntimeError('pilot frozen policy hash changed')
    if file_sha(PILOT / 'edges/pilot_edge_values.csv') != PILOT_EDGE_SHA:
        raise RuntimeError('pilot edge table hash changed')
    if file_sha(PILOT / 'prompt_split.json') != PILOT_SPLIT_SHA:
        raise RuntimeError('pilot split hash changed')
    frozen = read(PILOT / 'frozen_policies.json')
    if (frozen['edge_table_sha256'] != PILOT_EDGE_SHA or
        frozen['prompt_split_sha256'] != PILOT_SPLIT_SHA or
        frozen['config_sha256'] != file_sha(CONFIG)):
        raise RuntimeError('pilot frozen provenance mismatch')
    for relative, expected in frozen['code_sha256'].items():
        if file_sha(REPO / relative) != expected:
            raise RuntimeError(f'frozen scientific code changed: {relative}')
    branch = subprocess.check_output(['git','branch','--show-current'], cwd=REPO, text=True).strip()
    commit = subprocess.check_output(['git','rev-parse','HEAD'], cwd=REPO, text=True).strip()
    if branch != 'ngocminh' or commit != BASE_COMMIT:
        raise RuntimeError(f'wrong branch/commit {branch} {commit}')
    status = subprocess.check_output(['git','status','--short'], cwd=REPO, text=True)
    protected = ('scripts/diffusion_classical_search_t2i/tpu_runner.py',
                 'configs/diffusion_classical_search_t2i_table12.json',
                 'T2I_RESULTS.md')
    if any(line[3:].startswith(protected) for line in status.splitlines()):
        raise RuntimeError('protected baseline modified')
    if file_sha(STABILITY / 'prompt_split.json') != PILOT_SPLIT_SHA:
        raise RuntimeError('stability TRAIN split differs')
    return frozen, branch, commit


def make_split(run_root: Path):
    split = read(PILOT / 'prompt_split.json')
    train = sorted(split['train_indices'])
    fit = train[:12]
    remaining = train[12:]
    if (len(train), len(fit), len(remaining), len(split['validation_indices']),
        len(split['test_indices'])) != (60,12,48,20,20):
        raise RuntimeError('invalid split cardinalities')
    if set(fit) & set(remaining) or set(train) & (
        set(split['validation_indices']) | set(split['test_indices'])):
        raise RuntimeError('TRAIN overlap with held-out split')
    ordered = sorted(remaining, key=lambda p: (sha256(
        f'STATIC_BFS_PHASE2_SPLIT_V1|{p}'.encode()).hexdigest(), p))
    cal, audit = sorted(ordered[:24]), sorted(ordered[24:])
    result = {'protocol':'STATIC_BFS_PHASE2_SPLIT_V1',
              'method':'SHA256(STATIC_BFS_PHASE2_SPLIT_V1|prompt_index), first 24 calibration',
              'graph_fit_12':fit,'phase2_calibration':cal,'phase2_audit':audit,
              'validation_test_used':False}
    write_once(run_root/'phase2_prompt_split.json', result)
    return result


def make_ranking(run_root: Path):
    cfg = read(CONFIG)
    grid = cfg['pilot']
    steps = tuple(grid['candidate_steps'])
    taus = tuple(map(float, grid['taus']))
    edge_rows = list(csv.DictReader((PILOT/'edges/pilot_edge_values.csv').open()))
    values = {row['edge_key']: float(row['mean_delta']) for row in edge_rows}
    if len(values) != 92 or len(all_edges(steps, taus)) != 92:
        raise RuntimeError('invalid graph edge count')
    by_k = {}
    all_rows = []
    for k, expected in ((2,189),(3,945),(4,2835)):
        rows = []
        for path in all_paths(steps, taus, k):
            policy = policy_from_path(path, 4)
            weights = [values[e.key] for e in path]
            if not all(math.isfinite(w) for w in weights):
                raise RuntimeError('nonfinite edge weight')
            rows.append({'policy_id':policy.id,'K':k,
                         'steps':list(policy.resampling_steps),
                         'taus':[[s,t] for s,t in policy.temperature_by_step],
                         'additive_graph_score':sum(weights),
                         'worst_edge_score':max(weights),
                         'lexicographic_vector':sorted(weights,reverse=True)})
        rows.sort(key=lambda r:(r['additive_graph_score'],r['policy_id']))
        if len(rows)!=expected or len({r['policy_id'] for r in rows})!=expected:
            raise RuntimeError(f'wrong feasible K={k} count')
        for rank,row in enumerate(rows,1):
            row['rank_within_K']=rank
        by_k[k]=rows
        all_rows += rows
    if len(all_rows)!=3969 or len({r['policy_id'] for r in all_rows})!=3969:
        raise RuntimeError('wrong feasible total')
    fields=('policy_id','K','rank_within_K','steps','taus','additive_graph_score',
            'worst_edge_score','lexicographic_vector')
    path=run_root/'all_feasible_policy_ranking.csv'
    lines=[','.join(fields)]
    import io
    buf=io.StringIO();writer=csv.DictWriter(buf,fieldnames=fields,lineterminator='\n')
    writer.writeheader()
    for row in all_rows:
        writer.writerow({key:json.dumps(row[key]) if isinstance(row[key],list) else row[key]
                         for key in fields})
    freeze_text(path,buf.getvalue())
    return by_k


def make_comparison(run_root: Path, k3):
    n=len(k3)
    rng=random.Random(COMP_SEED)
    comparisons=[]
    strata=((.10,.25),(.25,.50),(.50,.75),(.75,1.0))
    for low,high in strata:
        eligible=[r for r in k3 if low <= (r['rank_within_K']-1)/n < high
                  and r['rank_within_K']>20]
        if len(eligible)<5:
            raise RuntimeError('comparison stratum too small')
        for row in rng.sample(eligible,5):
            comparisons.append({**row,'stratum':f'{low:.2f}-{high:.2f}'})
    if len(comparisons)!=20 or len({r['policy_id'] for r in comparisons})!=20:
        raise RuntimeError('comparison selection failed')
    result={'seed':COMP_SEED,'comparison_per_stratum':5,
            'rank_percentile_definition':'(rank_within_K-1)/945',
            'outside_graph_top20':True,'comparison_policies':comparisons}
    write_once(run_root/'comparison_policy_plan.json',result)
    return result


def make_interactions(run_root: Path, by_k):
    parents=[]
    for k in (3,4):
        choices=[r for r in by_k[k] if 60 in r['steps'] and 90 in r['steps']]
        parents.extend(choices[:2])
    if len(parents)!=4:
        raise RuntimeError('missing four interaction parent contexts')
    catalogue={}
    contexts=[]
    for row in parents:
        steps=row['steps'];tau=dict(row['taus'])
        variants={}
        for label,removed in (('parent',()),('without_60',(60,)),
                              ('without_90',(90,)),('without_60_90',(60,90))):
            kept=tuple(s for s in steps if s not in removed)
            policy=StaticBFSPolicy(kept,tuple((s,tau[s]) for s in kept),4)
            rec=policy_record(policy)
            catalogue[policy.id]=rec
            variants[label]=policy.id
        contexts.append({'parent':row,'variants':variants})
    result={'hypothesis':'prespecified positive 60-90 reward interaction',
            'parent_selection':'top two additive K3 and top two additive K4 containing steps 60 and 90',
            'contexts':contexts,'unique_policies':sorted(catalogue.values(),key=lambda r:r['policy_id']),
            'validation_test_used':False}
    write_once(run_root/'interaction/interaction_plan.json',result)
    return result


def make_reuse_map(run_root: Path, candidates, split):
    wanted={r['policy_id']:r for r in candidates}
    allowed=set(split['phase2_calibration']+split['phase2_audit'])
    found={}
    for stage in ('stability_reference','stability_edges'):
        for path in sorted((STABILITY/'raw'/stage/'n4').glob('seed*/*.json')):
            row=read(path)
            pid=row['policy_id']
            if pid not in wanted:
                continue
            p,s=int(row['prompt_index']),int(row['trial_seed'])
            if p not in allowed or s not in (42,43,44,45):
                continue
            item=wanted[pid]
            policy=StaticBFSPolicy(tuple(item['steps']),
                                   tuple((int(x),float(y)) for x,y in item['taus']),4)
            expected=static_record(policy)
            if (row['unit_key']['policy']!=expected or row['rng_protocol']!=RNG_PROTOCOL or
                row['execution_dtype']!='torch.bfloat16' or row['diffusion_NFE']!=400 or
                not math.isfinite(row['selected_final_score'])):
                raise RuntimeError(f'incompatible reuse record: {path}')
            key=f'{pid}|{p}|{s}'
            relative=str(path.relative_to(REPO))
            if key in found:
                previous=read(REPO/found[key])
                if (previous['final_particle_scores']!=row['final_particle_scores'] or
                    previous['selected_image_rgb_sha256']!=row['selected_image_rgb_sha256']):
                    raise RuntimeError(f'conflicting reuse records: {key}')
            else:
                found[key]=relative
    result={'protocol':'STATIC_BFS_PHASE2_EXACT_REUSE_V1','source_run':'stability-v1',
            'source_stages':['stability_reference','stability_edges'],
            'records':found,'n_records':len(found)}
    write_once(run_root/'reuse_map.json',result)
    return result


def preflight(run_root: Path):
    frozen,branch,commit=validate_sources()
    split=make_split(run_root)
    by_k=make_ranking(run_root)
    comparison=make_comparison(run_root,by_k[3])
    interaction=make_interactions(run_root,by_k)
    top20=by_k[3][:20]
    mixed=[*by_k[2][:5],*by_k[3][:5],*by_k[4][:5]]
    candidates={r['policy_id']:r for r in [*top20,*comparison['comparison_policies'],
                                           *mixed,*interaction['unique_policies']]}
    reuse=make_reuse_map(run_root,candidates.values(),split)
    plan={'protocol':'STATIC_BFS_PHASE2_DIAGNOSTIC_V1',
          'graph_source_sha256':file_sha(REPO/'scripts/static_bfs_graph/graph_builder.py'),
          'graph_search_sha256':file_sha(REPO/'scripts/static_bfs_graph/graph_search.py'),
          'pilot_frozen_sha256':PILOT_FROZEN_SHA,'edge_table_sha256':PILOT_EDGE_SHA,
          'original_prompt_split_sha256':PILOT_SPLIT_SHA,
          'phase2_prompt_split_sha256':file_sha(run_root/'phase2_prompt_split.json'),
          'all_feasible_ranking_sha256':file_sha(run_root/'all_feasible_policy_ranking.csv'),
          'comparison_plan_sha256':file_sha(run_root/'comparison_policy_plan.json'),
          'interaction_plan_sha256':file_sha(run_root/'interaction/interaction_plan.json'),
          'reuse_map_sha256':file_sha(run_root/'reuse_map.json'),
          'all_feasible_policy_count':3969,'feasible_count_by_K':{'2':189,'3':945,'4':2835},
          'primary_K':3,'primary_objective':'additive','top20_K3':top20,
          'comparison_policy_ids':[r['policy_id'] for r in comparison['comparison_policies']],
          'comparison_policies':comparison['comparison_policies'],
          'mixed_K_secondary_pool':mixed,'M_values':[1,2,5,10],
          'graph_fit_12':split['graph_fit_12'],
          'phase2_calibration':split['phase2_calibration'],
          'phase2_audit':split['phase2_audit'],
          'seeds':[42,43,44,45],
          'bootstrap_replicates':20000,'bootstrap_seed':BOOTSTRAP_SEED,
          'no_validation_test_N8':True}
    write_once(run_root/'shortlist_plan.json',plan)
    plan_sha=file_sha(run_root/'shortlist_plan.json')
    freeze_text(run_root/'shortlist_plan.sha256',plan_sha+'  shortlist_plan.json\n')
    import torch,torch_xla
    manifest={'protocol':'STATIC_BFS_PHASE2_DIAGNOSTIC_V1','branch':branch,
              'base_commit':commit,'host':'Hieu-tpu-v5-8-1','accelerator':'TPU v5p-8',
              'physical_tpu_chips_used':4,'torch':torch.__version__,
              'torch_xla':torch_xla.__version__,
              'model':'sd15','dtype':'bfloat16','N':4,'DDIM_steps':100,'eta':1.0,
              'verifier':'ImageReward','scoring':'max','resampling':'ssp',
              'candidate_steps':read(CONFIG)['pilot']['candidate_steps'],
              'taus':read(CONFIG)['pilot']['taus'],
              'rng_protocol':RNG_PROTOCOL,
              'edge_definition':'dense-reference selected terminal ImageReward minus single-edge intervention selected terminal ImageReward',
              'shortlist_plan_sha256':plan_sha,
              'baseline_config_sha256':file_sha(BASELINE_CONFIG),
              'baseline_source_sha256':file_sha(REPO/'scripts/diffusion_classical_search_t2i/tpu_runner.py'),
              'graph_fit_12_count':12,'phase2_calibration_count':24,'phase2_audit_count':24,
              'validation_touched':False,'test_touched':False,'N8_touched':False}
    write_once(run_root/'experiment_manifest.json',manifest)
    print('PHASE2_PREFLIGHT_PASS',plan_sha,'feasible',3969,'K3',945,
          'top20',len(top20),'comparison',20,'mixed',len(mixed),
          'interaction',len(interaction['unique_policies']),'reuse',reuse['n_records'],flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run-id',default='phase2-diagnostic-v1')
    args=ap.parse_args()
    preflight(REPO/'results/static-bfs-graph'/args.run_id)


if __name__=='__main__':main()
