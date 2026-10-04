"""Tiny graph diagnostics, paired policy metrics, and scientific stop gate."""
from __future__ import annotations

from pathlib import Path
import csv
import json
import math
import statistics

import numpy as np

from .evaluate_paths import stage_records, write_csv, write
from .graph_builder import dense_reference


def ranks(values):
    x=np.asarray(values,dtype=float);order=np.argsort(x,kind='stable');rank=np.empty(len(x),float)
    i=0
    while i<len(x):
        j=i+1
        while j<len(x) and x[order[j]]==x[order[i]]:j+=1
        rank[order[i:j]]=(i+j-1)/2+1
        i=j
    return rank


def corr(a,b):
    x=np.asarray(a,float);y=np.asarray(b,float)
    if len(x)<3 or np.std(x)==0 or np.std(y)==0:return float('nan')
    return float(np.corrcoef(x,y)[0,1])


def aggregate_tiny(run_root:Path,cfg:dict,split:dict):
    tiny=cfg['tiny'];pairs=sorted((p,s) for p in split['tiny_train_indices'] for s in tiny['seeds'])
    chosen=json.loads((run_root/'paths/tiny_selected.json').read_text())['evaluation_policies']
    ref=stage_records(run_root,'tiny_reference');evaluation=stage_records(run_root,'tiny_validate')
    reference=dense_reference(tuple(tiny['candidate_steps']),float(tiny['reference_tau']),4)
    failure_paths=list((run_root/'failures').glob('**/*.json')) if (run_root/'failures').exists() else []
    if failure_paths:raise RuntimeError(f'{len(failure_paths)} policy failure artifacts exist')
    manual_id='manual_bfs_g0p008_n4'
    for p,s in pairs:
        if (reference.id,p,s) not in ref or (manual_id,p,s) not in evaluation:
            raise RuntimeError(f'missing reference/manual pair {p},{s}')
    rows=[];composition=[];seen=set()
    for item in chosen:
        pid=item['policy_id']
        actual=[];reference_scores=[];manual_scores=[];runtimes=[];calls=[];nfes=[]
        for p,s in pairs:
            record=evaluation.get((pid,p,s))
            if record is None:raise RuntimeError(f'missing evaluated path {pid},{p},{s}')
            actual.append(record['selected_final_score'])
            reference_scores.append(ref[(reference.id,p,s)]['selected_final_score'])
            manual_scores.append(evaluation[(manual_id,p,s)]['selected_final_score'])
            runtimes.append(record['elapsed_seconds']);calls.append(record['verifier_calls'])
            nfes.append(record['diffusion_NFE'])
        steps=item['policy']['resampling_steps'] if item.get('policy') else cfg['manual_n4']['resampling_steps']
        temps=item['policy']['temperature_by_step'] if item.get('policy') else [
            [s,10*((1+cfg['manual_n4']['gamma'])**s-1)] for s in steps]
        actual_delta=statistics.mean(r-a for r,a in zip(reference_scores,actual))
        row={'method':item['method'],'N':4,'K':item['K'],'policy_id':pid,
             'resampling_steps':json.dumps(steps),'temperatures':json.dumps(temps),
             'predicted_delta':item['predicted_delta'],
             'actual_mean_reward':statistics.mean(actual),
             'sample_std':statistics.stdev(actual),
             'actual_delta_vs_reference':actual_delta,
             'actual_delta_vs_manual_bfs':statistics.mean(a-m for a,m in zip(actual,manual_scores)),
             'mean_wall_seconds':statistics.mean(runtimes),
             'verifier_calls':calls[0],'diffusion_NFE':nfes[0],
             'n_paired_prompt_seed_units':len(pairs)}
        rows.append(row)
        if item['method']!='manual_bfs' and pid not in seen:
            seen.add(pid)
            composition.append({'policy_id':pid,'method':item['method'],'K':item['K'],
                      'predicted_delta':item['predicted_delta'],'actual_delta':actual_delta,
                      'absolute_error':abs(item['predicted_delta']-actual_delta),
                      'resampling_steps':json.dumps(steps),'temperature_schedule':json.dumps(temps)})
    write_csv(run_root/'evaluation/path_scores.csv',rows)
    write_csv(run_root/'evaluation/compositionality.csv',composition)
    pred=[x['predicted_delta'] for x in composition]
    actual=[x['actual_delta'] for x in composition]
    pearson=corr(pred,actual);spearman=corr(ranks(pred),ranks(actual))
    by={(x['method'],x['K']):x for x in rows}
    reference_mean=statistics.mean(ref[(reference.id,p,s)]['selected_final_score'] for p,s in pairs)
    result={'status':'TINY_COMPLETE','train_prompts':len(split['tiny_train_indices']),
            'seeds':tiny['seeds'],'n_paired_units':len(pairs),'candidate_steps':tiny['candidate_steps'],
            'taus':tiny['taus'],'edge_count':len(list(csv.DictReader((run_root/'edges/edge_values.csv').open()))),
            'dense_reference_mean_reward':reference_mean,
            'manual_bfs_mean_reward':by[('manual_bfs',3)]['actual_mean_reward'],
            'additive':{str(k):by[('additive',k)] for k in tiny['k_values']},
            'lexicographic':{str(k):by[('lexicographic',k)] for k in tiny['k_values']},
            'compositionality':{'n_unique_static_policies':len(composition),
                                'pearson':pearson,'spearman':spearman,
                                'MAE':statistics.mean(x['absolute_error'] for x in composition)},
            'failed_policies':len(failure_paths),'rng_protocol':cfg['rng_protocol'],
            'pilot_launched':False}
    write(run_root/'evaluation/summary.json',result)
    lines=['# Tiny static BFS graph — SD1.5 TPU v5p-8','',
           f"Status: {result['status']}; **tiny TRAIN data only** (2 prompts × 4 seeds).",
           f"Branch: `ngocminh`; model BF16, DDIM100 eta=1; Max + SSP; `{cfg['rng_protocol']}`.",
           f"Graph: 4 candidate indices × 3 explicit temperatures; {result['edge_count']} evaluated edges; K=2 and K=3.",
           f"Dense reference mean final ImageReward: **{reference_mean:.6f}**. Manual BFS same-unit mean: **{result['manual_bfs_mean_reward']:.6f}**.",
           '', '| K | Objective | Events | Temperatures | Predicted delta | Actual mean IR |',
           '|---:|---|---|---|---:|---:|']
    for k in tiny['k_values']:
        for method in ('additive','lexicographic'):
            row=by[(method,k)]
            lines.append(f"| {k} | {method} | `{row['resampling_steps']}` | `{row['temperatures']}` | {row['predicted_delta']:+.6f} | {row['actual_mean_reward']:.6f} |")
    lines.extend(['',f"Predicted-versus-actual complete-path delta: Pearson **{pearson:.3f}**, Spearman **{spearman:.3f}**, MAE **{result['compositionality']['MAE']:.4f}** across {len(composition)} unique static policies.",
                  'All edge targets and path scores use selected **terminal** ImageReward from lossless in-memory PIL, never intermediate verifier values or JPEG scores.',
                  'This tiny comparison is a structural/scientific smoke test on TRAIN prompts, not a held-out performance claim. No pilot, validation, test, or N=8 generation was launched.',
                  '', 'See `edges/edge_values.csv`, `evaluation/path_scores.csv`, `evaluation/compositionality.csv`, `evaluation/summary.json`, and per-unit raw records.'])
    (run_root/'report.md').write_text('\n'.join(lines)+'\n')
    print('TINY_AGGREGATE_COMPLETE',result['compositionality'],flush=True)


def aggregate_pilot(run_root:Path,cfg:dict,split:dict):
    """TRAIN-only pilot diagnostics; does not inspect validation or test."""
    grid=cfg['pilot']
    pairs=sorted((p,s) for p in sorted(split['train_indices'])[:grid['train_prompts']]
                 for s in grid.get('seeds',cfg['tiny']['seeds']))
    selected=json.loads((run_root/'paths/pilot_selected.json').read_text())['evaluation_policies']
    reference=dense_reference(tuple(grid['candidate_steps']),float(grid['reference_tau']),4)
    ref=stage_records(run_root,'pilot_reference')
    evaluation=stage_records(run_root,'pilot_validate')
    failures=list((run_root/'failures').glob('**/*.json')) if (run_root/'failures').exists() else []
    if failures:raise RuntimeError(f'{len(failures)} failure artifacts exist')
    manual_id='manual_bfs_g0p008_n4'
    rows=[];composition=[];seen=set()
    for item in selected:
        pid=item['policy_id'];scores=[];references=[];manuals=[];times=[]
        for p,s in pairs:
            if (pid,p,s) not in evaluation or (reference.id,p,s) not in ref or (manual_id,p,s) not in evaluation:
                raise RuntimeError(f'missing paired pilot record {pid},{p},{s}')
            scores.append(evaluation[(pid,p,s)]['selected_final_score'])
            references.append(ref[(reference.id,p,s)]['selected_final_score'])
            manuals.append(evaluation[(manual_id,p,s)]['selected_final_score'])
            times.append(evaluation[(pid,p,s)]['elapsed_seconds'])
        policy=item.get('policy')
        steps=policy['resampling_steps'] if policy else cfg['manual_n4']['resampling_steps']
        temps=policy['temperature_by_step'] if policy else [
            [step,10*((1+cfg['manual_n4']['gamma'])**step-1)] for step in steps]
        actual_delta=statistics.mean(a-b for a,b in zip(references,scores))
        row={'method':item['method'],'N':4,'K':item['K'],'policy_id':pid,
             'resampling_steps':json.dumps(steps),'temperatures':json.dumps(temps),
             'predicted_delta':item['predicted_delta'],'actual_mean_reward':statistics.mean(scores),
             'sample_std':statistics.stdev(scores),'actual_delta_vs_reference':actual_delta,
             'actual_delta_vs_manual_bfs':statistics.mean(a-b for a,b in zip(scores,manuals)),
             'mean_wall_seconds':statistics.mean(times),'verifier_calls':4*(len(steps)+1),
             'diffusion_NFE':400,'n_paired_prompt_seed_units':len(pairs)}
        rows.append(row)
        if item['method']!='manual_bfs' and pid not in seen:
            seen.add(pid)
            composition.append({'policy_id':pid,'method':item['method'],'K':item['K'],
                                'predicted_delta':item['predicted_delta'],'actual_delta':actual_delta,
                                'absolute_error':abs(item['predicted_delta']-actual_delta),
                                'resampling_steps':json.dumps(steps),
                                'temperature_schedule':json.dumps(temps)})
    write_csv(run_root/'evaluation/pilot_path_scores.csv',rows)
    write_csv(run_root/'evaluation/pilot_compositionality.csv',composition)
    pred=[r['predicted_delta'] for r in composition]
    actual=[r['actual_delta'] for r in composition]
    result={'status':'PILOT_TRAIN_COMPLETE','train_prompts':grid['train_prompts'],
            'seeds':grid.get('seeds',cfg['tiny']['seeds']),'n_paired_units':len(pairs),
            'edge_count':len(list(csv.DictReader((run_root/'edges/pilot_edge_values.csv').open()))),
            'reference_mean_reward':statistics.mean(ref[(reference.id,p,s)]['selected_final_score'] for p,s in pairs),
            'manual_mean_reward':next(r['actual_mean_reward'] for r in rows if r['method']=='manual_bfs'),
            'path_scores':rows,'compositionality':{
                'n_unique_static_policies':len(composition),'pearson':corr(pred,actual),
                'spearman':corr(ranks(pred),ranks(actual)),
                'MAE':statistics.mean(r['absolute_error'] for r in composition)},
            'failed_policies':0,'validation_or_test_used':False,'rng_protocol':cfg['rng_protocol']}
    (run_root/'evaluation/pilot_summary.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    lines=['# Static BFS pilot — TRAIN-only diagnostic','',
           f"{grid['train_prompts']} TRAIN prompts × {len(grid.get('seeds',cfg['tiny']['seeds']))} seeds; no validation/test scores used.",
           f"Edges: {result['edge_count']}; reference mean IR: {result['reference_mean_reward']:.6f}; manual BFS mean IR: {result['manual_mean_reward']:.6f}.",
           f"Predicted/actual path Spearman: {result['compositionality']['spearman']:.3f}; MAE: {result['compositionality']['MAE']:.4f}.",
           '', '| Method | K | Events | Temperatures | Actual mean IR | Delta vs manual |',
           '|---|---:|---|---|---:|---:|']
    for row in rows:
        lines.append(f"| {row['method']} | {row['K']} | `{row['resampling_steps']}` | `{row['temperatures']}` | {row['actual_mean_reward']:.6f} | {row['actual_delta_vs_manual_bfs']:+.6f} |")
    lines.append('')
    lines.append('This is a TRAIN-only pilot diagnostic, not a held-out comparison. Validation/test require a separately frozen selection protocol.')
    (run_root/'pilot_report.md').write_text('\n'.join(lines)+'\n')
    print('PILOT_TRAIN_AGGREGATE_COMPLETE',result['compositionality'],flush=True)
