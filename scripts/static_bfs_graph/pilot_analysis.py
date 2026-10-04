"""Frozen TRAIN bootstrap, policy checkpoint, and held-out validation analysis."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from hashlib import sha256
import json
from pathlib import Path
import statistics

import numpy as np

from .aggregate import corr, ranks
from .evaluate_paths import stage_records, write_csv
from .graph_builder import Edge, all_edges, all_paths, policy_from_path

REPO = Path(__file__).resolve().parents[2]


def read(path: Path):
    return json.loads(path.read_text())


def hash_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def write_once(path: Path, obj) -> None:
    data=json.dumps(obj,sort_keys=True,indent=2)+'\n'
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if path.read_text()!=data:raise RuntimeError(f'frozen output differs: {path}')
    else:path.write_text(data)


def context(root: Path):
    config=read(REPO/'configs/static_bfs_graph_t2i.json')
    split=read(root/'prompt_split.json')
    protocol=read(root/'analysis_protocol.json')
    tiny_split=REPO/'results/static-bfs-graph/tiny-v1/prompt_split.json'
    if not tiny_split.exists() or tiny_split.read_bytes()!=(root/'prompt_split.json').read_bytes():
        raise RuntimeError('pilot split must match frozen tiny split byte-for-byte')
    if protocol['candidate_steps']!=config['pilot']['candidate_steps'] or protocol['taus']!=config['pilot']['taus']:
        raise RuntimeError('pilot graph grid differs from preregistration')
    train=sorted(split['train_indices'])[:config['pilot']['train_prompts']]
    val=split['validation_indices']
    if len(train)!=12 or len(val)!=20 or set(train)&set(val):
        raise RuntimeError('invalid train/validation split')
    return config,split,protocol,train,val


def policy_catalog(root: Path):
    selected=read(root/'paths/pilot_selected.json')['evaluation_policies']
    by_id={}; methods=defaultdict(list)
    for entry in selected:
        pid=entry['policy_id']
        if pid not in by_id:by_id[pid]=entry
        methods[pid].append((entry['method'],entry['K']))
    return selected,by_id,methods


def edge_prompt_matrix(root: Path,config:dict,train:list[int]):
    grid=config['pilot'];seeds=config['tiny']['seeds']
    steps=tuple(grid['candidate_steps']);taus=tuple(map(float,grid['taus']))
    edges=all_edges(steps,taus)
    references=stage_records(root,'pilot_reference')
    interventions=stage_records(root,'pilot_edges')
    refid=next(iter({key[0] for key in references}))
    matrix=np.empty((len(train),len(edges)),dtype=np.float64)
    unit_deltas={}
    for ei,edge in enumerate(edges):
        for pi,p in enumerate(train):
            deltas=[]
            for seed in seeds:
                a=references.get((refid,p,seed));b=interventions.get((edge.key,p,seed))
                if a is None or b is None:raise RuntimeError(f'missing paired edge {edge.key},{p},{seed}')
                deltas.append(a['selected_final_score']-b['selected_final_score'])
            matrix[pi,ei]=statistics.mean(deltas)
            unit_deltas[(edge.key,p)]=deltas
    return edges,matrix,unit_deltas


def path_tables(steps,taus,edges,k):
    edge_index={e.key:i for i,e in enumerate(edges)}
    paths=list(all_paths(tuple(steps),tuple(taus),k))
    paths.sort(key=lambda p:policy_from_path(p,4).id)
    ids=[policy_from_path(p,4).id for p in paths]
    indices=np.asarray([[edge_index[e.key] for e in p] for p in paths],dtype=np.int32)
    event_steps=[tuple(e.dst for e in p if e.dst!=100) for p in paths]
    event_taus=[tuple(float(e.tau_dst) for e in p if e.dst!=100) for p in paths]
    return ids,indices,event_steps,event_taus


def select_from_values(values,indices,objective):
    scores=values[indices]
    if objective=='additive':return int(np.argmin(np.sum(scores,axis=1)))
    ordered=np.sort(scores,axis=1)[:,::-1]
    return int(np.lexsort(tuple(ordered[:,j] for j in reversed(range(ordered.shape[1]))))[0])


def bootstrap_graph(root:Path,config:dict,protocol:dict,train:list[int],edges,matrix):
    selected,_,_=policy_catalog(root)
    chosen={(e['method'],e['K']):e['policy_id'] for e in selected
            if e['method'] in ('additive','lexicographic')}
    samples=np.random.default_rng(protocol['graph_bootstrap_seed']).integers(
        0,len(train),size=(protocol['graph_bootstraps'],len(train)))
    frequencies=[];timestep_rows=[];tau_rows=[];summary={}
    for k in config['pilot']['k_values']:
        ids,indices,steps,taus=path_tables(config['pilot']['candidate_steps'],
                                           config['pilot']['taus'],edges,k)
        for objective in ('additive','lexicographic'):
            full_idx=select_from_values(np.mean(matrix,axis=0),indices,objective)
            if ids[full_idx]!=chosen[(objective,k)]:
                raise RuntimeError(f'bootstrap full-data search disagrees with frozen TRAIN graph {objective},K={k}')
            picked=[]
            for sample in samples:
                weights=np.bincount(sample,minlength=len(train))/len(train)
                values=weights@matrix
                picked.append(select_from_values(values,indices,objective))
            counts=Counter(picked)
            for index,n in counts.most_common():
                frequencies.append({'K':k,'objective':objective,'policy_id':ids[index],
                                    'resampling_steps':json.dumps(steps[index]),
                                    'temperatures':json.dumps(taus[index]),
                                    'count':n,'frequency':n/len(samples)})
            original_steps=set(steps[full_idx]);same_steps=0;same_tau_when_steps=0
            jaccards=[];step_counts=Counter();tau_counts=Counter()
            for index in picked:
                s=set(steps[index]);same_steps+=s==original_steps
                if s==original_steps and taus[index]==taus[full_idx]:same_tau_when_steps+=1
                jaccards.append(len(s&original_steps)/len(s|original_steps))
                step_counts.update(steps[index]);tau_counts.update(taus[index])
            for step in config['pilot']['candidate_steps']:
                timestep_rows.append({'K':k,'objective':objective,'sampling_idx':step,
                                      'selection_probability':step_counts[step]/len(samples)})
            for tau in config['pilot']['taus']:
                tau_rows.append({'K':k,'objective':objective,'tau':tau,
                                 'event_fraction':tau_counts[float(tau)]/(k*len(samples)),
                                 'path_inclusion_probability':sum(tau in taus[i] for i in picked)/len(samples)})
            sorted_counts=counts.most_common()
            weighted_jaccard=0.0
            for ia,na in counts.items():
                sa=set(steps[ia])
                for ib,nb in counts.items():
                    sb=set(steps[ib])
                    weighted_jaccard+=(na/len(samples))*(nb/len(samples))*len(sa&sb)/len(sa|sb)
            chosen_frequency=counts[full_idx]/len(samples)
            key=f'{objective}_K{k}'
            summary[key]={'selected_policy_id':ids[full_idx],
                          'selected_path_frequency':chosen_frequency,
                          'stability_label':'very stable' if chosen_frequency>=.8 else
                                            'moderately stable' if chosen_frequency>=.5 else 'unstable',
                          'modal_path_frequency':sorted_counts[0][1]/len(samples),
                          'top2_cumulative_frequency':sum(n for _,n in sorted_counts[:2])/len(samples),
                          'mean_jaccard_vs_full_data_steps':statistics.mean(jaccards),
                          'mean_pairwise_bootstrap_step_jaccard':weighted_jaccard,
                          'same_timestep_set_frequency':same_steps/len(samples),
                          'temperature_agreement_given_same_steps':
                              same_tau_when_steps/same_steps if same_steps else None,
                          'n_unique_bootstrap_paths':len(counts)}
    write_csv(root/'analysis/bootstrap_path_frequencies.csv',frequencies)
    write_csv(root/'analysis/bootstrap_timestep_frequencies.csv',timestep_rows)
    write_csv(root/'analysis/bootstrap_temperature_frequencies.csv',tau_rows)
    write_once(root/'analysis/bootstrap_summary.json',summary)
    return summary


def rank_diagnostics(rows,epsilon:float):
    pred=np.asarray([r['predicted_delta'] for r in rows],float)
    actual=np.asarray([r['actual_delta'] for r in rows],float)
    pairs=correct=0
    for i in range(len(rows)):
        for j in range(i+1,len(rows)):
            pd=pred[i]-pred[j];ad=actual[i]-actual[j]
            if abs(pd)<=epsilon or abs(ad)<=epsilon:continue
            pairs+=1;correct+=int(pd*ad>0)
    sign_rows=[r for r in rows if abs(r['predicted_delta'])>epsilon
               and abs(r['actual_delta'])>epsilon]
    return {'n_policies':len(rows),'pearson':corr(pred,actual),
            'spearman':corr(ranks(pred),ranks(actual)),
            'MAE':float(np.mean(np.abs(pred-actual))),
            'pairwise_accuracy':correct/pairs if pairs else None,
            'pairwise_n_eligible':pairs,'pairwise_n_correct':correct,
            'sign_agreement':float(sum(r['predicted_delta']*r['actual_delta']>0 for r in sign_rows)/len(sign_rows)) if sign_rows else None,
            'sign_n_eligible':len(sign_rows),'tie_epsilon':epsilon}


def diagnostic_composition(root:Path,protocol:dict):
    catalog=read(root/'paths/pilot_selected.json')['evaluation_policies']
    edge_values={r['edge_key']:float(r['mean_delta']) for r in csv.DictReader(
        (root/'edges/pilot_edge_values.csv').open())}
    path_scores={r['policy_id']:r for r in csv.DictReader((root/'evaluation/pilot_path_scores.csv').open())}
    seen=set();rows=[]
    for e in catalog:
        pid=e['policy_id']
        if e['method']=='manual_bfs' or pid in seen:continue
        seen.add(pid)
        count=sum(abs(edge_values[x['key']])>1e-12 for x in e['path'])
        score=path_scores[pid]
        rows.append({'policy_id':pid,'method':e['method'],'K':e['K'],
                     'nonzero_measured_edges':count,
                     'predicted_delta':float(score['predicted_delta']),
                     'actual_delta':float(score['actual_delta_vs_reference']),
                     'resampling_steps':score['resampling_steps'],
                     'temperatures':score['temperatures']})
    if len(rows)<30:raise RuntimeError(f'need >=30 audited unique static policies, got {len(rows)}')
    multi=[r for r in rows if r['nonzero_measured_edges']>=2]
    if len(multi)<10:raise RuntimeError('too few multi-edge policies for ranking audit')
    summary={'all':rank_diagnostics(rows,protocol['pairwise_tie_epsilon']),
             'multi_nonzero_edge':rank_diagnostics(multi,protocol['pairwise_tie_epsilon'])}
    write_csv(root/'analysis/compositionality_paths.csv',rows)
    write_once(root/'analysis/compositionality_summary.json',summary)
    return summary


def temperature_diagnostics(root:Path,config:dict,edges,unit_deltas,train:list[int]):
    edge_means={r['edge_key']:float(r['mean_delta']) for r in csv.DictReader(
        (root/'edges/pilot_edge_values.csv').open())}
    groups=defaultdict(dict)
    for e in edges:
        if e.dst!=100:groups[(e.src,e.dst)][float(e.tau_dst)]=e
    rows=[];best_count=Counter()
    for (src,dst),options in sorted(groups.items()):
        ordered=sorted(options,key=lambda tau:(edge_means[options[tau].key],tau))
        best,second=ordered[:2];best_count[best]+=1
        paired=[];prompt_means=[]
        for p in train:
            a=unit_deltas[(options[second].key,p)]
            b=unit_deltas[(options[best].key,p)]
            differences=[x-y for x,y in zip(a,b)]
            paired.extend(differences)
            prompt_means.append(statistics.mean(differences))
        means={tau:edge_means[options[tau].key] for tau in (2.,8.,32.)}
        rows.append({'src':src,'dst':dst,'mean_delta_tau2':means[2.],
                     'mean_delta_tau8':means[8.],'mean_delta_tau32':means[32.],
                     'best_tau':best,'second_tau':second,
                     'difference_best_vs_second':statistics.mean(paired),
                     'paired_sem':statistics.stdev(prompt_means)/len(prompt_means)**0.5,
                     'unit_level_sem':statistics.stdev(paired)/len(paired)**0.5,
                     'n_paired_units':len(paired),'n_prompts':len(prompt_means)})
    write_csv(root/'analysis/temperature_matched_edges.csv',rows)
    summary={'n_matched_source_destination_edges':len(rows),
             'best_tau_counts':{str(int(t)):best_count[t] for t in (2.,8.,32.)},
             'best_tau_fractions':{str(int(t)):best_count[t]/len(rows) for t in (2.,8.,32.)}}
    write_once(root/'analysis/temperature_summary.json',summary)
    return summary


def diagnostics(root:Path):
    config,split,protocol,train,val=context(root)
    if (root/'raw/validation').exists():raise RuntimeError('diagnostics must precede validation')
    if list((root/'failures').glob('**/*.json')) if (root/'failures').exists() else []:
        raise RuntimeError('failure artifacts exist')
    edges,matrix,unit_deltas=edge_prompt_matrix(root,config,train)
    if len(edges)!=92 or len(stage_records(root,'pilot_edges'))!=92*12*4:
        raise RuntimeError('incomplete TRAIN graph')
    composition=diagnostic_composition(root,protocol)
    bootstrap=bootstrap_graph(root,config,protocol,train,edges,matrix)
    temperature=temperature_diagnostics(root,config,edges,unit_deltas,train)
    write_once(root/'analysis/diagnostics_summary.json',{
        'train_prompt_indices':train,'validation_prompt_indices_unread':val,
        'edge_count':len(edges),'train_pair_count':len(train)*4,
        'compositionality':composition,'bootstrap':bootstrap,'temperature':temperature,
        'analysis_protocol_sha256':hash_file(root/'analysis_protocol.json'),
        'edge_table_sha256':hash_file(root/'edges/pilot_edge_values.csv'),
        'prompt_split_sha256':hash_file(root/'prompt_split.json')})
    print('PILOT_DIAGNOSTICS_COMPLETE',len(edges),composition['multi_nonzero_edge'])


def freeze(root:Path):
    config,split,protocol,train,val=context(root)
    if (root/'raw/validation').exists():raise RuntimeError('cannot freeze after validation began')
    diag=read(root/'analysis/diagnostics_summary.json')
    selected,catalog,methods=policy_catalog(root)
    train_scores={r['policy_id']:r for r in csv.DictReader((root/'evaluation/pilot_path_scores.csv').open())}
    if len(catalog)<31:raise RuntimeError('frozen catalog lacks complete-policy audit')
    policies=[]
    for pid,entry in catalog.items():
        if pid not in train_scores:raise RuntimeError(f'missing TRAIN path score {pid}')
        aliases=[{'method':method,'K':k} for method,k in methods[pid]]
        bootstrap={f'{method}_K{k}':diag['bootstrap'][f'{method}_K{k}']['selected_path_frequency']
                   for method,k in methods[pid] if method in ('additive','lexicographic')}
        policies.append({'policy_id':pid,'kind':'manual' if entry['method']=='manual_bfs' else 'static',
                         'source':aliases,'K':entry['K'],'policy':entry['policy'],
                         'manual_policy':config['manual_n4'] if entry['method']=='manual_bfs' else None,
                         'graph_predicted_cost':entry['predicted_delta'],
                         'train_actual_mean_reward':float(train_scores[pid]['actual_mean_reward']),
                         'train_delta_vs_manual':float(train_scores[pid]['actual_delta_vs_manual_bfs']),
                         'bootstrap_selected_path_frequency':bootstrap})
    code_paths=sorted((REPO/'scripts/static_bfs_graph').glob('*.py'))
    code_paths.append(REPO/'scripts/static_bfs_graph/remote_driver.sh')
    code_hashes={str(p.relative_to(REPO)):hash_file(p) for p in code_paths}
    payload={'protocol':'STATIC_BFS_FROZEN_POLICIES_V1','branch':'ngocminh',
             'base_git_commit':read(root/'experiment_manifest.json')['git_commit'],
             'created_before_validation':True,'N':4,'execution_dtype':'bfloat16',
             'model':'sd15','scoring':'max','resampling':'ssp','DDIM_steps':100,'eta':1.0,
             'train_prompt_indices':train,'validation_prompt_indices':val,'seeds':protocol['seeds'],
             'analysis_protocol_sha256':hash_file(root/'analysis_protocol.json'),
             'config_sha256':hash_file(REPO/'configs/static_bfs_graph_t2i.json'),
             'prompt_split_sha256':hash_file(root/'prompt_split.json'),
             'edge_table_sha256':hash_file(root/'edges/pilot_edge_values.csv'),
             'code_sha256':code_hashes,'policies':policies}
    write_csv(root/'analysis/frozen_policy_table.csv',[
        {'policy_id':p['policy_id'],
         'method':'|'.join(x['method'] for x in p['source']),
         'K':p['K'],
         'resampling_steps':json.dumps(p['policy']['resampling_steps'] if p['policy'] else config['manual_n4']['resampling_steps']),
         'temperatures':json.dumps(p['policy']['temperature_by_step'] if p['policy'] else None),
         'predicted_graph_delta':p['graph_predicted_cost'],
         'train_mean_IR':p['train_actual_mean_reward'],
         'bootstrap_selected_path_frequency':json.dumps(p['bootstrap_selected_path_frequency'],sort_keys=True)}
        for p in policies])
    path=root/'frozen_policies.json';write_once(path,payload)
    digest=hash_file(path)
    hash_path=root/'frozen_policies.sha256'
    line=f'{digest}  frozen_policies.json\n'
    if hash_path.exists() and hash_path.read_text()!=line:raise RuntimeError('frozen hash differs')
    if not hash_path.exists():hash_path.write_text(line)
    if not any(any(x['method']=='manual_bfs' for x in p['source']) for p in policies):
        raise RuntimeError('manual BFS omitted from checkpoint')
    print('PILOT_FREEZE_COMPLETE',len(policies),digest)


def bootstrap_ci(prompt_deltas:np.ndarray,samples:np.ndarray):
    draws=prompt_deltas[samples].mean(axis=1)
    return np.quantile(draws,[0.025,0.975]).tolist()


def validation_aggregate(root:Path):
    config,split,protocol,train,val=context(root)
    frozen_path=root/'frozen_policies.json'
    frozen_hash=(root/'frozen_policies.sha256').read_text().split()[0]
    if hash_file(frozen_path)!=frozen_hash:raise RuntimeError('frozen policies hash changed')
    frozen=read(frozen_path)
    if frozen['prompt_split_sha256']!=hash_file(root/'prompt_split.json') or frozen['edge_table_sha256']!=hash_file(root/'edges/pilot_edge_values.csv'):
        raise RuntimeError('split or graph edge table changed after freeze')
    if frozen['config_sha256']!=hash_file(REPO/'configs/static_bfs_graph_t2i.json'):
        raise RuntimeError('experiment config changed after freeze')
    for relative,digest in frozen['code_sha256'].items():
        if hash_file(REPO/relative)!=digest:raise RuntimeError(f'code changed after freeze: {relative}')
    if frozen['train_prompt_indices']!=train or frozen['validation_prompt_indices']!=val:
        raise RuntimeError('frozen prompt assignment differs')
    failures=list((root/'failures').glob('**/*.json')) if (root/'failures').exists() else []
    if failures:raise RuntimeError(f'{len(failures)} hidden failed policies')
    records=stage_records(root,'validation')
    seeds=protocol['seeds'];policies=frozen['policies']
    expected={(x['policy_id'],p,s) for x in policies for p in val for s in seeds}
    if set(records)!=expected:raise RuntimeError(f'validation unit set mismatch missing={len(expected-set(records))} extra={len(set(records)-expected)}')
    manual_id='manual_bfs_g0p008_n4'
    if not any(x['policy_id']==manual_id for x in policies):raise RuntimeError('manual BFS absent')
    for key,row in records.items():
        if row['execution_dtype']!='torch.bfloat16' or row['diffusion_NFE']!=400:
            raise RuntimeError(f'backend/budget mismatch {key}')
        if row['selected_particle_index']!=int(np.argmax(row['final_particle_scores'])):
            raise RuntimeError(f'terminal selection mismatch {key}')
    sample_indices=np.random.default_rng(protocol['validation_bootstrap_seed']).integers(
        0,len(val),size=(protocol['validation_bootstraps'],len(val)))
    train_scores={r['policy_id']:r for r in csv.DictReader((root/'evaluation/pilot_path_scores.csv').open())}
    rows=[];gap_rows=[]
    manual_prompt=np.asarray([statistics.mean(records[(manual_id,p,s)]['selected_final_score'] for s in seeds)
                              for p in val])
    for policy in policies:
        pid=policy['policy_id']
        scores=[records[(pid,p,s)]['selected_final_score'] for p in val for s in seeds]
        prompt=np.asarray([statistics.mean(records[(pid,p,s)]['selected_final_score'] for s in seeds)
                           for p in val])
        deltas=prompt-manual_prompt
        lower,upper=bootstrap_ci(deltas,sample_indices)
        method='|'.join(f"{x['method']}:K{x['K']}" for x in policy['source'])
        row={'method':method,'policy_id':pid,'K':policy['K'],
             'resampling_steps':json.dumps(policy['policy']['resampling_steps'] if policy['policy'] else config['manual_n4']['resampling_steps']),
             'temperatures':json.dumps(policy['policy']['temperature_by_step'] if policy['policy'] else None),
             'mean_IR':statistics.mean(scores),'sample_std':statistics.stdev(scores),
             'paired_delta_vs_manual':float(np.mean(deltas)),
             'median_prompt_paired_delta':float(np.median(deltas)),
             'ci95_lower':lower,'ci95_upper':upper,
             'prompt_win_fraction':float(np.mean(deltas>0)),
             'prompt_tie_fraction':float(np.mean(deltas==0)),
             'n_validation_prompts':len(val),'n_prompt_seed_units':len(scores),
             'mean_elapsed_seconds':statistics.mean(records[(pid,p,s)]['elapsed_seconds'] for p in val for s in seeds),
             'verifier_calls':records[(pid,val[0],seeds[0])]['verifier_calls'],
             'diffusion_NFE':400}
        rows.append(row)
        tr=train_scores[pid]
        gap_rows.append({'policy_id':pid,'method':method,'K':policy['K'],
                         'train_mean_IR':float(tr['actual_mean_reward']),
                         'validation_mean_IR':row['mean_IR'],
                         'train_delta_vs_manual':float(tr['actual_delta_vs_manual_bfs']),
                         'validation_delta_vs_manual':row['paired_delta_vs_manual'],
                         'generalization_gap':float(tr['actual_delta_vs_manual_bfs'])-row['paired_delta_vs_manual']})
    write_csv(root/'evaluation/validation_results.csv',rows)
    write_csv(root/'evaluation/train_validation_gap.csv',gap_rows)
    corr_rank=corr(ranks([r['train_mean_IR'] for r in gap_rows]),
                   ranks([r['validation_mean_IR'] for r in gap_rows]))
    selected=read(root/'paths/pilot_selected.json')['selected']
    by_id={r['policy_id']:r for r in rows}
    graph={f"{e['method']}_K{e['K']}":by_id[e['policy_id']] for e in selected}
    diag=read(root/'analysis/diagnostics_summary.json')
    summary={'status':'VALIDATION_COMPLETE','branch':'ngocminh',
             'git_commit':frozen['base_git_commit'],'tpu_devices':4,
             'train_prompts':len(train),'validation_prompts':len(val),
             'n_graph_edges':diag['edge_count'],'n_audited_static_policies':diag['compositionality']['all']['n_policies'],
             'n_frozen_unique_policies':len(policies),'failed_policies':0,
             'manual_validation_mean_IR':by_id[manual_id]['mean_IR'],
             'graph_validation':graph,
             'multi_edge_ranking':diag['compositionality']['multi_nonzero_edge'],
             'bootstrap':diag['bootstrap'],'temperature':diag['temperature'],
             'train_validation_policy_rank_spearman':corr_rank,
             'frozen_policies_sha256':frozen_hash,
             'prompt_split_sha256':hash_file(root/'prompt_split.json'),
             'edge_table_sha256':hash_file(root/'edges/pilot_edge_values.csv'),
             'config_sha256':hash_file(REPO/'configs/static_bfs_graph_t2i.json'),
             'validation_bootstraps':protocol['validation_bootstraps'],
             'validation_bootstrap_seed':protocol['validation_bootstrap_seed'],
             'test_used':False,'N8_used':False}
    write_once(root/'evaluation/validation_summary.json',summary)
    write_report(root,summary,rows,gap_rows,selected)
    print('FINAL_VALIDATION_AGGREGATE_COMPLETE',len(policies),len(val),frozen_hash)


def write_report(root:Path,summary:dict,rows:list[dict],gaps:list[dict],selected:list[dict]):
    graph=summary['graph_validation'];manual=summary['manual_validation_mean_IR']
    ranking=summary['multi_edge_ranking'];bootstrap=summary['bootstrap']
    additive3=graph['additive_K3'];additive2=graph['additive_K2']
    if additive3['ci95_upper']<0:
        n8='No: the matched K=3 graph policy is worse than manual BFS on held-out VALIDATION.'
    elif ranking['spearman'] is not None and ranking['spearman']<0.3:
        n8='No: multi-edge ranking is too weak to justify scaling the same approximation.'
    elif bootstrap['additive_K3']['selected_path_frequency']<0.5:
        n8='No direct N=8 extension yet: the matched K=3 TRAIN-selected path is unstable under prompt bootstrap, even if its held-out mean is competitive.'
    elif additive3['ci95_lower']>0:
        n8='Yes, as a separate preregistered follow-up: matched K=3 improves on manual BFS.'
    else:
        n8='Not yet established: the matched K=3 uncertainty interval includes zero.'
    lines=['# Static BFS graph pilot — TRAIN graph, held-out VALIDATION','',
           'The 92-edge graph was estimated on 12 TRAIN prompts × four seeds. All policies were frozen and hashed before evaluating the 20 VALIDATION prompts × four seeds. TEST and N=8 were not run.',
           f"Branch `ngocminh`, base commit `{summary['git_commit']}`; SD1.5 BF16, 100-step DDIM eta=1, Max+SSP, ImageReward. Frozen policy SHA256: `{summary['frozen_policies_sha256']}`.",
           '',
           '## Decision gates','',
           f"1. **Ranking:** among multi-nonzero-edge complete policies, Spearman {ranking['spearman']:.3f}; pairwise ranking accuracy {ranking['pairwise_accuracy']:.3f} on {ranking['pairwise_n_eligible']} non-tied pairs (epsilon {ranking['tie_epsilon']}).",
           '2. **Path stability:** full-data selected-path bootstrap frequencies appear in the table below. Bootstrap resamples whole TRAIN prompts with their four seeds; no diffusion was rerun.',
           f"3. **Matched K=3:** additive graph minus manual BFS {additive3['paired_delta_vs_manual']:+.4f} ImageReward, prompt-bootstrap 95% CI [{additive3['ci95_lower']:+.4f}, {additive3['ci95_upper']:+.4f}].",
           f"4. **Lower-compute K=2:** additive graph minus manual K=3 {additive2['paired_delta_vs_manual']:+.4f}, 95% CI [{additive2['ci95_lower']:+.4f}, {additive2['ci95_upper']:+.4f}]. Diffusion NFE remains 400 for both; K=2 saves one verifier/resampling event, not diffusion steps.",
           '5. **Temperature identifiability:** see the matched source/destination tau table and bootstrap event shares below; tau=8 is the dense-reference setting, so frequent selection alone does not establish a learned increasing schedule.',
           '', '## Frozen graph policies and held-out results','',
           f"Manual BFS validation mean ImageReward: **{manual:.6f}**.",
           '', '| Objective | K | Steps | Tau | Predicted regret | Bootstrap path frequency | TRAIN mean IR | VALIDATION mean IR | Delta vs manual [95% CI] |',
           '|---|---:|---|---|---:|---:|---:|---:|---|']
    gap_by_id={r['policy_id']:r for r in gaps}
    for e in selected:
        key=f"{e['method']}_K{e['K']}";r=graph[key];g=gap_by_id[e['policy_id']]
        b=bootstrap[key]['selected_path_frequency']
        lines.append(f"| {e['method']} | {e['K']} | `{e['policy']['resampling_steps']}` | `{e['policy']['temperature_by_step']}` | {e['predicted_delta']:+.4f} | {b:.1%} | {g['train_mean_IR']:.4f} | {r['mean_IR']:.4f} | {r['paired_delta_vs_manual']:+.4f} [{r['ci95_lower']:+.4f}, {r['ci95_upper']:+.4f}] |")
    temp_rows=list(csv.DictReader((root/'analysis/bootstrap_temperature_frequencies.csv').open()))
    lines.extend(['',f"Across all {summary['n_frozen_unique_policies']} unique frozen policies, TRAIN/VALIDATION reward-rank Spearman is {summary['train_validation_policy_rank_spearman']:.3f}.",
                  'Generalization gaps (TRAIN paired delta minus VALIDATION paired delta) are in `evaluation/train_validation_gap.csv`; the manual baseline has zero by definition.',
                  '', 'TRAIN bootstrap temperature event shares:', '',
                  '| Objective | K | tau=2 | tau=8 | tau=32 |',
                  '|---|---:|---:|---:|---:|'])
    for objective in ('additive','lexicographic'):
        for k in (2,3,4):
            found={float(r['tau']):float(r['event_fraction']) for r in temp_rows
                   if r['objective']==objective and int(r['K'])==k}
            lines.append(f"| {objective} | {k} | {found[2.]:.1%} | {found[8.]:.1%} | {found[32.]:.1%} |")
    lines.extend([
                  '', '## Scientific answers','',
                  f"**Q1 — Edge ranking at pilot scale.** Multi-edge Spearman {ranking['spearman']:.3f}, pairwise accuracy {ranking['pairwise_accuracy']:.3f}; calibration error and sign agreement are in `analysis/compositionality_summary.json`.",
                  f"**Q2 — Path stability.** Selected-path frequencies range from {min(bootstrap[k]['selected_path_frequency'] for k in bootstrap):.1%} to {max(bootstrap[k]['selected_path_frequency'] for k in bootstrap):.1%} across six graph searches. Full path, timestep and tau marginals are saved separately; low frequency means the 12 TRAIN prompts do not identify a unique policy.",
                  f"**Q3 — Held-out generalization.** The matched additive K=3 delta is {additive3['paired_delta_vs_manual']:+.4f} with 95% CI [{additive3['ci95_lower']:+.4f}, {additive3['ci95_upper']:+.4f}]. TRAIN-to-VALIDATION gaps for every frozen policy are reported without retuning.",
                  f"**Q4 — K=2.** Its delta versus manual K=3 is {additive2['paired_delta_vs_manual']:+.4f} with 95% CI [{additive2['ci95_lower']:+.4f}, {additive2['ci95_upper']:+.4f}]. Reward by K is not assumed monotonic; see the table.",
                  '**Q5 — Timing versus temperature.** Read `analysis/temperature_matched_edges.csv` for paired source/destination comparisons, `analysis/temperature_summary.json` for best-tau fractions, and `analysis/bootstrap_temperature_frequencies.csv` for selected event shares. Flat tau=8 or ambiguous paired gaps would support timing dominance rather than temperature learning.',
                  f'**Q6 — N=8 follow-up.** {n8} This run stops after VALIDATION.',
                  '', '## Integrity and artifacts','',
                  f"Frozen split SHA256 `{summary['prompt_split_sha256']}`; TRAIN edge table SHA256 `{summary['edge_table_sha256']}`; config SHA256 `{summary['config_sha256']}`.",
                  'The graph used TRAIN records only. VALIDATION produced no edge values and did not change the frozen policy file. Every validation unit used N=4, BF16, Max+SSP, DDIM100 eta=1 and terminal lossless ImageReward. Hidden failed policies: 0.',
                  'Machine-readable plot data: `evaluation/validation_results.csv`, `analysis/compositionality_paths.csv`, `analysis/bootstrap_timestep_frequencies.csv`, `analysis/bootstrap_temperature_frequencies.csv`, and `evaluation/train_validation_gap.csv`. Plot script: `scripts/static_bfs_graph/plots.py`.',
                  'The 20 VALIDATION prompts are held out from graph fitting, but model choice among reported K/objectives on this split would require a further untouched TEST run before a final performance claim.',
                  ''])
    (root/'report.md').write_text('\n'.join(lines))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=('pilot_diagnostics','pilot_freeze','final_aggregate'))
    parser.add_argument('--run-id',default='pilot-v1')
    args=parser.parse_args()
    root=REPO/'results/static-bfs-graph'/args.run_id
    if args.stage=='pilot_diagnostics':diagnostics(root)
    elif args.stage=='pilot_freeze':freeze(root)
    else:validation_aggregate(root)


if __name__=='__main__':main()
