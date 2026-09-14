"""Pre-registered paired prompt bootstrap; no gamma/z-score/model selection."""
import csv
import json
from pathlib import Path

import numpy as np

from validation import check, save, sha, validate_pair


def write_csv(path,rows):
    with Path(path).open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def paired_statistics(bfs,vasr,seed=20260905,resamples=20000):
    bfs,vasr=np.asarray(bfs,dtype=np.float64),np.asarray(vasr,dtype=np.float64)
    check(bfs.shape==vasr.shape and bfs.ndim==2 and bfs.shape[1]==4,'expected prompt x four matched seeds')
    check(np.isfinite(bfs).all() and np.isfinite(vasr).all(),'nonfinite analysis values')
    deltas=(vasr-bfs).mean(axis=1)
    rng=np.random.default_rng(seed)
    boot=[]
    for start in range(0,resamples,500):
        sample=deltas[rng.integers(0,len(deltas),size=(min(500,resamples-start),len(deltas)))]
        boot.extend(sample.mean(axis=1).tolist())
    result={'mean_delta':float(deltas.mean()),'median_delta':float(np.median(deltas)),
            'win_fraction':float(np.mean(deltas>0)), 'tie_fraction':float(np.mean(deltas==0)),
            'bootstrap_ci95':np.quantile(boot,[.025,.975]).tolist(),
            'bootstrap_samples':resamples,'analysis_seed':seed,
            'bootstrap_unit':'prompt after averaging four matched-seed differences',
            'prompt_deltas':deltas.tolist()}
    for method,values in [('bfs',bfs),('vasr',vasr)]:
        means=values.mean(axis=0)
        result[method]={'mean':float(means.mean()),'std':float(means.std(ddof=1)),
                        'seed_means':means.tolist(),'std_ddof':1}
    return result


def analyze_n(output,protocol,k):
    output=Path(output)
    prereg=json.loads((output/'preregistration.json').read_text())
    records=[]
    curves={m:np.empty((100,4)) for m in ['bfs','vasr']}
    artifact_hashes={}
    for method in curves:
        expected={output/'raw'/f'n{k}'/method/f'seed{seed}'/f'prompt_{i:03d}.json'
                  for seed in protocol['seeds'] for i in range(100)}
        actual=set((output/'raw'/f'n{k}'/method).glob('seed*/prompt_*.json'))
        check(actual==expected,f'missing/extra result tuples for {method}/N{k}')
    for i in range(100):
        for j,seed in enumerate(protocol['seeds']):
            pair={}
            for method in curves:
                path=output/'raw'/f'n{k}'/method/f'seed{seed}'/f'prompt_{i:03d}.json'
                r=json.loads(path.read_text())
                check((r['particles'],r['trial_seed'],r['prompt_index'],r['method'])==(k,seed,i,method),'record key mismatch')
                prompt=prereg['prompts'][i].get('prompt',prereg['prompts'][i].get('text'))
                check(r['prompt']==prompt,'prompt text mismatch')
                check(r['protocol_sha256']==prereg['protocol_sha256'] and r['code_hashes']==prereg['code_hashes'],'provenance changed')
                check(sha(output/r['selected_image_path'])==r['selected_image_sha256'],'missing/changed selected artifact')
                artifact_hashes[str(path.relative_to(output))]=sha(path)
                curves[method][i,j]=r['selected_max_reward']
                records.append(r)
                pair[method]=r
            validate_pair(pair['bfs'],pair['vasr'],protocol)
    a=protocol['analysis']
    check(a['bootstrap_samples']==20000 and a['seed']==20260905 and a['seed_std_ddof']==1,'analysis policy changed')
    result=paired_statistics(curves['bfs'],curves['vasr'],a['seed'],a['bootstrap_samples'])
    result.update({'n':k,'prompts':100,'seeds':protocol['seeds'],'method_records':800,
                   'nfe_per_prompt_seed':100*k,'verifier_calls_per_prompt_seed':4*k,
                   'scientifically_valid':True,'retuning':False})
    mechanism=[]
    keys=['reward_mean','reward_std','reward_max','unique_parents','distinct_original_lineages',
          'offspring_count_variance','max_offspring_count','selection_entropy','selection_ess']
    for step in protocol['resampling_steps']:
        for method in ['bfs','vasr']:
            events=[e for r in records if r['method']==method for e in r['resampling_events'] if e['sampling_index']==step]
            check(len(events)==400,'mechanism grid incomplete')
            mechanism.append({'n':k,'method':method,'sampling_index':step,'observations':400,
                              **{key:float(np.mean([e[key] for e in events])) for key in keys}})
    result['mechanism']=mechanism
    directory=output/'analysis'/f'n{k}'
    directory.mkdir(parents=True,exist_ok=True)
    write_csv(directory/'paired_prompts.csv',[{'prompt_index':i,'bfs_mean_over_seeds':float(curves['bfs'][i].mean()),
                                              'vasr_mean_over_seeds':float(curves['vasr'][i].mean()),
                                              'paired_delta':result['prompt_deltas'][i]} for i in range(100)])
    write_csv(directory/'mechanism.csv',mechanism)
    write_csv(directory/'seed_means.csv',[{'method':m,'seed':s,'mean_max_imagereward':result[m]['seed_means'][j]}
                                        for m in ['bfs','vasr'] for j,s in enumerate(protocol['seeds'])])
    save(directory/'artifact_hashes.json',artifact_hashes,immutable=True)
    save(directory/'summary.json',result,immutable=True)
    return result


def combined_report(output,protocol):
    output=Path(output)
    summaries=[json.loads((output/'analysis'/f'n{k}'/'summary.json').read_text()) for k in [4,8]]
    rows=[]
    text=['# Soft VASR transferred to Classical Search\n\n',
          'This is NOT an official VASR text-to-image reproduction. '
          'Soft VASR is transferred to the Classical Search ImageReward benchmark under a matched resource protocol. '
          'The VASR paper uses VASR-MAX for its main Stable Diffusion experiments; a negative result here does not contradict that native result. '
          '[VASR v2](https://arxiv.org/html/2604.06779v2).\n\n',
          '| N | Method | Mean ImageReward | Std | Paired delta vs BFS | 95% CI | Win rate | NFE | Verifier calls |\n',
          '|---:|---|---:|---:|---:|---|---:|---:|---:|\n']
    for s in summaries:
        for method in ['bfs','vasr']:
            is_vasr=method=='vasr'
            ci=s['bootstrap_ci95'] if is_vasr else [0.,0.]
            row={'N':s['n'],'Method':method.upper(),'Mean ImageReward':s[method]['mean'],'Std':s[method]['std'],
                 'Paired delta vs BFS':s['mean_delta'] if is_vasr else 0.,'CI lower':ci[0],'CI upper':ci[1],
                 'Win rate':s['win_fraction'] if is_vasr else None,'NFE':s['nfe_per_prompt_seed'],
                 'Verifier calls':s['verifier_calls_per_prompt_seed']}
            rows.append(row)
            ci_text=f"[{ci[0]:+.6f}, {ci[1]:+.6f}]" if is_vasr else 'reference'
            win=f"{s['win_fraction']:.1%}" if is_vasr else '-'
            text.append(f"| {s['n']} | {method.upper()} | {s[method]['mean']:.6f} | {s[method]['std']:.6f} | "
                        f"{row['Paired delta vs BFS']:+.6f} | {ci_text} | {win} | {row['NFE']} | {row['Verifier calls']} |\n")
    text.extend(['\nStd is sample standard deviation (ddof=1) of four seed-wise means over 100 prompts. '
                 'NFE/verifier counts are per prompt/seed. CFG raw sample-forwards are twice the reported guided NFE. '
                 'CI uses 20,000 paired prompt bootstrap draws, seed 20260905, after averaging four matched-seed differences.\n\n',
                 'Sanity references only, not tuning targets: Classical Search paper BFS N=4: 0.882 +/- 0.029; N=8: 1.087 +/- 0.031.\n\n',
                 '## Paired and mechanism interpretation\n\n'])
    for s in summaries:
        ci=s['bootstrap_ci95']
        direction='positive' if s['mean_delta']>0 else 'negative or zero'
        supported='excludes zero' if ci[0]>0 or ci[1]<0 else 'includes zero'
        text.append(f"N={s['n']}: paired mean {s['mean_delta']:+.6f}, median {s['median_delta']:+.6f}; "
                    f"direction {direction}, 95% interval {supported}.\n\n")
        for key,label in [('distinct_original_lineages','retained original lineages'),('offspring_count_variance','realized cross-parent offspring-count variance')]:
            means={m:float(np.mean([e[key] for e in s['mechanism'] if e['method']==m])) for m in ['bfs','vasr']}
            text.append(f"N={s['n']} average {label} across the three events: BFS {means['bfs']:.6f}, VASR {means['vasr']:.6f}.\n\n")
    consistent=(summaries[0]['mean_delta']>0)==(summaries[1]['mean_delta']>0)
    text.append(f"The direction of the mean effect is {'consistent' if consistent else 'not consistent'} across N=4 and N=8; inspect both CIs separately.\n\n")
    text.extend(['The main two-arm comparison changes both allocation and resampler. It cannot establish an allocation advantage beyond the resampler without the separate 2x2 ablation; that ablation was not run.\n\n',
                 'A soft-VASR loss may be consistent with the motivation concerning noisy Tweedie rewards, but this experiment does not measure repeated stochastic continuations and cannot establish noise as the cause.\n\n',
                 'Realized variance across the K offspring counts is a concentration diagnostic, not a Monte Carlo estimate of conditional resampling variance.\n\n',
                 '## Fixed protocol and limitations\n\n',
                 '100 benchmark_ir prompts; seeds 42/43/44/45; events [20,40,80], zero-based, with no step 60 or final resampling. '
                 'BFS gamma is frozen from earlier results: 0.008 (N4), 0.024 (N8), temperature 10 with increasing tempering, Max scoring and SSP. '
                 'VASR lambda=1, zscore=False, ordering=True, persistent w/m weights, official low-level systematic resampling. '
                 'The ordering score is algebraically constant with this soft allocation, so stable ties retain original order.\n\n',
                 'Both arms use the unchanged Classical Search hook after scheduler.step, not a native VASR pipeline. '
                 'Both use TPU BF16, CPU ImageReward and identical lossless in-memory PIL scoring. Selected JPEGs are storage only. '
                 'Historical BFS results are not reused in the paired table. Historical hyperparameters were selected on this benchmark before VASR; this is not independent held-out tuning.\n'])
    write_csv(output/'results_table.csv',rows)
    (output/'report.md').write_text(''.join(text))
    save(output/'complete.json',{'complete':True,'sizes':[4,8],'summaries':summaries,
                               'interpretation':protocol['interpretation']},immutable=True)
