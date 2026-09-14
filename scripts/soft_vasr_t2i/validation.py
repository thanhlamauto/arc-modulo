"""Fail-closed, accelerator-independent validation shared by run and analysis."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np


def check(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def finite(value):
    if isinstance(value, float): return math.isfinite(value)
    if isinstance(value, dict): return all(finite(v) for v in value.values())
    if isinstance(value, (list,tuple)): return all(finite(v) for v in value)
    return True


def save(path, value, immutable=False):
    path = Path(path)
    check(finite(value), 'NaN/Inf in artifact')
    if immutable and path.exists():
        check(json.loads(path.read_text()) == value, f'immutable artifact changed: {path}')
        return
    path.parent.mkdir(parents=True,exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
    temp.replace(path)


def validate_record(record, protocol):
    check(finite(record),'nonfinite record')
    k = record['particles']
    check(k in protocol['particles'] and record['trial_seed'] in protocol['seeds'], 'unknown tuple')
    check(0 <= record['prompt_index'] < protocol['prompt_count'], 'unknown prompt')
    events = record['resampling_events']
    check([e['sampling_index'] for e in events] == protocol['resampling_steps'], 'event grid mismatch')
    c = record['counts']
    check(c['unet_invocations']==100 and c['unet_sample_forwards']==200*k, 'denoiser forward mismatch')
    check(c['nfe']==100*k and c['scheduler_steps']==100, 'NFE mismatch')
    check(c['intermediate_reward_calls']==3*k and c['final_reward_calls']==k, 'verifier calls mismatch')
    check(c['reward_batch_sizes']==[k]*4, 'unexpected verifier batching')
    check(c['vae_invocations']==4 and c['vae_decoded_particles']==4*k, 'decode path changed')
    check(c['callback_steps']==list(range(100)), 'denoising loop not complete')
    check(c['population_sizes']==[k]*100, 'population changed during denoising')
    check(len(record['final_particle_rewards'])==k, 'missing final rewards')
    check(record['selected_max_reward']==max(record['final_particle_rewards']), 'final selection not max')
    lineage=np.arange(k)
    persistent_weights=np.ones(k)
    for event in events:
        check(len(event['raw_rewards'])==k and len(event['parent_indices'])==k, 'event population mismatch')
        check(all(0<=i<k for i in event['parent_indices']), 'invalid parent')
        check(sum(event['offspring_counts'])==k, 'offspring total mismatch')
        check(len(event['lineage_ids_after'])==k, 'lineage total mismatch')
        parents=np.asarray(event['parent_indices'])
        check(np.array_equal(np.bincount(parents,minlength=k),event['offspring_counts']),'offspring vector mismatch')
        check(np.array_equal(lineage,event['lineage_ids_before']),'lost incoming lineage mapping')
        lineage=lineage[parents]
        check(np.array_equal(lineage,event['lineage_ids_after']),'incorrect lineage propagation')
        if record['method']=='vasr':
            check(np.allclose(persistent_weights,event['particle_weights_before'],rtol=1e-12,atol=0),'persistent weights reset')
            logits=np.log(persistent_weights)+np.asarray(event['raw_rewards'])
            masses=np.exp(logits-logits.max())
            masses=k*masses/masses.sum()
            check(np.allclose(masses,event['masses'],rtol=1e-12,atol=0),'soft VASR allocation mismatch')
            offspring=persistent_weights/masses
            check(np.allclose(offspring,event['offspring_weights_by_parent'],rtol=1e-12,atol=0),'w/m mismatch')
            check(np.array_equal(event['ordering_permutation'],np.arange(k)),'algebraic ordering ties changed')
            u=event['systematic_u']
            check(0<u<1,'invalid grid offset')
            expected_parents=np.searchsorted(np.cumsum(np.asarray(event['masses'])/k),(u+np.arange(k))/k,side='left')
            check(np.array_equal(parents,expected_parents),'systematic grid mapping mismatch')
            persistent_weights=offspring[parents]
            check(np.allclose(persistent_weights,event['particle_weights_after'],rtol=1e-12,atol=0),'offspring weights not carried')
    e=record['effective_protocol']
    for key in ('height','width','guidance_scale','eta','precision','scheduler','num_inference_steps','model','revision'):
        check(e[key]==protocol[key],f'protocol differs: {key}')
    check(e['backend']=='TPU' and e['verifier_device']=='cpu','backend changed')
    check(record['initial_latents_sha256'] and record['transition_rng_trace_sha256'], 'missing RNG evidence')
    if record['method']=='vasr':
        for event in events:
            check(event['lambda']==1 and not event['zscore'] and event['ordering_enabled'], 'VASR tuning detected')
    else:
        check(record['method']=='bfs' and record['gamma']==protocol['bfs']['gamma_by_n'][str(k)], 'BFS tuning detected')
    return True


def validate_pair(bfs, vasr, protocol):
    validate_record(bfs, protocol)
    validate_record(vasr, protocol)
    check((bfs['method'],vasr['method'])==('bfs','vasr'), 'wrong paired methods')
    for field in ('particles','trial_seed','prompt_index','prompt','prompt_seed','initial_latents_sha256',
                  'transition_rng_trace_sha256','effective_protocol','counts','protocol_sha256','code_hashes'):
        check(bfs[field]==vasr[field],f'paired fairness mismatch: {field}')
    return {'valid':True,'particles':bfs['particles'],'trial_seed':bfs['trial_seed'],
            'prompt_index':bfs['prompt_index'],'initial_latents_sha256':bfs['initial_latents_sha256'],
            'transition_rng_trace_sha256':bfs['transition_rng_trace_sha256'],
            'resampling_steps':protocol['resampling_steps'],'counts':bfs['counts']}
