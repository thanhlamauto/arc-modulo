"""Freeze the N=8 TRAIN-only graph construction plan before any rollout."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess

from .adapter import RNG_PROTOCOL
from .graph_builder import all_edges
from .phase2_plan import (PILOT, BASE_COMMIT, PILOT_EDGE_SHA,
                          PILOT_FROZEN_SHA, PILOT_SPLIT_SHA,
                          validate_sources, freeze_text)
from .runner import (REPO, CONFIG, BASELINE_CONFIG, file_sha,
                     read, write_once)

PROTOCOL = 'STATIC_BFS_N8_RETUNING_V1'
SEEDS = [42,43,44,45]
N4_POLICY = {'name':'TRANSFER4_TO_8','K':3,
             'steps':[40,60,90], 'taus':[[40,8.0],[60,8.0],[90,32.0]]}


def n8_config():
    source = read(CONFIG)
    if (source['particles'] != 4 or source['steps'] != 100 or
            source['eta'] != 1.0 or source['scoring'] != 'max' or
            source['resampling'] != 'ssp' or
            source['execution_dtype'] != 'bfloat16' or
            source['pilot']['candidate_steps'] != [10,20,30,40,60,80,90] or
            source['pilot']['taus'] != [2.0,8.0,32.0] or
            source['pilot']['reference_tau'] != 8.0):
        raise RuntimeError('N=4 graph configuration unexpectedly changed')
    cfg = {**source, 'particles':8, 'protocol_version':PROTOCOL}
    return cfg


def preflight(root: Path):
    frozen, branch, commit = validate_sources()
    cfg = n8_config()
    split = read(PILOT/'prompt_split.json')
    train = sorted(split['train_indices'])
    fit, heldout = train[:12], train[12:]
    if (len(fit),len(heldout)) != (12,48) or set(fit)&set(heldout) or \
            set(heldout)&set(split['validation_indices']+split['test_indices']):
        raise RuntimeError('invalid fixed TRAIN split')
    grid = cfg['pilot']
    edges = all_edges(tuple(grid['candidate_steps']),tuple(grid['taus']))
    if len(edges) != 92:
        raise RuntimeError('wrong graph size')
    plan = {'protocol':PROTOCOL,'N':8,'K':3,'seeds':SEEDS,
            'graph_fit_12':fit,'heldout_train_48':heldout,
            'candidate_steps':grid['candidate_steps'],'taus':grid['taus'],
            'reference_tau':grid['reference_tau'],'edge_keys':[e.key for e in edges],
            'edge_count':92,'DDIM_steps':100,'eta':1.0,'dtype':'bfloat16',
            'verifier':'ImageReward-v1.0','scoring':'Max','resampling':'SSP',
            'rng_protocol':RNG_PROTOCOL,'frozen_N4_edge_sha256':PILOT_EDGE_SHA,
            'frozen_N4_split_sha256':PILOT_SPLIT_SHA,
            'edge_definition':'dense-reference selected terminal IR minus single-edge intervention selected terminal IR',
            'transfer_N4_policy':N4_POLICY,
            'manual8':{'steps':[20,40,80],'base_temperature':10.0,
                       'tempering':'increase','gamma':0.024},
            'no_validation_test_external_pool':True}
    write_once(root/'graph_plan.json',plan)
    plan_sha = file_sha(root/'graph_plan.json')
    freeze_text(root/'graph_plan.sha256',plan_sha+'  graph_plan.json\n')
    import torch, torch_xla
    status = subprocess.check_output(['git','status','--short'],cwd=REPO,text=True)
    manifest = {'protocol':PROTOCOL,'branch':branch,'commit':commit,
                'git_status_before_run':status,'host':'Hieu-tpu-v5-8-1',
                'accelerator':'TPU v5p-8','physical_chips':4,
                'torch':torch.__version__,'torch_xla':torch_xla.__version__,
                'model':'sd15','N':8,'K':3,'dtype':'bfloat16',
                'DDIM_steps':100,'eta':1.0,'verifier':'ImageReward-v1.0',
                'scoring':'Max','resampling':'SSP','rng_protocol':RNG_PROTOCOL,
                'graph_plan_sha256':plan_sha,'pilot_frozen_sha256':PILOT_FROZEN_SHA,
                'N4_edge_table_sha256':PILOT_EDGE_SHA,
                'N4_split_sha256':PILOT_SPLIT_SHA,
                'N4_config_sha256':file_sha(CONFIG),
                'baseline_config_sha256':file_sha(BASELINE_CONFIG),
                'baseline_source_sha256':file_sha(REPO/'scripts/diffusion_classical_search_t2i/tpu_runner.py'),
                'graph_fit_12_count':12,'heldout_train_48_count':48,
                'validation_touched':False,'test_touched':False,
                'external_100_prompt_touched':False}
    write_once(root/'experiment_manifest.json',manifest)
    print('N8_PREFLIGHT_PASS',plan_sha,'edges',len(edges),'fit',len(fit),
          'heldout',len(heldout),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run-id',default='n8-retuning-v1')
    args=ap.parse_args();preflight(REPO/'results/static-bfs-graph'/args.run_id)


if __name__=='__main__':main()
