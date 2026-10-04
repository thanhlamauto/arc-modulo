"""Static scientific plots for the N=8 TRAIN-only graph calibration study."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def save(fig, out, stem):
    fig.savefig(out/f'{stem}.png',dpi=180)
    fig.savefig(out/f'{stem}.pdf')
    plt.close(fig)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True)
    args=ap.parse_args();root=args.run;out=root/'plots';out.mkdir(parents=True,exist_ok=True)
    with (root/'analysis/policy_by_fit_size.csv').open() as f:rows=list(csv.DictReader(f))
    n=np.asarray([int(r['n_fit']) for r in rows])
    p2=np.asarray([float(r['P_tau40_2']) for r in rows])
    p8=np.asarray([float(r['P_tau40_8']) for r in rows])
    p32=np.asarray([float(r['P_tau40_32']) for r in rows])
    exact=np.asarray([float(r['exact_path_frequency']) for r in rows])
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(7,4.5),constrained_layout=True)
    ax.plot(n,p2,'o-',color='#5C327E',label=r'$P(\tau_{40}=2)$')
    ax.plot(n,p8,'s-',color='#3267A4',label=r'$P(\tau_{40}=8)$')
    ax.plot(n,p32,'^-',color='#D17B19',label=r'$P(\tau_{40}=32)$')
    ax.set(xticks=n,ylim=(0,1),xlabel='Number of graph-fit TRAIN prompts',
           ylabel='Prompt-bootstrap selection frequency',
           title='N=8 preferred selection strength at step 40')
    ax.legend(frameon=False)
    save(fig,out,'01_tau40_probabilities')

    fig,ax=plt.subplots(figsize=(7,4.3),constrained_layout=True)
    ax.plot(n,exact,'o-',color='#5C327E')
    ax.set(xticks=n,ylim=(0,1),xlabel='Number of graph-fit TRAIN prompts',
           ylabel='Exact frozen-path bootstrap frequency',
           title='N=8 K=3 graph stability with more calibration prompts')
    save(fig,out,'02_exact_path_frequency')

    fig,ax=plt.subplots(figsize=(8,3.7),constrained_layout=True)
    for i,r in enumerate(rows):
        steps=json.loads(r['steps']);taus=json.loads(r['taus']);y=len(rows)-1-i
        ax.hlines(y,0,100,color='#CCCCCC',lw=1.5)
        ax.scatter(steps,[y]*3,color='#5C327E',s=70)
        for step,tau in zip(steps,taus):
            ax.annotate(f'{tau:g}',(step,y),xytext=(0,9),textcoords='offset points',ha='center')
    ax.set(xlim=(0,100),ylim=(-.5,len(rows)-.2),xticks=[0,10,20,30,40,60,80,90,100],
           yticks=list(range(len(rows))),yticklabels=[f'n={x}' for x in reversed(n)],
           xlabel='Zero-based DDIM sampling index',
           title='Full-data selected N=8 policies (labels: tau)')
    save(fig,out,'03_selected_policies')

    with (root/'analysis/bootstrap_paths.csv').open() as f:boots=list(csv.DictReader(f))
    p4='static_df7ddcbdb5f12bd4'
    p8='static_220ce94dc634388e'
    fig,ax=plt.subplots(figsize=(7,4.3),constrained_layout=True)
    for pid,label,color in ((p4,'N=4 transfer [8,8,32]','#3267A4'),
                            (p8,'Original N=8 [2,8,32]','#5C327E')):
        probs=[next((float(r['frequency']) for r in boots if int(r['n_fit'])==size and r['policy_id']==pid),0.0)
               for size in n]
        ax.plot(n,probs,'o-',color=color,label=label)
    ax.set(xticks=n,ylim=(0,1),xlabel='Number of graph-fit TRAIN prompts',
           ylabel='Prompt-bootstrap exact-policy frequency',
           title='Original N=4 and 12-prompt N=8 schedules across calibration sizes')
    ax.legend(frameon=False)
    save(fig,out,'04_original_policy_frequencies')
    print('N8_SAMPLE_EFFICIENCY_PLOTS_COMPLETE',4,flush=True)


if __name__=='__main__':main()
