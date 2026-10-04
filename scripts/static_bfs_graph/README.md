# Static BFS Graph: experiments and reproduction

This directory contains the static resampling-schedule graph for diffusion
test-time selection. The associated Modal GPU runners and offline analyses are
in [`modal_n8_gpu/`](../../modal_n8_gpu/). These scripts reproduce *completed*
experiments; reading this README or checking the code does not start a GPU job.

## What the graph measures

The SD1.5 protocol uses BF16, 100 DDIM steps (`eta=1`), Max scoring,
SSP resampling, and ImageReward. A node is a candidate zero-based sampling
step from `{10,20,30,40,60,80,90}`. A directed edge skips from one selected
event to the next and assigns the destination a temperature from `{2,8,32}`;
edges to `END=100` have no temperature. There are 92 possible edges. A K=3
schedule is a four-edge path from `START=-1` to `END`, chosen from 945 paths.

The **dense reference** resamples at all seven candidate steps with
temperature 8. For each prompt and seed, an edge intervention is run with
the same initial seed and changes the events between that edge's endpoints.
Its observed cost is terminal `ImageReward(reference) −
ImageReward(intervention)`. Negative edge costs are allowed. The reference
is an experimental anchor, **not an oracle or an assumed best policy**.
The graph sums first-order edge costs to rank complete K=3 paths; full-policy
audits test whether this additive ranking transfers.

## Completed results

| Run | Frozen comparison | Result | Scope |
| --- | --- | --- | --- |
| [N=4 TEST](../../docs/static-bfs-graph/n4-test-report.md) | K=3 `[40,60,90]`, tau `[8,8,32]` vs manual BFS | Delta ImageReward −0.0051, 95% CI [−0.0341,+0.0231] | 20 untouched TEST prompts × 4 seeds; inconclusive |
| [N=8 TPU retuning](../../docs/static-bfs-graph/n8-retuning-report.md) | Transfer N=4 policy vs N=8 retuned policy | Transfer 1.0048; retuned 0.9922; retuned minus transfer −0.0126 [−0.0263,−0.0006] | Graph fitted on 12 TRAIN prompts × 4 seeds; audit on a different 48 TRAIN prompts × 4 seeds |
| [N=8 Modal offline compiler](../../docs/static-bfs-graph/n8-offline-compiler-report.md) | G1/G5/G10/G20 vs manual and transfer | G20 bootstrap exact-path frequency 2.5%; transfer had the highest audit mean (0.9945), G20 0.9597 | 20 calibration trajectories; separate 20 TRAIN audit prompts × 4 seeds |
| [Adaptive edge racing](../../docs/static-bfs-graph/adaptive-edge-racing-report.md) | Targeted edge refinement after G5 | Five candidates survived; fallback retained transfer. No demonstrated audit advantage over manual | 20 fresh TRAIN racing trajectories and separate 20-prompt audit |
| [AER budget allocation](../../docs/static-bfs-graph/aer-budget-allocation-report.md) | Broad G5 vs G10 profiling plus targeted racing | Both arms underidentified; transfer fallback 1.0642, G10 top-1 1.0272 on audit (paired CI crosses zero) | 30 fresh TRAIN racing pairs; separate 20-prompt audit |

The rows use **different prompt/seed pools and sometimes different hardware**;
do not compare their absolute ImageReward means across rows. Except for the
frozen N=4 TEST row, the audits above are TRAIN diagnostics, not final TEST
confirmation. The N=8 TPU dense reference was measured only on the 12 graph-fit
prompts, **not** on its 48 held-out TRAIN prompts. Its fit-set mean 1.4048
cannot be compared with the held-out policy means above. Four of its 92
single-edge interventions had negative mean cost on the fit set, directly
showing the dense reference is not always best. The reference-versus-policy
paired held-out experiment discussed separately has **not** been run.

The offline compiler's shared-tree correctness gates passed 6/6 reference
replays, 5/5 full-from-start edge comparisons, and 3/3 temperature forks.
For its 20-trajectory graph, measured eight-particle denoising work fell
59.25% and destination verifier evaluations fell 66.7%; a controlled five-edge
sample ran 1.63× faster. Its measured compilation workload was 6.220 summed
L40S GPU-hours, or $15.675 at the run's frozen *estimated* rate. This is not
a Modal invoice. The policy-learning result remains limited: more calibration
did not identify a stable or superior N=8 schedule.

## Reproduce the Modal GPU experiment

The runnable starting point is the offline compiler. Install `uv` and use a
Modal account with available L40S quota and billing. This will incur charges.
All prompt IDs are from the frozen 60-prompt TRAIN pool in
[`train_pool.json`](../../modal_n8_gpu/train_pool.json); the scripts reject
changed hashes and do not need VALIDATION, TEST, PSP, or the external 100-prompt
pool. The historical source commit is `34f17bd984921aeba046e8a5b7936e4a7a8346c0`.

The Modal image packages the official FKD implementation. Check out its pinned
commit locally and point the packager at it:

```bash
git clone https://github.com/XiangchengZhang/Diffusion-inference-scaling.git ../Diffusion-inference-scaling
git -C ../Diffusion-inference-scaling checkout c158ab9eb058adb24ad0f8d48506ab837d7fae91
export STATIC_BFS_OFFICIAL_ROOT="$PWD/../Diffusion-inference-scaling"
export MODAL_PROFILE=thanhlamresearch  # or your own configured Modal profile
uvx --from modal modal profile current
```

From this repository's root, run the stages **in order**, checking each result
before moving on. The reference replay, edge equivalence, and cost gates must
pass before broad profiling. `STATIC_BFS_COMPILER_MAX_USD` sets the frozen
rate-based compiler cap; it is an estimate and excludes audit and startup.

```bash
uvx --from modal modal run modal_n8_gpu/modal_offline.py::preflight
uvx --from modal modal run modal_n8_gpu/modal_offline.py::lock_stop_rule
uvx --from modal modal run modal_n8_gpu/modal_offline.py::replay_test
uvx --from modal modal run modal_n8_gpu/modal_offline.py::tau_fork_test
uvx --from modal modal run modal_n8_gpu/modal_offline.py::profile_one
uvx --from modal modal run modal_n8_gpu/modal_offline.py::graph_analysis --max-n 1
STATIC_BFS_COMPILER_MAX_USD=25 uvx --from modal modal run modal_n8_gpu/modal_offline.py::cost_gate --measured-n 1
uvx --from modal modal run modal_n8_gpu/modal_offline.py::profile_stage --first-rank 1 --stop-rank 5
uvx --from modal modal run modal_n8_gpu/modal_offline.py::graph_analysis --max-n 5
```

Continue with n=10, apply the preregistered stop rule in
[`OFFLINE_RUNBOOK.md`](../../modal_n8_gpu/OFFLINE_RUNBOOK.md), and run n=20 only
if required and permitted by a new cost gate. Freeze the policy **before**
scoring the independent audit. The remaining entrypoints are
`freeze_audit`, `audit_cost_gate`, `audit_stage`, and `aggregate`; inspect
their arguments with `modal run … --help`. Download the result Volume, then
run `python -m modal_n8_gpu.offline_finalize <downloaded-run-dir>` and
`python -m modal_n8_gpu.offline_plots <downloaded-run-dir>` for integrity and
figures. The [historical runbook](../../modal_n8_gpu/OFFLINE_RUNBOOK.md)
records the original execution and repair; it should not be treated as a new
scientific protocol.

For AER budget allocation, [`aer_allocation_plan.py`](../../modal_n8_gpu/aer_allocation_plan.py)
first verifies the historical offline-compiler artifacts and freezes G5/G10
candidate sets and disjoint TRAIN pools. The Modal stages in
[`modal_aer_allocation.py`](../../modal_n8_gpu/modal_aer_allocation.py) are
`preflight`, `racing_batch`, `racing_cost_gate`, `analyze_round`,
`freeze_compiler`, `freeze_audit`, `audit_pilot`, `audit_cost_gate`,
`audit_stage`, and `audit_aggregate`. Follow the frozen plans and cost gates;
running individual stages out of order does not reproduce the report.

Large raw image/reward artifacts, model caches, GPU outputs, and credentials
are intentionally excluded from Git. The linked compact reports preserve
sample sizes, paired uncertainty, hashes, and limitations.
