# N=8 offline shared-tree BFS compiler (Modal GPU)

Scientific specification: frozen run manifest and plans in the experiment result Volume.
The previous `n8-lightweight-v1` prefix-only profiler is superseded. Its n=1
job was explicitly stopped; none of its edge values belongs to this run.

Workspace: `thanhlamtba`; Modal app: `static-bfs-n8-offline-compiler-v1`;
Volume: `static-bfs-n8-offline-compiler-v1-results`; remote path:
`n8-offline-compiler-v1`; local source: `modal_n8_gpu/modal_offline.py`.
Backend: SD1.5 BF16, DDIM100 eta=1, N=8, K=3, Max+SSP, ImageReward.
Only TRAIN prompts. Audit baselines: MANUAL8, TRANSFER4_TO_8. No PSP.

Preflight fixed 20 unique calibration prompts with one balanced seed each and
20 disjoint TRAIN audit prompts with seeds 42–45. Replay gate passed 6/6
reference resumes at 10/40/80 across two prompt-seed pairs, plus 5/5
shared-tree vs full-from-start edge checks. Exact event records, terminal scores,
latent/image hashes agreed. `replay/benchmark.json` reports 137.15 seconds
naive versus 84.29 seconds shared for the same five edge policies (1.63×).
The stop rule is frozen in `engineering_stop_rule.json`, hash
`aa9b1ef2823a61a0469bb6f17dac0894bd157dae584b8a3131f7e73555a76542`.

As of 2026-10-03 local time, profiling completed all 20 calibration
trajectories and validated 1,840 edge observations. The G20 policy is
`[30,80,90]` with tau `[8,8,8]`; exact bootstrap frequency is 2.5%.
Estimated compiler cost from 22,390 GPU-seconds is USD 15.675. Audit plan is
frozen (SHA256 `97ec7724df6d9e8bcf31b5bb0df00c622425c701ea095bb31967f2f936624834`).
The first audit app `ap-ZHzzBDjkpYrW7FiP96UEDr` stopped at MANUAL8 on the
first audit prompt: the GPU runtime mistakenly passed `selection_mode=raw_tau`
with no explicit temperatures. The GPU runtime now follows the frozen TPU
runner's `kind=manual` branch, using the original gamma-increase temperature
schedule and checking each observed event temperature. Existing static-policy
rows are retained; the frozen audit plan and scientific settings were not
changed. Resume app `ap-uE9iZKqvZi8kHBWPLrygiD` completed 400/400 audit
rows. Aggregate app `ap-rOfe9S1uPMYpCI84KOyAjT` validated and summarized
them. The separate tau-fork equivalence check
`ap-3ifhLk7NzIQ8XvPt2KlOnD` passed 3/3 edges for
`20 -> 60` at tau 2/8/32 using one shared verifier evaluation and three
independent suffixes. Its first attempt failed before scoring because a fresh
pipeline had not initialized the fixed guidance properties;
`offline_profiler/replay.py` now sets them before restore.

All artifacts were synced locally to
`results/static-bfs-graph/n8-offline-compiler-v1/` because the Extreme SSD
was not mounted. `offline_plots.py` generated six figures and
`offline_finalize.py` verified 20 references, 1,840 edge records, 400 audit
rows, frozen hashes, split membership, and scope. The full report answers
Q1–Q13 in `report.md`. No commit/push.

Commands from repository root:

```bash
MODAL_PROFILE=thanhlamtba uvx --from modal modal run modal_n8_gpu/modal_offline.py::cost_gate
MODAL_PROFILE=thanhlamtba uvx --from modal modal run modal_n8_gpu/modal_offline.py::graph_analysis --max-n 1
MODAL_PROFILE=thanhlamtba uvx --from modal modal run --detach modal_n8_gpu/modal_offline.py::profile_stage --first-rank 1 --stop-rank 5
```

Check CLI option spelling with `modal run ... --help`; do not infer job
completion from an empty log. Check app status and Volume output. Do not start
later stages if a correctness or cost gate fails.
