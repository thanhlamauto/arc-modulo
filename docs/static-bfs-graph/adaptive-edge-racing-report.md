# Adaptive Edge Racing v1 — final TRAIN-only report

## Outcome

AER completed on Modal L40S after moving the interrupted run from workspace `thanhlamtba` to `thanhlamresearch` at the user's request. The same frozen plans, seeds, candidate set, 16 discriminating edges, SSP sampler, SD1.5 backend, ImageReward verifier, additive path cost, and paired-bootstrap elimination rule were retained. Nine completed racing trajectories and the tenth reference checkpoint were migrated; completed work was not recomputed. Frozen JSON hashes matched before and after migration. No VALIDATION, TEST, external 100-prompt pool, PSP, or new method tuning was used.

The 5-trajectory global graph proposed eight graph schedules, but the eight-candidate cap covered only **0.881** of 5,000 bootstrap selections, below the intended 0.90 mass. Adding the fixed `TRANSFER4_TO_8` incumbent produced **nine** candidates. Their **16** discriminating edges were profiled on 20 fresh TRAIN prompt-seed trajectories. The number of survivors after 5/10/15/20 trajectories was **9/7/7/5**. Because the final five were not distinguishable under the prespecified simultaneous intervals, AER returned the incumbent `[40,60,90] / [8,8,32]` with `selection_status=underidentified`. This is a conservative fallback, not a certified unique optimum.

The selected policy was frozen before a separate 20-prompt, one-seed-per-prompt TRAIN audit. `AER_WINNER` and `TRANSFER4_TO_8` are the same policy, so four distinct policies were run. Integrity checks passed: 20 racing trajectories × 16 edges = 320 fresh edge observations, 20 audit timing records, and 80 distinct policy-prompt rollouts. Five plots plus four round-specific pairwise plots are in `plots/`.

| Frozen audit alias | Mean terminal ImageReward |
| --- | ---: |
| AER_WINNER = TRANSFER4_TO_8 | 1.006313 |
| COARSE_TOP1 | 0.966100 |
| G20_OLD_TOP1 | 0.985244 |
| MANUAL8 | 1.012835 |

Paired prompt bootstrap used 20,000 resamples. AER minus COARSE_TOP1 was **+0.040213**, 95% CI **[−0.016168, +0.094691]**, W/T/L **10/6/4**. AER minus old G20 top-1 was **+0.021069**, CI **[−0.025334, +0.067331]**, W/T/L **12/0/8**. AER minus MANUAL8 was **−0.006522**, CI **[−0.044370, +0.031248]**, W/T/L **10/0/10**. AER minus transfer is identically zero. None of the nontrivial intervals establishes an audit advantage on this 20-prompt sample; the MANUAL8 point estimate is slightly higher here.

The estimated compiler cost, including the historical Stage-A compute, was **$7.02 / 2.79 GPU-hours**, versus **$15.68** for uniform G20 profiling: a **0.448** cost ratio. AER used **780 logical edge observations** (460 historical Stage A + 320 new Stage B), versus 1,840 for G20. Stage-B new compute alone was estimated at **$3.16**; independent audit cost is separate. These are timer × frozen-rate estimates, not invoices. Modal's `thanhlamresearch` billing summary after execution showed about **$4.09** ephemeral-app metered use in October, offset by credits, and $0 billed at that moment; this includes workspace setup and audit, while earlier AER trajectories were charged to the original workspace.

## Required questions

1. **Useful five-trajectory candidate family?** It gave a small graph proposal family but did not establish that the best full-policy schedule was inside it. The strongest prior incumbent was added explicitly, outside graph proposal generation.
2. **How many schedules cover 90% bootstrap mass?** The frozen cap admitted eight graph schedules covering 88.1%, so the 90% target was not reached; adding the incumbent gave nine candidates.
3. **Edges requiring refinement?** Sixteen of 92, or 17.4%.
4. **Did fresh edge profiling resolve ordering?** Partly: two candidates were eliminated at trajectory 10 and two more at 20, but five remained.
5. **Trajectories until elimination stabilized?** There was no unique stabilized winner within the maximum 20. Survivor counts were 9, 7, 7, 5 at budgets 5, 10, 15, 20. The frozen fallback selected transfer at every budget in the offline budget ablation.
6. **Did AER outperform coarse top-1 on audit?** Point estimate +0.0402 ImageReward, but the 95% CI crossed zero; no established improvement on 20 prompts.
7. **Did AER outperform or retain transfer?** It retained transfer exactly. The result demonstrates conservative refusal to replace the incumbent, not superiority over it.
8. **Fraction of G20 cost?** Estimated 44.8% of compiler cost and 42.4% of logical edge observations. Audit cost is excluded from both compiler figures.
9. **Estimation uncertainty or additive composition?** This run did not directly separate them because edge racing and full-policy audit used disjoint prompt-seed sets. The preceding same-trajectory diagnostic favored estimation/winner-selection noise, but AER's five survivors show uncertainty remains.
10. **Support for sequential-elimination theory?** Shared-edge measurement was compute-efficient and permitted four eliminations, but not identification. Paired percentile-bootstrap intervals are a practical rule, not an exact finite-sample certificate.
11. **What supplied concentration theory applies directly?** The finite-action/adaptive-allocation motivation and the need to spend fresh samples on unresolved comparisons apply conceptually. The supplied exact multinomial certificate does not directly certify this experiment.
12. **What fails to apply under SSP?** Exact multinomial offspring-count assumptions and any implied terminal-reward guarantee for the deployed SSP BFS policy. The experiment measures terminal edge interventions empirically.
13. **Study shadow multinomial certification next?** It may be useful as an independent local risk signal, but it would not certify the deployed SSP draw or terminal reward. This is a future proposal, not an AER result or a reason to change the present policy.

Candidate-mass ablation at 0.70/0.80/0.90/0.95 required 5/7/8/8 graph candidates, 6/8/9/9 including transfer, and 13/15/16/16 discriminating edges. The 0.90 and 0.95 settings both hit the eight-graph-candidate cap. Estimated costs for this ablation scale measured edge-profile time by `|D|`; they do not model changed shared-tree geometry.

## Interpretation and artifacts

The result supports the engineering claim that targeted edge refinement is substantially cheaper than uniform G20 profiling. It does **not** show a statistically superior new N=8 schedule, a unique AER winner, or a reward gain over transfer. The correct deployment decision from this experiment is to retain `TRANSFER4_TO_8`; no VALIDATION or TEST confirmation was consumed.

The hashed plans and selected policy are in `stage_a/candidate_set.json`, `racing/discriminating_edges.json`, `racing/racing_trajectory_plan.json`, `racing/aer_compiled_policy.json`, and `audit/frozen_policy_plan.json`. Detailed paired results are in `audit/paired_results.csv`, cost and ablations in `analysis/`, reproducibility and migration details in `RUN_LOG.md`, theory limitations in `THEORY_BRIDGE.md`, and final integrity hashes in `integrity.json`.
