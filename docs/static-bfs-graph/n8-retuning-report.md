# Static BFS Graph: N=8 budget retuning diagnostic

**Outcome:** C: N=8 retuning underperforms zero-shot N=4 transfer. This is TRAIN-only exploratory evidence. No final method or TEST policy was selected.

Graph: 92 edges × 12 fit prompts × 4 seeds; frozen RETUNED8 `[40, 60, 90]` with tau `[[40, 2.0], [60, 8.0], [90, 32.0]]`, additive predicted regret -0.0097. Bootstrap exact full-path frequency 34.3%.

## Table A — Frozen N=8 policies

| Method | N | K | Steps | Temperature schedule | Source |
|---|---:|---:|---|---|---|
| MANUAL8 | 8 | 3 | [20, 40, 80] | {'base_temperature': 10, 'tempering': 'increase', 'gamma': 0.024} | historical manual BFS N=8 |
| TRANSFER4_TO_8 | 8 | 3 | [40, 60, 90] | [[40, 8.0], [60, 8.0], [90, 32.0]] | frozen N=4 graph policy, no retuning |
| RETUNED8 | 8 | 3 | [40, 60, 90] | [[40, 2.0], [60, 8.0], [90, 32.0]] | new N=8 graph on GRAPH_FIT_12 |

## Table B — HELDOUT_TRAIN_48 result

| Method | Mean IR | Prompt std | Δ vs MANUAL8 | 95% paired CI | Win fraction |
|---|---:|---:|---:|---:|---:|
| MANUAL8 | 0.9737 | 0.6716 | +0.0000 | [+0.0000, +0.0000] | 0.0% |
| TRANSFER4_TO_8 | 1.0048 | 0.6735 | +0.0311 | [+0.0096, +0.0535] | 66.7% |
| RETUNED8 | 0.9922 | 0.6757 | +0.0185 | [-0.0038, +0.0401] | 56.2% |

## Table C — Paired contrasts

| Comparison | Mean Δ | 95% CI | Wins/Ties/Losses |
|---|---:|---:|---:|
| PRIMARY: RETUNED8 - MANUAL8 | +0.0185 | [-0.0038, +0.0401] | 27/0/21 |
| SECONDARY: RETUNED8 - TRANSFER4_TO_8 | -0.0126 | [-0.0263, -0.0006] | 14/17/17 |
| SECONDARY: TRANSFER4_TO_8 - MANUAL8 | +0.0311 | [+0.0096, +0.0535] | 32/0/16 |

## Table D — Cross-N graph shift

| Metric | Value |
|---|---:|
| Edge Spearman | +0.819 |
| Edge Pearson | +0.866 |
| Edge sign agreement | 75.0% |
| Best-tau agreement | 57.1% |
| P4/P8 timestep Jaccard | 1.000 |

## Scientific questions

**Q1. Edge landscape shift?** Edge Spearman +0.819, Pearson +0.866, sign agreement 75.0%; see the full 92-edge scatter.
**Q2. Different schedule/tau?** P4 steps [40, 60, 90] tau [[40, 8.0], [60, 8.0], [90, 32.0]]; P8 steps [40, 60, 90] tau [[40, 2.0], [60, 8.0], [90, 32.0]]. Changed timesteps 0; changed tau at shared steps 1.
**Q3. RETUNED8 vs manual?** Primary paired Δ +0.0185, 95% CI [-0.0038, +0.0401].
**Q4. RETUNED8 vs transfer?** Secondary paired Δ -0.0126, 95% CI [-0.0263, -0.0006].
**Q5. Is N=8 retuning necessary?** C: N=8 retuning underperforms zero-shot N=4 transfer; policy difference alone is not evidence of useful retuning.
**Q6. Selection pressure shift?** Best tau matches across N in 57.1% of 28 src/dst pairs; event ESS and lineage summaries are descriptive in `mechanism/event_summary.csv`.
**Q7. Tau grid sufficient?** Saturation flag False; tau=32 wins 10/28 matched N=8 edge groups. The tau grid was not expanded.
**Q8. Path stability?** Full-data P8 appears in 34.3% of 2,000 prompt bootstraps; top two paths cover 69.2%.
**Q9. Final untouched N=8 evaluation?** Do not run automatically. Any future rule choosing P4 transfer or P8 must be frozen explicitly before an untouched evaluation.
**Q10. Less budget-specific hand tuning?** Interpret the paired MANUAL8, TRANSFER4_TO_8 and RETUNED8 contrasts above. These TRAIN prompts were previously explored, so this is mechanism evidence rather than final confirmation.

All three policies used N=8, K=3, 800 diffusion NFE and 32 verifier particle evaluations per prompt/seed. Wall time is in `evaluation/policy_results.csv`. No VALIDATION, TEST, external 100-prompt pool or N=4 records were run here; zero failed rollouts.

Six figures are in `plots/`.
