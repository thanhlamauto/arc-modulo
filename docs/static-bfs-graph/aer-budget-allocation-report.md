# AER-BUDGET-ALLOCATION-v1 — TRAIN-only result

Branch `ngocminh`, commit `34f17bd984921aeba046e8a5b7936e4a7a8346c0`; Modal workspace `thanhlamresearch`, L40S. SD1.5, N=8, K=3, DDIM100 eta=1, Max/SSP/ImageReward. No VALIDATION, TEST, PSP, or external pool used.

## Frozen design and source integrity

Historical nested G5/G10 92-edge matrices were reused without new broad GPU work. G5 has 9 graph candidates, mass 0.9176; with transfer, 10. G10 has 16 graph candidates, mass 0.7738; with transfer, 17. G10 proposal undercoverage is `true`. |D5|=17, |D10|=22, |D_union|=26. Candidate Jaccard 0.190, edge-rank Spearman 0.478.

Fresh racing measured 30 common TRAIN prompt-seed trajectories and 780 union-edge records. Independent audit: 20 TRAIN prompts, one new seed each, 3 unique frozen policies. Exact reference/edge replay correctness gate passed. Frozen compiler hash `6922a8a90b181a3c6045c04490a9461b9d64ea0c717fa6d1b0a087f96d33a494`; frozen audit-plan hash `b842dbcda4249cdbf6f35d907dd831b69856cbc5685d278a703f8833b08bd819`.

## Racing and total compilation cost

| Arm | Fresh racing | Survivors | Status | Selected/fallback | Broad USD | Targeted USD upper | Total USD upper |
|---|---:|---:|---|---|---:|---:|---:|
| AER5 | 30 | 7 | underidentified | `static_df7ddcbdb5f12bd4` | 3.86 | 5.19 | 9.05 |
| AER10 | 30 | 12 | underidentified | `static_df7ddcbdb5f12bd4` | 7.74 | 6.43 | 14.17 |

Pure G10_TOP1 compiles for historical 7.74 USD with no racing. New measured D_union racing plus independent audit cost 8.03 USD in GPU-time estimates, under the 12.00 USD cap. Broad G5/G10 costs are historical, not new spend. Arm-specific D5/D10 costs are bounds inferred from the measured union run: lower excludes union shared overhead; upper includes all of it. The equal-budget table and primary cost figure use the conservative upper estimate.

## Independent ImageReward audit

| Alias | Policy | Mean ImageReward |
|---|---|---:|
| G10_TOP1 | `static_937dd840e3b6069f` | 1.0272 |
| AER5_FINAL | `static_df7ddcbdb5f12bd4` | 1.0642 |
| AER10_FINAL | `static_df7ddcbdb5f12bd4` | 1.0642 |
| TRANSFER4_TO_8 | `static_df7ddcbdb5f12bd4` | 1.0642 |
| MANUAL8 | `manual_bfs_g0p024_n8` | 1.0618 |

Paired deltas use 20,000 prompt-level bootstrap resamples:

| Comparison | Delta IR | 95% CI | W/T/L |
|---|---:|---|---|
| AER5_FINAL − G10_TOP1 | 0.0371 | [-0.0081, 0.0876] | 11/3/6 |
| AER10_FINAL − G10_TOP1 | 0.0371 | [-0.0081, 0.0876] | 11/3/6 |
| AER5_FINAL − AER10_FINAL | 0.0000 | [0.0000, 0.0000] | 0/20/0 |
| AER5_FINAL − TRANSFER4_TO_8 | 0.0000 | [0.0000, 0.0000] | 0/20/0 |
| AER10_FINAL − TRANSFER4_TO_8 | 0.0000 | [0.0000, 0.0000] | 0/20/0 |

## Answers to the preregistered questions


**Q1.** No. G5 entropy 3.053 bits; G10 entropy 4.688 bits.

**Q2.** No. G5 reaches 90% with 9; G10 reaches only 77.38% at cap 16.

**Q3.** No. |D5|=17, |D10|=22.

**Q4.** At the preregistered $10 checkpoint: AER5 7 survivors after 30 racing trajectories; AER10 13 after 10. Arm-specific targeted costs are conservative upper estimates, since only D_union was run.

**Q5.** AER5: underidentified; AER10: underidentified at their frozen stopping points.

**Q6.** Pure G10_TOP1 scored 1.0272 versus 1.0642 for the shared AER fallback/transfer. The paired delta is 0.0371 with CI [-0.0081, 0.0876], so these 20 TRAIN prompts do not establish a reward advantage.

**Q7.** Highest observed audit mean 1.0642 is shared exactly by AER5_FINAL, AER10_FINAL, TRANSFER4_TO_8 because they are the same policy. At comparable compiler cost, AER5 has fewer survivors; audit provides no AER5-versus-AER10 reward distinction.

**Q8.** Median difficulty at the final checkpoint is 12.994 versus 1.615 for pairs eliminated in rounds 1–2. This is descriptive; the current-best comparator changes over rounds.

**Q9.** At about $10 total compiler budget, spending less on broad profiling and more on targeted edges leaves 7 rather than 13 survivors. This supports targeted allocation for decision efficiency here, but neither arm certifies a unique action and no SSP theorem follows.

**Q10.** Both: G10 has higher proposal entropy and only 77.38% mass at the 16-candidate cap, while 30 fresh racing trajectories still leave 7/12 survivors. The fixed action family contains hard-to-distinguish schedules.

**Q11.** Provisional compiler default: AER5; deployed fallback remains TRANSFER4_TO_8. Extra G10 broad profiling did not sharpen proposals or improve equal-budget elimination, and neither arm reached a certified choice.

**Q12.** No. Stop at 30 as frozen. Further scaling would require a new design question about the graph/action space; this run does not authorize more GPU rollouts.

## Limitations

The graph uses additive first-order terminal-regret edges; audit measures complete policies but does not rerank them. Bootstrap intervals are empirical and do not certify finite-sample optimality. Only one fresh seed per audit prompt was frozen. Arm-specific GPU cost cannot be measured exactly from a shared D_union run and is bracketed as described above. AER5 versus AER10 audit CI excludes zero: `false`.
