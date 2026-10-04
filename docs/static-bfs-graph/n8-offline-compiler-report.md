# N=8 Static BFS offline compiler — SD1.5 / Modal L40S

Source: `NguyenNgocMinh30012005/arc-modulo`, branch `ngocminh`, base commit
`34f17bd984921aeba046e8a5b7936e4a7a8346c0`; Modal workspace
`thanhlamtba`. This run builds a three-event Static BFS schedule from **20 TRAIN prompt–seed
calibration trajectories**. The graph is fitted only on those trajectories.
The separate audit uses 20 disjoint TRAIN prompts × four seeds and the five
unique policies frozen in `audit/audit_policy_plan.json` (P5=P10). No PSP,
VALIDATION, TEST, external 100-prompt pool, or TPU edge values enter this run.

## Frozen experiment and correctness

The backend is SD1.5 BF16 on NVIDIA L40S, DDIM100 with η=1, N=8, Max scoring,
SSP resampling, and ImageReward. The graph has 92 first-order intervention
edges over steps 10/20/30/40/60/80/90 and destination temperatures 2/8/32.
The dense reference temperature is 8. Replay restored all six reference
continuations at source steps 10/40/80 with **exact** event records, final
latents, and terminal scores. Five independent full-from-start edge checks
matched the shared-tree result, and three tau-fork checks matched while using
one shared verifier call for the destination and separate suffixes.

The audit was initially interrupted by a GPU runtime argument bug: MANUAL8
was passed as `raw_tau` despite its frozen gamma-increase schedule. The runtime
was corrected to use the existing TPU runner's `kind=manual` branch. The
frozen plan, MANUAL8 parameters, and saved static-policy results were retained.

## Offline computation and graph stability

The 20 reference passes produced 20 post-event snapshot files (353,438,472
bytes in total), 160 shared source trunks, 560 destination verifier
evaluations, 1,680 independently executed tau suffixes, and 1,840 logical
edge results. Without destination verifier sharing, those tau branches would
need 1,680 verifier evaluations, so 66.7% of destination evaluations were
avoided. The profiler executed 74,980 eight-particle denoising steps versus
184,000 for 1,840 independent 100-step edge runs, a 59.25% reduction in
denoising-step work. Tau suffixes were not deduplicated. The controlled
five-edge benchmark took 137.15 s independently versus 84.29 s with the
shared tree: **1.63× measured speedup** for that sample.

| Calibration trajectories | Frozen policy steps | Temperatures | Bootstrap exact-path frequency | Sum of measured GPU time | Rate-based compiler estimate |
|---:|---|---|---:|---:|---:|
| 1 | [60,80,90] | [2,2,32] | degenerate | 0.311 h | $0.784 |
| 5 | [30,40,90] | [8,2,32] | 8.0% | 1.533 h | $3.864 |
| 10 | [30,40,90] | [8,2,32] | 16.9% | 3.072 h | $7.742 |
| 20 | [30,80,90] | [8,8,8] | 2.5% | 6.220 h | $15.675 |

The G1/G5/G10 edge-rank Spearman correlations against G20 were 0.539,
0.334, and 0.795. The nonmonotonic values and only 2.5% exact-path bootstrap
frequency at n=20 show that a *particular* optimum path is not yet stable.
P5=P10, but the preregistered n=10 stop rule failed because bootstrap
frequency was 16.9% and G5–G10 edge-rank correlation was 0.478. Accordingly,
n=20 was profiled. The frozen deployment artifact is
`compiled/compiled_policy.json`: P20=[30,80,90], τ=[8,8,8]. The deployment
entrypoint loads this policy directly and does no graph construction or
calibration at inference time.

## Frozen held-out TRAIN audit

The frozen plan contains five unique policies: P1, P5=P10, P20, MANUAL8,
and TRANSFER4_TO_8. All 400 policy–prompt–seed units completed. Each mean
below uses 20 held-out TRAIN prompts × four paired seeds. Differences and
95% intervals use 20,000 bootstrap resamples of the **20 prompt means**;
wins/ties/losses likewise count prompts, not the 80 individual seed runs.

| Policy | Mean terminal ImageReward | Δ vs MANUAL8 [95% CI], W/T/L | Δ vs TRANSFER4_TO_8 [95% CI], W/T/L |
|---|---:|---|---|
| MANUAL8 | 0.94153 | reference | −0.05299 [−0.08963, −0.01982], 6/1/13 |
| TRANSFER4_TO_8 | **0.99451** | **+0.05299 [+0.01982, +0.08963], 13/1/6** | reference |
| P1 | 0.97830 | +0.03678 [−0.00888, +0.08651], 12/1/7 | −0.01621 [−0.05800, +0.01927], 8/3/9 |
| P5=P10 | 0.97119 | +0.02966 [−0.01551, +0.07712], 12/1/7 | −0.02332 [−0.06069, +0.01174], 5/3/12 |
| **P20 (pre-audit frozen)** | 0.95969 | +0.01816 [−0.03819, +0.07082], 10/1/9 | −0.03483 [−0.08440, +0.00845], 6/1/13 |

All compiled schedules have point-estimate means above MANUAL8, but none
has a positive 95% paired interval against it. TRANSFER4_TO_8 has the
highest mean and beats MANUAL8 on this TRAIN audit; its comparison with any
compiled schedule is inconclusive at 95%. P1 is the highest-scoring compiled
schedule *descriptively*, but choosing P1 after seeing the audit would break
the frozen-policy design. P20 remains the compiled deployment artifact.
No confirmation/TEST claim follows from this small TRAIN-only audit.

## Cost and scope

The 20 calibration trajectories took 22,390.4 summed GPU seconds (6.220
GPU-hours); multiplying by the run's frozen L40S/CPU/memory rate assumption
gives **$15.675 estimated compilation compute**, below the $25 compiler
cap. The audit's 400 scored units took 8,247.8 summed inference GPU seconds,
or **$5.774 estimated evaluation compute** excluding model loading. The
rate-based sum for these two measured components is **$21.449**. Replay
gates, model loading, container startup, and the failed audit attempts add
unmeasured cost. This is a compute estimate, **not an actual Modal invoice**
or a complete billed research total. The compiler method claim uses the
$15.675 component alone. The n20 profiling stage occupied roughly 57
calendar minutes on four workers (Modal app `ap-ezrg9Nkv2KZdNSVvQOUALd`);
the 6.220 GPU-hours above are the accurate cumulative measured runtime
across the whole compiler.

## Answers to the preregistered questions

1. **Q1 — Completely offline compiler?** Yes. Shared-tree profiling writes
   terminal-reward edge observations, and the CPU graph compiler consumes
   only the edge table. Deployment loads the frozen JSON schedule directly.
2. **Q2 — Exact scientific semantics?** Yes on the six reference replays,
   five full-from-start edge checks, and three tau-fork checks. Event scores,
   temperatures, weights, parents, final latents, and rewards matched in
   those tested cases; this is a correctness gate, not a proof over every
   possible trajectory.
3. **Q3 — Reuse across edges with the same source?** Across 20 trajectories,
   1,840 logical edges reused 20 reference passes and 160 source trunks.
   Denoising work fell from an estimated 184,000 to 74,980 steps (59.25%).
4. **Q4 — Verifier reuse across tau?** One destination verifier evaluation
   served three tau branches: 560 actual versus 1,680 independent destination
   evaluations (66.7% saved). Tau suffix propagation remained separate.
5. **Q5 — Measured profiler speedup?** The controlled five-edge check was
   137.15 s independent versus 84.29 s shared, or 1.63×. This sample
   speedup should not be projected as an exact whole-run wall-clock ratio.
6. **Q6 — Edge-rank stabilization sample size?** Not established by n=20.
   Spearman with G20 was 0.539/0.334/0.795 for G1/G5/G10, while P20's
   own exact-path bootstrap frequency was only 2.5%.
7. **Q7 — Trajectories needed for a competitive policy?** P1 already had a
   mean above MANUAL8, but none of P1/P5/P10/P20 had a positive 95% paired
   interval against MANUAL8 or TRANSFER4_TO_8. The required number for a
   reliably superior compiled policy is not identified within 1–20.
8. **Q8 — Is 10–20 enough?** No for a stable exact K=3 path or proven audit
   advantage. P10 exact-path frequency was 16.9%; P20 fell to 2.5%, and
   neither beat transfer in mean reward.
9. **Q9 — Actual time and cost?** Measured compiler workload was 6.220
   summed GPU-hours, estimated $15.675 at the frozen rate. The n20 stage
   occupied about 57 wall-clock minutes on four GPUs. A complete billed
   amount for all stages is unavailable; audit inference separately adds
   2.291 summed GPU-hours, estimated $5.774 before overhead.
10. **Q10 — Lightweight relative to brute-force tuning?** Yes as an
    engineering implementation: 20 rather than 240 calibration
    prompt–seed trajectories, shared reference/source/verifier work, and a
    measured 1.63× five-edge speedup. It did not deliver a stronger policy
    on this audit, so compute efficiency alone is not a method win.
11. **Q11 — Does N=8 rebuild beat zero-shot transfer?** No. P20 scored
    0.95969 versus transfer's 0.99451; paired Δ=−0.03483 with 95% CI
    [−0.08440,+0.00845]. P1 also remained below transfer in point estimate.
12. **Q12 — Prefer compile-once transfer?** For this frozen SD1.5/N=8
    protocol, transfer is the practical incumbent: higher audit mean and no
    new N=8 compilation cost. The paired interval against P20 still crosses
    zero, so this is a deployment preference from the TRAIN audit, not a
    general superiority claim or a TEST confirmation.
13. **Q13 — Keep the cancelled 60×4×92 job cancelled?** Yes. The 20-trajectory
    lightweight compiler did not establish an audit gain or stable exact
    policy, so the much larger sweep is not justified by this experiment.

The final frozen design is **P20=[30,80,90], τ=[8,8,8]** as an offline
compiler demonstration, with **no N=8 retuned policy ready for final TEST**.
The audit did not change the compiled artifact. The six figures in `plots/`
show the architecture, audit reward curve, cost curve, controlled benchmark,
edge convergence, and path stability. Integrity checks and hashes are in
`integrity.json`.

## Post-audit offline diagnostic

An exploratory analysis using only the existing calibration edges and audit
rows is in `diagnostics/OFFLINE_PATH_DIAGNOSTIC_REPORT.md`. It compares
predicted versus actual ranking of the four audited static policies,
decomposes the P20–transfer graph margin with paired uncertainty, and tests
an SEM-penalized edge cost as a sensitivity analysis. It neither changes the
frozen policy nor consumes additional prompts or GPU compute.
