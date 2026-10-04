# Frozen additive K=3 on untouched N=4 TEST

This is one preregistered, paired evaluation of the full-TRAIN additive K=3 policy as a representative of the TRAIN-identified policy family. The policy was frozen before TEST; TEST scores do not choose or alter its schedule. Manual BFS N=4 is the paired comparator.

| Method | Steps | Tau / schedule | Mean selected terminal ImageReward |
|---|---|---|---:|
| Frozen additive K=3 | `[40, 60, 90]` | `[8.0, 8.0, 32.0]` | 0.740661 |
| Manual BFS N=4 | `[20, 40, 80]` | base 10, increase gamma 0.008 | 0.745781 |

Paired difference, frozen K=3 minus manual: **-0.005120** ImageReward; 95% prompt-bootstrap CI **[-0.034109, +0.023060]**. Prompt wins/ties/losses: **11/0/9**. Preregistered descriptive label: **inconclusive**.

The 20 TEST prompts are the statistical units. Each prompt uses the same four trial seeds (42–45) for both policies, and the initial generator state hashes match within every pair. Both methods use SD1.5 BF16, DDIM100 eta=1, N=4, Max+SSP, terminal ImageReward, and 400 diffusion NFE per prompt/seed. The comparison estimates performance of this one frozen path; it does not identify which member of the near-equal TRAIN policy family would be best on TEST. Do not retune from these TEST results.

Integrity: 160/160 rollout records, no failures, 4 TPU chips. Frozen policy SHA256 `a2c3c7eccae865363d48336913e02b0873e249ead120cd951c6ee2e8d0afea44`; TEST protocol SHA256 `aa92c5fce29d9f072577584086041e9a455a74064b7207daa4f5c08abdd19e7c`. No N=8 run.

Machine-readable paired rows are in `evaluation/prompt_seed_scores.csv` and `evaluation/prompt_means.csv`; the preregistered protocol and execution manifest are beside this report.
