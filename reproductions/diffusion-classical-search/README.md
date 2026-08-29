# Diffusion Classical Search Reproduction

This scope reproduces arXiv:2505.23614v2 from the immutable official source
commit `c158ab9eb058adb24ad0f8d48506ab837d7fae91`.

The official source is never copied into this repository. Kaggle bundles are
built from `git archive` at the pinned commit. The external runner imports the
official Locomotion modules and changes orchestration only: device assignment,
run isolation, per-episode logging, provenance, and invariant checks.

Priority P0 is Locomotion Table 3. PointMaze and image scopes are deliberately
out of scope until Table 3 is reviewed.

The native D4RL `get_normalized_score` value is a fraction. Results retain that
fraction and also store `100 * fraction` as `paper_normalized_score`, because
Table 3 reports the 0--100 convention.

First-row gate, declared before execution:

- `REPRODUCED`: absolute mean delta at most 1.0 point;
- `CLOSE`: absolute mean delta greater than 1.0 and at most 3.0 points;
- `FAIL`: absolute mean delta greater than 3.0 points, incomplete episodes, or
  failed scientific invariants.

No retuning is permitted after observing results.
