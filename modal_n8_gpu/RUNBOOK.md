# N=8 graph sample efficiency on Modal GPU

Status on 2026-10-02: **CANCELLED** by the user's n8-lightweight-v1 protocol. PREPARED and canary passed, but full reference/edge stages were never launched. The canary is engineering evidence only and must not enter the lightweight aggregate. TPU results were never copied into this run.

Workspace: Modal profile `thanhlamtba`. Run ID: `n8-graph-sample-efficiency-gpu-v1`. Modal Volumes: `static-bfs-n8-gpu-cache` and `static-bfs-n8-gpu-results`. Source repo commit: `34f17bd984921aeba046e8a5b7936e4a7a8346c0`; the static-BFS files are the existing uncommitted research scripts copied into this checkout. No commit/push.

Frozen protocol: SD1.5 revision `451f4fe16113bff5a5d2269ed5ad43b0592e9a14`, BF16, DDIM100 eta=1, 8 particles, Max+SSP, ImageReward-v1.0 on CPU, four seeds 42–45, same 60 TRAIN indices, same 92 edges, K=3, tau grid {2,8,32}, fit sizes {12,24,36,48,60}, prompt-grouped 2,000-replicate bootstrap. Never use VALIDATION, TEST, external 100-prompt pool, or policy reward evaluation. The 48 additional TRAIN prompts have appeared in prior policy diagnostics, so findings are exploratory.

Canary: reference prompt 0/seed 42 35.47 sec, one tau-2 edge 28.89 sec on L40S. Both produced valid ImageReward, nonuniform output, seven correct resampling events and event seeds. The changed tau did not alter SSP ancestry on this one unit; selected image hashes matched. Five structural tests passed. The CPU preparation cache and canary are already in the Modal Volumes. The pricing-based full-run estimate is roughly USD 300–450 before any workspace credits or discount; it is not a bill or guarantee.

The commands below are historical notes and must not be used for the cancelled run:

```bash
MODAL_PROFILE=thanhlamtba uvx --from modal modal run --detach modal_n8_gpu/modal_app.py::launch_references
```

Wait for 240/240 GPU reference records across four seeds, verify zero failures, and check the cost rate. Only then launch the edge shards:

```bash
MODAL_PROFILE=thanhlamtba uvx --from modal modal run --detach modal_n8_gpu/modal_app.py::launch_edges
```

The edge stage submits 184 two-edge shards, with at most eight GPU containers. Every raw unit has a deterministic key and atomic output. If a shard fails, inspect failure records and rerun the affected shard after fixing deployment only; never change the scientific plan.

After 240 references and 22,080 physical edge records, run strict GPU-only integrity and bootstrap analysis:

```bash
MODAL_PROFILE=thanhlamtba uvx --from modal modal run modal_n8_gpu/modal_app.py::finish_analysis
```

Download Volume results to local/Extreme SSD, run `scripts/static_bfs_graph/n8_sample_efficiency_plots.py --run <downloaded-run-dir>` to create four plots, inspect `report.md` and `integrity.json`, then report the selected policy at each fit size, tau40 probabilities and exact-path frequencies. Do not compare numerical IR scores directly with TPU as if paired on one backend.
