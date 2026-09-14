#!/usr/bin/env bash
set -euo pipefail

ROOT=${DCS_REPRO_ROOT:-/home/nguyenngocminh2005bn/dcs-table12-repro-20260903}
HARNESS=${DCS_HARNESS_ROOT:-$ROOT/harness}
RUN_ID=${DCS_RUN_ID:-table12-tpu-20260903}
START_STAGE=${DCS_START_STAGE:-self_test}

source "$ROOT/venv/bin/activate"
export PJRT_DEVICE=TPU
export HF_HOME="$ROOT/cache/huggingface"
export TRANSFORMERS_CACHE="$ROOT/cache/huggingface/transformers"
export HF_HUB_CACHE="$ROOT/cache/huggingface/hub"
export GOOGLE_LOG_DIR="$ROOT/logs/tpu"
export TPU_LOG_DIR="$ROOT/logs/tpu"
export TOKENIZERS_PARALLELISM=false
export TF_CPP_MIN_LOG_LEVEL=1
export PYTHONDONTWRITEBYTECODE=1

mkdir -p "$ROOT/logs/tpu" "$ROOT/results/$RUN_ID" "$ROOT/cache/xla"

MANIFEST="$HARNESS/configs/diffusion_classical_search_t2i_table12.json"
RUNNER="$HARNESS/scripts/diffusion_classical_search_t2i/tpu_runner.py"
AGGREGATOR="$HARNESS/scripts/diffusion_classical_search_t2i/aggregate_table12.py"
COMMON=(
  --manifest "$MANIFEST"
  --official-root "$ROOT/official"
  --cache-root "$ROOT/cache"
  --output-root "$ROOT/results"
  --run-id "$RUN_ID"
)

stages=(self_test prefetch canary table1 table2_sd15 table2_sdxl)
started=false
for stage in "${stages[@]}"; do
  if [[ "$stage" == "$START_STAGE" ]]; then
    started=true
  fi
  if [[ "$started" != true ]]; then
    continue
  fi
  echo "[$(date --iso-8601=seconds)] starting $stage"
  python "$RUNNER" "${COMMON[@]}" --stage "$stage" \
    2>&1 | tee -a "$ROOT/logs/${RUN_ID}-${stage}.log"
  echo "[$(date --iso-8601=seconds)] completed $stage"
  if [[ "$stage" == "table1" ]]; then
    python "$AGGREGATOR" --run-root "$ROOT/results/$RUN_ID" --gate table1 \
      2>&1 | tee -a "$ROOT/logs/${RUN_ID}-table1-aggregate.log"
  fi
done

python "$AGGREGATOR" --run-root "$ROOT/results/$RUN_ID" \
  2>&1 | tee -a "$ROOT/logs/${RUN_ID}-aggregate.log"
df -h "$ROOT" | tee "$ROOT/results/$RUN_ID/disk-usage.txt"
echo "[$(date --iso-8601=seconds)] reproduction complete"
