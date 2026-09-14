#!/usr/bin/env bash
set -euo pipefail
root=/home/nguyenngocminh2005bn/soft-vasr-t2i-20260905
volatile=/dev/shm/soft-vasr-t2i-20260905
export PJRT_DEVICE=TPU
export TPU_VISIBLE_CHIPS=2
export HF_HOME="$volatile/cache/huggingface"
export TRANSFORMERS_CACHE="$volatile/cache/huggingface/transformers"
export HF_HUB_CACHE="$volatile/cache/huggingface/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1
export GOOGLE_LOG_DIR="$volatile/logs"
export TPU_LOG_DIR="$volatile/logs"
export TMPDIR="$volatile/tmp"
mkdir -p "$volatile/logs" "$volatile/tmp"
exec "$volatile/venv/bin/python" -u "$root/harness/scripts/soft_vasr_t2i/run.py" --stage "${1:-full}"
