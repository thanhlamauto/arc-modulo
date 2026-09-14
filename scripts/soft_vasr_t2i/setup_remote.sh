#!/usr/bin/env bash
set -euo pipefail
root=/home/nguyenngocminh2005bn/soft-vasr-t2i-20260905
volatile=/dev/shm/soft-vasr-t2i-20260905
uv=/home/nguyenngocminh2005bn/.local/bin/uv
old=/home/nguyenngocminh2005bn/dcs-table12-repro-20260903
mkdir -p "$root/logs" "$root/results" "$volatile/cache" "$volatile/uv-cache" "$volatile/tmp"
export UV_CACHE_DIR="$volatile/uv-cache"
export TMPDIR="$volatile/tmp"
if [[ ! -x "$volatile/venv/bin/python" ]]; then
  "$uv" venv --python /usr/bin/python3.10 "$volatile/venv"
fi
"$uv" pip install --python "$volatile/venv/bin/python" \
  'torch==2.4.0+cpu' 'torchvision==0.19.0+cpu' 'setuptools==69.5.1' 'wheel==0.48.0' \
  --index https://download.pytorch.org/whl/cpu --default-index https://pypi.org/simple --index-strategy unsafe-best-match
"$uv" pip install --python "$volatile/venv/bin/python" \
  -r "$old/harness/envs/diffusion-classical-search-t2i-tpu/requirements.txt" \
  'torch_xla @ https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.4.0-cp310-cp310-manylinux_2_28_x86_64.whl' \
  --no-build-isolation \
  --find-links https://storage.googleapis.com/libtpu-releases/index.html \
  --index https://download.pytorch.org/whl/cpu --default-index https://pypi.org/simple --index-strategy unsafe-best-match
"$uv" pip freeze --python "$volatile/venv/bin/python" > "$root/results/restored_environment.txt"
echo '[setup] restored pinned Classical Search core environment; no TPU science started'
