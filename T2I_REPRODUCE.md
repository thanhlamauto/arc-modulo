# Hướng dẫn chạy lại thí nghiệm T2I

Hướng dẫn này tái tạo đúng hai phase đã dùng để sinh số trong
[`T2I_RESULTS.md`](T2I_RESULTS.md):

1. Classical Search Tables 1–2 baseline.
2. Paired BFS vs Soft VASR trên SD1.5.

Không cần tải raw images/result cũ từ GitHub. Runner sẽ tạo artifact mới và
fail-closed nếu model, protocol, counts, paired RNG hoặc analysis grid lệch.

## 1. Yêu cầu

- Linux TPU VM, Python 3.10 và `uv`.
- TPU v5p-8; mỗi paired worker chỉ nhìn thấy một chip.
- Đủ CPU RAM cho ImageReward và đủ disk/cache cho SD1.5 + SDXL.
- Git và quyền tải model từ Hugging Face.

Code và protocol được pin tại branch `ngocminh`:

```bash
git clone --branch ngocminh --single-branch \
  https://github.com/NguyenNgocMinh30012005/arc-modulo.git arc-modulo
cd arc-modulo
export HARNESS_ROOT="$PWD"
```

Clone official code đúng commit:

```bash
export DCS_ROOT=/home/<user>/dcs-table12-repro
mkdir -p "$DCS_ROOT"
git clone https://github.com/XiangchengZhang/Diffusion-inference-scaling.git \
  "$DCS_ROOT/official"
git -C "$DCS_ROOT/official" checkout c158ab9eb058adb24ad0f8d48506ab837d7fae91
test -z "$(git -C "$DCS_ROOT/official" status --porcelain)"
```

## 2. Dựng environment đã pin

```bash
uv venv --python /usr/bin/python3.10 "$DCS_ROOT/venv"
export UV_CACHE_DIR=/dev/shm/t2i-uv-cache

uv pip install --python "$DCS_ROOT/venv/bin/python" \
  'torch==2.4.0+cpu' 'torchvision==0.19.0+cpu' \
  --index https://download.pytorch.org/whl/cpu \
  --default-index https://pypi.org/simple \
  --index-strategy unsafe-best-match

uv pip install --python "$DCS_ROOT/venv/bin/python" \
  -r envs/diffusion-classical-search-t2i-tpu/requirements.txt \
  'torch_xla @ https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.4.0-cp310-cp310-manylinux_2_28_x86_64.whl' \
  --no-build-isolation \
  --find-links https://storage.googleapis.com/libtpu-releases/index.html \
  --index https://download.pytorch.org/whl/cpu \
  --default-index https://pypi.org/simple \
  --index-strategy unsafe-best-match
```

Không copy `.venv` từ máy khác. Model/cache cũng không được commit vào Git.

## 3. Chạy Classical Search baseline

Harness chạy tuần tự `self_test → prefetch → canary → table1 → table2_sd15 →
table2_sdxl`, còn mỗi stage dùng các PJRT devices addressable để chia four trials.

```bash
export DCS_REPRO_ROOT="$DCS_ROOT"
export DCS_HARNESS_ROOT="$HARNESS_ROOT"
export DCS_RUN_ID=table12-tpu-reproduction
export DCS_START_STAGE=self_test

bash scripts/diffusion_classical_search_t2i/remote_driver.sh
```

Nếu job dừng, đặt `DCS_START_STAGE` bằng stage cần tiếp tục. Per-prompt seed là
deterministic và record đã hoàn thành được resume, không sinh lại.

Kết quả baseline nằm tại:

```bash
export BASELINE_RESULTS="$DCS_ROOT/results/$DCS_RUN_ID"
test -f "$BASELINE_RESULTS/experiment_manifest.json"
test -f "$BASELINE_RESULTS/asset_manifest.json"
test -f "$BASELINE_RESULTS/export/results/diffusion-classical-search/t2i/table1/table1_reproduced.csv"
```

## 4. Chuẩn bị paired BFS–Soft-VASR run

Run này tái sử dụng official checkout, verified model cache và gamma evidence từ
baseline. Nó không tái sử dụng historical BFS scores.

```bash
export VASR_ROOT=/home/<user>/soft-vasr-t2i
export VASR_CACHE="$DCS_ROOT/cache"
mkdir -p "$VASR_ROOT/results"

"$DCS_ROOT/venv/bin/python" scripts/t2i_search/bootstrap.py \
  --run-root "$VASR_ROOT" \
  --baseline-results "$BASELINE_RESULTS" \
  --cache-root "$VASR_CACHE"
```

Bootstrap chỉ thành công khi mọi SD1.5/ImageReward asset khớp byte size + SHA256
và `table1_reproduced.csv` khớp gamma-evidence hash đã freeze. Nó tạo:

- `results/restored_assets.json`
- `results/restored_environment.txt`
- `harness/provenance/table1_reproduced.csv`
- `results/bootstrap_status.json`

## 5. Plan, preflight, smoke và full run

Đặt các path override một lần bằng Bash array:

```bash
COMMON=(
  --run-root "$VASR_ROOT"
  --official-root "$DCS_ROOT/official"
  --baseline-results "$BASELINE_RESULTS"
  --cache-root "$VASR_CACHE"
  --tpu-chip 2
)

PY="$DCS_ROOT/venv/bin/python"
RUNNER=scripts/t2i_search/cli.py

"$PY" "$RUNNER" plan "${COMMON[@]}"
"$PY" "$RUNNER" prepare "${COMMON[@]}"
"$PY" "$RUNNER" smoke "${COMMON[@]}"
"$PY" "$RUNNER" run "${COMMON[@]}"
"$PY" "$RUNNER" status "${COMMON[@]}"
"$PY" "$RUNNER" analyze "${COMMON[@]}"
```

Expected plan:

- 800 paired units = `2 N × 100 prompts × 4 seeds`.
- 1.600 method runs = BFS + VASR trong mỗi paired unit.
- N=4 trước N=8; methods được xen kẽ theo prompt/seed parity.

`smoke` là tuple N=4/prompt 0/seed 42 và được reuse trong main grid. `run` và
`analyze` đều resume-safe. Analysis từ chối chạy nếu thiếu bất kỳ pair/raw record
nào và giữ đúng 20.000 bootstrap resamples, seed `20260905`.

## 6. Scale nhiều TPU workers

Shard unit là toàn bộ `(N, prompt, seed)` pair, nên BFS không bao giờ bị tách khỏi
VASR. Mỗi process cần một chip và XLA cache riêng. Model directories có thể là
read-only symlink tới verified baseline cache.

Ví dụ bốn workers:

```bash
for i in 0 1 2 3; do
  SHARD_CACHE="/dev/shm/soft-vasr-shard-$i/cache"
  mkdir -p "$SHARD_CACHE"
  ln -sfn "$DCS_ROOT/cache/huggingface" "$SHARD_CACHE/huggingface"
  ln -sfn "$DCS_ROOT/cache/ImageReward" "$SHARD_CACHE/ImageReward"

  "$PY" "$RUNNER" run \
    --run-root "$VASR_ROOT" \
    --official-root "$DCS_ROOT/official" \
    --baseline-results "$BASELINE_RESULTS" \
    --cache-root "$SHARD_CACHE" \
    --tpu-chip "$i" \
    --shard "$i/4" \
    >"$VASR_ROOT/shard-$i.log" 2>&1 &
done
wait
```

Nếu CPU RAM không đủ cho bốn ImageReward instances, giảm số shard/process. Sau
khi mọi worker xong, dùng một cache hợp lệ và chạy `status`, rồi `analyze` không
có `--shard`.

## 7. Output cần giữ

```text
$VASR_ROOT/results/
├── preregistration.json
├── orchestration_manifest.json
├── structural_tests.json
├── smoke_validation.json
├── raw/n{4,8}/{bfs,vasr}/seed*/...
├── pairs/n{4,8}/seed*/...
├── analysis/n{4,8}/summary.json
├── results_table.csv
├── report.md
└── complete.json
```

Không dùng JPEG đã lưu để tính lại reward: score chính được tính trên lossless
in-memory PIL images; JPEG chỉ là storage artifact.

## 8. Kiểm tra kết quả

Kết quả chạy đúng protocol trước đây phải gần bảng trong
[`T2I_RESULTS.md`](T2I_RESULTS.md), nhưng accelerator/backend khác có thể tạo
numeric variation. Trước khi so sánh score, bắt buộc kiểm tra:

- `status` báo đủ 800 pairs và 1.600 records.
- Không duplicate/missing prompt IDs.
- Counts đúng `100 × N` guided NFE và `4 × N` ImageReward calls mỗi method tuple.
- Initial-latent + transition RNG hashes match trong từng BFS/VASR pair.
- Mọi metric finite và selected images còn đúng SHA256.
- Không thay đổi protocol, gamma, lambda, z-score, event indices hoặc bootstrap policy.

## 9. Scientific caveats

- Đây là TPU BF16 backend reproduction; paper launcher dùng FP16.
- Paired experiment là Soft VASR transfer, không phải official native VASR-MAX run.
- Kết quả paired chính không cô lập allocation khỏi resampler.
- Không dùng follow-up mechanism results để retune hoặc chọn lại tham số.
