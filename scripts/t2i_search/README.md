# T2I search experiment runner

Đây là entrypoint gọn cho thí nghiệm Classical Search BFS vs Soft VASR. Nó chỉ
điều phối; implementation khoa học đã audit trong `soft_vasr_t2i/` và
`diffusion_classical_search_t2i/` được giữ nguyên để các artifact cũ vẫn có thể
kiểm tra provenance.

## Các thành phần

- `core.py`: đọc/validate config, tạo paired grid, shard và kiểm tra tiến độ.
- `backend.py`: adapter mỏng tới scientific code đã đóng băng.
- `cli.py`: một CLI cho prepare, smoke, run, resume và analysis.
- `configs/t2i_search.json`: đường dẫn/runtime; scientific protocol vẫn chỉ có
  một nguồn là `configs/soft_vasr_t2i.json`.

## Chạy

Mọi lệnh `plan` và `status` đều accelerator-free:

```bash
python scripts/t2i_search/cli.py plan
python scripts/t2i_search/cli.py status
python scripts/t2i_search/cli.py prepare
python scripts/t2i_search/cli.py smoke
python scripts/t2i_search/cli.py run
python scripts/t2i_search/cli.py analyze
```

Các lệnh giả định working directory là repo này (hoặc thư mục `harness/` trên
TPU).

`full` chạy tuần tự toàn grid rồi analyze. Các record đã hợp lệ được scientific
adapter tự resume, không sinh lại.

```bash
python scripts/t2i_search/cli.py full
```

Có thể override path vận hành mà không sửa protocol:

```bash
python scripts/t2i_search/cli.py plan \
  --run-root /data/my-run \
  --cache-root /dev/shm/my-run/cache
```

## Scale theo TPU

Shard chia theo paired unit `(N, prompt, seed)`, không chia BFS khỏi VASR. Vì vậy
paired fairness, thứ tự xen kẽ và RNG coupling không thay đổi. Mỗi worker dùng
một output root chung, một cache riêng, và một TPU chip riêng:

```bash
python scripts/t2i_search/cli.py run --shard 0/4 --tpu-chip 0 --cache-root /dev/shm/t2i-s0/cache
python scripts/t2i_search/cli.py run --shard 1/4 --tpu-chip 1 --cache-root /dev/shm/t2i-s1/cache
python scripts/t2i_search/cli.py run --shard 2/4 --tpu-chip 2 --cache-root /dev/shm/t2i-s2/cache
python scripts/t2i_search/cli.py run --shard 3/4 --tpu-chip 3 --cache-root /dev/shm/t2i-s3/cache
```

Phần preregistration dùng file lock dùng chung, còn raw artifact của các shard
không overlap. Có thể khởi động các worker đồng thời.

Sau khi tất cả worker hoàn tất:

```bash
python scripts/t2i_search/cli.py status
python scripts/t2i_search/cli.py analyze
```

Có thể chạy chọn lọc để retry hoặc chia theo scheduler ngoài bằng `--prompts
START:STOP`, `--particles 4,8`, và `--seeds 42,43`. Analysis luôn fail-closed nếu
không đủ toàn bộ grid đã đăng ký trước.

## Không thay đổi

- Protocol khoa học, budget, seed, prompt, timing scope và bootstrap policy.
- Scientific controller/adapter và validation đã dùng cho kết quả hiện có.
- Raw results, selected images, báo cáo và các script lịch sử phục vụ audit.
