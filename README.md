# Diffusion test-time alignment reproduction kit

> **Static BFS Graph (SD1.5/TPU/Modal GPU):**
> [experiment results and reproduction guide](scripts/static_bfs_graph/README.md).

> **T2I Classical Search / BFS vs Soft VASR:** xem
> [hướng dẫn chạy lại](T2I_REPRODUCE.md) và
> [kết quả đã hoàn thành](T2I_RESULTS.md). Compact CSV/JSON full precision nằm
> tại [`results/t2i-search/`](results/t2i-search/).

Bộ mã này đóng gói lần tái lập bốn paper theo một layout chạy được bằng `uv`.
`main` ghim dependency, runner, kết quả, tài liệu và bootstrap đúng source từ
bốn branch paper. Mỗi paper có một branch riêng bắt đầu từ **đúng lịch sử Git
upstream** tại commit đã dùng trong thí nghiệm.

## Branches

| Paper | Branch | Upstream commit | Kết luận đã kiểm toán |
|---|---|---|---|
| Null-TTA | [`paper/null-tta`](https://github.com/sontungkieu/arc-modulo/tree/paper/null-tta) | `337bf73037e9` | PASS, 3 seeds x 50 prompts, 4 scorers |
| DAS | [`paper/das`](https://github.com/sontungkieu/arc-modulo/tree/paper/das) | `2f4b2239f29e` | PASS_WITH_VARIANCE; Kaggle numeric FAIL |
| FK Steering | [`paper/fk-steering`](https://github.com/sontungkieu/arc-modulo/tree/paper/fk-steering) | `9413005dee1e` | PARTIAL, 1 prompt x 3 seeds, `k=4` |
| FK Correctors | [`paper/fk-correctors`](https://github.com/sontungkieu/arc-modulo/tree/paper/fk-correctors) | `aa6f5ed4a0eb` | PASS_WITH_VARIANCE, exact Table A1 |

Các branch paper chứa lịch sử tác giả gốc và thêm đúng một lớp
`reproduction/` ở phía trên. Branch `main` không mirror source lần thứ hai;
script bootstrap tạo bốn detached linked worktree dưới `upstream/` tại đúng
commit đã audit. Muốn nghiên cứu lịch sử hoặc sửa code từng paper, checkout
branch tương ứng.

FK Steering có gitlink `discrete_diffusion/mdlm`. Phạm vi text-to-image đã chạy
không dùng submodule này; trên branch `paper/fk-steering`, có thể lấy nó bằng
`git submodule update --init --recursive` nếu muốn thử phần discrete diffusion.

```bash
git clone https://github.com/sontungkieu/arc-modulo.git
cd arc-modulo
git switch main
python3 scripts/bootstrap_upstreams.py
```

Bootstrap fetch từng branch qua HTTP/1.1 và retry tối đa ba lần; nếu mạng đứt,
chạy lại cùng lệnh sẽ giữ các worktree đã hoàn thành và tiếp tục phần còn lại.

## Dựng môi trường bằng uv

Không copy `.venv` giữa máy. Python 3.10 và dependency đã được khóa trong ba
lockfile độc lập:

```bash
# DAS + FK Correctors + report tooling
uv sync --frozen

# Null-TTA
cd envs/null-tta
uv sync --frozen --no-install-project

# FK Steering
cd ../fk-steering
uv sync --frozen --no-install-project
```

Trên máy có ổ/cache lớn, nên đặt `UV_CACHE_DIR` và
`UV_PROJECT_ENVIRONMENT` vào filesystem phù hợp trước khi sync. Modal, Kaggle
và Talapas cũng dựng lại từ lockfile; không tải virtualenv từ repository.

Lệnh notebook, Modal và Kaggle cụ thể nằm trong
[`docs/SUITE_README.md`](docs/SUITE_README.md). Commit upstream đầy đủ được ghi
trong [`UPSTREAMS.toml`](UPSTREAMS.toml).

Nếu không muốn linked worktree, có thể clone từng branch paper riêng. Wrapper
trên `main` chỉ yêu cầu source xuất hiện đúng tại `upstream/<paper-id>`.

## Kết quả và tài liệu

- Tóm tắt verdict và bảng số: [`REPORT.md`](REPORT.md)
- Báo cáo PDF: [`docs/report/release/diffusion_smc_reproduction_report.pdf`](docs/report/release/diffusion_smc_reproduction_report.pdf)
- LaTeX, BibTeX, code dựng bảng/figure: [`docs/report/`](docs/report/)
- Slide PDF + Beamer source: [`docs/slides/`](docs/slides/)
- Script thuyết trình tiếng Việt: [`docs/slides/speaker_script_vi.md`](docs/slides/speaker_script_vi.md)
- Giải thích DAS EMD/Wasserstein-1: [`docs/das-emd/`](docs/das-emd/)
- Kết quả machine-readable, seed variance và provenance: [`results/`](results/)

`results/` là bằng chứng tổng hợp nhỏ, không phải checkpoint. Ảnh sinh hàng
loạt, model cache, virtualenv, credential, log remote đầy đủ và PDF paper bên
thứ ba không được đưa vào repository.

## Kiểm tra nhanh

```bash
uv lock --check
(cd envs/null-tta && uv lock --check)
(cd envs/fk-steering && uv lock --check)
sha256sum -c SHA256SUMS
uv run python scripts/verify_upstreams.py
```

`SHA256SUMS` kiểm tra file thuộc branch `main`; `verify_upstreams.py` kiểm tra
bốn worktree bootstrap có đúng commit và sạch hay không.

Đây là reproduction có phân biệt rõ `PASS`, `PARTIAL`, variance và portability
failure; không nên coi mọi job `COMPLETED` là paper đã được tái lập thành công.

Xem [`docs/THIRD_PARTY_LICENSE_STATUS.md`](docs/THIRD_PARTY_LICENSE_STATUS.md)
trước khi tái phân phối code upstream.
