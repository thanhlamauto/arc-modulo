# Kết quả thí nghiệm T2I đã hoàn thành

Tài liệu này ghi lại hai thí nghiệm đã chạy trên TPU v5p-8:

1. Tái lập Tables 1–2 của *Inference-time Scaling of Diffusion Models through
   Classical Search* (arXiv:2505.23614v2).
2. So sánh paired BFS với Soft VASR trên cùng Classical Search bed.

Các số dưới đây được lấy từ artifact gốc, không phải chạy lại hoặc làm tròn rồi
nhập tay. File CSV/JSON full precision nằm trong [`results/t2i-search/`](results/t2i-search/).

## 1. Classical Search Tables 1–2

### Protocol

- Official source commit: `c158ab9eb058adb24ad0f8d48506ab837d7fae91`.
- 100 `benchmark_ir` prompts; seeds `42, 43, 44, 45`.
- 100 DDIM steps, `eta=1`, resampling tại zero-based indices `[20, 40, 80]`.
- ImageReward chạy trên CPU; diffusion chạy TPU v5p-8 BF16.
- `N` là số particle/candidate được duy trì song song cho mỗi prompt.
- Paper nói bốn trials nhưng launcher commit chỉ nêu 42–44; seed 45 là phần
  reconstruction tối thiểu để đủ bốn trials liên tiếp.

### Table 2 reproduced means

Mỗi ô trình bày `mean ± sample std` của bốn seed means, mỗi seed mean được tính
trên đủ 100 prompts.

| Model | N | BoN | FK | DAS | TreeG | SVDD | BFS |
|---|---:|---:|---:|---:|---:|---:|---:|
| SD1.5 | 4 | 0.732 ± 0.038 | 0.685 ± 0.021 | 0.820 ± 0.049 | 0.788 ± 0.022 | 0.725 ± 0.027 | 0.793 ± 0.037 |
| SD1.5 | 8 | 0.914 ± 0.030 | 0.944 ± 0.046 | 1.038 ± 0.004 | 1.009 ± 0.017 | 0.979 ± 0.063 | 0.996 ± 0.039 |
| SDXL | 4 | 1.060 ± 0.021 | 1.086 ± 0.036 | 1.140 ± 0.025 | 1.133 ± 0.022 | 1.117 ± 0.043 | 1.138 ± 0.023 |
| SDXL | 8 | 1.159 ± 0.014 | 1.203 ± 0.021 | 1.276 ± 0.022 | 1.238 ± 0.015 | 1.224 ± 0.008 | 1.245 ± 0.011 |

BFS cao hơn BoN trong cả bốn `(model, N)` cells. Khoảng cách BFS so với paper
lần lượt là `-0.089`, `-0.091`, `-0.056`, và `-0.046`. Phần lớn cells gần số
paper theo ngưỡng audit đã đăng ký; SD1.5/N=8/SVDD là ngoại lệ lớn nhất với
absolute gap `0.204`.

Toàn bộ Table 1 (resampler, score, tempering) và Table 2 nằm ở:

- [Table 1 full precision](results/t2i-search/classical-search/table1_reproduced.csv)
- [Table 2 full precision](results/t2i-search/classical-search/table2_reproduced.csv)

## 2. Paired BFS vs Soft VASR

Đây **không phải** official VASR text-to-image reproduction. Soft VASR được
chuyển sang Classical Search ImageReward bed để so sánh paired, matched compute.
VASR paper dùng VASR-MAX cho các Stable Diffusion experiments chính, nên kết quả
transfer âm ở đây không phủ định kết quả native đó.

### Main result

| N | Method | Mean ImageReward | Std | Paired delta vs BFS | Bootstrap 95% CI | Prompt win fraction | NFE | Verifier calls |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 4 | BFS | 0.792512 | 0.037121 | reference | — | — | 400 | 16 |
| 4 | Soft VASR | 0.751540 | 0.043415 | -0.040971 | [-0.063977, -0.018576] | 36% | 400 | 16 |
| 8 | BFS | 0.996118 | 0.039299 | reference | — | — | 800 | 32 |
| 8 | Soft VASR | 0.967603 | 0.022950 | -0.028515 | [-0.053179, -0.004578] | 33% | 800 | 32 |

Paired delta là `VASR - BFS`. Với mỗi prompt, bốn matched-seed differences được
average trước; sau đó bootstrap 20.000 lần trên 100 prompt units với seed
`20260905`. Cả hai confidence intervals đều nằm dưới zero, nên BFS tốt hơn Soft
VASR trong bed này ở cả N=4 và N=8.

### Mechanism diagnostics

Mỗi ô là average của 400 events (100 prompts × 4 seeds).

| N | Sampling index | BFS lineages | VASR lineages | BFS offspring dispersion | VASR offspring dispersion |
|---:|---:|---:|---:|---:|---:|
| 4 | 20 | 3.8500 | 3.2925 | 0.085000 | 0.382500 |
| 4 | 40 | 3.2575 | 2.8375 | 0.446250 | 0.421250 |
| 4 | 80 | 2.1400 | 2.6200 | 1.190000 | 0.311250 |
| 8 | 20 | 6.9125 | 6.4750 | 0.738125 | 0.481250 |
| 8 | 40 | 4.4600 | 5.5750 | 2.504375 | 0.511875 |
| 8 | 80 | 2.1600 | 5.0700 | 4.626250 | 0.381250 |

VASR giữ nhiều lineage hơn ở event cuối, đặc biệt tại N=8, và có realized
offspring-count dispersion thấp hơn khi average ba events. Đây chỉ là
concentration diagnostic của một realized resampling draw, không phải ước lượng
conditional resampling variance.

### Integrity

- 800 matched `(N, prompt, seed)` pairs.
- 1.600 method records và 1.600 selected-image hashes đã được audit.
- Actual initial-latent hashes và DDIM generator traces match trong mọi pair.
- 17 structural/fairness tests và smoke gate đã pass trước main run.
- Recomputed means, std, paired median/mean, win fraction và bootstrap intervals
  khớp report đến `1e-12`.
- Không retune sau khi xem kết quả; historical BFS scores không được chèn vào
  paired table.

Machine-readable artifacts:

- [Full result table](results/t2i-search/soft-vasr/results_table.csv)
- [N=4 summary](results/t2i-search/soft-vasr/n4_summary.json)
- [N=8 summary](results/t2i-search/soft-vasr/n8_summary.json)
- [N=4 mechanism](results/t2i-search/soft-vasr/n4_mechanism.csv)
- [N=8 mechanism](results/t2i-search/soft-vasr/n8_mechanism.csv)

## Limitations

1. TPU BF16 thay cho FP16 của paper vì FP16 canary trên TPU tạo output
   non-finite/black. Vì vậy đây là backend reproduction, không phải bitwise paper run.
2. Bốn-seed set có một seed reconstructed (`45`) do launcher commit thiếu seed thứ tư.
3. Main BFS–VASR comparison thay đổi cả allocation lẫn resampler; riêng thí
   nghiệm này không xác định causal allocation effect.
4. Soft VASR chạy trong hook/timing của Classical Search, không phải native VASR pipeline.
5. Runtime scope không bao gồm mọi end-to-end overhead và không tương đương phép
   đo wall-clock H200 của paper.

Các mechanism/continuation-variance follow-up được thực hiện sau main study,
nhưng không thuộc entrypoint tối giản trong branch này. Không dùng chúng để sửa
tham số hoặc diễn giải lại paired result ở trên.
