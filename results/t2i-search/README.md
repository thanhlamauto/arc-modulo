# Compact published results

Thư mục này chỉ chứa bảng/summary nhỏ để kiểm tra kết quả đã công bố. Raw JSONL,
1.600 selected images, model weights, cache và environment không được đưa lên Git.

| File | SHA256 |
|---|---|
| `classical-search/table1_reproduced.csv` | `8e1fc2707f1da01e15d9e339de2dcc133fe171e878d3994a8e940ce2cb2a8118` |
| `classical-search/table2_reproduced.csv` | `6c861ff5e80d38f61e69563d90508f5e7d4fd3d25e21177904ac325f9249b272` |
| `soft-vasr/results_table.csv` | `d88ce60fdfaec7fac69eaafccfe2b8d5854e23968893f118d80b64fa8dc039d7` |
| `soft-vasr/n4_summary.json` | `530d6d0dcb0aa23209be63b8cfd238c03d306c64812094c556af8571593108e5` |
| `soft-vasr/n8_summary.json` | `d1f6a47202dfae3868833a519f12443ac41719de5bcc8711dacc392fb3c6e1eb` |
| `soft-vasr/n4_mechanism.csv` | `d526249deaf8534c64f67fb982b9df3e3c09bb8de7aade0934f42af929b311c5` |
| `soft-vasr/n8_mechanism.csv` | `6d494a6f9cdbb77ad16389c0566c99281c2813cbbf7def574405103b56522f21` |

`table1_reproduced.csv` cũng là frozen gamma-selection evidence cho paired run.
Nó phải giữ nguyên LF bytes và hash nêu trên.
