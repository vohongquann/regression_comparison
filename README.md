# Regression Comparison

Project nghiên cứu và so sánh 9 thuật toán hồi quy tuyến tính: **Linear, Polynomial, Ridge, Lasso,
Elastic Net, LARS, Huber, Quantile, Bayesian Linear**.

```
docs/      9 bài viết (.md + hình), chép nguyên văn từ repo AI-Algorithm
checks/    script kiểm chứng của 9 bài, chép nguyên văn từ AI-Algorithm/tools
src/       code so sánh: data.py · models.py · compare.py
scripts/   sync_docs.py — đồng bộ docs/ từ AI-Algorithm
results/   đầu ra của compare.py và của checks/run_all.py
```

## docs/ — đồng bộ một chiều

Sửa bài ở `AI-Algorithm`, rồi chạy:

```
python scripts/sync_docs.py --dry-run   # xem sẽ đổi gì
python scripts/sync_docs.py             # chép file mới/đổi
python scripts/sync_docs.py --prune     # xoá luôn file nguồn không còn
```

Đừng sửa tay trong `docs/`: lần đồng bộ sau sẽ ghi đè. Các liên kết tương đối trỏ ra ngoài 9 bài
(`../../_papers/…`, `../../../tools/…`, bài của nhóm khác) sẽ không mở được ở đây vì file `.md` được giữ nguyên.

## src/ — so sánh 9 thuật toán

```
python -m src.compare                          # cả hai bộ dữ liệu, CV 5 fold × 3 lần
python -m src.compare --data diabetes --quad   # mọi thuật toán chạy trên cột nở bậc 2
python -m src.compare --only Ridge Lasso LARS
```

| Bộ dữ liệu | Cỡ | Vì sao chọn |
|---|---|---|
| `diabetes` | 442 × 10 | Đúng bộ của Efron et al. 2004 (bài LARS), các bài Lasso/Elastic Net cũng chạy trên nó. Nhỏ, cột s1–s5 tương quan mạnh: thấy rõ Ridge co đều, Lasso chọn một, Elastic Net giữ nhóm. Bậc 2 ra 65 cột, gần mô hình m=64 của bài báo. |
| `california` | 5 000 × 8 (lấy mẫu từ 20 640) | Giá bị chặn ở 5,0 nên nhãn có cụm ngoại lai, nơi Huber và Quantile khác bình phương tối thiểu. Quan hệ phi tuyến nên Polynomial có việc làm. |

Mỗi thuật toán tự chọn siêu tham số bằng CV bên trong; CV ngoài dùng cùng các fold cho mọi thuật toán,
nên so sánh là so sánh ghép cặp.

## checks/ — script kiểm chứng

Các `v_*.py` dựng lại từng con số trong bài từ nguồn gốc (bài báo, sách). Chạy riêng từng file:
`cd checks && python v_linreg.py`. `run_all.py` chạy tất cả rồi đối chiếu với `_out_*.txt` (đầu ra lần chạy gốc).

Ngoài `v_*.py`, thư mục này còn có (chép nguyên văn, chưa chạy):

| Nhóm | File |
|---|---|
| Kiểm bài viết | `khoi7` `luat1` `xungho` `bangcot` `fixfence` `checklinks` `congthuc` `kytu` `danchieu` `kiempdf` |
| Đối chiếu bài báo | `kiemtrich` `sopaper` `sohinh` `pdfscan` |
| Sinh hình cho bài | `fig_*` (9 bài) |
| Demo | `demo_enet` `demo_shrink` |

Các script đọc PDF ở `AI algorithms/_papers/`, đọc bài ở `AI algorithms/Machine Learning/` hoặc ghi hình vào đó
đều tính đường dẫn theo bố cục của repo gốc. Bố cục này không có trong project, nên các script đó chưa chạy được ở đây.
