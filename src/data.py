"""Hai bộ dữ liệu dùng để học 9 thuật toán hồi quy. Cả hai nạp thẳng từ scikit-learn, không cần tải tay.

diabetes   442 dòng × 10 cột. Chính là bộ dữ liệu của Efron et al. 2004 (bài LARS) — đường đi hệ số
           của Lasso/LARS/Ridge/Elastic Net chạy trên đúng bộ này. Nhỏ, ít nhiễu cấu trúc, tương
           quan mạnh giữa s1–s5 (nên thấy rõ Ridge co đều, Lasso chọn một, Elastic Net giữ nhóm).
           Bậc 2 của 10 cột ra 65 cột — cỡ mô hình m=64 của bài báo — nên là chỗ thấy Polynomial
           quá khớp và chính quy hoá cứu nó.

california 20 640 dòng × 8 cột, lấy mẫu 5 000 dòng cho nhanh (đặt seed). Giá nhà bị chặn trên ở 5,0
           (kiểm duyệt) nên nhãn có một cụm ngoại lai ở đỉnh — chỗ Huber và Quantile khác bình
           phương tối thiểu. Quan hệ phi tuyến nên Polynomial có việc để làm.
"""
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing, load_diabetes

DATASETS = ["diabetes", "california"]
CALIFORNIA_N = 5_000
SEED = 0


def load(name: str):
    """Trả về (X: DataFrame, y: Series) với y ở đơn vị gốc của bộ dữ liệu."""
    if name == "diabetes":
        d = load_diabetes(as_frame=True)
        return d.data, d.target
    if name == "california":
        d = fetch_california_housing(as_frame=True)
        idx = np.random.RandomState(SEED).choice(len(d.data), CALIFORNIA_N, replace=False)
        X = d.data.iloc[idx].reset_index(drop=True)
        # Bốn cột lệch phải nặng; thô thì một dòng AveOccup=1243 nằm ở fold test làm RMSE của
        # mọi mô hình tuyến tính nổ (R2 ≈ -18). log1p là phép đơn điệu, không dùng nhãn.
        for c in ("AveRooms", "AveBedrms", "Population", "AveOccup"):
            X[c] = np.log1p(X[c])
        return X, d.target.iloc[idx].reset_index(drop=True)
    raise ValueError(f"không có bộ dữ liệu '{name}', chọn một trong {DATASETS}")


def describe(name: str) -> pd.DataFrame:
    X, y = load(name)
    return pd.concat([X, y.rename("target")], axis=1).describe().T
