# -*- coding: utf-8 -*-
"""Elastic net từ đầu, theo ĐÚNG quy ước của Zou & Hastie 2005:

        L(λ₁, λ₂, β) = |y − Xβ|² + λ₂|β|² + λ₁|β|₁          — công thức (3)

Ba phương pháp là ba góc của cùng một bài toán: λ₂ = 0 cho lasso, λ₁ = 0 cho ridge.
Chạy trong dưới một giây; không cần thư viện nào ngoài numpy.
"""
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")


def chuan_hoa(X, y):
    """Quy ước (2) tr.303: y trung bình 0, mỗi cột trung bình 0 và Σx² = 1."""
    Xc = X - X.mean(0)
    return Xc / np.sqrt((Xc ** 2).sum(0)), y - y.mean()


def soft(z, g):
    """Ngưỡng mềm: nghiệm của  min_b (b − z)² + 2g|b|."""
    return np.sign(z) * np.maximum(np.abs(z) - g, 0.0)


def enet(X, y, l1, l2, b0=None, tol=1e-13, max_sweep=100000):
    """Hạ theo toạ độ cho (3). Trả (β của bản NAIVE, số lượt quét).

    Cập nhật một toạ độ: giữ nguyên các hệ số khác, bài toán còn lại một chiều và có
    nghiệm đóng — đó là lý do vòng lặp này không cần bất kỳ phép giải hệ nào.
    b0 là KHỞI ĐỘNG ẤM: quét một lưới λ₁ từ lớn xuống nhỏ thì mỗi bước chỉ còn vài lượt.
    """
    p = X.shape[1]
    b = np.zeros(p) if b0 is None else b0.copy()
    r = y - X @ b
    nn = (X ** 2).sum(0)
    for sweep in range(1, max_sweep + 1):
        lon = 0.0
        for j in range(p):
            z = X[:, j] @ r + nn[j] * b[j]          # hệ số OLS một chiều của phần dư
            moi = soft(z, l1 / 2.0) / (nn[j] + l2)  # so với (6): ngưỡng TRƯỚC, chia SAU
            if moi != b[j]:
                r -= X[:, j] * (moi - b[j])
                lon = max(lon, abs(moi - b[j]))
                b[j] = moi
        if lon < tol:
            return b, sweep
    raise RuntimeError("chưa hội tụ sau %d lượt quét (lệch %.2e)" % (max_sweep, lon))


def enet_sua(X, y, l1, l2):
    """Bản ĐÃ NHÂN LẠI của bài báo — công thức (12): β̂(EN) = (1+λ₂)·β̂(naive)."""
    b, _ = enet(X, y, l1, l2)
    return (1.0 + l2) * b


def qua_du_lieu_tang_cuong(X, y, l1, l2):
    """Bổ đề 1 tr.304: giải bằng cách dựng (y*, X*) rồi chạy MỘT bài toán lasso."""
    n, p = X.shape
    Xs = np.vstack([X, np.sqrt(l2) * np.eye(p)]) / np.sqrt(1.0 + l2)
    ys = np.r_[y, np.zeros(p)]
    gam = l1 / np.sqrt(1.0 + l2)
    bs, _ = enet(Xs, ys, gam, 0.0)                  # λ₂ = 0 ⇒ đúng bài toán lasso
    return bs / np.sqrt(1.0 + l2)                   # β̂(naive) = β̂*/√(1+λ₂)


def ridge_dong(X, y, l2):
    p = X.shape[1]
    return np.linalg.solve(X.T @ X + l2 * np.eye(p), X.T @ y)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n, p = 60, 10
    X = rng.normal(size=(n, p))
    X[:, 1] = X[:, 0] + 0.02 * rng.normal(size=n)   # cột 1 gần trùng cột 0
    X[:, 2] = X[:, 0] + 0.02 * rng.normal(size=n)   # cột 2 cũng vậy
    beta = np.zeros(p)
    beta[[0, 1, 2, 6]] = [1.0, 1.0, 1.0, -2.0]
    y = X @ beta + 0.5 * rng.normal(size=n)
    X, y = chuan_hoa(X, y)
    print("n = %d, p = %d ; ba cột 0,1,2 tương quan %.4f ; hệ số thật khác 0 ở cột 0,1,2,6\n"
          % (n, p, np.corrcoef(X[:, 0], X[:, 1])[0, 1]))

    # 1. λ₂ = 0 phải cho đúng lasso; λ₁ = 0 phải cho đúng ridge dạng đóng
    for l2 in (0.05, 0.5, 5.0):
        b_lap, _ = enet(X, y, 0.0, l2)
        print("   λ₁ = 0, λ₂ = %-5g: hạ theo toạ độ so với nghiệm ridge dạng đóng — lệch %.2e"
              % (l2, np.abs(b_lap - ridge_dong(X, y, l2)).max()))

    # 2. Bổ đề 1: hai đường tính hoàn toàn khác nhau phải ra cùng một nghiệm
    print()
    for l1, l2 in [(0.2, 0.5), (0.05, 2.0)]:
        a = enet(X, y, l1, l2)[0]
        b = qua_du_lieu_tang_cuong(X, y, l1, l2)
        print("   Bổ đề 1, λ₁ = %-5g λ₂ = %-4g: hạ toạ độ so với dữ liệu tăng cường — lệch %.2e"
              % (l1, l2, np.abs(a - b).max()))
        assert np.allclose(a, b, atol=1e-10)

    # 3. ba phương pháp trên cùng bộ dữ liệu
    print("\n   Cùng mức phạt, ba nghiệm khác hẳn nhau (in 4 chữ số):")
    print("      %-26s %s" % ("cột", "  ".join("%7d" % j for j in range(p))))
    print("      %-26s %s" % ("hệ số thật", "  ".join("%7.2f" % v for v in beta)))
    for ten, l1, l2 in [("Lasso   (λ₂ = 0)", 0.2, 0.0),
                        ("Naive EN (λ₁,λ₂ = 0,2;1)", 0.2, 1.0),
                        ("Elastic net, đã nhân lại", 0.2, 1.0),
                        ("Ridge   (λ₁ = 0)", 0.0, 1.0)]:
        b = enet_sua(X, y, l1, l2) if ten.startswith("Elastic") else enet(X, y, l1, l2)[0]
        print("      %-26s %s   khác 0: %2d/%d"
              % (ten, "  ".join("%7.4f" % v for v in b), int((np.abs(b) > 1e-9).sum()), p))

    # 4. hiệu ứng nhóm: độ trải của ba hệ số gần trùng nhau
    print("\n   Độ trải giữa ba hệ số của ba cột gần trùng nhau (càng nhỏ càng gom nhóm):")
    for l2 in (0.0, 0.01, 0.1, 1.0, 10.0):
        b = enet(X, y, 0.2, l2)[0][:3]
        print("      λ₂ = %-6g β̂ = [%8.4f %8.4f %8.4f]   độ trải = %.4f"
              % (l2, b[0], b[1], b[2], b.max() - b.min()))

    # 5. trần n biến: chỉ phần λ₂ mới vượt qua được
    print("\n   Khi p > n: số hệ số khác 0 lớn nhất mà mỗi phương pháp đạt được")
    rng2 = np.random.default_rng(7)
    n2, p2 = 20, 60
    X2 = rng2.normal(size=(n2, p2))
    y2 = X2 @ np.r_[np.full(8, 2.0), np.zeros(p2 - 8)] + 0.3 * rng2.normal(size=n2)
    X2, y2 = chuan_hoa(X2, y2)
    for ten, l2 in [("lasso      (λ₂ = 0)", 0.0), ("elastic net (λ₂ = 0,1)", 0.1),
                    ("elastic net (λ₂ = 1)", 1.0)]:
        tot, b_am, dung = 0, None, ""
        for l1 in np.logspace(1, -4, 30):           # từ lớn xuống nhỏ, khởi động ấm
            try:
                b_am, _ = enet(X2, y2, l1, l2, b0=b_am, tol=1e-9, max_sweep=20000)
            except RuntimeError:
                dung = "  (dừng ở λ₁ = %.4g: chưa hội tụ — đúng bệnh của lasso khi cột gần phụ thuộc)" % l1
                break
            tot = max(tot, int((np.abs(b_am) > 1e-9).sum()))
        print("      %-24s %3d  (n = %d, p = %d)%s" % (ten, tot, n2, p2, dung))
