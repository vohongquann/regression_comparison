# -*- coding: utf-8 -*-
"""Ridge, Lasso, Elastic Net từ đầu — cùng một vòng lặp toạ độ, khác đúng một dòng."""
import sys

import numpy as np

# stdout của Windows mặc định là cp1252, không in nổi tiếng Việt có dấu. Thiếu dòng
# này thì script GÃY ngay ở lệnh print đầu tiên — và bài Ridge mục 6 lại trích dẫn
# chính nó làm nguồn cho bảng đối chiếu. Phát hiện ngày 2026-08-31.
sys.stdout.reconfigure(encoding="utf-8")


def soft(z, g):
    """Toán tử ngưỡng mềm — nghiệm đóng của  min ½(b−z)² + γ|b|."""
    return np.sign(z) * np.maximum(np.abs(z) - g, 0.0)


def enet(X, y, lam=1.0, alpha=1.0, n_iter=200000, tol=1e-12, return_iters=False):
    """Hạ theo toạ độ cho  ½‖y−Xβ‖² + λ(α‖β‖₁ + ½(1−α)‖β‖²).

    alpha = 1 -> Lasso;  alpha = 0 -> Ridge;  ở giữa -> Elastic Net.
    Giả định các cột của X đã được chuẩn hoá (trung bình 0).

    BÁO LỖI nếu hết n_iter mà chưa đạt tol. Không bao giờ trả về nghiệm chưa hội
    tụ một cách im lặng — đó là lỗi đã thật sự xảy ra trong repo này, xem mục 4.1.
    """
    n, p = X.shape
    b = np.zeros(p)
    r = y.copy()                               # phần dư, cập nhật dần cho rẻ
    nrm = (X ** 2).sum(0)                      # ‖xⱼ‖² — tính một lần
    for it in range(1, n_iter + 1):
        delta = 0.0
        for j in range(p):
            if nrm[j] == 0:
                continue
            r += X[:, j] * b[j]                # bỏ đóng góp của j ra khỏi phần dư
            z = X[:, j] @ r
            new = soft(z, lam * alpha) / (nrm[j] + lam * (1 - alpha))
            delta = max(delta, abs(new - b[j]))
            b[j] = new
            r -= X[:, j] * b[j]                # đưa lại vào
        if delta < tol:
            return (b, it) if return_iters else b
    raise RuntimeError("chưa hội tụ sau %d vòng (delta = %.2e > tol = %.0e)"
                       % (n_iter, delta, tol))


def ridge_closed(X, y, lam):
    """Ridge có nghiệm đóng — dùng để đối chiếu với vòng lặp toạ độ."""
    p = X.shape[1]
    return np.linalg.solve(X.T @ X + lam * np.eye(p), X.T @ y)


def demo():
    rng = np.random.default_rng(0)
    n, p = 80, 12
    X = rng.normal(size=(n, p))
    X = (X - X.mean(0)) / X.std(0)
    beta = np.zeros(p); beta[[0, 3, 7]] = [3.0, -2.0, 1.5]     # chỉ 3/12 hệ số khác 0
    y = X @ beta + rng.normal(size=n) * 1.0
    y = y - y.mean()

    print("n = %d, p = %d, hệ số thật khác 0 ở cột 0, 3, 7\n" % (n, p))

    # --- 1. vòng lặp toạ độ với alpha=0 phải trùng nghiệm đóng của ridge ---
    for lam in (0.1, 1.0, 10.0, 100.0):
        a = enet(X, y, lam=lam, alpha=0.0)
        b = ridge_closed(X, y, lam)
        print("   ridge λ=%-6g  vòng lặp toạ độ vs nghiệm đóng: lệch %.3e"
              % (lam, np.abs(a - b).max()))

    # --- 2. ba bộ trên cùng λ ---
    print("\nCùng λ = 8, ba bộ cho ba nghiệm khác hẳn nhau:")
    print("   %-22s %s" % ("", "  ".join("%6d" % j for j in range(p))))
    print("   %-22s %s" % ("hệ số thật", "  ".join("%6.2f" % v for v in beta)))
    for name, al in [("Ridge   (α = 0)", 0.0), ("Elastic Net (α=0.5)", 0.5), ("Lasso   (α = 1)", 1.0)]:
        b, it = enet(X, y, lam=8.0, alpha=al, return_iters=True)
        print("   %-22s %s   khác 0: %2d/%d  (%d vòng)"
              % (name, "  ".join("%6.2f" % v for v in b), (np.abs(b) > 1e-8).sum(), p, it))

    # --- 3. lasso thực sự đưa hệ số về ĐÚNG 0, ridge thì không ---
    bl = enet(X, y, lam=8.0, alpha=1.0)
    br = enet(X, y, lam=8.0, alpha=0.0)
    print("\n   Hệ số nhỏ nhất khác 0 của lasso: %.3e   (số hệ số đúng bằng 0: %d)"
          % (np.abs(bl[bl != 0]).min(), int((bl == 0).sum())))
    print("   Hệ số nhỏ nhất của ridge       : %.3e   (số hệ số đúng bằng 0: %d)"
          % (np.abs(br).min(), int((br == 0).sum())))
    assert (bl == 0).sum() > 0, "lasso phải cho hệ số bằng đúng 0"
    assert (br == 0).sum() == 0, "ridge không bao giờ cho đúng 0"

    # --- 4. đường đi của nghiệm khi λ tăng ---
    print("\n   Số biến lasso giữ lại khi λ tăng:")
    for lam in (0.5, 2, 8, 32, 128, 512):
        b = enet(X, y, lam=lam, alpha=1.0)
        sup = [j for j in range(p) if abs(b[j]) > 1e-8]
        print("      λ = %-6g -> %2d biến  %s" % (lam, len(sup), sup if len(sup) <= 5 else ""))
    print("      (giá đỡ thật là [0, 3, 7]; λ = 32 tìm ra ĐÚNG ba biến đó)")

    # --- 5. nhóm biến gần trùng nhau: chỗ lasso và elastic net khác nhau rõ nhất ---
    Xg = X.copy()
    Xg[:, 1] = X[:, 0] + 0.01 * rng.normal(size=n)      # cột 1 gần trùng cột 0
    Xg[:, 2] = X[:, 0] + 0.01 * rng.normal(size=n)      # cột 2 cũng vậy
    Xg = (Xg - Xg.mean(0)) / Xg.std(0)
    yg = Xg[:, :3].sum(1) + rng.normal(size=n) * 0.5
    yg = yg - yg.mean()
    print("\n   Ba cột 0,1,2 gần trùng nhau (tương quan %.4f), đóng góp thật chia đều:"
          % np.corrcoef(Xg[:, 0], Xg[:, 1])[0, 1])
    for name, al in [("Ridge  ", 0.0), ("ElNet  ", 0.4), ("Lasso  ", 1.0)]:
        b, it = enet(Xg, yg, lam=6.0, alpha=al, return_iters=True)
        print("      %s hệ số ba cột nhóm = [%7.4f %7.4f %7.4f]   hội tụ sau %6d vòng"
              % (name, b[0], b[1], b[2], it))


if __name__ == "__main__":
    demo()
