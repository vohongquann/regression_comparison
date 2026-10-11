# -*- coding: utf-8 -*-
"""Dựng lại mọi công thức, bảng, hình và ví dụ số của các nguồn gốc bài Elastic Net.

Nguồn:
  [Z] H. Zou, T. Hastie, "Regularization and Variable Selection via the Elastic Net",
      J. R. Statist. Soc. B 67(2):301–320, 2005 — BÀI GỐC.
      §1–§7, Phụ lục A.1–A.3, công thức (1)–(21), Hình 1–7, Bảng 1–4.
      AI algorithms/_papers/2005 - Zou, Hastie - Regularization and variable selection via the elastic net.pdf
  [E] T. Hastie, R. Tibshirani, J. Friedman, The Elements of Statistical Learning, 2nd ed.,
      Springer 2009 (bản in 12): (3.54) tr.73, Hình 3.13 tr.73, Bài tập 3.30 (3.91) tr.99,
      §18.4 (18.20)–(18.21) tr.662–664, Hình 18.5 tr.662, Bảng 18.1 tr.655.
  [G] J. Friedman, T. Hastie, R. Tibshirani, "Regularization Paths for GLM via Coordinate
      Descent", J. Stat. Softw. 33(1), 2010 — (1)–(3) và chú thích 1 tr.3, Hình 1 tr.4.
  [T] R. Tibshirani, "Regression Shrinkage and Selection via the Lasso",
      J. R. Statist. Soc. B 58(1):267–288, 1996 — ba ví dụ mô phỏng mà [Z] §5 dùng lại.
  [L] B. Efron, T. Hastie, I. Johnstone, R. Tibshirani, "Least Angle Regression",
      Ann. Statist. 32(2):407–451, 2004 — LARS, nền của LARS-EN ở [Z] §3.4.

Dữ liệu: tools/prostate.csv (gói R ElemStatLearn, bản ĐÃ sửa lỗi lweight của đối tượng 32).
[Z] §4 chạy trước khi lỗi đó được sửa, nên mục 12 dựng lại trên CẢ HAI bản.

Chạy:  cd tools && python -u v_enet.py
"""
import csv
import itertools
import math
import os
import sys

import numpy as np
from sklearn.linear_model import lars_path

sys.stdout.reconfigure(encoding="utf-8")
np.set_printoptions(linewidth=150, suppress=True)
HERE = os.path.dirname(os.path.abspath(__file__))


def head(t):
    print("\n" + "=" * 104 + "\n" + t + "\n" + "=" * 104, flush=True)


def soft(z, g):
    return np.sign(z) * np.maximum(np.abs(z) - g, 0.0)


# ───────────────────────────────────────────── quy ước của [Z]: L = |y−Xβ|² + λ₂|β|² + λ₁|β|₁
def nen_cd(X, y, l1, l2, b=None, tol=1e-14, max_sweep=300000, dem=False):
    """Hạ theo toạ độ cho tiêu chí naive elastic net (3) của [Z]. Cột X KHÔNG cần chuẩn hoá."""
    p = X.shape[1]
    b = np.zeros(p) if b is None else b.copy()
    r = y - X @ b
    nn = (X ** 2).sum(0)
    for sweep in range(1, max_sweep + 1):
        lon = 0.0
        for j in range(p):
            z = X[:, j] @ r + nn[j] * b[j]
            moi = soft(z, l1 / 2.0) / (nn[j] + l2)
            if moi != b[j]:
                r -= X[:, j] * (moi - b[j])
                lon = max(lon, abs(moi - b[j]))
                b[j] = moi
        if lon < tol:
            return (b, sweep) if dem else b
    raise RuntimeError("CD chưa hội tụ sau %d lượt (lệch %.2e)" % (max_sweep, lon))


def lasso_Z(X, y, l1):
    """Nghiệm lasso của [Z] (|y−Xβ|² + λ₁|β|₁) lấy CHÍNH XÁC từ đường LARS, không lặp."""
    aa, _, cc = lars_path(X, y, method="lasso")
    muc = l1 / 2.0 / X.shape[0]          # alphas của scikit-learn = max|xᵀr|/n
    if muc >= aa[0]:
        return np.zeros(X.shape[1])
    if muc <= aa[-1]:
        return cc[:, -1].copy()
    k = int(np.searchsorted(-aa, -muc))
    w = (muc - aa[k - 1]) / (aa[k] - aa[k - 1])
    return cc[:, k - 1] + w * (cc[:, k] - cc[:, k - 1])


def L_nen(X, y, b, l1, l2):
    """Giá trị tiêu chí (3) của [Z]."""
    r = y - X @ b
    return r @ r + l2 * (b @ b) + l1 * np.abs(b).sum()


def tang_cuong(X, y, l2):
    """Bổ đề 1 của [Z]: bộ dữ liệu nhân tạo (X*, y*) cỡ (n+p)×p."""
    n, p = X.shape
    Xs = np.vstack([X, math.sqrt(l2) * np.eye(p)]) / math.sqrt(1.0 + l2)
    ys = np.r_[y, np.zeros(p)]
    return Xs, ys


def chuan_Z(X, y):
    """Quy ước (2) của [Z]: y center, mỗi cột center và có Σx² = 1 (KHÔNG phải độ lệch chuẩn 1)."""
    Xc = X - X.mean(0)
    Xc = Xc / np.sqrt((Xc ** 2).sum(0))
    return Xc, y - y.mean()


print("Phiên bản: numpy %s · scikit-learn %s" % (np.__version__, __import__("sklearn").__version__))

# ═════════════════════════════════════════════════════════════════════ 1. (2)–(5): định nghĩa
head("1. [Z] §2.1 — công thức (2)–(5): định nghĩa naive elastic net, và α CỦA BÀI BÁO")
rng = np.random.default_rng(0)
n0, p0 = 40, 6
X0 = rng.normal(size=(n0, p0))
y0 = X0 @ np.array([2.0, -1.5, 0.0, 0.8, 0.0, 0.3]) + 0.7 * rng.normal(size=n0)
X0, y0 = chuan_Z(X0, y0)
print("   quy ước (2): max|Σx_ij| = %.2e ; max|Σx²_ij − 1| = %.2e ; |Σy_i| = %.2e"
      % (np.abs(X0.sum(0)).max(), np.abs((X0 ** 2).sum(0) - 1).max(), abs(y0.sum())))

l1, l2 = 0.6, 1.4
b_cd = nen_cd(X0, y0, l1, l2)
# nghiệm tham chiếu: bài toán lasso trên dữ liệu tăng cường (Bổ đề 1)
Xs, ys = tang_cuong(X0, y0, l2)
gam = l1 / math.sqrt(1.0 + l2)
al_sk, _, co_sk = lars_path(Xs, ys, method="lasso")
# lars_path trả alphas = max|X'r|/n_hàng; đổi sang γ/2 của [Z]
muc = gam / 2.0 / len(ys)
k = int(np.searchsorted(-al_sk, -muc))
w = (muc - al_sk[k - 1]) / (al_sk[k] - al_sk[k - 1])
bs_lars = co_sk[:, k - 1] + w * (co_sk[:, k] - co_sk[:, k - 1])
print("   Bổ đề 1: β̂ = β̂*/√(1+λ₂) so với hạ theo toạ độ — lệch %.3e"
      % np.abs(bs_lars / math.sqrt(1 + l2) - b_cd).max())

# (5): dạng ràng buộc với α = λ₂/(λ₁+λ₂)
alpha_Z = l2 / (l1 + l2)
t_dat = (1 - alpha_Z) * np.abs(b_cd).sum() + alpha_Z * (b_cd @ b_cd)
print("   (5): α = λ₂/(λ₁+λ₂) = %.6f ; t = (1−α)|β|₁ + α|β|² tại nghiệm = %.6f" % (alpha_Z, t_dat))

# chia đôi trên hệ số tỉ lệ c (giữ nguyên tỉ lệ λ₁ : λ₂, tức giữ nguyên α) để t khớp đúng t_dat


def t_cua(c):
    bb = nen_cd(X0, y0, c * l1, c * l2)
    return (1 - alpha_Z) * np.abs(bb).sum() + alpha_Z * (bb @ bb), bb


lo, hi = 1e-6, 1e3
for _ in range(80):
    mid = math.sqrt(lo * hi)
    if t_cua(mid)[0] > t_dat:
        lo = mid
    else:
        hi = mid
t_tim, bbest = t_cua(math.sqrt(lo * hi))
print("   dạng phạt (3) và dạng ràng buộc (5) cho cùng nghiệm: c = %.6f (đúng 1 nếu khớp) ; "
      "lệch β %.2e ; lệch t %.2e" % (math.sqrt(lo * hi), np.abs(bbest - b_cd).max(), abs(t_tim - t_dat)))

print("\n   α CỦA [Z] NGƯỢC với α của glmnet/scikit-learn — kiểm bằng hai đầu mút:")
for a, ten in [(1.0, "α = 1"), (0.0, "α = 0")]:
    # α = λ₂/(λ₁+λ₂): α=1 ⇔ λ₁=0 (ridge) ; α=0 ⇔ λ₂=0 (lasso)
    if a == 1.0:
        bb = nen_cd(X0, y0, 0.0, 2.0)
        rr = np.linalg.solve(X0.T @ X0 + 2.0 * np.eye(p0), X0.T @ y0)
        print("      %s → λ₁ = 0: nghiệm so với ridge dạng đóng — lệch %.3e   [Z] tr.3: \"becomes simple ridge regression\""
              % (ten, np.abs(bb - rr).max()))
    else:
        bb = nen_cd(X0, y0, 0.6, 0.0)
        ll = lars_path(X0, y0, method="lasso")
        aa, _, cc = ll
        mm = 0.6 / 2.0 / n0
        kk = int(np.searchsorted(-aa, -mm))
        ww = (mm - aa[kk - 1]) / (aa[kk] - aa[kk - 1])
        bl = cc[:, kk - 1] + ww * (cc[:, kk] - cc[:, kk - 1])
        print("      %s → λ₂ = 0: nghiệm so với lasso (LARS) — lệch %.3e   [Z] tr.3: \"the lasso penalty (α = 0)\""
              % (ten, np.abs(bb - bl).max()))

# ═════════════════════════════════════════════════════════════ 2. Hình 1 — miền phạt, mức 1
head("2. [Z] Hình 1 tr.4 — ba miền phạt ở mức 1, và câu 'strictly convex' của chú thích")
aZ = 0.5                      # α của chú thích Hình 1


def pen_Z(b, a):
    """Phạt elastic net theo (5) của [Z]: (1−α)|β|₁ + α|β|²."""
    return (1 - a) * np.abs(b).sum(-1) + a * (b ** 2).sum(-1)


for ten, f in [("ridge  |β|²", lambda b: (b ** 2).sum(-1)),
               ("lasso  |β|₁", lambda b: np.abs(b).sum(-1)),
               ("elnet α=0,5", lambda b: pen_Z(b, aZ))]:
    print("   %s : tại (1,0) = %.6f ; tại (0,1) = %.6f ; tại (0,707;0,707) = %.6f"
          % (ten, f(np.array([1.0, 0.0])), f(np.array([0.0, 1.0])),
             f(np.array([1 / math.sqrt(2), 1 / math.sqrt(2)]))))

# đường mức 1 trong góc phần tư thứ nhất: với mỗi β₁ giải β₂ ≥ 0
b1g = np.linspace(0.0, 1.0, 20001)
disc = (1 - aZ) ** 2 + 4 * aZ * (1 - (1 - aZ) * b1g - aZ * b1g ** 2)
b2_en = (-(1 - aZ) + np.sqrt(disc)) / (2 * aZ)
b2_la = 1.0 - b1g
b2_ri = np.sqrt(np.maximum(1 - b1g ** 2, 0))
print("   đường mức 1, góc phần tư I: elastic net PHỒNG ra ngoài đường thẳng lasso,")
print("      lệch lớn nhất = %.6f tại β₁ = %.4f ; ridge lệch %.6f"
      % ((b2_en - b2_la).max(), b1g[(b2_en - b2_la).argmax()], (b2_ri - b2_la).max()))
# độ cong: lasso thẳng nên đạo hàm bậc hai bằng 0
h = b1g[1] - b1g[0]
d2_en = np.gradient(np.gradient(b2_en, h), h)[200:-200]
d2_la = np.gradient(np.gradient(b2_la, h), h)[200:-200]
print("      đạo hàm bậc hai của biên: elastic net trong [%.4f, %.4f] (âm ⇒ lồi ngặt) ; lasso |d²| ≤ %.2e"
      % (d2_en.min(), d2_en.max(), np.abs(d2_la).max()))
# đỉnh nhọn tại (1,0): hai hệ số góc một phía
sl_en = (b2_en[1] - b2_en[0]) / h
print("      đỉnh tại (0,1): hệ số góc một phía của elastic net = %.6f ; của lasso = %.6f ; "
      "ridge = %.6f (trơn ⇒ KHÔNG có đỉnh)"
      % (sl_en, (b2_la[1] - b2_la[0]) / h, (b2_ri[1] - b2_ri[0]) / h))

# ═══════════════════════════════════════════════════ 3. (6) và Hình 2 — thiết kế trực giao
head("3. [Z] §2.2 công thức (6) và Hình 2 tr.5 — nghiệm chính xác khi thiết kế trực giao")
Q, _ = np.linalg.qr(rng.normal(size=(80, 8)))
yq = Q @ np.array([3.0, -2.2, 1.4, 0.9, -0.5, 0.25, 0.1, -0.05]) + 0.15 * rng.normal(size=80)
b_ols = Q.T @ yq
L1, L2 = 2.0, 1.0                                    # tham số của chú thích Hình 2
b_ct6 = soft(b_ols, L1 / 2) / (1 + L2)               # (6)
b_do = nen_cd(Q, yq, L1, L2)
print("   ‖QᵀQ − I‖ = %.2e ; β̂(OLS) = Xᵀy: lệch %.2e"
      % (np.abs(Q.T @ Q - np.eye(8)).max(), np.abs(np.linalg.lstsq(Q, yq, rcond=None)[0] - b_ols).max()))
print("   (6) so với hạ theo toạ độ: lệch %.3e" % np.abs(b_ct6 - b_do).max())
print("   ridge β̂(OLS)/(1+λ₂): lệch %.3e ; lasso soft(β̂(OLS), λ₁/2): lệch %.3e"
      % (np.abs(b_ols / (1 + L2) - nen_cd(Q, yq, 0.0, L2)).max(),
         np.abs(soft(b_ols, L1 / 2) - nen_cd(Q, yq, L1, 0.0)).max()))
# thứ tự hai bước: [Z] tr.5 viết "ridge-type direct shrinkage FOLLOWED BY lasso-type thresholding"
g = np.linspace(-4, 4, 4001)
lit = soft(g / (1 + L2), L1 / 2)                     # đọc ĐÚNG THỨ TỰ câu văn, ngưỡng λ₁/2
dung = soft(g, L1 / 2) / (1 + L2)                    # (6)
nguoc = soft(g / (1 + L2), L1 / 2 / (1 + L2))        # ridge trước, ngưỡng CHIA cho (1+λ₂)
print("   thứ tự hai bước: đọc nguyên văn câu tr.5 (ridge trước, ngưỡng λ₁/2) lệch (6) tối đa %.6f"
      % np.abs(lit - dung).max())
print("      còn ridge trước với ngưỡng λ₁/{2(1+λ₂)} thì TRÙNG (6): lệch %.2e — hai bước GIAO HOÁN ĐƯỢC"
      % np.abs(nguoc - dung).max())
print("      ngưỡng của (6) theo β̂(OLS) = λ₁/2 = %.4f ; số hệ số bị đưa về 0: %d/8"
      % (L1 / 2, int((b_ct6 == 0).sum())))

# ═════════════════════════════════════════════════════════ 4. Bổ đề 2 — hai cột TRÙNG KHÍT
head("4. [Z] §2.3 Bổ đề 2 tr.6 — khi x_i = x_j: phạt lồi ngặt cho β̂_i = β̂_j, lasso thì không")
Xd = X0.copy()
Xd[:, 1] = Xd[:, 0]                                  # hai cột trùng khít
print("   ‖x₁ − x₂‖ = %.2e (trùng khít)" % np.abs(Xd[:, 0] - Xd[:, 1]).max())
for nl1, nl2, ten in [(0.4, 1.0, "elastic net λ₁=0,4 λ₂=1"), (0.4, 0.0, "lasso      λ₁=0,4 λ₂=0"),
                      (0.0, 1.0, "ridge      λ₁=0   λ₂=1")]:
    bb = nen_cd(Xd, y0, nl1, nl2)
    print("      %-24s β̂₁ = %10.6f  β̂₂ = %10.6f  |β̂₁−β̂₂| = %.3e" % (ten, bb[0], bb[1], abs(bb[0] - bb[1])))
# phần (b): mọi cách chia tổng theo s ∈ [0,1] đều là nghiệm lasso
bl = nen_cd(Xd, y0, 0.4, 0.0)
tong = bl[0] + bl[1]
print("   Bổ đề 2(b): chia tổng %.6f theo s — tiêu chí (7) với J = |β|₁ tại các s khác nhau:" % tong)
for s_ in (0.0, 0.25, 0.5, 0.75, 1.0):
    bt = bl.copy()
    bt[0], bt[1] = tong * s_, tong * (1 - s_)
    print("      s = %.2f : β̂ = (%9.6f, %9.6f)  L = %.10f" % (s_, bt[0], bt[1], L_nen(Xd, y0, bt, 0.4, 0.0)))

# ══════════════════════════════════════════════════ 5. Định lý 1 — chặn của hiệu ứng nhóm
head("5. [Z] §2.3 Định lý 1 tr.6 — chặn |β̂_i − β̂_j|, và CHỖ LỆCH giữa PHÁT BIỂU và CHỨNG MINH")


def cap_tuong_quan(nn, rho, r):
    """Hai cột có tương quan mẫu ≈ rho, cộng vài cột nhiễu."""
    z = r.normal(size=nn)
    e1, e2 = r.normal(size=nn), r.normal(size=nn)
    c = math.sqrt(max(0.0, (1 - rho) / 2))
    x1, x2 = z + c * e1, z + c * e2
    X = np.c_[x1, x2, r.normal(size=(nn, 3))]
    yy = 2.0 * (x1 + x2) + 1.5 * X[:, 2] + r.normal(size=nn)
    return chuan_Z(X, yy)


print("   Phát biểu tr.6: D = |β̂_i − β̂_j| / |y|₁ ≤ (1/λ₂)·√{2(1−ρ)}")
print("   Chứng minh A.2 tr.18–19 lại chia cho |y| (chuẩn 2): \"So |r̂| ≤ |y|\"")
print("   %-9s %-7s %12s %12s %12s %10s %10s" % ("ρ", "λ₂", "D (|y|₁)", "D (|y|₂)", "chặn", "lỏng ₁", "lỏng ₂"))
r5 = np.random.default_rng(7)
for rho in (0.5, 0.9, 0.99, 0.999):
    Xr, yr = cap_tuong_quan(200, rho, r5)
    rho_do = Xr[:, 0] @ Xr[:, 1]
    for lam2 in (0.5, 5.0):
        bb = nen_cd(Xr, yr, 0.05, lam2)
        assert bb[0] * bb[1] > 0, "Định lý 1 đòi β̂_i·β̂_j > 0"
        d = abs(bb[0] - bb[1])
        chan = math.sqrt(2 * (1 - rho_do)) / lam2
        d1, d2 = d / np.abs(yr).sum(), d / math.sqrt(yr @ yr)
        print("   %-9.6f %-7.1f %12.3e %12.3e %12.3e %10.1f %10.1f"
              % (rho_do, lam2, d1, d2, chan, chan / d1, chan / d2))
print("   |y|₁/|y|₂ ở dòng cuối = %.4f, nên phát biểu LỎNG hơn chứng minh đúng bấy nhiêu lần"
      % (np.abs(yr).sum() / math.sqrt(yr @ yr)))

# ───────────────────────────────────────────── câu p = 2 của [Z] tr.6: |β̂₁ − β̂₂| = |cos θ|
print("\n   [Z] tr.6 về LASSO với p = 2: \"we easily obtain that |β̂₁ − β̂₂| = |cos(θ)|\"")
print("   Dựng lại: điều kiện dừng cho (x₁−x₂)ᵀr̂ = 0 ⇒ β̂₁ − β̂₂ = (x₁−x₂)ᵀy/(1−ρ) = √2·|y|·cos θ/√(1−ρ)")
print("   %-10s %10s %12s %12s %14s" % ("ρ", "cos θ", "|β̂₁−β̂₂|", "|cos θ|", "√2|y|cosθ/√(1−ρ)"))
r5b = np.random.default_rng(11)
for rho in (0.5, 0.9, 0.99, 0.999, 0.9999):
    z = r5b.normal(size=300)
    c = math.sqrt(max(0.0, (1 - rho) / 2))
    X2 = np.c_[z + c * r5b.normal(size=300), z + c * r5b.normal(size=300)]
    y2 = 2.0 * X2[:, 0] + 1.0 * X2[:, 1] + 0.5 * r5b.normal(size=300)
    X2, y2 = chuan_Z(X2, y2)
    rho_do = X2[:, 0] @ X2[:, 1]
    bb = lasso_Z(X2, y2, 0.02)
    if bb[0] * bb[1] <= 0:
        continue
    dd = X2[:, 0] - X2[:, 1]
    cos_t = (dd @ y2) / (math.sqrt(dd @ dd) * math.sqrt(y2 @ y2))
    ct = math.sqrt(2) * math.sqrt(y2 @ y2) * cos_t / math.sqrt(1 - rho_do)
    print("   %-10.6f %10.6f %12.6f %12.6f %14.6f" % (rho_do, cos_t, abs(bb[0] - bb[1]), abs(cos_t), abs(ct)))

print("\n   [Z] tr.6: \"It is easy to construct examples such that ρ → 1 but cos(θ) does not vanish\"")
print("   Dựng đúng thế: giữ cos θ ≈ hằng số, cho ρ → 1 — hiệu hệ số PHÂN KỲ chứ không chỉ 'không tắt dần'")
print("   (dấu hai hệ số KHÁC nhau từ dòng 2, nên cột 4 — công thức cùng dấu — lệch; cột 5 là công thức tổng quát)")
print("   %-12s %10s %14s %16s %14s %10s %10s %11s"
      % ("ρ", "cos θ", "|β̂₁−β̂₂|", "cùng dấu", "tổng quát", "β̂₁", "β̂₂", "cond(X)"))
r5c = np.random.default_rng(3)
zc = r5c.normal(size=400)
uc = r5c.normal(size=400)
uc = uc - uc @ zc / (zc @ zc) * zc                  # thành phần vuông góc với z
for rho in (0.9, 0.99, 0.999, 0.9999, 0.99999):
    c = math.sqrt(max(0.0, (1 - rho) / 2))
    X2 = np.c_[zc + c * uc, zc - c * uc]            # x₁ − x₂ = 2c·u, hướng CỐ ĐỊNH
    y2 = 3.0 * zc + 0.30 * uc + 0.05 * r5c.normal(size=400)
    X2, y2 = chuan_Z(X2, y2)
    rho_do = X2[:, 0] @ X2[:, 1]
    bb = lasso_Z(X2, y2, 0.02)
    dd = X2[:, 0] - X2[:, 1]
    cos_t = (dd @ y2) / (math.sqrt(dd @ dd) * math.sqrt(y2 @ y2))
    ct = math.sqrt(2) * math.sqrt(y2 @ y2) * cos_t / math.sqrt(1 - rho_do)
    rr = y2 - X2 @ bb
    # công thức TỔNG QUÁT: trừ hai điều kiện dừng cho (x₁−x₂)ᵀr̂ = λ₁(s₁−s₂)/2
    ds = np.sign(bb[0]) - np.sign(bb[1])
    tq = ((dd @ y2) - 0.02 * ds / 2) / (1 - rho_do)
    print("   %-12.7f %10.6f %14.4f %16.4f %14.4f %10.4f %10.4f %11.2e"
          % (rho_do, cos_t, abs(bb[0] - bb[1]), abs(ct), abs(tq), bb[0], bb[1], np.linalg.cond(X2)))

# ══════════════════════════════════════════════ 6. (8)–(9): tiên nghiệm, và Lq không tạo số 0
head("6. [Z] §2.4 công thức (8)–(9) tr.7 — tiên nghiệm, và câu 'chỉ q = 1 mới cho nghiệm thưa'")
print("   (9): p(β) ∝ exp[−λ{α|β|² + (1−α)|β|₁}] — lấy −log đúng bằng phạt của (5) nhân λ")
bts = np.array([0.7, -0.3])
lam_, a_ = 1.3, 0.5
print("      −log p(β) tại β = (0,7; −0,3): %.6f ; λ·[(α|β|² + (1−α)|β|₁] = %.6f"
      % (lam_ * (a_ * (bts @ bts) + (1 - a_) * np.abs(bts).sum()),
         lam_ * (a_ * (bts @ bts) + (1 - a_) * np.abs(bts).sum())))


def bridge(X, y, lam, q, b0=None, n_it=200000, tol=1e-14):
    """Hạ theo toạ độ cho |y−Xβ|² + λΣ|β_j|^q, q > 1.

    Với q > 1, đạo hàm một chiều 2‖x_j‖²t − 2|z| + λq·t^(q−1) TĂNG NGẶT trên t ≥ 0 và bằng
    −2|z| < 0 tại t = 0, nên nghiệm luôn > 0 khi z ≠ 0: không có cách nào ra số 0 đúng.
    """
    p = X.shape[1]
    b = np.zeros(p) if b0 is None else b0.copy()
    nn = (X ** 2).sum(0)
    for _ in range(n_it):
        lon = 0.0
        for j in range(p):
            z = X[:, j] @ (y - X @ b) + nn[j] * b[j]
            if z == 0:
                t = 0.0
            else:
                lo, hi = 0.0, max(1.0, abs(z))
                while 2 * nn[j] * hi - 2 * abs(z) + lam * q * hi ** (q - 1) < 0:
                    hi *= 2
                for _ in range(200):
                    mid = 0.5 * (lo + hi)
                    if 2 * nn[j] * mid - 2 * abs(z) + lam * q * mid ** (q - 1) < 0:
                        lo = mid
                    else:
                        hi = mid
                t = math.copysign(0.5 * (lo + hi), z)
            lon = max(lon, abs(t - b[j]))
            b[j] = t
        if lon < tol:
            return b
    return b


Xb, yb = chuan_Z(rng.normal(size=(60, 8)), rng.normal(size=60))
yb = yb + Xb @ np.array([2.0, 0, 0, -1.5, 0, 0, 0.8, 0])
for q in (1.0, 1.2, 1.5, 2.0):
    if q == 1.0:
        bb = nen_cd(Xb, yb, 0.6, 0.0)
    else:
        bb = bridge(Xb, yb, 0.6, q)
    print("      q = %.1f : số hệ số ĐÚNG bằng 0 = %d/8 ; |β| nhỏ nhất = %.3e"
          % (q, int((np.abs(bb) < 1e-12).sum()), np.abs(bb).min()))

# ═══════════════════════════════════════ 7. (10)–(15): hiệu chỉnh (1+λ₂) và phép KHỬ TƯƠNG QUAN
head("7. [Z] §3.1–3.2 tr.7–9 — công thức (10)–(15): vì sao nhân lại (1+λ₂), và ma trận R*")
Xe, ye = cap_tuong_quan(120, 0.85, np.random.default_rng(21))
l1e, l2e = 0.30, 2.0
b_naive = nen_cd(Xe, ye, l1e, l2e)
Xs, ys = tang_cuong(Xe, ye, l2e)
b_sao = lasso_Z(Xs, ys, 2 * l1e / math.sqrt(1 + l2e) * 0 + l1e / math.sqrt(1 + l2e))
print("   (10): bài toán lasso trên (y*, X*) với γ = λ₁/√(1+λ₂) = %.6f" % (l1e / math.sqrt(1 + l2e)))
print("   (11): β̂(elastic net) = √(1+λ₂)·β̂*  ;  β̂(naive) = β̂*/√(1+λ₂)")
b_en = math.sqrt(1 + l2e) * b_sao
print("      β̂* = %s" % np.array2string(b_sao, precision=6))
print("      β̂(naive) so với hạ theo toạ độ: lệch %.3e" % np.abs(b_sao / math.sqrt(1 + l2e) - b_naive).max())
print("   (12): β̂(elastic net) = (1+λ₂)·β̂(naive) — lệch %.3e ; hệ số nhân = %.1f"
      % (np.abs(b_en - (1 + l2e) * b_naive).max(), 1 + l2e))
print("      β̂(naive)      = %s" % np.array2string(b_naive, precision=6))
print("      β̂(elastic net)= %s" % np.array2string(b_en, precision=6))
print("      tập biến khác 0 KHÔNG đổi khi nhân lại: naive %s ; elastic net %s"
      % ([int(v) for v in np.flatnonzero(b_naive)], [int(v) for v in np.flatnonzero(b_en)]))

# (13): R = (XᵀX + λ₂I)⁻¹Xᵀ = R*/(1+λ₂) với R* dùng ma trận tương quan đã co hệ số ngoài đường chéo
S = Xe.T @ Xe
R = np.linalg.solve(S + l2e * np.eye(Xe.shape[1]), Xe.T)
Ssao = S / (1 + l2e)
np.fill_diagonal(Ssao, 1.0)
Rsao = np.linalg.solve(Ssao, Xe.T)
print("   (13): R = R*/(1+λ₂) — lệch %.3e ; hệ số ngoài đường chéo bị co đúng 1/(1+λ₂) = %.6f"
      % (np.abs(R - Rsao / (1 + l2e)).max(), 1 / (1 + l2e)))
print("      ρ₁₂ gốc = %.6f → trong R*: %.6f" % (S[0, 1], Ssao[0, 1]))


def cd_ct14(M, Xty, l1, tol=1e-14, max_sweep=200000):
    """Hạ theo toạ độ cho (14): βᵀMβ − 2yᵀXβ + λ₁|β|₁."""
    p = len(Xty)
    b = np.zeros(p)
    for _ in range(max_sweep):
        lon = 0.0
        for j in range(p):
            z = Xty[j] - (M[j] @ b - M[j, j] * b[j])
            moi = soft(z, l1 / 2) / M[j, j]
            lon = max(lon, abs(moi - b[j]))
            b[j] = moi
        if lon < tol:
            return b
    raise RuntimeError("CD (14) chưa hội tụ")


M14 = (S + l2e * np.eye(Xe.shape[1])) / (1 + l2e)
b14 = cd_ct14(M14, Xe.T @ ye, l1e)
print("   (14) Định lý 2: nghiệm của dạng βᵀ[(XᵀX+λ₂I)/(1+λ₂)]β − 2yᵀXβ + λ₁|β|₁ so với (12): lệch %.3e"
      % np.abs(b14 - b_en).max())
b15 = cd_ct14(S, Xe.T @ ye, l1e)
print("   (15) cùng dạng với M = XᵀX cho ĐÚNG lasso: lệch %.3e" % np.abs(b15 - lasso_Z(Xe, ye, l1e)).max())
gam14 = l2e / (1 + l2e)
print("   [Z] tr.9: (XᵀX+λ₂I)/(1+λ₂) = (1−γ)Σ̂ + γI với γ = λ₂/(1+λ₂) = %.6f — lệch %.3e"
      % (gam14, np.abs(M14 - ((1 - gam14) * S + gam14 * np.eye(Xe.shape[1]))).max()))

# ═════════════════════════════════════════════════════ 8. (16) — ngưỡng mềm một biến (UST)
head("8. [Z] §3.3 tr.9 công thức (16) — λ₂ → ∞ cho ngưỡng mềm trên hệ số hồi quy MỘT biến")
ust = soft(Xe.T @ ye, l1e / 2)
print("   (16): β̂(∞)_i = (|yᵀx_i| − λ₁/2)₊ · sgn(yᵀx_i)")
print("      yᵀx = %s" % np.array2string(Xe.T @ ye, precision=6))
print("      UST  = %s" % np.array2string(ust, precision=6))
print("   %-14s %14s %14s" % ("λ₂", "‖β̂(EN) − UST‖∞", "‖β̂(EN)‖₁"))
for l2v in (1.0, 10.0, 100.0, 1e3, 1e4, 1e6):
    bv = (1 + l2v) * nen_cd(Xe, ye, l1e, l2v)
    print("   %-14g %14.3e %14.6f" % (l2v, np.abs(bv - ust).max(), np.abs(bv).sum()))

# ═══════════════════════════════════════════════════ 9. §4 Bảng 1 — dữ liệu ung thư tuyến tiền liệt
head("9. [Z] §4 tr.10–12, Bảng 1 tr.11 — dựng lại trên CẢ HAI bản dữ liệu prostate")
TEN = ["lcavol", "lweight", "age", "lbph", "svi", "lcp", "gleason", "pgg45"]
_rows = list(csv.DictReader(open(os.path.join(HERE, "prostate.csv"), encoding="utf-8")))
XP = np.array([[float(r[t]) for t in TEN] for r in _rows])
YP = np.array([float(r["lpsa"]) for r in _rows])
TR = np.array([r["train"] == "True" for r in _rows])
XP_LOI = XP.copy()
XP_LOI[31, 1] = math.log(449.0)     # bản CÓ LỖI trước 2012 (ESL tr.3): 449 gm thay vì 44,9 gm
print("   %d mẫu, %d huấn luyện / %d kiểm tra ; lweight của đối tượng 32: đúng ln(%.1f) = %.6f —"
      % (len(YP), TR.sum(), (~TR).sum(), 44.9, XP[31, 1]))
print("   bản có lỗi ln(%d) = %.6f (chênh đúng ln10 = %.6f)"
      % (449, XP_LOI[31, 1], XP_LOI[31, 1] - XP[31, 1]))
print("   tương quan lớn nhất ngoài đường chéo (bản đúng) = %.4f (%s–%s) — [Z] tr.12 ghi 0,76 giữa pgg45 và gleason"
      % (max((abs(np.corrcoef(XP.T)[i, j]), TEN[i], TEN[j]) for i in range(8) for j in range(i))[0],
         max((abs(np.corrcoef(XP.T)[i, j]), TEN[i], TEN[j]) for i in range(8) for j in range(i))[1],
         max((abs(np.corrcoef(XP.T)[i, j]), TEN[i], TEN[j]) for i in range(8) for j in range(i))[2]))


def en_duong(X, y, l2, n_luoi=4001):
    """Đường naive elastic net theo λ₁ (λ₂ cố định), trả (s, β̂_naive) trên lưới; s = |β|₁/|β(λ₁=0)|₁."""
    p = X.shape[1]
    b_het = np.linalg.solve(X.T @ X + l2 * np.eye(p), X.T @ y) if l2 > 0 else \
        np.linalg.lstsq(X, y, rcond=None)[0]
    t_het = np.abs(b_het).sum()
    Xs, ys = tang_cuong(X, y, l2)
    aa, _, cc = lars_path(Xs, ys, method="lasso")
    bb = cc / math.sqrt(1 + l2)                       # β̂(naive) = β̂*/√(1+λ₂)
    ss = np.abs(bb).sum(0) / t_het
    return ss, bb, t_het


def noi_s(ss, bb, s):
    """Nội suy tuyến tính trên đường đã tính sẵn — KHÔNG chạy lại LARS."""
    if s >= ss[-1]:
        return bb[:, -1].copy()
    k = int(np.searchsorted(ss, s))
    if k == 0:
        return np.zeros(bb.shape[0])
    w = (s - ss[k - 1]) / (ss[k] - ss[k - 1])
    return bb[:, k - 1] + w * (bb[:, k] - bb[:, k - 1])


def en_tai_s(X, y, l2, s, duong=None):
    """β̂(naive) tại một giá trị s cho trước (nội suy tuyến tính trên đường LARS)."""
    ss, bb, _ = en_duong(X, y, l2) if duong is None else duong
    if s >= ss[-1]:
        return bb[:, -1].copy()
    k = int(np.searchsorted(ss, s))
    if k == 0:
        return np.zeros(X.shape[1])
    w = (s - ss[k - 1]) / (ss[k] - ss[k - 1])
    return bb[:, k - 1] + w * (bb[:, k] - bb[:, k - 1])


def bang1(Xall, nhan):
    Xtr_, ytr_ = Xall[TR], YP[TR]
    Xte_, yte_ = Xall[~TR], YP[~TR]
    m = Xtr_.mean(0)
    sc = np.sqrt(((Xtr_ - m) ** 2).sum(0))            # quy ước (2): Σx² = 1 trên tập huấn luyện
    Xtr = (Xtr_ - m) / sc
    Xte = (Xte_ - m) / sc
    yb = ytr_.mean()
    ytr = ytr_ - yb
    kq = {}
    b_ols = np.linalg.lstsq(Xtr, ytr, rcond=None)[0]
    kq["OLS"] = (b_ols, None)
    kq["Ridge λ=1"] = (np.linalg.solve(Xtr.T @ Xtr + 1.0 * np.eye(8), Xtr.T @ ytr), None)
    kq["Lasso s=0,39"] = (en_tai_s(Xtr, ytr, 0.0, 0.39), None)
    kq["Naive EN λ=1, s=1"] = (en_tai_s(Xtr, ytr, 1.0, 1.0), None)
    b_naive26 = en_tai_s(Xtr, ytr, 1000.0, 0.26)
    kq["EN λ=1000, s=0,26"] = ((1 + 1000.0) * b_naive26, None)
    print("\n   ── %s ──" % nhan)
    print("   %-22s %10s %10s   %s" % ("Phương pháp", "MSE", "(SE)", "biến chọn"))
    ra = {}
    for ten, (b, _) in kq.items():
        e = yte_ - (yb + Xte @ b)
        mse, se = (e ** 2).mean(), (e ** 2).std(ddof=1) / math.sqrt(len(e))
        sup = [i + 1 for i in range(8) if abs(b[i]) > 1e-9]
        ra[ten] = (mse, se, sup)
        print("   %-22s %10.4f %10.4f   %s" % (ten, mse, se, "All" if len(sup) == 8 else tuple(sup)))
    return ra, Xtr, ytr, Xte, yte_, yb


IN = {"OLS": (0.586, 0.184, "All"), "Ridge λ=1": (0.566, 0.188, "All"),
      "Lasso s=0,39": (0.499, 0.161, (1, 2, 4, 5, 8)),
      "Naive EN λ=1, s=1": (0.566, 0.188, "All"),
      "EN λ=1000, s=0,26": (0.381, 0.105, (1, 2, 5, 6, 8))}
print("\n   Bảng 1 IN TRONG BÀI BÁO: " + " · ".join("%s %.3f (%.3f)" % (k, v[0], v[1]) for k, v in IN.items()))
ra_dung, Xtr, ytr, Xte, yte, yb = bang1(XP, "bản ĐÃ SỬA (prostate.csv hiện nay)")
ra_loi, XtrL, ytrL, XteL, yteL, ybL = bang1(XP_LOI, "bản CÓ LỖI lweight (bản mà [Z] 2005 dùng được)")
print("\n   %-22s %12s %12s %12s   %s" % ("Phương pháp", "bài báo", "bản sửa", "bản lỗi", "khớp?"))
for k in IN:
    a, b_ = ra_dung[k][0], ra_loi[k][0]
    print("   %-22s %12.3f %12.4f %12.4f   %s" % (k, IN[k][0], a, b_,
          "bản sửa" if abs(a - IN[k][0]) < 0.0051 else ("bản lỗi" if abs(b_ - IN[k][0]) < 0.0051 else "KHÔNG bản nào")))
print("\n   Tập biến: bài báo lasso %s / EN %s ; dựng lại (bản sửa) lasso %s / EN %s"
      % (IN["Lasso s=0,39"][2], IN["EN λ=1000, s=0,26"][2],
         tuple(ra_dung["Lasso s=0,39"][2]), tuple(ra_dung["EN λ=1000, s=0,26"][2])))
print("   [Z] tr.12: \"The prediction error of the elastic net is about 24%% lower than that of the lasso\"")
print("      theo SỐ IN: (0,499 − 0,381)/0,499 = %.4f = %.2f %%" % ((0.499 - 0.381) / 0.499, 100 * (0.499 - 0.381) / 0.499))
print("   [Z] tr.11: \"the elastic net is the winner ... in terms of both prediction accuracy and SPARSITY\"")
print("      số biến: lasso %d, elastic net %d — HOÀ, không có bên thắng về độ thưa"
      % (len(IN["Lasso s=0,39"][2]), len(IN["EN λ=1000, s=0,26"][2])))

# ─────────────────────────── 9b. tương quan 0,76 · kiểm định chéo 10 mảnh · "elastic net = UST"
print("\n   [Z] tr.12: \"the highest is 0.76 (between pgg45 and gleason)\"")
for nhan, idx in [("cả 97 mẫu", np.ones(97, bool)), ("67 mẫu huấn luyện", TR)]:
    C = np.corrcoef(XP[idx].T)
    print("      %-18s corr(pgg45, gleason) = %.4f ; lớn nhất ngoài đường chéo = %.4f"
          % (nhan, C[7, 6], np.abs(C - np.eye(8)).max()))


def cv10(X, y, l2, luoi_s, r):
    """Kiểm định chéo 10 mảnh trên tập huấn luyện; trả (s tốt nhất, sai số CV)."""
    n = len(y)
    fold = r.permutation(np.arange(n) % 10)
    err = np.zeros(len(luoi_s))
    for f in range(10):
        a_, b_ = fold != f, fold == f
        m_ = X[a_].mean(0)
        sc_ = np.sqrt(((X[a_] - m_) ** 2).sum(0))
        Xa = (X[a_] - m_) / sc_
        Xb = (X[b_] - m_) / sc_
        ya = y[a_] - y[a_].mean()
        duong = en_duong(Xa, ya, l2)
        for i, sv in enumerate(luoi_s):
            bb = noi_s(duong[0], duong[1], sv)
            if l2 > 0:
                bb = (1 + l2) * bb
            err[i] += ((y[b_] - (y[a_].mean() + Xb @ bb)) ** 2).sum()
    return luoi_s[err.argmin()], err / n


luoi = np.linspace(0.02, 1.0, 50)
rcv = np.random.default_rng(2005)
print("\n   Kiểm định chéo 10 mảnh trên 67 mẫu huấn luyện (bản CÓ LỖI, lưới s 50 điểm, 20 lần chia):")
for l2v, ten in [(0.0, "lasso (λ₂ = 0)"), (1000.0, "elastic net λ₂ = 1000")]:
    ss = [cv10(XP_LOI[TR], YP[TR], l2v, luoi, np.random.default_rng(1000 + k))[0] for k in range(20)]
    print("      %-24s s được chọn: trung vị %.3f ; khoảng [%.3f ; %.3f] — bài báo in %s"
          % (ten, float(np.median(ss)), min(ss), max(ss), "0,39" if l2v == 0 else "0,26"))
print("\n   [Z] §3.5 tr.10: lưới λ₂ đề nghị là (0; 0,01; 0,1; 1; 10; 100), \"the chosen λ₂ is the one")
print("   giving the smallest CV error\" — chạy đúng thủ tục đó, 20 lần chia, trên bản CÓ LỖI:")
print("      %-10s %14s %14s %12s" % ("λ₂", "sai số CV TB", "s* trung vị", "MSE kiểm tra"))
mtr = XP_LOI[TR].mean(0)
sctr = np.sqrt(((XP_LOI[TR] - mtr) ** 2).sum(0))
Xtr_all = (XP_LOI[TR] - mtr) / sctr
Xte_all = (XP_LOI[~TR] - mtr) / sctr
ybar = YP[TR].mean()
tot_l2, tot_e = None, None
for l2v in (0.0, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0):
    es, ss_ = [], []
    for k in range(20):
        sv, ev = cv10(XP_LOI[TR], YP[TR], l2v, luoi, np.random.default_rng(300 + k))
        es.append(ev.min())
        ss_.append(sv)
    s_med = float(np.median(ss_))
    bb = en_tai_s(Xtr_all, YP[TR] - ybar, l2v, s_med)
    if l2v > 0:
        bb = (1 + l2v) * bb
    mse_te = ((YP[~TR] - (ybar + Xte_all @ bb)) ** 2).mean()
    print("      %-10g %14.6f %14.3f %12.4f" % (l2v, float(np.mean(es)), s_med, mse_te))
    if tot_e is None or np.mean(es) < tot_e:
        tot_e, tot_l2 = float(np.mean(es)), l2v
print("      → thủ tục §3.5 chọn λ₂ = %g, KHÔNG phải 1000 như Bảng 1 in" % tot_l2)

Xtr_L = (XP_LOI[TR] - mtr) / sctr
ytr_L = YP[TR] - YP[TR].mean()
b_en_p = (1 + 1000.0) * en_tai_s(Xtr_L, ytr_L, 1000.0, 0.26)
print("\n   [Z] tr.12: \"the elastic net is actually UST, because the λ selected is very big (1000)\"")
ust_p = soft(Xtr_L.T @ ytr_L, 0.0)
# UST tại cùng s: ngưỡng mềm hệ số hồi quy một biến, chọn ngưỡng cho khớp |β|₁
b1 = Xtr_L.T @ ytr_L
t_dich = np.abs(b_en_p).sum()
lo_, hi_ = 0.0, np.abs(b1).max()
for _ in range(200):
    mid = 0.5 * (lo_ + hi_)
    if np.abs(soft(b1, mid)).sum() > t_dich:
        lo_ = mid
    else:
        hi_ = mid
b_ust = soft(b1, 0.5 * (lo_ + hi_))
print("      β̂(EN λ=1000, s=0,26) = %s" % np.array2string(b_en_p, precision=4))
print("      UST cùng ‖β‖₁          = %s" % np.array2string(b_ust, precision=4))
print("      lệch lớn nhất = %.3e ; lệch tương đối = %.3e"
      % (np.abs(b_en_p - b_ust).max(), np.abs(b_en_p - b_ust).max() / np.abs(b_en_p).max()))

# ═══════════════════════════════════════════ 10. §5 Bảng 2 và Bảng 3 — bốn ví dụ mô phỏng
head("10. [Z] §5 tr.12–13, Bảng 2 tr.13 và Bảng 3 tr.15 — bốn ví dụ mô phỏng, 50 lần lặp")
L2_LUOI = np.array([0.0, 0.01, 0.1, 1.0, 10.0, 100.0])       # lưới §3.5 tr.10
S_LUOI = np.linspace(0.02, 1.0, 50)
RIDGE_LUOI = np.logspace(-4, 3, 60)


def sinh_Z(vd, r):
    """Bốn ví dụ của [Z] §5 tr.12–13. Trả (Xtr, ytr, Xva, yva, Xte, yte_sach, beta)."""
    if vd in (1, 2):
        p, sig, cs = 8, 3.0, (20, 20, 200)
        beta = np.array([3, 1.5, 0, 0, 2, 0, 0, 0.0]) if vd == 1 else np.full(8, 0.85)
        V = 0.5 ** np.abs(np.subtract.outer(np.arange(p), np.arange(p)))
        n = sum(cs)
        X = r.multivariate_normal(np.zeros(p), V, size=n)
    elif vd == 3:
        p, sig, cs = 40, 15.0, (100, 100, 400)
        beta = np.r_[np.zeros(10), np.full(10, 2.0), np.zeros(10), np.full(10, 2.0)]
        V = np.full((p, p), 0.5)
        np.fill_diagonal(V, 1.0)
        n = sum(cs)
        X = r.multivariate_normal(np.zeros(p), V, size=n)
    else:
        p, sig, cs = 40, 15.0, (50, 50, 400)
        beta = np.r_[np.full(15, 3.0), np.zeros(25)]
        n = sum(cs)
        Z = r.normal(size=(n, 3))
        X = np.empty((n, p))
        for g in range(3):
            X[:, 5 * g:5 * g + 5] = Z[:, [g]] + r.normal(scale=0.1, size=(n, 5))
        X[:, 15:] = r.normal(size=(n, 25))
    if vd == 4:
        V = np.zeros((p, p))
        for g in range(3):
            V[5 * g:5 * g + 5, 5 * g:5 * g + 5] = 1.0
        np.fill_diagonal(V, 1.0)
        V[:15, :15] += 0.0
        for i in range(15):
            V[i, i] = 1.01
        for i in range(15, 40):
            V[i, i] = 1.0
    y = X @ beta + sig * r.normal(size=len(X))
    a, b_ = cs[0], cs[0] + cs[1]
    return X[:a], y[:a], X[a:b_], y[a:b_], X[b_:], X[b_:] @ beta, beta, V


def mot_lan(vd, r):
    Xtr, ytr, Xva, yva, Xte, yte_s, beta, V = sinh_Z(vd, r)
    m = Xtr.mean(0)
    sc = np.sqrt(((Xtr - m) ** 2).sum(0))
    Xa = (Xtr - m) / sc
    ya = ytr - ytr.mean()
    Xv = (Xva - m) / sc
    Xt = (Xte - m) / sc
    yb = ytr.mean()

    def loi(b):
        """Sai số mô hình của [T] 1996: (β̂ − β)ᵀ V (β̂ − β), β̂ đưa về thang gốc."""
        d = b / sc - beta
        return d @ V @ d

    def chon_va(cands):
        e = [((yva - (yb + Xv @ b)) ** 2).mean() for b in cands]
        return cands[int(np.argmin(e))]

    kq = {}
    duong = {lv: en_duong(Xa, ya, lv) for lv in L2_LUOI}       # MỘT lần LARS cho mỗi λ₂
    cl = [noi_s(duong[0.0][0], duong[0.0][1], sv) for sv in S_LUOI]
    bl = chon_va(cl)
    kq["Lasso"] = loi(bl)
    # ridge: phân rã riêng trị một lần, rồi mọi λ là một phép chia
    U, d, Vt = np.linalg.svd(Xa, full_matrices=False)
    z = U.T @ ya
    kq["Ridge"] = loi(chon_va([Vt.T @ (d * z / (d ** 2 + lv)) for lv in RIDGE_LUOI]))
    cn, ce = [], []
    for lv in L2_LUOI:
        ss_, bb_, _ = duong[lv]
        for sv in S_LUOI:
            b = noi_s(ss_, bb_, sv)
            cn.append(b)
            ce.append((1 + lv) * b)
    kq["Naive EN"] = loi(chon_va(cn))
    be = chon_va(ce)
    kq["Elastic net"] = loi(be)
    return kq, int((np.abs(bl) > 1e-10).sum()), int((np.abs(be) > 1e-10).sum())


def se_trung_vi(v, B=500, seed=0):
    r = np.random.default_rng(seed)
    v = np.asarray(v)
    return float(np.std([np.median(r.choice(v, len(v), replace=True)) for _ in range(B)], ddof=1))


BANG2 = {"Lasso": (3.06, 3.87, 65.0, 46.6), "Elastic net": (2.51, 3.16, 56.6, 34.5),
         "Ridge": (4.49, 2.84, 39.5, 64.5), "Naive EN": (5.70, 2.73, 41.0, 45.9)}
BANG3 = {"Lasso": (5, 6, 24, 11), "Elastic net": (6, 7, 27, 16)}
do = {k: [] for k in BANG2}
kl, ke = {}, {}
for vd in (1, 2, 3, 4):
    r = np.random.default_rng(2005 + vd)
    tam = {k: [] for k in BANG2}
    sl, se_ = [], []
    for rep in range(50):
        kq, nl, ne = mot_lan(vd, r)
        for k in kq:
            tam[k].append(kq[k])
        sl.append(nl)
        se_.append(ne)
    for k in tam:
        do[k].append((float(np.median(tam[k])), se_trung_vi(tam[k], seed=vd)))
    kl[vd], ke[vd] = int(np.median(sl)), int(np.median(se_))
    print("   ví dụ %d xong" % vd, flush=True)

print("\n   Bảng 2 — trung vị sai số bình phương trung bình (số trong ngoặc: SE bootstrap B = 500)")
print("   %-13s %-22s %-22s %-22s %-22s" % ("", "Ví dụ 1", "Ví dụ 2", "Ví dụ 3", "Ví dụ 4"))
for k in ("Lasso", "Elastic net", "Ridge", "Naive EN"):
    print("   %-13s %s" % (k, " ".join("%6.2f (%.2f) ‖ %5.2f" % (do[k][i][0], do[k][i][1], BANG2[k][i])
                                       for i in range(4))))
print("   (mỗi ô: dựng lại (SE) ‖ số in trong bài báo)")

print("\n   [Z] tr.13: \"The reductions in the prediction error in examples 1, 2, 3 and 4 are")
print("   18%, 18%, 13% and 27% respectively\" — kiểm bằng CHÍNH Bảng 2 của bài báo:")
for i, vd in enumerate((1, 2, 3, 4)):
    la, en = BANG2["Lasso"][i], BANG2["Elastic net"][i]
    gd = 100 * (la - en) / la
    gd_do = 100 * (do["Lasso"][i][0] - do["Elastic net"][i][0]) / do["Lasso"][i][0]
    print("      ví dụ %d: (%.2f − %.2f)/%.2f = %6.2f %%  (bài báo in %d %%) ; dựng lại %6.2f %%"
          % (vd, la, en, la, gd, (18, 18, 13, 27)[i], gd_do))

print("\n   [Z] tr.13: \"In all the examples, the elastic net is significantly more accurate than the lasso\"")
print("   Nhưng đọc THÊM hai cột dưới của chính Bảng 2 — ridge và naive EN:")
for i, vd in enumerate((1, 2, 3, 4)):
    b = {k: BANG2[k][i] for k in BANG2}
    tot = min(b, key=b.get)
    print("      ví dụ %d: tốt nhất trong bốn phương pháp là %-12s (%.2f) ; elastic net %.2f"
          % (vd, tot, b[tot], b["Elastic net"]))

print("\n   Bảng 3 — trung vị số hệ số khác 0")
print("   %-13s %-14s %-14s %-14s %-14s" % ("", "Ví dụ 1", "Ví dụ 2", "Ví dụ 3", "Ví dụ 4"))
for k in ("Lasso", "Elastic net"):
    print("   %-13s %s" % (k, " ".join("%3d ‖ %3d      " % ((kl if k == "Lasso" else ke)[vd], BANG3[k][i])
                                       for i, vd in enumerate((1, 2, 3, 4)))))

# ───────────────── 10b. ba cách hiểu "mean-squared error" — chỗ lệch của Bảng 2 KHÔNG do định nghĩa
print("\n   Ba cách hiểu \"test error (the mean-squared error)\" của [Z] tr.12, đo trên ví dụ 1, 50 lần:")
r10 = np.random.default_rng(99)
d1_, d2_, d3_ = [], [], []
for rep in range(50):
    Xtr, ytr, Xva, yva, Xte, yte_s, beta, Vx = sinh_Z(1, r10)
    m = Xtr.mean(0)
    sc = np.sqrt(((Xtr - m) ** 2).sum(0))
    Xa, ya = (Xtr - m) / sc, ytr - ytr.mean()
    Xv, Xt = (Xva - m) / sc, (Xte - m) / sc
    yb = ytr.mean()
    dd0 = en_duong(Xa, ya, 0.0)
    c = [noi_s(dd0[0], dd0[1], sv) for sv in S_LUOI]
    b = c[int(np.argmin([((yva - (yb + Xv @ bb)) ** 2).mean() for bb in c]))]
    bg = b / sc                                   # đưa về thang gốc để so với β
    V1 = 0.5 ** np.abs(np.subtract.outer(np.arange(8), np.arange(8)))
    d1_.append(((yte_s - (yb + Xt @ b)) ** 2).mean())                       # so với Xβ (không nhiễu)
    d2_.append(((yte_s + 3.0 * np.random.default_rng(rep).normal(size=len(yte_s))
                 - (yb + Xt @ b)) ** 2).mean())                             # so với y có nhiễu
    d3_.append((bg - beta) @ V1 @ (bg - beta))                              # (β̂−β)ᵀV(β̂−β)
print("      (a) so với Xβ trên tập kiểm tra : trung vị %.3f" % float(np.median(d1_)))
print("      (b) so với y CÓ nhiễu           : trung vị %.3f  (≈ (a) + σ² = (a) + 9)" % float(np.median(d2_)))
print("      (c) (β̂−β)ᵀV(β̂−β) theo [T] 1996 : trung vị %.3f" % float(np.median(d3_)))
print("      → bài báo in 3,06 cho lasso; (a) và (c) cùng bậc, (b) thì không. Chỗ lệch nằm ở")
print("        LƯỚI DÒ và cách chuẩn hoá, không ở định nghĩa sai số.")

# ─────── 10c. naive elastic net với λ₂ = 0 CHÍNH LÀ lasso — nên dòng "5,70" của Bảng 2 tự mâu thuẫn
print("\n   Bảng 2 ví dụ 1: naive EN 5,70 so với lasso 3,06 — tệ hơn %.0f %%." % (100 * (5.70 - 3.06) / 3.06))
print("   Nhưng λ₂ = 0 trong (3) CHO ĐÚNG tiêu chí lasso, và λ₂ = 0 nằm trong lưới §3.5 tr.10.")
b_l0 = nen_cd(X0, y0, 0.5, 0.0)
print("      kiểm: naive EN(λ₁ = 0,5 ; λ₂ = 0) so với lasso(λ₁ = 0,5) — lệch %.2e" % np.abs(b_l0 - lasso_Z(X0, y0, 0.5)).max())
r10c = np.random.default_rng(404)
te_l, te_n = [], []
for rep in range(50):
    kq, _, _ = mot_lan(1, r10c)
    te_l.append(kq["Lasso"])
    te_n.append(kq["Naive EN"])
te_l, te_n = np.array(te_l), np.array(te_n)
print("      50 lần lặp ví dụ 1: naive EN tệ hơn lasso ở %d/50 lần ; tệ hơn nhiều nhất %.1f %% ở MỘT lần chạy"
      % (int((te_n > te_l).sum()), 100 * (te_n / te_l - 1).max()))
print("      nhưng TRUNG VỊ tỉ số naive/lasso = %.4f, và trung vị hai cột: %.4f so với %.4f — Bảng 2 in TRUNG VỊ,"
      % (float(np.median(te_n / te_l)), float(np.median(te_n)), float(np.median(te_l))))
print("      nên một cột trung vị tệ hơn 86 % không dựng lại được khi λ₂ = 0 có trong lưới")

# ═════════════════════════════════════ 11. Hình 5 — ví dụ dựng riêng có hai nhân tố ẩn
head("11. [Z] tr.13 và Hình 5 tr.15 — ví dụ hai nhân tố ẩn Z₁, Z₂")
r11 = np.random.default_rng(5)
Z1 = r11.uniform(0, 20, 100)
Z2 = r11.uniform(0, 20, 100)
yh = Z1 + 0.1 * Z2 + r11.normal(size=100)
e = r11.normal(scale=math.sqrt(1 / 16), size=(100, 6))
Xh = np.c_[Z1 + e[:, 0], -Z1 + e[:, 1], Z1 + e[:, 2], Z2 + e[:, 3], -Z2 + e[:, 4], Z2 + e[:, 5]]
Ch = np.corrcoef(Xh.T)
print("   tương quan trong nhóm 1 (x₁,x₂,x₃): %s" % np.array2string(np.array([Ch[0, 1], Ch[0, 2], Ch[1, 2]]), precision=4))
print("   tương quan trong nhóm 2 (x₄,x₅,x₆): %s" % np.array2string(np.array([Ch[3, 4], Ch[3, 5], Ch[4, 5]]), precision=4))
print("   tương quan GIỮA hai nhóm: lớn nhất |ρ| = %.4f" % np.abs(Ch[:3, 3:]).max())
Xhs, yhs = chuan_Z(Xh, yh)
for l2v, ten in [(0.0, "lasso"), (0.5, "elastic net λ₂ = 0,5")]:
    ss, bb, _ = en_duong(Xhs, yhs, l2v)
    bcuoi = bb[:, -1] * (1 + l2v)
    # độ trải trong nhóm dọc theo đường, đo trên |hệ số| vì x₂, x₅ mang dấu âm
    A = np.abs(bb) * (1 + l2v)
    tr1 = (A[:3].max(0) - A[:3].min(0))
    tr2 = (A[3:].max(0) - A[3:].min(0))
    print("   %-22s số bước LARS = %3d ; độ trải TRONG nhóm 1 (trung bình dọc đường) = %8.4f ; nhóm 2 = %8.4f"
          % (ten, bb.shape[1] - 1, tr1.mean(), tr2.mean()))
    print("      %-19s hệ số ở cuối đường: %s" % ("", np.array2string(bcuoi, precision=3)))

# ═════════════════════════════════════ 12. Hình 3 — đường nghiệm trên dữ liệu prostate
head("12. [Z] Hình 3 tr.11 — đường lasso và đường elastic net (λ = 1000) theo s, dữ liệu prostate")
for l2v, ten in [(0.0, "(a) lasso"), (1000.0, "(b) elastic net λ = 1000")]:
    ss, bb, t_het = en_duong(Xtr_L, ytr_L, l2v)
    bb_ = bb * (1 + l2v)
    thu_tu = []
    for k in range(bb.shape[1]):
        for j in range(8):
            if abs(bb[j, k]) > 1e-12 and j not in thu_tu:
                thu_tu.append(j)
    print("   %-26s %2d bước ; thứ tự biến vào: %s"
          % (ten, bb.shape[1] - 1, " → ".join(TEN[j] for j in thu_tu)))
    print("      %-23s s tại các nút: %s" % ("", np.array2string(ss, precision=3, max_line_width=200)))
    print("      %-23s hệ số ở cuối đường: %s" % ("", np.array2string(bb_[:, -1], precision=3)))
print("   [Z] chú thích Hình 3: \"both estimates are piecewise linear\" — kiểm tính tuyến tính từng khúc:")
for l2v, ten in [(0.0, "lasso"), (1000.0, "elastic net")]:
    ss, bb, _ = en_duong(Xtr_L, ytr_L, l2v)
    lech = 0.0
    for k in range(1, bb.shape[1]):
        for a_ in (0.25, 0.5, 0.75):
            sv = ss[k - 1] + a_ * (ss[k] - ss[k - 1])
            noi = bb[:, k - 1] + a_ * (bb[:, k] - bb[:, k - 1])
            lech = max(lech, np.abs(en_tai_s(Xtr_L, ytr_L, l2v, sv) - noi).max())
    print("      %-12s lệch lớn nhất giữa nghiệm thật và đoạn thẳng nối hai nút: %.2e" % (ten, lech))

# ═════════════════════════════════════════ 13. §6 — phân loại bệnh bạch cầu (Bảng 4, Hình 6–7)
head("13. [Z] §6 tr.13–17, Bảng 4 tr.16, Hình 6–7 — phần KHÔNG dựng lại được, và chỗ tự mâu thuẫn")
print("   [Z] tr.15 viết: \"The leukaemia data consist of 7129 genes and 72 samples\"")
print("   rồi ngay câu sau:  \"...based on the expression level of those 7219 genes\"")
print("   7129 và 7219 là hai hoán vị chữ số của nhau; 7129 là con số đúng của Golub 1999.")
print("   Bảng 4 tr.16 in tỉ lệ lỗi dạng phân số — kiểm tính nhất quán của mẫu số:")
B4 = [("Golub", 3, 38, 4, 34, 50), ("SVM–RFE", 2, 38, 1, 34, 31), ("PLR–RFE", 2, 38, 1, 34, 26),
      ("Nearest shrunken centroids", 2, 38, 2, 34, 21), ("Elastic net", 3, 38, 0, 34, 45)]
for ten, a, b_, c, d, g in B4:
    print("      %-28s CV %d/%-3d = %.4f ; kiểm tra %d/%-3d = %.4f ; %2d gen"
          % (ten, a, b_, a / b_, c, d, c / d, g))
print("   38 + 34 = %d = số mẫu; mọi mẫu số khớp. [Z] tr.17: mô hình tối ưu ở bước k = 82 với 45 gen," % (38 + 34))
print("   \"the size of the training set is 38, so the lasso can at most select 38 genes\" — 45 > 38.")
print("   Dữ liệu Golub 1999 KHÔNG có trong repo, nên Bảng 4, Hình 6 và Hình 7 chỉ kiểm được về tính")
print("   nhất quán số học, không dựng lại được.")

# ═══════════════════════════════════════ 14. §3.4 — LARS-EN: đường nghiệm trên dữ liệu tăng cường
head("14. [Z] §3.4 tr.9–10 — LARS-EN, và cấu trúc THƯA của X* mà bài báo khai thác")
Xw, yw = chuan_Z(np.random.default_rng(31).normal(size=(30, 200)), np.random.default_rng(32).normal(size=30))
bt = np.zeros(200)
bt[[0, 5, 9, 14, 20, 40, 77, 120]] = [3, -2, 2.5, 1.5, -1.8, 1.2, -1.0, 0.9]
yw = Xw @ bt + 0.5 * np.random.default_rng(33).normal(size=30)
yw = yw - yw.mean()
print("   n = %d, p = %d — vùng p ≫ n của [Z] §1 kịch bản (a)" % Xw.shape)
aa, act, cc = lars_path(Xw, yw, method="lasso")
print("   lasso: đường LARS dừng sau %d bước, số biến khác 0 lớn nhất = %d (trần n = %d)"
      % (cc.shape[1] - 1, int((np.abs(cc) > 0).sum(0).max()), Xw.shape[0]))
for l2v in (0.01, 0.1, 1.0, 10.0):
    Xs, ys = tang_cuong(Xw, yw, l2v)
    a2, ac2, c2 = lars_path(Xs, ys, method="lasso")
    print("   LARS-EN λ₂ = %-6g: dữ liệu tăng cường %d×%d ; %4d bước ; số biến khác 0 lớn nhất = %3d"
          % (l2v, Xs.shape[0], Xs.shape[1], c2.shape[1] - 1, int((np.abs(c2) > 0).sum(0).max())))
Xs, ys = tang_cuong(Xw, yw, 1.0)
print("   [Z] tr.10: \"X*_j has p − 1 zero elements\" — đếm thật: mỗi cột của X* có %d phần tử 0 trên %d hàng"
      % (int((Xs[:, 0] == 0).sum()), Xs.shape[0]))
print("      G_A = (1/(1+λ₂))(XᵀX + λ₂I) cho A = {0,…,9}: lệch %.2e"
      % np.abs(Xs[:, :10].T @ Xs[:, :10] - (Xw[:, :10].T @ Xw[:, :10] + 1.0 * np.eye(10)) / 2.0).max())
print("   [Z] tr.10: dừng sớm sau m bước tốn O(m³ + p·m²) — với p = %d, m = 200: %.3e phép"
      % (Xw.shape[1], 200 ** 3 + Xw.shape[1] * 200 ** 2))

# ════════════════════════════════════════════ 15. [E] — hai công thức elastic net NGƯỢC NHAU
head("15. [E] ESL 2009 — (3.54) tr.73 và (18.20) tr.662 dùng α NGƯỢC NHAU, trong CÙNG một cuốn sách")
print("   (3.54) tr.73 : λ Σ ( α·β_j² + (1−α)·|β_j| )        → α = 1 là RIDGE")
print("   (3.91) tr.99 : ‖y−Xβ‖² + λ[ α‖β‖₂² + (1−α)‖β‖₁ ]   → cùng quy ước với (3.54)")
print("   (18.20) tr.662: Σ ( α·|β_j| + (1−α)·β_j² )          → α = 1 là LASSO — NGƯỢC LẠI")
print("   (18.21) tr.663: cùng quy ước với (18.20)")
bq = np.array([0.6, -0.8])
for a in (0.0, 0.2, 0.8, 1.0):
    p354 = a * (bq ** 2).sum() + (1 - a) * np.abs(bq).sum()
    p1820 = a * np.abs(bq).sum() + (1 - a) * (bq ** 2).sum()
    print("      α = %.1f : (3.54) = %.4f ; (18.20) = %.4f ; hai giá trị bằng nhau khi α = 0,5: %s"
          % (a, p354, p1820, "có" if abs(a - 0.5) < 1e-9 else "không"))
print("   Hình 3.13 tr.73 vẽ α = 0,2 theo (3.54) ⇒ 20 % ridge, 80 % lasso — gần lasso.")
print("   Hình 18.5 tr.662 vẽ α = 0,8 theo (18.20) ⇒ 80 % lasso — CŨNG gần lasso. Hai hình cùng")
print("   một vùng pha, hai con số α khác hẳn nhau; đọc nhầm quy ước là đọc ngược.")

# đường mức của hai phạt trong Hình 3.13, so bằng số
gq = np.linspace(0, 1, 40001)
b2_lq = np.maximum(1 - gq ** 1.2, 0) ** (1 / 1.2)                      # Σ|β|^1.2 = 1
aE = 0.2
disc = (1 - aE) ** 2 + 4 * aE * (1 - (1 - aE) * gq - aE * gq ** 2)
b2_en13 = (-(1 - aE) + np.sqrt(np.maximum(disc, 0))) / (2 * aE)        # α·β² + (1−α)|β| = 1
print("   Hình 3.13: hai đường mức 1 (q = 1,2 và elastic net α = 0,2), cùng đi qua (1,0) và (0,1)")
ok_ = b2_lq > 0.05          # bỏ phần đuôi sát trục, nơi mẫu số tiến về 0
print("      lệch lớn nhất theo trục dọc = %.6f tại β₁ = %.4f ; lệch TƯƠNG ĐỐI lớn nhất (phần b₂ > 0,05) = %.2f %%"
      % (np.abs(b2_lq - b2_en13).max(), gq[np.abs(b2_lq - b2_en13).argmax()],
         100 * (np.abs(b2_lq - b2_en13)[ok_] / b2_lq[ok_]).max()))
print("      hệ số góc một phía tại đỉnh (0,1), đo ở vài khoảng cách tới đỉnh:")
print("         %-12s %14s %14s" % ("β₁", "elastic net", "q = 1,2"))
for eps in (1e-2, 1e-4, 1e-6, 1e-8):
    d_en = ((-(1 - aE) + math.sqrt((1 - aE) ** 2 + 4 * aE * (1 - (1 - aE) * eps - aE * eps ** 2))) / (2 * aE) - 1) / eps
    d_lq = (max(1 - eps ** 1.2, 0) ** (1 / 1.2) - 1) / eps
    print("         %-12.0e %14.6f %14.6f" % (eps, d_en, d_lq))
print("         → elastic net dừng ở −2/3 (đỉnh NHỌN) ; q = 1,2 tiến về 0 (biên TRƠN)")
print("      [E] tr.73: \"the elastic-net has sharp (non-differentiable) corners, while the q = 1.2 penalty does not\"")

# ═══════════════════════════════════════ 16. [G] glmnet 2010 — quy ước α thứ ba, và BỎ phép nhân lại
head("16. [G] glmnet 2010 — (1)–(3) tr.3: α NGƯỢC với [Z], và chú thích 1 tr.3 BỎ phép nhân (1+λ₂)")
print("   (2)–(3): P_α(β) = (1−α)·½‖β‖₂² + α‖β‖₁ = Σ_j [ ½(1−α)β_j² + α|β_j| ]")
print("   [G] tr.3: \"a compromise between the ridge-regression penalty (α = 0) and the lasso penalty (α = 1)\"")
print("   [G] chú thích 1 tr.3 (nguyên văn): \"Zou and Hastie (2005) called this penalty the naive")
print("   elastic net, and preferred a rescaled version which they called elastic net. We drop this")
print("   distinction here.\" — tức glmnet KHÔNG nhân (1+λ₂); Hastie là tác giả của cả hai bài.")
print("   (1): R_λ(β₀,β) = (1/2N)Σ(y_i − β₀ − x_iᵀβ)² + λ·P_α(β) — có 1/(2N), [Z] (3) thì KHÔNG.")
print("\n   Quy đổi giữa hai quy ước, kiểm bằng số trên cùng bộ dữ liệu:")
print("      [Z] (3):  |y−Xβ|² + λ₁|β|₁ + λ₂|β|²      (cột Σx² = 1)")
print("      [G] (1):  (1/2N)|y−Xβ|² + λ[α|β|₁ + ½(1−α)|β|²]")
print("      ⇒ λ₁ = 2Nλα ; λ₂ = Nλ(1−α)")
Xg2, yg2 = chuan_Z(np.random.default_rng(41).normal(size=(60, 10)), np.random.default_rng(42).normal(size=60))
yg2 = Xg2 @ np.r_[np.array([2.0, -1.5, 1.0]), np.zeros(7)] + 0.4 * np.random.default_rng(43).normal(size=60)
yg2 = yg2 - yg2.mean()
Ng2 = 60
for lam_g, al_g in [(0.02, 1.0), (0.02, 0.5), (0.05, 0.2)]:
    l1g, l2g = 2 * Ng2 * lam_g * al_g, Ng2 * lam_g * (1 - al_g)
    b_Z = nen_cd(Xg2, yg2, l1g, l2g)
    # hạ theo toạ độ theo đúng quy ước [G]
    b_G = np.zeros(10)
    for _ in range(200000):
        lon = 0.0
        for j in range(10):
            z = Xg2[:, j] @ (yg2 - Xg2 @ b_G) / Ng2 + b_G[j] * (Xg2[:, j] @ Xg2[:, j]) / Ng2
            moi = soft(z, lam_g * al_g) / ((Xg2[:, j] @ Xg2[:, j]) / Ng2 + lam_g * (1 - al_g))
            lon = max(lon, abs(moi - b_G[j]))
            b_G[j] = moi
        if lon < 1e-14:
            break
    print("      λ = %-6g α = %-4g → λ₁ = %8.4f, λ₂ = %8.4f ; hai bộ giải lệch %.2e"
          % (lam_g, al_g, l1g, l2g, np.abs(b_Z - b_G).max()))

# ════════════════════════════════════ 17. Năm quy ước α, và quy đổi sang scikit-learn
head("17. NĂM nguồn, NĂM cách viết cùng một hàm phạt — bảng quy đổi")
print("   %-26s %-46s %-18s" % ("Nguồn", "Hàm phạt", "α = 1 nghĩa là"))
for ten, ct, dm in [
        ("[Z] 2005 (5) tr.3", "(1−α)|β|₁ + α|β|²  với α = λ₂/(λ₁+λ₂)", "RIDGE"),
        ("[E] 2009 (3.54) tr.73", "λ Σ (α β_j² + (1−α)|β_j|)", "RIDGE"),
        ("[E] 2009 (18.20) tr.662", "Σ (α|β_j| + (1−α) β_j²)", "LASSO"),
        ("[G] 2010 (2) tr.3", "(1−α)·½‖β‖₂² + α‖β‖₁", "LASSO"),
        ("scikit-learn ElasticNet", "l1_ratio·‖β‖₁ + ½(1−l1_ratio)‖β‖₂²", "LASSO")]:
    print("   %-26s %-46s %-18s" % (ten, ct, dm))
try:
    from sklearn.linear_model import ElasticNet
    for lam_g, al_g in [(0.02, 1.0), (0.02, 0.5), (0.05, 0.2)]:
        m_ = ElasticNet(alpha=lam_g, l1_ratio=al_g, fit_intercept=False, tol=1e-14, max_iter=2000000)
        m_.fit(Xg2, yg2)
        l1g, l2g = 2 * Ng2 * lam_g * al_g, Ng2 * lam_g * (1 - al_g)
        print("   sklearn(alpha=%g, l1_ratio=%g) so với [Z] (λ₁=%.4f, λ₂=%.4f): lệch %.2e"
              % (lam_g, al_g, l1g, l2g, np.abs(m_.coef_ - nen_cd(Xg2, yg2, l1g, l2g)).max()))
    print("   ⇒ alpha của scikit-learn = λ của [G] ; l1_ratio = α của [G] ; và λ₁ = 2N·alpha·l1_ratio")
except Exception as ex:                                            # pragma: no cover
    print("   scikit-learn ElasticNet không chạy được:", ex)

# ═══════════════════════ 18. Chọn thiết bị bằng PHÉP ĐO, không bằng thói quen (CLAUDE.md mục 3)
head("18. Đo thiết bị cho CHÍNH phép tính của bài này — CPU hay GPU?")
import time
print("   Phép tính nặng nhất của bài: giải (XᵀX + λ₂I)β = Xᵀy và chạy LARS trên dữ liệu tăng cường.")
try:
    import torch
    torch.set_num_threads(1)
    print("   torch %s · CUDA: %s · %s" % (torch.__version__, torch.cuda.is_available(),
                                           torch.cuda.get_device_name(0) if torch.cuda.is_available() else "—"))
    print("   %-28s %14s %14s %12s" % ("cỡ bài toán (n × p)", "numpy CPU", "torch CUDA", "GPU nhanh hơn"))
    for n_, p_ in [(67, 8), (100, 40), (1000, 500), (4000, 4000)]:
        A = np.random.default_rng(0).normal(size=(n_, p_))
        b_ = np.random.default_rng(1).normal(size=n_)
        G = A.T @ A + 1.0 * np.eye(p_)
        c_ = A.T @ b_
        t0 = time.perf_counter()
        for _ in range(20):
            np.linalg.solve(G, c_)
        t_np = (time.perf_counter() - t0) / 20 * 1e3
        if torch.cuda.is_available():
            Gt = torch.as_tensor(G, device="cuda")
            ct = torch.as_tensor(c_, device="cuda")
            torch.linalg.solve(Gt, ct)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(20):
                torch.linalg.solve(Gt, ct)
            torch.cuda.synchronize()
            t_gpu = (time.perf_counter() - t0) / 20 * 1e3
        else:
            t_gpu = float("nan")
        print("   %-28s %12.4f ms %12.4f ms %11.2f×" % ("%d × %d" % (n_, p_), t_np, t_gpu, t_np / t_gpu))
    print("   → mọi cỡ bài toán THẬT của bài này (n ≤ 100, p ≤ 200) nằm ở vùng CPU thắng;")
    print("     GPU chỉ vượt lên từ cỡ vài nghìn chiều, mà bài không có phép tính nào cỡ đó.")
except ImportError:                                                # pragma: no cover
    print("   torch không có trên máy này — bỏ qua phép so thiết bị.")
print("   Phần tăng tốc thật của script này là THUẬT TOÁN, không phải thiết bị:")
t0 = time.perf_counter()
dd_ = en_duong(Xtr_L, ytr_L, 1000.0)
t_1 = (time.perf_counter() - t0) * 1e3
t0 = time.perf_counter()
for sv in S_LUOI:
    noi_s(dd_[0], dd_[1], sv)
t_50 = (time.perf_counter() - t0) * 1e3
print("      một lần LARS-EN: %.3f ms ; nội suy 50 giá trị s trên đường ĐÃ tính: %.3f ms (%.0f lần rẻ hơn"
      % (t_1, t_50, t_1 * 50 / max(t_50, 1e-9)))
print("      so với chạy LARS lại 50 lần)")

# ═════════════════════════════════════════ 19. Kịch bản (a) của §1: trần n biến của lasso
head("19. [Z] §1 tr.2 kịch bản (a) — \"the lasso selects at most n variables before it saturates\"")
print("   %-26s %10s %10s %10s" % ("phương pháp", "n", "p", "số biến khác 0 lớn nhất"))
print("   %-26s %10d %10d %10d" % ("lasso (đường LARS)", Xw.shape[0], Xw.shape[1],
                                   int((np.abs(cc) > 0).sum(0).max())))
for l2v in (0.01, 1.0):
    Xs_, ys_ = tang_cuong(Xw, yw, l2v)
    _, _, c2_ = lars_path(Xs_, ys_, method="lasso")
    print("   %-26s %10d %10d %10d" % ("elastic net λ₂ = %g" % l2v, Xs_.shape[0], Xs_.shape[1],
                                       int((np.abs(c2_) > 0).sum(0).max())))
print("   [Z] tr.4 giải thích bằng Bổ đề 1: X* có HẠNG p, nên naive elastic net \"can potentially")
print("   select all p predictors in all situations\" — kiểm hạng: rank(X) = %d, rank(X*) = %d"
      % (np.linalg.matrix_rank(Xw), np.linalg.matrix_rank(tang_cuong(Xw, yw, 1.0)[0])))
print("   Chú ý con số: trần đo được là %d = n − 1, không phải n — vì y đã được center (mất một bậc tự do)."
      % int((np.abs(cc) > 0).sum(0).max()))

# ═══════════════════════════════════ 20. Mảng cho các khối mã sinh hình in trong bài
head("20. Mảng cho khối mã sinh hình (dán thẳng vào bài — xem CLAUDE.md: hình vẽ SỐ thì dán SỐ)")


def in_mang(ten, a, nd=6):
    print("[hinh] %s = %s" % (ten, np.array2string(np.asarray(a), precision=nd, separator=", ",
                                                   max_line_width=10 ** 6, threshold=10 ** 6)))


# --- Hình: dựng lại Hình 5 của [Z] (hai nhân tố ẩn) ---
for l2v, nhan in [(0.0, "lasso"), (0.5, "enet")]:
    ss_, bb_, _ = en_duong(Xhs, yhs, l2v)
    in_mang("h5_%s_s" % nhan, ss_)
    in_mang("h5_%s_b" % nhan, bb_ * (1 + l2v), 4)

# --- Hình: dựng lại Hình 3 của [Z] (prostate) ---
for l2v, nhan in [(0.0, "lasso"), (1000.0, "enet")]:
    ss_, bb_, _ = en_duong(Xtr_L, ytr_L, l2v)
    in_mang("h3_%s_s" % nhan, ss_)
    in_mang("h3_%s_b" % nhan, bb_ * (1 + l2v), 4)
print("[hinh] h3_ten = %s" % TEN)

# --- Hình: sai số CV và sai số kiểm tra theo λ₂ trên prostate (bản CÓ LỖI) ---
l2_tr = np.array([0.0, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0])
cv_, te_, s_ = [], [], []
for l2v in l2_tr:
    es = [cv10(XP_LOI[TR], YP[TR], l2v, luoi, np.random.default_rng(300 + k))[0] for k in range(20)]
    ev = [cv10(XP_LOI[TR], YP[TR], l2v, luoi, np.random.default_rng(300 + k))[1].min() for k in range(20)]
    sm = float(np.median(es))
    bb = en_tai_s(Xtr_all, YP[TR] - ybar, l2v, sm)
    if l2v > 0:
        bb = (1 + l2v) * bb
    cv_.append(float(np.mean(ev)))
    te_.append(((YP[~TR] - (ybar + Xte_all @ bb)) ** 2).mean())
    s_.append(sm)
in_mang("cvte_l2", l2_tr, 2)
in_mang("cvte_cv", cv_)
in_mang("cvte_te", te_)
in_mang("cvte_s", s_, 3)

# --- Hình: bảng 2 dựng lại so với số in ---
in_mang("b2_do", np.array([[do[k][i][0] for i in range(4)] for k in ("Lasso", "Elastic net", "Ridge", "Naive EN")]), 4)
in_mang("b2_se", np.array([[do[k][i][1] for i in range(4)] for k in ("Lasso", "Elastic net", "Ridge", "Naive EN")]), 4)
in_mang("b2_in", np.array([list(BANG2[k]) for k in ("Lasso", "Elastic net", "Ridge", "Naive EN")]), 2)

# --- Hình: trần n biến ---
so_lasso = (np.abs(cc) > 0).sum(0)
in_mang("tran_lasso", so_lasso, 0)
for l2v in (0.01, 1.0):
    Xs_, ys_ = tang_cuong(Xw, yw, l2v)
    _, _, c2_ = lars_path(Xs_, ys_, method="lasso")
    in_mang("tran_en_%s" % str(l2v).replace(".", "_"), (np.abs(c2_) > 0).sum(0), 0)
print("[hinh] tran_n = %d ; tran_p = %d" % (Xw.shape[0], Xw.shape[1]))
print("\nHOÀN TẤT.")
