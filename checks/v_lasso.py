# -*- coding: utf-8 -*-
"""Dựng lại mọi phép toán, bảng và hình của các nguồn gốc bài Lasso Regression.

Nguồn:
  [T] R. Tibshirani, "Regression Shrinkage and Selection via the Lasso",
      J. R. Statist. Soc. B 58(1):267–288, 1996.
      AI algorithms/_papers/1996 - Tibshirani - Regression Shrinkage and Selection via the Lasso.pdf
      (bản tải JSTOR — CHỈ có trên máy cục bộ, không đẩy lên repo; xem .gitignore)
  [R] R. Tibshirani, "Regression shrinkage and selection via the lasso: a retrospective",
      J. R. Statist. Soc. B 73(3):273–282, 2011.
  [E] T. Hastie, R. Tibshirani, J. Friedman, The Elements of Statistical Learning, 2nd ed.,
      Springer 2009 (bản in 12): chú thích tr.3, §3.4.2–3.4.3 tr.68–73, Bảng 3.3 tr.63, Hình 3.7 tr.62.
  [L] B. Efron, T. Hastie, I. Johnstone, R. Tibshirani, "Least Angle Regression",
      Ann. Statist. 32(2):407–451, 2004 — Bảng 1, (1.5), Hình 1, §3.1.
  [G] J. Friedman, T. Hastie, R. Tibshirani, "Regularization Paths for Generalized Linear
      Models via Coordinate Descent", J. Stat. Softw. 33(1), 2010 — (1)–(10), §2.5–2.6.

Dữ liệu: tools/prostate.csv (ElemStatLearn 2015.6.26.2, bản ĐÃ sửa lỗi lweight của đối tượng 32),
tools/kyphosis.csv (gói R rpart qua Rdatasets, 81 dòng), bộ diabetes của scikit-learn.

Chạy:  cd tools && python v_lasso.py        (khoảng 10–20 phút, phần lớn là mục 9)
"""
import csv
import ctypes
import itertools
import math
import os
import sys

import numpy as np
from scipy import optimize, stats
from sklearn.datasets import load_diabetes
from sklearn.linear_model import lars_path

sys.stdout.reconfigure(encoding="utf-8")
np.set_printoptions(linewidth=120, suppress=True)
HERE = os.path.dirname(os.path.abspath(__file__))
PAPERS = os.path.join(os.path.dirname(HERE), "AI algorithms", "_papers")
ESL = os.path.join(PAPERS, "2009 - Hastie, Tibshirani, Friedman - The Elements of Statistical Learning (2nd ed, print 12).pdf")


def head(t):
    print("\n" + "=" * 100 + "\n" + t + "\n" + "=" * 100, flush=True)


def soft(z, g):
    return np.sign(z) * np.maximum(np.abs(z) - g, 0.0)


# ─────────────────────────────────────────────────────────────── đường lasso dạng RÀNG BUỘC
class Duong:
    """Đường lasso theo t = Σ|β_j| (dạng (1) của [T]), tính bằng LARS có sửa đổi lasso.

    X, y đã center. positive=True cho đường garotte (hệ số không âm).
    """

    def __init__(self, X, y, positive=False):
        _, _, co = lars_path(X, y, method="lasso", positive=positive)
        self.co = co
        self.tt = np.abs(co).sum(0)
        self.t0 = self.tt[-1]

    def tai(self, t):
        tt, co = self.tt, self.co
        if t >= tt[-1]:
            return co[:, -1].copy()
        if t <= 0:
            return np.zeros(co.shape[0])
        k = int(np.searchsorted(tt, t))
        w = (t - tt[k - 1]) / (tt[k] - tt[k - 1])
        return co[:, k - 1] + w * (co[:, k] - co[:, k - 1])


def lasso_cd(X, y, lam, b=None, tol=1e-13, max_sweep=200000, dem=False):
    """Hạ theo toạ độ cho ½‖y − Xβ‖² + lam·‖β‖₁ (cột KHÔNG cần chuẩn hoá)."""
    p = X.shape[1]
    b = np.zeros(p) if b is None else b.copy()
    r = y - X @ b
    nn = (X ** 2).sum(0)
    for sweep in range(1, max_sweep + 1):
        lon = 0.0
        for j in range(p):
            z = X[:, j] @ r + nn[j] * b[j]
            moi = soft(z, lam) / nn[j]
            if moi != b[j]:
                r -= X[:, j] * (moi - b[j])
                lon = max(lon, abs(moi - b[j]))
                b[j] = moi
        if lon < tol:
            return (b, sweep) if dem else b
    raise RuntimeError("CD chưa hội tụ")


# ─────────────────────────────────────────────────────────────── dữ liệu prostate
TEN = ["lcavol", "lweight", "age", "lbph", "svi", "lcp", "gleason", "pgg45"]
_rows = list(csv.DictReader(open(os.path.join(HERE, "prostate.csv"), encoding="utf-8")))
XP = np.array([[float(r[t]) for t in TEN] for r in _rows])
YP = np.array([float(r["lpsa"]) for r in _rows])
TR = np.array([r["train"] == "True" for r in _rows])
XP_LOI = XP.copy()
XP_LOI[31, 1] = math.log(449.0)          # bản CÓ LỖI mà ESL tr.3 kể: 449 gm thay vì 44,9 gm


def ols_se(A, y):
    """OLS có hệ số chặn; trả (hệ số, sai số chuẩn, σ̂²) với σ̂² = RSS/(N − p − 1)."""
    M = np.c_[np.ones(len(y)), A]
    b = np.linalg.lstsq(M, y, rcond=None)[0]
    r = y - M @ b
    s2 = r @ r / (len(y) - M.shape[1])
    return b, np.sqrt(np.diag(np.linalg.inv(M.T @ M)) * s2), s2


print("Phiên bản: numpy %s · scikit-learn %s · scipy %s"
      % (np.__version__, __import__("sklearn").__version__, __import__("scipy").__version__))

# ═══════════════════════════════════════════════════════════════════════════ PHẦN I — [T] 1996
head("1. [T] §2.2 công thức (3) và Hình 1 — thiết kế trực chuẩn XᵀX = I")
rng = np.random.default_rng(0)
Q, _ = np.linalg.qr(rng.normal(size=(60, 6)))
Q -= Q.mean(0)
Q, _ = np.linalg.qr(Q)
yq = Q @ np.array([3.0, -2.0, 1.2, 0.6, -0.3, 0.1]) + 0.2 * rng.normal(size=60)
yq -= yq.mean()
b0q = Q.T @ yq
dq = Duong(Q, yq)
print("   β̂° (OLS) = %s ; t0 = Σ|β̂°| = %.6f ; ‖QᵀQ − I‖ = %.2e"
      % (np.array2string(b0q, precision=4), dq.t0, np.abs(Q.T @ Q - np.eye(6)).max()))


def gamma_cho_t(b0, t):
    lo, hi = 0.0, np.abs(b0).max()
    for _ in range(200):
        g = (lo + hi) / 2
        if np.abs(soft(b0, g)).sum() > t:
            lo = g
        else:
            hi = g
    return (lo + hi) / 2


lech3 = 0.0
for s in (0.1, 0.3, 0.5, 0.7, 0.9):
    t = s * dq.t0
    g = gamma_cho_t(b0q, t)
    lech3 = max(lech3, np.abs(soft(b0q, g) - dq.tai(t)).max())
print("   (3) sign(β̂°)(|β̂°| − γ)⁺ so với nghiệm số của (1), 5 giá trị t: lệch lớn nhất %.3e" % lech3)
# ridge 1/(1+γ)
lam_r = 0.7
br = np.linalg.solve(Q.T @ Q + lam_r * np.eye(6), Q.T @ yq)
print("   ridge β̂°/(1+γ), γ = λ = 0,7: lệch %.3e" % np.abs(br - b0q / 1.7).max())
# garotte (1 − γ/β̂°²)⁺ β̂°  — giải (2) bằng đường lasso DƯƠNG trên Z_j = β̂°_j x_j
Zg = Q * b0q
dg = Duong(Zg, yq, positive=True)
lechg = 0.0
for s in (0.2, 0.5, 0.8):
    t = s * 6
    c = dg.tai(t)
    lo, hi = 0.0, (b0q ** 2).max()
    for _ in range(200):
        g = (lo + hi) / 2
        if np.maximum(1 - g / b0q ** 2, 0).sum() > t:
            lo = g
        else:
            hi = g
    lechg = max(lechg, np.abs(c * b0q - np.maximum(1 - g / b0q ** 2, 0) * b0q).max())
print("   garotte (1 − γ/β̂°²)⁺β̂° so với nghiệm số của (2), 3 giá trị t: lệch lớn nhất %.3e" % lechg)
print("   tổng c_j ở đầu đường garotte = %.6f (= p = 6: c = 1 là nghiệm không ràng buộc)" % dg.t0)
# Hình 1: giá trị các hàm tại β̂° = 5 (mép phải của hình)
bb = np.array([1.0, 2.0, 2.5, 5.0])
print("   Hình 1 — ngưỡng 2 ở cả ba panel (a), (c), (d):")
print("      β̂°            : %s" % bb)
print("      tập con λ = 2  : %s" % (bb * (np.abs(bb) > 2)))
print("      lasso  γ = 2   : %s" % soft(bb, 2.0))
print("      garotte γ = 4  : %s   (1 − 4/β̂°² = 0 đúng tại β̂° = 2)" % (np.maximum(1 - 4 / bb ** 2, 0) * bb))
print("      ridge γ = 2    : %s   (hình không ghi γ của panel b)" % (bb / 3))

head("2. [T] §2.3 — dấu của nghiệm lasso so với dấu của β̂°: p = 2 và p = 3")


def he_so_mau(p, rho, n, r):
    C = rho ** np.abs(np.subtract.outer(np.arange(p), np.arange(p)))
    X = r.multivariate_normal(np.zeros(p), C, size=n)
    X -= X.mean(0)
    X /= np.sqrt((X ** 2).mean(0))
    return X


def lech_dau(X, y, luoi=np.linspace(0.02, 0.98, 25)):
    """Có t nào mà một hệ số lasso khác 0 mang dấu NGƯỢC dấu β̂° không?"""
    d = Duong(X, y)
    b0 = d.co[:, -1]
    for s in luoi:
        b = d.tai(s * d.t0)
        if np.any((np.abs(b) > 1e-10) & (np.sign(b) != np.sign(b0))):
            return True, s, b, b0
    return False, None, None, b0


r2 = np.random.default_rng(1)
dem2 = 0
for _ in range(3000):
    X = he_so_mau(2, r2.uniform(-0.95, 0.95), 30, r2)
    y = X @ r2.normal(size=2) * 2 + r2.normal(size=30)
    dem2 += lech_dau(X, y - y.mean())[0]
print("   p = 2: 3000 bộ dữ liệu, tương quan ngẫu nhiên trong [−0,95; 0,95]: %d bộ có hệ số đổi dấu" % dem2)
r3 = np.random.default_rng(2)
dem3, vd = 0, None
N3 = 3000
for _ in range(N3):
    rho = r3.uniform(0.3, 0.95)
    X = he_so_mau(3, rho, 30, r3)
    y = X @ r3.normal(size=3) * 2 + r3.normal(size=30)
    kq = lech_dau(X, y - y.mean())
    if kq[0]:
        dem3 += 1
        if vd is None:
            vd = (rho, kq)
print("   p = 3, tương quan AR ρ ∈ [0,3; 0,95]: %d/%d bộ (%.4f) có hệ số đổi dấu" % (dem3, N3, dem3 / N3))
rho_vd, (_, s_vd, b_vd, b0_vd) = vd
print("   ví dụ đầu tiên: ρ = %.4f ; β̂° = %s ; tại s = %.2f lasso = %s"
      % (rho_vd, np.array2string(b0_vd, precision=4), s_vd, np.array2string(b_vd, precision=4)))

head("3. [T] §2.4 công thức (5), (6) và Hình 4 — hai biến tương quan")
r4 = np.random.default_rng(3)
X = he_so_mau(2, 0.7, 50, r4)
y = X @ np.array([2.0, 1.2]) + r4.normal(size=50)
y -= y.mean()
d4 = Duong(X, y)
b0 = d4.co[:, -1]
print("   ρ mẫu = %.4f ; β̂° = %s" % (np.corrcoef(X.T)[0, 1], np.array2string(b0, precision=6)))
lech5 = 0.0
for t in np.linspace(abs(b0[0] - b0[1]) + 1e-9, b0.sum(), 20):
    g = (b0.sum() - t) / 2
    lech5 = max(lech5, np.abs(np.maximum(b0 - g, 0) - d4.tai(t)).max())
print("   (5)/(6) với |β̂°₁ − β̂°₂| ≤ t ≤ β̂°₁ + β̂°₂: lệch lớn nhất với nghiệm số %.3e" % lech5)
print("   (6) khi t < β̂°₁ − β̂°₂ — ví dụ của Hình 4, β̂° = (6, 3):")
for t in (0.5, 1.0, 2.0, 2.9):
    in6 = np.maximum(np.array([t / 2 + 1.5, t / 2 - 1.5]), 0)
    print("      t = %.1f: (6) in ra (%.2f, %.2f), tổng %.2f > t ; nghiệm đúng (%.2f, 0.00)"
          % (t, in6[0], in6[1], in6.sum(), t))
# Hình 4
print("   Hình 4: 100 điểm, y = 6x₁ + 3x₂ không nhiễu, x chuẩn hoá theo mẫu")
duong_rho, dinh_ridge = {}, {}
for rho in (0.0, 0.23, 0.45, 0.68, 0.90):
    r5 = np.random.default_rng(4)
    X = he_so_mau(2, rho, 100, r5) if rho > 0 else he_so_mau(2, 1e-12, 100, r5)
    y = X @ np.array([6.0, 3.0])
    y -= y.mean()
    d5 = Duong(X, y)
    duong_rho[rho] = np.array([d5.tai(t) for t in np.linspace(0, 9, 91)])
    lams = np.r_[0, np.logspace(-3, 4, 600)]
    B = np.array([np.linalg.solve(X.T @ X + l * np.eye(2), X.T @ y) for l in lams])
    k = int(np.argmax(B[:, 1]))
    dinh_ridge[rho] = (np.corrcoef(X.T)[0, 1], B[k, 0], B[k, 1])
goc = duong_rho[0.0]
print("   lệch lớn nhất giữa đường lasso của 5 giá trị ρ: %.3e"
      % max(np.abs(v - goc).max() for v in duong_rho.values()))
print("   %8s %10s %14s %14s" % ("ρ in", "ρ mẫu", "β̂₂ ridge lớn nhất", "tại β̂₁ ="))
for rho, (rm, b1, b2) in dinh_ridge.items():
    print("   %8.2f %10.4f %14.6f %14.6f" % (rho, rm, b2, b1))
print("   -> β̂₂ ridge vượt 3 (TĂNG khi cận giảm) ở các ρ: %s"
      % [rho for rho, v in dinh_ridge.items() if v[2] > 3 + 1e-9])

head("4. [T] §3 Bảng 1 — dữ liệu prostate: bản CÓ LỖI lweight (đối tượng 32) và bản đã sửa")
print("   ESL tr.3 chú thích: lweight của đối tượng 32 in 6,1 (449 gm) — đúng là 44,9 gm.")
print("   ln(449) = %.6f ; csv đã sửa: %.10f ; exp = %.4f gm ; hiệu hai giá trị = %.6f ; ln 10 = %.6f"
      % (math.log(449), XP[31, 1], math.exp(XP[31, 1]), math.log(449) - XP[31, 1], math.log(10)))
IN_T1_LS = np.array([2.48, 0.69, 0.23, -0.15, 0.16, 0.32, -0.15, 0.03, 0.13])
IN_T1_Z = np.array([34.46, 6.68, 2.67, -1.76, 1.83, 3.14, -1.16, 0.29, 1.02])
IN_T1_LASSO = np.array([2.48, 0.56, 0.10, 0, 0, 0.16, 0, 0, 0])
for ten_bn, XX in (("bản ĐÃ SỬA (csv)", XP), ("bản CÓ LỖI (449 gm)", XP_LOI)):
    Xs = (XX - XX.mean(0)) / XX.std(0, ddof=1)
    b, se, s2 = ols_se(Xs, YP)
    z = b / se
    yc = YP - YP.mean()
    dd = Duong(Xs, yc)
    bl = dd.tai(0.44 * dd.t0)
    print("   %s" % ten_bn)
    print("      LS hệ số  : %s  khớp %d/9 ô" % (np.array2string(b, precision=4),
                                                (np.abs(np.round(b, 2) - IN_T1_LS) < 5e-3).sum()))
    print("      LS Z-score: %s  khớp %d/9 ô" % (np.array2string(z, precision=4),
                                                (np.abs(np.round(z, 2) - IN_T1_Z) < 5e-3).sum()))
    print("      lasso s = 0,44: %s  khớp %d/9 ô (t0 = %.6f)"
          % (np.array2string(np.r_[YP.mean(), bl], precision=4),
             (np.abs(np.round(np.r_[YP.mean(), bl], 2) - IN_T1_LASSO) < 5e-3).sum(), dd.t0))
XsL = (XP_LOI - XP_LOI.mean(0)) / XP_LOI.std(0, ddof=1)
ycL = YP - YP.mean()
dL = Duong(XsL, ycL)
b_ls_L, se_ls_L, s2_L = ols_se(XsL, YP)
print("   Z-score LS lấy từ số CHƯA làm tròn: 0,69/0,10 = %.2f nhưng in %.2f" % (0.69 / 0.10, 6.68))
# cột tập con
sub = [0, 1, 4]
bs, ses, _ = ols_se(XsL[:, sub], YP)
print("   Tập con {lcavol, lweight, svi}: hệ số %s ; SE %s ; Z %s"
      % (np.array2string(bs, precision=4), np.array2string(ses, precision=4),
         np.array2string(bs / ses, precision=4)))
print("      in: hệ số [2.48 0.65 0.25 0.28] ; SE [0.07 0.09 0.07 0.09] ; Z [34.05 7.39 3.39 3.18]")
print("   Cột lasso: Z in = thương hai số ĐÃ làm tròn?")
for ten_o, he, se_ in (("intcpt", 2.48, 0.07), ("lcavol", 0.56, 0.09), ("lweight", 0.10, 0.07), ("svi", 0.16, 0.09)):
    print("      %-8s %.2f/%.2f = %.4f" % (ten_o, he, se_, he / se_))
print("      in: 35.43 · 6.22 · 1.43 · 1.78")
print("   Câu tr.274 'hệ số và Z của tập con có xu hướng LỚN hơn mô hình đầy đủ' — đếm trên chính Bảng 1:")
print("      hệ số: lcavol 0,65 vs 0,69 · lweight 0,25 vs 0,23 · svi 0,28 vs 0,32 -> lớn hơn ở %d/3"
      % sum(a > b for a, b in [(0.65, 0.69), (0.25, 0.23), (0.28, 0.32)]))
print("      Z    : 7,39 vs 6,68 · 3,39 vs 2,67 · 3,18 vs 3,14 -> lớn hơn ở %d/3"
      % sum(a > b for a, b in [(7.39, 6.68), (3.39, 2.67), (3.18, 3.14)]))
print("   s = 0,44 giữ %d/8 biến khác 0 (%.4f) — bài báo: 'just under half'"
      % ((np.abs(dL.tai(0.44 * dL.t0)) > 1e-12).sum(), (np.abs(dL.tai(0.44 * dL.t0)) > 1e-12).sum() / 8))
# thứ tự vào và tính đơn điệu (Hình 5)
luoi_s = np.linspace(0, 1, 2001)
PL = np.array([dL.tai(s * dL.t0) for s in luoi_s])
nut_vao = []
for k in range(1, dL.co.shape[1]):
    for j in np.nonzero((np.abs(dL.co[:, k]) > 1e-12) & (np.abs(dL.co[:, k - 1]) <= 1e-12))[0]:
        nut_vao.append((dL.tt[k - 1] / dL.t0, TEN[j]))
print("   Hình 5 — biến rời khỏi 0 theo thứ tự (s tại nút): %s"
      % ", ".join("%s %.4f" % (t_, s_) for s_, t_ in nut_vao))
print("   Hình 5 — |β̂_j(s)| đơn điệu không giảm theo s ở %d/8 biến (lưới 2001 điểm)"
      % sum(np.all(np.diff(np.abs(PL[:, j])) >= -1e-12) for j in range(8)))
print("   Hình 5 — hệ số tại s = 1: %s" % np.array2string(PL[-1], precision=4))
print("   Hình 5 — hệ số tại s = 0,44: %s" % np.array2string(dL.tai(0.44 * dL.t0), precision=4))
# mảng cho hình tự vẽ
dS = Duong((XP - XP.mean(0)) / XP.std(0, ddof=1), YP - YP.mean())
print("   [hinh5] s = %s" % np.array2string(dL.tt / dL.t0, precision=10, separator=","))
print("   [hinh5] he so co loi (moi cot mot nut) =\n%s" % np.array2string(dL.co, precision=10, separator=","))
print("   [hinh5] s ban da sua = %s" % np.array2string(dS.tt / dS.t0, precision=10, separator=","))
print("   [hinh5] he so ban da sua =\n%s" % np.array2string(dS.co, precision=10, separator=","))

# đoạn thêm vào mục 4: quy ước chuẩn hoá
for dd_ in (0, 1):
    XsQ = (XP_LOI - XP_LOI.mean(0)) / XP_LOI.std(0, ddof=dd_)
    bq_, _, _ = ols_se(XsQ, YP)
    print("   chuẩn hoá ddof = %d (Σx²/%s = 1): hệ số LS khớp %d/9 ô Bảng 1"
          % (dd_, "N" if dd_ == 0 else "(N−1)", (np.abs(np.round(bq_, 2) - IN_T1_LS) < 5e-3).sum()))
N_P = len(YP)


def gcv_duong(X, y, d, luoi, cach="active"):
    """GCV (10) của [T] trên một lưới s; p(t) theo (9) — 'active': cột có hệ số 0 bị loại."""
    N = len(y)
    kq = []
    for s in luoi:
        b = d.tai(s * d.t0)
        nz = np.abs(b) > 1e-12
        rss = ((y - X @ b) ** 2).sum()
        if not nz.any():
            pt = 0.0
        else:
            g = X.T @ (y - X @ b)
            lam = np.abs(g[nz]).mean()          # (X'X + λW⁻)β = X'y  ⇔  X'(y − Xβ) = λ·sign(β)
            if cach == "active":
                A = X[:, nz]
                pt = np.trace(A @ np.linalg.solve(A.T @ A + lam * np.diag(1 / np.abs(b[nz])), A.T))
            else:
                Wm = np.zeros(X.shape[1])
                Wm[nz] = 1 / np.abs(b[nz])
                pt = np.trace(X @ np.linalg.pinv(X.T @ X + lam * np.diag(Wm)) @ X.T)
        kq.append((rss / N / (1 - pt / N) ** 2, pt))
    return np.array(kq)


head("5. [T] §2.5 công thức (7), Bảng 2 và Hình 6 — sai số chuẩn của lasso trên prostate (bản có lỗi)")
b44 = dL.tai(0.44 * dL.t0)
nz = np.abs(b44) > 1e-12
g44 = XsL.T @ (ycL - XsL @ b44)
lam44 = np.abs(g44[nz]).mean()
print("   tại s = 0,44: λ trong (X'X + λW⁻)β = X'y là %.6f (các |g_j| active: %s)"
      % (lam44, np.array2string(np.abs(g44[nz]), precision=8)))
A = XsL[:, nz]
Mi = np.linalg.inv(A.T @ A + lam44 * np.diag(1 / np.abs(b44[nz])))
print("   kiểm (X_A'X_A + λW⁻)⁻¹X_A'y = β̂_A: lệch %.3e" % np.abs(Mi @ A.T @ ycL - b44[nz]).max())
se7 = np.zeros(9)
se7[0] = math.sqrt(s2_L / N_P)
se7[1:][nz] = np.sqrt(np.diag(Mi @ A.T @ A @ Mi) * s2_L)
print("   σ̂² (RSS/(N−p−1) của LS đầy đủ) = %.6f" % s2_L)
IN_T2_7 = np.array([0.07, 0.09, 0.06, 0, 0, 0.07, 0, 0, 0])
print("   (7), cột có hệ số 0 bị loại: %s  khớp %d/9 ô cột (7) Bảng 2"
      % (np.array2string(se7, precision=4), (np.abs(np.round(se7, 2) - IN_T2_7) < 5e-3).sum()))
Wm = np.zeros(8)
Wm[nz] = 1 / np.abs(b44[nz])
Mg = np.linalg.pinv(XsL.T @ XsL + lam44 * np.diag(Wm))
se7g = np.sqrt(np.diag(Mg @ XsL.T @ XsL @ Mg) * s2_L)
print("   (7) thử các cách đọc khác (cột có hệ số 0 bị loại), in sai số chuẩn của lcavol, lweight, svi:")
for he_so in (0.25, 0.5, 2.0, 4.0):
    Mk = np.linalg.inv(A.T @ A + he_so * lam44 * np.diag(1 / np.abs(b44[nz])))
    print("      λ × %-4g: bánh kẹp %s ; bỏ thừa số giữa %s"
          % (he_so, np.array2string(np.sqrt(np.diag(Mk @ A.T @ A @ Mk) * s2_L), precision=4),
             np.array2string(np.sqrt(np.diag(Mk) * s2_L), precision=4)))
print("      λ × 1   : bỏ thừa số giữa %s ; Bảng 2 in 0.09 0.06 0.07"
      % np.array2string(np.sqrt(np.diag(Mi) * s2_L), precision=4))
print("   (7) đọc NGUYÊN VĂN W⁻ = nghịch đảo suy rộng (ô 0 → phạt 0): %s" % np.array2string(se7g, precision=4))
print("      -> các biến có hệ số 0 KHÔNG nhận phương sai 0, trái câu ngay dưới (7)")
print("   (7) in λX⁻ ở thừa số thứ hai: X⁻ có cỡ %d×%d, XᵀX cỡ 8×8 — phép cộng không xác định"
      % (XsL.shape[1], XsL.shape[0]))
# bootstrap phần dư
fit_ls = np.c_[np.ones(N_P), XsL] @ b_ls_L
res_ls = YP - fit_ls
LUOI_S = np.linspace(0.01, 1.0, 100)
rb = np.random.default_rng(11)
B_CO, B_BIEN = [], []
s_bien = []
for _ in range(200):
    ys = fit_ls + rb.choice(res_ls, N_P, replace=True)
    ysc = ys - ys.mean()
    db_ = Duong(XsL, ysc)
    B_CO.append(np.r_[ys.mean(), db_.tai(0.44 * db_.t0)])
    gv = gcv_duong(XsL, ysc, db_, LUOI_S)
    sb = LUOI_S[int(np.argmin(gv[:, 0]))]
    s_bien.append(sb)
    B_BIEN.append(np.r_[ys.mean(), db_.tai(sb * db_.t0)])
B_CO, B_BIEN = np.array(B_CO), np.array(B_BIEN)
IN_T2_CO = np.array([0.07, 0.08, 0.06, 0.04, 0.04, 0.09, 0.03, 0.02, 0.03])
IN_T2_BIEN = np.array([0.07, 0.10, 0.08, 0.05, 0.07, 0.09, 0.07, 0.05, 0.06])
IN_T1_SE = np.array([0.07, 0.09, 0.07, 0.01, 0.04, 0.09, 0.03, 0.02, 0.03])
print("   bootstrap 200 lần, phần dư LS, hạt giống 11:")
print("   %-9s %9s %9s %9s %9s %9s %9s %9s" % ("biến", "cố định", "in T2", "in T1", "đổi s", "in T2", "5%", "95%"))
for j, t_ in enumerate(["intcpt"] + TEN):
    q5, q95 = np.percentile(B_CO[:, j], [5, 95])
    print("   %-9s %9.4f %9.2f %9.2f %9.4f %9.2f %9.4f %9.4f"
          % (t_, B_CO[:, j].std(ddof=1), IN_T2_CO[j], IN_T1_SE[j], B_BIEN[:, j].std(ddof=1), IN_T2_BIEN[j], q5, q95))
se_co = np.round(B_CO.std(0, ddof=1), 2)
print("   cột 'cố định' làm tròn 2 chữ số khớp Bảng 1 ở %d/9 ô, khớp Bảng 2 'Fixed t' ở %d/9 ô"
      % ((np.abs(se_co - IN_T1_SE) < 5e-3).sum(), (np.abs(se_co - IN_T2_CO) < 5e-3).sum()))
print("   cột 'đổi s' làm tròn 2 chữ số khớp Bảng 2 'Varying t' ở %d/9 ô"
      % (np.abs(np.round(B_BIEN.std(0, ddof=1), 2) - IN_T2_BIEN) < 5e-3).sum())
chua0 = [t_ for j, t_ in enumerate(TEN, 1) if np.percentile(B_CO[:, j], 5) <= 0 <= np.percentile(B_CO[:, j], 95)]
print("   khoảng 5–95%% KHÔNG chứa 0: %s" % [t_ for t_ in TEN if t_ not in chua0])
print("   s chọn lại bằng GCV trên 200 mẫu bootstrap: trung vị %.2f, [%.2f; %.2f]"
      % (np.median(s_bien), min(s_bien), max(s_bien)))
print("   Bảng 1 cột SE của lasso so với Bảng 2 cột 'Fixed t' (văn bản nói hai cột là MỘT phép đo):")
print("      lệch ở %d/9 ô: %s" % ((np.abs(IN_T1_SE - IN_T2_CO) > 1e-9).sum(),
                                   [t_ for t_, a_, b_ in zip(["intcpt"] + TEN, IN_T1_SE, IN_T2_CO) if abs(a_ - b_) > 1e-9]))
print("   số 0 trong 200 bản bootstrap (s cố định): %s"
      % ", ".join("%s %d" % (t_, (np.abs(B_CO[:, j]) < 1e-12).sum()) for j, t_ in enumerate(TEN, 1)))

head("6. [T] §4 — chọn s: GCV (9)–(10), CV năm mảnh, công thức Stein (11)")
gv = gcv_duong(XsL, ycL, dL, LUOI_S)
k_ = int(np.argmin(gv[:, 0]))
print("   GCV (cột 0 bị loại khỏi p(t)): cực tiểu ở s = %.2f, GCV = %.6f, p(t) = %.4f"
      % (LUOI_S[k_], gv[k_, 0], gv[k_, 1]))
k44 = int(np.argmin(np.abs(LUOI_S - 0.44)))
print("   tại s = 0,44: GCV = %.6f, p(t) = %.4f" % (gv[k44, 0], gv[k44, 1]))
gvg = gcv_duong(XsL, ycL, dL, LUOI_S, cach="literal")
print("   GCV đọc W⁻ nguyên văn: cực tiểu ở s = %.2f ; tại 0,44: %.6f"
      % (LUOI_S[int(np.argmin(gvg[:, 0]))], gvg[k44, 0]))
L15 = np.linspace(0, 1, 15)
gv15 = gcv_duong(XsL, ycL, dL, L15)
print("   GCV trên lưới 15 điểm (tr.277 'a grid of 15 values'): cực tiểu ở s = %.4f" % L15[int(np.argmin(gv15[:, 0]))])
print("   [gcv] s = %s" % np.array2string(LUOI_S, precision=4, separator=","))
print("   [gcv] GCV = %s" % np.array2string(gv[:, 0], precision=8, separator=","))
# CV 5 mảnh
chon_cv = []
for seed in range(200):
    rc = np.random.default_rng(seed)
    fold = rc.permutation(np.arange(N_P) % 5)
    err = np.zeros(len(L15))
    for f in range(5):
        a_, b_ = fold != f, fold == f
        mx, my = XsL[a_].mean(0), YP[a_].mean()
        df_ = Duong(XsL[a_] - mx, YP[a_] - my)
        for i_, s in enumerate(L15):
            bb_ = df_.tai(s * df_.t0)
            err[i_] += ((YP[b_] - my - (XsL[b_] - mx) @ bb_) ** 2).sum()
    chon_cv.append(L15[int(np.argmin(err))])
chon_cv = np.array(chon_cv)
print("   CV 5 mảnh, lưới 15 điểm, 200 cách chia: s được chọn")
for v in np.unique(chon_cv):
    print("      s = %.4f : %3d lần" % (v, (chon_cv == v).sum()))
# Stein
sig2_N = ((YP - fit_ls) ** 2).sum() / (N_P - 8)
for dd_ in (1, 0):
    Xq = (XP_LOI - XP_LOI.mean(0)) / XP_LOI.std(0, ddof=dd_)
    b0s = np.linalg.lstsq(Xq, ycL, rcond=None)[0]
    tau = math.sqrt(sig2_N / N_P)
    z = b0s / tau
    ung = np.r_[0.0, np.sort(np.abs(z)) * (1 + 1e-12), np.linspace(0, np.abs(z).max() + 1, 4001)]
    ung = np.unique(ung)
    Rmin = np.array([8 - 2 * (np.abs(z) < g).sum() + (np.minimum(np.abs(z), g) ** 2).sum() for g in ung])
    Rmax = np.array([8 - 2 * (np.abs(z) < g).sum() + (np.maximum(np.abs(z), g) ** 2).sum() for g in ung])
    gmin, gmax = ung[int(np.argmin(Rmin))], ung[int(np.argmin(Rmax))]
    t_in = np.maximum(np.abs(b0s) - gmin, 0).sum()
    t_dv = np.maximum(np.abs(b0s) - tau * gmin, 0).sum()
    print("   Stein, chuẩn hoá ddof = %d: τ̂ = %.6f ; |β̂°/τ̂| = %s" % (dd_, tau, np.array2string(np.abs(z), precision=4)))
    print("      công thức đúng (min): γ̂ = %.6f ; t̂ = Σ(|β̂°| − τ̂γ̂)⁺ = %.6f -> ŝ = %.4f"
          % (gmin, t_dv, t_dv / np.abs(b0s).sum()))
    print("      t̂ in nguyên văn Σ(|β̂°| − γ̂)⁺ = %.6f -> ŝ = %.4f" % (t_in, t_in / np.abs(b0s).sum()))
    print("      công thức in (max): γ̂ = %.6f -> ŝ = %.4f"
          % (gmax, np.maximum(np.abs(b0s) - tau * gmax, 0).sum() / np.abs(b0s).sum()))
# Monte Carlo: công thức Stein (min) có phải ước lượng không chệch của rủi ro ngưỡng mềm không
rm = np.random.default_rng(12)
mu = np.r_[np.zeros(30), rm.normal(0, 3, 20)]
ZZ = mu + rm.normal(size=(20000, 50))
print("   Monte Carlo 20 000 lần, p = 50, E‖ngưỡng mềm − μ‖² so với trung bình công thức:")
print("   %6s %12s %14s %14s" % ("γ", "rủi ro thật", "công thức min", "công thức max"))
for g in (0.5, 1.0, 1.5, 2.0, 3.0):
    that = ((soft(ZZ, g) - mu) ** 2).sum(1).mean()
    fmin = (50 - 2 * (np.abs(ZZ) < g).sum(1) + (np.minimum(np.abs(ZZ), g) ** 2).sum(1)).mean()
    fmax = (50 - 2 * (np.abs(ZZ) < g).sum(1) + (np.maximum(np.abs(ZZ), g) ** 2).sum(1)).mean()
    print("   %6.1f %12.4f %14.4f %14.4f" % (g, that, fmin, fmax))
print("   tr.277: lưới 15 giá trị × CV 5 mảnh = %d lần tối ưu; Stein 1 lần; GCV %d lần" % (15 * 5, 15))
# lập luận Z = XV^{-1/2}, θ = βV^{-1/2}
V = XsL.T @ XsL
w_, U_ = np.linalg.eigh(V)
Vm = U_ @ np.diag(w_ ** -0.5) @ U_.T
Vp = U_ @ np.diag(w_ ** 0.5) @ U_.T
Zs = XsL @ Vm
bt = b_ls_L[1:]
print("   tr.277: Z = XV^(−1/2): ZᵀZ = I lệch %.3e" % np.abs(Zs.T @ Zs - np.eye(8)).max())
print("      θ = βV^(−1/2) (như in): ‖Zθ − Xβ‖ = %.4e ; θ = V^(1/2)β: ‖Zθ − Xβ‖ = %.3e"
      % (np.linalg.norm(Zs @ (Vm @ bt) - XsL @ bt), np.linalg.norm(Zs @ (Vp @ bt) - XsL @ bt)))

head("7. [T] §5 — lasso là mode hậu nghiệm với tiên nghiệm Laplace; Hình 7")
lam_rss = 2 * lam44          # dạng RSS + λΣ|β| của [T]: 2X'(y − Xβ) = λ·sign(β)
print("   prostate, s = 0,44: λ (dạng RSS + λΣ|β|) = %.6f ; σ̂² = %.6f" % (lam_rss, s2_L))
print("   −log hậu nghiệm = RSS/(2σ²) + Σ|β|/τ  ⇒  λ = 2σ²/τ  ⇒  τ = 2σ²/λ = %.6f" % (2 * s2_L / lam_rss))
print("   bài báo in τ = 1/λ = %.6f — hai giá trị bằng nhau chỉ khi σ² = 1/2" % (1 / lam_rss))
# kiểm số: mode hậu nghiệm tính trực tiếp bằng tối ưu so với lasso
tau_ = 2 * s2_L / lam_rss


def am_log_hn(b):
    r_ = ycL - XsL @ b
    return r_ @ r_ / (2 * s2_L) + np.abs(b).sum() / tau_


mode = optimize.minimize(am_log_hn, b44 + 0.01, method="Powell",
                         options={"xtol": 1e-12, "ftol": 1e-14, "maxiter": 200000}).x
print("   mode hậu nghiệm tìm bằng Powell (τ = 2σ²/λ) so với lasso s = 0,44: lệch %.3e"
      % np.abs(mode - b44).max())
bg = np.linspace(-30, 30, 600001)
dx = bg[1] - bg[0]
print("   một chiều, z ~ N(β, 1), β ~ Laplace(τ = 1): mode, trung bình, trung vị hậu nghiệm")
print("   %6s %10s %12s %12s" % ("z", "mode", "trung bình", "trung vị"))
for zv in (0.0, 0.25, 0.5, 1.0, 2.0, 3.0):
    lp = -(zv - bg) ** 2 / 2 - np.abs(bg)
    w = np.exp(lp - lp.max())
    w /= w.sum()
    cdf = np.cumsum(w)
    print("   %6.2f %10.4f %12.6f %12.6f"
          % (zv, soft(zv, 1.0), (w * bg).sum(), bg[int(np.searchsorted(cdf, 0.5))]))
for ten_p, tau_l in (("τ = 1 (đỉnh 0,5)", 1.0), ("cùng phương sai 1, τ = 1/√2", 1 / math.sqrt(2))):
    print("   Laplace %s so với N(0, 1):" % ten_p)
    print("      mật độ tại 0: %.4f vs %.4f ; P(|β| < 0,5): %.4f vs %.4f ; P(|β| > 3): %.4f vs %.4f"
          % (1 / (2 * tau_l), stats.norm.pdf(0), 1 - math.exp(-0.5 / tau_l), 2 * stats.norm.cdf(0.5) - 1,
             math.exp(-3 / tau_l), 2 * stats.norm.sf(3)))

head("8. [T] §6 — thuật toán ràng buộc dấu (Lawson–Hansen) và cách của David Gay")


def qp_tap_hoat(H, c, G, h, b):
    """min ½bᵀHb − cᵀb  với  Gb ≤ h ; phương pháp tập hoạt động nguyên thuỷ, giải tuyến tính chính xác.

    b phải khả thi lúc bắt đầu. Trả (b, số lần giải hệ KKT).
    """
    W = [i for i in range(len(h)) if abs(G[i] @ b - h[i]) < 1e-12 * max(1, abs(h[i]))]
    lap = 0
    while True:
        lap += 1
        k = len(W)
        K = np.zeros((len(b) + k, len(b) + k))
        K[:len(b), :len(b)] = H
        if k:
            K[:len(b), len(b):] = G[W].T
            K[len(b):, :len(b)] = G[W]
        rhs = np.r_[c, h[W]] if k else c
        nghiem = np.linalg.lstsq(K, rhs, rcond=None)[0]
        bm, nu = nghiem[:len(b)], nghiem[len(b):]
        d = bm - b
        if np.abs(d).max() < 1e-13 * max(1, np.abs(b).max()):
            if k == 0 or nu.min() >= -1e-12:
                return bm, lap
            W.pop(int(np.argmin(nu)))
            continue
        buoc, chan = 1.0, None
        for i in range(len(h)):
            if i in W:
                continue
            gd = G[i] @ d
            if gd > 1e-15:
                a = (h[i] - G[i] @ b) / gd
                if a < buoc:
                    buoc, chan = a, i
        b = b + buoc * d
        if chan is not None:
            W.append(chan)


def lh_lasso(X, y, t):
    """Đúng các bước (a)–(d) tr.278: thêm dần ràng buộc δᵢᵀβ ≤ t."""
    H, c = X.T @ X, X.T @ y
    b0 = np.linalg.solve(H, c)
    if np.abs(b0).sum() <= t:
        return b0, 0, 0
    dau = lambda b: np.where(b >= 0, 1.0, -1.0)
    E = [dau(b0)]
    so_0 = 0
    b = b0 * t / np.abs(b0).sum()          # điểm khả thi với MỌI ràng buộc dấu
    while True:
        G = np.array(E)
        b, _ = qp_tap_hoat(H, c, G, np.full(len(E), t), b)
        if np.abs(b).sum() <= t * (1 + 1e-10):
            return b, len(E), so_0
        b = np.where(np.abs(b) < 1e-12 * t, 0.0, b)
        moi = dau(b)
        so_0 += int((b == 0).any())
        if any(np.array_equal(moi, e) for e in E):
            raise RuntimeError("δ đã có trong E: Σ|β| − t = %.3e" % (np.abs(b).sum() - t))
        E.append(moi)
        # bước (b) lại bắt đầu từ điểm khả thi với mọi ràng buộc
        b = b * t / np.abs(b).sum()


def gay_lasso(X, y, t):
    p = X.shape[1]
    XX = np.c_[X, -X]
    f = lambda u: ((y - XX @ u) ** 2).sum()
    jac = lambda u: -2 * XX.T @ (y - XX @ u)
    kq = optimize.minimize(f, np.full(2 * p, t / (4 * p)), jac=jac, method="SLSQP",
                           bounds=[(0, None)] * (2 * p),
                           constraints=[{"type": "ineq", "fun": lambda u: t - u.sum(), "jac": lambda u: -np.ones(2 * p)}],
                           options={"ftol": 1e-15, "maxiter": 5000})
    return kq.x[:p] - kq.x[p:]


print("   p = 8: số ràng buộc dạng (1) = 2^p = %d ; cách của Gay: %d biến, %d ràng buộc" % (2 ** 8, 16, 17))
print("   %4s %6s %10s %10s %12s %14s %14s" % ("p", "s", "số vòng TB", "TB / p", "[min; max]", "lệch LARS", "lệch Gay"))
tong_vong = {}
for p in (4, 8, 16):
    for s in (0.25, 0.5, 0.75):
        rl = np.random.default_rng(100 + p)
        vong, lech_, lechg_ = [], 0.0, 0.0
        for _ in range(40):
            X = he_so_mau(p, 0.5, 100, rl)
            y = X @ rl.normal(size=p) + rl.normal(size=100)
            y -= y.mean()
            dd_ = Duong(X, y)
            t = s * dd_.t0
            b, k, _ = lh_lasso(X, y, t)
            vong.append(k)
            lech_ = max(lech_, np.abs(b - dd_.tai(t)).max())
            if p <= 8:
                lechg_ = max(lechg_, np.abs(gay_lasso(X, y, t) - dd_.tai(t)).max())
        tong_vong[(p, s)] = vong
        print("   %4d %6.2f %10.3f %10.3f %12s %14.3e %14s"
              % (p, s, np.mean(vong), np.mean(vong) / p, "[%d; %d]" % (min(vong), max(vong)), lech_,
                 "%.3e" % lechg_ if p <= 8 else "—"))

head("9. [T] §7 — bốn mô phỏng, Bảng 3–8: dựng lại và kiểm chéo")
L15 = np.linspace(0, 1, 15)
TO_HOP8 = [c for k in range(1, 9) for c in itertools.combinations(range(8), k)]


def sinh(vd, r, sig_thay=None):
    if vd == 4:
        n, p, sig = 100, 40, 15.0
        X = r.normal(size=(n, p)) + r.normal(size=(n, 1))
        beta = np.r_[np.zeros(10), np.full(10, 2.0), np.zeros(10), np.full(10, 2.0)]
        return X, X @ beta + sig * r.normal(size=n), beta, np.eye(p) + np.ones((p, p))
    n, p = 20, 8
    V = 0.5 ** np.abs(np.subtract.outer(np.arange(p), np.arange(p)))
    beta, sig = {1: (np.array([3, 1.5, 0, 0, 2, 0, 0, 0.0]), 3.0),
                 2: (np.full(8, 0.85), 3.0),
                 3: (np.array([5, 0, 0, 0, 0, 0, 0, 0.0]), 2.0)}[vd]
    sig = sig if sig_thay is None else sig_thay
    X = r.multivariate_normal(np.zeros(p), V, size=n)
    return X, X @ beta + sig * r.normal(size=n), beta, V


def chuan(X, y):
    m, sd = X.mean(0), X.std(0)
    return (X - m) / sd, y - y.mean(), m, sd


def df_to_lam(d, dv):
    if dv <= 0:
        return np.inf
    if dv >= (d > 1e-12).sum():
        return 0.0
    lo, hi = 0.0, 1e8
    for _ in range(100):
        mid = (lo + hi) / 2
        if (d ** 2 / (d ** 2 + mid)).sum() > dv:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def best_subsets(X, y):
    """Tập con tốt nhất (RSS nhỏ nhất) cho mỗi cỡ 0..p; X, y đã center."""
    tot = {0: ()}
    rss_tot = {0: y @ y}
    for c in TO_HOP8:
        A = X[:, c]
        b = np.linalg.lstsq(A, y, rcond=None)[0]
        rss = ((y - A @ b) ** 2).sum()
        if rss < rss_tot.get(len(c), np.inf):
            rss_tot[len(c)], tot[len(c)] = rss, c
    return tot


def khop_chon(vd, X, y, r):
    """Mọi phương pháp của [T] §7 trên một bộ dữ liệu; trả {tên: (β gốc thang, s hoặc None)}."""
    n, p = X.shape
    Xs, yc, m, sd = chuan(X, y)
    kq = {}
    b_ls = np.linalg.lstsq(Xs, yc, rcond=None)[0]
    kq["LS"] = (b_ls, None)
    d = Duong(Xs, yc)
    fold = r.permutation(np.arange(n) % 5)
    # CV cho lasso, garotte, ridge (và tập con nếu p = 8)
    e_la, e_ga, e_ri = np.zeros(15), np.zeros(15), np.zeros(15)
    e_bs = np.zeros(p + 1)
    for f in range(5):
        a_, b_ = fold != f, fold == f
        Xa, ya, ma, sa = chuan(X[a_], y[a_])
        Xt = (X[b_] - ma) / sa
        yt = y[b_] - y[a_].mean()
        da = Duong(Xa, ya)
        bo = np.linalg.lstsq(Xa, ya, rcond=None)[0]
        ga = Duong(Xa * bo, ya, positive=True)
        U, dd, Vt = np.linalg.svd(Xa, full_matrices=False)
        for i_, s in enumerate(L15):
            e_la[i_] += ((yt - Xt @ da.tai(s * da.t0)) ** 2).sum()
            e_ga[i_] += ((yt - Xt @ (ga.tai(s * p) * bo)) ** 2).sum()
            lv = df_to_lam(dd, s * p)
            br_ = np.zeros(p) if np.isinf(lv) else Vt.T @ ((dd / (dd ** 2 + lv)) * (U.T @ ya))
            e_ri[i_] += ((yt - Xt @ br_) ** 2).sum()
        if p == 8:
            for J, c in best_subsets(Xa, ya).items():
                if J == 0:
                    e_bs[J] += (yt ** 2).sum()
                else:
                    bc = np.linalg.lstsq(Xa[:, c], ya, rcond=None)[0]
                    e_bs[J] += ((yt - Xt[:, c] @ bc) ** 2).sum()
    s = L15[int(np.argmin(e_la))]
    kq["Lasso (CV)"] = (d.tai(s * d.t0), s)
    gv_ = gcv_duong(Xs, yc, d, L15)
    s = L15[int(np.argmin(gv_[:, 0]))]
    kq["Lasso (GCV)"] = (d.tai(s * d.t0), s)
    rss = ((yc - Xs @ b_ls) ** 2).sum()
    tau = math.sqrt(rss / (n - p) / n)
    z = np.abs(b_ls) / tau
    ung = np.unique(np.r_[0.0, z * (1 + 1e-12)])
    R_ = [p - 2 * (z < g).sum() + (np.minimum(z, g) ** 2).sum() for g in ung]
    g_ = ung[int(np.argmin(R_))]
    t_ = np.maximum(np.abs(b_ls) - tau * g_, 0).sum()
    kq["Lasso (Stein)"] = (d.tai(t_), t_ / d.t0)
    Z = Xs * b_ls
    gz = Duong(Z, yc, positive=True)
    s = L15[int(np.argmin(e_ga))]
    kq["Garotte"] = (gz.tai(s * p) * b_ls, None)
    if p == 8:
        J = int(np.argmin(e_bs))
        c = best_subsets(Xs, yc)[J]
        bb_ = np.zeros(p)
        if J:
            bb_[list(c)] = np.linalg.lstsq(Xs[:, c], yc, rcond=None)[0]
        kq["Best subset"] = (bb_, None)
    U, dd, Vt = np.linalg.svd(Xs, full_matrices=False)
    lv = df_to_lam(dd, L15[int(np.argmin(e_ri))] * p)
    kq["Ridge"] = (np.zeros(p) if np.isinf(lv) else Vt.T @ ((dd / (dd ** 2 + lv)) * (U.T @ yc)), None)
    return {k: (v / sd, s_) for k, (v, s_) in kq.items()}


IN_BANG = {
    1: {"LS": (2.79, 0.12, 0.0), "Lasso (CV)": (2.43, 0.14, 3.3), "Lasso (Stein)": (2.07, 0.10, 2.6),
        "Lasso (GCV)": (1.93, 0.09, 2.4), "Garotte": (2.29, 0.16, 3.9), "Best subset": (2.44, 0.16, 4.8),
        "Ridge": (3.21, 0.12, 0.0)},
    2: {"LS": (6.50, 0.64, 0.0), "Lasso (CV)": (5.30, 0.45, 3.0), "Lasso (Stein)": (5.85, 0.36, 2.7),
        "Lasso (GCV)": (4.87, 0.35, 2.3), "Garotte": (7.40, 0.48, 4.3), "Best subset": (9.05, 0.78, 5.2),
        "Ridge": (2.30, 0.22, 0.0)},
    3: {"LS": (2.89, 0.04, 0.0), "Lasso (CV)": (0.89, 0.01, 3.0), "Lasso (Stein)": (1.26, 0.02, 2.6),
        "Lasso (GCV)": (1.02, 0.02, 3.9), "Garotte": (0.52, 0.01, 5.5), "Best subset": (0.64, 0.02, 6.3),
        "Ridge": (3.53, 0.05, 0.0)},
    4: {"LS": (137.3, 7.3, 0.0), "Lasso (Stein)": (80.2, 4.9, 14.4), "Lasso (GCV)": (64.9, 2.3, 13.6),
        "Garotte": (94.8, 3.2, 22.9), "Ridge": (57.4, 1.4, 0.0)},
}
IN_S = {1: {"Lasso (CV)": 0.63, "Lasso (Stein)": 0.69, "Lasso (GCV)": 0.73},
        2: {"Lasso (CV)": 0.50, "Lasso (Stein)": 0.55, "Lasso (GCV)": 0.69},
        3: {"Lasso (CV)": 0.50, "Lasso (Stein)": 0.70, "Lasso (GCV)": 0.63},
        4: {"Lasso (Stein)": 0.55, "Lasso (GCV)": 0.60}}
KET_QUA_MP = {}
MO_HINH = {"Lasso (GCV)": [], "Best subset": []}
for vd, so_lan in ((1, 200), (2, 200), (3, 200), (4, 50)):
    r = np.random.default_rng(1000 + vd)
    ghi = {}
    for lan in range(so_lan):
        X, y, beta, V = sinh(vd, r)
        for k, (b, s_) in khop_chon(vd, X, y, r).items():
            me = (b - beta) @ V @ (b - beta)
            ghi.setdefault(k, []).append((me, int((b == 0).sum()), s_ if s_ is not None else np.nan))
            if vd == 1 and k in MO_HINH:
                MO_HINH[k].append(tuple(int(j) + 1 for j in np.nonzero(b)[0]))
            if vd == 4 and k == "Lasso (GCV)":
                ghi.setdefault("khoi", []).append([b[i:i + 10].mean() for i in (0, 10, 20, 30)])
    KET_QUA_MP[vd] = ghi
    print("   Ví dụ %d — %d bộ dữ liệu, hạt giống %d" % (vd, so_lan, 1000 + vd))
    print("   %-15s %10s %8s %10s %8s %9s %6s %8s %8s %6s"
          % ("phương pháp", "trung vị ME", "(SE)", "in", "(SE in)", "số 0 TB", "in", "s TB", "(SE)", "in"))
    rb = np.random.default_rng(7)
    for k in IN_BANG[vd]:
        if k not in ghi:
            continue
        me = np.array([v[0] for v in ghi[k]])
        se_med = np.std([np.median(rb.choice(me, len(me))) for _ in range(500)], ddof=1)
        z0 = np.mean([v[1] for v in ghi[k]])
        ss = np.array([v[2] for v in ghi[k]])
        pin = IN_BANG[vd][k]
        s_txt = ("%8.4f %8.4f %6.2f" % (np.nanmean(ss), np.nanstd(ss, ddof=1) / math.sqrt(len(ss)), IN_S[vd][k])
                 if k in IN_S[vd] else "%8s %8s %6s" % ("—", "—", "—"))
        print("   %-15s %10.4f %8.4f %10.2f %8.2f %9.3f %6.1f %s"
              % (k, np.median(me), se_med, pin[0], pin[1], z0, pin[2], s_txt))
    if vd == 4:
        khoi = np.array(ghi["khoi"])
        print("   ví dụ 4, lasso (GCV): trung bình hệ số trong 4 khối 10 biến = %s ; SE %s"
              % (np.array2string(khoi.mean(0), precision=4), np.array2string(khoi.std(0, ddof=1) / math.sqrt(50), precision=4)))
        print("      in: 0.50 (0.06), 0.92 (0.07), 1.56 (0.08), 2.33 (0.09) ; hệ số thật theo khối: 0, 2, 0, 2")
    print(flush=True)

print("   Bảng 4/5 dựng lại (ví dụ 1): mô hình hay gặp nhất")
for k, ds in MO_HINH.items():
    from collections import Counter
    cnt = Counter("".join(map(str, m_)) for m_ in ds)
    chua = np.mean([set((1, 2, 5)) <= set(m_) for m_ in ds])
    dung = np.mean([set(m_) == {1, 2, 5} for m_ in ds])
    print("      %-12s top 5: %s ; đúng mô hình {1,2,5}: %.3f ; chứa {1,2,5}: %.3f"
          % (k, ", ".join("%s %.3f" % (a, b / len(ds)) for a, b in cnt.most_common(5)), dung, chua))
print("      in: lasso GCV đúng 0.025, chứa 0.955 ; tập con đúng 0.240, chứa 0.535")
# giả thuyết: ví dụ 1 thật ra chạy với σ = 2
r = np.random.default_rng(1001)
ghi2, mh2 = {}, {"Lasso (GCV)": [], "Best subset": []}
for lan in range(200):
    X, y, beta, V = sinh(1, r, sig_thay=2.0)
    for k, (b, s_) in khop_chon(1, X, y, r).items():
        ghi2.setdefault(k, []).append(((b - beta) @ V @ (b - beta), int((b == 0).sum())))
        if k in mh2:
            mh2[k].append(tuple(int(j) + 1 for j in np.nonzero(b)[0]))
print("\n   Thử lại ví dụ 1 với σ = 2 (giả thuyết), hạt giống 1001:")
for k in IN_BANG[1]:
    me = [v[0] for v in ghi2[k]]
    print("      %-15s trung vị ME %.4f (in %.2f) ; số 0 TB %.3f (in %.1f)"
          % (k, np.median(me), IN_BANG[1][k][0], np.mean([v[1] for v in ghi2[k]]), IN_BANG[1][k][2]))
for k, ds in mh2.items():
    print("      %-12s đúng {1,2,5}: %.3f ; chứa {1,2,5}: %.3f"
          % (k, np.mean([set(m_) == {1, 2, 5} for m_ in ds]), np.mean([set((1, 2, 5)) <= set(m_) for m_ in ds])))
# kiểm chéo không cần mô phỏng
print("\n   Kiểm chéo tự lộ:")
for a_ in (0.055, 0.050, 0.045, 0.025, 0.240, 0.200, 0.095, 0.040, 0.955, 0.535):
    print("      tỉ lệ %.3f × 200 = %6.1f ; × 50 = %6.2f" % (a_, a_ * 200, a_ * 50))
for vd_, n_, se_in in ((1, 200, 0.01), (2, 200, 0.23), (3, 200, 0.04), (4, 50, 0.88), (4, 50, 0.02)):
    print("      ví dụ %d: s ∈ [0, 1] nên độ lệch chuẩn ≤ 0,5 ⇒ SE của trung bình ≤ %.4f (n = %d) ; in %.2f"
          % (vd_, 0.5 / math.sqrt(n_), n_, se_in))
rl = np.random.default_rng(55)
print("   Trung vị ME của LS chỉ phụ thuộc (n, p, σ, V) — KHÔNG phụ thuộc β (20 000 lần):")
for ten_, n_, p_, sg, V_ in (("ví dụ 1, 2 (σ = 3)", 20, 8, 3.0, 0.5 ** np.abs(np.subtract.outer(np.arange(8), np.arange(8)))),
                              ("ví dụ 3 (σ = 2)", 20, 8, 2.0, 0.5 ** np.abs(np.subtract.outer(np.arange(8), np.arange(8)))),
                              ("ví dụ 4 (σ = 15)", 100, 40, 15.0, np.eye(40) + np.ones((40, 40)))):
    Lc = np.linalg.cholesky(V_)
    me_ = []
    for _ in range(20000 if p_ == 8 else 4000):
        X = rl.normal(size=(n_, p_)) @ Lc.T
        e = sg * rl.normal(size=n_)
        M = np.c_[np.ones(n_), X]
        db = np.linalg.lstsq(M, e, rcond=None)[0][1:]
        me_.append(db @ V_ @ db)
    print("      %-20s trung vị %.4f ; trung bình %.4f" % (ten_, np.median(me_), np.mean(me_)))
print("      in: ví dụ 1 = 2.79 · ví dụ 2 = 6.50 · ví dụ 3 = 2.89 · ví dụ 4 = 137.3 ; 2.79 × 4/9 = %.4f" % (2.79 * 4 / 9))
print("   Tỉ số tín hiệu/nhiễu — bài báo không định nghĩa; bốn cách đọc:")
print("   %-10s %10s %10s %12s %10s %6s" % ("ví dụ", "β'Vβ/σ²", "căn", "1 + β'Vβ/σ²", "√(1+·)", "in"))
V8 = 0.5 ** np.abs(np.subtract.outer(np.arange(8), np.arange(8)))
for vd_, b_, sg, V_, pin in ((1, np.array([3, 1.5, 0, 0, 2, 0, 0, 0.0]), 3, V8, 5.7), (2, np.full(8, 0.85), 3, V8, 1.8),
                             (3, np.r_[5.0, np.zeros(7)], 2, V8, 7.0),
                             (4, np.r_[np.zeros(10), np.full(10, 2.0), np.zeros(10), np.full(10, 2.0)], 15,
                              np.eye(40) + np.ones((40, 40)), 9.0)):
    q = b_ @ V_ @ b_ / sg ** 2
    print("   %-10d %10.4f %10.4f %12.4f %10.4f %6.1f" % (vd_, q, math.sqrt(q), 1 + q, math.sqrt(1 + q), pin))
print("   ví dụ 1 nếu σ = 2: β'Vβ/σ² = %.4f" % (np.array([3, 1.5, 0, 0, 2, 0, 0, 0.0]) @ V8 @ np.array([3, 1.5, 0, 0, 2, 0, 0, 0.0]) / 4))
print("   Tương quan cặp ở ví dụ 4: cov(x_j, x_k) / var(x_j) = 1/2 = %.2f" % (1 / 2))
m4r = np.median([v[0] for v in KET_QUA_MP[4]["Ridge"]])
print("   ví dụ 4, ridge: in / dựng lại = 57.4 / %.4f = %.4f" % (m4r, 57.4 / m4r))

head("10. [T] §8.1 — hồi quy logistic trên dữ liệu kyphosis (81 dòng, không phải 83)")
kr = list(csv.DictReader(open(os.path.join(HERE, "kyphosis.csv"), encoding="utf-8")))
yk = np.array([r_["Kyphosis"] == "present" for r_ in kr], dtype=float)
Xk = np.array([[float(r_["Age"]), float(r_["Number"]), float(r_["Start"])] for r_ in kr])
Xk = Xk - Xk.mean(0)
Fk = np.c_[Xk, Xk ** 2]
Fk = (Fk - Fk.mean(0)) / Fk.std(0, ddof=1)
TEN_K = ["x1", "x2", "x3", "x1²", "x2²", "x3²"]


def logit_irls(F, y, cot=None, t=None, tol=1e-6):
    cot = list(range(F.shape[1])) if cot is None else cot
    A = F[:, cot]
    b0, b = 0.0, np.zeros(len(cot))
    for lap in range(1, 200):
        eta = b0 + A @ b
        pr = 1 / (1 + np.exp(-eta))
        w = pr * (1 - pr)
        zr = eta + (y - pr) / w
        sw = np.sqrt(w)
        if t is None:
            M = np.c_[np.ones(len(y)), A] * sw[:, None]
            moi = np.linalg.lstsq(M, zr * sw, rcond=None)[0]
            b0m, bm = moi[0], moi[1:]
        else:
            mw = (w * zr).sum() / w.sum()
            mx = (w[:, None] * A).sum(0) / w.sum()
            Aw = (A - mx) * sw[:, None]
            dw = Duong(Aw, (zr - mw) * sw)
            bm = dw.tai(t)
            b0m = mw - mx @ bm
        doi = ((np.r_[b0m, bm] - np.r_[b0, b]) ** 2).sum()
        b0, b = b0m, bm
        if doi < tol:
            break
    ll = (y * (b0 + A @ b) - np.log1p(np.exp(b0 + A @ b))).sum()
    return b0, b, lap, ll


b0k, bk, lapk, llk = logit_irls(Fk, yk)
print("   n = %d. Mô hình đầy đủ: %.4f + %s" % (len(yk), b0k, " ".join("%+.4f·%s" % (v, n_) for v, n_ in zip(bk, TEN_K))))
print("   in (83 dòng): −2.64 + 0.83x1 + 0.77x2 − 2.28x3 − 1.55x1² + 0.03x2² − 1.17x3²")
aic = {n_: None for n_ in TEN_K}
cot = list(range(6))
while True:
    base = -2 * logit_irls(Fk, yk, cot)[3] + 2 * (len(cot) + 1)
    thu = [(-2 * logit_irls(Fk, yk, [c for c in cot if c != j])[3] + 2 * len(cot), j) for j in cot]
    tot_ = min(thu)
    if tot_[0] >= base:
        break
    print("   lùi theo AIC: bỏ %s (AIC %.4f -> %.4f)" % (TEN_K[tot_[1]], base, tot_[0]))
    cot.remove(tot_[1])
b0a, ba, _, _ = logit_irls(Fk, yk, cot)
print("   mô hình sau khi lùi: %.4f + %s" % (b0a, " ".join("%+.4f·%s" % (v, TEN_K[c]) for v, c in zip(ba, cot))))
t33 = 0.33 * np.abs(bk).sum()
b0l, bl, lapl, _ = logit_irls(Fk, yk, t=t33)
print("   lasso s = 0,33: %.4f + %s ; hội tụ sau %d vòng IRLS (‖Δβ‖² < 1e−6)"
      % (b0l, " ".join("%+.4f·%s" % (v, n_) for v, n_ in zip(bl, TEN_K)), lapl))
print("   in: −1.42 + 0.03x1 + 0.31x2 − 0.48x3 − 0.28x1², năm vòng")

head("11. [T] §10 — công thức (13)–(16): ngưỡng cứng, ngưỡng mềm và rủi ro lý tưởng")
print("   (14) in z_j ~ N(0, σ²): với σ = 2, phương sai của σz_j là σ⁴ = %.0f, không phải σ² = %.0f" % (2 ** 4, 2 ** 2))
rt = np.random.default_rng(21)
p_ = 1024
for sg in (1.0,):
    beta = np.zeros(p_)
    beta[:20] = 5.0
    beta[20:60] = 0.5
    rdp = np.minimum(beta ** 2, sg ** 2).sum()
    g = sg * math.sqrt(2 * math.log(p_))
    Zm = beta + sg * rt.normal(size=(4000, p_))
    r_cung = ((Zm * (np.abs(Zm) > g) - beta) ** 2).sum(1).mean()
    r_mem = ((soft(Zm, g) - beta) ** 2).sum(1).mean()
    r_ls = ((Zm - beta) ** 2).sum(1).mean()
    print("   p = n = %d, σ = %.0f, 20 hệ số 5 + 40 hệ số 0,5: R_DP = %.4f ; γ = σ√(2 log p) = %.4f" % (p_, sg, rdp, g))
    print("   rủi ro: LS %.4f · ngưỡng cứng %.4f · ngưỡng mềm %.4f ; cận (16) (2 log p + 1)(σ² + R_DP) = %.4f"
          % (r_ls, r_cung, r_mem, (2 * math.log(p_) + 1) * (sg ** 2 + rdp)))
    print("   tỉ số rủi ro / R_DP: cứng %.4f · mềm %.4f ; 2 log p + 1 = %.4f"
          % (r_cung / rdp, r_mem / rdp, 2 * math.log(p_) + 1))
    print("   mềm / cứng = %.4f ; mềm / cận = %.4f ; cứng / cận = %.4f"
          % (r_mem / r_cung, r_mem / ((2 * math.log(p_) + 1) * (sg ** 2 + rdp)),
             r_cung / ((2 * math.log(p_) + 1) * (sg ** 2 + rdp))))

head("12. [T] §11 Hình 9 — miền Σ|β_j|^q ≤ 1 lồi khi nào")
for q in (4, 2, 1, 0.5, 0.1):
    gia = 2 * 0.5 ** q
    print("   q = %-4g: điểm giữa của (1,0) và (0,1) có Σ|β|^q = %.4f -> %s"
          % (q, gia, "trong miền" if gia <= 1 + 1e-12 else "NGOÀI miền: không lồi"))

# ═══════════════════════════════════════════════════════════════════════════ PHẦN II — [R] 2011
head("13. [R] 2011 — chuẩn tổng quát không trị tuyệt đối, và cách viết của Holmes (3)")
bv = np.array([1.0, -1.0])
print("   (Σβ_j^q)^(1/q) in không có trị tuyệt đối: với β = (1, −1), q = 1 cho %.1f ; Σ|β_j| = %.1f"
      % (bv.sum(), np.abs(bv).sum()))
# Holmes: max_β,η  −½RSS − Σ β_j²/η_j  với Ση_j = t  ⇔  min ½RSS + (Σ|β_j|)²/t
tH = 0.05
bH = b_ls_L[1:].copy()
for lap in range(20000):
    act = np.abs(bH) > 1e-13
    eta = np.zeros(8)
    eta[act] = tH * np.abs(bH[act]) / np.abs(bH[act]).sum()
    A = XsL[:, act]
    moi = np.zeros(8)
    moi[act] = np.linalg.solve(A.T @ A + 2 * np.diag(1 / eta[act]), A.T @ ycL)
    if np.abs(moi - bH).max() < 1e-15:
        bH = moi
        break
    bH = moi
lamH = 2 * np.abs(bH).sum() / tH
bL = lasso_cd(XsL, ycL, lamH)
print("   prostate (bản lỗi), t = %.2f: ridge lặp lại hội tụ sau %d vòng ; λ tương đương = 2Σ|β|/t = %.6f" % (tH, lap + 1, lamH))
print("   hệ số theo Holmes: %s" % np.array2string(bH, precision=6))
print("   lasso cùng λ     : %s" % np.array2string(bL, precision=6))
print("   lệch lớn nhất %.3e" % np.abs(bH - bL).max())

# ═══════════════════════════════════════════════════════════════════════════ PHẦN III — [E] ESL
import pypdfium2 as pdfium          # noqa: E402
import pypdfium2.raw as R           # noqa: E402

_doc_esl = pdfium.PdfDocument(ESL)


def trang(pn):
    """(đường: [(màu nét, [(x,y)…])], chấm chữ: [(màu tô, tâm x, tâm y)], chữ: (chuỗi, textpage))."""
    pg = _doc_esl[pn - 1]
    duong, cham = [], []

    def duyet(objs):
        for o in objs:
            if o.type == R.FPDF_PAGEOBJ_FORM:
                duyet(o.get_objects())
                continue
            if o.type == R.FPDF_PAGEOBJ_PATH:
                st = [ctypes.c_uint() for _ in range(4)]
                R.FPDFPageObj_GetStrokeColor(o.raw, *st)
                m = R.FS_MATRIX()
                R.FPDFPageObj_GetMatrix(o.raw, ctypes.byref(m))
                P = []
                for i in range(R.FPDFPath_CountSegments(o.raw)):
                    sg = R.FPDFPath_GetPathSegment(o.raw, i)
                    x, y = ctypes.c_float(), ctypes.c_float()
                    R.FPDFPathSegment_GetPoint(sg, x, y)
                    P.append((m.a * x.value + m.c * y.value + m.e, m.b * x.value + m.d * y.value + m.f))
                duong.append((tuple(c.value for c in st[:3]), P))
            elif o.type == R.FPDF_PAGEOBJ_TEXT:
                f = [ctypes.c_uint() for _ in range(4)]
                R.FPDFPageObj_GetFillColor(o.raw, *f)
                b = [ctypes.c_float() for _ in range(4)]
                R.FPDFPageObj_GetBounds(o.raw, *b)
                cham.append((tuple(c.value for c in f[:3]), (b[0].value + b[2].value) / 2, (b[1].value + b[3].value) / 2))
    duyet(pg.get_objects())
    tp = pg.get_textpage()
    return duong, cham, tp


head("14. [E] Bảng 3.3 tr.63, cột Lasso — dựng lại tới từng chữ số")
IN_E_LASSO = np.array([2.468, 0.533, 0.169, 0, 0.002, 0.094, 0, 0, 0])
X97 = (XP - XP.mean(0)) / XP.std(0, ddof=1)
A97 = X97[TR]
mA, sA = A97.mean(0), A97.std(0)
yTc = YP[TR] - YP[TR].mean()
dZ = Duong((A97 - mA) / sA, yTc)          # chuẩn hoá LẠI trên 67 mẫu huấn luyện
BZ = dZ.co / sA[:, None]                   # trả hệ số về thang của đầu vào (chuẩn hoá trên 97)
ttZ = np.abs(BZ).sum(0)


def tai_BZ(sv):
    t = sv * ttZ[-1]
    k = max(1, int(np.searchsorted(ttZ, t)))
    w = (t - ttZ[k - 1]) / (ttZ[k] - ttZ[k - 1])
    b = BZ[:, k - 1] + w * (BZ[:, k] - BZ[:, k - 1])
    return np.r_[YP[TR].mean() - mA @ b, b]


luoi_e = np.arange(0.30, 0.45, 1e-5)
khop_e = np.array([(np.abs(np.round(tai_BZ(sv), 3) - IN_E_LASSO) < 5e-4).sum() for sv in luoi_e])
cua_so = luoi_e[khop_e == 9]
print("   Giả thuyết: đầu vào chuẩn hoá trên 97 mẫu (như cột LS), bộ giải chuẩn hoá LẠI trên 67 mẫu huấn")
print("   luyện rồi trả hệ số về thang đầu vào; s tính trên thang đầu vào.")
print("   số ô khớp lớn nhất = %d/9 ; các s cho 9/9: [%.5f; %.5f]" % (khop_e.max(), cua_so.min(), cua_so.max()))
b_e = tai_BZ(cua_so.mean())
pr_e = b_e[0] + X97[~TR] @ b_e[1:]
e2e = (YP[~TR] - pr_e) ** 2
print("   tại s = %.5f: %s" % (cua_so.mean(), np.array2string(b_e, precision=4)))
print("   sai số kiểm tra %.6f (in 0.479) ; sai số chuẩn %.6f (in 0.164)" % (e2e.mean(), e2e.std(ddof=1) / math.sqrt(30)))
# cùng điểm đó đo trên thang chuẩn hoá 67 mẫu (ddof = 1, như Hình 3.10)
X67 = (XP - XP[TR].mean(0)) / XP[TR].std(0, ddof=1)
d67 = Duong(X67[TR], yTc)
b67_tai = b_e[1:] * XP[TR].std(0, ddof=1) / XP.std(0, ddof=1)
print("   cùng nghiệm quy về thang 67 mẫu: Σ|β| / t0 = %.5f (t0 = %.6f)" % (np.abs(b67_tai).sum() / d67.t0, d67.t0))
print("   các nút của đường theo s thang đầu vào: %s" % np.array2string(ttZ / ttZ[-1], precision=4))
print("   các nút của đường theo s thang 67 mẫu : %s" % np.array2string(d67.tt / d67.t0, precision=4))
for ten_q, XX, dd_ in (("chuẩn hoá trên 97 mẫu, giải thẳng", X97, 1), ("chuẩn hoá trên 67 mẫu, giải thẳng", X67, 1)):
    dq_ = Duong(XX[TR] - XX[TR].mean(0), yTc)
    tot = max((np.abs(np.round(np.r_[YP[TR].mean() - XX[TR].mean(0) @ dq_.tai(sv * dq_.t0), dq_.tai(sv * dq_.t0)], 3)
                      - IN_E_LASSO) < 5e-4).sum() for sv in np.arange(0.30, 0.45, 1e-4))
    thu_tu = []
    for k in range(1, dq_.co.shape[1]):
        for j in np.nonzero((np.abs(dq_.co[:, k]) > 1e-12) & (np.abs(dq_.co[:, k - 1]) <= 1e-12))[0]:
            thu_tu.append(TEN[j])
    print("   %-36s tốt nhất %d/9 ô ; thứ tự vào: %s" % (ten_q, tot, thu_tu))

head("15. [E] Hình 3.10 tr.70 — đọc toạ độ vectơ: nút, vạch đỏ, đầu mút")
dg10, _, tp10 = trang(89)
xtick = sorted({round(P[0][0], 2) for c, P in dg10 if c == (0, 0, 0) and len(P) == 2
                and abs(P[0][0] - P[1][0]) < 1e-6 and abs(P[0][1] - P[1][1]) < 6 and P[0][1] > 300})
ytick = sorted({round(P[0][1], 2) for c, P in dg10 if c == (0, 0, 0) and len(P) == 2
                and abs(P[0][1] - P[1][1]) < 1e-6 and abs(P[0][0] - P[1][0]) < 6})
print("   vạch chia trục ngang: %s (6 nhãn 0.0 … 1.0)" % xtick)
print("   vạch chia trục dọc : %s (5 nhãn −0.2 … 0.6)" % ytick)
fx = lambda x: (x - xtick[0]) / (xtick[-1] - xtick[0])
fy = lambda y: -0.2 + 0.8 * (y - ytick[0]) / (ytick[-1] - ytick[0])
xam = sorted(fx(P[0][0]) for c, P in dg10 if c == (211, 211, 211))
do = [fx(P[0][0]) for c, P in dg10 if c == (255, 0, 0)]
print("   vạch xám đứng (nút của đường) ở s = %s" % np.array2string(np.array(xam), precision=4))
print("   nút đường lasso chuẩn hoá 67 mẫu     : %s" % np.array2string(d67.tt / d67.t0, precision=4))
print("   lệch lớn nhất giữa hai dãy nút: %.2e" % np.abs(np.array(xam) - d67.tt / d67.t0).max())
print("   vạch ĐỎ đứng ở s = %.4f ; chú thích và tr.69 viết s = 0,36" % do[0])
xanh = [np.array([(fx(x), fy(y)) for x, y in P]) for c, P in dg10 if c == (0, 0, 255) and len(P) < 13]
cuoi = np.array([L_[-1, 1] for L_ in xanh])
ls67 = d67.co[:, -1]
gan = [int(np.argmin(np.abs(ls67 - v))) for v in cuoi]
print("   %d đường xanh ; đầu mút s = 1: %s" % (len(xanh), np.array2string(cuoi, precision=4)))
print("   LS chuẩn hoá 67 mẫu (ddof 1): %s" % np.array2string(ls67, precision=4))
print("   gán đường theo đầu mút: %s ; lệch đầu mút lớn nhất %.4f"
      % ([TEN[j] for j in gan], max(abs(ls67[j] - v) for j, v in zip(gan, cuoi))))
lech_duong = 0.0
for L_, j in zip(xanh, gan):
    for sv, v in L_:
        lech_duong = max(lech_duong, abs(d67.tai(sv * d67.t0)[j] - v))
print("   mọi điểm gãy của 8 đường so với đường dựng lại: lệch lớn nhất %.4f (thang trục dọc)" % lech_duong)
print("   |β̂_j(s)| không giảm theo s trên cả 8 đường in: %s"
      % all(np.all(np.diff(np.abs(L_[:, 1])) >= -1e-9) for L_ in xanh))
for sv in (0.36, do[0]):
    bb_ = d67.tai(sv * d67.t0)
    print("   tại s = %.4f: khác 0 = %s ; bằng 0: %d biến" % (sv, [TEN[j] for j in np.nonzero(np.abs(bb_) > 1e-12)[0]],
                                                          (np.abs(bb_) <= 1e-12).sum()))
print("   [hinh310] nut = %s" % np.array2string(np.array(xam), precision=6, separator=","))
for L_, j in zip(xanh, gan):
    print("   [hinh310] %s = %s" % (TEN[j], np.array2string(L_, precision=6, separator=",").replace("\n", "")))
print("   [hinh310] do = %.6f" % do[0])
print("   [hinh310] duong 67: s = %s" % np.array2string(d67.tt / d67.t0, precision=10, separator=","))
print("   [hinh310] duong 67: he so =\n%s" % np.array2string(d67.co, precision=10, separator=","))
# t = t0/2
b_nua = d67.tai(0.5 * d67.t0)
ti = np.abs(b_nua) / np.abs(ls67)
print("   tr.69 't = t0/2 … co khoảng 50%% trung bình': |β(t0/2)|/|β_LS| = %s" % np.array2string(ti, precision=4))
print("      trung bình %.4f ; trung vị %.4f ; số biến về 0: %d ; Σ|β|/t0 = %.4f"
      % (ti.mean(), np.median(ti), (ti == 0).sum(), np.abs(b_nua).sum() / d67.t0))

head("16. [E] Hình 3.7 tr.62, ô Lasso — đường CV, vạch tím")
dg7, ch7, tp7 = trang(81)
khung = [P for c, P in dg7 if c == (0, 0, 0) and len(P) == 5]
KH = [k_ for k_ in khung if abs(k_[0][0] - 158.36) < 0.1 and abs(k_[0][1] - 416.0) < 0.1][0]
x0k, y0k, x1k, y1k = KH[0][0], KH[0][1], KH[2][0], KH[2][1]
# vạch chia của mỗi trục là MỘT đường nhiều đoạn: các cặp điểm (trên trục, lệch ra ngoài 1,9 pt)
tx = sorted({round(x, 2) for c, P in dg7 if c == (0, 0, 0) and len(P) == 12 and abs(P[0][1] - y0k) < 0.1
             and x0k <= P[0][0] <= x1k for x, y in P if abs(y - y0k) < 0.1})
ty = sorted({round(y, 2) for c, P in dg7 if c == (0, 0, 0) and len(P) == 14 and abs(P[0][0] - x0k) < 0.1
             and y0k <= P[0][1] <= y1k for x, y in P if abs(x - x0k) < 0.1})
print("   khung ô Lasso: x %.2f–%.2f, y %.2f–%.2f ; %d vạch trục ngang, %d vạch trục dọc" % (x0k, x1k, y0k, y1k, len(tx), len(ty)))
gx7 = lambda x: (x - tx[0]) / (tx[-1] - tx[0])
gy7 = lambda y: 0.6 + 1.2 * (y - ty[0]) / (ty[-1] - ty[0])
cam = sorted((gx7(x), gy7(y)) for c, x, y in ch7 if c == (230, 159, 0) and x0k < x < x1k and y0k < y < y1k)
cam = np.array(cam)
print("   %d chấm cam: s = %s" % (len(cam), np.array2string(cam[:, 0], precision=4)))
print("                CV = %s" % np.array2string(cam[:, 1], precision=4))
tim = [P for c, P in dg7 if c == (160, 32, 240) and x0k - 0.1 <= P[0][0] <= x1k + 0.1 and y0k - 0.1 <= P[0][1] <= y1k + 0.1]
for P in tim:
    if abs(P[0][0] - P[1][0]) < 1e-6:
        print("   vạch tím ĐỨNG ở s = %.4f" % gx7(P[0][0]))
        s_tim = gx7(P[0][0])
    else:
        print("   vạch tím NGANG ở CV = %.4f" % gy7(P[0][1]))
        m_tim = gy7(P[0][1])
jm = int(np.argmin(cam[:, 1]))
print("   cực tiểu của đường in: s = %.4f, CV = %.4f ; vạch ngang − cực tiểu = %.4f" % (cam[jm, 0], cam[jm, 1], m_tim - cam[jm, 1]))
duoi = cam[cam[:, 1] <= m_tim + 1e-9]
print("   các chấm dưới vạch ngang: s = %s ; nhỏ nhất %.4f" % (np.array2string(duoi[:, 0], precision=4), duoi[:, 0].min()))

head("17. [E] tr.71–73 — Bảng 3.4, (3.53), Hình 3.12–3.13")
print("   Bảng 3.4 — ba công thức trực chuẩn: đã kiểm ở mục 1 (lasso, ridge) ; tập con: β̂_j·I(|β̂_j| ≥ |β̂_(M)|)")
bm = np.array([3.0, -0.4, 1.7, -2.2, 0.1])
M_ = 2
ng = np.sort(np.abs(bm))[::-1][M_ - 1]
print("   ví dụ tập con M = 2 trên β̂ = %s: %s" % (bm, bm * (np.abs(bm) >= ng)))
th = np.linspace(0, math.pi / 2, 20001)
c_, s_ = np.cos(th), np.sin(th)
r_q = (c_ ** 1.2 + s_ ** 1.2) ** (-1 / 1.2)
a_ = 0.2
A2 = a_ * (c_ ** 2 + s_ ** 2)
B1 = (1 - a_) * (c_ + s_)
r_en = (-B1 + np.sqrt(B1 ** 2 + 4 * A2)) / (2 * A2)
print("   Hình 3.13 — bán kính đường mức 1 theo góc: L_1.2 và elastic net α = 0,2 (αβ² + (1−α)|β|)")
print("      lệch lớn nhất %.5f (tương đối %.4f) tại góc %.2f° ; bán kính trên trục: %.4f và %.4f"
      % (np.abs(r_q - r_en).max(), (np.abs(r_q - r_en) / r_q).max(), np.degrees(th[int(np.argmax(np.abs(r_q - r_en)))]),
         r_q[0], r_en[0]))
dq12 = lambda t_: -((np.cos(t_) ** 1.2 + np.sin(t_) ** 1.2) ** (-1 / 1.2))
print("   độ dốc đường mức tại trục (dβ₂/dβ₁ khi β₂ → 0⁺): L_1.2 = 0 (khả vi) ; elastic net = −(1−α)/(2α·r + 1−α) = %.4f"
      % (-(1 - a_) / (2 * a_ * r_en[0] + (1 - a_))))

# ═══════════════════════════════════════════════════════════════════════════ PHẦN IV — [L] 2004
head("18. [L] Bảng 1, Hình 1, §3.1 — bộ diabetes")
dr = load_diabetes(scaled=False)
Xr, yr = dr.data, dr.target
IN_B1 = {1: [59, 2, 32.1, 101, 157, 93.2, 38, 4, 4.9, 87, 151], 2: [48, 1, 21.6, 87, 183, 103.2, 70, 3, 3.9, 69, 75],
         3: [72, 2, 30.5, 93, 156, 93.6, 41, 4, 4.7, 85, 141], 4: [24, 1, 25.3, 84, 198, 131.4, 40, 5, 4.9, 89, 206],
         5: [50, 1, 23.0, 101, 192, 125.4, 52, 4, 4.3, 80, 135], 6: [23, 1, 22.6, 89, 139, 64.8, 61, 2, 4.2, 68, 97],
         441: [36, 1, 30.0, 95, 201, 125.2, 42, 5, 5.1, 85, 220], 442: [36, 1, 19.6, 71, 250, 133.2, 97, 3, 4.6, 92, 57]}
khop_b1, tong_b1 = 0, 0
for i_, dong in IN_B1.items():
    v = list(Xr[i_ - 1]) + [yr[i_ - 1]]
    for a_in, a_th in zip(dong, v):
        so_le = len(str(a_in).split(".")[1]) if "." in str(a_in) else 0
        khop_b1 += abs(round(a_th, so_le) - a_in) < 1e-9
        tong_b1 += 1
print("   Bảng 1 (8 dòng in) so với load_diabetes(scaled=False): khớp %d/%d ô sau khi làm tròn như bản in" % (khop_b1, tong_b1))
print("   dòng 1 trong scikit-learn: %s" % np.array2string(np.r_[Xr[0], yr[0]], precision=4))
Xd = Xr - Xr.mean(0)
Xd = Xd / np.sqrt((Xd ** 2).sum(0))
yd = yr - yr.mean()
_, _, co_la = lars_path(Xd, yd, method="lasso")
_, _, co_lar = lars_path(Xd, yd, method="lar")
print("   (1.1): Σx = 0 lệch %.1e ; Σx² = 1 lệch %.1e ; Σy = %.1e"
      % (np.abs(Xd.sum(0)).max(), np.abs((Xd ** 2).sum(0) - 1).max(), abs(yd.sum())))
print("   số bước: lasso %d, LARS %d (bài báo: 12 và 10)" % (co_la.shape[1] - 1, co_lar.shape[1] - 1))
vao_ra = []
for k in range(1, co_la.shape[1]):
    moi_vao = np.nonzero((co_la[:, k] != 0) & (co_la[:, k - 1] == 0))[0]
    moi_ra = np.nonzero((co_la[:, k] == 0) & (co_la[:, k - 1] != 0))[0]
    vao_ra.append(("+" + ",".join(str(j + 1) for j in moi_vao) if len(moi_vao) else "")
                  + ("−" + ",".join(str(j + 1) for j in moi_ra) if len(moi_ra) else ""))
print("   thay đổi tập khác 0 ở mỗi bước: %s" % vao_ra)
tt_d = np.abs(co_la).sum(0)
print("   t tại bước cuối = %.4f (bài báo 3460.00) ; Σ|β̂_OLS| = %.4f"
      % (tt_d[-1], np.abs(np.linalg.lstsq(Xd, yd, rcond=None)[0]).sum()))
b_raw = np.linalg.lstsq(np.c_[np.ones(len(yr)), Xr], yr, rcond=None)[0][1:]
print("   Σ|β̂_OLS| trên biến THEO ĐƠN VỊ GỐC (Bảng 1) = %.4f ; trên biến chuẩn hoá độ dài 1 theo (1.1) = %.4f"
      % (np.abs(b_raw).sum(), np.abs(np.linalg.lstsq(Xd, yd, rcond=None)[0]).sum()))
k1 = int(np.searchsorted(tt_d, 1000))
w1 = (1000 - tt_d[k1 - 1]) / (tt_d[k1] - tt_d[k1 - 1])
c1000 = co_la[:, k1 - 1] + w1 * (co_la[:, k1] - co_la[:, k1 - 1])
print("   tại t = 1000 khác 0: %s (bài báo: 3, 9, 4, 7)" % sorted(int(j) + 1 for j in np.nonzero(np.abs(c1000) > 1e-9)[0]))
# mô hình bậc hai của §4
for ten_q, Zq in (("tích của biến GỐC", Xr), ("tích của biến ĐÃ TRỪ TRUNG BÌNH", Xr - Xr.mean(0))):
    cols = [Zq[:, j] for j in range(10)]
    cols += [Zq[:, j] ** 2 for j in range(10) if j != 1]          # bỏ bình phương của sex (nhị phân)
    cols += [Zq[:, i] * Zq[:, j] for i in range(10) for j in range(i + 1, 10)]
    XQ = np.array(cols).T
    XQ = XQ - XQ.mean(0)
    XQ = XQ / np.sqrt((XQ ** 2).sum(0))
    _, act_la, cq_la = lars_path(XQ, yd, method="lasso")
    _, act_lar, cq_lar = lars_path(XQ, yd, method="lar")
    nzl = (cq_la != 0).sum(0)
    print("   mô hình bậc hai (3.15), %s: %d cột, hạng %d ; số nút lasso %d (lần rời %d), LAR %d ; bài báo: 103 và 64"
          % (ten_q, XQ.shape[1], np.linalg.matrix_rank(XQ), cq_la.shape[1] - 1, int((np.diff(nzl) < 0).sum()),
             cq_lar.shape[1] - 1))
print("   -> LAR về lý thuyết cần đúng m = 64 bước khi n > m; scikit-learn trả nhiều nút hơn, nên số nút của nó")
print("      không phải 'số bước' theo nghĩa của bài báo — phép so này KHÔNG kết luận được")
print("   [hinh1] t = %s" % np.array2string(tt_d, precision=6, separator=","))
print("   [hinh1] he so =\n%s" % np.array2string(co_la, precision=6, separator=","))

# ═══════════════════════════════════════════════════════════════════════════ PHẦN V — [G] 2010
head("19. [G] (1)–(10), §2.5 — hạ theo toạ độ của glmnet")
Xg = (XP_LOI - XP_LOI.mean(0)) / XP_LOI.std(0)       # Σx = 0, Σx²/N = 1 đúng như [G] §2
yg = YP - YP.mean()
Ng = len(yg)


def en_cd(X, y, lam, al, b=None, cach="naive", tol=1e-13, max_sweep=100000):
    """(5) với cập nhật naive (7)–(8) hoặc cập nhật hiệp phương sai (9). Trả (β, số lượt quét, số lần cập nhật)."""
    N, p = X.shape
    b = np.zeros(p) if b is None else b.copy()
    r = y - X @ b
    if cach == "cov":
        XtX = X.T @ X
        Xty = X.T @ y
    dem = 0
    for sweep in range(1, max_sweep + 1):
        lon = 0.0
        for j in range(p):
            if cach == "naive":
                z = X[:, j] @ r / N + b[j]                     # (8)
            else:
                z = (Xty[j] - XtX[j] @ b) / N + b[j]           # (9)
            moi = soft(z, lam * al) / (1 + lam * (1 - al))     # (5)
            dem += 1
            if moi != b[j]:
                if cach == "naive":
                    r -= X[:, j] * (moi - b[j])
                lon = max(lon, abs(moi - b[j]))
                b[j] = moi
        if lon < tol:
            return b, sweep, dem
    raise RuntimeError("CD chưa hội tụ")


lmax = np.abs(Xg.T @ yg).max() / Ng
print("   λ_max = max|⟨x_j, y⟩|/(Nα), α = 1: %.6f" % lmax)
print("   tại λ_max: số hệ số khác 0 = %d ; tại 0,999·λ_max: %d"
      % ((en_cd(Xg, yg, lmax, 1.0)[0] != 0).sum(), (en_cd(Xg, yg, 0.999 * lmax, 1.0)[0] != 0).sum()))
alp, _, co_g = lars_path(Xg, yg, method="lasso")      # alphas của scikit-learn = max|X'r|/N
for lv in (0.5 * lmax, 0.1 * lmax, 0.01 * lmax):
    bn, sw_n, _ = en_cd(Xg, yg, lv, 1.0, cach="naive")
    bc, sw_c, _ = en_cd(Xg, yg, lv, 1.0, cach="cov")
    k = int(np.searchsorted(-alp, -lv))
    w = (lv - alp[k - 1]) / (alp[k] - alp[k - 1])
    bl_ = co_g[:, k - 1] + w * (co_g[:, k] - co_g[:, k - 1])
    kkt = np.abs(Xg.T @ (yg - Xg @ bn) / Ng)
    print("   λ = %.4f·λ_max: naive vs hiệp phương sai lệch %.1e (%d = %d lượt) ; vs LARS lệch %.1e ; max|x'r|/N = %.6f"
          % (lv / lmax, np.abs(bn - bc).max(), sw_n, sw_c, np.abs(bn - bl_).max(), kkt.max()))
# đường 100 giá trị λ, khởi động ấm
Lg = lmax * np.logspace(0, -3, 100)
b_, tong_am = np.zeros(8), 0
for lv in Lg:
    b_, _, dm = en_cd(Xg, yg, lv, 1.0, b=b_)
    tong_am += dm
_, _, lanh_cuoi = en_cd(Xg, yg, Lg[-1], 1.0)
tong_lanh = sum(en_cd(Xg, yg, lv, 1.0)[2] for lv in Lg)
print("   đường 100 λ từ λ_max tới 0,001·λ_max (prostate, p = 8): cập nhật khởi động ấm %d ; lạnh %d ; riêng λ nhỏ nhất lạnh %d"
      % (tong_am, tong_lanh, lanh_cuoi))
rw = np.random.default_rng(31)
Xw = he_so_mau(200, 0.9, 60, rw)
yw = Xw[:, :10] @ rw.normal(size=10) + rw.normal(size=60)
yw -= yw.mean()
lw = np.abs(Xw.T @ yw).max() / 60
Lw = lw * np.logspace(0, -2, 100)
b_, tong_am = np.zeros(200), 0
for lv in Lw:
    b_, _, dm = en_cd(Xw, yw, lv, 1.0, b=b_, tol=1e-10)
    tong_am += dm
_, _, lanh_cuoi = en_cd(Xw, yw, Lw[-1], 1.0, tol=1e-10)
print("   n = 60, p = 200, tương quan 0,9: cập nhật cả đường ấm %d ; riêng λ nhỏ nhất lạnh %d -> %s"
      % (tong_am, lanh_cuoi, "ĐƯỜNG rẻ hơn" if tong_am < lanh_cuoi else "một điểm rẻ hơn"))
# ridge với k cột trùng nhau
x1 = Xg[:, :1]
for k in (2, 3, 5):
    Xk_ = np.repeat(x1, k, axis=1)
    for lv in (0.01, 0.1, 1.0):
        bk_, _, _ = en_cd(Xk_, yg, lv, 0.0)
        b1_, _, _ = en_cd(x1, yg, lv, 0.0)
        print("   ridge (α = 0), %d cột trùng, λ = %-5g: mỗi hệ số / hệ số khi đứng một mình = %.4f ; 1/k = %.4f ; (1+λ)/(k+λ) = %.4f"
              % (k, lv, bk_[0] / b1_[0], 1 / k, (1 + lv) / (k + lv)))
# độ thưa theo α ở λ cố định
for ten_d, X_, y_ in (("prostate", Xg, yg), ("n = 60, p = 200", Xw, yw)):
    lmx = np.abs(X_.T @ y_).max() / len(y_)
    for lv in (0.1 * lmx, 0.3 * lmx):
        so0 = []
        for al in np.linspace(0, 1, 21):
            so0.append(int((en_cd(X_, y_, lv, al, tol=1e-10)[0] == 0).sum()))
        print("   %s, λ = %.1f·λ_max(α=1): số hệ số 0 theo α = 0, 0,05 … 1: %s ; không giảm: %s"
              % (ten_d, lv / lmx, so0, all(a <= b for a, b in zip(so0, so0[1:]))))

# ═══════════════════════════════════════════════════════════════════════════ PHẦN VI — mảng cho hình
head("20. Mảng cho các hình tự vẽ (in đủ chữ số)")
print("   [hinh4] t = %s" % np.array2string(np.linspace(0, 9, 91), precision=4, separator=","))
print("   [hinh4] lasso (β1, β2) =\n%s" % np.array2string(goc, precision=10, separator=","))
for rho in (0.0, 0.23, 0.45, 0.68, 0.90):
    r5 = np.random.default_rng(4)
    X = he_so_mau(2, rho, 100, r5) if rho > 0 else he_so_mau(2, 1e-12, 100, r5)
    y = X @ np.array([6.0, 3.0])
    y -= y.mean()
    lams = np.r_[0, np.logspace(-2, 4, 40)]
    B = np.array([np.linalg.solve(X.T @ X + l * np.eye(2), X.T @ y) for l in lams])
    print("   [hinh4] ridge rho=%.2f =\n%s" % (rho, np.array2string(B, precision=8, separator=",")))
print("   [mophong] trung vị ME (ví dụ, phương pháp, dựng lại, in):")
for vd, ghi in KET_QUA_MP.items():
    for k in IN_BANG[vd]:
        if k in ghi:
            print("   [mophong] %d | %s | %.6f | %.2f" % (vd, k, np.median([v[0] for v in ghi[k]]), IN_BANG[vd][k][0]))
for k in IN_BANG[1]:
    print("   [mophong] 1s2 | %s | %.6f | %.2f" % (k, np.median([v[0] for v in ghi2[k]]), IN_BANG[1][k][0]))

head("21. Ví dụ nhận–trả ở mục 0.1 — bộ diabetes của scikit-learn (cột đã chuẩn hoá độ dài 1)")
from sklearn.linear_model import Lasso, Ridge          # noqa: E402

ds = load_diabetes()
Xs_d, ys_d = ds.data, ds.target
print("   X %s, y %s ; chuẩn 2 của mỗi cột: %s" % (Xs_d.shape, ys_d.shape, np.array2string(np.sqrt(((Xs_d - Xs_d.mean(0)) ** 2).sum(0)), precision=6)))
for a in (0.05, 1.0, 2.0):
    m = Lasso(alpha=a, max_iter=100000).fit(Xs_d, ys_d)
    print("   alpha = %.2f (λ = n·alpha = %.1f): β̂ làm tròn = %s ; %d biến khác 0 ; hệ số chặn %.4f"
          % (a, len(ys_d) * a, np.round(m.coef_).astype(int), (m.coef_ != 0).sum(), m.intercept_))
m = Lasso(alpha=1.0, max_iter=100000).fit(Xs_d, ys_d)
print("   alpha = 1: giá đỡ %s ; hệ số chưa làm tròn %s" % ([ds.feature_names[j] for j in np.nonzero(m.coef_)[0]],
                                                          np.array2string(m.coef_[m.coef_ != 0], precision=4)))
print("   dự đoán: shape %s ; 3 giá trị đầu %s" % (m.predict(Xs_d).shape, np.array2string(m.predict(Xs_d)[:3], precision=4)))
print("   [0.5] ngưỡng mềm của [0.5, 1.0, 3.0, −2.0] với λ = 1: %s" % soft(np.array([0.5, 1.0, 3.0, -2.0]), 1.0))
print("   [0.5] ridge z/(1+λ) cùng đầu vào: %s" % (np.array([0.5, 1.0, 3.0, -2.0]) / 2))
print("   [0.5] hai biến, β̂° = (6, 3): t = 5 -> (%.1f, %.1f) ; t = 2 -> (%.1f, %.1f)"
      % tuple(np.r_[np.maximum(np.array([6.0, 3.0]) - (9 - 5) / 2, 0), 2.0, 0.0]))
print("\nXONG")
