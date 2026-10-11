# -*- coding: utf-8 -*-
"""Sinh hình cho bài "Polynomial Regression".

KHÔNG sửa file này bằng tay: nó được sinh ra từ chính khối code in trong bài, bằng
    python extract_fig.py "Polynomial Regression" fig_poly.py
Muốn đổi hình thì sửa khối code TRONG BÀI rồi chạy lại lệnh trên và regen.py.
"""
import os

os.chdir(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "AI algorithms", "Machine Learning", "Polynomial Regression"))

import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
sys.stdout.reconfigure(encoding="utf-8")

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})

# Bishop 2006, Bảng 1.1 tr.8, cột M = 9 (in 2 chữ số thập phân)
W9 = np.array([0.35, 232.37, -5321.83, 48568.31, -231639.30, 640042.26,
               -1061800.52, 1042400.18, -557682.99, 125201.43])
W3_IN = np.array([0.31, 7.99, -25.43, 17.37])          # cột in nhãn "M = 6"
R18_IN = np.array([0.35, 4.74, -0.77, -31.97, -3.89, 55.28, 41.32, -45.95, -91.53, 72.68])
R0_IN = np.array([0.13, -0.05, -0.06, -0.05, -0.03, -0.02, -0.01, -0.00, 0.00, 0.01])
# Hình 1.5 tr.8 số hoá bằng điểm ảnh (tools/v_poly.py mục 9)
HINH15_HUAN = np.array([0.4363, 0.3281, 0.3253, 0.1342, 0.1342, 0.1329, 0.1301, 0.1295, 0.1212, 0.0034])

x = np.linspace(0, 1, 10)
V9 = np.vander(x, 10, increasing=True)
t = V9 @ W9                                            # E(w*) = 0 nên đa thức M = 9 đi qua đúng 10 điểm


def khop(M, lam=0.0):
    V = np.vander(x, M + 1, increasing=True)
    A = np.vstack([V, np.sqrt(lam) * np.eye(M + 1)])
    return np.linalg.lstsq(A, np.r_[t, np.zeros(M + 1)], rcond=None)[0]


g = np.linspace(0, 1, 400)
fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.8))

# --- trái: 10 điểm dựng lại, M = 3 dựng lại chồng lên cột in "M = 6" ---
w3 = khop(3)
ax[0].plot(g, np.sin(2 * np.pi * g), color="#16a34a", lw=1.2, label="sin(2πx)")
ax[0].plot(g, np.vander(g, 4, increasing=True) @ W3_IN, color="#dc2626", lw=2.4, alpha=.45,
           label='cột in "M = 6"')
ax[0].plot(g, np.vander(g, 4, increasing=True) @ w3, "--", color="#111827", lw=1.1, label="M = 3 dựng lại")
ax[0].scatter(x, t, s=28, facecolors="none", edgecolors="#2563eb", lw=1.4, zorder=4, label="10 điểm dựng lại")
ax[0].set_ylim(-1.5, 1.5); ax[0].set_xlabel("x"); ax[0].set_ylabel("t"); ax[0].legend(fontsize=7)
ax[0].set_title('Từ cột M = 9 suy ra 10 điểm;\ncột in "M = 6" trùng M = 3 dựng lại', fontsize=9)

# --- giữa: Hình 1.5 số hoá so với RMSE và RMSE/√2 ---
rm = np.array([np.sqrt(((np.vander(x, M + 1, increasing=True) @ khop(M) - t) ** 2).mean()) for M in range(10)])
Ms = np.arange(10)
ax[1].plot(Ms, rm, "s-", color="#9ca3af", ms=4, lw=1.1, label="RMSE = √(2E/N), công thức (1.3)")
ax[1].plot(Ms, rm / np.sqrt(2), "o-", color="#2563eb", ms=4, lw=1.1, label="√(E/N) = RMSE/√2")
ax[1].plot(Ms, HINH15_HUAN, "x", color="#dc2626", ms=8, mew=1.8, label="Hình 1.5, đường huấn luyện (số hoá)")
ax[1].set_xlabel("M"); ax[1].set_ylabel("sai số huấn luyện"); ax[1].legend(fontsize=7)
ti = HINH15_HUAN[:9] / rm[:9]
ax[1].set_title("Hình 1.5 vẽ √(E/N), không phải (1.3)\ntỉ số hình / RMSE trung bình %.4f" % ti.mean(), fontsize=9)

# --- phải: quét λ cho hai cột của Bảng 1.2 ---
LL = np.arange(-30, 10.0001, 0.02)
for inb, l_in, mau, ten in ((R18_IN, -18, "#7c3aed", "cột ln λ = −18"), (R0_IN, 0, "#d97706", "cột ln λ = 0")):
    lech = np.array([np.abs(khop(9, np.exp(l)) - inb).max() for l in LL])
    i = int(lech.argmin())
    ax[2].semilogy(LL, lech, color=mau, lw=1.2, label="%s — tốt nhất ln λ = %.2f" % (ten, LL[i]))
    ax[2].axvline(l_in, color=mau, ls=":", lw=1)
    print("%s: ln λ tốt nhất %.2f, lệch max %.4f, cách nhãn in %+.2f" % (ten, LL[i], lech[i], LL[i] - l_in))
ax[2].axhline(0.005, color="#111827", ls="--", lw=.8)
ax[2].text(-29.5, 0.0065, "±0,005 (mức làm tròn)", fontsize=7)
ax[2].set_xlabel("ln λ đưa vào (1.4)"); ax[2].set_ylabel("max |hệ số dựng − hệ số in|"); ax[2].legend(fontsize=7)
ax[2].set_title("Bảng 1.2: nhãn in lệch khỏi λ của (1.4)\nvạch chấm = nhãn in, đáy = λ khớp nhất", fontsize=9)

fig.savefig("poly_bishop_dung_lai.png", dpi=130, bbox_inches="tight", facecolor="white")
print("M = 3 dựng lại:", " ".join("%.6f" % v for v in w3))
print("tỉ số Hình 1.5 / RMSE, M = 0..8:", " ".join("%.4f" % v for v in ti), "· trung bình %.4f" % ti.mean())
print("RMSE dựng lại M = 0..9:", " ".join("%.6f" % v for v in rm))

# ----------------------------------------------------------------------

import sys
import numpy as np, matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding="utf-8")

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})


def rmse(a, b):
    return float(np.sqrt(((np.asarray(a) - np.asarray(b)) ** 2).mean()))


def f_true(t):
    return np.sin(1.6 * t) + 0.35 * t


rng = np.random.default_rng(0)
ntr = 30
xtr = np.sort(rng.uniform(-3, 3, ntr))
ytr = f_true(xtr) + rng.normal(0, .35, ntr)
xte = np.sort(rng.uniform(-3, 3, 400))          # tap kiem tra tu CUNG phan phoi
yte = f_true(xte) + rng.normal(0, .35, 400)
grid = np.linspace(-3.2, 3.2, 400)

fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.7))

# --- bang trai: bon bac ve chong len nhau ---
for d, col in [(1, "#6b7280"), (3, "#2563eb"), (9, "#d97706"), (24, "#dc2626")]:
    cf = np.polyfit(xtr, ytr, d)
    axes[0].plot(grid, np.polyval(cf, grid), color=col, lw=1.4, label="bậc %d" % d)
axes[0].scatter(xtr, ytr, s=20, color="#111827", zorder=4)
axes[0].plot(grid, f_true(grid), "--", color="#059669", lw=1.2, label="hàm thật")
axes[0].set_ylim(-3, 3.2); axes[0].legend(fontsize=7.2, ncol=2)
axes[0].set_xlabel("x"); axes[0].set_ylabel("y")
axes[0].set_title("30 điểm huấn luyện\nbậc 24 đi qua gần hết — và vô dụng", fontsize=9)

# --- bang giua: RMSE huan luyen vs kiem tra theo bac ---
ds = range(1, 26)
tr_e, te_e = [], []
for d in ds:
    cf = np.polyfit(xtr, ytr, d)
    tr_e.append(rmse(np.polyval(cf, xtr), ytr))
    te_e.append(rmse(np.polyval(cf, xte), yte))
axes[1].semilogy(list(ds), tr_e, "o-", color="#2563eb", ms=3, lw=1.2, label="huấn luyện")
axes[1].semilogy(list(ds), te_e, "o-", color="#dc2626", ms=3, lw=1.2, label="kiểm tra")
axes[1].axhline(.35, color="#059669", ls="--", lw=1.0, label="mức nhiễu σ = 0.35")
best = int(np.argmin(te_e)) + 1
axes[1].axvline(best, color="#111827", lw=.8)
axes[1].legend(fontsize=7.2); axes[1].set_xlabel("bậc đa thức"); axes[1].set_ylabel("RMSE")
axes[1].set_title("Bậc tốt nhất trên tập kiểm tra: %d\n"
                  "sai số huấn luyện giảm mãi, kiểm tra thì không" % best, fontsize=9)

# --- bang phai: so dieu kien theo bac, doc lap voi du lieu ---
# Ba cach tinh cung mot dai luong, de doi chieu (xem canh bao duoi hinh).
conds, conds_direct, conds_sv = [], [], []
for d in ds:
    V = np.vander(xtr, d + 1, increasing=True)
    G = V.T @ V
    conds.append(np.linalg.cond(G))                       # (a) qua bien tam G
    conds_direct.append(np.linalg.cond(np.vander(xtr, d + 1, increasing=True).T
                                       @ np.vander(xtr, d + 1, increasing=True)))  # (b)
    sv = np.linalg.svd(V, compute_uv=False)
    conds_sv.append((sv[0] / sv[-1]) ** 2)                # (c) qua kappa(V)^2
axes[2].semilogy(list(ds), conds, "o-", color="#7c3aed", ms=3, lw=1.2)
axes[2].axhline(1 / np.finfo(float).eps, color="#dc2626", ls="--", lw=1.1)
axes[2].text(2, 1 / np.finfo(float).eps * 1.6,
             "1/ε của float64 — quá đây thì\nnghiệm không còn chữ số nào đúng",
             fontsize=7, color="#dc2626")
axes[2].set_xlabel("bậc đa thức"); axes[2].set_ylabel(r"$\kappa(V^\top V)$")
axes[2].set_title("Ma trận Vandermonde hỏng rất nhanh\n"
                  "Đây mới là lý do không dùng bậc cao", fontsize=9)

fig.savefig("poly_degree.png", dpi=130, bbox_inches="tight", facecolor="white")

d_break = next(d for d, c in zip(ds, conds) if c > 1 / np.finfo(float).eps)
print("bậc tốt nhất trên kiểm tra: %d (RMSE %.4f)" % (best, min(te_e)))
print("RMSE bậc 1  : huấn luyện %.4f  kiểm tra %.4f" % (tr_e[0], te_e[0]))
print("RMSE bậc 24 : huấn luyện %.4f  kiểm tra %.4f" % (tr_e[23], te_e[23]))
print("tỉ số kiểm tra / huấn luyện ở bậc 24: %.1f" % (te_e[23] / tr_e[23]))
print("κ(VᵀV) bậc 1 = %.2e  bậc 12 = %.2e  bậc 24 = %.2e" % (conds[0], conds[11], conds[23]))
print("bậc đầu tiên κ vượt 1/ε: %d" % d_break)
print("\nBa cách tính CÙNG một κ(VᵀV) — chỉ khớp khi κ còn dưới 1/ε:")
for d in (12, 24):
    i = d - 1
    print("   bậc %2d: (a) %.3e   (b) %.3e   (c) κ(V)² = %.3e"
          % (d, conds[i], conds_direct[i], conds_sv[i]))
