# -*- coding: utf-8 -*-
"""Sinh hình cho bài "Ridge Regression (Tikhonov Regularization)".

KHÔNG sửa file này bằng tay: nó được sinh ra từ chính khối code in trong bài, bằng
    python extract_fig.py "Ridge Regression (Tikhonov Regularization)" fig_ridge.py
Muốn đổi hình thì sửa khối code TRONG BÀI rồi chạy lại lệnh trên và regen.py.
"""
import os

os.chdir(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "AI algorithms", "Machine Learning", "Ridge Regression (Tikhonov Regularization)"))

import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding="utf-8")
plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})

# v_ridge.py mục 4(b) và mục 6: sai số đều max|z − z̄| theo α, nhiễu δ = 1e-3
alpha_k1 = 10.0 ** -np.arange(2, 15)
sai_k1 = np.array([1.303721, 0.777038, 0.286015, 0.066032, 0.022421, 0.030404, 0.067798,
                   0.119298, 0.231674, 0.712750, 1.549022, 3.399091, 8.546764])
alpha_k0 = 10.0 ** -np.arange(2, 11)
sai_k0 = np.array([0.858877, 0.408916, 0.219624, 0.180816, 0.208411, 0.460691, 1.386116,
                   2.124955, 5.386052])

# v_ridge.py mục 17: số hoá Hình 1 của Hoerl & Kennard 1970
k_hk = np.array([0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.95])
phuong_sai = np.array([8.24, 6.11, 4.92, 4.12, 3.64, 3.04, 2.47])
chech2 = np.array([51.83, 61.85, 70.85, 78.61, 85.16, 92.84, 105.87])
net_dut = np.array([59.74, 68.03, 75.45, 82.62, 88.78, 96.09, 108.10])
muc_ls = 86.38

# v_ridge.py mục 16: toạ độ vectơ bảng Ridge của Hình 3.7 ESL
df_37 = np.array([-0.004968, 1.001, 1.988, 3.022, 3.991, 4.996, 6.021, 7.146, 7.995])
cv_37 = np.array([1.4765, 1.0635, 0.8401, 0.7156, 0.6465, 0.6030, 0.5734, 0.5536, 0.5575])
vach_ngang = 0.6332

# v_ridge.py mục 16 và Bảng 3.3 tr.63: hệ số LS theo biến
bien = ["lcavol", "lweight", "age", "lbph", "svi", "lcp", "gleason", "pgg45"]
bang_33 = np.array([0.680, 0.263, -0.141, 0.210, 0.305, -0.288, -0.021, 0.267])
hinh_38 = np.array([0.7165, 0.2931, -0.1422, 0.2124, 0.3098, -0.2886, -0.0204, 0.2775])

fig, ax = plt.subplots(2, 2, figsize=(11.5, 7.6))

a = ax[0, 0]
a.loglog(alpha_k1, sai_k1, "o-", color="#2563eb", ms=4, lw=1.3, label="Tikhonov, k = 1")
a.loglog(alpha_k0, sai_k0, "s--", color="#dc2626", ms=4, lw=1.1, label="ridge, k = 0")
a.axvline(1e-6, color="#111827", lw=0.8)
a.invert_xaxis()
a.set_xlabel("α (giảm dần sang phải)")
a.set_ylabel("max |z − z̄|")
a.set_title("Tikhonov 1963 tr.503: δ = 1e-3 cố định, α → 0\nsai số chạm đáy ở α = δ² rồi tăng lại", fontsize=9)
a.legend(fontsize=8)

a = ax[0, 1]
a.axhline(muc_ls, color="#6b7280", lw=1.0, label="Least Squares (86,38)")
a.plot(k_hk, phuong_sai, "o", color="#059669", ms=5, label="Variance")
a.plot(k_hk, chech2, "o", color="#d97706", ms=5, label="Bias-Squared")
a.plot(k_hk, net_dut, "o", color="#2563eb", ms=6, mfc="none", label="Ridge (nét đứt)")
a.plot(k_hk, phuong_sai + chech2, "+", color="#111827", ms=10, label="Variance + Bias²")
a.set_xlabel("k")
a.set_ylabel("đơn vị σ²")
a.set_title("Hoerl & Kennard 1970 Hình 1, số hoá bản quét\nđường đứt = tổng hai đường liền", fontsize=9)
a.legend(fontsize=7.5, loc="center right")

a = ax[1, 0]
a.plot(df_37, cv_37, "o-", color="#d97706", ms=5, lw=1.2)
a.axhline(vach_ngang, color="#7c3aed", lw=1.0, ls="--", label="cực tiểu + 1 SE (0,6332)")
jm = int(np.argmin(cv_37))
j1 = int(np.nonzero(cv_37 <= vach_ngang)[0][0])
a.plot(df_37[jm], cv_37[jm], "o", color="#dc2626", ms=9, label="cực tiểu: df = %.3f" % df_37[jm])
a.plot(df_37[j1], cv_37[j1], "o", color="#2563eb", ms=9, label="1-SE: df = %.3f" % df_37[j1])
a.set_xlabel("df(λ)")
a.set_ylabel("CV error")
a.set_title("ESL Hình 3.7 (Ridge): tr.68 viết 'minimum at df = 5.0'\ncực tiểu thật ở df ≈ 7, df = 5 là quy tắc 1-SE", fontsize=9)
a.legend(fontsize=7.5)

a = ax[1, 1]
xs = np.arange(len(bien))
a.axhline(0, color="#111827", lw=0.7)
a.bar(xs, hinh_38 - bang_33, color="#93c5fd", edgecolor="#1d4ed8")
a.set_xticks(xs)
a.set_xticklabels(bien, rotation=35, fontsize=8)
a.set_ylabel("đầu mút Hình 3.8 − cột LS Bảng 3.3")
a.set_title("ESL: cùng mô hình LS, hai quy ước chuẩn hoá\nHình 3.8 (67 mẫu) so với Bảng 3.3 (97 mẫu)", fontsize=9)

fig.tight_layout()
fig.savefig("ridge_dung_lai.png", dpi=130, bbox_inches="tight", facecolor="white")

print("Tikhonov: α tốt nhất k=1 = %.0e (%.6f) ; k=0 = %.0e (%.6f) ; tỉ số %.4f"
      % (alpha_k1[np.argmin(sai_k1)], sai_k1.min(), alpha_k0[np.argmin(sai_k0)], sai_k0.min(),
         sai_k0.min() / sai_k1.min()))
print("Hoerl-Kennard: |nét đứt − tổng| lớn nhất = %.2f" % np.abs(net_dut - phuong_sai - chech2).max())
print("ESL Hình 3.7: cực tiểu df = %.3f ; 1-SE df = %.3f" % (df_37[jm], df_37[j1]))
print("ESL Hình 3.8 − Bảng 3.3: lệch lớn nhất %.4f ở %s"
      % (np.abs(hinh_38 - bang_33).max(), bien[int(np.argmax(np.abs(hinh_38 - bang_33)))]))

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


def collinear(n=100, p=8, rho=0.99, noise=1.0, seed=0):
    """Thiet ke gan suy bien - dung boi canh Hoerl & Kennard 1970 mo ta:
    8 bien sinh ra tu chi 2 huong doc lap, nen XtX gan suy bien."""
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, 2))
    X = np.zeros((n, p))
    for j in range(p):
        X[:, j] = rho * Z[:, j % 2] + np.sqrt(1 - rho ** 2) * rng.normal(size=n)
    beta = np.zeros(p); beta[:2] = [3.0, -2.0]
    return X, X @ beta + noise * rng.normal(size=n), beta


X, y, beta = collinear(n=50, p=8, rho=0.99, noise=1.0, seed=3)
n, p = X.shape
XtX, Xty = X.T @ X, X.T @ y

# --- mo phong: chech^2 va phuong sai cua ridge theo lambda, 4000 lan rut nhieu ---
lam = np.array(sorted(set(np.r_[0, np.logspace(-4, 3, 60)])))
rng = np.random.default_rng(11)
REP = 4000
BB = {i: [] for i in range(len(lam))}
for _ in range(REP):
    Xty_ = X.T @ (X @ beta + rng.normal(size=n))
    for i, l in enumerate(lam):
        BB[i].append(np.linalg.solve(XtX + l * np.eye(p), Xty_))
bias2 = np.array([((np.array(BB[i]).mean(0) - beta) ** 2).sum() for i in range(len(lam))])
var = np.array([np.array(BB[i]).var(0).sum() for i in range(len(lam))])
mse = np.array([((np.array(BB[i]) - beta) ** 2).sum(1).mean() for i in range(len(lam))])
ibest = int(np.argmin(mse))

fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.6))

# bang trai: VET RIDGE - he so theo lambda, khop tren du lieu that (khong mo phong)
lam2 = np.logspace(-4, 4, 120)
Bp = np.array([np.linalg.solve(XtX + l * np.eye(p), Xty) for l in lam2])
for j in range(p):
    axes[0].semilogx(lam2, Bp[:, j], lw=1.2)
axes[0].axhline(0, color="#111827", lw=.7)
axes[0].set_xlabel("k (λ)"); axes[0].set_ylabel(r"$\hat\beta_j(k)$")
axes[0].set_title("Vết ridge — 8 biến, tương quan ρ = 0.99\n"
                  "(Hoerl & Kennard gọi là ridge trace)", fontsize=9)

# bang giua: phan ra MSE = chech^2 + phuong sai
axes[1].semilogx(lam[1:], mse[1:], color="#2563eb", lw=1.6, label="MSE")
axes[1].semilogx(lam[1:], bias2[1:], color="#dc2626", lw=1.1, ls="--", label="chệch²")
axes[1].semilogx(lam[1:], var[1:], color="#059669", lw=1.1, ls=":", label="phương sai")
axes[1].axhline(mse[0], color="#6b7280", lw=1.0, label="MSE của OLS")
axes[1].axvline(lam[ibest], color="#111827", lw=.8)
axes[1].set_yscale("log"); axes[1].set_xlabel("k (λ)"); axes[1].legend(fontsize=7.5)
axes[1].set_title("MSE = chệch² + phương sai, %d lần lặp\nk* = %.3g giảm MSE %.1f lần"
                  % (REP, lam[ibest], mse[0] / mse[ibest]), fontsize=9)

# bang phai: he so co theo tung huong rieng (muc 3.1)
s_ = np.linalg.svd(X, full_matrices=False)[1]
for l, col in [(0.1, "#2563eb"), (1.0, "#d97706"), (10.0, "#dc2626")]:
    axes[2].plot(s_ ** 2, s_ ** 2 / (s_ ** 2 + l), "o-", color=col, ms=4, lw=1.1,
                 label="λ = %g" % l)
axes[2].set_xscale("log"); axes[2].legend(fontsize=7.5)
axes[2].set_xlabel(r"$d_j^2$ (giá trị riêng của $X^\top X$)")
axes[2].set_ylabel(r"$d_j^2/(d_j^2+\lambda)$")
axes[2].set_title("Ridge co hướng nào? Hướng phương sai nhỏ\n"
                  "bị co mạnh, hướng lớn gần như giữ nguyên", fontsize=9)

fig.savefig("ridge_trace.png", dpi=130, bbox_inches="tight", facecolor="white")
better = lam[mse < mse[0]]
print("κ(XᵀX) = %.3e" % np.linalg.cond(XtX))
print("OLS: MSE %.4f = chệch² %.4f + phương sai %.4f"
      % (mse[0], bias2[0], var[0]))
print("k* %.4g: MSE %.4f = chệch² %.4f + phương sai %.4f"
      % (lam[ibest], mse[ibest], bias2[ibest], var[ibest]))
print("k lớn nhất trên lưới còn thắng OLS: %.4g" % better.max())
