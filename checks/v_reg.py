# -*- coding: utf-8 -*-
"""Kiểm chứng số cho toàn bộ GĐ 1 — hồi quy.

Mỗi mục in ra "bài báo nói X / đo được Y" để đối chiếu được.
"""
import sys
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
from mllib import *

np.set_printoptions(precision=4, suppress=True, linewidth=130)


def head(s):
    print("\n" + "=" * 78 + "\n" + s + "\n" + "=" * 78)


# ============================================================ 1. PHƯƠNG TRÌNH CHUẨN
head("1. Phương trình chuẩn XᵀX bình phương số điều kiện")
for p, alpha in [(6, 1.0), (10, 1.0)]:
    # ma trận Vandermonde trên [0,1] — kinh điển về điều kiện xấu
    t = np.linspace(0, 1, 60)
    X = np.vander(t, p, increasing=True)
    k_X = np.linalg.cond(X)
    k_N = np.linalg.cond(X.T @ X)
    print("   Vandermonde bậc %-2d: κ(X) = %.3e   κ(XᵀX) = %.3e   tỉ số = %.3e  (κ² = %.3e)"
          % (p - 1, k_X, k_N, k_N / k_X, k_X ** 2 / k_X))

print("\n   Hệ quả trên số thực float64 — nghiệm nào đúng hơn?")
rng = np.random.default_rng(0)
t = np.linspace(0, 1, 60)
for p in (8, 12, 16):
    X = np.vander(t, p, increasing=True)
    beta = np.ones(p)
    y = X @ beta                                # KHÔNG nhiễu: sai số chỉ do số học
    b_normal = np.linalg.solve(X.T @ X, X.T @ y)
    Q, R = np.linalg.qr(X)
    b_qr = np.linalg.solve(R, Q.T @ y)
    print("      p=%-3d κ(X)=%.2e   sai số ptr chuẩn = %.3e   sai số QR = %.3e"
          % (p, np.linalg.cond(X), np.abs(b_normal - beta).max(), np.abs(b_qr - beta).max()))

# ============================================================ 2. GAUSS–MARKOV
head("2. Gauss–Markov: OLS có phương sai nhỏ nhất trong các ước lượng tuyến tính không chệch")
X, y, beta = make_reg(n=40, p=3, noise=1.0, seed=1)
n, p = X.shape
XtXi = np.linalg.inv(X.T @ X)
print("   Cov lý thuyết σ²(XᵀX)⁻¹, đường chéo:", np.diag(XtXi))
# mô phỏng 200 000 lần rút nhiễu, giữ nguyên X
rng = np.random.default_rng(7)
B_ols, B_alt = [], []
A_ols = XtXi @ X.T
# một ước lượng tuyến tính KHÁC, cũng không chệch: A + D với DX = 0
D = np.eye(p) @ np.zeros((p, n))
Z = rng.normal(size=(p, n))
D = Z - (Z @ X) @ XtXi @ X.T          # chiếu để DX = 0  => vẫn không chệch
assert np.abs(D @ X).max() < 1e-10
for _ in range(200000):
    e = rng.normal(size=n)
    yy = X @ beta + e
    B_ols.append(A_ols @ yy)
    B_alt.append((A_ols + D) @ yy)
B_ols, B_alt = np.array(B_ols), np.array(B_alt)
print("   OLS      : chệch = %.4f   tổng phương sai = %.4f" %
      (np.abs(B_ols.mean(0) - beta).max(), B_ols.var(0).sum()))
print("   không OLS: chệch = %.4f   tổng phương sai = %.4f" %
      (np.abs(B_alt.mean(0) - beta).max(), B_alt.var(0).sum()))
print("   lý thuyết: tổng phương sai OLS = tr((XᵀX)⁻¹) = %.4f" % np.trace(XtXi))
print("   -> cả hai không chệch, nhưng phương sai OLS nhỏ hơn %.2f lần"
      % (B_alt.var(0).sum() / B_ols.var(0).sum()))

# ============================================================ 3. RIDGE — Hoerl & Kennard
head("3. Hoerl & Kennard 1970: tồn tại k > 0 làm MSE ridge < MSE OLS")
X, y, beta = collinear(n=50, p=8, rho=0.99, noise=1.0, seed=3)
n, p = X.shape
print("   κ(XᵀX) = %.3e  (thiết kế gần suy biến)" % np.linalg.cond(X.T @ X))
XtX, Xty = X.T @ X, X.T @ y
lam = np.array(sorted(set(np.r_[0, np.logspace(-4, 3, 60)])))
rng = np.random.default_rng(11)
REP = 4000
mse = np.zeros(len(lam)); bias2 = np.zeros(len(lam)); var = np.zeros(len(lam))
BB = {k: [] for k in range(len(lam))}
for _ in range(REP):
    yy = X @ beta + rng.normal(size=n)
    Xty_ = X.T @ yy
    for i, l in enumerate(lam):
        b = np.linalg.solve(XtX + l * np.eye(p), Xty_)
        BB[i].append(b)
for i in range(len(lam)):
    B = np.array(BB[i])
    bias2[i] = ((B.mean(0) - beta) ** 2).sum()
    var[i] = B.var(0).sum()
    mse[i] = ((B - beta) ** 2).sum(1).mean()
i0 = 0
ibest = int(np.argmin(mse))
print("   k = 0 (OLS)      : MSE = %.4f   (chệch² %.4f + phương sai %.4f)"
      % (mse[i0], bias2[i0], var[i0]))
print("   k* = %.4g        : MSE = %.4f   (chệch² %.4f + phương sai %.4f)"
      % (lam[ibest], mse[ibest], bias2[ibest], var[ibest]))
print("   -> giảm %.1f lần MSE. Định lý 4.3 của bài báo được xác nhận trên thiết kế này."
      % (mse[i0] / mse[ibest]))
kmax = lam[mse < mse[i0]].max()
print("   Khoảng k cho MSE < MSE(OLS): (0, %.4g]" % kmax)

# --- hình: vết ridge + phân rã chệch/phương sai ---
fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.6))
lam2 = np.logspace(-4, 4, 120)
Bp = np.array([np.linalg.solve(XtX + l * np.eye(p), Xty) for l in lam2])
for j in range(p):
    axes[0].semilogx(lam2, Bp[:, j], lw=1.2)
axes[0].axhline(0, color="#111827", lw=.7)
axes[0].set_xlabel("k (λ)"); axes[0].set_ylabel(r"$\hat\beta_j(k)$")
axes[0].set_title("Vết ridge — 8 biến, tương quan ρ = 0.99\n(Hoerl & Kennard gọi là ridge trace)", fontsize=9)
axes[1].semilogx(lam[1:], mse[1:], color=C["Ridge"], lw=1.6, label="MSE")
axes[1].semilogx(lam[1:], bias2[1:], color="#dc2626", lw=1.1, ls="--", label="chệch²")
axes[1].semilogx(lam[1:], var[1:], color="#059669", lw=1.1, ls=":", label="phương sai")
axes[1].axhline(mse[0], color="#6b7280", lw=1.0, label="MSE của OLS")
axes[1].axvline(lam[ibest], color="#111827", lw=.8)
axes[1].set_yscale("log"); axes[1].set_xlabel("k (λ)")
axes[1].legend(fontsize=7.5)
axes[1].set_title("MSE = chệch² + phương sai, %d lần lặp\nk* = %.3g giảm MSE %.1f lần"
                  % (REP, lam[ibest], mse[0] / mse[ibest]), fontsize=9)
# hệ số thu nhỏ theo giá trị kỳ dị
U, s, Vt = np.linalg.svd(X, full_matrices=False)
for l, col in [(0.1, "#2563eb"), (1.0, "#d97706"), (10.0, "#dc2626")]:
    axes[2].plot(s ** 2, s ** 2 / (s ** 2 + l), "o-", color=col, ms=4, lw=1.1, label="λ = %g" % l)
axes[2].set_xscale("log"); axes[2].set_xlabel(r"$d_j^2$ (giá trị riêng của $X^\top X$)")
axes[2].set_ylabel(r"$d_j^2/(d_j^2+\lambda)$")
axes[2].legend(fontsize=7.5)
axes[2].set_title("Ridge co hướng nào? Hướng phương sai nhỏ\nbị co mạnh, hướng lớn gần như giữ nguyên", fontsize=9)
# Hình vết ridge của bài Ridge KHÔNG ghi ở đây nữa (2026-09-15): bản chuẩn là khối mã in trong bài,
# trích ra tools/fig_ridge.py. Không viết tên file hình vào chú thích này: sohinh.py dò script sinh
# hình bằng cách tìm TÊN FILE trong tools/*.py, và một chú thích có tên đó làm nó chạy lại cả script này.
plt.close(fig)

# ============================================================ 4. LASSO vs RIDGE
head("4. Zou & Hastie 2005, mục 2: ba giới hạn của lasso")
from sklearn.linear_model import Lasso, Ridge, ElasticNet, LassoLars, lars_path

# (a) p > n: lasso chọn tối đa n biến
rng = np.random.default_rng(0)
n_, p_ = 20, 60
Xa = rng.normal(size=(n_, p_))
ba = np.zeros(p_); ba[:5] = 3
ya = Xa @ ba + rng.normal(size=n_) * .5
nz_cd = []
for a in np.logspace(-3, 0, 40):
    m = Lasso(alpha=a, max_iter=50000).fit(Xa, ya)
    nz_cd.append(int((m.coef_ != 0).sum()))
_, _, cf = lars_path(Xa, ya, method="lasso")
nz_lars = [int((cf[:, k] != 0).sum()) for k in range(cf.shape[1])]
print("   (a) n = %d, p = %d — 'lasso chọn nhiều nhất n biến' (Zou & Hastie mục 2.1)" % (n_, p_))
print("       LARS (nghiệm chính xác)         : nhiều nhất %d biến  -> %s"
      % (max(nz_lars), "KHỚP" if max(nz_lars) <= n_ else "KHÔNG khớp"))
print("       Lasso(coordinate descent, sklearn): nhiều nhất %d biến  -> %s"
      % (max(nz_cd), "KHỚP" if max(nz_cd) <= n_ else "KHÔNG khớp — xem ghi chú"))
a_bad = np.logspace(-3, 0, 40)[int(np.argmax(nz_cd))]
m_bad = Lasso(alpha=a_bad, max_iter=50000).fit(Xa, ya)
tiny = np.abs(m_bad.coef_[m_bad.coef_ != 0])
print("       Ghi chú: ở α = %.4g, %d hệ số 'khác 0' của coordinate descent có biên độ"
      % (a_bad, len(tiny)))
print("       nhỏ nhất %.2e — đó là ngưỡng hội tụ chưa siết, không phải nghiệm lasso thật."
      % tiny.min())
print("       Siết tol xuống 1e-14: %d biến."
      % int((Lasso(alpha=a_bad, max_iter=2000000, tol=1e-14).fit(Xa, ya).coef_ != 0).sum()))

# (b) nhóm biến tương quan: lasso chọn một cách tuỳ tiện
rng = np.random.default_rng(1)
n_ = 100
z = rng.normal(size=n_)
Xb = np.c_[tuple(z + .01 * rng.normal(size=n_) for _ in range(3))]
Xb = np.c_[Xb, rng.normal(size=(n_, 3))]
yb = 3 * z + .5 * rng.normal(size=n_)
print("\n   (b) ba cột gần trùng nhau (tương quan %.4f), hệ số thật chia đều"
      % np.corrcoef(Xb[:, 0], Xb[:, 1])[0, 1])
for name, m in [("Lasso  α=0.3", Lasso(alpha=.3, max_iter=50000)),
                ("Ridge  α=0.3", Ridge(alpha=.3)),
                ("ElNet  α=0.3 l1=0.5", ElasticNet(alpha=.3, l1_ratio=.5, max_iter=50000))]:
    m.fit(Xb, yb)
    print("       %-20s hệ số 3 cột nhóm = %s" % (name, np.round(m.coef_[:3], 4)))

# (c) n > p, tương quan cao: Zou & Hastie nói ridge dự báo tốt hơn lasso.
# Chỉ so được nếu MỖI bộ đều được chọn siêu tham số công bằng -> dùng CV 5 mảnh.
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV
rng = np.random.default_rng(2)
res = {"Lasso": [], "Ridge": [], "ElasticNet": []}
GR = np.logspace(-3, 3, 40)
for rep in range(60):
    Xc, yc, bc = collinear(n=120, p=10, rho=0.995, noise=2.0, seed=100 + rep)
    tr, te = slice(0, 60), slice(60, 120)
    for nm, M in [("Lasso", LassoCV(alphas=GR, cv=5, max_iter=100000)),
                  ("Ridge", RidgeCV(alphas=GR)),
                  ("ElasticNet", ElasticNetCV(alphas=GR, l1_ratio=[.1, .3, .5, .7, .9],
                                              cv=5, max_iter=100000))]:
        M.fit(Xc[tr], yc[tr])
        res[nm].append(rmse(M.predict(Xc[te]), yc[te]))
print("\n   (c) n=120 > p=10, ρ = 0.995, 60 lần lặp, siêu tham số chọn bằng CV 5 mảnh")
print("       Zou & Hastie mục 2.3: 'the prediction performance of the lasso is dominated")
print("       by ridge regression' trong tình huống này. RMSE kiểm tra đo được:")
base = np.array(res["Lasso"])
for k, v in res.items():
    v = np.array(v)
    d = v - base
    print("       %-11s %.4f ± %.4f   (chênh với lasso: %+.4f ± %.4f)"
          % (k, v.mean(), v.std(), d.mean(), d.std() / np.sqrt(len(d))))
w = int((np.array(res["Ridge"]) < base).sum())
print("       Ridge thắng lasso ở %d/%d lần lặp." % (w, len(base)))
print("       -> KHÔNG tái hiện được. Lý do: collinear() ở trên chỉ có 2 nhân tố ẩn nên")
print("       hệ số thật THƯA (8/10 bằng 0) — đó là sân nhà của lasso, không phải của ridge.")


def dense_corr(n=120, p=10, rho=.9, noise=3.0, seed=0):
    """Cùng cấu trúc tương quan nhưng hệ số thật DÀY — đúng tình huống Zou & Hastie tả."""
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(rho ** np.abs(np.subtract.outer(np.arange(p), np.arange(p))))
    X = rng.normal(size=(n, p)) @ L.T
    b = rng.normal(size=p) * 0.8 + 1.0          # mọi hệ số đều khác 0
    return X, X @ b + noise * rng.normal(size=n), b


res2 = {"Lasso": [], "Ridge": [], "ElasticNet": []}
for rep in range(60):
    Xc, yc, bc = dense_corr(seed=300 + rep)
    tr, te = slice(0, 60), slice(60, 120)
    for nm, M in [("Lasso", LassoCV(alphas=GR, cv=5, max_iter=100000)),
                  ("Ridge", RidgeCV(alphas=GR)),
                  ("ElasticNet", ElasticNetCV(alphas=GR, l1_ratio=[.1, .3, .5, .7, .9],
                                              cv=5, max_iter=100000))]:
        M.fit(Xc[tr], yc[tr])
        res2[nm].append(rmse(M.predict(Xc[te]), yc[te]))
print("\n       Làm lại với hệ số thật DÀY (10/10 khác 0), ρ giảm dần theo khoảng cách 0.9^|i−j|:")
b2 = np.array(res2["Lasso"])
for k, v in res2.items():
    v = np.array(v)
    print("       %-11s %.4f ± %.4f   (chênh với lasso: %+.4f ± %.4f)"
          % (k, v.mean(), v.std(), (v - b2).mean(), (v - b2).std() / np.sqrt(len(v))))
w2 = int((np.array(res2["Ridge"]) < b2).sum())
print("       Ridge thắng lasso ở %d/%d lần lặp -> %s"
      % (w2, len(b2), "khớp Zou & Hastie" if w2 > 40 else "vẫn chưa dứt khoát"))

# ============================================================ 5. HIỆU ỨNG NHÓM
head("5. Zou & Hastie 2005, Định lý 1: chặn |β̂i − β̂j| của elastic net")
rng = np.random.default_rng(5)
n_ = 200
print("   Chặn:  |β̂i − β̂j| / ‖y‖₁  ≤  (1/λ₂)·√(2(1−ρ))")
print("   %-8s %-12s %-14s %-14s %s" % ("ρ", "λ₂", "đo được", "chặn", "thoả?"))
for rho in (0.9, 0.99, 0.999):
    zz = rng.normal(size=n_)
    x1 = rho * zz + np.sqrt(1 - rho ** 2) * rng.normal(size=n_)
    x2 = rho * zz + np.sqrt(1 - rho ** 2) * rng.normal(size=n_)
    Xg = np.c_[x1, x2, rng.normal(size=(n_, 4))]
    Xg = (Xg - Xg.mean(0)) / Xg.std(0)
    yg = 2 * zz + .3 * rng.normal(size=n_)
    yg = yg - yg.mean()
    r = np.corrcoef(x1, x2)[0, 1]
    for l2 in (0.5, 5.0):
        # elastic net của sklearn: alpha*l1_ratio = λ₁/n, alpha*(1-l1_ratio) = λ₂/(2n)
        l1 = 0.05
        a = (l1 + 2 * l2) / (2 * n_) if False else None
        alpha = (l1 / n_) + (l2 / n_)
        l1r = (l1 / n_) / alpha
        m = ElasticNet(alpha=alpha, l1_ratio=l1r, max_iter=200000, tol=1e-10).fit(Xg, yg)
        d = abs(m.coef_[0] - m.coef_[1]) / np.abs(yg).sum()
        bound = np.sqrt(2 * (1 - r)) / l2
        print("   %-8.4f %-12g %-14.3e %-14.3e %s"
              % (r, l2, d, bound, "có" if d <= bound else "KHÔNG"))

# ============================================================ 6. LARS trên diabetes
head("6. Efron 2004: dựng lại thứ tự biến vào mô hình trên bộ diabetes")
Xd, yd, names = diabetes()
print("   n = %d, p = %d, tên biến: %s" % (Xd.shape[0], Xd.shape[1], names))
alphas, active, coefs = lars_path(Xd, yd, method="lasso")
order, drops = [], []
prev = set()
for k in range(coefs.shape[1]):
    cur = set(np.nonzero(coefs[:, k])[0])
    for j in sorted(cur - prev):
        order.append(int(j) + 1)                # đánh số từ 1 như bài báo
    for j in sorted(prev - cur):
        drops.append((k, int(j) + 1))
    prev = cur
print("   Bài báo (Hình 1, chú thích): 'covariates enter ... in order j = 3, 9, 4, 7, ..., 1'")
print("   dựng lại thứ tự vào        :", order)
print("   bốn biến đầu %s -> %s" % (order[:4], "KHỚP" if order[:4] == [3, 9, 4, 7] else "LỆCH"))
print("   Bài báo nói biến CUỐI vào là 1; dựng lại: biến thứ 10 vào là %d -> %s"
      % (order[9], "KHỚP" if order[9] == 1 else "LỆCH"))
if drops:
    print("   *** Đường lasso có %d lần một biến RỜI mô hình rồi vào lại: %s"
          % (len(drops), ", ".join("biến %d rời ở bước %d" % (j, k) for k, j in drops)))
    print("       Đó là bước 'lasso modification' ở mục 3.1 của bài báo — LARS thuần")
    print("       không bao giờ bỏ biến, nên đường lasso dài hơn %d bước so với p = 10."
          % (len(order) - 10))
t_full = np.abs(coefs[:, -1]).sum()
print("   Bài báo: t chạy từ 0 tới t = 3460.00 (nơi β̂ = OLS)")
print("   dựng lại: Σ|β̂ⱼ| tại bước cuối = %.2f" % t_full)
b_ols = np.linalg.lstsq(Xd, yd, rcond=None)[0]
print("   Σ|β̂_OLS| = %.2f   (lệch %.3e)" % (np.abs(b_ols).sum(), abs(np.abs(b_ols).sum() - t_full)))
# tại t = 1000 bài báo nói chỉ 3,9,4,7 khác 0
tpath = np.abs(coefs).sum(0)
k1000 = np.searchsorted(tpath, 1000)
w = np.array(tpath)
i_lo = np.where(w <= 1000)[0][-1]
frac = (1000 - w[i_lo]) / (w[i_lo + 1] - w[i_lo])
c1000 = coefs[:, i_lo] + frac * (coefs[:, i_lo + 1] - coefs[:, i_lo])
print("   Tại t = 1000, biến khác 0: %s   (bài báo nói: 3, 9, 4, 7)"
      % sorted(int(v) for v in np.nonzero(np.abs(c1000) > 1e-9)[0] + 1))
print("   -> %s" % ("KHỚP" if sorted(np.nonzero(np.abs(c1000) > 1e-9)[0] + 1) == [3, 4, 7, 9] else "LỆCH"))

# --- hình: đường lasso vs ridge trên cùng dữ liệu ---
fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.8))
tt = np.abs(coefs).sum(0)
for j in range(coefs.shape[0]):
    axes[0].plot(tt, coefs[j], lw=1.2)
    axes[0].annotate(str(j + 1), (tt[-1], coefs[j, -1]), fontsize=6.5,
                     xytext=(3, -2), textcoords="offset points")
axes[0].axvline(1000, color="#111827", ls=":", lw=.9)
axes[0].text(1020, coefs.max() * .9, "t = 1000", fontsize=7)
axes[0].axhline(0, color="#111827", lw=.6)
axes[0].set_xlabel(r"$t=\sum_j|\hat\beta_j|$"); axes[0].set_ylabel(r"$\hat\beta_j$")
axes[0].set_title("Lasso qua LARS — bộ diabetes (Efron 2004, Hình 1 trái)\n"
                  "thứ tự vào: %s" % ", ".join(map(str, order[:6])), fontsize=9)
lam3 = np.logspace(-4, 2, 150)
Bp = np.array([np.linalg.solve(Xd.T @ Xd + l * np.eye(10), Xd.T @ yd) for l in lam3])
tr_ = np.abs(Bp).sum(1)
for j in range(10):
    axes[1].plot(tr_, Bp[:, j], lw=1.2)
    axes[1].annotate(str(j + 1), (tr_[0], Bp[0, j]), fontsize=6.5,
                     xytext=(3, -2), textcoords="offset points")
axes[1].axhline(0, color="#111827", lw=.6)
axes[1].set_xlabel(r"$\sum_j|\hat\beta_j|$"); axes[1].set_ylabel(r"$\hat\beta_j$")
axes[1].set_title("Ridge trên cùng dữ liệu — không hệ số nào\nchạm 0 đúng, đó là khác biệt cốt lõi", fontsize=9)
# Hình đường lasso/ridge của bài Lasso KHÔNG ghi ở đây nữa (2026-09-17): bản chuẩn là khối mã in
# trong bài. Chú thích này cố ý không nhắc tên file hình — sohinh.py dò script sinh hình theo tên file.
plt.close(fig)

# ============================================================ 7. HUBER
head("7. Huber 1964: δ = 1.345 cho hiệu suất 95 % tại phân phối chuẩn")
from sklearn.linear_model import HuberRegressor, LinearRegression, QuantileRegressor


def eff(delta, reps=20000, n=40, seed=0):
    """Hiệu suất tiệm cận = var(trung bình mẫu) / var(ước lượng Huber) khi nhiễu CHUẨN."""
    rng = np.random.default_rng(seed)
    v_mean, v_hub = [], []
    for _ in range(reps):
        x = rng.normal(size=n)
        v_mean.append(x.mean())
        # ước lượng M của Huber cho tham số vị trí, thang đo = 1
        mu = np.median(x)
        for _ in range(60):
            r = x - mu
            w = np.clip(delta / np.maximum(np.abs(r), 1e-12), None, 1.0)
            mu_new = (w * x).sum() / w.sum()
            if abs(mu_new - mu) < 1e-12:
                break
            mu = mu_new
        v_hub.append(mu)
    return np.var(v_mean) / np.var(v_hub)


for d in (0.5, 1.0, 1.345, 2.0, 5.0):
    print("   δ = %-6.3f hiệu suất tại phân phối chuẩn = %.4f" % (d, eff(d)))
print("   (giá trị δ = 1.345 là con số kinh điển cho 95 %; đo được ở trên)")

print("\n   Điểm sụp đổ — thêm 1 điểm ngoại lai vào n = 100:")
Xh, yh, bh = make_reg(n=100, p=1, noise=1.0, seed=4)
for shift in (0, 10, 100, 1000, 100000):
    yy = yh.copy()
    if shift:
        yy[0] += shift
    a = LinearRegression().fit(Xh, yy).coef_[0]
    b = HuberRegressor(epsilon=1.345, alpha=0.0).fit(Xh, yy).coef_[0]
    print("      lệch +%-8g  OLS = %-9.4f  Huber = %-9.4f  (hệ số thật %.1f)"
          % (shift, a, b, bh[0]))

# ============================================================ 8. QUANTILE
head("8. Koenker & Bassett: hồi quy phân vị")
rng = np.random.default_rng(0)
n_ = 400
xq = np.sort(rng.uniform(0, 10, n_))
yq = 1 + 0.5 * xq + (0.3 + 0.3 * xq) * rng.normal(size=n_)   # phương sai TĂNG theo x
Xq = xq[:, None]
print("   Dữ liệu phương sai không đều; hệ số thật của trung bình: 1 + 0.5x")
print("   %-6s %-22s %-14s %s" % ("τ", "hệ số (b₀, b₁)", "tỉ lệ y dưới", "số dư = 0"))
for tau in (0.1, 0.25, 0.5, 0.75, 0.9):
    m = QuantileRegressor(quantile=tau, alpha=0, solver="highs").fit(Xq, yq)
    pred = m.predict(Xq)
    below = np.mean(yq < pred)
    zeros = int(np.sum(np.abs(yq - pred) < 1e-8))
    print("   %-6.2f (%7.4f, %7.4f)      %-14.4f %d" % (tau, m.intercept_, m.coef_[0], below, zeros))
print("   Koenker: số dư đúng bằng 0 phải bằng số tham số p = 2 -> xem cột cuối")

# --- hình chung cho Huber + Quantile ---
fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.8))
Xh1, yh1, bh1 = make_reg(n=60, p=1, noise=1.0, seed=4)
yout = yh1.copy(); idx = [3, 17, 40]; yout[idx] += 18
xs = np.linspace(Xh1.min(), Xh1.max(), 100)[:, None]
axes[0].scatter(Xh1[:, 0], yout, s=14, color="#6b7280", zorder=3)
axes[0].scatter(Xh1[idx, 0], yout[idx], s=42, facecolor="none", edgecolor="#dc2626", lw=1.3, zorder=4)
for nm, M, col in [("OLS", LinearRegression(), C["OLS"]),
                   ("Huber δ=1.345", HuberRegressor(epsilon=1.345, alpha=0), C["Huber"]),
                   ("Phân vị τ=0.5", QuantileRegressor(quantile=.5, alpha=0, solver="highs"), C["Quantile"])]:
    M.fit(Xh1, yout)
    axes[0].plot(xs, M.predict(xs), lw=1.6, color=col,
                 label="%s (hệ số %.3f)" % (nm, M.coef_[0]))
axes[0].plot(xs, bh1[0] * xs, "--", color="#111827", lw=1.0, label="thật (%.1f)" % bh1[0])
axes[0].legend(fontsize=7.2); axes[0].set_xlabel("x"); axes[0].set_ylabel("y")
axes[0].set_title("3/60 điểm bị đẩy lên 18 đơn vị\nOLS lệch hẳn, hai bộ còn lại gần như không đổi", fontsize=9)

axes[1].scatter(xq, yq, s=7, color="#9ca3af")
for tau, col in [(.1, "#2563eb"), (.25, "#0891b2"), (.5, "#111827"), (.75, "#d97706"), (.9, "#dc2626")]:
    m = QuantileRegressor(quantile=tau, alpha=0, solver="highs").fit(Xq, yq)
    axes[1].plot(xq, m.predict(Xq), lw=1.4, color=col, label="τ = %.2f" % tau)
axes[1].legend(fontsize=7.2, ncol=2); axes[1].set_xlabel("x"); axes[1].set_ylabel("y")
axes[1].set_title("Phương sai tăng theo x: 5 phân vị TOẢ RA\nmột đường trung bình không mô tả nổi", fontsize=9)
save(fig, "Huber Regression", "robust_vs_quantile.png")

# ============================================================ 9. BAYES = RIDGE
head("9. Hồi quy tuyến tính Bayes: hậu nghiệm MAP đúng bằng ridge với λ = σ²/τ²")
X, y, beta = make_reg(n=60, p=4, noise=1.5, seed=9)
sigma2, tau2 = 1.5 ** 2, 2.0 ** 2
lam = sigma2 / tau2
n, p = X.shape
Sn = np.linalg.inv(X.T @ X / sigma2 + np.eye(p) / tau2)
mn = Sn @ (X.T @ y) / sigma2
b_ridge = np.linalg.solve(X.T @ X + lam * np.eye(p), X.T @ y)
print("   σ² = %.2f  τ² = %.2f  ->  λ = σ²/τ² = %.4f" % (sigma2, tau2, lam))
print("   trung bình hậu nghiệm :", np.round(mn, 6))
print("   ridge(λ)              :", np.round(b_ridge, 6))
print("   lệch lớn nhất         : %.3e" % np.abs(mn - b_ridge).max())
print("   độ lệch chuẩn hậu nghiệm:", np.round(np.sqrt(np.diag(Sn)), 4))
print("   hệ số thật              :", beta)
inside = np.abs(mn - beta) <= 1.96 * np.sqrt(np.diag(Sn))
print("   hệ số thật nằm trong khoảng 95 %%: %d/%d" % (inside.sum(), p))

print("\nXONG — mọi con số trên sinh từ file này.")
