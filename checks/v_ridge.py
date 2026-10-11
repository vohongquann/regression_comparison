# -*- coding: utf-8 -*-
"""Dựng lại mọi phép toán của hai nguồn gốc bài Ridge Regression (Tikhonov Regularization).

Nguồn:
  [T] А. Н. Тихонов, "О решении некорректно поставленных задач и методе регуляризации",
      Докл. АН СССР 151:3 (1963), 501–504.
      AI algorithms/_papers/1963 - Tikhonov - O reshenii nekorrektno postavlennykh zadach i metode regulyarizatsii.pdf
  [E] T. Hastie, R. Tibshirani, J. Friedman, The Elements of Statistical Learning, 2nd ed.,
      Springer 2009 (bản in 12), mục 3.4.1 Ridge Regression, tr.61–68.

Tikhonov 1963 không có ví dụ số nào. Nên các mục 1–6 dưới đây KHÔNG dựng lại số in; chúng
kiểm từng công thức và từng bước chứng minh của bài báo bằng một bài toán rời rạc cụ thể,
và dựng đúng "ví dụ" mà bài báo nói là "нетрудно убедиться на примерах" (tr.503) mà không in.

Chạy:  cd tools && python v_ridge.py
"""
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
np.set_printoptions(linewidth=120)


def head(t):
    print("\n" + "=" * 96 + "\n" + t + "\n" + "=" * 96)


# ─────────────────────────────────────────────────────────── bài toán rời rạc chung
# Phương trình Fredholm loại một (1) tr.501:  ∫_a^b K(x,s) z(s) ds = u(x),  c ≤ x ≤ d.
# Lưới đúng như mục 3 tr.503:  s_j = j h − 0,5 h,  x_i = i h1 − 0,5 h1.

def luoi(a, b, c, d, n, m):
    h = (b - a) / n            # bài báo in h = (a − b)/n — xem mục 1
    h1 = (d - c) / m
    s = a + np.arange(1, n + 1) * h - 0.5 * h
    x = c + np.arange(1, m + 1) * h1 - 0.5 * h1
    return h, h1, s, x


def nhan(x, s, w=0.08):
    return np.exp(-(x[:, None] - s[None, :]) ** 2 / (2 * w * w))


def z_that(s):
    return np.cos(np.pi * s) + 0.5 * np.cos(3 * np.pi * s)   # z'(0) = z'(1) = 0


def he_3phay(K, u, h, h1, alpha, k, p):
    """Lập ĐÚNG hệ (3′) tr.503 theo từng số hạng in, rồi giải.

    α{ (1/h²)[k_j(ẑ_{j+1} − ẑ_j) − k_{j−1}(ẑ_j − ẑ_{j−1})] − p_j ẑ_j }
      − { Σ_l K̄_jl ẑ_l h − b̂_j } = 0,     ẑ_0 = ẑ_1,  ẑ_{n+1} = ẑ_n
    K̄_jl = Σ_i K_ij K_il h1,   b̂_j = Σ_i K_ij û_i h1.
    k có n+1 phần tử: k[0] ứng k_0 (nhân với ẑ_1 − ẑ_0 = 0 nên không ảnh hưởng).
    """
    n = K.shape[1]
    Kb = K.T @ K * h1
    bb = K.T @ u * h1
    M = np.zeros((n, n))
    for j in range(n):                       # j ở đây là chỉ số 0..n-1, ứng j+1 của bài báo
        # số hạng k_j(ẑ_{j+1} − ẑ_j): với j = n thì ẑ_{n+1} = ẑ_n nên bằng 0
        if j + 1 < n:
            M[j, j + 1] += alpha * k[j + 1] / h ** 2
            M[j, j] -= alpha * k[j + 1] / h ** 2
        # số hạng − k_{j−1}(ẑ_j − ẑ_{j−1}): với j = 1 thì ẑ_0 = ẑ_1 nên bằng 0
        if j - 1 >= 0:
            M[j, j] -= alpha * k[j] / h ** 2
            M[j, j - 1] += alpha * k[j] / h ** 2
        M[j, j] -= alpha * p[j]
    M -= Kb * h
    return np.linalg.solve(M, -bb)


def phiem_ham(K, u, h, h1, alpha, k, p, z):
    """Phiếm hàm làm trơn sai phân M̂_h^α[ẑ, û] tr.503, tính trực tiếp theo công thức in."""
    n = len(z)
    N = (((K @ z) * h - u) ** 2).sum() * h1
    dz = np.r_[np.diff(z), 0.0]              # ẑ_{n+1} − ẑ_n = 0
    Om = (k[1:] * dz ** 2 / h + p * z ** 2 * h).sum()
    return N + alpha * Om, N, Om


def cuc_tieu_truc_tiep(K, u, h, h1, alpha, k, p):
    """Cực tiểu M̂ bằng bình phương tối thiểu trên hệ ghép — KHÔNG đi qua (3′)."""
    n = K.shape[1]
    D = np.zeros((n, n))
    for j in range(n - 1):
        D[j, j], D[j, j + 1] = -1.0, 1.0
    A = np.vstack([K * h * np.sqrt(h1),
                   np.sqrt(alpha) * np.sqrt(k[1:])[:, None] * D / np.sqrt(h),
                   np.sqrt(alpha * h) * np.diag(np.sqrt(p))])
    rhs = np.r_[u * np.sqrt(h1), np.zeros(2 * n)]
    return np.linalg.lstsq(A, rhs, rcond=None)[0]


# ═════════════════════════════════════════════════════ 1. Lưới tr.503 và dấu của h
head("1. Tikhonov tr.503 — lưới s_j = jh − 0,5h và công thức h = (a − b)/n in trong bài")
for (a, b, n) in [(0.0, 1.0, 10), (1.0, 3.0, 4)]:
    h_in = (a - b) / n
    h_dung = (b - a) / n
    s_in = np.arange(1, n + 1) * h_in - 0.5 * h_in
    s_dung = a + np.arange(1, n + 1) * h_dung - 0.5 * h_dung
    print("   (a, b) = (%g, %g), n = %d:  h in = %+.4f  -> s_1..s_n = %s"
          % (a, b, n, h_in, np.round(s_in, 4)))
    print("   %28s h đúng = %+.4f  -> s_1..s_n = %s" % ("", h_dung, np.round(s_dung, 4)))
print("   -> h = (a − b)/n ÂM vì a < b, và s_j = jh − 0,5h còn thiếu số hạng a: lưới in chỉ nằm")
print("      trong (a, b) khi a = 0 VÀ h lấy dấu dương. Tương tự h1 = (c − d)/m.")

# ═════════════════════════════════════════════════════ 2. Hệ (3′) đúng là điều kiện cực tiểu
head("2. Tikhonov tr.503 — hệ (3′) có đúng là điều kiện cực tiểu của phiếm hàm M̂_h^α không")
a, b, c, d = 0.0, 1.0, 0.0, 1.0
n, m = 80, 100
h, h1, s, x = luoi(a, b, c, d, n, m)
K = nhan(x, s)
zb = z_that(s)
ub = K @ zb * h                                   # ū = A[x, z̄] rời rạc, không nhiễu
kk = np.ones(n + 1)
pp = np.ones(n)
for alpha in (1e-2, 1e-5, 1e-8):
    z1 = he_3phay(K, ub, h, h1, alpha, kk, pp)
    z2 = cuc_tieu_truc_tiep(K, ub, h, h1, alpha, kk, pp)
    print("   α = %-6g  nghiệm hệ (3′) so với cực tiểu trực tiếp: lệch lớn nhất %.3e  (biên độ %.4f)"
          % (alpha, np.abs(z1 - z2).max(), np.abs(z1).max()))
# gradient số của M̂ tại nghiệm của (3′)
alpha = 1e-4
z1 = he_3phay(K, ub, h, h1, alpha, kk, pp)
g = np.zeros(n)
eps = 1e-6
for j in range(n):
    e = np.zeros(n); e[j] = eps
    g[j] = (phiem_ham(K, ub, h, h1, alpha, kk, pp, z1 + e)[0]
            - phiem_ham(K, ub, h, h1, alpha, kk, pp, z1 - e)[0]) / (2 * eps)
print("   α = 1e-4: chuẩn vô cùng của gradient số của M̂ tại nghiệm (3′) = %.3e" % np.abs(g).max())
print("   -> (3′) khớp: nó chính là ∂M̂/∂ẑ_j = 0 chia cho −2h. Bài báo không in bước chia đó.")

# ═════════════════════════════════════════════════════ 3. Bất đẳng thức tr.502 trong chứng minh định lý 2
head("3. Tikhonov tr.502 — chứng minh định lý 2: in ‖u^α − ū‖ ≤ αC; đúng ra phải là ≤ C√α")
C2 = phiem_ham(K, ub, h, h1, 1.0, kk, pp, zb)[2]
C = np.sqrt(C2)
print("   C² = Ω[z̄] = %.6f,  C = %.6f" % (C2, C))
print("   %10s %14s %14s %14s %10s %10s"
      % ("α", "‖u^α − ū‖", "αC (in)", "C√α (đúng)", "≤ αC ?", "≤ C√α ?"))
vi_pham_in = 0
vi_pham_dung = 0
ALPHAS = [10.0 ** e for e in range(-1, -11, -1)]
for alpha in ALPHAS:
    za = he_3phay(K, ub, h, h1, alpha, kk, pp)
    M_, N_, Om_ = phiem_ham(K, ub, h, h1, alpha, kk, pp, za)
    r = np.sqrt(N_)
    ok_in, ok_dung = r <= alpha * C, r <= C * np.sqrt(alpha)
    vi_pham_in += (not ok_in)
    vi_pham_dung += (not ok_dung)
    assert M_ <= alpha * C2 * (1 + 1e-9), "M^α[z^α] phải ≤ αC² (bước đầu của chứng minh)"
    assert Om_ <= C2 * (1 + 1e-9), "Ω[z^α] phải ≤ C² (hệ quả 1 của chứng minh)"
    print("   %10.0e %14.6e %14.6e %14.6e %10s %10s"
          % (alpha, r, alpha * C, C * np.sqrt(alpha), ok_in, ok_dung))
print("   Trên %d giá trị α: vi phạm ‖u^α − ū‖ ≤ αC ở %d; vi phạm ≤ C√α ở %d."
      % (len(ALPHAS), vi_pham_in, vi_pham_dung))
# phép thử thang đo: nhân Ω với c thì nghiệm giữ nguyên nếu α chia c
cc = 1e4
print("\n   Phép thử thang đo: thay (k, p) bằng (c·k, c·p) với c = %g và α bằng α/c." % cc)
print("   Nghiệm KHÔNG đổi, ‖u^α − ū‖ KHÔNG đổi, C nhân √c, nên αC chia √c còn C√α đứng yên.")
vi_pham_in_c = 0
for alpha in ALPHAS:
    za = he_3phay(K, ub, h, h1, alpha, kk, pp)
    zc = he_3phay(K, ub, h, h1, alpha / cc, cc * kk, cc * pp)
    assert np.abs(za - zc).max() < 1e-6 * max(1.0, np.abs(za).max())
    r = np.sqrt(phiem_ham(K, ub, h, h1, alpha, kk, pp, za)[1])
    Cc = C * np.sqrt(cc)
    vi_pham_in_c += r > (alpha / cc) * Cc
    assert r <= Cc * np.sqrt(alpha / cc) * (1 + 1e-9)
print("   Sau khi đổi thang: vi phạm ≤ αC ở %d/%d giá trị α; vi phạm ≤ C√α ở 0/%d."
      % (vi_pham_in_c, len(ALPHAS), len(ALPHAS)))
print("   -> Một bất đẳng thức đổi đúng/sai khi đổi đơn vị của Ω thì không thể là hệ quả của")
print("      M^α ≤ αC². Kết luận của định lý 2 (‖u^α − ū‖ → 0) vẫn đúng với cận C√α.")

# ═════════════════════════════════════════════════════ 4. Định lý 3 và câu "нетрудно убедиться на примерах"
head("4. Tikhonov tr.502–503 — định lý 3 (chọn α theo δ²) và ví dụ bài báo nói mà không in")
rng = np.random.default_rng(1963)
nhieu = rng.normal(size=m)
nhieu /= np.sqrt((nhieu ** 2).sum() * h1)         # ‖nhiễu‖ = 1 theo chuẩn L2 rời rạc tr.501
print("   (a) δ → 0 với α = δ²/γ, γ = 1 (tức γ1 = γ2 = 1): sai số đều max|z − z̄|")
print("   %10s %12s %16s" % ("δ", "α", "max|z − z̄|"))
sai_theo_delta = []
for delta in (1e-1, 1e-2, 1e-3, 1e-4, 1e-5):
    ud = ub + delta * nhieu
    alpha = delta ** 2
    za = he_3phay(K, ud, h, h1, alpha, kk, pp)
    e = np.abs(za - zb).max()
    sai_theo_delta.append(e)
    print("   %10.0e %12.0e %16.6f" % (delta, alpha, e))
print("   giảm đơn điệu theo δ: %s" % all(np.diff(sai_theo_delta) < 0))

print("\n   (b) δ = 1e-3 CỐ ĐỊNH, α → 0: nghiệm có rời khỏi ε-lân cận của z̄ không (tr.503)")
delta = 1e-3
ud = ub + delta * nhieu
print("   %10s %16s %12s" % ("α", "max|z − z̄|", "δ²/α"))
bang_b = []
for alpha in [10.0 ** e for e in range(-2, -15, -1)]:
    za = he_3phay(K, ud, h, h1, alpha, kk, pp)
    e = np.abs(za - zb).max()
    bang_b.append((alpha, e))
    print("   %10.0e %16.6f %12.3g" % (alpha, e, delta ** 2 / alpha))
a_tot, e_tot = min(bang_b, key=lambda t: t[1])
print("   tốt nhất ở α = %.0e (max|z − z̄| = %.6f); ở α = 1e-14 sai số gấp %.1f lần mức đó."
      % (a_tot, e_tot, bang_b[-1][1] / e_tot))
z_ls = np.linalg.lstsq(K * h, ud, rcond=None)[0]
print("   α = 0 (bình phương tối thiểu trần, lstsq): max|z − z̄| = %.3e" % np.abs(z_ls - zb).max())
print("   biên độ nghiệm thật max|z̄| = %.6f ; log10(sai số / biên độ) = %.4f"
      % (np.abs(zb).max(), np.log10(np.abs(z_ls - zb).max() / np.abs(zb).max())))
Kb_ = K.T @ K * h1 * h
print("   số điều kiện của Σ K̄ h (ma trận của bình phương tối thiểu trần) = %.3e" % np.linalg.cond(Kb_))

# ═════════════════════════════════════════════════════ 5. Dạng tích phân tr.504: dấu trước ∫p
head("5. Tikhonov tr.504 — dạng tích phân của (3): in  α{k z′ + ∫ p z}; kiểm cả hai dấu")
alpha = 1e-4
za = he_3phay(K, ud, h, h1, alpha, kk, pp)
Kb = K.T @ K * h1
bb = K.T @ ud * h1
Az = K @ za * h
R_cong, R_tru = [], []
for j in range(1, n):                              # biên phải của ô thứ j, t_j = a + j h
    # dạng tổng dồn chính xác của (3′): lấy tổng j' = 1..j rồi nhân h
    tp = h * (pp[:j] * za[:j]).sum()
    alpha_xs = K[:, :j].sum(1) * h               # α(x_i, t_j) = A[x, η_s]
    tA = (Az * alpha_xs).sum() * h1
    tU = (ud * alpha_xs).sum() * h1
    kzp = kk[j] * (za[j] - za[j - 1]) / h        # k_j (ẑ_{j+1} − ẑ_j)/h ở chỉ số bài báo
    R_cong.append(alpha * (kzp + tp) - (tA - tU))
    R_tru.append(alpha * (kzp - tp) - (tA - tU))
R_cong, R_tru = np.abs(R_cong), np.abs(R_tru)
print("   phần dư lớn nhất với dấu + (như in) : %.3e" % R_cong.max())
print("   phần dư lớn nhất với dấu − (tích phân (3) từ a tới s): %.3e" % R_tru.max())
print("   so với biên độ số hạng α·k z′ : %.3e" % (alpha * np.abs(np.diff(za) / h).max()))
print("   -> dấu − làm phương trình đúng tới sai số máy; dấu + để lại đúng 2α∫p z.")
print("   lệch giữa phần dư dấu + và 2α∫p z: %.3e"
      % np.abs(R_cong - np.abs([2 * alpha * h * (pp[:j] * za[:j]).sum() for j in range(1, n)])).max())

# ═════════════════════════════════════════════════════ 6. Ridge = trường hợp k = 0, p = 1
head("6. Từ Tikhonov sang ridge: k(s) = 0, p(s) = 1 cho đúng (XᵀX + λI)β = Xᵀy")
alpha = 1e-4
k0 = np.zeros(n + 1)
z_t = he_3phay(K, ud, h, h1, alpha, k0, pp)
X = K * h * np.sqrt(h1)
y = ud * np.sqrt(h1)
lam = alpha * h
z_r = np.linalg.solve(X.T @ X + lam * np.eye(n), X.T @ y)
print("   X = K·h·√h1 (%d×%d),  y = û·√h1,  λ = α·h = %.3e" % (m, n, lam))
print("   hệ (3′) với k = 0 so với công thức ridge: lệch lớn nhất %.3e (biên độ %.4f)"
      % (np.abs(z_t - z_r).max(), np.abs(z_r).max()))
try:
    from sklearn.linear_model import Ridge
    z_s = Ridge(alpha=lam, fit_intercept=False, solver="cholesky").fit(X, y).coef_
    print("   so với sklearn Ridge(alpha=λ, fit_intercept=False): lệch lớn nhất %.3e"
          % np.abs(z_s - z_r).max())
except ImportError:
    print("   (máy không có sklearn — bỏ phép so thứ ba)")
print("\n   Cùng δ = 1e-3, cùng lưới α: sai số đều khi CÓ và KHÔNG có số hạng đạo hàm k z′²")
print("   %10s %18s %18s" % ("α", "k = 1 (Tikhonov)", "k = 0 (ridge)"))
tot1, tot0 = [], []
for alpha in [10.0 ** e for e in range(-2, -11, -1)]:
    e1 = np.abs(he_3phay(K, ud, h, h1, alpha, kk, pp) - zb).max()
    e0 = np.abs(he_3phay(K, ud, h, h1, alpha, k0, pp) - zb).max()
    tot1.append(e1); tot0.append(e0)
    print("   %10.0e %18.6f %18.6f" % (alpha, e1, e0))
print("   tốt nhất: k = 1 -> %.6f ;  k = 0 -> %.6f ;  tỉ số %.4f"
      % (min(tot1), min(tot0), min(tot0) / min(tot1)))


# ═══════════════════════════════════════════════════════════════════════════════════════════
#  PHẦN II — Hoerl & Kennard 1970, Technometrics 12(1):55–67
#  AI algorithms/_papers/1970 - Hoerl, Kennard - Ridge Regression Biased Estimation for Nonorthogonal Problems.pdf
#  (bản quét JSTOR 13 trang, không có lớp text). Bài báo KHÔNG có ví dụ số: Hình 1 tr.61 được
#  chính tác giả gọi là "in qualitative form" (tr.60). Nên phần này kiểm từng công thức đánh số
#  trên một thiết kế cụ thể, và dựng lại những khẳng định bài báo nêu mà không đo.
# ═══════════════════════════════════════════════════════════════════════════════════════════

def thiet_ke_tuong_quan(n=50, p=8, rho=0.99, seed=3):
    """Thiết kế gần suy biến; X'X đưa về DẠNG TƯƠNG QUAN như tr.56 yêu cầu."""
    r = np.random.default_rng(seed)
    Z = r.normal(size=(n, 2))
    X = np.zeros((n, p))
    for j in range(p):
        X[:, j] = rho * Z[:, j % 2] + np.sqrt(1 - rho ** 2) * r.normal(size=n)
    X = X - X.mean(0)
    X = X / np.sqrt((X ** 2).sum(0))          # mỗi cột độ dài 1 -> X'X là ma trận tương quan
    return X


def gamma12(lam_, alpha_, sigma2, k):
    """(4.5) và (4.11): γ1 = σ²Σλ/(λ+k)², γ2 = k²Σα²/(λ+k)²."""
    return (sigma2 * (lam_ / (lam_ + k) ** 2).sum(),
            k * k * (alpha_ ** 2 / (lam_ + k) ** 2).sum())


head("7. Hoerl & Kennard tr.56–60 — kiểm từng công thức (1.x), (2.x), (3.x), (4.x) trên một thiết kế")
X = thiet_ke_tuong_quan()
n, p = X.shape
XtX = X.T @ X
lam_, Pt = np.linalg.eigh(XtX)                 # XtX = Pt diag(lam) Pt'
thu_tu = np.argsort(lam_)[::-1]
lam_, Pt = lam_[thu_tu], Pt[:, thu_tu]
P = Pt.T                                       # quy ước bài báo: X'X = P'ΛP
beta = np.array([3.0, -2.0, 0, 0, 0, 0, 0, 0])
sigma2 = 1.0
alpha_ = P @ beta                              # (4.12)
print("   n = %d, p = %d, X'X dạng tương quan (đường chéo = %s)" % (n, p, np.round(np.diag(XtX), 12)[:3]))
print("   trị riêng λ_i của X'X: %s" % np.array2string(lam_, precision=6))
print("   λ_max/λ_min = %.4e" % (lam_[0] / lam_[-1]))
rng = np.random.default_rng(1970)
y = X @ beta + np.sqrt(sigma2) * rng.normal(size=n)
k = 0.05
bhat = np.linalg.solve(XtX, X.T @ y)                               # (1.1)
W = np.linalg.inv(XtX + k * np.eye(p))
bstar = W @ X.T @ y                                                # (2.1)–(2.2)
Z = np.linalg.inv(np.eye(p) + k * np.linalg.inv(XtX))              # (2.3)
kiem = []
kiem.append(("(2.3)  β* = [I + k(X'X)⁻¹]⁻¹ β̂", np.abs(Z @ bhat - bstar).max()))
kiem.append(("(2.5)  ξ(W) = 1/(λ+k)", np.abs(np.sort(np.linalg.eigvalsh(W)) - np.sort(1 / (lam_ + k))).max()))
kiem.append(("(2.6)  ξ(Z) = λ/(λ+k)", np.abs(np.sort(np.linalg.eigvals(Z).real) - np.sort(lam_ / (lam_ + k))).max()))
kiem.append(("(2.7)  Z = I − kW", np.abs(Z - (np.eye(p) - k * W)).max()))
phi_star = ((y - X @ bstar) ** 2).sum()
kiem.append(("(2.11) φ*(k) = Y'Y − β*'X'Y − kβ*'β*", abs(phi_star - (y @ y - bstar @ X.T @ y - k * bstar @ bstar))))
phi_min = ((y - X @ bhat) ** 2).sum()
B = rng.normal(size=p)
kiem.append(("(3.1)  φ(B) = φ_min + (B−β̂)'X'X(B−β̂)",
             abs(((y - X @ B) ** 2).sum() - phi_min - (B - bhat) @ XtX @ (B - bhat))))
kiem.append(("(3.6)  φ*(k) = φ_min + k²β*'(X'X)⁻¹β*", abs(phi_star - phi_min - k * k * bstar @ np.linalg.solve(XtX, bstar))))
# (3.4)–(3.5): điều kiện dừng của Lagrange cho đúng β*
grad = 2 * bstar + (1 / k) * (2 * XtX @ bstar - 2 * XtX @ bhat)
kiem.append(("(3.4)  ∂F/∂B tại B = β* bằng 0", np.abs(grad).max()))
# (4.3)–(4.5)
Zb = Z @ beta - beta
g1a = sigma2 * np.trace(np.linalg.inv(XtX) @ Z.T @ Z)
g1b = sigma2 * (np.trace(W) - k * np.trace(W @ W))
g1, g2 = gamma12(lam_, alpha_, sigma2, k)
kiem.append(("(4.3)=(4.4) phần phương sai", abs(g1a - g1b)))
kiem.append(("(4.4)=(4.5) phần phương sai", abs(g1b - g1)))
kiem.append(("(4.3)=(4.5) phần chệch², và (4.11)", abs(Zb @ Zb - g2)))
kiem.append(("(4.8)  tr σ²Z(X'X)⁻¹Z' = γ1", abs(np.trace(sigma2 * Z @ np.linalg.inv(XtX) @ Z.T) - g1)))
kiem.append(("(4.13) γ2 = Σα²/[1+λ/k]²", abs((alpha_ ** 2 / (1 + lam_ / k) ** 2).sum() - g2)))
eps = 1e-6
d1 = (gamma12(lam_, alpha_, sigma2, k + eps)[0] - gamma12(lam_, alpha_, sigma2, k - eps)[0]) / (2 * eps)
d2 = (gamma12(lam_, alpha_, sigma2, k + eps)[1] - gamma12(lam_, alpha_, sigma2, k - eps)[1]) / (2 * eps)
kiem.append(("(4.14) dγ2/dk = 2kΣλα²/(λ+k)³ (tương đối)", abs(d2 - 2 * k * (lam_ * alpha_ ** 2 / (lam_ + k) ** 3).sum()) / abs(d2)))
kiem.append(("(4.15) dγ1/dk = −2σ²Σλ/(λ+k)³ (tương đối)", abs(d1 + 2 * sigma2 * (lam_ / (lam_ + k) ** 3).sum()) / abs(d1)))
for ten, v in kiem:
    print("   %-46s lệch %.3e" % (ten, v))
    assert v < 1e-6, ten
e0 = 1e-9
d1_0 = (gamma12(lam_, alpha_, sigma2, e0)[0] - gamma12(lam_, alpha_, sigma2, 0)[0]) / e0
print("   (4.9)  dγ1/dk tại k→0⁺: số %.6e · công thức −2σ²Σ1/λ² = %.6e"
      % (d1_0, -2 * sigma2 * (1 / lam_ ** 2).sum()))
print("   (4.9)  với X'X = I (trực giao), p = %d: −2pσ² = %.1f" % (p, -2 * p * sigma2))
print("   (2.8)  ‖β*‖² = %.6f < ‖β̂‖² = %.6f ; (2.9) cận ξ²max(Z)‖β̂‖² = %.6f"
      % (bstar @ bstar, bhat @ bhat, (lam_[0] / (lam_[0] + k)) ** 2 * (bhat @ bhat)))
print("   độ dài bình phương co %.4f lần ; k = %g nhỏ hơn λ_max %.4f lần"
      % ((bhat @ bhat) / (bstar @ bstar), k, lam_[0] / k))

print("\n   Hai chỗ in không khớp chiều ma trận:")
try:
    v = (B - bhat)[None, :] @ XtX @ (B - bhat)[None, :]
    print("   (3.3) như in chạy được?! shape", v.shape)
except ValueError as e:
    print("   (3.3) in (B − β̂)'X'X(B − β̂)′ — dấu phẩy thứ hai làm phép nhân gãy: %s" % str(e).split(" (")[0][:70])
Kd = np.diag(np.linspace(0.01, 0.2, p))
Lam = np.diag(lam_)
Wst = np.linalg.inv(Lam + Kd)
Z_in = np.eye(p) + np.linalg.inv(Lam) @ Kd                    # bảng ký hiệu tr.66 in: {I + [(X*)'(X*)]⁻¹K]}
Z_dung = np.linalg.inv(Z_in)
print("   Bảng ký hiệu tr.66: Z* = {I + [(X*)'(X*)]⁻¹K} (như in) so với I − KW*: lệch %.3e"
      % np.abs(Z_in - (np.eye(p) - Kd @ Wst)).max())
print("   Thêm nghịch đảo ngoài cùng, {I + [(X*)'(X*)]⁻¹K}⁻¹ so với I − KW*:       lệch %.3e"
      % np.abs(Z_dung - (np.eye(p) - Kd @ Wst)).max())

# (1.5)–(1.9) bằng Monte Carlo
REP = 40000
L2 = np.empty(REP)
bs_mc = np.empty((REP, p))
for i in range(REP):
    yy = X @ beta + rng.normal(size=n)
    b_ = np.linalg.solve(XtX, X.T @ yy)
    bs_mc[i] = b_
    L2[i] = ((b_ - beta) ** 2).sum()
print("\n   (1.8) E[L1²] = σ²Σ1/λ = %.4f ; Monte Carlo %d lần: %.4f" % ((1 / lam_).sum(), REP, L2.mean()))
print("   (1.9) VAR[L1²] = 2σ⁴Σ(1/λ)² = %.4f ; Monte Carlo: %.4f ; tỉ số %.4f"
      % (2 * (1 / lam_ ** 2).sum(), L2.var(), L2.var() / (2 * (1 / lam_ ** 2).sum())))
print("   cận dưới σ²/λ_min = %.4f ; 2σ⁴/λ²_min = %.4f" % (1 / lam_[-1], 2 / lam_[-1] ** 2))
print("   phần của trị riêng nhỏ nhất trong E[L1²]: %.4f / %.4f = %.2f%% ; sáu trị riêng nhỏ nhất: %.2f%%"
      % (1 / lam_[-1], (1 / lam_).sum(), 100 / lam_[-1] / (1 / lam_).sum(),
         100 * (1 / lam_[2:]).sum() / (1 / lam_).sum()))
print("   (1.5a) E[β̂'β̂] = β'β + σ²tr(X'X)⁻¹ = %.4f ; Monte Carlo: %.4f"
      % (beta @ beta + (1 / lam_).sum(), (bs_mc ** 2).sum(1).mean()))
print("   E[β̂'β̂] / β'β = %.4f lần" % ((beta @ beta + (1 / lam_).sum()) / (beta @ beta)))

head("8. Hoerl & Kennard tr.62–63 — định lý 4.3 và cận (4.16) k < σ²/α²max: cận đó chặt tới đâu")
kk_ = np.r_[0, np.logspace(-6, 2, 4001)]
mse = np.array([sum(gamma12(lam_, alpha_, sigma2, kv)) for kv in kk_])
i_best = int(np.argmin(mse))
thang = kk_[mse < mse[0]]
can = sigma2 / (alpha_ ** 2).max()
print("   E[L1²(0)] = σ²Σ1/λ = %.6f" % mse[0])
print("   k tối ưu trên lưới = %.6g, E[L1²] = %.6f (giảm %.4f lần)" % (kk_[i_best], mse[i_best], mse[0] / mse[i_best]))
print("   mọi k trên lưới (tới %.6g, đầu lưới) đều thắng OLS: %s ; lý do: giới hạn k → ∞ là β'β = %.4f < %.4f"
      % (thang.max(), bool((mse[1:] < mse[0]).all()), beta @ beta, mse[0]))
print("   cận đủ (4.16): σ²/α²max = %.6g" % can)
print("   k tối ưu / cận (4.16) = %.4f" % (kk_[i_best] / can))
dd = np.array([(sum(gamma12(lam_, alpha_, sigma2, kv + 1e-9)) - sum(gamma12(lam_, alpha_, sigma2, kv))) / 1e-9
               for kv in np.linspace(1e-6, can, 200)])
print("   dE/dk < 0 tại mọi điểm của (0, σ²/α²max] (200 điểm): %s" % bool((dd < 0).all()))

print("\n   Thiết kế của Hình 2 trong bài (cùng hàm sinh với khối mã in, seed 3, KHÔNG chuẩn hoá):")
r3 = np.random.default_rng(3)
Z3 = r3.normal(size=(50, 2))
X3 = np.zeros((50, 8))
for j in range(8):
    X3[:, j] = 0.99 * Z3[:, j % 2] + np.sqrt(1 - 0.99 ** 2) * r3.normal(size=50)
b3 = np.zeros(8); b3[:2] = [3.0, -2.0]
l3, V3 = np.linalg.eigh(X3.T @ X3)
a3 = V3.T @ b3
kk3 = np.r_[0, np.logspace(-4, 3, 200001)]
m3 = np.array([sum(gamma12(l3, a3, 1.0, kv)) for kv in kk3])
i3 = int(np.argmin(m3))
print("   E[L1²(0)] giải tích = %.4f ; k* giải tích = %.4f ; E[L1²(k*)] = %.4f ; giảm %.4f lần"
      % (m3[0], kk3[i3], m3[i3], m3[0] / m3[i3]))
g1s, g2s = gamma12(l3, a3, 1.0, kk3[i3])
print("   tại k*: γ1 (phương sai) = %.4f ; γ2 (chệch²) = %.4f" % (g1s, g2s))
print("   khoảng thắng OLS giải tích: (0, %.4f]" % kk3[m3 < m3[0]].max())
print("   cận (4.16) σ²/α²max = %.4f" % (1.0 / (a3 ** 2).max()))

head("9. Hoerl & Kennard tr.63 mục 4c — 'cực tiểu dời về k = 0 khi β'β tăng'")
print("   %10s %14s %14s" % ("β'β", "k tối ưu", "giảm MSE (lần)"))
kt_prev = np.inf
don_dieu = True
for c in (0.1, 0.3, 1, 3, 10, 30, 100):
    a_c = np.sqrt(c) * alpha_
    m_c = np.array([sum(gamma12(lam_, a_c, sigma2, kv)) for kv in kk_])
    ib = int(np.argmin(m_c))
    don_dieu &= kk_[ib] < kt_prev
    kt_prev = kk_[ib]
    print("   %10.4g %14.6g %14.4f" % (c * (beta @ beta), kk_[ib], m_c[0] / m_c[ib]))
print("   k tối ưu giảm đơn điệu khi β'β tăng: %s" % don_dieu)
kt_dau = kk_[int(np.argmin([sum(gamma12(lam_, np.sqrt(0.1) * alpha_, sigma2, kv)) for kv in kk_]))]
kt_cuoi = kk_[int(np.argmin([sum(gamma12(lam_, np.sqrt(100) * alpha_, sigma2, kv)) for kv in kk_]))]
print("   β'β tăng %.0f lần -> k tối ưu giảm %.1f lần" % (1000, kt_dau / kt_cuoi))

head("10. Hoerl & Kennard tr.63 mục 5 — k_i tối ưu = σ²/α_i², KHÔNG phụ thuộc λ_i")
for li, ai in [(0.01, 1.0), (1.0, 1.0), (100.0, 1.0), (1.0, 0.3), (1.0, 3.0)]:
    kg = np.logspace(-4, 3, 700001)
    mi = sigma2 * li / (li + kg) ** 2 + kg ** 2 * ai ** 2 / (li + kg) ** 2
    print("   λ_i = %-6g α_i = %-4g: argmin trên lưới k = %.5f ; σ²/α_i² = %.5f"
          % (li, ai, kg[np.argmin(mi)], sigma2 / ai ** 2))
ki = sigma2 / alpha_ ** 2
mse_tq = (sigma2 * lam_ / (lam_ + ki) ** 2 + ki ** 2 * alpha_ ** 2 / (lam_ + ki) ** 2).sum()
print("   Trên thiết kế mục 7: E[L1²] với k_i = σ²/α_i² riêng từng hướng = %.6f ;" % mse_tq)
print("   với MỘT k chung tốt nhất = %.6f ; OLS = %.6f" % (mse[i_best], mse[0]))
print("   (các hướng có α_i = 0 cho k_i = ∞: hệ số hướng đó bị đưa hẳn về 0)")

head("11. Hoerl & Kennard tr.65 mục 7 — vòng lặp k̂_i = σ̂²/(α̂*_i)² có điểm dừng không")
print("   Điểm cố định k = σ²(λ+k)²/(λ²α̂²) là nghiệm của σ²k² + (2σ²λ − λ²α̂²)k + σ²λ² = 0,")
print("   có nghiệm thực khi và chỉ khi λα̂²/σ² ≥ 4, tức |t| = |α̂|/(σ/√λ) ≥ 2.")
print("   %8s %8s %8s %8s %14s %16s %s" % ("λ", "α̂", "t²", "vòng", "k cuối", "α̂* cuối", "kết cục"))
for li, ah in [(1.0, 3.0), (1.0, 2.2), (1.0, 2.0), (1.0, 1.9), (1.0, 1.0), (0.05, 5.0), (0.05, 12.0)]:
    kv = sigma2 / ah ** 2
    for it in range(1, 100001):
        a_star = li * ah / (li + kv)
        k_new = sigma2 / a_star ** 2
        if abs(k_new - kv) < 1e-12 * max(1.0, kv) or k_new > 1e12:
            kv = k_new
            break
        kv = k_new
    t2 = li * ah ** 2 / sigma2
    ket = "dừng" if kv < 1e12 else "PHÂN KỲ (k → ∞)"
    print("   %8g %8g %8.3f %8d %14.6g %16.6g %s" % (li, ah, t2, it, kv, li * ah / (li + kv), ket))
t2 = lam_ * (P @ bhat) ** 2 / sigma2
print("   Trên thiết kế mục 7, một lần rút y: t² của %d hướng = %s" % (p, np.array2string(t2, precision=3)))
print("   số hướng có t² < 4 (vòng lặp phân kỳ): %d/%d" % ((t2 < 4).sum(), p))
a_hat_mc = bs_mc @ P.T
print("   'k̂ quá nhỏ vì α'α bị ước lượng quá': E[α̂'α̂] Monte Carlo %.4f so với α'α = %.4f + σ²Σ1/λ = %.4f"
      % ((a_hat_mc ** 2).sum(1).mean(), alpha_ @ alpha_, alpha_ @ alpha_ + (1 / lam_).sum()))

head("12. Hoerl & Kennard tr.64 mục 6 — Bayes: tiên nghiệm N(0, σ²/k·I) cho trung bình hậu nghiệm = β*")
k = 0.3
tien = (sigma2 / k) * np.eye(p)
hau_cov = np.linalg.inv(XtX / sigma2 + np.linalg.inv(tien))
hau_mean = hau_cov @ X.T @ y / sigma2
print("   trung bình hậu nghiệm so với [X'X + kI]⁻¹X'Y: lệch %.3e"
      % np.abs(hau_mean - np.linalg.solve(XtX + k * np.eye(p), X.T @ y)).max())
Rb = np.linalg.solve(XtX + k * np.eye(p), X.T @ y)
print("   cách nhìn ràng buộc: R² = ‖β*‖² = %.6f ; cực tiểu RSS trên mặt cầu bán kính R bằng Lagrange"
      % (Rb @ Rb))
# kiểm: mọi B trên mặt cầu có RSS ≥ RSS(β*)
tot = np.inf
for _ in range(20000):
    u_ = rng.normal(size=p)
    u_ *= np.sqrt(Rb @ Rb) / np.linalg.norm(u_)
    tot = min(tot, ((y - X @ u_) ** 2).sum())
print("   RSS(β*) = %.6f ; nhỏ nhất trên 20 000 điểm ngẫu nhiên cùng mặt cầu = %.6f"
      % (((y - X @ Rb) ** 2).sum(), tot))


# ═══════════════════════════════════════════════════════════════════════════════════════════
#  PHẦN III — Hastie, Tibshirani & Friedman, ESL 2nd ed. (bản in 12), mục 3.4.1, tr.61–68
#  AI algorithms/_papers/2009 - Hastie, Tibshirani, Friedman - The Elements of Statistical Learning (2nd ed, print 12).pdf
#  Dữ liệu prostate: tools/prostate.csv, trích từ gói R ElemStatLearn 2015.6.26.2 (CRAN archive),
#  bản đã sửa lỗi dữ liệu mà chú thích tr.3 của sách nhắc tới.
# ═══════════════════════════════════════════════════════════════════════════════════════════
import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
rows = list(csv.DictReader(open(os.path.join(HERE, "prostate.csv"), encoding="utf-8")))
TEN = ["lcavol", "lweight", "age", "lbph", "svi", "lcp", "gleason", "pgg45"]
Xall = np.array([[float(r[t]) for t in TEN] for r in rows])
yall = np.array([float(r["lpsa"]) for r in rows])
tr = np.array([r["train"] == "True" for r in rows])

head("13. ESL tr.49–51 và Bảng 3.3 tr.63 — dựng lại cột LS: chuẩn hoá theo quy ước nào")
IN_LS = np.array([2.465, 0.680, 0.263, -0.141, 0.210, 0.305, -0.288, -0.021, 0.267])
IN_RIDGE = np.array([2.452, 0.420, 0.238, -0.046, 0.162, 0.227, 0.000, 0.040, 0.133])
print("   %d mẫu, %d huấn luyện, %d kiểm tra" % (len(yall), tr.sum(), (~tr).sum()))


def chuan_hoa(cach):
    if cach == "toàn bộ 97, ddof=0":
        m, s = Xall.mean(0), Xall.std(0)
    elif cach == "toàn bộ 97, ddof=1":
        m, s = Xall.mean(0), Xall.std(0, ddof=1)
    elif cach == "67 huấn luyện, ddof=0":
        m, s = Xall[tr].mean(0), Xall[tr].std(0)
    else:
        m, s = Xall[tr].mean(0), Xall[tr].std(0, ddof=1)
    return (Xall - m) / s


def ls_fit(Xs):
    A = np.c_[np.ones(tr.sum()), Xs[tr]]
    return np.linalg.lstsq(A, yall[tr], rcond=None)[0]


CACH = ["toàn bộ 97, ddof=0", "toàn bộ 97, ddof=1", "67 huấn luyện, ddof=0", "67 huấn luyện, ddof=1"]
tot_cach, tot_lech = None, np.inf
for cach in CACH:
    b = ls_fit(chuan_hoa(cach))
    lech = np.abs(np.round(b, 3) - IN_LS).max()
    print("   %-22s LS = %s  lệch làm tròn lớn nhất %.3f" % (cach, np.array2string(b, precision=4), lech))
    if lech < tot_lech:
        tot_cach, tot_lech = cach, lech
print("   quy ước khớp cột LS in: %s" % tot_cach)
Xs = chuan_hoa(tot_cach)
b_ls = ls_fit(Xs)
pred = b_ls[0] + Xs[~tr] @ b_ls[1:]
e2 = (yall[~tr] - pred) ** 2
print("   sai số kiểm tra LS = %.6f (in 0.521) ; sai số chuẩn = %.6f (in 0.179)"
      % (e2.mean(), e2.std(ddof=1) / np.sqrt(len(e2))))
print("   mốc 'base error rate' (dự đoán bằng trung bình huấn luyện) = %.6f (in 1.057 tr.51)"
      % ((yall[~tr] - yall[tr].mean()) ** 2).mean())

head("14. ESL Bảng 3.3 cột Ridge và Hình 3.8 — tìm λ cho df(λ) = 5,0 bằng (3.50), rồi so từng hệ số")
Xc = Xs[tr] - Xs[tr].mean(0)
yc = yall[tr] - yall[tr].mean()
U, d, Vt = np.linalg.svd(Xc, full_matrices=False)
print("   giá trị kỳ dị d_j của X đã center (67×8): %s" % np.array2string(d, precision=4))


def df_lam(lv):
    return (d ** 2 / (d ** 2 + lv)).sum()


lo, hi = 0.0, 1e4
for _ in range(200):
    mid = (lo + hi) / 2
    if df_lam(mid) > 5.0:
        lo = mid
    else:
        hi = mid
lam5 = (lo + hi) / 2
b_r = Vt.T @ ((d / (d ** 2 + lam5)) * (U.T @ yc))
b0 = yall[tr].mean() - Xs[tr].mean(0) @ b_r
b_r_full = np.r_[b0, b_r]
print("   λ cho df = 5,0: %.6f (df kiểm lại %.10f)" % (lam5, df_lam(lam5)))
b_44 = np.linalg.solve(Xc.T @ Xc + lam5 * np.eye(8), Xc.T @ yc)
print("   (3.44) nghiệm đóng so với dạng SVD (3.47): lệch %.3e" % np.abs(b_44 - b_r).max())
print("   %-10s %9s %9s %9s %8s" % ("hệ số", "dựng lại", "in", "làm tròn", "khớp?"))
khop_r = 0
for t, v, iv in zip(["Intercept"] + TEN, b_r_full, IN_RIDGE):
    ok = abs(round(v, 3) - iv) < 5e-4
    khop_r += ok
    print("   %-10s %9.5f %9.3f %9.3f %8s" % (t, v, iv, round(v, 3), ok))
print("   khớp %d/9 ô sau khi làm tròn 3 chữ số" % khop_r)
pred_r = b_r_full[0] + Xs[~tr] @ b_r_full[1:]
e2r = (yall[~tr] - pred_r) ** 2
print("   sai số kiểm tra Ridge = %.6f (in 0.492) ; sai số chuẩn = %.6f (in 0.165)"
      % (e2r.mean(), e2r.std(ddof=1) / np.sqrt(len(e2r))))
# quét df quanh 5 để xem cột in nhạy tới đâu
print("   quét df quanh 5,0 — số ô khớp cột in:")
for dfv in (4.8, 4.9, 4.95, 5.0, 5.05, 5.1, 5.2):
    lo, hi = 0.0, 1e4
    for _ in range(200):
        mid = (lo + hi) / 2
        if df_lam(mid) > dfv:
            lo = mid
        else:
            hi = mid
    lv = (lo + hi) / 2
    bb_ = Vt.T @ ((d / (d ** 2 + lv)) * (U.T @ yc))
    full = np.r_[yall[tr].mean() - Xs[tr].mean(0) @ bb_, bb_]
    print("      df = %-5g λ = %9.5f  khớp %d/9  lcp = %+.5f" % (dfv, lv, (np.abs(np.round(full, 3) - IN_RIDGE) < 5e-4).sum(), full[6]))
print("\n   Hệ số chặn in 2.452 bằng đúng ȳ của 67 mẫu huấn luyện = %.6f, KHÔNG bằng hệ số chặn LS 2.465."
      % yall[tr].mean())
print("   Thử lại cột Ridge với các quy ước chuẩn hoá khác (df = 5,0):")
print("   %-34s %8s %s" % ("quy ước (trung bình · độ lệch chuẩn)", "khớp/9", "hệ số dựng lại"))
bien_the = {
    "TB 97 · SD 97 ddof=1 (như cột LS)": (Xall.mean(0), Xall.std(0, ddof=1)),
    "TB 67 · SD 97 ddof=1": (Xall[tr].mean(0), Xall.std(0, ddof=1)),
    "TB 67 · SD 67 ddof=1": (Xall[tr].mean(0), Xall[tr].std(0, ddof=1)),
    "TB 67 · SD 67 ddof=0": (Xall[tr].mean(0), Xall[tr].std(0)),
}
for ten_bt, (m_, s_) in bien_the.items():
    Xb = (Xall - m_) / s_
    Xbc = Xb[tr] - Xb[tr].mean(0)
    Ub, db, Vbt = np.linalg.svd(Xbc, full_matrices=False)
    lv = lam_cho_df_tam = None
    lo, hi = 0.0, 1e4
    for _ in range(200):
        mid = (lo + hi) / 2
        if (db ** 2 / (db ** 2 + mid)).sum() > 5.0:
            lo = mid
        else:
            hi = mid
    lv = (lo + hi) / 2
    bb_ = Vbt.T @ ((db / (db ** 2 + lv)) * (Ub.T @ yc))
    full = np.r_[yall[tr].mean() - Xb[tr].mean(0) @ bb_, bb_]
    nk = (np.abs(np.round(full, 3) - IN_RIDGE) < 5e-4).sum()
    print("   %-34s %8d %s" % (ten_bt, nk, np.array2string(full, precision=4)))
    e2b = (yall[~tr] - (full[0] + Xb[~tr] @ full[1:])) ** 2
    print("   %-34s lệch lớn nhất 8 hệ số %.5f ; sai số kiểm tra %.6f ; sai số chuẩn %.6f"
          % ("", np.abs(full[1:] - IN_RIDGE[1:]).max(), e2b.mean(), e2b.std(ddof=1) / np.sqrt(30)))
print("   (3.47) hệ số co d²/(d²+λ) theo từng hướng: %s" % np.array2string(d ** 2 / (d ** 2 + lam5), precision=4))
# câu tr.68 'ridge regression reduces the test error of the full least squares estimates by a small amount'
dif = e2 - e2r
print("   LS − Ridge trên 30 mẫu kiểm tra: chênh trung bình %.6f ; sai số chuẩn của chênh (cặp) %.6f ; tỉ số %.4f"
      % (dif.mean(), dif.std(ddof=1) / np.sqrt(30), dif.mean() / (dif.std(ddof=1) / np.sqrt(30))))
print("   số mẫu kiểm tra mà Ridge tốt hơn LS: %d/30" % (dif > 0).sum())
print("   (tr.64) đầu vào trực chuẩn: β̂ridge = β̂/(1+λ)")
Q, _ = np.linalg.qr(rng.normal(size=(40, 5)))
yq = rng.normal(size=40)
print("      lệch với nghiệm (3.44), λ = 0,7: %.3e"
      % np.abs(np.linalg.solve(Q.T @ Q + 0.7 * np.eye(5), Q.T @ yq) - (Q.T @ yq) / 1.7).max())

head("15. ESL Hình 3.7 bảng Ridge tr.62 và câu tr.68 'the minimum occurs at df(λ) = 5.0'")
DF_LUOI = np.arange(0, 9)


def lam_cho_df(dv, dd):
    if dv >= len(dd):
        return 0.0
    if dv <= 0:
        return np.inf
    lo, hi = 0.0, 1e7
    for _ in range(200):
        mid = (lo + hi) / 2
        if (dd ** 2 / (dd ** 2 + mid)).sum() > dv:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


Xtr, ytr = Xs[tr], yall[tr]
lam_luoi = [lam_cho_df(dv, d) for dv in DF_LUOI]     # λ theo df trên toàn tập huấn luyện
chon_min, chon_1se = [], []
for seed in range(500):
    r = np.random.default_rng(seed)
    fold = r.permutation(np.arange(67) % 10)
    err = np.zeros((10, len(DF_LUOI)))
    for f in range(10):
        a_, b_ = fold != f, fold == f
        mx, my = Xtr[a_].mean(0), ytr[a_].mean()
        Xa = Xtr[a_] - mx
        for j, lv in enumerate(lam_luoi):
            if np.isinf(lv):
                bb_ = np.zeros(8)
            else:
                bb_ = np.linalg.solve(Xa.T @ Xa + lv * np.eye(8), Xa.T @ (ytr[a_] - my))
            err[f, j] = ((ytr[b_] - my - (Xtr[b_] - mx) @ bb_) ** 2).mean()
    cv = err.mean(0)
    se = err.std(0, ddof=1) / np.sqrt(10)
    jm = int(np.argmin(cv))
    chon_min.append(DF_LUOI[jm])
    chon_1se.append(DF_LUOI[int(np.nonzero(cv <= cv[jm] + se[jm])[0][0])])
    if seed == 0:
        print("   hạt giống 0: CV theo df 0..8 = %s" % np.array2string(cv, precision=4))
        print("                sai số chuẩn   = %s" % np.array2string(se, precision=4))
chon_min, chon_1se = np.array(chon_min), np.array(chon_1se)
print("   500 cách chia 10 mảnh ngẫu nhiên, lưới df = 0..8 (trục ngang Hình 3.7):")
print("   %6s %16s %18s" % ("df", "argmin CV (lần)", "quy tắc 1-SE (lần)"))
for dv in DF_LUOI:
    print("   %6d %16d %18d" % (dv, (chon_min == dv).sum(), (chon_1se == dv).sum()))


# ─────────────────────────────────────────────────── đọc đồ thị VECTƠ của ESL: toạ độ lấy từ PDF
def doc_vecto(so_trang):
    """Trả (các chấm tròn theo màu, các đường theo màu nét, hộp chữ) của một trang PDF vectơ."""
    import ctypes
    import pypdfium2 as pdfium
    import pypdfium2.raw as R
    duong = os.path.join(os.path.dirname(HERE), "AI algorithms", "_papers",
                         "2009 - Hastie, Tibshirani, Friedman - The Elements of Statistical Learning (2nd ed, print 12).pdf")
    doc = pdfium.PdfDocument(duong)
    pg = doc[so_trang - 1]
    vat = []

    def duyet(objs):
        for o in objs:
            if o.type == R.FPDF_PAGEOBJ_FORM:
                duyet(o.get_objects())
                continue
            f = [ctypes.c_uint() for _ in range(4)]
            s = [ctypes.c_uint() for _ in range(4)]
            R.FPDFPageObj_GetFillColor(o.raw, *f)
            R.FPDFPageObj_GetStrokeColor(o.raw, *s)
            b = [ctypes.c_float() for _ in range(4)]
            R.FPDFPageObj_GetBounds(o.raw, *b)
            vat.append((o.type, tuple(c.value for c in f[:3]), tuple(c.value for c in s[:3]),
                        tuple(c.value for c in b)))
    duyet(pg.get_objects())
    tp = pg.get_textpage()
    chu = []
    for i in range(tp.count_chars()):
        chu.append((tp.get_text_range(i, 1), tp.get_charbox(i)))
    return vat, chu


def nhan_truc(chu, nhan, truc, gioi):
    """Tìm toạ độ tâm của các nhãn số trên một trục trong vùng gioi = (x0, y0, x1, y1)."""
    s = "".join(c for c, _ in chu)
    kq = {}
    for nh in nhan:
        start = 0
        while True:
            i = s.find(nh, start)
            if i < 0:
                break
            start = i + 1
            boxes = [chu[j][1] for j in range(i, i + len(nh))]
            x0 = min(b[0] for b in boxes); x1 = max(b[2] for b in boxes)
            y0 = min(b[1] for b in boxes); y1 = max(b[3] for b in boxes)
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            if gioi[0] <= cx <= gioi[2] and gioi[1] <= cy <= gioi[3]:
                # nhãn phải đứng riêng: ký tự trước/sau không phải chữ số hay dấu chấm
                trc = s[i - 1] if i > 0 else " "
                sau = s[i + len(nh)] if i + len(nh) < len(s) else " "
                if (trc.isdigit() or trc in ".−-") or (sau.isdigit() or sau == "."):
                    continue
                kq.setdefault(nh, []).append(cx if truc == "x" else cy)
    return kq


head("16. ESL Hình 3.8 tr.65 và Hình 3.7 tr.62 — đọc toạ độ vectơ trong PDF, so với phép dựng lại")
vat, chu = doc_vecto(84)
cham = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for t, f, s_, b in vat
        if t == 1 and f == (0, 0, 255) and abs((b[2] - b[0]) - 2.5) < 0.3]
print("   Hình 3.8: %d chấm xanh (8 biến × 25 điểm = %d)" % (len(cham), 8 * 25))
x_nhan = nhan_truc(chu, ["0", "2", "4", "6", "8"], "x", (150, 200, 450, 290))
y_nhan = nhan_truc(chu, ["−0.2", "0.0", "0.2", "0.4", "0.6"], "y", (120, 280, 175, 560))
gx = np.polyfit([float(k_) for k_ in x_nhan], [np.mean(v) for v in x_nhan.values()], 1)
gy = np.polyfit([float(k_.replace("−", "-")) for k_ in y_nhan], [np.mean(v) for v in y_nhan.values()], 1)
print("   hiệu chuẩn trục: %d nhãn x, %d nhãn y" % (len(x_nhan), len(y_nhan)))
cham = np.array(cham)
df_c = (cham[:, 0] - gx[1]) / gx[0]
he_c = (cham[:, 1] - gy[1]) / gy[0]
cuoi = he_c[np.abs(df_c - df_c.max()) < 0.02]
print("   các chấm ở df lớn nhất (df = %.4f): %s" % (df_c.max(), np.array2string(np.sort(cuoi), precision=4)))
ls_toan = ls_fit(chuan_hoa("toàn bộ 97, ddof=1"))[1:]
ls_huan = ls_fit(chuan_hoa("67 huấn luyện, ddof=1"))[1:]
ls_huan0 = ls_fit(chuan_hoa("67 huấn luyện, ddof=0"))[1:]
# gán chấm cuối cho biến theo NHÃN CHỮ in cạnh đầu mút: xếp nhãn theo toạ độ dọc, xếp chấm theo giá trị
s84 = "".join(c_ for c_, _ in chu)
y_nhan_bien = []
import re as _re
for t_ in TEN:
    # nhãn đứng riêng một dòng; tránh khớp "age" bên trong "Shrinkage" ở tiêu đề trang
    mm = [m_ for m_ in _re.finditer(r"(?<![A-Za-z0-9])" + t_ + r"(?![A-Za-z0-9])", s84)]
    assert len(mm) == 1, (t_, len(mm))
    i_ = mm[0].start()
    y_nhan_bien.append(np.mean([chu[j_][1][1] for j_ in range(i_, i_ + len(t_))]))
thu_tu_nhan = np.argsort(y_nhan_bien)                  # từ thấp lên cao trên trang
cham_bien = np.empty(8)
cham_bien[thu_tu_nhan] = np.sort(cuoi)
print("   thứ tự nhãn từ dưới lên trên: %s" % [TEN[i_] for i_ in thu_tu_nhan])
print("   thứ tự LS 97 mẫu từ nhỏ tới lớn : %s" % [TEN[i_] for i_ in np.argsort(ls_toan)])
print("   thứ tự LS 67 mẫu từ nhỏ tới lớn : %s" % [TEN[i_] for i_ in np.argsort(ls_fit(chuan_hoa("67 huấn luyện, ddof=1"))[1:])])
print("   gán chấm cho biến theo thứ tự nhãn chữ trong hình:")
for t_, v_ in zip(TEN, cham_bien):
    print("      %-8s chấm %.4f" % (t_, v_))
print("   LS 97 mẫu ddof=1 : %s" % np.array2string(ls_toan, precision=4))
print("   LS 67 mẫu ddof=1 : %s" % np.array2string(ls_huan, precision=4))
for ten_q, v in [("chuẩn hoá trên 97 mẫu, ddof=1 (khớp Bảng 3.3)", ls_toan),
                 ("chuẩn hoá trên 67 mẫu huấn luyện, ddof=1", ls_huan),
                 ("chuẩn hoá trên 67 mẫu huấn luyện, ddof=0", ls_huan0)]:
    print("   LS %-46s: lệch lớn nhất so với chấm cuối = %.4f"
          % (ten_q, np.abs(np.sort(cuoi) - np.sort(v)).max()))
# các chấm của lcavol dọc theo đường, so với ridge trên cả hai quy ước
for ten_q, cach in [("97 mẫu ddof=1", "toàn bộ 97, ddof=1"), ("67 mẫu ddof=0", "67 huấn luyện, ddof=0"),
                    ("67 mẫu ddof=1", "67 huấn luyện, ddof=1")]:
    Xq = chuan_hoa(cach)
    Xqc = Xq[tr] - Xq[tr].mean(0)
    Uq, dq, Vqt = np.linalg.svd(Xqc, full_matrices=False)
    sai = []
    for xv, yv in zip(df_c, he_c):
        if xv < 0.05 or xv > 7.98:
            continue
        lv = lam_cho_df(xv, dq)
        bq = Vqt.T @ ((dq / (dq ** 2 + lv)) * (Uq.T @ yc))
        sai.append(np.abs(bq - yv).min())
    print("   chấm giữa đường so với đường ridge dựng lại (%s): %d chấm, lệch trung vị %.4f, lớn nhất %.4f"
          % (ten_q, len(sai), np.median(sai), np.max(sai)))

vat, chu = doc_vecto(81)
# bảng Ridge nằm ở nửa phải hàng trên: tìm theo tiêu đề "Ridge Regression"
s_all = "".join(c for c, _ in chu)
i_t = s_all.find("Ridge Regression")
bx = chu[i_t][1]
print("   Hình 3.7: tiêu đề 'Ridge Regression' ở x = %.1f, y = %.1f" % (bx[0], bx[1]))
# khung của bảng Ridge nằm dưới tiêu đề: x ∈ [tiêu đề − 40, tiêu đề + 95], y ∈ [tiêu đề − 110, tiêu đề − 5]
vung = (bx[0] - 40, bx[1] - 110, bx[0] + 95, bx[1] - 5)
cam = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for t, f, s_, b in vat
       if t == 1 and f == (230, 159, 0) and abs((b[2] - b[0]) - 3.0) < 0.3
       and vung[0] <= b[0] <= vung[2] and vung[1] <= b[1] <= vung[3]]
x_nhan = nhan_truc(chu, ["0", "2", "4", "6", "8"], "x", (vung[0], bx[1] - 125, vung[2], bx[1] - 110))
y_nhan = nhan_truc(chu, ["0.6", "0.8", "1.0", "1.2", "1.4", "1.6", "1.8"], "y", (bx[0] - 50, vung[1], bx[0] - 35, vung[3]))
gx = np.polyfit([float(k_) for k_ in x_nhan], [np.mean(v) for v in x_nhan.values()], 1)
gy = np.polyfit([float(k_) for k_ in y_nhan], [np.mean(v) for v in y_nhan.values()], 1)
cam = sorted(cam)
dfp = np.array([(c_[0] - gx[1]) / gx[0] for c_ in cam])
cvp = np.array([(c_[1] - gy[1]) / gy[0] for c_ in cam])
print("   %d chấm cam trong bảng Ridge; %d nhãn x, %d nhãn y" % (len(cam), len(x_nhan), len(y_nhan)))
print("   df đọc được : %s" % np.array2string(dfp, precision=3))
print("   CV đọc được : %s" % np.array2string(cvp, precision=4))
# thanh sai số vẽ gộp trong một đường nhiều đoạn nên không tách được theo hộp bao; thay vào đó dùng
# VẠCH TÍM NGANG — theo chú thích Hình 3.7, nó nằm ở mức "cực tiểu + 1 sai số chuẩn"
tim = [b for t, f, s_, b in vat if t == 2 and s_ == (160, 32, 240)
       and vung[0] <= b[0] <= vung[2] and vung[1] - 5 <= b[1] <= vung[3]]
muc_1se = None
for b in tim:
    if b[2] - b[0] < 1.5:
        print("   vạch tím ĐỨNG ở df = %.3f" % (((b[0] + b[2]) / 2 - gx[1]) / gx[0]))
    else:
        muc_1se = ((b[1] + b[3]) / 2 - gy[1]) / gy[0]
        print("   vạch tím NGANG ở CV = %.4f" % muc_1se)
jm = int(np.argmin(cvp))
print("   cực tiểu của đường in: df = %.3f, CV = %.4f ; vạch ngang − cực tiểu = %.4f (một SE)"
      % (dfp[jm], cvp[jm], muc_1se - cvp[jm]))
duoi = np.nonzero(cvp <= muc_1se)[0]
print("   các điểm có CV ≤ vạch ngang: df = %s ; nhỏ nhất: df = %.3f"
      % (np.array2string(dfp[duoi], precision=3), dfp[duoi[0]]))
print("   -> tr.68 viết 'the minimum occurs at df(λ) = 5.0'; theo chính Hình 3.7, cực tiểu ở df ≈ %.0f,"
      % dfp[jm])
print("      còn df = 5 là lựa chọn của quy tắc MỘT SAI SỐ CHUẨN (đúng như mục 3.3.4 tr.61 và chú thích).")

head("17. Hoerl & Kennard Hình 1 tr.61 — số hoá bản quét: đường nét đứt có bằng TỔNG hai đường liền không")
from PIL import Image  # noqa: E402

anh = os.path.join(os.path.dirname(HERE), "AI algorithms", "Machine Learning",
                   "Ridge Regression (Tikhonov Regularization)", "Untitled", "image9.png")
g = np.array(Image.open(anh).convert("L"))
toi = g < 110
H_, W_ = toi.shape


def cum(chi_so):
    """Gom các chỉ số liên tiếp thành cụm, trả tâm từng cụm."""
    if len(chi_so) == 0:
        return []
    kq, dau = [], chi_so[0]
    for a_, b_ in zip(chi_so[:-1], chi_so[1:]):
        if b_ != a_ + 1:
            kq.append((dau + a_) / 2)
            dau = b_
    kq.append((dau + chi_so[-1]) / 2)
    return kq


def cap_so_cong(tam, so_luong):
    """Chọn `so_luong` tâm tạo cấp số cộng tốt nhất; trả (các tâm được chọn, các tâm bị loại)."""
    from itertools import combinations
    tot, chon = np.inf, None
    for bo in combinations(range(len(tam)), len(tam) - so_luong):
        con = [t_ for i_, t_ in enumerate(tam) if i_ not in bo]
        he_ = np.polyfit(np.arange(so_luong), con, 1)
        du = np.abs(np.polyval(he_, np.arange(so_luong)) - con).max()
        if du < tot:
            tot, chon = du, (con, [tam[i_] for i_ in bo])
    return chon, tot


def dinh(tong, nguong, cach=150):
    """Các đỉnh của một biên dạng tổng điểm tối, cao hơn ngưỡng, cách nhau ít nhất `cach` px."""
    chon = []
    for i_ in np.argsort(tong)[::-1]:
        if tong[i_] < nguong:
            break
        if all(abs(i_ - c_) > cach for c_ in chon):
            chon.append(int(i_))
    return sorted(chon)


# cửa sổ khung đồ thị (tỉ lệ trang, đọc từ ảnh): bỏ tiêu đề, nhãn trục và chú thích bên dưới
Y0, Y1, X0, X1 = int(0.08 * H_), int(0.60 * H_), int(0.15 * W_), int(0.83 * W_)
tam_hang = [Y0 + i_ for i_ in dinh(toi[Y0:Y1, X0:X1].sum(1), 600, cach=25)]
tam_cot = [X0 - 300 + i_ for i_ in dinh(toi[Y0:Y1, X0 - 300:X1 + 100].sum(0), 600)]
print("   ảnh %d×%d; %d hàng kẻ dài, %d cột kẻ dài" % (W_, H_, len(tam_hang), len(tam_cot)))
(hang_luoi, hang_loai), du_h = cap_so_cong(tam_hang, 11)
print("   11 hàng lưới (0, 11, …, 110) khớp cấp số cộng, dư lớn nhất %.2f px; hàng dài bị loại: %s"
      % (du_h, [round(v, 1) for v in hang_loai]))
cot_luoi = sorted(tam_cot)
k_cot = np.arange(len(cot_luoi)) / 10.0
du_c = np.abs(np.polyval(np.polyfit(k_cot, cot_luoi, 1), k_cot) - cot_luoi).max()
print("   %d cột lưới (k = 0,0 … %.1f), dư lớn nhất %.2f px" % (len(cot_luoi), k_cot[-1], du_c))
ay = np.polyfit(sorted(hang_luoi)[::-1], np.arange(0, 121, 11), 1)
gia_tri_LS = [float(np.polyval(ay, v)) for v in hang_loai]
print("   hàng dài không thuộc lưới = đường 'Least Squares' ở giá trị %s (đơn vị σ²)"
      % [round(v, 2) for v in gia_tri_LS])
LS_v = gia_tri_LS[0]
sach = toi.copy()
for r_ in hang_luoi:
    sach[int(r_) - 6: int(r_) + 7, :] = False
print("   %6s %10s %10s %12s %12s %12s" % ("k", "phương sai", "chệch²", "nét đứt", "tổng 2 đường", "đứt − tổng"))
lech_tong = []
for kv in (0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.95):
    x0 = np.polyval(np.polyfit(k_cot, cot_luoi, 1), kv)
    tim_duoc = None
    for dx in range(0, 60, 3):
        for sgn in (1, -1):
            xc = int(round(x0 + sgn * dx))
            cot = sach[:, xc - 2: xc + 3].any(1)
            idx = np.nonzero(cot)[0]
            chay = []                                   # (tâm, độ dày) của từng đoạn tối liên tục
            if len(idx):
                dau = idx[0]
                for a_, b_ in zip(idx[:-1], idx[1:]):
                    if b_ != a_ + 1:
                        chay.append(((dau + a_) / 2, a_ - dau + 1))
                        dau = b_
                chay.append(((dau + idx[-1]) / 2, idx[-1] - dau + 1))
            # nét vẽ dày ≥ 6 px; mảnh sót của hàng lưới đã xoá và chữ tiêu đề phía trên 110 bị loại
            tams = [np.polyval(ay, t_) for t_, day in chay if day >= 6]
            tams = [v for v in tams if -2 < v < 110.5 and abs(v - LS_v) > 1.2]
            if len(tams) == 3:
                tim_duoc = sorted(tams)
                break
        if tim_duoc:
            break
    if not tim_duoc:
        print("   %6.2f  (không tách được ba đường)" % kv)
        continue
    v_, b_, d_ = tim_duoc
    lech_tong.append(d_ - (v_ + b_))
    print("   %6.2f %10.2f %10.2f %12.2f %12.2f %+12.2f" % (kv, v_, b_, d_, v_ + b_, d_ - (v_ + b_)))
if lech_tong:
    print("   |nét đứt − tổng| lớn nhất trên %d cột: %.2f đơn vị (1 ô lưới = 11 đơn vị)"
          % (len(lech_tong), np.abs(lech_tong).max()))

head("18. Ví dụ tính tay của mục 0.5 trong bài: hai cột trùng hệt nhau, λ = 1")
from fractions import Fraction as Fr  # noqa: E402

Xh = [[1, 1], [2, 2], [3, 3]]
yh = [2, 4, 6]
XtX_h = [[sum(Fr(r[i]) * r[j] for r in Xh) for j in range(2)] for i in range(2)]
Xty_h = [sum(Fr(r[i]) * v for r, v in zip(Xh, yh)) for i in range(2)]
print("   XᵀX = %s ; Xᵀy = %s" % ([[int(v) for v in r] for r in XtX_h], [int(v) for v in Xty_h]))
A_h = [[XtX_h[i][j] + (1 if i == j else 0) for j in range(2)] for i in range(2)]
det = A_h[0][0] * A_h[1][1] - A_h[0][1] * A_h[1][0]
print("   XᵀX + I = %s ; định thức = %s" % ([[int(v) for v in r] for r in A_h], det))
b1 = (A_h[1][1] * Xty_h[0] - A_h[0][1] * Xty_h[1]) / det
b2 = (A_h[0][0] * Xty_h[1] - A_h[1][0] * Xty_h[0]) / det
print("   β̂ = (%s, %s) = (%.6f, %.6f)" % (b1, b2, float(b1), float(b2)))
yhat = [float(b1 * r[0] + b2 * r[1]) for r in Xh]
print("   ŷ = %s" % [round(v, 6) for v in yhat])
print("   RSS = %.6f ; phạt λ‖β̂‖² = %.6f ; tổng hàm mục tiêu = %.6f"
      % (sum((a_ - b_) ** 2 for a_, b_ in zip(yh, yhat)), float(b1 * b1 + b2 * b2),
         sum((a_ - b_) ** 2 for a_, b_ in zip(yh, yhat)) + float(b1 * b1 + b2 * b2)))
print("   hệ số co của hướng (1,1)/√2: d² = %s, d²/(d²+λ) = %s = %.6f"
      % (XtX_h[0][0] * 2, Fr(28, 29), 28 / 29))

head("19. Tikhonov tr.501 — ví dụ z₂ = z₁ + p·cos ωs: hai nghiệm cách xa, hai vế phải gần nhau")
a, b, c, d = 0.0, 1.0, 0.0, 1.0
n, m = 40000, 400
h, h1, s, x = luoi(a, b, c, d, n, m)
K = nhan(x, s)
p_bien = 1000.0
print("   nhân Gauss độ rộng 0,08 như mục 2 ; lưới mịn n = %d (ít nhất %d điểm mỗi chu kỳ ở ω lớn nhất)"
      % (n, int(2 * np.pi / 3000 / h)))
print("   p = %g, nên max|z₁ − z₂| = %g ở mọi ω" % (p_bien, p_bien))
print("   %8s %22s %26s %14s" % ("ω", "‖u₁ − u₂‖ (chuẩn L2 tr.501)", "‖u₁ − u₂‖ / max|z₁ − z₂|", "ω·‖u₁ − u₂‖"))
for om in (1, 10, 30, 100, 300, 1000, 3000):
    du = K @ (p_bien * np.cos(om * s)) * h
    chuan = np.sqrt((du ** 2).sum() * h1)
    print("   %8d %22.6e %26.6e %14.4f" % (om, chuan, chuan / p_bien, om * chuan))
# phần trong và phần sát biên: bỏ 0,2 ở mỗi đầu đoạn x
trong = (x > 0.2) & (x < 0.8)
for om in (30, 100, 1000):
    du = K @ (p_bien * np.cos(om * s)) * h
    print("   ω = %4d: ‖u₁ − u₂‖ chỉ tính x ∈ (0,2; 0,8) = %.6e ; phần sát biên chiếm %.4f bình phương chuẩn"
          % (om, np.sqrt((du[trong] ** 2).sum() * h1), (du[~trong] ** 2).sum() / (du ** 2).sum()))
