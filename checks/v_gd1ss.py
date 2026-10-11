# -*- coding: utf-8 -*-
"""Đo BẾ TẮC CỦA CÁCH CŨ cho từng bài GĐ 1 — khối SS của khung 7 khối.

CLAUDE.md mục 2.0 đòi khối SS nêu bế tắc của cách cũ **bằng SỐ, không bằng tính từ**.
Mỗi mục dưới đây dựng đúng tình huống mà cách cũ hỏng, rồi in ra con số để bài viết
trích. Không con số nào trong 9 bài GĐ 1 được phép gõ tay ngoài script này.

Chạy:  cd tools && python v_gd1ss.py
"""
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
from mllib import collinear, diabetes, rmse           # noqa: E402

np.set_printoptions(precision=4, suppress=True, linewidth=130)


def head(s):
    print("\n" + "=" * 78 + "\n" + s + "\n" + "=" * 78)


def ols(X, y):
    return np.linalg.lstsq(X, y, rcond=None)[0]


# ══════════════════════════════════════════════ 350. Machine Learning (Introduction)
# Cách cũ: viết LUẬT TAY. Bế tắc: mỗi luật thêm vào chỉ vá đúng vùng nó thấy.
head("350  Machine Learning (Introduction) — luật viết tay vs học từ dữ liệu")

rng = np.random.default_rng(0)


def sinh(m, seed):
    """Quyết định cho vay. Rủi ro thật phụ thuộc TRƠN vào 2 biến."""
    r = np.random.default_rng(seed)
    tn = r.uniform(0, 1, m)
    no = r.uniform(0, 1, m)
    p = 1 / (1 + np.exp(-(6 * tn - 5 * no - 0.5)))
    return np.c_[tn, no], (r.uniform(size=m) < p).astype(int)


Xtr, ytr = sinh(500, 0)          # chuyên gia chỉ nhìn thấy 500 hồ sơ cũ
Xte, yte = sinh(20000, 1)        # rồi hệ thống chạy trên hồ sơ MỚI


def luat(k):
    """k ô luật vuông góc, ngưỡng chốt trên tập cũ rồi đem áp cho hồ sơ mới."""
    d = int(np.ceil(np.sqrt(k)))
    def o(X):
        return (np.minimum((X[:, 0] * d).astype(int), d - 1) * d
                + np.minimum((X[:, 1] * d).astype(int), d - 1))
    otr, ote = o(Xtr), o(Xte)
    quyet = np.zeros(d * d, dtype=int)
    for c in range(d * d):
        m = otr == c
        if m.sum():
            quyet[c] = int(ytr[m].mean() >= 0.5)
    return (quyet[otr] == ytr).mean(), (quyet[ote] == yte).mean()


print("   Chuyên gia chốt luật trên 500 hồ sơ cũ, rồi áp cho 20 000 hồ sơ mới:")
print("       %-10s %14s %14s" % ("số luật", "trên hồ sơ cũ", "trên hồ sơ mới"))
for k in (1, 4, 16, 64, 256, 1024):
    a, b = luat(k)
    print("       %-10d %13.1f%% %13.1f%%" % (k, 100 * a, 100 * b))

from sklearn.linear_model import LogisticRegression    # noqa: E402
lr = LogisticRegression().fit(Xtr, ytr)
print("       %-10s %13.1f%% %13.1f%%"
      % ("2 hệ số học", 100 * lr.score(Xtr, ytr), 100 * lr.score(Xte, yte)))
a1024, b1024 = luat(1024)
print("   -> Thêm luật làm điểm trên hồ sơ CŨ tăng tới %.1f%%, còn trên hồ sơ MỚI"
      % (100 * a1024))
print("      thì TỤT xuống %.1f%%. Hai con số học được thắng %.1f điểm và không cần"
      % (100 * b1024, 100 * (lr.score(Xte, yte) - b1024)))
print("      ai ngồi viết luật. Đó là lý do ML tồn tại.")


# ══════════════════════════════════════════════════════ 370. Polynomial Regression
# Cách cũ: Linear Regression. Bế tắc: dữ liệu cong thì đường thẳng KHÔNG khớp nổi,
# và lỗi không giảm dù thêm bao nhiêu dữ liệu.
head("370  Polynomial Regression — đường thẳng trên dữ liệu cong")

rng = np.random.default_rng(1)
x = np.linspace(-1, 1, 200)
y_sach = np.sin(2.5 * x)
y = y_sach + 0.10 * rng.normal(size=x.size)

for bac in (1, 2, 3, 5):
    V = np.vander(x, bac + 1, increasing=True)
    r = rmse(y, V @ ols(V, y))
    print("   bậc %d: RMSE = %.4f" % (bac, r))

V1 = np.vander(x, 2, increasing=True)
r1 = rmse(y, V1 @ ols(V1, y))
V3 = np.vander(x, 4, increasing=True)
r3 = rmse(y, V3 @ ols(V3, y))
print("   -> bậc 1 sang bậc 3: RMSE %.4f -> %.4f, giảm %.1f lần." % (r1, r3, r1 / r3))
print("   Nhiễu thật có độ lệch chuẩn 0.10, nên bậc 3 đã chạm sàn; bậc 1 thì không.")

print("\n   Thêm dữ liệu có cứu được đường thẳng không?")
cuoi = None
for m in (50, 200, 2000, 20000, 200000):
    xx = np.linspace(-1, 1, m)
    yy = np.sin(2.5 * xx) + 0.10 * np.random.default_rng(2).normal(size=m)
    Vm = np.vander(xx, 2, increasing=True)
    cuoi = rmse(yy, Vm @ ols(Vm, yy))
    print("       n = %7d -> RMSE bậc 1 = %.4f" % (m, cuoi))
print("   -> KHÔNG. Sai số đứng yên ở %.2f dù n tăng %d lần — gấp %.1f lần mức nhiễu 0.10."
      % (cuoi, 200000 // 50, cuoi / 0.10))
print("      Thiếu bậc tự do, không phải thiếu dữ liệu: đây là CHỆCH, không phải phương sai.")


# ═══════════════════════════════════════════════════════════════ 380. Ridge
# Cách cũ: OLS. Bế tắc: cột gần trùng nhau -> hệ số nổ và đảo dấu giữa các mẫu.
head("380  Ridge Regression — OLS trên thiết kế gần suy biến")

B_THAT = np.array([1.0, 1.0])


def gan_trung(seed, n=50, jitter=0.01):
    """Hai cột gần trùng hệt nhau — đúng tình huống Hoerl & Kennard 1970 mô tả."""
    r = np.random.default_rng(seed)
    z = r.normal(size=n)
    X = np.c_[z, z + jitter * r.normal(size=n)]
    return X, X @ B_THAT + r.normal(size=n)


X0, _ = gan_trung(0)
print("   Hai cột tương quan %.6f · κ(X) = %.3e · κ(XᵀX) = %.3e"
      % (np.corrcoef(X0[:, 0], X0[:, 1])[0, 1],
         np.linalg.cond(X0), np.linalg.cond(X0.T @ X0)))
print("   Hệ số thật: β = [%+.1f, %+.1f]. TỔNG β₁+β₂ = %.1f xác định rõ; CÁCH CHIA thì không."
      % (B_THAT[0], B_THAT[1], B_THAT.sum()))

from sklearn.linear_model import Ridge                 # noqa: E402
bs, rs = [], []
for s in range(400):
    Xs, ys = gan_trung(s)
    bs.append(ols(Xs, ys))
    rs.append(Ridge(alpha=1.0, fit_intercept=False).fit(Xs, ys).coef_)
bs, rs = np.array(bs), np.array(rs)

print("\n   400 mẫu độc lập cùng phân phối:")
print("       %-14s %10s %10s %22s %12s"
      % ("", "β̂₁ t.bình", "độ lệch", "khoảng β̂₁", "‖β̂−β‖ t.bình"))
for ten_, m in (("OLS", bs), ("Ridge λ=1", rs)):
    print("       %-14s %+9.2f %10.2f   [%+8.2f, %+8.2f] %12.3f"
          % (ten_, m[:, 0].mean(), m[:, 0].std(), m[:, 0].min(), m[:, 0].max(),
             np.linalg.norm(m - B_THAT, axis=1).mean()))

print("       OLS cho β̂₁ NGƯỢC DẤU với hệ số thật ở %.0f%% số lần chạy."
      % (100 * (bs[:, 0] < 0).mean()))
print("       Ridge: %.0f%%." % (100 * (rs[:, 0] < 0).mean()))
print("   -> Tổng thì OLS ước lượng tốt (β̂₁+β̂₂ lệch chuẩn chỉ %.2f), nhưng TỪNG hệ số"
      % bs.sum(1).std())
print("      thì lệch chuẩn %.1f — gấp %.0f lần. Ridge đổi một chút chệch lấy sai số"
      % (bs[:, 0].std(), bs[:, 0].std() / bs.sum(1).std()))
print("      tổng nhỏ hơn %.1f lần."
      % (np.linalg.norm(bs - B_THAT, axis=1).mean()
         / np.linalg.norm(rs - B_THAT, axis=1).mean()))


# ═══════════════════════════════════════════════════════════════ 390. Lasso
# Cách cũ: ridge (co nhưng không xoá) và best-subset (xoá nhưng nổ tổ hợp).
head("390  Lasso Regression — ridge không xoá được biến, best-subset không chạy nổi")

Xd, yd, ten = diabetes()
from sklearn.linear_model import Lasso                 # noqa: E402

r = Ridge(alpha=1.0, fit_intercept=False).fit(Xd, yd)
print("   Ridge trên diabetes (10 biến): số hệ số đúng bằng 0 = %d/10"
      % int((r.coef_ == 0).sum()))
print("       hệ số nhỏ nhất theo trị tuyệt đối: %.4g  (nhỏ, nhưng KHÁC 0)"
      % np.abs(r.coef_).min())
for a in (0.05, 0.2, 0.5, 1.0, 2.0):
    l = Lasso(alpha=a, fit_intercept=False, max_iter=100000).fit(Xd, yd)
    print("   Lasso α=%-5.2f: số hệ số bằng 0 = %2d/10   giữ lại: %s"
          % (a, int((l.coef_ == 0).sum()),
             [ten[i] for i in np.where(l.coef_ != 0)[0]]))

print("\n   Best-subset — cách xoá biến có trước lasso:")
for p_ in (10, 20, 40, 60):
    print("       p = %2d -> phải thử 2^%d = %.3g tổ hợp" % (p_, p_, 2.0 ** p_))
print("   -> p=60 là %.1e tổ hợp. Lasso giải cùng bài toán bằng MỘT lần tối ưu lồi."
      % 2.0 ** 60)


# ═════════════════════════════════════════════════════════════ 400. Elastic Net
# Cách cũ: lasso. Hai bế tắc Zou & Hastie 2005 nêu ở mục 2.
head("400  Elastic Net — hai chỗ lasso tắc")

print("   (a) p > n: lasso chọn được NHIỀU NHẤT n biến (Zou & Hastie mục 2.1)")
rng = np.random.default_rng(0)
n_, p_ = 30, 200
Xa = rng.normal(size=(n_, p_))
b = np.zeros(p_)
b[:50] = 2.0                       # 50 biến thật khác 0 — nhiều hơn n
ya = Xa @ b + rng.normal(size=n_)
from sklearn.linear_model import lars_path, ElasticNet  # noqa: E402
_, _, cf = lars_path(Xa, ya, method="lasso")
print("       n = %d, số biến thật khác 0 = 50" % n_)
print("       Lasso chọn nhiều nhất: %d biến  -> chặn trên là n, KHÔNG phải 50"
      % int((cf != 0).sum(0).max()))
en = ElasticNet(alpha=0.05, l1_ratio=0.5, fit_intercept=False,
                max_iter=200000).fit(Xa, ya)
print("       Elastic Net (l1_ratio=0.5) chọn: %d biến" % int((en.coef_ != 0).sum()))

print("\n   (b) nhóm biến tương quan: lasso chọn 1, vứt phần còn lại")
rng = np.random.default_rng(3)
n2 = 100
z = rng.normal(size=n2)
Xb = np.c_[z + 0.01 * rng.normal(size=n2),
           z + 0.01 * rng.normal(size=n2),
           z + 0.01 * rng.normal(size=n2),
           rng.normal(size=n2)]
Xb = (Xb - Xb.mean(0)) / Xb.std(0)          # chuẩn hoá về độ lệch chuẩn 1
yb = 3 * Xb[:, 0] + 3 * Xb[:, 1] + 3 * Xb[:, 2] + 0.5 * rng.normal(size=n2)
yb -= yb.mean()
print("       Ba cột đầu tương quan %.4f; hệ số thật CHIA ĐỀU: [3, 3, 3, 0]."
      % np.corrcoef(Xb[:, 0], Xb[:, 1])[0, 1])
print("       %-6s %-34s %-34s" % ("α", "Lasso", "Elastic Net (l1_ratio=0.3)"))
for a in (0.1, 0.5, 1.0, 2.0):
    lb = Lasso(alpha=a, fit_intercept=False, max_iter=200000).fit(Xb, yb)
    eb = ElasticNet(alpha=a, l1_ratio=0.3, fit_intercept=False,
                    max_iter=200000).fit(Xb, yb)
    print("       %-6.1f %-34s %-34s"
          % (a, "%s  %d/3" % (np.round(lb.coef_[:3], 2), int((lb.coef_[:3] != 0).sum())),
             "%s  %d/3" % (np.round(eb.coef_[:3], 2), int((eb.coef_[:3] != 0).sum()))))
lb = Lasso(alpha=0.5, fit_intercept=False, max_iter=200000).fit(Xb, yb)
eb = ElasticNet(alpha=0.5, l1_ratio=0.3, fit_intercept=False,
                max_iter=200000).fit(Xb, yb)
tl = lb.coef_[:3].max() - lb.coef_[:3].min()
te = eb.coef_[:3].max() - eb.coef_[:3].min()
print("   -> KHÔNG phải chuyện lasso xoá mất cột (nó vẫn giữ 3/3 ở hầu hết α).")
print("      Chỗ hỏng là CHIA KHÔNG ĐỀU: ở α=0.5 lasso trải hệ số ra %.2f đơn vị"
      % tl)
print("      (%.2f … %.2f) giữa ba cột tương quan 0.9999, còn Elastic Net trải %.4f."
      % (lb.coef_[:3].min(), lb.coef_[:3].max(), te))
print("      Chênh nhau %.0f lần. Đó chính là 'hiệu ứng nhóm' Zou & Hastie chứng minh"
      % (tl / max(te, 1e-12)))
print("      ở Định lý 1: hai cột giống hệt nhau thì Elastic Net BẮT BUỘC cho hai hệ số")
print("      bằng nhau, còn lasso thì không có ràng buộc nào như vậy.")


# ═══════════════════════════════════════════════════════════════ 410. LARS
# Cách cũ: Forward Stagewise ε-nhỏ. Bế tắc: số bước.
head("410  LARS — Forward Stagewise cần bao nhiêu bước để tới cùng chỗ")

Xd, yd, ten = diabetes()


def stagewise(X, y, eps, n_max=2000000):
    b = np.zeros(X.shape[1])
    r = y.copy()
    for i in range(n_max):
        c = X.T @ r
        j = int(np.argmax(np.abs(c)))
        if np.abs(c[j]) < 1e-9:
            return b, i
        b[j] += eps * np.sign(c[j])
        r = y - X @ b
        if np.abs(b).sum() >= t_dich:
            return b, i + 1
    return b, n_max


b_ols = ols(Xd, yd)
t_dich = np.abs(b_ols).sum() * 0.30           # dừng ở 30% đường đi, cho so sánh được
print("   Đích: Σ|β̂| = %.2f (30%% quãng đường tới OLS)" % t_dich)
for eps in (10.0, 1.0, 0.1, 0.01):
    bb, k = stagewise(Xd, yd, eps)
    print("       Stagewise ε=%-6.2f -> %8d bước, Σ|β̂| = %.2f" % (eps, k, np.abs(bb).sum()))

_, _, cf = lars_path(Xd, yd, method="lar")
print("   LARS       -> %d bước là hết đường, tới đúng nghiệm OLS." % (cf.shape[1] - 1))
print("   -> ε=0.01 tốn %s bước cho 30%% đường; LARS đi CẢ đường trong %d bước."
      % ("{:,}".format(stagewise(Xd, yd, 0.01)[1]), cf.shape[1] - 1))


# ══════════════════════════════════════════════════════════════ 420. Huber
# Cách cũ: OLS. Bế tắc: MỘT điểm hỏng kéo cả đường.
head("420  Huber Regression — một điểm ngoại lai làm OLS lệch bao nhiêu")

rng = np.random.default_rng(0)
n = 60
x = rng.uniform(0, 10, n)
Xh = np.c_[np.ones(n), x]
y0 = 2.0 + 1.5 * x + rng.normal(scale=1.0, size=n)

from sklearn.linear_model import HuberRegressor        # noqa: E402
print("   Hệ số thật: chặn 2.00, góc 1.50")
b = ols(Xh, y0)
print("   Sạch          : OLS chặn %+.2f góc %+.2f" % (b[0], b[1]))
for shift in (20, 50, 100):
    yb2 = y0.copy()
    yb2[0] += shift
    b = ols(Xh, yb2)
    h = HuberRegressor(epsilon=1.35).fit(x.reshape(-1, 1), yb2)
    print("   +%3d ở 1/%d điểm: OLS chặn %+.2f góc %+.2f  |  Huber chặn %+.2f góc %+.2f"
          % (shift, n, b[0], b[1], h.intercept_, h.coef_[0]))
yb2 = y0.copy()
yb2[0] += 100
b = ols(Xh, yb2)
h = HuberRegressor(epsilon=1.35).fit(x.reshape(-1, 1), yb2)
print("   -> 1/60 điểm (1.7%% dữ liệu) đẩy hệ số góc OLS lệch %.0f%% khỏi giá trị thật;"
      % (100 * abs(b[1] - 1.5) / 1.5))
print("      Huber lệch %.0f%%." % (100 * abs(h.coef_[0] - 1.5) / 1.5))


# ═══════════════════════════════════════════════════════════ 430. Quantile
# Cách cũ: OLS. Bế tắc: OLS trả TRUNG BÌNH, không trả được khoảng.
head("430  Quantile Regression — OLS không trả lời được câu hỏi về khoảng")

rng = np.random.default_rng(0)
n = 3000
x = rng.uniform(0, 10, n)
# phương sai TĂNG theo x — tình huống OLS±hằng số sai nặng nhất
y = 2.0 + 1.5 * x + rng.normal(scale=0.5 + 0.6 * x, size=n)
Xq = np.c_[np.ones(n), x]

b = ols(Xq, y)
res = y - Xq @ b
s = res.std()
lo, hi = Xq @ b - 1.96 * s, Xq @ b + 1.96 * s
print("   OLS + khoảng ±1.96σ (giả định nhiễu đều): phủ %.1f%% tổng thể"
      % (100 * ((y >= lo) & (y <= hi)).mean()))

from sklearn.linear_model import QuantileRegressor     # noqa: E402
q05 = QuantileRegressor(quantile=0.05, alpha=0, solver="highs").fit(x.reshape(-1, 1), y)
q95 = QuantileRegressor(quantile=0.95, alpha=0, solver="highs").fit(x.reshape(-1, 1), y)
lo2 = q05.intercept_ + q05.coef_[0] * x
hi2 = q95.intercept_ + q95.coef_[0] * x
print("   Hồi quy phân vị 5%%–95%%                    : phủ %.1f%% tổng thể"
      % (100 * ((y >= lo2) & (y <= hi2)).mean()))

print("\n   Nhưng con số tổng thể giấu chỗ hỏng. Phủ theo từng khúc của x:")
print("       %-12s %10s %10s" % ("khúc x", "OLS±σ", "phân vị"))
for a_, b_ in [(0, 2), (2, 4), (4, 6), (6, 8), (8, 10)]:
    m = (x >= a_) & (x < b_)
    print("       [%2d, %2d)     %9.1f%% %9.1f%%"
          % (a_, b_, 100 * ((y >= lo) & (y <= hi))[m].mean(),
             100 * ((y >= lo2) & (y <= hi2))[m].mean()))
print("   -> OLS phủ quá NHIỀU ở x nhỏ và quá ÍT ở x lớn. Trung bình đúng, từng chỗ sai.")


# ══════════════════════════════════════════════════════ 440. Bayesian Linear Reg
# Cách cũ: OLS. Bế tắc: một con số, không kèm mức tin cậy, và nó sai lặng lẽ khi ít dữ liệu.
head("440  Bayesian Linear Regression — OLS im lặng ở chỗ nó không biết")

rng = np.random.default_rng(0)
sig = 1.0
x_tr = rng.uniform(-1, 1, 8)             # dữ liệu CHỈ nằm trong [-1, 1]
y_tr = 2.0 + 1.5 * x_tr + rng.normal(scale=sig, size=8)
Xt = np.c_[np.ones(8), x_tr]

b = ols(Xt, y_tr)
tau = 3.0                                 # độ lệch chuẩn tiên nghiệm
A = Xt.T @ Xt / sig ** 2 + np.eye(2) / tau ** 2
S = np.linalg.inv(A)
mu = S @ (Xt.T @ y_tr) / sig ** 2

print("   8 điểm huấn luyện, x chỉ nằm trong [-1, 1]. Hệ số thật: 2.00 và 1.50")
print("   OLS      : β̂ = [%+.3f, %+.3f]   — hai con số, không kèm gì" % (b[0], b[1]))
print("   Hậu nghiệm: β̄ = [%+.3f, %+.3f]  ± [%.3f, %.3f]"
      % (mu[0], mu[1], np.sqrt(S[0, 0]), np.sqrt(S[1, 1])))
print("\n   Dự đoán ở ba chỗ — trong vùng dữ liệu và ngoài xa:")
print("       %-8s %10s %14s" % ("x", "OLS", "độ lệch chuẩn hậu nghiệm"))
for xs in (0.0, 1.0, 5.0, 20.0):
    v = np.array([1.0, xs])
    print("       %-8.1f %+9.3f %14.3f" % (xs, v @ b, np.sqrt(v @ S @ v)))
print("   -> OLS đưa một con số ở x=20 y hệt cách nó đưa ở x=0. Hậu nghiệm nói rõ")
print("      bất định ở x=20 lớn hơn ở x=0 %.1f lần — đó là thông tin OLS vứt đi."
      % (np.sqrt(np.array([1., 20.]) @ S @ np.array([1., 20.]))
         / np.sqrt(np.array([1., 0.]) @ S @ np.array([1., 0.]))))

print("\n" + "=" * 78)
print("Xong. Mọi con số khối SS của 9 bài GĐ 1 lấy từ đây.")
print("=" * 78)
