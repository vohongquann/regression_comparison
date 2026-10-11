# -*- coding: utf-8 -*-
"""Sinh hình cho bài "Quantile Regression".

KHÔNG sửa file này bằng tay: nó được sinh ra từ chính khối code in trong bài, bằng
    python extract_fig.py "Quantile Regression" fig_quantile.py
Muốn đổi hình thì sửa khối code TRONG BÀI rồi chạy lại lệnh trên và regen.py.
"""
import os

os.chdir(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "AI algorithms", "Machine Learning", "Quantile Regression"))

# tools/fig_quantile.py — phần A
# Mọi con số dưới đây do tools/v_quantile.py mục 6 in ra; khối này chỉ VẼ.
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})

# nam quan sat dung lai duoc tu Bang II (v_quantile.py muc 6, BUOC 2)
BII_X = [1.0, 2.0, 4.0, 7.0, 9.0]
BII_Y = [3.0, 2.0, 7.0, 8.0, 6.0]
# bon nghiem cua Bang II tr.42
BII_B0 = [6 / 7, 21 / 8, 13 / 6, 17 / 3]
BII_B1 = [4 / 7, 3 / 8, 5 / 6, 1 / 3]
BII_MOC = [7 / 22, 1 / 2, 3 / 4]
NHAN = [r"$0<\theta\leq 7/22$", r"$7/22\leq\theta\leq 1/2$",
        r"$1/2\leq\theta\leq 3/4$", r"$3/4\leq\theta<1$"]
MAU = ["#2563eb", "#0891b2", "#d97706", "#dc2626"]

fig, ax = plt.subplots(1, 2, figsize=(11.4, 4.0))

xs = np.linspace(0, 10, 200)
for b0, b1, nh, c in zip(BII_B0, BII_B1, NHAN, MAU):
    ax[0].plot(xs, b0 + b1 * xs, lw=1.7, color=c, label=nh)
ax[0].scatter(BII_X, BII_Y, s=70, color="#111827", zorder=5)
for x, y in zip(BII_X, BII_Y):
    ax[0].annotate("(%d, %d)" % (x, y), (x, y), textcoords="offset points",
                   xytext=(6, -12), fontsize=8.2, color="#111827")
ax[0].set_xlim(0, 10); ax[0].set_ylim(0, 10)
ax[0].set_xlabel("x"); ax[0].set_ylabel("y")
ax[0].legend(fontsize=7.6, loc="upper left")
ax[0].set_title("Năm quan sát dựng lại được và bốn nghiệm của Bảng II\n"
                "mỗi đường đi qua ĐÚNG hai điểm — Định lý 3.1", fontsize=9)

canh = [0.0] + BII_MOC + [1.0]
for gt, ten, c, nl in ((BII_B0, r"$\beta^*_1$ (hệ số chặn)", "#be185d", "-"),
                       (BII_B1, r"$\beta^*_2$ (hệ số góc)", "#0f766e", "--")):
    for i, v in enumerate(gt):
        ax[1].hlines(v, canh[i], canh[i + 1], color=c, lw=2.2, linestyle=nl,
                     label=ten if i == 0 else None)
for m in BII_MOC:
    ax[1].axvline(m, color="#9ca3af", lw=0.9, linestyle=":")
    ax[1].annotate("%.4f" % m, (m, 0.12), fontsize=7.4, color="#6b7280",
                   ha="center", va="bottom",
                   bbox=dict(fc="white", ec="none", pad=1.0))
ax[1].set_xlim(0, 1); ax[1].set_ylim(0, 7.0)
ax[1].set_xlabel(r"$\theta$"); ax[1].set_ylabel("giá trị hệ số")
ax[1].legend(fontsize=8, loc="upper left")
ax[1].set_title("Hai hệ số theo θ: bậc thang, ba mốc\n"
                "hai hệ số nhảy NGƯỢC chiều nhau ở mỗi mốc", fontsize=9)

fig.savefig("quantile_bang_II.png", dpi=130, bbox_inches="tight", facecolor="white")

# ----------------------------------------------------------------------

# tools/fig_quantile.py — phần B
# Hình này thuần toán: không có dữ liệu nào, không có con số nào phải đi mượn.
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})

# Ham mat mat cua bai bao tr.38, viet gon mot dong (cach viet doi sau).
def rho(u, th):
    u = np.asarray(u, dtype=float)
    return u * (th - (u < 0))

u = np.linspace(-4, 4, 1601)
BO = [(0.1, "#2563eb"), (0.5, "#111827"), (0.9, "#dc2626")]

fig, ax = plt.subplots(1, 2, figsize=(11.4, 3.9))

for th, c in BO:
    ax[0].plot(u, rho(u, th), lw=1.9, color=c, label=r"$\theta$ = %.1f" % th)
ax[0].axvline(0, color="#9ca3af", lw=0.8)
ax[0].set_xlabel("phần dư  u = y − xb"); ax[0].set_ylabel(r"$\rho_\theta(u)$")
ax[0].legend(fontsize=8.4)
ax[0].set_title("Chữ V LỆCH, đỉnh tại u = 0\n"
                "θ = 0,9: đoán thiếu bị phạt gấp 9 lần đoán thừa", fontsize=9)

for th, c in BO:
    doc = np.where(u >= 0, th, th - 1.0)
    ax[1].step(u, doc, where="post", lw=1.9, color=c, label=r"$\theta$ = %.1f" % th)
    ax[1].annotate("%.1f" % th, (0.35, th), fontsize=7.6, color=c, va="bottom", ha="left",
                   bbox=dict(fc="white", ec="none", pad=0.8))
    ax[1].annotate("%.1f" % (th - 1), (-0.35, th - 1), fontsize=7.6, color=c,
                   va="top", ha="right", bbox=dict(fc="white", ec="none", pad=0.8))
ax[1].plot(u, 2 * u / 8.0, "--", color="#6b7280", lw=1.2,
           label="bình phương tối thiểu, chia 8")
ax[1].axvline(0, color="#9ca3af", lw=0.8)
ax[1].axhline(0, color="#9ca3af", lw=0.8)
ax[1].set_ylim(-1.15, 1.15)
ax[1].set_xlabel("phần dư  u"); ax[1].set_ylabel("độ dốc")
ax[1].legend(fontsize=8, loc="lower right")
ax[1].set_title("Độ dốc là BẬC THANG, bề cao luôn bằng 1\n"
                "đổi θ chỉ TRƯỢT bậc thang, không làm nó cao hơn", fontsize=9)

fig.savefig("quantile_ham_pinball.png", dpi=130, bbox_inches="tight", facecolor="white")

# ----------------------------------------------------------------------

# tools/fig_quantile.py — phần C
# Du lieu sinh lai tu dung hat giong cua v_quantile.py muc 17, nen khoi nay tu chua.
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import linprog

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})


def giai_lp(y, X, th):
    """Bai (P) cua Phu luc tr.48; beta tach doi vi no tu do ve dau."""
    T, K = X.shape
    c = np.r_[np.zeros(2 * K), th * np.ones(T), (1.0 - th) * np.ones(T)]
    A = np.hstack([X, -X, np.eye(T), -np.eye(T)])
    r = linprog(c, A_eq=A, b_eq=y, bounds=[(0, None)] * (2 * K + 2 * T), method="highs")
    return r.x[:K] - r.x[K:2 * K]


rng = np.random.default_rng(430430)
T = 400
x = np.sort(rng.uniform(0, 10, T))
y = 1 + 0.5 * x + (0.3 + 0.3 * x) * rng.normal(size=T)
X = np.c_[np.ones(T), x]
ols = np.linalg.lstsq(X, y, rcond=None)[0]

fig, ax = plt.subplots(1, 2, figsize=(11.6, 4.0))

ax[0].scatter(x, y, s=7, color="#9ca3af", zorder=2)
for th, c in ((0.10, "#2563eb"), (0.25, "#0891b2"), (0.50, "#111827"),
              (0.75, "#d97706"), (0.90, "#dc2626")):
    b = giai_lp(y, X, th)
    ax[0].plot(x, X @ b, lw=1.5, color=c, zorder=4,
               label=r"$\theta$ = %.2f  (hệ số góc %.4f)" % (th, b[1]))
# Duong OLS gan TRUNG duong theta = 0.50 (he so goc 0.5410 so voi 0.5543), nen phai
# doi mau va ve SAU CUNG, neu khong no bi che kin — loi da gap o Hinh 34 bai Huber.
ax[0].plot(x, X @ ols, "--", color="#7c3aed", lw=2.6, zorder=8, dashes=(5, 3),
           label="bình phương tối thiểu (%.4f)" % ols[1])
ax[0].set_xlabel("x"); ax[0].set_ylabel("y")
ax[0].legend(fontsize=7.2, ncol=2, loc="upper left")
ax[0].set_title("Phương sai tăng theo x: năm đường phân vị TOẢ RA\n"
                "bình phương tối thiểu chỉ cho MỘT đường", fontsize=9)

b05, b95 = giai_lp(y, X, 0.05), giai_lp(y, X, 0.95)
s = float(np.std(y - X @ ols, ddof=X.shape[1]))
trong_pv = (y >= X @ b05) & (y <= X @ b95)
trong_ols = np.abs(y - X @ ols) <= 1.96 * s
lo = np.arange(0, 10, 2)
pv = [float(np.mean(trong_pv[(x >= a) & (x < a + 2)])) for a in lo]
ls = [float(np.mean(trong_ols[(x >= a) & (x < a + 2)])) for a in lo]
w = 0.38
ax[1].bar(np.arange(5) - w / 2, pv, w, color="#0891b2", label="phân vị 5 %–95 %")
ax[1].bar(np.arange(5) + w / 2, ls, w, color="#9ca3af",
          label=r"bình phương tối thiểu $\pm 1{,}96\,s$")
ax[1].axhline(0.90, color="#dc2626", lw=1.2, linestyle="--", label="mức 90 % nhắm tới")
for i, (a, b) in enumerate(zip(pv, ls)):
    ax[1].annotate("%.3f" % a, (i - w / 2, a), ha="center", va="bottom", fontsize=7)
    ax[1].annotate("%.3f" % b, (i + w / 2, b), ha="center", va="bottom", fontsize=7)
ax[1].set_xticks(np.arange(5))
ax[1].set_xticklabels(["[%d, %d)" % (a, a + 2) for a in lo])
ax[1].set_ylim(0, 1.12); ax[1].set_xlabel("khúc x"); ax[1].set_ylabel("tỉ lệ phủ")
ax[1].legend(fontsize=7.6, loc="lower left")
ax[1].set_title("Độ phủ theo từng khúc x\n"
                "cột xám phủ QUÁ ở đầu này và phủ THIẾU ở đầu kia", fontsize=9)

fig.savefig("quantile_nam_phan_vi.png", dpi=130, bbox_inches="tight", facecolor="white")
