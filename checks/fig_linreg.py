# -*- coding: utf-8 -*-
"""Sinh hình cho bài "Linear Regression".

KHÔNG sửa file này bằng tay: nó được sinh ra từ chính khối code in trong bài, bằng
    python extract_fig.py "Linear Regression" fig_linreg.py
Muốn đổi hình thì sửa khối code TRONG BÀI rồi chạy lại lệnh trên và regen.py.
"""
import os

os.chdir(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "AI algorithms", "Machine Learning", "Linear Regression"))

import numpy as np, matplotlib.pyplot as plt, matplotlib.patches as mp

plt.rcParams.update({                          # kiểu chữ, lưới và màu nền của hình
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})

rng = np.random.default_rng(3)                 # cùng seed với mục 7, nên cùng dữ liệu
n = 25
x = np.sort(rng.uniform(0, 10, n))
y = 2.0 + 1.3 * x + rng.normal(0, 2.0, n)
X = np.c_[np.ones(n), x]
b = np.linalg.lstsq(X, y, rcond=None)[0]
yhat = X @ b
OLS = "#6b7280"

fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.7))

# --- bảng 1: không gian dữ liệu, phần dư vẽ THẲNG ĐỨNG ---
ax = axes[0]
ax.scatter(x, y, s=22, color="#374151", zorder=3)
ax.plot(x, yhat, color=OLS, lw=1.8, zorder=2)
for xi, yi, yh in zip(x, y, yhat):
    ax.plot([xi, xi], [yi, yh], color="#dc2626", lw=1.0, alpha=.75, zorder=1)
ax.set_xlabel("x"); ax.set_ylabel("y")
ax.text(.5, max(y) * .97, r"$\hat\beta$ = (%.3f, %.3f)" % (b[0], b[1]), fontsize=8)
ax.set_title("1. Trong không gian dữ liệu\nOLS cực tiểu TỔNG BÌNH PHƯƠNG đoạn đỏ", fontsize=9)

# --- bảng 2: đường mức của S(β) trên lưới 200x200 ---
ax = axes[1]
b0 = np.linspace(b[0] - 6, b[0] + 6, 200)
b1 = np.linspace(b[1] - 1.2, b[1] + 1.2, 200)
B0, B1 = np.meshgrid(b0, b1)
S = np.array([[np.sum((y - (i + j * x)) ** 2) for i in b0] for j in b1])
ax.contour(B0, B1, S, levels=np.percentile(S, [1, 3, 7, 14, 25, 40, 60, 85]),
           cmap="Greys", linewidths=.7)
ax.plot(b[0], b[1], "*", color=OLS, ms=16)
ax.set_xlabel(r"$\beta_0$"); ax.set_ylabel(r"$\beta_1$")
ax.set_title("2. Trong không gian tham số\n"
             r"$S(\beta)$ là parabôlôit lồi — đúng MỘT cực tiểu", fontsize=9)

# --- bảng 3: sơ đồ hình chiếu, toạ độ đặt tay ---
ax = axes[2]
ax.set_aspect("equal"); ax.grid(False); ax.set_xticks([]); ax.set_yticks([])
ax.add_patch(mp.Polygon([[0, 0], [3.4, .55], [4.7, 2.45], [1.3, 1.9]], closed=True,
                        fc="#e5e7eb", ec="#9ca3af", lw=.8, alpha=.7, zorder=0))
ax.plot([0, 3.4], [0, 0.55], color="#9ca3af", lw=1.4)     # cột x1
ax.plot([0, 1.3], [0, 1.9],  color="#9ca3af", lw=1.4)     # cột x2
ax.annotate("", (2.6, 1.35), (0, 0), arrowprops=dict(arrowstyle="->", color=OLS, lw=1.8))
ax.annotate("", (2.6, 3.3),  (0, 0), arrowprops=dict(arrowstyle="->", color="#111827", lw=1.8))
ax.annotate("", (2.6, 3.3), (2.6, 1.35), arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1.6))
ax.plot([2.6, 2.36], [1.35, 1.58], color="#dc2626", lw=.9)   # ký hiệu góc vuông
ax.plot([2.36, 2.6], [1.58, 1.82], color="#dc2626", lw=.9)
ax.text(2.72, 3.32, r"$y$", fontsize=10)
ax.text(2.05, 1.05, r"$\hat y=X\hat\beta$", fontsize=10, color=OLS)
ax.text(2.70, 2.30, r"$e=y-\hat y$", fontsize=10, color="#dc2626")
ax.text(3.55, 1.50, r"$\mathrm{col}(X)$", fontsize=9, color="#4b5563")  # tránh đè nhãn e
ax.text(3.45, 0.26, r"$x_1$", fontsize=9, color="#4b5563")
ax.text(1.02, 2.00, r"$x_2$", fontsize=9, color="#4b5563")
ax.set_xlim(-.35, 5.1); ax.set_ylim(-.35, 3.9)
ax.set_title("3. Trong hình học\ne vuông góc mọi cột của X — đó là phương trình chuẩn", fontsize=9)

fig.savefig("ols_three_views.png", dpi=130, bbox_inches="tight", facecolor="white")

# ----------------------------------------------------------------------

import numpy as np, matplotlib.pyplot as plt

plt.rcParams.update({                          # cùng kiểu vẽ với Hình 1
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})

rng = np.random.default_rng(11)
n = 120
xx = np.sort(rng.uniform(0, 10, n))
base_noise = rng.normal(0, 1, n)              # cùng nhiễu để so sánh công bằng
clean = 2 + 1.2 * xx + 1.5 * base_noise
yo = clean.copy()
yo[5] += 40                                   # chỉ thêm đúng 1 ngoại lai

sets = [
    ("Đúng giả định",            clean,
     "phần dư đều, không hình dạng"),
    ("Bỏ sót độ cong",           2 + 1.2 * xx + 0.35 * (xx - 5) ** 2 + 1.5 * base_noise,
     "đường thẳng bỏ sót cấu trúc hình chữ U"),
    ("Phương sai không đều",     2 + 1.2 * xx + (0.3 + 0.45 * xx) * base_noise,
     "SE cổ điển sai; dải nhiễu loe thành phễu"),
    ("Có điểm ngoại lai",        yo, "1 điểm có thể kéo cả đường"),
]

fig, axes = plt.subplots(1, 4, figsize=(14.5, 3.2))
for ax, (name, yv, note) in zip(axes, sets):
    Xv = np.c_[np.ones(n), xx]
    bb = np.linalg.lstsq(Xv, yv, rcond=None)[0]   # khớp OLS trên CHÍNH dữ liệu bị vỡ
    r = yv - Xv @ bb
    ax.scatter(Xv @ bb, r, s=9, color="#374151")
    ax.axhline(0, color="#6b7280", lw=1.3)
    ax.set_xlabel(r"$\hat y$"); ax.set_ylabel("phần dư" if ax is axes[0] else "")
    ax.set_title("%s\n%s" % (name, note), fontsize=8.4)

fig.savefig("ols_assumptions.png", dpi=130, bbox_inches="tight", facecolor="white")
