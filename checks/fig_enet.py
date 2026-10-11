# -*- coding: utf-8 -*-
"""Sinh hình cho bài "Elastic Net".

KHÔNG sửa file này bằng tay: nó được sinh ra từ chính khối code in trong bài, bằng
    python extract_fig.py "Elastic Net" fig_enet.py
Muốn đổi hình thì sửa khối code TRONG BÀI rồi chạy lại lệnh trên và regen.py.
"""
import os

os.chdir(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "AI algorithms", "Machine Learning", "Elastic Net"))

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})

tran_lasso = np.array([ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 12, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 26, 26, 26, 26, 25, 26, 27, 28, 28, 28, 28, 28, 29])
tran_en_0_01 = np.array([ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 12, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 33, 33, 34, 35, 36, 37, 38, 39, 40, 40, 40, 41, 42, 43, 44, 45, 45, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199])
tran_en_1_0 = np.array([ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200])
tran_n, tran_p = 30, 200

fig, ax = plt.subplots(figsize=(7.8, 4.4))
ax.plot(np.arange(len(tran_lasso)), tran_lasso, color="#dc2626", lw=1.8, label="lasso")
ax.plot(np.arange(len(tran_en_0_01)), tran_en_0_01, color="#059669", lw=1.8,
        label=r"elastic net, $\lambda_2 = 0{,}01$")
ax.plot(np.arange(len(tran_en_1_0)), tran_en_1_0, color="#2563eb", lw=1.8,
        label=r"elastic net, $\lambda_2 = 1$")
ax.axhline(tran_n, color="#111827", ls="--", lw=1.0)
ax.text(2, tran_n + 3, "n = %d — trần của lasso" % tran_n, fontsize=8)
ax.axhline(tran_p, color="#6b7280", ls=":", lw=1.0)
ax.text(2, tran_p - 12, "p = %d" % tran_p, fontsize=8, color="#6b7280")
ax.set_xlabel("bước của LARS / LARS-EN")
ax.set_ylabel("số hệ số khác 0")
ax.legend(fontsize=8, loc="lower right")
ax.set_title("Kịch bản (a) của bài báo: lasso bão hoà ở %d = n − 1 biến,\n"
             "còn elastic net đi tới %d = p" % (int(tran_lasso.max()), int(tran_en_1_0.max())), fontsize=9)
fig.tight_layout()
fig.savefig("enet_tran_n.png", dpi=130, bbox_inches="tight", facecolor="white")

# ----------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})

# Miền phạt mức 1 của (1−α)|β|₁ + α|β|² — quy ước α của [Z] 2005: α = 1 là RIDGE.
g = np.linspace(0.0, 1.0, 20001)
fig, ax = plt.subplots(1, 2, figsize=(10.6, 4.4))

for a, col, nhan in [(0.0, "#dc2626", "α = 0 — lasso"), (0.25, "#ea580c", "α = 0,25"),
                     (0.5, "#059669", "α = 0,5 — Hình 1 của bài báo"),
                     (0.75, "#0891b2", "α = 0,75"), (1.0, "#2563eb", "α = 1 — ridge")]:
    if a == 0.0:
        b2 = 1.0 - g
    else:
        d = (1 - a) ** 2 + 4 * a * (1 - (1 - a) * g - a * g ** 2)
        b2 = (-(1 - a) + np.sqrt(np.maximum(d, 0))) / (2 * a)
    x = np.r_[g, -g[::-1], -g, g[::-1]]
    y = np.r_[b2, b2[::-1], -b2, -b2[::-1]]
    ax[0].plot(x, y, color=col, lw=1.5, label=nhan)
ax[0].set_aspect("equal")
ax[0].legend(fontsize=7.6, loc="upper right")
ax[0].set_xlabel(r"$\beta_1$")
ax[0].set_ylabel(r"$\beta_2$")
ax[0].set_title("Miền phạt mức 1 biến dạng liên tục theo α\n"
                "(quy ước của Zou & Hastie 2005: α = 1 là ridge)", fontsize=9)

# hệ số góc một phía tại đỉnh (0,1) — đỉnh NHỌN khi α < 1, TRƠN khi α = 1
eps = np.logspace(-8, -1, 200)
for a, col in [(0.0, "#dc2626"), (0.25, "#ea580c"), (0.5, "#059669"), (0.75, "#0891b2"), (1.0, "#2563eb")]:
    if a == 0.0:
        d1 = np.full_like(eps, -1.0)
    elif a == 1.0:
        d1 = (np.sqrt(np.maximum(1 - eps ** 2, 0)) - 1) / eps
    else:
        d1 = ((-(1 - a) + np.sqrt((1 - a) ** 2 + 4 * a * (1 - (1 - a) * eps - a * eps ** 2))) / (2 * a) - 1) / eps
    ax[1].semilogx(eps, d1, color=col, lw=1.5)
    ax[1].text(1.3e-8, d1[0] + 0.03, "α = %.2f → %.3f" % (a, d1[0]), fontsize=7.4, color=col)
ax[1].set_xlabel(r"khoảng cách tới đỉnh, $\beta_1$")
ax[1].set_ylabel("hệ số góc một phía của biên")
ax[1].set_title("Đỉnh NHỌN khi α < 1, TRƠN khi α = 1\n"
                "đỉnh nhọn là thứ tạo ra hệ số bằng đúng 0", fontsize=9)
fig.tight_layout()
fig.savefig("enet_mien_phat.png", dpi=130, bbox_inches="tight", facecolor="white")

# ----------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})

h3_lasso_s = np.array([0. , 0.189317, 0.254388, 0.37237 , 0.376468, 0.557045, 0.599204, 0.943302, 1. ])
h3_lasso_b = np.array([[ 0. , 3.4767, 4.0742, 4.5577, 4.5672, 4.7094, 4.7642, 5.6818, 5.8201], [ 0. , 0. , 0.5975, 1.5255, 1.5353, 1.9954, 2.0898, 2.3641, 2.3774], [ 0. , 0. , 0. , 0. , 0. , 0. , -0.2607, -1.0865, -1.1581], [ 0. , 0. , 0. , 0. , 0.0291, 1.1658, 1.3312, 1.6754, 1.7224], [ 0. , 0. , 0. , 0.7552, 0.782 , 1.6272, 1.6917, 2.4397, 2.5154], [ 0. , 0. , 0. , 0. , 0. , 0. , 0. , -2.0835, -2.3479], [ 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , -0.1699], [ 0. , 0. , 0. , 0. , 0. , 0.7318, 0.8664, 1.9921, 2.2532]])
h3_enet_s = np.array([0. , 0.049857, 0.088162, 0.091433, 0.133553, 0.282815, 0.41681 , 0.487238, 1. ])
h3_enet_b = np.array([[0. , 1.7306, 2.3954, 2.4333, 2.7987, 3.8346, 4.6094, 4.9584, 7.1816], [0. , 0. , 0. , 0. , 0.3658, 1.403 , 2.1793, 2.5288, 4.755 ], [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 2.2254], [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0.3498, 2.5781], [0. , 0. , 0.6648, 0.7027, 1.0681, 2.1041, 2.8792, 3.2284, 5.453 ], [0. , 0. , 0. , 0.0378, 0.4033, 1.439 , 2.2138, 2.5628, 4.7862], [0. , 0. , 0. , 0. , 0. , 0. , 0.7753, 1.1245, 3.3487], [0. , 0. , 0. , 0. , 0. , 1.0362, 1.8112, 2.1602, 4.3837]])
h3_ten = ["lcavol", "lweight", "age", "lbph", "svi", "lcp", "gleason", "pgg45"]

fig, ax = plt.subplots(1, 2, figsize=(10.6, 4.4), sharey=True)
mau = plt.cm.tab10(np.linspace(0, 1, 10))
for k, (ss, bb, ten) in enumerate([(h3_lasso_s, h3_lasso_b, "(a) lasso"),
                                   (h3_enet_s, h3_enet_b, "(b) elastic net, λ = 1000")]):
    for j in range(8):
        ax[k].plot(ss, bb[j], "o-", color=mau[j], lw=1.4, ms=3.0)
    cuoi = bb[:, -1]
    tt = np.argsort(cuoi)
    vt = cuoi[tt].astype(float).copy()
    for i in range(1, 8):
        if vt[i] - vt[i - 1] < 0.52:
            vt[i] = vt[i - 1] + 0.52
    for i, j in enumerate(tt):
        ax[k].plot([1.0, 1.045], [cuoi[j], vt[i]], color=mau[j], lw=0.6, alpha=0.6)
        ax[k].text(1.055, vt[i], h3_ten[j], fontsize=7.2, color=mau[j], va="center")
    ax[k].axhline(0, color="#111827", lw=0.6)
    ax[k].set_xlabel(r"$s = \|\beta\|_1 / \max\|\beta\|_1$")
    ax[k].set_title(ten, fontsize=9)
    ax[k].set_xlim(-0.02, 1.34)
ax[0].set_ylabel("hệ số đã chuẩn hoá")
fig.suptitle("Dựng lại Hình 3 của Zou & Hastie 2005 — 67 mẫu huấn luyện của bộ prostate, bản CÓ LỖI lweight",
             fontsize=9.5)
fig.tight_layout()
fig.savefig("enet_prostate_duong.png", dpi=130, bbox_inches="tight", facecolor="white")

# ----------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})

cvte_l2 = np.array([ 0. , 0.01, 0.1 , 1. , 10. , 100. , 1000. ])
cvte_cv = np.array([0.579405, 0.579039, 0.58081 , 0.625034, 0.6873 , 0.699193, 0.700434])
cvte_te = np.array([0.553645, 0.552343, 0.507296, 0.448845, 0.38433 , 0.383545, 0.380516])
cvte_s = np.array([0.87, 0.88, 0.66, 0.62, 0.32, 0.26, 0.26])

x = np.arange(len(cvte_l2))
fig, ax = plt.subplots(figsize=(7.6, 4.4))
ax.plot(x, cvte_cv, "o-", color="#2563eb", lw=1.8, ms=5, label="sai số kiểm định chéo 10 mảnh (67 mẫu huấn luyện)")
ax.plot(x, cvte_te, "s-", color="#dc2626", lw=1.8, ms=5, label="sai số trên 30 mẫu KIỂM TRA")
for i in range(len(x)):
    ax.text(x[i], cvte_cv[i] + 0.013, "%.3f" % cvte_cv[i], fontsize=7, color="#2563eb", ha="center")
    ax.text(x[i], cvte_te[i] - 0.023, "%.3f" % cvte_te[i], fontsize=7, color="#dc2626", ha="center")
ax.axvline(int(np.argmin(cvte_cv)), color="#2563eb", ls="--", lw=1.0)
ax.axvline(int(np.argmin(cvte_te)), color="#dc2626", ls="--", lw=1.0)
ax.set_xticks(x)
ax.set_xticklabels(["0", "0,01", "0,1", "1", "10", "100", "1000"])
ax.set_xlabel(r"$\lambda_2$  (lưới §3.5 của bài báo là sáu giá trị đầu; 1000 nằm NGOÀI lưới đó)")
ax.set_ylabel("sai số bình phương trung bình")
ax.set_ylim(min(cvte_te) - 0.055, max(cvte_cv) + 0.055)
ax.legend(fontsize=8, loc="center right")
ax.set_title("Hai đường đi NGƯỢC nhau: thủ tục §3.5 chọn λ₂ = 0,01 (vạch xanh),\n"
             "còn Bảng 1 báo cáo λ = 1000 (vạch đỏ) — điểm tốt nhất trên tập kiểm tra", fontsize=9)
fig.tight_layout()
fig.savefig("enet_cv_vs_test.png", dpi=130, bbox_inches="tight", facecolor="white")

# ----------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})

b2_do = np.array([[ 3.1625, 3.3414, 46.2159, 59.6357], [ 2.9464, 3.2177, 34.8338, 32.1307], [ 3.894 , 2.0785, 23.414 , 74.3379], [ 2.9804, 2.3614, 24.3505, 56.9031]])
b2_se = np.array([[0.5145, 0.3354, 1.4841, 9.0214], [0.5069, 0.2171, 1.7975, 6.4702], [0.6044, 0.1935, 1.1252, 6.1998], [0.52 , 0.2022, 1.5188, 7.602 ]])
b2_in = np.array([[ 3.06, 3.87, 65. , 46.6 ], [ 2.51, 3.16, 56.6 , 34.5 ], [ 4.49, 2.84, 39.5 , 64.5 ], [ 5.7 , 2.73, 41. , 45.9 ]])
ten_pp = ["Lasso", "Elastic net", "Ridge", "Naive EN"]
mau = ["#dc2626", "#059669", "#2563eb", "#b45309"]

fig, ax = plt.subplots(1, 4, figsize=(13.2, 3.8))
for v in range(4):
    x = np.arange(4)
    ax[v].bar(x - 0.2, b2_in[:, v], 0.38, color=mau, alpha=0.45, label="bài báo in")
    ax[v].bar(x + 0.2, b2_do[:, v], 0.38, color=mau,
              yerr=b2_se[:, v], capsize=3, label="dựng lại")
    ax[v].set_xticks(x)
    ax[v].set_xticklabels(ten_pp, rotation=30, ha="right", fontsize=7.6)
    ax[v].set_title("Ví dụ %d" % (v + 1), fontsize=9)
    if v == 0:
        ax[v].set_ylabel("trung vị sai số mô hình")
        ax[v].legend(fontsize=7.4)
fig.suptitle("Bảng 2 của Zou & Hastie 2005 — cột nhạt là số in, cột đậm là bản dựng lại "
             "(50 lần lặp, thanh sai số là SE bootstrap B = 500)", fontsize=9.5)
fig.tight_layout()
fig.savefig("enet_bang2.png", dpi=130, bbox_inches="tight", facecolor="white")

# ----------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})

h5_lasso_s = np.array([0. , 0.162485, 0.484021, 0.598433, 0.747858, 0.967402, 1. ])
h5_lasso_b = np.array([[ 0. , 13.9606, 27.7738, 31.1637, 41.0485, 41.3486, 41.5049], [ 0. , 0. , -13.8132, -17.2663, -6.87 , -7.5914, -7.681 ], [ 0. , 0. , 0. , 2.9872, 9.8685, 8.8656, 8.6221], [ 0. , 0. , 0. , 0. , 0. , -9.4263, -10.2039], [ 0. , 0. , 0. , 0. , 0. , 0. , 0.6223], [ 0. , 0. , 0. , 0. , 6.4687, 15.8871, 17.2855]])
h5_enet_s = np.array([0. , 0.001181, 0.002702, 0.755778, 0.758222, 0.762535, 1. ])
h5_enet_b = np.array([[ 0. , 0.0998, 0.164 , 21.3797, 21.4197, 21.484 , 24.8849], [ 0. , 0. , -0.0643, -21.2821, -21.3208, -21.3825, -24.6347], [ 0. , 0. , 0. , 21.2053, 21.2451, 21.309 , 24.6894], [ 0. , 0. , 0. , 0. , 0. , 0. , 3.3459], [ 0. , 0. , 0. , 0. , 0. , -0.0873, -3.4307], [ 0. , 0. , 0. , 0. , 0.088 , 0.1754, 3.5196]])

fig, ax = plt.subplots(1, 2, figsize=(10.6, 4.2), sharey=True)
mau = ["#dc2626", "#ea580c", "#b45309", "#2563eb", "#0891b2", "#0e7490"]
for k, (ss, bb, ten) in enumerate([(h5_lasso_s, h5_lasso_b, "(a) lasso"),
                                   (h5_enet_s, h5_enet_b, "(b) elastic net, λ₂ = 0,5")]):
    for j in range(6):
        ax[k].plot(ss, bb[j], "o-", color=mau[j], lw=1.5, ms=3.2, label="$x_%d$" % (j + 1))
    cuoi = bb[:, -1]
    tt = np.argsort(cuoi)
    vt = cuoi[tt].astype(float).copy()
    for i in range(1, 6):                       # đẩy nhãn ra xa nhau, không cho đè
        if vt[i] - vt[i - 1] < 2.6:
            vt[i] = vt[i - 1] + 2.6
    for i, j in enumerate(tt):
        ax[k].text(1.03, vt[i], "%d" % (j + 1), fontsize=7.6, color=mau[j], va="center")
    ax[k].axhline(0, color="#111827", lw=0.6)
    ax[k].set_xlabel(r"$s = \|\beta\|_1 / \max\|\beta\|_1$")
    ax[k].set_title(ten, fontsize=9)
    ax[k].set_xlim(-0.02, 1.10)
ax[0].set_ylabel("hệ số đã chuẩn hoá")
ax[0].legend(fontsize=7.4, ncol=2, loc="lower left")
fig.suptitle("Dựng lại Hình 5 của Zou & Hastie 2005 — $x_1,x_2,x_3$ sinh từ $Z_1$; $x_4,x_5,x_6$ từ $Z_2$",
             fontsize=9.5)
fig.tight_layout()
fig.savefig("enet_hai_nhom.png", dpi=130, bbox_inches="tight", facecolor="white")
