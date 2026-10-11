# -*- coding: utf-8 -*-
"""Sinh hình cho bài "Least Angle Regression (LARS)".

KHÔNG sửa file này bằng tay: nó được sinh ra từ chính khối code in trong bài, bằng
    python extract_fig.py "Least Angle Regression (LARS)" fig_lars.py
Muốn đổi hình thì sửa khối code TRONG BÀI rồi chạy lại lệnh trên và regen.py.
"""
import os

os.chdir(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "AI algorithms", "Machine Learning", "Least Angle Regression (LARS)"))

import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import nnls
from sklearn.datasets import load_diabetes

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})

# chuẩn hoá đúng quy ước (1.1) tr.3: cột trừ trung bình rồi chia cho CHUẨN EUCLID
d = load_diabetes(scaled=False)
X = d.data - d.data.mean(0)
X = X / np.sqrt((X ** 2).sum(0))
y = d.target - d.target.mean()
def huong(X, A, c):                              # (2.4)–(2.6), (2.11)
    s = np.sign(c[A]); s[s == 0] = 1.0
    XA = X[:, A] * s
    iG = np.linalg.solve(XA.T @ XA, np.ones(len(A)))
    AA = 1.0 / np.sqrt(np.ones(len(A)) @ iG)
    wA = AA * iG
    uA = XA @ wA
    return s, AA, wA, uA, X.T @ uA


def lars(X, y, kieu="lar", eps=1e-12):
    """LARS của Efron 2004 theo đúng (2.4)–(2.13); kieu = lar | lasso | stage."""
    n, m = X.shape
    beta, mu, A, B, bo_ky = np.zeros(m), np.zeros(n), [], [np.zeros(m)], []
    vua_bo = -1
    while True:
        c = X.T @ (y - mu)
        if A and max(abs(c[j]) for j in A) < 1e-9 * np.abs(X.T @ y).max():
            break
        ngoai = [j for j in range(m) if j not in A and j != vua_bo]
        if ngoai:
            jb = max(ngoai, key=lambda j: abs(c[j]))
            if not A or abs(c[jb]) >= max(abs(c[j]) for j in A) - 1e-9:
                A = A + [jb]
        vua_bo = -1
        if not A:
            break
        s, AA, wA, uA, a = huong(X, A, c)
        C = max(abs(c[j]) for j in A)
        if kieu == "stage":                      # chiếu u_A vào nón lồi (3.12)
            P = nnls(X[:, A] * s, uA)[0]
            Bh = [A[i] for i in range(len(A)) if P[i] > 1e-10]
            if len(Bh) < len(A):
                bo_ky.append((len(B), [A[i] + 1 for i in range(len(A)) if P[i] <= 1e-10]))
            s, AA, wA, uA, a = huong(X, Bh, c)
            A = Bh
        con = [j for j in range(m) if j not in A]
        ung = [t / mm for j in con
               for t, mm in ((C - c[j], AA - a[j]), (C + c[j], AA + a[j]))
               if mm > eps and t / mm > eps]     # (2.13)
        gam = min(ung) if ung else C / AA
        d_ = np.zeros(m); d_[A] = s * wA         # (3.3)
        bo = None
        if kieu == "lasso":
            gt, jt = np.inf, None
            for j in A:
                if abs(d_[j]) > eps:
                    g = -beta[j] / d_[j]         # (3.4)–(3.5)
                    if eps < g < gt:
                        gt, jt = g, j
            if gt < gam:
                gam, bo = gt, jt                 # (3.6)
        beta = beta + gam * d_
        mu = mu + gam * uA
        if bo is not None:
            beta[bo] = 0.0
            A = [j for j in A if j != bo]
            bo_ky.append((len(B), [bo + 1]))
            vua_bo = bo
        B.append(beta.copy())
        if not con and bo is None:
            break
    return np.array(B).T, bo_ky


B = lars(X, y, "lasso")[0]
K = B.shape[1]

C = np.array([np.abs(X.T @ (y - X @ B[:, k])) for k in range(K)])     # K × 10
trong = np.array([np.abs(B[:, k]) > 1e-12 for k in range(K)])         # K × 10
SAN = 0.3                                        # sàn để vẽ được giá trị 0 trên trục log

fig, ax = plt.subplots(figsize=(10.6, 4.8))
for j in range(10):
    v = np.maximum(C[:, j], SAN)
    mau = plt.cm.tab10(j % 10)
    ax.plot(range(K), v, lw=1.0, alpha=.45, color="#9ca3af", zorder=1)
    on, off = trong[:, j], ~trong[:, j]
    ax.plot(np.arange(K)[on], v[on], "o", ms=6.2, zorder=3, color=mau)
    ax.plot(np.arange(K)[off], v[off], "o", ms=6.2, zorder=2,
            mfc="white", mec=mau, mew=1.1)
    kv = int(np.argmax(on))
    ax.annotate(str(j + 1), (kv, v[kv]), fontsize=7.5, fontweight="bold", color=mau,
                xytext=(0, 9), textcoords="offset points", ha="center", zorder=5)

do_trai = max(np.ptp(C[k, trong[k]]) for k in range(1, K) if trong[k].any())
tran = np.array([C[k, trong[k]].max() if trong[k].any() else C[k].max() for k in range(K)])
ax.plot(range(K), np.maximum(tran, SAN), "k--", lw=1.3, zorder=4,
        label=r"trần $\hat C_k$ — giá trị chung của mọi biến hoạt động")
ax.axvline(10, color="#ea580c", lw=1.4, ls=":", zorder=4)
ax.annotate("bước 10: (3.6) bỏ biến 7\nra khỏi tập hoạt động", (10, 3.0),
            xytext=(-14, 34), textcoords="offset points", fontsize=8,
            color="#ea580c", ha="right",
            arrowprops=dict(arrowstyle="->", color="#ea580c", lw=1.0))
ax.plot([], [], "o", ms=6.2, color="#374151", label="chấm ĐẶC = biến đang hoạt động")
ax.plot([], [], "o", ms=6.2, mfc="white", mec="#374151", mew=1.1,
        label="chấm RỖNG = biến ở ngoài")
ax.set_yscale("log")
ax.set_ylim(SAN * 0.8, C.max() * 2.6)
ax.set_xticks(range(K))
ax.set_xlabel("bước của đường lasso (LARS có sửa đổi (3.6))")
ax.set_ylabel(r"$|x_j^{\top}(y - X\hat\beta)|$")
ax.legend(fontsize=8, loc="lower left", ncol=1)
ax.set_title("Bất biến cách đều góc, đo trên chính bộ diabetes của bài báo\n"
             "mọi chấm đặc ở cùng một bước CHỒNG KHÍT lên nhau — độ trải lớn nhất "
             "%.2e\n" % do_trai +
             "biến ngoài không bao giờ vượt trần, và vào đúng lúc nó CHẠM trần",
             fontsize=9.5)
fig.savefig("lars_bat_bien.png", dpi=130, bbox_inches="tight", facecolor="white")
plt.close(fig)

for k in range(1, K):
    c = C[k, trong[k]]
    ngoai = C[k, ~trong[k]]
    assert c.max() - c.min() < 1e-9, "bất biến vỡ ở bước %d" % k
    assert len(ngoai) == 0 or ngoai.max() <= c.max() + 1e-9, "biến ngoài vượt trần ở bước %d" % k
print("bất biến đúng ở cả %d bước; độ trải lớn nhất %.2e"
      % (K - 1, max(np.ptp(C[k, trong[k]]) for k in range(1, K) if trong[k].any())))

# ----------------------------------------------------------------------

import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import nnls
from sklearn.datasets import load_diabetes

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})

d = load_diabetes(scaled=False)
X = d.data - d.data.mean(0)
X = X / np.sqrt((X ** 2).sum(0))
y = d.target - d.target.mean()


def huong(X, A, c):                              # (2.4)–(2.6), (2.11)
    s = np.sign(c[A]); s[s == 0] = 1.0
    XA = X[:, A] * s
    iG = np.linalg.solve(XA.T @ XA, np.ones(len(A)))
    AA = 1.0 / np.sqrt(np.ones(len(A)) @ iG)
    wA = AA * iG
    uA = XA @ wA
    return s, AA, wA, uA, X.T @ uA


def lars(X, y, kieu="lar", eps=1e-12):
    """LARS của Efron 2004 theo đúng (2.4)–(2.13); kieu = lar | lasso | stage."""
    n, m = X.shape
    beta, mu, A, B, bo_ky = np.zeros(m), np.zeros(n), [], [np.zeros(m)], []
    vua_bo = -1
    while True:
        c = X.T @ (y - mu)
        if A and max(abs(c[j]) for j in A) < 1e-9 * np.abs(X.T @ y).max():
            break
        ngoai = [j for j in range(m) if j not in A and j != vua_bo]
        if ngoai:
            jb = max(ngoai, key=lambda j: abs(c[j]))
            if not A or abs(c[jb]) >= max(abs(c[j]) for j in A) - 1e-9:
                A = A + [jb]
        vua_bo = -1
        if not A:
            break
        s, AA, wA, uA, a = huong(X, A, c)
        C = max(abs(c[j]) for j in A)
        if kieu == "stage":                      # chiếu u_A vào nón lồi (3.12)
            P = nnls(X[:, A] * s, uA)[0]
            Bh = [A[i] for i in range(len(A)) if P[i] > 1e-10]
            if len(Bh) < len(A):
                bo_ky.append((len(B), [A[i] + 1 for i in range(len(A)) if P[i] <= 1e-10]))
            s, AA, wA, uA, a = huong(X, Bh, c)
            A = Bh
        con = [j for j in range(m) if j not in A]
        ung = [t / mm for j in con
               for t, mm in ((C - c[j], AA - a[j]), (C + c[j], AA + a[j]))
               if mm > eps and t / mm > eps]     # (2.13)
        gam = min(ung) if ung else C / AA
        d_ = np.zeros(m); d_[A] = s * wA         # (3.3)
        bo = None
        if kieu == "lasso":
            gt, jt = np.inf, None
            for j in A:
                if abs(d_[j]) > eps:
                    g = -beta[j] / d_[j]         # (3.4)–(3.5)
                    if eps < g < gt:
                        gt, jt = g, j
            if gt < gam:
                gam, bo = gt, jt                 # (3.6)
        beta = beta + gam * d_
        mu = mu + gam * uA
        if bo is not None:
            beta[bo] = 0.0
            A = [j for j in A if j != bo]
            bo_ky.append((len(B), [bo + 1]))
            vua_bo = bo
        B.append(beta.copy())
        if not con and bo is None:
            break
    return np.array(B).T, bo_ky


fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.9), sharey=True)
for ax, kieu, ten in zip(axes, ("lar", "lasso", "stage"),
                         ("LARS gốc — mục 2\n(Hình 3 bảng trái)",
                          "Lasso — sửa đổi (3.6)\n(Hình 1 bảng trái)",
                          "Stagewise — chiếu vào nón lồi\n(Hình 1 bảng phải)")):
    B, bo_ky = lars(X, y, kieu)
    t = np.abs(B).sum(0)
    for j in range(10):
        ax.plot(t, B[j], lw=1.5, color=plt.cm.tab10(j % 10))
        ax.annotate(str(j + 1), (t[-1], B[j, -1]), fontsize=7.5, fontweight="bold",
                    color=plt.cm.tab10(j % 10), xytext=(4, -3), textcoords="offset points")
    for k, js in bo_ky:
        ax.annotate("bỏ %s" % ",".join(str(v) for v in js), (t[k], 0),
                    xytext=(0, -72), textcoords="offset points", fontsize=8,
                    color="#ea580c", ha="center",
                    bbox=dict(boxstyle="round,pad=0.22", fc="white",
                              ec="#ea580c", lw=0.7),
                    arrowprops=dict(arrowstyle="->", color="#ea580c", lw=1.0))
    ax.axhline(0, color="#6b7280", lw=0.8)
    ax.set_xlim(-60, t[-1] * 1.09)
    ax.set_title("%s\n%d bước" % (ten, B.shape[1] - 1), fontsize=9.5)
    ax.set_xlabel(r"$t = \sum_j |\hat\beta_j|$")
axes[0].set_ylabel(r"hệ số $\hat\beta_j$")
fig.suptitle("Ba đường nghiệm trên bộ diabetes — cùng một điểm đầu, cùng một điểm cuối, "
             "khác nhau ở giữa\nsố bước 10 · 12 · 13 khớp đúng ba con số bài báo nêu "
             "ở tr.9, tr.12 và tr.14", fontsize=10.5, y=1.10)
fig.savefig("lars_ba_duong.png", dpi=130, bbox_inches="tight", facecolor="white")
plt.close(fig)

for kieu, doi in (("lar", 10), ("lasso", 12), ("stage", 13)):
    B, _ = lars(X, y, kieu)
    assert B.shape[1] - 1 == doi, (kieu, B.shape[1] - 1)
    assert np.abs(B[:, -1] - np.linalg.lstsq(X, y, rcond=None)[0]).max() < 1e-8
print("ba đường: 10, 12, 13 bước; cả ba kết thúc đúng nghiệm OLS")

# ----------------------------------------------------------------------

import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import nnls
from sklearn.datasets import load_diabetes

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})

d = load_diabetes(scaled=False)
X = d.data - d.data.mean(0)
X = X / np.sqrt((X ** 2).sum(0))
y = d.target - d.target.mean()


def stagewise_eps(X, y, eps, t_dich, tran=400000):
    """(1.7) — cập nhật ĉ bằng ma trận Gram nên mỗi bước chỉ tốn O(m), không O(nm)."""
    G = X.T @ X
    c = X.T @ y
    b = np.zeros(X.shape[1])
    duong = [b.copy()]
    for i in range(tran):
        j = int(np.argmax(np.abs(c)))
        s = np.sign(c[j])
        b[j] += eps * s
        c -= eps * s * G[:, j]
        duong.append(b.copy())
        if np.abs(b).sum() >= t_dich:
            return np.array(duong).T, i + 1
    return np.array(duong).T, tran


def huong(X, A, c):
    s = np.sign(c[A]); s[s == 0] = 1.0
    XA = X[:, A] * s
    iG = np.linalg.solve(XA.T @ XA, np.ones(len(A)))
    AA = 1.0 / np.sqrt(np.ones(len(A)) @ iG)
    return s, AA, AA * iG, XA @ (AA * iG), X.T @ (XA @ (AA * iG))


def lars_stage(X, y, eps=1e-12):
    """Sửa đổi Stagewise: thay u_A bằng hình chiếu của nó vào nón lồi C_A, (3.12)."""
    n, m = X.shape
    beta, mu, A, B = np.zeros(m), np.zeros(n), [], [np.zeros(m)]
    while True:
        c = X.T @ (y - mu)
        if A and max(abs(c[j]) for j in A) < 1e-9 * np.abs(X.T @ y).max():
            break
        ngoai = [j for j in range(m) if j not in A]
        if ngoai:
            jb = max(ngoai, key=lambda j: abs(c[j]))
            if not A or abs(c[jb]) >= max(abs(c[j]) for j in A) - 1e-9:
                A = A + [jb]
        if not A:
            break
        s, AA, wA, uA, a = huong(X, A, c)
        P = nnls(X[:, A] * s, uA)[0]
        A = [A[i] for i in range(len(A)) if P[i] > 1e-10]
        s, AA, wA, uA, a = huong(X, A, c)
        C = max(abs(c[j]) for j in A)
        con = [j for j in range(m) if j not in A]
        ung = [t / mm for j in con
               for t, mm in ((C - c[j], AA - a[j]), (C + c[j], AA + a[j]))
               if mm > eps and t / mm > eps]
        gam = min(ung) if ung else C / AA
        dd = np.zeros(m); dd[A] = s * wA
        beta = beta + gam * dd
        mu = mu + gam * uA
        B.append(beta.copy())
        if not con:
            break
    return np.array(B).T


b_ols = np.linalg.lstsq(X, y, rcond=None)[0]
t_het = np.abs(b_ols).sum()
B_id = lars_stage(X, y)
t_id = np.abs(B_id).sum(0)

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.6, 4.8))
mau = {4.0: "#fca5a5", 1.0: "#f97316", 0.25: "#2563eb"}
for eps in (4.0, 1.0, 0.25):
    D, _ = stagewise_eps(X, y, eps, 0.62 * t_het)
    tt = np.abs(D).sum(0)
    ax.plot(tt, D[2], lw=1.2, color=mau[eps], label=r"$\varepsilon = %g$" % eps)
    ax.plot(tt, D[8], lw=1.2, color=mau[eps])
ax.plot(t_id, B_id[2], "k--", lw=1.8, label="đường lý tưởng (Định lý 2)")
ax.plot(t_id, B_id[8], "k--", lw=1.8)
ax.set_xlim(0, 0.62 * t_het)
ax.set_ylim(-20, 620)
ax.annotate("biến 3", (0.60 * t_het, B_id[2][np.argmin(np.abs(t_id - 0.60 * t_het))]),
            xytext=(-42, 10), textcoords="offset points", fontsize=8.5, fontweight="bold")
ax.annotate("biến 9", (0.60 * t_het, B_id[8][np.argmin(np.abs(t_id - 0.60 * t_het))]),
            xytext=(-42, -16), textcoords="offset points", fontsize=8.5, fontweight="bold")
ax.set_xlabel(r"$t = \sum_j |\hat\beta_j|$"); ax.set_ylabel(r"hệ số $\hat\beta_j$")
ax.set_title("Bước ε càng nhỏ, bậc thang càng phẳng ra thành đường lý tưởng\n"
             "(đây là thứ 'staircase' trong Hình 2 tr.6 vẽ bằng tay)", fontsize=9.5)
ax.legend(fontsize=8, loc="upper left")

# So tại CÙNG MỘT t cho mọi ε: điểm dừng của mỗi lần chạy rơi vào một t khác nhau,
# và so ở những t khác nhau thì dãy sai số không còn đơn điệu theo ε — lỗi của THƯỚC ĐO,
# không phải của thuật toán.
T_SO = 0.5 * t_het
eps_l, le_l, buoc_l = [], [], []
for eps in (8.0, 4.0, 2.0, 1.0, 0.5, 0.25, 0.125, 0.0625):
    bb, kk = stagewise_eps(X, y, eps, T_SO * 1.02)
    t_bb = np.abs(bb).sum(0)
    tai_T = np.array([np.interp(T_SO, t_bb, bb[j]) for j in range(10)])
    ref = np.array([np.interp(T_SO, t_id, B_id[j]) for j in range(10)])
    eps_l.append(eps); le_l.append(np.abs(tai_T - ref).max()); buoc_l.append(kk)
eps_l = np.array(eps_l); le_l = np.array(le_l)
tho = eps_l >= 4.0                               # hai giá trị ε quá thô, tách riêng
ax2.loglog(eps_l[~tho], le_l[~tho], "o-", ms=6, lw=1.6, color="#7c3aed",
           label=r"lệch đo được, $\varepsilon \leq 2$")
ax2.loglog(eps_l[tho], le_l[tho], "x", ms=9, mew=2.0, color="#dc2626",
           label=r"$\varepsilon \geq 4$: bước THÔ hơn cả một đoạn, phép so mất nghĩa")
r_ = eps_l[~tho]
ax2.loglog(r_, r_ * le_l[~tho][0] / r_[0], "--", lw=1.4, color="#6b7280",
           label=r"đường tham chiếu tỉ lệ THUẬN với $\varepsilon$")
for e, l, k in zip(eps_l, le_l, buoc_l):
    ax2.annotate("%d bước" % k, (e, l), xytext=(6, -11), textcoords="offset points",
                 fontsize=7.5, color="#4b5563")
ax2.set_xlabel(r"bước $\varepsilon$"); ax2.set_ylabel(r"lệch với đường lý tưởng tại $t$ cố định")
ax2.set_title("Với ε ≤ 2 sai số giảm TỈ LỆ THUẬN với ε, còn số bước tăng tỉ lệ NGHỊCH\n"
              "muốn mượt gấp đôi thì trả giá gấp đôi, mãi mãi", fontsize=9.5)
ax2.legend(fontsize=8, loc="upper left")
fig.savefig("lars_eps.png", dpi=130, bbox_inches="tight", facecolor="white")
plt.close(fig)

print("số bước ứng với ε = %s : %s" % (eps_l, buoc_l))
print("lệch ứng với ε = %s" % ["%.4f" % v for v in le_l])
print("tỉ số lệch/ε : %s" % ["%.4f" % (l / e) for e, l in zip(eps_l, le_l)])

# ----------------------------------------------------------------------

import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})

# Trung bình trên 100 lần lặp của mô hình (3.15)–(3.16); `tools/v_lars.py` mục 23 in ra.
pe_lar = [0.0, 0.16137326080068815, 0.7410025520685758, 0.8867110557927856, 0.9350087550592083,
          0.9499846647152862, 0.9576921960697135, 0.961458250035188, 0.9636565167276252,
          0.9655683013856813, 0.9663587533276058, 0.9664215946576782, 0.9661503758712067,
          0.9654225353654488, 0.9640054497548314, 0.9626560941512722, 0.9613381404251197,
          0.9596821500093218, 0.9578209438065511, 0.9555312256565532, 0.953829856691638,
          0.9516619196783792, 0.9493813851697237, 0.9473947158962684, 0.944808642708031,
          0.942532379020172, 0.9399211919494506, 0.9376799909928732, 0.9347056063648586,
          0.9322551935841894, 0.92888660295163, 0.9262251093004045, 0.9234794153767786,
          0.9200382777157533, 0.917162824402744, 0.9142059979443059, 0.9113782309499264,
          0.9090176699056954, 0.90635004834155, 0.9029951292396702, 0.8998684502813599]
pe_lasso = [0.0, 0.16137326080068815, 0.7410025520685758, 0.8867110557927856, 0.9350087550592083,
            0.9499846647152862, 0.9576921960697135, 0.9614400803090214, 0.9636134646553534,
            0.9655355499950666, 0.9662947328926659, 0.9664044544782391, 0.9661602846741324,
            0.9654263280071708, 0.9640803682703529, 0.9627765528151508, 0.9614611230281151,
            0.9599259226388019, 0.95816009017857, 0.9561762196016724, 0.9544014720077297,
            0.9528017238420513, 0.950522618801289, 0.9484176370486099, 0.9463630446934185,
            0.9442225225666627, 0.9420607607927226, 0.9396949756693962, 0.9375758627701591,
            0.935133210173877, 0.9328533035033751, 0.9305242608093822, 0.9280701859746268,
            0.9256971715638999, 0.9232161351710918, 0.9206081881532522, 0.9182239371560565,
            0.9161123746440438, 0.913886500003819, 0.9113732428732785, 0.9093210906935684]
pe_stage = [0.0, 0.16137326080068815, 0.7410025520685758, 0.8867110557927856, 0.9350087550592083,
            0.9499840971226687, 0.9576912805244342, 0.9614686599091726, 0.9635251843438235,
            0.9655947769729758, 0.9663695466621641, 0.9665137898860617, 0.9663243113102569,
            0.9657666868765616, 0.9645333747845557, 0.9635824702787835, 0.9621141891420484,
            0.9607368508067836, 0.9594665228566375, 0.9578473015979266, 0.9561974147949506,
            0.9546738175677503, 0.9528911186828534, 0.9511996861095068, 0.9493604565502447,
            0.9474698940539311, 0.9454835037929022, 0.9433090955182284, 0.9414471887306022,
            0.9391338237569948, 0.937429632502935, 0.9353875743740392, 0.9334003400713742,
            0.9310364538337607, 0.9285904200248525, 0.9264561485738453, 0.9240994742513574,
            0.921777964408827, 0.9198064748549973, 0.9172320116141034, 0.9150617103413135]
pe_fs = [0.0, 0.6972223593296433, 0.9425632100025563, 0.9560087865983492, 0.950345383351585,
         0.9446973008912662, 0.9357902713259486, 0.9299300087723165, 0.9222093416690079,
         0.915919906000439, 0.9101383405292719, 0.9048849248621356, 0.8991327651703804,
         0.8939620571742349, 0.8898345391015936, 0.8860798767522493, 0.8823185757173988,
         0.8788436064135609, 0.8754819561475565, 0.8725808590290292, 0.8698701757653601,
         0.8670366043797304, 0.8642586531316633, 0.8618924310041476, 0.8601317550832228,
         0.8580451195277611, 0.8558467400971653, 0.8538709460558905, 0.8513724244952986,
         0.8493424429577768, 0.8476798551333508, 0.8457922216722076, 0.8441174907694274,
         0.8424221616512232, 0.8402995083679639, 0.838385814882017, 0.8368641685799239,
         0.8351872297567746, 0.8335009276490701, 0.8315077280578658, 0.8298674871053177]
sd_lar = [0.0, 0.10073285876923689, 0.10789147513603316, 0.054496970677128796,
          0.03538153852684438, 0.026091836456359953, 0.022508801690708034,
          0.021275547435047885, 0.01998331144348188, 0.017926525845322124,
          0.016769326684056857, 0.016119133769422176, 0.015906472970491848,
          0.015545471239661608, 0.01555847665084218, 0.015152927440251375,
          0.015058925907182295, 0.015307847666781218, 0.015328259659380443,
          0.015031126999257466, 0.015086454536766417, 0.015391092129780754,
          0.01604804399043123, 0.016081781838553355, 0.01615365841910121,
          0.016399082484252963, 0.016263111791931687, 0.016316186523168314,
          0.01679884702322542, 0.01724328117076876, 0.017398788320719942,
          0.017661327999642453, 0.018414299873111765, 0.018812274656518267,
          0.01922733707726031, 0.019758383389068847, 0.01986770799186052,
          0.02052066078224579, 0.02021624304918996, 0.021006755132725888,
          0.021922781236303087]

k = np.arange(41)
fig, ax = plt.subplots(figsize=(10.2, 5.2))
ax.plot(k, pe_lar, "-", lw=2.2, color="#1d4ed8", label="LARS")
ax.plot(k, pe_lasso, "--", lw=1.6, color="#059669", label="Lasso")
ax.plot(k, pe_stage, ":", lw=2.0, color="#ea580c", label="Stagewise")
ax.plot(k, pe_fs, "-", lw=2.2, color="#b91c1c", dashes=(6, 3),
        label="Forward Selection cổ điển")
ax.plot(k, np.array(pe_lar) + np.array(sd_lar), ".", ms=3.0, color="#93c5fd")
ax.plot(k, np.array(pe_lar) - np.array(sd_lar), ".", ms=3.0, color="#93c5fd")
ax.plot([], [], ".", ms=6, color="#93c5fd", label="± 1 độ lệch chuẩn của LARS")

kb = int(np.argmax(pe_lar))
ax.plot([kb], [pe_lar[kb]], "o", ms=10, mfc="none", mec="#1d4ed8", mew=2.0)
ax.annotate("LARS cao nhất %.4f tại k = %d\n(bài báo: 0,963 tại k = 10)"
            % (pe_lar[kb], kb), (kb, pe_lar[kb]), xytext=(34, -28),
            textcoords="offset points", fontsize=8.5, color="#1d4ed8",
            arrowprops=dict(arrowstyle="->", color="#1d4ed8", lw=1.0))
kf = int(np.argmax(pe_fs))
ax.plot([kf], [pe_fs[kf]], "o", ms=10, mfc="none", mec="#b91c1c", mew=2.0)
ax.annotate("Forward Selection cao nhất %.4f tại k = %d\nrồi TỤT nhanh hơn hẳn "
            "(bài báo: 0,950 tại k = 3)" % (pe_fs[kf], kf), (kf, pe_fs[kf]),
            xytext=(46, 24), textcoords="offset points", fontsize=8.5, color="#b91c1c",
            arrowprops=dict(arrowstyle="->", color="#b91c1c", lw=1.0))

ax.set_xlabel("số bước k")
ax.set_ylabel(r"tỉ lệ giải thích được  $\mathrm{pe}(\hat\mu) = 1 - \|\hat\mu-\mu\|^2/\|\mu\|^2$  (3.17)")
ax.set_ylim(0.60, 1.005)
ax.set_title("Dựng lại Hình 5 tr.15 — 100 lần lặp mô hình (3.15)–(3.16), m = 64\n"
             "ba thuật toán LARS · Lasso · Stagewise gần như KHÔNG phân biệt được nhau",
             fontsize=10)
ax.legend(fontsize=8.5, loc="lower left", ncol=2)
fig.savefig("lars_mo_phong.png", dpi=130, bbox_inches="tight", facecolor="white")
plt.close(fig)

print("LARS đỉnh %.6f tại k = %d ; Forward Selection đỉnh %.6f tại k = %d"
      % (max(pe_lar), int(np.argmax(pe_lar)), max(pe_fs), int(np.argmax(pe_fs))))
print("khoảng cách ba đường LARS/Lasso/Stagewise tại k = 40: %.6f"
      % (max(pe_lar[40], pe_lasso[40], pe_stage[40])
         - min(pe_lar[40], pe_lasso[40], pe_stage[40])))

# ----------------------------------------------------------------------

import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})

# `tools/v_lars.py` mục 12 và 23 in ra ba mảng này
rho = [0.06332340658933504, 0.4907357695538847, 0.3021055872663274, 0.5882932706030122,
       0.3177236980040751, 0.22323214020969445, 0.7102700478790898, 0.7258650199952871,
       0.0710721109996991, 0.9999999999999966]
ve_trai = [0.30174599602085383, 0.10822872766909852, 0.06274549583049804,
           0.0125705398921373, 0.01382712633977834, 0.008815043969504055,
           0.00117392379690473, 0.00015877310695899816, 0.0020305496913188836,
           -7.771561172376096e-16]
delta_R2 = [0.04217776420452635, 0.30907878776564135, 0.0660803826640356,
            0.061590873697595305, 0.015876628878104482, 0.005794679233685929,
            0.012810775005050457, 0.0019539765120778485, 0.00032260034371911495,
            0.0020619539159140166]

rho = np.array(rho); ve_trai = np.array(ve_trai); delta_R2 = np.array(delta_R2)
dung = (1 - rho) ** 2 / (rho * (2 - rho)) * delta_R2      # (1 − ρ)²  — suy từ Hình 4
nhu_in = (1 - rho ** 2) / (rho * (2 - rho)) * delta_R2    # 1 − ρ²    — bản in tr.16
k = np.arange(1, 11)

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.7))
w = 0.27
ax.bar(k - w, ve_trai, w, color="#111827", label="VẾ TRÁI, đo trực tiếp")
ax.bar(k, dung, w, color="#059669", label=r"$(1-\rho_k)^2 / [\rho_k(2-\rho_k)]$ — suy lại")
ax.bar(k + w, nhu_in, w, color="#dc2626",
       label=r"$(1-\rho_k^2) / [\rho_k(2-\rho_k)]$ — BẢN IN tr.16")
ax.set_yscale("symlog", linthresh=1e-4)
ax.set_xticks(k)
ax.set_xlabel("bước k"); ax.set_ylabel(r"$R^2(\bar\beta_k) - R^2(\hat\beta_k)$")
ax.set_title("Công thức (3.20): cột xanh CHỒNG KHÍT cột đen, cột đỏ thì không\n"
             "(trục dọc log đối xứng vì các giá trị trải qua bốn bậc)", fontsize=9.5)
ax.legend(fontsize=8, loc="upper right")

ti = (1 + rho) / (1 - rho)
r = np.linspace(0.0, 0.95, 300)
ax2.plot(r, (1 + r) / (1 - r), lw=2.0, color="#dc2626",
         label=r"hệ số sai của bản in: $(1+\rho)/(1-\rho)$")
ax2.plot(rho[:-1], ti[:-1], "o", ms=7, color="#111827", zorder=3,
         label="10 bước đo được trên bộ diabetes")
for i, dx, dy, ha in ((1, -10, 16, "right"), (6, -16, -34, "right"), (7, 16, 14, "left")):
    ax2.annotate("k = %d\nsai %.2f lần" % (i + 1, ti[i]), (rho[i], ti[i]),
                 xytext=(dx, dy), textcoords="offset points", fontsize=8,
                 ha=ha, color="#111827",
                 arrowprops=dict(arrowstyle="-", color="#9ca3af", lw=0.8))
ax2.axhline(1, color="#059669", lw=1.4, ls="--", label="bằng nhau (chỉ khi ρ = 0)")
ax2.set_xlabel(r"$\rho_k = \hat\gamma_k / \bar\gamma_k$")
ax2.set_ylabel("bản in / bản đúng")
ax2.set_ylim(0, 12)
ax2.set_title("Hai bản chỉ trùng nhau tại ρ = 0 và phân kỳ khi ρ → 1\n"
              "bước nào LARS đi gần hết tới OLS thì chỗ sai càng lớn", fontsize=9.5)
ax2.legend(fontsize=8, loc="upper left")
fig.savefig("lars_ct320.png", dpi=130, bbox_inches="tight", facecolor="white")
plt.close(fig)

assert np.abs(dung - ve_trai).max() < 1e-12, "bản (1−ρ)² KHÔNG khớp"
print("bản (1−ρ)² khớp vế trái tới %.2e trên cả 10 bước" % np.abs(dung - ve_trai).max())
print("bản in 1−ρ² sai lớn nhất %.2f lần (tại k = %d)"
      % (ti[:-1].max(), int(np.argmax(ti[:-1])) + 1))

# ----------------------------------------------------------------------

import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})

# Mọi mảng dưới đây do `tools/v_lars.py` mục 23 in ra ĐỦ CHỮ SỐ — dán thẳng vào đây
# để hình dựng lại được trong vài giây thay vì chạy lại 500 lần bootstrap.
df10 = [0.9626394119323122, 2.0838681560558876, 2.8964655750846005, 3.9745861473434734,
        4.8370680925234915, 5.756363075530417, 6.609884404884125, 7.586668098713742,
        8.697189067358309, 10.08328952369671]
se10 = [0.07267620788047585, 0.1303442562425645, 0.11687749484606942, 0.09950925774330967,
        0.09948264527166381, 0.07455407498332303, 0.12344261727610725, 0.14544018423521277,
        0.16488380072704717, 0.14552895413275152]
K64 = [1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46, 49, 52, 55, 58, 61, 64]
df64 = [0.9184276721576267, 4.123928132236446, 7.025050709352413, 10.130360982340857,
        12.94850254008435, 15.800020455549076, 18.709049540931012, 21.5429314261086,
        24.32115577239908, 26.964381637815688, 29.78909610102815, 32.57381709645876,
        35.27402767107544, 38.04927227526457, 40.63874798438391, 43.39121718170785,
        46.21065414381041, 49.15659331116208, 52.35200449334537, 55.73711384050036,
        59.20610967788954, 63.53479562130559]
se64 = [0.06241106851223841, 0.15529491040040283, 0.15441617200028973, 0.1761963272825003,
        0.1881440319456216, 0.154697142741991, 0.1936298925761718, 0.2399923571000519,
        0.2456561825993646, 0.2943351984918918, 0.29981191920806727, 0.28462659101162546,
        0.32112996526608145, 0.30160150385999035, 0.34027278726413, 0.39114219835225034,
        0.38359475694981965, 0.3868975161583985, 0.37631916388591385, 0.42079963147802435,
        0.42030423362016994, 0.3843495497036518]
Cp10 = [451.72439585243205, 416.0290990203355, 141.79784615368544, 84.74019607957268,
        31.69492969416592, 19.505599141908647, 16.326752944624047, 6.87745079283394,
        7.131134315067243, 8.842818517775129, 8.999999999999318]
Cp64 = [483.01779975642194, 446.00261711330563, 162.09923690295034, 102.97370672400143,
        62.25868206827761, 56.13264296495191, 55.89647101419581, 47.76315559871239,
        48.40136756892156, 46.83347339528689, 44.61053546415087, 24.83953936665324,
        23.063090355561314, 19.86085846009979, 18.52852042215892, 16.200264094219392,
        17.83438734594864, 19.090842719861712, 19.627720616513102, 21.134815060609924,
        23.081667027472406, 24.306476103023726, 26.083256325759976, 26.96446268605831,
        28.063302423480934, 29.820392723963266, 30.876810621534787, 31.15739715056145,
        32.75546874163234, 33.532437685059676, 35.519280866496956, 36.40322952040407,
        34.05995057686505, 33.15599172137149, 30.21386211337517, 27.310184697722605,
        29.24229864711873, 29.615836408942698, 29.359056995840604, 31.146134150827436,
        29.849755044675362, 31.708895805707584, 32.183116942471486, 34.04255693211883,
        35.984634279864906, 37.174051663381476, 38.21492279066115, 40.109036910672444,
        41.744035512056826, 42.18977320082627, 43.444470422921995, 45.29818487078222,
        46.346092826292136, 46.931674174243994, 48.529010395402224, 48.99854396574949,
        50.695394804752766, 52.48290882074565, 53.85712289809453, 55.730280801809386,
        57.06821731543931, 58.27349242405313, 60.16648726557992, 61.00352338952126,
        62.9999999999996]

T9 = 2.262                                       # t Student 9 bậc tự do, hai phía 95 %
fig, ax = plt.subplots(2, 2, figsize=(12.4, 8.0))

for a, K, df, se, ten in ((ax[0, 0], list(range(1, 11)), df10, se10, "m = 10 — bộ diabetes"),
                          (ax[0, 1], K64, df64, se64, "m = 64 — mô hình bậc hai (3.15)")):
    K = np.array(K, float); df = np.array(df); se = np.array(se)
    a.plot([0, K.max()], [0, K.max()], "-", lw=1.6, color="#059669",
           label=r"xấp xỉ đơn giản (4.9): $\mathrm{df} = k$")
    a.errorbar(K, df, yerr=T9 * se, fmt="o", ms=5, lw=1.1, capsize=3,
               color="#1d4ed8", label="bootstrap (4.6)–(4.8), B = 500")
    ngoai = np.abs(df - K) > T9 * se
    if ngoai.any():
        a.plot(K[ngoai], df[ngoai], "o", ms=11, mfc="none", mec="#dc2626", mew=1.8,
               label="k NẰM NGOÀI khoảng tin cậy 95 %")
    a.set_xlabel("số bước k"); a.set_ylabel(r"$\mathrm{df}(\hat\mu_k)$")
    a.set_title("%s — dựng lại Hình 6 tr.19\nlệch lớn nhất %.4f"
                % (ten, np.abs(df - K).max()), fontsize=9.5)
    a.legend(fontsize=8, loc="upper left")

for a, Cp, doi, ten in ((ax[1, 0], Cp10, 7, "m = 10"), (ax[1, 1], Cp64, 16, "m = 64")):
    Cp = np.array(Cp)
    k = int(np.argmin(Cp))
    a.plot(range(len(Cp)), Cp, "-o", ms=4, lw=1.4, color="#7c3aed")
    a.plot([k], [Cp[k]], "o", ms=11, mfc="none", mec="#059669", mew=2.0,
           label="cực tiểu đo được: k = %d" % k)
    a.plot([doi], [Cp[doi]], "s", ms=9, mfc="none", mec="#dc2626", mew=1.8,
           label="bài báo nói: k = %d" % doi)
    a.set_xlabel("số bước k"); a.set_ylabel(r"$C_p(\hat\mu_k)$")
    a.set_ylim(Cp.min() - 6, min(Cp.max(), Cp[max(0, k - 6)] + 60))
    a.set_title("%s — dựng lại Hình 7 tr.20, công thức (4.10)" % ten, fontsize=9.5)
    a.legend(fontsize=8, loc="upper right")

fig.suptitle("Bậc tự do và $C_p$: xấp xỉ df ≈ k đúng tới đâu, và nó chọn ra mô hình nào",
             fontsize=11, y=0.995)
fig.tight_layout()
fig.savefig("lars_df_cp.png", dpi=130, bbox_inches="tight", facecolor="white")
plt.close(fig)

print("m=10: C_p nhỏ nhất tại k = %d ; m=64: tại k = %d"
      % (int(np.argmin(Cp10)), int(np.argmin(Cp64))))
print("m=10: %d/10 điểm có df = k nằm NGOÀI khoảng tin cậy 95%%"
      % int((np.abs(np.array(df10) - np.arange(1, 11)) > T9 * np.array(se10)).sum()))

# ----------------------------------------------------------------------

import itertools
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})

d = load_diabetes(scaled=False)
X = d.data - d.data.mean(0)
X = X / np.sqrt((X ** 2).sum(0))
G = X.T @ X

vp, tong, nho_nhat = [], [], []
for k in range(1, 11):
    v_, t_, m_ = 0, 0, []
    for A in itertools.combinations(range(10), k):
        t_ += 1
        z = np.linalg.solve(G[np.ix_(A, A)], np.ones(k))          # (4.11)
        m_.append(z.min())
        if (z <= 0).any():
            v_ += 1
    vp.append(v_); tong.append(t_); nho_nhat.append(min(m_))

k = np.arange(1, 11)
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.6))
ax.bar(k, tong, color="#e5e7eb", label="tổng số tập con cỡ k")
ax.bar(k, vp, color="#dc2626", label=r"số tập con VI PHẠM $G_{\mathcal{A}}^{-1}\mathbf{1} > 0$")
for i in range(10):
    if vp[i]:
        ax.annotate("%d/%d" % (vp[i], tong[i]), (k[i], max(vp[i], 6)),
                    xytext=(0, 5), textcoords="offset points", fontsize=7.5,
                    ha="center", color="#7f1d1d")
ax.set_xticks(k)
ax.set_xlabel("cỡ của tập con A"); ax.set_ylabel("số tập con")
ax.set_title("Điều kiện nón dương (4.11) trên CHÍNH bộ diabetes\n"
             "%d trên %d tập con vi phạm — %.2f %%"
             % (sum(vp), sum(tong), 100 * sum(vp) / sum(tong)), fontsize=9.5)
ax.legend(fontsize=8, loc="upper left")

ax2.plot(k, nho_nhat, "o-", ms=6, lw=1.6, color="#1d4ed8")
ax2.axhline(0, color="#dc2626", lw=1.4, ls="--",
            label="ngưỡng của (4.11): mọi phần tử phải DƯƠNG")
ax2.fill_between(k, nho_nhat, 0, where=np.array(nho_nhat) < 0, color="#fecaca", alpha=.7)
ax2.set_yscale("symlog", linthresh=1e-2)
ax2.set_xticks(k)
ax2.set_xlabel("cỡ của tập con A")
ax2.set_ylabel(r"phần tử NHỎ NHẤT của $G_{\mathcal{A}}^{-1}\mathbf{1}$ trong mọi tập cỡ k")
ax2.set_title("Vi phạm không phải sát ngưỡng: giá trị âm nhất là %.2f\n"
              "Định lý 4 tr.20 đòi (4.11), nên nó KHÔNG áp được cho bộ này"
              % min(nho_nhat), fontsize=9.5)
ax2.set_ylim(min(nho_nhat) * 9.0, 4.0)
ax2.legend(fontsize=8, loc="lower center")
fig.savefig("lars_non_duong.png", dpi=130, bbox_inches="tight", facecolor="white")
plt.close(fig)

print("vi phạm theo cỡ 1..10: %s" % vp)
print("tổng: %d/%d tập con vi phạm (%.2f%%); phần tử âm nhất %.6f"
      % (sum(vp), sum(tong), 100 * sum(vp) / sum(tong), min(nho_nhat)))

# ----------------------------------------------------------------------

import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.linear_model import lars_path

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})

d = load_diabetes(scaled=False)
X = d.data - d.data.mean(0)
X = X / np.sqrt((X ** 2).sum(0))
y = d.target - d.target.mean()
B = lars_path(X, y, method="lasso")[2]
T = np.abs(B).sum(0)
S = ((y[:, None] - X @ B) ** 2).sum(0)


def he_so(t):                                    # đường nghiệm TUYẾN TÍNH giữa hai nút
    return np.array([np.interp(t, T, B[j]) for j in range(10)])


luoi = np.linspace(0, T[-1], 1200)
S_luoi = np.array([((y - X @ he_so(t)) ** 2).sum() for t in luoi])
b1k = he_so(1000.0)
S1k = ((y - X @ b1k) ** 2).sum()
C1k = np.abs(X.T @ (y - X @ b1k)).max()          # Ĉ(1000)

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.6))
ax.plot(luoi, S_luoi, lw=2.0, color="#2563eb", zorder=2)
ax.plot(T, S, "o", ms=6.5, color="#1e3a8a", zorder=3,
        label="12 điểm gãy của đường lasso")
ax.plot([1000.0], [S1k], "^", ms=12, color="#dc2626", zorder=4,
        label=r"điểm $t = 1000$")
dx = np.array([-520.0, 760.0])
ax.plot(1000 + dx, S1k - 2 * C1k * dx, "--", lw=1.6, color="#dc2626", zorder=4,
        label=r"tiếp tuyến, độ dốc $\dot S = -2\hat C(t)$")
ax.annotate(r"$\dot S(1000) = -2 \times %.4f = %.4f$" % (C1k, -2 * C1k),
            (1000 + dx[1], S1k - 2 * C1k * dx[1]),
            xytext=(-6, 26), textcoords="offset points", fontsize=8.5, color="#dc2626",
            ha="right")
ax.set_xlabel(r"$T = \sum_j |\hat\beta_j|$")
ax.set_ylabel(r"$S = \|y - X\hat\beta\|^2$")
ax.set_title("Dựng lại Hình 8 tr.30 — đường (T, S) của lasso\n"
             "giảm, LỒI, và là một spline bậc hai", fontsize=9.5)
ax.legend(fontsize=8, loc="upper right")

# đạo hàm cấp hai đo bằng sai phân, so với công thức 2·A_A² của từng đoạn
dd = np.diff(S_luoi, 2) / (luoi[1] - luoi[0]) ** 2
ax2.plot(luoi[1:-1], dd, lw=1.4, color="#7c3aed", label=r"$\ddot S$ đo bằng sai phân")
for k in range(1, len(T)):
    A = list(np.flatnonzero(np.abs(B[:, k]) > 1e-12))
    if not A:
        continue
    c = X.T @ (y - X @ B[:, k - 1])
    s = np.sign(c[A]); s[s == 0] = 1.0
    XA = X[:, A] * s
    iG = np.linalg.solve(XA.T @ XA, np.ones(len(A)))
    AA2 = 1.0 / (np.ones(len(A)) @ iG)
    ax2.plot([T[k - 1], T[k]], [2 * AA2, 2 * AA2], lw=3.0, color="#059669",
             solid_capstyle="butt")
ax2.plot([], [], lw=3.0, color="#059669", label=r"công thức $2A_{\mathcal{A}}^2$ của từng đoạn")
ax2.axhline(0, color="#6b7280", lw=0.8)
ax2.set_xlabel(r"$T = \sum_j |\hat\beta_j|$")
ax2.set_ylabel(r"$\ddot S(T)$")
ax2.set_ylim(-0.25, 2.5)
ax2.set_title(r"$\ddot S$ là HẰNG SỐ trên mỗi đoạn và LUÔN DƯƠNG"
              "\nđó là toàn bộ lý do đường (T, S) lồi", fontsize=9.5)
ax2.legend(fontsize=8, loc="upper right")
fig.savefig("lars_ts_loi.png", dpi=130, bbox_inches="tight", facecolor="white")
plt.close(fig)

assert dd.min() > -1e-6, "đường (T, S) có chỗ LÕM"
# S là đa thức bậc hai TRÊN MỖI ĐOẠN, nên sai phân hai phía với bước nằm gọn trong đoạn
# cho đạo hàm CHÍNH XÁC; dùng lưới chung ở trên thì bước vắt qua điểm gãy và sai số lớn.
h = 20.0
doc = (((y - X @ he_so(1000 + h)) ** 2).sum() - ((y - X @ he_so(1000 - h)) ** 2).sum()) / (2 * h)
print("Ṡ(1000) đo %.6f · công thức −2Ĉ(1000) = −2×%.6f = %.6f · lệch %.2e"
      % (doc, C1k, -2 * C1k, abs(doc + 2 * C1k)))
print("đạo hàm cấp hai nhỏ nhất trên cả đường: %.4e -> LỒI" % dd.min())
