# -*- coding: utf-8 -*-
"""Sinh hình cho bài "Bayesian Linear Regression".

KHÔNG sửa file này bằng tay: nó được sinh ra từ chính khối code in trong bài, bằng
    python extract_fig.py "Bayesian Linear Regression" fig_bayes.py
Muốn đổi hình thì sửa khối code TRONG BÀI rồi chạy lại lệnh trên và regen.py.
"""
import os

os.chdir(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "AI algorithms", "Machine Learning", "Bayesian Linear Regression"))

# -*- coding: utf-8 -*-
# Mọi con số trong khối này do tools/v_bayes.py mục 23 [A] in ra; khối này chỉ VẼ.
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

XA = np.array([-0.20000000, -0.16666667, -0.13333333, -0.10000000, -0.06666667,
               -0.03333333, 0.00000000, 0.03333333, 0.06666667, 0.10000000,
               0.13333333, 0.16666667, 0.20000000, 0.23333333, 0.26666667,
               0.30000000, 0.33333333, 0.36666667, 0.40000000, 0.43333333,
               0.46666667, 0.50000000, 0.53333333, 0.56666667, 0.60000000,
               0.63333333, 0.66666667, 0.70000000, 0.73333333, 0.76666667,
               0.80000000, 0.83333333, 0.86666667, 0.90000000, 0.93333333,
               0.96666667, 1.00000000, 1.03333333, 1.06666667, 1.10000000,
               1.13333333, 1.16666667, 1.20000000])
# phần BẤT ĐỊNH THAM SỐ, tức phi^T S_N phi, ở N = 4 và N = 25
Q4 = np.array([0.44834905, 0.37499148, 0.30387827, 0.23794115, 0.17335376,
               0.11330627, 0.08777928, 0.15302167, 0.35614508, 0.68913412,
               1.07526870, 1.40076911, 1.56421824, 1.51422887, 1.26712384,
               0.90236729, 0.53170773, 0.25242669, 0.10982662, 0.08638062,
               0.11926351, 0.14006367, 0.11926351, 0.08638062, 0.10982662,
               0.25242669, 0.53170773, 0.90236729, 1.26712384, 1.51422887,
               1.56421824, 1.40076911, 1.07526870, 0.68913412, 0.35614508,
               0.15302167, 0.08777928, 0.11330627, 0.17335376, 0.23794115,
               0.30387827, 0.37499148, 0.44834905])
Q25 = np.array([0.39618272, 0.31616900, 0.23707402, 0.16945754, 0.11751071,
                0.07879148, 0.04999943, 0.03140507, 0.02393613, 0.02402522,
                0.02521367, 0.02449884, 0.02350580, 0.02361636, 0.02382469,
                0.02350078, 0.02343058, 0.02368906, 0.02352067, 0.02320255,
                0.02345499, 0.02375827, 0.02345499, 0.02320255, 0.02352067,
                0.02368906, 0.02343058, 0.02350078, 0.02382469, 0.02361636,
                0.02350580, 0.02449884, 0.02521367, 0.02402522, 0.02393613,
                0.03140507, 0.04999943, 0.07879148, 0.11751071, 0.16945754,
                0.23707402, 0.31616900, 0.39618272])
NHIEU = 0.09009009          # 1/beta voi beta = 11,1

fig, ax = plt.subplots(1, 2, figsize=(11.4, 4.1), sharey=True)
for a, Q, ten in ((ax[0], Q4, "N = 4"), (ax[1], Q25, "N = 25")):
    a.fill_between(XA, 0, NHIEU, color="#94a3b8", alpha=0.65,
                   label=r"$1/\beta$ — nhiễu, không giảm được")
    a.fill_between(XA, NHIEU, NHIEU + Q, color="#7c3aed", alpha=0.45,
                   label=r"$\phi^\top S_N\phi$ — bất định tham số")
    a.plot(XA, NHIEU + Q, color="#4c1d95", lw=2.0)
    a.axvspan(0.0, 1.0, color="#16a34a", alpha=0.07)
    a.axvline(0.0, color="#16a34a", lw=1.0, ls=(0, (4, 3)))
    a.axvline(1.0, color="#16a34a", lw=1.0, ls=(0, (4, 3)))
    a.set_title("%s — vùng xanh là nơi có dữ liệu" % ten)
    a.set_xlabel("x")
    a.set_yscale("log")
ax[0].set_ylabel(r"$\sigma_N^2(x)$, hai thành phần cộng lại")
ax[0].legend(loc="upper center")
print("phần bất định tham số tại x = 0,5: N=4 -> %.8f ; N=25 -> %.8f"
      % (Q4[21], Q25[21]))
print("tỉ số hai giá trị đó: %.6f lần" % (Q4[21] / Q25[21]))
print("so với nhiễu 1/beta tại x = 0,5: N=4 -> %.4f lần ; N=25 -> %.4f phần trăm"
      % (Q4[21] / NHIEU, 100 * Q25[21] / NHIEU))
print("tại x = 1,2 (ngoài dữ liệu): N=4 -> %.8f ; N=25 -> %.8f"
      % (Q4[-1], Q25[-1]))
fig.savefig("bayes_hai_nguon_bat_dinh.png", dpi=130, bbox_inches="tight",
            facecolor="white")

# ----------------------------------------------------------------------

# -*- coding: utf-8 -*-
# Mọi con số trong khối này do tools/v_bayes.py mục 23 [C] in ra; khối này chỉ VẼ.
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

ALC = np.array([1.000000e-12, 4.216965e-12, 1.778279e-11, 7.498942e-11, 3.162278e-10,
                1.333521e-09, 5.623413e-09, 2.371374e-08, 1.000000e-07, 4.216965e-07,
                1.778279e-06, 7.498942e-06, 3.162278e-05, 1.333521e-04, 5.623413e-04,
                2.371374e-03, 1.000000e-02, 4.216965e-02, 1.778279e-01, 7.498942e-01,
                3.162278e+00, 1.333521e+01, 5.623413e+01, 2.371374e+02, 1.000000e+03])
DC = np.array([5.646816e-12, 7.835954e-12, 9.912515e-12, 1.483169e-11, 8.092427e-11,
               3.404190e-10, 1.410850e-09, 5.958219e-09, 2.511531e-08, 1.059338e-07,
               4.466998e-07, 1.883383e-06, 7.936416e-06, 3.336588e-05, 1.389257e-04,
               5.563985e-04, 1.944724e-03, 4.959254e-03, 9.310149e-03, 1.603030e-02,
               2.676825e-02, 4.679721e-02, 8.473261e-02, 1.632411e-01, 3.301435e-01])
AL_H38 = 0.513333       # alpha dung lai duoc Hinh 3.8 (v_bayes.py muc 9)
DC_H38 = 1.395e-02      # do lech tai chinh alpha do (v_bayes.py muc 13)

fig, ax = plt.subplots(figsize=(7.8, 4.3))
ax.loglog(ALC, DC, color="#2563eb", lw=2.0, marker="o", ms=3.4,
          label=r"$\max_x\,|\sum_n k(x,x_n)-1|$")
ax.axvline(AL_H38, color="#dc2626", lw=1.8, ls=(0, (5, 3)),
           label=r"$\alpha$ dựng lại được Hình 3.8")
ax.plot([AL_H38], [DC_H38], "*", color="#dc2626", ms=17, zorder=6,
        label="độ lệch tại đúng $\\alpha$ đó")
ax.axhline(1e-2, color="#64748b", lw=1.0, ls=":")
ax.text(2.0e2, 1.25e-2, "mức 1 %", color="#64748b", fontsize=8, ha="right")
ax.set_xlabel(r"$\alpha$ — độ chính xác tiên nghiệm")
ax.set_ylabel(r"độ lệch khỏi ràng buộc (3.64)")
ax.set_title(r"(3.64) $\sum_n k(x,x_n)=1$ chỉ đúng trong giới hạn $\alpha\to 0$")
ax.legend(loc="upper left", fontsize=8)
print("do lech tai alpha nho nhat (%.3e): %.6e" % (ALC[0], DC[0]))
print("do lech tai alpha lon nhat (%.3e): %.6e" % (ALC[-1], DC[-1]))
print("so bac do lon giua hai dau: %.4f" % np.log10(DC[-1] / DC[0]))
print("do lech tai alpha = %.6f la %.6e, tuc %.4f phan tram"
      % (AL_H38, DC_H38, 100 * DC_H38))
fig.savefig("bayes_tong_nhan.png", dpi=130, bbox_inches="tight", facecolor="white")

# ----------------------------------------------------------------------

# -*- coding: utf-8 -*-
# Mọi con số trong khối này do tools/v_bayes.py mục 23 [B] in ra; khối này chỉ VẼ.
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

M = np.arange(10)
# ba day so, doc tu Hinh 3.14 va dung lai bang hai cach doc cong thuc
DOC = np.array([-24.036722, -19.729529, -23.075116, -17.612635, -18.307154,
                -19.869763, -21.412340, -22.828063, -24.170305, -25.479131])
CT386 = np.array([-23.123666, -17.879100, -20.306141, -13.934076, -13.713212,
                  -14.359022, -14.981523, -15.481400, -15.906108, -16.290483])
THIEU = np.array([-24.042604, -19.716977, -23.062957, -17.609830, -18.307904,
                  -19.872653, -21.414093, -22.832908, -24.176555, -25.479868])

fig, ax = plt.subplots(1, 2, figsize=(11.6, 4.3))

ax[0].plot(M, DOC, "o-", color="#111827", lw=2.4, ms=6, zorder=6,
           label="đọc từ Hình 3.14 (lớp vectơ)")
ax[0].plot(M, THIEU, "s--", color="#dc2626", lw=1.8, ms=5, zorder=5,
           label=r"(3.86) $-\ (M/2)\ln 2\pi$")
ax[0].plot(M, CT386, "^-", color="#2563eb", lw=1.8, ms=5, zorder=4,
           dashes=(5, 3), label="đúng công thức (3.86)")
ax[0].plot([int(np.argmax(DOC))], [DOC.max()], "*", color="#111827", ms=17, zorder=7)
ax[0].plot([int(np.argmax(CT386))], [CT386.max()], "*", color="#2563eb", ms=17, zorder=7)
ax[0].set_xlabel("bậc đa thức M")
ax[0].set_ylabel(r"$\ln p(\mathsf{t}\mid\alpha,\beta)$")
ax[0].set_title("Hình 3.14 và hai cách đọc công thức")
ax[0].legend(loc="lower left")

le = CT386 - DOC
ax[1].plot(M + 1, le, "o", color="#7c3aed", ms=7, zorder=5, label="lệch đo được")
A = np.c_[np.ones(10), M + 1]
ab = np.linalg.lstsq(A, le, rcond=None)[0]
xx = np.linspace(0.6, 10.4, 50)
ax[1].plot(xx, ab[0] + ab[1] * xx, color="#0891b2", lw=2.0,
           label="đường khớp: %+.6f %+.6f(M+1)" % (ab[0], ab[1]))
ax[1].plot(xx, 0.5 * np.log(2 * np.pi) * xx, color="#dc2626", lw=1.6,
           dashes=(6, 3), label=r"$\frac{1}{2}\ln(2\pi)\,(M+1)$")
ax[1].set_xlabel("số tham số M + 1")
ax[1].set_ylabel("(3.86) trừ giá trị đọc được")
ax[1].set_title("lệch TUYẾN TÍNH theo số tham số")
ax[1].legend(loc="upper left")

print("he so goc do duoc      : %.6f" % ab[1])
print("mot nua ln(2 pi)       : %.6f" % (0.5 * np.log(2 * np.pi)))
print("chenh hai con so       : %.6f" % abs(ab[1] - 0.5 * np.log(2 * np.pi)))
print("he so chan do duoc     : %+.6f" % ab[0])
print("phan du lon nhat       : %.6f" % np.abs(le - A @ ab).max())
print("dinh cua duong doc duoc: M = %d" % int(np.argmax(DOC)))
print("dinh theo dung (3.86)  : M = %d" % int(np.argmax(CT386)))
print("lech lon nhat ban thieu 2pi: %.6f" % np.abs(THIEU - DOC).max())
fig.savefig("bayes_dung_lai_hinh_3_14.png", dpi=130, bbox_inches="tight",
            facecolor="white")
