# -*- coding: utf-8 -*-
"""Sinh hình cho bài "Lasso Regression".

KHÔNG sửa file này bằng tay: nó được sinh ra từ chính khối code in trong bài, bằng
    python extract_fig.py "Lasso Regression" fig_lasso.py
Muốn đổi hình thì sửa khối code TRONG BÀI rồi chạy lại lệnh trên và regen.py.
"""
import os

os.chdir(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "AI algorithms", "Machine Learning", "Lasso Regression"))

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})
TEN = ["lcavol", "lweight", "age", "lbph", "svi", "lcp", "gleason", "pgg45"]
MAU = ["#dc2626", "#2563eb", "#059669", "#d97706", "#7c3aed", "#0891b2", "#be185d", "#4b5563"]


def nhan_phai(a, ys, x, gap):
    """Ghi tên biến ở mép phải, đẩy các nhãn gần nhau ra cho khỏi chồng."""
    thu_tu = np.argsort(ys)
    vi_tri = np.array(ys, dtype=float)[thu_tu]
    for i in range(1, len(vi_tri)):
        vi_tri[i] = max(vi_tri[i], vi_tri[i - 1] + gap)
    for j, v in zip(thu_tu, vi_tri):
        a.annotate(TEN[j], (x, ys[j]), xytext=(x + 0.03, v), textcoords="data", fontsize=7,
                   color=MAU[j], va="center")

# Mọi mảng dưới đây do tools/v_lasso.py in ra (mục 4, 9 và 20); khối này chỉ vẽ.
S_LOI = np.array([0, 0.2283801674, 0.3158224504, 0.4760973639, 0.4844883101, 0.6134985707, 0.7065472527, 0.7459998282, 1])
B_LOI = np.array([[0, 0.4211311879, 0.5017526205, 0.5753305642, 0.5782094339, 0.6082966537, 0.6239226518, 0.6286205146, 0.6918722977], [0, 0, 0, 0.1252127624, 0.1279517184, 0.1716733829, 0.1947184486, 0.2049275644, 0.2257160472], [0, 0, 0, 0, 0, 0, -0.0581457138, -0.0823370198, -0.1462060263], [0, 0, 0, 0, 0.0051942117, 0.0739202551, 0.1094600263, 0.1236325501, 0.1553065124], [0, 0, 0.0806214325, 0.177376375, 0.1820371748, 0.2348739212, 0.2486293464, 0.255055723, 0.3171829304], [0, 0, 0, 0, 0, 0, 0, 0, -0.14747849], [0, 0, 0, 0, 0, 0, 0, 0.0090542714, 0.0325973634], [0, 0, 0, 0, 0, 0.0425222065, 0.0679912522, 0.0719900271, 0.1276322725]])
S_SUA = np.array([0, 0.2270228333, 0.2706411284, 0.4917470828, 0.5123525536, 0.5996824157, 0.6932909827, 0.7472112829, 1])
B_SUA = np.array([[0, 0.4211311879, 0.4615875248, 0.5574525141, 0.5624760014, 0.5828329231, 0.5972691681, 0.6029232795, 0.6651466713], [0, 0, 0, 0.1718051892, 0.1860763342, 0.2100962793, 0.2326862311, 0.2462930422, 0.2664802617], [0, 0, 0, 0, 0, 0, -0.0598626448, -0.0937414612, -0.1581952246], [0, 0, 0, 0, 0, 0.0511503136, 0.088065044, 0.1081242209, 0.1403111691], [0, 0, 0.0404563369, 0.1829414359, 0.1912867675, 0.2292324519, 0.2435999107, 0.2526922409, 0.3153288818], [0, 0, 0, 0, 0, 0, 0, 0, -0.1482856849], [0, 0, 0, 0, 0, 0, 0, 0.0123116155, 0.035549169], [0, 0, 0, 0, 0.0105835323, 0.0391090357, 0.0645834769, 0.0700036823, 0.1257198175]])
RIDGE = {
    0.00: np.array([[6, 3], [5.99942485, 2.99975925], [5.99918039, 2.99965692], [5.99883203, 2.9995111], [5.99833567, 2.99930332], [5.99762844, 2.99900725], [5.99662085, 2.99858542], [5.99518553, 2.99798446], [5.99314125, 2.99712843], [5.99023037, 2.9959093], [5.98608698, 2.99417354], [5.98019222, 2.9917032], [5.97181179, 2.98818941], [5.95990967, 2.98319542], [5.94303038, 2.97610583], [5.91914163, 2.96605765], [5.88543047, 2.95184906], [5.83805228, 2.93182306], [5.77184724, 2.90372788], [5.6800717, 2.86456737], [5.55425045, 2.81047714], [5.38434477, 2.73670092], [5.1595329, 2.63779725], [4.86994195, 2.50825871], [4.50950392, 2.34371629], [4.07955829, 2.14272149], [3.59192247, 1.90866968], [3.06946717, 1.65089719], [2.54275136, 1.38387421], [2.04333211, 1.12423663], [1.59660947, 0.8868098], [1.2174085, 0.68151788], [0.90959093, 0.51238968], [0.66867011, 0.37849203], [0.48544301, 0.27577437], [0.34911614, 0.1988583], [0.2493321, 0.14229645], [0.17716824, 0.10125296], [0.12543204, 0.07175684], [0.08857233, 0.05070614], [0.06242838, 0.0357571]]),
    0.23: np.array([[6, 3], [5.99943816, 2.99988302], [5.99919936, 2.99983329], [5.99885907, 2.99976241], [5.99837421, 2.9996614], [5.99768337, 2.99951742], [5.99669917, 2.9993122], [5.99529721, 2.99901966], [5.99330052, 2.9986026], [5.9904576, 2.99800793], [5.98641132, 2.99715982], [5.98065546, 2.99594989], [5.97247401, 2.99422301], [5.96085753, 2.99175685], [5.94438934, 2.98823208], [5.9210943, 2.98318894], [5.88824415, 2.97596356], [5.84212048, 2.96559486], [5.77775157, 2.95068944], [5.68867121, 2.92923042], [5.56680155, 2.89832243], [5.4026385, 2.85389137], [5.18599549, 2.79042798], [4.90757368, 2.70099646], [4.56146135, 2.57789405], [4.14822205, 2.41437914], [3.67757807, 2.20749158], [3.16921263, 1.96104495], [2.65048729, 1.6869626], [2.15118307, 1.40343602], [1.69711978, 1.13027813], [1.30532791, 0.88381557], [0.98251136, 0.6737957], [0.72661657, 0.50312939], [0.52998459, 0.36959873], [0.38250384, 0.26813659], [0.27389725, 0.19272221], [0.19499822, 0.13757388], [0.13824721, 0.09772093], [0.09771872, 0.06916622], [0.06892379, 0.04883135]]),
    0.45: np.array([[6, 3], [5.99938721, 3.00002428], [5.99912676, 3.00003458], [5.99875566, 3.00004922], [5.9982269, 3.00007002], [5.99747359, 3.00009954], [5.9964005, 3.00014136], [5.99487216, 3.00020044], [5.99269598, 3.0002836], [5.98959848, 3.00040001], [5.9851919, 3.00056166], [5.97892756, 3.00078345], [5.97003154, 3.00108226], [5.95741683, 3.00147345], [5.93956615, 3.00196182], [5.91437996, 3.0025209], [5.87898889, 3.00304932], [5.82953964, 3.00328415], [5.76098448, 3.00263878], [5.66693915, 2.99991922], [5.53972214, 2.99287171], [5.37073329, 2.97755303], [5.15133437, 2.94764272], [4.87430597, 2.89406046], [4.53575889, 2.80553146], [4.13714788, 2.67076385], [3.68688054, 2.48227505], [3.20092811, 2.24063088], [2.70176776, 1.95678979], [2.21519913, 1.65062455], [1.76551933, 1.34579309], [1.37079643, 1.06344588], [1.04024567, 0.81777788], [0.77448945, 0.6148517], [0.56789552, 0.45407034], [0.41152254, 0.33073665], [0.29556484, 0.23841803], [0.21088907, 0.17055977], [0.1497523, 0.12133915], [0.10597235, 0.08597824], [0.07480638, 0.06074849]]),
    0.68: np.array([[6, 3], [5.99917991, 3.00030062], [5.99883142, 3.00042829], [5.99833493, 3.00061012], [5.99762766, 3.000869], [5.99662031, 3.00123743], [5.9951859, 3.00176146], [5.99314409, 3.00250622], [5.9902391, 3.00356342], [5.9861089, 3.00506164], [5.98024255, 3.00717981], [5.97192191, 3.0101643], [5.9601435, 3.01434909], [5.94351668, 3.02017638], [5.92013675, 3.02821072], [5.88743778, 3.03913122], [5.84204263, 3.05367235], [5.77964922, 3.07246239], [5.69502068, 3.09568517], [5.58217098, 3.12248307], [5.43482652, 3.15006604], [5.24715701, 3.17264816], [5.0146056, 3.18061203], [4.73452165, 3.16058107], [4.40643082, 3.09707654], [4.0322196, 2.97591903], [3.6168437, 2.78864805], [3.16975351, 2.53645371], [2.70611083, 2.2318581], [2.2461584, 1.89690404], [1.81192387, 1.55803389], [1.42240816, 1.23963861], [1.08967684, 0.9590529], [0.81759803, 0.72473255], [0.60316759, 0.53739621], [0.43911878, 0.39266294], [0.31648046, 0.28373094], [0.22638738, 0.20333184], [0.16105383, 0.1448385], [0.11412052, 0.10272378], [0.08063409, 0.07262789]]),
    0.90: np.array([[6, 3], [5.99793387, 3.00159696], [5.99705692, 3.00227449], [5.99580857, 3.00323865], [5.99403238, 3.00460987], [5.99150686, 3.00655832], [5.98791936, 3.00932358], [5.98283022, 3.0132412], [5.9756248, 3.01877761], [5.96545078, 3.02657419], [5.95113994, 3.0374991], [5.93111765, 3.0527005], [5.90331167, 3.07364514], [5.86508748, 3.10210932], [5.81326003, 3.14006436], [5.74425588, 3.18937291], [5.65450418, 3.25120562], [5.54107849, 3.32514411], [5.40245321, 3.40809734], [5.23900773, 3.49340186], [5.05278532, 3.57063062], [4.84625414, 3.62645408], [4.6204489, 3.6463668], [4.37346467, 3.61664586], [4.10030361, 3.52602534], [3.79451654, 3.36713933], [3.45134102, 3.13813929], [3.07137414, 2.84453499], [2.66327279, 2.50042586], [2.24385608, 2.12770503], [1.83488325, 1.75242731], [1.45762404, 1.3992861], [1.12777921, 1.08660441], [0.85296382, 0.82395253], [0.63318217, 0.61276594], [0.46315066, 0.44879537], [0.33497412, 0.32488656], [0.24023269, 0.2331471], [0.17122133, 0.16624584], [0.12148682, 0.11799378], [0.08592044, 0.08346851]]),
}
# (phương pháp, trung vị ME in ở Bảng 3, dựng lại với σ = 3, dựng lại với σ = 2)
BANG3 = [
    ('LS', 2.79, 6.336191, 2.816085),
    ('Lasso (CV)', 2.43, 4.023847, 2.026167),
    ('Lasso (Stein)', 2.07, 5.16904, 2.475672),
    ('Lasso (GCV)', 1.93, 3.405607, 1.673076),
    ('Garotte', 2.29, 4.776572, 2.217645),
    ('Best subset', 2.44, 6.160472, 2.165651),
    ('Ridge', 3.21, 3.934005, 2.273567),
]

fig, ax = plt.subplots(2, 2, figsize=(11.5, 8.8))

# (a) Hình 1 tr.270: bốn phép co trong thiết kế trực chuẩn
b = np.linspace(0, 5, 501)
a = ax[0, 0]
a.plot(b, b, ":", color="#9ca3af", lw=1, label="đường 45°")
a.plot(b, np.where(b > 2, b, 0), color="#111827", lw=1.3, label="tập con, ngưỡng 2")
a.plot(b, b / 3, color="#2563eb", lw=1.3, label="ridge, γ = 2 (hình gốc không ghi γ)")
a.plot(b, np.maximum(b - 2, 0), color="#dc2626", lw=1.8, label="lasso (3), γ = 2")
a.plot(b, np.maximum(1 - 4 / np.maximum(b, 1e-9) ** 2, 0) * b, color="#059669", lw=1.3, label="garotte, γ = 4")
a.set_xlabel(r"$\hat\beta^\circ_j$ (bình phương tối thiểu)")
a.set_ylabel(r"$\hat\beta_j$")
a.set_title("(a) Hình 1 tr.270 dựng lại từ công thức", fontsize=9)
a.legend(fontsize=7.5, loc="upper left")

# (b) Hình 4 tr.272: β̂° = (6, 3); lasso theo (5) đã sửa miền, ridge theo năm ρ
a = ax[0, 1]
t = np.linspace(0, 9, 91)
lasso = np.c_[np.where(t <= 3, t, (t + 3) / 2), np.where(t <= 3, 0, (t - 3) / 2)]
for (rho, B), m in zip(RIDGE.items(), ["#93c5fd", "#60a5fa", "#3b82f6", "#1d4ed8", "#1e3a8a"]):
    a.plot(B[:, 0], B[:, 1], "--", color=m, lw=1.2, label="ridge, ρ = %.2f" % rho)
a.plot(lasso[:, 0], lasso[:, 1], color="#111827", lw=2.2, label="lasso, cả năm ρ (trùng nhau)")
tt = np.linspace(0, 2.9, 30)
a.plot((tt + 3) / 2, np.zeros_like(tt), color="#dc2626", lw=5, alpha=.35,
       label="(6) in nguyên văn khi t < 3: β̂₁ = (t+3)/2 > t")
a.axhline(3, color="#9ca3af", lw=.8, ls=":")
a.set_xlabel(r"$\hat\beta_1$")
a.set_ylabel(r"$\hat\beta_2$")
a.set_title("(b) Hình 4 tr.272: đường lasso không phụ thuộc ρ; ridge thì có", fontsize=9)
a.legend(fontsize=7, loc="upper left")

# (c) Hình 5 tr.273: prostate, dữ liệu CÓ LỖI (liền) và ĐÃ SỬA (đứt)
a = ax[1, 0]
for j in range(8):
    a.plot(S_LOI, B_LOI[j], color=MAU[j], lw=1.5)
    a.plot(S_SUA, B_SUA[j], color=MAU[j], lw=1.0, ls="--")
nhan_phai(a, B_LOI[:, -1], 1.0, 0.035)
a.axvline(0.44, color="#111827", ls=":", lw=1)
a.text(0.45, 0.62, "s = 0,44", fontsize=7.5)
a.axhline(0, color="#111827", lw=.6)
a.set_xlim(0, 1.18)
a.set_xlabel(r"$s=t/\sum_j|\hat\beta^\circ_j|$")
a.set_ylabel("hệ số (biến chuẩn hoá)")
a.set_title("(c) Hình 5 tr.273: liền = lweight đối tượng 32 bằng ln 449 (khớp Bảng 1),\n"
            "đứt = bản đã sửa ln 44,9", fontsize=9)

# (d) Bảng 3 tr.279: trung vị ME của ví dụ 1
a = ax[1, 1]
y = np.arange(len(BANG3))[::-1]
a.scatter([r[1] for r in BANG3], y, marker="D", s=46, color="#111827", label="Bảng 3 in", zorder=3)
a.scatter([r[2] for r in BANG3], y + 0.15, marker="o", s=34, color="#2563eb", label="dựng lại, σ = 3 (như văn bản)")
a.scatter([r[3] for r in BANG3], y - 0.15, marker="s", s=34, color="#d97706", label="dựng lại, σ = 2")
a.set_yticks(y)
a.set_yticklabels([r[0] for r in BANG3])
a.set_xlabel("trung vị sai số ME trên 200 bộ dữ liệu")
a.set_title("(d) Bảng 3 tr.279 khớp σ = 2, không khớp σ = 3", fontsize=9)
a.legend(fontsize=7.5, loc="lower right")

fig.tight_layout()
fig.savefig("lasso_tib1996_dung_lai.png", dpi=130, bbox_inches="tight", facecolor="white")
print("đã ghi lasso_tib1996_dung_lai.png")

# ----------------------------------------------------------------------

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})
TEN = ["lcavol", "lweight", "age", "lbph", "svi", "lcp", "gleason", "pgg45"]
MAU = ["#dc2626", "#2563eb", "#059669", "#d97706", "#7c3aed", "#0891b2", "#be185d", "#4b5563"]


def nhan_phai(a, ys, x, gap):
    """Ghi tên biến ở mép phải, đẩy các nhãn gần nhau ra cho khỏi chồng."""
    thu_tu = np.argsort(ys)
    vi_tri = np.array(ys, dtype=float)[thu_tu]
    for i in range(1, len(vi_tri)):
        vi_tri[i] = max(vi_tri[i], vi_tri[i - 1] + gap)
    for j, v in zip(thu_tu, vi_tri):
        a.annotate(TEN[j], (x, ys[j]), xytext=(x + 0.03, v), textcoords="data", fontsize=7,
                   color=MAU[j], va="center")

# Toạ độ vectơ đọc thẳng từ PDF ESL (tools/v_lasso.py mục 15–16) và đường dựng lại (mục 15, mục 20).
NUT = np.array([1.3e-05, 0.189335, 0.254405, 0.372386, 0.376482, 0.557056, 0.599226, 0.943313, 1.000009])
DUONG_IN = {
    'lcavol': np.array([[1.3e-05, 4e-06], [0.189335, 0.427944], [0.254401, 0.5015], [0.372386, 0.561007], [0.376482, 0.562185], [0.557051, 0.579689], [0.599222, 0.58643], [0.943305, 0.699384], [1.000005, 0.716397]]),
    'lweight': np.array([[0.189335, 4e-06], [0.254401, 0.073541], [0.372386, 0.187769], [0.376482, 0.188985], [0.557051, 0.245624], [0.599222, 0.257222], [0.943305, 0.291], [1.000005, 0.292636]]),
    'age': np.array([[0.557056, 4e-06], [0.599222, -0.032085], [0.943309, -0.133741], [1.000009, -0.142553]]),
    'lbph': np.array([[0.372386, 4e-06], [0.376482, 0.003573], [0.557051, 0.143491], [0.599222, 0.163862], [0.943309, 0.206226], [1.000005, 0.211995]]),
    'svi': np.array([[0.254405, 4e-06], [0.372386, 0.092948], [0.376482, 0.09625], [0.557051, 0.200301], [0.599222, 0.20822], [0.943309, 0.300307], [1.000005, 0.30961]]),
    'lcp': np.array([[0.599226, 4e-06], [0.943313, -0.256456], [1.000009, -0.289002]]),
    'gleason': np.array([[0.943313, 4e-06], [1.000009, -0.020921]]),
    'pgg45': np.array([[0.376482, 4e-06], [0.557051, 0.090081], [0.599222, 0.106655], [0.943309, 0.245204], [1.000005, 0.27735]]),
}
DO = 0.375001
S67 = np.array([0, 0.1893167014, 0.2543884563, 0.3723704365, 0.3764680298, 0.5570445635, 0.5992041056, 0.9433024977, 1])
B67 = np.array([[0, 0.4279487498, 0.5014958165, 0.561008306, 0.5621810998, 0.579691685, 0.5864363211, 0.6993856607, 0.7164070125], [0, 0, 0.0735470667, 0.187775267, 0.1889809686, 0.2456216279, 0.2572313691, 0.2909974619, 0.2926424008], [0, 0, 0, 0, 0, 0, -0.0320869097, -0.13374448, -0.142549626], [0, 0, 0, 0, 0.0035792824, 0.1435012811, 0.1638641747, 0.2062244718, 0.2120076045], [0, 0, 0, 0.0929565461, 0.096261342, 0.2002992337, 0.2082280615, 0.3003101264, 0.3096195331], [0, 0, 0, 0, 0, 0, 0, -0.2564584594, -0.2890056157], [0, 0, 0, 0, 0, 0, 0, 0, -0.0209135198], [0, 0, 0, 0, 0, 0.0900805423, 0.1066488106, 0.2452063958, 0.2773459525]])
CAM_S = np.array([-0.0006, 0.124, 0.2497, 0.3743, 0.5, 0.6246, 0.7491, 0.8748, 0.9994])
CAM_CV = np.array([1.4687, 1.041, 0.7915, 0.6509, 0.5836, 0.5955, 0.6172, 0.6192, 0.6172])
TIM_NGANG, TIM_DUNG = 0.6753, 0.3749

fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2), gridspec_kw={"width_ratios": [1.35, 1]})
a = ax[0]
for j, ten in enumerate(TEN):
    a.plot(S67, B67[j], color=MAU[j], lw=1.3)
    P = DUONG_IN[ten]
    a.plot(P[:, 0], P[:, 1], "o", ms=4, mfc="none", mec=MAU[j])
nhan_phai(a, B67[:, -1], 1.0, 0.045)
for x in NUT:
    a.axvline(x, color="#d1d5db", lw=.7, zorder=0)
a.axvline(DO, color="#dc2626", lw=1.4, label="vạch đỏ trong Hình 3.10: s = %.4f" % DO)
a.axvline(0.36, color="#dc2626", lw=1, ls="--", label="s = 0,36 theo chú thích và tr.69")
a.axhline(0, color="#111827", lw=.6)
a.set_xlim(0, 1.16)
a.set_xlabel("s")
a.set_ylabel("hệ số")
a.set_title("(a) Hình 3.10 tr.70: vòng tròn = điểm gãy đọc từ PDF, đường = dựng lại\n"
            "(chuẩn hoá trên 67 mẫu huấn luyện)", fontsize=9)
a.legend(fontsize=7.5, loc="upper left")
a = ax[1]
a.plot(CAM_S, CAM_CV, "o-", color="#e69f00", lw=1.2, ms=5)
a.axhline(TIM_NGANG, color="#a020f0", lw=1, label="cực tiểu + 1 sai số chuẩn = %.4f" % TIM_NGANG)
a.axvline(TIM_DUNG, color="#a020f0", lw=1, ls="--", label="lựa chọn trong hình: s = %.4f" % TIM_DUNG)
k = int(np.argmin(CAM_CV))
a.plot(CAM_S[k], CAM_CV[k], "*", ms=12, color="#111827", label="cực tiểu: s = %.4f" % CAM_S[k])
a.set_xlabel("s")
a.set_ylabel("sai số kiểm định chéo 10 mảnh")
a.set_title("(b) Hình 3.7 tr.62, ô Lasso — lưới 9 điểm, bước 1/8", fontsize=9)
a.legend(fontsize=7.5, loc="upper right")
fig.tight_layout()
fig.savefig("lasso_esl_dung_lai.png", dpi=130, bbox_inches="tight", facecolor="white")
print("đã ghi lasso_esl_dung_lai.png")

# ----------------------------------------------------------------------

import sys
import numpy as np, matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.linear_model import lars_path

sys.stdout.reconfigure(encoding="utf-8")

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})

# Chuan hoa DUNG quy uoc Efron muc 2: cot trung binh 0 va chuan 2 bang 1, y tru trung binh.
d = load_diabetes()
Xd = d.data - d.data.mean(0)
Xd = Xd / np.sqrt((Xd ** 2).sum(0))
yd = d.target - d.target.mean()

alphas, active, coefs = lars_path(Xd, yd, method="lasso")   # các nút đường nghiệm theo số học máy

fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.8))

# --- bang trai: duong LASSO, truc hoanh t = sum|beta| giong Efron Hinh 1 ---
tt = np.abs(coefs).sum(0)
order, prev = [], set()
for k in range(coefs.shape[1]):
    cur = set(np.nonzero(coefs[:, k])[0])
    order += [int(j) + 1 for j in sorted(cur - prev)]
    prev = cur
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

# --- bang phai: RIDGE tren CUNG du lieu, cung truc hoanh de so sanh cong bang ---
lam3 = np.logspace(-4, 2, 150)
Bp = np.array([np.linalg.solve(Xd.T @ Xd + l * np.eye(10), Xd.T @ yd) for l in lam3])
tr_ = np.abs(Bp).sum(1)
for j in range(10):
    axes[1].plot(tr_, Bp[:, j], lw=1.2)
    axes[1].annotate(str(j + 1), (tr_[0], Bp[0, j]), fontsize=6.5,
                     xytext=(3, -2), textcoords="offset points")
axes[1].axhline(0, color="#111827", lw=.6)
axes[1].set_xlabel(r"$\sum_j|\hat\beta_j|$"); axes[1].set_ylabel(r"$\hat\beta_j$")
axes[1].set_title("Ridge trên cùng dữ liệu — không có đoạn\n"
                  "bị ghim ở 0 bởi một ngưỡng mềm", fontsize=9)

fig.savefig("lasso_vs_ridge_path.png", dpi=130, bbox_inches="tight", facecolor="white")
print("thứ tự biến vào:", order)
print("t cuối = %.2f" % tt[-1])
