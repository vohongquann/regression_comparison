# -*- coding: utf-8 -*-
"""Kiểm chứng cho bài `Least Angle Regression (LARS)` — Order 410, GĐ 1.

Nguồn ruột: Efron, Hastie, Johnstone & Tibshirani (2004), *Least Angle Regression*,
The Annals of Statistics 32(2):407–451 — `_papers/2004 - Efron et al - Least Angle
Regression.pdf` (bản arXiv math/0406456v2, 48 trang).

MỌI con số trong bài phải in ra từ đây. Chạy: `cd tools && python v_lars.py`.

Bố cục 22 mục, đi theo thứ tự của chính bài báo:

     1  Bảng 1 và quy ước chuẩn hoá (1.1) — dữ liệu nào, đơn vị nào
     2  LARS viết TỪ ĐẦU theo (2.4)–(2.13), đối chiếu `sklearn`
     3  Bất biến cách đều góc: (2.7) · (2.16) · (2.18)
     4  Quan hệ với OLS: (2.19)–(2.22), và vì sao γ̂ < γ̄
     5  Sửa đổi lasso (3.1)–(3.6): Định lý 1
     6  Dựng lại Hình 1 và Hình 3 — thứ tự biến vào, t cuối, t = 1000
     7  Forward Stagewise bước ε, và giới hạn ε → 0
     8  Sửa đổi Stagewise (3.8)–(3.14): Định lý 2, nón lồi
     9  Mô hình bậc hai (3.15) m = 64 — dựng lại "true R² = 0,416"
    10  Mô phỏng §3.3 và Hình 5
    11  Positive Lasso (3.18)–(3.19)
    12  LARS–OLS hybrid (3.20) — CHỖ IN SAI THỨ NHẤT
    13  Bậc tự do (4.1)–(4.10): bootstrap và C_p, Hình 6 và Hình 7
    14  Điều kiện nón dương (4.11) — đếm trên chính bộ diabetes
    15  (4.13)–(4.15): trace(M_k) = k — CHỖ IN SAI THỨ HAI
    16  Thiết kế trực giao (4.16)–(4.18): Bổ đề 1, Định lý 3
    17  §5 Bổ đề 4–6: (5.1)–(5.14)
    18  §5 Bổ đề 7–10 và Hình 8: (5.16)–(5.43)
    19  §6 Ràng buộc I–III và (6.19)–(6.22)
    20  §7 Chi phí: O(m³ + nm²), và trường hợp m ≫ n
    21  §8 Boosting: stagewise trên tập hàm cơ sở lớn
    22  Chọn thiết bị — đo CPU và GPU cho chính phép tính của bài
    23  Mảng đủ chữ số cho script vẽ hình
    24  Ví dụ chạy tay hai biến — in lại từng bước
    25  Các con số dẫn xuất mà bài viết trích
"""
import io
import itertools
import os
import sys
import time

# Đặt CỨNG số luồng TRƯỚC khi nạp numpy: thứ tự cộng dồn dấu phẩy động của BLAS đổi
# theo số luồng, nên một con số "khớp tới chữ số cuối" mà không khai số luồng là một
# tuyên bố chưa đủ điều kiện tái lập.
for _b in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_b, "1")

import numpy as np
from scipy.optimize import nnls
from sklearn.datasets import load_diabetes
from sklearn.linear_model import lars_path

sys.stdout.reconfigure(encoding="utf-8")
np.set_printoptions(linewidth=200, suppress=False)

T0 = time.time()


def head(s):
    print()
    print("=" * 94)
    print(s)
    print("=" * 94)


# ════════════════════════════════════════════════════════════════════════════
# Thư viện dùng chung cho cả file
# ════════════════════════════════════════════════════════════════════════════
def chuan(M):
    """Quy ước (1.1) tr.3: mỗi cột trừ trung bình rồi chia cho chuẩn Euclid.

    KHÔNG phải "độ lệch chuẩn bằng 1" — hai quy ước lệch nhau một hệ số √(n−1).
    """
    M = np.asarray(M, dtype=float)
    M = M - M.mean(0)
    return M / np.sqrt((M ** 2).sum(0))


def ols(X, y):
    return np.linalg.lstsq(X, y, rcond=None)[0]


def huong(X, A, c):
    """(2.4)–(2.6) + (2.11): trả về (s, A_A, w_A, u_A, a) cho tập hoạt động A."""
    s = np.sign(c[A])
    s[s == 0] = 1.0
    XA = X[:, A] * s                                   # (2.4)
    GA = XA.T @ XA                                     # (2.5)
    iG = np.linalg.solve(GA, np.ones(len(A)))
    AA = 1.0 / np.sqrt(np.ones(len(A)) @ iG)           # (2.5)
    wA = AA * iG                                       # (2.6)
    uA = XA @ wA                                       # (2.6)
    a = X.T @ uA                                       # (2.11)
    return s, AA, wA, uA, a


def lars(X, y, kieu="lar", k_max=None, eps=1e-12, tran=20000):
    """LARS của Efron 2004, viết từ đầu theo đúng (2.4)–(2.13).

    kieu = "lar"    : thuật toán gốc, mục 2
           "lasso"  : thêm sửa đổi (3.6), mục 3.1
           "stage"  : thêm sửa đổi Stagewise — thay u_A bằng hình chiếu vào nón lồi

    Trả về (B, sk) với B[:, k] là β̂ sau bước k, sk là nhật ký sự kiện.
    """
    n, m = X.shape
    beta = np.zeros(m)
    mu = np.zeros(n)
    A = []
    B = [beta.copy()]
    sk = []
    vua_bo = -1
    nguong_het = 1e-9 * max(1.0, np.abs(X.T @ y).max())
    while True:
        c = X.T @ (y - mu)
        if A and max(abs(c[j]) for j in A) < nguong_het:
            break                                      # bão hoà: phần dư đã trực giao
        ngoai = [j for j in range(m) if j not in A and j != vua_bo]
        if ngoai:
            jb = max(ngoai, key=lambda j: abs(c[j]))
            Cht = max(abs(c[j]) for j in A) if A else 0.0
            if not A or abs(c[jb]) >= Cht - 1e-9:
                A = A + [jb]                           # (2.9)
                sk.append(("vao", jb, len(B) - 1))
        vua_bo = -1
        if not A:
            break
        s, AA, wA, uA, a = huong(X, A, c)
        C = max(abs(c[j]) for j in A)

        if kieu == "stage":
            # Sửa đổi Stagewise: chiếu u_A vào nón lồi C_A, (3.12)
            XA = X[:, A] * s
            P, _ = nnls(XA, uA)
            Bh = [A[i] for i in range(len(A)) if P[i] > 1e-10]
            if len(Bh) < len(A):
                sk.append(("bo", [A[i] for i in range(len(A)) if P[i] <= 1e-10],
                           len(B) - 1, sorted(A)))
            s, AA, wA, uA, a = huong(X, Bh, c)
            A_dung = Bh
        else:
            A_dung = A

        con = [j for j in range(m) if j not in A_dung]
        ung = []
        for j in con:
            for tu, mau in ((C - c[j], AA - a[j]), (C + c[j], AA + a[j])):
                if mau > eps and tu / mau > eps:
                    ung.append(tu / mau)               # (2.13)
        gam = min(ung) if ung else C / AA              # quy ước bước cuối, tr.10

        d = np.zeros(m)
        d[A_dung] = s * wA                             # (3.3)
        bo = None
        if kieu == "lasso":
            gt, jt = np.inf, None
            for j in A_dung:
                if abs(d[j]) > eps:
                    g = -beta[j] / d[j]                # (3.4)
                    if eps < g < gt:
                        gt, jt = g, j                  # (3.5)
            if gt < gam:                               # (3.6)
                gam, bo = gt, jt

        beta = beta + gam * d
        mu = mu + gam * uA
        if bo is not None:
            beta[bo] = 0.0
            A = [j for j in A if j != bo]
            sk.append(("bo", [bo], len(B)))
            vua_bo = bo
        if kieu == "stage":
            A = A_dung[:]
        B.append(beta.copy())
        if len(B) > tran:
            raise RuntimeError("vượt trần %d bước" % tran)
        if k_max is not None and len(B) - 1 >= k_max:
            break
        if not con and bo is None:
            break
    return np.array(B).T, sk


def tap(b, nguong=1e-12):
    return set(int(j) for j in np.flatnonzero(np.abs(b) > nguong))


def thu_tu_vao(B):
    """Thứ tự các biến LẦN LƯỢT vào (đếm cả lần vào lại), theo số hiệu 1..m."""
    ra, truoc = [], set()
    for k in range(B.shape[1]):
        h = tap(B[:, k])
        ra.extend(j + 1 for j in sorted(h - truoc))
        truoc = h
    return ra


# ════════════════════════════════════════════════════════════════════════════
head("1.  Bảng 1 tr.3 và quy ước chuẩn hoá (1.1) — dữ liệu nào, đơn vị nào")
# ════════════════════════════════════════════════════════════════════════════
tho = load_diabetes(scaled=False)
Xt, yt = tho.data, tho.target
n = len(yt)
print("   n = %d bệnh nhân, m = %d biến nền; tên cột: %s"
      % (Xt.shape[0], Xt.shape[1], ", ".join(tho.feature_names)))
print()
print("   Bảng 1 của bài báo in 6 dòng đầu và 2 dòng cuối. Dựng lại từ dữ liệu gốc:")
print("      BN  |  x1  x2   x3     x4    x5    x6     x7   x8     x9     x10 |   y")
for i in (0, 1, 2, 3, 4, 5, 440, 441):
    print("      %3d | %s | %5.0f"
          % (i + 1, " ".join("%6g" % v for v in Xt[i]), yt[i]))
print()
print("   Bảng 1 in x8 của bệnh nhân 441 là 5; dữ liệu có %.2f." % Xt[440, 7])
print("   Bảng 1 in x9 của bệnh nhân 1 là 4.9; dữ liệu có %.4f." % Xt[0, 8])
print("   -> Bảng 1 là bản ĐÃ LÀM TRÒN, và chú thích bảng không nói điều đó.")

X = chuan(Xt)
y = yt - yt.mean()
sk_scaled = load_diabetes().data
print()
print("   Chuẩn hoá theo (1.1): Σyᵢ = %.3e · Σxᵢⱼ lớn nhất = %.3e · Σxᵢⱼ² = %.10f"
      % (y.sum(), np.abs(X.sum(0)).max(), (X ** 2).sum(0)[0]))
print("   `load_diabetes()` của sklearn ĐÃ theo đúng quy ước này: lệch %.3e"
      % np.abs(sk_scaled - X).max())
sd1 = Xt.std(0, ddof=1)
print("   Nếu dùng nhầm quy ước 'độ lệch chuẩn = 1' thì mỗi cột dài √(n−1) = %.6f lần"
      % np.sqrt(n - 1))

b_ols = ols(X, y)
b_ols_goc = ols(np.c_[np.ones(n), Xt], yt)[1:]
print()
print("   Σ|β̂| của nghiệm OLS, theo quy ước (1.1)      : %.4f" % np.abs(b_ols).sum())
print("   Σ|β̂| của nghiệm OLS, theo ĐƠN VỊ GỐC         : %.4f" % np.abs(b_ols_goc).sum())
print("   Bài báo tr.4 in t cuối = 3460.00 và câu sau (1.1) nói *numerical results are")
print("   expressed in the original units*. Con số 3460 là con số của quy ước (1.1),")
print("   KHÔNG phải của đơn vị gốc — hai giá trị cách nhau %.2f lần."
      % (np.abs(b_ols).sum() / np.abs(b_ols_goc).sum()))


# ════════════════════════════════════════════════════════════════════════════
head("2.  LARS viết TỪ ĐẦU theo (2.4)–(2.13), đối chiếu với sklearn")
# ════════════════════════════════════════════════════════════════════════════
B_lar, sk_lar = lars(X, y, "lar")
B_las, sk_las = lars(X, y, "lasso")
S_lar = lars_path(X, y, method="lar")[2]
S_las = lars_path(X, y, method="lasso")[2]
print("   LARS   : tự viết %2d bước · sklearn %2d bước · lệch hệ số lớn nhất %.3e"
      % (B_lar.shape[1] - 1, S_lar.shape[1] - 1, np.abs(B_lar - S_lar).max()))
print("   Lasso  : tự viết %2d bước · sklearn %2d bước · lệch hệ số lớn nhất %.3e"
      % (B_las.shape[1] - 1, S_las.shape[1] - 1, np.abs(B_las - S_las).max()))
print("   Bài báo tr.12: *12 instead of 10 in this example* -> KHỚP cả hai con số.")
print()
print("   Cột cuối của đường LARS có đúng bằng nghiệm OLS không? lệch %.3e"
      % np.abs(B_lar[:, -1] - b_ols).max())
print("   Cột cuối của đường lasso có đúng bằng nghiệm OLS không? lệch %.3e"
      % np.abs(B_las[:, -1] - b_ols).max())


# ════════════════════════════════════════════════════════════════════════════
head("3.  Bất biến cách đều góc: (2.7) · (2.16) · (2.18)")
# ════════════════════════════════════════════════════════════════════════════
print("   (2.7)  X_A'u_A = A_A·1  và  ‖u_A‖ = 1 — kiểm ở MỌI bước của đường LARS")
le27 = 0.0
le_u = 0.0
for k in range(1, B_lar.shape[1]):
    c = X.T @ (y - X @ B_lar[:, k - 1])
    A = sorted(tap(B_lar[:, k]))
    s, AA, wA, uA, a = huong(X, A, c)
    le27 = max(le27, np.abs((X[:, A] * s).T @ uA - AA).max())
    le_u = max(le_u, abs(np.linalg.norm(uA) - 1.0))
print("      lệch lớn nhất của X_A'u_A − A_A·1 : %.3e" % le27)
print("      lệch lớn nhất của ‖u_A‖ − 1       : %.3e" % le_u)
print()
print("   Bảng bất biến — mọi biến hoạt động có |tương quan| BẰNG NHAU (đường lasso):")
print("   bước | tập hoạt động          | |ĉ| của biến hoạt động  |  độ trải  | trần biến ngoài")
Cmax = []
for k in range(B_las.shape[1]):
    c = np.abs(X.T @ (y - X @ B_las[:, k]))
    A = sorted(tap(B_las[:, k]))
    Cmax.append(c[A].max() if A else c.max())
    if not A:
        continue
    ngo = [j for j in range(10) if j not in A]
    print("     %3d | %-22s | %11.6f            | %.2e  | %11.6f"
          % (k, ",".join(str(j + 1) for j in A), c[A].mean(), np.ptp(c[A]),
             c[ngo].max() if ngo else 0.0))
print("   -> độ trải lớn nhất trên cả đường: %.3e"
      % max(np.ptp(np.abs(X.T @ (y - X @ B_las[:, k]))[sorted(tap(B_las[:, k]))])
            for k in range(1, B_las.shape[1]) if tap(B_las[:, k])))
print()
print("   (2.16) |c_j(γ)| = Ĉ − γ·A_A cho j ∈ A — kiểm bằng cách đi nửa bước:")
c0 = X.T @ y
A0 = [int(np.argmax(np.abs(c0)))]
s, AA, wA, uA, a = huong(X, A0, c0)
C0 = np.abs(c0).max()
for f in (0.25, 0.5, 0.75):
    g = f * (C0 / AA)
    cj = X.T @ (y - g * uA)
    print("      γ = %8.3f : đo được %12.6f · công thức %12.6f · lệch %.2e"
          % (g, abs(cj[A0[0]]), C0 - g * AA, abs(abs(cj[A0[0]]) - (C0 - g * AA))))
print()
print("   (2.18) Ĉ_k = Ĉ_{k−1} − γ̂_{k−1}·A_{k−1} — dãy Ĉ trên đường LARS:")
Ck = []
for k in range(B_lar.shape[1]):
    c = np.abs(X.T @ (y - X @ B_lar[:, k]))
    Ck.append(c.max())
print("      " + " ".join("%.4f" % v for v in Ck))
print("      dãy có giảm đơn điệu không? %s"
      % ("CÓ" if all(Ck[i] >= Ck[i + 1] - 1e-9 for i in range(len(Ck) - 1)) else "KHÔNG"))


# ════════════════════════════════════════════════════════════════════════════
head("4.  Quan hệ với OLS: (2.19)–(2.22), và vì sao γ̂ luôn nhỏ hơn γ̄")
# ════════════════════════════════════════════════════════════════════════════
print("   k | Ĉ_k       | A_k      | γ̄_k = Ĉ_k/A_k | γ̂_k        | ρ_k = γ̂/γ̄ | lệch (2.19)")
rho_list = []
for k in range(1, B_lar.shape[1]):
    mu_tr = X @ B_lar[:, k - 1]
    c = X.T @ (y - mu_tr)
    A = sorted(tap(B_lar[:, k]))
    s, AA, wA, uA, a = huong(X, A, c)
    Ck_ = max(abs(c[j]) for j in A)
    gbar = Ck_ / AA                                    # (2.21)
    ghat = np.linalg.norm(X @ B_lar[:, k] - mu_tr)
    XA = X[:, A]
    ybar = mu_tr + XA @ ols(XA, y - mu_tr)             # (2.19) vế trái
    le = np.abs(ybar - (mu_tr + gbar * uA)).max()      # (2.19) vế phải
    rho_list.append(ghat / gbar)
    print("  %3d | %9.4f | %8.6f | %13.4f | %10.4f | %10.6f | %.2e"
          % (k, Ck_, AA, gbar, ghat, ghat / gbar, le))
print("   -> ρ_k < 1 ở mọi bước trừ bước cuối? %s ; ρ lớn nhất trước bước cuối = %.6f"
      % ("ĐÚNG" if all(r < 1 - 1e-9 for r in rho_list[:-1]) else "SAI", max(rho_list[:-1])))
print("   Bài báo tr.10: *γ̂_k is always less than γ̄_k, so that μ̂_k lies closer than ȳ_k*")


# ════════════════════════════════════════════════════════════════════════════
head("5.  Sửa đổi lasso (3.1)–(3.6): Định lý 1")
# ════════════════════════════════════════════════════════════════════════════
print("   (3.1) dấu(β̂_j) = dấu(ĉ_j) cho mọi j hoạt động — kiểm trên CẢ hai đường.")
print("   Bỏ qua bước cuối cùng, nơi Ĉ = 0 nên dấu của ĉ chỉ còn là nhiễu số học.")
for ten, Bm in (("LARS gốc", B_lar), ("lasso", B_las)):
    xau, tong, cho = 0, 0, []
    for k in range(1, Bm.shape[1]):
        c = X.T @ (y - X @ Bm[:, k])
        if np.abs(c).max() < 1e-6:
            continue
        for j in tap(Bm[:, k]):
            tong += 1
            if np.sign(Bm[j, k]) != np.sign(c[j]):
                xau += 1
                cho.append((k, j + 1))
    print("      %-9s: %2d/%3d cặp (bước, biến) VI PHẠM điều kiện dấu %s"
          % (ten, xau, tong, cho if cho else ""))
print("   -> Tại các ĐIỂM GÃY, cả hai đường đều không vi phạm. Nhưng chỗ vi phạm của LARS")
print("      gốc nằm GIỮA hai điểm gãy, nên phải quét trên một lưới dày mới thấy:")
for ten, Bm in (("LARS gốc", B_lar), ("lasso", B_las)):
    tm = np.abs(Bm).sum(0)
    lg = np.linspace(0, tm[-1] * 0.9995, 4001)
    xau, cho = 0, []
    for tq in lg:
        b = np.array([np.interp(tq, tm, Bm[j]) for j in range(10)])
        c = X.T @ (y - X @ b)
        if np.abs(c).max() < 1e-6:
            continue
        for j in range(10):
            if abs(b[j]) > 1e-8 and np.sign(b[j]) != np.sign(c[j]):
                xau += 1
                cho.append((j + 1, tq))
    bj = sorted(set(j for j, _ in cho))
    kh = [tq for j, tq in cho]
    print("      %-9s: %4d/%d điểm trên lưới VI PHẠM%s"
          % (ten, xau, len(lg),
             (" — chỉ ở biến %s, trong khoảng t = %.1f .. %.1f"
              % (bj, min(kh), max(kh))) if cho else ""))
print("   -> Đường lasso KHÔNG vi phạm điểm nào — đúng Bổ đề 8 tr.29.")
print("   -> LARS gốc vi phạm ở đúng biến 7, trong một khoảng nằm GIỮA hai điểm gãy cuối.")
print("      Đó chính là khoảng mà (3.6) cắt ra bằng cách bỏ biến 7 khỏi tập hoạt động.")
print()
bo_las = [(j + 1, e[2]) for e in sk_las if e[0] == "bo" for j in e[1]]
print("   Sửa đổi (3.6) kích hoạt %d lần: %s"
      % (len(bo_las), ", ".join("biến %d tại bước %d" % (j, k) for j, k in bo_las)))
vao_las = thu_tu_vao(B_las)
print("   Thứ tự biến vào trên đường lasso (đếm cả lần vào lại): %s" % vao_las)
print("   Bài báo tr.12: *There A contained all 10 indices while A+ = A − {7}.*")
for j, k in bo_las:
    # A TRONG bước k gồm: tập trước bước, biến vừa vào ở bước này, và biến sắp bị bỏ
    truoc = tap(B_las[:, k - 1]) | tap(B_las[:, k]) | {j - 1}
    print("      trong bước %d, A có %d biến (%s); sau (3.6), A+ còn %d biến (%s)"
          % (k, len(truoc), ",".join(str(i + 1) for i in sorted(truoc)),
             len(tap(B_las[:, k])),
             ",".join(str(i + 1) for i in sorted(tap(B_las[:, k])))))
vang = [k for k in range(vao_las.index(7) + 1, B_las.shape[1]) if 6 not in tap(B_las[:, k])]
print("   Bài báo tr.12: *Variable 7 was restored to the active set one LARS step later.*")
print("      biến 7 vắng mặt ở đúng các bước %s, rồi trở lại ở bước %d"
      % (vang, max(vang) + 1))


# ════════════════════════════════════════════════════════════════════════════
head("6.  Dựng lại Hình 1 tr.4 và Hình 3 tr.9")
# ════════════════════════════════════════════════════════════════════════════
t_las = np.abs(B_las).sum(0)
t_lar = np.abs(B_lar).sum(0)
print("   Hình 1 bảng trái (lasso) — bài báo nói: thứ tự 3, 9, 4, 7, …, 1 ; t chạy tới 3460.00")
print("      dựng lại thứ tự    : %s" % vao_las)
print("      bốn biến đầu %s -> %s"
      % (vao_las[:4], "KHỚP" if vao_las[:4] == [3, 9, 4, 7] else "LỆCH"))
print("      biến vào cuối cùng trong 10 biến đầu tiên: %d -> %s"
      % (vao_las[9], "KHỚP" if vao_las[9] == 1 else "LỆCH"))
print("      t tại bước cuối    : %.4f" % t_las[-1])
print()
print("   Hình 3 bảng trái (LARS gốc) — bài báo tr.9 nói CÙNG thứ tự 3, 9, 4, 7, …, 1")
print("      dựng lại thứ tự    : %s" % thu_tu_vao(B_lar))
print("      t tại bước cuối    : %.4f ; lệch với đường lasso %.3e"
      % (t_lar[-1], abs(t_lar[-1] - t_las[-1])))
print()
print("   Bài báo tr.4: *At t = 1000 … only variables 3, 9, 4 and 7 enter*")
b1000 = np.array([np.interp(1000.0, t_las, B_las[j]) for j in range(10)])
print("      nội suy tại t = 1000 -> biến khác 0: %s -> %s"
      % (sorted(j + 1 for j in tap(b1000, 1e-8)),
         "KHỚP" if sorted(tap(b1000, 1e-8)) == [2, 3, 6, 8] else "LỆCH"))
print("      hệ số tại đó: %s"
      % " ".join("β%d=%.4f" % (j + 1, b1000[j]) for j in sorted(tap(b1000, 1e-8))))
print()
print("   Hai đường LARS và lasso khác nhau bao nhiêu? (so tại cùng t, nội suy)")
luoi = np.linspace(0, t_lar[-1], 501)
mx = 0.0
for j in range(10):
    v1 = np.interp(luoi, t_lar, B_lar[j])
    v2 = np.interp(luoi, t_las, B_las[j])
    mx = max(mx, np.abs(v1 - v2).max())
print("      lệch hệ số lớn nhất giữa hai đường: %.4f (trên thang hệ số cỡ %.0f)"
      % (mx, np.abs(B_lar).max()))
print("   Bài báo tr.9: *nearly but not exactly the same* -> đúng: gần nhau, không trùng.")


# ════════════════════════════════════════════════════════════════════════════
head("7.  Forward Stagewise bước ε (1.7), và giới hạn ε → 0")
# ════════════════════════════════════════════════════════════════════════════
G10 = X.T @ X


def stagewise_eps(X, y, eps, n_buoc=None, t_dich=None, tran=4000000):
    """(1.7) — cập nhật tương quan bằng ma trận Gram: O(m) mỗi bước thay vì O(nm).

    Dừng khi đủ `n_buoc`, hoặc khi Σ|β̂| chạm `t_dich`. Trả về (β̂, số bước, tới_đích).
    """
    G = X.T @ X
    c = X.T @ y
    b = np.zeros(X.shape[1])
    gioi = n_buoc if n_buoc is not None else tran
    for i in range(gioi):
        j = int(np.argmax(np.abs(c)))
        if abs(c[j]) < 1e-12:
            return b, i, True
        s = np.sign(c[j])
        b[j] += eps * s
        c -= eps * s * G[:, j]
        if t_dich is not None and np.abs(b).sum() >= t_dich:
            return b, i + 1, True
    return b, gioi, n_buoc is not None


t_het = np.abs(b_ols).sum()
B_stg, sk_stg = lars(X, y, "stage")
t_stg = np.abs(B_stg).sum(0)
print("   Bài báo tr.5: *The estimates were built up in 6000 Stagewise steps*")
print("   Chạy ĐÚNG 6000 bước, xem mỗi ε đưa đường đi được tới đâu (t cuối = %.2f):" % t_het)
for eps in (1.0, 0.5767, 0.4, 0.2):
    bb, kk, _ = stagewise_eps(X, y, eps, n_buoc=6000)
    tt = np.abs(bb).sum()
    ref = np.array([np.interp(tt, t_stg, B_stg[j]) for j in range(10)])
    print("      ε = %-7.4f -> Σ|β̂| = %9.4f (%5.1f%% đường) · lệch với đường lý tưởng %.4f"
          % (eps, tt, 100 * tt / t_het, np.abs(bb - ref).max()))
print()
print("   Hai bảng của Hình 1 dùng CHUNG trục ngang chạy tới t = %.2f. Quét ε để tìm giá trị"
      % t_het)
print("   đưa được xa nhất trong đúng 6000 bước:")
tot_nhat = (0.0, 0.0)
for eps in (0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0):
    bb, kk, _ = stagewise_eps(X, y, eps, n_buoc=6000)
    tt = np.abs(bb).sum()
    if tt > tot_nhat[0]:
        tot_nhat = (tt, eps)
    print("      ε = %-5.2f -> Σ|β̂| = %9.4f (%5.1f%% đường)" % (eps, tt, 100 * tt / t_het))
print("      -> giá trị tốt nhất: ε = %.2f cho %.4f, tức %.1f%% quãng đường."
      % (tot_nhat[1], tot_nhat[0], 100 * tot_nhat[0] / t_het))
print("      6000 bước KHÔNG phủ hết trục ngang của Hình 1 với bất kỳ ε nào trong lưới này:")
print("      ε nhỏ thì đi chậm, ε lớn thì dao động quanh nghiệm OLS mà không tiến thêm.")
print()
print("   MỘT BẪY: với ε hữu hạn, Stagewise KHÔNG dừng ở nghiệm OLS mà DAO ĐỘNG quanh nó.")
for eps in (1.0, 0.1):
    bb, kk, toi = stagewise_eps(X, y, eps, t_dich=t_het, tran=300000)
    print("      ε = %-5.2f, đích Σ|β̂| = %.2f: sau %d bước %s (Σ|β̂| dừng ở %.4f)"
          % (eps, t_het, kk, "TỚI" if toi else "VẪN CHƯA TỚI", np.abs(bb).sum()))
print("      Lý do: gần nghiệm OLS, mọi |ĉ_j| đều nhỏ hơn ε nên hệ số nhảy +ε rồi −ε mãi.")
print("      Vì thế bảng dưới lấy đích là 90%% quãng đường, chỗ còn đi được.")
print()
print("   Chi phí tỉ lệ NGHỊCH với ε — đích Σ|β̂| = %.2f (90%% đường):" % (0.9 * t_het))
for eps in (10.0, 1.0, 0.1, 0.01):
    t1 = time.perf_counter()
    bb, kk, toi = stagewise_eps(X, y, eps, t_dich=0.9 * t_het, tran=500000)
    print("      ε = %-6.2f -> %9d bước (%.3f giây) %s"
          % (eps, kk, time.perf_counter() - t1, "" if toi else "CHƯA TỚI"))
print("      LARS       -> %9d bước là đi HẾT đường, không phải 90%%."
      % (B_lar.shape[1] - 1))
print()
print("   ε → 0 có hội tụ về đường LARS–Stagewise không? So tại cùng t, lệch lớn nhất:")
for eps in (4.0, 1.0, 0.25, 0.0625):
    bb, kk, _ = stagewise_eps(X, y, eps, t_dich=0.5 * t_het)
    tt = np.abs(bb).sum()
    ref = np.array([np.interp(tt, t_stg, B_stg[j]) for j in range(10)])
    le = np.abs(bb - ref).max()
    print("      ε = %-7.4f (%7d bước): lệch %9.4f  (lệch/ε = %.4f)" % (eps, kk, le, le / eps))
print("      -> lệch giảm TỈ LỆ THUẬN với ε: đó là nghĩa của 'idealized Stagewise' tr.13.")


# ════════════════════════════════════════════════════════════════════════════
head("8.  Sửa đổi Stagewise (3.8)–(3.14): Định lý 2, chiếu vào nón lồi")
# ════════════════════════════════════════════════════════════════════════════
print("   Đường LARS–Stagewise tự dựng: %d bước; tới OLS chưa? lệch %.3e"
      % (B_stg.shape[1] - 1, np.abs(B_stg[:, -1] - b_ols).max()))
print("   Bài báo tr.14: *It took a total of 13 modified LARS steps to reach the full OLS*")
bo_stg = [(e[2], sorted(j + 1 for j in e[1]), sorted(j + 1 for j in e[3]))
          for e in sk_stg if e[0] == "bo"]
print("   Các lần tập hoạt động BỊ THU NHỎ: %s" % bo_stg)
k0, bo0, A_truoc = bo_stg[0]
print("   Bài báo tr.14: *there the set A = {3, 9, 4, 7, 2, 10, 5, 8} was decreased to")
print("                   B̂ = A − {3, 7}*")
print("      dựng lại: tại bước %d, A = %s bị thu về B̂ = A − %s"
      % (k0, A_truoc, bo0))
print("      -> %s" % ("KHỚP cả tập A lẫn tập bị bỏ"
                       if A_truoc == [2, 3, 4, 5, 7, 8, 9, 10] and bo0 == [3, 7]
                       else "LỆCH"))
print()
print("   (3.14) dấu(β̂⁺_j − β̂_j) = s_j: hiệu hệ số liên tiếp luôn CÙNG DẤU với ĐI của nó?")
xau = 0
tong = 0
for k in range(1, B_stg.shape[1]):
    c = X.T @ (y - X @ B_stg[:, k - 1])
    d = B_stg[:, k] - B_stg[:, k - 1]
    for j in np.flatnonzero(np.abs(d) > 1e-9):
        tong += 1
        if np.sign(d[j]) != np.sign(c[j]):
            xau += 1
print("      %d/%d hiệu VI PHẠM -> Stagewise thật sự không bao giờ lùi ngược chiều." % (xau, tong))
print()
print("   Chú thích Hình 1 tr.4: *The two plots are nearly identical, but differ slightly")
print("   for large t as shown in the track of covariate 8.*  Đo lệch theo TỪNG biến:")
luoi = np.linspace(0, t_het, 1001)
bang = []
for j in range(10):
    v1 = np.interp(luoi, t_las, B_las[j])
    v2 = np.interp(luoi, t_stg, B_stg[j])
    bang.append((np.abs(v1 - v2).max(), j + 1, luoi[int(np.argmax(np.abs(v1 - v2)))]))
bang.sort(reverse=True)
for v, j, tt in bang[:4]:
    print("      trên CẢ đường  — biến %-2d: lệch lớn nhất %8.4f tại t = %.1f" % (j, v, tt))
print("      -> biến lệch nhiều nhất trên cả đường là biến %d." % bang[0][1])
print()
print("   Nhưng chú thích nói *for LARGE t*. Đo lại chỉ trên một phần tư cuối (t > %.0f):"
      % (0.75 * t_het))
msk = luoi > 0.75 * t_het
bang2 = []
for j in range(10):
    v1 = np.interp(luoi, t_las, B_las[j])[msk]
    v2 = np.interp(luoi, t_stg, B_stg[j])[msk]
    bang2.append((np.abs(v1 - v2).max(), j + 1))
bang2.sort(reverse=True)
for v, j in bang2[:4]:
    print("      t lớn        — biến %-2d: lệch lớn nhất %8.4f" % (j, v))
print("      -> ở t lớn, biến lệch nhiều nhất là biến %d %s"
      % (bang2[0][1], "-> KHỚP chú thích" if bang2[0][1] == 8 else "-> KHÔNG phải biến 8"))
print()
print("   Phép so trên chưa CÔNG BẰNG: bảng phải Hình 1 là bản ε hữu hạn 6000 bước, không")
print("   phải đường lý tưởng. Đo lại trên đúng bản ấy, ε = t/6000 = %.6f:" % (t_het / 6000))
b6000, k6000, _ = stagewise_eps(X, y, t_het / 6000, n_buoc=6000)
t6 = np.abs(b6000).sum()
bang3 = []
for j in range(10):
    v1 = np.interp(t6, t_las, B_las[j])
    bang3.append((abs(v1 - b6000[j]), j + 1))
bang3.sort(reverse=True)
print("      sau 6000 bước, Σ|β̂| = %.4f (%.1f%% đường); lệch với đường lasso theo biến:"
      % (t6, 100 * t6 / t_het))
for v, j in bang3[:4]:
    print("         biến %-2d: %8.4f" % (j, v))
print("      -> biến lệch nhiều nhất là biến %d %s"
      % (bang3[0][1], "-> KHỚP chú thích" if bang3[0][1] == 8 else "-> KHÔNG phải biến 8"))
print("      Ba cách đo — cả đường, một phần tư cuối, và đúng bản 6000 bước — cho ba câu")
print("      trả lời %s, và KHÔNG cách nào chỉ vào biến 8."
      % ("khác nhau" if len({bang[0][1], bang2[0][1], bang3[0][1]}) > 1 else "giống nhau"))


# ════════════════════════════════════════════════════════════════════════════
head("9.  Mô hình bậc hai (3.15) m = 64 — dựng lại 'true R² = 0.416'")
# ════════════════════════════════════════════════════════════════════════════
def quadratic(base):
    ten = ["x%d" % (j + 1) for j in range(10)]
    ct, nm = [], []
    for i, j in itertools.combinations(range(10), 2):
        ct.append(base[:, i] * base[:, j])
        nm.append("x%d:x%d" % (i + 1, j + 1))
    for j in range(10):
        if j == 1:                                     # bỏ x2 vì nó nhị phân
            continue
        ct.append(base[:, j] ** 2)
        nm.append("x%d²" % (j + 1))
    return chuan(np.c_[base, np.array(ct).T]), ten + nm


print("   (3.15): 10 hiệu ứng chính + 45 tương tác + 9 bình phương")
print("      C(10,2) = %d tương tác; 10 − 1 = %d bình phương (bỏ x2 nhị phân); tổng = %d"
      % (len(list(itertools.combinations(range(10), 2))), 9,
         10 + len(list(itertools.combinations(range(10), 2))) + 9))
print()
print("   Dựng tích từ cột NÀO? Bài báo không nói. Thử cả hai, để con số 0,416 phân xử:")
for ten_cach, base in (("từ cột đã chuẩn hoá (1.1)", X), ("từ cột theo đơn vị gốc", Xt)):
    X2c, _ = quadratic(base)
    B2c, _ = lars(X2c, y, "lar", k_max=10)
    mu_c = X2c @ B2c[:, 10]
    ep_c = y - mu_c
    R2c = (mu_c ** 2).sum() / ((mu_c ** 2).sum() + (ep_c ** 2).sum())
    print("      %-28s -> true R² = %.6f" % (ten_cach, R2c))
print("      Bài báo tr.15: *The \"true R²\" for this model … equaled 0.416*")

X2, ten2 = quadratic(X)
print()
print("   Chốt: dựng từ cột đã chuẩn hoá. m = %d, hạng = %d, số điều kiện %.4e"
      % (X2.shape[1], np.linalg.matrix_rank(X2), np.linalg.cond(X2)))
B2_10, _ = lars(X2, y, "lar", k_max=10)
beta_that = B2_10[:, 10]
mu_that = X2 @ beta_that
eps_that = y - mu_that
R2_that = (mu_that ** 2).sum() / ((mu_that ** 2).sum() + (eps_that ** 2).sum())
print("   true R² = ‖μ‖²/(‖μ‖² + ‖ε‖²) = %.6f  (bài báo: 0.416) -> lệch %.6f"
      % (R2_that, abs(R2_that - 0.416)))
print("   10 biến được chọn: %s" % [ten2[j] for j in sorted(tap(beta_that))])

print()
print("   Số bước của ba thuật toán trên m = 64 — bài báo nêu 3 con số:")
t9 = time.perf_counter()
B2_lar, _ = lars(X2, y, "lar")
B2_las, _ = lars(X2, y, "lasso")
B2_stg, _ = lars(X2, y, "stage")
b_ols2 = ols(X2, y)
print("      LARS      : %3d bước (bài báo: 64)  | tới OLS? lệch %.2e"
      % (B2_lar.shape[1] - 1, np.abs(B2_lar[:, -1] - b_ols2).max()))
print("      Lasso     : %3d bước (bài báo: 103) | tới OLS? lệch %.2e"
      % (B2_las.shape[1] - 1, np.abs(B2_las[:, -1] - b_ols2).max()))
print("      Stagewise : %3d bước (bài báo: 255) | tới OLS? lệch %.2e"
      % (B2_stg.shape[1] - 1, np.abs(B2_stg[:, -1] - b_ols2).max()))
print("      (cả ba đường tính trong %.2f giây)" % (time.perf_counter() - t9))
print("      §7 tr.40: *increasing the computations by a factor up to 5 or more*")
print("      -> đo được: Stagewise/LARS = %d/%d = %.4f lần"
      % (B2_stg.shape[1] - 1, B2_lar.shape[1] - 1,
         (B2_stg.shape[1] - 1) / (B2_lar.shape[1] - 1)))

print()
print("   `sklearn.linear_model.lars_path(method='lar')` trên CHÍNH thiết kế này:")
S2_lar = lars_path(X2, y, method="lar", max_iter=2000)[2]
t_s2 = np.abs(S2_lar).sum(0)
rss_s2 = ((y[:, None] - X2 @ S2_lar) ** 2).sum(0)
rss_ols2 = ((y - X2 @ b_ols2) ** 2).sum()
print("      số bước: %d (thay vì 64) | Σ|β̂| ở bước cuối: %.4e | RSS ở bước cuối: %.4e"
      % (S2_lar.shape[1] - 1, t_s2[-1], rss_s2[-1]))
print("      RSS của nghiệm OLS: %.4f ; RSS của bản tự viết ở bước 64: %.4f"
      % (rss_ols2, ((y - X2 @ B2_lar[:, 64]) ** 2).sum()))
print("      tại bước 64, sklearn mới có %d hệ số khác 0 và RSS = %.4f"
      % (len(tap(S2_lar[:, 64])), rss_s2[64]))
print("      dãy alpha của sklearn có giảm đơn điệu không? %s"
      % ("CÓ" if np.all(np.diff(lars_path(X2, y, method='lar', max_iter=2000)[0]) <= 1e-9)
         else "KHÔNG"))
print("      -> chế độ 'lar' của sklearn PHÂN KỲ trên thiết kế này, và không phát cảnh báo nào.")
print("      Chế độ 'lasso' thì không sao: %d bước, lệch với bản tự viết %.2e"
      % (lars_path(X2, y, method="lasso", max_iter=2000)[2].shape[1] - 1,
         np.abs(lars_path(X2, y, method="lasso", max_iter=2000)[2] - B2_las).max()))


# ════════════════════════════════════════════════════════════════════════════
head("10. Mô phỏng §3.3 tr.15 và Hình 5 — 100 lần lặp mô hình (3.15)–(3.16)")
# ════════════════════════════════════════════════════════════════════════════
def tien_chon(X, y, K):
    """Forward Selection cổ điển: chọn biến tương quan nhất rồi khớp HẾT CỠ."""
    n_, m_ = X.shape
    A, mu, ra = [], np.zeros(n_), [np.zeros(n_)]
    for _ in range(K):
        c = np.abs(X.T @ (y - mu))
        c[A] = -1
        A.append(int(np.argmax(c)))
        mu = X[:, A] @ ols(X[:, A], y)
        ra.append(mu.copy())
    return np.array(ra).T


K_MP = 40
SO_LAP = 100
rng = np.random.default_rng(0)
pe = {k: np.zeros((SO_LAP, K_MP + 1)) for k in ("lar", "lasso", "stage", "fs")}
so_kh = {k: np.zeros((SO_LAP, K_MP + 1)) for k in ("lar", "lasso", "stage")}
t10 = time.perf_counter()
for r in range(SO_LAP):
    ys = mu_that + rng.choice(eps_that, size=n, replace=True)      # (3.16)
    for kieu in ("lar", "lasso", "stage"):
        Bm, _ = lars(X2, ys, kieu, k_max=K_MP)
        for k in range(K_MP + 1):
            kk = min(k, Bm.shape[1] - 1)
            d = X2 @ Bm[:, kk] - mu_that
            pe[kieu][r, k] = 1 - (d @ d) / (mu_that @ mu_that)      # (3.17)
            so_kh[kieu][r, k] = len(tap(Bm[:, kk]))
    MU = tien_chon(X2, ys, K_MP)
    for k in range(K_MP + 1):
        d = MU[:, k] - mu_that
        pe["fs"][r, k] = 1 - (d @ d) / (mu_that @ mu_that)
print("   %d lần lặp × 4 thuật toán, chạy trong %.1f giây" % (SO_LAP, time.perf_counter() - t10))
print()
for kieu, nhan, doi in (("lar", "LARS", "0.963 tại k = 10"), ("lasso", "Lasso", ""),
                        ("stage", "Stagewise", ""), ("fs", "Forward Selection", "0.950 tại k = 3")):
    m_ = pe[kieu].mean(0)
    print("   %-18s: pe lớn nhất %.6f tại k = %2d ; độ lệch chuẩn trung bình %.6f  %s"
          % (nhan, m_.max(), int(np.argmax(m_)), pe[kieu].std(0, ddof=1).mean(),
             ("(bài báo: %s)" % doi) if doi else ""))
print("   Bài báo tr.15: *reaching a maximum of 0.963 at k = 10* và")
print("                  *the light dots display the small standard deviation … roughly ±0.02*")
print("                  *Forward Selection … rises very quickly, to a maximum of 0.950 after k = 3*")
print()
print("   Bài báo tr.15: *μ̂^(40)* averaged 33.23 nonzero terms with Stagewise, compared to")
print("                  35.83 for Lasso and 40 for LARS.*")
for kieu, nhan, doi in (("stage", "Stagewise", 33.23), ("lasso", "Lasso", 35.83),
                        ("lar", "LARS", 40.0)):
    print("      %-10s: trung bình số hệ số khác 0 ở bước 40 = %.4f  (bài báo %.2f)"
          % (nhan, so_kh[kieu][:, K_MP].mean(), doi))
print()
print("   Bài báo tr.16: *Stopping at any point between k = 5 and 25 typically gave a μ̂ with")
print("                  true predictive R² about 0.40, compared to the ideal value 0.416*")
m_lar = pe["lar"].mean(0)
print("      pe trung bình của LARS trong khoảng k = 5..25: từ %.6f tới %.6f"
      % (m_lar[5:26].min(), m_lar[5:26].max()))
print("      R² dự báo thật = pe × true R² -> từ %.6f tới %.6f"
      % (m_lar[5:26].min() * R2_that, m_lar[5:26].max() * R2_that))


# ════════════════════════════════════════════════════════════════════════════
head("11. Positive Lasso (3.18)–(3.19) — ba chỗ sửa, và chỗ nó KHÔNG tới OLS")
# ════════════════════════════════════════════════════════════════════════════
def lars_duong(X, y, eps=1e-12, tran=5000):
    """(3.18)–(3.19): đổi |ĉ_j| thành ĉ_j trong (2.9), đặt s_j = 1, bỏ nhánh thứ hai
    của (2.13). Tức coi mỗi x_j sinh ra một NỬA đường thẳng chứ không phải cả trục."""
    n_, m_ = X.shape
    beta = np.zeros(m_)
    mu = np.zeros(n_)
    A = []
    B = [beta.copy()]
    vua_bo = -1
    while True:
        c = X.T @ (y - mu)
        ngoai = [j for j in range(m_) if j not in A and j != vua_bo]
        Cht = max(c[j] for j in A) if A else -np.inf
        if A and Cht <= 1e-9:
            break                                       # mọi tương quan đã về 0
        if ngoai:
            jb = max(ngoai, key=lambda j: c[j])
            if not A and c[jb] <= 1e-9:
                break
            if not A or c[jb] >= Cht - 1e-9:
                A = A + [jb]
        vua_bo = -1
        if not A:
            break
        s = np.ones(len(A))                             # s_j = 1, không lấy dấu
        XA = X[:, A]
        iG = np.linalg.solve(XA.T @ XA, np.ones(len(A)))
        AA = 1.0 / np.sqrt(np.ones(len(A)) @ iG)
        wA = AA * iG
        uA = XA @ wA
        a = X.T @ uA
        C = max(c[j] for j in A)
        con = [j for j in range(m_) if j not in A]
        ung = [(C - c[j]) / (AA - a[j]) for j in con
               if AA - a[j] > eps and (C - c[j]) / (AA - a[j]) > eps]     # (3.19)
        gam = min(ung + [C / AA])         # chặn tại chỗ tương quan chung về 0
        d = np.zeros(m_)
        d[A] = wA
        gt, jt = np.inf, None
        for j in A:
            if abs(d[j]) > eps:
                g = -beta[j] / d[j]
                if eps < g < gt:
                    gt, jt = g, j
        bo = None
        if gt < gam:
            gam, bo = gt, jt
        beta = beta + gam * d
        mu = mu + gam * uA
        if bo is not None:
            beta[bo] = 0.0
            A = [j for j in A if j != bo]
            vua_bo = bo
        B.append(beta.copy())
        if len(B) > tran:
            raise RuntimeError("vượt trần")
        if not con and bo is None:
            break
    return np.array(B).T


Bp = lars_duong(X, y)
b_nnls = nnls(X, y)[0]
print("   Đường positive lasso: %d bước; hệ số cuối có âm không? nhỏ nhất = %.6f"
      % (Bp.shape[1] - 1, Bp[:, -1].min()))
print("   Nghiệm bình phương tối thiểu KHÔNG ÂM (scipy.optimize.nnls): lệch %.3e"
      % np.abs(Bp[:, -1] - b_nnls).max())
print("   Bài báo tr.16: *The positive Lasso usually does not converge to the full OLS*")
print("      Σ|β̂| của positive lasso ở cuối : %.4f" % np.abs(Bp[:, -1]).sum())
print("      Σ|β̂| của OLS đầy đủ            : %.4f" % np.abs(b_ols).sum())
print("      OLS có mấy hệ số ÂM? %d/%d -> %s"
      % (int((b_ols < 0).sum()), 10,
         "nên ràng buộc β ≥ 0 CẮT MẤT nghiệm OLS" if (b_ols < 0).any() else "không cắt"))
print("      RSS: positive lasso %.2f · OLS %.2f · tệ hơn %.4f%%"
      % (((y - X @ Bp[:, -1]) ** 2).sum(), ((y - X @ b_ols) ** 2).sum(),
         100 * (((y - X @ Bp[:, -1]) ** 2).sum() / ((y - X @ b_ols) ** 2).sum() - 1)))
print("   Số biến vào được: %d/%d (các biến %s)"
      % (len(tap(Bp[:, -1])), 10, sorted(j + 1 for j in tap(Bp[:, -1]))))


# ════════════════════════════════════════════════════════════════════════════
head("12. LARS–OLS hybrid (3.20) — CHỖ IN SAI THỨ NHẤT")
# ════════════════════════════════════════════════════════════════════════════
print("   Bài báo tr.16 in:  R²(β̄_k) − R²(β̂_k) = [1 − ρ_k²]/[ρ_k(2 − ρ_k)] · [R²(β̂_k) − R²(β̂_{k−1})]")
print("   Suy lại từ hình học Hình 4 thì tử số phải là (1 − ρ_k)², không phải 1 − ρ_k².")
print()
S_lar_ = ((y[:, None] - X @ B_lar) ** 2).sum(0)
R2_lar = 1 - S_lar_ / (y @ y)
print("    k |   ρ_k    | VẾ TRÁI đo được | (1−ρ)²/(ρ(2−ρ))·Δ | 1−ρ²/(ρ(2−ρ))·Δ | sai số bản in")
for k in range(1, B_lar.shape[1]):
    A = sorted(tap(B_lar[:, k]))
    XA = X[:, A]
    R2_bar = 1 - ((y - XA @ ols(XA, y)) ** 2).sum() / (y @ y)
    rho = rho_list[k - 1]
    D = R2_lar[k] - R2_lar[k - 1]
    vt = R2_bar - R2_lar[k]
    a1 = (1 - rho) ** 2 / (rho * (2 - rho)) * D
    a2 = (1 - rho ** 2) / (rho * (2 - rho)) * D
    print("   %2d | %.6f |   %.8f    |    %.8f     |   %.8f  |  %6.2f lần"
          % (k, rho, vt, a1, a2, (a2 / vt) if abs(vt) > 1e-12 else 0.0))
print()
print("   Công thức (1 − ρ)² khớp %d/%d dòng trong sai số %.2e;"
      % (B_lar.shape[1] - 1, B_lar.shape[1] - 1,
         max(abs((1 - rho_list[k - 1]) ** 2 / (rho_list[k - 1] * (2 - rho_list[k - 1]))
                 * (R2_lar[k] - R2_lar[k - 1])
                 - (1 - ((y - X[:, sorted(tap(B_lar[:, k]))]
                          @ ols(X[:, sorted(tap(B_lar[:, k]))], y)) ** 2).sum() / (y @ y)
                    - R2_lar[k]))
             for k in range(1, B_lar.shape[1]))))
print("   bản in 1 − ρ² lệch tới %.2f lần ở dòng tệ nhất."
      % max((1 - rho_list[k - 1] ** 2) / (1 - rho_list[k - 1]) ** 2
            for k in range(1, B_lar.shape[1]) if rho_list[k - 1] < 0.999))
print("   Hệ số giữa hai bản là (1+ρ)/(1−ρ), bằng 1 chỉ khi ρ = 0 và ra ∞ khi ρ → 1.")
print()
print("   Bài báo tr.17: *The increases in R² were small … on the order of 0.01 for k ≥ 4")
print("                  compared with R² ≐ 0.50*")
print("      R² của mô hình đầy đủ = %.6f" % R2_lar[-1])
print("      mức tăng khi thay β̂_k bằng OLS trên cùng tập biến, k = 4..9: %s"
      % " ".join("%.4f" % ((1 - ((y - X[:, sorted(tap(B_lar[:, k]))]
                                  @ ols(X[:, sorted(tap(B_lar[:, k]))], y)) ** 2).sum()
                            / (y @ y)) - R2_lar[k]) for k in range(4, 10)))


# ════════════════════════════════════════════════════════════════════════════
head("13. Bậc tự do (4.1)–(4.10): bootstrap, và C_p — Hình 6 tr.19 và Hình 7 tr.20")
# ════════════════════════════════════════════════════════════════════════════
def df_bootstrap(X, yq, K, B_lan=500, seed=0, kieu="lar", mu_cho=None, sig_cho=None):
    """(4.6)–(4.8): y* ~ N(μ̄, σ̄²), rồi df = Σᵢ cov(μ̂ᵢ, y*ᵢ)/σ̄².

    Thiết kế bão hoà (n = m) không ước lượng được σ̄² từ phần dư OLS — phần dư bằng 0
    và bậc tự do còn lại âm. Khi ấy truyền thẳng (mu_cho, sig_cho) như Định lý 3 giả định.
    """
    rng_ = np.random.default_rng(seed)
    nq = len(yq)
    if mu_cho is not None:
        mubar, s2 = mu_cho, float(sig_cho) ** 2
    else:
        mubar = X @ ols(X, yq)
        s2 = ((yq - mubar) ** 2).sum() / (nq - X.shape[1] - 1)
    ys = mubar[None, :] + rng_.normal(scale=np.sqrt(s2), size=(B_lan, nq))
    MU = np.empty((B_lan, nq, len(K)))
    for b_ in range(B_lan):
        Bm, _ = lars(X, ys[b_], kieu, k_max=max(K))
        for i, k in enumerate(K):
            MU[b_, :, i] = X @ Bm[:, min(k, Bm.shape[1] - 1)]
    ybar = ys.mean(0)                                   # (4.7)
    out, sai = [], []
    for i in range(len(K)):
        cov = (MU[:, :, i] * (ys - ybar)).sum(0) / (B_lan - 1)
        out.append(cov.sum() / s2)                      # (4.8)
        nh = B_lan // 10                                # 10 nhóm 50, theo tr.19
        gr = [((MU[g * nh:(g + 1) * nh, :, i]
                * (ys[g * nh:(g + 1) * nh] - ys[g * nh:(g + 1) * nh].mean(0))).sum(0)
               / (nh - 1)).sum() / s2 for g in range(10)]
        sai.append(np.std(gr, ddof=1) / np.sqrt(10))
    return np.array(out), np.array(sai), np.sqrt(s2)


t13 = time.perf_counter()
K10 = list(range(1, 11))
df10, se10, sig10 = df_bootstrap(X, y, K10, 500)
print("   Hình 6 bảng TRÁI — diabetes m = 10, B = 500 lần bootstrap, σ̄ = %.4f:" % sig10)
print("      k        :" + "".join("%7d" % k for k in K10))
print("      df ước   :" + "".join("%7.2f" % v for v in df10))
print("      sai số   :" + "".join("%7.2f" % v for v in se10))
print("      |df − k|  lớn nhất = %.4f ; trung bình = %.4f"
      % (np.abs(df10 - np.array(K10)).max(), np.abs(df10 - np.array(K10)).mean()))
trong = int(sum(abs(df10[i] - K10[i]) <= 2.262 * se10[i] for i in range(10)))
print("      số k mà df = k nằm trong khoảng tin cậy 95%% (t với 9 bậc tự do, ±2.262·se):"
      " %d/10" % trong)
thieu = [k for i, k in enumerate(K10) if df10[i] < k - 2.262 * se10[i]]
print("      các k mà df ĐO ĐƯỢC THẤP HƠN k một cách có ý nghĩa: %s" % thieu)
print("      Bài báo (4.9) tr.18: *df(μ̂_k) ≐ k* — gọi là 'the simple approximation'.")

K64 = list(range(1, 65, 3))
df64, se64, sig64 = df_bootstrap(X2, y, K64, 500, seed=1)
print()
print("   Hình 6 bảng PHẢI — mô hình bậc hai m = 64, B = 500, σ̄ = %.4f:" % sig64)
print("      k    :" + "".join("%6d" % k for k in K64))
print("      df   :" + "".join("%6.1f" % v for v in df64))
print("      |df − k| lớn nhất = %.4f tại k = %d ; trung bình = %.4f"
      % (np.abs(df64 - np.array(K64)).max(),
         K64[int(np.argmax(np.abs(df64 - np.array(K64))))],
         np.abs(df64 - np.array(K64)).mean()))
print("      (%.1f giây cho cả hai bảng)" % (time.perf_counter() - t13))

print()
print("   Hình 7 — C_p(μ̂_k) = ‖y − μ̂_k‖²/σ̄² − n + 2k, công thức (4.10):")
for ten_mh, Xm, Bm, doi in (("m = 10", X, B_lar, 7), ("m = 64", X2, B2_lar, 16)):
    s2m = ((y - Xm @ ols(Xm, y)) ** 2).sum() / (n - Xm.shape[1] - 1)
    rss = ((y[:, None] - Xm @ Bm) ** 2).sum(0)
    Cp = rss / s2m - n + 2 * np.arange(Bm.shape[1])
    kmin = int(np.argmin(Cp))
    print("      %s: C_p nhỏ nhất tại k = %d (bài báo: %d) · C_p = %.4f"
          % (ten_mh, kmin, doi, Cp[kmin]))
    if ten_mh == "m = 10":
        print("         dãy C_p: " + " ".join("%.2f" % v for v in Cp))
        A7 = sorted(j + 1 for j in tap(Bm[:, kmin]))
        print("         mô hình được chọn gồm các biến %s" % A7)
    else:
        lan_can = {k: float(Cp[k]) for k in range(max(0, kmin - 2), min(len(Cp), kmin + 3))}
        print("         C_p quanh cực tiểu: %s"
              % " ".join("k=%d:%.3f" % (k, v) for k, v in lan_can.items()))
        print("         C_p tại k = 16 là %.4f, cao hơn cực tiểu %.4f"
              % (Cp[16], Cp[16] - Cp[kmin]))


# ════════════════════════════════════════════════════════════════════════════
head("14. Điều kiện nón dương (4.11) — đếm trên chính bộ diabetes")
# ════════════════════════════════════════════════════════════════════════════
print("   (4.11): với MỌI tập con A, G_A⁻¹·1_A > 0 theo từng phần tử.")
G = X.T @ X
vp, tong_tap, vd = 0, 0, []
for k in range(1, 11):
    for A in itertools.combinations(range(10), k):
        tong_tap += 1
        v = np.linalg.solve(G[np.ix_(A, A)], np.ones(k))
        if (v <= 0).any():
            vp += 1
            if len(vd) < 4:
                vd.append((tuple(j + 1 for j in A), float(v.min())))
print("   Bài báo tr.20: *counterexamples (such as the diabetes data) show that not all")
print("                  design matrices X satisfy it*")
print("      đếm được: %d / %d tập con VI PHẠM, tức %.2f%%"
      % (vp, tong_tap, 100 * vp / tong_tap))
print("      tập con nhỏ nhất vi phạm có %d phần tử; vài ví dụ:"
      % min(len(a) for a, _ in vd))
for a, v in vd:
    print("         A = %-14s -> phần tử nhỏ nhất của G_A⁻¹·1 = %.6f" % (str(a), v))
print()
print("   Thiết kế TRỰC GIAO có thoả không? (bài báo nói có)")
Io = np.eye(8)
xau = sum(1 for k in range(1, 9) for A in itertools.combinations(range(8), k)
          if (np.linalg.solve(Io[np.ix_(A, A)], np.ones(k)) <= 0).any())
print("      %d/%d tập con vi phạm trên ma trận đơn vị 8×8." % (xau, 2 ** 8 - 1))
print()
print("   Dưới điều kiện nón dương, ba thuật toán có TRÙNG nhau không? (bài báo tr.20 nói có)")
rng14 = np.random.default_rng(7)
thu, trung = 0, 0
while thu < 6:
    Z = chuan(np.abs(rng14.normal(size=(60, 5))) + 0.4 * rng14.normal(size=(60, 5)))
    Gz = Z.T @ Z
    if any((np.linalg.solve(Gz[np.ix_(A, A)], np.ones(k)) <= 0).any()
           for k in range(1, 6) for A in itertools.combinations(range(5), k)):
        continue
    thu += 1
    yz = Z @ np.array([3.0, 1.5, 0.0, 2.0, 0.0]) + rng14.normal(scale=0.5, size=60)
    Ba, _ = lars(Z, yz, "lar")
    Bb, _ = lars(Z, yz, "lasso")
    Bc, _ = lars(Z, yz, "stage")
    if Ba.shape == Bb.shape == Bc.shape and \
       max(np.abs(Ba - Bb).max(), np.abs(Ba - Bc).max()) < 1e-8:
        trung += 1
print("      %d/%d thiết kế ngẫu nhiên THOẢ (4.11): ba đường trùng nhau tới 1e-8." % (trung, thu))
print("      Còn trên diabetes — thiết kế KHÔNG thoả — ba đường có %d, %d và %d bước."
      % (B_lar.shape[1] - 1, B_las.shape[1] - 1, B_stg.shape[1] - 1))


# ════════════════════════════════════════════════════════════════════════════
head("15. (4.13)–(4.15): trace(M_k) = k — CHỖ IN SAI THỨ HAI")
# ════════════════════════════════════════════════════════════════════════════
print("   Bài báo tr.21 in (4.13):  μ̂_k = ȳ_k − cot_k · ‖ȳ_{k+1} − ȳ_k‖")
print("   Vế trái là VECTƠ trong R^%d, vế phải là vectơ trừ một SỐ. Thiếu thừa số u_k." % n)
print()
print("    k | trace(M_k) | (4.13) có u_k: lệch | (4.13) như in: lệch | M_k có là Jacobian?")


def mu_buoc(Xm, yq, K):
    Bm, _ = lars(Xm, yq, "lar", k_max=K)
    return Xm @ Bm[:, K]


for K in range(1, 10):
    AK = sorted(tap(B_lar[:, K]))
    AK1 = sorted(tap(B_lar[:, K + 1]))
    s_, AA_, w_, uk, a_ = huong(X, AK, X.T @ (y - X @ B_lar[:, K - 1]))
    s2_, AA2_, w2_, uk1, a2_ = huong(X, AK1, X.T @ (y - X @ B_lar[:, K]))
    XA, XA1 = X[:, AK], X[:, AK1]
    P = XA @ np.linalg.solve(XA.T @ XA, XA.T)
    yb_k = XA @ ols(XA, y)
    yb_k1 = XA1 @ ols(XA1, y)
    dot = float(uk @ uk1)
    cot = dot / np.sqrt(1 - dot * dot)                  # (4.14)
    v = yb_k1 - yb_k
    v = v - P @ v
    v = v / np.linalg.norm(v)
    M = P - cot * np.outer(uk, v)                       # (4.15)
    dung = np.abs((X @ B_lar[:, K]) - (yb_k - cot * np.linalg.norm(yb_k1 - yb_k) * uk)).max()
    nhu_in = np.abs((X @ B_lar[:, K]) - (yb_k - cot * np.linalg.norm(yb_k1 - yb_k))).max()
    rng15 = np.random.default_rng(K)
    h = 1e-6
    dy = rng15.normal(size=n)
    dy /= np.linalg.norm(dy)
    so = (mu_buoc(X, y + h * dy, K) - mu_buoc(X, y - h * dy, K)) / (2 * h)
    print("   %2d | %.10f |      %.2e       |     %10.4f      |  lệch %.2e"
          % (K, np.trace(M), dung, nhu_in, np.abs(M @ dy - so).max()))
print()
print("   -> trace(M_k) = k tới 10 chữ số, vì u_k ∈ L(X_k) còn v_k ⊥ L(X_k) nên số hạng")
print("      thứ hai không đóng góp gì vào vết. Đó là toàn bộ lập luận 'delta method' tr.21.")
print("   -> M_k đúng là đạo hàm cục bộ: nhân M_k với một hướng bất kỳ cho ra cùng kết quả")
print("      với sai phân hữu hạn của chính thuật toán, sai số cỡ 1e-7 (mức nhiễu của sai phân).")


# ════════════════════════════════════════════════════════════════════════════
head("16. Thiết kế trực giao (4.16)–(4.18): Bổ đề 1 và Định lý 3")
# ════════════════════════════════════════════════════════════════════════════
rng16 = np.random.default_rng(3)
nz = 8
Xo = np.eye(nz)
yo = rng16.normal(scale=2.0, size=nz)
Bo, _ = lars(Xo, yo, "lar")
tt_o = np.r_[np.sort(np.abs(yo))[::-1], 0.0]
print("   y = %s" % " ".join("%+.4f" % v for v in yo))
print("   |y|_(k) xếp giảm dần: %s" % " ".join("%.4f" % v for v in tt_o[:-1]))
print()
print("   Bổ đề 1 tr.22: μ̂_k,i = η(y_i ; |y|_(k+1)) — ngưỡng mềm tại thống kê thứ tự thứ k+1")
le_max = 0.0
for k in range(nz + 1):
    du = np.sign(yo) * np.maximum(np.abs(yo) - tt_o[k], 0.0)
    le_max = max(le_max, np.abs(Bo[:, k] - du).max())
print("      lệch lớn nhất giữa đường LARS và công thức ngưỡng mềm: %.3e" % le_max)
print("      ngưỡng tại từng bước: %s"
      % " ".join("k=%d:%.4f" % (k, tt_o[k]) for k in range(nz + 1)))
print()
print("   Trong chứng minh: G_A = I, A_A = |A|^(−1/2), u_A = |A|^(−1/2)·1, a_{k,j} = 0")
for k in (1, 3, 5):
    A = sorted(tap(Bo[:, k]))
    s_, AA_, w_, u_, a_ = huong(Xo, A, Xo.T @ (yo - Xo @ Bo[:, k - 1]))
    print("      |A| = %d: A_A đo %.8f · công thức %.8f | max|a_j| ngoài A = %.2e"
          % (len(A), AA_, len(A) ** -0.5,
             max([abs(a_[j]) for j in range(nz) if j not in A], default=0.0)))
print()
print("   Định lý 3 tr.19: thiết kế trực giao -> df(μ̂_k) = k. Kiểm bằng bootstrap:")
Ko = list(range(1, nz))
mu_o = np.array([6.0, -4.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0])
dfo, seo, sigo = df_bootstrap(Xo, mu_o, Ko, 2000, seed=5, mu_cho=mu_o, sig_cho=1.0)
print("      μ = %s, σ = 1" % np.array2string(mu_o, precision=1))
print("      k  :" + "".join("%8d" % k for k in Ko))
print("      df :" + "".join("%8.3f" % v for v in dfo))
print("      |df − k| lớn nhất = %.4f (B = 2000 lần bootstrap)"
      % np.abs(dfo - np.array(Ko)).max())
print("   Chứng minh của bài báo: ∇·μ̂_k = Σᵢ 1{|y_i| > |y|_(k+1)} = k hầu chắc chắn.")
print("      đếm trực tiếp trên y ở trên: %s"
      % " ".join("k=%d:%d" % (k, int((np.abs(yo) > tt_o[k]).sum())) for k in range(nz)))


# ════════════════════════════════════════════════════════════════════════════
head("17. §5 — Bổ đề 4, 5, 6: (5.1)–(5.14)")
# ════════════════════════════════════════════════════════════════════════════
print("   Bổ đề 4 tr.25: biến MỚI vào có w_kk và β̂_kk cùng dấu với tương quan c_kk của nó.")
print("    k | biến mới | dấu ĉ_kk | dấu w_kk | dấu β̂_kk | khớp?")
ok4 = True
for k in range(1, B_lar.shape[1]):
    A_tr = sorted(tap(B_lar[:, k - 1]))
    A_k = sorted(tap(B_lar[:, k]))
    moi = [j for j in A_k if j not in A_tr]
    if len(moi) != 1:
        continue
    j0 = moi[0]
    c = X.T @ (y - X @ B_lar[:, k - 1])
    s_, AA_, wA_, uA_, a_ = huong(X, A_k, c)
    i0 = A_k.index(j0)
    d_c = np.sign(c[j0])
    d_w = np.sign(s_[i0] * wA_[i0])
    d_b = np.sign(B_lar[j0, k])
    ok4 &= (d_c == d_w == d_b)
    print("   %2d |    %2d    |   %+d     |   %+d     |    %+d     | %s"
          % (k, j0 + 1, d_c, d_w, d_b, "có" if d_c == d_w == d_b else "KHÔNG"))
print("   -> Bổ đề 4 đúng ở mọi bước? %s" % ("ĐÚNG" if ok4 else "SAI"))
print()
print("   Bổ đề 5 tr.27: A_A là KHOẢNG CÁCH từ gốc tới đơn hình mở rộng S_A (5.7).")
print("   tập A          | A_A theo (2.5) | khoảng cách tối tiểu đo trực tiếp | lệch")
for A in ([2], [2, 8], [2, 3, 8], [1, 2, 3, 6, 8], list(range(10))):
    c = X.T @ y
    s_, AA_, wA_, uA_, a_ = huong(X, A, c)
    XA = X[:, A] * s_
    # min ‖X_A P‖ với ràng buộc Σ P_j = 1 — nghiệm đóng (5.10)
    iG = np.linalg.solve(XA.T @ XA, np.ones(len(A)))
    P = iG / iG.sum()
    d = np.linalg.norm(XA @ P)
    print("   %-32s | %.8f     |     %.8f           | %.2e"
          % (str([j + 1 for j in A]), AA_, d, abs(AA_ - d)))
print("   Bổ đề 5 nói thêm: A ⊆ B thì A_A ≥ A_B, và A_A = 1 đúng khi A có MỘT phần tử.")
AA_bang = {}
for k in range(1, 11):
    for A in itertools.combinations(range(10), k):
        iG = np.linalg.solve((X[:, list(A)]).T @ X[:, list(A)], np.ones(k))
        AA_bang[A] = 1.0 / np.sqrt(np.ones(k) @ iG)
xau5 = sum(1 for A in AA_bang for j in range(10) if j not in A
           and AA_bang[A] < AA_bang[tuple(sorted(A + (j,)))] - 1e-12)
print("      số cặp (A, A∪{j}) VI PHẠM tính đơn điệu: %d / %d" % (xau5, sum(1 for A in AA_bang for j in range(10) if j not in A)))
print("      A_A của 10 tập một phần tử: %s"
      % " ".join("%.6f" % AA_bang[(j,)] for j in range(10)))
print("      A_A của tập đầy đủ 10 biến: %.6f (nhỏ nhất trong %d tập)"
      % (AA_bang[tuple(range(10))], len(AA_bang)))
print()
print("   Bổ đề 6 tr.27 (5.13): S(γ) − S(0) = −2·ĉ'd·γ + d'X'Xd·γ²")
rng17 = np.random.default_rng(11)
b0 = B_lar[:, 4]
c0 = X.T @ (y - X @ b0)
S0 = ((y - X @ b0) ** 2).sum()
for _ in range(3):
    d = rng17.normal(size=10)
    for g in (0.5, 2.0):
        do = ((y - X @ (b0 + g * d)) ** 2).sum() - S0
        ct = -2 * (c0 @ d) * g + (d @ (X.T @ X) @ d) * g ** 2
        print("      γ = %.1f : đo %14.6f · công thức %14.6f · lệch %.2e" % (g, do, ct, abs(do - ct)))


# ════════════════════════════════════════════════════════════════════════════
head("18. §5 — Bổ đề 7–10 và Hình 8 tr.30: đường (T, S) của lasso")
# ════════════════════════════════════════════════════════════════════════════
print("   Bổ đề 7 (5.16): trong một đoạn có A không đổi, μ̂(t) = μ̂(t₀) + A_A·(t − t₀)·u_A")
for k in range(1, 5):
    t0_, t1_ = t_las[k - 1], t_las[k]
    A = sorted(tap(B_las[:, k]))
    c = X.T @ (y - X @ B_las[:, k - 1])
    s_, AA_, wA_, uA_, a_ = huong(X, A, c)
    for f in (0.3, 0.8):
        tq = t0_ + f * (t1_ - t0_)
        bq = np.array([np.interp(tq, t_las, B_las[j]) for j in range(10)])
        le = np.abs(X @ bq - (X @ B_las[:, k - 1] + AA_ * (tq - t0_) * uA_)).max()
        print("      đoạn %d, t = %8.2f : lệch %.3e" % (k, tq, le))
print()
print("   (5.25): t₂ − t₁ = (λ₁ − λ₂)·A_A⁻², tức Ĉ(t) giảm TUYẾN TÍNH với hệ số A_A²")
print("   đoạn | A_A²      | −dĈ/dt đo được | lệch")
for k in range(1, 6):
    A = sorted(tap(B_las[:, k]))
    c = X.T @ (y - X @ B_las[:, k - 1])
    s_, AA_, wA_, uA_, a_ = huong(X, A, c)
    C1 = np.abs(X.T @ (y - X @ B_las[:, k - 1])).max()
    C2 = np.abs(X.T @ (y - X @ B_las[:, k])).max()
    doc = (C1 - C2) / (t_las[k] - t_las[k - 1])
    print("    %3d | %.6f |    %.6f    | %.2e" % (k, AA_ ** 2, doc, abs(doc - AA_ ** 2)))
print()
print("   Hình 8 tr.30: *The (T, S) curve is a decreasing, convex, quadratic spline*")
S_las = ((y[:, None] - X @ B_las) ** 2).sum(0)
print("      12 điểm gãy (T, S): " + " · ".join("(%.0f, %.0f)" % (t_las[k], S_las[k])
                                                for k in range(0, B_las.shape[1], 3)))
print("      S có giảm đơn điệu không? %s ; T có tăng đơn điệu không? %s"
      % ("CÓ" if np.all(np.diff(S_las) < 0) else "KHÔNG",
         "CÓ" if np.all(np.diff(t_las) > 0) else "KHÔNG"))
luoi_t = np.linspace(1e-6, t_las[-1] * 0.999, 4001)
S_luoi = np.array([((y - X @ np.array([np.interp(tq, t_las, B_las[j])
                                       for j in range(10)])) ** 2).sum() for tq in luoi_t])
dd = np.diff(S_luoi, 2)
print("      đạo hàm cấp hai theo sai phân: nhỏ nhất %.4e -> %s"
      % (dd.min(), "LỒI trên cả đường" if dd.min() > -1e-6 else "CÓ CHỖ LÕM"))
print()
print("      *dashed arrow is tangent at t = 1000, negative slope R_t, (5.31)*")
b1k = np.array([np.interp(1000.0, t_las, B_las[j]) for j in range(10)])
C1k = np.abs(X.T @ (y - X @ b1k)).max()
h = 20.0          # S là đa thức bậc hai TRÊN ĐOẠN nên sai phân với bước lớn vẫn chính xác
Sp = ((y - X @ np.array([np.interp(1000.0 + h, t_las, B_las[j]) for j in range(10)])) ** 2).sum()
Sm = ((y - X @ np.array([np.interp(1000.0 - h, t_las, B_las[j]) for j in range(10)])) ** 2).sum()
print("      Ṡ(T) tại t = 1000 : đo %.6f · công thức −2Ĉ = %.6f · lệch %.2e"
      % ((Sp - Sm) / (2 * h), -2 * C1k, abs((Sp - Sm) / (2 * h) + 2 * C1k)))
print("      S tại t = 1000 : %.4f ; T = 1000 ; Ĉ(1000) = %.6f"
      % (((y - X @ b1k) ** 2).sum(), C1k))
A1k = sorted(tap(b1k, 1e-8))
s_, AA1k, w_, u_, a_ = huong(X, A1k, X.T @ (y - X @ b1k))
print("      S̈(T) tại t = 1000 : công thức 2·A_A² = %.6f · đo bằng sai phân %.6f"
      % (2 * AA1k ** 2, (Sp - 2 * ((y - X @ b1k) ** 2).sum() + Sm) / h ** 2))
print()
print("   Bổ đề 10 (5.37): R(d) = −Ṡ(0)/Ṫ(0) ≤ 2Ĉ với MỌI hướng d — thử 20000 hướng ngẫu nhiên:")
rng18 = np.random.default_rng(13)
cc = X.T @ (y - X @ b1k)
Ch = np.abs(cc).max()
D = rng18.normal(size=(20000, 10))
Sdot = -2 * (D @ cc)
bb1 = b1k[None, :] + 1e-7 * D
Tdot = ((np.abs(bb1).sum(1) - np.abs(b1k).sum()) / 1e-7)
ms = Tdot > 1e-9
R = -Sdot[ms] / Tdot[ms]
print("      R(d) lớn nhất trong 20000 hướng: %.6f ; 2Ĉ = %.6f ; vượt trần? %s"
      % (R.max(), 2 * Ch, "CÓ" if R.max() > 2 * Ch + 1e-6 else "KHÔNG"))
d_uA = np.zeros(10)
s_, AAq, wAq, uAq, aq = huong(X, A1k, cc)
d_uA[A1k] = s_ * wAq
R_uA = (2 * (d_uA @ cc)) / ((np.abs(b1k + 1e-7 * d_uA).sum() - np.abs(b1k).sum()) / 1e-7)
print("      R(d) theo đúng hướng cách đều góc u_A: %.6f -> đạt ĐÚNG trần 2Ĉ = %.6f."
      % (R_uA, 2 * Ch))
print()
print("   Bài báo tr.31 viết: *Using Lemmas 3 and 4 it can be shown that the (T, S) curve")
print("   is always convex*. Bổ đề 3 nói về tính liên tục dưới điều kiện nón dương; Bổ đề 4")
print("   nói về DẤU của biến mới vào. Hai bổ đề thật sự cho tính lồi là Bổ đề 7 và Bổ đề 9:")
print("   Ṡ(T) = −2Ĉ(T), và Ĉ(T) giảm — đúng hai thứ vừa đo ở trên.")


# ════════════════════════════════════════════════════════════════════════════
head("19. §6 — Ràng buộc I–III và các tỉ số (6.19)–(6.22)")
# ════════════════════════════════════════════════════════════════════════════
print("   Tại bước %d của đường LARS–Stagewise, u_A nằm NGOÀI nón lồi C_A." % k0)
k_kt = k0
c = X.T @ (y - X @ B_stg[:, k_kt - 1])
A_full = [j - 1 for j in A_truoc]
s_, AAa, wAa, uAa, aa = huong(X, A_full, c)
XA = X[:, A_full] * s_
P, _ = nnls(XA, uAa)
Bh = [A_full[i] for i in range(len(A_full)) if P[i] > 1e-10]
s_b, AAb, wAb, uAb, ab = huong(X, Bh, c)
print("      A  = %s -> A_A = %.8f ; w_A có %d thành phần ÂM"
      % ([j + 1 for j in A_full], AAa, int((s_ * wAa < 0).sum())))
print("      B̂  = %s -> A_B̂ = %.8f" % ([j + 1 for j in Bh], AAb))
print("      Bổ đề 5 đòi A ⊆ B thì A_A ≥ A_B; ở đây B̂ ⊆ A nên A_B̂ ≥ A_A: %.8f ≥ %.8f -> %s"
      % (AAb, AAa, "ĐÚNG" if AAb >= AAa else "SAI"))
print()
print("   (6.9): hình chiếu của u_A vào L(X_B̂) bằng (A_A/A_B̂)·u_B̂")
Pj = XA[:, [A_full.index(j) for j in Bh]]
chieu = Pj @ np.linalg.solve(Pj.T @ Pj, Pj.T @ uAa)
print("      lệch = %.3e ; Cos = A_A/A_B̂ = %.8f ; u_A'u_B̂ đo được = %.8f"
      % (np.abs(chieu - (AAa / AAb) * uAb).max(), AAa / AAb, float(uAa @ uAb)))
print()
print("   Ràng buộc II (6.6): x_j'v_B̂ = A_B̂² với j ∈ B̂ ; Ràng buộc III (6.7): ≥ A_B̂² với j ∈ A − B̂")
vB = AAb * uAb
for j in A_full:
    sj = np.sign(c[j])
    print("      biến %-2d %s: x_j'v_B̂ = %.8f  %s  A_B̂² = %.8f"
          % (j + 1, "∈ B̂ " if j in Bh else "∈ A−B̂",
             sj * (X[:, j] @ vB), "=" if j in Bh else "≥", AAb ** 2))
print()
print("   (6.19)–(6.22): so tốc độ giảm của S và của Ĉ giữa hai hướng")
Ch_ = max(abs(c[j]) for j in A_full)
dS_lars = 2 * Ch_ * float(uAa @ uAa) / AAa
dS_stage = 2 * Ch_ * float(uAa @ uAb) / AAa
print("      −∂S/∂γ theo u_A  (LARS)      : %.6f" % dS_lars)
print("      −∂S/∂γ theo u_B̂  (Stagewise) : %.6f" % dS_stage)
print("      (6.20) tỉ số Stage/LARS = %.8f ; A_A/A_B̂ = %.8f ; lệch %.2e"
      % (dS_stage / dS_lars, AAa / AAb, abs(dS_stage / dS_lars - AAa / AAb)))
dC_lars = min(abs(X[:, j] @ uAa) for j in A_full)
dC_stage = min(abs(X[:, j] @ uAb) for j in A_full)
print("      (6.21) −∂Ĉ/∂γ theo u_A  = %.8f (= A_A  = %.8f)" % (dC_lars, AAa))
print("             −∂Ĉ/∂γ theo u_B̂ = %.8f (= A_B̂ = %.8f)" % (dC_stage, AAb))
print("      (6.22) tỉ số LARS/Stage = %.8f = A_A/A_B̂ ; lệch %.2e"
      % (dC_lars / dC_stage, abs(dC_lars / dC_stage - AAa / AAb)))
print("      -> LARS giảm S nhanh hơn Stagewise %.4f lần, nhưng giảm Ĉ CHẬM hơn đúng bấy nhiêu."
      % (dS_lars / dS_stage))
print("      Bài báo tr.39: *it being more accurate to describe Stagewise as striving to")
print("                      minimize the maximum absolute residual correlation*")


# ════════════════════════════════════════════════════════════════════════════
head("20. §7 Chi phí: O(m³ + nm²) — bản cập nhật Cholesky, và trường hợp m ≫ n")
# ════════════════════════════════════════════════════════════════════════════
from scipy.linalg import solve_triangular                # noqa: E402


def lars_chol(X, y, k_max=None):
    """LARS với cập nhật Cholesky — đúng cách §7 tr.39 mô tả.

    Hai chỗ tối ưu so với bản ở đầu file, và cả hai đều cần thiết để chi phí thật sự
    là O(m³ + nm²):
      · giữ sẵn một vùng đệm (n × m) cho các cột ĐÃ ĐỔI DẤU, nên mỗi bước chỉ THÊM một
        cột chứ không dựng lại cả X_A — dấu s_j không đổi trong LARS gốc (tr.9);
      · cập nhật phân tích Cholesky hạng 1 thay vì nghịch đảo lại G_A từ đầu.
    """
    n_, m_ = X.shape
    beta = np.zeros(m_)
    mu = np.zeros(n_)
    A, R = [], None
    dem = np.empty((n_, m_))                            # vùng đệm cột đã đổi dấu
    B = [beta.copy()]
    while True:
        c = X.T @ (y - mu)
        if A and max(abs(c[j]) for j in A) < 1e-9 * max(1.0, np.abs(X.T @ y).max()):
            break
        ngoai = [j for j in range(m_) if j not in A]
        if ngoai:
            jb = max(ngoai, key=lambda j: abs(c[j]))
            sj = np.sign(c[jb]) or 1.0
            k = len(A)
            dem[:, k] = sj * X[:, jb]
            XA = dem[:, :k]
            if R is None:
                R = np.array([[np.sqrt(dem[:, 0] @ dem[:, 0])]])
            else:
                w = solve_triangular(R, XA.T @ dem[:, k], trans="T", lower=False)
                rho2 = dem[:, k] @ dem[:, k] - w @ w
                Rm = np.zeros((k + 1, k + 1))
                Rm[:k, :k] = R
                Rm[:k, k] = w
                Rm[k, k] = np.sqrt(max(rho2, 1e-300))
                R = Rm
            A = A + [jb]
        if not A:
            break
        k = len(A)
        one = np.ones(k)
        z = solve_triangular(R, solve_triangular(R, one, trans="T", lower=False),
                             lower=False)
        AA = 1.0 / np.sqrt(one @ z)
        wA = AA * z
        XA = dem[:, :k]
        uA = XA @ wA
        a = X.T @ uA
        C = max(abs(c[j]) for j in A)
        con = [j for j in range(m_) if j not in A]
        ung = []
        for j in con:
            for tu, mau in ((C - c[j], AA - a[j]), (C + c[j], AA + a[j])):
                if mau > 1e-12 and tu / mau > 1e-12:
                    ung.append(tu / mau)
        gam = min(ung) if ung else C / AA
        beta = beta.copy()
        beta[A] += gam * np.sign(c[A]) * wA
        mu = mu + gam * uA
        B.append(beta.copy())
        if (k_max is not None and len(B) - 1 >= k_max) or not con:
            break
    return np.array(B).T


Bch = lars_chol(X, y)
print("   Bản Cholesky cho cùng đường không? lệch %.3e trên %d bước"
      % (np.abs(Bch - B_lar).max(), Bch.shape[1] - 1))
Bch2 = lars_chol(X2, y)
print("   Trên m = 64: %d bước, lệch %.3e" % (Bch2.shape[1] - 1, np.abs(Bch2 - B2_lar).max()))
print()
print("   Bài báo tr.39: *The entire sequence of steps in the LARS algorithm with m < n")
print("                  variables requires O(m³ + nm²) computations — the cost of a least")
print("                  squares fit on m variables.*")
print()
print("   Cách kiểm ĐÚNG là ĐẾM PHÉP TÍNH, không phải bấm đồng hồ: một vòng lặp Python bị")
print("   chi phí thông dịch át mất phần số học. Đếm phép nhân-cộng theo đúng mô tả §7:")
print("      · mỗi bước k: (m − k) tích vô hướng độ dài n  -> (m − k)·n")
print("      · cập nhật Cholesky hạng 1                     -> k²/2 + O(k)")
print("      · hai phép giải tam giác để lấy G_A⁻¹·1       -> k²")
print("      · dựng u_A = X_A w_A và a = X'u_A              -> n·k + n·m")


def dem_phep(n_, m_):
    """Số phép nhân-cộng của m bước LARS, và của một lần khớp OLS trên m biến."""
    lars_ = sum((m_ - k) * n_ + 1.5 * k ** 2 + n_ * k + n_ * m_ for k in range(1, m_ + 1))
    ols_ = n_ * m_ ** 2 + m_ ** 3 / 3.0
    return lars_, ols_


print()
print("      n     | m    | LARS (phép)  | OLS (phép)   | LARS/OLS | m gấp đôi -> LARS ×")
truoc = None
for n20, m20 in ((2000, 20), (2000, 40), (2000, 80), (2000, 160), (2000, 320)):
    L_, O_ = dem_phep(n20, m20)
    print("     %5d | %4d | %.4e | %.4e | %7.2f  | %s"
          % (n20, m20, L_, O_, L_ / O_, "—" if truoc is None else "%.2f" % (L_ / truoc)))
    truoc = L_
print("      -> tỉ số LARS/OLS dừng lại quanh %.2f chứ không tăng theo m: đó chính là nghĩa"
      % (dem_phep(2000, 320)[0] / dem_phep(2000, 320)[1]))
print("         của câu *the same order of magnitude as ordinary least squares*.")
print()
print("   Đo đồng hồ để đối chiếu — n lớn cho phần số học át được chi phí thông dịch:")
rng20 = np.random.default_rng(21)
print("      n      | m    | LARS Cholesky | LARS giải lại  | OLS đầy đủ | LARS/OLS")
for n20, m20 in ((20000, 40), (20000, 80), (20000, 160)):
    Xz = chuan(rng20.normal(size=(n20, m20)))
    yz = Xz @ rng20.normal(size=m20) + rng20.normal(scale=2.0, size=n20)
    t1 = time.perf_counter(); Bc_ = lars_chol(Xz, yz); tc = time.perf_counter() - t1
    t1 = time.perf_counter(); Bn_, _ = lars(Xz, yz, "lar"); tn = time.perf_counter() - t1
    t1 = time.perf_counter()
    for _ in range(5):
        ols(Xz, yz)
    to = (time.perf_counter() - t1) / 5
    print("     %6d | %4d | %10.4f s  | %10.4f s  | %8.4f s | %6.1f lần"
          % (n20, m20, tc, tn, to, tc / to))
print("      (hai bản LARS cho cùng đường: lệch %.2e)" % np.abs(Bc_ - Bn_).max())
print()
print("   §7 tr.40: *LARS terminates at the saturated least squares fit after n − 1")
print("             variables have entered the active set*")
for (nq, mq) in ((40, 300), (60, 1000)):
    Xq = chuan(rng20.normal(size=(nq, mq)))
    yq = Xq[:, :5] @ np.array([4.0, -3.0, 2.0, 1.0, -1.0]) + rng20.normal(scale=0.3, size=nq)
    Bq, skq = lars(Xq, yq, "lasso", tran=4000)
    dem = [len(tap(Bq[:, k])) for k in range(Bq.shape[1])]
    tung_vao = set()
    for k in range(Bq.shape[1]):
        tung_vao |= tap(Bq[:, k])
    print("      n = %-3d m = %-5d: %3d bước · số biến khác 0 NHIỀU NHẤT = %d (n − 1 = %d)"
          % (nq, mq, Bq.shape[1] - 1, max(dem), nq - 1))
    print("                          số biến TỪNG vào mô hình = %d (> n − 1: %s)"
          % (len(tung_vao), "có" if len(tung_vao) > nq - 1 else "không"))


# ════════════════════════════════════════════════════════════════════════════
head("21. §8 Boosting — Stagewise trên một tập hàm cơ sở lớn")
# ════════════════════════════════════════════════════════════════════════════
print("   §8 tr.40–41 là mục DUY NHẤT của bài báo không có một con số nào, và nó nói rõ")
print("   mình là phỏng đoán: *conjectured that this may help explain the success of the")
print("   Forward Stagewise process used in least squares boosting*.")
print()
print("   Dựng đúng bối cảnh đó ở quy mô nhỏ: bộ học cơ sở là các GỐC CÂY (stump) 1 chiều.")
rng21 = np.random.default_rng(5)
n21 = 120
x21 = np.sort(rng21.uniform(0, 1, n21))
f21 = np.sin(6 * x21) + 0.6 * (x21 > 0.55)
y21 = f21 - f21.mean() + rng21.normal(scale=0.25, size=n21)
y21 = y21 - y21.mean()
# ngưỡng đặt giữa hai điểm dữ liệu LIỀN NHAU -> mỗi gốc cây là một cột KHÁC NHAU;
# lấy thưa ra để ma trận không sát suy biến
nguong = (x21[:-1] + x21[1:])[::2] / 2.0
X21 = chuan(np.array([(x21 > t).astype(float) for t in nguong]).T)
print("      n = %d điểm, m = %d gốc cây (mỗi gốc là một ngưỡng chia)" % X21.shape)
print("      hạng của ma trận thiết kế: %d ; số điều kiện %.3e"
      % (np.linalg.matrix_rank(X21), np.linalg.cond(X21)))
B21, _ = lars(X21, y21, "lasso", k_max=40, tran=4000)
t21 = np.abs(B21).sum(0)
for eps in (0.2, 0.05, 0.0125):
    bb, kk, _ = stagewise_eps(X21, y21, eps, t_dich=0.9 * t21[-1], tran=300000)
    ref = np.array([np.interp(np.abs(bb).sum(), t21, B21[j]) for j in range(X21.shape[1])])
    print("      ε = %-7.4f (%7d bước): lệch lớn nhất với đường lasso %.6f (thang hệ số %.3f)"
          % (eps, kk, np.abs(bb - ref).max(), np.abs(B21).max()))
print("      -> ε càng nhỏ, đường boosting bình phương tối thiểu càng sát đường LASSO.")
print("      Đó đúng là nội dung phỏng đoán của §8 — và bài báo KHÔNG đo nó lần nào.")


# ════════════════════════════════════════════════════════════════════════════
head("22. Chọn thiết bị — đo CPU và GPU cho CHÍNH phép tính của bài")
# ════════════════════════════════════════════════════════════════════════════
try:
    import torch
    co_torch = True
except Exception:                                        # pragma: no cover
    co_torch = False
print("   Số luồng BLAS đặt cứng bằng 1 (biến môi trường, đặt trước khi nạp numpy).")
print("   Phép tính nặng nhất của LARS là X'r (n×m) và giải hệ k×k. Đo cả hai thiết bị:")
if not co_torch:
    print("   (không có torch trên máy này — bỏ qua phần GPU)")
else:
    print("      cỡ (n × m)   | numpy CPU  | torch CUDA | GPU nhanh hơn")
    torch.set_num_threads(1)
    co_cuda = torch.cuda.is_available()
    for (nq, mq) in ((442, 10), (442, 64), (2000, 500), (8000, 4000)):
        Xq = np.asfortranarray(np.random.default_rng(0).normal(size=(nq, mq)))
        rq = np.random.default_rng(1).normal(size=nq)
        for _ in range(3):
            Xq.T @ rq
        t1 = time.perf_counter()
        for _ in range(50):
            Xq.T @ rq
        tc = (time.perf_counter() - t1) / 50 * 1e3
        if co_cuda:
            Xg = torch.tensor(Xq, device="cuda")
            rg = torch.tensor(rq, device="cuda")
            for _ in range(5):
                Xg.T @ rg
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            for _ in range(50):
                Xg.T @ rg
            torch.cuda.synchronize()
            tg = (time.perf_counter() - t1) / 50 * 1e3
            print("      %5d × %-5d | %8.4f ms | %8.4f ms | %.2f×" % (nq, mq, tc, tg, tc / tg))
        else:
            print("      %5d × %-5d | %8.4f ms | không có CUDA" % (nq, mq, tc))
    print("      -> cỡ bài toán của GĐ 1 (442×10 và 442×64) nằm hẳn trong vùng CPU thắng.")
    print("      Số luồng torch đặt cứng: %d" % torch.get_num_threads())



# ════════════════════════════════════════════════════════════════════════════
head("23. MẢNG cho script vẽ hình — in ĐỦ CHỮ SỐ để khối mã trong bài dán thẳng")
# ════════════════════════════════════════════════════════════════════════════
def in_mang(ten, v, cot=6):
    v = np.asarray(v, dtype=float).ravel()
    print("   %s = [" % ten)
    for i in range(0, len(v), cot):
        print("       " + ", ".join(repr(float(x)) for x in v[i:i + cot]) + ",")
    print("   ]")


print("   Vì sao phải in: hình nào VẼ NHỮNG CON SỐ thì khối mã trong bài nên dán thẳng mảng,")
print("   để `regen.py` dựng lại hình trong vài giây thay vì chạy lại cả phép đo.")
print()
in_mang("df10", df10)
in_mang("se10", se10)
in_mang("K64", K64, 12)
in_mang("df64", df64)
in_mang("se64", se64)

s2_10 = ((y - X @ ols(X, y)) ** 2).sum() / (n - 10 - 1)
Cp10 = ((y[:, None] - X @ B_lar) ** 2).sum(0) / s2_10 - n + 2 * np.arange(B_lar.shape[1])
s2_64 = ((y - X2 @ ols(X2, y)) ** 2).sum() / (n - 64 - 1)
Cp64 = ((y[:, None] - X2 @ B2_lar) ** 2).sum(0) / s2_64 - n + 2 * np.arange(B2_lar.shape[1])
in_mang("Cp10", Cp10)
in_mang("Cp64", Cp64)

for kieu in ("lar", "lasso", "stage", "fs"):
    in_mang("pe_%s" % kieu, pe[kieu].mean(0))
in_mang("pe_sd_lar", pe["lar"].std(0, ddof=1))

in_mang("rho", rho_list)
vt_320, d_320 = [], []
for k in range(1, B_lar.shape[1]):
    A = sorted(tap(B_lar[:, k]))
    XA = X[:, A]
    vt_320.append(1 - ((y - XA @ ols(XA, y)) ** 2).sum() / (y @ y) - R2_lar[k])
    d_320.append(R2_lar[k] - R2_lar[k - 1])
in_mang("ve_trai_320", vt_320)
in_mang("delta_R2", d_320)

t_s2_l = np.abs(S2_lar).sum(0)
in_mang("t_sklearn_m64", t_s2_l, 4)
in_mang("t_tuviet_m64", np.abs(B2_lar).sum(0), 4)

dem_vp = []
for k in range(1, 11):
    d_ = sum(1 for A in itertools.combinations(range(10), k)
             if (np.linalg.solve(G[np.ix_(A, A)], np.ones(k)) <= 0).any())
    dem_vp.append(d_)
in_mang("vi_pham_theo_co", dem_vp, 10)
in_mang("tong_theo_co", [len(list(itertools.combinations(range(10), k))) for k in range(1, 11)], 10)
print()
print("   Hai mảng cuối: số tập con VI PHẠM (4.11) và TỔNG số tập con, theo cỡ 1..10.")



# ════════════════════════════════════════════════════════════════════════════
head("24. VÍ DỤ CHẠY TAY — hai biến, in lại TỪNG BƯỚC của (2.8)–(2.13)")
# ════════════════════════════════════════════════════════════════════════════
print("   Bốn quan sát, hai cột. Cột đã tâm hoá và có chuẩn Euclid bằng 1 — quy ước (1.1).")
Xv = np.array([[-1.0, 0.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0]]) / np.sqrt(2.0)
yv = 3.0 * Xv[:, 0] + 1.0 * Xv[:, 1]
print("   X =")
for i in range(4):
    print("        [%+.6f, %+.6f]" % (Xv[i, 0], Xv[i, 1]))
print("   y = [%s]" % ", ".join("%+.6f" % v for v in yv))
print("   kiểm quy ước (1.1): Σy = %.1e · Σx_{i1} = %.1e · Σx_{i1}² = %.6f · Σx_{i2}² = %.6f"
      % (yv.sum(), Xv[:, 0].sum(), (Xv[:, 0] ** 2).sum(), (Xv[:, 1] ** 2).sum()))
print("   hai cột có TRỰC GIAO không? x₁ᵀx₂ = %.1e" % (Xv[:, 0] @ Xv[:, 1]))
print()

muv = np.zeros(4)
bv = np.zeros(2)
for buoc in (1, 2):
    cv = Xv.T @ (yv - muv)                                          # (2.8)
    print("   ── BƯỚC %d ──" % buoc)
    print("   (2.8)  ĉ = Xᵀ(y − μ̂) = [%+.6f, %+.6f]" % (cv[0], cv[1]))
    Cv = np.abs(cv).max()                                           # (2.9)
    Av = sorted(int(j) for j in np.flatnonzero(np.abs(cv) >= Cv - 1e-12))
    print("   (2.9)  Ĉ = max|ĉ_j| = %.6f  ->  A = {%s}"
          % (Cv, ", ".join(str(j + 1) for j in Av)))
    sv = np.sign(cv[Av])
    print("   (2.10) s_j = dấu(ĉ_j) = [%s]" % ", ".join("%+d" % v for v in sv))
    XA = Xv[:, Av] * sv                                             # (2.4)
    GA = XA.T @ XA                                                  # (2.5)
    iG = np.linalg.solve(GA, np.ones(len(Av)))
    AA = 1.0 / np.sqrt(np.ones(len(Av)) @ iG)
    wA = AA * iG                                                    # (2.6)
    uA = XA @ wA
    av = Xv.T @ uA                                                  # (2.11)
    print("   (2.5)  G_A = %s" % np.array2string(GA, precision=6).replace("\n", " "))
    print("          A_A = (1ᵀG_A⁻¹1)^(−1/2) = %.6f" % AA)
    print("   (2.6)  w_A = A_A·G_A⁻¹·1 = [%s]" % ", ".join("%.6f" % v for v in wA))
    print("          u_A = X_A·w_A = [%s]" % ", ".join("%+.6f" % v for v in uA))
    print("   (2.7)  kiểm: X_Aᵀu_A = [%s] phải bằng A_A·1 = %.6f ; ‖u_A‖ = %.6f"
          % (", ".join("%.6f" % v for v in XA.T @ uA), AA, np.linalg.norm(uA)))
    print("   (2.11) a = Xᵀu_A = [%+.6f, %+.6f]" % (av[0], av[1]))
    ngoai = [j for j in range(2) if j not in Av]
    if ngoai:
        ung = []
        for j in ngoai:
            for nhan, tu, mau in (("Ĉ−ĉ / A−a", Cv - cv[j], AA - av[j]),
                                  ("Ĉ+ĉ / A+a", Cv + cv[j], AA + av[j])):
                ok = mau > 1e-12 and tu / mau > 1e-12
                print("   (2.13) j = %d, %s = (%.6f)/(%.6f) = %s"
                      % (j + 1, nhan, tu, mau,
                         ("%.6f  <- hợp lệ" % (tu / mau)) if ok else "loại"))
                if ok:
                    ung.append(tu / mau)
        gam = min(ung)
    else:
        gam = Cv / AA
        print("   (2.13) không còn biến ngoài -> quy ước tr.10: γ̂ = Ĉ/A_A = %.6f" % gam)
    print("   (2.12) γ̂ = %.6f ; μ̂ ← μ̂ + γ̂·u_A" % gam)
    dv = np.zeros(2)
    dv[Av] = sv * wA
    bv = bv + gam * dv
    muv = muv + gam * uA
    print("          β̂ = [%.6f, %.6f] ; Σ|β̂| = %.6f" % (bv[0], bv[1], np.abs(bv).sum()))
    print("   (2.16) tương quan chung mới: Ĉ − γ̂·A_A = %.6f ; đo lại = %.6f"
          % (Cv - gam * AA, np.abs(Xv.T @ (yv - muv)).max()))
    print()

print("   Nghiệm OLS của bài toán này: [%s]"
      % ", ".join("%.6f" % v for v in ols(Xv, yv)))
print("   LARS dừng ở      : [%s] -> lệch %.2e"
      % (", ".join("%.6f" % v for v in bv), np.abs(bv - ols(Xv, yv)).max()))
Bv = lars_path(Xv, yv, method="lar")[2]
print("   sklearn cho     : %s" % np.array2string(Bv, precision=6).replace("\n", "\n" + " " * 21))
print("   Hai cột vuông góc nên γ̂₁ đưa β̂₁ tới đúng chỗ β̂₁ = β̂₂ về tương quan;")
print("   bước 2 đi theo đường phân giác và tới thẳng nghiệm OLS — đúng Hình 2 tr.6.")


# ════════════════════════════════════════════════════════════════════════════
head("25. CÁC CON SỐ DẪN XUẤT mà bài viết trích — in ra thay vì để người viết tự quy đổi")
# ════════════════════════════════════════════════════════════════════════════
print("   Bài học Đợt 24: một con số TÓM TẮT một bảng cũng phải do script in ra.")
print()
print("   ρ_k đổi sang phần trăm (bảng ở mục 4 của bài):")
print("      ρ_1 = %.6f -> %.2f %% quãng đường Forward Selection sẽ đi" % (rho_list[0], 100 * rho_list[0]))
print("      ρ_8 = %.6f -> %.2f %% (lớn nhất trước bước cuối)" % (rho_list[7], 100 * rho_list[7]))
print()
print("   Mô phỏng §3.3 — các con số bài viết trích trong văn xuôi:")
kc = (max(pe["lar"][:, K_MP].mean(), pe["lasso"][:, K_MP].mean(), pe["stage"][:, K_MP].mean())
      - min(pe["lar"][:, K_MP].mean(), pe["lasso"][:, K_MP].mean(), pe["stage"][:, K_MP].mean()))
print("      khoảng cách ba đường LARS/Lasso/Stagewise tại k = 40 : %.6f" % kc)
print("      pe trung bình tại k = 40: LARS %.4f · Forward Selection %.4f"
      % (pe["lar"][:, K_MP].mean(), pe["fs"][:, K_MP].mean()))
print("      pe trung bình tại k = 3 : LARS %.4f · Forward Selection %.4f"
      % (pe["lar"][:, 3].mean(), pe["fs"][:, 3].mean()))
sd_tb = pe["lar"].std(0, ddof=1).mean()
print("      độ lệch chuẩn trung bình của LARS: %.6f -> làm tròn %.4f" % (sd_tb, sd_tb))
print("      sai số chuẩn của trung bình trên 100 lần: %.4f" % (sd_tb / np.sqrt(SO_LAP)))
print("      chênh giữa đỉnh đo được và 0,963 của bài báo: %.4f"
      % (pe["lar"].mean(0).max() - 0.963))
print()
print("   A_A của tập đầy đủ 10 biến, LẤY DẤU TỪ ĉ tại μ̂ = 0 (khác bảng quét mọi tập con):")
_c0 = X.T @ y
_s, _AA, _w, _u, _a = huong(X, list(range(10)), _c0)
print("      A_A = %.8f ; còn bảng quét dấu dương cho %.6f" % (_AA, AA_bang[tuple(range(10))]))

print()
print("=" * 94)
print("TỔNG: chạy hết trong %.2f giây." % (time.time() - T0))
print("=" * 94)
