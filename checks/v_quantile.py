# -*- coding: utf-8 -*-
"""Kiem chung bai `Quantile Regression` (Order 430) theo Luat 6.

Nguon ruot: Koenker & Bassett, "Regression Quantiles", Econometrica 46(1):33-50, 1978.
    AI algorithms/_papers/1978 - Koenker, Bassett - Regression Quantiles.pdf
    (ban JSTOR, KHONG truy cap mo — nam trong .gitignore, chi co tren may dung bai.)

Bai bao co 0 HINH, 2 BANG (Bang I tr.37, Bang II tr.42), 8 dinh ly, 22 cong thuc
danh so ((1.1), (3.1)-(3.9), (4.1)-(4.12)) cong ba nhan (P), (D), (D').

MUC TIEU: moi con so bai viet trich deu do script nay in ra.

Ky hieu cua BAI BAO giu nguyen trong script: theta (khong phai tau), T (khong phai n),
K (khong phai p), b (bien toi uu), beta*(theta) (nghiem).

Chay:  cd tools && python v_quantile.py
"""
import os
import sys

for _b in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_b, "1")

sys.stdout.reconfigure(encoding="utf-8")

import itertools
import re
import time
from fractions import Fraction as F

import numpy as np
import scipy
from scipy.optimize import linprog
from scipy.stats import norm

T0 = time.time()


def tieu_de(s):
    print()
    print("=" * 92)
    print(s)
    print("=" * 92)


# ---------------------------------------------------------------- 0. moi truong
tieu_de("0. MOI TRUONG")
print("numpy %s · scipy %s · python %s" % (np.__version__, scipy.__version__,
                                           sys.version.split()[0]))
try:
    import sklearn
    from sklearn.linear_model import QuantileRegressor, LinearRegression
    CO_SKLEARN = True
    print("scikit-learn %s" % sklearn.__version__)
except ImportError:                                            # pragma: no cover
    CO_SKLEARN = False
    print("KHONG co scikit-learn — bo qua phan doi chieu thu vien")
print("so luong torch/BLAS ghim ve 1 truoc khi import numpy")


# ============================================================ ham nen dung chung
def pv_mau_don(s_sx, th):
    """phan vi mau thu theta cua tung dong ma tran DA SAP XEP; T*theta nguyen thi
       lay trung binh hai thong ke thu tu ke nhau."""
    tt = s_sx.shape[1]
    k = tt * th
    if abs(k - round(k)) < 1e-9:
        lo = int(round(k)) - 1
        return 0.5 * (s_sx[:, lo] + s_sx[:, lo + 1])
    return s_sx[:, int(np.ceil(k)) - 1]


def rho(u, th):
    """Ham mat mat cua bai bao, viet theo dung dong tr.38:
       sum_{y>=xb} theta|u| + sum_{y<xb} (1-theta)|u|  =  u(theta - 1[u<0])."""
    u = np.asarray(u, dtype=float)
    return u * (th - (u < 0))


def psi(b, th, y, X):
    """psi(b; theta, y, X) cua bai bao tr.39."""
    return float(np.sum(rho(y - X @ b, th)))


def giai_lp(y, X, th):
    """Bai (P) cua PHU LUC tr.48:  min [theta*i'r+ + (1-theta)*i'r-]
       s.t.  y = Xb + r+ - r-,  (b, r+, r-) in R^K x R^{2T}_+.
       b tu do ve dau nen phai tach doi b = b+ - b- de moi bien deu >= 0."""
    Tn, K = X.shape
    c = np.r_[np.zeros(2 * K), th * np.ones(Tn), (1.0 - th) * np.ones(Tn)]
    A = np.hstack([X, -X, np.eye(Tn), -np.eye(Tn)])
    r = linprog(c, A_eq=A, b_eq=y, bounds=[(0, None)] * (2 * K + 2 * Tn),
                method="highs")
    assert r.status == 0, r.message
    b = r.x[:K] - r.x[K:2 * K]
    return b, r.x[2 * K:2 * K + Tn], r.x[2 * K + Tn:], r.fun


def giai_dual(y, X, th):
    """Bai (D') tr.48-49:  max [y'Delta]  s.t.  X'Delta = (1-theta)X'i,  Delta in [0,1]^T."""
    Tn, K = X.shape
    r = linprog(-y, A_eq=X.T, b_eq=(1.0 - th) * (X.T @ np.ones(Tn)),
                bounds=[(0, 1)] * Tn, method="highs")
    assert r.status == 0, r.message
    return r.x, -r.fun


# ------------------------------------------- 1. dinh nghia tr.38, mo hinh vi tri
tieu_de("1. DINH NGHIA tr.38 — bai toan cuc tieu, va no THU GON ve phan vi mau")
print("Bai bao dinh nghia phan vi mau thu theta la NGHIEM BAT KY cua")
print("    min_b [ sum_{t: y_t >= b} theta|y_t - b| + sum_{t: y_t < b} (1-theta)|y_t - b| ]")
print("va phan vi hoi quy thu theta bang cach thay b boi x_t b. Trong mo hinh vi tri")
print("(K = 1, x_t = 1 voi moi t) hai bai toan TRUNG NHAU — tr.38.")
print()
rng = np.random.default_rng(430)
y1 = rng.normal(size=19)
X1 = np.ones((19, 1))
print("%-9s %-18s %-18s %-13s" % ("theta", "nghiem bai bao", "np.quantile", "lech"))
LECH_VITRI = []
for th in (0.10, 0.25, 0.50, 0.75, 0.90):
    b, _, _, _ = giai_lp(y1, X1, th)
    q = float(np.quantile(y1, th, method="inverted_cdf"))
    LECH_VITRI.append(abs(b[0] - q))
    print("%-9.2f %-18.10f %-18.10f %-13.2e" % (th, b[0], q, abs(b[0] - q)))
print("lech toi da tren 5 muc theta = %.3e   (T = 19 nen T*theta khong nguyen o ca 5 muc)"
      % max(LECH_VITRI))
assert max(LECH_VITRI) < 1e-9


# ---------------------------------------------------- 2. Dinh ly 3.1 tr.39
tieu_de("2. DINH LY 3.1 tr.39 — beta*(theta) = X(h)^{-1} y(h) voi mot h nao do")
print('"If X has rank K then the set of regression quantiles, B*(theta), has at least')
print(' one element of the form beta*(theta) = X(h)^{-1} y(h) for some h in H."')
print()
print("Kiem: giai LP, roi di tim tap con h gom K quan sat ma duong di QUA ca K diem.")
print()
print("%-6s %-6s %-7s %-9s %-11s %-14s %-9s" %
      ("T", "K", "theta", "|h| tim", "= K ?", "lech beta", "psi"))
DL31_OK, DL31_ZK = 0, 0
rng = np.random.default_rng(1978)
for Tn, K in ((30, 2), (40, 3), (55, 4), (80, 2), (120, 3)):
    for th in (0.15, 0.5, 0.85):
        X = np.c_[np.ones(Tn), rng.normal(size=(Tn, K - 1))]
        y = X @ np.arange(1.0, K + 1) + rng.standard_t(4, size=Tn)
        b, _, _, _ = giai_lp(y, X, th)
        e = y - X @ b
        h = np.flatnonzero(np.abs(e) < 1e-9)
        lech = np.nan
        if len(h) >= K:
            hh = h[:K]
            if abs(np.linalg.det(X[hh])) > 1e-12:
                lech = float(np.max(np.abs(np.linalg.solve(X[hh], y[hh]) - b)))
        DL31_OK += int(lech == lech and lech < 1e-8)
        DL31_ZK += int(len(h) == K)
        print("%-6d %-6d %-7.2f %-9d %-11s %-14.3e %-9.4f" %
              (Tn, K, th, len(h), "co" if len(h) == K else "KHONG", lech, psi(b, th, y, X)))
print()
print("%d/15 truong hop co X(h)^{-1}y(h) trung nghiem LP toi 1e-8" % DL31_OK)
print("%d/15 truong hop co DUNG K phan du bang 0" % DL31_ZK)
assert DL31_OK == 15


# ---------------------------------------------------- 3. Dinh ly 3.2 tr.39 (i)-(iv)
tieu_de("3. DINH LY 3.2 tr.39 — bon tinh chat dong bien doi (i) (ii) (iii) (iv)")
rng = np.random.default_rng(7)
Tn, K = 60, 3
X = np.c_[np.ones(Tn), rng.normal(size=(Tn, K - 1))]
y = X @ np.array([2.0, -1.0, 0.5]) + rng.laplace(size=Tn)
th = 0.35
b0, _, _, _ = giai_lp(y, X, th)
lam_duong, lam_am = 2.75, -1.6
gam = np.array([0.7, -2.1, 3.3])
A = np.array([[1.0, 0.4, -0.2], [0.0, 1.3, 0.6], [0.0, -0.5, 1.1]])

t_i = giai_lp(lam_duong * y, X, th)[0]
t_ii = giai_lp(lam_am * y, X, 1.0 - th)[0]
t_iii = giai_lp(y + X @ gam, X, th)[0]
t_iv = giai_lp(y, X @ A, th)[0]
E = [float(np.max(np.abs(t_i - lam_duong * b0))),
     float(np.max(np.abs(t_ii - lam_am * b0))),
     float(np.max(np.abs(t_iii - (b0 + gam)))),
     float(np.max(np.abs(t_iv - np.linalg.solve(A, b0))))]
NHAN = ["(i)   beta*(theta, lam*y, X) = lam*beta*(theta, y, X),  lam = %.2f >= 0" % lam_duong,
        "(ii)  beta*(1-theta, lam*y, X) = lam*beta*(theta, y, X), lam = %.2f <= 0" % lam_am,
        "(iii) beta*(theta, y + X*gam, X) = beta*(theta, y, X) + gam",
        "(iv)  beta*(theta, y, X*A) = A^{-1} beta*(theta, y, X)"]
for nh, e in zip(NHAN, E):
    print("%-72s lech = %.3e" % (nh, e))
print()
print("lech toi da tren ca bon tinh chat = %.3e" % max(E))
assert max(E) < 1e-8
print()
print("He qua tr.40: (i) va (ii) cho beta*(1/2) dong bien doi theo THANG DO;")
print("(iii) cho dong bien doi theo VI TRI; (iv) cho dong bien doi khi doi tham so thiet ke.")


# ---------------------------------------------------- 4. Dinh ly 3.4 tr.41, cong thuc (3.7)
tieu_de("4. DINH LY 3.4 tr.41 — cong thuc (3.7):  N(u*) <= T*theta <= T - P(u*) = N + Z")
print("Day la phat bieu CHAT CHE ve 'ti le nam duoi', thay cho moi cong thuc dem tu nghi.")
print("Dieu kien: X co mot cot toan 1.")
print()
print("%-5s %-4s %-7s %-6s %-6s %-6s %-9s %-24s" %
      ("T", "K", "theta", "N", "Z", "P", "T*theta", "(3.7) N <= T*th <= N+Z"))
rng = np.random.default_rng(3141)
V37, N37 = 0, 0
HANG37 = []
for Tn, K in ((25, 2), (40, 2), (60, 3), (100, 4)):
    for th in (0.10, 0.37, 0.50, 0.75, 0.90):
        X = np.c_[np.ones(Tn), rng.normal(size=(Tn, K - 1))]
        y = X @ np.ones(K) + rng.normal(size=Tn) * (0.5 + 0.5 * np.abs(X[:, 1]))
        b, _, _, _ = giai_lp(y, X, th)
        u = y - X @ b
        Nn = int(np.sum(u < -1e-9))
        Zn = int(np.sum(np.abs(u) <= 1e-9))
        Pn = int(np.sum(u > 1e-9))
        ok = (Nn <= Tn * th <= Nn + Zn)
        N37 += 1
        V37 += int(ok)
        HANG37.append((Tn, K, th, Nn, Zn, Pn))
        print("%-5d %-4d %-7.2f %-6d %-6d %-6d %-9.2f %-24s" %
              (Tn, K, th, Nn, Zn, Pn, Tn * th, "DUNG" if ok else "SAI"))
print()
print("(3.7) dung o %d/%d truong hop." % (V37, N37))
assert V37 == N37
print("Z(u*) = K o %d/%d truong hop — dung Remark tr.41 ('Z(u*) = K with probability one')."
      % (sum(1 for h in HANG37 if h[1] == h[4]), N37))

print()
print("REMARK tr.41 co HAI cach doc, thu ca hai roi de SO CHO SAI quyet dinh:")
print("  (A) nhu ban in: 'at least T*theta observations BELOW ... at most T*theta + K ABOVE'")
print("  (B) suy thang tu (3.7): so quan sat duoi-hoac-tren-duong N+Z >= T*theta,")
print("      va so quan sat PHIA TREN P <= T(1-theta)")
saiA1 = sum(1 for (Tn, K, th, Nn, Zn, Pn) in HANG37 if not (Nn >= Tn * th))
saiA2 = sum(1 for (Tn, K, th, Nn, Zn, Pn) in HANG37 if not (Pn <= Tn * th + K))
saiB1 = sum(1 for (Tn, K, th, Nn, Zn, Pn) in HANG37 if not (Nn + Zn >= Tn * th))
saiB2 = sum(1 for (Tn, K, th, Nn, Zn, Pn) in HANG37 if not (Pn <= Tn * (1 - th)))
print("  (A) ve 'below':  %d/%d truong hop SAI" % (saiA1, N37))
print("  (A) ve 'above':  %d/%d truong hop SAI" % (saiA2, N37))
print("  (B) ve 'below':  %d/%d truong hop SAI" % (saiB1, N37))
print("  (B) ve 'above':  %d/%d truong hop SAI" % (saiB2, N37))
print("  -> cach doc %s la cach khop voi (3.7)."
      % ("B" if (saiB1 + saiB2) < (saiA1 + saiA2) else "A"))


# ------------------------------------- 4b. Dinh ly 3.3 tr.40 va cong thuc (3.1), (3.6)
tieu_de("4b. DINH LY 3.3 tr.40 — dieu kien (3.1) cho nghiem DUY NHAT, va (3.6)")
print("(3.1)  (theta-1) iota'_K  <  sum_{t in h_ngang} [1/2 - 1/2 sgn(y_t - x_t beta*) - theta]")
print("                             x_t X(h)^{-1}   <  theta iota'_K")
print()
print("Ve giua la mot vecto K chieu; bat dang thuc doc theo TUNG toa do. Kiem: tinh ve giua")
print("tai nghiem LP, roi xem no co nam han trong khoang (theta-1, theta) khong.")
print()


def ve_giua(y, X, th, b, h):
    Tn, K = X.shape
    Xi = np.linalg.inv(X[h])
    u = y - X @ b
    ngoai = [t for t in range(Tn) if t not in set(h)]
    v = np.zeros(K)
    for t in ngoai:
        v += (0.5 - 0.5 * np.sign(u[t]) - th) * (X[t] @ Xi)
    return v


rng = np.random.default_rng(31)
print("%-5s %-5s %-7s %-40s %-14s %-12s" %
      ("T", "K", "theta", "ve giua, tung toa do", "trong khoang?", "bien gan nhat"))
V31_OK, V31_N = 0, 0
for Tn, K in ((30, 2), (45, 3), (70, 2), (90, 3)):
    for th in (0.2, 0.5, 0.8):
        X = np.c_[np.ones(Tn), rng.normal(size=(Tn, K - 1))]
        y = X @ np.ones(K) + rng.standard_t(5, size=Tn)
        b, _, _, _ = giai_lp(y, X, th)
        h = list(np.flatnonzero(np.abs(y - X @ b) < 1e-9))[:K]
        v = ve_giua(y, X, th, b, h)
        ok = bool(np.all(v > th - 1 + 1e-12) and np.all(v < th - 1e-12))
        V31_N += 1
        V31_OK += int(ok)
        bien = float(min(np.min(v - (th - 1)), np.min(th - v)))
        print("%-5d %-5d %-7.2f %-40s %-14s %-12.6f"
              % (Tn, K, th, " ".join("%8.5f" % z for z in v), "DUNG" if ok else "SAI", bien))
print()
print("(3.1) dung o %d/%d truong hop." % (V31_OK, V31_N))
assert V31_OK == V31_N
print()
print("PHEP THU NGUOC — lay mot tap h KHONG toi uu roi tinh lai ve giua:")
X = np.c_[np.ones(40), rng.normal(size=(40, 1))]
y = X @ np.array([1.0, 2.0]) + rng.standard_t(5, size=40)
th = 0.3
b, _, _, _ = giai_lp(y, X, th)
h_that = list(np.flatnonzero(np.abs(y - X @ b) < 1e-9))[:2]
vi_pham = 0
thu = 0
for h in itertools.combinations(range(40), 2):
    if set(h) == set(h_that) or abs(np.linalg.det(X[list(h)])) < 1e-9:
        continue
    thu += 1
    bh = np.linalg.solve(X[list(h)], y[list(h)])
    v = ve_giua(y, X, th, bh, list(h))
    if not (np.all(v > th - 1) and np.all(v < th)):
        vi_pham += 1
print("  thu %d tap h khac: %d tap VI PHAM (3.1), %d tap thoa." % (thu, vi_pham, thu - vi_pham))
print("  -> (3.1) %s" % ("phan biet duoc nghiem toi uu voi cac tap con khac"
                          if vi_pham > 0.9 * thu else "KHONG phan biet duoc, phep thu that bai"))

print()
print("CONG THUC (3.6) tr.40 — ban thu gon cua (3.1) ve MO HINH VI TRI:")
print("   theta - 1 < sum_{t in h_ngang} [1/2 - 1/2 sgn(y_t - y(h)) - theta] < theta")
print("Bai bao tr.41 doc no thanh: 'the number of y_t's less that y(h) be strictly between")
print("T*theta - 1 and T*theta. This in turn demands that T*theta be nonintegral.'")
print()
print("%-5s %-8s %-11s %-13s %-30s %-11s" %
      ("T", "theta", "T*theta", "nguyen?", "so y_t < phan vi mau", "trong (T*th-1, T*th)?"))
rng = np.random.default_rng(36)
for Tn in (19, 20, 40):
    for th in (0.25, 0.5, 0.9):
        yv = rng.normal(size=Tn)
        b, _, _, _ = giai_lp(yv, np.ones((Tn, 1)), th)
        dem = int(np.sum(yv < b[0] - 1e-12))
        ngn = abs(Tn * th - round(Tn * th)) < 1e-9
        ok = (Tn * th - 1) < dem < (Tn * th)
        print("%-5d %-8.2f %-11.2f %-13s %-30d %-11s"
              % (Tn, th, Tn * th, "CO" if ngn else "khong", dem,
                 "dung" if ok else "SAI (T*theta nguyen)"))
print()
print("Doc cot cuoi: moi dong co T*theta NGUYEN deu truot bat dang thuc ngat — dung nhu")
print("cau tr.41 noi. Do khong phai loi cua bai bao ma la dieu bai bao BAO TRUOC.")


# ---------------------------------------------------- 5. Dinh ly 3.5 tr.41
tieu_de("5. DINH LY 3.5 tr.41 — day quan sat RA XA duong thi duong KHONG doi")
print('"beta*(theta) in B*(theta, X beta* + D u*, X) where D is any T x T diagonal matrix')
print(' with nonnegative elements." Remark tr.42 goi day la mot tinh chat hinh hoc:')
print("keo mot quan sat ra xa ma KHONG doi ben thi duong dung yen.")
print()
rng = np.random.default_rng(1964)
Tn, K, th = 45, 2, 0.30
X = np.c_[np.ones(Tn), rng.uniform(0, 10, Tn)]
y = X @ np.array([1.0, 0.5]) + rng.normal(size=Tn)
b, _, _, _ = giai_lp(y, X, th)
u = y - X @ b
print("%-28s %-22s %-14s" % ("he so d_t cua ma tran D", "beta* sau khi day", "lech"))
LECH35 = []
for ten, d in (("d_t = 1 (khong doi)", np.ones(Tn)),
               ("d_t = 3 (day xa 3 lan)", 3.0 * np.ones(Tn)),
               ("d_t = 25", 25.0 * np.ones(Tn)),
               ("d_t ngau nhien trong [0,50]", rng.uniform(0, 50, Tn))):
    y2 = X @ b + d * u
    b2, _, _, _ = giai_lp(y2, X, th)
    L = float(np.max(np.abs(b2 - b)))
    LECH35.append(L)
    print("%-28s [%8.5f %8.5f] %-14.3e" % (ten, b2[0], b2[1], L))
print()
print("lech toi da = %.3e" % max(LECH35))
assert max(LECH35) < 1e-8
print("PHEP THU NGUOC — cho mot he so AM (D khong con khong am) thi tinh chat phai GAY:")
d = np.ones(Tn)
d[int(np.argmax(np.abs(u)))] = -4.0
b3, _, _, _ = giai_lp(X @ b + d * u, X, th)
print("  doi dau mot he so d_t: lech = %.6f  -> %s"
      % (float(np.max(np.abs(b3 - b))),
         "duong DA doi, dung nhu du doan" if np.max(np.abs(b3 - b)) > 1e-6
         else "duong khong doi (phep thu nguoc KHONG phan biet duoc)"))


# ---------------------------------------------------- 6. BANG II tr.42 — dung lai
tieu_de("6. BANG II tr.42 — DUNG LAI CA BANG, VA DUNG LAI CA DU LIEU BAI BAO KHONG IN")
print("Bai bao viet: 'an illustration in Table II of the regression quantiles for all")
print("theta in (0,1) in a simple bivariate model with five observations. Note that for")
print("theta in {7/22, 1/2, 3/4} the problem admits multiple solutions.' (tr.42)")
print("Bang in NGHIEM cho bon khoang theta — nhung KHONG in nam quan sat.")
print()

BANG_II = [(F(6, 7), F(4, 7)), (F(21, 8), F(3, 8)), (F(13, 6), F(5, 6)), (F(17, 3), F(1, 3))]
MOC_II = [F(7, 22), F(1, 2), F(3, 4)]
print("Bang II nhu bai bao in:")
print("            0 < th <= 7/22   7/22 <= th <= 1/2   1/2 <= th <= 3/4   3/4 <= th < 1")
print("beta*_1 :   %-16s %-19s %-18s %s" % tuple(str(b[0]) for b in BANG_II))
print("beta*_2 :   %-16s %-19s %-18s %s" % tuple(str(b[1]) for b in BANG_II))
print()

print("BUOC 1 — hai nghiem lien tiep phai CHUNG dung mot quan sat (K - 1 = 1 diem).")
print("Giao diem cua hai duong lien tiep vi the la mot quan sat cua bai bao:")
CHUNG = []
for i in range(3):
    (a1, a2), (c1, c2) = BANG_II[i], BANG_II[i + 1]
    x = (c1 - a1) / (a2 - c2)
    yv = a1 + a2 * x
    CHUNG.append((x, yv))
    print("   nghiem %d & %d  ->  (x, y) = (%s, %s)" % (i + 1, i + 2, x, yv))
print("Ca ba deu la SO NGUYEN, chua can gia thiet gi.")
print()


def duong_nghiem(pts):
    """Duong nghiem chinh xac theo theta. psi(h, theta) = theta*A_h + C_h tuyen tinh
       theo theta, nen nghiem toi uu la BAO DUOI cua C(T,K) duong thang."""
    cand = []
    for h in itertools.combinations(range(len(pts)), 2):
        (x1, ya), (x2, yb) = pts[h[0]], pts[h[1]]
        if x1 == x2:
            continue
        b2 = (yb - ya) / (x2 - x1)
        b1 = ya - b2 * x1
        e = [yv - (b1 + b2 * xv) for xv, yv in pts]
        cand.append((b1, b2, sum(e), sum(-v for v in e if v < 0)))
    ngat = {F(0), F(1)}
    for (_, _, a1, c1) in cand:
        for (_, _, a2, c2) in cand:
            if a1 != a2:
                t = F(c2 - c1, a1 - a2)
                if 0 < t < 1:
                    ngat.add(t)
    ngat = sorted(ngat)
    gop = []
    for i in range(len(ngat) - 1):
        th = (ngat[i] + ngat[i + 1]) / 2
        m = min(th * a + c for (_, _, a, c) in cand)
        sol = sorted({(b1, b2) for (b1, b2, a, c) in cand if th * a + c == m})
        if gop and gop[-1][2] == sol:
            gop[-1][1] = ngat[i + 1]
        else:
            gop.append([ngat[i], ngat[i + 1], sol])
    return gop


print("BUOC 2 — hai quan sat con lai nam tren duong dau va duong cuoi. Quet chung.")
print("Mien quet: x huu ty p/q voi |p/q| <= 20 va q <= 6 (y suy ra tu phuong trinh duong).")
UNG_VIEN = sorted({F(p, q) for q in range(1, 7) for p in range(-120, 121)
                   if abs(F(p, q)) <= 20})
print("So gia tri x duoc thu cho moi dau: %d  ->  %d cap." % (len(UNG_VIEN), len(UNG_VIEN) ** 2))
def duong_nghiem_f(pts):
    """Ban FLOAT cua duong_nghiem, dung de LOC TRUOC. Nhanh hon ~60 lan; moi ung
       vien qua duoc deu phai xac nhan lai bang so huu ty o duoi."""
    cand = []
    for h in itertools.combinations(range(5), 2):
        (x1, ya), (x2, yb) = pts[h[0]], pts[h[1]]
        if x1 == x2:
            continue
        b2 = (yb - ya) / (x2 - x1)
        b1 = ya - b2 * x1
        e = [yv - (b1 + b2 * xv) for xv, yv in pts]
        cand.append((b1, b2, sum(e), sum(-v for v in e if v < 0)))
    ngat = [0.0, 1.0]
    for (_, _, a1, c1) in cand:
        for (_, _, a2, c2) in cand:
            if abs(a1 - a2) > 1e-12:
                t = (c2 - c1) / (a1 - a2)
                if 1e-12 < t < 1 - 1e-12:
                    ngat.append(t)
    ngat = sorted(set(round(t, 12) for t in ngat))
    gop = []
    for i in range(len(ngat) - 1):
        th = 0.5 * (ngat[i] + ngat[i + 1])
        val = [th * a + c for (_, _, a, c) in cand]
        m = min(val)
        k = min(j for j, v in enumerate(val) if v < m + 1e-11)
        sol = (round(cand[k][0], 9), round(cand[k][1], 9))
        if gop and gop[-1][1] == sol:
            gop[-1][0] = ngat[i + 1]
        else:
            gop.append([ngat[i + 1], sol])
    return gop


BII_F = [(round(float(a), 9), round(float(b), 9)) for a, b in BANG_II]
MOC_F = [round(float(m), 12) for m in MOC_II]
CHUNG_F = [(float(a), float(b)) for a, b in CHUNG]
khop_nghiem, khop_het = [], []
for xa in UNG_VIEN:
    ya = BANG_II[0][0] + BANG_II[0][1] * xa
    xaf, yaf = float(xa), float(ya)
    for xe in UNG_VIEN:
        ye = BANG_II[3][0] + BANG_II[3][1] * xe
        if len({xa, xe, CHUNG[0][0], CHUNG[1][0], CHUNG[2][0]}) != 5:
            continue
        gf = duong_nghiem_f([(xaf, yaf)] + CHUNG_F + [(float(xe), float(ye))])
        if [q[1] for q in gf] != BII_F:
            continue
        pts = [(xa, ya), CHUNG[0], CHUNG[1], CHUNG[2], (xe, ye)]
        g = duong_nghiem(pts)                       # xac nhan lai bang SO HUU TY
        if [q[2][0] for q in g] != BANG_II:
            continue
        khop_nghiem.append(pts)
        if [q[0] for q in g[1:]] == MOC_II:
            khop_het.append((pts, g))
print()
print("So bo du lieu cho ra DUNG bon nghiem cua Bang II, dung thu tu : %d" % len(khop_nghiem))
print("So bo du lieu DONG THOI khop ca ba moc 7/22, 1/2, 3/4          : %d" % len(khop_het))
print()
print("-> Bon nghiem KHONG du de xac dinh du lieu; ba moc moi la bang chung phan xu.")
assert len(khop_het) == 1
PTS, DN = khop_het[0]
PTS_SX = sorted(PTS)
print()
print("NAM QUAN SAT CUA BAI BAO, dung lai duoc:")
print("   t  :  1   2   3   4   5")
print("   x_t:  %s" % "   ".join(str(p[0]) for p in PTS_SX))
print("   y_t:  %s" % "   ".join(str(p[1]) for p in PTS_SX))
print("   (ca muoi con so deu NGUYEN — khong phai gia thiet, la ket qua cua phep quet)")
print()
print("BUOC 3 — dung lai TOAN BO Bang II tu nam diem do:")
print("%-22s %-14s %-14s %-12s" % ("khoang theta", "beta*_1", "beta*_2", "so nghiem co so"))
DUNG_LAI = []
for lo, hi, sol in DN:
    print("%-22s %-14s %-14s %-12d" % ("%s <= theta <= %s" % (lo, hi),
                                       str(sol[0][0]), str(sol[0][1]), len(sol)))
    DUNG_LAI.append((sol[0][0], sol[0][1]))
print()
print("So o khop Bang II: %d/8"
      % sum((a[0] == b[0]) + (a[1] == b[1]) for a, b in zip(DUNG_LAI, BANG_II)))
print("So moc khop     : %d/3" % sum(1 for m, s in zip(MOC_II, [s[0] for s in DN[1:]]) if m == s))
assert DUNG_LAI == BANG_II

print()
print("BUOC 4 — kiem cau 'for theta in {7/22, 1/2, 3/4} the problem admits multiple")
print("solutions': dem so nghiem co so toi uu DUNG TAI moc.")
for m in MOC_II:
    cand = []
    for h in itertools.combinations(range(5), 2):
        (x1, ya), (x2, yb) = PTS_SX[h[0]], PTS_SX[h[1]]
        if x1 == x2:
            continue
        b2 = (yb - ya) / (x2 - x1)
        b1 = ya - b2 * x1
        e = [yv - (b1 + b2 * xv) for xv, yv in PTS_SX]
        cand.append((b1, b2, m * sum(e) + sum(-v for v in e if v < 0)))
    mn = min(c[2] for c in cand)
    tie = sorted({(c[0], c[1]) for c in cand if c[2] == mn})
    print("   theta = %-5s : %d nghiem co so toi uu  ->  %s"
          % (str(m), len(tie), " ; ".join("(%s, %s)" % (a, b) for a, b in tie)))
print("Ca ba moc deu co NHIEU HON MOT nghiem — dung nhu bai bao noi.")

print()
print("BUOC 5 — PHEP THU NGUOC. Doi mot toa do di mot don vi roi dung lai Bang II:")
for i in range(5):
    for dv in (-1, 1):
        p2 = list(PTS_SX)
        p2[i] = (p2[i][0], p2[i][1] + dv)
        if i == 0 and dv == -1:
            pass
        g2 = duong_nghiem(p2)
        khop = ([s[2][0] for s in g2] == BANG_II) and ([s[0] for s in g2[1:]] == MOC_II)
        if khop:
            print("   doi y_%d %+d : VAN KHOP  <-- phep thu nguoc THAT BAI" % (i + 1, dv))
print("   khong bo nao trong 10 bo lech mot don vi con khop -> nam diem la duy nhat")
print("   trong mien quet.")

print()
print("BUOC 6 — doi chieu voi bo giai LP so thuc (khong dung phan so):")
Xq = np.array([[1.0, float(p[0])] for p in PTS_SX])
yq = np.array([float(p[1]) for p in PTS_SX])
print("%-9s %-16s %-16s %-16s %-16s %-10s" %
      ("theta", "beta*_1 LP", "beta*_1 Bang II", "beta*_2 LP", "beta*_2 Bang II", "lech"))
LECH_LP = []
for th, (b1f, b2f) in zip((0.2, 0.4, 0.6, 0.9), BANG_II):
    bb, _, _, _ = giai_lp(yq, Xq, th)
    L = max(abs(bb[0] - float(b1f)), abs(bb[1] - float(b2f)))
    LECH_LP.append(L)
    print("%-9.2f %-16.10f %-16.10f %-16.10f %-16.10f %-10.2e"
          % (th, bb[0], float(b1f), bb[1], float(b2f), L))
print("lech toi da = %.2e" % max(LECH_LP))
assert max(LECH_LP) < 1e-9

print()
print("BUOC 7 — in lai TUNG BUOC cua vi du chay tay o muc 0.4 cua bai. Moi vi du so")
print("chay tay in trong bai deu phai co mot muc script in lai tung buoc (luat YOLOv5).")
print("theta = 0.2, nghiem beta* = (6/7, 4/7):")
_th = F(1, 5)
_b1, _b2 = BANG_II[0]
_tong = F(0)
print("   %-4s %-6s %-6s %-10s %-10s %-9s %-12s" %
      ("t", "x_t", "y_t", "y_mu", "u_t", "dau", "dong gop"))
for _t, (_x, _yv) in enumerate(PTS_SX, start=1):
    _ym = _b1 + _b2 * _x
    _u = _yv - _ym
    _gop = _u * (_th - (1 if _u < 0 else 0))
    _tong += _gop
    print("   %-4d %-6s %-6s %-10s %-10s %-9s %-12s"
          % (_t, _x, _yv, _ym, _u,
             "duong" if _u > 0 else ("am" if _u < 0 else "bang 0"), _gop))
print("   tong = %s = %.10f" % (_tong, float(_tong)))
print("   so phan du bang 0 = %d (= K = 2) · N = %d · T*theta = %s"
      % (sum(1 for _x, _yv in PTS_SX if _yv == _b1 + _b2 * _x),
         sum(1 for _x, _yv in PTS_SX if _yv < _b1 + _b2 * _x), F(5) * _th))


# ---------------------------------------------------- 7. PHU LUC tr.48 — (P), (D), (D')
tieu_de("7. PHU LUC tr.48-49 — bai doi ngau (D'), thu ma bai viet cu bo qua hoan toan")
print("(P)  min [theta*i'r+ + (1-theta)*i'r-]   s.t.  y = Xb + r+ - r-,  r+, r- >= 0")
print("(D)  max [y'd]                           s.t.  X'd = 0,  d in [theta-1, theta]^T")
print("(D') max [y'Delta]                       s.t.  X'Delta = (1-theta)X'i, Delta in [0,1]^T")
print("     voi Delta = d + 1 - theta  (bai bao tr.48).")
print()
rng = np.random.default_rng(48)
Tn, K = 50, 3
X = np.c_[np.ones(Tn), rng.normal(size=(Tn, K - 1))]
y = X @ np.array([1.0, 2.0, -0.5]) + rng.standard_t(3, size=Tn)
print("Doi bien Delta = d + 1 - theta lam ham muc tieu doi mot hang so: ")
print("  y'Delta = y'd + (1-theta) * sum(y_t).  Nen gia tri (D') KHONG bang gia tri (P);")
print("  hieu cua chung phai bang DUNG (1-theta)*sum(y). Kiem dung cot cuoi.")
print("  sum(y_t) = %.10f" % float(np.sum(y)))
print()
print("%-8s %-16s %-16s %-15s %-15s %-8s %-8s %-7s" %
      ("theta", "gia tri (P)", "gia tri (D')", "hieu", "(1-th)*sum(y)", "D=0", "D=1", "0<D<1"))
LECH_DUAL, LECH_HIEU = [], []
for th in (0.1, 0.25, 0.5, 0.75, 0.9):
    b, rp, rm, fP = giai_lp(y, X, th)
    D, fD = giai_dual(y, X, th)
    u = y - X @ b
    hs = fD - fP
    dk = (1.0 - th) * float(np.sum(y))
    LECH_HIEU.append(abs(hs - dk))
    n0 = int(np.sum(D < 1e-9))
    n1 = int(np.sum(D > 1 - 1e-9))
    nm = Tn - n0 - n1
    LECH_DUAL.append((th, fP, fD, hs, n0, n1, nm))
    print("%-8.2f %-16.10f %-16.10f %-15.10f %-15.10f %-8d %-8d %-7d"
          % (th, fP, fD, hs, dk, n0, n1, nm))
print()
print("Lech toi da giua hieu do duoc va (1-theta)*sum(y): %.3e" % max(LECH_HIEU))
assert max(LECH_HIEU) < 1e-8
print()
print("So toa do Delta nam HAN trong (0,1) bang K = %d o %d/5 muc theta." %
      (K, sum(1 for r in LECH_DUAL if r[6] == K)))
print("Do la dung cac quan sat co phan du bang 0 — tuc tap h cua Dinh ly 3.1.")
print()
print("QUAN HE DAU — do ra chu KHONG doan. Bu tru chat che cho:")
print("  phan du DUONG  (r+ > 0) -> d_t chan tren = theta   -> Delta_t = 1")
print("  phan du AM     (r- > 0) -> d_t chan duoi = theta-1 -> Delta_t = 0")
print("  phan du BANG 0          -> d_t tu do trong khoang  -> 0 < Delta_t < 1")
b, _, _, _ = giai_lp(y, X, 0.35)
D, _ = giai_dual(y, X, 0.35)
u = y - X @ b
sai = int(np.sum((u > 1e-9) & (D < 0.5))) + int(np.sum((u < -1e-9) & (D > 0.5)))
nguoc = int(np.sum((u < -1e-9) & (D < 0.5))) + int(np.sum((u > 1e-9) & (D > 0.5)))
print("  theta = 0.35: so quan sat vi pham quan he tren : %d / %d" % (sai, Tn))
print("  cach doc NGUOC lai (Delta=1 khi phan du am)    : %d / %d vi pham" % (nguoc, Tn))
print("  so quan sat co 0 < Delta < 1                   : %d (= K = %d)"
      % (int(np.sum((D > 1e-9) & (D < 1 - 1e-9))), K))
assert sai == 0
print()
print('"parametric variation of the right-hand-side of (D\') permits economical solution')
print(' for many, or indeed all, theta in [0,1]" (tr.49) — ve phai (1-theta)X\'i la duong')
print("thang theo theta, nen doi theta chi doi VE PHAI, khong doi ma tran rang buoc.")


# ---------------------------------------------------- 8. Dinh ly 4.1 tr.42
tieu_de("8. DINH LY 4.1 tr.42 — omega_ij = theta_i(1-theta_j) / [f(xi_i) f(xi_j)],  i <= j")


def omega(ths, f_tai_xi):
    M = len(ths)
    O = np.empty((M, M))
    for i in range(M):
        for j in range(M):
            a, b = min(i, j), max(i, j)
            O[i, j] = ths[a] * (1 - ths[b]) / (f_tai_xi[i] * f_tai_xi[j])
    return O


THS = np.array([0.25, 0.5, 0.75])
f_chuan = norm.pdf(norm.ppf(THS))
OM = omega(THS, f_chuan)
print("Chuan N(0,1), theta = 0.25, 0.50, 0.75 — ma tran Omega theo cong thuc:")
for r in OM:
    print("   " + "  ".join("%12.8f" % v for v in r))
print()
print("Kiem bang Monte-Carlo: T = 4000, 40000 lan lap, lay phan vi mau roi nhan sqrt(T).")
rng = np.random.default_rng(41)
Tmc, R = 4000, 40000
mau = rng.normal(size=(R, Tmc))
mau.sort(axis=1)
idx = [int(np.ceil(Tmc * t)) - 1 for t in THS]
Q = mau[:, idx] * np.sqrt(Tmc)
OM_mc = np.cov(Q, rowvar=False)
print("Ma tran hiep phuong sai DO DUOC:")
for r in OM_mc:
    print("   " + "  ".join("%12.8f" % v for v in r))
print()
print("Lech tuong doi toi da: %.6f %%" % (100 * np.max(np.abs(OM_mc - OM) / OM)))
assert np.max(np.abs(OM_mc - OM) / OM) < 0.05
print()
print("Truong hop rieng tr.43: trung vi co phuong sai tiem can [2 f(xi(1/2))]^{-2}.")
print("  Chuan N(0,1) : [2*f(0)]^{-2} = %.10f" % (1.0 / (2 * norm.pdf(0)) ** 2))
print("  Laplace      : [2*f(0)]^{-2} = %.10f   (f(0) = 1/2)" % (1.0 / (2 * 0.5) ** 2))
print("  Cauchy       : [2*f(0)]^{-2} = %.10f   (f(0) = 1/pi)" % (1.0 / (2 / np.pi) ** 2))


# ---------------------------------------------------- 9. Dinh ly 4.2 tr.43
tieu_de("9. DINH LY 4.2 tr.43 — hiep phuong sai la Omega(theta;F) KRONECKER Q^{-1}")
print("sqrt(T)[beta*_T(theta) - beta(theta)] -> N(0, Omega tich Kronecker Q^{-1}),")
print("voi Q = lim T^{-1} X'X. Dieu kien (ii) doi cot dau cua X la cot toan 1.")
print()
rng = np.random.default_rng(42)
Tmc, K, R = 300, 3, 3000
Xb = np.c_[np.ones(Tmc), rng.normal(size=(Tmc, K - 1))]
Qm = Xb.T @ Xb / Tmc
Qi = np.linalg.inv(Qm)
beta_that = np.array([1.0, 2.0, -1.0])
THS2 = np.array([0.3, 0.7])
f2 = norm.pdf(norm.ppf(THS2))
OM2 = omega(THS2, f2)
LY_THUYET = np.kron(OM2, Qi)
mau_b = np.empty((R, len(THS2) * K))
for r in range(R):
    yb = Xb @ beta_that + rng.normal(size=Tmc)
    for m, th in enumerate(THS2):
        bb, _, _, _ = giai_lp(yb, Xb, th)
        xi = norm.ppf(th)
        mau_b[r, m * K:(m + 1) * K] = np.sqrt(Tmc) * (bb - (beta_that + np.r_[xi, np.zeros(K - 1)]))
DO_DUOC = np.cov(mau_b, rowvar=False)
print("So chieu cua ma tran: %d x %d  (M*K = %d*%d)" % (LY_THUYET.shape + (len(THS2), K)))
print()
print("%-8s %-8s %-18s %-18s %-14s" % ("i", "j", "ly thuyet", "do duoc", "lech t.doi %"))
LECH42 = []
for i in range(len(THS2) * K):
    for j in range(i, len(THS2) * K):
        lt, dd = LY_THUYET[i, j], DO_DUOC[i, j]
        if abs(lt) > 1e-6:
            L = 100 * abs(dd - lt) / abs(lt)
            LECH42.append(L)
            if i == j or (i, j) in ((0, 3), (1, 4), (2, 5)):
                print("%-8d %-8d %-18.10f %-18.10f %-14.4f" % (i, j, lt, dd, L))
print()
LON = np.max(np.abs(LY_THUYET))
MANH = np.abs(LY_THUYET) > 0.1 * LON
L_MANH = 100 * np.max(np.abs((DO_DUOC - LY_THUYET)[MANH] / LY_THUYET[MANH]))
print("Lech TUONG DOI chi doc duoc o nhung o ly thuyet khong gan 0. Tren %d o co"
      % int(MANH.sum()))
L_TB = 100 * float(np.mean(np.abs((DO_DUOC - LY_THUYET)[MANH] / LY_THUYET[MANH])))
print("|ly thuyet| > 10%% cua o lon nhat: lech tuong doi TRUNG BINH %.4f %%,"
      " LON NHAT %.4f %%" % (L_TB, L_MANH))
print("Tren TOAN BO 36 o: lech TUYET DOI toi da %.6f, trong khi o lon nhat la %.6f"
      % (float(np.max(np.abs(DO_DUOC - LY_THUYET))), float(LON)))
print("(Bao ca hai thuoc do la bat buoc: mot o co gia tri ly thuyet 0.0004 cho lech")
print(" tuong doi 488 %% ma lech tuyet doi chi 0.002 — con so tuong doi o do vo nghia.)")
print("(R = %d lan lap, T = %d — sai so Monte-Carlo cua mot phuong sai co ~ sqrt(2/R) = %.4f %%)"
      % (R, Tmc, 100 * np.sqrt(2.0 / R)))
print()
print("PHEP THU NGUOC — bo cot toan 1 khoi X (vi pham dieu kien (ii)) thi Dinh ly 3.4")
print("khong con ap dung duoc:")
Xn = rng.normal(size=(Tmc, 2))
bn, _, _, _ = giai_lp(Xn @ np.array([1.0, 2.0]) + rng.normal(size=Tmc), Xn, 0.3)
un = (Xn @ np.array([1.0, 2.0]) + 0) * 0
yn = Xn @ np.array([1.0, 2.0]) + rng.normal(size=Tmc)
bn, _, _, _ = giai_lp(yn, Xn, 0.3)
un = yn - Xn @ bn
print("  khong co cot 1: N = %d, T*theta = %.1f, (3.7) doi N <= T*theta -> %s"
      % (int(np.sum(un < -1e-9)), Tmc * 0.3,
         "van dung" if np.sum(un < -1e-9) <= Tmc * 0.3 else "SAI, dung nhu du doan"))


# ---------------------------------------------------- 10. Dinh ly 4.3 tr.46
tieu_de("10. DINH LY 4.3 tr.46 — uoc luong to hop pi'Omega pi . Q^{-1}")
print("Remark tr.46 neu dich danh hai bo trong so:")
print("  Gastwirth: (pi(1/3), pi(1/2), pi(2/3)) = (.3, .4, .3)")
print("  Trimean  : (pi(1/4), pi(1/2), pi(3/4)) = (1/4, 1/2, 1/4)")
print()
GAST_TH = np.array([1 / 3, 0.5, 2 / 3])
GAST_PI = np.array([0.3, 0.4, 0.3])
TRIM_TH = np.array([0.25, 0.5, 0.75])
TRIM_PI = np.array([0.25, 0.5, 0.25])


def mat_do(ten):
    """tra ve (ham mat do, ham phan vi, phuong sai, thong tin Fisher) cua F."""
    if ten == "Chuan":
        return norm.pdf, norm.ppf, 1.0, 1.0
    if ten == "Laplace":
        return (lambda x: 0.5 * np.exp(-np.abs(x)),
                lambda p: np.where(p < 0.5, np.log(2 * p), -np.log(2 * (1 - p))),
                2.0, 1.0)
    if ten == "Logistic":
        return (lambda x: np.exp(-x) / (1 + np.exp(-x)) ** 2,
                lambda p: np.log(p / (1 - p)),
                np.pi ** 2 / 3, 1.0 / 3.0)
    if ten == "Cauchy":
        return (lambda x: 1.0 / (np.pi * (1 + np.asarray(x, dtype=float) ** 2)),
                lambda p: np.tan(np.pi * (np.asarray(p, dtype=float) - 0.5)),
                np.inf, 0.5)
    raise ValueError(ten)


def pv_tiem_can(ths, pis, ten):
    f, qf, _, _ = mat_do(ten)
    return float(pis @ omega(np.asarray(ths), f(qf(np.asarray(ths)))) @ pis)


print("%-12s %-16s %-16s %-16s %-16s" %
      ("F", "trung binh", "trung vi", "Gastwirth", "Trimean"))
PV = {}
for ten in ("Chuan", "Laplace", "Logistic", "Cauchy"):
    f, qf, var, _ = mat_do(ten)
    pv_tv = 1.0 / (2 * f(0.0)) ** 2
    pv_g = pv_tiem_can(GAST_TH, GAST_PI, ten)
    pv_t = pv_tiem_can(TRIM_TH, TRIM_PI, ten)
    PV[ten] = (var, pv_tv, pv_g, pv_t)
    print("%-12s %-16s %-16.10f %-16.10f %-16.10f"
          % (ten, ("vo cuc" if not np.isfinite(var) else "%.10f" % var), pv_tv, pv_g, pv_t))
print()
print("Kiem Gastwirth bang Monte-Carlo tren phan phoi Chuan (T = 4000, 20000 lan lap):")
rng = np.random.default_rng(46)
Tmc, R = 4000, 20000
mau = rng.normal(size=(R, Tmc))
mau.sort(axis=1)
ig = [int(np.ceil(Tmc * t)) - 1 for t in GAST_TH]
gst = mau[:, ig] @ GAST_PI
print("  ly thuyet T*Var = %.10f   do duoc = %.10f   lech = %.4f %%"
      % (PV["Chuan"][2], Tmc * np.var(gst),
         100 * abs(Tmc * np.var(gst) - PV["Chuan"][2]) / PV["Chuan"][2]))


# ---------------------------------------------------- 11. cau "nearly eighty per cent" tr.36
tieu_de("11. TR.36 — 'asymptotic efficiency of NEARLY EIGHTY PER CENT for the Gaussian,")
print("     Laplace, logistic, AND Cauchy distributions' — noi ve Gastwirth.")
print("Hieu suat = can duoi Cramer-Rao / phuong sai tiem can = 1 / [I(F) * V].")
print()
print("%-12s %-16s %-18s %-16s %-16s" %
      ("F", "I(F)", "V(Gastwirth)", "hieu suat", "'gan 80%' ?"))
HS_GAST = []
for ten in ("Chuan", "Laplace", "Logistic", "Cauchy"):
    _, _, _, IF = mat_do(ten)
    V = PV[ten][2]
    hs = 1.0 / (IF * V)
    HS_GAST.append(hs)
    print("%-12s %-16.10f %-18.10f %-16.10f %-16s"
          % (ten, IF, V, hs, "%.2f %%" % (100 * hs)))
print()
print("Khoang hieu suat tren bon phan phoi: [%.10f ; %.10f], do rong %.10f"
      % (min(HS_GAST), max(HS_GAST), max(HS_GAST) - min(HS_GAST)))
print()
print("CACH DOC. Cau tr.36 phai doc la mot SAN, khong phai mot khoang: 'nearly eighty")
print("per cent' dung cho CA BON phan phoi cung mot luc, nen no noi ve gia tri NHO NHAT.")
print("  gia tri nho nhat trong bon = %.10f  ->  %.2f %%" % (min(HS_GAST), 100 * min(HS_GAST)))
print("  so phan phoi co hieu suat >= 0.79 : %d/4" % sum(1 for h in HS_GAST if h >= 0.79))
print("  phan phoi CAO nhat la Logistic voi %.10f — bai bao noi NHE hon thuc te o day."
      % max(HS_GAST))
print()
print("Doi chieu: trung binh mau, cung thuoc do.")
print("%-12s %-18s %-16s" % ("F", "V(trung binh)", "hieu suat"))
for ten in ("Chuan", "Laplace", "Logistic", "Cauchy"):
    _, _, _, IF = mat_do(ten)
    var = PV[ten][0]
    print("%-12s %-18s %-16s"
          % (ten, ("vo cuc" if not np.isfinite(var) else "%.10f" % var),
             ("0 (hieu suat bang khong)" if not np.isfinite(var) else "%.10f" % (1.0 / (IF * var)))))
print()
print("Do la y cua cau tr.36: trung binh mau co hieu suat 1 o Chuan nhung 0 o Cauchy;")
print("Gastwirth giu %.4f - %.4f tren CA BON." % (min(HS_GAST), max(HS_GAST)))


# ---------------------------------------------------- 12. BANG I tr.37
tieu_de("12. BANG I tr.37 — dung lai 30 o bang Monte-Carlo, T = 20")
BANG_I = {
    "Mean": (1.00, 1.88, 11.54, 2.10, 12548.0),
    "10% trimmed mean": (1.06, 1.31, 1.46, 1.60, 7.3),
    "25% trimmed mean": (1.20, 1.41, 1.47, 1.33, 3.1),
    "Median": (1.50, 1.70, 1.80, 1.37, 2.9),
    "Gastwirth": (1.23, 1.45, 1.51, 1.35, 3.1),
    "Trimean": (1.15, 1.37, 1.48, 1.43, 3.9),
}
PHAN_PHOI = ("Normal", "10% 3sigma", "10% 10sigma", "Laplace", "Cauchy")
print("Bang I nhu bai bao in (chu thich a: 'Abstracted from Exhibit 5 in Andrews, et al. [3]'):")
print("%-20s %-10s %-12s %-13s %-10s %-12s" % (("Estimators",) + PHAN_PHOI))
for k, v in BANG_I.items():
    print("%-20s %-10.2f %-12.2f %-13.2f %-10.2f %-12.1f" % ((k,) + v))
print()
print("Chu thich bang: b '10% 3sigma' = hon hop Gauss .9*Phi(1) + .1*Phi(3);")
print("                c '10% 10sigma' = .9*Phi(1) + .1*Phi(10);")
print("                d Gastwirth = .3*beta*(1/3) + .4*beta*(1/2) + .3*beta*(2/3);")
print("                e Trimean   = 1/4*beta*(1/4) + 1/2*beta*(1/2) + 1/4*beta*(3/4).")
print()

TT = 20
print("BUOC 1 — dieu kien duy nhat cua chinh bai bao (tr.41): T*theta phai KHONG NGUYEN.")
print("Voi T = %d:" % TT)
for nh, ths in (("Median", (0.5,)), ("Gastwirth", tuple(GAST_TH)), ("Trimean", tuple(TRIM_TH))):
    hang = ["%s: T*theta = %.2f %s" % (("%.4f" % t), TT * t,
                                       "(NGUYEN -> khong duy nhat)" if abs(TT * t - round(TT * t)) < 1e-9
                                       else "(khong nguyen -> duy nhat)") for t in ths]
    print("   %-11s %s" % (nh, " | ".join(hang)))
print("-> quy uoc chon nghiem trong doan la mot LUA CHON, khong phai mot dinh nghia.")
print("   Do ca hai quy uoc roi de TONG SAI LECH chon (luat 'thu ca hai cach doc').")
print()


def sinh(ten_pp, r, tt, rg):
    if ten_pp == "Normal":
        return rg.normal(size=(r, tt))
    if ten_pp == "10% 3sigma":
        z = rg.normal(size=(r, tt))
        s = np.where(rg.random((r, tt)) < 0.1, 3.0, 1.0)
        return z * s
    if ten_pp == "10% 10sigma":
        z = rg.normal(size=(r, tt))
        s = np.where(rg.random((r, tt)) < 0.1, 10.0, 1.0)
        return z * s
    if ten_pp == "Laplace":
        return rg.laplace(size=(r, tt))
    if ten_pp == "Cauchy":
        return rg.standard_cauchy(size=(r, tt))
    raise ValueError(ten_pp)


def pv_mau(s, th, quy_uoc):
    """phan vi mau thu theta cua tung dong cua ma tran DA SAP XEP."""
    tt = s.shape[1]
    k = tt * th
    if abs(k - round(k)) < 1e-9:                    # T*theta nguyen -> ca mot doan
        lo, hi = int(round(k)) - 1, int(round(k))
        if quy_uoc == "trung binh":
            return 0.5 * (s[:, lo] + s[:, hi])
        return s[:, lo]                              # quy uoc "lay dau duoi"
    return s[:, int(np.ceil(k)) - 1]


def bo_uoc_luong(s, quy_uoc):
    tt = s.shape[1]
    out = {}
    out["Mean"] = s.mean(axis=1)
    for a, ten in ((0.10, "10% trimmed mean"), (0.25, "25% trimmed mean")):
        g = int(round(a * tt))
        out[ten] = s[:, g:tt - g].mean(axis=1)
    out["Median"] = pv_mau(s, 0.5, quy_uoc)
    out["Gastwirth"] = sum(w * pv_mau(s, t, quy_uoc) for w, t in zip(GAST_PI, GAST_TH))
    out["Trimean"] = sum(w * pv_mau(s, t, quy_uoc) for w, t in zip(TRIM_PI, TRIM_TH))
    return out


R = 120000
print("BUOC 2 — chay lai Monte-Carlo: T = %d, %d lan lap, 3 hat giong." % (TT, R))
KQ = {}
for quy_uoc in ("trung binh", "lay dau duoi"):
    KQ[quy_uoc] = {}
    for hg in (0, 1, 2):
        rg = np.random.default_rng(1978 + hg)
        for pp in PHAN_PHOI:
            s = sinh(pp, R, TT, rg)
            s.sort(axis=1)
            for ten, v in bo_uoc_luong(s, quy_uoc).items():
                KQ[quy_uoc].setdefault((ten, pp), []).append(TT * float(np.var(v)))

print()
print("Tong sai lech tuyet doi so voi Bang I, TRU cot Cauchy (cot do co phuong sai vo cuc):")
for quy_uoc in ("trung binh", "lay dau duoi"):
    tong = 0.0
    for ten, v in BANG_I.items():
        for j, pp in enumerate(PHAN_PHOI):
            if pp == "Cauchy":
                continue
            tong += abs(float(np.mean(KQ[quy_uoc][(ten, pp)])) - v[j])
    print("   quy uoc '%-12s' : tong sai lech tren 24 o = %.6f" % (quy_uoc, tong))
QU = min(("trung binh", "lay dau duoi"),
         key=lambda q: sum(abs(float(np.mean(KQ[q][(t, p)])) - BANG_I[t][j])
                           for t in BANG_I for j, p in enumerate(PHAN_PHOI) if p != "Cauchy"))
print("-> quy uoc khop hon: '%s'" % QU)
print()
print("BANG I DUNG LAI (quy uoc '%s', trung binh 3 hat giong):" % QU)
print("%-20s %-10s %-12s %-13s %-10s %-12s" % (("Estimators",) + PHAN_PHOI))
LECH_I, LECH_I_CHITIET = [], []
for ten in BANG_I:
    hang = [float(np.mean(KQ[QU][(ten, pp)])) for pp in PHAN_PHOI]
    print("%-20s %-10.4f %-12.4f %-13.4f %-10.4f %-12.1f" % ((ten,) + tuple(hang)))
    for j, pp in enumerate(PHAN_PHOI):
        if pp != "Cauchy":
            LECH_I.append(abs(hang[j] - BANG_I[ten][j]))
            LECH_I_CHITIET.append((ten, pp, BANG_I[ten][j], hang[j], hang[j] - BANG_I[ten][j]))
print()
print("24 o ngoai cot Cauchy: lech tuyet doi trung binh %.6f, lon nhat %.6f"
      % (float(np.mean(LECH_I)), float(np.max(LECH_I))))
print("So o khop den chu so thap phan thu hai bai bao in: %d/24"
      % sum(1 for (_, _, a, b, _) in LECH_I_CHITIET if round(b, 2) == a))
print("So o lech duoi 0.10 : %d/24" % sum(1 for L in LECH_I if L < 0.10))
print("So o lech duoi 0.20 : %d/24" % sum(1 for L in LECH_I if L < 0.20))
print()
print("Nam o lech nhieu nhat:")
for ten, pp, a, b, d in sorted(LECH_I_CHITIET, key=lambda r: -abs(r[4]))[:5]:
    print("   %-20s %-13s bai bao %8.2f   do lai %10.4f   lech %+8.4f" % (ten, pp, a, b, d))
print()
print("CHAN DOAN o lech nhat — '10% trimmed mean' tren hon hop 10% 10sigma.")
print("Cat 10%% cua T = %d la %d quan sat moi dau. So diem bi nhiem la Binomial(%d, 0.1)."
      % (TT, int(round(0.10 * TT)), TT))
from math import comb as _comb
_p3 = sum(_comb(TT, _k) * 0.1 ** _k * 0.9 ** (TT - _k) for _k in range(3, TT + 1))
print("  P(so diem nhiem >= 3) = %.10f — tuc %.2f %% so mau co diem nhiem CON LAI"
      % (_p3, 100 * _p3))
print("  sau khi cat. Do la ly do phep do cho %.4f thay vi %.2f."
      % (float(np.mean(KQ[QU][("10% trimmed mean", "10% 10sigma")])), BANG_I["10% trimmed mean"][2]))
print()
print("GIA TRI LY THUYET CHINH XAC cua hai cot hon hop, cho hang 'Mean'. Phuong sai cua")
print("mot hon hop Gauss .9*Phi(1) + .1*Phi(c) la .9*1 + .1*c^2, dung voi MOI T:")
for _ten, _c in (("10% 3sigma", 3.0), ("10% 10sigma", 10.0)):
    _lt = 0.9 * 1.0 + 0.1 * _c * _c
    _bi = BANG_I["Mean"][PHAN_PHOI.index(_ten)]
    _do = float(np.mean(KQ[QU][("Mean", _ten)]))
    print("   %-13s ly thuyet %-12.6f Bang I %-10.2f do lai %-12.6f"
          % (_ten, _lt, _bi, _do))
    print("   %-13s Bang I lech ly thuyet %+.6f (%+.4f %%) · do lai lech %+.6f (%+.4f %%)"
          % ("", _bi - _lt, 100 * (_bi - _lt) / _lt, _do - _lt, 100 * (_do - _lt) / _lt))
print()
print("LUU Y VE NGUON: chu thich a cua Bang I ghi 'Abstracted from Exhibit 5 in")
print("Andrews, et al. [3]'. Bang nay KHONG phai phep do cua Koenker & Bassett; chenh")
print("lech o tren la chenh voi Nghien cuu Ben vung Princeton, khong phai voi bai bao.")
print()
print("COT CAUCHY phai doc rieng — phuong sai LY THUYET cua trung binh la VO CUC,")
print("nen con so 12548.0 khong tai lap duoc. Ba hat giong cho:")
for ten in BANG_I:
    v = KQ[QU][(ten, "Cauchy")]
    print("   %-20s bai bao %9.1f   ba lan chay: %s   do rong khoang %s"
          % (ten, BANG_I[ten][4],
             " ".join("%11.2f" % x for x in v), "%.2f" % (max(v) - min(v))))
print()
print("Ti so do rong khoang / gia tri bai bao, cot Cauchy:")
for ten in BANG_I:
    v = KQ[QU][(ten, "Cauchy")]
    print("   %-20s %.6f" % (ten, (max(v) - min(v)) / BANG_I[ten][4]))
print()
print("BUOC 3 — so gia tri TIEM CAN (Dinh ly 4.1) voi gia tri T = 20 cua Bang I.")
print("Bai bao tr.43 noi dich danh: 'The asymptotic variance of the Gastwirth or trimean")
print("estimator of Table I can be easily calculated in a similar manner from Theorem 4.1'.")
print()
PP_MAP = {"Normal": "Chuan", "Laplace": "Laplace"}
print("%-20s %-12s %-16s %-16s %-14s" %
      ("uoc luong", "F", "tiem can", "Bang I (T=20)", "chenh"))
for ten, lay in (("Mean", lambda t: PV[t][0]), ("Median", lambda t: PV[t][1]),
                 ("Gastwirth", lambda t: PV[t][2]), ("Trimean", lambda t: PV[t][3])):
    for pp, tkey in PP_MAP.items():
        tc = lay(tkey)
        bi = BANG_I[ten][PHAN_PHOI.index(pp)]
        print("%-20s %-12s %-16.10f %-16.2f %+14.6f" % (ten, pp, tc, bi, bi - tc))
print()
print("Cot Cauchy, gia tri tiem can:")
print("   Mean      : VO CUC          (bai bao in %.1f — mot con so cua T = 20, khong hoi tu)"
      % BANG_I["Mean"][4])
print("   Median    : %.10f  (bai bao in %.1f)" % (PV["Cauchy"][1], BANG_I["Median"][4]))
print("   Gastwirth : %.10f  (bai bao in %.1f)" % (PV["Cauchy"][2], BANG_I["Gastwirth"][4]))
print("   Trimean   : %.10f  (bai bao in %.1f)" % (PV["Cauchy"][3], BANG_I["Trimean"][4]))
print()
print("Ti so 'trung binh / trung vi' o cot Cauchy: Bang I cho %.4f;"
      % (BANG_I["Mean"][4] / BANG_I["Median"][4]))
print("tiem can cho VO CUC. Do la o duy nhat cua bang ma ly thuyet va so do khac nhau")
print("ve BAN CHAT chu khong ve do lon.")


# ---------------------------------------------------- 13. tr.35 — dieu kien Laplace 1818
tieu_de("13. TR.35 — dieu kien cua Laplace nam 1818:  [2 f(0)]^{-1} < sigma")
print("'In 1818 Laplace proved that in the simple model of bivariate regression through")
print(" the origin, this least absolute error (LAE) estimator had smaller asymptotic")
print(" variance than the least squares estimator if the error law of the model had")
print(" variance, sigma^2, and density at the median, f(0), satisfying [2f(0)]^{-1} < sigma.'")
print()
print("%-12s %-18s %-18s %-14s %-22s" %
      ("F", "[2 f(0)]^{-1}", "sigma", "LAE thang ?", "ti so phuong sai LS/LAE"))
for ten in ("Chuan", "Laplace", "Logistic", "Cauchy"):
    f, _, var, _ = mat_do(ten)
    ve_trai = 1.0 / (2 * float(f(0.0)))
    sg = np.sqrt(var)
    ts = var / ve_trai ** 2
    print("%-12s %-18.10f %-18s %-14s %-22s"
          % (ten, ve_trai, ("vo cuc" if not np.isfinite(sg) else "%.10f" % sg),
             "co" if ve_trai < sg else "KHONG",
             ("vo cuc" if not np.isfinite(ts) else "%.10f" % ts)))
print()
print("Chuan la phan phoi DUY NHAT trong bon cai tren ma LAE thua — va no thua dung")
print("he so %.10f, tuc mat %.4f %% hieu suat." %
      (1.0 / (2 * norm.pdf(0)) ** 2, 100 * ((1.0 / (2 * norm.pdf(0)) ** 2) - 1)))


# ---------------------------------------------------- 14. tr.46 — cau in nghieng
tieu_de("14. TR.46 — cau in nghieng, phat bieu chinh thuc dau tien theo loi tac gia")
print("'the regression median is seen to be more efficient than the least squares estimator")
print(" in the linear model for any distribution for which the median is more efficient than")
print(" the mean in the location model'")
print()
print("Ly do ngan: hiep phuong sai cua trung vi hoi quy la [2f(0)]^{-2} Q^{-1} (tr.46),")
print("cua binh phuong toi thieu la sigma^2 Q^{-1}. Q^{-1} la CHUNG, nen no TRIET TIEU:")
print("ti so hieu suat trong mo hinh tuyen tinh bang dung ti so trong mo hinh vi tri,")
print("KHONG phu thuoc ma tran thiet ke.")
print()
print("Kiem bang so: bon ma tran thiet ke rat khac nhau, cung mot phan phoi nhieu Laplace.")
rng = np.random.default_rng(46046)
Tmc, R2 = 300, 800
print("%-34s %-16s %-18s %-18s %-14s" %
      ("thiet ke", "so dieu kien Q", "T*Var(trung vi)", "T*Var(OLS)", "ti so"))
TI_SO, DIEU_KIEN = [], []
for ten_tk, lam_X in (
        ("cot 1 + chuan doc lap", lambda: np.c_[np.ones(Tmc), rng.normal(size=(Tmc, 2))]),
        ("cot 1 + deu [0,10]", lambda: np.c_[np.ones(Tmc), rng.uniform(0, 10, (Tmc, 2))]),
        ("cot 1 + tuong quan 0.95", lambda: (lambda z: np.c_[np.ones(Tmc), z[:, 0],
                                                             0.95 * z[:, 0] + 0.3122 * z[:, 1]])(
            rng.normal(size=(Tmc, 2)))),
        ("cot 1 + thang do lech 100 lan", lambda: np.c_[np.ones(Tmc), rng.normal(size=Tmc),
                                                        100 * rng.normal(size=Tmc)])):
    Xd = lam_X()
    Qd = Xd.T @ Xd / Tmc
    bt = np.ones(Xd.shape[1])
    bm, bo = [], []
    for _ in range(R2):
        yd = Xd @ bt + rng.laplace(size=Tmc)
        bm.append(giai_lp(yd, Xd, 0.5)[0])
        bo.append(np.linalg.lstsq(Xd, yd, rcond=None)[0])
    vm = Tmc * float(np.var(np.array(bm)[:, 1]))
    vo = Tmc * float(np.var(np.array(bo)[:, 1]))
    TI_SO.append(vo / vm)
    DIEU_KIEN.append(float(np.linalg.cond(Qd)))
    print("%-34s %-16.4f %-18.10f %-18.10f %-14.10f"
          % (ten_tk, DIEU_KIEN[-1], vm, vo, vo / vm))
print()
print("Ti so ly thuyet trong mo hinh VI TRI voi Laplace: sigma^2 / [2f(0)]^{-2} = %.10f"
      % (2.0 / 1.0))
print()
print("Bon ti so do duoc deu DUOI 2. Do KHONG phai chenh giua hai mo hinh — do la thieu")
print("hut HUU HAN MAU cua chinh trung vi. Phep so dung la so voi MO HINH VI TRI o CUNG")
print("mot T, vi Dinh ly 4.2 noi hai ti so bang nhau, chu khong noi ti so bang 2 o T huu han.")
print()
print("%-10s %-22s %-22s %-16s" % ("T", "vi tri: T*Var(t.vi)", "vi tri: T*Var(t.binh)", "ti so"))
TS_VITRI = {}
for Tv, Rv in ((100, 200000), (300, 200000), (1000, 100000), (10000, 20000)):
    mv = rng.laplace(size=(Rv, Tv))
    mv.sort(axis=1)
    tv = pv_mau_don(mv, 0.5)
    tb = mv.mean(axis=1)
    a_, b_ = Tv * float(np.var(tv)), Tv * float(np.var(tb))
    TS_VITRI[Tv] = b_ / a_
    print("%-10d %-22.10f %-22.10f %-16.10f" % (Tv, a_, b_, b_ / a_))
print("Ti so trong mo hinh vi tri tien den 2 khi T tang: %s"
      % " -> ".join("%.4f" % TS_VITRI[t] for t in (100, 300, 1000, 10000)))
print()
print("SO SANH CHAT: o T = 300, mo hinh VI TRI cho ti so %.10f;" % TS_VITRI[300])
print("bon thiet ke cua mo hinh TUYEN TINH cho [%.10f ; %.10f]."
      % (min(TI_SO), max(TI_SO)))
print("Khoang cua mo hinh tuyen tinh %s gia tri cua mo hinh vi tri."
      % ("CHUA" if min(TI_SO) <= TS_VITRI[300] <= max(TI_SO) else "KHONG chua"))
print()
print("HAI CAU HOI PHAI TACH RA, va ca hai deu do duoc.")
print()
print("(a) Do rong 0.1728 giua bon THIET KE co lon hon nhieu Monte-Carlo khong?")
print("    Chay LAI CUNG MOT thiet ke voi bon hat giong khac nhau:")
DK_HAT = []
for hg in range(4):
    rg2 = np.random.default_rng(900 + hg)
    Xv = np.c_[np.ones(300), rg2.normal(size=(300, 2))]
    bv = np.ones(3)
    bm, bo = [], []
    for _ in range(R2):
        yv = Xv @ bv + rg2.laplace(size=300)
        bm.append(giai_lp(yv, Xv, 0.5)[0])
        bo.append(np.linalg.lstsq(Xv, yv, rcond=None)[0])
    DK_HAT.append((300 * float(np.var(np.array(bo)[:, 1])))
                  / (300 * float(np.var(np.array(bm)[:, 1]))))
    print("      hat giong %d : ti so %.10f" % (hg, DK_HAT[-1]))
print("    do rong theo HAT GIONG  : %.10f" % (max(DK_HAT) - min(DK_HAT)))
print("    do rong theo THIET KE   : %.10f" % (max(TI_SO) - min(TI_SO)))
print("    -> %s"
      % ("hai do rong cung co: khong ket luan duoc rang thiet ke co anh huong"
         if (max(TI_SO) - min(TI_SO)) < 2 * (max(DK_HAT) - min(DK_HAT))
         else "do rong theo thiet ke LON hon han — thiet ke CO anh huong"))
print()
print("(b) Vi sao bon ti so tuyen tinh deu THAP hon mo hinh vi tri o cung T = 300?")
print("    Gia thuyet: so THAM SO K an bot bac tu do, va hieu ung do co bac K/T nen o")
print("    T nho no phai RO HON. Do o T = 150, moi K hai hat giong de co do rong.")
TV, RV = 150, 2000
print("%-6s %-10s %-20s %-20s %-16s" % ("K", "hat giong", "T*Var(trung vi)", "T*Var(OLS)", "ti so"))
TS_K = {}
for Kv in (1, 2, 3, 6, 12):
    TS_K[Kv] = []
    for hg in (0, 1):
        rg2 = np.random.default_rng(2000 + 100 * Kv + hg)
        Xv = (np.c_[np.ones(TV), rg2.normal(size=(TV, Kv - 1))] if Kv > 1
              else np.ones((TV, 1)))
        bv = np.ones(Kv)
        bm, bo = [], []
        for _ in range(RV):
            yv = Xv @ bv + rg2.laplace(size=TV)
            bm.append(giai_lp(yv, Xv, 0.5)[0])
            bo.append(np.linalg.lstsq(Xv, yv, rcond=None)[0])
        vm = TV * float(np.var(np.array(bm)[:, 0]))
        vo = TV * float(np.var(np.array(bo)[:, 0]))
        TS_K[Kv].append(vo / vm)
        print("%-6d %-10d %-20.10f %-20.10f %-16.10f" % (Kv, hg, vm, vo, vo / vm))
print()
print("%-6s %-18s %-18s" % ("K", "ti so trung binh", "do rong hai hat giong"))
for Kv in (1, 2, 3, 6, 12):
    print("%-6d %-18.10f %-18.10f"
          % (Kv, float(np.mean(TS_K[Kv])), max(TS_K[Kv]) - min(TS_K[Kv])))
_tb = {k: float(np.mean(v)) for k, v in TS_K.items()}
_rong = max(max(v) - min(v) for v in TS_K.values())
print()
print("    Chenh giua K = 1 va K = 12 : %.10f" % (_tb[1] - _tb[12]))
print("    Do rong hat giong LON NHAT : %.10f" % _rong)
print("    -> %s"
      % ("hieu ung K DO DUOC: no lon hon nhieu hat giong"
         if (_tb[1] - _tb[12]) > 2 * _rong else
         "KHONG ket luan duoc: chenh theo K khong lon hon nhieu hat giong"))
print("    Ti so giam don dieu theo K o %d/4 buoc lien tiep."
      % sum(1 for a, b in zip((1, 2, 3, 6), (2, 3, 6, 12)) if _tb[a] > _tb[b]))
print("    K = 1 CHINH LA mo hinh vi tri; no cho %.10f o T = %d." % (_tb[1], TV))
print("    Dinh ly 4.2 la phat bieu TIEM CAN — no khong noi gi ve T huu han, nen")
print("    chenh lech nay khong phai bang chung chong lai no.")
print("Bon ti so DO DUOC: khoang [%.10f ; %.10f], do rong %.10f"
      % (min(TI_SO), max(TI_SO), max(TI_SO) - min(TI_SO)))
print("So dieu kien cua Q chay tu %.4f toi %.4f — gap %.4f lan — ma ti so van"
      % (min(DIEU_KIEN), max(DIEU_KIEN), max(DIEU_KIEN) / min(DIEU_KIEN)))
print("nam trong mot khoang rong %.10f. Do la Q^{-1} triet tieu, do bang so." %
      (max(TI_SO) - min(TI_SO)))


# ---------------------------------------------------- 15. tr.47 — binh phuong toi thieu cat duoi
tieu_de("15. TR.47 — binh phuong toi thieu CAT DUOI, dung hai phan vi hoi quy")
print("'Let alpha be the desired trimming proportion. Calculate beta*(alpha) and")
print(" beta*(1-alpha) for the sample. By Theorem 3.4 there will be approximately T*alpha")
print(" observations below the former and above the latter. These observations are set")
print(" aside, and the remaining observations are subjected to a least squares fit.'")
print()
print("Day la de xuat thuat toan chinh cua muc 5 bai bao — va no chi ton THEM HAI bai LP.")
print()
rng = np.random.default_rng(47)
Tmc, R3 = 200, 400
Xd = np.c_[np.ones(Tmc), rng.uniform(0, 10, Tmc)]
bt = np.array([1.0, 0.5])
print("%-16s %-20s %-20s %-20s %-14s" %
      ("nhieu", "OLS: T*Var(he so)", "cat 10%: T*Var", "cat 25%: T*Var", "loi/hai OLS"))
for ten_nh, sinh_nh in (("Chuan", lambda n: rng.normal(size=n)),
                        ("Laplace", lambda n: rng.laplace(size=n)),
                        ("10% 10sigma", lambda n: rng.normal(size=n) *
                         np.where(rng.random(n) < 0.1, 10.0, 1.0))):
    ket = {0.0: [], 0.10: [], 0.25: []}
    for _ in range(R3):
        yd = Xd @ bt + sinh_nh(Tmc)
        ket[0.0].append(np.linalg.lstsq(Xd, yd, rcond=None)[0])
        for al in (0.10, 0.25):
            blo = giai_lp(yd, Xd, al)[0]
            bhi = giai_lp(yd, Xd, 1 - al)[0]
            giu = (yd >= Xd @ blo - 1e-9) & (yd <= Xd @ bhi + 1e-9)
            ket[al].append(np.linalg.lstsq(Xd[giu], yd[giu], rcond=None)[0])
    v = {a: Tmc * float(np.var(np.array(ket[a])[:, 1])) for a in ket}
    print("%-16s %-20.10f %-20.10f %-20.10f %-14.6f"
          % (ten_nh, v[0.0], v[0.10], v[0.25], v[0.0] / min(v[0.10], v[0.25])))
print()
print("Cot cuoi la ti so 'phuong sai OLS / phuong sai ban cat tot nhat'. Lon hon 1 nghia la")
print("cat co loi; nho hon 1 nghia la cat phai TRA GIA. Do la chu 'insurance' o muc 5 tr.48.")


# ---------------------------------------------------- 16. cau chu — muc 4 den 6
tieu_de("16. TR.37 — 'In Sections 4 through 6 below' nhung bai bao chi co NAM muc")
print("Danh sach muc danh so thuc te cua bai bao, doc tu ban in:")
MUC_BAI_BAO = [
    ("1", "INTRODUCTION", 33), ("2", "BACKGROUND AND MOTIVATION", 34),
    ("3", "REGRESSION QUANTILES", 38),
    ("4", "THE ASYMPTOTIC DISTRIBUTION THEORY OF REGRESSION QUANTILES", 42),
    ("5", "CONCLUSION", 48),
]
for so, ten, tr in MUC_BAI_BAO:
    print("   muc %s  tr.%-4d %s" % (so, tr, ten))
print("   PHU LUC   tr.48  COMPUTATIONAL ASPECTS OF REGRESSION QUANTILES")
print("   THAM KHAO tr.49  43 muc")
print()
print("So muc danh so: %d. Cau tr.37 hua 'Sections 4 through 6' — muc 6 KHONG TON TAI."
      % len(MUC_BAI_BAO))
print("Ban thao nhan thang 2/1975, ban sua nhan thang 1/1977 (tr.48) — hai nam giua hai moc.")


# ---------------------------------------------------- 17. cau noi sang thuc hanh hom nay
tieu_de("17. NOI SANG THUC HANH HOM NAY — doi chieu voi scikit-learn")
if CO_SKLEARN:
    rng = np.random.default_rng(430430)
    Tmc = 400
    xq = np.sort(rng.uniform(0, 10, Tmc))
    yq = 1 + 0.5 * xq + (0.3 + 0.3 * xq) * rng.normal(size=Tmc)
    Xq2 = xq[:, None]
    Xa = np.c_[np.ones(Tmc), xq]
    print("Du lieu: T = %d, y = 1 + 0.5x + (0.3 + 0.3x)*N(0,1) — phuong sai TANG theo x." % Tmc)
    print()
    print("%-8s %-16s %-16s %-14s %-12s %-10s %-10s" %
          ("theta", "beta*_1", "beta*_2", "lech sklearn", "ti le duoi", "N", "Z"))
    ARR_TH, ARR_B0, ARR_B1, ARR_DUOI = [], [], [], []
    for th in (0.10, 0.25, 0.50, 0.75, 0.90):
        b, _, _, _ = giai_lp(yq, Xa, th)
        m = QuantileRegressor(quantile=th, alpha=0, solver="highs").fit(Xq2, yq)
        L = max(abs(b[0] - m.intercept_), abs(b[1] - m.coef_[0]))
        u = yq - Xa @ b
        duoi = float(np.mean(u < -1e-9))
        ARR_TH.append(th); ARR_B0.append(b[0]); ARR_B1.append(b[1]); ARR_DUOI.append(duoi)
        print("%-8.2f %-16.10f %-16.10f %-14.2e %-12.6f %-10d %-10d"
              % (th, b[0], b[1], L, duoi, int(np.sum(u < -1e-9)),
                 int(np.sum(np.abs(u) <= 1e-9))))
    print()
    print("He so goc di tu %.10f (theta = 0.10) toi %.10f (theta = 0.90), tuc gap %.10f lan."
          % (ARR_B1[0], ARR_B1[-1], ARR_B1[-1] / ARR_B1[0]))
    ols = np.linalg.lstsq(Xa, yq, rcond=None)[0]
    print("OLS tren cung du lieu: he so chan %.10f, he so goc %.10f — DUNG MOT duong."
          % (ols[0], ols[1]))
    print()
    print("Do phu: khoang [beta*(0.05), beta*(0.95)] so voi khoang OLS +- 1.96 s.")
    b05 = giai_lp(yq, Xa, 0.05)[0]
    b95 = giai_lp(yq, Xa, 0.95)[0]
    s = float(np.std(yq - Xa @ ols, ddof=Xa.shape[1]))
    trong_pv = (yq >= Xa @ b05) & (yq <= Xa @ b95)
    trong_ols = np.abs(yq - Xa @ ols) <= 1.96 * s
    print("   toan bo du lieu : phan vi %.6f   OLS %.6f" %
          (float(np.mean(trong_pv)), float(np.mean(trong_ols))))
    print()
    print("%-14s %-16s %-16s %-10s" % ("khuc x", "phan vi 5-95%", "OLS +- 1.96 s", "so diem"))
    for lo in range(0, 10, 2):
        m_ = (xq >= lo) & (xq < lo + 2)
        print("%-14s %-16.6f %-16.6f %-10d"
              % ("[%d, %d)" % (lo, lo + 2), float(np.mean(trong_pv[m_])),
                 float(np.mean(trong_ols[m_])), int(m_.sum())))
    print()
    print("Chenh lech giua o te nhat va o tot nhat cua tung cach:")
    pv_khuc = [float(np.mean(trong_pv[(xq >= lo) & (xq < lo + 2)])) for lo in range(0, 10, 2)]
    ols_khuc = [float(np.mean(trong_ols[(xq >= lo) & (xq < lo + 2)])) for lo in range(0, 10, 2)]
    print("   phan vi : %.6f  (tu %.6f toi %.6f)"
          % (max(pv_khuc) - min(pv_khuc), min(pv_khuc), max(pv_khuc)))
    print("   OLS     : %.6f  (tu %.6f toi %.6f)"
          % (max(ols_khuc) - min(ols_khuc), min(ols_khuc), max(ols_khuc)))
    print("   ti so hai do rong: %.10f lan" % ((max(ols_khuc) - min(ols_khuc)) /
                                               (max(pv_khuc) - min(pv_khuc))))
    print()
    print("MANG CHO KHOI MA SINH HINH (day thang, khong go tay):")
    print("X_HINH = %r" % [round(float(v), 10) for v in xq[:8]])
    print("THETA  = %r" % ARR_TH)
    print("B0     = %r" % [round(float(v), 10) for v in ARR_B0])
    print("B1     = %r" % [round(float(v), 10) for v in ARR_B1])
    print("PHU_PV = %r" % [round(v, 10) for v in pv_khuc])
    print("PHU_OLS= %r" % [round(v, 10) for v in ols_khuc])
    print("BII_X  = %r" % [float(p[0]) for p in PTS_SX])
    print("BII_Y  = %r" % [float(p[1]) for p in PTS_SX])
    print("BII_B0 = %r" % [float(b[0]) for b in BANG_II])
    print("BII_B1 = %r" % [float(b[1]) for b in BANG_II])
    print("BII_MOC= %r" % [float(m) for m in MOC_II])
else:
    print("bo qua — khong co scikit-learn")


# ---------------------------------------------------- 18. ham pinball, cach viet hien dai
tieu_de("18. HAM MAT MAT — cach viet cua bai bao va cach viet hien dai la MOT")
print("Bai bao tr.38 viet bang HAI TONG tach theo dau phan du;")
print("cach viet hien dai la rho_theta(u) = u(theta - 1[u < 0]). Kiem chung tren luoi:")
u = np.linspace(-5, 5, 100001)
for th in (0.1, 0.5, 0.9):
    hai_tong = np.where(u >= 0, th * np.abs(u), (1 - th) * np.abs(u))
    mot_dong = u * (th - (u < 0))
    print("   theta = %.1f : lech toi da tren %d diem = %.3e"
          % (th, len(u), float(np.max(np.abs(hai_tong - mot_dong)))))
    assert np.max(np.abs(hai_tong - mot_dong)) < 1e-12
print()
print("Voi theta = 1/2 thi rho = |u|/2, tuc uoc luong LAE — 'the regression quantile for")
print("theta = 1/2' (tr.38). Gia tri cu the:")
for v in (-4.0, -1.0, 0.0, 1.0, 4.0):
    print("   rho_0.10(%5.1f) = %10.6f   rho_0.50(%5.1f) = %10.6f   rho_0.90(%5.1f) = %10.6f"
          % (v, float(rho(v, 0.1)), v, float(rho(v, 0.5)), v, float(rho(v, 0.9))))


# ---------------------------------------------------- 18b. gia phai tra: thoi gian
tieu_de("18b. GIA PHAI TRA — do thoi gian binh phuong toi thieu so voi quy hoach tuyen tinh")
print("Con so nay KHONG tai lap duoc giua cac may va cac lan chay; day la MOT lan chay,")
print("va bai viet phai ghi ro nhu the (luat ve cot thoi gian).")
print()
print("%-14s %-8s %-16s %-16s %-16s %-12s" %
      ("T", "K", "OLS (ms)", "1 phan vi (ms)", "9 phan vi (ms)", "9 pv / OLS"))
rng = np.random.default_rng(1818)
for Tn, K in ((200, 3), (1000, 3), (5000, 3), (1000, 10)):
    X = np.c_[np.ones(Tn), rng.normal(size=(Tn, K - 1))]
    y = X @ np.ones(K) + rng.normal(size=Tn)
    t0 = time.time()
    for _ in range(5):
        np.linalg.lstsq(X, y, rcond=None)
    t_ols = (time.time() - t0) / 5 * 1000
    t0 = time.time()
    giai_lp(y, X, 0.5)
    t_1 = (time.time() - t0) * 1000
    t0 = time.time()
    for th in np.arange(0.1, 0.95, 0.1):
        giai_lp(y, X, float(th))
    t_9 = (time.time() - t0) * 1000
    print("%-14d %-8d %-16.4f %-16.4f %-16.4f %-12.2f"
          % (Tn, K, t_ols, t_1, t_9, t_9 / t_ols))
print()
print("Cot cuoi la con so dang nho: muon CA MOT DAI chin muc phan vi thi phai tra bao")
print("nhieu lan so voi mot lan binh phuong toi thieu.")


# ---------------------------------------------------- 19. kiem ke bai bao
tieu_de("19. KIEM KE BAI BAO — nhung con so bai viet trich o bang doi chieu")
import glob as _glob
_thu_muc = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "AI algorithms", "Machine Learning", "Quantile Regression")
_khung = sorted(_glob.glob(os.path.join(_thu_muc, "paper_kb1978_*.png")))
print("So khung da cat tu ban in (dem file, KHONG go tay): %d" % len(_khung))
_pdf = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "AI algorithms", "_papers",
                    "1978 - Koenker, Bassett - Regression Quantiles.pdf")
if os.path.exists(_pdf):
    import pypdfium2 as _pdfium
    _d = _pdfium.PdfDocument(_pdf)
    _dem = [len(re.sub(r"\s", "", _d[i].get_textpage().get_text_range())) for i in range(len(_d))]
    print("So trang PDF                                      : %d" % len(_dem))
    print("So trang than bai (tr.33-50)                      : %d" % 18)
    print("So trang than bai co lop chu doc duoc             : %d"
          % sum(1 for c in _dem[1:19] if c > 0))
    print("So ky tu trung binh moi trang than bai            : %.1f"
          % (sum(_dem[1:19]) / 18.0))
    print("So trang CO lop chu (bia + phu luc dan lien ket)  : %d" % sum(1 for c in _dem if c > 0))
    print("Mat do ky tu trung binh CA FILE — con so danh bai")
    print("  phep do cu cua kiemtrich.py                     : %.1f" % (sum(_dem) / len(_dem)))
else:
    print("(khong co PDF tren may nay — xem muc 0.5 cua bai)")
print()
print("Nhung con so sau DOC TAY tu ban in, vi bai bao la ban quet nen khong cong cu nao")
print("trong tools/ dem ho duoc. Chung duoc khai o day de bai viet co mot nguon:")
print("   so HINH VE cua bai bao                 : 0")
print("   so BANG so cua bai bao                 : 2")
print("   so muc DANH SO                         : 5")
print("   so phu luc                             : 1")
print("   so DINH LY                             : 8")
print("   so cong thuc DANH SO                   : 22")
print("   so nhan cong thuc khong danh so (P/D/D'): 3")
print("   so muc tai lieu tham khao              : 43")
print("   so o cua Bang I                        : 30")
print("   so o cua Bang II                       : 8")


print()
print("=" * 92)
print("XONG — %.1f giay" % (time.time() - T0))
print("=" * 92)
