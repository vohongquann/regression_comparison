# -*- coding: utf-8 -*-
"""Mọi con số của bài Linear Regression — dựng lại từ NGUỒN GỐC và từ ví dụ trong bài.

Luật 6 (CLAUDE.md 2.0) đòi phần ruột theo nguồn gốc. Với hồi quy tuyến tính, nguồn gốc
không phải một bài báo có hình mà là hai bản in, cộng một bài báo lịch sử làm trọng tài:

* Legendre 1805 — mục LVII–LVIII (sao chổi 1769) và phụ lục "Sur la méthode des moindres
  quarrés", gồm cả "Application à la mesure des degrés du méridien" với bốn hệ đánh số
  (a)–(d) và một lần giải lại khi GHIM độ dẹt.
* Gauss 1809, Theoria Motus, quyển II mục III, điều 172–187 — luật sai số (177), nguyên lý
  hợp lý cực đại (179), phương trình chuẩn (180), độ chính xác (182–183), ví dụ số (184),
  và so sánh với nguyên lý của Laplace/Boscovich (186).
* Stigler 1981 — dựng lại chính bốn cung kinh tuyến Pháp bằng công thức (3.1).

Mỗi mục in con số của NGUỒN cạnh con số DỰNG LẠI. Không con số nào của bài được gõ tay
ngoài đầu ra của script này.

Ba lỗi script này đã bắt được, giữ lại để bài kể:
1. Bản đầu cho phương trình "praecisio dimidia" của Gauss trọng số 1/2 — sai. Độ chính xác
   h tỉ lệ 1/sigma, nên một nửa h là trọng số 1/4, đúng bằng việc chia phương trình cho 2.
2. Bản đầu đọc độ dẹt ghim của Legendre là 1/334 từ ảnh độ phân giải thấp. Bản in ghi 1/320,
   và hằng số của hệ (e) chỉ khớp với 1/320.
3. Con số Vandermonde bậc 15 của bài cũ (17,67 · 4,238e-6 · 1,38e18) không tái lập được trên
   máy này — kappa(XtX) ~ 1e18 vượt độ phân giải float64, chữ số phụ thuộc BLAS và CPU.

Chạy:  cd tools && python v_linreg.py
"""
import math
import platform
import sys
from itertools import combinations

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")


def head(s):
    print("\n" + "=" * 78 + "\n" + s + "\n" + "=" * 78)


def dms(d, m, s):
    return d + m / 60 + s / 3600


def du_lieu_25():
    rng = np.random.default_rng(3)
    n = 25
    x = np.sort(rng.uniform(0, 10, n))
    y = 2.0 + 1.3 * x + rng.normal(0, 2.0, n)
    return x, y


print("numpy %s · %s · %s" % (np.__version__, platform.platform(), platform.processor()))

# ════════════════════════════════════════════════════════════ 1. bốn chiếc máy
head("1. Ví dụ bốn chiếc máy — phép tính tay của bài")
x = np.array([1.0, 2.0, 3.0, 4.0])
y = np.array([3.0, 4.0, 5.0, 8.0])
X = np.c_[np.ones(4), x]
XtX, Xty = X.T @ X, X.T @ y
b = np.linalg.solve(XtX, Xty)
e = y - X @ b
H = X @ np.linalg.solve(XtX, X.T)
rss, tss = e @ e, ((y - y.mean()) ** 2).sum()
s2 = rss / (4 - 2)
inv = np.linalg.inv(XtX)
se1 = math.sqrt(s2 * inv[1, 1])
print("XtX =", XtX.tolist(), " Xty =", Xty.tolist(), " det = %.6f" % np.linalg.det(XtX))
print("inv(XtX) =", inv.round(6).tolist())
print("beta = %.6f %.6f" % tuple(b))
print("yhat =", (X @ b).round(6).tolist())
print("e    =", e.round(6).tolist())
print("h_ii =", np.diag(H).round(6).tolist(), " tong = %.6f" % np.trace(H))
print("sum e = %.2e   sum e*x = %.2e" % (e.sum(), e @ x))
print("RSS = %.6f  TSS = %.6f  R2 = %.6f  sigma2 = %.6f" % (rss, tss, 1 - rss / tss, s2))
print("SE(b1) = %.6f  t1 = %.6f" % (se1, b[1] / se1))
print("du doan tai x=2.5: %.6f" % (np.array([1, 2.5]) @ b))
for bb in ([1.0, 1.6], [0.0, 2.0], [1.5, 1.4]):
    ee = y - X @ bb
    print("   duong b0=%.1f b1=%.1f: tong binh phuong phan du = %.6f" % (bb[0], bb[1], ee @ ee))
assert np.allclose(b, [1, 1.6]) and abs(np.trace(H) - 2) < 1e-12

# ════════════════════════════════════════ 2. Legendre 1805, mục LVII–LVIII tr.64
head("2. Legendre 1805 mục LVIII tr.64 (PDF 76) — sao chổi 1769, hai ẩn z, u")
a = np.array([2.248, 0.006, 1.418])
A = np.array([[1.499, 2.996], [0.059, 3.040], [4.780, -1.167]])
N, r = A.T @ A, A.T @ a
th = np.linalg.solve(N, -r)
E = a + A @ th
print("in:       0 = 10.148 + z(25.099) - u(0.908)   |  0 = 5.098 - z(0.908) + u(19.580)")
print("dung lai: 0 = %.3f + z(%.3f) %+.3f u  |  0 = %.3f %+.3f z + u(%.3f)"
      % (r[0], N[0, 0], N[0, 1], r[1], N[1, 0], N[1, 1]))
print("in:       z = -0.4144  u = -0.2796")
print("dung lai: z = %.4f  u = %.4f   (du chu so: %.6f %.6f)" % (th[0], th[1], th[0], th[1]))
print("in tr.65: E' = 0.789  E'' = -0.868  E''' = -0.236")
print("dung lai: E' = %.5f  E'' = %.5f  E''' = %.5f" % tuple(E))
print("RSS truoc (z=u=0) = %.4f   sau = %.4f   giam %.4f lan" % (a @ a, E @ E, (a @ a) / (E @ E)))
print("|E| lon nhat truoc = %.3f phut cung (in: 's'environ 2 1/4'), sau = %.3f (in: 'ne va pas a une minute')"
      % (np.abs(a).max(), np.abs(E).max()))
lst = np.linalg.lstsq(A, -a, rcond=None)[0]
print("lstsq lech so voi giai phuong trinh chuan: %.2e" % np.abs(lst - th).max())
print("cat 3 chu so E''' = %.3f   lam tron = %.3f" % (math.trunc(E[2] * 1000) / 1000, round(E[2], 3)))

# ═══════════════════════════════ 3. Legendre 1805 phụ lục — kinh tuyến tr.76–79
head("3. Legendre 1805 phụ lục tr.76–79 (PDF 88–91) — đo cung kinh tuyến Pháp")
ten = ["Dunkerque", "Pantheon", "Evaux", "Carcassonne", "Montjouy"]
lat = [dms(51, 2, 10.50), dms(48, 50, 49.75), dms(46, 10, 42.50),
       dms(43, 12, 54.40), dms(41, 21, 44.80)]
S = [62472.59, 76145.74, 84424.55, 52749.48]
in_dL = ["2 11 20.75", "2 40 7.25", "2 57 48.10", "1 51 9.60"]
in_sL = ["99 53 0", "95 1 32", "89 23 37", "84 34 39"]
in_a = [(0.002923, 2.192, -0.563), (0.003100, 2.672, -0.351),
        (-0.001096, 2.962, 0.047), (-0.001808, 1.851, 0.263)]
print("cong thuc (a): L'-L = S/28500 + c*S/28500 + a*(270/pi)*sin(L'-L)*cos(L'+L)")
print("  hang so (a) = S/28500 - (L'-L) quan sat ; he so c = S/28500 ; he so a = (270/pi) sin cos")
print("%-24s %10s %9s %9s | %10s %9s %9s %9s" % ("cung", "hang(in)", "c(in)", "a(in)",
                                                "hang", "c", "a[3/2]", "a[1/2]"))
coef = []
for k in range(4):
    dL, sL = lat[k] - lat[k + 1], lat[k] + lat[k + 1]
    c0, cz = S[k] / 28500 - dL, S[k] / 28500
    k32 = 270 / math.pi * math.sin(math.radians(dL)) * math.cos(math.radians(sL))
    k12 = 90 / math.pi * math.sin(math.radians(dL)) * math.cos(math.radians(sL))
    coef.append((c0, cz, k32))
    print("%-24s %10.6f %9.3f %9.3f | %10.6f %9.4f %9.4f %9.4f"
          % (ten[k][:11] + "-" + ten[k + 1][:11], *in_a[k], c0, cz, k32, k12))
    print("   L'-L in %s = %.6f do ; L'+L in %s = %.6f do" % (in_dL[k], dL, in_sL[k], sL))
C = np.array(coef)
d = np.zeros((5, 3))
d[1] = C[1]
d[0] = C[0] + C[1]
d[3] = -C[2]
d[4] = -C[2] - C[3]
in_b = [(0.006023, 4.864, -0.914), (0.003100, 2.672, -0.351), (0, 0, 0),
        (0.001096, -2.962, -0.047), (0.002904, -4.813, -0.310)]
nhan = ["E'", "E''", "E'''", "Eiv", "Ev"]
print("\nhe (b) — moi sai so viet qua E''' (in | dung lai):")
for j in range(5):
    print("   %-4s = E''' %+.6f %+.3f c %+.3f a  |  %+.6f %+.4f c %+.4f a" % (nhan[j], *in_b[j], *d[j]))
m = d.sum(0) / 5
print("cuc tieu theo E''': 0 = 5E''' %+.6f %+.4f c %+.4f a   (in: 0 = 5E''' + 0.013123 - c(0.239) - a(1.622))"
      % tuple(5 * m))
print("   => E''' = %+.6f %+.4f c %+.4f a   (in: -0.002625 + c(0.048) + a(0.324))" % tuple(-m))
ce = d - m
in_c = [(0.003398, 4.912, -0.590), (0.000475, 2.720, -0.027), (-0.002625, 0.048, 0.324),
        (-0.001529, -2.914, 0.277), (0.000279, -4.765, 0.014)]
print("he (c) (in | dung lai):")
for j in range(5):
    print("   %-4s = %+.6f %+.3f c %+.3f a  |  %+.6f %+.4f c %+.4f a" % (nhan[j], *in_c[j], *ce[j]))
M, v = ce[:, 1:], ce[:, 0]
Nd, rd = M.T @ M, M.T @ v
print("he (d) in:       0 = 0.020983 + c(62.726) - a(3.830) ;  0 = -0.003287 - c(3.830) + a(0.531)")
print("he (d) dung lai: 0 = %+.6f + c(%.4f) %+.4f a ;  0 = %+.6f %+.4f c + a(%.4f)"
      % (rd[0], Nd[0, 0], Nd[0, 1], rd[1], Nd[1, 0], Nd[1, 1]))
zc, al = np.linalg.solve(Nd, -rd)
zp, ap = np.linalg.solve(np.array([[62.726, -3.830], [-3.830, 0.531]]), -np.array([0.020983, -0.003287]))
print("in tr.78:            a = 0.00675   c = 0.0000778   do det 1/148   D = 28497.78")
print("giai he (d) SO IN:   a = %.6f  c = %.8f  do det 1/%.2f  D = %.4f" % (ap, zp, 1 / ap, 28500 / (1 + zp)))
print("dung lai tu bang:    a = %.6f  c = %.8f  do det 1/%.2f  D = %.4f" % (al, zc, 1 / al, 28500 / (1 + zc)))
Ein = np.array([t[0] + t[1] * 0.0000778 + t[2] * 0.00675 for t in in_c])
Ed = v + M @ [zc, al]
print("sai so (giay) in tr.79:          -0.73  1.83  -1.55  0.42  0.03 ; trung binh |.| in 0.91")
print("sai so (giay) he (c) voi so in:  %s ; trung binh |.| %.4f"
      % (" ".join("%.4f" % (3600 * t) for t in Ein), np.abs(3600 * Ein).mean()))
print("sai so (giay) dung lai het:      %s ; trung binh |.| %.4f ; |max| %.4f"
      % (" ".join("%.4f" % (3600 * t) for t in Ed), np.abs(3600 * Ed).mean(), np.abs(3600 * Ed).max()))
print("trung binh |.| cua 5 so in: %.4f" % np.mean(np.abs([-0.73, 1.83, -1.55, 0.42, 0.03])))
print("\nghim do det (tr.79). Do det ngam dinh boi tung hang so cua he (e) da in:")
in_e = [0.001554, 0.000391, -0.001612, -0.000663, 0.000337]
for j in range(5):
    imp = (in_c[j][0] - in_e[j]) / -in_c[j][2]
    print("   %-4s: (c) %.6f -> (e) %.6f  => a = %.6f = 1/%.1f" % (nhan[j], in_c[j][0], in_e[j], imp, 1 / imp))
for den, nhan_den in ((320, "mau so IN o tr.79"), (334, "doi chung — so doc nham ban dau")):
    a0 = 1 / den
    zz = -(rd[0] + Nd[0, 1] * a0) / Nd[0, 0]
    Eg = v + M @ [zz, a0]
    print("   a = 1/%d (%s): c = %.7f  D = 28500(1-c) = %.2f ; giay %s ; |max| %.2f ; tong binh phuong %.2f"
          % (den, nhan_den, zz, 28500 * (1 - zz), " ".join("%.2f" % (3600 * t) for t in Eg),
             np.abs(3600 * Eg).max(), (3600 * Eg) @ (3600 * Eg)))
zp2 = -(0.009010) / 62.726
print("   in tr.79: 0 = 0.009010 + c(62.726) -> c = %.7f ; D = %.2f ; sai so in 3.06 0.00 -5.83 -0.88 3.62"
      % (zp2, 28500 * (1 - zp2)))
print("   tong binh phuong (giay^2) cua 5 so in khi ghim: %.4f ; khi toi uu ca hai: %.4f"
      % (np.sum(np.array([3.06, 0.00, -5.83, -0.88, 3.62]) ** 2), np.sum(np.array([-0.73, 1.83, -1.55, 0.42, 0.03]) ** 2)))

# ═══════════════════════════════ 3b. Stigler 1981 — cùng bốn cung, công thức (3.1)
head("3b. Stigler 1981 tr.467–470 — cùng bốn cung, công thức (3.1) a = z + y sin^2 L")
dS = [2.18910, 2.66868, 2.96336, 1.85266]
Lm = [dms(49, 56, 30), dms(47, 30, 46), dms(44, 41, 48), dms(42, 17, 20)]
print("Bang 1 Stigler d: %s ; dung lai tu vi do Legendre: %s"
      % (" ".join("%.5f" % t for t in dS),
         " ".join("%.5f" % (lat[k] - lat[k + 1]) for k in range(4))))
print("Bang 1 Stigler L giua: %s ; dung lai: %s"
      % (" ".join("%.4f" % t for t in Lm), " ".join("%.4f" % ((lat[k] + lat[k + 1]) / 2) for k in range(4))))
for nhan_s, S2 in (("so lieu da sua (76145.74)", 76145.74), ("so lieu in sai (76545.74)", 76545.74)):
    SS = [62472.59, S2, 84424.55, 52749.48]
    aa = np.array([SS[k] / dS[k] for k in range(4)])
    Z = np.c_[np.ones(4), np.sin(np.radians(Lm)) ** 2]
    z, yy = np.linalg.lstsq(Z, aa, rcond=None)[0]
    print("   %s: z = %.5f  y = %.7f  1/do det = 3z/y + 3/2 = %.2f  gon kinh tuyen = 90(z+y/2) = %.2f"
          % (nhan_s, z, yy, 3 * z / yy + 1.5, 90 * (z + yy / 2)))
print("   in tr.469-470: da sua z = 28227.16205 y = 541.2639353 -> 157.95, 2564801.46 ;"
      " in sai z = 28074.82697 y = 906.7901314 -> 94.38, 2567539.98")
print("   Gauss 1799 (Stigler tr.466): 1/187 va 2565006 ; Legendre 1805 cach viet sai so vi do: 1/%.2f" % (1 / al))

# ════════════════════════════════════ 4. Gauss 1809 điều 177 — tiên đề trung bình
head("4. Gauss 1809 điều 177–179 tr.211–213 — trung bình cộng => luật Gauss => bình phương tối thiểu")
rng = np.random.default_rng(0)
mau = rng.normal(5.0, 2.0, 11)
luoi = np.linspace(0, 10, 1001)
ll_g = np.array([-(((mau - t) ** 2).sum()) for t in luoi])
ll_l = np.array([-(np.abs(mau - t).sum()) for t in luoi])
print("11 quan sat: mean = %.6f  median = %.6f" % (mau.mean(), np.median(mau)))
print("argmax hop ly GAUSS (luoi 0.01)   = %.2f  -> trung binh cong" % luoi[ll_g.argmax()])
print("argmax hop ly LAPLACE (luoi 0.01) = %.2f  -> trung vi" % luoi[ll_l.argmax()])
tt = np.linspace(-6, 6, 120001)
for h in (1.0, 2.0):
    phi = h / math.sqrt(math.pi) * np.exp(-(h * tt) ** 2)
    trong = np.abs(tt) <= 0.5 / h
    print("h = %.1f: tich phan phi = %.6f ; sigma = 1/(h sqrt2) = %.6f ; P(|D| <= 0.5/h) = %.6f"
          % (h, np.trapezoid(phi, tt), 1 / (h * math.sqrt(2)), np.trapezoid(phi * trong, tt)))

# ══════════════════════════════════════════ 5. Gauss 1809 điều 184 — ví dụ số
head("5. Gauss 1809 điều 184 tr.219–220 — bốn phương trình, ba ẩn, một phương trình kém chính xác")
G = np.array([[1, -1, 2], [3, 2, -5], [4, 1, 4], [-2, 6, 6]], float)
g = np.array([3, 5, 21, 28], float)
for wt in (0.5, 0.25):
    Nw = G.T @ np.diag([1, 1, 1, wt]) @ G
    print("trong so %.2f cho phuong trinh 4: N = %s  rhs = %s"
          % (wt, Nw.tolist(), (G.T @ (np.array([1, 1, 1, wt]) * g)).tolist()))
w = np.array([1, 1, 1, 0.25])
Ng, rg = G.T @ np.diag(w) @ G, G.T @ (w * g)
print("in: P = 27p + 6q - 88 ; Q = 6p + 15q + r - 70 ; R = q + 54r - 107  (trong so 1/4 khop)")
sol = np.linalg.solve(Ng, rg)
print("in:       p = 2.470  q = 3.551  r = 1.916")
print("dung lai: p = %.6f  q = %.6f  r = %.6f" % tuple(sol))
print("dang phan so Gauss: 49154/19899 = %.6f ; 2617/737 = %.6f ; 76242/39798 = %.6f"
      % (49154 / 19899, 2617 / 737, 76242 / 39798))
iv = np.linalg.inv(Ng)
print("det N = %.3f ; N^-1 duong cheo = %.8f %.8f %.8f"
      % (np.linalg.det(Ng), *np.diag(iv)))
print("   phan so Gauss: 809/19899 = %.8f ; 54/737 = %.8f ; 491/13266 = %.8f"
      % (809 / 19899, 54 / 737, 491 / 13266))
print("in do chinh xac: p 4.96  q 3.69  r 5.20")
print("dung lai 1/sqrt(diag N^-1): %.6f %.6f %.6f" % tuple(1 / np.sqrt(np.diag(iv))))
Gw, gw = G * np.sqrt(w)[:, None], g * np.sqrt(w)
res = gw - Gw @ sol
print("phan du (sau khi chia phuong trinh 4 cho 2): %s ; tong binh phuong %.6f"
      % (" ".join("%.6f" % t for t in res), res @ res))
sol_u = np.linalg.lstsq(G, g, rcond=None)[0]
print("neu KHONG ha trong so phuong trinh 4: p = %.6f q = %.6f r = %.6f" % tuple(sol_u))

# ═══════════════════════════════════════════════ 6. 25 điểm, bốn bộ giải
head("6. Dữ liệu 25 điểm (seed 3) — bốn bộ giải cho cùng một nghiệm")
xx, yy = du_lieu_25()
n = len(xx)
XX = np.c_[np.ones(n), xx]
bn = np.linalg.solve(XX.T @ XX, XX.T @ yy)
Q, R = np.linalg.qr(XX)
bq = np.linalg.solve(R, Q.T @ yy)
U, sv, Vt = np.linalg.svd(XX, full_matrices=False)
bs = Vt.T @ ((1 / sv) * (U.T @ yy))
lr = 1.0 / np.linalg.eigvalsh(XX.T @ XX / n)[-1]
bg = np.zeros(2)
for _ in range(2000):
    bg -= lr * (XX.T @ (XX @ bg - yy)) / n
for tname, bv in (("phuong trinh chuan", bn), ("QR", bq), ("SVD", bs), ("ha gradient", bg)):
    print("   %-20s beta = [%.6f, %.6f]" % (tname, *bv))
print("   lech lon nhat so voi QR: %.3e" % max(np.abs(t - bq).max() for t in (bn, bs, bg)))
rr = yy - XX @ bq
rss2, tss2 = rr @ rr, ((yy - yy.mean()) ** 2).sum()
s22 = rss2 / (n - 2)
se = np.sqrt(np.diag(s22 * np.linalg.inv(XX.T @ XX)))
print("   R2 = %.4f  R2adj = %.4f  sigma = %.4f  sigma(chia n) = %.4f"
      % (1 - rss2 / tss2, 1 - (rss2 / (n - 2)) / (tss2 / (n - 1)), math.sqrt(s22), math.sqrt(rss2 / n)))
print("   SE = [%.4f, %.4f]  t = [%.3f, %.3f]" % (*se, *(bq / se)))
print("   khoang beta +- 1.96 SE chua he so that [2.0, 1.3]: %s"
      % bool(np.all(np.abs(bq - [2.0, 1.3]) < 1.96 * se)))
print("   chia n thay n-p: he so (n-p)/n = %.4f -> chech %.1f %%" % ((n - 2) / n, 100 * 2 / n))
print("   Sum e = %.2e  e.x = %.2e" % (rr.sum(), rr @ xx))

# ═══════════════════════════════════ 7. Gauss 1809 điều 186 — L2 so với L1
head("7. Gauss 1809 điều 186 tr.221 — nguyên lý Boscovich/Laplace khớp ĐÚNG số phương trình bằng số ẩn")
tot = None
for i, j in combinations(range(n), 2):
    bb = np.linalg.solve(XX[[i, j]], yy[[i, j]])
    val = np.abs(yy - XX @ bb).sum()
    if tot is None or val < tot[0]:
        tot = (val, bb)
r1 = yy - XX @ tot[1]
print("so cap diem phai do de tim nghiem L1: %d" % len(list(combinations(range(n), 2))))
print("L1 tu do (25 diem): beta = [%.6f, %.6f] ; sum|e| = %.6f ; so phan du = 0: %d (so an p = 2)"
      % (tot[1][0], tot[1][1], tot[0], int((np.abs(r1) < 1e-9).sum())))
print("   OLS: sum|e| = %.6f ; so phan du = 0: %d" % (np.abs(rr).sum(), int((np.abs(rr) < 1e-9).sum())))
xm, ym = xx.mean(), yy.mean()
cands = (yy - ym) / (xx - xm)
best = min(cands, key=lambda s: np.abs(yy - ym - s * (xx - xm)).sum())
rL = yy - ym - best * (xx - xm)
print("L1 + dieu kien Laplace (tong phan du = 0): beta = [%.6f, %.6f] ; sum|e| = %.6f ; tong e = %.2e ; so phan du = 0: %d"
      % (ym - best * xm, best, np.abs(rL).sum(), rL.sum(), int((np.abs(rL) < 1e-9).sum())))
print("   (Gauss: dieu kien them lam so phuong trinh khop dung it hon so an MOT don vi)")

# ════════════════════════════════════════════ 8. Vandermonde và điều kiện số
head("8. Số điều kiện — phương trình chuẩn bình phương kappa (chữ số PHỤ THUỘC MÁY ở bậc 15)")
xv = np.linspace(0, 1, 60)
for bac in (5, 9):
    V = np.vander(xv, bac + 1, increasing=True)
    k1, k2 = np.linalg.cond(V), np.linalg.cond(V.T @ V)
    print("   bac %d: k(X) = %.3e  k(XtX) = %.3e  k(X)^2 = %.3e" % (bac, k1, k2, k1 ** 2))
V = np.vander(xv, 16, increasing=True)
yv = V @ np.ones(16)
en = np.abs(np.linalg.solve(V.T @ V, V.T @ yv) - 1).max()
Qv, Rv = np.linalg.qr(V)
eq = np.abs(np.linalg.solve(Rv, Qv.T @ yv) - 1).max()
Uv, sv2, Vtv = np.linalg.svd(V, full_matrices=False)
es = np.abs(Vtv.T @ (np.where(sv2 > 1e-12 * sv2[0], 1 / sv2, 0) * (Uv.T @ yv)) - 1).max()
print("   bac 15 khong nhieu: normal %.3e  QR %.3e  SVD %.3e" % (en, eq, es))
print("   k(X) = %.2e  k(XtX) = %.2e  k(X)^2 = %.2e" % (np.linalg.cond(V), np.linalg.cond(V.T @ V),
                                                     np.linalg.cond(V) ** 2))
print("   bai cu (may Windows, 2026-08-31): normal 1.767e+01  QR 4.238e-06  SVD 6.164e-06  k(XtX) 1.38e+18")
print("   ti so sai so normal/QR tren may nay: %.3e" % (en / eq))

# ═══════════════════════════════════════════ 9. ma trận chiếu H và bộ nhớ
head("9. Ma trận chiếu H — đường chéo lấy từ Q, và bảng bộ nhớ")
rng = np.random.default_rng(0)
n9, p9 = 400, 5
X9 = np.c_[np.ones(n9), rng.normal(size=(n9, p9 - 1))]
H9 = X9 @ np.linalg.solve(X9.T @ X9, X9.T)
Q9, _ = np.linalg.qr(X9)
h9 = (Q9 ** 2).sum(1)
print("   lech diag(H) vs Q: %.3e ; tr(H) = %.12f ; |H@H - H| = %.3e"
      % (np.abs(np.diag(H9) - h9).max(), np.trace(H9), np.abs(H9 @ H9 - H9).max()))
for nn, pp in ((10_000, 100), (100_000, 100), (1_000_000, 100)):
    print("   n=%d p=%d: X %.2f MB  XtX %.4f MB  H %.1f MB  ti so X/XtX = %.0f"
          % (nn, pp, nn * pp * 8 / 1e6, pp * pp * 8 / 1e6, nn * nn * 8 / 1e6, (nn * pp) / (pp * pp)))
# Ba mục dưới chuyển từ v_gd1ss.py mục 370/380/420 sang numpy thuần: script đó gọi sklearn
# ngay ở mục 350 nên gãy trên máy không có sklearn, và năm con số bài trích mất đường chạy lại.


def rmse(a_, b_):
    return float(np.sqrt(np.mean((a_ - b_) ** 2)))


def ols(A_, b_):
    return np.linalg.lstsq(A_, b_, rcond=None)[0]


# ═══════════════════════════════════════ 10. chệch không giảm khi thêm dữ liệu
head("10. Đường thẳng trên dữ liệu cong — chệch KHÔNG giảm khi thêm dữ liệu (v_gd1ss 370)")
rng = np.random.default_rng(1)
x10 = np.linspace(-1, 1, 200)
y10 = np.sin(2.5 * x10) + 0.10 * rng.normal(size=x10.size)
for bac in (1, 2, 3, 5):
    V10 = np.vander(x10, bac + 1, increasing=True)
    print("   bac %d, n = 200: RMSE = %.4f" % (bac, rmse(y10, V10 @ ols(V10, y10))))
cuoi = None
for m_ in (50, 200, 2000, 20000, 200000):
    xm_ = np.linspace(-1, 1, m_)
    ym_ = np.sin(2.5 * xm_) + 0.10 * np.random.default_rng(2).normal(size=m_)
    Vm_ = np.vander(xm_, 2, increasing=True)
    cuoi = rmse(ym_, Vm_ @ ols(Vm_, ym_))
    print("   n = %7d -> RMSE bac 1 = %.4f" % (m_, cuoi))
print("   RMSE cuoi / muc nhieu 0.10 = %.4f lan ; n tang %d lan" % (cuoi / 0.10, 200000 // 50))

# ════════════════════════════════════ 11. hai cột gần trùng — OLS so với Ridge
head("11. Hai cột gần trùng — OLS nổ và đảo dấu; Ridge lambda = 1 dạng đóng (v_gd1ss 380)")
B_THAT = np.array([1.0, 1.0])


def gan_trung(seed, n_=50, jitter=0.01):
    r_ = np.random.default_rng(seed)
    z_ = r_.normal(size=n_)
    X_ = np.c_[z_, z_ + jitter * r_.normal(size=n_)]
    return X_, X_ @ B_THAT + r_.normal(size=n_)


X0, _ = gan_trung(0)
print("   tuong quan hai cot = %.6f ; k(X) = %.3e ; k(XtX) = %.3e"
      % (np.corrcoef(X0[:, 0], X0[:, 1])[0, 1], np.linalg.cond(X0), np.linalg.cond(X0.T @ X0)))
bs_, rs_ = [], []
for s_ in range(400):
    Xs, ys_ = gan_trung(s_)
    bs_.append(ols(Xs, ys_))
    # sklearn Ridge(alpha=1, fit_intercept=False) cực tiểu ||y - Xw||^2 + ||w||^2: đúng dạng đóng này
    rs_.append(np.linalg.solve(Xs.T @ Xs + np.eye(2), Xs.T @ ys_))
bs_, rs_ = np.array(bs_), np.array(rs_)
for ten_, mm in (("OLS", bs_), ("Ridge lambda=1", rs_)):
    print("   %-15s b1 tb %+.4f  lech chuan %.4f  khoang [%+.2f, %+.2f]  ||b - b_that|| tb %.4f  b1 < 0: %.1f %%"
          % (ten_, mm[:, 0].mean(), mm[:, 0].std(), mm[:, 0].min(), mm[:, 0].max(),
             np.linalg.norm(mm - B_THAT, axis=1).mean(), 100 * (mm[:, 0] < 0).mean()))
print("   OLS: lech chuan cua b1 + b2 = %.4f ; ti so lech chuan b1 / tong = %.2f"
      % (bs_.sum(1).std(), bs_[:, 0].std() / bs_.sum(1).std()))
print("   ti so sai so trung binh OLS / Ridge = %.2f lan"
      % (np.linalg.norm(bs_ - B_THAT, axis=1).mean() / np.linalg.norm(rs_ - B_THAT, axis=1).mean()))

# ═══════════════════════════════════════════ 12. một điểm ngoại lai — Huber IRLS
head("12. Một điểm ngoại lai — OLS so với Huber IRLS (thang MAD, delta = 1.345) (v_gd1ss 420)")
rng = np.random.default_rng(0)
n12 = 60
x12 = rng.uniform(0, 10, n12)
X12 = np.c_[np.ones(n12), x12]
y12 = 2.0 + 1.5 * x12 + rng.normal(scale=1.0, size=n12)


def huber_irls(A_, b_, delta=1.345, lap=500):
    """Huber qua bình phương tối thiểu có trọng số lặp. KHÔNG phải sklearn.HuberRegressor:
    bản sklearn ước lượng thang đo cùng lúc và có phạt L2 alpha = 1e-4."""
    w_ = ols(A_, b_)
    for _ in range(lap):
        r_ = b_ - A_ @ w_
        s_ = np.median(np.abs(r_ - np.median(r_))) / 0.6745
        u_ = np.abs(r_) / (delta * s_)
        wt_ = np.where(u_ <= 1, 1.0, 1.0 / u_)
        moi = np.linalg.solve(A_.T @ (wt_[:, None] * A_), A_.T @ (wt_ * b_))
        if np.abs(moi - w_).max() < 1e-13:
            return moi
        w_ = moi
    return w_


for shift in (0, 20, 50, 100):
    yb_ = y12.copy()
    yb_[0] += shift
    bo_, bh_ = ols(X12, yb_), huber_irls(X12, yb_)
    print("   +%3d o 1/%d diem: OLS chan %+.4f goc %+.4f (lech %.1f %%) | Huber chan %+.4f goc %+.4f (lech %.1f %%)"
          % (shift, n12, bo_[0], bo_[1], 100 * abs(bo_[1] - 1.5) / 1.5,
             bh_[0], bh_[1], 100 * abs(bh_[1] - 1.5) / 1.5))
print("   1 diem tren %d = %.2f %% du lieu" % (n12, 100 / n12))

# ═══════════════════════════════════════ 13. Legendre tr.78 — hệ quả cho mét
head("13. Legendre 1805 tr.78 — độ dẹt 1/148 kéo theo mét phải giảm bao nhiêu")
D_phap, D_peru = 28497.78, 28504.10
print("   D chi do o Phap = %.2f ; D Phap + Peru (in) = %.2f ; ti le = 1/%.1f (in: 'environ un 4500eme')"
      % (D_phap, D_peru, 1 / (D_peru / D_phap - 1)))
print("   dung lai: D = %.4f -> 1/%.1f" % (28500 / (1 + zc), 1 / (D_peru / (28500 / (1 + zc)) - 1)))
print("\nXong.")
