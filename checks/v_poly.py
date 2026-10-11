# -*- coding: utf-8 -*-
"""Kiểm chứng bài Polynomial Regression — dựng lại MỌI con số của hai nguồn ruột.

Nguồn ruột:
  [G] J. D. Gergonne 1815, "Application de la méthode des moindres quarrés à l'interpolation
      des suites", Annales de Math. pures et appliquées 6:242–252
      (_papers/1815 - Gergonne - ...pdf, PDF trang = trang in − 240).
  [L] A.-M. Legendre 1805, Nouvelles méthodes..., lời nói đầu tr.iv–v (Gergonne trích).
  [B] C. M. Bishop 2006, PRML, mục 1.1 (PDF trang 24–31).

Mục:
  1  [G] tr.246 — hệ số đạo hàm nội suy 3/5/7 tung độ, sai số E khi một tung độ lệch β
  2  [G] tr.247–248 — khoảng cách a tăng theo số tự nhiên
  3  [G] tr.248 — "bỏ bớt quan sát" so với bình phương tối thiểu, trên đoạn cố định
  4  [G] tr.250 — "l'emporter en précision sur ces données elles-mêmes"
  5  [G] tr.251 — tách hệ số chẵn/lẻ khi x cách đều
  6  [G] tr.252 — công thức đổi biến in trên trang
  7  [B] Bảng 1.1 — dựng lại 10 điểm từ hệ số M=9, khớp lại M=0,1,3,6
  8  [B] Bảng 1.2 — Ridge ln λ = −18 và 0
  9  [B] Hình 1.5, 1.6, 1.8 — dạng đường cong trên dữ liệu tự sinh cùng quy trình
  10 Ví dụ tay 4 điểm của bài
  11 Con số lẻ của bài (tương quan cột, tổ hợp, 1/ε)
Chạy: cd tools && python v_poly.py
"""
import sys
import platform
from fractions import Fraction as F
from math import comb, log

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
print("numpy", np.__version__, "·", platform.platform())


def head(s):
    print("\n" + "=" * 92 + "\n" + s + "\n" + "=" * 92)


def giai(A, b):
    """Khử Gauss trên phân số chính xác."""
    n = len(A)
    M = [list(map(F, A[i])) + [F(b[i])] for i in range(n)]
    for c in range(n):
        p = next(r for r in range(c, n) if M[r][c] != 0)
        M[c], M[p] = M[p], M[c]
        for r in range(n):
            if r != c and M[r][c] != 0:
                k = M[r][c] / M[c][c]
                M[r] = [M[r][j] - k * M[c][j] for j in range(n + 1)]
    return [M[i][n] / M[i][i] for i in range(n)]


def nghich(A):
    n = len(A)
    cot = [giai(A, [1 if i == j else 0 for i in range(n)]) for j in range(n)]
    return [[cot[j][i] for j in range(n)] for i in range(n)]


def trong_so_dao_ham(n):
    """Nội suy bậc 2n qua 2n+1 nút −n..n (bước 1). Trả về hệ số của từng tung độ trong
    y'(0) và y''(0): dict nút -> (hệ số y', hệ số y'')."""
    nut = list(range(-n, n + 1))
    V = [[F(k) ** j for j in range(2 * n + 1)] for k in nut]
    out = {}
    for i, k in enumerate(nut):
        e = [1 if r == i else 0 for r in range(len(nut))]
        c = giai(V, e)                       # đa thức nội suy tung độ đơn vị tại nút k
        out[k] = (c[1], 2 * c[2])
    return out


def fs(x):
    return "%d" % x.numerator if x.denominator == 1 else "%d/%d" % (x.numerator, x.denominator)


# ═══════════════════════════════════════════════════════════════════════════════ 1
head("1  [G] tr.246 — hệ số đạo hàm nội suy và sai số khi MỘT tung độ b' lệch β")
IN_TR246 = {  # đúng như trang in: (mẫu, hệ số y' theo b'−b_, b''−b__, ...), (mẫu, −y'': b, b'+b_, ...)
    1: ((2, [1]), None),
    2: ((12, [8, -1]), (12, [30, -16, 1])),
    3: ((60, [45, -9, 1]), (180, [490, -270, 27, -2])),
}
for n in range(1, 7):
    w = trong_so_dao_ham(n)
    d1 = [w[k][0] for k in range(1, n + 1)]
    d2 = [w[k][1] for k in range(0, n + 1)]
    assert all(w[-k][0] == -w[k][0] and w[-k][1] == w[k][1] for k in range(1, n + 1))
    if n in IN_TR246:
        (m1, h1), g2 = IN_TR246[n]
        assert d1 == [F(h, m1) for h in h1], (n, d1)
        if g2:
            m2, h2 = g2
            assert d2 == [F(-h, m2) for h in h2], (n, d2)
        else:
            assert d2 == [F(-2), F(1)]
        khop = "KHỚP trang in"
    else:
        khop = "(trang in dừng ở 7, 'et ainsi de suite')"
    print("   %2d tung độ: y' theo b',b'',...: %-28s y'' theo b,b',b'',...: %-34s %s"
          % (2 * n + 1, " ".join(fs(x) for x in d1), " ".join(fs(x) for x in d2), khop))

print("\n   Sai số khi riêng b' lệch β — trang in: 1/2 β, 2/2 β · 2/3 β, 4/3 β · 3/4 β, 6/4 β")
print("   %-10s %-10s %-10s %-12s %-12s %s" % ("tung độ", "E.y'/β", "E.y''/β", "E.y''/E.y'", "n/(n+1)", "tung độ ảnh hưởng y'' NHẤT"))
IN_E = {1: (F(1, 2), F(2, 2)), 2: (F(2, 3), F(4, 3)), 3: (F(3, 4), F(6, 4))}
for n in range(1, 7):
    w = trong_so_dao_ham(n)
    e1, e2 = w[1]
    if n in IN_E:
        assert (e1, e2) == IN_E[n]
    assert e1 == F(n, n + 1) and e2 == 2 * e1
    lon1 = max(range(-n, n + 1), key=lambda k: abs(w[k][0]))
    lon2 = max(range(0, n + 1), key=lambda k: abs(w[k][1]))
    print("   %-10d %-10s %-10s %-12s %-12s nút %d (|hệ số| %s, còn b' là %s)"
          % (2 * n + 1, fs(e1), fs(e2), fs(e2 / e1), fs(F(n, n + 1)), lon2, fs(abs(w[lon2][1])), fs(abs(e2))))
    assert abs(w[lon1][0]) == abs(e1)          # với y' thì b' đúng là tung độ nặng nhất
print("   -> 'croissent comme 1/2, 2/3, 3/4, 4/5' ĐÚNG; 'tendent vers β' ĐÚNG (n/(n+1) -> 1);")
print("      'constamment double' ĐÚNG cho mọi n đã thử. NHƯNG câu tr.247 'l'ordonnée fautive est")
print("      précisément celle dont la valeur exerce l'influence la plus notable sur nos DEUX")
print("      coefficiens' chỉ đúng cho y': với y'' tung độ GIỮA b nặng hơn b' ở mọi n.")

print("\n   [L] Legendre tr.v: sai số 5 quan sát 'plus grande dans le rapport de 4 à 3'")
w1, w2 = trong_so_dao_ham(1), trong_so_dao_ham(2)
r1, r2 = w2[1][0] / w1[1][0], w2[1][1] / w1[1][1]
print("   y': %s : %s = %s     y'': %s : %s = %s" % (fs(w2[1][0]), fs(w1[1][0]), fs(r1),
                                                    fs(w2[1][1]), fs(w1[1][1]), fs(r2)))
assert r1 == r2 == F(4, 3)

print("\n   Sai số NGẪU NHIÊN độc lập, cùng phương sai σ² ở mọi tung độ (bước 1) — phương sai của")
print("   ước lượng = σ² · tổng bình phương hệ số (Gergonne tr.247 nói Legendre loại trừ 'compensation'):")
print("   %-10s %-16s %-16s" % ("tung độ", "Var(y')/σ²", "Var(y'')/σ²"))
for n in range(1, 7):
    w = trong_so_dao_ham(n)
    v1 = sum(w[k][0] ** 2 for k in w)
    v2 = sum(w[k][1] ** 2 for k in w)
    print("   %-10d %-16.6f %-16.6f" % (2 * n + 1, float(v1), float(v2)))
w3_, w13_ = trong_so_dao_ham(1), trong_so_dao_ham(6)
print("   Var(y') 13 tung độ / 3 tung độ = %.4f" % float(sum(w13_[k][0] ** 2 for k in w13_) / sum(w3_[k][0] ** 2 for k in w3_)))
print("   -> bước cố định thì phương sai CŨNG tăng theo số tung độ: bù trừ ngẫu nhiên không cứu được.")

# ═══════════════════════════════════════════════════════════════════════════════ 2
head("2  [G] tr.247–248 — khoảng cách a tăng theo số tự nhiên (a = n, lấy a = 1 khi có 3 quan sát)")
print("   trang in tr.248: 3 obs 1/2 β, 1/1 β · 5 obs 1/3 β, 1/3 β · 7 obs 1/4 β, 1/6 β")
IN_248 = {1: (F(1, 2), F(1, 1)), 2: (F(1, 3), F(1, 3)), 3: (F(1, 4), F(1, 6))}
print("   %-8s %-6s %-10s %-10s %-14s %-16s %-16s" % ("obs", "a", "E.y'/β", "E.y''/β", "2/(n(n+1))", "sd(y')/σ", "sd(y'')/σ"))
for n in range(1, 7):
    w = trong_so_dao_ham(n)
    a = n
    e1, e2 = w[1][0] / a, w[1][1] / a ** 2
    if n in IN_248:
        assert (e1, e2) == IN_248[n], (n, e1, e2)
    assert e1 == F(1, n + 1) and e2 == F(2, n * (n + 1))
    v1 = sum(w[k][0] ** 2 for k in w) / a ** 2
    v2 = sum(w[k][1] ** 2 for k in w) / a ** 4
    print("   %-8d %-6d %-10s %-10s %-14s %-16.6f %-16.6f"
          % (2 * n + 1, a, fs(e1), fs(e2), fs(F(2, n * (n + 1))), float(v1) ** .5, float(v2) ** .5))
print("   -> 'inverses des nombres naturels' và 'inverses des nombres triangulaires' ĐÚNG, khớp 3/3 dòng in.")
print("   Cái giá Gergonne nói ngay sau: đoạn quan sát trải từ −n·a tới n·a = −n² tới n² — với 7 quan sát")
print("   đã là [−9, 9], gấp 9 lần đoạn [−1, 1] của 3 quan sát.")

# ═══════════════════════════════════════════════════════════════════════════════ 3
head("3  [G] tr.248 — trên ĐOẠN CỐ ĐỊNH [−1, 1] có 21 quan sát cách đều: bỏ bớt, hay dùng hết?")
X21 = [F(i, 10) for i in range(-10, 11)]


def ls_var_dao_ham(xs, bac):
    """Var(y'(0))/σ², Var(y''(0))/σ², Var(y(0))/σ² của bình phương tối thiểu bậc `bac` trên xs."""
    V = [[x ** j for j in range(bac + 1)] for x in xs]
    G = [[sum(V[i][r] * V[i][c] for i in range(len(xs))) for c in range(bac + 1)] for r in range(bac + 1)]
    Gi = nghich(G)
    return Gi[1][1], 4 * Gi[2][2], Gi[0][0], Gi


def ls_uoc_luong(xs, ys, bac):
    V = [[x ** j for j in range(bac + 1)] for x in xs]
    G = [[sum(V[i][r] * V[i][c] for i in range(len(xs))) for c in range(bac + 1)] for r in range(bac + 1)]
    h = [sum(V[i][r] * ys[i] for i in range(len(xs))) for r in range(bac + 1)]
    return giai(G, h)


# hàm thử để đo CHỆCH (không nhiễu): f = e^x, f'(0) = f''(0) = 1
fx = [F(np.exp(float(x))) for x in X21]
print("   σ = 1. Chệch đo trên f(x) = e^x không nhiễu (f'(0) = f''(0) = 1).")
print("   %-44s %-12s %-12s %-14s %-14s" % ("cách", "sd(y')", "sd(y'')", "chệch y'", "chệch y''"))
for n_obs, a in ((3, F(1)), (5, F(1, 2)), (11, F(1, 5)), (21, F(1, 10))):
    n = (n_obs - 1) // 2
    w = trong_so_dao_ham(n)
    v1 = float(sum(w[k][0] ** 2 for k in w) / a ** 2)
    v2 = float(sum(w[k][1] ** 2 for k in w) / a ** 4)
    y1 = sum(w[k][0] * F(np.exp(float(k * a))) for k in w) / a
    y2 = sum(w[k][1] * F(np.exp(float(k * a))) for k in w) / a ** 2
    print("   %-44s %-12.6f %-12.6f %-+14.3e %-+14.3e" % ("nội suy %d quan sát, bước %s, BỎ %d" % (n_obs, fs(a), 21 - n_obs),
                                                   v1 ** .5, v2 ** .5, float(y1) - 1, float(y2) - 1))
for bac in (2, 4):
    v1, v2, v0, _ = ls_var_dao_ham(X21, bac)
    c = ls_uoc_luong(X21, fx, bac)
    print("   %-44s %-12.6f %-12.6f %-+14.3e %-+14.3e" % ("bình phương tối thiểu bậc %d trên CẢ 21" % bac,
                                                   float(v1) ** .5, float(v2) ** .5, float(c[1]) - 1, float(2 * c[2]) - 1))
w10 = trong_so_dao_ham(10)
v2_noi = float(sum(w10[k][1] ** 2 for k in w10) / F(1, 10) ** 4) ** .5
v2_ls = float(ls_var_dao_ham(X21, 2)[1]) ** .5
print("   sd(y'') nội suy 21 điểm / bình phương tối thiểu bậc 2 trên 21 điểm = %.4f / %.4f = %.2f"
      % (v2_noi, v2_ls, v2_noi / v2_ls))
print("   -> nội suy qua cả 21 điểm (bước 0,1) cho sd(y'') hàng trăm; bình phương tối thiểu bậc 2 dùng")
print("      đủ 21 điểm cho sd nhỏ nhất bảng, và trả bằng CHỆCH lớn hơn — nên số số hạng phải chọn,")
print("      đúng câu hỏi tr.252 để ngỏ: 'quel est le plus probablement le degré de cette courbe'.")

# ═══════════════════════════════════════════════════════════════════════════════ 4
head("4  [G] tr.250 — 'on pourra le plus souvent espérer ici de l'emporter en précision sur ces données'")
print("   sd của đa thức khớp tại x = 0 (giữa) và x = n (mép), tính theo σ của MỘT quan sát, x = −n..n:")
print("   %-8s %-16s %-16s %-16s %-16s" % ("obs", "bậc 2: giữa", "bậc 2: mép", "bậc 4: giữa", "bậc 4: mép"))
for n in (1, 2, 3, 5, 10, 20):
    xs = [F(k) for k in range(-n, n + 1)]
    hang = []
    for bac in (2, 4):
        if 2 * n + 1 <= bac:
            hang += [float("nan"), float("nan")]
            continue
        _, _, v0, Gi = ls_var_dao_ham(xs, bac)
        e = [F(n) ** j for j in range(bac + 1)]
        vm = sum(e[r] * Gi[r][c] * e[c] for r in range(bac + 1) for c in range(bac + 1))
        hang += [float(v0) ** .5, float(vm) ** .5]
    print("   %-8d %-16.6f %-16.6f %-16.6f %-16.6f" % (2 * n + 1, *hang))
print("   -> ở giữa: từ 5 quan sát trở lên đa thức bậc 2 CHÍNH XÁC HƠN chính dữ liệu (sd < 1). Ở mép thì")
print("      chậm hơn hẳn — khớp với cảnh báo tr.244: đừng dùng công thức 'se rapprochant trop' hai đầu.")

# ═══════════════════════════════════════════════════════════════════════════════ 5
head("5  [G] tr.251 — x cách đều −n..n: tổng luỹ thừa lẻ bằng 0, hệ số chẵn và lẻ tách được")
for n in range(1, 8):
    s = {m: sum(F(k) ** m for k in range(-n, n + 1)) for m in range(0, 9)}
    assert all(s[m] == 0 for m in (1, 3, 5, 7))
    assert s[2] == 2 * F(n * (n + 1) * (2 * n + 1), 6)
    assert s[4] == 2 * F(n * (n + 1) * (2 * n + 1) * (3 * n * n + 3 * n - 1), 30)
print("   Σa = Σa³ = Σa⁵ = Σa⁷ = 0 và Σa² = 2Σn², Σa⁴ = 2Σn⁴ (công thức đóng) — đúng với n = 1..7.")
xs = [F(k) for k in range(-3, 4)]
ys = [F(v) for v in (4, 2, 1, 1, 3, 6, 12)]
full = ls_uoc_luong(xs, ys, 3)
s = {m: sum(x ** m for x in xs) for m in range(7)}
sy = {m: sum(x ** m * y for x, y in zip(xs, ys)) for m in range(4)}
chan = giai([[s[0], s[2]], [s[2], s[4]]], [sy[0], sy[2]])          # A, C
le = giai([[s[2], s[4]], [s[4], s[6]]], [sy[1], sy[3]])            # B, D
print("   Ví dụ 7 điểm x = −3..3, y = 4 2 1 1 3 6 12, bậc 3:")
print("     hệ đủ 4×4 : A=%s  B=%s  C=%s  D=%s" % tuple(fs(v) for v in full))
print("     hai hệ 2×2: A=%s  C=%s | B=%s  D=%s" % (fs(chan[0]), fs(chan[1]), fs(le[0]), fs(le[1])))
assert [chan[0], le[0], chan[1], le[1]] == full
print("     ma trận chuẩn (Σa^(i+j)):")
for r in range(4):
    print("       " + " ".join("%6s" % fs(s[r + c]) for c in range(4)))
print("   -> KHỚP tuyệt đối: giải một hệ 4×4 bằng hai hệ 2×2, đúng như tr.251 hứa.")

# ═══════════════════════════════════════════════════════════════════════════════ 6
head("6  [G] tr.252 — công thức đổi biến: trang in có mẫu 'dλ/dx' và số mũ 2 ở mẫu của y''")
for ten, X, Y, z, y1, y2 in (
        ("x = z², y = z³, z = 2", lambda z: (z * z, 2 * z, F(2)), lambda z: (z ** 3, 3 * z * z, 6 * z),
         F(2), F(3), F(3, 8)),
        ("x = 3z + 1, y = z⁴, z = 1", lambda z: (3 * z + 1, F(3), F(0)), lambda z: (z ** 4, 4 * z ** 3, 12 * z * z),
         F(1), F(4, 3), F(12, 9))):
    x0, x1, x2 = X(z)
    _, t1, t2 = Y(z)
    in_mu2 = (x1 * t2 - t1 * x2) / x1 ** 2
    dung_mu3 = (x1 * t2 - t1 * x2) / x1 ** 3
    print("   %-26s dy/dx thật %-6s y'/x' = %-6s | d²y/dx² thật %-6s  mũ 2 (in) = %-6s  mũ 3 = %-6s"
          % (ten, fs(y1), fs(t1 / x1), fs(y2), fs(in_mu2), fs(dung_mu3)))
    assert t1 / x1 == y1 and dung_mu3 == y2 and in_mu2 != y2
print("   -> mẫu số đúng là (dx/dz)³. Mũ 2 chỉ cho đúng khi dx/dz = 1, tức khi z CHÍNH LÀ x.")

# ═══════════════════════════════════════════════════════════════════════════════ 7
head("7  [B] Bảng 1.1 tr.8 — dựng lại 10 điểm huấn luyện từ cột M = 9, rồi khớp lại các cột khác")
W9 = np.array([0.35, 232.37, -5321.83, 48568.31, -231639.30, 640042.26,
               -1061800.52, 1042400.18, -557682.99, 125201.43])
IN_M0 = np.array([0.19])
IN_M1 = np.array([0.82, -1.27])
IN_M6COT = np.array([0.31, 7.99, -25.43, 17.37])       # cột mang nhãn "M = 6" nhưng chỉ có 4 dòng
IN_R18 = np.array([0.35, 4.74, -0.77, -31.97, -3.89, 55.28, 41.32, -45.95, -91.53, 72.68])
IN_R0 = np.array([0.13, -0.05, -0.06, -0.05, -0.03, -0.02, -0.01, -0.00, 0.00, 0.01])


def khop(x, t, M, lam=0.0, w0_phat=True):
    V = np.vander(x, M + 1, increasing=True)
    if lam == 0.0:
        return np.linalg.lstsq(V, t, rcond=None)[0]
    D = np.sqrt(lam) * np.eye(M + 1)
    if not w0_phat:
        D[0, 0] = 0.0
    return np.linalg.lstsq(np.vstack([V, D]), np.r_[t, np.zeros(M + 1)], rcond=None)[0]


LUOI = {
    "linspace(0,1,10)": np.linspace(0, 1, 10),
    "(i+0.5)/10": (np.arange(10) + .5) / 10,
    "i/10, i=0..9": np.arange(10) / 10,
    "(i+1)/11": (np.arange(10) + 1) / 11,
}
print("   %-18s %-10s %-10s %-10s %-10s %-9s %-9s" % ("lưới x", "lệch M0", "lệch M1", "lệch M3", "lệch M6", "sd nhiễu", "tb nhiễu"))
for ten, x in LUOI.items():
    t = np.vander(x, 10, increasing=True) @ W9
    d0 = np.abs(khop(x, t, 0) - IN_M0).max()
    d1 = np.abs(khop(x, t, 1) - IN_M1).max()
    d3 = np.abs(khop(x, t, 3) - IN_M6COT).max()
    d6 = np.abs(khop(x, t, 6)[:4] - IN_M6COT).max()
    nh = t - np.sin(2 * np.pi * x)
    print("   %-18s %-10.4f %-10.4f %-10.4f %-10.4f %-9.4f %-+9.4f" % (ten, d0, d1, d3, d6, nh.std(ddof=1), nh.mean()))

x = LUOI["linspace(0,1,10)"]
t = np.vander(x, 10, increasing=True) @ W9
print("\n   Với lưới linspace(0,1,10):")
print("   x = " + " ".join("%.6f" % v for v in x))
print("   t = " + " ".join("%.6f" % v for v in t))
print("   nhiễu t − sin(2πx) = " + " ".join("%+.6f" % v for v in t - np.sin(2 * np.pi * x)))
for M, inb in ((0, IN_M0), (1, IN_M1), (3, IN_M6COT)):
    w = khop(x, t, M)
    print("   M = %d dựng lại: %s   (in: %s)" % (M, " ".join("%.4f" % v for v in w), " ".join("%.2f" % v for v in inb)))
w6 = khop(x, t, 6)
print("   M = 6 dựng lại: %s" % " ".join("%.2f" % v for v in w6))
print("   -> cột in nhãn 'M = 6' có 4 hệ số và khớp M = 3; M = 6 thật có 7 hệ số, w1 = %.2f." % w6[1])
print("   phần dư của M = 9 trên dữ liệu dựng lại: %.3e (Bishop tr.6: E(w*) = 0)"
      % np.abs(np.vander(x, 10, increasing=True) @ khop(x, t, 9) - t).max())
print("   κ(V) của M = 9 trên lưới này: %.4e" % np.linalg.cond(np.vander(x, 10, increasing=True)))

rng = np.random.default_rng(0)
lo = {k: None for k in ("M0", "M1", "M3", "R18", "R0")}
for _ in range(4000):
    w9p = W9 + rng.uniform(-.005, .005, 10)
    tp = np.vander(x, 10, increasing=True) @ w9p
    ket = {"M0": khop(x, tp, 0), "M1": khop(x, tp, 1), "M3": khop(x, tp, 3),
           "R18": khop(x, tp, 9, np.exp(-18)), "R0": khop(x, tp, 9, 1.0)}
    for k, v in ket.items():
        lo[k] = np.c_[v, v] if lo[k] is None else np.c_[np.minimum(lo[k][:, 0], v), np.maximum(lo[k][:, 1], v)]
print("\n   Độ nhạy với phép làm tròn 2 chữ số của cột M = 9 (4000 lần rút ±0,005 mỗi hệ số):")
for k, inb in (("M0", IN_M0), ("M1", IN_M1), ("M3", IN_M6COT)):
    for j, v in enumerate(inb):
        a, b = lo[k][j]
        print("     %-3s w%d: in %8.2f   khoảng dựng lại [%.4f ; %.4f]  %s"
              % (k, j, v, a, b, "trong" if a - .005 <= v <= b + .005 else "NGOÀI"))

# ═══════════════════════════════════════════════════════════════════════════════ 8
head("8  [B] Bảng 1.2 tr.11 — Ridge (1.4) trên cùng 10 điểm dựng lại")
for tenl, lam, inb, k in (("ln λ = −18", np.exp(-18), IN_R18, "R18"), ("ln λ = 0", 1.0, IN_R0, "R0")):
    w_co = khop(x, t, 9, lam, True)
    w_ko = khop(x, t, 9, lam, False)
    print("   %s" % tenl)
    print("     in             : " + " ".join("%8.2f" % v for v in inb))
    print("     phạt cả w0     : " + " ".join("%8.2f" % v for v in w_co) + "   lệch max %.4f" % np.abs(w_co - inb).max())
    print("     không phạt w0  : " + " ".join("%8.2f" % v for v in w_ko) + "   lệch max %.4f" % np.abs(w_ko - inb).max())
    ngoai = [j for j in range(10) if not (lo[k][j, 0] - .005 <= inb[j] <= lo[k][j, 1] + .005)]
    print("     khoảng làm tròn: " + " ".join("[%.2f;%.2f]" % (a, b) for a, b in lo[k]))
    print("     số hệ số in nằm NGOÀI khoảng làm tròn: %d / 10" % len(ngoai))

print("\n   8b. Chính các hệ số IN của Bảng 1.2, đem đo trên 10 điểm dựng lại:")
V9 = np.vander(x, 10, increasing=True)
for ten, w in (("ln λ = −∞ (in)", W9), ("ln λ = −18 (in)", IN_R18), ("ln λ = 0 (in)", IN_R0)):
    r = np.sqrt(((V9 @ w - t) ** 2).mean())
    print("     %-18s RMSE huấn luyện %.6f   RMSE/√2 %.6f   y(0) = %.4f   y(1) = %.4f"
          % (ten, r, r / np.sqrt(2), w[0], w.sum()))
print("   Quét ln λ từ −45 tới 15 (bước 0,01), tìm λ làm (1.4) cho ra cột in gần nhất:")
LL = np.arange(-45, 15.0001, 0.01)
for tenl, inb in (("ln λ = −18", IN_R18), ("ln λ = 0", IN_R0)):
    for w0p in (True, False):
        lech = np.array([np.abs(khop(x, t, 9, np.exp(l), w0p) - inb).max() for l in LL])
        i = int(lech.argmin())
        wb = khop(x, t, 9, np.exp(LL[i]), w0p)
        lo_, hi_ = wb.copy(), wb.copy()
        r = np.random.default_rng(11)
        for _ in range(2000):
            v = khop(x, V9 @ (W9 + r.uniform(-.005, .005, 10)), 9, np.exp(LL[i]), w0p)
            lo_, hi_ = np.minimum(lo_, v), np.maximum(hi_, v)
        ngoai = int(np.sum((inb < lo_ - .005) | (inb > hi_ + .005)))
        print("     cột %-11s %-15s ln λ tốt nhất %7.2f   lệch max %.4f   NGOÀI khoảng làm tròn %d / 10   ln λ tốt − ln λ in = %+.2f"
              % (tenl, "phạt cả w0" if w0p else "không phạt w0", LL[i], lech[i], ngoai,
                 LL[i] - (-18.0 if "18" in tenl else 0.0)))
print("   ln N = %.4f   ln(N−1) = %.4f" % (np.log(10), np.log(9)))
print("   Giả thuyết: mã sinh bảng nhân λ với N = 10 (tức cực tiểu hoá E(w)/N + λ/2·‖w‖²).")
print("   Thử đúng tại ln λ + ln N, và cả ln λ + ln(N−1), với khoảng làm tròn của cột M = 9:")
for tenl, l0, inb in (("ln λ = −18", -18.0, IN_R18), ("ln λ = 0", 0.0, IN_R0)):
    for tenc, c in (("λ theo (1.4)", 1.0), ("λ·(N−1)", 9.0), ("λ·N", 10.0)):
        wc = khop(x, t, 9, np.exp(l0) * c)
        lo_, hi_ = wc.copy(), wc.copy()
        r = np.random.default_rng(7)
        for _ in range(2000):
            tp = V9 @ (W9 + r.uniform(-.005, .005, 10))
            v = khop(x, tp, 9, np.exp(l0) * c)
            lo_, hi_ = np.minimum(lo_, v), np.maximum(hi_, v)
        ngoai = int(np.sum((inb < lo_ - .005) | (inb > hi_ + .005)))
        print("     %-11s %-13s lệch max %.4f   hệ số in NGOÀI khoảng làm tròn: %d / 10   dựng: %s"
              % (tenl, tenc, np.abs(wc - inb).max(), ngoai, " ".join("%.2f" % v for v in wc)))
print("   Huấn luyện của Ridge TĂNG ĐƠN ĐIỆU theo λ (λ lớn hơn chỉ thêm ràng buộc), nên RMSE ở")
print("   ln λ = −18 phải ≥ RMSE ở ln λ = −20 nếu hai con số tính trên CÙNG dữ liệu và CÙNG thang λ.")

# ═══════════════════════════════════════════════════════════════════════════════ 9
head("9  [B] Hình 1.5 · 1.6 · 1.8 — số hoá điểm ảnh, rồi so với dữ liệu dựng lại")
try:
    import pypdfium2 as pdfium
    from scipy import ndimage
    import os
    PDF_B = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "AI algorithms", "_papers",
                         "2006 - Bishop - Pattern Recognition and Machine Learning.pdf")
    doc = pdfium.PdfDocument(PDF_B)

    def vung(trang, box, scale=6):
        im = np.asarray(doc[trang - 1].render(scale=scale).to_pil().convert("RGB")).astype(int)
        H, W = im.shape[:2]
        return im[int(box[1] * H):int(box[3] * H), int(box[0] * W):int(box[2] * W)]

    def khung(im):
        toi = (im.max(axis=2) < 90)
        hang = np.where(toi.mean(axis=1) > .5)[0]
        cot = np.where(toi.mean(axis=0) > .5)[0]
        return hang.min(), hang.max(), cot.min(), cot.max()

    def mau(im, kenh):
        r, g, b = im[..., 0], im[..., 1], im[..., 2]
        if kenh == "xanh":
            return (b > 180) & (r < 90) & (g < 90)
        return (r > 180) & (g < 90) & (b < 90)

    # Hình 1.5, trang 28: trục dọc từ 0 (đáy khung) tới 1 (đỉnh khung). Vòng tròn đánh dấu
    # dính vào đường nối nên không tách thành khối riêng: lấy tâm vòng ĐẦU và CUỐI từ mép trái/phải
    # của nét màu, chia đều 9 khoảng, rồi đọc đỉnh–đáy vòng ở từng cột M.
    im = vung(28, (0.55, 0.10, 0.94, 0.32))
    top, bot, lef, rig = khung(im)
    doc_so = {}
    for kenh in ("xanh", "do"):
        m = mau(im, kenh)
        m[:, :lef + 3] = False
        m[:, rig - 2:] = False
        m[:int(top + .30 * (bot - top)), :int(lef + .50 * (rig - lef))] = False   # bỏ chú giải góc trên trái
        cot = np.where(m.any(axis=0))[0]
        trai = np.where(m[:, cot.min():cot.min() + 60].any(axis=1))[0]
        ban_kinh = (trai.max() - trai.min()) / 2
        x0, x9 = cot.min() + ban_kinh, cot.max() - ban_kinh
        gia_tri = []
        for M in range(10):
            xc = int(round(x0 + M * (x9 - x0) / 9))
            hang = np.where(m[:, xc - 1:xc + 2].any(axis=1))[0]
            hang = hang[np.abs(hang - np.median(hang)) <= ban_kinh * 1.3]
            gia_tri.append((bot - (hang.min() + hang.max()) / 2) / (bot - top))
        doc_so[kenh] = gia_tri
        print("   Hình 1.5 nét %-4s: %d điểm ảnh, bán kính vòng %.1f px" % (kenh, int(m.sum()), ban_kinh))
    print("   %-4s %-12s %-14s %-14s %-12s %-14s" % ("M", "huấn (hình)", "RMSE dựng lại", "RMSE/√2", "kiểm (hình)", "kiểm tự sinh/√2"))
    for M in range(10):
        w = khop(x, t, M)
        tr = np.sqrt(((np.vander(x, M + 1, increasing=True) @ w - t) ** 2).mean())
        te = np.sqrt(((np.vander(xte_ := np.random.default_rng(0).uniform(0, 1, 100), M + 1, increasing=True) @ w
                       - (np.sin(2 * np.pi * xte_) + np.random.default_rng(0).normal(0, .3, 100))) ** 2).mean())
        print("   %-4d %-12.4f %-14.4f %-14.4f %-12.4f %-14.4f"
              % (M, doc_so["xanh"][M], tr, tr / np.sqrt(2), doc_so["do"][M], te / np.sqrt(2)))
    hs = [np.sqrt(((np.vander(x, M + 1, increasing=True) @ khop(x, t, M) - t) ** 2).mean()) for M in range(9)]
    ti = np.array([doc_so["xanh"][M] / hs[M] for M in range(9)])
    print("   tỉ số huấn (hình) / RMSE dựng lại, M = 0..8: " + " ".join("%.4f" % v for v in ti))
    print("   trung bình %.4f   (1/√2 = %.4f)" % (ti.mean(), 1 / np.sqrt(2)))

    # Hình 1.8, trang 31: đầu mút phải của đường xanh (huấn) và đỏ (kiểm)
    im = vung(31, (0.55, 0.69, 0.94, 0.905))
    top, bot, lef, rig = khung(im)
    for kenh, ten in (("xanh", "huấn"), ("do", "kiểm")):
        m = mau(im, kenh)
        m[:, :lef + 3] = False
        m[:, rig - 2:] = False
        m[:int(top + .25 * (bot - top)), int(lef + .55 * (rig - lef)):] = False   # bỏ chú giải góc trên phải
        cot = np.where(m.any(axis=0))[0]
        c_p, c_t = cot.max(), cot.min()
        yp = np.where(m[:, c_p])[0].mean()
        yt = np.where(m[:, c_t])[0].mean()
        print("   Hình 1.8 đường %-4s: đầu trái %.4f · đầu phải (ln λ ≈ −20) %.4f"
              % (ten, (bot - yt) / (bot - top), (bot - yp) / (bot - top)))
except Exception as ex:                                         # báo ra, không im lặng
    print("   KHÔNG số hoá được hình: %r" % (ex,))

print("\n   Tập kiểm tra tự sinh (Hình 1.5 ở trên dùng lại đúng tập này):")
print("   Hình 1.5: E_RMS huấn luyện trên 10 điểm dựng lại, kiểm tra trên 100 điểm sinh bằng hạt giống 0")
rng = np.random.default_rng(0)
xte = rng.uniform(0, 1, 100)
tte = np.sin(2 * np.pi * xte) + rng.normal(0, .3, 100)
print("   %-4s %-12s %-12s %-14s" % ("M", "E_RMS huấn", "E_RMS kiểm", "max |w|"))
for M in range(10):
    w = khop(x, t, M)
    tr = np.sqrt(((np.vander(x, M + 1, increasing=True) @ w - t) ** 2).mean())
    te = np.sqrt(((np.vander(xte, M + 1, increasing=True) @ w - tte) ** 2).mean())
    print("   %-4d %-12.6f %-12.6f %-14.4f" % (M, tr, te, np.abs(w).max()))
print("   (E_RMS = sqrt(2E(w)/N) của (1.3) chính là RMSE thường.)")

print("\n   Hình 1.6: M = 9, tăng N — trung vị trên 200 hạt giống của E_RMS kiểm tra (1000 điểm)")
for N in (10, 15, 30, 100):
    ds = []
    for s in range(200):
        r = np.random.default_rng(1000 + s)
        xs_ = r.uniform(0, 1, N)
        ts_ = np.sin(2 * np.pi * xs_) + r.normal(0, .3, N)
        xk = r.uniform(0, 1, 1000)
        tk = np.sin(2 * np.pi * xk) + r.normal(0, .3, 1000)
        w = khop(xs_, ts_, 9)
        ds.append(np.sqrt(((np.vander(xk, 10, increasing=True) @ w - tk) ** 2).mean()))
    print("     N = %-4d trung vị E_RMS kiểm tra = %.6f" % (N, float(np.median(ds))))

print("\n   Hình 1.8: M = 9 trên 10 điểm dựng lại, quét ln λ")
best = None
for ll in range(-40, 1, 2):
    w = khop(x, t, 9, np.exp(ll))
    te = np.sqrt(((np.vander(xte, 10, increasing=True) @ w - tte) ** 2).mean())
    tr = np.sqrt(((np.vander(x, 10, increasing=True) @ w - t) ** 2).mean())
    best = (te, ll) if best is None or te < best[0] else best
    print("     ln λ = %4d  huấn %.6f  kiểm %.6f" % (ll, tr, te))
print("   -> ln λ tốt nhất trên tập kiểm tra tự sinh: %d (E_RMS %.6f)" % (best[1], best[0]))

# ═══════════════════════════════════════════════════════════════════════════════ 10
head("10  Ví dụ tay của bài: x = (−1, 0, 1, 2), y = (3, 1, 2, 9), bậc 2")
xs = [F(v) for v in (-1, 0, 1, 2)]
ys = [F(v) for v in (3, 1, 2, 9)]
V = [[x_ ** j for j in range(3)] for x_ in xs]
G = [[sum(V[i][r] * V[i][c] for i in range(4)) for c in range(3)] for r in range(3)]
h = [sum(V[i][r] * ys[i] for i in range(4)) for r in range(3)]
b = giai(G, h)
yh = [sum(V[i][j] * b[j] for j in range(3)) for i in range(4)]
e = [ys[i] - yh[i] for i in range(4)]
rss = sum(v * v for v in e)
tss = sum((v - sum(ys) / 4) ** 2 for v in ys)
print("   VᵀV = %s   Vᵀy = %s" % ([[fs(v) for v in r] for r in G], [fs(v) for v in h]))
print("   β̂ = %s   ŷ = %s   e = %s" % ([fs(v) for v in b], [fs(v) for v in yh], [fs(v) for v in e]))
print("   Σe = %s  Σex = %s  Σex² = %s" % tuple(fs(sum(e[i] * xs[i] ** j for i in range(4))) for j in range(3)))
print("   RSS = %s = %.6f   TSS = %s   R² = %.6f   κ(V) = %.6f"
      % (fs(rss), float(rss), fs(tss), 1 - float(rss / tss),
         np.linalg.cond(np.array([[float(v) for v in r] for r in V]))))

# ═══════════════════════════════════════════════════════════════════════════════ 11
head("11  Con số lẻ của bài")
xx = np.linspace(0, 1, 1000)
print("   tương quan x^10 và x^11 trên lưới 1000 điểm [0,1]: %.6f" % np.corrcoef(xx ** 10, xx ** 11)[0, 1])
print("   2000^6 = %.4e" % 2000.0 ** 6)
print("   C(10+3,3) = %d   C(100+3,3) = %d" % (comb(13, 3), comb(103, 3)))
print("   1/ε float64 = %.6e" % (1 / np.finfo(float).eps))
print("   ln λ = −18 -> λ = %.6e" % np.exp(-18))
print("   Gergonne tr.248: 7 quan sát với a = 3 trải đoạn dài %d (so với %d của 3 quan sát)" % (2 * 3 * 3, 2))
print("   log10 của max|w| ở Bảng 1.1 M = 9: %.4f" % log(abs(W9).max(), 10))
print("   Gergonne 1815 -> Bishop 2006: %d năm" % (2006 - 1815))
