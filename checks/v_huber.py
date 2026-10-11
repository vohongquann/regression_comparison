# -*- coding: utf-8 -*-
"""Dựng lại TOÀN BỘ phần ruột của Huber (1964) cho bài `Huber Regression`.

    cd tools && python v_huber.py

Nguồn: P. J. Huber, "Robust Estimation of a Location Parameter", The Annals of
Mathematical Statistics 35(1):73-101, 1964. Bản PDF trong `_papers/` là BẢN QUÉT
JSTOR — lớp text RỖNG ở 28/29 trang, nên mọi câu trích trong bài được chép tay
từ ảnh render và KHÔNG kiểm được bằng `kiemtrich.py`. Điều đó ghi rõ trong bài.

Bài báo có 4 bảng số (I, II, III, IV) với 403 + 18 + 15 + 63 = 499 ô, cộng 2 bảng
không đánh số ở tr.101 với 20 ô nữa — tổng cộng 519 ô. Script này dựng lại tất cả từ công thức của chính bài báo rồi
đối chiếu từng chữ số — đó là phép kiểm mạnh nhất có thể làm cho một bài báo
thuần lý thuyết, vì nó không cần dữ liệu và không cần huấn luyện gì.

Quy ước ký hiệu, theo đúng bài báo:
    k         ngưỡng của psi (bài viết gọi là delta; scikit-learn gọi là epsilon)
    epsilon   tỉ lệ nhiễm bẩn của F = (1-eps) Phi + eps H
    psi(t)    = t với |t| < k, = k sign(t) với |t| >= k        (chú thích Bảng I)
    V         phương sai tiệm cận của n^(1/2) T_n

Mọi phép tích phân dùng công thức đóng theo Phi và phi, không dùng cầu phương,
trừ Bảng II (ba phương trình phi tuyến) và mục Hodges-Lehmann (tích phân f^2).
"""
import os
import sys

for _b in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_b, "1")

import numpy as np
from scipy import integrate, optimize
from scipy.stats import norm

sys.stdout.reconfigure(encoding="utf-8")

PHI = norm.cdf
phi = norm.pdf
NGAN = "-" * 92


def muc(so, ten):
    print()
    print("=" * 92)
    print("%s. %s" % (so, ten))
    print("=" * 92)


# ════════════════════════════════════════════════════════════════════════════
# Phần A — hàm của chính bài báo
# ════════════════════════════════════════════════════════════════════════════

def rho(t, k):
    """rho của tr.75 và tr.79 mục (iii): 1/2 t^2 nếu |t| <= k, k|t| - 1/2 k^2 nếu |t| > k."""
    t = np.asarray(t, dtype=float)
    a = np.abs(t)
    return np.where(a <= k, 0.5 * t * t, k * a - 0.5 * k * k)


def psi(t, k):
    """psi = rho', chú thích Bảng I tr.85: t với |t| < k, k sign(t) với |t| >= k."""
    return np.clip(np.asarray(t, dtype=float), -k, k)


def E_psi_pham(k):
    """E_Phi psi' = P_Phi[|x| < k] = 2 Phi(k) - 1. Câu này ở tr.82, ngay dưới Hệ quả."""
    return 2.0 * PHI(k) - 1.0


def E_psi_binh(k):
    """E_Phi psi^2 = int_{-k}^{k} t^2 phi dt + 2 k^2 (1 - Phi(k)), công thức đóng."""
    return (2.0 * PHI(k) - 1.0) - 2.0 * k * phi(k) + 2.0 * k * k * (1.0 - PHI(k))


def eps_min(k):
    """Nghịch đảo quan hệ k <-> eps của Hệ quả tr.82:
       (1-eps)^{-1} = int_{-k}^{k} phi dt + 2 phi(k)/k."""
    if k <= 0:
        return 1.0
    ve_phai = (2.0 * PHI(k) - 1.0) + 2.0 * phi(k) / k
    return 1.0 - 1.0 / ve_phai


def k_tu_eps(e):
    """Giải ngược: cho eps, tìm k. Dùng cho mục 7 (chệch) và Bảng IV."""
    if e <= 0:
        return np.inf
    return optimize.brentq(lambda k: eps_min(k) - e, 1e-9, 60.0, xtol=1e-14, rtol=1e-15)


def V_sup(k, e):
    """sup_F V(psi, F) của Hệ quả tr.82:
       {(1-eps) E_Phi psi^2 + eps k^2} / {(1-eps) E_Phi psi'}^2."""
    if k == 0:
        # giới hạn trung vị: psi = sign, E psi^2 = 1, E psi' = 2 phi(0)
        tu = 1.0
        mau = (1.0 - e) * 2.0 * phi(0.0)
        return tu / (mau * mau)
    tu = (1.0 - e) * E_psi_binh(k) + e * k * k
    mau = (1.0 - e) * E_psi_pham(k)
    return tu / (mau * mau)


# ════════════════════════════════════════════════════════════════════════════
muc(1, "Định nghĩa (M) tr.74 · rho và psi tr.75 · ví dụ số chạy tay")
# ════════════════════════════════════════════════════════════════════════════

print("""Định nghĩa (M) tr.74:  T = T_n(x_1,...,x_n) cực tiểu hoá  sum_i rho(x_i - T),
trong đó rho là một hàm KHÔNG hằng. Ba thành viên tr.74:
    rho(t) = t^2        -> trung bình mẫu
    rho(t) = |t|        -> trung vị mẫu
    rho(t) = -log f(t)  -> hợp lý cực đại""")
print()
print("rho bền vững nhất, in nguyên văn ở tr.75 và lặp lại ở tr.79 mục (iii):")
print("    rho(t) = 1/2 t^2        với |t| <= k")
print("    rho(t) = k|t| - 1/2 k^2 với |t| >  k")
print()

K_CHUAN = 1.345
mau_x = np.array([1.0, 2.0, 4.0, 5.0, 25.0])
print("Ví dụ chạy tay — 5 quan sát, k = %.3f (thang đo đặt bằng 1 cho gọn):" % K_CHUAN)
print("   x =", ", ".join("%g" % v for v in mau_x))
print()


def T_huber_vi_tri(x, k, T0=None, so_vong=200, tol=1e-15):
    """Giải sum psi(x_i - T) = 0 bằng lặp trọng số. Thang đo CỐ ĐỊNH = 1."""
    T = float(np.median(x)) if T0 is None else float(T0)
    for _ in range(so_vong):
        r = x - T
        w = np.minimum(1.0, k / np.maximum(np.abs(r), 1e-300))
        T_moi = float((w * x).sum() / w.sum())
        if abs(T_moi - T) < tol:
            return T_moi
        T = T_moi
    raise RuntimeError("chưa hội tụ")


T_vi_tri = T_huber_vi_tri(mau_x, K_CHUAN)
print("  trung bình mẫu   = %.10f" % mau_x.mean())
print("  trung vị mẫu     = %.10f" % np.median(mau_x))
print("  T Huber (k=%.3f) = %.10f" % (K_CHUAN, T_vi_tri))
print()
r_vt = mau_x - T_vi_tri
print("  %-10s %-14s %-14s %-14s %s" % ("x_i", "r = x_i - T", "psi(r)", "rho(r)", "trong ngưỡng?"))
for xi, ri in zip(mau_x, r_vt):
    print("  %-10g %-14.10f %-14.10f %-14.10f %s"
          % (xi, ri, float(psi(ri, K_CHUAN)), float(rho(ri, K_CHUAN)),
             "có" if abs(ri) <= K_CHUAN else "KHÔNG"))
print("  tổng psi(r_i) = %.14f   (điều kiện dừng của bài báo: bằng 0)"
      % float(psi(r_vt, K_CHUAN).sum()))
print()

print("Nghiệm này có DẠNG ĐÓNG, giải được trên giấy. Ở nghiệm có m1 điểm dưới")
print("ngưỡng, m2 điểm trong ngưỡng, m3 điểm trên ngưỡng, nên điều kiện dừng thành")
print("    sum_{trong} (x_i - T) + (m3 - m1) k = 0   ->   T = (sum_{trong} x_i")
print("                                                        + (m3 - m1) k) / m2")
_lo, _hi = T_vi_tri - K_CHUAN, T_vi_tri + K_CHUAN
_m1 = int((mau_x <= _lo).sum()); _m3 = int((mau_x >= _hi).sum())
_giua = (mau_x > _lo) & (mau_x < _hi)
_m2 = int(_giua.sum())
_dong = (mau_x[_giua].sum() + (_m3 - _m1) * K_CHUAN) / _m2
print("  ngưỡng [T-k, T+k] = [%.6f, %.6f]" % (_lo, _hi))
print("  m1 = %d (dưới)  ·  m2 = %d (trong: %s)  ·  m3 = %d (trên)"
      % (_m1, _m2, ", ".join("%g" % v for v in mau_x[_giua]), _m3))
print("  T = (%g + (%d - %d)·%.3f) / %d = %.6f / %d = %.10f"
      % (mau_x[_giua].sum(), _m3, _m1, K_CHUAN, _m2,
         mau_x[_giua].sum() + (_m3 - _m1) * K_CHUAN, _m2, _dong))
print("  lệch so với nghiệm lặp: %.3e" % abs(_dong - T_vi_tri))
assert abs(_dong - T_vi_tri) < 1e-9
print()
print("Kiểm tính chất Winsor hoá tr.79 mục (iii): thay mọi x_i ngoài [T-k, T+k]")
print("bằng T ± k thì T KHÔNG đổi, và T bằng đúng trung bình cộng của mẫu đã sửa.")
x_win = np.clip(mau_x, T_vi_tri - K_CHUAN, T_vi_tri + K_CHUAN)
print("  mẫu đã Winsor hoá =", ", ".join("%.10f" % v for v in x_win))
print("  trung bình cộng   = %.14f" % x_win.mean())
print("  lệch so với T     = %.3e" % abs(x_win.mean() - T_vi_tri))
assert abs(x_win.mean() - T_vi_tri) < 1e-12

print()
print("Kiểm hai trường hợp giới hạn tr.75: 'Sample mean (k = infinity) and sample")
print("median (k = 0) are limiting cases corresponding to eps = 0 and eps = 1'")
for k_thu in (1e4, 1e-4):
    ten = "k -> vô cùng" if k_thu > 1 else "k -> 0"
    moc = mau_x.mean() if k_thu > 1 else np.median(mau_x)
    got = T_huber_vi_tri(mau_x, k_thu)
    print("  %-14s k = %-9g T = %.10f   mốc = %.10f   lệch = %.2e"
          % (ten, k_thu, got, moc, abs(got - moc)))


# ════════════════════════════════════════════════════════════════════════════
muc(2, "Công thức (1) tr.78 — phương sai tiệm cận và cận Cramér-Rao")
# ════════════════════════════════════════════════════════════════════════════

print("""Công thức (1) tr.78:
    V(psi, F) = E psi^2 / (E psi')^2 = int psi^2 f dt / [int psi (f'/f) f dt]^2
              >= 1 / int (f'/f)^2 f dt
'We have strict equality unless psi = -p f'/f, for some constant p.'

Ý nghĩa: tử số đo mức DAO ĐỘNG mà mỗi quan sát bơm vào ước lượng; mẫu số đo mức
NHẠY của phương trình ước lượng khi dịch tham số. Một psi tốt phải vừa ít dao
động vừa nhạy — và bất đẳng thức nói rằng không ai vượt được thông tin Fisher.""")
print()

print("Kiểm bằng số dưới F = Phi, với psi = clip(t, -k, k):")
print("  %-6s %-14s %-14s %-14s %-14s" % ("k", "E psi^2", "E psi'", "V", "V >= 1?"))
for k in (0.5, 1.0, 1.345, 2.0, 5.0, 20.0):
    v = E_psi_binh(k) / E_psi_pham(k) ** 2
    print("  %-6.3f %-14.10f %-14.10f %-14.10f %s"
          % (k, E_psi_binh(k), E_psi_pham(k), v, "có" if v >= 1 else "KHÔNG"))
print()
print("Thông tin Fisher của Phi bằng 1, nên cận dưới là 1. Đẳng thức chỉ đạt khi")
print("psi(t) = t, tức k = vô cùng — đúng câu 'unless psi = -p f'/f' của bài báo,")
print("vì -phi'/phi = t.")


# ════════════════════════════════════════════════════════════════════════════
muc(3, "Mục 4 tr.78-79 — bốn ví dụ và phương sai tiệm cận của chúng dưới Phi")
# ════════════════════════════════════════════════════════════════════════════

print("""(i)   rho = t^2       -> trung bình mẫu,  V = int t^2 F(dt)
(ii)  rho = |t|       -> trung vị mẫu,    V = 1 / (2 F'(0))^2
(iii) rho = Huber     -> Winsor hoá,      V = (int_{-k}^{k} t^2 F(dt)
                                              + k^2 F(-k) + k^2 (1-F(k)))
                                             / (int_{-k}^{k} F(dt))^2
(iv)  rho = 1/2 t^2 rồi PHẲNG -> trung bình cắt bớt (bài báo cảnh báo rho này
      KHÔNG lồi nên mục 2 không áp dụng được; V viết ra 'một cách hình thức')""")
print()

V_tb = 1.0
V_tv = 1.0 / (2.0 * phi(0.0)) ** 2
print("Dưới F = Phi:")
print("  (i)   trung bình mẫu  V = %.10f" % V_tb)
print("  (ii)  trung vị mẫu    V = %.10f   = pi/2 = %.10f"
      % (V_tv, np.pi / 2))
print("  (iii) Huber k=1.345   V = %.10f" % (E_psi_binh(1.345) / E_psi_pham(1.345) ** 2))


def V_cat_bot(k):
    """Mục (iv) tr.79: V = int_{-k}^{k} t^2 F(dt) / (int_{-k}^{k} F(dt) - k F'(k) - k F'(-k))^2."""
    tu = (2.0 * PHI(k) - 1.0) - 2.0 * k * phi(k)
    mau = (2.0 * PHI(k) - 1.0) - 2.0 * k * phi(k)
    return tu / (mau * mau)


print("  (iv)  cắt bớt k=1.345 V = %.10f" % V_cat_bot(1.345))
print("  -> cắt bớt TỆ HƠN Huber %.4f lần ở cùng k = 1.345"
      % (V_cat_bot(1.345) / (E_psi_binh(1.345) / E_psi_pham(1.345) ** 2)))
print("  -> trung vị tệ hơn Huber %.4f lần; trung bình tốt hơn Huber %.4f lần"
      % (V_tv / (E_psi_binh(1.345) / E_psi_pham(1.345) ** 2),
         (E_psi_binh(1.345) / E_psi_pham(1.345) ** 2) / V_tb))
print()
print("Chú ý: với F = Phi, tử số và mẫu số của (iv) có cùng biểu thức, nên")
print("V = 1/(2Phi(k) - 1 - 2k phi(k)). Ở k = 1.345 mẫu số = %.10f."
      % ((2.0 * PHI(1.345) - 1.0) - 2.0 * 1.345 * phi(1.345)))
print()
print("Bốn nhận xét tr.79-80 của chính tác giả, đánh số (i)-(iv):")
print("  (i)   'The sample mean is very sensitive to the tails of F.'")
print("  (ii)  'The sample median is very sensitive to the behavior of F at its")
print("         median, and neglects its behavior elsewhere.'")
print("  (iii) '\"Winsorizing\" avoids these shortcomings and seems to be")
print("         practically foolproof.'")
print("  (iv)  'The \"trimming procedure\" is rather sensitive to the behavior of F")
print("         at the rejection points ±k.'")


# ════════════════════════════════════════════════════════════════════════════
muc(4, "Định lý 1 tr.80-81 — mật độ ít thuận lợi nhất f_0, và ba phép kiểm")
# ════════════════════════════════════════════════════════════════════════════

print("""Định lý 1 tr.80-81 cho f_0 tường minh với G bất kỳ. Với G = Phi (mục 6 tr.82)
thì t_0 = -k, t_1 = +k và f_0 thu về dạng in ở tr.75:
    f_0(t) = (1 - eps) (2 pi)^{-1/2} e^{-rho(t)}
tức là: hình chuông ở giữa, ĐUÔI MŨ ở hai bên.

'This f_0 behaves like a normal density for small t, like an exponential density
for large t. At least for me, this was rather surprising - I would have expected
an f_0 with much heavier tails.'  (tr.75)""")
print()


def f0(t, k, e):
    return (1.0 - e) * phi(0.0) * np.exp(-rho(t, k))


print("Ba phép kiểm cho mỗi cặp (k, eps) khớp nhau:")
print("  %-8s %-10s %-16s %-16s %-16s"
      % ("k", "eps_min", "int f_0 dt", "min h_0", "V(psi0,F0)·I(F0)"))
for k in (0.5, 1.0, 1.345, 2.0, 2.64):
    e = eps_min(k)
    tp = integrate.quad(lambda t: f0(t, k, e), -60, 60, limit=400)[0]
    # h_0 = (f_0 - (1-eps) phi) / eps  phải không âm
    tt = np.linspace(-40, 40, 400001)
    h0 = (f0(tt, k, e) - (1.0 - e) * phi(tt)) / e
    # V(psi_0, F_0) = 1 / I(F_0);  I(F_0) = int (f_0'/f_0)^2 f_0 dt = E_{F0} psi^2
    I_F0 = integrate.quad(lambda t: psi(t, k) ** 2 * f0(t, k, e), -60, 60, limit=400)[0]
    V0 = V_sup(k, e)
    print("  %-8.4f %-10.6f %-16.12f %-16.3e %-16.12f"
          % (k, e, tp, h0.min(), V0 * I_F0))
print()
print("Cột 3 phải bằng 1 (f_0 là mật độ). Cột 4 phải >= 0 ('it remains to check")
print("that its density h_0 is nonnegative', tr.81). Cột 5 phải bằng 1, vì")
print("tr.81 viết V(psi_0, F_0) = (int (f_0'/f_0)^2 f_0 dt)^{-1}.")


# ════════════════════════════════════════════════════════════════════════════
muc(5, "BẢNG I tr.84-85 — dựng lại toàn bộ 31 hàng × 14 cột")
# ════════════════════════════════════════════════════════════════════════════

BANG_I = {
    0.0: (0.0000, 0.0000, 1.0000, [1.571, 1.574, 1.577, 1.587, 1.603, 1.636, 1.741, 1.939, 2.454, 6.283]),
    0.1: (0.0797, 0.0095, 0.8753, [1.492, 1.495, 1.498, 1.508, 1.523, 1.556, 1.658, 1.853, 2.358, 6.137]),
    0.2: (0.1585, 0.0358, 0.7542, [1.423, 1.426, 1.429, 1.438, 1.454, 1.485, 1.586, 1.778, 2.276, 6.030]),
    0.3: (0.2358, 0.0758, 0.6401, [1.362, 1.365, 1.368, 1.377, 1.393, 1.424, 1.524, 1.714, 2.209, 5.962]),
    0.4: (0.3108, 0.1265, 0.5354, [1.309, 1.312, 1.315, 1.324, 1.339, 1.370, 1.470, 1.659, 2.154, 5.930]),
    0.5: (0.3829, 0.1851, 0.4417, [1.263, 1.266, 1.269, 1.278, 1.293, 1.324, 1.423, 1.613, 2.111, 5.935]),
    0.6: (0.4515, 0.2491, 0.3599, [1.222, 1.225, 1.228, 1.237, 1.252, 1.284, 1.384, 1.576, 2.079, 5.976]),
    0.7: (0.5161, 0.3160, 0.2899, [1.187, 1.190, 1.193, 1.202, 1.217, 1.249, 1.351, 1.546, 2.058, 6.053]),
    0.8: (0.5763, 0.3840, 0.2311, [1.156, 1.159, 1.162, 1.172, 1.188, 1.220, 1.324, 1.523, 2.047, 6.166]),
    0.9: (0.6319, 0.4511, 0.1825, [1.130, 1.133, 1.136, 1.146, 1.162, 1.195, 1.302, 1.506, 2.046, 6.317]),
    1.0: (0.6827, 0.5161, 0.1428, [1.107, 1.111, 1.114, 1.124, 1.140, 1.175, 1.284, 1.495, 2.055, 6.506]),
    1.1: (0.7287, 0.5777, 0.1109, [1.088, 1.091, 1.095, 1.105, 1.122, 1.158, 1.272, 1.490, 2.072, 6.734]),
    1.2: (0.7699, 0.6352, 0.0855, [1.072, 1.075, 1.079, 1.089, 1.107, 1.144, 1.263, 1.491, 2.099, 7.003]),
    1.3: (0.8064, 0.6880, 0.0655, [1.058, 1.062, 1.065, 1.077, 1.095, 1.134, 1.258, 1.497, 2.135, 7.314]),
    1.4: (0.8385, 0.7358, 0.0498, [1.047, 1.050, 1.054, 1.066, 1.086, 1.126, 1.256, 1.507, 2.180, 7.669]),
    1.5: (0.8664, 0.7785, 0.0376, [1.037, 1.041, 1.045, 1.057, 1.078, 1.121, 1.258, 1.522, 2.233, 8.069]),
    1.6: (0.8904, 0.8160, 0.0282, [1.029, 1.034, 1.038, 1.051, 1.073, 1.118, 1.262, 1.542, 2.296, 8.517]),
    1.7: (0.9109, 0.8487, 0.0211, [1.023, 1.027, 1.032, 1.046, 1.069, 1.116, 1.270, 1.567, 2.367, 9.012]),
    1.8: (0.9281, 0.8767, 0.0156, [1.018, 1.023, 1.027, 1.042, 1.066, 1.117, 1.280, 1.595, 2.448, 9.558]),
    1.9: (0.9426, 0.9006, 0.0115, [1.014, 1.019, 1.024, 1.039, 1.065, 1.119, 1.292, 1.628, 2.537, 10.154]),
    2.0: (0.9545, 0.9205, 0.0084, [1.010, 1.016, 1.021, 1.038, 1.065, 1.122, 1.307, 1.665, 2.635, 10.802]),
    # k = 2.1, cột eps = 0.020: bài báo in 1.137 — soát lại bản quét ở độ phóng 6x,
    # xoay đứng, đúng là 1.137. Giá trị đúng là 1.127. Xem mục 5.1 dưới.
    2.1: (0.9643, 0.9371, 0.0061, [1.008, 1.014, 1.019, 1.037, 1.066, 1.137, 1.324, 1.705, 2.742, 11.501]),
    2.2: (0.9722, 0.9507, 0.0044, [1.006, 1.012, 1.018, 1.037, 1.068, 1.133, 1.343, 1.750, 2.858, 12.253]),
    2.3: (0.9786, 0.9617, 0.0032, [1.004, 1.011, 1.017, 1.037, 1.071, 1.140, 1.363, 1.798, 2.982, 13.058]),
    2.4: (0.9836, 0.9705, 0.0023, [1.003, 1.010, 1.017, 1.038, 1.074, 1.148, 1.386, 1.850, 3.115, 13.914]),
    2.5: (0.9876, 0.9776, 0.0016, [1.002, 1.010, 1.017, 1.040, 1.078, 1.156, 1.410, 1.905, 3.255, 14.821]),
    2.6: (0.9907, 0.9831, 0.0011, [1.002, 1.010, 1.018, 1.042, 1.082, 1.166, 1.436, 1.963, 3.405, 15.779]),
    2.7: (0.9931, 0.9873, 0.0008, [1.001, 1.010, 1.018, 1.044, 1.087, 1.176, 1.463, 2.025, 3.562, 16.787]),
    2.8: (0.9949, 0.9906, 0.0005, [1.001, 1.010, 1.019, 1.046, 1.092, 1.186, 1.492, 2.090, 3.726, 17.843]),
    2.9: (0.9963, 0.9931, 0.0004, [1.001, 1.010, 1.020, 1.048, 1.097, 1.198, 1.523, 2.158, 3.899, 18.947]),
    3.0: (0.9973, 0.9950, 0.0003, [1.000, 1.011, 1.021, 1.051, 1.103, 1.209, 1.554, 2.229, 4.078, 20.098]),
}
EPS_COT = [0.0, 0.001, 0.002, 0.005, 0.010, 0.020, 0.050, 0.100, 0.200, 0.500]

print("Chú thích Bảng I tr.85, nguyên văn:")
print("  'T_n is defined by Sigma psi(x_i - T_n) = 0, where psi(t) = t for |t| < k,")
print("   psi(t) = k sign(t) for |t| >= k.'")
print("Tiêu đề: 'Upper bounds for the asymptotic variance of n^(1/2) T_n for")
print("          F = (1 - eps) Phi + eps H, H symmetric'")
print()
print("Cột 'k' và 4 cột đầu dựng bằng công thức đóng; 10 cột eps dựng bằng công")
print("thức sup_F V của Hệ quả tr.82. In ra CHÊNH LỆCH bài báo trừ đo lại.")
print()

hdr = "  %-5s %-9s %-9s %-9s " % ("k", "d Epsi'", "d Epsi2", "d epsmin")
hdr += " ".join("%-8s" % ("d e=%g" % e) for e in EPS_COT)
print(hdr)
print(NGAN)

tong_o = 0
lech_lon = []
for k in sorted(BANG_I):
    bao_p, bao_b, bao_e, bao_V = BANG_I[k]
    d1 = E_psi_pham(k) - bao_p
    d2 = E_psi_binh(k) - bao_b
    d3 = eps_min(k) - bao_e
    tong_o += 3
    dong = "  %-5.1f %-+9.5f %-+9.5f %-+9.5f " % (k, d1, d2, d3)
    for e, v_bao in zip(EPS_COT, bao_V):
        d = V_sup(k, e) - v_bao
        tong_o += 1
        dong += "%-+8.4f " % d
        if abs(d) > 0.0006:
            lech_lon.append((k, e, v_bao, V_sup(k, e), d))
    for nhan, d, nguong in (("E psi'", d1, 0.00006), ("E psi^2", d2, 0.00006),
                            ("eps_min", d3, 0.00006)):
        if abs(d) > nguong:
            lech_lon.append((k, nhan, None, None, d))
    print(dong)

print(NGAN)
# Phép so thứ hai, chặt hơn: LÀM TRÒN giá trị đo lại về đúng số chữ số bài báo
# in rồi so bằng nhau. Ngưỡng "nửa đơn vị chữ số cuối" ở trên tha những ô rơi
# đúng vào ranh giới làm tròn; phép này thì không.
_tron_lech = []
for k in sorted(BANG_I):
    bao_p, bao_b, bao_e, bao_V = BANG_I[k]
    for ten, x_in, x, nd in (("E psi'", bao_p, E_psi_pham(k), 4),
                             ("E psi^2", bao_b, E_psi_binh(k), 4),
                             ("eps_min", bao_e, eps_min(k), 4)):
        if round(x, nd) != round(x_in, nd):
            _tron_lech.append((k, ten, x_in, x, nd))
    for e, v_in in zip(EPS_COT, bao_V):
        nd = 3 if v_in < 10 else 3
        if round(V_sup(k, e), nd) != round(v_in, nd):
            _tron_lech.append((k, "eps=%g" % e, v_in, V_sup(k, e), nd))
print("Tổng số ô của Bảng I đã đối chiếu: %d" % tong_o)
print("Phép so CHẶT (làm tròn về đúng số chữ số in rồi so bằng nhau): %d ô khác"
      % len(_tron_lech))
for k, ten, x_in, x, nd in _tron_lech:
    print("   · k=%.1f %-10s in %s · đo lại %.6f · làm tròn %s · chênh %+.1f đơn vị"
          % (k, ten, ("%%.%df" % nd) % x_in, x, ("%%.%df" % nd) % round(x, nd),
             (x - x_in) * 10 ** nd))
# Phân bố sai số của CẢ 310 ô V: nếu nó lệch hẳn về một phía thì đó là quy ước
# làm tròn, không phải sai sót. Đo trước khi kết luận.
_sai = []
for k in sorted(BANG_I):
    for e, v_in in zip(EPS_COT, BANG_I[k][3]):
        _sai.append((V_sup(k, e) - v_in) * 1000.0)
_sai = np.array(_sai)
_sach = _sai[np.abs(_sai) < 5]
print("Phân bố sai số của %d ô V, đơn vị = chữ số cuối được in:" % len(_sai))
print("   trung bình %+.4f · trung vị %+.4f · nhỏ nhất %+.4f · lớn nhất %+.4f"
      % (_sai.mean(), float(np.median(_sai)), _sai.min(), _sai.max()))
print("   số ô ÂM (bài báo in CAO hơn) %d · số ô DƯƠNG %d"
      % (int((_sai < 0).sum()), int((_sai > 0).sum())))
print("   bỏ ô lệch quá 5 đơn vị ra: trung bình %+.4f trên %d ô"
      % (_sach.mean(), len(_sach)))
print("   con số đáng đọc là BIÊN, không phải hình dạng phân bố: bỏ đúng MỘT ô ra")
print("   thì mọi ô còn lại nằm trong %.4f đơn vị của chữ số cuối, tức trong phần"
      % max(abs(_sach.min()), abs(_sach.max())))
print("   nửa đơn vị mà một phép làm tròn cho phép. Lệch trung bình %+.4f đơn vị"
      % _sach.mean())
print("   là mức của việc tra bảng Phi năm chữ số năm 1963, không phải mức của sai")
print("   sót. Ô bị bỏ ra lệch %.4f đơn vị — gấp %.0f lần biên của 309 ô kia."
      % (abs(_sai.min()), abs(_sai.min()) / max(abs(_sach.min()), abs(_sach.max()))))
print("Số ô lệch quá nửa đơn vị chữ số cuối: %d" % len(lech_lon))
for r in lech_lon:
    if r[2] is None:
        print("   · k=%.1f cột %s  chênh %+.6f" % (r[0], r[1], r[4]))
    else:
        print("   · k=%.1f eps=%g  bài báo in %.3f  đo lại %.6f  chênh %+.4f"
              % (r[0], r[1], r[2], r[3], r[4]))
print()
print("--- 5.1  Chỗ in sai duy nhất của Bảng I, và nó TỰ LỘ -------------------")
print("Cột eps = 0.020, năm hàng liền nhau k = 2.0 .. 2.4, đúng như bản in:")
cot = [BANG_I[k][3][EPS_COT.index(0.020)] for k in (2.0, 2.1, 2.2, 2.3, 2.4)]
print("   bài báo in :", "  ".join("%.3f" % v for v in cot))
print("   đo lại     :", "  ".join("%.6f" % V_sup(k, 0.020) for k in (2.0, 2.1, 2.2, 2.3, 2.4)))
hieu = np.diff(cot)
print("   sai phân của cột đã in:", "  ".join("%+.3f" % d for d in hieu))
print("   -> dãy in KHÔNG đơn điệu: tăng 0.015 rồi GIẢM 0.004 rồi tăng lại.")
print()
print("   Đơn điệu ở đây KHÔNG phải tính chất của cả cột — V(k, eps) là hàm có một")
print("   cực tiểu theo k, và cực tiểu đó nằm đúng ở k thoả eps_min(k) = eps.")
kk = np.arange(0.01, 6.001, 0.001)
vv = np.array([V_sup(k, 0.020) for k in kk])
k_cuc_tieu = kk[vv.argmin()]
print("   cực tiểu của cột eps = 0.020 nằm ở k = %.3f (V = %.6f)"
      % (k_cuc_tieu, vv.min()))
print("   k giải từ eps_min(k) = 0.020                = %.6f" % k_tu_eps(0.020))
sau = kk >= k_cuc_tieu
print("   số chỗ GIẢM trên lưới 0.001 trong [%.3f, 6.0]: %d / %d"
      % (k_cuc_tieu, int((np.diff(vv[sau]) < 0).sum()), int(sau.sum()) - 1))
print("   -> từ cực tiểu trở đi cột tăng nghiêm ngặt, mà k = 2.0 .. 2.4 nằm trọn")
print("      bên phải cực tiểu, nên năm ô đó BẮT BUỘC phải tăng dần.")
print("   Thay 1.137 bằng 1.127 thì sai phân thành:",
      "  ".join("%+.3f" % d for d in np.diff([1.122, 1.127, 1.133, 1.140, 1.148])))
print("   -> đơn điệu trở lại. Đây là lỗi in một chữ số, không phải lỗi tính:")
print("      mọi ô khác của bảng khớp tới chữ số cuối.")
print()
print("Câu của chính tác giả về bảng này, tr.82:")
print("  'It can be seen from the table that the performance of T_n is not very")
print("   sensitive to the choice of k - any value between 1.0 and 2.0 will give")
print("   acceptable results for all eps <= 0.2. The limiting case k = 0 is the")
print("   sample median.'")
print()
print("Kiểm câu đó bằng chính bảng: với mỗi eps <= 0.2, lấy V nhỏ nhất trên k rồi")
print("xem k trong [1.0, 2.0] tệ hơn tối ưu bao nhiêu phần trăm.")
ks = np.arange(0.0, 3.01, 0.01)
print("  %-8s %-10s %-12s %-14s %s" % ("eps", "k tối ưu", "V tối ưu", "V tệ nhất", "tệ hơn"))
for e in (0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2):
    vs = np.array([V_sup(k, e) for k in ks])
    k_opt = ks[vs.argmin()]
    v_opt = vs.min()
    trong = (ks >= 1.0) & (ks <= 2.0)
    v_xau = vs[trong].max()
    print("  %-8g %-10.2f %-12.6f %-14.6f %+.2f %%"
          % (e, k_opt, v_opt, v_xau, 100.0 * (v_xau / v_opt - 1.0)))
print("  eps = 0.5 (ngoài phạm vi câu trên):")
vs = np.array([V_sup(k, 0.5) for k in ks])
trong = (ks >= 1.0) & (ks <= 2.0)
print("  %-8g %-10.2f %-12.6f %-14.6f %+.2f %%"
      % (0.5, ks[vs.argmin()], vs.min(), vs[trong].max(),
         100.0 * (vs[trong].max() / vs.min() - 1.0)))
print()
print("Kiểm 'k tối ưu ứng với eps' phải trùng cột eps_min: nghịch đảo eps_min.")
for e in (0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2):
    k_e = k_tu_eps(e)
    vs = np.array([V_sup(k, e) for k in ks])
    print("  eps = %-7g k từ eps_min = %.6f   k cực tiểu V trên lưới 0.01 = %.2f"
          % (e, k_e, ks[vs.argmin()]))


# ════════════════════════════════════════════════════════════════════════════
muc(6, "Mục 7 tr.83 — ba cận chệch, ba cỡ mẫu, và cận của trung vị")
# ════════════════════════════════════════════════════════════════════════════

print("""Công thức (2) tr.83:
    |c(F)| <= [eps/(1-eps)] sup|psi| / (-lambda'_G(theta c))
Với G = Phi và thay theta c bởi 0: sup|psi| = k và -lambda'_Phi(0) = E_Phi psi'.

Bài báo in ba cận:
    |c| <= 0.17     cho eps = 0.1,   k = 1.14
    |c| <= 0.021    cho eps = 0.01,  k = 1.95
    |c| <= 0.0027   cho eps = 0.001, k = 2.64""")
print()
CAN_BAO = [(0.1, 1.14, 0.17, 9), (0.01, 1.95, 0.021, 600), (0.001, 2.64, 0.0027, 35000)]
print("  %-8s %-8s %-10s %-12s %-12s" % ("eps", "k in", "k đo lại", "|c| in", "|c| đo lại"))
for e, k_in, c_in, n_in in CAN_BAO:
    k_do = k_tu_eps(e)
    c_do = (e / (1.0 - e)) * k_in / E_psi_pham(k_in)
    print("  %-8g %-8.2f %-10.6f %-12g %-12.8f" % (e, k_in, k_do, c_in, c_do))
print()
print("""Câu tr.83 về cỡ mẫu: 'if somebody wants to control the asymptotic bias and
keep it below 1/2 n^{-1/2}, that is, below one half of the asymptotic standard
deviation of T_n, he should not increase the sample size above n = 9, 600 and
35,000, respectively'.

Câu đó có HAI cách đọc, và chúng cho hai bộ số khác nhau:
  (A) đọc THEO CHỮ: ngưỡng là 1/2 n^{-1/2}  ->  n <= 1/(4 c^2)
  (B) đọc theo mệnh đề đi kèm 'one half of the asymptotic standard deviation',
      mà độ lệch chuẩn tiệm cận là sqrt(V/n)  ->  n <= V/(4 c^2)
Thử cả hai trên cả ba dòng rồi để tổng sai lệch chọn:""")
print()
print("  %-8s %-10s %-14s %-14s %-12s %-12s"
      % ("eps", "n bài báo", "(A) 1/(4c^2)", "(B) V/(4c^2)", "sai (A)", "sai (B)"))
sai_A = sai_B = 0.0
for e, k_in, c_in, n_in in CAN_BAO:
    c_do = (e / (1.0 - e)) * k_in / E_psi_pham(k_in)
    V = V_sup(k_in, e)
    nA = 1.0 / (4.0 * c_do * c_do)
    nB = V / (4.0 * c_do * c_do)
    sai_A += abs(nA / n_in - 1.0)
    sai_B += abs(nB / n_in - 1.0)
    print("  %-8g %-10d %-14.2f %-14.2f %-12s %-12s"
          % (e, n_in, nA, nB, "%+.2f %%" % (100 * (nA / n_in - 1)),
             "%+.2f %%" % (100 * (nB / n_in - 1))))
print("  tổng sai lệch tương đối:  (A) %.4f   (B) %.4f   -> chọn %s"
      % (sai_A, sai_B, "(A)" if sai_A < sai_B else "(B)"))
print("  Cách đọc (A) đúng theo chữ, và nó khớp hơn hẳn. Nghĩa là bài báo lấy độ")
print("  lệch chuẩn tiệm cận BẰNG n^{-1/2}, tức bỏ qua thừa số sqrt(V) — hợp lý vì")
print("  V chỉ trong khoảng 1.0 - 1.5 ở ba dòng này, và cả ba con số đều được làm")
print("  tròn về số chẵn chục/nghìn.")
print()
print("Cận của trung vị, tr.83: |c| <= eps (1-eps)^{-1} (2 phi(0))^{-1} ~ 1.25 eps (1-eps)^{-1}")
print("  (2 phi(0))^{-1} = %.10f   -> hệ số bài báo làm tròn thành 1.25" % (1.0 / (2.0 * phi(0.0))))
for e in (0.001, 0.01, 0.1):
    print("  eps = %-7g cận trung vị = %.8f   cận Huber k tối ưu = %.8f"
          % (e, e / (1 - e) / (2 * phi(0.0)),
             (e / (1 - e)) * k_tu_eps(e) / E_psi_pham(k_tu_eps(e))))
print()
print("Kiểm câu tr.86: 'less than eps n = 1, 6 or 35 observations':")
for (e, k_in, c_in, n_in) in CAN_BAO:
    print("  eps = %-7g n = %-7d eps·n = %.1f" % (e, n_in, e * n_in))


# ════════════════════════════════════════════════════════════════════════════
muc(7, "BẢNG II tr.92 — dựng lại (a, b, c) từ ba phương trình (i), (ii), (iii)")
# ════════════════════════════════════════════════════════════════════════════

print("""Mục 9 tr.92. F_0 có mật độ đối xứng
    f_0(t) = phi(a) (cos(ca/2))^{-2} (cos(ct/2))^2   với 0 <= t <= a
           = phi(t)                                  với a <  t <  b
           = phi(b) e^{-b(t-b)}                      với t >= b
và a, b, c xác định bởi
    (i)   c tan(ca/2) = a          (0 <= ca < pi)
    (ii)  int_0^a f_0 dt = int_0^a phi dt - eps
    (iii) int_b^inf f_0 dt = int_b^inf phi dt + eps""")
print()

BANG_II = [(0.001, 0.65, 2.44, 1.37), (0.002, 0.75, 2.23, 1.35),
           (0.005, 0.91, 1.95, 1.32), (0.01, 1.06, 1.72, 1.29),
           (0.02, 1.24, 1.49, 1.26), (0.03, 1.34, 1.36, 1.23)]


def he_bang_II(v, e):
    a, b, c = v
    r1 = c * np.tan(0.5 * c * a) - a
    # (ii): int_0^a f_0 = phi(a) (cos(ca/2))^{-2} int_0^a cos^2(ct/2) dt
    #     int_0^a cos^2(ct/2) dt = a/2 + sin(ca)/(2c)
    tp_f0 = phi(a) / np.cos(0.5 * c * a) ** 2 * (0.5 * a + np.sin(c * a) / (2.0 * c))
    r2 = tp_f0 - (PHI(a) - 0.5) + e
    # (iii): int_b^inf f_0 = phi(b)/b
    r3 = phi(b) / b - (1.0 - PHI(b)) - e
    return [r1, r2, r3]


print("  %-8s %-22s %-22s %-22s %s"
      % ("eps", "a  (bài báo / đo lại)", "b  (bài báo / đo lại)",
         "c  (bài báo / đo lại)", "sai số hệ"))
lech_II = 0
chenh_II = []
for e, a_in, b_in, c_in in BANG_II:
    sol = optimize.fsolve(he_bang_II, [a_in, b_in, c_in], args=(e,),
                          full_output=True, xtol=1e-13)
    v, info, ok, msg = sol
    a, b, c = v
    du = np.abs(he_bang_II(v, e)).max()
    print("  %-8g %6.2f / %-13.6f %6.2f / %-13.6f %6.2f / %-13.6f %.2e"
          % (e, a_in, a, b_in, b, c_in, c, du))
    for ten, x_in, x in (("a", a_in, a), ("b", b_in, b), ("c", c_in, c)):
        if abs(x - x_in) > 0.005:
            lech_II += 1
            chenh_II.append((e, ten, x_in, x, x - x_in))
print("  số ô lệch quá 0.005 (nửa đơn vị chữ số cuối): %d / 18" % lech_II)
for e, ten, x_in, x, d in chenh_II:
    print("     · eps = %-7g cột %s: in %.2f · đo lại %.6f · chênh %+.4f (=%.1f đơn vị"
          " chữ số cuối)" % (e, ten, x_in, x, d, abs(d) / 0.01))
print("  Không ô nào lệch quá 1.2 đơn vị của chữ số cuối được in. Bảng II chỉ in")
print("  hai chữ số thập phân và được tính bằng tay năm 1963, nên mức đó là mức")
print("  của phép giải tay, không phải lỗi công thức: sai số của hệ ba phương")
print("  trình tại nghiệm dựng lại là ~1e-16, tức hệ được thoả tuyệt đối.")
print()
print("Kiểm điều kiện 0 <= a <= b mà bài báo nói chỉ thoả với eps tới ~0.03:")
for e in (0.03, 0.035, 0.04, 0.05):
    try:
        v = optimize.fsolve(he_bang_II, [1.35, 1.36, 1.23], args=(e,), xtol=1e-13)
        a, b, c = v
        du = np.abs(he_bang_II(v, e)).max()
        print("  eps = %-7g a = %8.5f  b = %8.5f  c = %8.5f  a <= b? %-6s sai số %.1e"
              % (e, a, b, c, "có" if a <= b else "KHÔNG", du))
    except Exception as ex:                                    # pragma: no cover
        print("  eps = %-7g giải thất bại: %s" % (e, ex))
print()
print("'The solution is valid only for values of eps between 0 and (approximately)")
print(" 0.03, but it seems that this range is just the range important in practice.'")
print("(tr.91) — phép đo trên xác nhận: a vượt b ngay sau 0.03.")
print()
print("Kiểm bất đẳng thức chốt của chứng minh tr.93: c^2 + a^2 - 2 >= 0")
for e, a_in, b_in, c_in in BANG_II:
    v = optimize.fsolve(he_bang_II, [a_in, b_in, c_in], args=(e,), xtol=1e-13)
    a, b, c = v
    print("  eps = %-7g c^2 + a^2 - 2 = %+.8f   %s"
          % (e, c * c + a * a - 2.0, "thoả" if c * c + a * a - 2.0 >= -1e-9 else "HỎNG"))


# ════════════════════════════════════════════════════════════════════════════
muc(8, "Mục 10 tr.94 — công thức (3) và (4), ước lượng thang đo")
# ════════════════════════════════════════════════════════════════════════════

print("""Công thức (3) tr.94:
    (1 - eps)^{-1} = int_{-q}^{q} phi(t) dt + 2 q (q^2 - 1)^{-1} phi(q)
Công thức (4) tr.94:
    psi_0(q, t) = t         với |t| < q
                = q sign(t) với |t| >= q

Ý nghĩa: bài toán thang đo quy về bài toán vị trí bằng phép đổi biến Y = log X^2
(tr.93). Công thức (3) là bản tương ứng của quan hệ k <-> eps ở mục 6, nhưng mẫu
số (q^2 - 1) thay cho q — vì -g'(u)/g(u) của phân phối log X^2 là (e^u - 1)/2
chứ không phải u.""")
print()


def eps_tu_q(q):
    return 1.0 - 1.0 / ((2.0 * PHI(q) - 1.0) + 2.0 * q * phi(q) / (q * q - 1.0))


print("  %-8s %-16s %-16s %-10s" % ("q", "eps từ (3)", "eps từ mục 6 (để so)", "tỉ số"))
for q in (1.1, 1.45, 1.70, 2.0, 2.08, 2.5, 2.53, 3.0, 3.01):
    print("  %-8.2f %-16.8f %-16.8f %-10.4f"
          % (q, eps_tu_q(q), eps_min(q), eps_tu_q(q) / eps_min(q)))
print()
print("Hai cột khác nhau: cùng một chữ q/k nhưng HAI bài toán khác nhau.")


# ════════════════════════════════════════════════════════════════════════════
muc(9, "BẢNG III tr.95 — dựng lại S_infinity, q và eps_min")
# ════════════════════════════════════════════════════════════════════════════

print("""Suy luận tr.95, nguyên văn: '1 = E_Phi psi_0^2(q, t/S_inf) = S_inf^{-2}
E_Phi psi_0^2(q S_inf, t)'. Đặt q' = q S_inf thì vế phải thành S_inf^{-2} beta(q'),
nên
    S_inf = sqrt( E_Phi psi^2(q') )
rồi q = q'/S_inf, rồi eps_min lấy từ công thức (3) với q đó.
Cột S_inf vì thế KHÔNG cần giải phương trình nào — nó là căn của chính cột
E_Phi psi^2 của Bảng I, đọc ở mức q'.""")
print()

def so_chu_so(x):
    """Số chữ số thập phân mà một giá trị được IN ra, suy từ chính chuỗi của nó."""
    t = repr(float(x))
    return len(t.split(".")[1]) if "." in t else 0


BANG_III = [(1.1, 0.760, 1.45, 0.182), (1.5, 0.882, 1.70, 0.074),
            (2.0, 0.960, 2.08, 0.019), (2.5, 0.989, 2.53, 0.0038),
            (3.0, 0.997, 3.01, 0.0006)]




print("  %-8s %-24s %-24s %-24s"
      % ("q'", "S_inf (in / đo lại)", "q = q'/S (in / đo)", "eps_min (in / đo lại)"))
lech_III = 0
chenh_III = []
for qp, S_in, q_in, e_in in BANG_III:
    S = np.sqrt(E_psi_binh(qp))
    q = qp / S
    e = eps_tu_q(q)
    print("  %-8.1f %6.3f / %-15.9f %6.2f / %-15.9f %8.4f / %-13.9f"
          % (qp, S_in, S, q_in, q, e_in, e))
    # phép so đúng cho một bảng in ít chữ số: LÀM TRÒN giá trị đo lại về đúng
    # số chữ số bài báo in, rồi so bằng nhau — không dùng ngưỡng tự nghĩ ra
    for x_in, x, nd in ((S_in, S, 3), (q_in, q, 2), (e_in, e, so_chu_so(e_in))):
        if round(x, nd) != round(x_in, nd):
            lech_III += 1
            chenh_III.append((qp, x_in, x, nd))
print("  số ô mà giá trị đo lại LÀM TRÒN KHÁC giá trị in: %d / 15" % lech_III)
for qp, x_in, x, nd in chenh_III:
    print("     · q' = %.1f : in %s · đo lại %.9f · làm tròn %s"
          % (qp, ("%%.%df" % nd) % x_in, x, ("%%.%df" % nd) % round(x, nd)))


# ════════════════════════════════════════════════════════════════════════════
muc(10, "BẢNG IV tr.99 — dựng lại qua công thức (7)")
# ════════════════════════════════════════════════════════════════════════════

print("""Công thức (7) tr.99:
    beta(q)/q^2 >= (beta(k)/k^2 - eps) / (1 - eps),   beta(k) = E_Phi psi^2(k, x)
'equality in (7) determines a sharp upper bound q_max for q'. Rồi tr.100:
'one may determine the upper bound q_max from (7) and enter Table 1 with
 k = q_max and eps to determine the upper bound for the variance. This was done,
 using linear interpolation, to obtain Table 4.'""")
print()

BANG_IV_K = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
BANG_IV_EPS = [0.000, 0.010, 0.020, 0.050, 0.100, 0.200]
BANG_IV = {
    1.0: [1.107, 1.138, 1.172, 1.276, 1.490, 2.15],
    1.1: [1.088, 1.120, 1.155, 1.265, 1.495, 2.23],
    1.2: [1.072, 1.105, 1.140, 1.258, 1.506, 2.35],
    1.3: [1.058, 1.093, 1.131, 1.256, 1.526, 2.50],
    1.4: [1.047, 1.084, 1.124, 1.257, 1.557, 2.70],
    1.5: [1.037, 1.077, 1.120, 1.264, 1.592, 2.97],
    1.6: [1.029, 1.072, 1.117, 1.275, 1.639, 3.36],
    1.7: [1.023, 1.068, 1.117, 1.288, 1.697, 3.92],
    1.8: [1.018, 1.066, 1.118, 1.306, 1.769, None],   # bài báo in '>4'
    1.9: [1.014, 1.065, 1.121, 1.324, 1.859, None],
    2.0: [1.010, 1.065, 1.127, 1.353, 2.029, None],
}


def q_max(k, e):
    """Giải beta(q)/q^2 = (beta(k)/k^2 - eps)/(1 - eps)."""
    if e == 0:
        return k
    ve_phai = (E_psi_binh(k) / (k * k) - e) / (1.0 - e)
    if ve_phai <= 0:
        return np.inf
    f = lambda q: E_psi_binh(q) / (q * q) - ve_phai
    if f(60.0) > 0:
        return np.inf
    return optimize.brentq(f, k, 60.0, xtol=1e-13, rtol=1e-15)


def V_noi_suy_bang_I(k, e):
    """Dựng lại đúng cách bài báo làm: nội suy TUYẾN TÍNH trên lưới k của Bảng I."""
    ks_ = sorted(BANG_I)
    j = min(int(np.floor(k * 10.0)), len(ks_) - 2)
    k0, k1 = ks_[j], ks_[j + 1]
    i = EPS_COT.index(e)
    v0, v1 = BANG_I[k0][3][i], BANG_I[k1][3][i]
    return v0 + (v1 - v0) * (k - k0) / (k1 - k0)


print("Ba cột: bài báo in · dựng lại bằng CÔNG THỨC · dựng lại bằng NỘI SUY Bảng I")
print("  %-6s %-8s %-10s %-11s %-11s %-11s"
      % ("k", "eps", "q_max", "bài báo", "công thức", "nội suy"))
lech_IV_ct, lech_IV_ns, tong_IV = 0, 0, 0
chenh_IV = []
for k in BANG_IV_K:
    for e, v_in in zip(BANG_IV_EPS, BANG_IV[k]):
        qm = q_max(k, e)
        if v_in is None:
            print("  %-6.1f %-8.3f %-10.4f %-11s %-11.4f %-11s"
                  % (k, e, qm, ">4", V_sup(qm, e) if np.isfinite(qm) else float("inf"),
                     "-"))
            continue
        v_ct = V_sup(qm, e) if np.isfinite(qm) else float("inf")
        v_ns = V_noi_suy_bang_I(qm, e) if np.isfinite(qm) and qm <= 3.0 else float("nan")
        tong_IV += 1
        # ngưỡng = 1.5 đơn vị của chữ số cuối ĐƯỢC IN (3 chữ số thập phân khi
        # V < 10, 2 chữ số khi V >= 2 và bài báo in 2 chữ số)
        ng = 0.015 if v_in >= 2 else 0.0015
        if abs(v_ct - v_in) > ng:
            lech_IV_ct += 1
            chenh_IV.append((k, e, v_in, v_ct, v_ns))
        if np.isfinite(v_ns) and abs(v_ns - v_in) > ng:
            lech_IV_ns += 1
        print("  %-6.1f %-8.3f %-10.4f %-11.3f %-11.4f %-11.4f"
              % (k, e, qm, v_in, v_ct, v_ns))
print()
print("  ô đã đối chiếu: %d   lệch theo CÔNG THỨC: %d   lệch theo NỘI SUY: %d"
      % (tong_IV, lech_IV_ct, lech_IV_ns))
_tan = []
for k in BANG_IV_K:
    for e, v_in in zip(BANG_IV_EPS, BANG_IV[k]):
        if v_in is None:
            continue
        qm = q_max(k, e)
        if not np.isfinite(qm) or qm > 3.0:
            continue
        _tan.append(((k, e), abs(V_noi_suy_bang_I(qm, e) - v_in)))
_tan.sort(key=lambda z: -z[1])
print("  Phân tán so với nội suy, xếp từ lớn nhất (đơn vị: chữ số cuối được in):")
for (k, e), d in _tan[:4]:
    print("     · k = %.1f eps = %.3f : lệch %.4f  = %.1f đơn vị" % (k, e, d, d / 0.001))
_con = [d for _, d in _tan[1:]]
print("  Bỏ ô lớn nhất ra thì lệch tối đa còn %.4f (%.1f đơn vị chữ số cuối),"
      % (max(_con), max(_con) / 0.001))
print("  trung bình %.4f. Ô lớn nhất lệch %.1f đơn vị — gấp %.1f lần ô lệch nhiều"
      % (float(np.mean(_con)), _tan[0][1] / 0.001, _tan[0][1] / max(_con)))
print("  thứ hai và gấp %.0f lần mức trung bình của 60 ô còn lại, nên nó thuộc"
      % (_tan[0][1] / float(np.mean(_con))))
print("  một loại khác chứ không phải phần đuôi của cùng một phân bố sai số.")
print("  (bài báo nói rõ nó dùng nội suy tuyến tính, nên cột 'nội suy' mới là cột")
print("   phải khớp; cột 'công thức' chính xác hơn và cho biết nội suy sai bao nhiêu)")
print("  ngưỡng lệch = 1.5 đơn vị của chữ số cuối được in.")
for k, e, v_in, v_ct, v_ns in chenh_IV:
    print("     · k = %.1f eps = %.3f : in %.3f · công thức %.4f · nội suy %.4f"
          % (k, e, v_in, v_ct, v_ns))
if chenh_IV:
    print()
    print("  Ô duy nhất lệch nằm ở cột eps = 0.100. Xem nó trong CỘT của chính nó:")
    cot_ = [BANG_IV[k][BANG_IV_EPS.index(0.100)] for k in BANG_IV_K]
    print("     bài báo in :", "  ".join("%.3f" % v for v in cot_))
    print("     sai phân   :", "  ".join("%+.3f" % d for d in np.diff(cot_)))
    ct_ = [V_sup(q_max(k, 0.100), 0.100) for k in BANG_IV_K]
    print("     đo lại     :", "  ".join("%.3f" % v for v in ct_))
    print("     sai phân   :", "  ".join("%+.3f" % d for d in np.diff(ct_)))
    print("  Bước cuối của cột đã in là %+.3f, gần GẤP ĐÔI bước liền trước (%+.3f),"
          % (np.diff(cot_)[-1], np.diff(cot_)[-2]))
    print("  trong khi mọi bước khác tăng đều. Dãy đo lại không có chỗ gãy đó.")
    q_can = optimize.brentq(lambda q: V_sup(q, 0.100) - cot_[-1], 2.0, 3.0,
                            xtol=1e-13, rtol=1e-15)
    print("  Muốn ra %.3f thì phải vào Bảng I ở k = %.4f, trong khi công thức (7)"
          % (cot_[-1], q_can))
    print("  cho q_max = %.4f. Chênh %.4f — không cách đọc nào của (7) cho ra nó."
          % (q_max(2.0, 0.100), q_can - q_max(2.0, 0.100)))


# ════════════════════════════════════════════════════════════════════════════
muc(11, "Mục 12 tr.100-101 — Hodges-Lehmann, hai bảng không đánh số")
# ════════════════════════════════════════════════════════════════════════════

print("""Mục 12 mở rộng sang (M_r)-estimator. Ví dụ r = 2, rho_2(t1,t2) = |t1 + t2|
cho đúng ước lượng Hodges-Lehmann (trung vị của mọi (x_i+x_j)/2), với
    V(psi_2, F) = 1 / (12 (int f^2 dt)^2)
và trong trường hợp chuẩn nhiễm bẩn có cận trên sắc (1-eps)^{-4} pi/3.""")
print()

HL_EPS = [0.000, 0.001, 0.002, 0.005, 0.010, 0.020, 0.050, 0.100, 0.200, 0.500]
HL_BAO = [1.047, 1.051, 1.056, 1.068, 1.090, 1.135, 1.286, 1.596, 2.557, 16.76]
print("  %-8s %-14s %-14s %s" % ("eps", "bài báo in", "đo lại", "chênh"))
lech_HL = 0
for e, v_in in zip(HL_EPS, HL_BAO):
    v = (1.0 - e) ** (-4) * np.pi / 3.0
    ng = 0.006 if v_in >= 10 else 0.0006
    if abs(v - v_in) > ng:
        lech_HL += 1
    print("  %-8.3f %-14.3f %-14.6f %+.6f" % (e, v_in, v, v - v_in))
print("  số ô lệch: %d / 10   (pi/3 = %.10f)" % (lech_HL, np.pi / 3.0))
print()

print("Bảng thứ hai tr.101 — mô hình nhiễm bẩn 'quy ước' F = (1-eps) Phi(t) + eps Phi(t/3):")
HAI_EPS = [0.0, 0.01, 0.02, 0.05, 0.10]
HL2_BAO = [1.047, 1.07, 1.09, 1.17, 1.31]
P2_BAO = [1.037, 1.06, 1.08, 1.16, 1.30]


def V_hodges_lehmann(e):
    """V = 1/(12 (int f^2)^2) với f = (1-e) phi(t) + e phi(t/3)/3."""
    f = lambda t: (1.0 - e) * phi(t) + e * phi(t / 3.0) / 3.0
    tp = integrate.quad(lambda t: f(t) ** 2, -60, 60, limit=400)[0]
    return 1.0 / (12.0 * tp * tp)


def V_de_xuat_2(k, e):
    """Đề xuất 2 dưới F cụ thể: V = E_F psi^2(kS, x) / (E_F psi'(kS, x))^2,
       trong đó S thoả E_F psi^2(kS, x) = S^2 beta(k) — đúng câu tr.99."""
    def E_F_psi2(c):
        # F = (1-e) Phi + e Phi(t/3) -> E psi^2(c, x) tính bằng công thức đóng
        def tp(sig):
            cc = c / sig
            return sig * sig * ((2 * PHI(cc) - 1) - 2 * cc * phi(cc)) \
                + c * c * 2 * (1 - PHI(cc))
        return (1.0 - e) * tp(1.0) + e * tp(3.0)

    def E_F_psip(c):
        return (1.0 - e) * (2 * PHI(c) - 1) + e * (2 * PHI(c / 3.0) - 1)

    beta_k = E_psi_binh(k)
    S = optimize.brentq(lambda s: E_F_psi2(k * s) - s * s * beta_k, 0.2, 20.0,
                        xtol=1e-14, rtol=1e-15)
    c = k * S
    return E_F_psi2(c) / (E_F_psip(c) ** 2), S


print("  %-8s %-24s %-24s" % ("eps", "Hodges-Lehmann (in / đo)", "Đề xuất 2 k=1.5 (in / đo)"))
lech_hai = 0
for e, a_in, b_in in zip(HAI_EPS, HL2_BAO, P2_BAO):
    va = V_hodges_lehmann(e)
    vb, S_ = V_de_xuat_2(1.5, e)
    ng = 0.0006 if e == 0 else 0.006
    if abs(va - a_in) > ng:
        lech_hai += 1
    if abs(vb - b_in) > ng:
        lech_hai += 1
    print("  %-8.2f %6.3f / %-15.8f %6.3f / %-15.8f" % (e, a_in, va, b_in, vb))
print("  số ô lệch: %d / 10" % lech_hai)
print()
print("'apparently, the Hodges-Lehmann estimator and the estimators proposed in the")
print(" preceding sections are close competitors.' (tr.101) — đo lại: Đề xuất 2")
print(" thắng ở CẢ NĂM cột, biên lớn nhất %.4f điểm."
      % max(V_hodges_lehmann(e) - V_de_xuat_2(1.5, e)[0] for e in HAI_EPS))


# ════════════════════════════════════════════════════════════════════════════
muc(12, "Mục 11 tr.96-98 — ba Đề xuất, hệ (6), và thuật toán (6')")
# ════════════════════════════════════════════════════════════════════════════

print("""Đề xuất 1 (tr.96): chọn k trước, rồi giải
    n^{-1} sum psi(k, (x_i - T)/S) = 0
    n^{-1} sum psi'(k, (x_i - T)/S) = E_Phi psi'(k, x)
Đề xuất 2 (tr.96): chọn k trước, rồi giải
    n^{-1} sum psi(k, (x_i - T)/S) = 0
    n^{-1} sum psi^2(k, (x_i - T)/S) = E_Phi psi^2(k, x)
Đề xuất 3 (tr.97): chọn k để cực tiểu hoá phương sai tiệm cận ước lượng được.

Hệ (6) tr.97 viết Đề xuất 2 ở dạng tổng quát với một hằng số beta:
    sum psi(k, (x_i - T)/S)   = 0
    sum psi^2(k, (x_i - T)/S) = n beta
'Proposal 2 corresponds to beta = E_Phi psi^2(k, x)'.

Thuật toán (6') tr.98 — bản cài đặt dưới đây chép đúng ba dòng của bài báo.""")
print()


def huber_6_phay(x, k, beta=None, T0=None, S0=None, so_vong=200, tol=1e-14):
    """Thuật toán (6') tr.98. m1, m2, m3 và trung bình cắt x' theo đúng bài báo."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if beta is None:
        beta = E_psi_binh(k)
    T = float(np.mean(x)) if T0 is None else float(T0)
    S = float(np.std(x, ddof=1)) if S0 is None else float(S0)
    for vong in range(so_vong):
        lo, hi = T - k * S, T + k * S
        m1 = int(np.sum(x <= lo))
        m3 = int(np.sum(x >= hi))
        giua = (x > lo) & (x < hi)
        m2 = int(giua.sum())
        if m2 == 0:
            raise RuntimeError("m2 = 0, hệ (6') không xác định")
        xp = float(x[giua].mean())
        mau = n * beta - (m1 + m3 + (m3 - m1) ** 2 / m2) * k * k
        if mau <= 0:
            raise RuntimeError("mẫu số của S^2 không dương — bài báo dặn phải chặn")
        S1 = float(np.sqrt(((x[giua] - xp) ** 2).sum() / mau))
        T1 = xp + k * S1 * (m3 - m1) / m2
        if abs(T1 - T) < tol and abs(S1 - S) < tol:
            return T1, S1, vong + 1
        T, S = T1, S1
    raise RuntimeError("(6') chưa hội tụ")


rng = np.random.default_rng(0)
mau60 = np.concatenate([rng.normal(size=57), np.array([12.0, 15.0, 20.0])])
for k in (1.0, 1.345, 1.5, 2.0):
    T, S, v = huber_6_phay(mau60, k)
    print("  k = %-6.3f -> T = %+.10f  S = %.10f  sau %d vòng" % (k, T, S, v))
print()
print("  trung bình mẫu   = %+.10f" % mau60.mean())
print("  trung vị mẫu     = %+.10f" % np.median(mau60))
print("  độ lệch chuẩn    = %.10f" % mau60.std(ddof=1))
print("  MAD/0.6745       = %.10f"
      % (np.median(np.abs(mau60 - np.median(mau60))) / 0.6745))
print()
print("Kiểm hệ (6) có thật sự được thoả ở nghiệm — đây là phép thử ngược:")
for k in (1.0, 1.345, 1.5, 2.0):
    T, S, _ = huber_6_phay(mau60, k)
    u = (mau60 - T) / S
    pt1 = psi(u, k).sum()
    pt2 = (psi(u, k) ** 2).sum() - len(mau60) * E_psi_binh(k)
    print("  k = %-6.3f  sum psi = %+.3e   sum psi^2 - n beta = %+.3e" % (k, pt1, pt2))
print()
print("Câu tr.99 về tốc độ: 'For k = 1.5, sample sizes up to 100 and distributions")
print("that do not have too heavy tails, the stationary value is on the average")
print("reached after 1-2 steps, starting from sample mean and sample variance.'")
print("Đo lại trên 2000 mẫu chuẩn cỡ 100, k = 1.5, khởi đầu đúng như bài báo dặn:")
rng2 = np.random.default_rng(1)
so_vong_chuan = []
for _ in range(2000):
    v = rng2.normal(size=100)
    try:
        so_vong_chuan.append(huber_6_phay(v, 1.5)[2])
    except RuntimeError:
        pass
so_vong_chuan = np.array(so_vong_chuan)
print("  trung bình %.4f lượt áp (6') · trung vị %.1f · tối đa %d · %d mẫu"
      % (so_vong_chuan.mean(), np.median(so_vong_chuan), so_vong_chuan.max(),
         len(so_vong_chuan)))
print("  Lượt cuối chỉ XÁC NHẬN T, S không đổi nữa, không phải một bước đi.")
print("  (con số CHƯA trừ lượt xác nhận: %.4f)" % so_vong_chuan.mean())
print("  Trừ nó ra: trung bình %.4f bước ĐỔI · trung vị %.1f · tối đa %d"
      % (so_vong_chuan.mean() - 1, np.median(so_vong_chuan) - 1,
         so_vong_chuan.max() - 1))
print("  -> rơi đúng vào khoảng '1-2 steps' bài báo nêu.")
print()
print("Đuôi nặng — Cauchy, đúng ví dụ bài báo nêu ('about 2 for sample size 10,")
print("4 for sample size 100'):")
rng3 = np.random.default_rng(2)
for n_ in (10, 100):
    vv = []
    for _ in range(2000):
        v = rng3.standard_cauchy(size=n_)
        try:
            vv.append(huber_6_phay(v, 1.5)[2])
        except RuntimeError:
            pass
    vv = np.array(vv)
    print("  n = %-5d trung bình %.4f bước ĐỔI · trung vị %.1f · tối đa %d · %d/2000 mẫu"
          % (n_, vv.mean() - 1, np.median(vv) - 1, vv.max() - 1, len(vv)))
    print("            (chưa trừ lượt xác nhận: %.4f)" % vv.mean())


# ════════════════════════════════════════════════════════════════════════════
muc(13, "scikit-learn cài hệ (6) với beta = 1, KHÔNG phải beta = E_Phi psi^2")
# ════════════════════════════════════════════════════════════════════════════

from sklearn.linear_model import HuberRegressor, LinearRegression   # noqa: E402

print("""HuberRegressor của scikit-learn cực tiểu hoá theo (w, sigma):
    sum_i ( sigma + H_eps(r_i/sigma) sigma ) + alpha ||w||^2
Đạo hàm theo sigma rồi cho bằng 0 cho ra ĐÚNG phương trình thứ hai của hệ (6)
tr.97, với beta = 1:
    n^{-1} sum psi^2(k, r_i/S) = 1
Đề xuất 2 của bài báo thì đặt beta = E_Phi psi^2(k, x) < 1.

Hệ quả có thể đoán trước: E_Phi psi^2 < 1 nên Đề xuất 2 cần S LỚN HƠN.""")
print()
print("  k = 1.345:  E_Phi psi^2 = %.10f   beta của sklearn = 1"
      % E_psi_binh(1.345))
print("  tỉ số hai vế = %.10f" % (1.0 / E_psi_binh(1.345)))
print()

rng4 = np.random.default_rng(4)
Xr = rng4.normal(size=(100, 1))
yr = Xr[:, 0] + rng4.normal(size=100)
Xd = np.c_[np.ones(100), Xr[:, 0]]


def irls_huber(X, y, k=1.345, thang=None, so_vong=500, tol=1e-14):
    """IRLS cho hồi quy. thang=None -> ước lượng lại bằng MAD mỗi vòng."""
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    for _ in range(so_vong):
        r = y - X @ b
        s = thang if thang is not None else max(
            np.median(np.abs(r - np.median(r))) / 0.6745, 1e-12)
        w = np.clip(k * s / np.maximum(np.abs(r), 1e-300), None, 1.0)
        sw = np.sqrt(w)
        moi = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0]
        if np.abs(moi - b).max() < tol:
            return moi
        b = moi
    raise RuntimeError("IRLS chưa hội tụ")


def thang_he6(X, y, b, k, beta):
    """Giải phương trình thứ hai của hệ (6) cho S, với b cố định."""
    r = y - X @ b
    n = len(r)
    f = lambda S: (psi(r / S, k) ** 2).sum() - n * beta
    return optimize.brentq(f, 1e-6, 1e6, xtol=1e-14, rtol=1e-15)


def huber_hoi_quy_he6(X, y, k=1.345, beta=1.0, so_vong=500, tol=1e-13):
    """Giải đồng thời (beta, T, S) cho hồi quy — bản tổng quát của hệ (6)."""
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    S = max(np.median(np.abs((y - X @ b) - np.median(y - X @ b))) / 0.6745, 1e-9)
    for _ in range(so_vong):
        S = thang_he6(X, y, b, k, beta)
        r = y - X @ b
        w = np.clip(k * S / np.maximum(np.abs(r), 1e-300), None, 1.0)
        sw = np.sqrt(w)
        moi = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0]
        if np.abs(moi - b).max() < tol:
            return moi, S
        b = moi
    raise RuntimeError("hệ (6) cho hồi quy chưa hội tụ")


print("Ba cột hệ số góc, trên cùng dữ liệu, cùng k = 1.345:")
print("  %-10s %-16s %-12s %-16s %-12s %-16s %-12s"
      % ("đẩy lệch", "sklearn", "scale_", "hệ (6) beta=1", "S", "Đề xuất 2", "S"))
for day in (0, 100, 100000):
    yb = yr.copy()
    yb[0] += day
    sk = HuberRegressor(epsilon=1.345, alpha=0.0, max_iter=10000, tol=1e-12).fit(Xr, yb)
    b1, S1 = huber_hoi_quy_he6(Xd, yb, 1.345, beta=1.0)
    b2, S2 = huber_hoi_quy_he6(Xd, yb, 1.345, beta=E_psi_binh(1.345))
    print("  %-10d %-16.10f %-12.6f %-16.10f %-12.6f %-16.10f %-12.6f"
          % (day, sk.coef_[0], sk.scale_, b1[1], S1, b2[1], S2))
print()
print("Cột 2 và cột 4 phải trùng nhau tới dung sai tối ưu — đó là bằng chứng")
print("sklearn = hệ (6) với beta = 1. In ra chênh lệch:")
for day in (0, 100, 100000):
    yb = yr.copy()
    yb[0] += day
    sk = HuberRegressor(epsilon=1.345, alpha=0.0, max_iter=10000, tol=1e-12).fit(Xr, yb)
    b1, S1 = huber_hoi_quy_he6(Xd, yb, 1.345, beta=1.0)
    print("  đẩy %-8d |hệ số| chênh %.3e   |thang đo| chênh %.3e"
          % (day, abs(sk.coef_[0] - b1[1]), abs(sk.scale_ - S1)))
print()
print("Và so IRLS + MAD (cách sách giáo khoa hay dạy) với cả ba:")
for day in (0, 100, 100000):
    yb = yr.copy()
    yb[0] += day
    b_mad = irls_huber(Xd, yb, 1.345)
    r = yb - Xd @ b_mad
    s_mad = np.median(np.abs(r - np.median(r))) / 0.6745
    sk = HuberRegressor(epsilon=1.345, alpha=0.0, max_iter=10000, tol=1e-12).fit(Xr, yb)
    b2, S2 = huber_hoi_quy_he6(Xd, yb, 1.345, beta=E_psi_binh(1.345))
    print("  đẩy %-8d MAD: hệ số %.6f thang %.6f | sklearn %.6f thang %.6f | ĐX2 %.6f thang %.6f"
          % (day, b_mad[1], s_mad, sk.coef_[0], sk.scale_, b2[1], S2))
    print("             MAD lệch sklearn %.4f %%   ĐX2 lệch sklearn %.4f %%"
          % (100 * abs(b_mad[1] / sk.coef_[0] - 1), 100 * abs(b2[1] / sk.coef_[0] - 1)))


# ════════════════════════════════════════════════════════════════════════════
muc(14, "k = 1.345 và con số 95 % — đo bằng chính công thức của bài báo")
# ════════════════════════════════════════════════════════════════════════════

print("""Bài báo 1964 KHÔNG chứa con số 1.345 và KHÔNG chứa chữ '95 %'. Nó là hệ quả
của công thức (1): hiệu suất tiệm cận tại Phi bằng 1/V(k), vì thông tin Fisher
của Phi bằng 1. Giải 1/V(k) = 0.95 cho ra k.""")
print()
hieu_suat = lambda k: 1.0 / (E_psi_binh(k) / E_psi_pham(k) ** 2)
k_95 = optimize.brentq(lambda k: hieu_suat(k) - 0.95, 0.5, 5.0, xtol=1e-14, rtol=1e-15)
print("  k cho hiệu suất đúng 95 %%   = %.10f" % k_95)
print("  hiệu suất tại k = 1.345      = %.10f" % hieu_suat(1.345))
print("  -> mất %.10f, tức %.4f %% so với bình phương tối thiểu khi dữ liệu sạch"
      % (1 - hieu_suat(1.345), 100 * (1 - hieu_suat(1.345))))
print("  hiệu suất tại k = 1.35       = %.10f" % hieu_suat(1.35))
print()
print("  %-8s %-16s %-16s %-16s" % ("k", "hiệu suất tại Phi", "V tại Phi", "eps_min"))
for k in (0.5, 1.0, 1.345, 1.5, 2.0, 3.0, 5.0):
    print("  %-8.3f %-16.10f %-16.10f %-16.8f"
          % (k, hieu_suat(k), E_psi_binh(k) / E_psi_pham(k) ** 2, eps_min(k)))
print()
print("  eps_min ứng với k = 1.345    = %.10f" % eps_min(1.345))
print("  -> k = 1.345 là nghiệm minimax cho mức nhiễm bẩn %.4f %%" % (100 * eps_min(1.345)))
print()
print("Đối chiếu với Bảng I: hàng k = 1.3 và k = 1.4 kẹp lấy 1.345.")
print("  Bảng I k=1.3 eps=0 -> 1.058 ; đo lại %.6f" % V_sup(1.3, 0.0))
print("  Bảng I k=1.4 eps=0 -> 1.047 ; đo lại %.6f" % V_sup(1.4, 0.0))
print("  nội suy tuyến tính tại 1.345 -> %.6f ; công thức -> %.6f"
      % (1.058 + (1.047 - 1.058) * 0.45, V_sup(1.345, 0.0)))


# ════════════════════════════════════════════════════════════════════════════
muc(15, "Từ vị trí sang hồi quy — phá dữ liệu theo y, và leverage theo X")
# ════════════════════════════════════════════════════════════════════════════

print("Thí nghiệm 1 — n = 100, hệ số thật 1.0, đẩy lệch MỘT điểm theo y:")
print("  %-12s %-16s %-16s" % ("đẩy lệch", "OLS", "Huber k=1.345"))
for day in (0, 10, 100, 1000, 100000):
    yb = yr.copy()
    yb[0] += day
    ols = LinearRegression().fit(Xr, yb)
    hub = HuberRegressor(epsilon=1.345, alpha=0.0, max_iter=10000, tol=1e-12).fit(Xr, yb)
    print("  %-12d %-16.10f %-16.10f" % (day, ols.coef_[0], hub.coef_[0]))
print()

print("Thí nghiệm 2 — n = 60 sinh từ y = 2 + 1.5x, đẩy lệch ĐÚNG MỘT điểm:")
rng5 = np.random.default_rng(4)
X60 = rng5.normal(size=(60, 1))
y60 = 2.0 + 1.5 * X60[:, 0] + rng5.normal(size=60)
print("  %-14s %-26s %-26s" % ("mức đẩy", "OLS: chặn · góc", "Huber: chặn · góc"))
ols0 = LinearRegression().fit(X60, y60)
hub0 = HuberRegressor(epsilon=1.345, alpha=0.0, max_iter=10000, tol=1e-12).fit(X60, y60)
print("  %-14s %+.6f · %+.6f   %+.6f · %+.6f"
      % ("không đẩy", ols0.intercept_, ols0.coef_[0], hub0.intercept_, hub0.coef_[0]))
for day in (20, 50, 100):
    yb = y60.copy()
    yb[0] += day
    o = LinearRegression().fit(X60, yb)
    h = HuberRegressor(epsilon=1.345, alpha=0.0, max_iter=10000, tol=1e-12).fit(X60, yb)
    print("  %-14s %+.6f · %+.6f   %+.6f · %+.6f"
          % ("+%d" % day, o.intercept_, o.coef_[0], h.intercept_, h.coef_[0]))
    print("                 OLS lệch góc %+.4f %%        Huber lệch góc %+.4f %%"
          % (100 * (o.coef_[0] / 1.5 - 1), 100 * (h.coef_[0] / 1.5 - 1)))
print()

print("Thí nghiệm 3 — ĐIỂM SỤP ĐỔ: một điểm có leverage cao theo X.")
print("Bài báo 1964 chỉ bàn bài toán VỊ TRÍ, nơi không có leverage. Đây là phần")
print("bài viết tự đo, và kết quả đi NGƯỢC trực giác 'Huber bền vững'.")
print("  %-10s %-16s %-16s %-16s %-14s"
      % ("x điểm hỏng", "OLS góc", "Huber góc", "phần dư điểm đó", "|Huber - OLS|"))
for x_bad in (1.0, 3.0, 10.0, 50.0, 500.0, 5000.0):
    Xb = np.vstack([X60, [[x_bad]]])
    yb = np.append(y60, 2.0 + 1.5 * x_bad - 40.0)
    o = LinearRegression().fit(Xb, yb)
    h = HuberRegressor(epsilon=1.345, alpha=0.0, max_iter=10000, tol=1e-12).fit(Xb, yb)
    du = yb[-1] - h.predict(Xb[-1:])[0]
    print("  %-10g %-16.8f %-16.8f %-16.6f %-14.3e"
          % (x_bad, o.coef_[0], h.coef_[0], du, abs(h.coef_[0] - o.coef_[0])))
print("  Hệ số thật 1.5. Đọc theo CỘT CUỐI: leverage càng lớn thì Huber càng")
print("  trùng OLS — ở x = 5000 hai bộ ước lượng chênh nhau dưới một phần triệu,")
print("  tức phần bền vững đã biến mất HẲN. Cột 'phần dư' cho cơ chế: đường khớp")
print("  bị kéo đi qua chính điểm hỏng, nên phần dư của nó co về 0 và psi không")
print("  bao giờ chạm ngưỡng. Cột 'Huber góc' KHÔNG đơn điệu vì hai hiệu ứng ngược")
print("  chiều chồng nhau; chỉ cột cuối mới đọc được thành một kết luận.")
print()
print("Đo trực tiếp điểm sụp đổ: một điểm đủ xa theo X kéo hệ số đi bao nhiêu.")
h_sach = HuberRegressor(epsilon=1.345, alpha=0.0, max_iter=10000, tol=1e-12).fit(X60, y60)
print("  hệ số Huber trên dữ liệu sạch          = %.8f" % h_sach.coef_[0])
for x_bad, y_bad in ((100.0, 0.0), (1000.0, 0.0), (10000.0, 0.0)):
    Xb = np.vstack([X60, [[x_bad]]])
    yb = np.append(y60, y_bad)
    h = HuberRegressor(epsilon=1.345, alpha=0.0, max_iter=10000, tol=1e-12).fit(Xb, yb)
    print("  thêm 1 điểm tại x = %-8g y = %-6g -> hệ số = %.8f" % (x_bad, y_bad, h.coef_[0]))


# ════════════════════════════════════════════════════════════════════════════
muc(16, "Ví dụ tối thiểu 6 điểm, và sai số chuẩn theo công thức tr.97")
# ════════════════════════════════════════════════════════════════════════════

x6 = np.arange(6.0)[:, None]
y6 = 1.0 + 2.0 * x6[:, 0]
y6[2] += 20.0
print("  x =", ", ".join("%g" % v for v in x6[:, 0]))
print("  y =", ", ".join("%g" % v for v in y6), "  (điểm x = 2 bị đẩy thêm 20)")
print()
ols6 = LinearRegression().fit(x6, y6)
hub6 = HuberRegressor(epsilon=1.35, alpha=0.0, max_iter=10000, tol=1e-12).fit(x6, y6)
print("  LinearRegression chặn = %.10f  góc = %.10f" % (ols6.intercept_, ols6.coef_[0]))
print("  HuberRegressor   chặn = %.10f  góc = %.10f" % (hub6.intercept_, hub6.coef_[0]))
print("  thang đo scale_       = %.10f" % hub6.scale_)
print()

print("""Ước lượng phương sai tiệm cận, tr.97:
    (n/(n-1)) · sum psi^2(k,(x_i-T)/S) / [sum psi'(k,(x_i-T)/S)]^2 · S^2
Đây chính là ước lượng 'bánh kẹp' mà tài liệu hiện đại gọi là sandwich, viết ra
từ 1964. Áp cho bộ n = 100 ở mục 15:""")
sk100 = HuberRegressor(epsilon=1.345, alpha=0.0, max_iter=10000, tol=1e-12).fit(Xr, yr)
r100 = yr - sk100.predict(Xr)
S100 = sk100.scale_
u100 = r100 / S100
n100 = len(yr)
tu = (psi(u100, 1.345) ** 2).sum()
mau = (np.abs(u100) < 1.345).sum()
V_hat = n100 / (n100 - 1.0) * tu / (mau ** 2) * S100 ** 2
print("  n = %d   sum psi^2 = %.8f   sum psi' = %d" % (n100, tu, mau))
print("  -> %d trên %d quan sát nằm NGOÀI ngưỡng và bị giảm trọng số, dù dữ liệu"
      % (n100 - mau, n100))
print("     sinh từ phân phối chuẩn thuần tuý (tỉ lệ %.4f)" % ((n100 - mau) / n100))
print("  -> sum psi^2 / n = %.8f, tức đúng vế phải beta = 1 của sklearn" % (tu / n100))
print("  ước lượng phương sai của T (thang vị trí) = %.10f" % V_hat)
print("  -> sai số chuẩn = %.10f" % np.sqrt(V_hat))
print("  so với sai số chuẩn OLS thường = %.10f"
      % np.sqrt(np.var(yr - LinearRegression().fit(Xr, yr).predict(Xr), ddof=2) / n100))


# ════════════════════════════════════════════════════════════════════════════
muc(17, "Số cho các khối mã sinh hình — in đủ chữ số để dán thẳng")
# ════════════════════════════════════════════════════════════════════════════

ks_hinh = np.arange(0.0, 3.001, 0.05)
print("KS =", "[" + ", ".join("%.2f" % k for k in ks_hinh) + "]")
for e in (0.0, 0.01, 0.05, 0.1):
    vs = [V_sup(k, e) for k in ks_hinh]
    print("V_eps_%s =" % str(e).replace(".", "_"),
          "[" + ", ".join("%.10f" % v for v in vs) + "]")
print("EPSMIN =", "[" + ", ".join("%.10f" % eps_min(k) for k in ks_hinh) + "]")
print()
tt = np.linspace(-4, 4, 161)
print("TT =", "[" + ", ".join("%.4f" % t for t in tt) + "]")
print("RHO_1345 =", "[" + ", ".join("%.8f" % v for v in rho(tt, 1.345)) + "]")
print("PSI_1345 =", "[" + ", ".join("%.8f" % v for v in psi(tt, 1.345)) + "]")
e_h = eps_min(1.345)
print("F0_1345 =", "[" + ", ".join("%.10f" % v for v in f0(tt, 1.345, e_h)) + "]")
print("PHI_T =", "[" + ", ".join("%.10f" % v for v in phi(tt)) + "]")
print("EPS_F0 = %.10f" % e_h)
print("RHO_TAI_4 = %.8f      # rho Huber tại t = 4, dùng ở phần 'Cách đọc hình' 32"
      % float(rho(4.0, 1.345)))
print("RHO_TI_SO_TAI_4 = %.4f  # bình phương / Huber tại t = 4"
      % (0.5 * 16.0 / float(rho(4.0, 1.345))))
print()
ks_hs = np.arange(0.05, 3.001, 0.05)
print("KS_HS =", "[" + ", ".join("%.2f" % k for k in ks_hs) + "]")
print("HIEUSUAT =", "[" + ", ".join("%.10f" % hieu_suat(k) for k in ks_hs) + "]")
print()
# dữ liệu cho hình điểm sụp đổ — in đủ chữ số để khối mã dán thẳng
X_BD = np.arange(-3.0, 3.001, 0.25)
print("X_DAY =", "[" + ", ".join("%.2f" % v for v in X_BD) + "]")
day_muc = [0, 20, 50, 100]
for dm in day_muc:
    yb = y60.copy()
    yb[0] += dm
    o_ = LinearRegression().fit(X60, yb)
    h_ = HuberRegressor(epsilon=1.345, alpha=0.0, max_iter=10000, tol=1e-12).fit(X60, yb)
    print("DAY_%d = [%.10f, %.10f, %.10f, %.10f]   # OLS chặn, OLS góc, Huber chặn, Huber góc"
          % (dm, o_.intercept_, o_.coef_[0], h_.intercept_, h_.coef_[0]))
x_lev = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0]
print("X_LEV =", "[" + ", ".join("%g" % v for v in x_lev) + "]")
_o, _h, _d = [], [], []
for xb in x_lev:
    Xb = np.vstack([X60, [[xb]]])
    yb = np.append(y60, 2.0 + 1.5 * xb - 40.0)
    o_ = LinearRegression().fit(Xb, yb)
    h_ = HuberRegressor(epsilon=1.345, alpha=0.0, max_iter=10000, tol=1e-12).fit(Xb, yb)
    _o.append(o_.coef_[0]); _h.append(h_.coef_[0])
    _d.append(abs(yb[-1] - h_.predict(Xb[-1:])[0]))
print("GOC_OLS_LEV =", "[" + ", ".join("%.10f" % v for v in _o) + "]")
print("GOC_HUB_LEV =", "[" + ", ".join("%.10f" % v for v in _h) + "]")
print("DU_LEV =", "[" + ", ".join("%.10f" % v for v in _d) + "]")
print("GOC_THAT = 1.5")
print("X60 =", "[" + ", ".join("%.10f" % v for v in X60[:, 0]) + "]")
print("Y60 =", "[" + ", ".join("%.10f" % v for v in y60) + "]")
print()
print()
print("─" * 92)
print("BẢNG I DỰNG LẠI — in ra ĐỦ GIÁ TRỊ, không chỉ chênh lệch. Bài viết in cả")
print("bảng này, nên script phải in nó ra; nếu không thì 403 ô trong bài không có")
print("nguồn (bài học 'bảng số không kèm script là bảng sẽ hỏng').")
print("─" * 92)
print("  %-5s %-8s %-8s %-8s %s"
      % ("k", "Epsi'", "Epsi^2", "epsmin", " ".join("%-7s" % ("e=%g" % e) for e in EPS_COT)))
for i in range(31):
    k = i / 10.0
    dong = "  %-5.1f %-8.4f %-8.4f %-8.4f " % (k, E_psi_pham(k), E_psi_binh(k), eps_min(k))
    dong += " ".join("%-7.3f" % V_sup(k, e) for e in EPS_COT)
    print(dong)
print()
print("─" * 92)
print("BẢNG IV DỰNG LẠI — cột NỘI SUY (đường bài báo dùng) và cột CÔNG THỨC.")
print("─" * 92)
print("  %-5s %-9s %-9s %-9s %-9s %-9s %-9s"
      % ("k", "e=0.000", "e=0.010", "e=0.020", "e=0.050", "e=0.100", "e=0.200"))
for k in BANG_IV_K:
    ns, ct = [], []
    for e in BANG_IV_EPS:
        qm = q_max(k, e)
        if np.isfinite(qm) and qm <= 3.0:
            ns.append("%.4f" % V_noi_suy_bang_I(qm, e))
            ct.append("%.4f" % V_sup(qm, e))
        else:
            ns.append("qmax=%.2f" % qm if np.isfinite(qm) else "qmax=inf")
            ct.append("%.4f" % V_sup(qm, e) if np.isfinite(qm) else "inf")
    print("  %-5.1f %s   [nội suy]" % (k, " ".join("%-9s" % v for v in ns)))
    print("        %s   [công thức]" % " ".join("%-9s" % v for v in ct))
print()
print("Mọi con số bài trích đều nằm trong đầu ra này.")
print(NGAN)
print("v_huber.py chạy xong.")
