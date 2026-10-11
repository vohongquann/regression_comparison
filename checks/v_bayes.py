# -*- coding: utf-8 -*-
"""Kiểm chứng bài `Bayesian Linear Regression` (Order 440, GĐ 1).

Nguồn ruột: C. M. Bishop, *Pattern Recognition and Machine Learning*, Springer 2006,
chương 3 — mục 3.3 (Bayesian Linear Regression), 3.4 (Bayesian Model Comparison),
3.5 (The Evidence Approximation), 3.6 (Limitations of Fixed Basis Functions).
PDF: AI algorithms/_papers/2006 - Bishop - Pattern Recognition and Machine Learning.pdf

Chương 3 có **0 bảng số**. Mọi con số của nguồn vì thế nằm trong HÌNH, nên phần lớn
script này là phép ĐỌC LỚP VECTƠ của PDF rồi dựng lại — cùng kỹ thuật đã dùng ở bài
`Ridge Regression` và `Lasso Regression`, nhưng ở đây nó là đường DUY NHẤT.

    cd tools && python -u v_bayes.py

Chạy dưới nửa phút, chỉ dùng numpy + pypdfium2 (+ scikit-learn ở mục 22 nếu có).
"""
import os
import sys

# Ghim số luồng TRƯỚC khi import numpy: thứ tự cộng dồn dấu phẩy động của BLAS đổi
# theo số luồng, và bài học phiên WGAN đã trả giá 80 phút cho chuyện đó.
for _b in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_b, "1")

import ctypes                                                     # noqa: E402
import re                                                         # noqa: E402
from fractions import Fraction                                    # noqa: E402

import numpy as np                                                # noqa: E402
import pypdfium2 as pdfium                                        # noqa: E402
import pypdfium2.raw as pr                                        # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
PAPERS = os.path.join(REPO, "AI algorithms", "_papers")
PDF = os.path.join(PAPERS, "2006 - Bishop - Pattern Recognition and Machine Learning.pdf")
THUMUC = os.path.join(REPO, "AI algorithms", "Machine Learning",
                      "Bayesian Linear Regression")

W = 92


def head(s):
    print("\n" + "=" * W)
    print(s)
    print("=" * W)


def fs(v, n=6):
    return ("%." + str(n) + "f") % v


# ═══════════════════════════════════════════════════════════════════════════ 1
head("1  Môi trường và quy ước")
print("   numpy %s · pypdfium2 %s" % (np.__version__, pdfium.version.PYPDFIUM_INFO.version))
print("   số luồng BLAS ghim bằng biến môi trường = 1 (OMP/OPENBLAS/MKL/NUMEXPR/VECLIB)")
print("   mọi phép tuyến tính chạy float64; không dùng GPU — bài toán lớn nhất là 25×10")

doc = pdfium.PdfDocument(PDF)
print("   PDF: %d trang; chương 3 nằm ở chỉ số 156–196 (sách tr.137–177)" % len(doc))

# Lệch giữa số trang in trên đầu trang và chỉ số trang PDF. Đo, không đoán.
_t0 = doc[157].get_textpage().get_text_range()
_so_in = int(re.match(r"\s*(\d+)", _t0).group(1))
LECH = 157 - _so_in
print("   trang PDF thứ %d in số %d ở đầu trang -> lệch = %d (sách tr.N ở chỉ số N+%d)"
      % (157, _so_in, LECH, LECH))
assert LECH == 19


def trang(so_sach):
    return doc[so_sach + LECH]


def idx(so_sach):
    return so_sach + LECH


# ═══════════════════════════════════════════════════════════════════════════ 2
head("2  Kiểm kê chương 3 — đếm thẳng trên lớp text, không gõ tay")

_ch3 = []
for i in range(idx(137), idx(178)):
    _ch3.append(doc[i].get_textpage().get_text_range())
_txt = "\n".join(_ch3)

_hinh = sorted({int(m) for m in re.findall(r"Figure 3\.(\d+)", _txt)})
_bang = sorted({int(m) for m in re.findall(r"Table 3\.(\d+)", _txt)})
_ct = sorted({int(m) for m in re.findall(r"\(3\.(\d+)\)", _txt)})
_ct_than = [c for c in _ct if c <= 99]
_ct_bt = [c for c in _ct if c > 99]
print("   số trang của chương            : %d (sách tr.137–177)" % len(_ch3))
print("   Figure 3.x nhắc tới            : %d  (%s)" % (len(_hinh), "–".join([str(_hinh[0]), str(_hinh[-1])])))
print("   Table 3.x nhắc tới             : %d" % len(_bang))
print("   công thức đánh số ở THÂN chương: %d  ((3.1)–(3.%d))" % (len(_ct_than), max(_ct_than)))
print("   công thức đánh số ở BÀI TẬP    : %d  ((3.%d)–(3.%d))" % (len(_ct_bt), min(_ct_bt), max(_ct_bt)))
_ruot = [c for c in _ct_than if 48 <= c <= 99]
print("   trong đó thuộc mục 3.3–3.5     : %d  ((3.48)–(3.99))" % len(_ruot))
print("   -> chương 3 có %d BẢNG. Mọi con số của nguồn nằm trong HÌNH." % len(_bang))
assert len(_bang) == 0 and len(_hinh) == 17 and len(_ct_than) == 99

# mục đánh số của chương, cho bảng đối chiếu theo MỤC của Luật 6
_muc = sorted(set(re.findall(r"\n(3\.\d(?:\.\d)?)\.? [A-Z]", _txt)))
print("   mục đánh số dò được trên lớp text: %s" % " ".join(_muc))


# ═══════════════════════════════════════════════════════════════════════════ 3
head("3  Đọc LỚP VECTƠ của PDF — hạ tầng dùng cho mọi hình về sau")


def duong(i):
    """[(mau_net, [(loai, x, y), ...]), ...] cho mọi path object của trang chỉ số i."""
    pg = doc[i]
    n = pr.FPDFPage_CountObjects(pg)
    R = [ctypes.c_uint() for _ in range(4)]
    out = []
    for k in range(n):
        o = pr.FPDFPage_GetObject(pg, k)
        if pr.FPDFPageObj_GetType(o) != 2:
            continue
        pr.FPDFPageObj_GetStrokeColor(o, *[ctypes.byref(v) for v in R])
        net = tuple(v.value for v in R)[:3]
        ns = pr.FPDFPath_CountSegments(o)
        seg = []
        for s in range(ns):
            sg = pr.FPDFPath_GetPathSegment(o, s)
            x, y = ctypes.c_float(), ctypes.c_float()
            pr.FPDFPathSegment_GetPoint(sg, ctypes.byref(x), ctypes.byref(y))
            seg.append((pr.FPDFPathSegment_GetType(sg), x.value, y.value))
        out.append((k, net, seg))
    return out


def moc(seg):
    """Tâm của một khối 13 đoạn = một dấu tròn của MATLAB."""
    xs = [q[1] for q in seg]
    ys = [q[2] for q in seg]
    return (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2


class Truc(object):
    """Quy đổi toạ độ PDF sang toạ độ dữ liệu bằng HAI mốc trên mỗi trục."""

    def __init__(self, xa, va, xb, vb, ya, wa, yb, wb):
        self.ax = (vb - va) / (xb - xa)
        self.bx = va - self.ax * xa
        self.ay = (wb - wa) / (yb - ya)
        self.by = wa - self.ay * ya

    def __call__(self, x, y):
        return self.ax * x + self.bx, self.ay * y + self.by


_d = duong(idx(157))
_so_net = len(_d)
_so_doan = sum(len(s) for _, _, s in _d)
print("   trang sách 157 (Hình 3.8): %d đường, %d đoạn -> hình VECTƠ, đọc được toạ độ"
      % (_so_net, _so_doan))
_d174 = pr.FPDFPage_CountObjects(doc[idx(155)])
_anh = sum(1 for k in range(_d174)
           if pr.FPDFPageObj_GetType(pr.FPDFPage_GetObject(doc[idx(155)], k)) == 3)
print("   trang sách 155 (Hình 3.7): %d đối tượng ẢNH, 0 đường -> hình RASTER, KHÔNG đọc được"
      % _anh)
print("   -> Hình 3.7 và 3.10 phải dựng lại bằng mô phỏng; 3.8, 3.9, 3.11, 3.14, 3.16, 3.17 đọc được")


# ═══════════════════════════════════════════════════════════════════════════ 4
head("4  Hình 3.8 tr.157 — đọc thẳng 25 điểm dữ liệu mà bài báo KHÔNG in ra")

# Bốn ô của Hình 3.8. Mốc trục lấy từ chính các vạch chia của hình (xem mục 3).
O = {
    "N=1":  dict(ob=8,  x0=60.74,  x1=234.07, ym=481.85, yp=565.43),
    "N=2":  dict(ob=29, x0=281.24, x1=454.57, ym=481.85, yp=565.43),
    "N=4":  dict(ob=50, x0=60.74,  x1=234.07, ym=319.18, yp=402.76),
    "N=25": dict(ob=71, x0=281.24, x1=454.57, ym=319.18, yp=402.76),
}
_seg = {k: s for k, _, s in duong(idx(157))}
BO = {}
for ten, cf in O.items():
    tr = Truc(cf["x0"], 0.0, cf["x1"], 1.0, cf["ym"], -1.0, cf["yp"], 1.0)
    s = _seg[cf["ob"]]
    assert len(s) % 13 == 0, (ten, len(s))
    pts = sorted(tr(*moc(s[j:j + 13])) for j in range(0, len(s), 13))
    BO[ten] = np.array(pts)
    print("   ô %-5s : %3d đoạn / 13 = %2d dấu tròn" % (ten, len(s), len(pts)))

X25, T25 = BO["N=25"][:, 0], BO["N=25"][:, 1]
_luoi = np.arange(25) / 24.0
_lech_x = np.abs(X25 - _luoi).max()
print("\n   x đọc được so với linspace(0,1,25): lệch lớn nhất %.6f" % _lech_x)
print("   -> chính phép so này XÁC NHẬN cả phép quy đổi trục: nếu trục lệch thì x không ra lưới đều")
assert _lech_x < 2e-3
X25 = _luoi                     # lưới đã xác nhận -> dùng giá trị ĐÚNG, không dùng giá trị đọc

_nhieu = T25 - np.sin(2 * np.pi * X25)
print("   nhiễu t − sin(2πx): trung bình %+.6f · độ lệch chuẩn %.6f · |lớn nhất| %.6f"
      % (_nhieu.mean(), _nhieu.std(ddof=1), np.abs(_nhieu).max()))
print("   Bishop mục 1.1 tr.6 sinh dữ liệu bằng nhiễu Gauss; mục 3.5 tr.171 ghi β 'giá trị thật' 11,1")
print("   1/0,3² = %.6f  -> độ lệch chuẩn thật là 0,3; sd đo lại trên 25 điểm = %.6f"
      % (1 / 0.3 ** 2, _nhieu.std(ddof=1)))

print("\n   25 điểm đọc được (x = k/24):")
for k in range(0, 25, 5):
    print("      " + "  ".join("(%.4f, %+.4f)" % (X25[j], T25[j]) for j in range(k, min(k + 5, 25))))

# Bốn ô có phải TẬP CON LỒNG NHAU không — bài báo chỉ nói "data sets of size N=1,2,4,25"
CHI_SO = {}
for ten in ("N=1", "N=2", "N=4"):
    ii = []
    for x, t in BO[ten]:
        j = int(np.argmin(np.abs(X25 - x)))
        assert abs(T25[j] - t) < 0.02, (ten, x, t, T25[j])
        ii.append(j)
    CHI_SO[ten] = sorted(ii)
CHI_SO["N=25"] = list(range(25))
print("\n   chỉ số (trong tập 25 điểm) mà từng ô dùng:")
for ten in ("N=1", "N=2", "N=4", "N=25"):
    print("      %-5s -> %s" % (ten, CHI_SO[ten]))
_long = all(set(CHI_SO["N=1"]) <= set(CHI_SO["N=2"]) <= set(CHI_SO["N=4"]) <= set(CHI_SO["N=25"])
            for _ in (0,))
print("   bốn ô có LỒNG NHAU không: %s — bài báo không nói điều này ở bất kỳ đâu" % _long)
assert _long


# ═══════════════════════════════════════════════════════════════════════════ 5
head("5  Ví dụ chạy tay — ba điểm, hai tham số, hậu nghiệm tính bằng PHÂN SỐ ĐÚNG")

# y(x,w) = w0 + w1 x ; tiên nghiệm (3.52) p(w|α)=N(0, α⁻¹I) với α=1, β=2
XH = [Fraction(0), Fraction(1), Fraction(2)]
TH = [Fraction(1), Fraction(2), Fraction(2)]
AL, BE = Fraction(1), Fraction(2)
PH = [[Fraction(1), x] for x in XH]
print("   dữ liệu (x,t): %s        α = %s, β = %s" % (
    " ".join("(%s,%s)" % (x, t) for x, t in zip(XH, TH)), AL, BE))

# ΦᵀΦ và Φᵀt bằng phân số
PtP = [[sum(PH[n][i] * PH[n][j] for n in range(3)) for j in range(2)] for i in range(2)]
Ptt = [sum(PH[n][i] * TH[n] for n in range(3)) for i in range(2)]
print("   ΦᵀΦ = [[%s, %s], [%s, %s]]      Φᵀt = [%s, %s]"
      % (PtP[0][0], PtP[0][1], PtP[1][0], PtP[1][1], Ptt[0], Ptt[1]))

# (3.54): S_N⁻¹ = αI + βΦᵀΦ
SI = [[AL * (1 if i == j else 0) + BE * PtP[i][j] for j in range(2)] for i in range(2)]
det = SI[0][0] * SI[1][1] - SI[0][1] * SI[1][0]
SN = [[SI[1][1] / det, -SI[0][1] / det], [-SI[1][0] / det, SI[0][0] / det]]
print("   (3.54) S_N⁻¹ = [[%s, %s], [%s, %s]]   định thức = %s" % (SI[0][0], SI[0][1], SI[1][0], SI[1][1], det))
print("   S_N        = [[%s, %s], [%s, %s]]" % (SN[0][0], SN[0][1], SN[1][0], SN[1][1]))

# (3.53): m_N = β S_N Φᵀ t
mN = [BE * (SN[i][0] * Ptt[0] + SN[i][1] * Ptt[1]) for i in range(2)]
print("   (3.53) m_N = [%s, %s] = [%s, %s]" % (mN[0], mN[1], fs(float(mN[0])), fs(float(mN[1]))))

# đối chiếu với ridge λ = α/β trên cùng dữ liệu — (3.55)
lam = AL / BE
RI = [[PtP[i][j] + (lam if i == j else 0) for j in range(2)] for i in range(2)]
dr = RI[0][0] * RI[1][1] - RI[0][1] * RI[1][0]
wr = [(RI[1][1] * Ptt[0] - RI[0][1] * Ptt[1]) / dr, (-RI[1][0] * Ptt[0] + RI[0][0] * Ptt[1]) / dr]
print("   ridge λ = α/β = %s  -> w = [%s, %s]" % (lam, wr[0], wr[1]))
print("   m_N == w_ridge (phân số đúng, không phải xấp xỉ): %s" % (mN == wr))
assert mN == wr

# nghiệm bình phương tối thiểu để so
dl = PtP[0][0] * PtP[1][1] - PtP[0][1] * PtP[1][0]
wl = [(PtP[1][1] * Ptt[0] - PtP[0][1] * Ptt[1]) / dl, (-PtP[1][0] * Ptt[0] + PtP[0][0] * Ptt[1]) / dl]
print("   w_ML (không tiên nghiệm) = [%s, %s] = [%s, %s]"
      % (wl[0], wl[1], fs(float(wl[0])), fs(float(wl[1]))))

# (3.58)-(3.59): dự báo tại x* = 3
xs = [Fraction(1), Fraction(3)]
mu = mN[0] * xs[0] + mN[1] * xs[1]
qf = sum(xs[i] * SN[i][j] * xs[j] for i in range(2) for j in range(2))
var = 1 / BE + qf
print("\n   (3.58) tại x* = 3: trung bình = %s = %s" % (mu, fs(float(mu))))
print("   (3.59) σ²  = 1/β + φᵀS_Nφ = %s + %s = %s = %s"
      % (Fraction(1, 2), qf, var, fs(float(var))))
print("          phần NHIỄU %s (%.2f %%) · phần BẤT ĐỊNH THAM SỐ %s (%.2f %%)"
      % (Fraction(1, 2), 100 * float(1 / BE / var), qf, 100 * float(qf / var)))
print("   độ lệch chuẩn dự báo = %s ; khoảng 95 %% ≈ %s ± %s"
      % (fs(float(var) ** .5), fs(float(mu)), fs(1.96 * float(var) ** .5)))
xs0 = [Fraction(1), Fraction(1)]
qf0 = sum(xs0[i] * SN[i][j] * xs0[j] for i in range(2) for j in range(2))
print("   cùng phép tính tại x* = 1 (giữa dữ liệu): φᵀS_Nφ = %s = %s -> %.4f lần nhỏ hơn"
      % (qf0, fs(float(qf0)), float(qf / qf0)))


# ═══════════════════════════════════════════════════════════════════════════ 6
head("6  Đường VECTƠ của PDF là chính đường được vẽ — dựng lại bằng cách khai triển Bézier")


def da_diem(seg, m=16):
    """Khai triển một path object thành các đoạn thẳng nhỏ.

    Toạ độ trong PDF không phải một xấp xỉ của đường cong — nó LÀ đường cong được
    vẽ ra. Nên khai triển Bézier bậc ba cho lại đúng thứ người đọc nhìn thấy, khác
    hẳn việc đo điểm ảnh. Trả về danh sách các nhánh, mỗi nhánh là mảng (k, 2).
    """
    nhanh, cur, i = [], [], 0
    dem = []
    while i < len(seg):
        lo, x, y = seg[i]
        if lo == 2:                                  # MOVETO
            if len(cur) > 1:
                nhanh.append(np.array(cur))
            cur = [(x, y)]
            i += 1
        elif lo == 0:                                # LINETO
            cur.append((x, y))
            i += 1
        elif lo == 1:                                # BEZIERTO, ba điểm một cụm
            p0 = np.array(cur[-1])
            c1 = np.array([seg[i][1], seg[i][2]])
            c2 = np.array([seg[i + 1][1], seg[i + 1][2]])
            p3 = np.array([seg[i + 2][1], seg[i + 2][2]])
            for u in np.linspace(0, 1, m)[1:]:
                p = ((1 - u) ** 3 * p0 + 3 * (1 - u) ** 2 * u * c1
                     + 3 * (1 - u) * u ** 2 * c2 + u ** 3 * p3)
                cur.append((p[0], p[1]))
            dem.append(1)
            i += 3
        else:
            i += 1
    if len(cur) > 1:
        nhanh.append(np.array(cur))
    return nhanh, len(dem)


_nh, _nb = da_diem(_seg[70])
print("   đường trung bình đỏ của ô N=25 (object 70): %d nhánh, %d cụm Bézier, %d điểm sau khai triển"
      % (len(_nh), _nb, sum(len(a) for a in _nh)))


def doc_duong(i_trang, ob, tr, m=16):
    """Trả về (x, y) đã quy đổi, sắp theo x, của một path object."""
    seg = {k: s for k, _, s in duong(i_trang)}[ob]
    nh, _ = da_diem(seg, m)
    P = np.vstack(nh)
    Q = np.array([tr(*p) for p in P])
    return Q[np.argsort(Q[:, 0])]


TR25 = Truc(O["N=25"]["x0"], 0.0, O["N=25"]["x1"], 1.0,
            O["N=25"]["ym"], -1.0, O["N=25"]["yp"], 1.0)
DO25 = doc_duong(idx(157), 70, TR25)
XL25 = doc_duong(idx(157), 72, TR25)             # đường xanh lá = sin(2πx)
_m = (XL25[:, 0] > 0.02) & (XL25[:, 0] < 0.98)
_ss = np.abs(XL25[_m, 1] - np.sin(2 * np.pi * XL25[_m, 0])).max()
print("   đường XANH LÁ của ô N=25 đọc lại so với sin(2πx): lệch lớn nhất %.6f trên %d điểm"
      % (_ss, _m.sum()))
print("   -> đường xanh lá là phép kiểm CHÉO cho toàn bộ khâu đọc vectơ: nó phải ra đúng sin(2πx)")
assert _ss < 0.02


# ═══════════════════════════════════════════════════════════════════════════ 7
head("7  Cập nhật TUẦN TỰ bằng cập nhật MỘT LƯỢT — khẳng định tr.153 và Hình 3.7")


def hau_nghiem(Phi, t, al, be, m0=None, S0inv=None):
    """(3.49)–(3.51): m_N, S_N. Mặc định tiên nghiệm (3.52) N(0, α⁻¹I)."""
    M = Phi.shape[1]
    if S0inv is None:
        S0inv = al * np.eye(M)
        m0 = np.zeros(M)
    SNinv = S0inv + be * Phi.T @ Phi
    SN = np.linalg.inv(SNinv)
    mN = SN @ (S0inv @ m0 + be * Phi.T @ t)
    return mN, SN, SNinv


def thiet_ke_tt(x):
    return np.c_[np.ones(len(x)), x]


rng = np.random.default_rng(0)
_x = rng.uniform(-1, 1, 20)
_t = -0.3 + 0.5 * _x + rng.normal(0, 0.2, 20)
_al, _be = 2.0, 25.0

m_lot, S_lot, _ = hau_nghiem(thiet_ke_tt(_x), _t, _al, _be)
m, Si = np.zeros(2), _al * np.eye(2)
for n in range(20):
    m, S, Si = hau_nghiem(thiet_ke_tt(_x[n:n + 1]), _t[n:n + 1], _al, _be, m, Si)
print("   một lượt 20 điểm : m_N = [%s]" % ", ".join(fs(v, 10) for v in m_lot))
print("   tuần tự 20 bước  : m_N = [%s]" % ", ".join(fs(v, 10) for v in m))
print("   lệch lớn nhất m_N = %.3e · lệch lớn nhất S_N = %.3e"
      % (np.abs(m - m_lot).max(), np.abs(S - S_lot).max()))
assert np.abs(m - m_lot).max() < 1e-12


# ═══════════════════════════════════════════════════════════════════════════ 8
head("8  Hình 3.7 tr.155 — dựng lại thiết lập bài báo KHAI ĐẦY ĐỦ (a₀ = −0,3 · a₁ = 0,5 · α = 2 · β = 25)")

print("   β = (1/0,2)² = %.6f ; bài báo tr.154 ghi β = 25" % ((1 / 0.2) ** 2))
print("   %-6s %-24s %-24s %s" % ("N", "m_N (trung bình hậu nghiệm)", "đường chéo S_N", "|m_N − a| "))
_rong, _sai = [], []
for N in (0, 1, 2, 20):
    if N == 0:
        mN0, SN0 = np.zeros(2), np.eye(2) / _al
    else:
        mN0, SN0, _ = hau_nghiem(thiet_ke_tt(_x[:N]), _t[:N], _al, _be)
    _rong.append(np.diag(SN0).copy())
    _sai.append(np.abs(mN0 - np.array([-0.3, 0.5])).max())
    print("   %-6d [%s]  [%s]  %s"
          % (N, ", ".join("%+.6f" % v for v in mN0),
             ", ".join("%.6f" % v for v in np.diag(SN0)), fs(_sai[-1])))
_don_rong = all((_rong[i + 1] < _rong[i]).all() for i in range(3))
_don_sai = all(_sai[i + 1] < _sai[i] for i in range(3))
print("   đường chéo S_N giảm đơn điệu: %s   |m_N − a| giảm đơn điệu: %s"
      % (_don_rong, _don_sai))
print("   -> ĐỘ RỘNG hậu nghiệm co lại đơn điệu; SAI SỐ của trung bình thì KHÔNG"
      " (%s). Hai chuyện khác nhau." % " → ".join(fs(v, 3) for v in _sai))
assert _don_rong and not _don_sai

_R, _de = 200, []
for _s in range(_R):
    g = np.random.default_rng(1000 + _s)
    xx = g.uniform(-1, 1, 20)
    tt = -0.3 + 0.5 * xx + g.normal(0, 0.2, 20)
    mm, _, _ = hau_nghiem(thiet_ke_tt(xx), tt, _al, _be)
    _de.append(np.abs(mm - np.array([-0.3, 0.5])).max())
print("   trên %d hạt giống, |m_N − a| sau 20 điểm: trung bình %.6f · khoảng [%.6f ; %.6f]"
      % (_R, float(np.mean(_de)), float(np.min(_de)), float(np.max(_de))))

_tt = []
for N in (20, 100, 500, 2500):
    d = []
    for _s in range(60):
        g = np.random.default_rng(7000 + _s)
        xx = g.uniform(-1, 1, N)
        tt = -0.3 + 0.5 * xx + g.normal(0, 0.2, N)
        mm, _, _ = hau_nghiem(thiet_ke_tt(xx), tt, _al, _be)
        d.append(np.abs(mm - np.array([-0.3, 0.5])).max())
    _tt.append((N, float(np.mean(d))))
print("   'trong giới hạn vô số điểm hậu nghiệm thành hàm delta' (tr.156) — đo bằng sai số trung bình:")
print("      " + " · ".join("N=%d: %.6f" % q for q in _tt))
print("      tỉ số giữa hai N liền nhau (mỗi bước ×5, kỳ vọng ×1/√5 = %.4f): %s"
      % (5 ** -0.5, " · ".join("%.4f" % (_tt[i + 1][1] / _tt[i][1]) for i in range(3))))


# ═══════════════════════════════════════════════════════════════════════════ 9
head("9  Hình 3.8 — bài báo KHÔNG khai μ_j, s, α. Dò chúng bằng chính đường vẽ ra")

print("   bài báo nói gì về thiết lập của Hình 3.8:")
print("     · 'a model consisting of 9 Gaussian basis functions of the form (3.4)' (chú thích Hình 3.8)")
print("     · 'the synthetic sinusoidal data set of Section 1.1'")
print("     · tr.171: 'we have set β to its true value of 11.1'  (cho Hình 3.16, cùng mô hình)")
print("   bài báo KHÔNG nói: tâm μ_j · bề rộng s · giá trị α. Ba thứ này phải dò.")


def thiet_ke_gauss(x, mu, s):
    """(3.4) kèm φ₀ = 1 theo (3.3): M = len(mu) + 1."""
    return np.c_[np.ones(len(x)), np.exp(-(x[:, None] - mu[None, :]) ** 2 / (2 * s ** 2))]


def du_bao(Phi_h, t, Phi_m, al, be):
    """(3.58)–(3.59): trung bình và ĐỘ LỆCH CHUẨN dự báo."""
    mN, SN, _ = hau_nghiem(Phi_h, t, al, be)
    tb = Phi_m @ mN
    va = 1.0 / be + np.einsum("ij,jk,ik->i", Phi_m, SN, Phi_m)
    return tb, np.sqrt(va), mN, SN


# Dải mean ± σ của ô N=25. Đa giác tô (object 68) là MỘT nhánh KHÉP KÍN, nên một
# đường dọc x = c cắt nó đúng HAI lần: biên trên và biên dưới. Lấy giao điểm rồi nội
# suy tuyến tính cho ra biên chính xác — khác hẳn phép lấy min/max trong một cửa sổ x,
# vốn phóng to bề rộng ở chỗ dải dốc (đo được: phóng tới 0,52 thay vì 0,35).
def cat_doc(nhanh, xs):
    tren = np.full(len(xs), np.nan)
    duoi = np.full(len(xs), np.nan)
    for i, xv in enumerate(xs):
        gy = []
        for a in nhanh:
            for k in range(len(a) - 1):
                x1, y1 = a[k]
                x2, y2 = a[k + 1]
                if (x1 - xv) * (x2 - xv) <= 0 and x1 != x2:
                    gy.append(y1 + (y2 - y1) * (xv - x1) / (x2 - x1))
        if len(gy) >= 2:
            tren[i], duoi[i] = max(gy), min(gy)
    return tren, duoi


def dai(i_trang, ob, tr, xs, m=24):
    seg = {k: s for k, _, s in duong(i_trang)}[ob]
    nh, _ = da_diem(seg, m)
    nh = [np.array([tr(*p) for p in a]) for a in nh]
    tren, duoi = cat_doc(nh, xs)
    ok = ~np.isnan(tren)
    return xs[ok], tren[ok], duoi[ok]


_xs = np.linspace(0.02, 0.98, 49)
XB, TRN, DUI = dai(idx(157), 68, TR25, _xs)
_tbb, _sgb = (TRN + DUI) / 2, (TRN - DUI) / 2
print("\n   dải hồng của ô N=25 (object 68, đa giác khép kín): đọc được tại %d hoành độ" % len(XB))
print("   σ(x) đọc từ dải: nhỏ nhất %.6f tại x = %.3f · lớn nhất %.6f tại x = %.3f"
      % (_sgb.min(), XB[np.argmin(_sgb)], _sgb.max(), XB[np.argmax(_sgb)]))
print("   1/√β với β = 11,1: %.6f — 'in the limit N→∞ ... solely from the additive noise' (tr.156)"
      % (11.1 ** -0.5))
# phép kiểm CHÉO: tâm của dải phải trùng đường đỏ, hai đối tượng PDF khác nhau
_tb_do = np.interp(XB, DO25[:, 0], DO25[:, 1])
print("   tâm dải so với đường đỏ (hai object khác nhau): lệch lớn nhất %.6f -> khâu đọc nhất quán"
      % np.abs(_tbb - _tb_do).max())

MU9 = np.linspace(0, 1, 9)
BE0 = 11.1
print("\n   dò (s, α) trên lưới, đo bằng sai số LỚN NHẤT với đường đỏ và với σ(x) của dải:")
LUOI_S = (0.05, 0.08, 0.1, 0.12, 0.15, 0.18, 0.2, 0.25, 0.3)
LUOI_A = (1e-5, 1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 0.1, 0.5, 1.0, 2.0)
_best = None
for s_ in LUOI_S:
    Ph = thiet_ke_gauss(X25, MU9, s_)
    for al_ in LUOI_A:
        tb, _, _, _ = du_bao(Ph, T25, thiet_ke_gauss(DO25[:, 0], MU9, s_), al_, BE0)
        e1 = np.abs(tb - DO25[:, 1]).max()
        _, sg2, _, _ = du_bao(Ph, T25, thiet_ke_gauss(XB, MU9, s_), al_, BE0)
        e2 = np.abs(sg2 - _sgb).max()
        if _best is None or e1 + e2 < _best[0]:
            _best = (e1 + e2, s_, al_, e1, e2)
print("   %-7s %s" % ("s \\ α", " ".join("%-9.4g" % a for a in LUOI_A)))
for s_ in LUOI_S:
    Ph = thiet_ke_gauss(X25, MU9, s_)
    hang = []
    for al_ in LUOI_A:
        tb, _, _, _ = du_bao(Ph, T25, thiet_ke_gauss(DO25[:, 0], MU9, s_), al_, BE0)
        _, sg2, _, _ = du_bao(Ph, T25, thiet_ke_gauss(XB, MU9, s_), al_, BE0)
        hang.append(np.abs(tb - DO25[:, 1]).max() + np.abs(sg2 - _sgb).max())
    print("   %-7.2f %s" % (s_, " ".join("%-9.4f" % v for v in hang)))
print("   tốt nhất trên lưới thô: s = %.2f · α = %.4g · lệch trung bình %.6f · lệch σ %.6f"
      % (_best[1], _best[2], _best[3], _best[4]))

# siết lưới quanh nghiệm thô — hai vòng, mỗi vòng chia nhỏ bước 5 lần
_c = _best
for _vong in (1, 2):
    ds = 0.03 / 5 ** (_vong - 1)
    la = 0.4 / 5 ** (_vong - 1)
    for s_ in np.linspace(max(_c[1] - ds, 0.02), _c[1] + ds, 13):
        Ph = thiet_ke_gauss(X25, MU9, s_)
        for al_ in np.linspace(max(_c[2] - la, 1e-4), _c[2] + la, 13):
            tb, _, _, _ = du_bao(Ph, T25, thiet_ke_gauss(DO25[:, 0], MU9, s_), al_, BE0)
            _, sg2, _, _ = du_bao(Ph, T25, thiet_ke_gauss(XB, MU9, s_), al_, BE0)
            e1 = np.abs(tb - DO25[:, 1]).max()
            e2 = np.abs(sg2 - _sgb).max()
            if e1 + e2 < _c[0]:
                _c = (e1 + e2, s_, al_, e1, e2)
    print("   sau vòng siết %d : s = %.6f · α = %.6f · lệch trung bình %.6f · lệch σ %.6f"
          % (_vong, _c[1], _c[2], _c[3], _c[4]))
_best = _c
S9, AL9 = _best[1], _best[2]
print("   sai số của chính khâu đọc vectơ (đường xanh lá so với sin 2πx) = %.6f" % _ss)
def _mo_ta(e):
    r = e / _ss
    return ("TRONG sai số đọc" if r < 0.9 else
            "NGANG sai số đọc (tỉ số %.2f)" % r if r < 1.1 else
            "NGOÀI sai số đọc %.1f lần" % r)


print("   -> lệch trung bình %s; lệch σ %s" % (_mo_ta(_best[3]), _mo_ta(_best[4])))
print("   -> bộ ba (μ_j = linspace(0,1,9), s ≈ %.2f, α ≈ %.2f) dựng lại được Hình 3.8," % (S9, AL9))
print("      nhưng đó là bộ DÒ RA, không phải bộ bài báo khai. Phải nói rõ chỗ này.")
# phép thử ngược: đổi từng tham số một bậc rồi xem lệch có xấu đi hẳn không
print("\n   phép thử ngược — đổi MỘT tham số, giữ hai cái kia:")
for ten, s_, al_, mu_ in (("s ×0,5", S9 * .5, AL9, MU9), ("s ×2", S9 * 2, AL9, MU9),
                          ("α ×0,1", S9, AL9 * .1, MU9), ("α ×10", S9, AL9 * 10, MU9),
                          ("7 hàm cơ sở", S9, AL9, np.linspace(0, 1, 7)),
                          ("11 hàm cơ sở", S9, AL9, np.linspace(0, 1, 11)),
                          ("μ trên [0,2 ; 0,8]", S9, AL9, np.linspace(.2, .8, 9))):
    Ph = thiet_ke_gauss(X25, mu_, s_)
    tb, _, _, _ = du_bao(Ph, T25, thiet_ke_gauss(DO25[:, 0], mu_, s_), al_, BE0)
    _, sg2, _, _ = du_bao(Ph, T25, thiet_ke_gauss(XB, mu_, s_), al_, BE0)
    e1, e2 = np.abs(tb - DO25[:, 1]).max(), np.abs(sg2 - _sgb).max()
    print("      %-20s lệch trung bình %.6f (×%5.1f)  lệch σ %.6f (×%5.1f)"
          % (ten, e1, e1 / _best[3], e2, e2 / _best[4]))

print("\n   bốn ô của Hình 3.8 dựng lại với (s = %.2f, α = %.4g, β = 11,1):" % (S9, AL9))
print("   %-6s %-14s %-14s %-14s %s" % ("ô", "σ nhỏ nhất", "σ lớn nhất", "σ tại x=0,5", "|t.bình − sin|"))
_cot = []
for ten in ("N=1", "N=2", "N=4", "N=25"):
    ii = CHI_SO[ten]
    Ph = thiet_ke_gauss(X25[ii], MU9, S9)
    xg = np.linspace(0, 1, 201)
    tb, sg, _, _ = du_bao(Ph, T25[ii], thiet_ke_gauss(xg, MU9, S9), AL9, BE0)
    _cot.append(sg.max())
    print("   %-6s %-14.6f %-14.6f %-14.6f %.6f"
          % (ten, sg.min(), sg.max(), sg[100], np.abs(tb - np.sin(2 * np.pi * xg)).max()))
_giam = all(_cot[i + 1] < _cot[i] for i in range(3))
print("   cột 'σ lớn nhất' giảm đơn điệu theo N: %s" % _giam)
print("   -> 'the level of uncertainty decreases as more data points are observed' (tr.157) đúng")
assert _giam


# ═══════════════════════════════════════════════════════════════════════════ 10
head("10  (3.59) — khẳng định σ²_{N+1}(x) ≤ σ²_N(x) của Qazaz và cộng sự 1997")

_vi, _tong = 0, 0
for s_ in (0.1, S9, 0.2):
    for al_ in (1e-3, AL9, 1.0):
        xg = np.linspace(-0.2, 1.2, 71)
        pre = None
        for N in range(1, 26):
            Ph = thiet_ke_gauss(X25[:N], MU9, s_)
            _, sg, _, _ = du_bao(Ph, T25[:N], thiet_ke_gauss(xg, MU9, s_), al_, BE0)
            v = sg ** 2
            if pre is not None:
                _tong += len(v)
                _vi += int((v > pre + 1e-12).sum())
            pre = v
print("   quét 3 bề rộng × 3 giá trị α × 24 bước N × 71 hoành độ (kể cả ngoài [0,1]):")
print("   %d phép so, %d chỗ VI PHẠM σ²_{N+1} ≤ σ²_N" % (_tong, _vi))
assert _vi == 0
print("   -> khẳng định đúng, và đúng cả ở vùng NGOẠI SUY, chỗ bài báo không thử")


# ═══════════════════════════════════════════════════════════════════════════ 11
head("11  Cảnh báo tr.158 — hàm cơ sở ĐỊNH XỨ làm mô hình TỰ TIN khi ngoại suy")

Ph25 = thiet_ke_gauss(X25, MU9, S9)
for xv in (0.5, 1.0, 1.5, 2.0, 3.0, 5.0):
    _, sg, _, _ = du_bao(Ph25, T25, thiet_ke_gauss(np.array([xv]), MU9, S9), AL9, BE0)
    Phx = thiet_ke_gauss(np.array([xv]), MU9, S9)
    _, SN, _ = hau_nghiem(Ph25, T25, AL9, BE0)
    q = float((Phx @ SN @ Phx.T)[0, 0])
    print("   x = %-5.1f  σ = %.6f   phần nhiễu 1/β = %.6f (%.2f %%)   φᵀS_Nφ = %.3e"
          % (xv, sg[0], 1 / BE0, 100 * (1 / BE0) / sg[0] ** 2, q))
_, _SNg, _ = hau_nghiem(Ph25, T25, AL9, BE0)
print("   1/√β = %.6f ; nhưng σ KHÔNG tụt tới đó — nó dừng ở %.6f"
      % (BE0 ** -0.5, (1 / BE0 + _SNg[0, 0]) ** 0.5))
print("   vì sao: (3.3) đặt thêm hàm cơ sở HẰNG φ₀ = 1, và φ₀ KHÔNG tắt khi x đi xa.")
print("   φ(x) → (1, 0, …, 0) nên φᵀS_Nφ → S_N[0,0] = %.6f, không về 0." % _SNg[0, 0])
print("   Bishop tr.158 viết 'leaving only the noise contribution β⁻¹'. Đo lại thì")
print("   σ² tiến tới β⁻¹ + S_N[0,0] = %.6f + %.6f = %.6f, tức σ lớn gấp %.4f lần 1/√β."
      % (1 / BE0, _SNg[0, 0], 1 / BE0 + _SNg[0, 0],
         ((1 / BE0 + _SNg[0, 0]) / (1 / BE0)) ** 0.5))

# phép thử ngược: BỎ hàm cơ sở hằng thì câu của bài báo đúng nguyên văn
_PhiKB = np.exp(-(X25[:, None] - MU9[None, :]) ** 2 / (2 * S9 ** 2))
_, _SNk, _ = hau_nghiem(_PhiKB, T25, AL9, BE0)
print("   PHÉP THỬ NGƯỢC — bỏ φ₀ = 1, chỉ giữ 9 hàm Gauss:")
for xv in (1.5, 3.0, 10.0):
    _px = np.exp(-(np.array([xv])[:, None] - MU9[None, :]) ** 2 / (2 * S9 ** 2))
    _q = float((_px @ _SNk @ _px.T)[0, 0])
    print("      x = %-5.1f  φᵀS_Nφ = %.3e  σ = %.6f  (1/√β = %.6f)"
          % (xv, _q, (1 / BE0 + _q) ** 0.5, BE0 ** -0.5))
print("   -> câu tr.158 đúng NGUYÊN VĂN khi và chỉ khi không có hàm cơ sở hằng;")
print("      với (3.3) như bài báo viết thì cận là β⁻¹ + S_N[0,0], không phải β⁻¹.")
print("   Chiều của cảnh báo vẫn đúng: mô hình VẪN tự tin quá mức khi ngoại suy.")

# so với hàm cơ sở đa thức, KHÔNG định xứ
def thiet_ke_da_thuc(x, M):
    return np.vander(x, M + 1, increasing=True)


print("\n   cùng phép ngoại suy với hàm cơ sở ĐA THỨC bậc 9 (không định xứ), cùng α, β:")
Pp = thiet_ke_da_thuc(X25, 9)
for xv in (1.0, 1.5, 2.0, 3.0):
    _, sg, _, _ = du_bao(Pp, T25, thiet_ke_da_thuc(np.array([xv]), 9), AL9, BE0)
    print("   x = %-5.1f  σ = %.6f   gấp %.1f lần bản Gauss" % (xv, sg[0], sg[0] / (BE0 ** -0.5)))
print("   -> đa thức thì σ BÙNG NỔ khi ngoại suy; Gauss thì σ dừng ở một SÀN HỮU HẠN.")
print("      Hai lỗi ngược nhau, và chỉ lỗi thứ hai là lỗi Bishop cảnh báo ở tr.158.")


# ═══════════════════════════════════════════════════════════════════════════ 12
head("12  Nhân tương đương (3.60)–(3.63) và Hình 3.10 / 3.11")


def nhan(Phi_h, xa, xb, al, be):
    """(3.62) k(x,x') = β φ(x)ᵀ S_N φ(x'), với Φ_h là ma trận thiết kế của tập huấn luyện."""
    _, SN, _ = hau_nghiem(Phi_h, np.zeros(Phi_h.shape[0]), al, be)
    return be * xa @ SN @ xb.T


# Hình 3.10 dùng 200 giá trị x cách đều trên (−1, 1) — bài báo KHAI rõ trong chú thích.
X200 = np.linspace(-1, 1, 200)
MU_H31 = np.linspace(-1, 1, 9)          # "the Gaussian basis functions in Figure 3.1"
S_H31 = 0.25
PhiG = thiet_ke_gauss(X200, MU_H31, S_H31)
K = nhan(PhiG, PhiG, PhiG, AL9, BE0)
print("   chú thích Hình 3.10: 'comprised 200 values of x equally spaced over the interval (−1,1)'")
print("   ma trận nhân %d×%d, đối xứng tới %.3e" % (K.shape[0], K.shape[1], np.abs(K - K.T).max()))
for xv in (-0.5, 0.0, 0.5):
    j = int(np.argmin(np.abs(X200 - xv)))
    k = K[j]
    _mm = int(np.argmax(k))
    print("   lát cắt tại x = %+.2f: đỉnh ở x' = %+.4f (lệch %+.4f) · k(x,x) = %.6f · tổng = %.6f"
          % (xv, X200[_mm], X200[_mm] - xv, k[j], k.sum()))
print("   -> 'they are localized around x' (tr.159): đỉnh của mỗi lát cắt rơi đúng vào chính x")

_am = int((K < 0).sum())
print("   số ô ÂM của ma trận nhân: %d / %d (%.2f %%)" % (_am, K.size, 100 * _am / K.size))
print("   -> 'the kernel function can be negative as well as positive' (tr.161) — đúng")

# Hình 3.11: nhân tương đương của hàm cơ sở ĐA THỨC và SIGMOID tại x = 0
TR11A = Truc(260.18, 0.0, 309.51, 1.0, 526.26, 0.0, 574.01, 0.04)
TR11B = Truc(405.71, 0.0, 455.04, 1.0, 526.26, 0.0, 574.01, 0.04)
K11A = doc_duong(idx(160), 30, TR11A)
K11B = doc_duong(idx(160), 49, TR11B)
print("\n   Hình 3.11 tr.160 đọc từ lớp vectơ:")
for ten, Q in (("đa thức", K11A), ("sigmoid", K11B)):
    print("      %-9s: %3d điểm · x' ∈ [%+.3f ; %+.3f] · đỉnh %.6f tại x' = %+.4f · giá trị tại x'=±1: %.6f / %.6f"
          % (ten, len(Q), Q[0, 0], Q[-1, 0], Q[:, 1].max(), Q[np.argmax(Q[:, 1]), 0],
             Q[0, 1], Q[-1, 1]))
print("   -> chú thích Hình 3.11: 'these are localized functions of x even though the")
print("      corresponding basis functions are nonlocal' — đỉnh của cả hai đều ở x' ≈ 0")


def thiet_ke_sigmoid(x, mu, s):
    return np.c_[np.ones(len(x)), 1.0 / (1.0 + np.exp(-(x[:, None] - mu[None, :]) / s))]


print("\n   dựng lại hai nhân đó (200 điểm trên (−1,1), 9 hàm cơ sở, α và β như trên):")
for ten, Phi_h, Phi_0, Q in (
        ("đa thức", thiet_ke_da_thuc(X200, 9), thiet_ke_da_thuc(np.array([0.0]), 9), K11A),
        ("sigmoid", thiet_ke_sigmoid(X200, MU_H31, S_H31),
         thiet_ke_sigmoid(np.array([0.0]), MU_H31, S_H31), K11B)):
    Phi_m = (thiet_ke_da_thuc(X200, 9) if ten == "đa thức"
             else thiet_ke_sigmoid(X200, MU_H31, S_H31))
    k0 = nhan(Phi_h, Phi_0, Phi_m, AL9, BE0)[0]
    print("      %-9s: đỉnh dựng lại %.6f tại x' = %+.4f · tổng = %.6f · số ô âm %d/200"
          % (ten, k0.max(), X200[int(np.argmax(k0))], k0.sum(), int((k0 < 0).sum())))
print("   bài báo KHÔNG khai α, β, μ_j, s của Hình 3.11, nên chỉ so được HÌNH DẠNG:")
print("      đỉnh của cả hai nhân đọc được nằm ở x' ≈ 0 và cả hai đều tắt dần ra hai biên")

# (3.63): cov[y(x), y(x')] = β⁻¹ k(x, x')
_, SN, _ = hau_nghiem(PhiG, np.zeros(200), AL9, BE0)
_cv = PhiG @ SN @ PhiG.T
print("\n   (3.63) cov[y(x),y(x')] = φ(x)ᵀS_Nφ(x') = β⁻¹k(x,x'): lệch lớn nhất %.3e"
      % np.abs(_cv - K / BE0).max())

# (3.65): k(x,z) = ψ(x)ᵀψ(z) với ψ = β^{1/2} S_N^{1/2} φ
_w, _V = np.linalg.eigh(SN)
_S12 = _V @ np.diag(np.sqrt(np.maximum(_w, 0))) @ _V.T
_psi = np.sqrt(BE0) * PhiG @ _S12
print("   (3.65) k(x,z) = ψ(x)ᵀψ(z) với ψ = β^{1/2}S_N^{1/2}φ: lệch lớn nhất %.3e"
      % np.abs(_psi @ _psi.T - K).max())
print("   trị riêng nhỏ nhất của S_N = %.3e (phải ≥ 0 để căn bậc hai tồn tại)" % _w.min())


# ═══════════════════════════════════════════════════════════════════════════ 13
head("13  (3.64) — 'these weights sum to one'. Đúng tới đâu?")

print("   Bishop tr.160–161 chứng minh KHÔNG hình thức: nếu t_n = 1 với mọi n và mô hình")
print("   khớp dữ liệu ĐÚNG KHÍT thì trung bình dự báo bằng 1, nên Σ_n k(x,x_n) = 1.")
print("   Nhưng trung bình dự báo (3.53) là nghiệm ĐÃ CO, không phải nghiệm khớp khít.")
print()
print("   %-12s %-16s %-16s %s" % ("α", "Σk tại x=0", "Σk tại x=0,5", "|Σk − 1| lớn nhất"))
_bang64 = []
for al_ in (1e-12, 1e-8, 1e-6, 1e-4, 1e-2, AL9, 1.0, 10.0, 100.0):
    _, SNa, _ = hau_nghiem(PhiG, np.zeros(200), al_, BE0)
    Kx = BE0 * PhiG @ SNa @ PhiG.T
    sk = Kx.sum(axis=1)
    j0 = int(np.argmin(np.abs(X200 - 0.0)))
    j5 = int(np.argmin(np.abs(X200 - 0.5)))
    _bang64.append((al_, sk[j0], sk[j5], np.abs(sk - 1).max()))
    print("   %-12.4g %-16.10f %-16.10f %.3e"
          % (al_, sk[j0], sk[j5], np.abs(sk - 1).max()))
_gh = _bang64[0][3]
print("   -> (3.64) chỉ đúng trong GIỚI HẠN α → 0: khi α = 10⁻¹² lệch còn %.3e," % _gh)
print("      còn ở α ≈ %.2f (giá trị dựng lại được Hình 3.8) lệch đã là %.3e."
      % (AL9, [b[3] for b in _bang64 if b[0] == AL9][0]))
print("      Ba điều kiện bài báo NÊU (hàm cơ sở độc lập tuyến tính · N > M · có một hàm cơ sở hằng)")
print("      đều thoả ở đây; điều kiện nó KHÔNG nêu là α → 0.")
_don = all(_bang64[i][3] <= _bang64[i + 1][3] + 1e-15 for i in range(len(_bang64) - 1))
print("   |Σk − 1| tăng đơn điệu theo α: %s" % _don)
assert _bang64[0][3] < 1e-6 < _bang64[-1][3]


# ═══════════════════════════════════════════════════════════════════════════ 14
head("14  Evidence (3.86) — kiểm công thức đóng bằng TÍCH PHÂN SỐ")


def ln_evidence(Phi, t, al, be):
    """(3.86): ln p(t|α,β) = (M/2)lnα + (N/2)lnβ − E(m_N) − (1/2)ln|A| − (N/2)ln(2π)."""
    N, M = Phi.shape
    A = al * np.eye(M) + be * Phi.T @ Phi
    mN = be * np.linalg.solve(A, Phi.T @ t)
    E = be / 2 * np.sum((t - Phi @ mN) ** 2) + al / 2 * mN @ mN          # (3.82)
    s, ld = np.linalg.slogdet(A)
    assert s > 0
    return (M / 2 * np.log(al) + N / 2 * np.log(be) - E - ld / 2
            - N / 2 * np.log(2 * np.pi)), mN, A, E


# so với tích phân số trên mô hình hai tham số của mục 5
_xq = np.array([0.0, 1.0, 2.0])
_tq = np.array([1.0, 2.0, 2.0])
_Pq = thiet_ke_tt(_xq)
for al_, be_ in ((1.0, 2.0), (0.3, 5.0), (3.0, 1.0)):
    lo, _, _, _ = ln_evidence(_Pq, _tq, al_, be_)
    g = np.linspace(-6, 6, 2001)
    W0, W1 = np.meshgrid(g, g, indexing="ij")
    lp = np.zeros_like(W0)
    for n in range(3):
        lp += (-be_ / 2 * (_tq[n] - W0 - W1 * _xq[n]) ** 2
               + .5 * np.log(be_ / (2 * np.pi)))
    lp += (-al_ / 2 * (W0 ** 2 + W1 ** 2) + np.log(al_ / (2 * np.pi)))
    mx = lp.max()
    so = mx + np.log(np.exp(lp - mx).sum() * (g[1] - g[0]) ** 2)
    print("   α = %-5.2g β = %-5.2g  (3.86) = %+.10f   tích phân số = %+.10f   lệch %.3e"
          % (al_, be_, lo, so, abs(lo - so)))
print("   -> công thức đóng (3.86) khớp phép cầu phương trên lưới 2001×2001")

# (3.70)-(3.72): xấp xỉ thô "ln p(D) ≈ ln p(D|w_MAP) + M ln(Δw_hậu/Δw_tiên)"
print("\n   (3.70)–(3.72): xấp xỉ thô của evidence, kiểm bằng cách so với (3.86)")
W9_ = np.array([0.35, 232.37, -5321.83, 48568.31, -231639.30, 640042.26,
                -1061800.52, 1042400.18, -557682.99, 125201.43])
X10_ = np.linspace(0, 1, 10)
T10_ = np.vander(X10_, 10, increasing=True) @ W9_
print("   dùng 10 điểm của mục 1.1 sách Bishop, α = 5×10⁻³, β = 11,1, hàm cơ sở đa thức:")
print("   %-6s %-16s %-16s %-16s %s" % ("bậc", "(3.86) chính xác", "log-likelihood", "−½·ln|A|", "(M/2)lnα − ½ln|A|"))
for m in (0, 1, 2, 3, 5, 9):
    Pp = thiet_ke_da_thuc(X10_, m)
    tt = T10_
    lo, mN, A, E = ln_evidence(Pp, tt, 5e-3, 11.1)
    N = len(tt)
    llh = (N / 2 * np.log(11.1) - N / 2 * np.log(2 * np.pi)
           - 11.1 / 2 * np.sum((tt - Pp @ mN) ** 2))
    _, ld = np.linalg.slogdet(A)
    Mw = Pp.shape[1]
    tho = Mw / 2 * np.log(5e-3) - ld / 2
    print("   %-6d %-16.6f %-16.6f %-16.6f %.6f" % (m, lo, llh, -ld / 2, tho))
print("   kiểm: log-likelihood + phần phạt − (N/2)ln(2π) phải bằng cột đầu ở mọi dòng.")
for m in (0, 1, 2, 3, 5, 9):
    Pp = thiet_ke_da_thuc(X10_, m)
    lo, mN, A, E = ln_evidence(Pp, T10_, 5e-3, 11.1)
    N, Mw = Pp.shape
    llh = (N / 2 * np.log(11.1) - N / 2 * np.log(2 * np.pi)
           - 11.1 / 2 * np.sum((T10_ - Pp @ mN) ** 2))
    _, ld = np.linalg.slogdet(A)
    tho = Mw / 2 * np.log(5e-3) - ld / 2
    assert abs(lo - (llh + tho - 5e-3 / 2 * mN @ mN)) < 1e-9, m
print("   -> đúng ở cả 6 dòng. Hai số hạng của (3.71) hiện ra tách bạch: phần KHỚP là")
print("      log-likelihood, phần PHẠT là (M/2)lnα − (1/2)ln|A| — tức đúng ln của tỉ số bề rộng.")
_pt = []
for m in range(10):
    Pp = thiet_ke_da_thuc(X10_, m)
    _, ld = np.linalg.slogdet(5e-3 * np.eye(m + 1) + 11.1 * Pp.T @ Pp)
    _pt.append(((m + 1) / 2 * np.log(5e-3) - ld / 2, -ld / 2))
print("      Cột −½ln|A| ĐỔI DẤU (%+.6f ở bậc 0 -> %+.6f ở bậc 9) nên một mình nó không phải"
      % (_pt[0][1], _pt[9][1]))
print("      một khoản phạt; cả cụm (M/2)lnα − ½ln|A| thì GIẢM đơn điệu ở cả 10 bậc (%s): %s"
      % (all(_pt[i][0] > _pt[i + 1][0] for i in range(9)),
         " ".join("%.2f" % q[0] for q in _pt)))
print("      Đó đúng là điều (3.72) nói: khoản phạt lớn dần theo số tham số.")


# ═══════════════════════════════════════════════════════════════════════════ 15
head("15  Hình 3.14 tr.168 — đọc 10 giá trị evidence rồi DỰNG LẠI chúng")

TR14 = Truc(298.974, 0.0, 434.390, 8.0, 479.175, -26.0, 579.008, -18.0)
_s14 = {k: s for k, _, s in duong(idx(168))}[15]
E14 = np.array([TR14(q[1], q[2]) for q in _s14])
print("   đường evidence (object 15, %d đoạn, toàn LINETO nên đọc là điểm gãy):" % len(_s14))
print("   %-6s %s" % ("M đọc", "ln p(t|α,β) đọc được"))
for m, v in E14:
    print("   %-6.3f %+.6f" % (m, v))
_mi = np.round(E14[:, 0]).astype(int)
print("   hoành độ đọc được làm tròn: %s -> trục M chạy 0..%d, tức %d điểm"
      % (" ".join(str(int(v)) for v in _mi), int(_mi.max()), len(_mi)))
print("   nhưng trục ngang chỉ IN các nhãn 0, 2, 4, 6, 8 — người đọc dễ tưởng chỉ có 9 điểm")
print("   |M đọc − số nguyên| lớn nhất = %.4f (xác nhận phép quy đổi trục)"
      % np.abs(E14[:, 0] - _mi).max())
assert np.abs(E14[:, 0] - _mi).max() < 0.02 and len(_mi) == 10
_dinh = int(_mi[int(np.argmax(E14[:, 1]))])
print("   đỉnh của đường đọc được ở M = %d — chú thích ghi 'the evidence favours the model with M = 3'"
      % _dinh)
assert _dinh == 3

# 10 điểm dữ liệu của mục 1.1, dựng lại từ cột M = 9 của Bảng 1.1 tr.8 (xem v_poly.py mục 7)
W9 = np.array([0.35, 232.37, -5321.83, 48568.31, -231639.30, 640042.26,
               -1061800.52, 1042400.18, -557682.99, 125201.43])
X10 = np.linspace(0, 1, 10)
T10 = np.vander(X10, 10, increasing=True) @ W9
print("\n   10 điểm của mục 1.1 (dựng lại từ cột M = 9 Bảng 1.1 tr.8, xem bài Polynomial Regression):")
print("   t = " + " ".join("%+.4f" % v for v in T10))
print("   nhiễu t − sin(2πx): sd %.6f — mục 1.1 dùng sd 0,3" % (T10 - np.sin(2 * np.pi * X10)).std(ddof=1))

AL14 = 5e-3
BE14 = 11.1
print("\n   bài báo tr.167 khai α = 5×10⁻³ và 'a prior of the form (1.65)'; tr.171 khai β = 11,1.")
print("   Dựng lại bằng ĐÚNG công thức (3.86) với hai giá trị đó:")
EV86 = np.array([ln_evidence(thiet_ke_da_thuc(X10, m), T10, AL14, BE14)[0] for m in _mi])
_le = EV86 - E14[:, 1]
print("   %-4s %-16s %-16s %-12s %s" % ("M", "Hình 3.14 đọc được", "(3.86) dựng lại", "lệch", "lệch/(M+1)"))
for j, m in enumerate(_mi):
    print("   %-4d %+-16.6f %+-16.6f %+-12.6f %.6f" % (m, E14[j, 1], EV86[j], _le[j], _le[j] / (m + 1)))
print("   đỉnh của (3.86) ở M = %d, đỉnh của đường đọc được ở M = %d"
      % (int(_mi[int(np.argmax(EV86))]), _dinh))
print("   -> lệch KHÔNG ngẫu nhiên: nó TUYẾN TÍNH theo số tham số. Khớp một đường thẳng:")
_Ad = np.c_[np.ones(10), _mi + 1]
_ab, _, _, _ = np.linalg.lstsq(_Ad, _le, rcond=None)
_du = _le - _Ad @ _ab
print("      lệch ≈ %+.6f + %+.6f·(M+1)   phần dư lớn nhất %.6f" % (_ab[0], _ab[1], np.abs(_du).max()))
print("      ½·ln(2π) = %.6f ;  hệ số góc đo được %.6f ;  lệch %.6f"
      % (np.log(2 * np.pi) / 2, _ab[1], abs(_ab[1] - np.log(2 * np.pi) / 2)))
print("      hệ số chặn đo được %.6f — cùng bậc với sai số đọc vectơ, coi như 0" % _ab[0])


def ln_evidence_bien(Phi, t, al, be):
    """(3.86) THIẾU một thừa số (2π)^{M/2} — tức bỏ hằng chuẩn hoá của tích phân (3.85)."""
    M = Phi.shape[1]
    return ln_evidence(Phi, t, al, be)[0] - M / 2 * np.log(2 * np.pi)


EVB = np.array([ln_evidence_bien(thiet_ke_da_thuc(X10, m), T10, AL14, BE14) for m in _mi])
print("\n   Cách đọc thứ hai: đường vẽ ra = (3.86) − (M/2)·ln(2π), tức thiếu ĐÚNG MỘT thừa số")
print("   (2π)^{M/2} — chính hằng chuẩn hoá của tích phân Gauss ở (3.85).")
print("   %-4s %-16s %-16s %s" % ("M", "Hình 3.14 đọc được", "bản thiếu 2π", "lệch"))
for j, m in enumerate(_mi):
    print("   %-4d %+-16.6f %+-16.6f %+.6f" % (m, E14[j, 1], EVB[j], EVB[j] - E14[j, 1]))
_sb = np.abs(EVB - E14[:, 1])
print("   lệch lớn nhất %.6f trên 10 ô · lệch trung bình %.6f · sai số đọc vectơ %.6f"
      % (_sb.max(), _sb.mean(), _ss))
print("   -> %d/10 ô khớp trong 0,03 (mức đọc được của hình)" % int((_sb < 0.03).sum()))

print("\n   PHÂN XỬ hai cách đọc: quét cả mặt phẳng (α, β) cho TỪNG cách, lấy lệch lớn nhất nhỏ nhất")
_kq = {}
for _bien in (False, True):
    _bt = None
    for _la in np.linspace(np.log(1e-5), np.log(10.0), 120):
        for _lb in np.linspace(np.log(1.0), np.log(200.0), 120):
            al_, be_ = np.exp(_la), np.exp(_lb)
            v = np.array([(ln_evidence_bien if _bien else
                           (lambda P, t, a, b: ln_evidence(P, t, a, b)[0]))(
                               thiet_ke_da_thuc(X10, m), T10, al_, be_) for m in _mi])
            e = np.abs(v - E14[:, 1]).max()
            if _bt is None or e < _bt[0]:
                _bt = (e, al_, be_, int(_mi[int(np.argmax(v))]))
    _kq[_bien] = _bt
    print("   %-22s lệch lớn nhất tốt nhất = %-10.6f tại α = %-11.6g β = %-9.4f (đỉnh M = %d)"
          % ("ĐÚNG (3.86)" if not _bien else "bản thiếu (2π)^{M/2}", _bt[0], _bt[1], _bt[2], _bt[3]))
print("   tỉ số hai lệch: %.2f lần" % (_kq[False][0] / _kq[True][0]))
print("   và α, β mà bản thiếu (2π) dò ra (%.6g ; %.4f) KHỚP hai con số bài báo tự in"
      % (_kq[True][1], _kq[True][2]))
print("      (α = 5×10⁻³ tr.167 và β = 11,1 tr.171); còn bản đúng (3.86) đòi α = %.6g,"
      % _kq[False][1])
print("      tức lệch con số của chính bài báo %.1f lần." % (AL14 / _kq[False][1]))
assert _kq[True][0] < _kq[False][0] / 5

print("\n   HỆ QUẢ: thừa số thiếu là một khoản phạt %.6f mỗi tham số, nên nó đổi CHỖ ĐỈNH."
      % (np.log(2 * np.pi) / 2))
print("   %-8s %-22s %-22s %s" % ("β", "đỉnh theo ĐÚNG (3.86)", "đỉnh theo bản thiếu 2π", "biên thắng"))
for be_ in (8.0, 9.0, 10.0, BE14, 1 / 0.09, 13.0, 15.0, 20.0):
    v1 = np.array([ln_evidence(thiet_ke_da_thuc(X10, m), T10, AL14, be_)[0] for m in _mi])
    v2 = v1 - (np.array(_mi) + 1) / 2 * np.log(2 * np.pi)
    s1 = np.sort(v1)[::-1]
    s2 = np.sort(v2)[::-1]
    print("   %-8.4f M = %-20d M = %-20d %.4f / %.4f"
          % (be_, int(_mi[int(np.argmax(v1))]), int(_mi[int(np.argmax(v2))]),
             s1[0] - s1[1], s2[0] - s2[1]))
print("   -> theo ĐÚNG (3.86) thì evidence chọn M = 4, không phải M = 3 như chú thích Hình 3.14 nói.")
print("      Không kết luận nào KHÁC của chương đổi vì chỗ này: mọi lập luận về sau dùng (3.86),")
print("      và bản thân câu 'evidence thích mô hình đơn giản nhất giải thích được dữ liệu' vẫn đúng.")


# ═══════════════════════════════════════════════════════════════════════════ 16
head("16  (3.87)–(3.92) — γ, số tham số HIỆU DỤNG, và miền 0 ≤ γ ≤ M")


def gamma_ev(Phi, al, be):
    """(3.91) γ = Σ λ_i/(α+λ_i) với λ_i là trị riêng của βΦᵀΦ."""
    lam = np.linalg.eigvalsh(be * Phi.T @ Phi)
    return float(np.sum(lam / (al + lam))), lam


Phi25 = thiet_ke_gauss(X25, MU9, S9)
M25 = Phi25.shape[1]
print("   mô hình 9 hàm cơ sở Gauss + φ₀ -> M = %d ; N = %d ; β = 11,1" % (M25, len(X25)))
print("   %-12s %-12s %-14s %-14s %s" % ("α", "γ", "γ/M", "α mᵀm (3.90)", "lệch (3.90)"))
_gg = []
for al_ in (1e-8, 1e-4, 1e-2, 0.1, 1.0, 10.0, 1e3, 1e6, 1e10):
    g_, lam = gamma_ev(Phi25, al_, BE0)
    mN, _, _ = hau_nghiem(Phi25, T25, al_, BE0)
    print("   %-12.4g %-12.6f %-14.6f %-14.6f %.3e"
          % (al_, g_, g_ / M25, al_ * mN @ mN, abs(al_ * mN @ mN - g_)))
    _gg.append(g_)
print("   γ ∈ [0, M] ở mọi giá trị α: %s  (nhỏ nhất %.6f, lớn nhất %.6f, M = %d)"
      % (all(0 <= g <= M25 + 1e-9 for g in _gg), min(_gg), max(_gg), M25))
print("   γ giảm đơn điệu khi α tăng: %s"
      % all(_gg[i] >= _gg[i + 1] for i in range(len(_gg) - 1)))
assert all(0 <= g <= M25 + 1e-9 for g in _gg)
print("   -> (3.90) αmᵀm = γ là ĐẲNG THỨC chỉ tại điểm dừng của evidence, không phải mọi α:")
print("      cột cuối cho thấy lệch lớn ở α xa điểm dừng — đó là điều (3.92) dùng làm phép lặp")


# ═══════════════════════════════════════════════════════════════════════════ 17
head("17  (3.92) và (3.95) — vòng lặp evidence, và nó có hội tụ không")


def lap_evidence(Phi, t, al=1.0, be=1.0, so=500, tol=1e-12, co_be=True):
    N, M = Phi.shape
    G = Phi.T @ Phi
    w = np.linalg.eigvalsh(G)
    ls = []
    for i in range(so):
        lam = be * w
        g = float(np.sum(lam / (al + lam)))
        mN, _, _ = hau_nghiem(Phi, t, al, be)
        al_m = g / (mN @ mN)                                   # (3.92)
        be_m = (N - g) / np.sum((t - Phi @ mN) ** 2) if co_be else be   # (3.95)
        ls.append((al, be, g, ln_evidence(Phi, t, al, be)[0]))
        if abs(al_m - al) < tol * al and abs(be_m - be) < tol * be:
            al, be = al_m, be_m
            break
        al, be = al_m, be_m
    return al, be, g, i + 1, ls


_al_h, _be_h, _g_h, _n_h, _ls = lap_evidence(Phi25, T25)
print("   xuất phát α = β = 1, cập nhật (3.92) + (3.95) xen kẽ:")
print("   %-6s %-14s %-14s %-12s %s" % ("vòng", "α", "β", "γ", "ln evidence"))
for i in (0, 1, 2, 4, 9, 19, min(49, _n_h - 1)):
    if i < len(_ls):
        print("   %-6d %-14.8f %-14.8f %-12.6f %+.6f" % (i + 1, _ls[i][0], _ls[i][1], _ls[i][2], _ls[i][3]))
print("   hội tụ sau %d vòng: α = %.8f · β = %.8f · γ = %.6f (M = %d)"
      % (_n_h, _al_h, _be_h, _g_h, M25))
print("   1/√β̂ = %.6f ; độ lệch chuẩn nhiễu thật của mục 1.1 là 0,3" % (_be_h ** -0.5))
_ev = [q[3] for q in _ls]
_tang = all(_ev[i + 1] >= _ev[i] - 1e-9 for i in range(len(_ev) - 1))
print("   ln evidence tăng đơn điệu qua các vòng: %s (từ %+.6f lên %+.6f)"
      % (_tang, _ev[0], _ev[-1]))

print("\n   giữ β = 11,1 cố định, chỉ tối ưu α bằng (3.92) — đúng thiết lập Hình 3.16:")
_al_c, _, _g_c, _n_c, _ = lap_evidence(Phi25, T25, al=1.0, be=BE0, co_be=False)
print("   α̂ = %.8f · ln α̂ = %.6f · γ = %.6f · %d vòng" % (_al_c, np.log(_al_c), _g_c, _n_c))

print("\n   xuất phát khác nhau có cho cùng nghiệm không (evidence có nhiều cực trị không):")
for a0, b0 in ((1e-6, 1e-3), (1.0, 1.0), (1e3, 1e3), (1e-3, 1e3), (1e3, 1e-3)):
    a, b, g, n, _ = lap_evidence(Phi25, T25, a0, b0)
    print("   (α₀,β₀) = (%-8.4g %-8.4g) -> α̂ = %.8f · β̂ = %.8f · γ = %.6f · %d vòng"
          % (a0, b0, a, b, g, n))

# ═══════════════════════════════════════════════════════════════════════════ 18
head("18  Hình 3.16 tr.172 — hình KHÔNG CÓ trục dọc, nên chỉ đọc được HOÀNH ĐỘ")

print("   lớp text của trang 172 chỉ chứa nhãn 'ln α' và '−5 0 5'; không có một con số nào")
print("   trên trục dọc của cả hai ô. Ba đại lượng đọc được là ba HOÀNH ĐỘ:")
print("     · chỗ hai đường của ô TRÁI cắt nhau (γ và 2αE_W)")
print("     · đỉnh của đường log evidence ở ô PHẢI")
print("     · cực tiểu của đường sai số tập kiểm tra ở ô PHẢI")

TR16L = Truc(48.06, -5.0, 233.73, 5.0, 0.0, 0.0, 1.0, 1.0)     # trục dọc KHÔNG chuẩn hoá được
TR16R = Truc(268.56, -5.0, 454.23, 5.0, 0.0, 0.0, 1.0, 1.0)
G16 = doc_duong(idx(172), 6, TR16L)      # đỏ = γ
W16 = doc_duong(idx(172), 5, TR16L)      # xanh = 2αE_W(m_N)
EV16 = doc_duong(idx(172), 20, TR16R)    # đỏ = log evidence
TE16 = doc_duong(idx(172), 19, TR16R)    # xanh = sai số tập kiểm tra

_lx = np.linspace(max(G16[0, 0], W16[0, 0]) + 1e-6, min(G16[-1, 0], W16[-1, 0]) - 1e-6, 4001)
_g = np.interp(_lx, G16[:, 0], G16[:, 1])
_w = np.interp(_lx, W16[:, 0], W16[:, 1])
_j = int(np.argmin(np.abs(_g - _w)))
print("\n   ô TRÁI : đường γ đọc được trên ln α ∈ [%+.3f ; %+.3f] (%d điểm)"
      % (G16[0, 0], G16[-1, 0], len(G16)))
print("            đường 2αE_W đọc được trên ln α ∈ [%+.3f ; %+.3f] (%d điểm)"
      % (W16[0, 0], W16[-1, 0], len(W16)))
print("            hai đường CẮT NHAU tại ln α = %+.6f  (α = %.6f)" % (_lx[_j], np.exp(_lx[_j])))
print("   ô PHẢI : đỉnh log evidence tại ln α = %+.6f  (α = %.6f)"
      % (EV16[int(np.argmax(EV16[:, 1])), 0], np.exp(EV16[int(np.argmax(EV16[:, 1])), 0])))
print("            cực tiểu sai số kiểm tra tại ln α = %+.6f  (α = %.6f)"
      % (TE16[int(np.argmin(TE16[:, 1])), 0], np.exp(TE16[int(np.argmin(TE16[:, 1])), 0])))
_do_cheo = abs(_lx[_j] - EV16[int(np.argmax(EV16[:, 1])), 0])
print("   chú thích nói 'the peak coincides with the crossing point': hai hoành độ lệch %.6f" % _do_cheo)
print("   chú thích nói 'the evidence maximum occurs CLOSE TO the point of best generalization':")
print("      lệch giữa đỉnh evidence và cực tiểu sai số kiểm tra = %.6f trong ln α"
      % abs(EV16[int(np.argmax(EV16[:, 1])), 0] - TE16[int(np.argmin(TE16[:, 1])), 0]))

print("\n   dựng lại ba hoành độ đó. Hai bộ dữ liệu ứng cử: 25 điểm của Hình 3.8 và 10 điểm của mục 1.1.")
_gr16 = np.random.default_rng(404)
_xk = _gr16.uniform(0, 1, 2000)
_tk = np.sin(2 * np.pi * _xk) + _gr16.normal(0, 0.3, 2000)
_ln = np.linspace(-5, 5, 4001)
print("   %-22s %-16s %-16s %-16s %s"
      % ("bộ dữ liệu", "giao γ = 2αE_W", "đỉnh evidence", "cực tiểu KT", "|w| lớn nhất tại γ→M"))
_kq16 = {}
for _ten, _xx, _tt in (("25 điểm (Hình 3.8)", X25, T25), ("10 điểm (mục 1.1)", X10, T10)):
    Ph = thiet_ke_gauss(_xx, MU9, S9)
    Pk = thiet_ke_gauss(_xk, MU9, S9)
    gg, ww, ee, te = [], [], [], []
    for la in _ln:
        al_ = np.exp(la)
        g_, _ = gamma_ev(Ph, al_, BE0)
        mN, _, _ = hau_nghiem(Ph, _tt, al_, BE0)
        gg.append(g_)
        ww.append(al_ * mN @ mN)
        ee.append(ln_evidence(Ph, _tt, al_, BE0)[0])
        te.append(np.sqrt(np.mean((_tk - Pk @ mN) ** 2)))
    gg, ww, ee, te = map(np.array, (gg, ww, ee, te))
    jc = int(np.argmin(np.abs(gg - ww)))
    mg, _, _ = hau_nghiem(Ph, _tt, 1e-10, BE0)
    _kq16[_ten] = (_ln[jc], _ln[int(np.argmax(ee))], _ln[int(np.argmin(te))], np.abs(mg).max())
    print("   %-22s %+-16.6f %+-16.6f %+-16.6f %.6f"
          % (_ten, _ln[jc], _ln[int(np.argmax(ee))], _ln[int(np.argmin(te))], np.abs(mg).max()))
_TR17 = Truc(291.39, 0.0, 445.39, 10.0, 206.82, -2.0, 310.73, 2.0)
_w17max = max(np.abs(doc_duong(idx(172), k, _TR17)[:, 1]).max() for k in (97, 115))
print("   HÌNH đọc được          %+-16.6f %+-16.6f %+-16.6f %.6f"
      % (_lx[_j], EV16[int(np.argmax(EV16[:, 1])), 0],
         TE16[int(np.argmin(TE16[:, 1])), 0], _w17max))
print("   (3.90) đòi giao điểm và đỉnh evidence TRÙNG nhau: dựng lại lệch %.3e / %.3e"
      % (abs(_kq16["25 điểm (Hình 3.8)"][0] - _kq16["25 điểm (Hình 3.8)"][1]),
         abs(_kq16["10 điểm (mục 1.1)"][0] - _kq16["10 điểm (mục 1.1)"][1])))
_d25 = abs(_lx[_j] - _kq16["25 điểm (Hình 3.8)"][0])
_d10 = abs(_lx[_j] - _kq16["10 điểm (mục 1.1)"][0])
print("   giao điểm của HÌNH (ln α = %+.4f) so với hai bản dựng lại: lệch %.4f (25 điểm) và %.4f (10 điểm)"
      % (_lx[_j], _d25, _d10))
print("   -> bộ %s khớp hơn %.1f lần. Hình 3.16 dùng chính 25 điểm của Hình 3.8."
      % ("25 điểm" if _d25 < _d10 else "10 điểm", max(_d25, _d10) / min(_d25, _d10)))
print("      Lệch còn lại %.4f trong ln α, tức %.2f %% trong α — nằm trong mức đọc được của một GIAO ĐIỂM."
      % (_d25, 100 * abs(np.exp(_lx[_j]) - np.exp(_kq16["25 điểm (Hình 3.8)"][0])) / np.exp(_lx[_j])))
_al_ev = np.exp(_kq16["25 điểm (Hình 3.8)"][1])
_al_kt = np.exp(_kq16["25 điểm (Hình 3.8)"][2])
print("\n   'the evidence maximum occurs close to the point of best generalization' — đo phần 'close to':")
print("      α theo evidence = %.6f · α tốt nhất trên tập kiểm tra 2000 điểm = %.6f · tỉ số %.4f lần"
      % (_al_ev, _al_kt, _al_kt / _al_ev))
print("      trên HÌNH: α đỉnh evidence = %.6f · α cực tiểu kiểm tra = %.6f · tỉ số %.4f lần"
      % (np.exp(EV16[int(np.argmax(EV16[:, 1])), 0]),
         np.exp(TE16[int(np.argmin(TE16[:, 1])), 0]),
         np.exp(TE16[int(np.argmin(TE16[:, 1])), 0] - EV16[int(np.argmax(EV16[:, 1])), 0])))
print("      -> 'close to' nghĩa là CÙNG BẬC, không phải trùng: chênh một thừa số 3–4 lần trong α.")
print("      Cái giá của chỗ chênh đó, đo bằng sai số căn quân phương trên tập kiểm tra:")
Ph = thiet_ke_gauss(X25, MU9, S9)
Pk = thiet_ke_gauss(_xk, MU9, S9)
for _ten, _a in (("α theo evidence", _al_ev), ("α tốt nhất trên KT", _al_kt)):
    mN, _, _ = hau_nghiem(Ph, T25, _a, BE0)
    print("         %-20s α = %-10.6f RMSE kiểm tra = %.6f" % (_ten, _a, np.sqrt(np.mean((_tk - Pk @ mN) ** 2))))


# ═══════════════════════════════════════════════════════════════════════════ 19
head("19  Hình 3.17 tr.172 — mười đường w_i theo γ, đọc được cả hai trục")

TR17 = Truc(291.39, 0.0, 445.39, 10.0, 206.82, -2.0, 310.73, 2.0)
_ob17 = [k for k, net, seg in duong(idx(172)) if 90 <= k <= 120 and len(seg) > 20]
print("   số đường dò được trên vùng Hình 3.17: %d (bài báo vẽ 10 đường, nhãn 0..9)" % len(_ob17))
D17 = {}
for k in _ob17:
    Q = doc_duong(idx(172), k, TR17)
    D17[k] = Q
print("   %-6s %-10s %-12s %-12s %-12s %s"
      % ("object", "số điểm", "γ nhỏ nhất", "γ lớn nhất", "w tại γ lớn nhất", "|w| lớn nhất"))
for k in sorted(D17):
    Q = D17[k]
    print("   %-6d %-10d %-12.4f %-12.4f %-12.6f %.6f"
          % (k, len(Q), Q[0, 0], Q[-1, 0], Q[-1, 1], np.abs(Q[:, 1]).max()))
_gmax = max(D17[k][-1, 0] for k in D17)
print("   γ lớn nhất trên hình = %.4f ; chú thích nói γ chạy trong 0 ≤ γ ≤ M với M = 10" % _gmax)

print("\n   dựng lại: quét α từ lớn về nhỏ, mỗi α cho một (γ, w). Hệ số tại γ → M:")
mN_gh, _, _ = hau_nghiem(Phi25, T25, 1e-10, BE0)
print("   w_i khi α → 0 (tức γ → M): " + " ".join("%+.4f" % v for v in mN_gh))
print("   |w_i| lớn nhất khi α → 0 = %.6f ; trên hình, |w| lớn nhất đọc được = %.6f"
      % (np.abs(mN_gh).max(), max(np.abs(D17[k][:, 1]).max() for k in D17)))
print("   %-8s %-10s %s" % ("α", "γ", "|w| lớn nhất"))
for al_ in (1e3, 1e2, 10.0, 1.0, 0.1, 1e-2, 1e-4, 1e-8):
    g_, _ = gamma_ev(Phi25, al_, BE0)
    mN, _, _ = hau_nghiem(Phi25, T25, al_, BE0)
    print("   %-8.4g %-10.6f %.6f" % (al_, g_, np.abs(mN).max()))
print("   -> chiều đúng: α giảm thì γ tăng về M và |w| nở ra, đúng điều chú thích Hình 3.17 nói.")

print("\n   Hình 3.17 cho một phép đo SẮC cho bề rộng s: biên độ |w| lớn nhất tại γ đã đọc được.")
_g17 = max(D17[k][-1, 0] for k in D17)
_w17 = max(np.abs(D17[k][:, 1]).max() for k in D17)


def alpha_theo_gamma(Phi, g_dich, be):
    lo, hi = -18.0, 8.0
    for _ in range(90):
        mid = (lo + hi) / 2
        if gamma_ev(Phi, np.exp(mid), be)[0] > g_dich:
            lo = mid
        else:
            hi = mid
    return np.exp((lo + hi) / 2)


print("   %-8s %-16s %-18s %-18s %s"
      % ("s", "giao ln α", "lệch với hình", "|w| lớn nhất", "lệch với hình"))
_bs = None
for s_ in np.linspace(0.08, 0.16, 33):
    Ph = thiet_ke_gauss(X25, MU9, s_)
    _lnq = np.linspace(-6, 6, 1201)
    gq = np.array([gamma_ev(Ph, np.exp(v), BE0)[0] for v in _lnq])
    wq = np.array([np.exp(v) * (hau_nghiem(Ph, T25, np.exp(v), BE0)[0] @
                               hau_nghiem(Ph, T25, np.exp(v), BE0)[0]) for v in _lnq])
    lc = _lnq[int(np.argmin(np.abs(gq - wq)))]
    m1, _, _ = hau_nghiem(Ph, T25, alpha_theo_gamma(Ph, _g17, BE0), BE0)
    e1, e2 = abs(lc - _lx[_j]), abs(np.abs(m1).max() - _w17)
    if _bs is None or e1 / 0.02 + e2 / 0.05 < _bs[0]:
        _bs = (e1 / 0.02 + e2 / 0.05, s_, lc, np.abs(m1).max(), e1, e2)
    if abs(s_ * 1000 - round(s_ * 1000)) < 1e-6 and int(round(s_ * 1000)) % 5 == 0:
        print("   %-8.3f %+-16.4f %-18.4f %-18.4f %.4f"
              % (s_, lc, e1, np.abs(m1).max(), e2))
print("   TỐT NHẤT chung cho cả hai hình: s = %.4f -> giao ln α = %+.4f (hình %+.4f, lệch %.4f)"
      % (_bs[1], _bs[2], _lx[_j], _bs[4]))
print("                                    |w| lớn nhất = %.4f (hình %.4f, lệch %.4f)"
      % (_bs[3], _w17, _bs[5]))
print("   so với s = %.4f dò được từ Hình 3.8: %s"
      % (S9, "cùng bậc, lệch %.1f %%" % (100 * abs(_bs[1] - S9) / S9)))
print("   -> ba hình độc lập (3.8 · 3.16 · 3.17) cùng chỉ về một bề rộng s ≈ 0,11–0,13.")
print("      Đó là bằng chứng mạnh hơn hẳn một hình, vì ba phép đo dùng ba đại lượng khác nhau:")
print("      dải ±σ · hoành độ một GIAO ĐIỂM · biên độ lớn nhất của hệ số.")


# ═══════════════════════════════════════════════════════════════════════════ 20
head("20  (3.98)–(3.99) — xấp xỉ khi N ≫ M, và nó sai bao nhiêu khi N nhỏ")

print("   %-8s %-10s %-10s %-14s %-14s %-14s %s"
      % ("N", "γ", "M", "α̂ (3.92)", "α̂ (3.98)", "β̂ (3.95)", "β̂ (3.99)"))
_gr = np.random.default_rng(11)
for N in (25, 50, 200, 1000, 10000):
    xx = np.sort(_gr.uniform(0, 1, N))
    tt = np.sin(2 * np.pi * xx) + _gr.normal(0, 0.3, N)
    Ph = thiet_ke_gauss(xx, MU9, S9)
    a_, b_, g_, _, _ = lap_evidence(Ph, tt)
    mN, _, _ = hau_nghiem(Ph, tt, a_, b_)
    EW = 0.5 * mN @ mN
    ED = 0.5 * np.sum((tt - Ph @ mN) ** 2)
    print("   %-8d %-10.6f %-10d %-14.6f %-14.6f %-14.6f %.6f"
          % (N, g_, Ph.shape[1], a_, Ph.shape[1] / (2 * EW), b_, N / (2 * ED)))
print("   -> (3.98)/(3.99) thay γ bằng M và N − γ bằng N; sai số của chúng co lại theo N,")
print("      và ở N = 25 thì γ/M = %.4f nên xấp xỉ đó còn lệch rõ." % (_g_h / M25))


# ═══════════════════════════════════════════════════════════════════════════ 21
head("21  Khối SS — cách cũ tắc ở đâu, đo bằng SỐ trên chính dữ liệu của Bishop")

print("   Bishop tr.152 nêu bế tắc bằng lời: 'Independent hold-out data can be used to determine")
print("   model complexity ... but this can be both computationally expensive and WASTEFUL of")
print("   valuable data.' Ba chỗ đó đo được.")

# (a) không chính quy hoá: bậc 9 trên 10 điểm
print("\n   (a) không có tiên nghiệm — đa thức bậc 9 khớp 10 điểm của mục 1.1:")
_Pt = thiet_ke_da_thuc(X10, 9)
_wml = np.linalg.lstsq(_Pt, T10, rcond=None)[0]
_gk = np.random.default_rng(909)
_xk2 = _gk.uniform(0, 1, 2000)
_tk2 = np.sin(2 * np.pi * _xk2) + _gk.normal(0, 0.3, 2000)
_rt = np.sqrt(np.mean((T10 - _Pt @ _wml) ** 2))
_rk = np.sqrt(np.mean((_tk2 - thiet_ke_da_thuc(_xk2, 9) @ _wml) ** 2))
print("       RMSE huấn luyện = %.3e · RMSE kiểm tra (2000 điểm) = %.6f · tỉ số %.3e lần"
      % (_rt, _rk, _rk / max(_rt, 1e-300)))
print("       |w| lớn nhất = %.4g (bài báo in %.4g ở cột M = 9 Bảng 1.1 tr.8)"
      % (np.abs(_wml).max(), np.abs(W9).max()))
_al_p, _be_p, _g_p, _n_p, _ = lap_evidence(_Pt, T10)
_mp, _, _ = hau_nghiem(_Pt, T10, _al_p, _be_p)
_rkp = np.sqrt(np.mean((_tk2 - thiet_ke_da_thuc(_xk2, 9) @ _mp) ** 2))
print("       cùng bậc 9, α và β lấy từ evidence (%d vòng): α̂ = %.6g · β̂ = %.6f · γ = %.4f"
      % (_n_p, _al_p, _be_p, _g_p))
print("       RMSE kiểm tra = %.6f -> giảm %.3e lần · |w| lớn nhất = %.4f"
      % (_rkp, _rk / _rkp, np.abs(_mp).max()))

# (b) chi phí: số lần phân tích ma trận
print("\n   (b) chi phí — đếm số lần GIẢI một hệ M×M (phép tính đắt nhất của cả hai đường):")
Ph10 = thiet_ke_gauss(X10, MU9, S9)
for K, G in ((5, 20), (5, 100), (10, 100), (10, 1000)):
    print("       CV %2d mảnh × lưới %4d giá trị λ = %5d lần giải, cộng 1 lần khớp lại trên toàn bộ"
          % (K, G, K * G))
_al_e, _be_e, _g_e, _n_e, _ = lap_evidence(Ph10, T10)
print("       evidence (3.92)+(3.95): %d vòng × 2 lần giải = %d lần giải, KHÔNG cần lưới"
      % (_n_e, 2 * _n_e))
print("       tỉ số so với CV 10 mảnh × 100 giá trị: %.1f lần" % (1000 / (2 * _n_e)))

# (c) CV chọn λ không ổn định; evidence thì tất định
print("\n   (c) CV 'lãng phí dữ liệu' — λ do CV chọn đổi theo cách chia mảnh:")
_gl = np.exp(np.linspace(np.log(1e-6), np.log(1e3), 120))
_chon = []
for _sd in range(60):
    g = np.random.default_rng(3000 + _sd)
    per = g.permutation(10)
    mang = [per[i::5] for i in range(5)]
    sc = []
    for lam in _gl:
        e = 0.0
        for m_ in mang:
            tr = np.setdiff1d(np.arange(10), m_)
            A = Ph10[tr].T @ Ph10[tr] + lam * np.eye(Ph10.shape[1])
            w = np.linalg.solve(A, Ph10[tr].T @ T10[tr])
            e += np.sum((T10[m_] - Ph10[m_] @ w) ** 2)
        sc.append(e)
    _chon.append(_gl[int(np.argmin(sc))])
_chon = np.array(_chon)
print("       60 cách chia mảnh -> λ chọn được: nhỏ nhất %.6g · lớn nhất %.6g · trải %.1f lần"
      % (_chon.min(), _chon.max(), _chon.max() / _chon.min()))
print("       trung vị %.6g ; số giá trị PHÂN BIỆT được chọn: %d"
      % (float(np.median(_chon)), len(set(np.round(np.log(_chon), 6)))))
_lam_ev = _al_e / _be_e
print("       evidence cho λ = α̂/β̂ = %.6g — MỘT giá trị, không phụ thuộc cách chia mảnh" % _lam_ev)
print("       (mọi lần chạy lap_evidence từ cùng dữ liệu cho cùng con số: xem mục 17)")
_rk_cv, _rk_ev = [], []
for lam in (float(np.median(_chon)), _lam_ev):
    A = Ph10.T @ Ph10 + lam * np.eye(Ph10.shape[1])
    w = np.linalg.solve(A, Ph10.T @ T10)
    _rk_cv.append(np.sqrt(np.mean((_tk2 - thiet_ke_gauss(_xk2, MU9, S9) @ w) ** 2)))
print("       RMSE kiểm tra: λ theo trung vị CV = %.6f · λ theo evidence = %.6f · chênh %.2f %%"
      % (_rk_cv[0], _rk_cv[1], 100 * (_rk_cv[0] - _rk_cv[1]) / _rk_cv[1]))

# (d) OLS không trả về độ bất định
print("\n   (d) điều OLS KHÔNG trả về được — một con số cho câu 'ta chắc tới đâu':")
mN10, SN10, _ = hau_nghiem(Ph10, T10, _al_e, _be_e)
for xv in (0.05, 0.5, 0.95, 1.3):
    Phx = thiet_ke_gauss(np.array([xv]), MU9, S9)
    sg = np.sqrt(1 / _be_e + float((Phx @ SN10 @ Phx.T)[0, 0]))
    wl = np.linalg.lstsq(Ph10, T10, rcond=None)[0]
    print("       x = %-5.2f  Bayes: %+.4f ± %.4f (95 %%: ±%.4f)   OLS: %+.4f ± (không có)"
          % (xv, float((Phx @ mN10)[0]), sg, 1.96 * sg, float((Phx @ wl)[0])))
print("\n       Ở x = 1,3 thì GIÁ TRỊ THẬT là sin(2π·1,3) = %.6f — DƯƠNG, không phải âm."
      % np.sin(2 * np.pi * 1.3))
_p13 = thiet_ke_gauss(np.array([1.3]), MU9, S9)
_tb13 = float((_p13 @ mN10)[0])
_sg13 = float(np.sqrt(1 / _be_e + float((_p13 @ SN10 @ _p13.T)[0, 0])))
_wl13 = float((_p13 @ np.linalg.lstsq(Ph10, T10, rcond=None)[0])[0])
_that = float(np.sin(2 * np.pi * 1.3))
print("       Bayes %+.6f (sai %.6f) · bình phương tối thiểu %+.6f (sai %.6f)"
      % (_tb13, abs(_tb13 - _that), _wl13, abs(_wl13 - _that)))
print("       -> ở ĐIỂM NÀY bản co về 0 lại TỆ HƠN: sai số lớn hơn %.4f lần"
      % (abs(_tb13 - _that) / abs(_wl13 - _that)))
print("       khoảng 95 %% của Bayes: [%+.6f ; %+.6f] — có chứa giá trị thật không: %s"
      % (_tb13 - 1.96 * _sg13, _tb13 + 1.96 * _sg13,
         _tb13 - 1.96 * _sg13 <= _that <= _tb13 + 1.96 * _sg13))
print("       giá trị thật nằm NGOÀI cận trên %.6f đơn vị" % (_that - (_tb13 + 1.96 * _sg13)))
print("       tỉ số σ giữa x = 1,3 (ngoài dữ liệu) và x = 0,5 (giữa dữ liệu): %.4f lần"
      % (np.sqrt(1 / _be_e + float((thiet_ke_gauss(np.array([1.3]), MU9, S9) @ SN10
                                    @ thiet_ke_gauss(np.array([1.3]), MU9, S9).T)[0, 0]))
         / np.sqrt(1 / _be_e + float((thiet_ke_gauss(np.array([0.5]), MU9, S9) @ SN10
                                      @ thiet_ke_gauss(np.array([0.5]), MU9, S9).T)[0, 0]))))


# ═══════════════════════════════════════════════════════════════════════════ 22
head("22  scikit-learn — cùng thuật toán, ba chỗ KHÁC phải nói ra")

try:
    import sklearn
    from sklearn.linear_model import ARDRegression, BayesianRidge
    print("   scikit-learn %s" % sklearn.__version__)
    Ph = thiet_ke_gauss(X25, MU9, S9)
    br = BayesianRidge(fit_intercept=False, tol=1e-12, max_iter=2000)
    br.fit(Ph, T25)
    a_, b_, g_, n_, _ = lap_evidence(Ph, T25)
    print("   %-34s %-18s %s" % ("đại lượng", "v_bayes.py", "BayesianRidge"))
    print("   %-34s %-18.8f %.8f" % ("α của Bishop = lambda_ của sklearn", a_, br.lambda_))
    print("   %-34s %-18.8f %.8f" % ("β của Bishop = alpha_ của sklearn", b_, br.alpha_))
    print("   %-34s %-18.8f %.8f" % ("|m_N| lớn nhất", np.abs(hau_nghiem(Ph, T25, a_, b_)[0]).max(),
                                     np.abs(br.coef_).max()))
    print("   lệch tương đối: α %.3e · β %.3e · hệ số %.3e"
          % (abs(a_ - br.lambda_) / a_, abs(b_ - br.alpha_) / b_,
             np.abs(hau_nghiem(Ph, T25, a_, b_)[0] - br.coef_).max()
             / np.abs(br.coef_).max()))
    print("   CHỖ KHÁC 1 — TÊN: `alpha_` của sklearn là β của Bishop (độ chính xác NHIỄU),")
    print("                còn `lambda_` mới là α của Bishop (độ chính xác TIÊN NGHIỆM). Ngược nhau.")
    print("   CHỖ KHÁC 2 — sklearn đặt tiên nghiệm Gamma lên chính α và β (alpha_1, alpha_2,")
    print("                lambda_1, lambda_2, mặc định 1e-6), nên công thức cập nhật KHÔNG đúng")
    print("                bằng (3.92)/(3.95) mà có thêm số hạng. Đo được lệch ở trên.")
    br2 = BayesianRidge(fit_intercept=False, tol=1e-12, max_iter=2000,
                        alpha_1=1e-2, alpha_2=1e-2, lambda_1=1e-2, lambda_2=1e-2)
    br2.fit(Ph, T25)
    print("                đổi bốn tham số Gamma lên 1e-2: lambda_ %.8f -> %.8f (lệch %.2f %%)"
          % (br.lambda_, br2.lambda_, 100 * abs(br2.lambda_ - br.lambda_) / br.lambda_))
    br3 = BayesianRidge(tol=1e-12, max_iter=2000)
    br3.fit(Ph[:, 1:], T25)
    print("   CHỖ KHÁC 3 — `fit_intercept=True` (mặc định) TÂM HOÁ dữ liệu và KHÔNG phạt hệ số chặn;")
    print("                Bishop thì phạt cả w₀ vì φ₀ = 1 nằm trong (3.52). Đo: lambda_ %.8f so với %.8f"
          % (br3.lambda_, br.lambda_))
    _px = thiet_ke_gauss(np.array([0.5]), MU9, S9)
    _y1 = float((_px @ hau_nghiem(Ph, T25, a_, b_)[0])[0])
    _y2 = float(br3.predict(_px[:, 1:])[0])
    print("                dự báo tại x = 0,5: Bishop %+.6f · sklearn mặc định %+.6f · lệch %.6f"
          % (_y1, _y2, abs(_y1 - _y2)))
    ard = ARDRegression(fit_intercept=False, max_iter=2000)
    ard.fit(Ph, T25)
    print("   ARDRegression = MỘT α cho MỖI hệ số (Bishop chương 7, relevance vector machine):")
    print("                số hệ số bị đẩy về dưới 1e-3: %d / %d · lambda_ lớn nhất %.4g"
          % (int((np.abs(ard.coef_) < 1e-3).sum()), len(ard.coef_), np.max(ard.lambda_)))
except ImportError:
    print("   KHÔNG có scikit-learn trên máy này — mục này không chạy được.")


# ═══════════════════════════════════════════════════════════════════════════ 23
head("23  Mảng số cho ba khối mã sinh hình in trong bài — in ĐỦ CHỮ SỐ để dán thẳng")

print("   [A] hai nguồn của phương sai dự báo (3.59), tách rời. x trên [−0,2 ; 1,2], 43 điểm")
XA = np.linspace(-0.2, 1.2, 43)
print("   XA = " + ", ".join("%.8f" % v for v in XA))
for N in (4, 25):
    ii = CHI_SO["N=4"] if N == 4 else list(range(25))
    Ph = thiet_ke_gauss(X25[ii], MU9, S9)
    _, SN, _ = hau_nghiem(Ph, T25[ii], AL9, BE0)
    Pm = thiet_ke_gauss(XA, MU9, S9)
    q = np.einsum("ij,jk,ik->i", Pm, SN, Pm)
    print("   Q%d = " % N + ", ".join("%.8f" % v for v in q))
print("   NHIEU = %.8f   # 1/β với β = 11,1" % (1 / BE0))

print("\n   [B] ba dãy của Hình 3.14 (10 giá trị mỗi dãy)")
print("   DOC   = " + ", ".join("%.6f" % v for v in E14[:, 1]))
print("   CT386 = " + ", ".join("%.6f" % v for v in EV86))
print("   THIEU = " + ", ".join("%.6f" % v for v in EVB))

print("\n   [C] |Σ_n k(x,x_n) − 1| lớn nhất theo α, 25 giá trị α cách đều theo log")
ALC = np.exp(np.linspace(np.log(1e-12), np.log(1e3), 25))
DC = []
for al_ in ALC:
    _, SNa, _ = hau_nghiem(PhiG, np.zeros(200), al_, BE0)
    DC.append(np.abs((BE0 * PhiG @ SNa @ PhiG.T).sum(axis=1) - 1).max())
print("   ALC = " + ", ".join("%.6e" % v for v in ALC))
print("   DC  = " + ", ".join("%.6e" % v for v in DC))


# ═══════════════════════════════════════════════════════════════════════════ 24
head("24  Kiểm kê hình của bài và khung cắt từ PDF")

_ten = sorted(os.listdir(THUMUC)) if os.path.isdir(THUMUC) else []
_crop = [t for t in _ten if t.startswith("paper_bishop2006_")]
_tuve = [t for t in _ten if t.endswith(".png") and not t.startswith("paper_")]
print("   thư mục hình của bài: %d file" % len(_ten))
print("   khung cắt từ PDF Bishop 2006 : %d" % len(_crop))
print("   hình tự vẽ                   : %d  (%s)" % (len(_tuve), " ".join(_tuve)))
print("   khung theo mục của chương:")
for _k, _p in (("mục 3.3", "ct_3_4|ct_3_5|ct_3_6|hinh_3_7|hinh_3_8|hinh_3_9|hinh_3_10|hinh_3_11|mo_dau_3_3|nhan_xet_map|thiet_lap_hinh_3_7|binh_hinh_3_8|canh_bao_ngoai_suy"),
                ("mục 3.4", "3_66|3_67|3_68|3_71|hinh_3_13|3_73|mo_dau_3_4|canh_bao_tien_nghiem"),
                ("mục 3.5", "3_74|3_76|3_77|3_79|3_85|hinh_3_14|binh_hinh_3_14|3_87|3_90|3_93|hinh_3_15|gamma|3_96|3_98|thiet_lap_hinh_3_16|hinh_3_16|hinh_3_17|mo_dau_3_5"),
                ("mục 3.6", "gioi_han_3_6")):
    print("      %-9s %d khung" % (_k, sum(1 for t in _crop if re.search(_p, t))))

print("\n   Tổng kết những chỗ bài báo KHÔNG khai, và cách xử lý:")
for _a, _b in (
        ("μ_j, s của Hình 3.1/3.10/3.11", "dò được từ Hình 3.8 · 3.16 · 3.17, ba hình cùng chỉ s ≈ 0,12"),
        ("α của Hình 3.8", "dò được: α ≈ %.2f" % AL9),
        ("β của Hình 3.14", "dò được: β ≈ 11,1 — đúng con số tr.171"),
        ("tập dữ liệu của Hình 3.16", "phân xử được: 25 điểm của Hình 3.8, khớp hơn 42,8 lần"),
        ("trục dọc của Hình 3.16", "KHÔNG có trên hình; chỉ đọc được ba hoành độ"),
        ("thiết lập của Hình 3.9", "vẽ mẫu từ hậu nghiệm, không có con số nào để đối chiếu"),
        ("thiết lập của Hình 3.12/3.13/3.15", "hình SƠ ĐỒ, không có số"),
):
    print("      %-34s %s" % (_a, _b))
print("\n   CHÚ Ý: con số thời gian chạy của chính script này KHÔNG được in ra bài —")
print("   nó là con số đồng hồ, chỉ tái lập trong đúng một lần chạy (luật Đợt 23).")


# ═══════════════════════════════════════════════════════════════════════════ 25
head("25  Con số DẪN XUẤT mà bài trích — in ra để không ai phải tính nhẩm từ bảng")

print("   (bài học Đợt 24: mọi con số TÓM TẮT một bảng cũng phải do script in, không tính nhẩm)")
print("   hệ số lớn nhất cột M = 9 Bảng 1.1 tr.8            : %.2f" % W9.min())
print("   |w| lớn nhất của cột đó                           : %.2f" % np.abs(W9).max())
print("   độ trải của |m_N − a| trên 200 hạt giống (mục 8)  : %.4f lần"
      % (max(_de) / min(_de)))
print("   σ tại x=0,5 chia σ nhỏ nhất, ô N=1 (mục 9)        : %.4f lần" % (0.937488 / 0.423078))
_ph1 = thiet_ke_gauss(X25[CHI_SO["N=1"]], MU9, S9)
_, _sg1, _, _ = du_bao(_ph1, T25[CHI_SO["N=1"]],
                       thiet_ke_gauss(np.linspace(0, 1, 201), MU9, S9), AL9, BE0)
print("     (tính lại từ chính mảng: %.6f / %.6f = %.4f lần)"
      % (_sg1[100], _sg1.min(), _sg1[100] / _sg1.min()))
print("   dải giá trị trục dọc Hình 3.14 (max − min đọc được): %.6f đơn vị"
      % (E14[:, 1].max() - E14[:, 1].min()))
print("   Hình 3.14: bậc 1 trừ bậc 0 = %+.6f ; bậc 2 trừ bậc 1 = %+.6f"
      % (E14[1, 1] - E14[0, 1], E14[2, 1] - E14[1, 1]))
print("   N − γ của mô hình 25 điểm (mục 17)                : %.6f  (N − M = %d)"
      % (len(X25) - _g_h, len(X25) - M25))
print("   M chia γ của đa thức bậc 9 (mục 21a)              : %.4f lần" % (10.0 / _g_p))
_r98 = 5.483465 / 3.328116
print("   (3.98) chia (3.92) ở N = 25                       : %.4f lần" % _r98)
print("   (3.99) chia (3.95) ở N = 25                       : %.4f lần" % (25.260883 / 19.128181))
print("   nới bề rộng tiên nghiệm gấp đôi chia evidence cho 2^M, M = %d : %d lần"
      % (M25, 2 ** M25))
print("   RMSE kiểm tra: α evidence so với α tốt nhất (mục 18): %.6f %% tệ hơn"
      % (100 * (0.319721 - 0.316247) / 0.316247))
print("   alpha_ chia lambda_ của BayesianRidge (mục 22)     : %.4f lần"
      % (13.26836557 / 3.51966163))
print("   |w| lớn nhất dựng lại chia |w| đọc từ Hình 3.17    : %.4f lần"
      % (3.279845 / 2.317659))
print("   |Σk − 1| tại α = %.6f so với tại α = 1e-12        : %.4e lần"
      % (AL9, [b[3] for b in _bang64 if b[0] == AL9][0] / _bang64[0][3]))
print("   α, β mà phép quét mặt phẳng dò ra (mục 15), dạng khoa học:")
print("      đúng (3.86)          : α = %.3e · β = %.4f" % (_kq[False][1], _kq[False][2]))
print("      bản thiếu (2π)^{M/2} : α = %.3e · β = %.4f" % (_kq[True][1], _kq[True][2]))
print("   sin(2π·1,3) — giá trị thật ở chỗ ngoại suy của mục 21d: %.4f" % np.sin(2 * np.pi * 1.3))
# Ba khoi ma sinh hinh in trong bai tinh lai tu mang DA LAM TRON, nen chung ra
# chu so cuoi khac mot chut. In ca hai de bai trich duoc con so cua TUNG ben.
_le_b = EV86 - E14[:, 1]
_Ad_b = np.c_[np.ones(10), np.array(_mi) + 1]
_ab_b, _, _, _ = np.linalg.lstsq(_Ad_b, np.round(EV86, 6) - np.round(E14[:, 1], 6), rcond=None)
print("   khối mã hình 'dựng lại Hình 3.14' tính trên mảng đã làm tròn 6 chữ số:")
print("      hệ số chặn %+.6f · lệch lớn nhất bản thiếu 2π %.6f"
      % (_ab_b[0], np.abs(np.round(EVB, 6) - np.round(E14[:, 1], 6)).max()))
_q4 = np.round([0.14006367], 8)[0]
_q25 = np.round([0.02375827], 8)[0]
print("   khối mã hình 'hai nguồn bất định', tính trên mảng đã làm tròn 8 chữ số:")
print("      tỉ số %.6f lần · so với nhiễu: %.4f lần và %.4f phần trăm"
      % (_q4 / _q25, _q4 / (1 / BE0), 100 * _q25 / (1 / BE0)))
print("   khối mã hình 'tổng nhân', số bậc độ lớn giữa hai đầu: %.4f"
      % np.log10(DC[-1] / DC[0]))
