# -*- coding: utf-8 -*-
"""Dò ký tự điều khiển lọt vào file .md — lệnh LaTeX bị NUỐT MẤT DẤU GẠCH CHÉO.

VÌ SAO SCRIPT NÀY TỒN TẠI (Đợt 27, 2026-08-31)
----------------------------------------------
CLAUDE.md mục 3 đã ghi bẫy heredoc: một dãy thoát trong chuỗi Python viết qua
`python - <<'PY'` có thể bị nuốt thành ký tự điều khiển thật. Bài học đó chép
lại một lần **thất bại ồn ào** — `assert s.count(old) == 1` gãy vì chuỗi tìm
kiếm đã hỏng, nên người viết biết ngay.

Đây là biến thể **IM LẶNG**, và nó nguy hiểm hơn hẳn: khi ký tự bị nuốt nằm ở
chuỗi THAY THẾ chứ không phải chuỗi tìm kiếm, `assert` vẫn xanh và phần hỏng
**được ghi thẳng vào bài**. Không phép kiểm nào trong `tools/` có dấu hiệu để
bắt: `fixfence` đếm dấu ```, `luat1`/`xungho` dò từ tiếng Việt, `kiem_gd*` trích
con số. Một ký tự 0x09 nằm giữa `$-6{,}6` và `imes10^{-3}$` không phải cái nào
trong số đó — bài vẫn render, chỉ là công thức hiện ra sai.

Đo lần đầu, ngày 2026-08-31: **8 bài hỏng thật**, cộng một sơ đồ ASCII dùng TAB
hợp lệ.

Sáu lệnh LaTeX bắt đầu bằng đúng những chữ mà Python coi là dãy thoát:

    tab  ->  \\times \\tau \\to \\text \\tfrac \\theta      (0x09)
    vt   ->  \\varepsilon \\varphi \\vec                   (0x0b)
    bs   ->  \\bar \\beta \\bf \\binom \\bullet             (0x08)
    cr   ->  \\rVert \\rho \\right \\rangle                 (0x0d)
    ff   ->  \\frac \\forall                               (0x0c)
    bel  ->  \\alpha \\approx \\angle                       (0x07)

Chạy:  cd tools && python kytu.py
       cd tools && python kytu.py --sua      # sửa những chỗ đoán chắc chắn
"""
import glob
import io
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(os.path.dirname(HERE), "AI algorithms", ".flat")

# ký tự điều khiển -> (tên, chữ cái đứng sau dấu gạch chéo trong dãy thoát Python)
CTRL = {"\x07": ("BEL", "a"), "\x08": ("BS", "b"), "\x09": ("TAB", "t"),
        "\x0b": ("VT", "v"), "\x0c": ("FF", "f"), "\x0d": ("CR", "r")}

# Chỉ sửa khi phần đuôi khớp một lệnh LaTeX CÓ THẬT. Đoán mò thì báo, không sửa.
LENH = {
    "t": ["imes", "au", "o", "ext", "frac", "heta", "ilde", "op"],
    "v": ["arepsilon", "arphi", "ec", "arnothing", "artheta", "space"],
    "b": ["ar", "eta", "inom", "ullet", "f", "ig", "igl", "igr", "ot", "reve"],
    "r": ["Vert", "ho", "ight", "angle", "brace", "bracket", "vert"],
    "f": ["rac", "orall", "lat", "box"],
    "a": ["lpha", "pprox", "ngle", "rrow", "st", "leph", "rccos", "rcsin", "rctan"],
}


def quet(t):
    """Trả về [(vị trí, ký tự, lệnh đoán được hoặc None)] theo thứ tự xuất hiện."""
    ra = []
    for i, c in enumerate(t):
        if c not in CTRL:
            continue
        chu = CTRL[c][1]
        duoi = t[i + 1:i + 14]
        # dài trước ngắn sau: "\to" không được nuốt mất "\top"
        doan = next((l for l in sorted(LENH[chu], key=len, reverse=True)
                     if duoi.startswith(l)), None)
        ra.append((i, c, ("\\" + chu + doan) if doan else None))
    return ra


def main():
    sua = "--sua" in sys.argv
    n_file = n_chac = n_ngo = 0
    print("KÝ TỰ ĐIỀU KHIỂN LỌT VÀO .md — lệnh LaTeX mất dấu gạch chéo")
    print("-" * 94)
    # CHO MU DA DO DUOC: ban dau script chi quet thu muc bai viet, nen ba file
    # chuan o goc repo — CLAUDE.md, `Ke hoach viet.md`, `Theo doi chat luong.md`
    # — va `Dan y mau` KHONG BAO GIO duoc quet. Do lan dau ngay 2026-09-07:
    # `Ke hoach viet.md` co 6 ky tu dieu khien, `Dan y mau` co 2 — trong do mot
    # cho nam ngay trong DONG VI DU cua chuan viet duy nhat.
    goc = os.path.dirname(HERE)
    ds = sorted(glob.glob(os.path.join(DB, "**", "*.md"), recursive=True))
    ds += sorted(glob.glob(os.path.join(goc, "*.md")))
    ds += sorted(glob.glob(os.path.join(goc, "AI algorithms", "*.md")))
    for f in ds:
        goc = io.open(f, encoding="utf-8", newline="").read()
        # CR đi liền LF là kết thúc dòng của Windows, KHÔNG phải một lệnh LaTeX
        # bắt đầu bằng chữ r bị nuốt. Không lọc bước này thì mọi file CRLF báo
        # hỏng từng dòng một — 28 dòng đầu của `A Search` ở lần chạy thử.
        #
        # Và phải GHI LẠI đúng kiểu xuống dòng cũ. Bản đầu ghi thẳng LF, làm
        # `K‑Means.md` — một trong vài file còn CRLF trong index — đổi cả 838
        # dòng trong `git diff` để sửa đúng 2 ký tự. Một phép sửa mà diff của nó
        # không đọc được thì không ai soát được nó.
        CRLF = chr(13) + chr(10)
        crlf = CRLF in goc
        t = goc.replace(CRLF, chr(10))
        hit = quet(t)
        if not hit:
            continue
        n_file += 1
        goc_repo = os.path.dirname(HERE)
        ten = os.path.relpath(f, DB if f.startswith(DB) else goc_repo)[:-3]
        for i, c, doan in hit:
            dong = t.count("\n", 0, i) + 1
            quanh = re.sub(r"[\x00-\x1f]", "·", t[max(0, i - 26):i + 20])
            n_chac, n_ngo = (n_chac + 1, n_ngo) if doan else (n_chac, n_ngo + 1)
            print("  %-38s dòng %-5d %-3s → %-12s …%s…%s"
                  % (ten[:38], dong, CTRL[c][0], doan or "?", quanh,
                     "" if doan else "   ⚠ KHÔNG đoán được"))
        if sua and any(d for _, _, d in hit):
            for i, c, doan in reversed(hit):
                if doan:
                    t = t[:i] + "\\" + CTRL[c][1] + t[i + 1:]
            io.open(f, "w", encoding="utf-8", newline="").write(
                t.replace(chr(10), CRLF) if crlf else t)
    print("-" * 94)
    print("  %d file · %d chỗ đoán CHẮC CHẮN · %d chỗ phải soát tay%s"
          % (n_file, n_chac, n_ngo, "  (đã sửa)" if sua else ""))
    return 1 if n_ngo or (n_chac and not sua) else 0


if __name__ == "__main__":
    sys.exit(main())
