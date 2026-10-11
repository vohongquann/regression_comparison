# -*- coding: utf-8 -*-
"""Dò bảng markdown bị LỆCH SỐ CỘT — bảng render hỏng mà đọc file thô không thấy.

Nguyên nhân gần như luôn là **dấu `|` của LaTeX nằm trong ô bảng**: `$|D|$`,
`$\\frac{1}{|A||B|}$`, `$\\|x\\|$`. Markdown cắt ô theo `|`, nên một công thức
chuẩn xác về toán học vẫn làm vỡ bảng. Bản render mất cột, và bản `.md` thô thì
trông hoàn toàn bình thường — đúng loại lỗi mà `CLAUDE.md` mục 2.6 nói chỉ đọc
mới thấy, trừ khi có phép kiểm.

Cách chữa: thay `|` bằng `\\lvert … \\rvert` (hoặc `\\vert`) bên trong công thức.

Không phải công cụ này bắt được mọi bảng hỏng — bảng có số cột **đều nhau nhưng
sai nội dung** thì nó chịu. Nó chỉ bắt lỗi cấu trúc, giống `fixfence.py`.

Chạy:  cd tools && python bangcot.py            # toàn database
       cd tools && python bangcot.py "GĐ 4"     # một giai đoạn
"""
import io
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(os.path.dirname(HERE), "AI algorithms", ".flat")

# `|` chưa bị escape mới là dấu cắt ô
SPLIT = re.compile(r"(?<!\\)\|")


def gd(t):
    m = re.search(r"(?m)^Giai đoạn:\s*(.+)$", t)
    return m.group(1).strip() if m else "?"


def bo_fence(s):
    """Thay mọi dòng nằm trong code fence bằng dòng rỗng."""
    out, trong = [], False
    for line in s.split("\n"):
        if line.lstrip().startswith("```"):
            trong = not trong
            out.append("")
            continue
        out.append("" if trong else line)
    return out


def quet_bai(s):
    """Trả list (dòng bắt đầu bảng, list số cột) cho mọi bảng lệch."""
    loi, blk = [], []

    def xong(blk):
        if len(blk) < 2:
            return
        w = [len(SPLIT.split(l)) - 2 for _, l in blk]
        if len(set(w)) > 1:
            loi.append((blk[0][0], w))

    for i, line in enumerate(bo_fence(s), 1):
        t = line.strip()
        if t.startswith("|") and t.endswith("|"):
            blk.append((i, t))
        else:
            xong(blk)
            blk = []
    xong(blk)
    return loi


def main():
    loc = sys.argv[1] if len(sys.argv) > 1 else None
    tong = hong = 0
    bai_hong = []
    for f in sorted(os.listdir(DB)):
        if not f.endswith(".md"):
            continue
        s = io.open(os.path.join(DB, f), encoding="utf-8").read()
        if loc and gd(s) != loc:
            continue
        # đếm tổng số bảng để con số "0 hỏng" có nghĩa
        blk, n = [], 0
        for line in bo_fence(s):
            t = line.strip()
            if t.startswith("|") and t.endswith("|"):
                blk.append(t)
            else:
                if len(blk) >= 2:
                    n += 1
                blk = []
        if len(blk) >= 2:
            n += 1
        tong += n
        loi = quet_bai(s)
        if loi:
            bai_hong.append(f[:-3])
            for dong, w in loi:
                print("  %-42s dòng %-5d số cột %s" % (f[:-3][:42], dong, w))
                hong += 1
    print("-" * 78)
    print("  %d bảng lệch cột / %d bảng%s · %d bài dính"
          % (hong, tong, (" (%s)" % loc) if loc else "", len(bai_hong)))
    # Cái gì bỏ qua được thì phải BÁO RA (CLAUDE.md mục 4). Tham số lọc ở đây là
    # tên GIAI ĐOẠN; đưa vào tên một BÀI thì không khớp gì cả và script in ra
    # "0 bảng lệch cột / 0 bảng" — một dòng xanh hoàn toàn vô nghĩa. Đo được
    # ngày 2026-09-03 khi gọi `bangcot.py "Variational Autoencoder (VAE)"`.
    if loc and tong == 0:
        print("  ⚠ KHÔNG bài nào có `Giai đoạn: %s` — con số 0 ở trên VÔ NGHĨA." % loc)
        print("    Tham số của script này là tên GIAI ĐOẠN (`\"GĐ 4\"`), không phải tên bài.")
        return 2
    if hong:
        print("  Sửa: thay `|` trong công thức bằng \\lvert … \\rvert")
    return 1 if hong else 0


if __name__ == "__main__":
    sys.exit(main())
