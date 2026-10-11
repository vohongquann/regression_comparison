# -*- coding: utf-8 -*-
"""Dò dẫn chiếu tới một file PDF trong `_papers/` mà file đó KHÔNG tồn tại.

VÌ SAO CẦN CÔNG CỤ NÀY
----------------------
Mục 2.1 của CLAUDE.md đòi mỗi paper được trích dẫn phải có PDF thật trong
`AI algorithms/_papers/`. Bài viết tuân thủ bằng cách in đường dẫn đầy đủ vào
mục tài liệu tham khảo và vào caption hình. Nhưng **không phép kiểm nào đối
chiếu đường dẫn đó với thư mục thật**:

* `checklinks.py` chỉ soát liên kết markdown dạng `[text](<path>)`. Đường dẫn
  PDF nằm trong dấu backtick hoặc trong văn xuôi, không phải liên kết.
* `kiem_gd*.py` trích con số, không quan tâm tên file.
* `danchieu.py` dò "mục N", không dò tên file.

Hệ quả đo được ngày 2026-09-01: ba tên file khác nhau được nhiều bài dẫn suốt
nhiều đợt trong khi thư mục chứa tên khác — "He et al" so với
"He, Zhang, Ren, Sun", và bản SENet thiếu hậu tố "(SENet)". Người đọc bấm theo
đường dẫn đó sẽ không tìm ra file, đúng lớp lỗi mà CLAUDE.md gọi là **dẫn chiếu
chết**.

Đây là lớp lỗi ĐẶC BIỆT im lặng: một tên file sai trông y hệt một tên file
đúng, và bài vẫn "có nguồn" theo mọi phép kiểm khác.

CÁCH DÙNG
---------
    cd tools && python kiempdf.py                # toàn database
    cd tools && python kiempdf.py "GĐ 6"         # một giai đoạn
    cd tools && python kiempdf.py --goiy         # kèm gợi ý tên gần đúng nhất

PHÉP THỬ LỖI GIẢ (bắt buộc mỗi lần sửa script này)
--------------------------------------------------
Đổi một ký tự trong một tên file PDF ở GIỮA thân một bài rồi chạy lại — script
phải bắt. Không có bước đó thì "0 chỗ" không phân biệt được với script hỏng.
"""
import difflib
import io
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
AI = os.path.join(os.path.dirname(HERE), "AI algorithms")
DB = os.path.join(AI, ".flat")
PAPERS = os.path.join(AI, "_papers")

# Bắt mọi cách bài viết dẫn một file trong `_papers/`. Ba khuôn gặp thật:
#   `AI algorithms/_papers/<tên>.pdf`   — đầy đủ, trong backtick
#   `_papers/<tên>.pdf`                 — rút gọn
#   AI algorithms/_papers/<tên>.pdf     — trần, trong caption nhiều dòng
# Tên file chứa dấu cách, dấu phẩy, ngoặc đơn và gạch ngang, nên không thể
# dùng lớp "không phải khoảng trắng". Cắt tại `.pdf` thay vì tại khoảng trắng.
DAN = re.compile(r"_papers/\s*([^`\n]*?\.pdf)")


def bai_trong(gd):
    out = []
    for f in sorted(os.listdir(DB)):
        if not f.endswith(".md"):
            continue
        s = io.open(os.path.join(DB, f), encoding="utf-8").read()
        if gd and ("Giai đoạn: " + gd) not in s:
            continue
        out.append((f, s))
    return out


def main():
    tham = [a for a in sys.argv[1:] if not a.startswith("--")]
    gd = tham[0] if tham else None
    goiy = "--goiy" in sys.argv

    if not os.path.isdir(PAPERS):
        sys.exit("Không thấy %s" % PAPERS)
    co = set(os.listdir(PAPERS))

    print("DẪN CHIẾU TỚI FILE PDF KHÔNG TỒN TẠI TRONG `_papers/`")
    print("-" * 94)

    ds = bai_trong(gd)
    if gd and not ds:
        # Đợt 34: gọi `kiempdf.py --bai BigGAN` thì `BigGAN` rơi vào chỗ của GIAI
        # ĐOẠN, không khớp bài nào, và script in "0 dẫn chiếu" — một dòng xanh vô
        # nghĩa. Luật Đợt 14: cái gì bỏ qua được thì phải BÁO RA.
        sys.exit("Không bài nào có `Giai đoạn: %s`. Tham số vị trí của script này là\n"
                 "GIAI ĐOẠN (ví dụ \"GĐ 6\"), không phải tên bài." % gd)

    hong, tong, bai_hong = 0, 0, set()
    for f, s in ds:
        # Tên file có thể bị ngắt dòng trong caption; nối lại trước khi dò.
        phang = re.sub(r"\n\s*", " ", s)
        for m in DAN.finditer(phang):
            ten = m.group(1).strip()
            tong += 1
            if ten in co:
                continue
            hong += 1
            bai_hong.add(f)
            print("  %-46s %s" % (f[:46], ten[:80]))
            if goiy:
                gan = difflib.get_close_matches(ten, co, n=1, cutoff=0.55)
                print("  %-46s   -> gợi ý: %s" % ("", gan[0] if gan else "(không có)"))

    print("-" * 94)
    print("  %d dẫn chiếu PDF được kiểm · %d chỗ HỎNG ở %d bài%s"
          % (tong, hong, len(bai_hong), ("  (%s)" % gd) if gd else ""))
    # Chiều ngược lại cũng đáng biết: PDF nằm trong kho mà không bài nào dẫn.
    return 1 if hong else 0


if __name__ == "__main__":
    sys.exit(main())
