# -*- coding: utf-8 -*-
"""Đối chiếu bài viết với BÀI BÁO GỐC: hình/bảng nào của paper chưa vào bài.

VÌ SAO CÔNG CỤ NÀY TỒN TẠI
--------------------------
Ngày 2026-09-01 đại ca bác toàn bộ kết quả đọc soát 18 bài đầu GĐ 6, với lý do
đúng và không cãi được: bài **đúng số, đủ bảy khối, có bảng ký hiệu** — mà phần
**cấu trúc mạng** vẫn chỉ là bản tóm tắt. Người đọc không dựng lại được mô hình
chỉ bằng bài, và nhiều hình của bài báo gốc chưa từng được đưa vào.

Không phép kiểm nào trong `tools/` bắt được chuyện đó:

* `khoi7.py` bật ✓ cho khối CT khi thấy **một** sơ đồ hoặc **một** bảng tầng. Nó
  không biết bài báo có mười hình và bài mới dùng một.
* `kiem_gd*.py` hỏi *"con số này có nguồn không"*, không hỏi *"bài báo còn con số
  nào bài chưa nhắc"*.
* `kiempdf.py` hỏi *"file PDF này có tồn tại không"*, không hỏi *"đã trích hết
  chưa"*.

Đây là **chỗ mù có hệ thống**: mọi phép kiểm hiện có đều đo thứ ĐÃ CÓ trong bài,
không đo thứ CÒN THIẾU so với nguồn. Script này đo đúng phần còn thiếu.

CÁCH ĐO
-------
1. Quét lớp text của PDF, gom mọi số hiệu `Figure N` và `Table N` mà **caption**
   của chính paper khai báo.
2. Đếm hình đã cắt vào thư mục bài (`paper_*.png`).
3. Đọc thân bài, gom mọi số hiệu Figure/Table mà bài có nhắc tới.
4. In ra phần chênh — hình/bảng paper có mà bài không nhắc.

HẠN CHẾ, ghi rõ để con số không bị đọc thành lời hứa
---------------------------------------------------
* PDF **quét** hoặc có lớp text vỡ trả về 0 caption. Script BÁO RA điều đó thay
  vì im lặng coi là "đủ" — xem cột `?`.
* Bài nhắc "Figure 2" trong văn xuôi mà không cắt hình vẫn tính là **có nhắc**.
  Cột `đã cắt` là thứ nói bài có hình thật hay không.
* Một số paper đánh số hình theo mục (Fig. 3.1); mẫu hiện chỉ bắt số nguyên.

Chạy:  cd tools && python sopaper.py "GĐ 6"
       cd tools && python sopaper.py --bai "ResNet - Residual Networks"
"""
import glob
import io
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
AI = os.path.join(os.path.dirname(HERE), "AI algorithms")
DB = os.path.join(AI, ".flat")

# Số hiệu hình/bảng mà bài báo khai là CÓ: "Figure 3:", "Fig. 3.", "Table 1:" — và
# cả "Fig. 1 We detect…", "Table 1 Ablation on…" KHÔNG có dấu nào sau số, khuôn mà
# Springer/ECCV dùng. Trần cũ đòi dấu câu ngay sau số nên bài báo CornerNet báo 4
# hình 0 bảng thay vì 11 hình 7 bảng — đo trên toàn `_papers/`: **+140 mục ở 38
# file, MẤT 0**, đọc tay toàn bộ phần thêm.
#
# Hệ quả phải ghi ra: nhánh mới nuốt luôn câu DẪN CHIẾU trong thân bài
# ("Table 11 reports…"). Với việc sopaper làm — đếm bài báo CÓ những hình/bảng nào
# — đó là đúng chứ không phải báo nhầm, vì một số hiệu được thân bài dẫn tới thì
# nó tồn tại. Nhưng cột này KHÔNG còn đọc được là "số caption".
# Chỗ mù thứ HAI của lớp phân cách, đo trên bài báo FCOS: caption ở đó viết
# "Figure 1 – As shown..." với dấu gạch NGANG NGẮN (en dash, U+2013), trong khi
# lớp cũ chỉ có gạch DÀI (em dash, U+2014) và gạch nối. Hệ quả: bài báo đó báo
# 4 hình 5 bảng thay vì 8 hình 8 bảng — tức hơn NỬA phần ruột vô hình với
# phép đối chiếu của Luật 6. Thêm cả gạch ngang dài U+2015 cho chắc.
CAP = re.compile(r"\b(Fig(?:ure)?\.?|Table|TABLE)\s*(\d{1,2})\s*"
                 r"(?:[:.\u2013\u2014\u2015-]|(?=[A-Z“‘\"']))", re.I)
# Bài viết dẫn tới hình/bảng của paper bằng tiếng Việt lẫn tiếng Anh.
DAN = re.compile(r"\b(?:Figure|Fig\.?|Hình|Table|Bảng)\s*(\d{1,2})\b", re.I)


def caption_pdf(p):
    """{('fig'|'tab', số)} mà lớp text của PDF khai báo là caption."""
    import pypdfium2 as pdfium      # cung thu vien ma pdfscan.py dung
    ra = set()
    try:
        d = pdfium.PdfDocument(p)
    except Exception as e:
        print("  ⚠ không mở được %s: %s" % (os.path.basename(p)[:50], e))
        return ra
    for i in range(len(d)):
        try:
            t = d[i].get_textpage().get_text_range()
        except Exception:
            continue
        for kind, so in CAP.findall(t):
            ra.add(("tab" if kind.lower().startswith("tab") else "fig", int(so)))
    return ra


def pdf_cua_bai(s):
    """Mọi đường dẫn `_papers/....pdf` mà bài dẫn tới."""
    phang = re.sub(r"\n\s*", " ", s)
    ra = []
    for m in re.finditer(r"_papers/\s*([^`\n]*?\.pdf)", phang):
        ten = m.group(1).strip()
        p = os.path.join(AI, "_papers", ten)
        if os.path.exists(p) and p not in ra:
            ra.append(p)
    return ra


def than_bai(s):
    """Bỏ RIÊNG mục tài liệu tham khảo — ở đó số hiệu hình chỉ là danh mục.

    CHO MU DA DO DUOC (2026-09-01): ban dau cat TAT CA tu "Tai lieu tham khao"
    tro di. Nhung Luat 6 doi moi bai kien truc co mot muc "Doi chieu voi bai bao
    goc" o cuoi, va muc do thuong nam SAU muc tham khao — nen no bi cat mat, va
    script bao "8 hinh chua nhac" trong khi ca 8 da duoc liet ke day du.

    Nay chi cat tu tieu de tham khao toi tieu de `## ` KE TIEP.
    """
    i = s.find("Tài liệu tham khảo")
    if i < 0:
        return s
    j = s.find("\n## ", i)
    return s[:i] + (s[j:] if j > 0 else "")


def mot_bai(ten, s):
    pdfs = pdf_cua_bai(s)
    if not pdfs:
        print("  %-44s ⚠ KHÔNG dẫn PDF nào trong `_papers/`" % ten[:44])
        return None
    # chỉ đối chiếu với paper GỐC — file đầu tiên bài dẫn tới
    goc = pdfs[0]
    co = caption_pdf(goc)
    thu_muc = os.path.join(DB, ten)
    da_cat = len(glob.glob(os.path.join(thu_muc, "paper_*.png")))
    dan = {int(x) for x in DAN.findall(than_bai(s))}

    fig = sorted(n for k, n in co if k == "fig")
    tab = sorted(n for k, n in co if k == "tab")
    thieu_f = [n for n in fig if n not in dan]
    thieu_t = [n for n in tab if n not in dan]
    print("  %-44s paper: %2d hình %2d bảng · đã cắt %d · bài nhắc %2d"
          % (ten[:44], len(fig), len(tab), da_cat, len(dan)))
    if not co:
        print("  %-44s   ⚠ lớp text PDF rỗng hoặc vỡ — phải soát tay bằng pdfsheet.py"
              % "")
    if thieu_f:
        print("  %-44s   → HÌNH bài KHÔNG nhắc: %s"
              % ("", ", ".join("Fig %d" % n for n in thieu_f)))
    if thieu_t:
        print("  %-44s   → BẢNG bài KHÔNG nhắc: %s"
              % ("", ", ".join("Table %d" % n for n in thieu_t)))
    return len(fig), len(tab), da_cat, len(thieu_f), len(thieu_t), bool(co)


def main():
    lo, hi = None, None
    if "--order" in sys.argv:
        i = sys.argv.index("--order")
        lo, hi = int(sys.argv[i + 1]), int(sys.argv[i + 2])
    if "--bai" in sys.argv:
        loc_bai = sys.argv[sys.argv.index("--bai") + 1]
        gd = None
    else:
        loc_bai = None
        tham = [a for a in sys.argv[1:] if not a.startswith("--")
                and not a.isdigit()]
        gd = tham[0] if tham else None

    print("ĐỐI CHIẾU BÀI VIẾT VỚI BÀI BÁO GỐC — hình/bảng nào chưa vào bài")
    print("-" * 100)
    tf = tt = tc = xf = xt = 0
    n_bai = n_vo = 0
    for f in sorted(os.listdir(DB)):
        if not f.endswith(".md"):
            continue
        s = io.open(os.path.join(DB, f), encoding="utf-8").read()
        ten = f[:-3]
        if gd and ("Giai đoạn: " + gd) not in s:
            continue
        if loc_bai and loc_bai not in ten:
            continue
        if lo is not None:
            m = re.search(r"^Order:\s*(\d+)", s, re.M)
            if not m or not (lo <= int(m.group(1)) <= hi):
                continue
        r = mot_bai(ten, s)
        if not r:
            continue
        a, b, c, d, e, doc_duoc = r
        tf += a
        tt += b
        tc += c
        xf += d
        xt += e
        n_bai += 1
        if not doc_duoc:
            n_vo += 1
    print("-" * 100)
    print("  %d bài · paper có %d hình + %d bảng · bài đã cắt %d hình gốc"
          % (n_bai, tf, tt, tc))
    print("  CHƯA NHẮC TỚI: %d hình · %d bảng%s"
          % (xf, xt, ("  ·  %d PDF lớp text vỡ, phải soát tay" % n_vo) if n_vo else ""))
    return 1 if (xf or xt) else 0


if __name__ == "__main__":
    sys.exit(main())
