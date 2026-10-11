# -*- coding: utf-8 -*-
"""Dò dẫn chiếu "mục N" trỏ tới một mục KHÔNG TỒN TẠI trong chính bài.

VÌ SAO SCRIPT NÀY TỒN TẠI
-------------------------
CLAUDE.md ghi hẳn một mục cho lớp lỗi này sau Đợt 26: *"DẪN CHIẾU MỤC SAI LÀ MỘT
LỚP LỖI RIÊNG, VÀ KHÔNG PHÉP KIỂM NÀO TRONG `tools/` CÓ DẤU HIỆU ĐỂ BẮT."* GĐ 3
có **tám** chỗ, tập trung ở hai bài dài nhất, và nguyên nhân chung là các mục
được chèn thêm qua nhiều đợt làm số mục dịch đi mà không ai dò lại.

Bài học đó kết thúc bằng một mẹo `grep` chạy tay. Đợt 22 đã đo được điều gì xảy
ra với một quy ước không có phép kiểm đứng sau, nên nay nó thành script.

BỐN KHUÔN PHẢI LOẠI — và cả bốn đều tìm ra bằng cách ĐỌC TAY kết quả bản đầu
--------------------------------------------------------------------------
Bản đầu báo **44 chỗ** ở GĐ 4; đọc tay 17 chỗ thì **cả 17 đều là báo nhầm**.
Bốn khuôn, xếp theo số lần gặp:

  1. `[Bài khác mục 6.1](<Bài khác.md>)`   — "mục N" nằm TRONG chữ của liên kết
  2. `mã giả mục 4.1 của bài báo`          — chữ "bài báo" đứng SAU, không phải trước
  3. `tools/v_gd4ss.py` mục 2.1`           — mục của một SCRIPT
  4. `mục 2.3.1 của CLAUDE.md`             — mục của tài liệu chuẩn

Khuôn 1 và 2 là hai khuôn mà một mẫu chỉ nhìn về phía TRƯỚC không thể thấy. Đó
là lý do docstring này liệt kê chúng ra: một phép kiểm chỉ soi chỗ người viết đã
nghĩ tới thì không kiểm gì cả.

HẠN CHẾ ĐÃ ĐO ĐƯỢC — đọc trước khi tin con số "0 chỗ"
-----------------------------------------------------
`NGUON` tìm từ khoá ở **bất kỳ đâu trong 200 ký tự đứng trước**, nên một dẫn
chiếu sai vẫn bị tha nếu đoạn văn ngay trước nó tình cờ nói về một bài báo.
Đo được ngày 2026-08-31: cài `mục 47` vào **cuối** bài Momentum thì script
KHÔNG bắt, vì checklist đóng bài ngay trên có cụm *"Hình từ paper gốc"*; cài
đúng câu đó vào **giữa** thân bài thì bắt ngay.

Đây là đánh đổi có chủ ý — thu hẹp vùng nhìn lại làm trượt khuôn thật *"Bài báo
có nói tới chuyện này ở mục 4.1"* — nhưng nó có nghĩa là **"0 chỗ" đọc là "0 chỗ
ngoài những đoạn đang nói về một nguồn khác"**, không phải "0 chỗ" tuyệt đối.

Phép thử bắt buộc mỗi lần vá, và phải cài lỗi giả vào **giữa thân bài**:

    cài `Chi tiết nằm ở mục 47.` ngay trên `## 1. Tổng quan` của một bài
    -> script phải bắt; gỡ ra sau khi kiểm

Chạy:  cd tools && python danchieu.py            # toàn database
       cd tools && python danchieu.py "GĐ 4"     # một giai đoạn
"""
import glob
import io
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(os.path.dirname(HERE), "AI algorithms", ".flat")

# ── Đã ĐỌC TAY và xác nhận là báo nhầm, kèm chỗ nó thật sự trỏ tới.
#
# Danh sách này đóng vai trò như `MIEN_TRU` của `kiem_gd*.py`: nó làm cho đầu ra
# của script đọc được là *"mọi thứ ngoài năm dòng này đều là lỗi thật"*. Không có
# nó thì năm dòng nhiễu cố định sẽ dạy người đọc bỏ qua cả bảng.
#
# Không nới thêm mẫu loại trừ để tha năm chỗ này. Bài học Đợt 22: câu phải hỏi khi
# nới một mẫu MIỄN TRỪ không phải "nó có bắt được thứ tôi định bắt không" mà "nó
# còn tha những gì ngoài thứ tôi định tha" — và mẫu đủ rộng để tha "mục 10 của bài
# đó" sẽ tha luôn mọi dẫn chiếu nội bộ đứng sau một liên kết.
# Khoá là (tên bài, mục được dẫn) — KHÔNG dùng số dòng. Bản đầu khoá theo số dòng
# và nó hỏng ngay lượt sửa sau: thêm một bảng ký hiệu vào đầu bài là mọi số dòng
# dịch đi, và cả năm miễn trừ mất hiệu lực cùng lúc. Một khoá phải bền trước
# những thay đổi không liên quan tới thứ nó khoá.
MIEN_TRU = {
    ("Gaussian Mixture Model (GMM)", "9.2.1"): "mục của Bishop 2006, chương 9",
    ("Incremental PCA", "3.1"): "mục của Ross et al. 2008",
    ("Mini‑Batch K‑Means", "10"): "mục 10 của bài K‑Means",
    ("Sparse PCA", "6.2"): "mục 6.2 của bài PCA",
    ("t‑SNE", "2.1"): "mục 2.1 của CLAUDE.md",

    # --- Đợt 28, GĐ 5: ba chỗ còn lại sau khi vá hai mẫu, đã đọc tay cả ba.
    ("Adam", "2.1"): "dòng ghi nguồn của một trích dẫn tiếng Anh — mục 2.1 của "
                     "bài báo Adam 2015",
    ("Adam", "2.3.2"): "mục 2.3.2 của CLAUDE.md, gọi bằng cụm đúng tinh thần",
    ("Stochastic Gradient Descent (SGD)", "2.2"):
        "mục 2.1 và 2.2 của bài báo Goyal 2017, dẫn trong khối tham khảo",
    ("Nadam", "3.2"): "mục 3.1 và 3.2 của bài báo Dozat 2016, mô tả bố cục "
                      "bài báo trong khối tài liệu tham khảo",
    ("RMSProp", "6.3"): "mục 5 và 6.3 của bài báo Adam 2015, dẫn trong khối "
                        "tài liệu tham khảo",
}

MUC = re.compile(r"mục\s+(\d+(?:\.\d+)*)")
TIEU_DE = re.compile(r"^#{2,6}\s+(\d+(?:\.\d+)*)\.?\s", re.M)
LIEN_KET = re.compile(r"\[[^\]]*\]\((?:<[^>]*>|[^)]*)\)")

# Chữ đứng NGAY TRƯỚC "mục N" cho biết nó trỏ ra ngoài bài.
TRUOC = re.compile(
    r"(?:\.py`?|CLAUDE\.md`?|bài báo|báo cáo|paper|luận án|sách|chương \d+|"
    r"\.md>?\)?`?|"
    r"[A-ZĐ][\wÀ-ỹ]+(?:\s+(?:et al\.?|&|và)\s+[A-ZĐ][\wÀ-ỹ]+)*\s+\(?\d{4}\)?)"
    r"[\s,)—-]*$", re.I)

# Chữ đứng NGAY SAU cũng cho biết điều đó — khuôn "mục 4.1 CỦA BÀI BÁO",
# và biến thể không có chữ "của": "mục 4.2 bài báo".
# TÁM trong 11 chỗ còn lại sau khi vá NGUON đều theo đúng một khuôn — `mục 7.1
# bài gốc`, `mục 8 bài đó`, `mục 2.3.1 của chuẩn viết` — tức từ khoá đứng SAU,
# và bản trước chỉ nhận ba chữ "bài báo / paper / luận án" ở NGAY sát.
# Cho phép vài từ chen giữa ("và 6.3 của bài báo"), tối đa 30 ký tự: đủ để bắt
# khuôn thật, chưa đủ để vắt sang một câu khác.
# CHỖ MÙ DẤU CHẤM, lần thứ HAI trong cùng một đợt (lần đầu ở `luat1.py`, nơi
# `[^.\n]` chặn mọi liên kết `.md`). Ở đây nạn nhân là *"mục 6.1 và 6.3 của bài
# báo"*: dấu chấm nằm trong **chính số hiệu mục thứ hai**, nên lớp ký tự loại
# dấu chấm không đi qua nổi. Giới hạn 30 ký tự đã đủ chặn việc vắt sang câu
# khác, nên không cần loại dấu chấm thêm một lần nữa.
SAU = re.compile(r"^[^\n]{0,30}?"
                 r"(?:bài báo|báo cáo|bài gốc|bài đó|bài họ|paper|luận án|sách|chuẩn viết|"
                 r"`?CLAUDE\.md|`?tools/|`?[\w]+\.py)", re.I)

# "báo cáo" thêm ngày 2026-09-30 cho bài SARSA: nguồn gốc là một BÁO CÁO KỸ THUẬT (Rummery &
# Niranjan 1994, CUED/F-INFENG/TR 166), và bài gọi nó đúng tên. Đo bán kính hai chiều trên toàn
# database trước khi nhận — xem CLAUDE.md, bài học bài SARSA.
# Dấu blockquote và xuống dòng nằm giữa từ khoá và "mục N" làm hai mẫu trên trượt.
# Bốn trong bảy chỗ báo nhầm của bản trước là đúng lỗi này — từ khoá ở dòng TRÊN.
NHIEU = re.compile(r"[\s>]+")

# Cùng bốn từ khoá, nhưng tìm ở BẤT KỲ ĐÂU trong câu đứng trước chứ không sát bên.
# Đọc tay 25 chỗ ở GĐ 5 (Đợt 28) làm lộ ra BỐN khuôn nữa, và cả bốn đều là báo
# nhầm. Ghi ra đây vì chúng không đoán được từ bốn khuôn cũ:
#
#   5. `mục 7.1 bài gốc`          — "bài gốc"/"bài đó"/"bài họ" đều trỏ BÀI BÁO
#   6. `mục 2.3 của chuẩn viết`   — trỏ CLAUDE.md, gọi bằng tên khác
#   7. `Kingma và Ba ... mục 2.1` — tên tác giả KHÔNG kèm năm; mẫu cũ đòi \d{4}
#   8. `Algorithm 2 ... mục 7.1`  — dẫn theo số hiệu thuật toán của bài báo
#
# Khuôn 7 là khuôn nguy hiểm nhất khi nới mẫu, nên chỉ nới cho **tên kép** (có
# "và" hoặc "&"): một tên đơn không kèm năm sẽ tha quá nhiều.
NGUON = re.compile(r"bài báo|báo cáo|paper|luận án|`?CLAUDE\.md|`?tools/|\w+\.py|\.md>|"
                   r"bài gốc|bài đó|bài họ|chuẩn viết|Algorithm|slide|lecture|"
                   r"[A-ZĐ][\wÀ-ỹ]+\s+(?:và|&)\s+[A-ZĐ][\wÀ-ỹ]+", re.I)


def vung_lien_ket(s):
    """Các khoảng ký tự nằm trong một liên kết markdown — kể cả phần chữ."""
    return [(m.start(), m.end()) for m in LIEN_KET.finditer(s)]


def quet(s):
    """Trả [(dòng, mục được dẫn)] cho những dẫn chiếu KHÔNG khớp tiêu đề nào."""
    co = set()
    for m in TIEU_DE.finditer(s):
        n = m.group(1)
        co.add(n)
        for i in range(1, n.count(".") + 1):        # "3.1" cũng làm "3" tồn tại
            co.add(".".join(n.split(".")[:i]))
    lk = vung_lien_ket(s)
    ra = []
    for m in MUC.finditer(s):
        if any(a <= m.start() < b for a, b in lk):
            continue                                # khuôn 1
        if SAU.match(NHIEU.sub(" ", s[m.end():m.end() + 70])):
            continue                                # khuôn 2
        # Từ khoá "bài báo" / tên script không nhất thiết đứng SÁT "mục N" —
        # "Bài báo có nói tới chuyện này ở mục 4.1" là khuôn thật, gặp ở DBSCAN.
        # Nên tìm nó ở bất kỳ đâu trong CÙNG MỘT CÂU đứng trước.
        cau = NHIEU.sub(" ", s[max(0, m.start() - 200):m.start()]).rsplit(". ", 1)[-1]
        if TRUOC.search(cau) or NGUON.search(cau):
            continue                                # khuôn 3 và 4
        if m.group(1) not in co:
            ra.append((s.count("\n", 0, m.start()) + 1, m.group(1)))
    return ra


def main():
    gd = sys.argv[1] if len(sys.argv) > 1 else None
    print("DẪN CHIẾU TỚI MỘT MỤC KHÔNG TỒN TẠI TRONG CHÍNH BÀI")
    print("-" * 94)
    tong = n_bai = n_mien = 0
    for f in sorted(glob.glob(os.path.join(DB, "*.md"))):
        s = io.open(f, encoding="utf-8").read()
        if gd and ("Giai đoạn: " + gd) not in s:
            continue
        xau = quet(s)
        if not xau:
            continue
        ten = os.path.basename(f)[:-3]
        that = [x for x in xau if (ten, x[1]) not in MIEN_TRU]
        n_mien += len(xau) - len(that)
        if not that:
            continue
        n_bai += 1
        tong += len(that)
        for dong, muc in that:
            quanh = s.split("\n")[dong - 1].strip()[:56]
            print("  %-40s dòng %-5d → mục %-6s %s" % (ten[:40], dong, muc, quanh))
    print("-" * 94)
    print("  %d chỗ ở %d bài · %d chỗ miễn trừ đã đọc tay%s"
          % (tong, n_bai, n_mien, ("  (%s)" % gd) if gd else ""))
    return 1 if tong else 0


if __name__ == "__main__":
    sys.exit(main())
