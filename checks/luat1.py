# -*- coding: utf-8 -*-
"""Dò vi phạm Luật 1 của CLAUDE.md mục 2.0 — "bài đọc được một mình".

Luật 1 cấm câu kiểu *"bài này giả định đã biết X"* và *"đọc bài X trước"*, vì
chúng đẩy phần giải thích sang bài khác. Liên kết chỉ để **đào sâu**.

BA THỨ KHÔNG PHẢI VI PHẠM, và đây là phần khó của phép dò:

1. **Dẫn trong CÙNG bài.** "Đọc mục 2 trước rồi quay lại mục 1" là hướng dẫn thứ
   tự đọc nội bộ, hoàn toàn hợp lệ.
2. **Chính tiêu đề khối KN.** `Dàn ý mẫu` quy định khuôn
   `### 0.x Bảy từ cần biết trước khi đọc tiếp` — nó chứa cụm "cần biết trước"
   nhưng nó chính là thứ Luật 1 ĐÒI phải có. Đợt 20 đã ghi lại chuyện mẫu dò
   ad-hoc bắt nhầm khuôn này (báo cáo Đợt 21 mục 4); nay xử lý hẳn ở đây.
3. **Câu tự khẳng định "Cần biết trước? Không."** — cũng là khuôn của nhà.

Đó chính là bài học CLAUDE.md mục 4 nói: phép kiểm và tài liệu chuẩn phải nói
CÙNG một ngôn ngữ. Trước khi thêm mẫu vào đây, mở `Dàn ý mẫu` ra đọc.

Chạy:  cd tools && python luat1.py           # toàn database
       cd tools && python luat1.py "GĐ 4"    # một giai đoạn
"""
import io
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(os.path.dirname(HERE), "AI algorithms", ".flat")

# Câu đẩy người đọc sang BÀI KHÁC.
#
# CẨN THẬN: dùng W = khoảng trắng KHÔNG gồm xuống dòng, đừng dùng `\s`.
# `\s` nuốt được cả `\n`, nên mẫu khớp VẮT QUA HAI DÒNG và báo nhầm — đã dính
# đúng một lần ngày 2026-08-30 ở bài Affinity Propagation, nơi chữ "đọc" cuối
# một dòng ghép với chữ "trước" ở dòng sau thành một "vi phạm" không có thật.
W = r"[^\S\n]"

# ── CHO MU LON NHAT CUA BAN TRUOC, tim ra ngay 2026-08-31 (Dot 28).
#
# Lop ky tu `[^.\n]` giua "doc" va "truoc" duoc viet de mau khong vat qua HAI
# CAU. Nhung no cung loai luon dau cham ben trong `.md`, ma **moi lien ket
# markdown deu chua dau cham do**. Hau qua: khuon vi pham Luat 1 PHO BIEN NHAT
# — *"Doc [Bai X](<Bai X.md>) truoc"* — chua tung duoc kiem mot lan nao.
#
# Do ban kinh truoc khi va, theo luat Dot 21: **10 cho o 10 bai** toan
# database, doc tay ca 10 thi **ca 10 deu la vi pham that**, khong co bao
# nham. Bay trong so do nam o GD 5; ba cho con lai (Hierarchical Clustering
# Divisive, Incremental PCA, OPTICS) nam o GD 4 — mot giai doan vua duoc
# tuyen bo sach o dot truoc.
#
# Cach va: xoa lien ket khoi ban dung DE DO, giu nguyen do dai de moi vi tri
# khop van anh xa 1-1 sang van ban goc. Khong noi lop ky tu `[^.\n]`, vi noi
# no ra `[^\n]` se cho mau vat qua nhieu cau trong cung mot dong.
LK = re.compile(r"\[[^\]]*\]\((?:<[^>]*>|[^)\s]*)\)")
DAU = "\x01"          # ky tu khong bao gio co trong `.md`, dung lam vet lien ket


def bo_lien_ket(t):
    """Thay moi lien ket bang DAU lap lai — cung do dai, khong con dau cham."""
    return LK.sub(lambda m: DAU * (m.end() - m.start()), t)


NGHI = re.compile(
    r"[Đđ]ọc" + W + r"+[^.\n]{0,70}?\btrước\b"
    # "doc [Bai X] roi quay lai" tuong duong "doc [Bai X] truoc" ve mat Luat 1.
    # Doi phai co vet lien ket \x01 o giua, neu khong se bat nham bon cho dan
    # TRONG CUNG BAI ("doc muc 2 truoc roi quay lai cong thuc o muc 1").
    r"|[Đđ]ọc" + W + r"+[^.\n]{0,60}?" + DAU + r"[^.\n]{0,40}?rồi quay lại"
    # Nhan "Tien quyet:" hop thuc hoa viec day phan giai thich sang bai khac —
    # chinh `Dan y mau` muc 3.3 da xoa dong checklist tuong duong vi ly do do.
    r"|(?:^|\n)Tiên quyết" + W + r"*:"
    r"|giả định" + W + r"+(?:bạn" + W + r"+|người đọc" + W + r"+)?đã" + W +
    r"+(?:biết|đọc|nắm)"
    r"|cần" + W + r"+(?:đọc|nắm|biết)" + W + r"+[^.\n]{0,45}?trước khi"
    r"|nên đọc" + W + r"+[^.\n]{0,45}?trước"
    r"|phải đọc" + W + r"+[^.\n]{0,45}?trước")

# Miễn trừ — xem docstring. Mỗi mẫu kèm lý do để người sau không xoá nhầm.
MIEN_TRU = [
    (r"(?:Sáu|Bảy|Tám|Chín|Mười|\d+)\s+từ\s+cần biết trước khi đọc tiếp",
     "tiêu đề khối KN, khuôn của Dàn ý mẫu"),
    (r"Cần biết trước\?\s*Không",
     "câu tự khẳng định của khối KN"),
    # 2026-08-30 (Đợt 22): bản đầu có "hai|ba|cả" đứng RIÊNG trong danh sách này,
    # nên nó miễn trừ luôn câu *"Đọc hai bài đó trước"* ở bài Nadam — một vi phạm
    # thật, bị chính phép miễn trừ giấu đi. Đó là lỗi cùng lớp với các lần vá
    # detector trước: mẫu miễn trừ rộng hơn thứ nó định miễn. Nay lượng từ chỉ
    # được miễn khi đi kèm một danh từ TRỎ TRONG BÀI.
    (r"[Đđ]ọc\s+(?:hai|ba|cả|)\s*(?:mục|bảng|hình|dòng|khối|đoạn|chương|"
     r"kết quả|danh sách|cột|hàng|ô)",
     "dẫn trong cùng bài"),
    (r"[Đđ]ọc\s+(?:kỹ|lại|nó|chậm|tiếp|theo|thẳng|từ trên|nguyên|đúng|xuôi|ngược)",
     "hướng dẫn cách đọc, không trỏ sang bài khác"),
    (r"cần đọc trước khi (?:so số|dùng|chạy|đối chiếu)",
     "cảnh báo về một mục trong cùng bài"),
    (r"cần đọc trước khi dùng",
     "cảnh báo về một mục trong cùng bài"),
    # Mot BAI DAN NHAP co nhiem vu chi lo trinh; "Vi sao doc truoc" o day la
    # TIEU DE COT cua bang lo trinh, khong phai cau day nguoi doc sang bai khac.
    # Mien tru khoanh dung o cua bang, khong khoanh theo ten bai — mien tru theo
    # ten bai la mot cho tron.
    (r"\|\s*Vì sao đọc trước\s*\|",
     "tiêu đề CỘT của bảng lộ trình trong một bài dẫn nhập"),
]
BO = [(re.compile(p), ly_do) for p, ly_do in MIEN_TRU]


def gd(t):
    m = re.search(r"(?m)^Giai đoạn:\s*(.+)$", t)
    return m.group(1).strip() if m else "?"


def quet(loc=None):
    vi_pham, mien = [], 0
    for f in sorted(os.listdir(DB)):
        if not f.endswith(".md"):
            continue
        s = io.open(os.path.join(DB, f), encoding="utf-8").read()
        if loc and gd(s) != loc:
            continue
        # CHO MU THU BA cua mau nay, do duoc ngay 2026-09-01.
        #
        # Dot 28 xoa lien ket khoi ban dung DE DO, nhung giu nguyen DO DAI de vi
        # tri khop van anh xa 1-1 sang van ban goc. Gioi han 70 ky tu cua mau
        # vi the van dem ca phan lien ket da bi thay. Mot cau co HAI lien ket
        # dai thi khoang cach "doc" -> "truoc" vuot 70 va lot luoi:
        #
        #   "hay doc [CNN nen tang](<...>) va [ResNet](<...>) truoc"
        #
        # — mot vi pham Luat 1 NGUYEN VAN, o bai ShuffleNet.
        #
        # Cach chua: quet theo TUNG DONG, va trong moi dong thay mot lien ket
        # bang DUNG MOT ky tu. Khong con can anh xa vi tri, vi don vi hien thi
        # da la ca dong. Mau hien tai khong mau nao vat qua hai dong (lop
        # `[^.\n]` va `(?:^|\n)` deu chan dieu do), nen phep doi nay khong lam
        # mat cho nao — da do: 28 -> 29 cho, dung mot cho moi va la vi pham that.
        for i, dong_goc in enumerate(s.split("\n"), 1):
            if not NGHI.search(LK.sub(DAU, dong_goc)):
                continue
            dong = dong_goc.strip()
            if any(p.search(dong) for p, _ in BO):
                mien += 1
                continue
            vi_pham.append((f[:-3], i, dong))
    return vi_pham, mien


if __name__ == "__main__":
    loc = sys.argv[1] if len(sys.argv) > 1 else None
    vi_pham, mien = quet(loc)
    print("LUẬT 1 — bài phải đọc được một mình%s" % (" · %s" % loc if loc else ""))
    print("-" * 78)
    for ten, dong, txt in vi_pham:
        print("  %-40s dòng %-5d %s" % (ten[:40], dong, txt[:70]))
    print("-" * 78)
    print("  %d chỗ vi phạm · %d chỗ được miễn trừ (khuôn hợp lệ, xem docstring)"
          % (len(vi_pham), mien))
    if not vi_pham:
        print("  Sạch. Nhưng đây là dấu hiệu BỀ MẶT — CLAUDE.md mục 2.6 vẫn áp dụng:")
        print("  một bài mượn khái niệm của bài khác mà không nói ra thì mẫu nào cũng trượt.")
