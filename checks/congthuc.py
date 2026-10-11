# -*- coding: utf-8 -*-
"""Dò công thức tách dòng `$$…$$` KHÔNG có câu diễn giải ý nghĩa ngay dưới.

VÌ SAO CÔNG CỤ NÀY TỒN TẠI
--------------------------
`Dàn ý mẫu` đã quy định từ lâu: *"Mỗi phương trình quan trọng phải có một câu
diễn giải bằng tiếng Việt ngay bên dưới — không phải đọc lại công thức thành
lời, mà nói **nó có nghĩa gì**"*. Và cũng đã ghi rõ đây là lỗi thường gặp:
*"Công thức thả trần — ký hiệu xuất hiện chưa từng được định nghĩa"*.

Nhưng **không phép kiểm nào đo nó**, nên suốt 29 đợt không ai biết con số thật.
Ngày 2026-09-01 đại ca hỏi thẳng *"kiểm lại xem có chú thích ý nghĩa trừu tượng
của công thức chưa"* — và câu trả lời trung thực lúc đó là: không ai biết.

CÁCH ĐO, và vì sao nó chỉ là DẤU HIỆU BỀ MẶT
--------------------------------------------
Script tìm mọi khối `$$…$$` rồi nhìn **đoạn văn ngay sau nó**. Một công thức
được coi là **có diễn giải** nếu đoạn ngay sau:

* dài hơn `NGAN` ký tự (mặc định 60), và
* KHÔNG mở đầu bằng một công thức khác, một tiêu đề, một bảng hay một khối mã.

Cái nó KHÔNG làm được, phải nói ra:

* Nó **không** phân biệt được câu *"trong đó $\\eta$ là tốc độ học"* — chỉ đọc
  lại ký hiệu — với câu nói **ý nghĩa** của cả phương trình. Đó là thứ chỉ đọc
  mới thấy, đúng tinh thần CLAUDE.md mục 2.6.
* Nó bỏ qua công thức viết inline `$…$`, vì gần như câu nào cũng có.

Nên đọc con số này là *"số công thức chắc chắn THẢ TRẦN"*, một cận dưới, không
phải *"số công thức chưa đạt"*.

Chạy:  cd tools && python congthuc.py                # toàn database
       cd tools && python congthuc.py "GĐ 6"         # một giai đoạn
       cd tools && python congthuc.py --order 1350 1520
       cd tools && python congthuc.py --chi-tiet     # in cả công thức bị bắt
"""
import io
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(os.path.dirname(HERE), "AI algorithms", ".flat")

NGAN = 60          # đoạn ngắn hơn thế này thì chưa gọi là một câu diễn giải

# Công thức tách dòng. Bắt cả dạng `$$…$$` trên một dòng lẫn dạng nhiều dòng.
CT = re.compile(r"\$\$(.+?)\$\$", re.S)


def bo_ma(s):
    """Bỏ khối mã — trong đó `$$` không phải công thức."""
    ra, trong = [], False
    for line in s.split("\n"):
        if line.lstrip().startswith("```"):
            trong = not trong
            ra.append("")
            continue
        ra.append("" if trong else line)
    return "\n".join(ra)


def sau_cong_thuc(s, ket):
    """Đoạn văn ngay sau công thức kết thúc ở vị trí `ket`.

    ĐI QUA BẢNG KÝ HIỆU — vá 2026-09-01. Luật 6 của CLAUDE.md **bắt buộc** mỗi
    công thức đánh số phải có đủ ba phần theo đúng thứ tự: công thức · bảng ký
    hiệu kèm chiều · đoạn nói ý nghĩa trừu tượng. Bản đầu của script nhìn thấy
    một bảng ngay dưới công thức là báo THẢ TRẦN, tức nó phạt đúng khuôn mà
    chuẩn viết đòi — cùng lớp lỗi với chuỗi công thức liên tiếp ở `mot_bai`, và
    cách chữa duy nhất bên phía bài lại là **làm bài xấu đi** (chèn một câu vô
    nghĩa vào giữa công thức và bảng).

    Đo bán kính trước khi vá, theo luật Đợt 21: **14 chỗ lật ở 11 bài**. Đọc tay
    cả 14 thì **12 là lật đúng** (Conjugate Gradient · Decision Tree Regression ·
    Huber · Logistic · LogitBoost · Naive Bayes · QDA · ResNeXt bốn công thức ·
    Voting), và **2 là lật GIẢ** — `Standard Deviation` và `Variance`, nơi đoạn
    sau bảng là một chú thích dạng trích dẫn `> …` chỉ giải thích một ký hiệu,
    không nói ý nghĩa của cả công thức. Nên phép đi-qua-bảng vẫn giữ nguyên mọi
    ràng buộc cũ của `mot_bai`, trong đó có ràng buộc loại `> `; hai bài kia ở
    lại danh sách thả trần, và đó là kết quả đúng.
    """
    con = s[ket:]
    # bỏ các dòng trống ngay sau công thức
    con = con.lstrip("\n")
    khoi = con.split("\n\n")
    # Bảng ký hiệu là phần BẮT BUỘC giữa công thức và câu diễn giải, nên nó
    # không tính là "thứ chắn". Đi qua mọi khối bảng liên tiếp rồi mới xét.
    for k in khoi:
        if k.strip().startswith("|"):
            continue
        return k.strip()
    return ""


def mot_bai(ten, s):
    """(số công thức, danh sách công thức thả trần).

    CHUOI CONG THUC LIEN TIEP — bao nham da do duoc ngay 2026-09-01. Muc 0.3
    cua bai MLP la BA BUOC cua CUNG MOT phep tinh chay tay:

        x = (0.5, 0.8), W1 = …      <- buoc 1
        z1 = 0.1*0.5 + …            <- buoc 2
        a1 = sigma(z1) = …          <- buoc 3
        y_hat = sigma(z2) = 0.7051  <- buoc cuoi, CO dien giai ngay duoi

    Ban dau bat ca ba buoc dau la "tha tran", va cach chua duy nhat la chen mot
    cau vo nghia vao giua cac buoc tinh — tuc lam bai XAU DI de lam vui long
    phep kiem. Nay mot chuoi cong thuc lien tiep chi tinh la mot don vi: neu
    cong thuc CUOI cua chuoi co dien giai thi ca chuoi dat.
    """
    than = bo_ma(s)
    ds = list(CT.finditer(than))
    tran = []
    for k, m in enumerate(ds):
        doan = sau_cong_thuc(than, m.end())
        d = doan.lstrip()
        # Neu ngay sau la mot cong thuc khac thi day la mot BUOC giua chuoi —
        # trach nhiem dien giai thuoc ve cong thuc cuoi chuoi, khong phai no.
        if d.startswith("$$"):
            continue
        if not d:
            tran.append(m.group(1))
            continue
        if d.startswith(("#", "|", "```", "![", "> ")):
            tran.append(m.group(1))
            continue
        if len(d) < NGAN:
            tran.append(m.group(1))
    return len(ds), tran


def main():
    lo = hi = None
    if "--order" in sys.argv:
        i = sys.argv.index("--order")
        lo, hi = int(sys.argv[i + 1]), int(sys.argv[i + 2])
    chi_tiet = "--chi-tiet" in sys.argv
    tham = [a for a in sys.argv[1:] if not a.startswith("--") and not a.isdigit()]
    gd = tham[0] if tham else None

    print("CÔNG THỨC TÁCH DÒNG KHÔNG CÓ CÂU DIỄN GIẢI Ý NGHĨA NGAY DƯỚI")
    print("-" * 96)
    tong = tran_tong = 0
    bai_hong = 0
    for f in sorted(os.listdir(DB)):
        if not f.endswith(".md"):
            continue
        s = io.open(os.path.join(DB, f), encoding="utf-8").read()
        if gd and ("Giai đoạn: " + gd) not in s:
            continue
        if lo is not None:
            m = re.search(r"^Order:\s*(\d+)", s, re.M)
            if not m or not (lo <= int(m.group(1)) <= hi):
                continue
        n, tran = mot_bai(f[:-3], s)
        tong += n
        tran_tong += len(tran)
        if tran:
            bai_hong += 1
            print("  %-46s %2d/%2d công thức thả trần" % (f[:-3][:46], len(tran), n))
            if chi_tiet:
                for c in tran:
                    print("  %-46s   %s" % ("", " ".join(c.split())[:60]))
    print("-" * 96)
    print("  %d công thức tách dòng · %d THẢ TRẦN ở %d bài"
          % (tong, tran_tong, bai_hong))
    print("  (dấu hiệu BỀ MẶT — script không phân biệt được câu nói Ý NGHĨA")
    print("   với câu chỉ đọc lại ký hiệu; xem docstring)")
    return 1 if tran_tong else 0


if __name__ == "__main__":
    sys.exit(main())
