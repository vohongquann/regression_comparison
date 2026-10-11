# -*- coding: utf-8 -*-
"""Quét 7 khối bắt buộc của một bài viết (chuẩn 2026-08-29).

CẢNH BÁO ĐỌC TRƯỚC KHI DÙNG SỐ CỦA SCRIPT NÀY
----------------------------------------------
Đây là phép quét **dấu hiệu bề mặt**, đúng nghĩa CLAUDE.md mục 2.6. Một dấu ✓ ở đây
CHỈ có nghĩa "bài có tiêu đề / có bảng / có trích số hiệu hình" — nó KHÔNG chứng minh
nội dung đúng hay đủ. Bài `ShuffleNet` từng ✓ mọi tín hiệu tự động trong khi mô tả sai
cơ chế ở 7 chỗ.

Dùng script này để trả lời "còn bao nhiêu việc", KHÔNG dùng để kết luận "bài đạt".
Cột `Kết luận` trong `Theo dõi chất lượng.md` vẫn chỉ điền sau khi ĐỌC HẾT bài.

Bảy khối, theo thứ tự câu hỏi người mới đặt ra:

  1 IO   Nhận gì – trả ra gì
  2 KN   Khái niệm mới được định nghĩa tại chỗ
  3 SS   Cải thiện gì so với cách cũ, và cách cũ tắc ở đâu
  4 CT   Cấu trúc: gồm những gì, ghép ra sao
  5 VS   Vì sao đổi cái đó lại ăn thua (cơ chế / ablation)
  6 KQ   Kết quả định lượng + định tính TỪ paper gốc, có số hiệu Bảng/Hình
  7 GIA  Giá phải trả, và khi nào KHÔNG dùng

Chạy:  cd tools && python khoi7.py            # tóm tắt theo giai đoạn
       cd tools && python khoi7.py --thieu    # liệt kê bài thiếu nhiều nhất
"""
import io
import os
import re
import sys
import collections

sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DB = os.path.join(ROOT, "AI algorithms", ".flat")

TEN = ["IO", "KN", "SS", "CT", "VS", "KQ", "GIA"]
MO_TA = {
    "IO": "nhận gì – trả gì",
    "KN": "khái niệm mới định nghĩa tại chỗ",
    "SS": "so với cách cũ + bế tắc cũ",
    "CT": "cấu trúc / sơ đồ / bảng tầng",
    "VS": "vì sao cải thiện được",
    "KQ": "kết quả từ paper, có số hiệu Bảng/Hình",
    "GIA": "giá phải trả · khi không dùng",
}

H = r"(?m)^#{2,4}\s.*"          # mọi dòng tiêu đề


def _tieu_de(t, mau):
    return any(re.search(mau, d, re.I) for d in re.findall(H, t))


def quet(t):
    """Trả về dict {tên khối: bool} cho nội dung một bài."""
    r = {}

    # 1 IO — có mục 0, hoặc nêu rõ cả đầu vào lẫn đầu ra
    #
    # 2026-08-30 (Đợt 23): lần thứ NĂM cùng một lớp lỗi — mẫu hẹp hơn cách tài liệu
    # thật sự viết. Toàn khối CNN của GĐ 6 dùng khuôn nhà `## 1. Đọc nhanh: …` và nêu
    # IO bằng chữ "nhận"/"trả" thay vì đúng hai từ "đầu vào"/"đầu ra": VGG viết
    # *"nhận $(N,3,224,224)$ và trả $(N,1000)$ điểm lớp"*. Mẫu cũ trượt cả tám bài.
    #
    # Đã đo bán kính TRƯỚC khi vá, đúng luật Đợt 21: mẫu mới lật đúng 8 bài, tất cả
    # trong GĐ 6. Đọc tay cả tám thì BẢY có IO thật (AlexNet · EfficientNet ·
    # GoogLeNet · RegNet · ResNet · SENet · VGG) và MỘT là lật giả — DenseNet, có
    # tiêu đề "Đọc nhanh: DenseNet nối cái gì?" nhưng §1 chỉ tả cơ chế nối, không nói
    # mạng nhận gì trả gì. Cách xử lý: KHÔNG nới mẫu để tha DenseNet, mà sửa bài
    # DenseNet cho nó nêu IO thật. Đúng nguyên tắc "sửa bài, đừng sửa thước".
    r["IO"] = bool(re.search(r"(?m)^##\s*0\.", t)) or bool(
        re.search(r"[Đđ]ầu vào|[Nn]hận", t) and re.search(r"[Đđ]ầu ra|trả", t)
        and _tieu_de(t, r"nhận gì|đầu vào|input|đọc nhanh|trả gì|nhận vào")
    )

    # 2 KN — có bảng thuật ngữ / ký hiệu, hoặc mục khái niệm
    #
    # 2026-08-29: bản đầu BỎ SÓT chính khuôn mà `Dàn ý mẫu` quy định —
    # "### 0.x Từ mới trong bài này" với bảng `| Từ | Nghĩa … |`. Nó cũng bỏ sót
    # kiểu viết của nhà: "Bảy từ cần biết trước khi đọc tiếp". Hai dạng đó là
    # bảng thuật ngữ thật, nên con số KN 8% đo ngày 2026-08-29 là ĐẾM THIẾU.
    # Xem CLAUDE.md mục 4: phép kiểm bỏ qua đầu vào lạ phải báo ra cái nó bỏ qua.
    r["KN"] = _tieu_de(t, r"thuật ngữ|khái niệm|ký hiệu|từ vựng|"
                          r"từ mới|từ cần biết|từ cần phân biệt|từ vựng") or bool(
        re.search(r"(?m)^\|\s*(Ký hiệu|Thuật ngữ|Khái niệm|Từ)\s*\|", t))

    # 3 SS — có mục so sánh với cái có trước
    #
    # 2026-08-29 (Đợt 19): mẫu cũ THIẾU chính cách nói mà CLAUDE.md mục 2.0 dùng để
    # định nghĩa khối này — *"Trước nó người ta làm sao, tắc ở đâu?"*. Tiêu đề viết
    # đúng theo chuẩn ("Trước Fisher, người ta làm gì — và nó tắc ở đâu", "Cách cũ
    # là LDA — nó tắc ở đâu") đều TRƯỢT, vì mẫu chỉ bắt cụm ghép "bế tắc".
    # Đây là lần thứ BA trong hai đợt cùng một lớp lỗi (KN ở Đợt 18, GIÁ ở trên).
    # 2026-08-30 (Đợt 21): lần thứ TƯ cùng một lớp lỗi. Khuôn viết SS thật sự của
    # database này là "Chỗ X thắng Y" / "Chỗ X bó tay" / "Vì sao X thất bại" — đọc
    # tay 7 mục dạng đó (Kernel PCA, GMM, HDBSCAN, ICA, t-SNE, MDS, Factor Analysis)
    # thì CẢ BẢY đều nêu bế tắc của cách cũ BẰNG SỐ, đúng yêu cầu mục 2.0. Mẫu cũ
    # trượt hết. Không mở rộng sang "Ý tưởng"/"Đánh đổi" — hai thứ đó là mô tả và
    # khối GIÁ, mở ra là đo một thứ khác chứ không phải đếm đủ hơn.
    r["SS"] = _tieu_de(t, r"so với|so sánh|cải tiến|khác gì|điểm khác|thay đổi gì|"
                          r"được gì.*mất gì|tiền thân|bế tắc|"
                          r"cách cũ|trước nó|tắc ở đâu|trước khi có|"
                          r"trước \w+,? người ta|"
                          r"chỗ .{0,45}(thắng|thua|bó tay|sập|thất bại|khác)|"
                          r"vì sao .{0,35}thất bại")

    # 4 CT — có sơ đồ khối, hoặc bảng tầng/tensor, hoặc mục kiến trúc
    co_so_do = bool(re.search(r"[│├└┌┐┘─►▼▲]", t))
    co_bang_tang = bool(re.search(r"(?m)^\|[^\n]*(shape|kích thước|tensor|tham số|"
                                  r"đầu ra|output)[^\n]*\|", t, re.I))
    r["CT"] = co_so_do or co_bang_tang or _tieu_de(t, r"kiến trúc|cấu trúc|sơ đồ|luồng")

    # 5 VS — giải thích cơ chế, không chỉ mô tả
    r["VS"] = _tieu_de(t, r"vì sao|tại sao|cơ chế|nguyên lý|ablation|hoạt động") or bool(
        re.search(r"ablation|bỏ .{0,20} đi thì", t, re.I))

    # 6 KQ — trích số hiệu Bảng/Hình cụ thể của paper (không phải "bảng dưới đây")
    r["KQ"] = bool(re.search(r"(Bảng|Table|Hình|Figure)\s*\d", t))

    # 7 GIA — nhược điểm / đánh đổi / khi nào không dùng
    #
    # 2026-08-29 (Đợt 19): bản đầu KHÔNG có cờ re.I, nên một bài chỉ có tiêu đề
    # "## 4. Giới hạn" (chữ G hoa) bị đếm là THIẾU khối GIÁ. Đo được **18 bài**
    # dính lỗi này, trong đó có `Passive-Aggressive Classifier` — bài có hẳn một
    # mục Giới hạn 4 điểm. Con số GIÁ 62% đo ngày 2026-08-29 là ĐẾM THIẾU.
    #
    # Đây là lần thứ HAI cùng một lớp lỗi: Đợt 18 đã vá khối KN vì detector
    # không nói cùng ngôn ngữ với `Dàn ý mẫu`. Bài học ở CLAUDE.md mục 4 —
    # phép kiểm bỏ qua đầu vào lạ thì phải BÁO RA cái nó bỏ qua — nay áp thêm
    # một vế: **phép kiểm phải khớp hoa/thường với cách tài liệu thật sự viết.**
    r["GIA"] = bool(re.search(r"nhược điểm|trả giá|đánh đổi|hạn chế|giới hạn|"
                              r"khi không|khi nào không|không nên dùng|thất bại",
                              t, re.I))
    return r


def hinh_paper(ten_bai):
    """Bài có hình cắt từ paper gốc chưa (file paper_*.png trong thư mục cùng tên)."""
    d = os.path.join(DB, ten_bai)
    return os.path.isdir(d) and any(f.startswith("paper_") for f in os.listdir(d))


def doc_het():
    """Trả về list (tên, giai đoạn, tier, nhãn, số dòng, dict 7 khối, có hình paper)."""
    out = []
    for f in sorted(os.listdir(DB)):
        if not f.endswith(".md"):
            continue
        ten = f[:-3]
        t = io.open(os.path.join(DB, f), encoding="utf-8", errors="replace").read()
        g = lambda k: (re.search(k + r":\s*(.+)", t).group(1).strip()
                       if re.search(k + r":\s*(.+)", t) else "")
        out.append((ten, g("Giai đoạn"), g("Tier"), g("Trạng thái"),
                    t.count("\n") + 1, quet(t), hinh_paper(ten)))
    return out


if __name__ == "__main__":
    bai = doc_het()
    n = len(bai)

    print("QUÉT 7 KHỐI BẮT BUỘC — %d bài\n" % n)
    print("Đây là dấu hiệu BỀ MẶT. ✓ không có nghĩa bài đạt (CLAUDE.md 2.6).\n")

    print("%-4s %-38s %6s %6s" % ("Khối", "Nội dung", "Có", "%"))
    print("-" * 58)
    for k in TEN:
        c = sum(1 for b in bai if b[5][k])
        print("%-4s %-38s %4d/%d %5.0f%%" % (k, MO_TA[k], c, n, 100.0 * c / n))
    c = sum(1 for b in bai if b[6])
    print("%-4s %-38s %4d/%d %5.0f%%" % ("PPR", "có hình gốc cắt từ paper", c, n,
                                         100.0 * c / n))

    print("\nSỐ KHỐI ĐẠT / BÀI")
    print("-" * 58)
    pp = collections.Counter(sum(b[5].values()) for b in bai)
    for k in range(8):
        print("  %d/7 khối : %3d bài  %s" % (k, pp.get(k, 0), "█" * (pp.get(k, 0) // 3)))

    du7 = [b for b in bai if sum(b[5].values()) == 7]
    print("\nĐủ 7/7 dấu hiệu: %d bài (%.0f%%)" % (len(du7), 100.0 * len(du7) / n))

    print("\nTHEO GIAI ĐOẠN")
    print("-" * 58)
    print("%-16s %5s %8s %10s" % ("Giai đoạn", "bài", "TB khối", "đủ 7/7"))
    gg = collections.defaultdict(list)
    for b in bai:
        gg[b[1]].append(b)
    for ge in sorted(gg):
        v = gg[ge]
        tb = sum(sum(x[5].values()) for x in v) / float(len(v))
        d7 = sum(1 for x in v if sum(x[5].values()) == 7)
        print("%-16s %5d %8.1f %10s" % (ge or "(trống)", len(v), tb,
                                        "%d (%.0f%%)" % (d7, 100.0 * d7 / len(v))))

    # bài mang nhãn Xong nhưng thiếu từ 3 khối trở lên
    xau = [b for b in bai if b[3] == "Xong" and sum(b[5].values()) <= 4]
    print("\nNhãn `Xong` nhưng thiếu từ 3 khối trở lên: %d bài" % len(xau))

    if "--thieu" in sys.argv:
        print("\n%-46s %-8s %4s %5s  thiếu" % ("Bài", "GĐ", "Tier", "dòng"))
        print("-" * 100)
        for b in sorted(xau, key=lambda x: (sum(x[5].values()), x[0]))[:60]:
            thieu = ",".join(k for k in TEN if not b[5][k])
            print("%-46s %-8s %4s %5d  %s" % (b[0][:46], b[1], b[2], b[4], thieu))
