# -*- coding: utf-8 -*-
"""Dò xưng hô lọt vào thân bài — Luật 3 của `Dàn ý mẫu`.

Luật 3: *"Xưng hô là quy ước của cuộc trò chuyện, không phải giọng văn của tài
liệu. Trong file `.md`: không gọi tên người đọc, không 'đại ca', không 'mình',
không 'em'."* Cách viết đúng là câu vô nhân xưng, hoặc "người đọc" / "người dùng".

Ngày 2026-08-29 đã dọn 25 chỗ **"đại ca"** ở 14 bài. Nhưng đó là lớp dễ thấy
nhất. Lớp đông hơn nhiều là **"tôi"** (người viết tự xưng, hay gặp ở mục kết quả
âm) và **"bạn"** (gọi thẳng người đọc, hay gặp ở mục nhập môn) — chưa lần nào
được đếm, vì chưa có phép kiểm nào.

BA THỨ KHÔNG PHẢI VI PHẠM, và đây là phần khó:

1. **"mình" phản thân, chỉ người thứ ba.** *"Zeiler tự đo bước đi của mình"*,
   *"Hestenes và Stiefel biết mình làm cùng một việc"* — "mình" ở đây thay cho
   chủ ngữ đã nêu, không phải người viết. Hợp lệ.
2. **"một mình" là trạng từ.** *"mã in trong bài không chạy được một mình"*,
   *"bài đứng độc lập, đọc được một mình"* — cụm cố định, và chính là thứ Luật 1
   đòi hỏi. Mẫu ad-hoc ngày 2026-08-30 bắt nhầm 4 chỗ dạng này.
3. **Trích dẫn nguyên văn.** Câu tiếng Anh trong khối trích của tác giả (*"we
   don't need to correct…"*) và lời tự nhận của chính tác giả không phải giọng
   của bài.

Cùng bài học ở CLAUDE.md mục 4: phép kiểm và tài liệu chuẩn phải nói cùng một
ngôn ngữ, và phép kiểm nào **bỏ qua** đầu vào lạ thì phải **báo ra** cái nó bỏ.

Chạy:  cd tools && python xungho.py           # toàn database
       cd tools && python xungho.py "GĐ 5"    # một giai đoạn
"""
import io
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(os.path.dirname(HERE), "AI algorithms", ".flat")

# Xưng hô thật. "mình" và "em" CỐ Ý không nằm ở đây — xem miễn trừ 1 và 2 trong
# docstring; chúng cần ngữ cảnh mới phân biệt được, và bắt bừa thì phép kiểm mất
# giá trị. Ai muốn soát tay hai từ đó thì chạy với --rong.
NGHI = re.compile(r"(?<![\w])(tôi|Tôi|bạn|Bạn|đại ca|Đại ca)(?![\w])")
RONG = re.compile(r"(?<![\w])(mình|Mình|em|Em|chúng ta|Chúng ta)(?![\w])")

# Cụm cố định chứa "mình" nhưng không phải xưng hô — chỉ dùng ở chế độ --rong.
BO_RONG = re.compile(r"một mình|của mình|biết mình|thấy mình|nghĩ mình|"
                     r"tự mình|mình đang|mình sắp|mình làm|mình muốn")


def gd(t):
    m = re.search(r"(?m)^Giai đoạn:\s*(.+)$", t)
    return m.group(1).strip() if m else "?"


def bo_trich(s):
    """Thay code fence và trích nguyên văn bằng dòng rỗng, giữ nguyên số dòng.

    Trích nguyên văn = cụm *"…"* — giọng của tác giả bài báo, không phải giọng
    của bài viết.

    BẢN VÁ THỨ BA, Đợt 25 (2026-08-31): BỎ miễn trừ theo dấu '>'
    ------------------------------------------------------------
    Bản trước tha **mọi dòng bắt đầu bằng '>'**, với lý do blockquote là chỗ đặt
    trích dẫn. Lý do đó sai ở database này: dấu '>' còn là khuôn của **hộp nhấn
    mạnh do chính bài viết ra** — "> **Đánh đổi thật.** Nếu bạn cần một mô hình
    để đọc…" — tức đúng giọng mà Luật 3 cấm.

    Miễn trừ ấy còn **thừa**: phép xoá cụm *"…"* ở trên đã chạy trên TOÀN văn bản
    trước khi lọc theo dòng, nên trích dẫn nguyên văn đã biến mất rồi. Giữ thêm
    nhánh '>' chỉ tạo ra một lỗ.

    Đo bán kính trước khi vá, theo luật Đợt 21: toàn database **195 → 248** chỗ
    (+53, +6 bài); riêng GĐ 2 **11 → 20**. Đọc tay cả 9 chỗ mới của GĐ 2:
    **9/9 là vi phạm THẬT, không có lật giả** — "thí nghiệm của tôi", "Tôi đã thử
    đo", "Nếu lý do bạn chọn…". Đây là lớp lỗi CLAUDE.md ghi ở Đợt 22 lặp lại
    lần nữa: *mẫu miễn trừ rộng hơn thứ nó định miễn*.

    LỊCH SỬ HAI BẢN VÁ, Đợt 23 (2026-08-30)
    ---------------------------------------
    Bản đầu dò `\*"[^"]*"\*` TRÊN TỪNG DÒNG. Một câu trích dài trải hai dòng thì
    lọt lưới — đo được ở bài AlexNet, nơi câu dịch nguyên văn *"chúng tôi đã không
    thể thử nghiệm những mạng nơ-ron lớn như thế này…"* bị báo vi phạm Luật 3,
    trong khi "chúng tôi" là giọng Krizhevsky chứ không phải giọng bài viết.

    **Bản vá thứ nhất đã SAI và bị bỏ.** Nó theo dõi trạng thái "đang trong cụm
    trích" bằng parity số dấu `*"` và `"*` trên mỗi dòng. Trạng thái đó **rò** qua
    những dòng không liên quan, và kết quả là nó tha thêm **10 vi phạm THẬT** ở 4
    bài (Balanced Random Forest, Bayesian Linear Regression, DBSCAN, Hierarchical
    Clustering, Huber Regression) — đúng lớp lỗi mà CLAUDE.md ghi ở Đợt 22: *mẫu
    miễn trừ rộng hơn thứ nó định miễn*. Phát hiện nhờ đọc tay từng dấu ✓ mới, theo
    đúng quy tắc bắt buộc sau mỗi lần vá detector.

    Bản hiện tại **khoanh đúng vùng ký tự**: tìm mọi cụm `*"…"*` trên toàn văn bản
    bằng `re.DOTALL`, xoá đúng những vùng đó, rồi mới cắt dòng. Không có trạng thái
    nào để rò, và cụm trích dài bao nhiêu dòng cũng đúng.
    """
    # Xoá trọn các cụm *"…"* kể cả khi chúng trải nhiều dòng. Giữ ký tự xuống dòng
    # để số dòng không đổi — cột "dòng" trong báo cáo phải trỏ đúng chỗ.
    def _xoa(m):
        return "".join(c if c == "\n" else " " for c in m.group())

    # Cụm trích phải: KHÔNG chứa dấu nháy kép bên trong, KHÔNG bắc qua dòng
    # trống (tức không vượt ranh giới đoạn văn), và dài tối đa 1600 ký tự.
    # Thiếu ba ràng buộc này thì một dấu `*"` lạc sẽ nuốt tới dấu `"*` xa hàng
    # trăm dòng và giấu mất mọi vi phạm ở giữa — đã đo được 3 chỗ như thế ở bài
    # Bayesian Linear Regression khi thử bản không ràng buộc.
    #
    # Trần 400 ký tự của bản trước viết cho những câu trích ngắn. **Luật 6** đổi
    # điều đó: từ nay mỗi hình ruột phải dán NGUYÊN VĂN cả caption gốc lẫn đoạn
    # thân bài tác giả bình luận về nó, và caption một bài báo CNN thường dài
    # 600–1200 ký tự. Trần 400 vì thế cắt đôi đúng những đoạn phải được miễn, và
    # báo giọng tác giả ("chúng tôi") là vi phạm Luật 3.
    #
    # Đo bán kính trước khi nới, theo luật Đợt 21 — nới 400 → 1600 trên toàn
    # database: **9 chỗ được tha thêm, tất cả ở một bài (ZFNet), tất cả là câu
    # dịch nguyên văn của Zeiler & Fergus.** Không có chỗ nào bị tha oan, vì hai
    # ràng buộc còn lại (không có `"` bên trong, không bắc qua dòng trống) mới
    # là hai ràng buộc thật sự chặn việc nuốt.
    s = re.sub(r'\*"(?:(?!\n\s*\n)[^"]){1,1600}?"\*', _xoa, s)

    out, trong = [], False
    for line in s.split("\n"):
        if line.lstrip().startswith("```"):
            trong = not trong
            out.append("")
            continue
        if trong:
            out.append("")
            continue
        out.append(line)
    return out


def quet(loc=None, rong=False):
    vi_pham, bo = [], 0
    mau = RONG if rong else NGHI
    for f in sorted(os.listdir(DB)):
        if not f.endswith(".md"):
            continue
        s = io.open(os.path.join(DB, f), encoding="utf-8", errors="replace").read()
        if loc and gd(s) != loc:
            continue
        for i, line in enumerate(bo_trich(s), 1):
            m = mau.search(line)
            if not m:
                continue
            if rong and BO_RONG.search(line):
                bo += 1
                continue
            vi_pham.append((f[:-3], i, m.group(0), line.strip()))
    return vi_pham, bo


def main():
    args = [a for a in sys.argv[1:] if a != "--rong"]
    rong = "--rong" in sys.argv
    loc = args[0] if args else None
    vi_pham, bo = quet(loc, rong)

    print("LUẬT 3 — không xưng hô trong file .md%s%s"
          % (" · " + loc if loc else "", " · chế độ --rong" if rong else ""))
    print("-" * 78)
    bai = set()
    for ten, dong, tu, noi_dung in vi_pham:
        bai.add(ten)
        print("  %-34s dòng %-5d %-7s %s" % (ten[:34], dong, tu, noi_dung[:60]))
    print("-" * 78)
    print("  %d chỗ vi phạm · %d bài%s"
          % (len(vi_pham), len(bai),
             " · %d chỗ được miễn trừ (cụm cố định)" % bo if rong else ""))
    if not rong:
        print("  Hai từ 'mình' và 'em' cần ngữ cảnh: chạy thêm `--rong` rồi soát tay.")
    return 1 if vi_pham else 0


if __name__ == "__main__":
    sys.exit(main())
