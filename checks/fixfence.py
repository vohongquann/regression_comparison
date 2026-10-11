# -*- coding: utf-8 -*-
"""Dò và sửa code fence bị vỡ trong bài .md nhập từ Notion.

**Vì sao cần.** Bản nhập Notion thỉnh thoảng chèn thừa một dấu ``` — thường là
một cặp fence RỖNG (mở rồi đóng ngay, giữa chỉ có dòng trắng), hoặc cắt đôi một
câu văn. Hậu quả không dừng ở chỗ đó: **mọi khối mã phía sau bị lệch một nhịp**.
Tiêu đề bị kẹt trong khối mã, mã Python render thành văn xuôi, và file có thể
kết thúc khi khối vẫn đang mở.

Phát hiện ngày 2026-08-26 khi đọc soát 16 bài CNN: **5 bài hỏng** — RegNet
(34 tiêu đề kẹt, ~780 dòng render sai), ConvNeXt (45), EfficientNet (48),
GoogLeNet (43), ResNeXt (28). Không script nào trước đó bắt được, và đọc bản
.md thô cũng khó thấy vì nội dung vẫn còn nguyên.

Chạy:
    cd tools && python fixfence.py              # chỉ BÁO CÁO, không sửa
    cd tools && python fixfence.py --fix        # sửa thật
    cd tools && python fixfence.py "<tên bài>" --fix
"""
import io
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(os.path.dirname(HERE), "AI algorithms", ".flat")


def diagnose(lines):
    """Trả về (số fence, số tiêu đề kẹt trong khối mã, file có kết thúc khi đang mở)."""
    inb = False
    stuck = 0
    for l in lines:
        if l.startswith("```"):
            inb = not inb
            continue
        # CHỈ đếm dấu hiệu chắc chắn là markdown: đoạn in đậm mở đầu dòng, hoặc
        # tiêu đề mục. KHÔNG dùng "#" trần — chú thích Python cũng bắt đầu bằng "#"
        # và sẽ cho hàng loạt dương tính giả trong khối mã hợp lệ.
        if inb and (l.startswith("**") or l.startswith("## ") or l.startswith("### ")):
            stuck += 1
    return sum(1 for l in lines if l.startswith("```")), stuck, inb


def empty_pairs(lines):
    """Các cặp fence chỉ chứa dòng trắng — chữ ký của fence thừa."""
    idx = [i for i, l in enumerate(lines) if l.startswith("```")]
    return [(a, b) for a, b in zip(idx, idx[1:])
            if all(not lines[k].strip() for k in range(a + 1, b))]


def swallowing_fence(lines):
    """Chỉ số dòng ``` đã MỞ khối mã đang nuốt tiêu đề đầu tiên (None nếu sạch)."""
    inb, opened = False, None
    for i, l in enumerate(lines):
        if l.startswith("```"):
            if not inb:
                opened = i
            inb = not inb
            continue
        if inb and (l.startswith("**") or l.startswith("## ") or l.startswith("### ")):
            return opened
    return None


def repair(lines):
    """Xoá fence thừa cho tới khi hết tiêu đề bị kẹt.

    Chiến lược: mỗi vòng, tìm khối mã đang **nuốt** một tiêu đề rồi xoá đúng dấu
    ``` đã mở khối đó. Chỉ xoá dòng chứa ```, không đụng nội dung.

    Vì sao không dùng cách khác:

    * *Lấy fence rỗng đầu tiên* — sai. Thử ngày 2026-08-26: cách này đẩy số tiêu
      đề kẹt của ConvNeXt từ 52 lên **63**.
    * *Thử mọi vị trí rồi chấm điểm* — bỏ sót bài hỏng ở NHIỀU chỗ. EfficientNet
      có hai chỗ vỡ và tổng fence vẫn **chẵn**; xoá chỗ thứ nhất làm file tạm
      thời kết thúc khi khối còn mở, nên phép chấm điểm từ chối nước đi đó và
      không bao giờ tới được chỗ thứ hai.

    Sau khi hết chỗ kẹt, nếu file vẫn kết thúc khi khối đang mở thì thêm một
    fence đóng ở cuối.

    Trả về (lines mới, danh sách số dòng đã xoá).
    """
    removed = []
    for _ in range(40):                       # chặn trên, tránh lặp vô hạn
        i = swallowing_fence(lines)
        if i is None:
            break
        del lines[i]
        removed.append(i + 1)
    if diagnose(lines)[2]:                    # cuối file vẫn đang mở khối
        lines.append("```")
    return lines, removed


def run(path, name, fix):
    lines = io.open(path, encoding="utf-8", newline="").read().split("\n")
    n0 = len(lines)
    f0, s0, o0 = diagnose(lines)
    if s0 == 0 and not o0 and f0 % 2 == 0:
        return 0

    print("\n### %s" % name)
    print("   fence %d (%s) · %d tiêu đề kẹt trong khối mã · cuối file %s"
          % (f0, "lẻ" if f0 % 2 else "chẵn", s0, "ĐANG MỞ" if o0 else "đóng"))
    if not fix:
        print("   (chạy với --fix để sửa)")
        return 1

    lines, removed = repair(lines)
    f1, s1, o1 = diagnose(lines)
    assert abs(len(lines) - (n0 - len(removed))) <= 1, "chỉ được xoá dòng fence, không mất nội dung"
    io.open(path, "w", encoding="utf-8", newline="\n").write("\n".join(lines))
    print("   đã xoá %d fence thừa (dòng %s) -> fence %d (%s) · kẹt %d · cuối file %s · dòng %d -> %d"
          % (len(removed), removed, f1, "lẻ" if f1 % 2 else "chẵn", s1,
             "ĐANG MỞ" if o1 else "đóng", n0, len(lines)))
    if s1 or o1:
        print("   ⚠ CHƯA SẠCH — cần sửa tay, dạng vỡ không phải fence rỗng")
    return 1


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--fix"]
    fix = "--fix" in sys.argv
    want = args[0] if args else None
    n = 0
    for f in sorted(os.listdir(DB)):
        if not f.endswith(".md"):
            continue
        if want and want not in f:
            continue
        n += run(os.path.join(DB, f), f[:-3], fix)
    print("\n%d bài hỏng cấu trúc code fence%s" % (n, "" if fix else " (chưa sửa)"))
