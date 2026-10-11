# -*- coding: utf-8 -*-
"""So md5 giữa HAI đường sinh hình: khối mã in trong bài, và script trong tools/.

VÌ SAO SCRIPT NÀY TỒN TẠI
-------------------------
CLAUDE.md mục 2.3.2 đòi khối mã in trong bài chạy ra **đúng** hình đang nằm trong
repo. Đợt 14 đã ghi rằng `regen.py` báo `5/5 hình sinh lại được, 0 lỗi` rồi mới lộ
ra hai đường sinh hình cho hai ảnh khác nhau — **dòng `OK` chỉ chứng minh mã CHẠY
ĐƯỢC**, không chứng minh hình đúng. Bài học đó kết thúc bằng một quy ước:
*"phép kiểm thật là so md5 giữa hai đường sinh hình"*.

Nhưng Đợt 22 đã đo được điều gì xảy ra với một quy ước không có phép kiểm đứng
sau: bản vá va chạm tên hình của Đợt 10 **hỏng lại sau bốn đợt**, vì thứ được gỡ
hôm đó chỉ là một câu văn. Nên quy ước md5 nay thành script này.

Nó bắt lại đúng lớp lỗi cũ ngay lần chạy đầu, ở bài `K‑Means` (Đợt 27): năm khối mã
kết thúc bằng `plt.savefig(..., dpi=150)` trong khi `unlib.save()` ghi bằng
`fig.savefig(..., bbox_inches="tight")` với `savefig.dpi=130` đặt qua `rcParams`.
Mã chạy được, `regen.py` báo `OK`, và **cả năm hình đều khác**.

    cd tools && python sohinh.py "K‑Means"        # một bài
    cd tools && python sohinh.py --gd "GĐ 4"      # cả một giai đoạn

Chạy chậm: mỗi bài phải chạy hai lần toàn bộ mã sinh hình.
"""
import glob
import hashlib
import io
import os
import re
import shutil
import subprocess
import sys
import tempfile

sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(os.path.dirname(HERE), "AI algorithms", ".flat")


# ── Hình KHÔNG THỂ khớp md5 theo cấu tạo, kèm lý do.
#
# Chỉ có một lý do hợp lệ: hình tự vẽ vào mình một đại lượng không tái lập được.
# Mọi lý do khác ("chênh ít thôi", "chắc do phiên bản") là chưa truy nguyên xong.
# Danh sách này phải NGẮN; nó dài ra nghĩa là đang có hình không kiểm được.
MIEN_TRU = {
    "K‑Means/clustering_grid.png":
        "mỗi ô in thời gian chạy của thuật toán đó (`%.2fs`), nên hai lần chạy "
        "khác nhau theo cấu tạo — cùng lý do khiến cột thời gian bị loại khỏi "
        "`kiem_gd4.py`",
    "Principal Component Analysis (PCA)/so_tay_giam_chieu.png":
        "mỗi ô in thời gian chạy (`%.1fs`) trong tiêu đề — cùng lý do như trên",
    "Transformer/speed_comparison.png":
        "hình vẽ CHÍNH cột thời gian đo được — sáu đường là sáu lần đo giây trên "
        "GPU, nên hai lần chạy cho hai file khác nhau theo cấu tạo. Cùng lý do "
        "khiến cột giây của bài phải đọc theo BẬC ĐỘ LỚN chứ không theo chữ số",
}


def md5(p):
    return hashlib.md5(io.open(p, "rb").read()).hexdigest()


def anh(thu_muc):
    """{tên hình: md5} của mọi .png không phải hình cắt từ paper."""
    if not os.path.isdir(thu_muc):
        return {}
    return {os.path.basename(p): md5(p)
            for p in sorted(glob.glob(os.path.join(thu_muc, "*.png")))
            if not os.path.basename(p).startswith("paper_")}


def script_sinh(ten_hinh):
    """Script nào trong tools/ ghi ra tên hình này."""
    ra = []
    for p in sorted(glob.glob(os.path.join(HERE, "*.py"))):
        if os.path.basename(p) in ("sohinh.py", "regen.py"):
            continue
        if ten_hinh in io.open(p, encoding="utf-8", errors="replace").read():
            ra.append(os.path.basename(p))
    return ra


def chay(lenh, *args):
    return subprocess.run([sys.executable, lenh] + list(args), cwd=HERE,
                          capture_output=True, text=True,
                          encoding="utf-8", errors="replace")


def mot_bai(ten):
    """Trả (số hình LỆCH, số hình MỒ CÔI) — hai loại hỏng KHÁC NHAU.

    Bản đầu gộp chung một bộ đếm, nên dòng tổng của GĐ 6 ghi "3 hình LỆCH"
    trong khi cả ba thật ra là hình MỒ CÔI: không script nào trong `tools/`
    sinh ra chúng, tức chưa bao giờ có đường sinh thứ hai để mà so.

    Hai chuyện khác hẳn nhau — một cái là *"hai đường cho kết quả khác nhau"*,
    một cái là *"chỉ có một đường"* — và gộp chung làm người đọc báo cáo hiểu
    sai con số theo hướng nhẹ đi: hình mồ côi nghe như một chỗ lệch nhỏ, trong
    khi nó là hình **không kiểm được**, đúng thứ mục 2.3.2 của CLAUDE.md cấm.
    """
    d = os.path.join(DB, ten)
    truoc = anh(d)
    if not truoc:
        print("  %-40s (không có hình tự vẽ)" % ten[:40])
        return 0, 0

    # đường A — script trong tools/
    scripts = sorted({s for h in truoc for s in script_sinh(h)})
    if not scripts:
        print("  %-40s ⚠ KHÔNG script nào trong tools/ sinh: %s"
              % (ten[:40], " ".join(sorted(truoc))))
        return 0, len(truoc)
    for sc in scripts:
        chay(sc)
    A = anh(d)

    # đường B — khối mã in trong chính bài
    sao = tempfile.mkdtemp(prefix="sohinh_")
    for h in A:
        shutil.copy2(os.path.join(d, h), sao)
    r = chay("regen.py", ten)
    B = anh(d)

    # CHỖ MÙ TÌM RA Ở ĐỢT 31 — đường B có thể KHÔNG TỒN TẠI mà vẫn báo "khớp".
    # `regen.py` dựng hình từ khối mã in trong bài. Bài nào KHÔNG in mã thì nó
    # sinh 0 hình, không ghi đè gì cả, nên B == A và mọi hình được tính là khớp.
    # Nói cách khác: phép so hai đường vẫn báo xanh khi chỉ có MỘT đường.
    # Đo được lần đầu trên hai bài vừa viết theo Luật 6 (Transformer, BERT):
    # regen in "0 hình sinh lại được" cho cả hai, mà sohinh báo 5/5 "khớp".
    # Đúng lớp lỗi mà CLAUDE.md mục 4 đã ghi hai lần: cái gì bỏ qua được thì
    # phải BÁO RA, một `continue` im lặng làm con số "0 hỏng" thành vô nghĩa.
    so_regen = 0
    for dong in (r.stdout or "").split("\n"):
        m = re.search(r"(\d+)\s+hình sinh lại được", dong)
        if m:
            so_regen = int(m.group(1))
    # Nhánh này CHẨN ĐOÁN SAI nếu bài CÓ in mã mà mã gãy hết: đo được ngày
    # 2026-09-03 ở bài β‑VAE, nơi regen dựng 0 hình vì cả hai khối đều
    # `NameError`, và dòng in ra vẫn ghi "BÀI KHÔNG IN MÃ SINH HÌNH". Hai chuyện
    # khác nhau — "không có mã" so với "có mã nhưng không chạy nổi" — và cách
    # chữa cũng khác nhau. Nên phải đọc số dòng LỖI trước khi kết luận.
    so_loi = len(re.findall(r"(?m)^\s*LỖI\s+\S+\.png", r.stdout or ""))
    if so_regen == 0:
        if so_loi:
            print("  %-40s ⚠ MÃ SINH HÌNH CỦA BÀI KHÔNG CHẠY ĐƯỢC — regen.py gãy %d/%d khối"
                  % (ten[:40], so_loi, so_loi))
        else:
            print("  %-40s ⚠ BÀI KHÔNG IN MÃ SINH HÌNH — regen.py dựng lại 0 hình, nên"
                  % ten[:40])
        print("  %-40s   %d hình dưới đây CHƯA BAO GIỜ được kiểm: %s"
              % ("", len(truoc), " ".join(sorted(truoc))))
        return 0, len(truoc)

    # Biến thể TỪNG PHẦN của chỗ mù trên, đo được ngày 2026-09-03 ở bài VAE:
    # `regen.py` dựng được 1 hình và GÃY ở 2 hình (`NameError: 'kq8'` — khối mã
    # in trong bài là một đoạn TRÍCH, không chạy được một mình). Nhánh
    # `so_regen == 0` ở trên không bắt được vì so_regen = 1, nên hai hình gãy
    # vẫn giữ nguyên bản của đường A và được in ra là "khớp". Dòng tổng khi ấy
    # ghi "0 hình LỆCH" trong khi hai hình CHƯA BAO GIỜ có đường thứ hai để so.
    # Cùng lớp lỗi Đợt 31 nói: đừng gộp "so được và khớp" với "không so được".
    gay, dung = set(), set()
    for dong in (r.stdout or "").split("\n"):
        m = re.match(r"\s*LỖI\s+(\S+\.png)", dong)
        if m:
            gay.add(m.group(1))
        m = re.match(r"\s*OK\s+(\S+\.png)", dong)
        if m:
            dung.add(m.group(1))

    # BIẾN THỂ THỨ BA của cùng chỗ mù, đo được ngày 2026-09-03 ở bài Autoencoder:
    # regen dựng lại 1 hình, gãy 0 hình — và BA hình còn lại của bài không có khối
    # mã nào để dựng. `regen.py` bắt khối theo `savefig(`, nên khối kết thúc bằng
    # `dlib.save()` bị BỎ QUA ÂM THẦM: không OK, không LỖI, không dòng nào cả.
    # Ba hình ấy vì thế giữ nguyên bản của đường A, B == A, và được in ra là
    # "khớp". Nhánh `so_regen == 0` không cứu được vì so_regen = 1.
    # Cách chữa duy nhất đúng: so với tập hình regen THẬT SỰ dựng lại.
    thieu = {h for h in A if h not in dung and h not in gay}

    lech = [h for h in sorted(A)
            if h not in gay and h not in thieu
            and A.get(h) != B.get(h) and (ten + "/" + h) not in MIEN_TRU]
    for h in sorted(A):
        khoa = ten + "/" + h
        if h in gay:
            print("  %-40s %-26s ⚠ KHÔNG KIỂM ĐƯỢC — regen.py gãy ở khối mã của bài"
                  % (ten[:40], h[:26]))
        elif h in thieu:
            print("  %-40s %-26s ⚠ KHÔNG KIỂM ĐƯỢC — bài không in khối mã có savefig()"
                  % (ten[:40], h[:26]))
        elif h in lech:
            print("  %-40s %-26s LỆCH  (script: %s)"
                  % (ten[:40], h[:26], " ".join(scripts)))
        elif khoa in MIEN_TRU:
            print("  %-40s %-26s miễn trừ — %s"
                  % (ten[:40], h[:26], MIEN_TRU[khoa].split(",")[0]))
        else:
            print("  %-40s %-26s khớp" % (ten[:40], h[:26]))
    if r.returncode:
        print("  %-40s ⚠ regen.py thoát %d" % (ten[:40], r.returncode))
    # trả hình về bản của script trong tools/, để lần chạy này không đổi repo
    for h in A:
        shutil.copy2(os.path.join(sao, h), d)
    shutil.rmtree(sao, ignore_errors=True)
    # Hình regen gãy, và hình bài không in mã, đều đi vào bộ đếm MỒ CÔI — không
    # vào bộ đếm LỆCH và tuyệt đối không vào cột "khớp": chúng KHÔNG KIỂM ĐƯỢC.
    return len(lech), len(gay) + len(thieu)


def main():
    if "--gd" in sys.argv:
        gd = sys.argv[sys.argv.index("--gd") + 1]
        ten = [os.path.basename(p)[:-3] for p in sorted(glob.glob(os.path.join(DB, "*.md")))
               if "Giai đoạn: " + gd in io.open(p, encoding="utf-8").read()]
    else:
        ten = [a for a in sys.argv[1:] if not a.startswith("--")]
        # Gọi `sohinh.py "GĐ 6"` mà quên cờ `--gd` là lỗi dễ mắc, và trước ngày
        # 2026-09-01 script IM LẶNG coi đó là tên một bài: nó khớp mờ được đúng
        # một bài rồi in "1 bài · 0 hình LỆCH" — một con số xanh hoàn toàn vô
        # nghĩa. Đúng lớp lỗi mà CLAUDE.md mục 4 cấm: cái gì bỏ qua được thì
        # phải BÁO RA.
        for a in ten:
            if a.strip().lower().startswith(("gđ ", "gd ", "nhánh ", "nhanh ")):
                sys.exit("Có vẻ đây là tên GIAI ĐOẠN, không phải tên bài: %r\n"
                         "Dùng:  python sohinh.py --gd %r" % (a, a))
    if not ten:
        sys.exit(__doc__)
    thieu = [t for t in ten if not os.path.exists(os.path.join(DB, t + ".md"))]
    if thieu:
        sys.exit("Không thấy bài: %s" % " · ".join(repr(t) for t in thieu))

    print("SO MD5 GIỮA HAI ĐƯỜNG SINH HÌNH")
    print("-" * 92)
    kq = [mot_bai(t) for t in ten]
    lech = sum(a for a, _ in kq)
    mo_coi = sum(b for _, b in kq)
    print("-" * 92)
    print("  %d bài · %d hình LỆCH giữa hai đường sinh"
          " · %d hình MỒ CÔI (không có đường sinh thứ hai để so)"
          % (len(ten), lech, mo_coi))
    return 1 if (lech or mo_coi) else 0


if __name__ == "__main__":
    sys.exit(main())
