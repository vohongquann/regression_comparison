# -*- coding: utf-8 -*-
"""Đối chiếu MỌI đoạn trích tiếng Anh trong một bài với lớp text của PDF gốc.

Luật 6 đòi mỗi hình ruột phải dán **nguyên văn** caption và đoạn thân bài mà tác giả
dùng để bình luận hình đó. Trước script này, không công cụ nào trong `tools/` kiểm
điều đó: `kiem_gd6` trích con số, `luat1` và `xungho` dò từ tiếng Việt, `kiempdf` chỉ
hỏi file PDF có tồn tại không, `sopaper` đếm hình. Một câu trích sai một chữ trông y
hệt một câu trích đúng.

Chạy lần đầu trên bài WGAN: 46 đoạn trích, **2 sai thật**, và cả hai cùng một khuôn —
người viết cắt câu sớm hơn bài báo rồi đặt dấu chấm vào chỗ bài báo đặt dấu phẩy:

  · *"…assigns the highest values."*  — bài báo viết *"…assigns the highest values, as
    observed by [4] and highlighted in [11]."*
  · *"…collapse to a single nonsensical image."* — bài báo có thêm *"[4]"* ở cuối.

Bốn chỗ phải xử lý, nếu không thì phép so chuỗi con báo nhầm hàng loạt:

  1. **Công thức LaTeX** — bài viết thay ký hiệu toán của bài báo bằng `$...$`. Xoá đi
     thì hai vế dính liền và không khớp; phải biến thành KÝ TỰ ĐẠI DIỆN.
  2. **Dấu ba chấm** đánh dấu đoạn lược bớt — cũng là ký tự đại diện.
  3. **Dấu nối từ bị cắt ngang dòng.** Ở PDF của WGAN nó là **U+FFFE**, không phải
     U+FFFD như thường gặp — `running experi<U+FFFE>ments`. Phải dò từng file.
  4. **Số trang dính vào từ đầu của trang sau**: lớp text cho ra
     `"Note also that the / 11JS estimate usually stays…"`. Không cắt thì mọi đoạn
     trích vắt qua ranh giới trang đều báo nhầm.

Chạy:
    cd tools && python kiemtrich.py --bai "WGAN" --pdf "2017 - Arjovsky, ....pdf"
    cd tools && python kiemtrich.py --bai "WGAN"      # tự dò PDF từ mục tham khảo
"""
import argparse
import io
import os
import re
import unicodedata
import sys

sys.stdout.reconfigure(encoding="utf-8")

import pypdfium2 as pdfium  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DB = os.path.join(REPO, "AI algorithms", ".flat")
PAPERS = os.path.join(REPO, "AI algorithms", "_papers")
MOC = "@@DAIDIEN@@"
# Nhung ky tu ma lop text cua PDF chen vao giua mot tu bi cat ngang dong. Danh sach
# nay MO RONG DAN: moi file PDF dung mot ky tu khac nhau, va U+FFFE la ky tu ma bai
# WGAN dung — khong doan duoc, phai do.
NOI_TU = "­¬¸�￾"
# BỘ MÃ CHỮ GHÉP HỎNG của bản PDF Sutton & Barto (2018): lớp text in "ff" thành
# U+21B5 (↵), "fi"/"ffi" thành U+FFFE — chung ký tự với dấu nối ngắt dòng —, và
# π thành U+21E1 (⇡). "different" đọc ra là "di↵erent", "efficient" là "e￾cient".
# Không phục hồi được "ffi" từ U+FFFE, nên cách so là BỎ các cụm chữ ghép ở CẢ HAI
# vế: "efficient" -> "ecient", "e￾cient" -> "ecient". Chỉ bật khi lớp text của PDF
# có nhiều ký tự ↵ (bộ mã này), để không đổi phép so của mọi PDF khác. Thêm
# 2026-09-30 cho bài Monte Carlo Control: 5 trên 6 đoạn "không khớp" là do bộ mã này.
GHEP_CHU = False


def chuan(t):
    """Chuẩn hoá cả hai vế trước khi so."""
    for a, b in (("ﬀ", "ff"), ("ﬁ", "fi"), ("ﬂ", "fl"),
                 ("‐", "-"), ("‑", "-"), ("–", "-"), ("—", "-"),
                 ("‘", "'"), ("’", "'"), ("“", '"'), ("”", '"'),
                 ("…", "...")):
        t = t.replace(a, b)
    if GHEP_CHU:
        t = t.replace("\u21b5", "ff").replace("\u21e1", "π")
    t = re.sub("[" + NOI_TU + "]", "", t)
    if GHEP_CHU:
        t = re.sub(r"ffi|ffl|ff|fi|fl", "", t)
    # Lớp text của PDF tạp chí Science mã hoá dấu nháy cong THÀNH CHỮ CÁI: câu
    # 'an adaptive, multilayer "encoder" network' đọc ra là 'multilayer Bencoder['.
    # Không xử lý thì mọi câu có từ đặt trong nháy của bài Hinton 2006 đều báo nhầm.
    t = re.sub(r"B([A-Za-z][A-Za-z-]*)\[", r"\1", t)
    # Dấu nháy quanh một từ là TRÌNH BÀY, không phải chữ — bỏ ở cả hai vế.
    t = t.replace("''", "").replace('"', "")
    # DẤU NỐI của một từ ghép biến mất khi PDF ngắt dòng ngay tại đó: bài báo
    # Hinton 2006 in "fine-tuning", lớp text cho ra "fine<U+FFFE>tuning", và sau
    # khi bỏ ký tự nối thì thành "finetuning" — không còn dấu nối để khớp với
    # "fine-tuning" của bài viết. Bỏ dấu nối ở CẢ HAI vế thì hai bên gặp nhau.
    # CHỈ bỏ dấu nối nằm giữa hai CHỮ CÁI. Bỏ hết thì chuỗi kiến trúc
    # "625-2000-1000-500-30" của chính bài báo đó dính thành một khối chữ số và
    # sinh ra một lớp báo nhầm mới — đo được ngay khi thử bản bỏ hết.
    # Phai khong phan biet HOA THUONG: `chuan()` ha chu ve thuong o CUOI, nen luc
    # nay "DC-IGN" van con hoa va luat chi-chu-thuong khong ban toi no — trong khi
    # lop text cua bai beta-VAE cho ra "DCIGN" lien mot khoi.
    t = re.sub(r"(?<=[A-Za-z])-(?=[A-Za-z])", "", t)
    # Bai viet hay IN DAM mot cum ben trong doan trich de chi cho dang doc. Dau ** va
    # _ la trinh bay, khong phai chu — bo di truoc khi so, neu khong thi moi cho nhan
    # manh deu bao nham.
    t = t.replace("**", "").replace("__", "")
    # Chu co DAU HAI CHAM tren dau bi lop text tach lam hai: JRSS-B in "naive" kieu
    # Phap ("naïve") va lop text cho ra "na¨ıve" — dau ¨ dung rieng, cong chu i KHONG
    # CHAM (U+0131). Khong xu ly thi MOI doan trich cua bai Zou & Hastie 2005 co chu do
    # deu bao nham: do duoc 7 tren 90 doan. Bo dau ket hop va dua i-khong-cham ve i.
    t = unicodedata.normalize("NFD", t)
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = t.replace("\u00a8", "").replace("\u0131", "i")
    return re.sub(r"\s+", " ", t).lower()


def doc_pdf(p):
    d = pdfium.PdfDocument(p)
    trang = []
    for i in range(len(d)):
        v = d[i].get_textpage().get_text_range().rstrip()
        v = re.sub(r"\s*\n\s*\d{1,3}\s*$", "", v)     # cắt số trang ở cuối trang
        # SỐ HIỆU CHÚ THÍCH CHÂN TRANG dính vào cuối từ ("DCGANs1 using") và vào đầu
        # đường dẫn ("are set 1https://github.com/..."). Chúng không thuộc câu, nên
        # bài viết trích ĐÚNG khi bỏ chúng — phải bỏ ở vế PDF để hai vế gặp nhau.
        # Đòi ÍT NHẤT HAI chữ cái trước con số. Bản đầu chỉ đòi một, và nó cắt
        # luôn CHỈ SỐ DƯỚI của biến: câu "the discrete code c1 captures" của bài
        # báo InfoGAN thành "code c captures", làm hai đoạn trích đúng bị báo sai.
        # Số hiệu chú thích thì luôn đứng sau một TỪ, không sau một chữ cái đơn.
        v = re.sub(r"(?<=[a-zA-Z][a-zA-Z])\d{1,2}(?=[\s(])", "", v)
        v = re.sub(r"\d{1,2}(?=https?://)", "", v)
        # So hieu chu thich con dinh vao sau DAU CHAM cuoi cau: bai CVAE co
        # "our proposed deep CGMs.5 In terms of...". Doi mot CHU CAI truoc dau
        # cham, neu khong thi "3.5 " cung bi cat mat chu so 5.
        # Chữ cái trước dấu chấm phải là chữ THƯỜNG, tức cuối một từ. Bản đầu nhận
        # cả chữ hoa và vì thế ăn luôn số hiệu phụ lục "A.1 MODEL ARCHITECTURE
        # DETAILS" của bài β‑VAE — một bản vá tự tạo ra chỗ mù mới ngay lượt sau.
        v = re.sub(r"(?<=[a-z]\.)\d{1,2}(?=\s)", "", v)
        trang.append(v)
    return "\n".join(trang)


def tim_pdf(than):
    """Dò tên file PDF từ chính mục tài liệu tham khảo của bài."""
    for m in re.finditer(r"_papers/(.+?\.pdf)", than.replace("\n", " ")):
        ten = re.sub(r"\s+", " ", m.group(1)).strip()
        if os.path.exists(os.path.join(PAPERS, ten)):
            return ten
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bai", required=True)
    # MOT BAI CO THE TRICH TU NHIEU BAI BAO. Bai SAGAN la bai dau tien: bai bao
    # goc KHONG in kien truc, no noi "similar to (Miyato & Koyama, 2018)", nen
    # phan RUOT theo Luat 6 nam o BA PDF. Truyen --pdf nhieu lan thi moi doan
    # trich duoc hoi tren HOP cua ca ba kho text.
    #
    # Khong doi hanh vi cua bat ky bai nao khac: khi khong truyen --pdf, tim_pdf
    # van tra ve DUNG MOT file nhu truoc — nen ban kinh cua thay doi nay bang 0
    # cho moi bai hien co.
    ap.add_argument("--pdf", action="append")
    # LỚP CHỮ VỠ NHƯNG CÓ BẢN OCR. Bài báo Williams 1992 (REINFORCE) dùng font Type3
    # không có bảng ToUnicode: lớp chữ đầy ký tự rác, mật độ cao nên phép dò bản quét
    # KHÔNG nổ và mọi đoạn trích bị báo sai. --txt nhận một file văn bản (đầu ra OCR,
    # ví dụ tools/_work_rf/w92_plain.txt) làm kho đối chiếu THAY cho lớp chữ của PDF.
    # Bán kính bằng 0 cho mọi bài khác: không truyền --txt thì hành vi y như cũ.
    ap.add_argument("--txt", action="append")
    ap.add_argument("--khe", type=int, default=400,
                    help="số ký tự tối đa cho mỗi chỗ lược bớt (mặc định 400)")
    a = ap.parse_args()

    p = os.path.join(DB, a.bai + ".md")
    if not os.path.exists(p):
        sys.exit("Không có bài %s" % p)
    than = io.open(p, encoding="utf-8").read()
    # CAT KHOI MA TRUOC KHI QUET DAU TRICH. Mau `*"..."*` khong phan biet duoc
    # mot doan trich voi hai dau nhay lac trong Python: `plt.plot(..., "*",
    # color="#7c3aed", ...)` cho ra mot cap `*"` ... `"*` hoan hao, va bai ProGAN
    # bi bao SAI vi mot dau cham hinh sao cua matplotlib.
    #
    # Do ban kinh HAI CHIEU truoc khi va (luat Dot 33), tren toan database:
    #   MAT  2 doan o 2 file — ca hai deu la manh MA (Micrograd, ProGAN). Dung.
    #   THEM 4 doan o 4 file (Adadelta, Adam, FCOS, SGD) — ca bon deu la trich
    #        NGUYEN VAN that, truoc day bi mot span vat qua khoi ma nuot mat.
    # Nen ban va vua bo bao nham vua THEM doan duoc kiem; khong co chieu nao xau.
    than_quet = re.sub(r"```.*?```", "\n", than, flags=re.S)

    tens = a.pdf or ([] if a.txt else ([tim_pdf(than)] if tim_pdf(than) else []))
    if not tens and not a.txt:
        sys.exit("Không dò được PDF từ mục tham khảo — truyền --pdf")
    print("bài: %s" % a.bai)
    khos = []
    for ten in tens:
        pp = os.path.join(PAPERS, ten)
        if not os.path.exists(pp):
            sys.exit("Không có PDF: %s" % pp)
        print("PDF: %s" % ten)
        tho = doc_pdf(pp)
        if tho.count("\u21b5") >= 20:
            global GHEP_CHU
            GHEP_CHU = True
            print("  -> lớp text dùng bộ mã chữ ghép hỏng (%d ký tự ↵): bỏ cụm ff/fi/ffi ở CẢ HAI vế"
                  % tho.count("\u21b5"))
        khos.append(chuan(tho))
    for ten in (a.txt or []):
        print("TXT (kho OCR thay lớp chữ PDF): %s" % ten)
        khos.append(chuan(io.open(ten, encoding="utf-8").read()))
    print("-" * 92)

    kho = "\n@@@\n".join(khos)

    # BẢN QUÉT: lớp text rỗng hoặc vỡ. Không có bản vá nào cứu được — mọi đoạn
    # trích sẽ bị báo "KHÔNG KHỚP" hàng loạt, và một danh sách sai dài đúng bằng
    # một danh sách thiếu thật. Phải BÁO RA rằng không kiểm được, đừng báo đỏ.
    # Gặp lần đầu ở Huber 1964 (bản quét JSTOR, 28/29 trang có 0 ký tự) — cùng
    # lớp với Rumelhart 1986 và Pearson 1901 đã ghi trong CLAUDE.md. Thêm 2026-09-20.
    # Phép đo phải theo TỪNG TRANG, không theo trung bình cả file. Bản quét JSTOR
    # của Koenker & Bassett 1978 có 18/18 trang thân bài rỗng nhưng kèm một trang
    # bìa và năm trang "linked citations" CÓ lớp text, nên trung bình lên 360,2 ký
    # tự mỗi trang và phép dò theo trung bình KHÔNG nổ. Đếm số trang rỗng thì nổ
    # đúng. Sửa 2026-09-21; bán kính hai chiều đo ở docstring.
    _trang = []
    for t in tens:
        _d = pdfium.PdfDocument(os.path.join(PAPERS, t))
        for _i in range(len(_d)):
            _tx = _d[_i].get_textpage().get_text_range()
            _trang.append(len(re.sub(r"\s", "", _tx)))
    _so_trang = len(_trang)
    _mat_do = sum(_trang) / max(_so_trang, 1)
    _rong = sum(1 for _c in _trang if _c < 100)
    if tens and (_mat_do < 100 or _rong * 2 >= _so_trang):
        print("LỚP TEXT CỦA PDF RỖNG HOẶC VỠ — %d/%d trang có dưới 100 ký tự;"
              " trung bình %.1f ký tự mỗi trang."
              % (_rong, _so_trang, _mat_do))
        print("Đây là BẢN QUÉT: không có gì để đối chiếu, nên script KHÔNG kiểm được")
        print("đoạn trích nào. Con số dưới đây là số đoạn TÌM THẤY, không phải số đoạn")
        print("đã kiểm. Phải soát tay bằng ảnh render (pdffig.py / pdfsheet.py) và ghi")
        print("rõ hạn chế đó trong checklist của bài.")
        print("-" * 92)
        _n = len(re.findall(r'\*"(.+?)"\*', than_quet, re.S))
        print("%d đoạn trích tiếng Anh TÌM THẤY · 0 đã kiểm · 0 không khớp"
              " (không kiểm được)" % _n)
        return
    # Ban CHI CON TU cua kho, de so cum bon tu: hai ve phai tach tu giong nhau,
    # neu khong thi "lsun 31 and" khong khop "LSUN [31] and" cua PDF.
    kho_tu = " " + " ".join(re.findall(r"[a-z0-9]+", kho)) + " "
    tong = xau = bo = long = thutu = 0
    for m in re.finditer(r'\*"(.+?)"\*', than_quet, re.S):
        q = m.group(1)
        # Lớp ký tự này phải ĐỦ mọi dấu tiếng Việt. Bản đầu thiếu các chữ có hai
        # dấu (ấ, ẫ, ộ, ượ…), nên một ô bảng như "lấy mẫu - no difficulties" lọt
        # qua bộ lọc rồi bị đem đi so với bài báo tiếng Anh.
        if re.search("[àáảãạăằắẳẵặâầấẩẫậđèéẻẽẹêềếểễệìíỉĩị"
                     "òóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵ]", q.lower()):
            bo += 1            # câu dịch hoặc câu của người viết, không phải trích
            continue
        q2 = chuan(re.sub(r"^> ?", "", q, flags=re.M))
        q2 = re.sub(r"\$[^$]*\$", MOC, q2)
        q2 = q2.replace("...", MOC)
        q2 = re.sub(r"\s+", " ", q2).strip()
        if len(q2.replace(MOC, "")) < 25:
            bo += 1
            continue
        tong += 1
        phan = [x.strip() for x in q2.split(MOC) if x.strip()]
        mau = (".{0,%d}?" % a.khe).join(re.escape(x) for x in phan)
        if re.search(mau, kho, re.S):
            continue
        # Phep so CHAT that bai. Thu lai theo TUNG TU, cho phep khe nho giua cac tu:
        # lop text cua PDF chen NHAN HINH va CHU THICH CHAN TRANG vao giua cau theo
        # thu tu doc, va dinh so hieu chu thich vao chinh tu ("DCGANs1 using"). Hai
        # thu do khong phai loi trich dan, nen phai tach ra khoi loi that.
        # Phep do CHAT hon mot mau long: cat doan trich thanh cac cum BON TU lien
        # tiep roi hoi tung cum co trong PDF khong. Chen nhan hinh vao giua cau lam
        # HONG dung mot vai cum tai cho chen; viet sai mot chu lam hong MOI cum chua
        # chu do. Hai truong hop vi the phan biet duoc bang ti le cum thieu.
        # Cụm bốn từ dựng THEO TỪNG MẢNH, không bắc qua chỗ lược bớt hay chỗ có công
        # thức: bắc qua thì sinh ra những cụm chưa bao giờ tồn tại ("following dcgans
        # for adam" ở chỗ bài báo viết "Following DCGANs, β1 for Adam").
        cum = []
        for mieng in phan:
            tu = re.findall(r"[a-z0-9]+", mieng)
            cum += [" ".join(tu[i:i + 4]) for i in range(max(0, len(tu) - 3))]
        thieu = [c for c in cum if (" " + c + " ") not in kho_tu]
        if cum and len(thieu) <= max(2, len(cum) // 12):
            long += 1
            print("KHỚP THEO CỤM (%d/%d cụm bốn từ có trong PDF — lớp text chèn"
                  " nhãn hình hoặc số hiệu chú thích vào giữa):"
                  % (len(cum) - len(thieu), len(cum)))
            print("   %s" % q2[:100])
            for c in thieu:
                print("   hụt tại: %s" % c)
            continue
        # TẦNG BA. Câu NGẮN mà lớp text chèn nhãn hình vào giữa thì tầng cụm bốn
        # từ cũng hỏng: câu "two stride=1 deconvolutional layers are added" của
        # bài LSGAN chỉ có bốn cụm, và 350 ký tự nhãn hình chen vào phá ba trong
        # bốn. Tầng này hỏi câu yếu hơn hẳn — mọi TỪ có xuất hiện ĐÚNG THỨ TỰ,
        # mỗi khe không quá `--khe` ký tự — nên nó phải được ĐẾM RIÊNG, đừng gộp
        # vào cột khớp.
        tu = re.findall(r"[a-z0-9]+", q2.replace(MOC, " "))
        mau_tu = (".{0,%d}?" % a.khe).join(re.escape(w) for w in tu)
        mt = re.search(mau_tu, kho, re.S) if tu else None
        if mt:
            # PHẢI báo ra ĐỘ DÀI phần chèn, đừng chỉ báo "khớp". Hai trường hợp
            # rất khác nhau cùng lọt qua tầng này:
            #   · chèn LỚN (hàng trăm ký tự) = nhãn hình xen vào giữa câu theo thứ
            #     tự đọc của lớp text — bài viết trích ĐÚNG;
            #   · chèn NHỎ (chục ký tự) = bài viết BỎ MẤT một cụm của bài báo, hay
            #     gặp nhất là một dẫn chiếu "(15, 16)" — bài viết trích SAI.
            # Bản đầu của tầng này gộp cả hai và vì thế nuốt mất một lỗi thật ở bài
            # `Autoencoder`. Con số dưới đây là thứ phân biệt chúng.
            chen = len(mt.group(0)) - len("".join(tu)) - (len(tu) - 1)
            thutu += 1
            print("KHỚP THEO THỨ TỰ TỪ — chèn thêm %d ký tự (%s):"
                  % (chen, "nhãn hình xen giữa câu" if chen >= 60
                     else "ĐỌC TAY: chèn ngắn, nhiều khả năng bài viết BỎ SÓT"))
            print("   %s" % q2[:100])
            continue
        xau += 1
        print("KHÔNG KHỚP:")
        for x in phan:
            print("   %s %s" % ("có " if x in kho else "SAI", x[:110]))
        for c in thieu[:6]:
            print("      cụm bốn từ KHÔNG có trong PDF: %s" % c)
    print("-" * 92)
    print("%d đoạn trích tiếng Anh đã kiểm · %d bỏ qua (câu dịch, đoạn quá ngắn) · "
          "%d khớp theo cụm · %d khớp theo thứ tự · %d KHÔNG khớp"
          % (tong, bo, long, thutu, xau))
    if xau:
        print()
        print("Dòng 'SAI' là đoạn KHÔNG có trong PDF. Dòng 'có' mà cả cụm vẫn không")
        print("khớp nghĩa là hai mảnh đều có nhưng cách nhau xa hơn %d ký tự — nới"
              % a.khe)
        print("`--khe` rồi đọc lại, đừng sửa bài trước khi biết là chỗ nào.")


if __name__ == "__main__":
    main()
