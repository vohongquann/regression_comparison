# -*- coding: utf-8 -*-
"""Dò xem PDF paper có hình/bảng ở TRANG NÀO, để biết cắt trang nào bằng pdffig.py.

Vì sao cần: mục 2.3.1 bắt mọi bài kiến trúc phải có hình cắt từ paper gốc, và bước tốn
thời gian nhất không phải cắt mà là TÌM. Trước đây phải render mù từng trang một. Script
này đọc thẳng lớp text của PDF, bắt các dòng bắt đầu bằng "Figure N" / "Fig. N" /
"Table N" / "Algorithm N", rồi in ra số trang kèm dòng caption.

    python pdfscan.py "<paper.pdf>"                 # liệt kê mọi caption theo trang
    python pdfscan.py "<paper.pdf>" --page 5        # in toàn bộ text trang 5
    python pdfscan.py "<paper.pdf>" --grep margin   # chỉ dòng chứa từ khoá

Sau khi biết trang, xem khung bằng:  python pdffig.py "<paper.pdf>" <trang> --grid
"""
import argparse
import io
import os
import re
import sys

import pypdfium2 as pdfium

sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
PAPERS = os.path.join(os.path.dirname(HERE), "AI algorithms", "_papers")

CAP = re.compile(r"^\s*(figure|fig\.?|table|algorithm|alg\.?)\s*\.?\s*(\d+)", re.I)

ap = argparse.ArgumentParser()
ap.add_argument("pdf", help="đường dẫn PDF, hoặc chỉ tên file trong _papers/")
ap.add_argument("--page", type=int, help="in toàn bộ text của một trang")
ap.add_argument("--grep", help="chỉ in dòng chứa chuỗi này (không phân biệt hoa thường)")
ap.add_argument("--muc", action="store_true",
                help="liệt kê MỤC đánh số của bài báo, cho bảng đối chiếu của Luật 6")
a = ap.parse_args()

path = a.pdf if os.path.exists(a.pdf) else os.path.join(PAPERS, a.pdf)
if not os.path.exists(path):
    sys.exit("không thấy file: %s" % a.pdf)

# Mot file .pdf co the KHONG PHAI PDF. Do duoc ngay trong kho: file Korf 1985 dai
# 16 byte va chua dung chuoi "error code: 522" — mot trang loi Cloudflare luu nham
# duoi .pdf. Truoc ban va nay, pdfscan nem mot traceback cua pypdfium2, tuc bao ra
# dung thu nguoi doc khong hanh dong duoc. Luat Dot 14: cai gi bo qua duoc thi phai
# BAO RA, va bao bang cau nguoi doc hieu.
try:
    doc = pdfium.PdfDocument(path)
except Exception as e:
    co = os.path.getsize(path)
    dau = io.open(path, "rb").read(64)
    # Thong bao nay di ra STDERR, ma stderr KHONG duoc reconfigure sang utf-8 nhu
    # stdout — nen no phai thuan ASCII, khong thi Windows in ra dau hoi.
    sys.exit("KHONG DOC DUOC PDF: %s\n  co %d byte, 64 byte dau: %r\n  loi: %s\n"
             "  Neu co qua nho hoac 64 byte dau khong bat dau bang %%PDF thi day KHONG\n"
             "  phai mot file PDF - nhieu kha nang la mot trang loi tai ve luu nham ten."
             % (os.path.basename(path), co, dau, e))
print("%s — %d trang" % (os.path.basename(path), len(doc)))
print("-" * 78)

if a.page:
    print(doc[a.page - 1].get_textpage().get_text_range())
    sys.exit()

if a.muc:
    # Tieu de muc cua mot bai bao: mot dong bat dau bang so muc roi den chu HOA
    # hoac chu dau viet hoa. Bo qua dong qua dai (do la cau van, khong phai
    # tieu de) va dong toan so (bang bieu).
    # HAI kieu danh muc, va phai bat ca hai:
    #   (a) bai bao hien dai: "3.1 Residual Learning"
    #   (b) tap chi cu: mot dong CHU HOA, khong danh so
    #       ("THE ORGANIZATION OF A PERCEPTRON")
    #
    # CHO MU DA DO DUOC: mau (a) mot minh bat NHAM moi danh sach danh so trong
    # van xuoi. Rosenblatt 1958 cho 18 dong, va doc tay thi CA 18 deu bao nham
    # — do la cac muc "1. Stimuli impinge on a retina..." cua mot danh sach.
    # Nen mau (a) doi tieu de phai NGAN va khong ket thuc bang tu noi.
    #
    # CHO MU THU HAI, do duoc ngay 2026-09-03 tren Kingma & Welling 2013:
    # (a) MUC1 doi moi TU cua tieu de nam trong [A-Za-z-], nen mot dau HAI CHAM
    #     lam truot ca dong — "3 Example: Variational Auto-Encoder" khong khop,
    #     va bai bao do bao 10 muc thay vi 11.
    # (b) PHU LUC danh bang CHU CAI ("A Visualisations", "C.1 Bernoulli MLP as
    #     decoder") khong co mau nao bat — chung VO HINH voi --muc. Day dung la
    #     lop loi da lam hong bai RegNet: buoc thu tu cua phep dung lai nam o
    #     Phu luc D, va bai viet ket luan "khong tai tao duoc" vi chua doc no.
    # MOT BAN VA DA THU VA BI CHINH PHEP DO BAC, ghi lai de khong ai thu lai:
    # lop [^.] loai MOI dau cham, ke ca dau cham KET THUC tieu de, nen
    # "5.2. Self-attention mechanism." cua SAGAN truot. Cho phep dung MOT dau
    # cham o cuoi (`([A-Z][^.]{2,60})\.?\s*$`) thi THEM 17 dong — va 12 trong
    # so do la BAI TAP cua Boyd ("3.19 Nonnegative weighted sums and
    # integrals.", "2.24 Supporting hyperplanes."), chi 5 la muc that. Siet
    # them "tieu de ngan <= 30 ky tu" cung khong cuu duoc: giu 1 muc that va 3
    # bai tap. Tieu de mot muc va tieu de mot bai tap KHONG khac nhau ve cau
    # truc, nen cho mu nay CO THAT ma KHONG va duoc — muc 5.2 cua SAGAN phai
    # ghi tay vao bang doi chieu.
    #
    # So muc CHINH phai >= 1. Bon dong "0.x" duy nhat trong 301 file deu la bao
    # nham, doc tay ca bon: "0.75 S@pot5" (rac cua lop text), "0.5 Single Model"
    # (mot dong bang), "0.12 Et[KLt]" (mot cong thuc), va mot chu thich hinh cua
    # DenseNet. Rang buoc nay giet ca bon.
    MUC = re.compile(r"^\s*([1-9]\d?(?:\.\d{1,2})+)\.?\s+([A-Z][^.]{2,60})\s*$")
    MUC1 = re.compile(r"^\s*(\d{1,2})\.?\s+([A-Z][A-Za-z:'’-]*(?:\s+[A-Za-z:'’-]+){0,6})\s*$")
    # CHO MU THU BA, do duoc ngay 2026-09-03 tren Radford 2015 (DCGAN): tran 6 TU
    # cat mat tieu de dai — "6 INVESTIGATING AND VISUALIZING THE INTERNALS OF THE
    # NETWORKS" co 8 tu, nen muc 6 VO HINH va bai bao bao 22 muc thay vi 23.
    #
    # HAI CACH VA DA THU VA BAC, do ban kinh truoc khi tin (luat Dot 21):
    #   (a) noi tran 6 -> 12 tu: THEM 148 dong o 54 file, doc tay thi gan nhu tat
    #       ca la DANH SACH DANH SO trong van xuoi ("1. Since only the frequent
    #       items will play a role in the"). Bo.
    #   (b) noi tran 12 tu KEM rang buoc THU TU 1,2,3... nhu PHULUC: them 53 dong
    #       nhung MAT 417 dong o 104 file — bo dem thu tu bi mot dong khop nham
    #       som lam lech, roi moi muc that phia sau deu truot. Bo.
    #
    # Cai tach duoc tieu de dai voi dong danh sach KHONG phai do dai ma la CHU
    # HOA: tieu de muc cua bao ICLR/NIPS in chu hoa nho, lop text tra ve toan hoa;
    # con dong danh sach la cau van thuong. Nen chi noi tran khi tieu de TOAN HOA.
    MUC1C = re.compile(r"^\s*(\d{1,2})\.?\s+([A-Z][A-Z0-9 :'’\-]{8,70})\s*$")
    # CHO MU THU TU, do duoc ngay 2026-09-12 tren Karras 2017 (ProGAN): tran 60
    # KY TU cua MUC cat mat tieu de MUC CON dai, va MUC1C — nhanh noi tran cho
    # tieu de toan hoa — chi nhan nhan KHONG CO DAU CHAM. Nen mot muc con toan
    # hoa dai hon 60 ky tu KHONG co duong nao khop: "6.1 IMPORTANCE OF INDIVIDUAL
    # CONTRIBUTIONS IN TERMS OF STATISTICAL SIMILARITY" (72 ky tu) vo hinh, va
    # bai bao bao 23 muc thay vi 25.
    #
    # Nhanh nay AN TOAN HON MUC1C chu khong nguy hiem hon, vi nhan CO DAU CHAM
    # ("6.1") dung dau dong hiem hon han mot so tran ("1.") trong van xuoi —
    # dung lop bao nham ma MUC1C phai dung rang buoc CHU HOA de giet. Do ban
    # kinh hai chieu: them 5 dong o 4 file (ProGAN, AdamW hai muc, DeBERTa,
    # DDIM), MAT 0.
    MUC2C = re.compile(r"^\s*(\d{1,2}(?:\.\d{1,2})+)\.?\s+([A-Z][A-Z0-9 :'’\-]{8,90})\s*$")
    # Phu luc: mot chu cai in hoa (co the kem .1) roi den tieu de.
    #
    # Mau nay MOT MINH bat nham rat nhieu: do bán kinh tren 290 PDF thi no them
    # 1 565 muc o 181 file, va rieng Boyd "Convex Optimization" them 302 — gan
    # nhu tat ca la dong CONG THUC ("T A1x <= 1 + 1", "W Ax + b"), vi chu chuyen
    # vi T hay mot ma tran W dung dau dong trong giong het nhan phu luc.
    # Khong co mau ky tu nao tach duoc hai loai do; thu tach bang "tieu de phai
    # toan chu" thi lai vut mat "B Solution of -D_KL(q||p), Gaussian case".
    #
    # Cai tach duoc chung KHONG nam trong dong, ma nam giua cac dong: PHU LUC
    # DANH THEO THU TU A, B, C... Nen chi nhan mot dong khi chu cai cua no dung
    # bang chu ky vong tiep theo (hoac bang chu hien tai, cho muc con C.1).
    # Voi rang buoc do, Boyd tu 302 xuong 3 — va ca 3 deu la phu luc that.
    PHULUC = re.compile(r"^\s*([A-Z](?:\.\d{1,2})?)\.?\s+([A-Z][^.]{2,60})\s*$")
    # Cung ngay, cung bai bao, mot chi em cua cho mu tren: PHULUC doi tieu de bat
    # dau bang [A-Z], nen "A.1 1024 x 1024 NETWORKS USED FOR CELEBA-HQ" truot vi
    # tieu de bat dau bang CHU SO. Va vi bo dem thu tu khi do ket o "A", muc con
    # A.2 van khop nhung A.1 thi mat — dung khuon "bo dem trang thai hong thi
    # hong lan" cua Dot 33, chi khac la o day no hong MOT phan.
    #
    # Chi noi cho nhan CO PHAN .N ("A.1"), khong noi cho nhan tran ("A"). Do la
    # cho phan biet: mot dong cong thuc bat dau bang "A 1024..." thi co that,
    # con "A.1 1024..." thi gan nhu khong. Do ban kinh hai chieu: +4 dong o 2
    # file (ProGAN, Imagen ba muc), MAT 0 — nho han han ban va cho nhan tran ma Dot 33 da BAC.
    PHULUCD = re.compile(r"^\s*([A-Z]\.\d{1,2})\.?\s+(\d[^.]{2,60})\s*$")
    # Lop [^.] o tren loai MOI dau cham, ke ca dau cham cua mot SO HIEU MUC nam
    # trong chinh tieu de phu luc: "A Proof of Lemma 5.1" cua bai bao InfoGAN
    # truot, va vi bo dem thu tu khi do ket o "A" nen B va C truot theo — mot bo
    # dem trang thai hong thi hong lan (bai hoc Dot 33). Do ban kinh: bai bao do
    # bao 10 muc thay vi 18.
    #
    # Ban va thu NHAT — cho phep moi dau cham nam giua hai chu so — da bi BAC
    # bang chinh phep do hai chieu: no nhan them 5 HANG BANG SO cua StyleGAN
    # ("A Baseline Progressive GAN [30] 7.79 8.04"), va chung nuot khe "A" nen
    # phu luc that "A The FFHQ dataset" BI MAT.
    #
    # Cai tach duoc hai loai: dau cham cua mot tieu de phu luc luon thuoc mot
    # DAN CHIEU CHEO ("Lemma 5.1", "Section 3.3"), khong bao gio la mot o du
    # lieu. Nen chuan hoa dong TRUOC khi khop — rut "Lemma 5.1" thanh "Lemma 5"
    # — thay vi noi lop ky tu. Do lai: +22 dong o 3 file, MAT 0, doc tay ca 22
    # deu la phu luc that.
    XREF = re.compile(r"\b(Section|Sec|Lemma|Equation|Eq|Theorem|Figure|Fig|"
                      r"Table|Appendix|Chapter|Definition|Corollary|"
                      r"Proposition|Remark|Algorithm|Assumption)\.?\s+"
                      r"(\d{1,2})(?:\.\d{1,2})+")
    # Chỗ mù thứ TƯ, đo trên bài báo CenterNet: phụ lục viết ĐỦ CHỮ "Appendix A:".
    # PHULUC ở trên chỉ bắt nhãn TRẦN ("A Model Architecture"), nên bài báo đó báo
    # 11 mục thay vì 18 — mất trọn 5 phụ lục, và đó là nơi Phụ lục C chứa toàn bộ
    # thí nghiệm va chạm tâm mà thân bài dựa vào.
    #
    # Đo bán kính HAI CHIỀU trên 299 PDF của `_papers/`: mẫu thô thêm 65 dòng ở 23
    # file, đọc tay thì 6 là dẫn chiếu trong văn xuôi ("Appendix A.2 for
    # reference.", "Appendix B.3)."). Cả 6 đều có tiêu đề bắt đầu bằng CHỮ THƯỜNG
    # hoặc CHỮ SỐ, nên ràng buộc "tiêu đề phải bắt đầu bằng chữ hoa" giết hết mà
    # không mất dòng nào. Một chỗ nữa — "Appendix E KKT conditions are given by" —
    # đã bị bộ lọc NOI sẵn có chặn, vì nó kết thúc bằng từ nối "by".
    # Tiêu đề bắt đầu bằng CHỮ SỐ vẫn hợp lệ ("Appendix B: 3D BBox Estimation
    # Details"), nên lớp đầu là [A-Z0-9]; chỉ CHỮ THƯỜNG mới là dấu hiệu văn xuôi.
    PLTEN = re.compile(r"^\s*Appendix\s+([A-Z](?:\.\d{1,2})?)\s*[:.]?\s+"
                       r"([A-Z0-9][^.]{2,60})\s*$")
    # Giu nguyen tieu de GOC de in ra, chi dung ban rut gon de khop.
    PLGOC = re.compile(r"^\s*(?:Appendix\s+)?[A-Z](?:\.\d{1,2})?[:.]?\s+(.*?)\s*$")
    pl_hien, pl_kv = "", "A"
    HOA = re.compile(r"^\s*([A-Z][A-Z \-,'&]{8,60})\s*$")
    NOI = ("the", "a", "an", "of", "and", "or", "in", "to", "is", "are", "which",
           "that", "for", "with", "by", "from", "this", "these", "such", "no")
    # CHO MU THU SAU, do duoc 2026-09-12 tren Zhang 2018 (SAGAN): mot tieu de VAT
    # QUA HAI DONG thi dong dau ket thuc bang tu noi, va bo loc NOI ngay duoi —
    # thu viet ra de loai cau van — giet dung no. Bai bao do bao 7 muc thay vi 11:
    #   "4. Techniques to Stabilize the Training of" + "GANs"
    #   "4.1. Spectral normalization for both generator and" + "discriminator"
    #   "4.2. Imbalanced learning rate for generator and" + "discriminator updates"
    # Bo loc NOI dang lam DUNG viec cua no; thu sai la tieu de bi CAT. Nen cach
    # chua khong phai noi bo loc ma la NOI LAI hai dong roi hoi lai. Rang buoc de
    # khong an cau van: chi noi khi dong dau DA khop mot mau tieu de, dong sau
    # ngan (<= 40 ky tu), khong chua dau cham, va phan noi lai khong con ket thuc
    # bang tu noi. Tran ky tu cua MUC noi rong rieng cho nhanh nay, vi "4.2" ghep
    # lai dai 64 ky tu — con 60 la tran viet cho tieu de MOT dong.
    def noi_dong(t, t2):
        # len(t2) >= 3: giet "5.11 Derive a dual problem for minimize" + "PN" cua
        # Boyd — mot manh cong thuc, khong phai phan con lai cua tieu de. Tieu de
        # that ngan nhat do duoc trong 301 file la "GANs" (4 ky tu).
        if not t2 or len(t2) > 40 or len(t2) < 3 or "." in t2 or not t2[:1].isalpha():
            return None
        g = t + " " + t2
        m2 = MUCN.match(g) or MUC1.match(g)
        if m2 and m2.group(2).split()[-1].lower() not in NOI:
            return m2
        return None

    # Do ban kinh HAI CHIEU tren 301 PDF (luat Dot 33), sau khi da siet:
    #   THEM 18 dong o 15 file — doc tay ca 18: 17 la muc THAT, 1 bao nham
    #        ("5.11 Derive a dual problem for minimize PN", mot BAI TAP cua Boyd).
    #   MAT  4 dong — ca 4 la bao nham "0.x" ke tren.
    #
    # Ban noi tran ky tu, CHI dung cho nhanh noi hai dong o tren. Tran 60 ky tu
    # cua MUC viet cho tieu de MOT dong; tieu de 4.2 cua SAGAN ghep lai dai 64.
    MUCN = re.compile(r"^\s*([1-9]\d?(?:\.\d{1,2})+)\.?\s+([A-Z][^.]{2,90})\s*$")
    thay_muc, hoa = [], []
    for i in range(len(doc)):
        dong = doc[i].get_textpage().get_text_range().splitlines()
        for _j, line in enumerate(dong):
            sau = dong[_j + 1].strip() if _j + 1 < len(dong) else ""
            t = line.strip()
            m = MUC.match(t) or MUC1.match(t)
            if not m:
                mc = MUC1C.match(t) or MUC2C.match(t)
                # Hai bao nham duy nhat cua MUC1C khi do ban kinh la NHAN TRUC
                # do thi: "1 O0 1000 10000 100000" va "10 M 30 M 100 M 300 M".
                # Ca hai khong co tu nao tu 3 CHU CAI tro len; moi tieu de that
                # thi co. Rang buoc do giet ca hai ma khong mat dong nao.
                if mc and re.search(r"[A-Z]{3,}", mc.group(2)):
                    m = mc
            if not m:
                # Bo qua TRANG 1: phu luc khong bao gio bat dau o trang dau, con
                # tieu de bai bao thi hay bat dau bang "A " — "A New Approach to
                # Self-Supervised Learning" cua BYOL la mot bao nham do duoc.
                tx = XREF.sub(lambda x: x.group(1) + " " + x.group(2), t)
                pt = PLTEN.match(tx) if i > 0 else None
                p = pt or (PHULUC.match(tx) or PHULUCD.match(tx) if i > 0 else None)
                if p:
                    ch = p.group(1)[0]
                    nhan = False
                    # Nhãn ĐỦ CHỮ "Appendix X" KHÔNG cần ràng buộc thứ tự: chính chữ
                    # "Appendix" đã là thứ phân biệt mà nhãn trần không có. Bỏ ràng
                    # buộc đó ở đây là cố ý, và nó chữa đúng lớp hỏng-lan của Đợt 33:
                    # "Appendix B: 3D BBox..." từng bị loại vì tiêu đề bắt đầu bằng
                    # chữ số, rồi C, D, E trượt theo vì bộ đếm kẹt ở "B".
                    if pt:
                        pl_hien = ch
                        pl_kv = chr(ord(ch) + 1)
                        nhan = True
                    elif ch == pl_kv:                    # phu luc tiep theo
                        pl_hien, pl_kv = ch, chr(ord(ch) + 1)
                        nhan = True
                    elif ch == pl_hien:                  # muc con: C.1 sau C
                        # Đã THỬ siết nhánh này thành 'nhãn phải có dấu chấm' để giết
                        # một dẫn chiếu trần ("Appendix E KKT conditions are given
                        # by"), và phép đo hai chiều BÁC: mất 40 dòng ở 20 file, trong
                        # đó Bellemare 2017 mất 10. Dẫn chiếu ấy vốn đã bị bộ lọc NOI
                        # chặn sẵn vì nó kết thúc bằng từ nối "by". Giữ nguyên.
                        nhan = True
                    if nhan:
                        g = PLGOC.match(t)
                        m = (p.group(1), g.group(1) if g else p.group(2))
            if m:
                # m la mot match, hoac mot cap (so hieu, tieu de) tu nhanh phu luc
                so, ten = m if isinstance(m, tuple) else (m.group(1), m.group(2))
                ten = ten.strip()
                # tieu de muc khong ket thuc bang tu noi — cau van thi co
                if ten and ten.split()[-1].lower() not in NOI and not ten.isdigit():
                    thay_muc.append((i + 1, so, ten))
                elif ten and not isinstance(m, tuple):
                    mn = noi_dong(t, sau)          # tieu de vat qua hai dong
                    if mn:
                        thay_muc.append((i + 1, mn.group(1), mn.group(2).strip()))
                continue
            h = HOA.match(t)
            if h and not any(c.isdigit() for c in t):
                hoa.append((i + 1, "—", h.group(1).strip()))
    if len(hoa) > len(thay_muc):
        print("  (bai bao dung TIEU DE CHU HOA, khong danh so muc)")
        thay_muc = hoa
    if not thay_muc:
        print("  KHONG doc duoc muc nao — lop text vo, hoac bai bao khong danh so muc.")
        print("  Phai liet ke tay: dung `pdfsheet.py` roi doc muc luc.")
    for tr, so, ten in thay_muc:
        print("  trang %-3d | %-8s %s" % (tr, so, ten[:80]))
    print("-" * 78)
    print("  %d muc danh so" % len(thay_muc))
    sys.exit()

hits = 0
for i in range(len(doc)):
    text = doc[i].get_textpage().get_text_range()
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if a.grep:
            if a.grep.lower() in line.lower():
                print("  trang %-3d | %s" % (i + 1, line[:110]))
                hits += 1
        elif CAP.match(line):
            print("  trang %-3d | %s" % (i + 1, line[:110]))
            hits += 1

print("-" * 78)
print("%d dòng khớp. Xem khung: python pdffig.py \"%s\" <trang> --grid"
      % (hits, os.path.basename(path)))
