# -*- coding: utf-8 -*-
"""Doi chieu MOI con so trong khoi SS cua 9 bai GD1 voi dau ra that cua v_gd1ss.py.

CLAUDE.md 2.1: so trong bai phai den tu code da chay. Script nay CHUNG MINH dieu do,
thay vi de nguoi doc tin loi. Bat ca truong hop go tay nham (da xay ra mot lan
o bai Lasso, xem bao cao dot).
"""
import io
import os
import re
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = HERE
DB = os.path.join(os.path.dirname(HERE), "AI algorithms", ".flat")

SS = ""
for _s in ("v_gd1ss.py", "v_enet.py", "v_lars.py", "v_huber.py", "v_quantile.py",
           "v_bayes.py"):
    # v_enet.py la nguon khoi SS bai Elastic Net, v_lars.py la nguon bai LARS,
    # v_huber.py la nguon bai Huber Regression, v_quantile.py la nguon bai Quantile
    # Regression — ca bon sau khi viet lai theo Luat 6 (2026-09-19, 2026-09-20,
    # 2026-09-21); bon bai do khong con lay so tu v_gd1ss.py.
    # v_quantile.py chay khoang 257 giay. CO Y khong khai cac con so THOI GIAN cua
    # muc 18b bai Quantile: chung khong tai lap duoc giua cac lan chay, va mot dong
    # khai nhu the chi la mot phep kiem xanh gia dua vao cache (luat Dot 23).
    # Ghi chu: v_lars.py chay khoang 23 giay, nen ca script nay cham theo.
    out = subprocess.run([sys.executable, _s], cwd=TOOLS,
                         capture_output=True, text=True, encoding="utf-8")
    assert out.returncode == 0, (_s, out.stderr[-2000:])
    SS += out.stdout

# Moi dong: (ten bai, chuoi trong BAI, chuoi phai co trong DAU RA script)
KIEM = [
    ("Machine Learning (Introduction)", "95{,}8\\%", "95.8%"),
    ("Machine Learning (Introduction)", "61{,}2\\%", "61.2%"),
    ("Machine Learning (Introduction)", "80{,}4\\%", "80.4%"),
    ("Machine Learning (Introduction)", "34,6", None),          # 95.8 - 61.2
    ("Machine Learning (Introduction)", "19,2", "19.2"),
    ("Polynomial Regression", "0,2962", "0.2962"),
    ("Polynomial Regression", "0,0937", "0.0937"),
    ("Polynomial Regression", "3,2 lần", "3.2 lần"),
    ("Polynomial Regression", "0,3013", "0.3013"),
    ("Polynomial Regression", "0,2938", "0.2938"),
    ("Ridge Regression (Tikhonov Regularization)", "0,999939", "0.999939"),
    ("Ridge Regression (Tikhonov Regularization)", "14{,}27", "14.27"),
    ("Ridge Regression (Tikhonov Regularization)", "-39{,}09", "-39.09"),
    ("Ridge Regression (Tikhonov Regularization)", "+42{,}57", "+42.57"),
    ("Ridge Regression (Tikhonov Regularization)", "16{,}096", "16.096"),
    ("Ridge Regression (Tikhonov Regularization)", "0{,}097", "0.097"),
    ("Ridge Regression (Tikhonov Regularization)", "48%", "48%"),
    ("Ridge Regression (Tikhonov Regularization)", "99 lần", "99 lần"),
    ("Ridge Regression (Tikhonov Regularization)", "165 lần", "165.3 lần"),
    ("Lasso Regression", "5{,}91", "5.91"),
    ("Lasso Regression", "1{,}15\\cdot10^{18}", "1.15e+18"),
    ("Lasso Regression", "1{,}1\\cdot10^{12}", "1.1e+12"),
    # viet lai 2026-09-19: khoi SS nay dung so cua v_enet.py muc 4 (Bo de 2)
    ("Elastic Net", "11,654039", "11.654039"),
    ("Elastic Net", "116,7424224940", "116.7424224940"),
    ("Elastic Net", "3,705786", "3.705786"),
    ("Elastic Net", "3,777738", "3.777738"),
    # viet lai 2026-09-19: moi con so cua bai LARS lay tu v_lars.py
    ("Least Angle Regression (LARS)", "356 310", "356310"),
    ("Least Angle Regression (LARS)", "35 634", "35634"),
    ("Least Angle Regression (LARS)", "3 566", "3566"),
    ("Least Angle Regression (LARS)", "3459.9776", "3459.9776"),
    ("Least Angle Regression (LARS)", "107.1213", "107.1213"),
    ("Least Angle Regression (LARS)", "0.415630", "0.415630"),
    ("Least Angle Regression (LARS)", "1{,}49\\times10^{-12}", "1.49e-12"),
    ("Least Angle Regression (LARS)", "40,86", "40.86"),
    ("Least Angle Regression (LARS)", "0,85624153", "0.85624153"),
    ("Least Angle Regression (LARS)", "0,966422", "0.966422"),
    ("Least Angle Regression (LARS)", "0,153899", "0.153899"),
    ("Least Angle Regression (LARS)", "8.3768e+14", "8.3768e+14"),
    # Bai Huber Regression viet lai theo Luat 6 ngay 2026-09-20: moi con so khoi SS
    # nay sinh tu v_huber.py, khong con tu v_gd1ss.py. Ba dong cu ("1{,}74",
    # "+2{,}53", "16%") da bo — chung la so cua ban truoc, va dong "1{,}74" con
    # khop NHAM voi o "1,740" cua Bang I dung lai, tuc mot lan bao xanh gia.
    ("Huber Regression", "1{,}1762358737", "1.1762358737"),
    ("Huber Regression", "573{,}3022934887", "573.3022934887"),
    ("Huber Regression", "1{,}1010559667", "1.1010559667"),
    ("Huber Regression", "1{,}3449975085", "1.3449975085"),
    ("Huber Regression", "0{,}9500002597", "0.9500002597"),
    ("Huber Regression", "0{,}0579148930", "0.0579148930"),
    ("Huber Regression", "0{,}7101645483", "0.7101645483"),
    ("Huber Regression", "1{,}4081243600", "1.4081243600"),
    ("Huber Regression", "2{,}4546", "2.4546"),
    ("Huber Regression", "1{,}50055665", "1.50055665"),
    ("Huber Regression", "0{,}01336733", "0.01336733"),
    ("Huber Regression", "1{,}127172", "1.127172"),
    ("Huber Regression", "1{,}9681", "1.9681"),
    ("Huber Regression", "1{,}6960", "1.6960"),
    ("Huber Regression", "0{,}0000443698", "0.0000443698"),
    # Bai Quantile Regression viet lai theo Luat 6 ngay 2026-09-21: nguon doi tu
    # v_gd1ss.py sang v_quantile.py, va bon dong cu ("93{,}2\\%", "90{,}1\\%",
    # "79{,}8\\%", "89{,}4\\%") khong con con so nao trong bai.
    ("Quantile Regression", "0{,}887500", "0.887500"),
    ("Quantile Regression", "0{,}812500", "0.812500"),
    ("Quantile Regression", "4{,}2537123647", "4.2537123647"),
    ("Quantile Regression", "3{,}7242857143", "3.7242857143"),
    ("Quantile Regression", "0{,}7957459726", "0.7957459726"),
    ("Quantile Regression", "1{,}2566824519", "1.2566824519"),
    ("Quantile Regression", "0{,}3230731948", "0.3230731948"),
    ("Quantile Regression", "1{,}5707963268", "1.5707963268"),
    ("Quantile Regression", "2{,}0006", "2.0006"),
    ("Quantile Regression", "0{,}072600", "0.072600"),
    ("Quantile Regression", "0{,}630925", "0.630925"),
    ("Quantile Regression", "0{,}3265748385", "0.3265748385"),
    ("Quantile Regression", "0{,}2317138908", "0.2317138908"),
    # Con so nay nam NGOAI cong thuc toan trong bai nen viet dau phay thuong,
    # khong phai "{,}" — day la mot cho de sai giua hai quy uoc viet so.
    ("Quantile Regression", "1,160629", "1.160629"),
    ("Quantile Regression", "7{,}855541", "7.855541"),
    ("Quantile Regression", "61{,}8568301855", "61.8568301855"),
    ("Quantile Regression", "1{,}9706585265", "1.9706585265"),
    ("Quantile Regression", "0{,}8571428571", "0.8571428571"),
    # viet lai 2026-09-22 theo Luat 6: khoi SS nay dung so cua v_bayes.py muc 21,
    # do tren chinh 10 diem du lieu cua muc 1.1 sach Bishop.
    ("Bayesian Linear Regression", "1,291709", "1.291709"),
    ("Bayesian Linear Regression", "0,501726", "0.501726"),
    ("Bayesian Linear Regression", "0,9596", "0.9596"),
    ("Bayesian Linear Regression", "2,9511", "2.9511"),
    ("Bayesian Linear Regression", "29,4 lần", "29.4 lần"),
    ("Bayesian Linear Regression", "0,331901", "0.331901"),
    ("Bayesian Linear Regression", "1,1231", "1.1231"),
    ("Bayesian Linear Regression", "0,565684", "0.565684"),
    ("Bayesian Linear Regression", "0,398823", "0.398823"),
    ("Bayesian Linear Regression", "0,393342", "0.393342"),
    # Cho ngoai suy: bai GIU NGUYEN ket qua nguoc voi ky vong — ban Bayes te hon
    # binh phuong toi thieu tai x = 1,3, va khoang 95 % KHONG chua gia tri that.
    ("Bayesian Linear Regression", "0{,}951057", "0.951057"),
    ("Bayesian Linear Regression", "0,194356", "0.194356"),
    ("Bayesian Linear Regression", "0,807115", "0.807115"),
    ("Bayesian Linear Regression", "4,1528 lần", "4.1528 lần"),
    ("Bayesian Linear Regression", "0,077127", "0.077127"),
    # Cac con so cua muc 24 — chinh la phat hien chinh cua bai — cung phai kiem.
    ("Bayesian Linear Regression", "0,917810", "0.917810"),
    # con so nay LUON nam trong $...$ nen bai viet `0{,}918939`; con 0,917810 thi
    # xuat hien o CA HAI dang. Cung khuon lech quy uoc da ghi o bai Quantile.
    ("Bayesian Linear Regression", "0{,}918939", "0.918939"),
    ("Bayesian Linear Regression", "0,012553", "0.012553"),
    ("Bayesian Linear Regression", "12,75 lần", "12.75 lần"),
]

doc = {}
loi = 0
for bai, trong_bai, trong_script in KIEM:
    if bai not in doc:
        doc[bai] = io.open(os.path.join(DB, bai + ".md"),
                           encoding="utf-8", errors="replace").read()
    a = trong_bai in doc[bai]
    b = True if trong_script is None else (trong_script in SS)
    ok = a and b
    if not ok:
        loi += 1
        print("  LỆCH  %-46s bài:%s  script:%s  (%r)"
              % (bai[:46], "✓" if a else "KHÔNG", "✓" if b else "KHÔNG", trong_bai))

print("\nĐối chiếu %d con số khối SS trên 9 bài: %d lệch" % (len(KIEM), loi))
if loi == 0:
    print("Mọi con số trích trong bài đều có mặt trong đầu ra thật của v_gd1ss.py.")
sys.exit(1 if loi else 0)
