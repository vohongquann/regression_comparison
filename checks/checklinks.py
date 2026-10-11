# -*- coding: utf-8 -*-
import io, os, re, sys, urllib.parse
sys.stdout.reconfigure(encoding='utf-8')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# dạng ](target)  và dạng ](<target có dấu cách>)
#
# Nhánh (?P<a>...) cho phép MỘT cấp ngoặc lồng bên trong đích. Trước Đợt 14 nó là
# [^)\s]+ nên mọi đường dẫn Notion có ngoặc trong tên thư mục -- ví dụ
# ![x](Scatterplot%20(Biểu%20đồ%20phân%20tán)/image.png) -- bị cắt ở dấu ) đầu tiên,
# thành một đích cụt không có đuôi file, rồi bị bỏ qua âm thầm. Tức là toàn bộ hình
# image*.png của những bài có ngoặc trong tiêu đề CHƯA BAO GIỜ được kiểm.
# Chú ý: lớp ký tự phải loại CẢ '(' — nếu chỉ loại ')' thì phần đầu tham lam nuốt
# luôn dấu '(' rồi dừng ở ')' đầu tiên, và nhánh ngoặc lồng không bao giờ chạy.
pat = re.compile(r'\]\((?:<(?P<b>[^>]+)>|(?P<a>[^()\s]*(?:\([^()\s]*\)[^()\s]*)*))\)')
EXT = ('.md', '.csv', '.png', '.pdf', '.jpg', '.jpeg', '.gif', '.svg',
       '.py', '.ipynb', '.txt', '.json', '/')
tot = bad = susp = 0
for dp, dn, fn in os.walk(ROOT):
    # bỏ qua thư mục không phải nội dung: .git, thư mục làm việc tạm của tools/
    if any(x in dp for x in ('.git', '_work', '__pycache__', '.flat')):
        continue
    for f in fn:
        if not f.endswith('.md'):
            continue
        p = os.path.join(dp, f)
        txt = io.open(p, encoding='utf-8', errors='replace').read()
        # bỏ fenced code block và inline code span -- ví dụ minh họa trong đó
        # không phải liên kết thật (âm tính giả gặp ở CLAUDE.md mục 4)
        txt = re.sub(r'```.*?```', '', txt, flags=re.S)
        txt = re.sub(r'`[^`\n]*`', '', txt)
        for m in pat.finditer(txt):
            t = m.group('b') or m.group('a')
            if re.match(r'^(https?|mailto|data:|#)', t) or not t:
                continue
            if os.path.isdir(os.path.join(dp, urllib.parse.unquote(t))):
                continue                      # trỏ tới thư mục, hợp lệ
            if not t.lower().endswith(EXT):
                # Âm tính giả đã ghi ở CLAUDE.md mục 4: trước đây mọi đích không
                # kết thúc bằng đuôi hợp lệ đều bị BỎ QUA ÂM THẦM, nên một liên kết
                # nội bộ quên .md không bị báo. Đợt 14 gặp một ca thật trong bài
                # "Seq2Seq with Attention". Giờ báo ra thay vì im lặng.
                susp += 1
                fix = os.path.exists(os.path.join(dp, urllib.parse.unquote(t) + '.md'))
                print("NGHI NGỜ %-40s -> %s%s" % (
                    f[:40], urllib.parse.unquote(t)[:80],
                    "   (thêm '.md' là đúng)" if fix else ""))
                continue
            tot += 1
            if not os.path.exists(os.path.join(dp, urllib.parse.unquote(t))):
                bad += 1
                print("BROKEN  %-42s -> %s" % (f[:42], urllib.parse.unquote(t)[:95]))
print("internal links: %d   broken: %d   nghi ngờ (không có đuôi file): %d"
      % (tot, bad, susp))
