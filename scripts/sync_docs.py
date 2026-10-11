"""Đồng bộ docs/ từ repo AI-Algorithm — một chiều, nguyên văn, chỉ 9 bài hồi quy của project này.

    python scripts/sync_docs.py            # chép file mới/đổi
    python scripts/sync_docs.py --dry-run  # chỉ liệt kê, không ghi
    python scripts/sync_docs.py --prune    # xoá luôn file trong docs/ mà nguồn không còn
    python scripts/sync_docs.py --md       # chỉ các file .md (nhanh khi chỉ sửa chữ)
    python scripts/sync_docs.py --src DIR  # đổi thư mục nguồn (hoặc đặt biến AI_ALGORITHM_DIR)

Ngoài 9 thư mục bài còn sinh docs/"Theo dõi chất lượng.md" — 9 dòng lọc từ bảng thẩm định của nguồn.

Sửa bài ở AI-Algorithm rồi chạy lệnh này; ĐỪNG sửa tay trong docs/ vì lần đồng bộ sau sẽ ghi đè.
"""
import argparse
import filecmp
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SRC = Path(os.environ.get("AI_ALGORITHM_DIR", Path.home() / "Documents/GitHub/AI-Algorithm"))
GROUP = Path("AI algorithms/Machine Learning")
ARTICLES = [
    "Linear Regression", "Polynomial Regression", "Ridge Regression (Tikhonov Regularization)",
    "Lasso Regression", "Elastic Net", "Least Angle Regression (LARS)", "Huber Regression",
    "Quantile Regression", "Bayesian Linear Regression",
]


def files(root: Path):
    return sorted(p.relative_to(root) for p in root.rglob("*") if p.is_file())


SHEET = "Theo dõi chất lượng.md"
LINK_PREFIX = "AI algorithms/Machine Learning/"


def quality_sheet(src_root: Path) -> str:
    """Bảng thẩm định của AI-Algorithm, lọc còn 9 dòng của project này.

    Giữ nguyên chữ của từng dòng; chỉ bỏ tiền tố đường dẫn để liên kết trỏ được vào docs/.
    Phần chú giải cột lấy nguyên từ đầu file nguồn.
    """
    lines = (src_root / SHEET).read_text(encoding="utf-8").split("\n")
    legend_start = next(i for i, l in enumerate(lines) if l.startswith("| Ký hiệu | Nghĩa |"))
    legend_end = next(i for i in range(legend_start, len(lines)) if lines[i].startswith("Quy tắc:"))
    gd1 = next(i for i, l in enumerate(lines) if l.startswith("## GĐ 1 "))
    header = next(i for i in range(gd1, len(lines)) if lines[i].startswith("| # | Bài |"))
    rows = {}
    for l in lines[header + 2:]:
        if not l.startswith("|"):
            break
        for art in ARTICLES:
            if f"[{art}](<{LINK_PREFIX}{art}/{art}.md>)" in l:
                rows[art] = l.replace(f"<{LINK_PREFIX}", "<")
    missing = [a for a in ARTICLES if a not in rows]
    if missing:
        raise SystemExit(f"bảng thẩm định không có dòng cho: {missing}")
    out = [
        "# Theo dõi chất lượng — 9 bài hồi quy", "",
        f"> Trích từ `AI-Algorithm/{SHEET}` (nhóm GĐ 1), chỉ 9 dòng của project này. Chữ của từng dòng giữ nguyên,",
        "> chỉ đổi đường dẫn bài thành tương đối so với thư mục này. Sinh bởi `scripts/sync_docs.py` — đừng sửa tay.", "",
        *lines[legend_start:legend_end + 1], "",
        *lines[header:header + 2],
        *[rows[a] for a in ARTICLES],
    ]
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--prune", action="store_true")
    ap.add_argument("--md", action="store_true")
    a = ap.parse_args()

    base = a.src / GROUP
    if not base.is_dir():
        raise SystemExit(f"không thấy {base} — dùng --src hoặc AI_ALGORITHM_DIR")
    new = upd = same = gone = 0
    for art in ARTICLES:
        s, d = base / art, ROOT / "docs" / art
        if not s.is_dir():
            print(f"THIẾU nguồn: {s}")
            continue
        for rel in files(s):
            if a.md and rel.suffix != ".md":
                continue
            src, dst = s / rel, d / rel
            if dst.exists() and filecmp.cmp(src, dst, shallow=False):
                same += 1
                continue
            if dst.exists():
                upd += 1
            else:
                new += 1
            print(("sửa  " if dst.exists() else "mới  ") + f"{art}/{rel}")
            if not a.dry_run:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
        if a.prune and d.is_dir():
            keep = set(files(s))
            for rel in files(d):
                if rel not in keep and not (a.md and rel.suffix != ".md"):
                    gone += 1
                    print(f"xoá  {art}/{rel}")
                    if not a.dry_run:
                        (d / rel).unlink()
    sheet, dst = quality_sheet(a.src), ROOT / "docs" / SHEET
    if not dst.exists() or dst.read_text(encoding="utf-8") != sheet:
        print(("sửa  " if dst.exists() else "mới  ") + SHEET)
        upd, new = upd + dst.exists(), new + (not dst.exists())
        if not a.dry_run:
            dst.write_text(sheet, encoding="utf-8")
    else:
        same += 1
    tag = " (dry-run)" if a.dry_run else ""
    print(f"\nmới {new} · sửa {upd} · giữ nguyên {same} · xoá {gone}{tag}")


if __name__ == "__main__":
    main()
