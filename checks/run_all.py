"""Chạy các script kiểm chứng và đối chiếu đầu ra với bản đã lưu từ lần chạy gốc (_out_*.txt).

    python checks/run_all.py [tên ...]        # mặc định: tất cả

Mỗi script chạy ở một tiến trình riêng, song song. Ghi ra results/checks/<tên>.txt (stdout+stderr)
và results/checks/summary.csv. Cột `lech` = số dòng khác bản gốc sau khi bỏ dòng chỉ chứa số giây.
"""
import csv
import difflib
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "results" / "checks"
# script -> bản đầu ra gốc (nếu có)
SCRIPTS = {
    "v_linreg": "_out_linreg.txt", "v_poly": "_out_poly.txt", "v_ridge": "_out_ridge.txt",
    "v_lasso": "_out_lasso.txt", "v_enet": "_out_enet.txt", "v_lars": "_out_lars.txt",
    "v_huber": None, "v_quantile": None, "v_bayes": None,
    "v_gd1ss": "_out_gd1ss.txt", "v_reg": None,
}
TIMING = re.compile(r"\d+(?:[.,]\d+)?\s*(?:s|ms|giây)\b")


def norm(text):
    return [l.rstrip() for l in text.replace("\r\n", "\n").split("\n") if not TIMING.search(l)]


def run(name):
    t0 = time.perf_counter()
    p = subprocess.run([sys.executable, "-X", "utf8", f"{name}.py"], cwd=HERE, capture_output=True,
                       text=True, encoding="utf-8", errors="replace")
    dt = time.perf_counter() - t0
    text = p.stdout + (("\n--- stderr ---\n" + p.stderr) if p.stderr.strip() else "")
    (OUT / f"{name}.txt").write_text(text, encoding="utf-8")
    ref, lech = SCRIPTS[name], ""
    if ref:
        a, b = norm((HERE / ref).read_text(encoding="utf-8")), norm(p.stdout)
        lech = sum(1 for l in difflib.unified_diff(a, b, n=0, lineterm="") if l[:1] in "+-" and l[:3] not in ("+++", "---"))
    tail = (p.stderr.strip().splitlines() or [""])[-1][:160] if p.returncode else ""
    return dict(script=name, exit=p.returncode, seconds=round(dt, 1), lines=len(p.stdout.splitlines()),
                lech=lech, loi=tail)


def main():
    names = sys.argv[1:] or list(SCRIPTS)
    OUT.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as ex:
        rows = list(ex.map(run, names))
    with open(OUT / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"{r['script']:10s} exit={r['exit']} {r['seconds']:7.1f}s lines={r['lines']:5d} lech={r['lech']!s:5s} {r['loi']}")


if __name__ == "__main__":
    main()
