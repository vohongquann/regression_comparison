"""Chạy:  python -m src.compare [--data diabetes california] [--quad] [--repeats 3] [--only Ridge Lasso]

CV lặp bên ngoài (RepeatedKFold 5 fold) trên cùng các fold cho mọi thuật toán, nên sai khác
giữa hai thuật toán là sai khác ghép cặp. Kết quả ghi ra results/<data>[_quad]_metrics.csv,
_folds.csv và .png.
"""
import argparse
import time
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import RepeatedKFold

from .data import DATASETS, load
from .models import NAMES, build, chosen_of, coef_of

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results"


def evaluate(data, names, repeats, quad, seed=42):
    X, y = load(data)
    folds = list(RepeatedKFold(n_splits=5, n_repeats=repeats, random_state=seed).split(X))
    rows = []
    for name in names:
        proto = build(name, quad)
        for k, (tr, te) in enumerate(folds):
            m = clone(proto)
            t0 = time.perf_counter()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m.fit(X.iloc[tr], y.iloc[tr])
            fit_s = time.perf_counter() - t0
            pred = m.predict(X.iloc[te])
            coef = coef_of(m)
            rows.append(dict(model=name, fold=k, rmse=root_mean_squared_error(y.iloc[te], pred),
                             mae=mean_absolute_error(y.iloc[te], pred), r2=r2_score(y.iloc[te], pred),
                             fit_s=fit_s, n_coef=coef.size, nonzero=int((np.abs(coef) > 1e-8).sum()),
                             max_abs_coef=float(np.abs(coef).max()), chosen=str(chosen_of(m))))
        r = pd.DataFrame(rows).query("model == @name")
        print(f"  {name:11s} RMSE {r.rmse.mean():9.4f} ± {r.rmse.std():7.4f}   R2 {r.r2.mean():7.4f}   "
              f"nonzero {r.nonzero.mean():5.1f}/{r.n_coef.iloc[0]}   {r.fit_s.mean():6.2f}s/fit", flush=True)
    return pd.DataFrame(rows)


def summarize(folds):
    s = folds.groupby("model", sort=False).agg(
        rmse=("rmse", "mean"), rmse_std=("rmse", "std"), mae=("mae", "mean"), r2=("r2", "mean"),
        fit_s=("fit_s", "mean"), n_coef=("n_coef", "first"), nonzero=("nonzero", "mean"),
        max_abs_coef=("max_abs_coef", "median"))
    return s.sort_values("rmse")


def plot(folds, summary, title, path):
    order = list(summary.index)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    ax[0].boxplot([folds.query("model == @m").rmse for m in order], tick_labels=order)
    ax[0].set_ylabel("RMSE mỗi fold (thấp hơn là tốt hơn)")
    ax[0].set_title(f"{title} — sai số CV")
    ax[1].bar(order, summary.nonzero, color="tab:orange")
    ax[1].set_ylabel("số hệ số khác 0 (trung bình các fold)")
    ax[1].set_title("Độ thưa")
    for a in ax:
        a.tick_params(axis="x", rotation=45)
        a.grid(axis="y", alpha=.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", choices=DATASETS, default=DATASETS)
    ap.add_argument("--quad", action="store_true", help="cho mọi thuật toán chạy trên cột nở bậc 2")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--only", nargs="+", choices=NAMES)
    a = ap.parse_args()
    names = a.only or [n for n in NAMES if not (a.quad and n == "Polynomial")]  # Polynomial ≡ Linear khi quad
    OUT.mkdir(exist_ok=True)
    for data in a.data:
        tag = data + ("_quad" if a.quad else "")
        print(f"== {tag}")
        folds = evaluate(data, names, a.repeats, a.quad)
        summary = summarize(folds)
        folds.to_csv(OUT / f"{tag}_folds.csv", index=False)
        summary.round(5).to_csv(OUT / f"{tag}_metrics.csv")
        plot(folds, summary, tag, OUT / f"{tag}.png")
        print(summary.round(4).to_string(), "\n")


if __name__ == "__main__":
    main()
