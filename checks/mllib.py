# -*- coding: utf-8 -*-
"""Dữ liệu + tiện ích dùng chung cho toàn bộ Đợt 2 (GĐ 1–3).

Cùng nguyên tắc như optlib.py của Đợt 1: mỗi bộ dữ liệu và mỗi hàm đo chỉ có MỘT
cài đặt, nên con số trong bài và con số trong hình luôn từ cùng một nguồn.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys

# stdout mặc định của Windows là cp1252, không in nổi tiếng Việt có dấu.
# Thiếu dòng này thì script GÃY ở lệnh print đầu tiên. Thêm Đợt 25.
sys.stdout.reconfigure(encoding="utf-8")

DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                  "AI algorithms", ".flat")

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False,
})

C = {
    "OLS":       "#6b7280",
    "Ridge":     "#2563eb",
    "Lasso":     "#dc2626",
    "ElasticNet": "#7c3aed",
    "LARS":      "#0891b2",
    "Huber":     "#d97706",
    "Quantile":  "#be185d",
    "Bayes":     "#059669",
    "LogReg":    "#2563eb",
    "LDA":       "#059669",
    "QDA":       "#d97706",
    "KNN":       "#7c3aed",
    "NB":        "#be185d",
    "SVM":       "#dc2626",
    "Tree":      "#111827",
    "Bagging":   "#0891b2",
    "RF":        "#059669",
    "ExtraTrees": "#4d7c0f",
    "AdaBoost":  "#d97706",
    "GBM":       "#dc2626",
    "XGBoost":   "#2563eb",
    "LightGBM":  "#7c3aed",
    "CatBoost":  "#ea580c",
    "Stacking":  "#111827",
}


def save(fig, folder, name):
    d = os.path.join(DB, folder)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, name)
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("  ->", folder + "/" + name, os.path.getsize(p), "B")


# ------------------------------------------------------------------ dữ liệu

def diabetes():
    """442 bệnh nhân, 10 đặc trưng — bộ dữ liệu của chính bài LARS (Efron 2004).

    Trả về X đã chuẩn hoá cột (trung bình 0, ‖cột‖₂ = 1) và y đã trừ trung bình,
    đúng quy ước Efron dùng ở mục 2 của bài báo.
    """
    from sklearn.datasets import load_diabetes
    d = load_diabetes()
    X = d.data.copy()
    X = X - X.mean(0)
    X = X / np.sqrt((X ** 2).sum(0))
    y = d.target - d.target.mean()
    return X, y, list(d.feature_names)


def make_reg(n=100, p=3, noise=1.0, seed=0, outliers=0, out_shift=20.0):
    """Hồi quy tuyến tính có nhiễu Gauss, tuỳ chọn thêm điểm ngoại lai."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    beta = np.arange(1, p + 1, dtype=float)
    y = X @ beta + noise * rng.normal(size=n)
    if outliers:
        idx = rng.choice(n, outliers, replace=False)
        y[idx] += out_shift
    return X, y, beta


def collinear(n=100, p=8, rho=0.99, noise=1.0, seed=0):
    """Ma trận thiết kế gần suy biến — bối cảnh Hoerl & Kennard 1970 nói tới."""
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, 2))
    X = np.zeros((n, p))
    for j in range(p):
        base = Z[:, j % 2]
        X[:, j] = rho * base + np.sqrt(1 - rho ** 2) * rng.normal(size=n)
    beta = np.zeros(p)
    beta[:2] = [3.0, -2.0]
    y = X @ beta + noise * rng.normal(size=n)
    return X, y, beta


def two_moons(n=400, noise=0.25, seed=0):
    from sklearn.datasets import make_moons
    return make_moons(n_samples=n, noise=noise, random_state=seed)


def gauss_blobs(n=400, seed=0, cov_a=None, cov_b=None, mu=2.0):
    """Hai lớp Gauss — dùng để tách bạch LDA (cùng hiệp phương sai) với QDA."""
    rng = np.random.default_rng(seed)
    A = np.array([[1.0, 0.0], [0.0, 1.0]]) if cov_a is None else np.asarray(cov_a)
    B = A if cov_b is None else np.asarray(cov_b)
    Xa = rng.multivariate_normal([-mu / 2, 0], A, n // 2)
    Xb = rng.multivariate_normal([+mu / 2, 0], B, n // 2)
    X = np.vstack([Xa, Xb])
    y = np.r_[np.zeros(n // 2, int), np.ones(n // 2, int)]
    return X, y


def adult(n_max=None):
    """Bộ Adult (thu nhập >50K) — có cột hạng mục thật, cần cho CatBoost."""
    from sklearn.datasets import fetch_openml
    d = fetch_openml("adult", version=2, as_frame=True, parser="auto")
    X, y = d.data, (d.target == ">50K").astype(int).to_numpy()
    if n_max:
        X, y = X.iloc[:n_max], y[:n_max]
    cat = [c for c in X.columns if str(X[c].dtype) in ("category", "object")]
    return X, y, cat


# ------------------------------------------------------------------- hàm đo

def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def acc(a, b):
    return float(np.mean(np.asarray(a) == np.asarray(b)))


def boundary(ax, clf, X, y, h=0.02, pad=0.6, cmap="RdBu", alpha=0.25, proba=False):
    """Vẽ biên quyết định của một bộ phân loại đã fit."""
    x0, x1 = X[:, 0].min() - pad, X[:, 0].max() + pad
    y0, y1 = X[:, 1].min() - pad, X[:, 1].max() + pad
    xx, yy = np.meshgrid(np.arange(x0, x1, h), np.arange(y0, y1, h))
    G = np.c_[xx.ravel(), yy.ravel()]
    if proba and hasattr(clf, "predict_proba"):
        Z = clf.predict_proba(G)[:, 1]
        ax.contourf(xx, yy, Z.reshape(xx.shape), levels=20, cmap=cmap, alpha=alpha)
        ax.contour(xx, yy, Z.reshape(xx.shape), levels=[0.5], colors="#111827", linewidths=1.1)
    else:
        Z = clf.predict(G).reshape(xx.shape)
        ax.contourf(xx, yy, Z, levels=[-.5, .5, 1.5], cmap=cmap, alpha=alpha)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, s=11, edgecolor="#111827", linewidth=.3)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
