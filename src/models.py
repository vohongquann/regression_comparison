"""9 thuật toán cần so sánh. Mỗi mục là một pipeline đã gồm cả phần tự chọn siêu tham số
(CV bên trong), nên khi đem đi CV bên ngoài thì phép đo là nested — không rò nhãn."""
import numpy as np
from sklearn.linear_model import (BayesianRidge, ElasticNetCV, HuberRegressor, LarsCV,
                                  LassoCV, LinearRegression, QuantileRegressor, RidgeCV)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

CV = 5
ALPHAS = np.logspace(-4, 1, 40)

NAMES = ["Linear", "Polynomial", "Ridge", "Lasso", "ElasticNet", "LARS", "Huber", "Quantile", "Bayesian"]


def _regressor(name):
    return {
        "Linear": lambda: LinearRegression(),
        "Polynomial": lambda: LinearRegression(),  # khác Linear ở chỗ đầu vào đã nở bậc 2
        "Ridge": lambda: RidgeCV(alphas=np.logspace(-3, 3, 40)),
        "Lasso": lambda: LassoCV(alphas=ALPHAS, cv=CV, max_iter=100_000, random_state=0),
        "ElasticNet": lambda: ElasticNetCV(l1_ratio=[.1, .3, .5, .7, .9], alphas=ALPHAS, cv=CV,
                                           max_iter=100_000, random_state=0),
        "LARS": lambda: LarsCV(cv=CV, max_n_alphas=500),
        "Huber": lambda: GridSearchCV(HuberRegressor(epsilon=1.35, max_iter=5000),
                                      {"alpha": [1e-4, 1e-2, 1, 100]}, cv=CV,
                                      scoring="neg_root_mean_squared_error"),
        "Quantile": lambda: GridSearchCV(QuantileRegressor(quantile=0.5, solver="highs"),
                                         {"alpha": [0, 1e-3, 1e-2, 1e-1]}, cv=CV,
                                         scoring="neg_root_mean_squared_error"),
        "Bayesian": lambda: BayesianRidge(),
    }[name]()


def build(name, quad=False):
    """Pipeline chưa fit cho thuật toán `name`.

    quad=False: Polynomial tự nở bậc 2, tám thuật toán còn lại dùng cột gốc.
    quad=True : MỌI thuật toán dùng cột đã nở bậc 2 (thí nghiệm "chính quy hoá cứu đa thức").
    Chuẩn hoá trước và sau khi nở, vì mọi mô hình có phạt đều nhạy với thang đo.
    """
    steps = [("scale", StandardScaler())]
    if quad or name == "Polynomial":
        steps += [("poly", PolynomialFeatures(degree=2, include_bias=False)),
                  ("scale2", StandardScaler())]
    return Pipeline(steps + [("reg", _regressor(name))])


def coef_of(fitted):
    """Vector hệ số của bước hồi quy cuối, kể cả khi nó bọc trong GridSearchCV."""
    reg = fitted.named_steps["reg"]
    reg = getattr(reg, "best_estimator_", reg)
    return np.ravel(reg.coef_)


def chosen_of(fitted):
    """Siêu tham số mà CV bên trong đã chọn (để in cho dễ soi)."""
    reg = fitted.named_steps["reg"]
    if hasattr(reg, "best_params_"):
        return dict(reg.best_params_)
    out = {}
    for k in ("alpha_", "l1_ratio_", "alpha", "lambda_"):
        v = getattr(reg, k, None)
        if v is not None and np.ndim(v) == 0:
            out[k] = round(float(v), 6)
    return out
