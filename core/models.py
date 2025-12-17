# core/models.py
# Model training + loading + interval predictions.

import importlib
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from .config import (
    TOL_PCT,
    TOL_MIN_OP_HOURS,
)
from .features import build_training_data


@dataclass
class CatBoostCQRBundle:
    kind: str
    target: str
    version: str
    feature_names: List[str]
    cat_feature_names: List[str]
    pi_level: float
    alpha: float
    q_lo: float
    q_hi: float
    qhat: float
    model_lo: Any
    model_mid: Any
    model_hi: Any
    abs_err_calib_sorted: np.ndarray
    n_calib: int
    n_test: int
    mapie_model: Any


def _require_catboost():
    spec = importlib.util.find_spec("catboost")
    if spec is None:
        raise ImportError(
            "catboost is required for training and inference. Please install catboost."
        )
    catboost = importlib.import_module("catboost")
    return catboost.CatBoostRegressor, catboost.Pool


def _require_mapie():
    spec = importlib.util.find_spec("mapie.quantile_regression")
    if spec is None:
        raise ImportError(
            "mapie is required for conformal quantile calibration. Please install mapie."
        )
    return importlib.import_module("mapie.quantile_regression").MapieQuantileRegressor


def _prepare_cat_features_inplace(X: pd.DataFrame, cat_features: Sequence[str]):
    for col in cat_features:
        if col not in X.columns:
            X[col] = "missing"
        X[col] = X[col].astype(str).fillna("missing")


def _make_pool(
    X: pd.DataFrame, feature_names: Sequence[str], cat_feature_names: Sequence[str]
) -> Any:
    CatBoostRegressor, Pool = _require_catboost()
    X_ordered = X.reindex(columns=feature_names)
    cat_indices = [feature_names.index(c) for c in cat_feature_names if c in feature_names]
    return Pool(X_ordered, cat_features=cat_indices)


def _prepare_array(
    X: pd.DataFrame, feature_names: Sequence[str], cat_features: Sequence[str]
) -> np.ndarray:
    X_arr = X.copy()
    _prepare_cat_features_inplace(X_arr, cat_features)
    X_arr = X_arr.reindex(columns=feature_names)
    return X_arr.values


def _quantile_higher(values: np.ndarray, q: float) -> float:
    try:
        return float(np.quantile(values, q, method="higher"))
    except TypeError:
        return float(np.quantile(values, q, interpolation="higher"))


def _compute_qhat(nonconformity: np.ndarray, alpha: float) -> float:
    n = len(nonconformity)
    if n == 0:
        return 0.0
    q_level = math.ceil((n + 1) * (1 - alpha)) / n
    q_level = min(q_level, 1.0)
    return _quantile_higher(nonconformity, q_level)


def _calibrated_interval(
    mapie_model: Any, X_arr: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y_pred, y_pis = mapie_model.predict(X_arr, ensemble=False)
    # y_pis shape: (n_samples, n_alpha, 2)
    lower = y_pis[:, 0, 0]
    upper = y_pis[:, -1, 1]

    p50 = np.clip(y_pred, 0, None)
    p10 = np.clip(lower, 0, None)
    p90 = np.clip(upper, 0, None)

    p10 = np.minimum(p10, p50)
    p90 = np.maximum(p90, p50)

    std = (p90 - p10) / 3.29
    return p50, p10, p90, std


def _within_tol_rule() -> str:
    return f"max({int(TOL_PCT * 100)}% of p50, {TOL_MIN_OP_HOURS}h)"


class _CatBoostQuantileWrapper:
    """
    Lightweight wrapper so MAPIE can call .predict on a single estimator.
    It routes requested quantiles to prefit CatBoost quantile models.
    """

    def __init__(
        self,
        feature_names: Sequence[str],
        cat_features: Sequence[str],
        model_lo: Any,
        model_mid: Any,
        model_hi: Any,
    ):
        self.feature_names = list(feature_names)
        self.cat_features = list(cat_features)
        self.model_lo = model_lo
        self.model_mid = model_mid
        self.model_hi = model_hi

    def fit(self, X, y):
        return self

    def predict(self, X, alpha=None):
        X_df = pd.DataFrame(X, columns=self.feature_names)
        _prepare_cat_features_inplace(X_df, self.cat_features)
        pool = _make_pool(X_df, self.feature_names, self.cat_features)
        mid = self.model_mid.predict(pool)
        if alpha is None:
            return mid

        # alpha expected iterable; only supports requested quantiles 0.05/0.95/0.5.
        out = []
        for a in alpha:
            if a <= 0.5:
                out.append(self.model_lo.predict(pool))
            elif a >= 0.5:
                out.append(self.model_hi.predict(pool))
            else:
                out.append(mid)
        return np.vstack(out).T


def train_one_op(
    df_master: pd.DataFrame,
    target: str,
    models_dir: str = "models",
    version: str = "v1",
) -> Optional[Dict]:
    """
    Train a CatBoost Quantile + Conformalized Quantile Regression bundle.
    Saves a bundle per target and returns metrics.
    """
    X, y, num_features, cat_features, sub = build_training_data(
        df_master, target
    )
    if X is None:
        print(f"Skipping {target}: not enough data.")
        return None

    feature_names = num_features + cat_features
    X = X.copy()
    _prepare_cat_features_inplace(X, cat_features)

    n_rows = len(X)
    CatBoostRegressor, Pool = _require_catboost()
    MapieQuantileRegressor = _require_mapie()

    if n_rows >= 25:
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42
        )
        X_train, X_calib, y_train, y_calib = train_test_split(
            X_temp, y_temp, test_size=0.20, random_state=42
        )
    elif n_rows >= 10:
        X_train, X_calib, y_train, y_calib = train_test_split(
            X, y, test_size=0.20, random_state=42
        )
        X_test, y_test = None, None
    else:
        X_train, X_calib, y_train, y_calib = X, X, y, y
        X_test, y_test = None, None

    train_pool = _make_pool(X_train, feature_names, cat_features)
    calib_pool = _make_pool(X_calib, feature_names, cat_features)
    test_pool = (
        _make_pool(X_test, feature_names, cat_features) if X_test is not None else None
    )

    base_params = {
        "iterations": 2000,
        "learning_rate": 0.05,
        "depth": 6,
        "random_seed": 42,
        "thread_count": -1,
        "allow_writing_files": False,
        "verbose": False,
    }

    model_lo = CatBoostRegressor(loss_function="Quantile:alpha=0.05", **base_params)
    model_mid = CatBoostRegressor(loss_function="Quantile:alpha=0.50", **base_params)
    model_hi = CatBoostRegressor(loss_function="Quantile:alpha=0.95", **base_params)

    model_lo.fit(train_pool, y_train)
    model_mid.fit(train_pool, y_train)
    model_hi.fit(train_pool, y_train)

    lo_hat = model_lo.predict(calib_pool)
    hi_hat = model_hi.predict(calib_pool)
    nonconformity = np.maximum(0, np.maximum(lo_hat - y_calib, y_calib - hi_hat))

    pi_level = 0.90
    alpha = 1 - pi_level
    qhat = _compute_qhat(nonconformity, alpha)

    mid_calib = model_mid.predict(calib_pool)
    abs_err = np.abs(y_calib - mid_calib)
    abs_err_calib_sorted = np.sort(abs_err)

    eval_df = X_test if X_test is not None else X_calib
    eval_y = y_test if y_test is not None else y_calib
    wrapper = _CatBoostQuantileWrapper(feature_names, cat_features, model_lo, model_mid, model_hi)
    mapie = MapieQuantileRegressor(
        estimator=wrapper,
        alpha=[alpha / 2, 1 - alpha / 2],
        method="quantile",
        cv="prefit",
    )
    X_calib_arr = _prepare_array(X_calib, feature_names, cat_features)
    mapie.fit(X_calib_arr, y_calib)

    p50_eval, p10_eval, p90_eval, _ = _calibrated_interval(
        mapie, _prepare_array(eval_df, feature_names, cat_features)
    )
    mae = mean_absolute_error(eval_y, p50_eval)
    r2 = r2_score(eval_y, p50_eval) if len(eval_y) > 1 else float("nan")
    coverage_pi = float(np.mean((eval_y >= p10_eval) & (eval_y <= p90_eval)))
    avg_pi_width = float(np.mean(p90_eval - p10_eval))

    tol_eval = np.maximum(TOL_PCT * np.abs(p50_eval), TOL_MIN_OP_HOURS)
    within_tol_rate = float(np.mean(np.abs(eval_y - p50_eval) <= tol_eval))

    within_tol_rule = _within_tol_rule()

    bundle = CatBoostCQRBundle(
        kind="catboost_cqr_v1",
        target=target,
        version=version,
        feature_names=feature_names,
        cat_feature_names=cat_features,
        pi_level=pi_level,
        alpha=alpha,
        q_lo=alpha / 2,
        q_hi=1 - alpha / 2,
        qhat=qhat,
        model_lo=model_lo,
        model_mid=model_mid,
        model_hi=model_hi,
        abs_err_calib_sorted=abs_err_calib_sorted,
        n_calib=len(X_calib),
        n_test=len(X_test) if X_test is not None else 0,
        mapie_model=mapie,
    )

    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, f"{target}_{version}.joblib")
    joblib.dump(bundle, model_path)

    metrics = {
        "target": target,
        "version": version,
        "rows": int(len(sub)),
        "mae": float(mae),
        "r2": float(r2),
        "model_path": model_path,
        "pi_level": pi_level,
        "coverage_pi": coverage_pi,
        "avg_pi_width": avg_pi_width,
        "within_tol_rule": within_tol_rule,
        "within_tol_rate": within_tol_rate,
    }
    return metrics


def _prepare_bundle_features(
    X_df: pd.DataFrame, bundle: CatBoostCQRBundle
) -> Tuple[pd.DataFrame, Pool]:
    X = X_df.copy()
    _prepare_cat_features_inplace(X, bundle.cat_feature_names)
    X = X.reindex(columns=bundle.feature_names)
    pool = _make_pool(X, bundle.feature_names, bundle.cat_feature_names)
    return X, pool


def predict_with_interval(model_obj: CatBoostCQRBundle, X_df: pd.DataFrame):
    """
    Return calibrated p50/p10/p90/std for CatBoost CQR bundles using MAPIE.
    """
    if not (
        isinstance(model_obj, CatBoostCQRBundle)
        or (hasattr(model_obj, "kind") and getattr(model_obj, "kind") == "catboost_cqr_v1")
    ):
        raise ValueError("Unsupported model object for prediction (CatBoost CQR only).")

    bundle: CatBoostCQRBundle = model_obj  # type: ignore[assignment]
    X_arr = _prepare_array(X_df, bundle.feature_names, bundle.cat_feature_names)
    return _calibrated_interval(bundle.mapie_model, X_arr)


def empirical_within_tol_prob(model_obj, tol_hours: float) -> Optional[float]:
    if isinstance(model_obj, CatBoostCQRBundle) or (
        hasattr(model_obj, "kind") and getattr(model_obj, "kind") == "catboost_cqr_v1"
    ):
        arr = model_obj.abs_err_calib_sorted
        n = len(arr)
        if n == 0:
            return None
        idx = np.searchsorted(arr, tol_hours, side="right")
        return idx / n
    return None


def load_model(
    target: str, version: str = "v1", models_dir: str = "models"
) -> CatBoostCQRBundle:
    """Load a persisted model or bundle for a given operation."""
    model_path = os.path.join(models_dir, f"{target}_{version}.joblib")
    return joblib.load(model_path)
