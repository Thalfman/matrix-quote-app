"""
core/models.py
Model training + loading + calibrated interval predictions.
"""

import datetime
import importlib
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from .config import CONFIDENCE_LEVELS, DEFAULT_CONFIDENCE
from .features import build_training_data


@dataclass
class IntervalModel:
    confidence: float
    alpha: float
    q_lo: float
    q_hi: float
    qhat: float
    model_lo: Any
    model_hi: Any


@dataclass
class CatBoostCQRBundle:
    kind: str
    target: str
    version: str
    feature_names: List[str]
    cat_feature_names: List[str]
    point_model: Any
    intervals: Dict[float, IntervalModel]
    training_rows: int
    calibration_rows: int
    evaluation_rows: int
    trained_at: str

    @property
    def model_mid(self):
        """Backward-compatibility alias for the median/point model."""
        pm = getattr(self, "point_model", None)
        if pm is not None:
            return pm

        legacy = self.__dict__.get("model_mid", None)
        if legacy is not None:
            return legacy

        raise AttributeError(
            "CatBoostCQRBundle has neither point_model nor legacy model_mid"
        )


def _require_catboost():
    spec = importlib.util.find_spec("catboost")
    if spec is None:
        raise ImportError(
            "catboost is required for training and inference. Please install catboost."
        )
    catboost = importlib.import_module("catboost")
    return catboost.CatBoostRegressor, catboost.Pool


def _prepare_cat_features_inplace(X: pd.DataFrame, cat_features: Sequence[str]):
    for col in cat_features:
        if col not in X.columns:
            X[col] = "missing"
        X[col] = X[col].astype(str).fillna("missing")


def _make_pool(
    X: pd.DataFrame,
    feature_names: Sequence[str],
    cat_feature_names: Sequence[str],
    y: Optional[Sequence[float]] = None,
) -> Any:
    CatBoostRegressor, Pool = _require_catboost()
    X_ordered = X.reindex(columns=feature_names)
    cat_indices = [feature_names.index(c) for c in cat_feature_names if c in feature_names]
    if y is not None:
        return Pool(X_ordered, label=y, cat_features=cat_indices)
    return Pool(X_ordered, cat_features=cat_indices)


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


def _calibrated_bounds(
    estimate_raw: np.ndarray,
    lo_raw: np.ndarray,
    hi_raw: np.ndarray,
    qhat: float,
) -> Dict[str, np.ndarray]:
    estimate = np.maximum(0, np.array(estimate_raw, dtype=float))
    lo = np.maximum(0, np.array(lo_raw, dtype=float) - qhat)
    hi = np.array(hi_raw, dtype=float) + qhat

    hi = np.maximum(hi, lo)
    estimate = np.clip(estimate, lo, hi)
    plus_minus = np.maximum(estimate - lo, hi - estimate)

    return {
        "estimate": estimate,
        "lo": lo,
        "hi": hi,
        "plus_minus": plus_minus,
    }


def train_one_op(
    df_master: pd.DataFrame,
    target: str,
    models_dir: str = "models",
    version: str = "v1",
) -> Optional[Dict]:
    """
    Train CatBoost quantile models + CQR calibration for all supported confidence levels.
    Persists a bundle per target and returns metrics.
    """
    X, y, num_features, cat_features, sub = build_training_data(df_master, target)
    if X is None:
        print(f"Skipping {target}: not enough data.")
        return None

    feature_names = num_features + cat_features
    X = X.copy()
    _prepare_cat_features_inplace(X, cat_features)

    n_rows = len(X)
    CatBoostRegressor, _ = _require_catboost()

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

    train_pool = _make_pool(X_train, feature_names, cat_features, y_train)
    calib_pool = _make_pool(X_calib, feature_names, cat_features)
    eval_df = X_test if X_test is not None else X_calib
    eval_y = y_test if X_test is not None else y_calib
    eval_pool = _make_pool(eval_df, feature_names, cat_features)

    base_params = {
        "iterations": 1200,
        "learning_rate": 0.05,
        "depth": 6,
        "random_seed": 42,
        "thread_count": -1,
        "allow_writing_files": False,
        "verbose": False,
    }

    point_model = CatBoostRegressor(loss_function="Quantile:alpha=0.50", **base_params)
    point_model.fit(train_pool)

    intervals: Dict[float, IntervalModel] = {}
    coverage_metrics: Dict[str, float] = {}
    width_metrics: Dict[str, float] = {}
    qhat_metrics: Dict[str, float] = {}

    point_pred_eval = np.maximum(0, point_model.predict(eval_pool))

    for confidence in CONFIDENCE_LEVELS:
        alpha = 1 - confidence
        q_lo = (1 - confidence) / 2
        q_hi = 1 - q_lo

        model_lo = CatBoostRegressor(
            loss_function=f"Quantile:alpha={q_lo}", **base_params
        )
        model_hi = CatBoostRegressor(
            loss_function=f"Quantile:alpha={q_hi}", **base_params
        )

        model_lo.fit(train_pool)
        model_hi.fit(train_pool)

        lo_hat = model_lo.predict(calib_pool)
        hi_hat = model_hi.predict(calib_pool)
        nonconformity = np.maximum(lo_hat - y_calib, y_calib - hi_hat)
        qhat = _compute_qhat(nonconformity, alpha)

        intervals[confidence] = IntervalModel(
            confidence=confidence,
            alpha=alpha,
            q_lo=q_lo,
            q_hi=q_hi,
            qhat=qhat,
            model_lo=model_lo,
            model_hi=model_hi,
        )

        eval_lo_raw = model_lo.predict(eval_pool)
        eval_hi_raw = model_hi.predict(eval_pool)
        calibrated = _calibrated_bounds(point_pred_eval, eval_lo_raw, eval_hi_raw, qhat)

        coverage = float(
            np.mean(
                (eval_y.values >= calibrated["lo"])
                & (eval_y.values <= calibrated["hi"])
            )
        )
        avg_width = float(np.mean(calibrated["hi"] - calibrated["lo"]))

        key = f"{int(confidence * 100)}"
        coverage_metrics[f"coverage_{key}"] = coverage
        width_metrics[f"avg_width_{key}"] = avg_width
        qhat_metrics[f"qhat_{key}"] = float(qhat)

    mae = mean_absolute_error(eval_y, point_pred_eval)
    r2 = r2_score(eval_y, point_pred_eval) if len(eval_y) > 1 else float("nan")

    bundle = CatBoostCQRBundle(
        kind="catboost_cqr_v2",
        target=target,
        version=version,
        feature_names=feature_names,
        cat_feature_names=cat_features,
        point_model=point_model,
        intervals=intervals,
        training_rows=len(X_train),
        calibration_rows=len(X_calib),
        evaluation_rows=len(eval_df),
        trained_at=datetime.datetime.utcnow().isoformat(),
    )

    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, f"{target}_{version}.joblib")
    joblib.dump(bundle, model_path)

    metrics: Dict[str, Any] = {
        "target": target,
        "version": version,
        "rows": int(len(sub)),
        "mae": float(mae),
        "r2": float(r2),
        "model_path": model_path,
        "default_confidence": DEFAULT_CONFIDENCE,
        "eval_rows": int(len(eval_df)),
    }
    metrics.update(coverage_metrics)
    metrics.update(width_metrics)
    metrics.update(qhat_metrics)
    return metrics


def predict_with_interval(
    model_obj: CatBoostCQRBundle,
    X_df: pd.DataFrame,
    confidence_level: float = DEFAULT_CONFIDENCE,
):
    """
    Return calibrated estimate/lo/hi/plus_minus arrays for the requested confidence level.
    """
    if not (
        isinstance(model_obj, CatBoostCQRBundle)
        or (hasattr(model_obj, "kind") and getattr(model_obj, "kind") == "catboost_cqr_v2")
    ):
        raise ValueError("Unsupported model object for prediction (CatBoost CQR v2 only).")

    bundle: CatBoostCQRBundle = model_obj  # type: ignore[assignment]
    if confidence_level not in bundle.intervals:
        raise ValueError(
            f"Confidence level {confidence_level} not trained for target {bundle.target}."
        )

    X = X_df.copy()
    _prepare_cat_features_inplace(X, bundle.cat_feature_names)
    pool = _make_pool(X, bundle.feature_names, bundle.cat_feature_names)

    interval = bundle.intervals[confidence_level]
    estimate_raw = bundle.point_model.predict(pool)
    lo_raw = interval.model_lo.predict(pool)
    hi_raw = interval.model_hi.predict(pool)

    return _calibrated_bounds(estimate_raw, lo_raw, hi_raw, interval.qhat)


def load_model(
    target: str, version: str = "v1", models_dir: str = "models"
) -> CatBoostCQRBundle:
    """Load a persisted model or bundle for a given operation."""
    model_path = os.path.join(models_dir, f"{target}_{version}.joblib")
    return joblib.load(model_path)
