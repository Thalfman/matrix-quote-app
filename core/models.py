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


# Stores one confidence level's quantile models and its CQR calibration constant (qhat).
# For example, at 90% confidence: alpha=0.10, q_lo=0.05, q_hi=0.95.
@dataclass
class IntervalModel:
    confidence: float    # e.g. 0.90
    alpha: float         # = 1 - confidence; the miscoverage rate
    q_lo: float          # lower quantile target (alpha / 2)
    q_hi: float          # upper quantile target (1 - alpha / 2)
    qhat: float          # CQR calibration offset learned on the held-out calibration set
    model_lo: Any        # CatBoostRegressor trained at Quantile:alpha=q_lo
    model_hi: Any        # CatBoostRegressor trained at Quantile:alpha=q_hi


# Serializable bundle that packages everything needed to make a calibrated prediction
# for one operation target (e.g. "me10_actual_hours"). Persisted as a .joblib file.
@dataclass
class CatBoostCQRBundle:
    kind: str                                # schema tag, always "catboost_cqr_v2"
    target: str                              # operation column name this model predicts
    version: str                             # model version string, e.g. "v3"
    feature_names: List[str]                 # ordered list of all features (numeric + categorical)
    cat_feature_names: List[str]             # subset of feature_names that are categorical
    point_model: Any                         # CatBoostRegressor at Quantile:alpha=0.50 (median)
    intervals: Dict[float, IntervalModel]    # keyed by confidence level (0.80, 0.90, 0.95)
    training_rows: int
    calibration_rows: int
    evaluation_rows: int
    trained_at: str                          # ISO timestamp of training run

    # Backward-compatibility alias so older code that references .model_mid still works
    # after we renamed the field to .point_model.
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


# Lazily import catboost only when needed so the app can start even if catboost
# isn't installed (e.g. for UI-only browsing of existing results).
def _require_catboost():
    spec = importlib.util.find_spec("catboost")
    if spec is None:
        raise ImportError(
            "catboost is required for training and inference. Please install catboost."
        )
    catboost = importlib.import_module("catboost")
    return catboost.CatBoostRegressor, catboost.Pool


# Ensure every categorical column exists and is a clean string (no NaN).
# CatBoost requires categorical features to be strings, not numeric or null.
def _prepare_cat_features_inplace(X: pd.DataFrame, cat_features: Sequence[str]):
    for col in cat_features:
        if col not in X.columns:
            X[col] = "missing"
        X[col] = X[col].astype(str).fillna("missing")


# Build a CatBoost Pool object with columns in the exact order the model expects.
# The Pool is CatBoost's internal data structure for efficient training/prediction.
def _make_pool(
    X: pd.DataFrame,
    feature_names: Sequence[str],
    cat_feature_names: Sequence[str],
    y: Optional[Sequence[float]] = None,
) -> Any:
    CatBoostRegressor, Pool = _require_catboost()
    # Reindex ensures columns appear in the same order used during training
    X_ordered = X.reindex(columns=feature_names)
    # CatBoost needs categorical features identified by their column index positions
    cat_indices = [feature_names.index(c) for c in cat_feature_names if c in feature_names]
    if y is not None:
        return Pool(X_ordered, label=y, cat_features=cat_indices)
    return Pool(X_ordered, cat_features=cat_indices)


# Numpy quantile with "higher" interpolation. The try/except handles the API change
# between older numpy (interpolation= kwarg) and newer numpy (method= kwarg).
def _quantile_higher(values: np.ndarray, q: float) -> float:
    try:
        return float(np.quantile(values, q, method="higher"))
    except TypeError:
        return float(np.quantile(values, q, interpolation="higher"))


# Compute the CQR calibration constant (qhat) from nonconformity scores.
# qhat is the (1 - alpha)-quantile of the nonconformity scores on the calibration set,
# adjusted with a finite-sample correction factor of (n+1)/n per the CQR paper.
# A larger qhat widens the prediction interval to achieve the target coverage.
def _compute_qhat(nonconformity: np.ndarray, alpha: float) -> float:
    n = len(nonconformity)
    if n == 0:
        return 0.0
    q_level = math.ceil((n + 1) * (1 - alpha)) / n
    q_level = min(q_level, 1.0)
    return _quantile_higher(nonconformity, q_level)


# Apply the CQR calibration offset (qhat) to raw quantile predictions to produce
# the final prediction interval. qhat widens the interval symmetrically:
#   calibrated_lo = raw_lo - qhat,  calibrated_hi = raw_hi + qhat
# Also enforces non-negativity (hours can't be < 0) and ensures hi >= lo.
def _calibrated_bounds(
    estimate_raw: np.ndarray,
    lo_raw: np.ndarray,
    hi_raw: np.ndarray,
    qhat: float,
) -> Dict[str, np.ndarray]:
    # Floor the point estimate at zero (negative hours are nonsensical)
    estimate = np.maximum(0, np.array(estimate_raw, dtype=float))
    # Widen the interval by qhat, flooring the lower bound at zero
    lo = np.maximum(0, np.array(lo_raw, dtype=float) - qhat)
    hi = np.array(hi_raw, dtype=float) + qhat

    # Ensure the interval is valid (hi >= lo) and the estimate sits inside it
    hi = np.maximum(hi, lo)
    estimate = np.clip(estimate, lo, hi)
    # plus_minus = the larger of (estimate - lo) or (hi - estimate)
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
    # Extract feature matrix (X) and target vector (y) from the master dataset.
    # build_training_data filters to rows with non-zero hours and selects only
    # the quote-time features that the model is allowed to use.
    X, y, num_features, cat_features, sub = build_training_data(df_master, target)
    if X is None:
        print(f"Skipping {target}: not enough data.")
        return None

    # Combine numeric + categorical into the final ordered feature list and
    # ensure all categorical columns are clean strings for CatBoost.
    feature_names = num_features + cat_features
    X = X.copy()
    _prepare_cat_features_inplace(X, cat_features)

    n_rows = len(X)
    CatBoostRegressor, _ = _require_catboost()

    # Adaptive data splitting based on dataset size:
    #   >= 25 rows: 64% train / 16% calibration / 20% test (three-way split)
    #   10-24 rows: 80% train / 20% calibration, no held-out test set
    #   < 10 rows:  all data used for both training and calibration (no holdout)
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

    # Build CatBoost Pool objects for each split.
    # - train_pool: used to fit models
    # - calib_pool: unlabeled, used for CQR nonconformity score prediction
    # - calib_pool_labeled: labeled, used as the early-stopping eval set
    # - eval_pool: used for final metric computation (test set if available, else calib)
    train_pool = _make_pool(X_train, feature_names, cat_features, y_train)
    calib_pool = _make_pool(X_calib, feature_names, cat_features)
    calib_pool_labeled = _make_pool(X_calib, feature_names, cat_features, y_calib)
    eval_df = X_test if X_test is not None else X_calib
    eval_y = y_test if X_test is not None else y_calib
    eval_pool = _make_pool(eval_df, feature_names, cat_features)

    # Shared hyperparameters for all CatBoost models in this bundle.
    # early_stopping_rounds=50 halts training if the eval metric doesn't improve
    # for 50 consecutive iterations, preventing overfitting on small datasets.
    base_params = {
        "iterations": 1200,
        "learning_rate": 0.05,
        "depth": 6,
        "random_seed": 42,
        "thread_count": -1,
        "allow_writing_files": False,
        "verbose": False,
        "early_stopping_rounds": 50,
    }

    # Train the point (median) model — this produces the "estimate" in predictions.
    # Uses Quantile loss at alpha=0.50, which targets the conditional median of hours.
    point_model = CatBoostRegressor(loss_function="Quantile:alpha=0.50", **base_params)
    point_model.fit(train_pool, eval_set=calib_pool_labeled)

    # Accumulators for per-confidence-level metrics and trained interval models
    intervals: Dict[float, IntervalModel] = {}
    coverage_metrics: Dict[str, float] = {}
    width_metrics: Dict[str, float] = {}
    qhat_metrics: Dict[str, float] = {}

    # Point predictions on the eval set (used for MAE / R² and as the "estimate"
    # inside the calibrated bounds at each confidence level)
    point_pred_eval = np.maximum(0, point_model.predict(eval_pool))

    # Train a pair of quantile models (lo + hi) for each supported confidence level
    # (80%, 90%, 95%) and calibrate them using Conformal Quantile Regression (CQR).
    for confidence in CONFIDENCE_LEVELS:
        alpha = 1 - confidence
        # Symmetric quantiles around the median: e.g. 90% -> q_lo=0.05, q_hi=0.95
        q_lo = (1 - confidence) / 2
        q_hi = 1 - q_lo

        # Train the lower-bound and upper-bound quantile regressors
        model_lo = CatBoostRegressor(
            loss_function=f"Quantile:alpha={q_lo}", **base_params
        )
        model_hi = CatBoostRegressor(
            loss_function=f"Quantile:alpha={q_hi}", **base_params
        )

        model_lo.fit(train_pool, eval_set=calib_pool_labeled)
        model_hi.fit(train_pool, eval_set=calib_pool_labeled)

        # CQR calibration: compute nonconformity scores on the calibration set.
        # Nonconformity = max(how far below lo the actual is, how far above hi the actual is).
        # If the actual falls inside [lo, hi], nonconformity is negative (good).
        lo_hat = model_lo.predict(calib_pool)
        hi_hat = model_hi.predict(calib_pool)
        nonconformity = np.maximum(lo_hat - y_calib, y_calib - hi_hat)
        # qhat is the calibration constant that widens the interval to hit target coverage
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

        # Evaluate calibrated coverage and interval width on the held-out eval set
        eval_lo_raw = model_lo.predict(eval_pool)
        eval_hi_raw = model_hi.predict(eval_pool)
        calibrated = _calibrated_bounds(point_pred_eval, eval_lo_raw, eval_hi_raw, qhat)

        # Coverage = fraction of eval samples where actual hours fall inside [lo, hi]
        coverage = float(
            np.mean(
                (eval_y.values >= calibrated["lo"])
                & (eval_y.values <= calibrated["hi"])
            )
        )
        # Average interval width measures precision — narrower is better if coverage is met
        avg_width = float(np.mean(calibrated["hi"] - calibrated["lo"]))

        key = f"{int(confidence * 100)}"
        coverage_metrics[f"coverage_{key}"] = coverage
        width_metrics[f"avg_width_{key}"] = avg_width
        qhat_metrics[f"qhat_{key}"] = float(qhat)

    # Compute overall point-prediction accuracy metrics on the eval set
    mae = mean_absolute_error(eval_y, point_pred_eval)
    r2 = r2_score(eval_y, point_pred_eval) if len(eval_y) > 1 else float("nan")

    # Package all models, metadata, and calibration constants into a single bundle
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

    # Persist the bundle to disk as a joblib file (e.g. models/me10_actual_hours_v3.joblib)
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, f"{target}_{version}.joblib")
    joblib.dump(bundle, model_path)

    # Return a flat metrics dict for this operation (written to metrics_summary.csv by the caller)
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


# ── Inference ──
# Run all three models (point + lo + hi) on new data and apply CQR calibration.
# Returns a dict with arrays: estimate, lo, hi, plus_minus.
def predict_with_interval(
    model_obj: CatBoostCQRBundle,
    X_df: pd.DataFrame,
    confidence_level: float = DEFAULT_CONFIDENCE,
):
    """
    Return calibrated estimate/lo/hi/plus_minus arrays for the requested confidence level.
    """
    # Guard: only CatBoost CQR v2 bundles are supported
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

    # Prepare input features identically to how training data was prepared
    X = X_df.copy()
    _prepare_cat_features_inplace(X, bundle.cat_feature_names)
    pool = _make_pool(X, bundle.feature_names, bundle.cat_feature_names)

    # Run the three models: median (point estimate), lower quantile, upper quantile
    interval = bundle.intervals[confidence_level]
    estimate_raw = bundle.point_model.predict(pool)
    lo_raw = interval.model_lo.predict(pool)
    hi_raw = interval.model_hi.predict(pool)

    # Apply the CQR calibration offset and return the final bounds
    return _calibrated_bounds(estimate_raw, lo_raw, hi_raw, interval.qhat)


# ── Model versioning ──
# Each retrain writes a new version string (v1, v2, ...) to current_version.txt.
# Model loading defaults to the current version so callers don't need to track it.

def _current_version(models_dir: str = "models") -> str:
    """Read the current model version from the version file, defaulting to 'v1'."""
    version_path = os.path.join(models_dir, "current_version.txt")
    if os.path.exists(version_path):
        return open(version_path).read().strip()
    return "v1"


def load_model(
    target: str, version: str = "", models_dir: str = "models"
) -> CatBoostCQRBundle:
    """Load a persisted model or bundle for a given operation."""
    if not version:
        version = _current_version(models_dir)
    model_path = os.path.join(models_dir, f"{target}_{version}.joblib")
    return joblib.load(model_path)


# ── Cached model loading ──
# Uses an LRU cache keyed on (file_path, file_mtime) so that:
#   1. Repeated predictions don't reload from disk (fast path).
#   2. After a retrain changes the file, the new mtime evicts the stale entry.

def load_model_cached(
    target: str, version: str = "", models_dir: str = "models"
) -> CatBoostCQRBundle:
    """Load a model with in-memory caching (keyed by path + mtime)."""
    if not version:
        version = _current_version(models_dir)
    model_path = os.path.join(models_dir, f"{target}_{version}.joblib")
    mtime = os.path.getmtime(model_path)
    return _load_model_from_cache(model_path, mtime)


from functools import lru_cache


@lru_cache(maxsize=32)
def _load_model_from_cache(model_path: str, mtime: float) -> CatBoostCQRBundle:
    return joblib.load(model_path)


# ── Feature importance ──
# Extracts CatBoost's built-in feature importance from the point model.
# First tries the no-data path (works if the model was trained with it stored internally).
# Falls back to computing importance from a sample of the master dataset if needed.
def get_feature_importance(
    bundle: CatBoostCQRBundle,
    df_master: Optional[pd.DataFrame] = None,
) -> Optional[pd.DataFrame]:
    """
    Return a DataFrame with columns ['feature', 'importance'] sorted descending,
    or None if importance cannot be computed.
    """
    if not isinstance(bundle, CatBoostCQRBundle):
        return None

    # Get the underlying CatBoost model (point_model or legacy model_mid)
    cb_model = getattr(bundle, "point_model", None) or bundle.__dict__.get("model_mid")
    if cb_model is None:
        return None

    feature_names = getattr(bundle, "feature_names", [])
    if not feature_names:
        return None

    # Try getting importance without data (CatBoost stores it internally for some modes)
    try:
        importances = cb_model.get_feature_importance()
    except Exception:
        # Fallback: compute importance using a sample of the master dataset as a Pool
        if df_master is None:
            return None
        try:
            cat_feature_names = getattr(bundle, "cat_feature_names", [])
            # Cap at 5000 rows to keep computation fast
            df_pool_source = (
                df_master.sample(n=min(len(df_master), 5000), random_state=42)
                if len(df_master) > 5000
                else df_master
            )
            df_pool = df_pool_source.copy()
            for col in feature_names:
                if col not in df_pool.columns:
                    df_pool[col] = 0
            _prepare_cat_features_inplace(df_pool, cat_feature_names)
            pool = _make_pool(df_pool, feature_names, cat_feature_names)
            importances = cb_model.get_feature_importance(pool)
        except Exception:
            return None

    # Ensure the importance array aligns with the feature list
    if len(feature_names) != len(importances):
        return None

    return (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
