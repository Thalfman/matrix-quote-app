# core/models.py
# Model training + loading + interval predictions.

import os
from typing import Optional, Dict, Any, TypedDict, Union, Tuple

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from scipy import sparse

from .features import build_training_data
from . import cqr

DEFAULT_MODEL_VERSION = "v1"
DEFAULT_ALPHA = 0.10
MIN_ROWS = 5


class CQRArtifact(TypedDict):
    preprocessor: Any
    model_lo: Any
    model_mid: Any
    model_hi: Any
    alpha: float
    qhat: float
    meta: Dict[str, Any]


def build_preprocessor(num_features, cat_features) -> ColumnTransformer:
    """Build a ColumnTransformer to handle numeric + categorical features."""
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )

    try:
        onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        onehot = OneHotEncoder(handle_unknown="ignore", sparse=False)

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", onehot),
        ]
    )

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_features),
            ("cat", categorical_transformer, cat_features),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )

    return pre


def _ensure_dense_array(features: Any) -> np.ndarray:
    if sparse.issparse(features):
        return features.toarray()
    return np.asarray(features)


def _count_positive_rows(df_master: pd.DataFrame, target: str) -> int:
    if target not in df_master.columns:
        return 0
    series = pd.to_numeric(df_master[target], errors="coerce").fillna(0)
    return int((series > 0).sum())


def _safe_mean(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.mean(values))


def _empty_cqr_metrics(
    target: str,
    version: str,
    rows: int,
    alpha: float,
) -> Dict[str, Any]:
    return {
        "target": target,
        "version": version,
        "trained": False,
        "rows": int(rows),
        "mae": float("nan"),
        "coverage": float("nan"),
        "interval_width": float("nan"),
        "qhat": float("nan"),
        "alpha": float(alpha),
        "r2": float("nan"),
    }


def get_target_model_path(target: str, model_dir: str) -> str:
    return os.path.join(
        model_dir,
        f"{target}_{DEFAULT_MODEL_VERSION}.joblib",
    )


def save_target_artifact(
    target: str,
    artifact: CQRArtifact,
    model_dir: str,
) -> None:
    os.makedirs(model_dir, exist_ok=True)
    model_path = get_target_model_path(target, model_dir)
    joblib.dump(artifact, model_path)


def load_target_artifact(
    target: str,
    model_dir: str,
) -> Optional[Union[CQRArtifact, Any]]:
    model_path = get_target_model_path(target, model_dir)
    if not os.path.exists(model_path):
        return None
    try:
        return joblib.load(model_path)
    except Exception:
        return None


def _split_train_calib_test(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = 42,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    Optional[pd.DataFrame],
    Optional[pd.Series],
]:
    n_samples = len(y)
    if n_samples < 8:
        X_train, X_calib, y_train, y_calib = train_test_split(
            X, y, test_size=0.2, random_state=random_state
        )
        return X_train, X_calib, y_train, y_calib, None, None

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=random_state
    )
    if len(y_temp) < 2:
        return X_train, X_temp, y_train, y_temp, None, None

    X_calib, X_test, y_calib, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=random_state
    )
    return X_train, X_calib, y_train, y_calib, X_test, y_test


def train_one_op_cqr(
    df_master: pd.DataFrame,
    target: str,
    models_dir: str = "models",
    version: str = DEFAULT_MODEL_VERSION,
    alpha: float = DEFAULT_ALPHA,
    min_rows: int = MIN_ROWS,
) -> Optional[Dict[str, Any]]:
    """
    Train quantile Gradient Boosting models + CQR calibration for one target.
    Saves a CQR artifact per target and returns basic metrics.
    """
    rows_available = _count_positive_rows(df_master, target)
    X, y, num_features, cat_features, sub = build_training_data(
        df_master, target
    )
    if X is None:
        print(f"Skipping {target}: not enough data.")
        return _empty_cqr_metrics(target, version, rows_available, alpha)

    mask = y > 0
    X = X.loc[mask]
    y = y.loc[mask]
    if len(y) < min_rows:
        print(f"Skipping {target}: not enough positive rows.")
        return _empty_cqr_metrics(target, version, int(len(y)), alpha)

    X_train, X_calib, y_train, y_calib, X_test, y_test = (
        _split_train_calib_test(X, y, random_state=42)
    )

    pre = build_preprocessor(num_features, cat_features)
    X_train_proc = _ensure_dense_array(pre.fit_transform(X_train))
    X_calib_proc = _ensure_dense_array(pre.transform(X_calib))
    X_test_proc = (
        _ensure_dense_array(pre.transform(X_test))
        if X_test is not None
        else None
    )

    q_lo = alpha / 2
    q_hi = 1 - alpha / 2

    model_lo = cqr.make_quantile_regressor(q_lo, random_state=42)
    model_mid = cqr.make_quantile_regressor(0.5, random_state=42)
    model_hi = cqr.make_quantile_regressor(q_hi, random_state=42)

    model_lo.fit(X_train_proc, y_train)
    model_mid.fit(X_train_proc, y_train)
    model_hi.fit(X_train_proc, y_train)

    calib_lo = model_lo.predict(X_calib_proc)
    calib_hi = model_hi.predict(X_calib_proc)
    scores = cqr.conformity_scores(y_calib.to_numpy(), calib_lo, calib_hi)
    qhat = cqr.finite_sample_qhat(scores, alpha=alpha)

    print(
        f"{target}: n={len(y)}, alpha={alpha:.2f}, qhat={qhat:.2f}"
    )

    artifact: CQRArtifact = {
        "preprocessor": pre,
        "model_lo": model_lo,
        "model_mid": model_mid,
        "model_hi": model_hi,
        "alpha": float(alpha),
        "qhat": float(qhat),
        "meta": {
            "rows": int(len(y)),
            "num_features": list(num_features),
            "cat_features": list(cat_features),
            "feature_columns": list(X.columns),
            "split_sizes": {
                "train": int(len(y_train)),
                "calib": int(len(y_calib)),
                "test": int(len(y_test)) if y_test is not None else 0,
            },
            "model_type": type(model_mid).__name__,
        },
    }
    save_target_artifact(target, artifact, models_dir)

    if X_test_proc is not None and y_test is not None and len(y_test) > 0:
        X_eval_proc = X_test_proc
        y_eval = y_test.to_numpy()
    else:
        X_eval_proc = X_calib_proc
        y_eval = y_calib.to_numpy()

    if X_eval_proc is not None and y_eval.size > 0:
        eval_pred_mid = model_mid.predict(X_eval_proc)
        eval_pred_lo = model_lo.predict(X_eval_proc)
        eval_pred_hi = model_hi.predict(X_eval_proc)
        eval_lo, eval_hi = cqr.apply_cqr(eval_pred_lo, eval_pred_hi, qhat)
        mae = _safe_mean(np.abs(y_eval - eval_pred_mid))
        coverage = _safe_mean((y_eval >= eval_lo) & (y_eval <= eval_hi))
        interval_width = _safe_mean(eval_hi - eval_lo)
    else:
        mae = float("nan")
        coverage = float("nan")
        interval_width = float("nan")

    metrics = {
        "target": target,
        "version": version,
        "trained": True,
        "rows": int(len(y)),
        "mae": float(mae),
        "coverage": float(coverage),
        "interval_width": float(interval_width),
        "qhat": float(qhat),
        "alpha": float(alpha),
        "r2": float("nan"),
    }
    return metrics


def train_one_op(
    df_master: pd.DataFrame,
    target: str,
    models_dir: str = "models",
    version: str = DEFAULT_MODEL_VERSION,
    alpha: float = DEFAULT_ALPHA,
    min_rows: int = MIN_ROWS,
) -> Optional[Dict[str, Any]]:
    """Default training entrypoint using CQR."""
    return train_one_op_cqr(
        df_master=df_master,
        target=target,
        models_dir=models_dir,
        version=version,
        alpha=alpha,
        min_rows=min_rows,
    )


def _format_confidence(alpha: Optional[float]) -> str:
    if alpha is None:
        alpha = DEFAULT_ALPHA
    confidence = int(round((1 - alpha) * 100))
    return f"{confidence}% calibrated"


def _is_cqr_artifact(artifact: Any) -> bool:
    if not isinstance(artifact, dict):
        return False
    required_keys = {"preprocessor", "model_lo", "model_mid", "model_hi", "qhat"}
    return required_keys.issubset(set(artifact.keys()))


def predict_target_with_interval(
    target_artifact: Any,
    X_df: pd.DataFrame,
    alpha: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """
    Predict calibrated interval bounds and point estimate for one target.
    Returns None when a model artifact is missing or invalid.
    """
    if target_artifact is None:
        return None

    if _is_cqr_artifact(target_artifact):
        pre = target_artifact["preprocessor"]
        X_proc = _ensure_dense_array(pre.transform(X_df))

        model_lo = target_artifact["model_lo"]
        model_mid = target_artifact["model_mid"]
        model_hi = target_artifact["model_hi"]
        qhat = target_artifact["qhat"]

        pred_lo = model_lo.predict(X_proc)
        pred_mid = model_mid.predict(X_proc)
        pred_hi = model_hi.predict(X_proc)
        p10, p90 = cqr.apply_cqr(pred_lo, pred_hi, qhat)
        std = (p90 - p10) / 3.29
        return {
            "p10": p10,
            "p50": pred_mid,
            "p90": p90,
            "std": std,
            "confidence": _format_confidence(
                alpha if alpha is not None else target_artifact.get("alpha")
            ),
        }

    if _is_legacy_model(target_artifact):
        p50, p10, p90, std = _predict_with_legacy_model(
            target_artifact, X_df
        )
        return {
            "p10": p10,
            "p50": p50,
            "p90": p90,
            "std": std,
            "confidence": "legacy",
        }

    return None


def _is_legacy_model(artifact: Any) -> bool:
    if artifact is None:
        return False
    return isinstance(artifact, Pipeline) or hasattr(artifact, "predict")


def _predict_with_legacy_model(
    model: Any, X_df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Best-effort legacy prediction for older artifacts without CQR metadata.
    Produces wide bounds to avoid implied calibration.
    """
    if isinstance(model, Pipeline):
        try:
            p50 = model.predict(X_df)
        except Exception:
            if hasattr(model, "named_steps") and "preprocess" in model.named_steps:
                pre = model.named_steps["preprocess"]
                X_proc = _ensure_dense_array(pre.transform(X_df))
                if "model" in model.named_steps:
                    p50 = model.named_steps["model"].predict(X_proc)
                else:
                    p50 = model.predict(X_proc)
            else:
                raise
    else:
        p50 = model.predict(X_df)

    p50 = np.asarray(p50)
    spread = np.maximum(np.abs(p50) * 0.5, 1.0)
    p10 = p50 - spread
    p90 = p50 + spread
    std = spread / 1.645
    return p50, p10, p90, std


def load_model(
    target: str,
    version: str = DEFAULT_MODEL_VERSION,
    models_dir: str = "models",
) -> Any:
    """Load a persisted model artifact for a given operation."""
    artifact = load_target_artifact(target, models_dir)
    if artifact is not None:
        return artifact
    model_path = os.path.join(models_dir, f"{target}_{version}.joblib")
    if not os.path.exists(model_path):
        return None
    try:
        return joblib.load(model_path)
    except Exception:
        return None


def predict_with_interval(model: Any, X_df: pd.DataFrame):
    """Predict p50/p10/p90/std from either a CQR artifact or RF pipeline."""
    result = predict_target_with_interval(model, X_df)
    if result is None:
        return None
    return result["p50"], result["p10"], result["p90"], result["std"]
