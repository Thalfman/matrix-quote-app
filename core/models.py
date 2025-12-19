# core/models.py
# Model training + loading + interval predictions.

import os
from typing import Optional, Dict, Any, TypedDict, Union, Tuple

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .config import TARGETS
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

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_features),
            ("cat", categorical_transformer, cat_features),
        ],
        remainder="drop",
    )

    return pre


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
    X, y, num_features, cat_features, sub = build_training_data(
        df_master, target
    )
    if X is None:
        print(f"Skipping {target}: not enough data.")
        return None

    mask = y > 0
    X = X.loc[mask]
    y = y.loc[mask]
    if len(y) < min_rows:
        print(f"Skipping {target}: not enough positive rows.")
        return None

    X_train, X_calib, y_train, y_calib, X_test, y_test = (
        _split_train_calib_test(X, y, random_state=42)
    )

    pre = build_preprocessor(num_features, cat_features)
    X_train_proc = pre.fit_transform(X_train)
    X_calib_proc = pre.transform(X_calib)
    X_test_proc = pre.transform(X_test) if X_test is not None else None

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

    metrics = {
        "target": target,
        "version": version,
        "rows": int(len(y)),
        "alpha": float(alpha),
        "qhat": float(qhat),
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


def train_one_op_rf(
    df_master: pd.DataFrame,
    target: str,
    models_dir: str = "models",
    version: str = DEFAULT_MODEL_VERSION,
) -> Optional[Dict[str, Any]]:
    """
    Legacy RandomForest training for a single operation's actual hours.
    Saves a pipeline per target and returns basic metrics.
    """
    X, y, num_features, cat_features, sub = build_training_data(
        df_master, target
    )
    if X is None:
        print(f"Skipping {target}: not enough data.")
        return None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    pre = build_preprocessor(num_features, cat_features)

    rf = RandomForestRegressor(
        n_estimators=400,
        random_state=42,
        n_jobs=-1,
    )

    pipe = Pipeline(steps=[("preprocess", pre), ("model", rf)])

    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    mae = float(np.mean(np.abs(y_test - pred)))
    r2 = float(np.corrcoef(y_test, pred)[0, 1] ** 2) if len(y_test) > 1 else 0.0

    print(f"{target}: n={len(sub)}, MAE={mae:.1f}, R2={r2:.2f}")

    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, f"{target}_{version}.joblib")
    joblib.dump(pipe, model_path)

    metrics = {
        "target": target,
        "version": version,
        "rows": int(len(sub)),
        "mae": float(mae),
        "r2": float(r2),
        "model_path": model_path,
    }
    return metrics


def _predict_with_cqr_artifact(
    artifact: CQRArtifact, X_df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pre = artifact["preprocessor"]
    X_proc = pre.transform(X_df)

    model_lo = artifact["model_lo"]
    model_mid = artifact["model_mid"]
    model_hi = artifact["model_hi"]
    qhat = artifact["qhat"]

    pred_lo = model_lo.predict(X_proc)
    pred_mid = model_mid.predict(X_proc)
    pred_hi = model_hi.predict(X_proc)
    p10, p90 = cqr.apply_cqr(pred_lo, pred_hi, qhat)
    std = (p90 - p10) / 2.0
    return pred_mid, p10, p90, std


def _predict_with_rf_pipeline(
    pipe: Pipeline, X_df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Use individual trees from the RandomForest to get a prediction distribution:
    - p50: central estimate (mean)
    - p10/p90: lower/upper bounds
    - std: spread
    """
    pre = pipe.named_steps["preprocess"]
    rf = pipe.named_steps["model"]

    X_proc = pre.transform(X_df)
    tree_preds = np.stack(
        [tree.predict(X_proc) for tree in rf.estimators_],
        axis=1,
    )

    p50 = np.mean(tree_preds, axis=1)
    p10 = np.percentile(tree_preds, 10, axis=1)
    p90 = np.percentile(tree_preds, 90, axis=1)
    std = np.std(tree_preds, axis=1)

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
    return joblib.load(model_path)


def predict_with_interval(model: Any, X_df: pd.DataFrame):
    """Predict p50/p10/p90/std from either a CQR artifact or RF pipeline."""
    if isinstance(model, dict) and "model_lo" in model:
        return _predict_with_cqr_artifact(model, X_df)
    return _predict_with_rf_pipeline(model, X_df)
