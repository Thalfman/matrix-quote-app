# core/models.py
# Model training + loading + conformalized interval predictions using CatBoost.

import os
from dataclasses import dataclass
from math import ceil
from typing import Optional, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from .features import build_training_data, MISSING_TOKEN


@dataclass
class CQRModelBundle:
    feature_names: List[str]
    cat_features: List[str]
    point_model: CatBoostRegressor
    low_model: CatBoostRegressor
    high_model: CatBoostRegressor
    base_quantiles: Tuple[float, float]
    calib_scores_sorted: List[float]
    calib_n: int


def _prepare_features_for_catboost(
    X: pd.DataFrame, cat_features: List[str]
) -> pd.DataFrame:
    df = X.copy()
    for col in cat_features:
        if col not in df.columns:
            df[col] = MISSING_TOKEN
        df[col] = df[col].fillna(MISSING_TOKEN).astype(str)
    for col in df.columns:
        if col not in cat_features:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def train_one_op(
    df_master: pd.DataFrame,
    target: str,
    models_dir: str = "models",
    version: str = "v2",
    base_quantiles: Tuple[float, float] = (0.10, 0.90),
    calib_frac: float = 0.25,
    min_rows: int = 8,
) -> Optional[Dict]:
    """Train CatBoost models + conformal calibration for a single operation."""

    X, y, num_features, cat_features, sub = build_training_data(df_master, target)
    if X is None or sub is None:
        print(f"Skipping {target}: not enough data.")
        return None

    n_rows = len(sub)
    if n_rows < min_rows:
        print(f"Skipping {target}: only {n_rows} rows (<{min_rows}).")
        return None

    # Determine calibration split size ensuring both splits have at least 3 rows.
    calib_size = max(3, int(np.ceil(n_rows * calib_frac)))
    calib_size = min(calib_size, n_rows - 3)
    if calib_size < 3 or n_rows - calib_size < 3:
        print(f"Skipping {target}: insufficient rows for train/calib split.")
        return None

    X_train, X_calib, y_train, y_calib = train_test_split(
        X, y, test_size=calib_size, random_state=42
    )

    X_train_cb = _prepare_features_for_catboost(X_train, cat_features)
    X_calib_cb = _prepare_features_for_catboost(X_calib, cat_features)

    base_q_low, base_q_high = base_quantiles

    point_model = CatBoostRegressor(
        loss_function="MAE", random_seed=42, verbose=False
    )
    low_model = CatBoostRegressor(
        loss_function=f"Quantile:alpha={base_q_low}", random_seed=42, verbose=False
    )
    high_model = CatBoostRegressor(
        loss_function=f"Quantile:alpha={base_q_high}", random_seed=42, verbose=False
    )

    point_model.fit(X_train_cb, y_train, cat_features=cat_features)
    low_model.fit(X_train_cb, y_train, cat_features=cat_features)
    high_model.fit(X_train_cb, y_train, cat_features=cat_features)

    point_pred = point_model.predict(X_calib_cb)
    q_low = low_model.predict(X_calib_cb)
    q_high = high_model.predict(X_calib_cb)

    mae = mean_absolute_error(y_calib, point_pred)
    r2 = r2_score(y_calib, point_pred)

    scores = np.maximum(q_low - y_calib, y_calib - q_high, 0)
    calib_scores_sorted = np.sort(scores)
    calib_n = len(calib_scores_sorted)

    bundle = CQRModelBundle(
        feature_names=list(X_train_cb.columns),
        cat_features=cat_features,
        point_model=point_model,
        low_model=low_model,
        high_model=high_model,
        base_quantiles=(base_q_low, base_q_high),
        calib_scores_sorted=calib_scores_sorted.tolist(),
        calib_n=int(calib_n),
    )

    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, f"{target}_{version}.joblib")
    joblib.dump(bundle, model_path)

    metrics = {
        "target": target,
        "version": version,
        "rows": int(n_rows),
        "calib_n": int(calib_n),
        "mae": float(mae),
        "r2": float(r2),
        "base_q_low": float(base_q_low),
        "base_q_high": float(base_q_high),
        "model_path": model_path,
    }
    return metrics


def load_model(target: str, version: str = "v2", models_dir: str = "models"):
    model_path = os.path.join(models_dir, f"{target}_{version}.joblib")
    return joblib.load(model_path)


def predict_with_conformal_interval(
    bundle: CQRModelBundle, X_df: pd.DataFrame, confidence: float = 0.90
):
    df = X_df.copy()
    for col in bundle.feature_names:
        if col not in df.columns:
            df[col] = np.nan
    df = df[bundle.feature_names]

    df = _prepare_features_for_catboost(df, bundle.cat_features)

    point_pred = bundle.point_model.predict(df)
    q_low = bundle.low_model.predict(df)
    q_high = bundle.high_model.predict(df)

    if bundle.calib_n > 0:
        k = ceil((bundle.calib_n + 1) * confidence)
        k = min(max(1, k), bundle.calib_n)
        t = bundle.calib_scores_sorted[k - 1]
    else:
        t = 0.0

    low = np.clip(q_low - t, 0, None)
    high = np.clip(q_high + t, 0, None)
    estimate = np.clip(point_pred, 0, None)

    return estimate, low, high
