# core/models.py
# Model training + loading + interval predictions.

import os
from typing import Optional, Dict

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .config import TARGETS
from .features import build_training_data


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


def train_one_op(
    df_master: pd.DataFrame,
    target: str,
    models_dir: str = "models",
    version: str = "v1",
) -> Optional[Dict]:
    """
    Train a GradientBoosting model for a single operation's actual hours.
    Saves a bundle (pipeline + quantile models) per target and returns basic metrics.
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

    gbr_main = GradientBoostingRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.1, random_state=42
    )
    gbr_q10 = GradientBoostingRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.1,
        loss="quantile", alpha=0.1, random_state=42,
    )
    gbr_q90 = GradientBoostingRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.1,
        loss="quantile", alpha=0.9, random_state=42,
    )

    pipe = Pipeline(steps=[("preprocess", pre), ("model", gbr_main)])
    pipe.fit(X_train, y_train)

    # Fit quantile models on preprocessed training data
    X_train_proc = pre.transform(X_train)
    gbr_q10.fit(X_train_proc, y_train)
    gbr_q90.fit(X_train_proc, y_train)

    pred = pipe.predict(X_test)

    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)

    print(f"{target}: n={len(sub)}, MAE={mae:.1f}, R2={r2:.2f}")

    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, f"{target}_{version}.joblib")
    bundle = {"pipeline": pipe, "q10": gbr_q10, "q90": gbr_q90}
    joblib.dump(bundle, model_path)

    metrics = {
        "target": target,
        "version": version,
        "rows": int(len(sub)),
        "mae": float(mae),
        "r2": float(r2),
        "model_path": model_path,
    }
    return metrics


def predict_with_interval(bundle: dict, X_df: pd.DataFrame):
    """
    Predict p50/p10/p90/std, supporting two saved bundle shapes:

    - Legacy: ``{"pipeline", "q10", "q90"}`` produced by the local
      ``train_one_op`` — a main GBR pipeline plus two quantile GBRs.
    - CQR: ``{"preprocessor", "model_mid", "model_lo", "model_hi",
      "qhat", "alpha", "meta"}`` — a conformalized-quantile-regression
      bundle. ``qhat`` shifts the raw quantile predictions to produce
      calibrated prediction intervals.
    """
    if "pipeline" in bundle:
        pipe = bundle["pipeline"]
        pre = pipe.named_steps["preprocess"]
        X_proc = pre.transform(X_df)
        p50 = pipe.named_steps["model"].predict(X_proc)
        p10 = bundle["q10"].predict(X_proc)
        p90 = bundle["q90"].predict(X_proc)
    else:
        pre = bundle["preprocessor"]
        X_proc = pre.transform(X_df)
        p50 = bundle["model_mid"].predict(X_proc)
        p10_raw = bundle["model_lo"].predict(X_proc)
        p90_raw = bundle["model_hi"].predict(X_proc)
        qhat = float(bundle.get("qhat") or 0.0)
        p10 = p10_raw - qhat
        p90 = p90_raw + qhat

    std = (p90 - p10) / 2.56
    return p50, p10, p90, std


def load_model(
    target: str, version: str = "v1", models_dir: str = "models"
) -> Optional[dict]:
    """Load a persisted model bundle for a given operation.

    Returns ``None`` if the model file does not exist on disk.
    """
    model_path = os.path.join(models_dir, f"{target}_{version}.joblib")
    if not os.path.exists(model_path):
        return None
    return joblib.load(model_path)
