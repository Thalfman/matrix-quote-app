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
    Train GradientBoosting models for a single operation's actual hours.
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

    pre_point = build_preprocessor(num_features, cat_features)
    pre_lower = build_preprocessor(num_features, cat_features)
    pre_upper = build_preprocessor(num_features, cat_features)

    point_model = GradientBoostingRegressor(
        n_estimators=400,
        random_state=42,
    )

    lower_model = GradientBoostingRegressor(
        n_estimators=400,
        random_state=42,
        loss="quantile",
        alpha=0.1,
    )

    upper_model = GradientBoostingRegressor(
        n_estimators=400,
        random_state=42,
        loss="quantile",
        alpha=0.9,
    )

    point_pipe = Pipeline(
        steps=[("preprocess", pre_point), ("model", point_model)]
    )
    lower_pipe = Pipeline(
        steps=[("preprocess", pre_lower), ("model", lower_model)]
    )
    upper_pipe = Pipeline(
        steps=[("preprocess", pre_upper), ("model", upper_model)]
    )

    point_pipe.fit(X_train, y_train)
    lower_pipe.fit(X_train, y_train)
    upper_pipe.fit(X_train, y_train)
    pred = point_pipe.predict(X_test)

    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)

    print(f"{target}: n={len(sub)}, MAE={mae:.1f}, R2={r2:.2f}")

    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, f"{target}_{version}.joblib")
    bundle = {
        "point": point_pipe,
        "lower": lower_pipe,
        "upper": upper_pipe,
        "interval_level": 0.8,
        "interval_method": "quantile",
    }
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


def predict_with_interval(pipe: Pipeline, X_df: pd.DataFrame):
    """
    Use quantile GradientBoosting models to get prediction intervals:
    - p50: central estimate
    - p10/p90: lower/upper quantile bounds
    - std: estimated spread based on P10/P90 width
    """
    if isinstance(pipe, dict):
        point = pipe["point"]
        lower = pipe["lower"]
        upper = pipe["upper"]

        p50 = point.predict(X_df)
        p10 = lower.predict(X_df)
        p90 = upper.predict(X_df)

        width = p90 - p10
        std = width / (2 * 1.281551565545)
        return p50, p10, p90, std

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
    target: str, version: str = "v1", models_dir: str = "models"
) -> Pipeline:
    """Load a persisted pipeline for a given operation."""
    model_path = os.path.join(models_dir, f"{target}_{version}.joblib")
    return joblib.load(model_path)
