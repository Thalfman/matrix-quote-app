# core/models.py
# Model training + loading + interval predictions.

import os
from typing import Optional, Dict

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
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
    Train a RandomForest model for a single operation's actual hours.
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

    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)

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


def predict_with_interval(pipe: Pipeline, X_df: pd.DataFrame):
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
    target: str, version: str = "v1", models_dir: str = "models"
) -> Pipeline:
    """Load a persisted pipeline for a given operation."""
    model_path = os.path.join(models_dir, f"{target}_{version}.joblib")
    return joblib.load(model_path)
