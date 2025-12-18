"""Quick sanity check for conformal coverage on stored v2 models."""

import os

import pandas as pd
from sklearn.model_selection import train_test_split

from core.config import TARGETS
from core.features import engineer_features_for_training, build_training_data
from core.models import load_model, predict_with_conformal_interval, CQRModelBundle

MASTER_DATA_PATH = os.path.join("data", "master", "projects_master.parquet")


def main(confidence: float = 0.9):
    if not os.path.exists(MASTER_DATA_PATH):
        print("Master dataset not found at", MASTER_DATA_PATH)
        return

    df_master = engineer_features_for_training(pd.read_parquet(MASTER_DATA_PATH))

    for target in TARGETS:
        model_path = os.path.join("models", f"{target}_v2.joblib")
        if not os.path.exists(model_path):
            print(f"Skipping {target}: v2 model not found.")
            continue

        X, y, num_features, cat_features, sub = build_training_data(df_master, target)
        if X is None or len(sub) < 6:
            print(f"Skipping {target}: not enough rows for eval.")
            continue

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=123
        )

        bundle: CQRModelBundle = load_model(target, version="v2")
        est, low, high = predict_with_conformal_interval(
            bundle, X_test, confidence=confidence
        )
        coverage = float(((y_test >= low) & (y_test <= high)).mean())
        width = float((high - low).mean())
        print(
            f"{target}: calib_n={bundle.calib_n}, requested={confidence:.2f}, "
            f"empirical_coverage={coverage:.2f}, avg_width={width:.2f}"
        )


if __name__ == "__main__":
    conf = float(os.environ.get("CONFIDENCE", 0.9))
    main(confidence=conf)
