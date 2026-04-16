"""Shared fixtures for the matrix-quote-app test suite."""

import sys
import os

import numpy as np
import pandas as pd
import pytest

# Ensure the repo root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.config import (
    TARGETS,
    QUOTE_NUM_FEATURES,
    QUOTE_CAT_FEATURES,
    REQUIRED_TRAINING_COLS,
)
from core.features import engineer_features_for_training
from core.models import train_one_op


@pytest.fixture()
def synthetic_training_df() -> pd.DataFrame:
    """Return ~20 rows of realistic synthetic training data.

    Contains all REQUIRED_TRAINING_COLS, most QUOTE_NUM_FEATURES,
    all QUOTE_CAT_FEATURES, and at least 3 TARGETS columns with
    non-zero hours (enough rows per op for RandomForest training).
    """
    rng = np.random.RandomState(42)
    n = 20

    data: dict = {}

    # --- Required columns ---
    data["project_id"] = [f"PRJ-{i:04d}" for i in range(n)]
    data["include_in_training"] = ["yes"] * n
    data["dataset_role"] = ["actuals"] * n

    # --- Categorical features ---
    data["industry_segment"] = rng.choice(
        ["Automotive", "Food & Bev", "Pharma"], n
    ).tolist()
    data["system_category"] = rng.choice(
        ["Palletizing", "Assembly", "Packaging"], n
    ).tolist()
    data["automation_level"] = rng.choice(
        ["Full", "Semi", "Manual"], n
    ).tolist()
    data["plc_family"] = rng.choice(
        ["Allen-Bradley", "Siemens", "Mitsubishi"], n
    ).tolist()
    data["hmi_family"] = rng.choice(
        ["PanelView", "KTP", "GOT"], n
    ).tolist()
    data["vision_type"] = rng.choice(
        ["None", "2D", "3D"], n
    ).tolist()

    # --- Numeric features ---
    data["stations_count"] = rng.randint(1, 8, n).tolist()
    data["robot_count"] = rng.randint(0, 5, n).tolist()
    data["fixture_sets"] = rng.randint(0, 4, n).tolist()
    data["part_types"] = rng.randint(1, 6, n).tolist()
    data["servo_axes"] = rng.randint(0, 12, n).tolist()
    data["pneumatic_devices"] = rng.randint(0, 10, n).tolist()
    data["safety_doors"] = rng.randint(0, 4, n).tolist()
    data["weldment_perimeter_ft"] = (rng.rand(n) * 50).round(1).tolist()
    data["fence_length_ft"] = (rng.rand(n) * 100).round(1).tolist()
    data["conveyor_length_ft"] = (rng.rand(n) * 60).round(1).tolist()
    data["product_familiarity_score"] = rng.randint(1, 6, n).tolist()
    data["product_rigidity"] = rng.randint(1, 6, n).tolist()
    data["is_product_deformable"] = rng.choice([0, 1], n).tolist()
    data["is_bulk_product"] = rng.choice([0, 1], n).tolist()
    data["bulk_rigidity_score"] = rng.randint(0, 4, n).tolist()
    data["has_tricky_packaging"] = rng.choice([0, 1], n).tolist()
    data["process_uncertainty_score"] = rng.randint(1, 6, n).tolist()
    data["changeover_time_min"] = (rng.rand(n) * 30).round(1).tolist()
    data["safety_devices_count"] = rng.randint(0, 6, n).tolist()
    data["custom_pct"] = (rng.rand(n) * 100).round(0).tolist()
    data["duplicate"] = rng.choice([0, 1], n).tolist()
    data["has_controls"] = [1] * n
    data["has_robotics"] = rng.choice([0, 1], n).tolist()
    data["Retrofit"] = rng.choice([0, 1], n).tolist()
    data["complexity_score_1_5"] = rng.randint(1, 6, n).tolist()
    data["vision_systems_count"] = rng.randint(0, 3, n).tolist()
    data["panel_count"] = rng.randint(1, 4, n).tolist()
    data["drive_count"] = rng.randint(0, 6, n).tolist()
    data["quoted_materials_cost"] = (rng.rand(n) * 200000 + 10000).round(0).tolist()

    # --- Target columns (actual hours) ---
    # Make at least 3 targets have non-zero hours across all 20 rows so
    # RandomForest training succeeds (needs >= 5 non-zero rows per op).
    for target in TARGETS:
        # Base hours correlated with station count for realism
        base = np.array(data["stations_count"]) * rng.uniform(10, 60)
        noise = rng.rand(n) * 40
        hours = (base + noise).round(1)
        # Ensure all values are positive (non-zero)
        hours = np.clip(hours, 5.0, None)
        data[target] = hours.tolist()

    return pd.DataFrame(data)


@pytest.fixture()
def tmp_models_dir(tmp_path):
    """Return a temporary directory path for model artifacts."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    return str(models_dir)


@pytest.fixture()
def trained_models(synthetic_training_df, tmp_models_dir):
    """Train models using synthetic data and return the metrics list.

    Trains models for all TARGETS that have enough data.
    """
    df_eng = engineer_features_for_training(synthetic_training_df)
    metrics_list = []
    for target in TARGETS:
        m = train_one_op(df_eng, target, models_dir=tmp_models_dir)
        if m is not None:
            metrics_list.append(m)
    return metrics_list
