"""
Lightweight smoke test for CatBoost CQR training + prediction.

Usage:
    python scripts/smoke_test_train_predict.py
"""

import numpy as np
import pandas as pd

from core.config import TOL_MIN_OP_HOURS, TOL_PCT
from core.features import engineer_features_for_training, prepare_quote_features
from core.models import (
    empirical_within_tol_prob,
    load_model,
    predict_with_interval,
    train_one_op,
)


def build_synthetic_df(rows: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    data = {
        "project_id": [f"proj_{i}" for i in range(rows)],
        "include_in_training": ["yes"] * rows,
        "dataset_role": ["actuals"] * rows,
        "industry_segment": rng.choice(
            ["Automotive", "Food & Beverage", "General"], size=rows
        ),
        "system_category": rng.choice(
            ["Machine Tending", "End of Line Automation"], size=rows
        ),
        "automation_level": rng.choice(
            ["Semi-Automatic", "Robotic", "Hard Automation"], size=rows
        ),
        "plc_family": rng.choice(["AB Compact Logix", "Siemens"], size=rows),
        "hmi_family": rng.choice(["AB PanelView Plus", "None"], size=rows),
        "vision_type": rng.choice(["None", "2D", "3D"], size=rows),
        "stations_count": rng.integers(1, 8, size=rows),
        "robot_count": rng.integers(0, 5, size=rows),
        "fixture_sets": rng.integers(0, 4, size=rows),
        "part_types": rng.integers(1, 5, size=rows),
        "servo_axes": rng.integers(0, 6, size=rows),
        "pneumatic_devices": rng.integers(0, 6, size=rows),
        "safety_doors": rng.integers(0, 4, size=rows),
        "weldment_perimeter_ft": rng.uniform(0, 200, size=rows),
        "fence_length_ft": rng.uniform(0, 150, size=rows),
        "conveyor_length_ft": rng.uniform(0, 80, size=rows),
        "product_familiarity_score": rng.integers(1, 6, size=rows),
        "product_rigidity": rng.integers(1, 6, size=rows),
        "is_product_deformable": rng.integers(0, 2, size=rows),
        "is_bulk_product": rng.integers(0, 2, size=rows),
        "bulk_rigidity_score": rng.integers(1, 6, size=rows),
        "has_tricky_packaging": rng.integers(0, 2, size=rows),
        "process_uncertainty_score": rng.integers(1, 6, size=rows),
        "changeover_time_min": rng.uniform(0, 120, size=rows),
        "safety_devices_count": rng.integers(0, 6, size=rows),
        "custom_pct": rng.integers(0, 100, size=rows),
        "duplicate": rng.integers(0, 2, size=rows),
        "has_controls": rng.integers(0, 2, size=rows),
        "has_robotics": rng.integers(0, 2, size=rows),
        "Retrofit": rng.integers(0, 2, size=rows),
        "complexity_score_1_5": rng.integers(1, 6, size=rows),
        "vision_systems_count": rng.integers(0, 3, size=rows),
        "panel_count": rng.integers(0, 5, size=rows),
        "drive_count": rng.integers(0, 5, size=rows),
        "stations_robot_index": rng.uniform(0, 10, size=rows),
        "mech_complexity_index": rng.uniform(0, 10, size=rows),
        "controls_complexity_index": rng.uniform(0, 10, size=rows),
        "physical_scale_index": rng.uniform(0, 10, size=rows),
        "log_quoted_materials_cost": rng.uniform(0, 10, size=rows),
    }

    base = (
        data["stations_count"]
        + data["robot_count"] * 8
        + data["mech_complexity_index"]
        + data["controls_complexity_index"] * 0.5
    )
    noise = rng.normal(0, 5, size=rows)
    data["me10_actual_hours"] = np.maximum(1, base + noise)
    return pd.DataFrame(data)


def main():
    df_raw = build_synthetic_df()
    df_train = engineer_features_for_training(df_raw)
    metrics = train_one_op(df_train, "me10_actual_hours")
    print("Training metrics:", metrics)

    model = load_model("me10_actual_hours")
    df_quote = prepare_quote_features(df_raw.head(1))
    p50, p10, p90, std = predict_with_interval(model, df_quote)
    tol = max(TOL_PCT * abs(p50[0]), TOL_MIN_OP_HOURS)
    within_prob = empirical_within_tol_prob(model, tol)

    print(f"Predicted p10/p50/p90/std: {p10[0]:.2f}, {p50[0]:.2f}, {p90[0]:.2f}, {std[0]:.2f}")
    print(f"Empirical within ±{tol:.1f}h probability:", within_prob)


if __name__ == "__main__":
    main()
