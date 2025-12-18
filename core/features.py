# core/features.py
# Feature engineering: convert raw Excel into a training-ready / quote-ready DataFrame.

import numpy as np
import pandas as pd

from .config import QUOTE_NUM_FEATURES, QUOTE_CAT_FEATURES

MISSING_TOKEN = "__MISSING__"

# Columns that may be stored as "yes/no/true/false/0/1" but we want 0/1 ints.
_BOOL_STR_COLS = [
    "has_controls",
    "has_robotics",
    "duplicate",
    "Retrofit",
    "is_product_deformable",
    "is_bulk_product",
    "has_tricky_packaging",
]


def _to_bool01(series: pd.Series) -> pd.Series:
    """Map yes/no/true/false/1/0/etc. to 1 or 0."""
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({"yes": 1, "true": 1, "1": 1, "no": 0, "false": 0, "0": 0})
        .fillna(0)
        .astype(int)
    )


def _compute_indices_inplace(df: pd.DataFrame) -> None:
    """
    Compute composite indices he already uses (station/robot, mech, controls, physical).
    All modifications are in-place on the given df.
    """
    numeric_cols = [
        "stations_count",
        "robot_count",
        "servo_axes",
        "fixture_sets",
        "pneumatic_devices",
        "safety_devices_count",
        "vision_systems_count",
        "i_o_points_est",
        "conveyor_length_ft",
        "fence_length_ft",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Station/robot/axes combined index
    for c in ["stations_count", "robot_count", "servo_axes"]:
        if c not in df.columns:
            df[c] = 0.0
    df["stations_robot_index"] = (
        df["stations_count"].fillna(0)
        + df["robot_count"].fillna(0)
        + df["servo_axes"].fillna(0)
    )

    # Mechanical complexity index
    for c in ["fixture_sets", "pneumatic_devices", "safety_devices_count"]:
        if c not in df.columns:
            df[c] = 0.0
    df["mech_complexity_index"] = (
        df["fixture_sets"].fillna(0)
        + df["pneumatic_devices"].fillna(0)
        + df["safety_devices_count"].fillna(0)
    )

    # Controls complexity index
    for c in ["vision_systems_count", "i_o_points_est", "servo_axes"]:
        if c not in df.columns:
            df[c] = 0.0
    df["controls_complexity_index"] = (
        df["servo_axes"].fillna(0)
        + df["vision_systems_count"].fillna(0)
        + df["i_o_points_est"].fillna(0) / 75.0
    )

    # Physical scale index (conveyors + fencing)
    for c in ["conveyor_length_ft", "fence_length_ft"]:
        if c not in df.columns:
            df[c] = 0.0
    df["physical_scale_index"] = (
        df["conveyor_length_ft"].fillna(0)
        + df["fence_length_ft"].fillna(0)
    )


def _prep_categoricals_inplace(df: pd.DataFrame) -> None:
    """Ensure categorical columns are strings with a stable missing token."""
    for col in QUOTE_CAT_FEATURES:
        if col not in df.columns:
            continue
        df[col] = df[col].astype(str)
        df[col] = df[col].replace(["nan", "NaN", "None", ""], MISSING_TOKEN)
        df[col] = df[col].fillna(MISSING_TOKEN)


def engineer_features_for_training(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the project-hours dataset for training:
    - Filter to include_in_training + actuals.
    - Convert flags to 0/1.
    - Compute indices.
    - Compute log materials cost if needed.
    """
    df = df_raw.copy()

    if "include_in_training" in df.columns:
        df["include_in_training"] = (
            df["include_in_training"].astype(str).str.strip().str.lower()
        )
        df = df[df["include_in_training"].isin(["yes", "1", "true"])]

    if "dataset_role" in df.columns:
        df["dataset_role"] = (
            df["dataset_role"].astype(str).str.strip().str.lower()
        )
        df = df[df["dataset_role"] == "actuals"].copy()

    for col in _BOOL_STR_COLS:
        if col in df.columns:
            df[col] = _to_bool01(df[col])

    # Force numeric on the quote-time numeric features (if present)
    for col in QUOTE_NUM_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    _compute_indices_inplace(df)
    _prep_categoricals_inplace(df)

    # If log cost is missing, derive it from quoted_materials_cost if possible.
    if (
        "log_quoted_materials_cost" not in df.columns
        or df["log_quoted_materials_cost"].isna().all()
    ):
        if "quoted_materials_cost" in df.columns:
            raw = (
                df["quoted_materials_cost"]
                .astype(str)
                .replace(r"[\$,]", "", regex=True)
            )
            raw = pd.to_numeric(raw, errors="coerce").fillna(0)
            df["log_quoted_materials_cost"] = np.log1p(raw)
        else:
            df["log_quoted_materials_cost"] = 0.0

    return df


def prepare_quote_features(df_quote: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the same basic transformations to quote-time inputs:
    - Flags to 0/1.
    - Numeric coercion.
    - Indices.
    - log_quoted_materials_cost.
    """
    df = df_quote.copy()

    for col in _BOOL_STR_COLS:
        if col in df.columns:
            df[col] = _to_bool01(df[col])

    for col in QUOTE_NUM_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    _compute_indices_inplace(df)
    _prep_categoricals_inplace(df)

    if (
        "log_quoted_materials_cost" not in df.columns
        or df["log_quoted_materials_cost"].isna().all()
    ):
        if "quoted_materials_cost" in df.columns:
            raw = (
                df["quoted_materials_cost"]
                .astype(str)
                .replace(r"[\$,]", "", regex=True)
            )
            raw = pd.to_numeric(raw, errors="coerce").fillna(0)
            df["log_quoted_materials_cost"] = np.log1p(raw)
        else:
            df["log_quoted_materials_cost"] = 0.0

    return df


def build_training_data(df_master: pd.DataFrame, target_col: str):
    """
    Build X, y for one operation's model, using only quote-time features.
    Returns (X, y, num_features, cat_features, subset_df) or Nones if not enough data.
    """
    if target_col not in df_master.columns:
        return None, None, None, None, None

    df = df_master.copy()
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    sub = df[df[target_col] > 0].copy()

    if len(sub) < 5:
        return None, None, None, None, None

    num_features = [
        c
        for c in QUOTE_NUM_FEATURES
        if c in sub.columns and not sub[c].isna().all()
    ]
    cat_features = [
        c
        for c in QUOTE_CAT_FEATURES
        if c in sub.columns and not sub[c].isna().all()
    ]

    X = sub[num_features + cat_features]
    y = sub[target_col]

    return X, y, num_features, cat_features, sub
