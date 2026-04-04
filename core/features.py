# core/features.py
# Feature engineering: convert raw Excel uploads into training-ready or quote-ready
# DataFrames. Both the training pipeline and the prediction pipeline run through
# the same transforms (_apply_common_transforms) to ensure consistency.

import numpy as np
import pandas as pd

from .config import QUOTE_NUM_FEATURES, QUOTE_CAT_FEATURES

# Boolean-like columns that Excel may store as "yes"/"no"/"true"/"false"/0/1 strings.
# We normalize them to integer 0 or 1 so CatBoost treats them as numeric flags.
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


# ── Derived composite indices ──
# These are engineered features that combine multiple raw inputs into single
# complexity scores. They are auto-computed (overwritten) on every call so the
# model always sees values consistent with the underlying raw features.
def _compute_indices_inplace(df: pd.DataFrame) -> None:
    """
    Compute composite indices (station/robot, mech, controls, physical).
    All modifications are in-place on the given df.
    """
    # Coerce index-component columns to numeric (they may arrive as strings from Excel)
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

    # stations_robot_index: overall system size (stations + robots + servo axes)
    for c in ["stations_count", "robot_count", "servo_axes"]:
        if c not in df.columns:
            df[c] = 0.0
    df["stations_robot_index"] = (
        df["stations_count"].fillna(0)
        + df["robot_count"].fillna(0)
        + df["servo_axes"].fillna(0)
    )

    # mech_complexity_index: mechanical tooling complexity (fixtures + pneumatics + safety)
    for c in ["fixture_sets", "pneumatic_devices", "safety_devices_count"]:
        if c not in df.columns:
            df[c] = 0.0
    df["mech_complexity_index"] = (
        df["fixture_sets"].fillna(0)
        + df["pneumatic_devices"].fillna(0)
        + df["safety_devices_count"].fillna(0)
    )

    # controls_complexity_index: electrical/controls complexity
    # i_o_points_est is divided by 75 to normalize it to a similar scale as the other terms
    for c in ["vision_systems_count", "i_o_points_est", "servo_axes"]:
        if c not in df.columns:
            df[c] = 0.0
    df["controls_complexity_index"] = (
        df["servo_axes"].fillna(0)
        + df["vision_systems_count"].fillna(0)
        + df["i_o_points_est"].fillna(0) / 75.0
    )

    # physical_scale_index: physical footprint of the system (conveyor + fence length)
    for c in ["conveyor_length_ft", "fence_length_ft"]:
        if c not in df.columns:
            df[c] = 0.0
    df["physical_scale_index"] = (
        df["conveyor_length_ft"].fillna(0)
        + df["fence_length_ft"].fillna(0)
    )


# ── Shared transform pipeline ──
# Both training and quote-time data go through this exact same sequence of
# transformations so the model sees identically prepared features at train and predict time.
def _apply_common_transforms(df: pd.DataFrame) -> pd.DataFrame:
    """
    Shared transformations for both training and quote-time data:
    - Convert boolean string columns to 0/1.
    - Coerce numeric features.
    - Compute composite indices.
    - Derive log_quoted_materials_cost if missing.
    """
    # Step 1: Normalize boolean-like columns (e.g. "yes" -> 1, "no" -> 0)
    for col in _BOOL_STR_COLS:
        if col in df.columns:
            df[col] = _to_bool01(df[col])

    # Step 2: Force all quote-time numeric features to numeric dtype
    # (handles strings, currency formatting, etc. from raw Excel)
    for col in QUOTE_NUM_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Step 3: Compute/overwrite the 4 derived composite indices from raw components
    _compute_indices_inplace(df)

    # Step 4: Derive log(1 + materials_cost) if not already present.
    # The log transform reduces skew in the cost feature for better model performance.
    if (
        "log_quoted_materials_cost" not in df.columns
        or df["log_quoted_materials_cost"].isna().all()
    ):
        if "quoted_materials_cost" in df.columns:
            # Strip dollar signs and commas before converting to numeric
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


# ── Training entry point ──
# Filters the raw upload to only rows flagged for training with actual hours,
# then applies the shared feature transforms.
def engineer_features_for_training(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the project-hours dataset for training:
    - Filter to include_in_training + actuals.
    - Apply common feature transforms.
    """
    df = df_raw.copy()

    # Keep only rows explicitly flagged for training (the client marks these in Excel)
    if "include_in_training" in df.columns:
        df["include_in_training"] = (
            df["include_in_training"].astype(str).str.strip().str.lower()
        )
        df = df[df["include_in_training"].isin(["yes", "1", "true"])]

    # Keep only rows with actual (not estimated/quoted) hours
    if "dataset_role" in df.columns:
        df["dataset_role"] = (
            df["dataset_role"].astype(str).str.strip().str.lower()
        )
        df = df[df["dataset_role"] == "actuals"].copy()

    return _apply_common_transforms(df)


# ── Quote-time entry point ──
# Applies the same transforms as training but without any row filtering,
# since at quote time every row in the input should be predicted on.
def prepare_quote_features(df_quote: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the same feature transforms to quote-time inputs.
    """
    df = df_quote.copy()
    return _apply_common_transforms(df)


# ── Build X/y for one operation ──
# Called by train_one_op() for each of the 12 target operations (e.g. me10_actual_hours).
# Filters to rows that have non-zero hours for this specific operation and selects
# only the features from QUOTE_NUM_FEATURES / QUOTE_CAT_FEATURES that are present.
def build_training_data(df_master: pd.DataFrame, target_col: str):
    """
    Build X, y for one operation's model, using only quote-time features.
    Returns (X, y, num_features, cat_features, subset_df) or Nones if not enough data.
    """
    if target_col not in df_master.columns:
        return None, None, None, None, None

    df = df_master.copy()
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    # Only train on rows where this operation actually had hours logged (> 0)
    sub = df[df[target_col] > 0].copy()

    # Require at least 5 rows to train a meaningful model for this operation
    if len(sub) < 5:
        return None, None, None, None, None

    # Select only the features that exist in this dataset and have at least one non-null value
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
