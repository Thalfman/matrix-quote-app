# core/config.py
# Central configuration for model targets, feature lists, and confidence levels.
# This file is the single source of truth for what the models predict and which
# features they use. Change these lists to add new operations or features.

# ── Model targets ──
# Each entry is a column in the training dataset containing actual logged hours
# for one engineering operation. The prefix (e.g. "me10", "ee20") is the internal
# department code; the model trains one CatBoost regressor per target.
TARGETS = [
    "me10_actual_hours",    # Mechanical engineering (design)
    "me15_actual_hours",    # Mechanical engineering (detailing)
    "me230_actual_hours",   # Mechanical engineering (other)
    "ee20_actual_hours",    # Electrical engineering
    "rb30_actual_hours",    # Robotics programming
    "cp50_actual_hours",    # Controls programming
    "bld100_actual_hours",  # Build / assembly
    "shp150_actual_hours",  # Shipping
    "inst160_actual_hours", # Installation
    "trv180_actual_hours",  # Travel
    "doc190_actual_hours",  # Documentation
    "pm200_actual_hours",   # Project management
]

# ── Sales buckets ──
# Operations are rolled up into these higher-level categories for the Sales team.
SALES_BUCKETS = [
    "ME",
    "EE",
    "PM",
    "Docs",
    "Build",
    "Robot",
    "Controls",
    "Install",
    "Travel",
]

# Maps each operation prefix to its parent Sales bucket. Multiple operations can
# roll up into the same bucket (e.g. me10, me15, me230 all -> "ME").
SALES_BUCKET_MAP = {
    "me10": "ME",
    "me15": "ME",
    "me230": "ME",
    "ee20": "EE",
    "rb30": "Robot",
    "cp50": "Controls",
    "bld100": "Build",
    "shp150": "Build",
    "inst160": "Install",
    "trv180": "Travel",
    "doc190": "Docs",
    "pm200": "PM",
}

# Tolerance rules for "within ±T" confidence derived from held-out data.
TOL_PCT = 0.10
TOL_MIN_OP_HOURS = 5.0
TOL_MIN_TOTAL_HOURS = 10.0

# Supported confidence levels for prediction intervals (lo/hi bounds).
CONFIDENCE_LEVELS = [0.80, 0.90, 0.95]
DEFAULT_CONFIDENCE = 0.90

# ── Quote-time features ──
# These are the only features the models are allowed to see. They must all be
# known (or estimable) at the time a quote is being prepared — no actuals allowed.
QUOTE_NUM_FEATURES = [
    # Raw project specifications (entered by the user)
    "stations_count",
    "robot_count",
    "fixture_sets",
    "part_types",
    "servo_axes",
    "pneumatic_devices",
    "safety_doors",
    "weldment_perimeter_ft",
    "fence_length_ft",
    "conveyor_length_ft",
    "product_familiarity_score",
    "product_rigidity",
    "is_product_deformable",
    "is_bulk_product",
    "bulk_rigidity_score",
    "has_tricky_packaging",
    "process_uncertainty_score",
    "changeover_time_min",
    "safety_devices_count",
    "custom_pct",
    "duplicate",
    "has_controls",
    "has_robotics",
    "Retrofit",
    "complexity_score_1_5",
    "vision_systems_count",
    "panel_count",
    "drive_count",
    # Derived composite indices (auto-computed by _compute_indices_inplace in features.py)
    "stations_robot_index",
    "mech_complexity_index",
    "controls_complexity_index",
    "physical_scale_index",
    # Log-transformed materials cost (derived from quoted_materials_cost)
    "log_quoted_materials_cost",
]

# Categorical features that are known at quote time.
QUOTE_CAT_FEATURES = [
    "industry_segment",
    "system_category",
    "automation_level",
    "plc_family",
    "hmi_family",
    "vision_type",
]

# Minimal columns that must exist in the project-hours dataset before we can train.
REQUIRED_TRAINING_COLS = [
    "project_id",
    "include_in_training",
    "dataset_role",
    "industry_segment",
    "system_category",
    "stations_count",
    "robot_count",
    "me10_actual_hours",  # at least one hours column is required
]
