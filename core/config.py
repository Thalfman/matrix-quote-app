# core/config.py
# Central place for model targets and feature lists.

# Columns containing actual hours per operation.
TARGETS = [
    "me10_actual_hours",
    "me15_actual_hours",
    "me230_actual_hours",
    "ee20_actual_hours",
    "rb30_actual_hours",
    "cp50_actual_hours",
    "bld100_actual_hours",
    "shp150_actual_hours",
    "inst160_actual_hours",
    "trv180_actual_hours",
    "doc190_actual_hours",
    "pm200_actual_hours",
]

# Numeric features that should be available (or reasonably estimable) at quote time.
# These are the only numeric inputs the quote-time models are allowed to use.
QUOTE_NUM_FEATURES = [
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
    "stations_robot_index",
    "mech_complexity_index",
    "controls_complexity_index",
    "physical_scale_index",
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
