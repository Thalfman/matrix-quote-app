"""Tests for core.features -- feature engineering and validation."""

import numpy as np
import pandas as pd
import pytest

from core.config import QUOTE_NUM_FEATURES, QUOTE_CAT_FEATURES, TARGETS
from core.features import (
    engineer_features_for_training,
    prepare_quote_features,
    validate_training_data,
    validate_batch_input,
    build_training_data,
)


# ---------------------------------------------------------------------------
# engineer_features_for_training
# ---------------------------------------------------------------------------


class TestEngineerFeaturesForTraining:
    """Tests for engineer_features_for_training."""

    def test_filters_include_in_training(self, synthetic_training_df):
        """Rows with include_in_training != yes are excluded."""
        df = synthetic_training_df.copy()
        df.loc[0, "include_in_training"] = "no"
        df.loc[1, "include_in_training"] = "no"
        result = engineer_features_for_training(df)
        assert len(result) == len(df) - 2

    def test_filters_dataset_role(self, synthetic_training_df):
        """Rows with dataset_role != actuals are excluded."""
        df = synthetic_training_df.copy()
        df.loc[0, "dataset_role"] = "budget"
        result = engineer_features_for_training(df)
        assert len(result) == len(df) - 1

    def test_computes_indices(self, synthetic_training_df):
        """Composite indices are added to the output."""
        result = engineer_features_for_training(synthetic_training_df)
        for col in [
            "stations_robot_index",
            "mech_complexity_index",
            "controls_complexity_index",
            "physical_scale_index",
        ]:
            assert col in result.columns

    def test_converts_bool_columns(self, synthetic_training_df):
        """Boolean-like columns are converted to 0/1 ints."""
        df = synthetic_training_df.copy()
        df["has_controls"] = "yes"
        df["has_robotics"] = "no"
        result = engineer_features_for_training(df)
        assert set(result["has_controls"].unique()).issubset({0, 1})
        assert set(result["has_robotics"].unique()).issubset({0, 1})

    def test_all_rows_pass_with_clean_data(self, synthetic_training_df):
        """With clean fixture data all 20 rows survive filtering."""
        result = engineer_features_for_training(synthetic_training_df)
        assert len(result) == len(synthetic_training_df)


# ---------------------------------------------------------------------------
# prepare_quote_features
# ---------------------------------------------------------------------------


class TestPrepareQuoteFeatures:
    """Tests for prepare_quote_features."""

    def test_computes_indices(self, synthetic_training_df):
        """Composite indices are computed on quote-time input."""
        result = prepare_quote_features(synthetic_training_df)
        assert "stations_robot_index" in result.columns
        assert "mech_complexity_index" in result.columns

    def test_handles_numeric_coercion(self):
        """String-valued numeric cols are coerced to numeric types."""
        df = pd.DataFrame(
            {
                "stations_count": ["3", "5"],
                "robot_count": ["1", "bad"],
                "industry_segment": ["Automotive", "Pharma"],
                "system_category": ["Assembly", "Packaging"],
            }
        )
        result = prepare_quote_features(df)
        assert np.issubdtype(result["stations_count"].dtype, np.number)
        # "bad" should become NaN
        assert pd.isna(result["robot_count"].iloc[1])

    def test_log_cost_derived(self):
        """log_quoted_materials_cost is derived from quoted_materials_cost
        when the log column is missing."""
        df = pd.DataFrame(
            {
                "stations_count": [3],
                "robot_count": [1],
                "quoted_materials_cost": ["$50,000"],
            }
        )
        result = prepare_quote_features(df)
        assert "log_quoted_materials_cost" in result.columns
        assert result["log_quoted_materials_cost"].iloc[0] > 0


# ---------------------------------------------------------------------------
# validate_training_data
# ---------------------------------------------------------------------------


class TestValidateTrainingData:
    """Tests for validate_training_data."""

    def test_clean_data_no_errors_or_warnings(self, synthetic_training_df):
        """Clean synthetic data produces zero errors and zero warnings."""
        result = validate_training_data(synthetic_training_df)
        assert result["errors"] == []
        assert result["warnings"] == []
        assert result["stats"]["row_count"] == len(synthetic_training_df)

    def test_negative_hours_produce_error(self, synthetic_training_df):
        """Negative hour values are blocking errors."""
        df = synthetic_training_df.copy()
        df.loc[0, "me10_actual_hours"] = -10
        result = validate_training_data(df)
        assert len(result["errors"]) > 0
        assert "negative" in result["errors"][0].lower()

    def test_duplicate_project_ids_produce_warning(self, synthetic_training_df):
        """Duplicate project_id values generate a warning."""
        df = synthetic_training_df.copy()
        df.loc[1, "project_id"] = df.loc[0, "project_id"]
        result = validate_training_data(df)
        assert len(result["warnings"]) > 0
        assert "duplicate" in result["warnings"][0].lower()


# ---------------------------------------------------------------------------
# validate_batch_input
# ---------------------------------------------------------------------------


class TestValidateBatchInput:
    """Tests for validate_batch_input."""

    def test_clean_data_no_warnings(self, synthetic_training_df):
        """Clean data produces empty warnings and errors."""
        result = validate_batch_input(synthetic_training_df)
        assert result["errors"] == []
        assert result["warnings"] == []

    def test_non_numeric_values_produce_warning(self, synthetic_training_df):
        """Non-numeric values in numeric columns trigger a warning."""
        df = synthetic_training_df.copy()
        df.loc[0, "stations_count"] = "INVALID"
        df.loc[1, "robot_count"] = "N/A"
        result = validate_batch_input(df)
        assert len(result["warnings"]) > 0
        assert "non-numeric" in result["warnings"][0].lower()
        assert result["stats"]["non_numeric_values"] == 2


# ---------------------------------------------------------------------------
# build_training_data
# ---------------------------------------------------------------------------


class TestBuildTrainingData:
    """Tests for build_training_data."""

    def test_returns_correct_shapes(self, synthetic_training_df):
        """X and y have matching row counts and expected column types."""
        df_eng = engineer_features_for_training(synthetic_training_df)
        target = "me10_actual_hours"
        X, y, num_f, cat_f, sub = build_training_data(df_eng, target)
        assert X is not None
        assert len(X) == len(y)
        assert len(num_f) > 0
        assert len(cat_f) > 0
        assert X.shape[1] == len(num_f) + len(cat_f)

    def test_returns_none_for_insufficient_data(self):
        """Fewer than 5 non-zero rows returns all Nones."""
        df = pd.DataFrame(
            {
                "me10_actual_hours": [0, 0, 0, 1, 0],
                "stations_count": [1, 2, 3, 4, 5],
                "industry_segment": ["A"] * 5,
            }
        )
        X, y, nf, cf, sub = build_training_data(df, "me10_actual_hours")
        assert X is None
        assert y is None

    def test_returns_none_for_missing_target(self, synthetic_training_df):
        """A target column not present in the DataFrame returns Nones."""
        df_eng = engineer_features_for_training(synthetic_training_df)
        X, y, nf, cf, sub = build_training_data(df_eng, "nonexistent_target")
        assert X is None
