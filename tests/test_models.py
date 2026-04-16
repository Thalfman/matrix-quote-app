"""Tests for core.models -- model training, loading, and prediction."""

import os

import numpy as np
import pandas as pd
import pytest

from core.config import TARGETS
from core.features import engineer_features_for_training, build_training_data
from core.models import train_one_op, load_model, predict_with_interval


class TestTrainOneOp:
    """Tests for train_one_op."""

    def test_trains_and_returns_metrics(self, synthetic_training_df, tmp_models_dir):
        """Training succeeds and returns a dict with expected metric keys."""
        df_eng = engineer_features_for_training(synthetic_training_df)
        target = "me10_actual_hours"
        metrics = train_one_op(df_eng, target, models_dir=tmp_models_dir)
        assert metrics is not None
        for key in ("target", "version", "rows", "mae", "r2", "model_path"):
            assert key in metrics, f"Missing key: {key}"
        assert metrics["target"] == target
        assert metrics["rows"] > 0

    def test_saves_joblib_file(self, synthetic_training_df, tmp_models_dir):
        """A .joblib file is created in the models directory."""
        df_eng = engineer_features_for_training(synthetic_training_df)
        target = "me10_actual_hours"
        metrics = train_one_op(df_eng, target, models_dir=tmp_models_dir)
        assert os.path.exists(metrics["model_path"])
        assert metrics["model_path"].endswith(".joblib")

    def test_returns_none_for_insufficient_data(self, tmp_models_dir):
        """If there is not enough data, train_one_op returns None."""
        df = pd.DataFrame(
            {
                "me10_actual_hours": [0, 0, 0, 1, 0],
                "stations_count": [1, 2, 3, 4, 5],
                "industry_segment": ["A"] * 5,
            }
        )
        result = train_one_op(df, "me10_actual_hours", models_dir=tmp_models_dir)
        assert result is None


class TestLoadModel:
    """Tests for load_model."""

    def test_loads_trained_model(self, trained_models, tmp_models_dir):
        """A previously trained model can be loaded."""
        target = "me10_actual_hours"
        pipe = load_model(target, models_dir=tmp_models_dir)
        assert pipe is not None

    def test_returns_none_for_missing_model(self, tmp_models_dir):
        """Returns None when no model file exists on disk."""
        pipe = load_model("nonexistent_target", models_dir=tmp_models_dir)
        assert pipe is None


class TestPredictWithInterval:
    """Tests for predict_with_interval."""

    def test_returns_correct_shape(self, synthetic_training_df, trained_models, tmp_models_dir):
        """p50, p10, p90, std are arrays with one element per input row."""
        df_eng = engineer_features_for_training(synthetic_training_df)
        target = "me10_actual_hours"
        X, y, num_f, cat_f, sub = build_training_data(df_eng, target)
        pipe = load_model(target, models_dir=tmp_models_dir)
        assert pipe is not None

        p50, p10, p90, std = predict_with_interval(pipe, X)
        assert len(p50) == len(X)
        assert len(p10) == len(X)
        assert len(p90) == len(X)
        assert len(std) == len(X)

    def test_p10_leq_p50_leq_p90(self, synthetic_training_df, trained_models, tmp_models_dir):
        """p10 <= p50 <= p90 for every row (basic sanity)."""
        df_eng = engineer_features_for_training(synthetic_training_df)
        target = "me10_actual_hours"
        X, y, num_f, cat_f, sub = build_training_data(df_eng, target)
        pipe = load_model(target, models_dir=tmp_models_dir)

        p50, p10, p90, std = predict_with_interval(pipe, X)
        assert np.all(p10 <= p50 + 1e-6)
        assert np.all(p50 <= p90 + 1e-6)
