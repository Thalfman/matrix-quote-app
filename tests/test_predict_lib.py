"""Integration tests for service.predict_lib."""

import pandas as pd
import pytest

from core.config import TARGETS, SALES_BUCKETS, SALES_BUCKET_MAP
from core.schemas import QuoteInput, QuotePrediction
from service.predict_lib import predict_quote, predict_quotes_df


class TestPredictQuote:
    """Tests for predict_quote (single-project prediction)."""

    def test_returns_quote_prediction(self, trained_models, tmp_models_dir, monkeypatch):
        """predict_quote returns a QuotePrediction with ops and totals."""
        # Monkeypatch load_model so it looks in tmp_models_dir
        import core.models as _models_mod
        _original_load = _models_mod.load_model
        monkeypatch.setattr(
            "service.predict_lib.load_model",
            lambda target, version="v1", models_dir="models": _original_load(
                target, version=version, models_dir=tmp_models_dir
            ),
        )

        qi = QuoteInput(
            industry_segment="Automotive",
            system_category="Assembly",
            automation_level="Full",
            plc_family="Allen-Bradley",
            hmi_family="PanelView",
            vision_type="2D",
            stations_count=4,
            robot_count=2,
        )
        result = predict_quote(qi)
        assert isinstance(result, QuotePrediction)
        assert len(result.ops) > 0
        assert result.total_p50 > 0

        # Check that sales_buckets are populated
        assert len(result.sales_buckets) > 0

        # Every predicted op should map to a known sales bucket
        for op_name in result.ops:
            assert op_name in SALES_BUCKET_MAP


class TestPredictQuotesDf:
    """Tests for predict_quotes_df (batch prediction)."""

    def test_returns_dataframe_with_prediction_columns(
        self, trained_models, tmp_models_dir, monkeypatch
    ):
        """predict_quotes_df adds per-op and rollup columns to the DataFrame."""
        import core.models as _models_mod
        _original_load = _models_mod.load_model
        monkeypatch.setattr(
            "service.predict_lib.load_model",
            lambda target, version="v1", models_dir="models": _original_load(
                target, version=version, models_dir=tmp_models_dir
            ),
        )

        # Build input rows from QuoteInput.model_dump() to ensure all
        # columns the trained preprocessor expects are present.
        row1 = QuoteInput(
            industry_segment="Automotive",
            system_category="Assembly",
            automation_level="Full",
            plc_family="Allen-Bradley",
            hmi_family="PanelView",
            vision_type="2D",
            stations_count=4,
            robot_count=2,
        ).model_dump()
        row2 = QuoteInput(
            industry_segment="Pharma",
            system_category="Packaging",
            automation_level="Semi",
            plc_family="Siemens",
            hmi_family="KTP",
            vision_type="None",
            stations_count=2,
            robot_count=1,
        ).model_dump()
        df_in = pd.DataFrame([row1, row2])
        result = predict_quotes_df(df_in)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2

        # Per-op prediction columns should exist
        assert "me10_p50" in result.columns
        assert "me10_p10" in result.columns
        assert "me10_p90" in result.columns
        assert "me10_std" in result.columns

        # Totals should exist
        assert "total_p50" in result.columns
        assert "total_p10" in result.columns
        assert "total_p90" in result.columns

        # At least some bucket rollup columns should exist
        bucket_cols = [
            c for c in result.columns if any(c.startswith(b + "_") for b in SALES_BUCKETS)
        ]
        assert len(bucket_cols) > 0
