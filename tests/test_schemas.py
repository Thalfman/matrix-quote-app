"""Tests for core.schemas -- Pydantic data models."""

import pytest

from core.schemas import QuoteInput, OpPrediction, SalesBucketPrediction, QuotePrediction


class TestQuoteInput:
    """Tests for the QuoteInput schema."""

    def test_creation_with_defaults(self):
        """All categorical fields required; numeric fields have defaults."""
        qi = QuoteInput(
            industry_segment="Automotive",
            system_category="Assembly",
            automation_level="Full",
            plc_family="Allen-Bradley",
            hmi_family="PanelView",
            vision_type="None",
        )
        assert qi.stations_count == 0
        assert qi.robot_count == 0
        assert qi.has_controls == 1
        assert qi.has_robotics == 1
        assert qi.complexity_score_1_5 == 3
        assert qi.project_id is None

    def test_creation_with_all_fields(self):
        """All fields (including overridden defaults) are accepted."""
        qi = QuoteInput(
            project_id="PRJ-0001",
            industry_segment="Food & Bev",
            system_category="Palletizing",
            automation_level="Semi",
            plc_family="Siemens",
            hmi_family="KTP",
            vision_type="2D",
            stations_count=4,
            robot_count=2,
            fixture_sets=3,
            part_types=2,
            servo_axes=6,
            pneumatic_devices=5,
            safety_doors=2,
            weldment_perimeter_ft=25.0,
            fence_length_ft=80.0,
            conveyor_length_ft=40.0,
            product_familiarity_score=3.0,
            product_rigidity=4.0,
            is_product_deformable=0,
            is_bulk_product=1,
            bulk_rigidity_score=2.0,
            has_tricky_packaging=0,
            process_uncertainty_score=2.0,
            changeover_time_min=15.0,
            safety_devices_count=3.0,
            custom_pct=50.0,
            duplicate=0,
            has_controls=1,
            has_robotics=0,
            Retrofit=1,
            complexity_score_1_5=4.0,
            vision_systems_count=1.0,
            panel_count=2.0,
            drive_count=4.0,
            stations_robot_index=12.0,
            mech_complexity_index=11.0,
            controls_complexity_index=7.0,
            physical_scale_index=120.0,
            log_quoted_materials_cost=10.5,
        )
        assert qi.project_id == "PRJ-0001"
        assert qi.stations_count == 4
        assert qi.Retrofit == 1
        assert qi.log_quoted_materials_cost == 10.5

    def test_missing_required_field_raises(self):
        """Omitting a required categorical field raises a validation error."""
        with pytest.raises(Exception):
            QuoteInput(
                # industry_segment intentionally missing
                system_category="Assembly",
                automation_level="Full",
                plc_family="Allen-Bradley",
                hmi_family="PanelView",
                vision_type="None",
            )


class TestQuotePrediction:
    """Tests for the QuotePrediction schema."""

    def test_structure(self):
        """QuotePrediction holds ops dict and totals."""
        op = OpPrediction(
            p50=100.0, p10=80.0, p90=120.0, std=10.0, rel_width=0.4, confidence="medium"
        )
        bucket = SalesBucketPrediction(
            p50=100.0, p10=80.0, p90=120.0, rel_width=0.4, confidence="medium"
        )
        qp = QuotePrediction(
            ops={"me10": op},
            total_p50=100.0,
            total_p10=80.0,
            total_p90=120.0,
            sales_buckets={"ME": bucket},
        )
        assert "me10" in qp.ops
        assert qp.total_p50 == 100.0
        assert qp.ops["me10"].confidence == "medium"
        assert "ME" in qp.sales_buckets
        assert qp.sales_buckets["ME"].p50 == 100.0

    def test_empty_ops_allowed(self):
        """An empty ops dict is valid (no models trained yet)."""
        qp = QuotePrediction(
            ops={},
            total_p50=0.0,
            total_p10=0.0,
            total_p90=0.0,
        )
        assert qp.ops == {}
        assert qp.sales_buckets == {}
