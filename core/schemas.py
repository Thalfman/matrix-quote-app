# core/schemas.py
# Pydantic models for structured inputs/outputs.

from typing import Dict, Optional

from pydantic import BaseModel, Field


class QuoteInput(BaseModel):
    """Fields that describe a project at quote time (used for prediction)."""

    project_id: Optional[str] = None

    industry_segment: str
    system_category: str
    automation_level: str
    plc_family: str
    hmi_family: str
    vision_type: str

    stations_count: float = 0
    robot_count: float = 0
    fixture_sets: float = 0
    part_types: float = 0
    servo_axes: float = 0
    pneumatic_devices: float = 0
    safety_doors: float = 0
    weldment_perimeter_ft: float = 0
    fence_length_ft: float = 0
    conveyor_length_ft: float = 0
    product_familiarity_score: float = 0
    product_rigidity: float = 0
    is_product_deformable: int = 0
    is_bulk_product: int = 0
    bulk_rigidity_score: float = 0
    has_tricky_packaging: int = 0
    process_uncertainty_score: float = 0
    changeover_time_min: float = 0
    safety_devices_count: float = 0
    custom_pct: float = 0
    duplicate: int = 0
    has_controls: int = 1
    has_robotics: int = 1
    Retrofit: int = 0
    complexity_score_1_5: float = 3
    vision_systems_count: float = 0
    panel_count: float = 0
    drive_count: float = 0
    stations_robot_index: float = 0
    mech_complexity_index: float = 0
    controls_complexity_index: float = 0
    physical_scale_index: float = 0
    log_quoted_materials_cost: float = 0


class OpPrediction(BaseModel):
    """Prediction output for a single operation."""

    estimate: float = Field(..., description="Point prediction (median hours)")
    lo: float = Field(..., description="Calibrated lower bound at selected confidence")
    hi: float = Field(..., description="Calibrated upper bound at selected confidence")
    plus_minus: float = Field(
        ..., description="Half-width of calibrated interval (max side)"
    )
    confidence: float = Field(..., description="Selected confidence level (0-1)")


class SalesBucketPrediction(BaseModel):
    """Aggregated prediction output for a Sales bucket."""

    estimate: float = Field(..., description="Point prediction across bucket")
    lo: float = Field(..., description="Calibrated lower bound for bucket")
    hi: float = Field(..., description="Calibrated upper bound for bucket")
    plus_minus: float = Field(
        ..., description="Half-width of calibrated bucket interval"
    )
    confidence: float = Field(..., description="Selected confidence level (0-1)")


class QuotePrediction(BaseModel):
    """All operation predictions plus project totals."""

    ops: Dict[str, OpPrediction]
    total_estimate: float
    total_lo: float
    total_hi: float
    total_plus_minus: float
    confidence: float = Field(..., description="Selected confidence level (0-1)")
    sales_buckets: Dict[str, SalesBucketPrediction] = Field(
        default_factory=dict,
        description="Rollups of operation predictions by Sales bucket",
    )
