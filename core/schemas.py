# core/schemas.py
# Pydantic models for structured inputs/outputs.

from typing import Dict, Optional, List

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

    estimate: float = Field(..., description="Point prediction for hours")
    low: float = Field(..., description="Lower bound of conformal PI")
    high: float = Field(..., description="Upper bound of conformal PI")
    confidence: float = Field(..., description="Requested prediction interval coverage")
    calib_n: int = Field(..., description="Calibration rows used to build conformal scores")


class SalesBucketPrediction(BaseModel):
    """Aggregated prediction output for a Sales bucket (heuristic sum)."""

    estimate: float = Field(..., description="Summed point estimates across bucket")
    low: float = Field(..., description="Summed lower bounds (heuristic)")
    high: float = Field(..., description="Summed upper bounds (heuristic)")
    confidence: float = Field(..., description="Requested coverage passed through")


class QuotePrediction(BaseModel):
    """All operation predictions plus project totals."""

    ops: Dict[str, OpPrediction]
    total_estimate: float
    total_low: float
    total_high: float
    confidence: float
    sales_buckets: Dict[str, SalesBucketPrediction] = Field(
        default_factory=dict,
        description="Rollups of operation predictions by Sales bucket (heuristic sums)",
    )
    missing_models: List[str] = Field(
        default_factory=list,
        description="Operations with no trained model available",
    )
