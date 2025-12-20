# service/predict_lib.py
# Small library that exposes prediction functions for single and batch inputs.

from typing import Dict

import numpy as np
import pandas as pd

from core.config import TARGETS, QUOTE_NUM_FEATURES, QUOTE_CAT_FEATURES
from core.features import prepare_quote_features
from core.models import load_model, predict_target_with_interval
from core.schemas import QuoteInput, QuotePrediction, OpPrediction


def _quote_to_df(q: QuoteInput) -> pd.DataFrame:
    """Convert QuoteInput into a one-row DataFrame with the expected columns."""
    data = q.dict()
    cols = list(set(QUOTE_NUM_FEATURES + QUOTE_CAT_FEATURES))
    row = {c: data.get(c, None) for c in cols}
    df = pd.DataFrame([row])
    df = prepare_quote_features(df)
    return df


def _compute_rel_width(p10: float, p50: float, p90: float) -> float:
    """Compute relative interval width for reporting."""
    eps = 1e-6
    width = p90 - p10
    denom = max(abs(p50), eps)
    return width / denom


def _parse_confidence_pct(confidence_label: str) -> float | None:
    """Extract the numeric percent from a confidence label like '90% calibrated'."""
    if not confidence_label:
        return None
    percent = confidence_label.split("%", 1)[0].strip()
    if not percent.replace(".", "", 1).isdigit():
        return None
    return float(percent)


def predict_quote(q: QuoteInput) -> QuotePrediction:
    """
    Predict hours for a single project described by QuoteInput.
    Returns per-operation P10/P50/P90/std/confidence plus totals.
    """
    df = _quote_to_df(q)

    ops: Dict[str, OpPrediction] = {}
    total_p50 = total_p10 = total_p90 = 0.0

    for target in TARGETS:
        artifact = load_model(target)
        result = predict_target_with_interval(artifact, df)
        if result is None:
            p10 = p50 = p90 = std = 0.0
            rel_width = 0.0
            conf_label = "not trained"
            trained = False
            confidence_pct = None
        else:
            p10 = float(result["p10"][0])
            p50 = float(result["p50"][0])
            p90 = float(result["p90"][0])
            std = float(result["std"][0])
            rel_width = _compute_rel_width(p10, p50, p90)
            conf_label = result.get("confidence") or "90% calibrated"
            trained = True
            confidence_pct = (
                _parse_confidence_pct(result.get("confidence", ""))
                or _parse_confidence_pct(conf_label)
            )

        op_name = target.replace("_actual_hours", "")
        ops[op_name] = OpPrediction(
            p50=p50,
            p10=p10,
            p90=p90,
            std=std,
            rel_width=rel_width,
            trained=trained,
            confidence_pct=confidence_pct,
            confidence=conf_label,
        )

        total_p50 += p50
        total_p10 += p10
        total_p90 += p90

    return QuotePrediction(
        ops=ops,
        total_p50=float(total_p50),
        total_p10=float(total_p10),
        total_p90=float(total_p90),
    )


def predict_quotes_df(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Batch prediction for a DataFrame with quote-time columns.
    Adds per-operation P10/P50/P90/std and project totals.
    """
    df = prepare_quote_features(df_in)

    df["total_p50"] = 0.0
    df["total_p10"] = 0.0
    df["total_p90"] = 0.0

    for target in TARGETS:
        artifact = load_model(target)
        result = predict_target_with_interval(artifact, df)
        if result is None:
            p10_arr = p50_arr = p90_arr = std_arr = np.zeros(len(df))
            conf_arr = ["not trained"] * len(df)
            trained_arr = np.array([False] * len(df))
            conf_pct_arr = np.array([None] * len(df), dtype=object)
        else:
            p10_arr = result["p10"]
            p50_arr = result["p50"]
            p90_arr = result["p90"]
            std_arr = result["std"]
            confidence_label = result.get("confidence") or "90% calibrated"
            conf_arr = [confidence_label] * len(df)
            trained_arr = np.array([True] * len(df))
            conf_pct_arr = np.array(
                [
                    _parse_confidence_pct(result.get("confidence", ""))
                    or _parse_confidence_pct(conf_arr[0])
                ]
                * len(df),
                dtype=object,
            )

        op_name = target.replace("_actual_hours", "")
        df[f"{op_name}_p50"] = p50_arr
        df[f"{op_name}_p10"] = p10_arr
        df[f"{op_name}_p90"] = p90_arr
        df[f"{op_name}_std"] = std_arr
        df[f"{op_name}_confidence"] = conf_arr
        df[f"{op_name}_trained"] = trained_arr
        df[f"{op_name}_confidence_pct"] = conf_pct_arr

        df["total_p50"] += p50_arr
        df["total_p10"] += p10_arr
        df["total_p90"] += p90_arr

    return df
