"""
service/predict_lib.py
Prediction helpers for single and batch quotes with calibrated intervals.
"""

from typing import Dict

import numpy as np
import pandas as pd

from core.config import (
    CONFIDENCE_LEVELS,
    DEFAULT_CONFIDENCE,
    QUOTE_CAT_FEATURES,
    QUOTE_NUM_FEATURES,
    SALES_BUCKETS,
    SALES_BUCKET_MAP,
    TARGETS,
)
from core.features import prepare_quote_features
from core.models import load_model, predict_with_interval
from core.schemas import (
    OpPrediction,
    QuoteInput,
    QuotePrediction,
    SalesBucketPrediction,
)


def _quote_to_df(q: QuoteInput) -> pd.DataFrame:
    """Convert QuoteInput into a one-row DataFrame with the expected columns."""
    data = q.dict()
    cols = list(set(QUOTE_NUM_FEATURES + QUOTE_CAT_FEATURES))
    row = {c: data.get(c, None) for c in cols}
    df = pd.DataFrame([row])
    df = prepare_quote_features(df)
    return df


def _validate_confidence(confidence_level: float) -> float:
    if confidence_level not in CONFIDENCE_LEVELS:
        raise ValueError(
            f"Confidence level {confidence_level} not supported. "
            f"Choose from {CONFIDENCE_LEVELS}."
        )
    return confidence_level


def predict_quote(
    q: QuoteInput, confidence_level: float = DEFAULT_CONFIDENCE
) -> QuotePrediction:
    """
    Predict hours for a single project described by QuoteInput.
    Returns per-operation calibrated intervals + Sales/total rollups.
    """
    confidence_level = _validate_confidence(confidence_level)
    df = _quote_to_df(q)

    ops: Dict[str, OpPrediction] = {}
    bucket_totals = {
        bucket: {"estimate": 0.0, "lo": 0.0, "hi": 0.0} for bucket in SALES_BUCKETS
    }
    total_estimate = total_lo = total_hi = 0.0

    for target in TARGETS:
        model_obj = load_model(target)
        preds = predict_with_interval(model_obj, df, confidence_level)

        estimate = float(preds["estimate"][0])
        lo = float(preds["lo"][0])
        hi = float(preds["hi"][0])
        plus_minus = float(preds["plus_minus"][0])

        op_name = target.replace("_actual_hours", "")
        ops[op_name] = OpPrediction(
            estimate=estimate,
            lo=lo,
            hi=hi,
            plus_minus=plus_minus,
            confidence=confidence_level,
        )

        bucket = SALES_BUCKET_MAP.get(op_name)
        if bucket in bucket_totals:
            bucket_totals[bucket]["estimate"] += estimate
            bucket_totals[bucket]["lo"] += lo
            bucket_totals[bucket]["hi"] += hi

        total_estimate += estimate
        total_lo += lo
        total_hi += hi

    sales_buckets: Dict[str, SalesBucketPrediction] = {}
    for bucket in SALES_BUCKETS:
        totals = bucket_totals.get(bucket, {"estimate": 0.0, "lo": 0.0, "hi": 0.0})
        bucket_plus_minus = max(
            totals["estimate"] - totals["lo"], totals["hi"] - totals["estimate"]
        )
        sales_buckets[bucket] = SalesBucketPrediction(
            estimate=float(totals["estimate"]),
            lo=float(totals["lo"]),
            hi=float(totals["hi"]),
            plus_minus=float(bucket_plus_minus),
            confidence=confidence_level,
        )

    total_plus_minus = max(total_estimate - total_lo, total_hi - total_estimate)

    return QuotePrediction(
        ops=ops,
        total_estimate=float(total_estimate),
        total_lo=float(total_lo),
        total_hi=float(total_hi),
        total_plus_minus=float(total_plus_minus),
        confidence=confidence_level,
        sales_buckets=sales_buckets,
    )


def predict_quotes_df(
    df_in: pd.DataFrame, confidence_level: float = DEFAULT_CONFIDENCE
) -> pd.DataFrame:
    """
    Batch prediction for a DataFrame with quote-time columns.
    Adds per-operation calibrated intervals, Sales rollups, and project totals.
    """
    confidence_level = _validate_confidence(confidence_level)
    df = prepare_quote_features(df_in)
    df["confidence_level"] = confidence_level

    df["total_estimate"] = 0.0
    df["total_lo"] = 0.0
    df["total_hi"] = 0.0

    bucket_totals = {
        bucket: {"estimate": np.zeros(len(df)), "lo": np.zeros(len(df)), "hi": np.zeros(len(df))}
        for bucket in SALES_BUCKETS
    }

    for target in TARGETS:
        bundle = load_model(target)
        preds = predict_with_interval(bundle, df, confidence_level)

        op_name = target.replace("_actual_hours", "")
        df[f"{op_name}_estimate"] = preds["estimate"]
        df[f"{op_name}_lo"] = preds["lo"]
        df[f"{op_name}_hi"] = preds["hi"]
        df[f"{op_name}_plus_minus"] = preds["plus_minus"]

        bucket = SALES_BUCKET_MAP.get(op_name)
        if bucket in bucket_totals:
            bucket_totals[bucket]["estimate"] += preds["estimate"]
            bucket_totals[bucket]["lo"] += preds["lo"]
            bucket_totals[bucket]["hi"] += preds["hi"]

        df["total_estimate"] += preds["estimate"]
        df["total_lo"] += preds["lo"]
        df["total_hi"] += preds["hi"]

    for bucket, totals in bucket_totals.items():
        plus_minus = np.maximum(
            totals["estimate"] - totals["lo"], totals["hi"] - totals["estimate"]
        )
        df[f"{bucket}_estimate"] = totals["estimate"]
        df[f"{bucket}_lo"] = totals["lo"]
        df[f"{bucket}_hi"] = totals["hi"]
        df[f"{bucket}_plus_minus"] = plus_minus

    df["total_plus_minus"] = np.maximum(
        df["total_estimate"] - df["total_lo"], df["total_hi"] - df["total_estimate"]
    )

    return df
