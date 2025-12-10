# service/predict_lib.py
# Small library that exposes prediction functions for single and batch inputs.

from typing import Dict

import pandas as pd

from core.config import (
    TARGETS,
    QUOTE_NUM_FEATURES,
    QUOTE_CAT_FEATURES,
    SALES_BUCKET_MAP,
    SALES_BUCKET_ORDER,
)
from core.features import prepare_quote_features
from core.models import load_model, predict_with_interval
from core.schemas import (
    QuoteInput,
    QuotePrediction,
    OpPrediction,
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


def _compute_confidence(p10: float, p50: float, p90: float):
    """
    Derive a heuristic confidence from the relative width of the interval.
    Smaller (p90 - p10) / |p50| => higher confidence.
    """
    eps = 1e-6
    width = p90 - p10
    denom = max(abs(p50), eps)
    rel_width = width / denom

    if rel_width < 0.3:
        label = "high"
    elif rel_width < 0.6:
        label = "medium"
    else:
        label = "low"

    return rel_width, label


def predict_quote(q: QuoteInput) -> QuotePrediction:
    """
    Predict hours for a single project described by QuoteInput.
    Returns per-operation P10/P50/P90/std/confidence plus totals.
    """
    df = _quote_to_df(q)

    ops: Dict[str, OpPrediction] = {}
    bucket_preds: Dict[str, SalesBucketPrediction] = {}
    total_p50 = total_p10 = total_p90 = 0.0

    for target in TARGETS:
        pipe = load_model(target)
        p50_arr, p10_arr, p90_arr, std_arr = predict_with_interval(pipe, df)

        p50 = float(p50_arr[0])
        p10 = float(p10_arr[0])
        p90 = float(p90_arr[0])
        std = float(std_arr[0])
        rel_width, conf_label = _compute_confidence(p10, p50, p90)

        op_name = target.replace("_actual_hours", "")
        ops[op_name] = OpPrediction(
            p50=p50,
            p10=p10,
            p90=p90,
            std=std,
            rel_width=rel_width,
            confidence=conf_label,
        )

        total_p50 += p50
        total_p10 += p10
        total_p90 += p90

    # Aggregate by Sales bucket using the configured mapping
    bucket_to_ops: Dict[str, list[str]] = {}
    for op_name, bucket in SALES_BUCKET_MAP.items():
        bucket_to_ops.setdefault(bucket, []).append(op_name)

    for bucket in SALES_BUCKET_ORDER:
        op_list = bucket_to_ops.get(bucket, [])
        if not op_list:
            continue

        p10_sum = sum(ops[op].p10 for op in op_list if op in ops)
        p50_sum = sum(ops[op].p50 for op in op_list if op in ops)
        p90_sum = sum(ops[op].p90 for op in op_list if op in ops)
        rel_width, conf_label = _compute_confidence(p10_sum, p50_sum, p90_sum)

        bucket_preds[bucket] = SalesBucketPrediction(
            p10=p10_sum,
            p50=p50_sum,
            p90=p90_sum,
            rel_width=rel_width,
            confidence=conf_label,
        )

    return QuotePrediction(
        ops=ops,
        sales_buckets=bucket_preds,
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
        pipe = load_model(target)
        p50_arr, p10_arr, p90_arr, std_arr = predict_with_interval(pipe, df)

        op_name = target.replace("_actual_hours", "")
        df[f"{op_name}_p50"] = p50_arr
        df[f"{op_name}_p10"] = p10_arr
        df[f"{op_name}_p90"] = p90_arr
        df[f"{op_name}_std"] = std_arr

        df["total_p50"] += p50_arr
        df["total_p10"] += p10_arr
        df["total_p90"] += p90_arr

    return df
