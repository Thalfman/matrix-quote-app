# service/predict_lib.py
# Small library that exposes prediction functions for single and batch inputs.

from typing import Dict, List

import pandas as pd

from core.config import (
    TARGETS,
    QUOTE_NUM_FEATURES,
    QUOTE_CAT_FEATURES,
    SALES_BUCKETS,
    SALES_BUCKET_MAP,
)
from core.features import prepare_quote_features
from core.models import load_model, predict_with_conformal_interval, CQRModelBundle
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


def predict_quote(q: QuoteInput, confidence: float = 0.90) -> QuotePrediction:
    """
    Predict hours for a single project described by QuoteInput.
    Returns per-operation conformal intervals plus totals.
    """
    df = _quote_to_df(q)

    ops: Dict[str, OpPrediction] = {}
    bucket_totals = {bucket: {"estimate": 0.0, "low": 0.0, "high": 0.0} for bucket in SALES_BUCKETS}
    total_estimate = total_low = total_high = 0.0
    missing_models: List[str] = []

    for target in TARGETS:
        op_name = target.replace("_actual_hours", "")
        try:
            bundle: CQRModelBundle = load_model(target, version="v2")
        except FileNotFoundError:
            missing_models.append(op_name)
            bundle = None

        if bundle is None:
            estimate = low = high = 0.0
            calib_n = 0
        else:
            estimate_arr, low_arr, high_arr = predict_with_conformal_interval(
                bundle, df, confidence=confidence
            )
            estimate = float(estimate_arr[0])
            low = float(low_arr[0])
            high = float(high_arr[0])
            calib_n = int(bundle.calib_n)

        ops[op_name] = OpPrediction(
            estimate=estimate,
            low=low,
            high=high,
            confidence=confidence,
            calib_n=calib_n,
        )

        bucket = SALES_BUCKET_MAP.get(op_name)
        if bucket in bucket_totals:
            bucket_totals[bucket]["estimate"] += estimate
            bucket_totals[bucket]["low"] += low
            bucket_totals[bucket]["high"] += high

        total_estimate += estimate
        total_low += low
        total_high += high

    sales_buckets: Dict[str, SalesBucketPrediction] = {}
    for bucket in SALES_BUCKETS:
        totals = bucket_totals.get(bucket, {"estimate": 0.0, "low": 0.0, "high": 0.0})
        sales_buckets[bucket] = SalesBucketPrediction(
            estimate=float(totals["estimate"]),
            low=float(totals["low"]),
            high=float(totals["high"]),
            confidence=confidence,
        )

    return QuotePrediction(
        ops=ops,
        total_estimate=float(total_estimate),
        total_low=float(total_low),
        total_high=float(total_high),
        confidence=confidence,
        sales_buckets=sales_buckets,
        missing_models=missing_models,
    )


def predict_quotes_df(df_in: pd.DataFrame, confidence: float = 0.90) -> pd.DataFrame:
    """
    Batch prediction for a DataFrame with quote-time columns.
    Adds per-operation estimate/low/high and project totals.
    """
    df = prepare_quote_features(df_in)

    for target in TARGETS:
        op_name = target.replace("_actual_hours", "")
        try:
            bundle: CQRModelBundle = load_model(target, version="v2")
        except FileNotFoundError:
            bundle = None

        if bundle is None:
            estimate_arr = [0.0] * len(df)
            low_arr = [0.0] * len(df)
            high_arr = [0.0] * len(df)
        else:
            estimate_arr, low_arr, high_arr = predict_with_conformal_interval(
                bundle, df, confidence=confidence
            )

        df[f"{op_name}_estimate"] = estimate_arr
        df[f"{op_name}_low"] = low_arr
        df[f"{op_name}_high"] = high_arr

    op_cols_est = [f"{target.replace('_actual_hours', '')}_estimate" for target in TARGETS]
    op_cols_low = [f"{target.replace('_actual_hours', '')}_low" for target in TARGETS]
    op_cols_high = [f"{target.replace('_actual_hours', '')}_high" for target in TARGETS]

    df["total_estimate"] = df[op_cols_est].sum(axis=1)
    df["total_low"] = df[op_cols_low].sum(axis=1)
    df["total_high"] = df[op_cols_high].sum(axis=1)
    df["confidence"] = confidence

    return df
