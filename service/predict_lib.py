# service/predict_lib.py
# Small library that exposes prediction functions for single and batch inputs.

from typing import Dict

import numpy as np
import pandas as pd

from core.config import (
    TARGETS,
    QUOTE_NUM_FEATURES,
    QUOTE_CAT_FEATURES,
    SALES_BUCKETS,
    SALES_BUCKET_MAP,
    TOL_PCT,
    TOL_MIN_OP_HOURS,
    TOL_MIN_TOTAL_HOURS,
)
from core.features import prepare_quote_features
from core.models import (
    empirical_within_tol_prob,
    load_model,
    predict_with_interval,
)
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


def _op_tol(p50: float) -> float:
    return max(TOL_PCT * abs(p50), TOL_MIN_OP_HOURS)


def _total_tol(total_p50: float) -> float:
    return max(TOL_PCT * abs(total_p50), TOL_MIN_TOTAL_HOURS)


def predict_quote(q: QuoteInput) -> QuotePrediction:
    """
    Predict hours for a single project described by QuoteInput.
    Returns per-operation P10/P50/P90/std/confidence plus totals.
    """
    df = _quote_to_df(q)

    ops: Dict[str, OpPrediction] = {}
    bucket_totals = {
        bucket: {"p10": 0.0, "p50": 0.0, "p90": 0.0} for bucket in SALES_BUCKETS
    }
    total_p50 = total_p10 = total_p90 = 0.0
    total_within = []

    for target in TARGETS:
        model_obj = load_model(target)
        p50_arr, p10_arr, p90_arr, std_arr = predict_with_interval(
            model_obj, df
        )

        p50 = float(p50_arr[0])
        p10 = float(p10_arr[0])
        p90 = float(p90_arr[0])
        std = float(std_arr[0])
        eps = 1e-6
        rel_width = (p90 - p10) / max(abs(p50), eps)

        tol = _op_tol(p50)
        p_within = empirical_within_tol_prob(model_obj, tol)

        if p_within is not None:
            confidence = float(p_within)
        else:
            confidence = None

        op_name = target.replace("_actual_hours", "")
        ops[op_name] = OpPrediction(
            p50=p50,
            p10=p10,
            p90=p90,
            std=std,
            rel_width=rel_width,
            confidence=confidence,
            tol_hours=tol,
        )

        bucket = SALES_BUCKET_MAP.get(op_name)
        if bucket in bucket_totals:
            bucket_totals[bucket]["p10"] += p10
            bucket_totals[bucket]["p50"] += p50
            bucket_totals[bucket]["p90"] += p90
        if p_within is not None:
            total_within.append(p_within)

        total_p50 += p50
        total_p10 += p10
        total_p90 += p90

    sales_buckets: Dict[str, SalesBucketPrediction] = {}
    for bucket in SALES_BUCKETS:
        totals = bucket_totals.get(bucket, {"p10": 0.0, "p50": 0.0, "p90": 0.0})
        p10 = float(totals["p10"])
        p50 = float(totals["p50"])
        p90 = float(totals["p90"])
        eps = 1e-6
        rel_width = (p90 - p10) / max(abs(p50), eps)
        calibrated_conf = [
            op_pred.confidence for op_pred in ops.values() if op_pred.confidence is not None
        ]
        bucket_conf = min(calibrated_conf) if calibrated_conf else None

        sales_buckets[bucket] = SalesBucketPrediction(
            p50=p50,
            p10=p10,
            p90=p90,
            rel_width=rel_width,
            confidence=bucket_conf if bucket_conf is not None else 0.0,
            tol_hours=_total_tol(p50),
        )

    tol_total = _total_tol(total_p50)
    total_within_prob = None
    if total_within:
        total_within_prob = min(total_within)

    return QuotePrediction(
        ops=ops,
        total_p50=float(total_p50),
        total_p10=float(total_p10),
        total_p90=float(total_p90),
        total_confidence=total_within_prob,
        sales_buckets=sales_buckets,
        tol_hours=tol_total,
        within_tol_prob=total_within_prob,
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

    within_probs_total = []

    for target in TARGETS:
        pipe = load_model(target)
        p50_arr, p10_arr, p90_arr, std_arr = predict_with_interval(pipe, df)

        op_name = target.replace("_actual_hours", "")
        df[f"{op_name}_p50"] = p50_arr
        df[f"{op_name}_p10"] = p10_arr
        df[f"{op_name}_p90"] = p90_arr
        df[f"{op_name}_std"] = std_arr

        tol_arr = np.maximum(TOL_PCT * np.abs(p50_arr), TOL_MIN_OP_HOURS)
        within_prob = empirical_within_tol_prob(pipe, tol_arr.mean())
        if within_prob is not None:
            df[f"{op_name}_within_tol_prob"] = within_prob
            df[f"{op_name}_tol_hours"] = tol_arr

        df["total_p50"] += p50_arr
        df["total_p10"] += p10_arr
        df["total_p90"] += p90_arr

        if within_prob is not None:
            within_probs_total.append(within_prob)

    if within_probs_total:
        df["total_tol_hours"] = np.maximum(
            TOL_PCT * np.abs(df["total_p50"]), TOL_MIN_TOTAL_HOURS
        )
        df["total_within_tol_prob"] = min(within_probs_total)

    return df
