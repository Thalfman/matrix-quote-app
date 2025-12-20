#!/usr/bin/env python3
"""End-to-end smoke test: train models and run single + batch predictions."""

from __future__ import annotations

import contextlib
import os
import sys
import tempfile
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core.config import (  # noqa: E402
    QUOTE_CAT_FEATURES,
    QUOTE_NUM_FEATURES,
    REQUIRED_TRAINING_COLS,
    TARGETS,
)
from core import cqr  # noqa: E402,F401
from core.features import engineer_features_for_training  # noqa: E402
from core.models import train_one_op  # noqa: E402
from core.schemas import QuoteInput  # noqa: E402
from service import predict_lib  # noqa: E402
from service.predict_lib import predict_quote, predict_quotes_df  # noqa: E402


_BOOL_COLS = {
    "has_controls",
    "has_robotics",
    "duplicate",
    "Retrofit",
    "is_product_deformable",
    "is_bulk_product",
    "has_tricky_packaging",
}


@contextlib.contextmanager
def _chdir(path: str) -> Iterable[None]:
    prior = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prior)


def _make_synthetic_df(rows: int, zero_target: str) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    data: dict[str, list] = {}

    data["project_id"] = [f"proj_{i:03d}" for i in range(rows)]
    data["include_in_training"] = ["yes"] * rows
    data["dataset_role"] = ["actuals"] * rows

    cat_choices = {
        "industry_segment": ["food", "auto", "consumer"],
        "system_category": ["assembly", "packaging", "inspection"],
        "automation_level": ["low", "medium", "high"],
        "plc_family": ["allen", "siemens", "mitsubishi"],
        "hmi_family": ["panelview", "wincc", "got"],
        "vision_type": ["none", "2d", "3d"],
    }
    for col in QUOTE_CAT_FEATURES:
        data[col] = rng.choice(cat_choices[col], size=rows).tolist()

    for col in QUOTE_NUM_FEATURES:
        if col in _BOOL_COLS:
            data[col] = rng.integers(0, 2, size=rows).tolist()
        elif col in {"custom_pct"}:
            data[col] = rng.uniform(0, 100, size=rows).tolist()
        else:
            data[col] = rng.uniform(0, 20, size=rows).tolist()

    for target in TARGETS:
        if target == zero_target:
            data[target] = [0.0] * rows
        else:
            data[target] = rng.uniform(10, 200, size=rows).tolist()

    df = pd.DataFrame(data)
    missing = [c for c in REQUIRED_TRAINING_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Synthetic data missing required columns: {missing}")
    return df


def _build_quote_input(df: pd.DataFrame) -> QuoteInput:
    row = df.iloc[0].to_dict()
    payload = {col: row.get(col) for col in QUOTE_CAT_FEATURES + QUOTE_NUM_FEATURES}
    payload["project_id"] = row.get("project_id")
    return QuoteInput(**payload)


def main() -> None:
    zero_target = TARGETS[0]
    df_raw = _make_synthetic_df(rows=30, zero_target=zero_target)
    df_train = engineer_features_for_training(df_raw)

    targets_present = [t for t in TARGETS if t in df_train.columns]
    if targets_present:
        hours_mat = (
            df_train[targets_present]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0)
        )
        df_train = df_train[hours_mat.gt(0).any(axis=1)]

    with tempfile.TemporaryDirectory() as tmpdir:
        os.makedirs(os.path.join(tmpdir, "models"), exist_ok=True)
        with _chdir(tmpdir):
            trained_targets: list[str] = []
            for target in TARGETS:
                metrics = train_one_op(
                    df_train,
                    target,
                    models_dir="models",
                    version="v1",
                )
                if metrics and metrics.get("trained", True):
                    trained_targets.append(target)

            if not trained_targets:
                raise RuntimeError("No models trained; cannot run predictions.")

            quote = _build_quote_input(df_raw)
            single_pred = predict_quote(quote)

            trained_target = trained_targets[0]
            trained_op_name = trained_target.replace("_actual_hours", "")
            trained_op = single_pred.ops[trained_op_name]
            assert all(
                value is not None
                for value in (trained_op.p10, trained_op.p50, trained_op.p90)
            ), "Single prediction missing p10/p50/p90."
            assert (
                trained_op.p10 <= trained_op.p50 <= trained_op.p90
            ), "Single prediction interval ordering violated."
            assert np.isfinite(
                trained_op.p90 - trained_op.p10
            ), "Single prediction interval width not finite."
            assert (
                trained_op.p90 - trained_op.p10 >= 0.0
            ), "Single prediction interval width negative."
            assert (
                single_pred.total_p10
                <= single_pred.total_p50
                <= single_pred.total_p90
            ), "Total interval ordering violated."

            batch_input = df_raw[
                QUOTE_NUM_FEATURES + QUOTE_CAT_FEATURES
            ].head(8)
            batch_pred = predict_quotes_df(batch_input)

            for suffix in ("p10", "p50", "p90"):
                col_name = f"{trained_op_name}_{suffix}"
                if col_name not in batch_pred.columns:
                    raise AssertionError(
                        f"Batch prediction missing column: {col_name}"
                    )

            p10 = batch_pred[f"{trained_op_name}_p10"].to_numpy()
            p50 = batch_pred[f"{trained_op_name}_p50"].to_numpy()
            p90 = batch_pred[f"{trained_op_name}_p90"].to_numpy()
            assert np.all(
                np.isfinite(p90 - p10)
            ), "Batch interval width not finite."
            assert np.all(
                (p10 <= p50) & (p50 <= p90)
            ), "Batch interval ordering violated."
            assert np.all(
                (p90 - p10) >= 0.0
            ), "Batch interval width negative."
            assert np.all(
                batch_pred["total_p10"]
                <= batch_pred["total_p50"]
            ) and np.all(
                batch_pred["total_p50"]
                <= batch_pred["total_p90"]
            ), "Batch total interval ordering violated."

            zero_op_name = zero_target.replace("_actual_hours", "")
            zero_op = single_pred.ops[zero_op_name]
            assert (
                zero_op.p10 == 0.0
                and zero_op.p50 == 0.0
                and zero_op.p90 == 0.0
            ), "Untrained target should return zero bounds."
            assert (
                "not trained" in (zero_op.confidence or "").lower()
                or zero_op.trained is False
            ), "Untrained target missing not trained indicator."

            assert np.allclose(
                batch_pred[f"{zero_op_name}_p10"].to_numpy(), 0.0
            ) and np.allclose(
                batch_pred[f"{zero_op_name}_p50"].to_numpy(), 0.0
            ) and np.allclose(
                batch_pred[f"{zero_op_name}_p90"].to_numpy(), 0.0
            ), "Batch untrained target should return zero bounds."
            assert np.all(
                batch_pred[f"{zero_op_name}_confidence"]
                .astype(str)
                .str.contains("not trained", case=False)
            ) or np.all(
                batch_pred[f"{zero_op_name}_trained"] == False
            ), "Batch untrained target missing not trained indicator."

            coverage_input = df_raw[
                QUOTE_NUM_FEATURES + QUOTE_CAT_FEATURES
            ].head(20)
            coverage_pred = predict_quotes_df(coverage_input)
            actuals = df_raw[trained_target].head(20).to_numpy()
            cov_p10 = coverage_pred[f"{trained_op_name}_p10"].to_numpy()
            cov_p90 = coverage_pred[f"{trained_op_name}_p90"].to_numpy()
            coverage = np.mean((actuals >= cov_p10) & (actuals <= cov_p90))
            assert (
                coverage >= 0.70
            ), f"Coverage below threshold: {coverage:.2f}"

        print(
            "OK: trained models in temp dir, "
            f"trained_targets={len(trained_targets)} "
            f"(skipped_zero_target={zero_target})."
        )


if __name__ == "__main__":
    main()
