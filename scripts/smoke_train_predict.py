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
                if metrics:
                    trained_targets.append(target)

            if not trained_targets:
                raise RuntimeError("No models trained; cannot run predictions.")

            original_targets = predict_lib.TARGETS
            predict_lib.TARGETS = trained_targets
            try:
                quote = _build_quote_input(df_raw)
                single_pred = predict_quote(quote)

                first_op = next(iter(single_pred.ops.values()))
                assert all(
                    value is not None
                    for value in (first_op.p10, first_op.p50, first_op.p90)
                ), "Single prediction missing p10/p50/p90."

                batch_input = df_raw[
                    QUOTE_NUM_FEATURES + QUOTE_CAT_FEATURES
                ].head(5)
                batch_pred = predict_quotes_df(batch_input)

                op_name = trained_targets[0].replace("_actual_hours", "")
                for suffix in ("p10", "p50", "p90"):
                    col_name = f"{op_name}_{suffix}"
                    if col_name not in batch_pred.columns:
                        raise AssertionError(
                            f"Batch prediction missing column: {col_name}"
                        )
            finally:
                predict_lib.TARGETS = original_targets

        print(
            "OK: trained models in temp dir, "
            f"trained_targets={len(trained_targets)} "
            f"(skipped_zero_target={zero_target})."
        )


if __name__ == "__main__":
    main()
