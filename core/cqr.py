"""Utilities for conformalized quantile regression (CQR)."""

from __future__ import annotations

from math import ceil
from typing import Tuple

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor


try:
    from sklearn.ensemble import HistGradientBoostingRegressor
except ImportError:  # pragma: no cover - depends on sklearn version
    HistGradientBoostingRegressor = None


def make_quantile_regressor(
    q: float, random_state: int = 42
):
    """Create a quantile regressor with scikit-learn GBM models.

    Prefers HistGradientBoostingRegressor when quantile loss is supported,
    otherwise falls back to GradientBoostingRegressor.
    """
    if HistGradientBoostingRegressor is not None:
        try:
            return HistGradientBoostingRegressor(
                loss="quantile",
                quantile=q,
                random_state=random_state,
            )
        except TypeError:
            pass

    return GradientBoostingRegressor(
        loss="quantile",
        alpha=q,
        random_state=random_state,
    )


def conformity_scores(
    y: np.ndarray, q_lo: np.ndarray, q_hi: np.ndarray
) -> np.ndarray:
    """Compute CQR conformity scores as max(q_lo - y, y - q_hi, 0)."""
    return np.maximum.reduce((q_lo - y, y - q_hi, np.zeros_like(y)))


def finite_sample_qhat(scores: np.ndarray, alpha: float) -> float:
    """Compute the finite-sample quantile for CQR calibration."""
    if scores.size == 0:
        raise ValueError("scores must be non-empty")

    n = scores.size
    k = int(ceil((n + 1) * (1 - alpha)))
    index = min(max(k - 1, 0), n - 1)
    sorted_scores = np.sort(scores)
    return float(sorted_scores[index])


def apply_cqr(
    q_lo: np.ndarray, q_hi: np.ndarray, qhat: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply CQR calibration to produce final bounds."""
    return q_lo - qhat, q_hi + qhat
