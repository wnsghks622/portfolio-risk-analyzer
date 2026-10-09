"""Diversification: correlation and Euler risk contribution per asset."""

from __future__ import annotations

import numpy as np
import pandas as pd

from risk.metrics import TRADING_DAYS


def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    return returns.corr()


def risk_contribution(returns: pd.DataFrame, weights: pd.Series) -> pd.DataFrame:
    """Euler decomposition: RC_i = w_i (Σw)_i / σ_p, which sums exactly to σ_p (annualized)."""
    assets = returns[list(weights.index)]
    cov = assets.cov().to_numpy(dtype=float) * TRADING_DAYS
    w = weights.to_numpy(dtype=float)
    sigma_p = float(np.sqrt(w @ cov @ w))
    if sigma_p < 1e-12:
        contribution = np.full(len(w), np.nan)
        pct = np.full(len(w), np.nan)
    else:
        contribution = w * (cov @ w) / sigma_p
        pct = contribution / sigma_p
    return pd.DataFrame(
        {"weight": w, "contribution": contribution, "pct_of_risk": pct},
        index=weights.index,
    )
