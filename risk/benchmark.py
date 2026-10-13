"""Benchmark-relative statistics: CAPM alpha/beta, tracking error, rolling beta."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from risk.metrics import TRADING_DAYS


@dataclass(frozen=True)
class CapmResult:
    alpha: float  # annualized
    beta: float


def _aligned(r: pd.Series, bench: pd.Series) -> tuple[pd.Series, pd.Series]:
    both = pd.concat([r, bench], axis=1, join="inner").dropna()
    return both.iloc[:, 0], both.iloc[:, 1]


def capm(r: pd.Series, bench: pd.Series, rf: float) -> CapmResult:
    """OLS of excess returns: (r - rf) = alpha + beta (bench - rf) + error."""
    r, bench = _aligned(r, bench)
    rf_daily = rf / TRADING_DAYS
    y = r.to_numpy(dtype=float) - rf_daily
    x = bench.to_numpy(dtype=float) - rf_daily
    if len(x) < 2 or np.var(x, ddof=1) < 1e-18:
        return CapmResult(alpha=float("nan"), beta=float("nan"))
    beta = np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1)
    alpha_daily = y.mean() - beta * x.mean()
    return CapmResult(alpha=float(alpha_daily * TRADING_DAYS), beta=float(beta))


def tracking_error(r: pd.Series, bench: pd.Series) -> float:
    r, bench = _aligned(r, bench)
    return float((r - bench).std(ddof=1) * np.sqrt(TRADING_DAYS))


def rolling_beta(r: pd.Series, bench: pd.Series, window: int = 126) -> pd.Series:
    r, bench = _aligned(r, bench)
    beta = r.rolling(window).cov(bench) / bench.rolling(window).var()
    return beta.dropna().rename("Rolling beta")
