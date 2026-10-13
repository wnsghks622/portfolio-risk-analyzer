"""Return and risk statistics for daily simple-return series."""

from __future__ import annotations

import numpy as np
import pandas as pd

from risk.portfolio import CASH

TRADING_DAYS = 252
_EPS = 1e-12


def _ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(denominator) or denominator < _EPS:
        return float("nan")
    return float(numerator / denominator)


def portfolio_returns(returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """Daily portfolio return with constant weights (rebalanced daily)."""
    values = returns[list(weights.index)].to_numpy() @ weights.to_numpy(dtype=float)
    return pd.Series(values, index=returns.index, name="Portfolio")


def add_cash(returns: pd.DataFrame, rf: float) -> pd.DataFrame:
    """Add a CASH column that earns the daily risk-free rate with zero volatility."""
    out = returns.copy()
    out[CASH] = rf / TRADING_DAYS
    return out


def equity_curve(r: pd.Series, start: float = 100.0) -> pd.Series:
    return start * (1 + r).cumprod()


def cagr(r: pd.Series) -> float:
    if len(r) == 0:
        return float("nan")
    growth = float((1 + r).prod())
    return growth ** (TRADING_DAYS / len(r)) - 1


def annualized_vol(r: pd.Series) -> float:
    return float(r.std(ddof=1) * np.sqrt(TRADING_DAYS))


def _excess(r: pd.Series, rf: float) -> pd.Series:
    return r - rf / TRADING_DAYS


def sharpe(r: pd.Series, rf: float) -> float:
    excess = _excess(r, rf)
    return _ratio(excess.mean() * TRADING_DAYS, excess.std(ddof=1) * np.sqrt(TRADING_DAYS))


def sortino(r: pd.Series, rf: float) -> float:
    excess = _excess(r, rf)
    downside = np.minimum(excess.to_numpy(), 0.0)
    downside_dev = np.sqrt(np.mean(downside**2)) * np.sqrt(TRADING_DAYS)
    return _ratio(excess.mean() * TRADING_DAYS, downside_dev)


def drawdown_series(r: pd.Series) -> pd.Series:
    equity = (1 + r).cumprod()
    peak = equity.cummax().clip(lower=1.0)  # the starting capital counts as a peak
    return (equity / peak - 1).rename("Drawdown")


def max_drawdown(r: pd.Series) -> float:
    if len(r) == 0:
        return float("nan")
    return float(drawdown_series(r).min())
