"""Value at Risk and Conditional VaR (Expected Shortfall), reported as positive loss fractions."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from scipy import stats

HISTORICAL = "Historical"
PARAMETRIC = "Parametric (normal)"
MONTE_CARLO = "Monte Carlo (Student-t)"


@dataclass(frozen=True)
class VarResult:
    method: str
    var: float
    cvar: float


def _empirical(values: np.ndarray, conf: float) -> tuple[float, float]:
    """VaR = loss at the (1 - conf) quantile; CVaR = average loss at or beyond it."""
    cutoff = np.quantile(values, 1 - conf)
    return float(-cutoff), float(-values[values <= cutoff].mean())


def historical_var(r: pd.Series, conf: float) -> VarResult:
    var, cvar = _empirical(r.to_numpy(dtype=float), conf)
    return VarResult(HISTORICAL, var, cvar)


def parametric_var(r: pd.Series, conf: float) -> VarResult:
    alpha = 1 - conf
    mu = float(r.mean())
    sigma = float(r.std(ddof=1))
    z = stats.norm.ppf(alpha)
    var = -(mu + z * sigma)
    cvar = -mu + sigma * stats.norm.pdf(z) / alpha
    return VarResult(PARAMETRIC, float(var), float(cvar))


def scale_horizon(result: VarResult, days: int) -> VarResult:
    """Square-root-of-time rule: assumes i.i.d. daily returns with negligible mean."""
    factor = float(np.sqrt(days))
    return replace(result, var=result.var * factor, cvar=result.cvar * factor)
