"""Value at Risk and Conditional VaR (Expected Shortfall), reported as positive loss fractions."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from scipy import stats

from risk.metrics import portfolio_returns

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


DEFAULT_DF = 5.0
DF_BOUNDS = (3.0, 30.0)


def fit_t_df(r: pd.Series) -> float:
    """Student-t degrees of freedom fitted by maximum likelihood, clipped to DF_BOUNDS.

    The lower bound of 3 also guarantees df > 2, which the variance rescaling needs.
    """
    try:
        nu, _loc, _scale = stats.t.fit(r.to_numpy(dtype=float))
    except Exception:
        return DEFAULT_DF
    if not np.isfinite(nu):
        return DEFAULT_DF
    return float(np.clip(nu, *DF_BOUNDS))


def simulate_t_returns(
    returns: pd.DataFrame, df: float, n_sims: int = 10_000, seed: int = 42
) -> np.ndarray:
    """Multivariate Student-t scenarios with the same mean and covariance as `returns`.

    X = mu + Z / sqrt(W / df), Z ~ MVN(0, cov * (df - 2) / df), W ~ chi2(df).
    The (df - 2) / df factor makes the simulated covariance equal the sample covariance;
    the single W per scenario makes all assets fall together in bad draws.
    """
    rng = np.random.default_rng(seed)
    mu = returns.mean().to_numpy(dtype=float)
    cov = returns.cov().to_numpy(dtype=float)
    z = rng.multivariate_normal(np.zeros(len(mu)), cov * (df - 2) / df, size=n_sims)
    w = rng.chisquare(df, size=n_sims)
    return mu + z / np.sqrt(w / df)[:, None]


def monte_carlo_var(
    returns: pd.DataFrame,
    weights: pd.Series,
    conf: float,
    df: float,
    n_sims: int = 10_000,
    seed: int = 42,
) -> VarResult:
    assets = returns[list(weights.index)]
    scenarios = simulate_t_returns(assets, df, n_sims, seed) @ weights.to_numpy(dtype=float)
    var, cvar = _empirical(scenarios, conf)
    return VarResult(MONTE_CARLO, var, cvar)


def var_table(
    returns: pd.DataFrame, weights: pd.Series, conf: float, days: int
) -> tuple[pd.DataFrame, float]:
    """Side-by-side VaR/CVaR for all three methods at the chosen horizon, plus the fitted df."""
    r = portfolio_returns(returns, weights)
    df = fit_t_df(r)
    results = [
        historical_var(r, conf),
        parametric_var(r, conf),
        monte_carlo_var(returns, weights, conf, df),
    ]
    scaled = [scale_horizon(result, days) for result in results]
    table = pd.DataFrame(
        {"VaR": [x.var for x in scaled], "CVaR": [x.cvar for x in scaled]},
        index=pd.Index([x.method for x in scaled], name="Method"),
    )
    return table, df
