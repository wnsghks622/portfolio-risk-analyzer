# Portfolio Risk Analyzer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Streamlit dashboard that computes return, tail-risk (VaR/CVaR by three methods), diversification and benchmark statistics for a long-only portfolio, backed by a tested pure-Python `risk/` package and deployed to Hugging Face Spaces.

**Architecture:** A pure analytics package `risk/` (pandas in, numbers/DataFrames out; only `risk/data.py` does I/O) is consumed by `charts.py` (Plotly figure builders, no math) and `app.py` (Streamlit wiring, no math). Prices come live from yfinance with automatic fallback to a bundled parquet sample. GitHub Actions runs pytest and, after tests pass on `main`, pushes to the HF Space (Docker SDK).

**Tech Stack:** Python 3.12, pandas, numpy, scipy, yfinance, pyarrow, Streamlit, Plotly, pytest, Docker, GitHub Actions, git-lfs.

**Spec:** `docs/superpowers/specs/2026-10-09-portfolio-risk-analyzer-design.md`

## Global Constraints

- Python 3.12 everywhere: local venv created with `py -3.12 -m venv .venv`; Docker base image `python:3.12-slim`; CI `python-version: "3.12"`.
- Exact pins: `pandas==3.0.6`, `numpy==2.5.3`, `scipy==1.18.1`, `yfinance==1.7.0`, `pyarrow==26.0.0`, `streamlit==1.65.0`, `plotly==7.1.0`, `pytest==9.1.1`. No other runtime dependencies (no statsmodels).
- `risk/` never imports streamlit. `risk/data.py` is the only module that does I/O. `charts.py` does no math beyond unit formatting. `app.py` does no math.
- Long-only portfolios; weights normalized to sum to 1.
- 252 trading days per year. Minimum 60 aligned return days. Rolling beta window 126 days. Monte Carlo: 10,000 scenarios, seed 42. Student-t df: MLE fit clipped to [3, 30], fallback 5.
- VaR and CVaR are positive loss fractions (0.021 = 2.1% loss); CVaR ≥ VaR.
- Tests never touch the network (yfinance is always monkeypatched).
- UI copy, verbatim:
  - Sample banner: `Live data unavailable — showing bundled sample through <YYYY-MM-DD>.`
  - VaR caption: `At 95%, VaR barely differs between methods. CVaR averages the losses beyond the cutoff, so it shows the fat tail. This is why Basel's market-risk rules moved from VaR to Expected Shortfall.`
  - Horizon note: `10-day figures use the square-root-of-time rule, which assumes independent daily returns and understates risk when volatility clusters.`
  - Footer: `Assumes constant weights, rebalanced daily.`
- Sample tickers (21): SPY, AGG, AAPL, MSFT, GOOGL, AMZN, META, NVDA, VTI, VXUS, BND, GLD, VNQ, TLT, QQQ, IWM, EFA, JPM, XOM, JNJ, BRK-B.
- `*.parquet` and `*.png` are tracked with git-lfs (Hugging Face rejects plain binary files).
- Every commit message ends with the trailer `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` (passed as a second `-m`).
- All shell commands run from the repo root in Git Bash with the venv active: `source .venv/Scripts/activate`.

**Interface adjustments vs. the spec (allowed by spec §5, stated here):**
- `Portfolio.from_inputs` takes an iterable of `(ticker, weight)` pairs instead of a dict, so duplicate tickers can be summed; adds `Portfolio.weight_series()`.
- `PriceResult.as_of` is `date | None` (None when no data) and gains `limiting_ticker: str | None` for the clipping message. `get_prices` gains a `sample_path` keyword for tests. `risk/data.py` adds `load_sample()` and `available_sample_tickers()`.
- `risk/var.py` adds `simulate_t_returns()` (the testable Monte Carlo draw).
- `charts.rolling_beta_fig` takes a `benchmark_name` argument.
- New file `presets.py` (preset definitions shared by the app and tests); new `requirements-dev.txt`, `pyproject.toml`, `.gitattributes`, `.dockerignore`.

## Review Focus

1. **Blank or half-filled rows from the dynamic holdings editor** (`None`/NaN ticker or weight) — expected: silently ignored, never a crash. Pinned in Task 1 (`test_blank_rows_from_the_editor_are_ignored`, `test_missing_or_zero_weight_drops_the_ticker`).
2. **Benchmark that is also a holding, or typed in lowercase** (`spy` with SPY in "60/40 ETF") — expected: one download per ticker, analysis works. Pinned in Task 7 (`test_tickers_are_deduplicated_and_uppercased`) and the app uppercases the benchmark input (Task 10).
3. **Every holding invalid while the benchmark is valid** — expected: clear error naming the tickers, no traceback. Pinned in Task 10 (`test_all_holdings_missing_shows_error`).
4. **Recent IPO / fewer than 60 return days** — expected: blocking message, no traceback. Pinned in Task 10 (`test_short_history_is_blocked_with_a_message`).
5. **Single-asset portfolio** — expected: all statistics compute (1×1 correlation, 100% risk share, 1-asset Monte Carlo). Pinned in Task 4 (`test_single_asset_portfolio_works`) and Task 5 (`test_single_asset_carries_all_risk`).

## File Structure

| File | Responsibility | Task |
|---|---|---|
| `requirements.txt`, `requirements-dev.txt`, `pyproject.toml`, `.gitattributes`, `.gitignore` (exists) | deps, pytest config, LFS/line endings | 1 |
| `risk/__init__.py`, `risk/portfolio.py` | Portfolio validation/normalization | 1 |
| `risk/metrics.py` | portfolio returns, CAGR, vol, Sharpe, Sortino, drawdown | 2 |
| `risk/var.py` | VaR/CVaR: historical, parametric, Student-t Monte Carlo, horizon, table | 3, 4 |
| `risk/attribution.py` | correlation, Euler risk contribution | 5 |
| `risk/benchmark.py` | CAPM alpha/beta, tracking error, rolling beta | 6 |
| `risk/data.py` | live fetch, sample fallback, alignment, returns | 7 |
| `scripts/refresh_sample.py`, `data/sample_prices.parquet` | bundled fallback data | 8 |
| `charts.py` | Plotly figures | 9 |
| `presets.py`, `app.py` | Streamlit UI | 10 |
| `Dockerfile`, `.dockerignore`, `.github/workflows/ci.yml` | container, CI, deploy | 11 |
| `README.md`, `docs/screenshots/*.png` | showcase docs | 12 |
| — | publish GitHub repo + HF Space (user-gated) | 13 |

Tests: `tests/test_<module>.py` per task.

---

### Task 1: Project scaffold and `Portfolio`

**Files:**
- Create: `requirements.txt`, `requirements-dev.txt`, `pyproject.toml`, `.gitattributes`, `risk/__init__.py`, `risk/portfolio.py`
- Test: `tests/test_portfolio.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `Portfolio(tickers: tuple[str, ...], weights: tuple[float, ...])` — frozen dataclass, weights sum to 1.
  - `Portfolio.from_inputs(holdings: Iterable[tuple[object, object]]) -> Portfolio` — raises `ValueError`.
  - `Portfolio.drop(tickers: Iterable[str]) -> Portfolio` — raises `ValueError("None of the holdings have price data.")` if nothing remains.
  - `Portfolio.weight_series() -> pd.Series` — index = tickers, name `"weight"`.

- [ ] **Step 1: Create the venv and dependency files**

`requirements.txt`:
```
pandas==3.0.6
numpy==2.5.3
scipy==1.18.1
yfinance==1.7.0
pyarrow==26.0.0
streamlit==1.65.0
plotly==7.1.0
```

`requirements-dev.txt`:
```
-r requirements.txt
pytest==9.1.1
```

`pyproject.toml`:
```toml
[project]
name = "portfolio-risk-analyzer"
version = "0.1.0"
requires-python = ">=3.12"

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
```

`.gitattributes`:
```
* text=auto eol=lf
*.parquet filter=lfs diff=lfs merge=lfs -text
*.png filter=lfs diff=lfs merge=lfs -text
```

`risk/__init__.py`:
```python
"""Pure analytics package: pandas in, numbers and DataFrames out."""
```

Run:
```bash
py -3.12 -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
git lfs install --local
python --version
```
Expected: `Python 3.12.x`; install finishes without errors.

- [ ] **Step 2: Write the failing tests**

`tests/test_portfolio.py`:
```python
import pytest

from risk.portfolio import Portfolio


def test_weights_are_normalized_to_one():
    p = Portfolio.from_inputs([("SPY", 60), ("AGG", 40)])
    assert p.tickers == ("SPY", "AGG")
    assert p.weights == pytest.approx((0.6, 0.4))


def test_any_weight_scale_is_normalized():
    p = Portfolio.from_inputs([("A", 1), ("B", 3)])
    assert p.weights == pytest.approx((0.25, 0.75))


def test_tickers_are_stripped_and_uppercased():
    p = Portfolio.from_inputs([("  spy ", 1)])
    assert p.tickers == ("SPY",)


def test_duplicate_tickers_are_summed():
    p = Portfolio.from_inputs([("SPY", 30), ("AGG", 40), ("spy", 30)])
    assert p.tickers == ("SPY", "AGG")
    assert p.weights == pytest.approx((0.6, 0.4))


def test_blank_rows_from_the_editor_are_ignored():
    rows = [("SPY", 50), (None, None), ("", 10), (float("nan"), 5), ("AGG", 50)]
    p = Portfolio.from_inputs(rows)
    assert p.tickers == ("SPY", "AGG")
    assert p.weights == pytest.approx((0.5, 0.5))


def test_missing_or_zero_weight_drops_the_ticker():
    p = Portfolio.from_inputs([("SPY", 100), ("AGG", None), ("GLD", 0), ("TLT", float("nan"))])
    assert p.tickers == ("SPY",)
    assert p.weights == (1.0,)


def test_negative_weight_is_rejected():
    with pytest.raises(ValueError, match="Negative weight for AGG"):
        Portfolio.from_inputs([("SPY", 100), ("AGG", -10)])


def test_non_numeric_weight_is_rejected():
    with pytest.raises(ValueError, match="Weight for SPY is not a number"):
        Portfolio.from_inputs([("SPY", "lots")])


def test_zero_total_is_rejected():
    with pytest.raises(ValueError, match="positive weight"):
        Portfolio.from_inputs([("SPY", 0), ("AGG", 0)])


def test_no_holdings_is_rejected():
    with pytest.raises(ValueError, match="positive weight"):
        Portfolio.from_inputs([])


def test_drop_renormalizes_remaining():
    p = Portfolio.from_inputs([("A", 50), ("B", 25), ("C", 25)]).drop(["B"])
    assert p.tickers == ("A", "C")
    assert p.weights == pytest.approx((2 / 3, 1 / 3))


def test_dropping_every_holding_raises():
    with pytest.raises(ValueError, match="None of the holdings have price data"):
        Portfolio.from_inputs([("A", 1)]).drop(["A"])


def test_weight_series():
    s = Portfolio.from_inputs([("A", 1), ("B", 1)]).weight_series()
    assert list(s.index) == ["A", "B"]
    assert s.tolist() == pytest.approx([0.5, 0.5])
    assert s.name == "weight"
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/test_portfolio.py -v`
Expected: collection error `ModuleNotFoundError: No module named 'risk.portfolio'`.

- [ ] **Step 4: Implement `risk/portfolio.py`**

```python
"""Portfolio definition: tickers with normalized, long-only weights."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import pandas as pd


def _clean_ticker(raw: object) -> str:
    if raw is None or (not isinstance(raw, str) and pd.isna(raw)):
        return ""
    return str(raw).strip().upper()


def _clean_weight(ticker: str, raw: object) -> float:
    if raw is None or (not isinstance(raw, str) and pd.isna(raw)):
        return 0.0
    try:
        return float(raw)
    except (TypeError, ValueError):
        raise ValueError(f"Weight for {ticker} is not a number.") from None


@dataclass(frozen=True)
class Portfolio:
    tickers: tuple[str, ...]
    weights: tuple[float, ...]

    @classmethod
    def from_inputs(cls, holdings: Iterable[tuple[object, object]]) -> Portfolio:
        totals: dict[str, float] = {}
        for raw_ticker, raw_weight in holdings:
            ticker = _clean_ticker(raw_ticker)
            if not ticker:
                continue
            weight = _clean_weight(ticker, raw_weight)
            if weight < 0:
                raise ValueError(
                    f"Negative weight for {ticker}: short positions are not supported."
                )
            totals[ticker] = totals.get(ticker, 0.0) + weight
        return cls._normalized(totals, "Add at least one holding with a positive weight.")

    @classmethod
    def _normalized(cls, totals: dict[str, float], empty_message: str) -> Portfolio:
        positive = {t: w for t, w in totals.items() if w > 0}
        total = sum(positive.values())
        if not positive or total <= 0:
            raise ValueError(empty_message)
        return cls(tuple(positive), tuple(w / total for w in positive.values()))

    def drop(self, tickers: Iterable[str]) -> Portfolio:
        remove = set(tickers)
        kept = {t: w for t, w in zip(self.tickers, self.weights) if t not in remove}
        return self._normalized(kept, "None of the holdings have price data.")

    def weight_series(self) -> pd.Series:
        return pd.Series(self.weights, index=list(self.tickers), name="weight", dtype=float)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_portfolio.py -v`
Expected: 13 passed.

- [ ] **Step 6: Commit**

```bash
git add requirements.txt requirements-dev.txt pyproject.toml .gitattributes risk/__init__.py risk/portfolio.py tests/test_portfolio.py
git commit -m "feat: scaffold project and add validated Portfolio" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: `risk/metrics.py`

**Files:**
- Create: `risk/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Consumes: nothing from earlier tasks (weights arrive as `pd.Series`, which `Portfolio.weight_series()` produces).
- Produces (all `r` are daily simple-return `pd.Series`; `rf` is an annual decimal):
  - `TRADING_DAYS: int = 252`
  - `portfolio_returns(returns: pd.DataFrame, weights: pd.Series) -> pd.Series` (name `"Portfolio"`)
  - `equity_curve(r, start: float = 100.0) -> pd.Series`
  - `cagr(r) -> float`, `annualized_vol(r) -> float`
  - `sharpe(r, rf: float) -> float`, `sortino(r, rf: float) -> float` (NaN when denominator is zero)
  - `drawdown_series(r) -> pd.Series` (name `"Drawdown"`, ≤ 0), `max_drawdown(r) -> float` (≤ 0)

- [ ] **Step 1: Write the failing tests**

`tests/test_metrics.py`:
```python
import numpy as np
import pandas as pd
import pytest

from risk import metrics


def series(values):
    return pd.Series(values, index=pd.bdate_range("2024-01-01", periods=len(values)), dtype=float)


def test_portfolio_returns_is_weighted_sum_regardless_of_column_order():
    idx = pd.bdate_range("2024-01-01", periods=2)
    returns = pd.DataFrame({"A": [0.01, 0.02], "B": [-0.01, 0.04]}, index=idx)
    weights = pd.Series({"B": 0.25, "A": 0.75})
    r = metrics.portfolio_returns(returns, weights)
    assert r.tolist() == pytest.approx([0.005, 0.025])
    assert r.name == "Portfolio"
    assert r.index.equals(idx)


def test_equity_curve_compounds_from_100():
    eq = metrics.equity_curve(series([0.10, -0.20, 0.25]))
    assert eq.tolist() == pytest.approx([110.0, 88.0, 110.0])


def test_cagr_of_one_year_of_constant_return():
    assert metrics.cagr(series([0.001] * 252)) == pytest.approx(1.001**252 - 1)


def test_cagr_of_21_percent_over_two_years_is_10_percent():
    assert metrics.cagr(series([0.21] + [0.0] * 503)) == pytest.approx(0.10)


def test_annualized_vol():
    # mean 0.005; deviations 0.005, -0.015, 0.015, -0.005; sum of squares 5e-4; sample var = 5e-4 / 3
    r = series([0.01, -0.01, 0.02, 0.0])
    assert metrics.annualized_vol(r) == pytest.approx(np.sqrt(5e-4 / 3) * np.sqrt(252))


def test_sharpe_with_zero_risk_free():
    r = series([0.01, -0.01, 0.02, 0.0])
    expected = 0.005 * 252 / (np.sqrt(5e-4 / 3) * np.sqrt(252))
    assert metrics.sharpe(r, 0.0) == pytest.approx(expected)


def test_sharpe_subtracts_daily_risk_free():
    r = series([0.01, -0.01, 0.02, 0.0])
    # rf 2.52% a year = 0.0001 a day, so mean excess = 0.0049; std unchanged
    expected = 0.0049 * 252 / (np.sqrt(5e-4 / 3) * np.sqrt(252))
    assert metrics.sharpe(r, 0.0252) == pytest.approx(expected)


def test_sortino_uses_downside_deviation_over_all_days():
    r = series([0.02, -0.01, 0.03, -0.03])
    # mean 0.0025; downside = [0, -0.01, 0, -0.03]; mean of squares over ALL 4 days = 2.5e-4
    expected = 0.0025 * 252 / (np.sqrt(2.5e-4) * np.sqrt(252))
    buggy = 0.0025 * 252 / (np.std([-0.01, -0.03], ddof=1) * np.sqrt(252))
    result = metrics.sortino(r, 0.0)
    assert result == pytest.approx(expected)
    assert result != pytest.approx(buggy)


def test_ratios_are_nan_when_denominator_is_zero():
    flat = series([0.001] * 10)
    assert np.isnan(metrics.sharpe(flat, 0.0))
    assert np.isnan(metrics.sortino(flat, 0.0))


def test_max_drawdown_peak_to_trough():
    assert metrics.max_drawdown(series([0.10, -0.20, 0.25])) == pytest.approx(-0.20)


def test_drawdown_counts_loss_from_initial_capital():
    r = series([-0.10, 0.05])
    assert metrics.drawdown_series(r).tolist() == pytest.approx([-0.10, -0.055])
    assert metrics.max_drawdown(r) == pytest.approx(-0.10)


def test_no_drawdown_when_always_rising():
    assert metrics.max_drawdown(series([0.01, 0.02])) == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_metrics.py -v`
Expected: `ImportError: cannot import name 'metrics' from 'risk'`.

- [ ] **Step 3: Implement `risk/metrics.py`**

```python
"""Return and risk statistics for daily simple-return series."""

from __future__ import annotations

import numpy as np
import pandas as pd

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_metrics.py -v`
Expected: 12 passed.

- [ ] **Step 5: Commit**

```bash
git add risk/metrics.py tests/test_metrics.py
git commit -m "feat: add return and risk metrics" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: `risk/var.py` — historical, parametric, horizon scaling

**Files:**
- Create: `risk/var.py`
- Test: `tests/test_var.py`

**Interfaces:**
- Consumes: nothing yet.
- Produces:
  - `VarResult(method: str, var: float, cvar: float)` — frozen dataclass, positive loss fractions.
  - Method labels: `HISTORICAL = "Historical"`, `PARAMETRIC = "Parametric (normal)"`, `MONTE_CARLO = "Monte Carlo (Student-t)"`.
  - `historical_var(r: pd.Series, conf: float) -> VarResult`
  - `parametric_var(r: pd.Series, conf: float) -> VarResult`
  - `scale_horizon(result: VarResult, days: int) -> VarResult`
  - private `_empirical(values: np.ndarray, conf: float) -> tuple[float, float]` (reused in Task 4)

- [ ] **Step 1: Write the failing tests**

`tests/test_var.py`:
```python
import numpy as np
import pandas as pd
import pytest

from risk import var


def series(values):
    return pd.Series(values, index=pd.bdate_range("2020-01-01", periods=len(values)), dtype=float)


def test_historical_var_and_cvar_on_known_values():
    values = np.linspace(-0.05, 0.05, 101)  # step 0.001
    np.random.default_rng(0).shuffle(values)  # order must not matter
    result = var.historical_var(series(values), 0.95)
    # 5% quantile of 101 evenly spaced points sits at index 5 -> -0.045
    assert result.method == "Historical"
    assert result.var == pytest.approx(0.045)
    # mean of the six worst values -0.050 ... -0.045 is -0.0475
    assert result.cvar == pytest.approx(0.0475)


def test_parametric_matches_normal_formulas():
    r = series(np.random.default_rng(1).normal(0.0004, 0.012, 2_000))
    mu, sigma = r.mean(), r.std(ddof=1)
    at95 = var.parametric_var(r, 0.95)
    assert at95.method == "Parametric (normal)"
    assert at95.var == pytest.approx(-(mu - 1.6448536269514722 * sigma))
    assert at95.cvar == pytest.approx(-mu + sigma * 2.0627, rel=1e-4)
    at99 = var.parametric_var(r, 0.99)
    assert at99.var == pytest.approx(-(mu - 2.3263478740408408 * sigma))
    assert at99.cvar == pytest.approx(-mu + sigma * 2.6652, rel=1e-4)


def test_historical_close_to_parametric_on_normal_data():
    r = series(np.random.default_rng(2).normal(0.0005, 0.01, 100_000))
    hist = var.historical_var(r, 0.95)
    para = var.parametric_var(r, 0.95)
    assert hist.var == pytest.approx(para.var, rel=0.02)
    assert hist.cvar == pytest.approx(para.cvar, rel=0.02)


@pytest.mark.parametrize("conf", [0.95, 0.99])
def test_cvar_is_never_below_var(conf):
    rng = np.random.default_rng(3)
    for r in (series(rng.normal(0, 0.01, 5_000)), series(rng.standard_t(4, 5_000) * 0.01)):
        for result in (var.historical_var(r, conf), var.parametric_var(r, conf)):
            assert result.cvar >= result.var


def test_scale_horizon_uses_square_root_of_time():
    one_day = var.VarResult("Historical", 0.02, 0.03)
    ten_day = var.scale_horizon(one_day, 10)
    assert ten_day.method == "Historical"
    assert ten_day.var == pytest.approx(0.02 * np.sqrt(10))
    assert ten_day.cvar == pytest.approx(0.03 * np.sqrt(10))
    assert var.scale_horizon(one_day, 1) == one_day
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_var.py -v`
Expected: `ImportError: cannot import name 'var' from 'risk'`.

- [ ] **Step 3: Implement `risk/var.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_var.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add risk/var.py tests/test_var.py
git commit -m "feat: add historical and parametric VaR/CVaR with horizon scaling" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: `risk/var.py` — fitted Student-t Monte Carlo and comparison table

**Files:**
- Modify: `risk/var.py` (append; add one import)
- Test: `tests/test_var.py` (append)

**Interfaces:**
- Consumes: `risk.metrics.portfolio_returns(returns, weights) -> pd.Series` (Task 2); `VarResult`, `_empirical`, `historical_var`, `parametric_var`, `scale_horizon` (Task 3).
- Produces:
  - `DEFAULT_DF = 5.0`, `DF_BOUNDS = (3.0, 30.0)`
  - `fit_t_df(r: pd.Series) -> float`
  - `simulate_t_returns(returns: pd.DataFrame, df: float, n_sims: int = 10_000, seed: int = 42) -> np.ndarray` — shape `(n_sims, n_assets)`, columns in `returns` column order.
  - `monte_carlo_var(returns: pd.DataFrame, weights: pd.Series, conf: float, df: float, n_sims: int = 10_000, seed: int = 42) -> VarResult`
  - `var_table(returns: pd.DataFrame, weights: pd.Series, conf: float, days: int) -> tuple[pd.DataFrame, float]` — index `Method` (rows in order Historical, Parametric (normal), Monte Carlo (Student-t)), columns `["VaR", "CVaR"]`; second value is the fitted df.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_var.py`)

```python
def t_series(df, n, scale=0.01, seed=0):
    return series(np.random.default_rng(seed).standard_t(df, size=n) * scale)


def two_asset_returns(n=1_000, seed=5):
    cov = np.array([[1e-4, 3e-5], [3e-5, 4e-5]])
    data = np.random.default_rng(seed).multivariate_normal([0.0004, 0.0002], cov, size=n)
    return pd.DataFrame(data, columns=["A", "B"], index=pd.bdate_range("2020-01-01", periods=n))


def test_fit_recovers_df_of_t_data():
    assert var.fit_t_df(t_series(4, 5_000)) == pytest.approx(4, abs=1)


def test_fit_on_normal_data_stays_in_bounds():
    nu = var.fit_t_df(series(np.random.default_rng(1).normal(0, 0.01, 2_000)))
    assert 3.0 <= nu <= 30.0


def test_fit_failure_falls_back_to_5(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("fit failed")

    monkeypatch.setattr(var.stats.t, "fit", boom)
    assert var.fit_t_df(t_series(4, 500)) == 5.0


def test_non_finite_fit_falls_back_to_5(monkeypatch):
    monkeypatch.setattr(var.stats.t, "fit", lambda *a, **k: (float("nan"), 0.0, 1.0))
    assert var.fit_t_df(t_series(4, 500)) == 5.0


def test_fit_is_clipped_to_lower_bound(monkeypatch):
    monkeypatch.setattr(var.stats.t, "fit", lambda *a, **k: (1.5, 0.0, 1.0))
    assert var.fit_t_df(t_series(4, 500)) == 3.0


def test_simulated_covariance_matches_input():
    hist = two_asset_returns(n=5_000)
    sims = var.simulate_t_returns(hist, df=10, n_sims=200_000, seed=3)
    assert sims.shape == (200_000, 2)
    assert np.cov(sims, rowvar=False) == pytest.approx(hist.cov().to_numpy(), rel=0.05)


def test_monte_carlo_is_reproducible_with_seed():
    returns = two_asset_returns()
    w = pd.Series({"A": 0.6, "B": 0.4})
    first = var.monte_carlo_var(returns, w, 0.95, df=5)
    second = var.monte_carlo_var(returns, w, 0.95, df=5)
    assert first == second
    assert first.method == "Monte Carlo (Student-t)"
    assert first.cvar >= first.var


def test_student_t_var_crosses_normal_between_95_and_99():
    r = t_series(5, 20_000, seed=4)
    returns = r.to_frame("A")
    w = pd.Series({"A": 1.0})
    nu = var.fit_t_df(r)
    mc95 = var.monte_carlo_var(returns, w, 0.95, nu, n_sims=100_000)
    mc99 = var.monte_carlo_var(returns, w, 0.99, nu, n_sims=100_000)
    # vol-matched fat tails: smaller VaR at 95%, bigger at 99%, bigger CVaR at 95%
    assert mc95.var < var.parametric_var(r, 0.95).var
    assert mc99.var > var.parametric_var(r, 0.99).var
    assert mc95.cvar > var.parametric_var(r, 0.95).cvar


def test_var_table_shape_and_horizon():
    returns = two_asset_returns()
    w = pd.Series({"A": 0.5, "B": 0.5})
    one_day, nu = var.var_table(returns, w, 0.95, 1)
    ten_day, _ = var.var_table(returns, w, 0.95, 10)
    assert list(one_day.index) == ["Historical", "Parametric (normal)", "Monte Carlo (Student-t)"]
    assert one_day.index.name == "Method"
    assert list(one_day.columns) == ["VaR", "CVaR"]
    assert 3.0 <= nu <= 30.0
    assert ten_day.to_numpy() == pytest.approx(one_day.to_numpy() * np.sqrt(10))
    assert (one_day["CVaR"] >= one_day["VaR"]).all()


def test_single_asset_portfolio_works():
    returns = series(np.random.default_rng(6).normal(0.0003, 0.01, 750)).to_frame("SPY")
    table, _ = var.var_table(returns, pd.Series({"SPY": 1.0}), 0.99, 1)
    assert table.notna().all().all()
    assert (table > 0).all().all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_var.py -v`
Expected: the 10 new tests fail with `AttributeError: module 'risk.var' has no attribute ...`; the 6 Task 3 tests still pass.

- [ ] **Step 3: Implement** — add the import below the existing imports in `risk/var.py`:

```python
from risk.metrics import portfolio_returns
```

and append to `risk/var.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_var.py -v`
Expected: 16 passed (scipy may print RuntimeWarnings during `t.fit`; that is fine).

- [ ] **Step 5: Commit**

```bash
git add risk/var.py tests/test_var.py
git commit -m "feat: add fitted Student-t Monte Carlo VaR and comparison table" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: `risk/attribution.py`

**Files:**
- Create: `risk/attribution.py`
- Test: `tests/test_attribution.py`

**Interfaces:**
- Consumes: `risk.metrics.TRADING_DAYS`; tests also use `metrics.portfolio_returns`, `metrics.annualized_vol` (Task 2).
- Produces:
  - `correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame`
  - `risk_contribution(returns: pd.DataFrame, weights: pd.Series) -> pd.DataFrame` — index = tickers in `weights` order; columns `["weight", "contribution", "pct_of_risk"]`; `contribution` in annualized-vol units, sums to σₚ; `pct_of_risk` sums to 1.

- [ ] **Step 1: Write the failing tests**

`tests/test_attribution.py`:
```python
import numpy as np
import pandas as pd
import pytest

from risk import attribution, metrics


def frame(data, columns):
    return pd.DataFrame(data, columns=columns, index=pd.bdate_range("2015-01-01", periods=len(data)))


def correlated_returns(n=1_500, seed=0):
    cov = np.array([[2e-4, 6e-5, 1e-5], [6e-5, 1e-4, 2e-5], [1e-5, 2e-5, 3e-5]])
    return frame(np.random.default_rng(seed).multivariate_normal([0, 0, 0], cov, size=n), ["A", "B", "C"])


def test_contributions_sum_to_portfolio_vol():
    returns = correlated_returns()
    w = pd.Series({"A": 0.5, "B": 0.3, "C": 0.2})
    rc = attribution.risk_contribution(returns, w)
    vol = metrics.annualized_vol(metrics.portfolio_returns(returns, w))
    assert rc["contribution"].sum() == pytest.approx(vol, rel=1e-9)
    assert rc["pct_of_risk"].sum() == pytest.approx(1.0)
    assert list(rc.columns) == ["weight", "contribution", "pct_of_risk"]
    assert list(rc.index) == ["A", "B", "C"]
    assert rc["weight"].tolist() == pytest.approx([0.5, 0.3, 0.2])


def test_identical_assets_split_risk_evenly():
    x = np.random.default_rng(1).normal(0, 0.01, 500)
    returns = frame(np.column_stack([x, x]), ["A", "B"])
    rc = attribution.risk_contribution(returns, pd.Series({"A": 0.5, "B": 0.5}))
    assert rc["pct_of_risk"].tolist() == pytest.approx([0.5, 0.5])


def test_stocks_dominate_risk_in_60_40():
    rng = np.random.default_rng(2)
    stocks = rng.normal(0, 0.16 / np.sqrt(252), 2_520)
    bonds = rng.normal(0, 0.05 / np.sqrt(252), 2_520)
    returns = frame(np.column_stack([stocks, bonds]), ["STOCKS", "BONDS"])
    rc = attribution.risk_contribution(returns, pd.Series({"STOCKS": 0.6, "BONDS": 0.4}))
    assert rc.loc["STOCKS", "pct_of_risk"] > 0.9


def test_single_asset_carries_all_risk():
    returns = frame(np.random.default_rng(3).normal(0, 0.01, 300), ["SPY"])
    rc = attribution.risk_contribution(returns, pd.Series({"SPY": 1.0}))
    assert rc.loc["SPY", "pct_of_risk"] == pytest.approx(1.0)


def test_correlation_matrix_is_symmetric_with_unit_diagonal():
    corr = attribution.correlation_matrix(correlated_returns())
    assert np.diag(corr.to_numpy()) == pytest.approx([1.0, 1.0, 1.0])
    assert corr.to_numpy() == pytest.approx(corr.to_numpy().T)
    assert list(corr.columns) == ["A", "B", "C"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_attribution.py -v`
Expected: `ImportError: cannot import name 'attribution' from 'risk'`.

- [ ] **Step 3: Implement `risk/attribution.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_attribution.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add risk/attribution.py tests/test_attribution.py
git commit -m "feat: add correlation matrix and Euler risk contribution" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: `risk/benchmark.py`

**Files:**
- Create: `risk/benchmark.py`
- Test: `tests/test_benchmark.py`

**Interfaces:**
- Consumes: `risk.metrics.TRADING_DAYS`.
- Produces:
  - `CapmResult(alpha: float, beta: float)` — frozen dataclass; alpha annualized; both NaN if benchmark variance is zero.
  - `capm(r: pd.Series, bench: pd.Series, rf: float) -> CapmResult`
  - `tracking_error(r: pd.Series, bench: pd.Series) -> float`
  - `rolling_beta(r: pd.Series, bench: pd.Series, window: int = 126) -> pd.Series` (name `"Rolling beta"`, length `n - window + 1`)
  - All three inner-join `r` and `bench` on dates first.

- [ ] **Step 1: Write the failing tests**

`tests/test_benchmark.py`:
```python
import numpy as np
import pandas as pd
import pytest

from risk import benchmark


def bench_series(n=500, seed=0):
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.Series(np.random.default_rng(seed).normal(0.0004, 0.01, n), index=idx, name="SPY")


def test_benchmark_against_itself_has_beta_one_alpha_zero():
    b = bench_series()
    fit = benchmark.capm(b, b, rf=0.04)
    assert fit.beta == pytest.approx(1.0)
    assert fit.alpha == pytest.approx(0.0, abs=1e-12)


def test_double_benchmark_has_beta_two():
    b = bench_series()
    fit = benchmark.capm(2 * b, b, rf=0.0)
    assert fit.beta == pytest.approx(2.0)
    assert fit.alpha == pytest.approx(0.0, abs=1e-12)


def test_constant_outperformance_is_annualized_alpha():
    b = bench_series()
    fit = benchmark.capm(b + 0.0001, b, rf=0.0)
    assert fit.beta == pytest.approx(1.0)
    assert fit.alpha == pytest.approx(0.0001 * 252)


def test_risk_free_rate_does_not_change_beta():
    b = bench_series()
    r = 0.5 * b + pd.Series(np.random.default_rng(1).normal(0, 0.005, len(b)), index=b.index)
    assert benchmark.capm(r, b, 0.0).beta == pytest.approx(benchmark.capm(r, b, 0.05).beta)


def test_flat_benchmark_gives_nan():
    b = bench_series()
    flat = pd.Series(0.0, index=b.index)
    fit = benchmark.capm(b, flat, 0.0)
    assert np.isnan(fit.beta) and np.isnan(fit.alpha)


def test_inputs_are_aligned_on_dates():
    b = bench_series()
    longer = pd.concat([pd.Series([0.5], index=[pd.Timestamp("2019-12-31")]), 2 * b])
    assert benchmark.capm(longer, b, 0.0).beta == pytest.approx(2.0)


def test_tracking_error_is_zero_for_constant_gap():
    b = bench_series()
    assert benchmark.tracking_error(b + 0.0002, b) == pytest.approx(0.0, abs=1e-12)


def test_tracking_error_known_value():
    b = bench_series(n=100)
    gap = pd.Series([0.001, -0.001] * 50, index=b.index)
    # every deviation is ±0.001: sample variance = 100 * 1e-6 / 99
    expected = np.sqrt(100 * 1e-6 / 99) * np.sqrt(252)
    assert benchmark.tracking_error(b + gap, b) == pytest.approx(expected)


def test_rolling_beta_length_and_value():
    b = bench_series(n=300)
    beta = benchmark.rolling_beta(2 * b, b, window=126)
    assert len(beta) == 300 - 126 + 1
    assert beta.to_numpy() == pytest.approx(np.full(len(beta), 2.0))
    assert beta.name == "Rolling beta"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_benchmark.py -v`
Expected: `ImportError: cannot import name 'benchmark' from 'risk'`.

- [ ] **Step 3: Implement `risk/benchmark.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_benchmark.py -v`
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add risk/benchmark.py tests/test_benchmark.py
git commit -m "feat: add CAPM alpha/beta, tracking error and rolling beta" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: `risk/data.py`

**Files:**
- Create: `risk/data.py`
- Test: `tests/test_data.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `SAMPLE_PATH: Path` = `<repo>/data/sample_prices.parquet`; `CLIP_TOLERANCE_DAYS = 5`
  - `PriceResult(prices: pd.DataFrame, source: Literal["live", "sample"], as_of: date | None, missing: list[str], clipped_start: date | None, limiting_ticker: str | None)` — frozen dataclass.
  - `get_prices(tickers: list[str], start: date, end: date, sample_path: Path = SAMPLE_PATH) -> PriceResult`
  - `to_returns(prices: pd.DataFrame) -> pd.DataFrame`
  - `load_sample(path: Path = SAMPLE_PATH) -> pd.DataFrame`
  - `available_sample_tickers(path: Path = SAMPLE_PATH) -> list[str]` (sorted)
  - private seam `_download_live(tickers: list[str], start: date, end: date) -> pd.DataFrame` — tests and the app smoke test monkeypatch this.

- [ ] **Step 1: Write the failing tests**

`tests/test_data.py`:
```python
from datetime import date

import numpy as np
import pandas as pd
import pytest

from risk import data


def make_prices(tickers, start="2015-01-01", end="2025-06-30", seed=0):
    idx = pd.bdate_range(start, end, name="Date")
    steps = np.random.default_rng(seed).normal(0.0003, 0.01, size=(len(idx), len(tickers)))
    return pd.DataFrame(100 * np.exp(np.cumsum(steps, axis=0)), index=idx, columns=tickers)


@pytest.fixture
def sample_path(tmp_path):
    path = tmp_path / "sample.parquet"
    make_prices(["SPY", "AGG", "AAPL"]).to_parquet(path)
    return path


def offline(*args, **kwargs):
    raise ConnectionError("no network in tests")


def test_live_failure_falls_back_to_sample(monkeypatch, sample_path):
    monkeypatch.setattr(data, "_download_live", offline)
    result = data.get_prices(["SPY", "AGG"], date(2021, 1, 1), date(2024, 1, 1), sample_path=sample_path)
    assert result.source == "sample"
    assert list(result.prices.columns) == ["SPY", "AGG"]
    assert result.missing == []


def test_empty_live_result_falls_back(monkeypatch, sample_path):
    monkeypatch.setattr(data, "_download_live", lambda tickers, start, end: pd.DataFrame())
    result = data.get_prices(["SPY"], date(2021, 1, 1), date(2024, 1, 1), sample_path=sample_path)
    assert result.source == "sample"


def test_all_nan_live_result_falls_back(monkeypatch, sample_path):
    idx = pd.bdate_range("2023-01-02", periods=50)
    nan_frame = pd.DataFrame(np.nan, index=idx, columns=["SPY", "AGG"])
    monkeypatch.setattr(data, "_download_live", lambda tickers, start, end: nan_frame)
    result = data.get_prices(["SPY", "AGG"], date(2021, 1, 1), date(2024, 1, 1), sample_path=sample_path)
    assert result.source == "sample"
    assert result.missing == []


def test_sample_window_keeps_length_but_ends_at_sample_end(monkeypatch, sample_path):
    monkeypatch.setattr(data, "_download_live", offline)
    result = data.get_prices(["SPY"], date(2021, 1, 1), date(2026, 1, 1), sample_path=sample_path)
    assert result.as_of == date(2025, 6, 30)
    first = result.prices.index[0]
    assert pd.Timestamp("2020-06-29") <= first <= pd.Timestamp("2020-07-08")
    assert result.clipped_start is None


def test_ticker_not_in_sample_is_missing(monkeypatch, sample_path):
    monkeypatch.setattr(data, "_download_live", offline)
    result = data.get_prices(["SPY", "ZZZZ"], date(2021, 1, 1), date(2024, 1, 1), sample_path=sample_path)
    assert result.missing == ["ZZZZ"]
    assert list(result.prices.columns) == ["SPY"]


def test_live_ticker_with_no_data_is_missing_but_stays_live(monkeypatch, sample_path):
    live = make_prices(["SPY", "AGG"], start="2023-01-02", end="2023-12-29")
    live["BAD"] = np.nan
    monkeypatch.setattr(data, "_download_live", lambda tickers, start, end: live)
    result = data.get_prices(["SPY", "BAD", "AGG"], date(2023, 1, 1), date(2024, 1, 1), sample_path=sample_path)
    assert result.source == "live"
    assert result.missing == ["BAD"]
    assert list(result.prices.columns) == ["SPY", "AGG"]


def test_tickers_are_deduplicated_and_uppercased(monkeypatch, sample_path):
    seen = {}

    def fake(tickers, start, end):
        seen["tickers"] = tickers
        return make_prices(tickers, start="2023-01-02", end="2023-12-29")

    monkeypatch.setattr(data, "_download_live", fake)
    result = data.get_prices(["spy", "AGG", " SPY "], date(2023, 1, 1), date(2024, 1, 1), sample_path=sample_path)
    assert seen["tickers"] == ["SPY", "AGG"]
    assert list(result.prices.columns) == ["SPY", "AGG"]


def test_window_clips_to_shortest_history(monkeypatch, sample_path):
    live = make_prices(["SPY", "NEW"], start="2023-01-02", end="2023-12-29")
    live.loc[:"2023-03-31", "NEW"] = np.nan
    monkeypatch.setattr(data, "_download_live", lambda tickers, start, end: live)
    result = data.get_prices(["SPY", "NEW"], date(2023, 1, 1), date(2024, 1, 1), sample_path=sample_path)
    assert result.prices.index[0] == pd.Timestamp("2023-04-03")
    assert result.clipped_start == date(2023, 4, 3)
    assert result.limiting_ticker == "NEW"
    assert not result.prices.isna().any().any()


def test_small_gap_is_not_reported_as_clipping(monkeypatch, sample_path):
    live = make_prices(["SPY", "NEW"], start="2023-01-02", end="2023-12-29")
    live.iloc[:2, 1] = np.nan
    monkeypatch.setattr(data, "_download_live", lambda tickers, start, end: live)
    result = data.get_prices(["SPY", "NEW"], date(2023, 1, 1), date(2024, 1, 1), sample_path=sample_path)
    assert result.clipped_start is None
    assert result.limiting_ticker is None
    assert result.prices.index[0] == live.index[2]


def test_timezone_aware_index_is_made_naive(monkeypatch, sample_path):
    live = make_prices(["SPY"], start="2023-01-02", end="2023-12-29")
    live.index = live.index.tz_localize("America/New_York")
    monkeypatch.setattr(data, "_download_live", lambda tickers, start, end: live)
    result = data.get_prices(["SPY"], date(2023, 1, 1), date(2024, 1, 1), sample_path=sample_path)
    assert result.prices.index.tz is None
    assert result.as_of == date(2023, 12, 29)


def test_no_valid_tickers_gives_empty_result(monkeypatch, sample_path):
    monkeypatch.setattr(data, "_download_live", offline)
    result = data.get_prices(["ZZZZ"], date(2021, 1, 1), date(2024, 1, 1), sample_path=sample_path)
    assert result.prices.empty
    assert result.missing == ["ZZZZ"]
    assert result.as_of is None


def test_to_returns():
    prices = pd.DataFrame({"A": [100.0, 110.0, 99.0]}, index=pd.bdate_range("2024-01-01", periods=3))
    returns = data.to_returns(prices)
    assert returns["A"].tolist() == pytest.approx([0.10, -0.10])
    assert returns.index[0] == prices.index[1]


def test_available_sample_tickers(sample_path):
    assert data.available_sample_tickers(sample_path) == ["AAPL", "AGG", "SPY"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_data.py -v`
Expected: `ImportError: cannot import name 'data' from 'risk'`.

- [ ] **Step 3: Implement `risk/data.py`**

```python
"""Price loading: live from Yahoo Finance, falling back to a bundled sample.

This is the only module in `risk/` that performs I/O.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal

import pandas as pd
import yfinance as yf

SAMPLE_PATH = Path(__file__).resolve().parent.parent / "data" / "sample_prices.parquet"
CLIP_TOLERANCE_DAYS = 5


@dataclass(frozen=True)
class PriceResult:
    prices: pd.DataFrame  # adjusted close, columns = valid tickers, aligned dates, no NaN
    source: Literal["live", "sample"]
    as_of: date | None  # last date in prices; None when there is no data
    missing: list[str]  # requested tickers with no data
    clipped_start: date | None  # set when the window was shortened to the shortest history
    limiting_ticker: str | None  # the ticker with the shortest history, when clipped


def _download_live(tickers: list[str], start: date, end: date) -> pd.DataFrame:
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False, timeout=10)
    if raw is None or raw.empty:
        return pd.DataFrame()
    close = raw["Close"]
    if isinstance(close, pd.Series):
        close = close.to_frame(name=tickers[0])
    return close


def load_sample(path: Path = SAMPLE_PATH) -> pd.DataFrame:
    return pd.read_parquet(path)


def available_sample_tickers(path: Path = SAMPLE_PATH) -> list[str]:
    return sorted(load_sample(path).columns)


def _sample_window(tickers: list[str], start: date, end: date, path: Path) -> pd.DataFrame:
    """Same window length as requested, shifted to end at the sample's last date if needed."""
    sample = load_sample(path)
    length = pd.Timestamp(end) - pd.Timestamp(start)
    end_ts = min(pd.Timestamp(end), sample.index.max())
    start_ts = end_ts - length
    present = [t for t in tickers if t in sample.columns]
    return sample.loc[start_ts:end_ts, present]


def _align(raw: pd.DataFrame, tickers: list[str], source: Literal["live", "sample"]) -> PriceResult:
    raw = raw.copy()
    raw.index = pd.DatetimeIndex(raw.index)
    if raw.index.tz is not None:
        raw.index = raw.index.tz_localize(None)
    raw = raw.sort_index()

    missing = [t for t in tickers if t not in raw.columns or raw[t].isna().all()]
    valid = [t for t in tickers if t not in missing]
    if not valid:
        return PriceResult(pd.DataFrame(), source, None, missing, None, None)

    first_valid = {t: raw[t].first_valid_index() for t in valid}
    limiting = max(first_valid, key=first_valid.get)
    prices = raw.loc[first_valid[limiting]:, valid].dropna(how="any")
    if prices.empty:
        return PriceResult(prices, source, None, missing, None, None)

    skipped = int((raw.index < prices.index[0]).sum())
    clipped = skipped > CLIP_TOLERANCE_DAYS
    return PriceResult(
        prices=prices,
        source=source,
        as_of=prices.index[-1].date(),
        missing=missing,
        clipped_start=prices.index[0].date() if clipped else None,
        limiting_ticker=limiting if clipped else None,
    )


def get_prices(
    tickers: list[str], start: date, end: date, sample_path: Path = SAMPLE_PATH
) -> PriceResult:
    wanted = list(dict.fromkeys(t.strip().upper() for t in tickers if t and t.strip()))
    source: Literal["live", "sample"] = "live"
    try:
        raw = _download_live(wanted, start, end)
    except Exception:
        raw = pd.DataFrame()
    if raw.empty or raw.isna().all().all():
        source = "sample"
        raw = _sample_window(wanted, start, end, sample_path)
    return _align(raw, wanted, source)


def to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily simple returns; the first row (no prior price) is dropped."""
    return (prices / prices.shift(1) - 1).iloc[1:]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_data.py -v`
Expected: 13 passed.

- [ ] **Step 5: Run the whole suite**

Run: `python -m pytest -q`
Expected: all tests from Tasks 1–7 pass.

- [ ] **Step 6: Commit**

```bash
git add risk/data.py tests/test_data.py
git commit -m "feat: add price loading with live fetch and sample fallback" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Bundled sample data

**Files:**
- Create: `scripts/refresh_sample.py`, `data/sample_prices.parquet` (generated, git-lfs)
- Test: `tests/test_sample_data.py`

**Interfaces:**
- Consumes: `risk.data.SAMPLE_PATH`, `risk.data.load_sample()` (Task 7).
- Produces: `scripts.refresh_sample.SAMPLE_TICKERS: list[str]` (the 21 tickers); the committed parquet file (DatetimeIndex named `Date`, one float column per ticker).

- [ ] **Step 1: Write the failing test**

`tests/test_sample_data.py`:
```python
from risk.data import SAMPLE_PATH, load_sample
from scripts.refresh_sample import SAMPLE_TICKERS


def test_sample_tickers_list():
    assert len(SAMPLE_TICKERS) == 21
    assert len(set(SAMPLE_TICKERS)) == 21
    assert "SPY" in SAMPLE_TICKERS


def test_sample_file_has_every_ticker_and_about_ten_years():
    sample = load_sample()
    assert set(SAMPLE_TICKERS) <= set(sample.columns)
    assert sample.index.is_monotonic_increasing
    assert (sample.index[-1] - sample.index[0]).days >= 3_600
    assert sample.isna().mean().max() < 0.01
    assert SAMPLE_PATH.stat().st_size < 1_000_000
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_sample_data.py -v`
Expected: `ModuleNotFoundError: No module named 'scripts.refresh_sample'`.

- [ ] **Step 3: Write `scripts/refresh_sample.py`**

```python
"""Regenerate data/sample_prices.parquet from Yahoo Finance.

Run locally (needs network):  python scripts/refresh_sample.py
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from risk.data import SAMPLE_PATH  # noqa: E402

SAMPLE_TICKERS = [
    "SPY", "AGG", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "VTI", "VXUS", "BND",
    "GLD", "VNQ", "TLT", "QQQ", "IWM", "EFA", "JPM", "XOM", "JNJ", "BRK-B",
]
YEARS = 10


def main() -> None:
    end = date.today()
    start = (pd.Timestamp(end) - pd.DateOffset(years=YEARS)).date()
    raw = yf.download(SAMPLE_TICKERS, start=start, end=end, auto_adjust=True, progress=False)
    close = raw["Close"]
    failed = [t for t in SAMPLE_TICKERS if t not in close.columns or close[t].isna().all()]
    if failed:
        raise SystemExit(f"Download failed for: {', '.join(failed)}. Try again later.")

    prices = close[SAMPLE_TICKERS].dropna(how="all").astype(float)
    prices.index = pd.DatetimeIndex(prices.index)
    if prices.index.tz is not None:
        prices.index = prices.index.tz_localize(None)
    prices.index.name = "Date"
    prices.columns.name = None

    SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(SAMPLE_PATH)
    size_kb = SAMPLE_PATH.stat().st_size / 1024
    print(
        f"Wrote {SAMPLE_PATH} — {prices.shape[1]} tickers, {len(prices)} days, "
        f"{prices.index[0]:%Y-%m-%d} to {prices.index[-1]:%Y-%m-%d}, {size_kb:.0f} KB"
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Generate the sample file (needs network)**

Run: `python scripts/refresh_sample.py`
Expected: `Wrote ...sample_prices.parquet — 21 tickers, ~2510 days, <start> to <end>, <size> KB` with size under 1000 KB. If it exits with `Download failed for: ...`, wait a few minutes and rerun (Yahoo rate limit).

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_sample_data.py -v`
Expected: 2 passed.

- [ ] **Step 6: Commit through git-lfs and verify**

```bash
git add scripts/refresh_sample.py data/sample_prices.parquet tests/test_sample_data.py
git commit -m "feat: add bundled 10-year sample prices and refresh script" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git lfs ls-files
```
Expected: `git lfs ls-files` lists `data/sample_prices.parquet`.

---

### Task 9: `charts.py`

**Files:**
- Create: `charts.py`
- Test: `tests/test_charts.py`

**Interfaces:**
- Consumes: shapes produced by Tasks 2, 3, 5, 6 (Series/DataFrames only; no imports from `risk/`).
- Produces (all return `plotly.graph_objects.Figure`):
  - `equity_curve_fig(portfolio: pd.Series, benchmark: pd.Series, benchmark_name: str)`
  - `drawdown_fig(drawdown: pd.Series)` (drawdown as fractions; plotted in %)
  - `var_histogram_fig(r: pd.Series, var_value: float, cvar_value: float, conf: float)` (two vertical lines at `-var*100`, `-cvar*100`)
  - `correlation_heatmap_fig(corr: pd.DataFrame)`
  - `risk_contribution_fig(rc: pd.DataFrame)` (expects columns `weight`, `pct_of_risk`)
  - `rolling_beta_fig(beta: pd.Series, benchmark_name: str)`

- [ ] **Step 1: Write the failing tests**

`tests/test_charts.py`:
```python
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

import charts


def series(values, name=None):
    return pd.Series(values, index=pd.bdate_range("2024-01-01", periods=len(values)), name=name, dtype=float)


def test_equity_curve_has_portfolio_and_benchmark():
    fig = charts.equity_curve_fig(series([100, 101, 102]), series([100, 100.5, 101]), "SPY")
    assert isinstance(fig, go.Figure)
    assert [trace.name for trace in fig.data] == ["Portfolio", "SPY"]


def test_drawdown_is_plotted_in_percent():
    fig = charts.drawdown_fig(series([0.0, -0.1, -0.05]))
    assert list(fig.data[0].y) == pytest.approx([0.0, -10.0, -5.0])


def test_var_histogram_has_two_reference_lines():
    r = series(np.random.default_rng(0).normal(0, 0.01, 500))
    fig = charts.var_histogram_fig(r, 0.02, 0.03, 0.95)
    assert len(fig.data) == 1
    assert len(fig.layout.shapes) == 2
    assert fig.layout.shapes[0].x0 == pytest.approx(-2.0)
    assert fig.layout.shapes[1].x0 == pytest.approx(-3.0)


def test_correlation_heatmap_uses_fixed_scale():
    corr = pd.DataFrame([[1.0, 0.3], [0.3, 1.0]], index=["A", "B"], columns=["A", "B"])
    fig = charts.correlation_heatmap_fig(corr)
    assert fig.data[0].zmin == -1 and fig.data[0].zmax == 1
    assert list(fig.data[0].x) == ["A", "B"]


def test_risk_contribution_groups_weight_and_risk():
    rc = pd.DataFrame({"weight": [0.6, 0.4], "contribution": [0.1, 0.01], "pct_of_risk": [0.9, 0.1]}, index=["SPY", "AGG"])
    fig = charts.risk_contribution_fig(rc)
    assert fig.layout.barmode == "group"
    assert [trace.name for trace in fig.data] == ["Weight", "Share of risk"]
    assert list(fig.data[1].y) == pytest.approx([90.0, 10.0])


def test_rolling_beta_has_reference_line_at_one():
    fig = charts.rolling_beta_fig(series([0.9, 1.1, 1.0], name="Rolling beta"), "SPY")
    assert len(fig.data) == 1
    assert fig.layout.shapes[0].y0 == 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_charts.py -v`
Expected: `ModuleNotFoundError: No module named 'charts'`.

- [ ] **Step 3: Implement `charts.py`**

```python
"""Plotly figure builders: data in, figures out. No risk math here."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

PORTFOLIO_COLOR = "#2563eb"
BENCHMARK_COLOR = "#9ca3af"
LOSS_COLOR = "#dc2626"
VAR_COLOR = "#f59e0b"


def _layout(fig: go.Figure, title: str, yaxis_title: str, **overrides) -> go.Figure:
    layout = dict(
        title=title,
        yaxis_title=yaxis_title,
        margin=dict(l=10, r=10, t=60, b=10),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    layout.update(overrides)
    fig.update_layout(**layout)
    return fig


def equity_curve_fig(portfolio: pd.Series, benchmark: pd.Series, benchmark_name: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=portfolio.index, y=portfolio, name="Portfolio",
                             line=dict(color=PORTFOLIO_COLOR, width=2)))
    fig.add_trace(go.Scatter(x=benchmark.index, y=benchmark, name=benchmark_name,
                             line=dict(color=BENCHMARK_COLOR, width=1.5)))
    return _layout(fig, "Growth of 100", "Value")


def drawdown_fig(drawdown: pd.Series) -> go.Figure:
    fig = go.Figure(go.Scatter(x=drawdown.index, y=drawdown * 100, name="Drawdown", fill="tozeroy",
                               line=dict(color=LOSS_COLOR, width=1)))
    return _layout(fig, "Drawdown from previous peak", "%", showlegend=False)


def var_histogram_fig(r: pd.Series, var_value: float, cvar_value: float, conf: float) -> go.Figure:
    fig = go.Figure(go.Histogram(x=r * 100, nbinsx=80, name="Daily returns",
                                 marker_color=PORTFOLIO_COLOR, opacity=0.75))
    fig.add_vline(x=-var_value * 100, line=dict(color=VAR_COLOR, dash="dash"),
                  annotation_text=f"VaR {conf:.0%}: -{var_value:.2%}", annotation_position="top left")
    fig.add_vline(x=-cvar_value * 100, line=dict(color=LOSS_COLOR, dash="dot"),
                  annotation_text=f"CVaR: -{cvar_value:.2%}", annotation_position="bottom left")
    return _layout(fig, "Distribution of daily returns (1-day, historical)", "Days",
                   xaxis_title="Daily return (%)", hovermode="closest", showlegend=False)


def correlation_heatmap_fig(corr: pd.DataFrame) -> go.Figure:
    labels = list(corr.columns)
    fig = go.Figure(go.Heatmap(
        z=corr.to_numpy(), x=labels, y=labels, zmin=-1, zmax=1, colorscale="RdBu_r",
        text=corr.round(2).to_numpy(), texttemplate="%{text}",
        hovertemplate="%{y} / %{x}: %{z:.2f}<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed")
    return _layout(fig, "Correlation of daily returns", "", hovermode="closest")


def risk_contribution_fig(rc: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=rc.index, y=rc["weight"] * 100, name="Weight", marker_color=BENCHMARK_COLOR))
    fig.add_trace(go.Bar(x=rc.index, y=rc["pct_of_risk"] * 100, name="Share of risk", marker_color=LOSS_COLOR))
    return _layout(fig, "Weight vs share of portfolio risk", "%", barmode="group")


def rolling_beta_fig(beta: pd.Series, benchmark_name: str) -> go.Figure:
    fig = go.Figure(go.Scatter(x=beta.index, y=beta, name="Rolling beta", line=dict(color=PORTFOLIO_COLOR)))
    fig.add_hline(y=1.0, line=dict(color=BENCHMARK_COLOR, dash="dash"),
                  annotation_text=f"β = 1 (moves with {benchmark_name})")
    return _layout(fig, f"Rolling 6-month beta vs {benchmark_name}", "Beta", showlegend=False)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_charts.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add charts.py tests/test_charts.py
git commit -m "feat: add Plotly chart builders" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 10: Streamlit app (`presets.py`, `app.py`)

**Files:**
- Create: `presets.py`, `app.py`
- Test: `tests/test_app.py`

**Interfaces:**
- Consumes: `Portfolio.from_inputs`, `Portfolio.drop`, `Portfolio.weight_series` (Task 1); `metrics.*` (Task 2); `var.var_table`, `var.historical_var` (Tasks 3–4); `attribution.*` (Task 5); `benchmark.*` (Task 6); `get_prices`, `to_returns`, `available_sample_tickers`, `load_sample`, `PriceResult`, `_download_live` seam (Task 7); sample file (Task 8); `charts.*` (Task 9).
- Produces: `presets.PRESETS: dict[str, dict[str, float]]` (weights in percent), `presets.CUSTOM = "Custom"`, `presets.DEFAULT_PRESET = "60/40 ETF"`; widget keys used by tests: `preset`, `range`, `conf`, `horizon`, `rf`, `benchmark`; session-state keys `holdings` (DataFrame with columns `Ticker`, `Weight %`) and `editor_version`.

- [ ] **Step 1: Create `presets.py`**

```python
"""Preset portfolios shown in the sidebar. Weights are in percent."""

BIG_TECH = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA"]

PRESETS: dict[str, dict[str, float]] = {
    "60/40 ETF": {"SPY": 60.0, "AGG": 40.0},
    "Big Tech": {ticker: 100 / len(BIG_TECH) for ticker in BIG_TECH},
    "Diversified": {"VTI": 30.0, "VXUS": 20.0, "BND": 25.0, "GLD": 10.0, "VNQ": 10.0, "TLT": 5.0},
}
CUSTOM = "Custom"
DEFAULT_PRESET = "60/40 ETF"
```

- [ ] **Step 2: Write the failing tests**

`tests/test_app.py`:
```python
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import risk.data
from presets import PRESETS
from risk.data import load_sample

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def offline(*args, **kwargs):
    raise ConnectionError("no network in tests")


@pytest.fixture(autouse=True)
def fresh_cache():
    st.cache_data.clear()
    yield
    st.cache_data.clear()


def new_app():
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    return at


def test_every_preset_ticker_and_default_benchmark_is_in_sample():
    sample = set(load_sample().columns)
    for weights in PRESETS.values():
        assert set(weights) <= sample
    assert "SPY" in sample


@pytest.mark.parametrize("preset", list(PRESETS))
def test_app_renders_each_preset_from_sample_data(monkeypatch, preset):
    monkeypatch.setattr(risk.data, "_download_live", offline)
    at = new_app()
    at.selectbox(key="preset").set_value(preset).run()
    assert not at.exception
    assert not at.error
    assert list(at.session_state["holdings"]["Ticker"]) == list(PRESETS[preset])
    assert any("Live data unavailable" in w.value for w in at.warning)
    assert len(at.tabs) == 4


def test_99_percent_ten_day_path_shows_horizon_note(monkeypatch):
    monkeypatch.setattr(risk.data, "_download_live", offline)
    at = new_app()
    at.radio(key="conf").set_value("99%")
    at.radio(key="horizon").set_value("10-day").run()
    assert not at.exception
    assert any("square-root-of-time" in c.value for c in at.caption)


def test_short_history_is_blocked_with_a_message(monkeypatch):
    def recent_ipo(tickers, start, end):
        idx = pd.bdate_range(end=pd.Timestamp(end) - pd.Timedelta(days=1), periods=40)
        return pd.DataFrame({t: np.linspace(100, 110, 40) for t in tickers}, index=idx)

    monkeypatch.setattr(risk.data, "_download_live", recent_ipo)
    at = new_app()
    assert not at.exception
    assert any("days of returns" in e.value for e in at.error)


def test_all_holdings_missing_shows_error(monkeypatch):
    def only_qqq(tickers, start, end):
        idx = pd.bdate_range(end=pd.Timestamp(end) - pd.Timedelta(days=1), periods=300)
        frame = pd.DataFrame(np.nan, index=idx, columns=tickers, dtype=float)
        if "QQQ" in frame.columns:
            frame["QQQ"] = np.linspace(100, 130, 300)
        return frame

    monkeypatch.setattr(risk.data, "_download_live", only_qqq)
    at = new_app()
    at.text_input(key="benchmark").set_value("qqq").run()
    assert not at.exception
    assert any("None of your holdings have price data" in e.value for e in at.error)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/test_app.py -v`
Expected: the preset-in-sample test passes; the AppTest tests fail (no `app.py` — AppTest reports the missing script / `at.exception`).

- [ ] **Step 4: Implement `app.py`**

```python
"""Streamlit UI for the Portfolio Risk Analyzer.

Wiring only: sidebar inputs -> risk/ -> charts.py. No risk math lives here.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import streamlit as st

import charts
from presets import CUSTOM, DEFAULT_PRESET, PRESETS
from risk import attribution, benchmark, metrics, var
from risk.data import PriceResult, available_sample_tickers, get_prices, to_returns
from risk.portfolio import Portfolio

RANGES = {"1Y": 1, "3Y": 3, "5Y": 5, "10Y": 10}
CONFIDENCE = {"95%": 0.95, "99%": 0.99}
HORIZON = {"1-day": 1, "10-day": 10}
MIN_RETURN_DAYS = 60
ROLLING_WINDOW = 126

VAR_CAPTION = (
    "At 95%, VaR barely differs between methods. CVaR averages the losses beyond the cutoff, "
    "so it shows the fat tail. This is why Basel's market-risk rules moved from VaR to Expected Shortfall."
)
HORIZON_NOTE = (
    "10-day figures use the square-root-of-time rule, which assumes independent daily returns "
    "and understates risk when volatility clusters."
)
FOOTER = "Assumes constant weights, rebalanced daily."


@st.cache_data(ttl=3600, show_spinner="Loading prices…")
def load_prices(tickers: tuple[str, ...], start: date, end: date) -> PriceResult:
    return get_prices(list(tickers), start, end)


def holdings_frame(preset: str) -> pd.DataFrame:
    weights = PRESETS[preset]
    return pd.DataFrame({"Ticker": list(weights), "Weight %": list(weights.values())})


def init_state() -> None:
    if "preset" not in st.session_state:
        st.session_state.preset = DEFAULT_PRESET
        st.session_state.holdings = holdings_frame(DEFAULT_PRESET)
        st.session_state.editor_version = 0


def on_preset_change() -> None:
    name = st.session_state.preset
    if name in PRESETS:
        st.session_state.holdings = holdings_frame(name)
        st.session_state.editor_version += 1  # new editor key discards old edits


def on_holdings_edit() -> None:
    st.session_state.preset = CUSTOM


def pct(x: float) -> str:
    return "n/a" if not np.isfinite(x) else f"{x:.1%}"


def ratio(x: float) -> str:
    return "n/a" if not np.isfinite(x) else f"{x:.2f}"


def sample_hint(result: PriceResult) -> str:
    if result.source != "sample":
        return ""
    return " Not in the bundled sample. Available: " + ", ".join(available_sample_tickers()) + "."


def sidebar() -> dict:
    sb = st.sidebar
    sb.header("Portfolio")
    sb.selectbox("Preset", [*PRESETS, CUSTOM], key="preset", on_change=on_preset_change)
    holdings = sb.data_editor(
        st.session_state.holdings,
        key=f"holdings_{st.session_state.editor_version}",
        num_rows="dynamic",
        hide_index=True,
        on_change=on_holdings_edit,
        column_config={
            "Ticker": st.column_config.TextColumn("Ticker"),
            "Weight %": st.column_config.NumberColumn("Weight %", format="%.2f"),
        },
    )
    sb.header("Settings")
    range_label = sb.radio("Date range", list(RANGES), index=2, horizontal=True, key="range")
    conf_label = sb.radio("VaR confidence", list(CONFIDENCE), horizontal=True, key="conf")
    horizon_label = sb.radio("VaR horizon", list(HORIZON), horizontal=True, key="horizon")
    rf_pct = sb.number_input("Risk-free rate (annual %)", value=4.0, step=0.25, key="rf")
    bench = sb.text_input("Benchmark", value="SPY", key="benchmark")
    return {
        "holdings": holdings,
        "years": RANGES[range_label],
        "conf": CONFIDENCE[conf_label],
        "conf_label": conf_label,
        "days": HORIZON[horizon_label],
        "horizon_label": horizon_label,
        "rf": rf_pct / 100,
        "benchmark": bench.strip().upper(),
    }


def render_overview(r: pd.Series, bench_r: pd.Series, bench: str, rf: float) -> None:
    cols = st.columns(5)
    cols[0].metric("CAGR", pct(metrics.cagr(r)))
    cols[1].metric("Volatility", pct(metrics.annualized_vol(r)))
    cols[2].metric("Sharpe", ratio(metrics.sharpe(r, rf)))
    cols[3].metric("Sortino", ratio(metrics.sortino(r, rf)))
    cols[4].metric("Max drawdown", pct(metrics.max_drawdown(r)))
    st.plotly_chart(charts.equity_curve_fig(metrics.equity_curve(r), metrics.equity_curve(bench_r), bench))
    st.plotly_chart(charts.drawdown_fig(metrics.drawdown_series(r)))


def render_tail_risk(asset_returns: pd.DataFrame, weights: pd.Series, r: pd.Series, inputs: dict) -> None:
    table, nu = var.var_table(asset_returns, weights, inputs["conf"], inputs["days"])
    st.subheader(f"{inputs['horizon_label']} loss at {inputs['conf_label']} confidence")
    st.dataframe(table.map(lambda x: f"{x:.2%}"))
    st.caption(
        f"Monte Carlo: 10,000 scenarios from a multivariate Student-t with df = {nu:.1f}, "
        "fitted to this portfolio's daily returns."
    )
    st.caption(VAR_CAPTION)
    if inputs["days"] > 1:
        st.caption(HORIZON_NOTE)
    hist = var.historical_var(r, inputs["conf"])
    st.plotly_chart(charts.var_histogram_fig(r, hist.var, hist.cvar, inputs["conf"]))


def render_diversification(asset_returns: pd.DataFrame, weights: pd.Series) -> None:
    left, right = st.columns(2)
    with left:
        st.plotly_chart(charts.correlation_heatmap_fig(attribution.correlation_matrix(asset_returns)))
    with right:
        st.plotly_chart(charts.risk_contribution_fig(attribution.risk_contribution(asset_returns, weights)))


def render_benchmark(r: pd.Series, bench_r: pd.Series, bench: str, rf: float) -> None:
    fit = benchmark.capm(r, bench_r, rf)
    cols = st.columns(3)
    cols[0].metric("Beta", ratio(fit.beta))
    cols[1].metric("Alpha (annualized)", pct(fit.alpha))
    cols[2].metric("Tracking error", pct(benchmark.tracking_error(r, bench_r)))
    if len(r) < ROLLING_WINDOW:
        st.info(f"Rolling beta needs at least {ROLLING_WINDOW} days of returns.")
    else:
        st.plotly_chart(charts.rolling_beta_fig(benchmark.rolling_beta(r, bench_r, ROLLING_WINDOW), bench))


def main() -> None:
    st.set_page_config(page_title="Portfolio Risk Analyzer", layout="wide")
    init_state()
    inputs = sidebar()
    st.title("Portfolio Risk Analyzer")
    st.caption("Return, tail-risk, diversification and benchmark analytics for a long-only portfolio.")

    holdings = inputs["holdings"]
    try:
        portfolio = Portfolio.from_inputs(zip(holdings["Ticker"], holdings["Weight %"]))
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    bench = inputs["benchmark"]
    if not bench:
        st.error("Enter a benchmark ticker.")
        st.stop()

    end = date.today()
    start = (pd.Timestamp(end) - pd.DateOffset(years=inputs["years"])).date()
    result = load_prices((*portfolio.tickers, bench), start, end)

    if result.source == "sample" and result.as_of is not None:
        st.warning(f"Live data unavailable — showing bundled sample through {result.as_of:%Y-%m-%d}.")
    if bench in result.missing:
        st.error(f"No price data for benchmark {bench}." + sample_hint(result))
        st.stop()
    dropped = [t for t in portfolio.tickers if t in result.missing]
    if dropped:
        try:
            portfolio = portfolio.drop(dropped)
        except ValueError:
            st.error(f"None of your holdings have price data ({', '.join(dropped)})." + sample_hint(result))
            st.stop()
        st.warning(f"No price data for {', '.join(dropped)} — dropped; remaining weights renormalized." + sample_hint(result))
    if result.clipped_start is not None:
        st.info(
            f"Window starts {result.clipped_start:%Y-%m-%d} "
            f"(shortest available history: {result.limiting_ticker})."
        )

    returns = to_returns(result.prices)
    if len(returns) < MIN_RETURN_DAYS:
        st.error(
            f"Only {len(returns)} days of returns available; at least {MIN_RETURN_DAYS} "
            "are needed for stable statistics."
        )
        st.stop()

    weights = portfolio.weight_series()
    asset_returns = returns[list(portfolio.tickers)]
    bench_r = returns[bench]
    r = metrics.portfolio_returns(asset_returns, weights)

    st.sidebar.caption("Normalized weights")
    st.sidebar.dataframe((weights * 100).round(2).rename("Weight %"))

    overview, tail, diversification, bench_tab = st.tabs(
        ["Overview", "VaR / CVaR", "Diversification", "Benchmark"]
    )
    with overview:
        render_overview(r, bench_r, bench, inputs["rf"])
    with tail:
        render_tail_risk(asset_returns, weights, r, inputs)
    with diversification:
        render_diversification(asset_returns, weights)
    with bench_tab:
        render_benchmark(r, bench_r, bench, inputs["rf"])
    st.caption(FOOTER)


main()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_app.py -v`
Expected: 7 passed (1 sample check + 3 presets + 99%/10-day + short history + all-missing).

- [ ] **Step 6: Run the whole suite**

Run: `python -m pytest -q`
Expected: all tests pass.

- [ ] **Step 7: Manual check in the browser**

Run: `streamlit run app.py`
Expected: browser opens at `http://localhost:8501`; the 60/40 ETF preset renders all four tabs within ~10 s (live data if Yahoo responds, otherwise the sample banner). Then:
- Switch to "Big Tech": holdings table updates to 6 tickers.
- Edit a weight in the table: preset dropdown switches to "Custom", charts update.
- Add a row with ticker `ZZZZ`, weight 10: warning names `ZZZZ`, analysis continues.
- Select 99% and 10-day: table values grow; horizon note appears.
Stop the server with Ctrl+C.

- [ ] **Step 8: Commit**

```bash
git add presets.py app.py tests/test_app.py
git commit -m "feat: add Streamlit dashboard with presets and smoke tests" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 11: Docker image and CI/CD workflow

**Files:**
- Create: `Dockerfile`, `.dockerignore`, `.github/workflows/ci.yml`
- Test: `tests/test_deploy_config.py`

**Interfaces:**
- Consumes: `requirements.txt`, `requirements-dev.txt`, `app.py`, `data/sample_prices.parquet`.
- Produces: container serving Streamlit on port 7860 (health at `/_stcore/health`); CI job `test`; job `deploy` that pushes `main` to the HF Space named by the GitHub Actions variable `HF_SPACE` (format `<hf-user>/<space-name>`) using secret `HF_TOKEN`, skipped while `HF_SPACE` is unset.

- [ ] **Step 1: Write the failing test**

`tests/test_deploy_config.py`:
```python
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def read(name):
    return (ROOT / name).read_text(encoding="utf-8").replace("\r\n", "\n")


def test_dockerfile_serves_streamlit_on_7860_with_python_312():
    dockerfile = read("Dockerfile")
    assert dockerfile.startswith("FROM python:3.12-slim")
    assert "--server.port=7860" in dockerfile
    assert "--server.address=0.0.0.0" in dockerfile
    assert "EXPOSE 7860" in dockerfile


def test_dockerignore_keeps_sample_data_in_image():
    ignored = read(".dockerignore").split()
    assert "data" not in ignored
    assert ".venv" in ignored


def test_ci_checks_out_lfs_and_deploys_only_after_tests():
    ci = read(".github/workflows/ci.yml")
    assert "lfs: true" in ci
    assert "needs: test" in ci
    assert 'python-version: "3.12"' in ci
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_deploy_config.py -v`
Expected: `FileNotFoundError` for `Dockerfile`.

- [ ] **Step 3: Write `Dockerfile`, `.dockerignore`, `.github/workflows/ci.yml`**

`Dockerfile`:
```dockerfile
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Hugging Face Spaces runs containers as uid 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .

EXPOSE 7860
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true"]
```

`.dockerignore`:
```
.git
.github
.venv
.pytest_cache
__pycache__
tests
docs
```

`.github/workflows/ci.yml`:
```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
  workflow_dispatch:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          lfs: true
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
          cache: pip
          cache-dependency-path: requirements*.txt
      - run: pip install -r requirements-dev.txt
      - run: pytest -q

  deploy:
    needs: test
    if: github.ref == 'refs/heads/main' && github.event_name != 'pull_request' && vars.HF_SPACE != ''
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
          lfs: true
      - name: Push to Hugging Face Space
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
          HF_SPACE: ${{ vars.HF_SPACE }}
        run: |
          git lfs install --local
          git push --force "https://${HF_SPACE%%/*}:${HF_TOKEN}@huggingface.co/spaces/${HF_SPACE}" HEAD:main
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_deploy_config.py -v`
Expected: 3 passed.

- [ ] **Step 5: Build and smoke-test the container** (Docker Desktop must be running)

```bash
docker build -t portfolio-risk-analyzer .
docker run -d --name pra -p 7860:7860 portfolio-risk-analyzer
for i in $(seq 1 30); do curl -sf http://localhost:7860/_stcore/health && break; sleep 2; done
docker logs pra | tail -n 5
docker rm -f pra
```
Expected: `curl` prints `ok`; logs show Streamlit listening on port 7860. Optionally open `http://localhost:7860` before removing the container and confirm the dashboard renders.

- [ ] **Step 6: Commit**

```bash
git add Dockerfile .dockerignore .github/workflows/ci.yml tests/test_deploy_config.py
git commit -m "build: add Docker image for HF Spaces and CI/deploy workflow" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 12: README and screenshots

**Files:**
- Create: `README.md`, `docs/screenshots/overview.png`, `docs/screenshots/var.png`, `docs/screenshots/diversification.png`, `docs/screenshots/benchmark.png` (git-lfs)
- Test: `tests/test_readme.py`

**Interfaces:**
- Consumes: Dockerfile port (Task 11), CI workflow name `ci.yml`.
- Produces: README with HF frontmatter. Assumed identifiers (confirm with the user in Step 1): GitHub repo `wnsghks622/portfolio-risk-analyzer`, HF Space `wnsghks622/portfolio-risk-analyzer`.

- [ ] **Step 1: Confirm identifiers with the user**

Ask: "I'll use GitHub repo `wnsghks622/portfolio-risk-analyzer` and Hugging Face Space `wnsghks622/portfolio-risk-analyzer` (Space URL `https://huggingface.co/spaces/wnsghks622/portfolio-risk-analyzer`). What is your Hugging Face username, if it differs?" Use the answers in the README below (replace the two `wnsghks622` occurrences in the HF links if the HF username differs).

- [ ] **Step 2: Write the failing test**

`tests/test_readme.py`:
```python
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def readme():
    return (ROOT / "README.md").read_text(encoding="utf-8").replace("\r\n", "\n")


def frontmatter():
    text = readme()
    assert text.startswith("---\n")
    block = text.split("---\n")[1]
    return dict(line.split(": ", 1) for line in block.strip().splitlines())


def test_frontmatter_matches_docker_space():
    meta = frontmatter()
    assert meta["sdk"] == "docker"
    assert meta["app_port"] == "7860"
    assert len(meta["short_description"]) <= 60


def test_referenced_screenshots_exist():
    paths = re.findall(r"\((docs/screenshots/[^)]+\.png)\)", readme())
    assert len(paths) == 4
    for path in paths:
        assert (ROOT / path).exists(), path
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_readme.py -v`
Expected: `FileNotFoundError` for `README.md`.

- [ ] **Step 4: Write `README.md`**

````markdown
---
title: Portfolio Risk Analyzer
emoji: 📉
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
short_description: Portfolio VaR, CVaR, drawdown and risk attribution
---

# Portfolio Risk Analyzer

[![CI](https://github.com/wnsghks622/portfolio-risk-analyzer/actions/workflows/ci.yml/badge.svg)](https://github.com/wnsghks622/portfolio-risk-analyzer/actions/workflows/ci.yml)

**Live demo:** https://huggingface.co/spaces/wnsghks622/portfolio-risk-analyzer

An interactive dashboard that measures how risky a stock/ETF portfolio is: returns and drawdowns, tail risk (VaR and CVaR by three methods), where the risk comes from, and how the portfolio moves against a benchmark. Pick a preset or type your own holdings.

| Overview | VaR / CVaR |
|---|---|
| ![Overview tab](docs/screenshots/overview.png) | ![VaR tab](docs/screenshots/var.png) |
| **Diversification** | **Benchmark** |
| ![Diversification tab](docs/screenshots/diversification.png) | ![Benchmark tab](docs/screenshots/benchmark.png) |

## What it shows

- **Overview:** CAGR, volatility, Sharpe, Sortino, max drawdown; growth of 100 vs benchmark; drawdown chart.
- **VaR / CVaR:** historical, parametric (normal) and Monte Carlo (Student-t) VaR and CVaR side by side, at 95%/99% and 1-day/10-day.
- **Diversification:** correlation heatmap; each holding's weight next to its share of total risk.
- **Benchmark:** beta, alpha, tracking error; rolling 6-month beta.

## Methodology

**Returns.** Simple daily returns from dividend- and split-adjusted closes. Simple returns add up across holdings (portfolio return = Σ weight × asset return), which log returns do not. Weights are held constant (rebalanced daily).

**Annualizing.** 252 trading days. Volatility = daily standard deviation × √252. CAGR is the compound growth rate of the equity curve. Sharpe uses the arithmetic mean: mean daily excess return × 252 ÷ annualized volatility. CAGR and the Sharpe numerator differ on purpose; compounding makes CAGR lower when returns are volatile.

**Sortino.** Like Sharpe, but the denominator only penalizes returns below the risk-free rate: downside deviation = √(mean of min(return − rf, 0)²) taken over *all* days, × √252.

**Max drawdown.** The worst fall from a previous peak (the starting capital counts as a peak).

**VaR and CVaR.** VaR at 95% is the daily loss exceeded on only 5% of days. CVaR (Expected Shortfall) is the average loss on those worst 5% of days: "when a bad day comes, how bad is it on average?"
- *Historical:* read straight off past returns, no distribution assumed; assumes the past window is representative.
- *Parametric:* assumes returns are normal. Fast, but real returns have fatter tails.
- *Monte Carlo (Student-t):* 10,000 simulated days from a multivariate Student-t with the same means and covariances as the data. Its degrees of freedom are fitted to the portfolio's returns by maximum likelihood (clipped to 3–30). A normal Monte Carlo would only reproduce the parametric number, so the t-distribution is what makes the comparison informative. All assets share one random "shock size" per scenario, so they tend to crash together, as in real markets.
- *What to look for:* at 95%, VaR barely differs across methods (a fat-tailed distribution has *less* mass right at the 95% cutoff), but CVaR is higher under the fat-tailed models. VaR only marks the cutoff; CVaR looks past it. This is why Basel's market-risk rules (FRTB) replaced 99% VaR with 97.5% Expected Shortfall.
- *10-day horizon:* 1-day figures × √10, which assumes independent days and understates risk when volatility clusters.

**Risk contribution.** Euler decomposition: contributionᵢ = wᵢ (Σw)ᵢ / σₚ. Contributions add up exactly to portfolio volatility. In a 60/40 stock/bond portfolio, stocks usually carry 90%+ of the risk despite being 60% of the money.

**Beta, alpha, tracking error.** OLS regression of daily excess portfolio returns on excess benchmark returns (CAPM form). Alpha is annualized (× 252). Tracking error = annualized standard deviation of (portfolio − benchmark) returns. Rolling beta uses a 126-day (6-month) window.

## Data

Prices come live from Yahoo Finance via `yfinance`. If that fails (Yahoo often rate-limits shared cloud servers), the app falls back to a bundled 10-year sample of 21 tickers and says so in a banner. Refresh the sample with `python scripts/refresh_sample.py`.

## Run locally

```bash
py -3.12 -m venv .venv            # or: python3.12 -m venv .venv
source .venv/Scripts/activate     # macOS/Linux: source .venv/bin/activate
pip install -r requirements-dev.txt
streamlit run app.py
pytest
```

## Project layout

```
app.py          Streamlit UI (wiring only)
charts.py       Plotly figures
presets.py      preset portfolios
risk/           pure analytics package (no Streamlit)
  data.py         live prices + sample fallback
  portfolio.py    holdings validation
  metrics.py      CAGR, volatility, Sharpe, Sortino, drawdown
  var.py          VaR / CVaR (historical, parametric, Student-t Monte Carlo)
  attribution.py  correlation, risk contribution
  benchmark.py    beta, alpha, tracking error
tests/          pytest suite (no network)
```

## Limitations

- Constant weights rebalanced daily; a buy-and-hold portfolio would drift.
- 10-day figures use √10 scaling, which assumes independent daily returns.
- The risk-free rate is a constant you choose, not a historical T-bill series.
- Monte Carlo uses the sample covariance and one shared degrees-of-freedom value for all assets.
- Sample data is only as fresh as the last refresh.
````

- [ ] **Step 5: Capture screenshots (USER STEP — ask the user to do this)**

Ask the user to:
1. Run `streamlit run app.py`, open `http://localhost:8501` in a window about 1400 px wide, preset "60/40 ETF".
2. Screenshot each tab's main area and save as PNG to exactly: `docs/screenshots/overview.png`, `docs/screenshots/var.png`, `docs/screenshots/diversification.png`, `docs/screenshots/benchmark.png`.
3. Reply when done. Wait for the reply before Step 6.

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest tests/test_readme.py -v`
Expected: 2 passed.

- [ ] **Step 7: Commit (PNGs go through git-lfs)**

```bash
git add README.md docs/screenshots/*.png tests/test_readme.py
git commit -m "docs: add README with methodology and screenshots" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git lfs ls-files
```
Expected: `git lfs ls-files` lists the four PNGs and the parquet file.

---

### Task 13: Publish to GitHub and Hugging Face (user-gated)

Creating public repositories and pushing code is outward-facing. **Get explicit user approval before Step 2.**

**Files:** none (configuration only; a README link fix if identifiers changed).

**Interfaces:**
- Consumes: CI workflow from Task 11 (`HF_SPACE` variable, `HF_TOKEN` secret).
- Produces: public GitHub repo, live HF Space.

- [ ] **Step 1: Final local verification**

Run: `python -m pytest -q`
Expected: all tests pass. Show the output to the user.

- [ ] **Step 2: Ask for approval to publish**

Ask: "Ready to create the public GitHub repo `wnsghks622/portfolio-risk-analyzer` and push `main`? This makes the code public." Wait for yes.

- [ ] **Step 3: Create the GitHub repo and push**

```bash
gh repo create wnsghks622/portfolio-risk-analyzer --public --source . --remote origin --push
gh run list --limit 1
```
Expected: repo created; one CI run listed. `gh run watch` until the `test` job succeeds; `deploy` is skipped because `HF_SPACE` is unset.

- [ ] **Step 4: USER STEP — create the Space and token**

Ask the user to:
1. Create a Space at https://huggingface.co/new-space: name `portfolio-risk-analyzer`, SDK **Docker**, template **Blank**, visibility Public.
2. Create a token with **write** access at https://huggingface.co/settings/tokens.
3. Run in this session: `! gh secret set HF_TOKEN` and paste the token when prompted (keeps the token out of the transcript).

- [ ] **Step 5: Point CI at the Space and deploy**

```bash
gh variable set HF_SPACE --body "<hf-username>/portfolio-risk-analyzer"
gh workflow run ci.yml --ref main
gh run watch
```
(`<hf-username>` is the Hugging Face username confirmed in Task 12 Step 1.)
Expected: both `test` and `deploy` jobs succeed.

- [ ] **Step 6: Verify the live demo**

Open `https://huggingface.co/spaces/<hf-username>/portfolio-risk-analyzer`, wait for the build to finish (Space status "Running"), then confirm: the preset portfolio renders all four tabs within ~10 s and there are no error messages (the sample banner is acceptable). If the build fails, read the Space's build logs before changing anything.

- [ ] **Step 7: Fix README links if identifiers changed, then push**

Only if the HF username differs from the README: update the two HF links in `README.md`, then:
```bash
git add README.md
git commit -m "docs: point README at live Space" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push
```
Expected: CI passes and redeploys.
