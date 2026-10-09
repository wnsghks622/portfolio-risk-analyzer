# Portfolio Risk Analyzer — Design Spec

- **Date:** 2026-10-09
- **Status:** Approved in brainstorming (sections 1–4); awaiting written-spec review
- **Source notes:** `docs/brainstorm/2026-10-08-portfolio-risk-analyzer-notes.md`

## 1. Purpose and success criteria

A portfolio/resume project aimed at data science and analytics roles. The interactive dashboard and clear visuals are the main signal; clean code layout, tests, and CI are secondary signal.

**Success:** a recruiter opens the public link, picks a preset portfolio, and sees all charts within 10 seconds with no errors, even when live market data is unavailable.

**Constraints:** Python only; solo project shippable in a few weekends; equities/ETFs with daily data only; long-only portfolios.

**In scope (v1):**

- A. Core stats: CAGR, annualized volatility, Sharpe, Sortino, max drawdown, equity curve, drawdown chart
- B. VaR and CVaR by three methods (historical, parametric normal, Monte Carlo Student-t), compared side by side
- C. Correlation heatmap and per-asset risk contribution
- D. Benchmark comparison: beta, alpha, tracking error, rolling beta

**Out of scope (v1):** stress-test scenario replay (2008 / 2020 / 2022), portfolio optimization / efficient frontier, short positions, intraday data, non-equity instruments, user accounts or saved portfolios.

## 2. Approach

Streamlit + Plotly UI on top of a separate pure-Python analytics package (`risk/`) with pytest coverage, deployed to Hugging Face Spaces via the Docker SDK.

Rejected alternatives: Plotly Dash (callback boilerplate for little gain at this size), Gradio (built for ML demos, weak for multi-tab dashboards).

## 3. Architecture

```
portfolio_risk_analyzer/
├── app.py                       # Streamlit UI only: inputs, layout, wiring
├── risk/                        # pure analytics package, no Streamlit imports
│   ├── __init__.py
│   ├── data.py                  # prices: yfinance → bundled sample fallback
│   ├── portfolio.py             # Portfolio dataclass: tickers, weights, validation
│   ├── metrics.py               # returns, CAGR, vol, Sharpe, Sortino, drawdown
│   ├── var.py                   # historical / parametric / Monte Carlo VaR + CVaR
│   ├── attribution.py           # correlation matrix, risk contribution
│   └── benchmark.py             # beta, alpha, tracking error, rolling beta
├── charts.py                    # Plotly figure builders: DataFrames in, figures out
├── data/sample_prices.parquet   # bundled fallback prices (see 5.1)
├── scripts/refresh_sample.py    # regenerates the sample file locally
├── tests/                       # pytest, one file per risk module + app smoke test
├── .github/workflows/           # CI (pytest) and HF Space sync
├── Dockerfile
├── requirements.txt             # pinned versions
└── README.md                    # HF frontmatter, screenshots, methodology, live link
```

**Boundaries:**

- `risk/` takes pandas objects and returns numbers, Series, DataFrames, or small dataclasses. `data.py` is the only module that does I/O.
- `charts.py` builds figures only; it does no math beyond formatting.
- `app.py` wires inputs to `risk/` and `charts.py`; it does no math.
- Each `risk/` module should stay around 50–150 lines.

**Dependencies:** pandas, numpy, scipy, yfinance, pyarrow, streamlit, plotly; pytest for tests. No statsmodels (OLS is done with numpy).

## 4. UI and data flow

### 4.1 Sidebar inputs

| Input | Control | Default |
|---|---|---|
| Preset | dropdown: "60/40 ETF", "Big Tech", "Diversified", "Custom" | "60/40 ETF" |
| Holdings | `st.data_editor` table: ticker, weight | filled from preset |
| Date range | 1Y / 3Y / 5Y / 10Y (end date = today) | 5Y |
| VaR confidence | 95% / 99% | 95% |
| VaR horizon | 1-day / 10-day | 1-day |
| Risk-free rate | number, annual % | 4.0 |
| Benchmark | ticker text | SPY |

**Presets:**

- **60/40 ETF:** SPY 60%, AGG 40%
- **Big Tech:** AAPL, MSFT, GOOGL, AMZN, META, NVDA, equal weight
- **Diversified:** VTI 30%, VXUS 20%, BND 25%, GLD 10%, VNQ 10%, TLT 5%

Editing the holdings table of a preset switches the preset label to "Custom". Weights are normalized to sum to 100%, and the normalized weights are displayed.

### 4.2 Flow

1. Inputs → `Portfolio` (validated, normalized).
2. `data.get_prices(tickers + [benchmark], start, end)` → `PriceResult` with a `source` flag (`"live"` or `"sample"`). `app.py` wraps this call in `st.cache_data` (TTL 1 hour, keyed on tickers + dates) so `risk/` stays Streamlit-free.
3. Prices → aligned daily simple returns.
4. Returns → `metrics`, `var`, `attribution`, `benchmark`.
5. Results → `charts.py` → four tabs.

### 4.3 Tabs

1. **Overview:** KPI tiles (CAGR, volatility, Sharpe, Sortino, max drawdown); equity curve vs benchmark (both start at 100); drawdown chart.
2. **VaR / CVaR:** comparison table (rows: historical, parametric, Monte Carlo; columns: VaR, CVaR) at the chosen confidence and horizon; histogram of daily portfolio returns with VaR and CVaR lines; the fitted Student-t df is shown under the table. A caption explains: "At 95%, VaR barely differs between methods. CVaR averages the losses beyond the cutoff, so it shows the fat tail. This is why Basel's market-risk rules moved from VaR to Expected Shortfall."
3. **Diversification:** correlation heatmap; grouped bar chart of weight % vs risk contribution % per asset.
4. **Benchmark:** beta, annualized alpha, tracking error tiles; rolling 126-day (6-month) beta chart.

**Banners and notes:**

- When `source == "sample"`: "Live data unavailable — showing bundled sample through `<as_of date>`."
- Under the VaR tab when horizon is 10-day: "10-day figures use the square-root-of-time rule, which assumes independent daily returns and understates risk when volatility clusters."
- Footer: "Assumes constant weights, rebalanced daily."

## 5. Module interfaces

Signatures are the contract between modules; names may be adjusted during planning if the plan states the change. `returns` means a DataFrame of daily simple returns (rows = dates, columns = tickers); `r` means a Series of daily portfolio returns. All rates passed in are annual decimals (0.04 = 4%) unless named `_daily`.

### 5.1 `risk/data.py`

```python
@dataclass
class PriceResult:
    prices: pd.DataFrame        # adjusted close, columns = valid tickers, aligned dates
    source: Literal["live", "sample"]
    as_of: date                 # last date in prices
    missing: list[str]          # requested tickers with no data
    clipped_start: date | None  # set when the window was shortened to fit the shortest history

def get_prices(tickers: list[str], start: date, end: date) -> PriceResult
def to_returns(prices: pd.DataFrame) -> pd.DataFrame   # pct_change, drop first row
```

- Live path: `yfinance.download(tickers, start, end, auto_adjust=True, progress=False)`, close prices only.
- Requested tickers are de-duplicated first (the benchmark may also be a holding, e.g. SPY in "60/40 ETF").
- Fallback to the sample file on any exception, timeout, or an empty result. In sample mode the window keeps its requested length but is shifted to end at the sample's last date, so a 5Y request still gets 5 years even if the sample is a few months old.
- A ticker whose column is entirely NaN goes in `missing`. In sample mode, tickers not in the sample also go in `missing`, and the app message lists available sample tickers.
- Alignment: drop `missing` tickers, start the window at the latest first-valid date across remaining tickers, then drop any rows that still contain NaN. Set `clipped_start` when the resulting start is more than 5 trading days after the requested start.
- **Sample file:** 21 tickers, 10 years of daily adjusted close, under 1 MB: SPY, AGG, AAPL, MSFT, GOOGL, AMZN, META, NVDA, VTI, VXUS, BND, GLD, VNQ, TLT, QQQ, IWM, EFA, JPM, XOM, JNJ, BRK-B. Covers every preset plus extras for custom portfolios. Regenerated with `scripts/refresh_sample.py` and committed.

### 5.2 `risk/portfolio.py`

```python
@dataclass(frozen=True)
class Portfolio:
    tickers: tuple[str, ...]
    weights: tuple[float, ...]   # normalized, sum to 1

    @classmethod
    def from_inputs(cls, holdings: dict[str, float]) -> "Portfolio"
    def drop(self, tickers: list[str]) -> "Portfolio"   # remove + renormalize
```

Validation in `from_inputs`: tickers stripped and uppercased; duplicate tickers have their weights summed; empty ticker rows ignored; any negative weight → `ValueError` (long-only); no tickers or a zero weight sum → `ValueError`; otherwise normalize.

### 5.3 `risk/metrics.py`

```python
def portfolio_returns(returns: pd.DataFrame, weights: pd.Series) -> pd.Series   # r_p = returns @ w
def equity_curve(r: pd.Series, start: float = 100.0) -> pd.Series             # start * cumprod(1 + r)
def cagr(r: pd.Series) -> float
def annualized_vol(r: pd.Series) -> float
def sharpe(r: pd.Series, rf: float) -> float
def sortino(r: pd.Series, rf: float) -> float
def drawdown_series(r: pd.Series) -> pd.Series
def max_drawdown(r: pd.Series) -> float        # negative number, e.g. -0.34
```

### 5.4 `risk/var.py`

```python
@dataclass
class VarResult:
    method: str        # "Historical" | "Parametric (normal)" | "Monte Carlo (Student-t)"
    var: float         # positive loss fraction, e.g. 0.021 = 2.1%
    cvar: float        # positive loss fraction, always >= var

def historical_var(r: pd.Series, conf: float) -> VarResult
def parametric_var(r: pd.Series, conf: float) -> VarResult
def fit_t_df(r: pd.Series) -> float
def monte_carlo_var(returns: pd.DataFrame, weights: pd.Series, conf: float,
                    df: float, n_sims: int = 10_000, seed: int = 42) -> VarResult
def scale_horizon(result: VarResult, days: int) -> VarResult
def var_table(returns: pd.DataFrame, weights: pd.Series, conf: float, days: int) -> tuple[pd.DataFrame, float]
                    # returns (table, fitted df)
```

### 5.5 `risk/attribution.py`

```python
def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame
def risk_contribution(returns: pd.DataFrame, weights: pd.Series) -> pd.DataFrame
                    # columns: weight, contribution (annualized vol units), pct_of_risk
```

### 5.6 `risk/benchmark.py`

```python
@dataclass
class CapmResult:
    alpha: float       # annualized
    beta: float

def capm(r: pd.Series, bench: pd.Series, rf: float) -> CapmResult
def tracking_error(r: pd.Series, bench: pd.Series) -> float
def rolling_beta(r: pd.Series, bench: pd.Series, window: int = 126) -> pd.Series
```

### 5.7 `charts.py`

One function per figure, each taking the data it plots and returning a `plotly.graph_objects.Figure`: `equity_curve_fig`, `drawdown_fig`, `var_histogram_fig`, `correlation_heatmap_fig`, `risk_contribution_fig`, `rolling_beta_fig`.

## 6. Methodology

This section is also the basis of the README's methodology write-up, so each choice includes its reason.

**Returns.** Simple daily returns from adjusted close (adjusted for dividends and splits). Simple returns are used rather than log returns because they aggregate across assets: r_p = Σ wᵢrᵢ. Weights are held constant (rebalanced daily).

**Annualization.** 252 trading days.

- Volatility = σ_daily × √252 (sample std, ddof = 1).
- CAGR (the KPI tile) is geometric: (equity_end / equity_start)^(252 / n_days) − 1.
- Sharpe uses the arithmetic mean: mean(rₜ − rf_daily) × 252 / (σ_daily × √252), with rf_daily = rf / 252. CAGR and the Sharpe numerator therefore differ by design; the README states both definitions.

**Sortino.** Numerator: mean(rₜ − rf_daily) × 252. Denominator: downside deviation over **all** days, DD = √(mean(min(rₜ − rf_daily, 0)²)) × √252. (Taking the std of only the negative days is a common bug and is tested against.)

**Max drawdown.** Worst peak-to-trough decline of the equity curve: min over t of (equityₜ / running_max(equity)ₜ − 1).

**VaR and CVaR.** Reported as positive loss fractions at confidence `conf` (α = 1 − conf).

- **Historical:** VaR = −quantile(r, α) (numpy default linear interpolation). CVaR = −mean of returns ≤ −VaR. No distribution assumption; assumes the past window is representative. Current weights are applied to past asset returns.
- **Parametric (normal):** with sample μ and σ of r and z = `norm.ppf(α)` (−1.645 at 95%, −2.326 at 99%): VaR = −(μ + zσ); CVaR = −μ + σ·φ(z)/α. Fast, but assumes thin tails.
- **Monte Carlo (Student-t):** 10,000 one-day scenarios, seed 42.
  - Draw: X = μ + Z / √(W / ν), with Z ~ MVN(0, Σ·(ν − 2)/ν) and W ~ χ²(ν) shared by all assets in a scenario. μ and Σ are the sample mean vector and covariance of asset returns.
  - The (ν − 2)/ν factor makes the simulated covariance equal Σ, so only tail shape differs from the normal model.
  - The shared W means a bad draw hits every asset at once (tail dependence: crashes come together), which a multivariate normal lacks.
  - Portfolio scenario returns = X @ w; VaR and CVaR are computed from them exactly as in the historical method.
  - **Why not a normal Monte Carlo:** for a linear portfolio, wᵀX under a multivariate normal is itself normal with mean wᵀμ and variance wᵀΣw, so it reproduces the parametric answer up to sampling noise (~1–2% at 10,000 draws) and adds nothing to the comparison.
- **Degrees of freedom ν (`fit_t_df`):** maximum-likelihood fit with `scipy.stats.t.fit` on the portfolio's daily returns, keeping only the df estimate, clipped to [3, 30]. If the fit raises or returns a non-finite value, use ν = 5. The lower clip at 3 is required, because the rescaling needs ν > 2. Known bias: volatility clustering makes returns look heavier-tailed than they are, so the fitted ν tends to be low, which errs conservative. The fit is noisy on 1Y windows; clipping bounds it.
- **Expected result (stated in the UI caption):** a vol-matched Student-t has *lower* VaR than the normal at 95% (1.561σ vs 1.645σ at ν = 5) and *higher* VaR at 99% (2.606σ vs 2.326σ); the crossover is near 97.3%. CVaR at 95% is higher under the t (2.239σ vs 2.063σ). VaR only marks the cutoff, while CVaR averages the losses beyond it, so CVaR is what reveals fat tails at 95%. This matches the reason Basel's FRTB replaced 99% VaR with 97.5% Expected Shortfall (ES also satisfies subadditivity, which VaR can violate).
- **10-day horizon:** multiply the 1-day VaR and CVaR of every method by √10 (square-root-of-time rule, treating the daily mean as negligible). The UI note in 4.3 states the i.i.d. assumption.

**Correlation.** Pearson correlation of daily asset returns.

**Risk contribution (Euler decomposition).** With annualized covariance Σ and σₚ = √(wᵀΣw): RCᵢ = wᵢ(Σw)ᵢ / σₚ. Contributions sum exactly to σₚ; displayed as % of σₚ next to weight %. Expected insight: in a 60/40 stock/bond portfolio, stocks carry roughly 90%+ of the risk (checked: σ 16% vs 5%, correlation 0–0.2 → 94–96%).

**CAPM beta and alpha.** OLS via numpy of daily excess returns: (rₜ − rf_daily) = α + β(bₜ − rf_daily) + εₜ. Alpha is annualized as α_daily × 252.

**Tracking error.** std(rₜ − bₜ) × √252.

**Rolling beta.** The same OLS slope over a trailing 126-day window, recomputed daily.

**Undefined values.** If a denominator is zero (zero volatility, zero downside deviation, zero benchmark variance), the metric is NaN and shown as "n/a".

## 7. Error handling

| Condition | Behavior |
|---|---|
| Unknown or delisted ticker | Warn naming it, drop it, renormalize the remaining weights (shown), continue |
| No valid tickers left | Error message, stop rendering |
| Benchmark ticker invalid | Error message naming it, stop rendering |
| Live fetch fails (exception, rate limit, timeout, empty) | Fall back to sample data, show sample banner |
| Ticker not in sample (sample mode) | Treated as missing; message lists available sample tickers |
| Window shorter than requested (`clipped_start` set) | Info message: "Window starts `<date>` (shortest available history: `<ticker>`)" |
| Fewer than 60 aligned return days | Block with message: too little data for stable statistics |
| Negative weight | Validation error (long-only in v1) |
| Weights sum to zero / no holdings | Validation error |

## 8. Testing

pytest, one file per `risk/` module plus an app smoke test. Tests never touch the network.

- **Hand-calculated values:** small return arrays with answers worked out by hand for CAGR, annualized vol, Sharpe, Sortino (including a case where "std of negative days only" gives a different number), and max drawdown on a constructed price path.
- **Invariants:** risk contributions sum to σₚ; benchmark regressed on itself gives β = 1, α = 0; a series equal to 2× the benchmark gives β = 2; CVaR ≥ VaR for every method; the same seed gives identical Monte Carlo results.
- **Statistical checks on synthetic data:**
  - On a large normal sample, historical VaR ≈ parametric VaR within tolerance.
  - Rescaled Student-t draws preserve the target covariance within tolerance.
  - `fit_t_df` recovers ν within ±1 on simulated t(4) data; returns a value in [3, 30] on normal data; returns 5 when the fit fails.
  - On t(5) data, the Monte Carlo VaR is below parametric at 95% and above it at 99% (locks in the crossover behavior from section 6).
- **`data.py`:** yfinance mocked to raise → result has `source == "sample"`; all-NaN column → ticker in `missing`; alignment clips to the shortest history and sets `clipped_start`.
- **`portfolio.py`:** negative weight rejected; zero sum rejected; normalization; duplicate tickers summed; `drop` renormalizes.
- **App smoke test:** `streamlit.testing.v1.AppTest` runs `app.py` headless with yfinance mocked to fail (forcing sample data), once per preset, and asserts no exceptions. This enforces the "demo never breaks" requirement.
- **CI:** GitHub Actions runs pytest on every push and pull request. README shows the status badge.

## 9. Deployment

- **Hugging Face Spaces, Docker SDK.** `Dockerfile` based on `python:3.12-slim`: install pinned `requirements.txt`, copy the app, run `streamlit run app.py --server.port 7860 --server.address 0.0.0.0`.
- **README frontmatter** required by HF: `sdk: docker`, `app_port: 7860`, plus title/emoji/colors. GitHub renders it as a small table at the top of the README; accepted.
- **Repo flow:** GitHub is the primary repository. A GitHub Actions job pushes to the HF Space only after the test job passes, so a broken build never reaches the live demo. The HF token is stored as a GitHub Actions secret.
- **Sample data:** regenerated locally with `scripts/refresh_sample.py` and committed; not fetched at build time.
- **README contents:** live demo link, test badge, screenshots of each tab, methodology section (from section 6, plain language), how to run locally (`pip install -r requirements.txt`, `streamlit run app.py`, `pytest`), and a limitations section (daily rebalancing, i.i.d. horizon scaling, constant risk-free rate, sample data can be stale).

## 10. Known limitations (documented, accepted for v1)

- Constant weights with daily rebalancing; the equity curve differs from buy-and-hold.
- √10 horizon scaling assumes i.i.d. returns.
- Constant user-supplied risk-free rate, not a historical T-bill series.
- Sample covariance and a single shared ν across assets in the Monte Carlo model.
- Sample data is only as fresh as the last `refresh_sample.py` run.
