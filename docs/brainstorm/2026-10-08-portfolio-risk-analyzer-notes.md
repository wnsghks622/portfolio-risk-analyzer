# Portfolio Risk Analyzer — Brainstorming Notes (DRAFT, IN PROGRESS)

> Status: brainstorming paused 2026-10-08. NOT an approved spec.
> Path: architectural (new project) → questions → approaches → design sections → written spec → implementation plan.
> No code written yet. Resume at "Next steps" below.

## Agreed understanding

**Purpose:** portfolio/resume project to show employers.
**Target roles:** data science / analytics → interactive dashboard + clear visuals is the priority signal.
**Data:** live `yfinance`, with automatic fallback to a bundled sample dataset (demo must never break; Yahoo often throttles shared cloud IPs like HF Spaces).
**v1 features:**
- A. Core stats: annualized return (CAGR), volatility, Sharpe, Sortino, max drawdown, equity curve, drawdown chart
- B. VaR + CVaR — historical, parametric, Monte Carlo, compared side by side
- C. Correlation heatmap + per-asset risk contribution
- D. Benchmark vs SPY: beta, alpha, tracking error

**Stretch (not v1):** stress-test scenarios (2008 / 2020 / 2022 replay), portfolio optimization / efficient frontier.
**Showcase:** public live demo on **Hugging Face Spaces** + GitHub repo with polished README and screenshots.

**Assumptions (confirmed by user):** Python stack; solo, shippable in a few weekends; equities/ETFs, daily data only; success = recruiter opens link, picks preset, sees charts in under 10s with no errors; clean code layout + tests are bonus signal.

## Approach — APPROVED

Streamlit + Plotly UI on top of a separate pure-Python analytics package (`risk/`) with pytest coverage. Deployed to HF Spaces via the Docker SDK.
Rejected: Plotly Dash (too much callback boilerplate), Gradio (ML-demo oriented, weak for dashboards).

## Section 1: Architecture & components — APPROVED

```
portfolio_risk_analyzer/
├── app.py                  # Streamlit UI only: inputs, layout, charts
├── risk/                   # pure analytics package, no Streamlit imports
│   ├── data.py             # fetch prices: yfinance → cache → bundled sample fallback
│   ├── portfolio.py        # Portfolio dataclass: tickers, weights, validation
│   ├── metrics.py          # return, vol, Sharpe, Sortino, max drawdown
│   ├── var.py              # historical / parametric / Monte Carlo VaR + CVaR
│   ├── attribution.py      # correlation matrix, risk contribution per asset
│   └── benchmark.py        # beta, alpha, tracking error vs SPY
├── charts.py               # Plotly figure builders: DataFrames in, figures out
├── data/sample_prices.parquet   # bundled fallback: ~20 tickers + SPY, 10 years
├── scripts/refresh_sample.py    # regenerates sample file locally
├── tests/                  # pytest, one file per risk module
├── Dockerfile              # HF Spaces (Docker SDK)
├── requirements.txt
└── README.md               # screenshots, methodology, live link
```

Boundaries: `risk/` takes pandas returns, returns numbers/DataFrames, no I/O except `data.py`. `charts.py` = figures only, no math. `app.py` = wiring only, no math. Each `risk/` module ~50–150 lines.

## Section 2: Data flow & UI — APPROVED

**Sidebar inputs:**
- Preset dropdown: "60/40 ETF", "Big Tech", "Diversified", "Custom" (preset = zero typing for recruiters)
- Holdings table (`st.data_editor`): ticker, weight; weights normalized to 100%
- Date range: 1Y / 3Y / 5Y / 10Y (default 5Y)
- VaR confidence 95% / 99%; horizon 1-day / 10-day
- Risk-free rate (default 4%)
- Benchmark (default SPY)

**Flow:** inputs → `Portfolio` (validated) → `data.get_prices(tickers + benchmark, start, end)` returns prices + `source` flag (`live`/`sample`) → daily returns → `risk/` modules → `charts.py` → tabs.

**Tabs:**
1. Overview — KPI tiles (CAGR, vol, Sharpe, Sortino, max DD), equity curve vs benchmark, drawdown chart
2. VaR / CVaR — 3-method comparison table + return histogram with VaR/CVaR lines
3. Diversification — correlation heatmap, risk-contribution bars next to weight bars
4. Benchmark — beta, alpha, tracking error, rolling 6-month beta chart

**Banner** when `source == sample`: "Live data unavailable — showing bundled sample through <date>."
**Caching:** `st.cache_data` on price fetch, 1h TTL, keyed on tickers + dates.

## Section 3: Methodology & error handling — APPROVED 2026-10-09 (after double-check corrections)

**Methodology:**
- Simple daily returns from adjusted close (adjusted = corrected for dividends/splits; simple returns aggregate across assets: r_p = Σ wᵢrᵢ). Vol annualized × √252.
- CAGR tile = geometric: (end/start)^(252/n) − 1. Sharpe uses arithmetic mean: mean(daily excess) × 252 / (σ_daily × √252). Both stated in spec so numbers don't look inconsistent.
- Sharpe excess return uses daily rf = annual / 252. Sortino: numerator = annualized mean excess return; denominator = downside deviation DD = √(mean(min(rₜ − rf_daily, 0)²)) over ALL days (not std of negative days only), annualized × √252.
- Max drawdown = worst peak-to-trough decline.
- Historical VaR = empirical quantile of past portfolio returns (current weights applied to history). No distribution assumption; assumes past repeats.
- Parametric VaR = normal distribution: VaR = −(μ + zσ), z = −1.645 (95%), −2.326 (99%). Fast but understates fat tails at high confidence.
- Monte Carlo VaR = 10,000 scenarios from **multivariate Student-t**, fixed seed. DECIDED: df fitted from data via `scipy.stats.t.fit` on portfolio daily returns (MLE), clipped to [3, 30]; fallback df = 5 if fit fails/non-finite. Fitted df shown in UI. Draw: X = μ + Z/√(W/df), Z ~ MVN(0, Σ·(df−2)/df), W ~ χ²(df) shared across assets → same covariance as data, fatter tails, joint crashes (tail dependence). Reason: normal MC reproduces parametric exactly for a linear portfolio (wᵀX normal with mean wᵀμ, var wᵀΣw), so it adds nothing.
  - Caveat: volatility clustering inflates apparent kurtosis → fitted df tends low (conservative). Fit noisy on 1Y windows; clipping bounds it.
- CVaR (Expected Shortfall) = mean of returns beyond VaR ("when a bad day happens, how bad on average"). Basel FRTB replaced 99% VaR with 97.5% ES (captures tail beyond cutoff; subadditive).
- KEY CORRECTION: vol-matched t has LOWER VaR than normal at 95% (1.561σ vs 1.645σ) but higher at 99% (2.606σ vs 2.326σ); crossover ≈ 97.3%. CVaR 95% is higher under t (2.239σ vs 2.063σ). UI caption under VaR table explains: at 95% VaR barely differs across methods, CVaR reveals the fat tail — that's why ES matters.
- 10-day horizon = 1-day × √10, with a UI note that it assumes i.i.d. returns (understates risk under volatility clustering).
- Risk contribution (Euler decomposition): RCᵢ = wᵢ(Σw)ᵢ / σₚ; contributions sum to σₚ, shown as % of total. Key insight: 60/40 stocks/bonds → ~90%+ of risk from stocks (checked: σ 16%/5%, ρ 0–0.2 → 94–96%).
- Beta/alpha: OLS of excess portfolio returns on excess benchmark returns (CAPM form): (Rₚ − R_f) = α + β(R_SPY − R_f) + ε. Alpha annualized. Tracking error = annualized std of (Rₚ − R_bench). Rolling 6-month beta.
- Assumption: constant weights, rebalanced daily — stated in UI.

**OPEN DECISION:** Monte Carlo with a multivariate normal converges to the parametric answer, so the comparison is redundant. Recommendation: draw the Monte Carlo from a **multivariate Student-t (df ≈ 5)** so the table shows that the normal model understates tail risk compared with historical and Student-t. User has not decided yet.

**Error handling:**
- Unknown/delisted ticker → warn naming it, continue with the rest; none valid → error + stop.
- Short history → clip window to shortest asset's history, inform user.
- Live fetch fails (rate limit / timeout / exception) → fallback to sample; ticker not in sample → "not in sample set, available: …".
- Weights: negatives rejected (long-only v1); zero sum → error; otherwise normalize.
- Fewer than 60 return days → block with message (too little data for stable stats).

## Next steps

1. ~~Resolve Monte Carlo decision, approve Section 3~~ DONE 2026-10-09 (Student-t, fitted df).
2. Present Section 4: testing strategy + deployment (pytest against known values, HF Spaces Docker setup, README).
3. Write the formal spec to `docs/superpowers/specs/2026-10-08-portfolio-risk-analyzer-design.md`, self-review, and get user review.
4. Invoke writing-plans for the implementation plan.
