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
- **Shareable portfolios:** "Save as link" puts your holdings in the page URL (`?p=SPY:60,AGG:40`). Bookmark or send the link to reopen the same portfolio; nothing is stored on the server.

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
portfolio_link.py  encode/decode holdings in shareable links
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
