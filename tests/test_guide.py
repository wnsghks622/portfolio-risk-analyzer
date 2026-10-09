import math
from dataclasses import replace

import pytest

import guide

STATS = guide.GuideStats(
    cagr=0.083, growth=1.49, years=5.0, vol=0.111, sharpe=0.41, sortino=0.59, max_drawdown=-0.205,
    rf=0.04, conf=0.95, days=1,
    var=0.0107, cvar=0.0158, var_normal=0.0112, var_t=0.0107, df=4.4,
    avg_corr=0.22, n_assets=2, top_ticker="SPY", top_weight=0.6, top_risk=0.9,
    beta=0.63, alpha=-0.02, tracking_error=0.068, benchmark="SPY", cash_weight=0.0,
)


def text(stats=STATS):
    return "\n".join(body for _, body in guide.sections(stats)).replace("\$", "$")


def test_every_term_is_explained():
    body = text()
    for term in [
        "CAGR", "Volatility", "Sharpe ratio", "Sortino ratio", "Max drawdown", "Risk-free rate",
        "Value at Risk", "CVaR", "Confidence level and horizon", "Three ways to estimate",
        "Correlation", "Risk contribution", "Beta", "Alpha", "Tracking error", "Rolling beta",
    ]:
        assert term in body, term


def test_dollar_figures_use_a_10000_portfolio():
    body = text()
    assert "$107" in body  # 1.07% VaR
    assert "$158" in body  # 1.58% CVaR
    assert "$14,900" in body  # growth of 1.49x
    assert "$7,950" in body  # after a 20.5% drawdown


def test_var_frequency_matches_confidence_and_horizon():
    assert "1 trading day in 20" in text()
    assert "1 ten-day period in 100" in text(replace(STATS, conf=0.99, days=10))


def test_sharpe_verdicts():
    assert "0.30, which counts as weak" in text(replace(STATS, sharpe=0.3))
    assert "0.75, which counts as decent" in text(replace(STATS, sharpe=0.75))
    assert "1.40, which counts as strong" in text(replace(STATS, sharpe=1.4))
    assert "risk-free rate did better" in text(replace(STATS, sharpe=-0.2))


def test_undefined_values_are_explained_not_printed_as_nan():
    body = text(replace(STATS, sharpe=math.nan, sortino=math.nan, beta=math.nan, alpha=math.nan))
    assert "nan" not in body.lower().replace("financial", "")
    assert "can't be computed" in body


def test_single_holding_has_no_correlation_to_report():
    body = text(replace(STATS, avg_corr=math.nan, n_assets=1))
    assert "only one holding" in body


def test_cash_is_mentioned_when_held():
    assert "is held as cash" in text(replace(STATS, cash_weight=0.2))
    assert "is held as cash" not in text()


def test_tooltips_cover_every_kpi_tile():
    assert set(guide.TOOLTIPS) == {
        "CAGR", "Volatility", "Sharpe", "Sortino", "Max drawdown", "Beta", "Alpha (annualized)", "Tracking error",
    }
    assert all(len(tip) < 120 for tip in guide.TOOLTIPS.values())
