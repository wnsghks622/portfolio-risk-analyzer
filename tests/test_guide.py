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


def card(term, stats=STATS):
    for items in guide.cards(stats).values():
        for c in items:
            if c.term == term:
                return c
    raise KeyError(term)


def plain(s):
    return s.replace("\\$", "$")


def test_cards_are_grouped_into_four_tabs_with_every_term():
    groups = guide.cards(STATS)
    assert list(groups) == ["Returns", "Tail risk", "Diversification", "Benchmark"]
    terms = {c.term for items in groups.values() for c in items}
    assert terms == {
        "CAGR", "Volatility", "Sharpe ratio", "Sortino ratio", "Max drawdown",
        "Value at Risk (VaR)", "CVaR (Expected Shortfall)", "Fat tails",
        "Correlation", "Risk contribution", "Beta", "Alpha", "Tracking error",
    }


def test_summaries_are_one_short_sentence():
    for items in guide.cards(STATS).values():
        for c in items:
            assert len(plain(c.summary)) <= 110, c.term
            assert c.tone in {"good", "ok", "bad", "info"}
            assert c.details  # the long explanation lives behind "Learn more"


def test_dollar_figures_use_a_10000_portfolio():
    assert "$14,900" in plain(card("CAGR").summary)
    assert "$7,950" in plain(card("Max drawdown").summary)
    assert "$107" in plain(card("Value at Risk (VaR)").summary)
    assert "$158" in plain(card("CVaR (Expected Shortfall)").summary)


def test_var_frequency_matches_confidence_and_horizon():
    assert card("Value at Risk (VaR)").verdict == "95%, 1-day"
    assert "About 1 trading day in 20" in card("Value at Risk (VaR)").summary
    ten_day = card("Value at Risk (VaR)", replace(STATS, conf=0.99, days=10))
    assert ten_day.verdict == "99%, 10-day"
    assert "About 1 ten-day period in 100" in ten_day.summary


@pytest.mark.parametrize(
    ("sharpe", "verdict", "tone"),
    [(-0.2, "Negative", "bad"), (0.3, "Weak", "bad"), (0.75, "Decent", "ok"), (1.4, "Strong", "good")],
)
def test_sharpe_verdicts(sharpe, verdict, tone):
    c = card("Sharpe ratio", replace(STATS, sharpe=sharpe))
    assert (c.verdict, c.tone) == (verdict, tone)


def test_fat_tail_cvar_is_flagged():
    assert card("CVaR (Expected Shortfall)").verdict == "Fat tail"  # 1.58 / 1.07 = 1.48 > bell curve's 1.25
    thin = card("CVaR (Expected Shortfall)", replace(STATS, cvar=0.0132))  # ratio 1.23
    assert thin.verdict == "Bell-curve tail"


def test_concentrated_risk_is_flagged():
    assert card("Risk contribution").verdict == "Concentrated"  # 60% of money, 90% of risk
    assert card("Risk contribution", replace(STATS, top_risk=0.65)).verdict == "Balanced"


def test_undefined_values_show_na_and_explain():
    stats = replace(STATS, sharpe=math.nan, sortino=math.nan, beta=math.nan, alpha=math.nan)
    for term in ["Sharpe ratio", "Sortino ratio", "Beta", "Alpha"]:
        c = card(term, stats)
        assert c.value == "n/a"
        assert "can't be computed" in c.summary.lower()
        assert "nan" not in c.summary.lower()


def test_single_holding_has_no_correlation_to_report():
    c = card("Correlation", replace(STATS, avg_corr=math.nan, n_assets=1))
    assert c.value == "n/a"
    assert "only one holding" in c.summary


def test_cash_card_appears_only_when_cash_is_held():
    assert "Cash" not in {c.term for c in guide.cards(STATS)["Diversification"]}
    cash = card("Cash", replace(STATS, cash_weight=0.2))
    assert cash.value == "20%"


def test_assumptions_mention_rebalancing_and_not_a_forecast():
    text = guide.assumptions(STATS)
    assert "rebalanc" in text.lower()
    assert "not investment advice" in text


def test_tooltips_cover_every_kpi_tile():
    assert set(guide.TOOLTIPS) == {
        "CAGR", "Volatility", "Sharpe", "Sortino", "Max drawdown", "Beta", "Alpha (annualized)", "Tracking error",
    }
    assert all(len(tip) < 120 for tip in guide.TOOLTIPS.values())
