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
