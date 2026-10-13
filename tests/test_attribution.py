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


def test_cash_carries_no_risk():
    returns = correlated_returns()[["A", "B"]].copy()
    returns["Cash"] = 0.04 / 252
    rc = attribution.risk_contribution(returns, pd.Series({"A": 0.4, "B": 0.4, "Cash": 0.2}))
    assert rc.loc["Cash", "pct_of_risk"] == pytest.approx(0.0, abs=1e-9)
    assert rc["pct_of_risk"].sum() == pytest.approx(1.0)


def test_average_correlation_uses_off_diagonal_pairs():
    corr = pd.DataFrame([[1.0, 0.2, 0.4], [0.2, 1.0, 0.6], [0.4, 0.6, 1.0]], index=list("ABC"), columns=list("ABC"))
    assert attribution.average_correlation(corr) == pytest.approx(0.4)


def test_average_correlation_of_one_asset_is_nan():
    corr = pd.DataFrame([[1.0]], index=["A"], columns=["A"])
    assert np.isnan(attribution.average_correlation(corr))
