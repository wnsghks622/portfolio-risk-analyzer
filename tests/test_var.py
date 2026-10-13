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


def test_monte_carlo_handles_zero_variance_cash_column():
    returns = two_asset_returns()
    returns["Cash"] = 0.04 / 252
    w = pd.Series({"A": 0.4, "B": 0.4, "Cash": 0.2})
    table, _ = var.var_table(returns, w, 0.95, 1)
    assert table.notna().all().all()
    no_cash, _ = var.var_table(returns[["A", "B"]], pd.Series({"A": 0.5, "B": 0.5}), 0.95, 1)
    assert table.loc["Parametric (normal)", "VaR"] < no_cash.loc["Parametric (normal)", "VaR"]
