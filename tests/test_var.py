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
