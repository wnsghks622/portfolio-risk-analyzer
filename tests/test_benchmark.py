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
