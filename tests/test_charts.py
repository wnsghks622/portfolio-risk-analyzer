import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

import charts


def series(values, name=None):
    return pd.Series(values, index=pd.bdate_range("2024-01-01", periods=len(values)), name=name, dtype=float)


def test_equity_curve_has_portfolio_and_benchmark():
    fig = charts.equity_curve_fig(series([100, 101, 102]), series([100, 100.5, 101]), "SPY")
    assert isinstance(fig, go.Figure)
    assert [trace.name for trace in fig.data] == ["Portfolio", "SPY"]


def test_drawdown_is_plotted_in_percent():
    fig = charts.drawdown_fig(series([0.0, -0.1, -0.05]))
    assert list(fig.data[0].y) == pytest.approx([0.0, -10.0, -5.0])


def test_var_histogram_has_two_reference_lines():
    r = series(np.random.default_rng(0).normal(0, 0.01, 500))
    fig = charts.var_histogram_fig(r, 0.02, 0.03, 0.95)
    assert len(fig.data) == 1
    assert len(fig.layout.shapes) == 2
    assert fig.layout.shapes[0].x0 == pytest.approx(-2.0)
    assert fig.layout.shapes[1].x0 == pytest.approx(-3.0)


def test_correlation_heatmap_uses_fixed_scale():
    corr = pd.DataFrame([[1.0, 0.3], [0.3, 1.0]], index=["A", "B"], columns=["A", "B"])
    fig = charts.correlation_heatmap_fig(corr)
    assert fig.data[0].zmin == -1 and fig.data[0].zmax == 1
    assert list(fig.data[0].x) == ["A", "B"]


def test_risk_contribution_groups_weight_and_risk():
    rc = pd.DataFrame({"weight": [0.6, 0.4], "contribution": [0.1, 0.01], "pct_of_risk": [0.9, 0.1]}, index=["SPY", "AGG"])
    fig = charts.risk_contribution_fig(rc)
    assert fig.layout.barmode == "group"
    assert [trace.name for trace in fig.data] == ["Weight", "Share of risk"]
    assert list(fig.data[1].y) == pytest.approx([90.0, 10.0])


def test_rolling_beta_has_reference_line_at_one():
    fig = charts.rolling_beta_fig(series([0.9, 1.1, 1.0], name="Rolling beta"), "SPY")
    assert len(fig.data) == 1
    assert fig.layout.shapes[0].y0 == 1.0
