"""Plotly figure builders: data in, figures out. No risk math here."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

PORTFOLIO_COLOR = "#2563eb"
BENCHMARK_COLOR = "#9ca3af"
LOSS_COLOR = "#dc2626"
VAR_COLOR = "#f59e0b"


def _layout(fig: go.Figure, title: str, yaxis_title: str, **overrides) -> go.Figure:
    layout = dict(
        title=title,
        yaxis_title=yaxis_title,
        margin=dict(l=10, r=10, t=60, b=10),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    layout.update(overrides)
    fig.update_layout(**layout)
    return fig


def equity_curve_fig(portfolio: pd.Series, benchmark: pd.Series, benchmark_name: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=portfolio.index, y=portfolio, name="Portfolio",
                             line=dict(color=PORTFOLIO_COLOR, width=2)))
    fig.add_trace(go.Scatter(x=benchmark.index, y=benchmark, name=benchmark_name,
                             line=dict(color=BENCHMARK_COLOR, width=1.5)))
    return _layout(fig, "Growth of 100", "Value")


def drawdown_fig(drawdown: pd.Series) -> go.Figure:
    fig = go.Figure(go.Scatter(x=drawdown.index, y=drawdown * 100, name="Drawdown", fill="tozeroy",
                               line=dict(color=LOSS_COLOR, width=1)))
    return _layout(fig, "Drawdown from previous peak", "%", showlegend=False)


def var_histogram_fig(r: pd.Series, var_value: float, cvar_value: float, conf: float) -> go.Figure:
    fig = go.Figure(go.Histogram(x=r * 100, nbinsx=80, name="Daily returns",
                                 marker_color=PORTFOLIO_COLOR, opacity=0.75))
    fig.add_vline(x=-var_value * 100, line=dict(color=VAR_COLOR, dash="dash"),
                  annotation_text=f"VaR {conf:.0%}: -{var_value:.2%}", annotation_position="top left")
    fig.add_vline(x=-cvar_value * 100, line=dict(color=LOSS_COLOR, dash="dot"),
                  annotation_text=f"CVaR: -{cvar_value:.2%}", annotation_position="bottom left")
    return _layout(fig, "Distribution of daily returns (1-day, historical)", "Days",
                   xaxis_title="Daily return (%)", hovermode="closest", showlegend=False)


def correlation_heatmap_fig(corr: pd.DataFrame) -> go.Figure:
    labels = list(corr.columns)
    fig = go.Figure(go.Heatmap(
        z=corr.to_numpy(), x=labels, y=labels, zmin=-1, zmax=1, colorscale="RdBu_r",
        text=corr.round(2).to_numpy(), texttemplate="%{text}",
        hovertemplate="%{y} / %{x}: %{z:.2f}<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed")
    return _layout(fig, "Correlation of daily returns", "", hovermode="closest")


def risk_contribution_fig(rc: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=rc.index, y=rc["weight"] * 100, name="Weight", marker_color=BENCHMARK_COLOR))
    fig.add_trace(go.Bar(x=rc.index, y=rc["pct_of_risk"] * 100, name="Share of risk", marker_color=LOSS_COLOR))
    return _layout(fig, "Weight vs share of portfolio risk", "%", barmode="group")


def rolling_beta_fig(beta: pd.Series, benchmark_name: str) -> go.Figure:
    fig = go.Figure(go.Scatter(x=beta.index, y=beta, name="Rolling beta", line=dict(color=PORTFOLIO_COLOR)))
    fig.add_hline(y=1.0, line=dict(color=BENCHMARK_COLOR, dash="dash"),
                  annotation_text=f"β = 1 (moves with {benchmark_name})")
    return _layout(fig, f"Rolling 6-month beta vs {benchmark_name}", "Beta", showlegend=False)
