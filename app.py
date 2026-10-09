"""Streamlit UI for the Portfolio Risk Analyzer.

Wiring only: sidebar inputs -> risk/ -> charts.py. No risk math lives here.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import streamlit as st

import charts
from portfolio_link import decode_holdings, encode_holdings
from presets import CUSTOM, DEFAULT_PRESET, PRESETS
from risk import attribution, benchmark, metrics, var
from risk.data import PriceResult, available_sample_tickers, get_prices, to_returns
from risk.portfolio import CASH, Portfolio

RANGES = {"1Y": 1, "3Y": 3, "5Y": 5, "10Y": 10}
CONFIDENCE = {"95%": 0.95, "99%": 0.99}
HORIZON = {"1-day": 1, "10-day": 10}
MIN_RETURN_DAYS = 60
ROLLING_WINDOW = 126

VAR_CAPTION = (
    "At 95%, VaR barely differs between methods. CVaR averages the losses beyond the cutoff, "
    "so it shows the fat tail. This is why Basel's market-risk rules moved from VaR to Expected Shortfall."
)
HORIZON_NOTE = (
    "10-day figures use the square-root-of-time rule, which assumes independent daily returns "
    "and understates risk when volatility clusters."
)
FOOTER = "Assumes constant weights, rebalanced daily."
LINK_ERROR = "Couldn't read the saved portfolio in this link — showing the default preset."


@st.cache_data(ttl=3600, show_spinner="Loading prices…")
def load_prices(tickers: tuple[str, ...], start: date, end: date) -> PriceResult:
    return get_prices(list(tickers), start, end)


def holdings_frame(preset: str) -> pd.DataFrame:
    weights = PRESETS[preset]
    return pd.DataFrame({"Ticker": list(weights), "Weight %": list(weights.values())})


def init_state() -> None:
    if "preset" in st.session_state:
        return
    st.session_state.editor_version = 0
    st.session_state.link_error = False
    st.session_state.cash = st.query_params.get("cash") == "1"
    link = st.query_params.get("p")
    if link:
        try:
            pairs = decode_holdings(link)
            Portfolio.from_inputs(pairs)  # same validation as typed holdings
        except ValueError:
            st.session_state.link_error = True
        else:
            st.session_state.preset = CUSTOM
            st.session_state.holdings = pd.DataFrame(
                {"Ticker": [t for t, _ in pairs], "Weight %": [w for _, w in pairs]}
            )
            return
    st.session_state.preset = DEFAULT_PRESET
    st.session_state.holdings = holdings_frame(DEFAULT_PRESET)


def on_preset_change() -> None:
    name = st.session_state.preset
    st.session_state.link_error = False
    if name in PRESETS:
        st.session_state.holdings = holdings_frame(name)
        st.session_state.editor_version += 1  # new editor key discards old edits


def on_holdings_edit() -> None:
    st.session_state.preset = CUSTOM


def pct(x: float) -> str:
    return "n/a" if not np.isfinite(x) else f"{x:.1%}"


def ratio(x: float) -> str:
    return "n/a" if not np.isfinite(x) else f"{x:.2f}"


def sample_hint(result: PriceResult) -> str:
    if result.source != "sample":
        return ""
    return " Not in the bundled sample. Available: " + ", ".join(available_sample_tickers()) + "."


def sidebar() -> dict:
    sb = st.sidebar
    sb.header("Portfolio")
    sb.selectbox("Preset", [*PRESETS, CUSTOM], key="preset", on_change=on_preset_change)
    holdings = sb.data_editor(
        st.session_state.holdings,
        key=f"holdings_{st.session_state.editor_version}",
        num_rows="dynamic",
        hide_index=True,
        on_change=on_holdings_edit,
        column_config={
            "Ticker": st.column_config.TextColumn("Ticker"),
            "Weight %": st.column_config.NumberColumn("Weight %", format="%.2f"),
        },
    )
    cash = sb.checkbox(
        "Treat unallocated weight as cash",
        key="cash",
        help="Weights are read as % of the portfolio; anything below 100% is held as cash earning the risk-free rate.",
    )
    if sb.button("Save as link", key="save_link"):
        save_link(holdings, cash)
    sb.header("Settings")
    range_label = sb.radio("Date range", list(RANGES), index=2, horizontal=True, key="range")
    conf_label = sb.radio("VaR confidence", list(CONFIDENCE), horizontal=True, key="conf")
    horizon_label = sb.radio("VaR horizon", list(HORIZON), horizontal=True, key="horizon")
    rf_pct = sb.number_input("Risk-free rate (annual %)", value=4.0, step=0.25, key="rf")
    bench = sb.text_input("Benchmark", value="SPY", key="benchmark")
    return {
        "holdings": holdings,
        "cash": cash,
        "years": RANGES[range_label],
        "conf": CONFIDENCE[conf_label],
        "conf_label": conf_label,
        "days": HORIZON[horizon_label],
        "horizon_label": horizon_label,
        "rf": rf_pct / 100,
        "benchmark": bench.strip().upper(),
    }


def save_link(holdings: pd.DataFrame, cash: bool) -> None:
    try:
        portfolio = Portfolio.from_inputs(zip(holdings["Ticker"], holdings["Weight %"]), cash_remainder=cash)
    except ValueError as exc:
        st.sidebar.error(str(exc))
        return
    st.query_params["p"] = encode_holdings(portfolio)
    if cash:
        st.query_params["cash"] = "1"
    elif "cash" in st.query_params:
        del st.query_params["cash"]
    st.session_state.link_error = False
    st.sidebar.success("Saved. Copy the link from your browser's address bar to reopen or share this portfolio.")


def render_overview(r: pd.Series, bench_r: pd.Series, bench: str, rf: float) -> None:
    cols = st.columns(5)
    cols[0].metric("CAGR", pct(metrics.cagr(r)))
    cols[1].metric("Volatility", pct(metrics.annualized_vol(r)))
    cols[2].metric("Sharpe", ratio(metrics.sharpe(r, rf)))
    cols[3].metric("Sortino", ratio(metrics.sortino(r, rf)))
    cols[4].metric("Max drawdown", pct(metrics.max_drawdown(r)))
    st.plotly_chart(charts.equity_curve_fig(metrics.equity_curve(r), metrics.equity_curve(bench_r), bench))
    st.plotly_chart(charts.drawdown_fig(metrics.drawdown_series(r)))


def render_tail_risk(asset_returns: pd.DataFrame, weights: pd.Series, r: pd.Series, inputs: dict) -> None:
    table, nu = var.var_table(asset_returns, weights, inputs["conf"], inputs["days"])
    st.subheader(f"{inputs['horizon_label']} loss at {inputs['conf_label']} confidence")
    st.dataframe(table.map(lambda x: f"{x:.2%}"))
    st.caption(
        f"Monte Carlo: 10,000 scenarios from a multivariate Student-t with df = {nu:.1f}, "
        "fitted to this portfolio's daily returns."
    )
    st.caption(VAR_CAPTION)
    if inputs["days"] > 1:
        st.caption(HORIZON_NOTE)
    hist = var.historical_var(r, inputs["conf"])
    st.plotly_chart(charts.var_histogram_fig(r, hist.var, hist.cvar, inputs["conf"]))


def render_diversification(asset_returns: pd.DataFrame, weights: pd.Series) -> None:
    left, right = st.columns(2)
    with left:
        risky = asset_returns.drop(columns=[CASH], errors="ignore")  # cash never moves: correlation undefined
        st.plotly_chart(charts.correlation_heatmap_fig(attribution.correlation_matrix(risky)))
    with right:
        st.plotly_chart(charts.risk_contribution_fig(attribution.risk_contribution(asset_returns, weights)))


def render_benchmark(r: pd.Series, bench_r: pd.Series, bench: str, rf: float) -> None:
    fit = benchmark.capm(r, bench_r, rf)
    cols = st.columns(3)
    cols[0].metric("Beta", ratio(fit.beta))
    cols[1].metric("Alpha (annualized)", pct(fit.alpha))
    cols[2].metric("Tracking error", pct(benchmark.tracking_error(r, bench_r)))
    if len(r) < ROLLING_WINDOW:
        st.info(f"Rolling beta needs at least {ROLLING_WINDOW} days of returns.")
    else:
        st.plotly_chart(charts.rolling_beta_fig(benchmark.rolling_beta(r, bench_r, ROLLING_WINDOW), bench))


def main() -> None:
    st.set_page_config(page_title="Portfolio Risk Analyzer", layout="wide")
    init_state()
    inputs = sidebar()
    st.title("Portfolio Risk Analyzer")
    st.caption("Return, tail-risk, diversification and benchmark analytics for a long-only portfolio.")
    if st.session_state.get("link_error"):
        st.warning(LINK_ERROR)

    holdings = inputs["holdings"]
    try:
        portfolio = Portfolio.from_inputs(
            zip(holdings["Ticker"], holdings["Weight %"]), cash_remainder=inputs["cash"]
        )
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    if inputs["cash"] and portfolio.input_total > 100:
        st.info(f"Weights add to {portfolio.input_total:.4g}%: scaled to 100%, no cash.")
    bench = inputs["benchmark"]
    if not bench:
        st.error("Enter a benchmark ticker.")
        st.stop()

    end = date.today()
    start = (pd.Timestamp(end) - pd.DateOffset(years=inputs["years"])).date()
    tickers = tuple(t for t in portfolio.tickers if t != CASH)
    result = load_prices((*tickers, bench), start, end)

    if result.source == "sample" and result.as_of is not None:
        st.warning(f"Live data unavailable — showing bundled sample through {result.as_of:%Y-%m-%d}.")
    if bench in result.missing:
        st.error(f"No price data for benchmark {bench}." + sample_hint(result))
        st.stop()
    dropped = [t for t in portfolio.tickers if t in result.missing]
    if dropped:
        try:
            portfolio = portfolio.drop(dropped)
        except ValueError:
            st.error(f"None of your holdings have price data ({', '.join(dropped)})." + sample_hint(result))
            st.stop()
        st.warning(f"No price data for {', '.join(dropped)} — dropped; remaining weights renormalized." + sample_hint(result))
    if result.clipped_start is not None:
        st.info(
            f"Window starts {result.clipped_start:%Y-%m-%d} "
            f"(shortest available history: {result.limiting_ticker})."
        )

    returns = metrics.add_cash(to_returns(result.prices), inputs["rf"])
    if len(returns) < MIN_RETURN_DAYS:
        st.error(
            f"Only {len(returns)} days of returns available; at least {MIN_RETURN_DAYS} "
            "are needed for stable statistics."
        )
        st.stop()

    weights = portfolio.weight_series()
    asset_returns = returns[list(portfolio.tickers)]
    bench_r = returns[bench]
    r = metrics.portfolio_returns(asset_returns, weights)

    st.sidebar.caption("Normalized weights")
    st.sidebar.dataframe((weights * 100).round(2).rename("Weight %"))

    overview, tail, diversification, bench_tab = st.tabs(
        ["Overview", "VaR / CVaR", "Diversification", "Benchmark"]
    )
    with overview:
        render_overview(r, bench_r, bench, inputs["rf"])
    with tail:
        render_tail_risk(asset_returns, weights, r, inputs)
    with diversification:
        render_diversification(asset_returns, weights)
    with bench_tab:
        render_benchmark(r, bench_r, bench, inputs["rf"])
    st.caption(FOOTER)


main()
