"""Plain-language explanations of every risk term as scorecards, filled in with the current numbers.

Each card has a headline (the number, a verdict chip, one sentence) and longer details shown on demand.
Pure text building: stats in, text out. No Streamlit and no risk math here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

PORTFOLIO_VALUE = 10_000

TOOLTIPS = {
    "CAGR": "Average yearly growth rate, with compounding. See the Guide tab for details.",
    "Volatility": "How much returns swing over a year (annualized standard deviation). Higher = bumpier.",
    "Sharpe": "Return above the risk-free rate per unit of volatility. Above 1 is strong.",
    "Sortino": "Like Sharpe, but only downside swings count as risk.",
    "Max drawdown": "Worst fall from a peak to a later low during this period.",
    "Beta": "How much the portfolio tends to move when the benchmark moves 1%.",
    "Alpha (annualized)": "Yearly return beyond what the portfolio's beta to the benchmark explains.",
    "Tracking error": "How far the portfolio's returns stray from the benchmark's, per year.",
}

# CVaR / VaR for a bell curve at each confidence level; a clearly bigger ratio means a fat tail
_BELL_CURVE_TAIL_RATIO = {0.95: 1.25, 0.99: 1.15}


@dataclass(frozen=True)
class GuideStats:
    cagr: float
    growth: float  # ending value of 1 invested at the start
    years: float
    vol: float
    sharpe: float
    sortino: float
    max_drawdown: float  # negative fraction
    rf: float
    conf: float
    days: int
    var: float  # historical, at the chosen confidence and horizon
    cvar: float
    var_normal: float
    var_t: float
    df: float
    avg_corr: float
    n_assets: int
    top_ticker: str
    top_weight: float
    top_risk: float
    beta: float
    alpha: float
    tracking_error: float
    benchmark: str
    cash_weight: float


@dataclass(frozen=True)
class Card:
    term: str
    value: str
    verdict: str
    tone: str  # "good" | "ok" | "bad" | "info"
    summary: str  # one short sentence
    details: str  # Markdown shown under "Learn more"


def _ok(x: float) -> bool:
    return x is not None and math.isfinite(x)


def _pct(x: float, digits: int = 1) -> str:
    return f"{x:.{digits}%}" if _ok(x) else "n/a"


def _num(x: float) -> str:
    return f"{x:.2f}" if _ok(x) else "n/a"


def _money(fraction_of_portfolio: float) -> str:
    # "\$" keeps Streamlit's Markdown from reading dollar signs as LaTeX
    return f"\\${fraction_of_portfolio * PORTFOLIO_VALUE:,.0f}"


def _period(s: GuideStats) -> str:
    return {1: "trading day", 10: "ten-day period"}.get(s.days, f"{s.days}-day period")


def _returns(s: GuideStats) -> list[Card]:
    if s.cagr < s.rf:
        cagr_verdict = ("Below cash", "bad")
    elif s.cagr < s.rf + 0.03:
        cagr_verdict = ("Modest", "ok")
    else:
        cagr_verdict = ("Good", "good")
    grew = "grew" if s.growth >= 1 else "shrank"

    vol_verdict = ("Low", "good") if s.vol < 0.08 else ("Moderate", "ok") if s.vol < 0.16 else ("High", "bad")

    if not _ok(s.sharpe):
        sharpe = ("n/a", "info", "Can't be computed: the returns barely varied.")
    elif s.sharpe < 0:
        sharpe = ("Negative", "bad", "Cash at the risk-free rate did better.")
    elif s.sharpe < 0.5:
        sharpe = ("Weak", "bad", "Paid only a little for the risk it took.")
    elif s.sharpe < 1:
        sharpe = ("Decent", "ok", "Paid reasonably for the risk it took.")
    else:
        sharpe = ("Strong", "good", "Well paid for the risk it took.")

    if not (_ok(s.sortino) and _ok(s.sharpe)):
        sortino = ("n/a", "info", "Can't be computed: no days fell below the risk-free rate.")
    elif s.sortino > s.sharpe:
        sortino = ("Upside-heavy", "good", f"Above the Sharpe ratio ({s.sharpe:.2f}): more swings were gains.")
    else:
        sortino = ("Downside-heavy", "ok", f"Not above the Sharpe ratio ({s.sharpe:.2f}): losses drove the swings.")

    dd = s.max_drawdown
    dd_verdict = ("Mild", "good") if dd > -0.10 else ("Moderate", "ok") if dd > -0.25 else ("Severe", "bad")

    return [
        Card("CAGR", _pct(s.cagr), *cagr_verdict,
             f"{_money(1)} {grew} to about {_money(s.growth)} in {s.years:.1f} years.",
             "**What it is:** the steady yearly growth rate that turns the starting value into the ending "
             "value, smoothing out the ups and downs.\n\n"
             f"**How to read it:** higher is better. Below the risk-free rate ({_pct(s.rf, 2)}) means the "
             "portfolio took risk without being paid for it."),
        Card("Volatility", _pct(s.vol), *vol_verdict,
             f"Most years, the return lands within about ±{_pct(s.vol, 0)} of its average.",
             "**What it is:** how much the return swings around its average over a year (annualized "
             "standard deviation of daily returns), the most common single measure of risk.\n\n"
             "**How to read it:** roughly two years out of three fall within one volatility of the average. "
             "The US stock market usually runs about 15–20%; high-quality bonds about 4–7%."),
        Card("Sharpe ratio", _num(s.sharpe), sharpe[0], sharpe[1], sharpe[2],
             "**What it is:** return above the risk-free rate for each unit of volatility: "
             "(average return − risk-free rate) ÷ volatility. It answers \"was the risk worth it?\"\n\n"
             "**How to read it (rule of thumb):** below 0.5 is weak, 0.5–1 is decent, above 1 is strong.\n\n"
             "**Risk-free rate:** what you could earn with no risk, such as US Treasury bills "
             f"({_pct(s.rf, 2)} in the sidebar)."),
        Card("Sortino ratio", _num(s.sortino), sortino[0], sortino[1], sortino[2],
             "**What it is:** like Sharpe, but only days below the risk-free rate count as risk, since "
             "upside surprises don't hurt anyone.\n\n"
             "**How to read it:** higher is better. Sortino well above Sharpe means much of the volatility "
             "came from good days."),
        Card("Max drawdown", _pct(dd), *dd_verdict,
             f"At the worst point, {_money(1)} at the peak fell to {_money(1 + dd)}.",
             "**What it is:** the largest fall from a peak to a later low: the loss of someone who bought "
             "at the top and sold at the bottom. The drawdown chart shows this for every day.\n\n"
             "**How to read it:** the \"could I stomach this?\" number. For reference, the S&P 500 fell about "
             "34% in early 2020 and about 25% during 2022."),
    ]


def _tail_risk(s: GuideStats) -> list[Card]:
    period = _period(s)
    one_in = round(1 / (1 - s.conf))
    ratio = s.cvar / s.var if s.var > 0 else math.nan
    bell = _BELL_CURVE_TAIL_RATIO.get(s.conf, 1.2)
    fat_cvar = _ok(ratio) and ratio > bell * 1.08
    if s.df <= 6:
        tails = ("Fat tails", "ok")
    elif s.df <= 15:
        tails = ("Some fat tails", "ok")
    else:
        tails = ("Near bell curve", "good")

    return [
        Card("Value at Risk (VaR)", _pct(s.var, 2), f"{s.conf:.0%}, {s.days}-day", "info",
             f"About 1 {period} in {one_in}, {_money(1)} would lose more than {_money(s.var)}.",
             f"**What it is:** a loss threshold. At {s.conf:.0%} confidence it's the loss exceeded only "
             f"1 {period} in {one_in}: how bad a normal bad day gets, not the worst case. The headline uses "
             "the historical method (your actual past returns).\n\n"
             "**Confidence and horizon:** 95% vs 99% picks how rare the bad day is (1 in 20 vs 1 in 100). "
             "10-day figures are the 1-day figure × √10 ≈ 3.16, which assumes independent days; real "
             "markets bunch bad days together, so treat them as a floor."),
        Card("CVaR (Expected Shortfall)", _pct(s.cvar, 2),
             "Fat tail" if fat_cvar else "Bell-curve tail", "ok" if fat_cvar else "good",
             f"When those bad {period}s happen, the loss averages about {_money(s.cvar)}.",
             "**What it is:** the average loss beyond the VaR line. VaR says where the danger zone starts; "
             "CVaR says how deep it goes. Banking rules (Basel) moved from VaR to Expected Shortfall for "
             "this reason.\n\n"
             f"**How to read it:** CVaR ÷ VaR is {_num(ratio)} here. A bell curve gives about {bell:.2f}; "
             "a clearly bigger ratio means the worst days are worse than a bell curve suggests."),
        Card("Fat tails", f"df {s.df:.1f}", *tails,
             f"Bell-curve VaR {_pct(s.var_normal, 2)} vs fat-tailed model {_pct(s.var_t, 2)}.",
             "**Three ways to estimate VaR:** *historical* replays past returns; *parametric* assumes a "
             "bell curve (fast, but real markets have more extreme days); *Monte Carlo* simulates 10,000 "
             "days from a fat-tailed Student-t fitted to your data.\n\n"
             "**How to read it:** the fitted degrees of freedom (df) measure tail fatness: about 3–6 is "
             "typical for daily stock returns, 30 is close to a bell curve. Fat tails show most clearly at "
             "99% and in CVaR; at 95% the fat-tailed VaR can even be slightly lower."),
    ]


def _diversification(s: GuideStats) -> list[Card]:
    if s.n_assets < 2 or not _ok(s.avg_corr):
        corr = ("n/a", "n/a", "info", "There's only one holding, so there is nothing to compare.")
    elif s.avg_corr > 0.7:
        corr = (_num(s.avg_corr), "High", "bad", "Holdings tend to fall together, so they protect each other less.")
    elif s.avg_corr >= 0.3:
        corr = (_num(s.avg_corr), "Moderate", "ok", "Holdings partly move together.")
    else:
        corr = (_num(s.avg_corr), "Low", "good", "Holdings often move independently: good for diversification.")

    if _ok(s.top_risk):
        concentrated = s.top_risk - s.top_weight > 0.15
        rc = (f"{s.top_ticker} {_pct(s.top_risk, 0)}", "Concentrated" if concentrated else "Balanced",
              "ok" if concentrated else "good",
              f"{s.top_ticker} is {_pct(s.top_weight, 0)} of the money but {_pct(s.top_risk, 0)} of the risk.")
    else:
        rc = ("n/a", "n/a", "info", "Can't be computed: the returns barely varied.")

    cards = [
        Card("Correlation", corr[0], corr[1], corr[2], corr[3],
             "**What it is:** how closely two holdings move together, from −1 (opposite) through 0 "
             "(unrelated) to +1 (lockstep). The headline is the average over all pairs.\n\n"
             "**How to read it:** diversification only helps when holdings aren't highly correlated. In the "
             "heatmap, red means moving together and blue means moving apart."),
        Card("Risk contribution", rc[0], rc[1], rc[2], rc[3],
             "**What it is:** each holding's share of the portfolio's total volatility. It depends on the "
             "weight, how volatile the holding is, and how it moves with the rest, so it can differ a lot "
             "from the weight.\n\n"
             "**How to read it:** a holding whose risk share is far above its weight dominates how the "
             "portfolio behaves."),
    ]
    if s.cash_weight > 0:
        cards.append(Card("Cash", _pct(s.cash_weight, 0), "Zero risk", "good",
                          f"Earns the {_pct(s.rf, 2)} risk-free rate and never moves.",
                          "**What it is:** the unallocated part of the portfolio when \"Treat unallocated weight "
                          "as cash\" is on. It lowers volatility, drawdown and VaR, and pulls returns toward the "
                          "risk-free rate."))
    return cards


def _benchmark(s: GuideStats) -> list[Card]:
    b = s.benchmark
    if not _ok(s.beta):
        beta = ("n/a", "info", f"Can't be computed: {b} barely moved.")
    elif s.beta < 0.8:
        beta = (f"Calmer than {b}", "info", f"When {b} moves 1%, this typically moves {s.beta:.2f}%.")
    elif s.beta <= 1.2:
        beta = (f"Moves with {b}", "info", f"When {b} moves 1%, this typically moves {s.beta:.2f}%.")
    else:
        beta = (f"Amplifies {b}", "info", f"When {b} moves 1%, this typically moves {s.beta:.2f}%.")

    if not _ok(s.alpha):
        alpha = ("n/a", "info", "Can't be computed because beta can't.")
    elif abs(s.alpha) < 0.02:
        alpha = ("About zero", "info", f"Within ±2% a year of what its beta predicts: mostly noise.")
    elif s.alpha > 0:
        alpha = ("Positive", "good", f"Beat what its beta predicts by {_pct(s.alpha)} a year.")
    else:
        alpha = ("Negative", "bad", f"Lagged what its beta predicts by {_pct(-s.alpha)} a year.")

    te = s.tracking_error
    if not _ok(te):
        track = ("n/a", "info", "Can't be computed.")
    elif te < 0.03:
        track = ("Hugs benchmark", "info", f"Stays close to {b} year to year.")
    elif te < 0.08:
        track = ("Own path", "info", f"Moderately different from {b}.")
    else:
        track = ("Very different", "info", f"Follows a very different path from {b}.")

    return [
        Card("Beta", _num(s.beta), *beta,
             f"**What it is:** how much the portfolio tends to move when {b} moves 1%. 1 moves in step, "
             "0.6 is calmer, 1.3 amplifies.\n\n"
             "**Rolling beta:** the chart on the Benchmark tab recomputes beta over every trailing 6 months. "
             "A flat line means a stable relationship; big swings mean it changes with market conditions."),
        Card("Alpha", _pct(s.alpha), *alpha,
             "**What it is:** the yearly return left over after accounting for how much benchmark risk "
             "(beta) the portfolio took.\n\n"
             f"**How to read it:** positive beat what beta predicts, negative lagged. Over {s.years:.1f} "
             "years, a few percent either way is easily noise: a hint, not proof of skill."),
        Card("Tracking error", _pct(te), *track,
             f"**What it is:** how much the portfolio's return differs from {b}'s, per year.\n\n"
             "**How to read it:** about 1–3% closely follows the benchmark; much higher goes its own way."),
    ]


def cards(s: GuideStats) -> dict[str, list[Card]]:
    return {
        "Returns": _returns(s),
        "Tail risk": _tail_risk(s),
        "Diversification": _diversification(s),
        "Benchmark": _benchmark(s),
    }


def assumptions(s: GuideStats) -> str:
    cash = f" Right now {_pct(s.cash_weight, 0)} is held as cash." if s.cash_weight > 0 else ""
    return (
        "- **Weights** are rescaled to 100% keeping their ratios (60 and 20 become 75% and 25%), unless "
        f"\"Treat unallocated weight as cash\" is on, in which case the gap is cash.{cash}\n"
        "- **Rebalancing:** weights are assumed to be reset every day; a buy-and-hold portfolio would drift.\n"
        "- **History is not a forecast:** every number describes the chosen past window. Different windows "
        "give different answers, and this is not investment advice."
    )
