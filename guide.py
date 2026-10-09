"""Plain-language explanations of every risk term, filled in with the current portfolio's numbers.

Pure text building: stats in, Markdown out. No Streamlit and no risk math here.
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


def _ok(x: float) -> bool:
    return x is not None and math.isfinite(x)


def _pct(x: float, digits: int = 1) -> str:
    return f"{x:.{digits}%}" if _ok(x) else "n/a"


def _money(fraction_of_portfolio: float) -> str:
    # "\$" keeps Streamlit's Markdown from reading dollar signs as LaTeX
    return f"\\${fraction_of_portfolio * PORTFOLIO_VALUE:,.0f}"


def _sharpe_verdict(sharpe: float) -> str:
    if not _ok(sharpe):
        return "the Sharpe ratio can't be computed here (the portfolio's returns barely varied)."
    if sharpe < 0:
        return (f"a Sharpe ratio of {sharpe:.2f}, which means the risk-free rate did better "
                "than this portfolio over the period: the risk was not rewarded.")
    verdict = "weak" if sharpe < 0.5 else "decent" if sharpe < 1 else "strong"
    return f"a Sharpe ratio of {sharpe:.2f}, which counts as {verdict}."


def _return_and_risk(s: GuideStats) -> str:
    sortino_line = (
        f"a Sortino ratio of {s.sortino:.2f} versus a Sharpe ratio of {s.sharpe:.2f}."
        + (" Sortino is higher, so a good part of the swings were on the upside."
           if _ok(s.sharpe) and s.sortino > s.sharpe else "")
        if _ok(s.sortino) and _ok(s.sharpe)
        else "the Sortino ratio can't be computed here (there were no days below the risk-free rate, "
             "or returns barely varied)."
    )
    return f"""
**CAGR (compound annual growth rate)**: the steady yearly growth rate that would turn the starting value
into the ending value. It squeezes all the ups and downs into one number.

*How to read it:* higher is better. Compare it with the benchmark and with the risk-free rate; a CAGR
below the risk-free rate means the portfolio took risk without being paid for it.

*Your portfolio:* {_pct(s.cagr)} a year. {_money(1)} invested {s.years:.1f} years ago would now be worth
about {_money(s.growth)}.

**Volatility**: how much the portfolio's return swings around its average over a year (the annualized
standard deviation of daily returns). It is the most common single measure of risk.

*How to read it:* roughly, in about two years out of three the yearly return lands within one volatility
of its average. For reference, the US stock market usually runs about 15–20% and high-quality bonds
about 4–7%.

*Your portfolio:* {_pct(s.vol)}.

**Sharpe ratio**: the return earned above the risk-free rate for each unit of volatility,
(average return − risk-free rate) ÷ volatility. It answers "was the extra risk worth it?"

*How to read it (rule of thumb, long periods):* below 0.5 is weak, 0.5 to 1 is decent, above 1 is strong.

*Your portfolio:* {_sharpe_verdict(s.sharpe)}

**Sortino ratio**: like Sharpe, but only days below the risk-free rate count as risk, because
surprises on the upside don't hurt anyone.

*How to read it:* higher is better. If Sortino is clearly above Sharpe, much of the volatility came from
good days rather than bad ones.

*Your portfolio:* {sortino_line}

**Max drawdown**: the largest fall from a peak to a later low during the period, i.e. the loss of
someone who bought at the top and sold at the bottom. The drawdown chart shows, for every day, how far
the portfolio sat below its previous high.

*How to read it:* this is the "could I stomach this?" number. For reference, the S&P 500 fell about 34%
in early 2020 and about 25% during 2022.

*Your portfolio:* {_pct(s.max_drawdown)}. At the worst point, {_money(1)} at the peak would have fallen
to about {_money(1 + s.max_drawdown)}.

**Risk-free rate**: what you could earn with essentially no risk, such as short-term US Treasury bills.
Sharpe, Sortino and alpha measure performance above it, and cash in the portfolio earns it.

*Your setting:* {_pct(s.rf, 2)} a year.
"""


def _tail_risk(s: GuideStats) -> str:
    period = {1: "trading day", 10: "ten-day period"}.get(s.days, f"{s.days}-day period")
    one_in = round(1 / (1 - s.conf))
    horizon = f"{s.days}-day"
    return f"""
**Value at Risk (VaR)**: a loss threshold. At {s.conf:.0%} confidence over a {horizon} horizon, it is the
loss you would expect to exceed only 1 {period} in {one_in}. It describes how bad a "normal bad
{period}" gets, not the worst case.

*Your portfolio:* the {s.conf:.0%} {horizon} VaR is {_pct(s.var, 2)} (historical method): on a
{_money(1)} portfolio, about 1 {period} in {one_in} you would lose more than {_money(s.var)}.

**CVaR (Conditional VaR, also called Expected Shortfall)**: the average loss on those rare bad
{period}s beyond the VaR line. VaR tells you where the danger zone starts; CVaR tells you how deep it
goes. This is why Basel's market-risk rules for banks moved from VaR to Expected Shortfall.

*Your portfolio:* when a loss beyond VaR happens, it averages {_pct(s.cvar, 2)}, about {_money(s.cvar)}
on {_money(1)}.

**Confidence level and horizon**: 95% versus 99% picks how rare the bad day is (1 in 20 versus 1 in 100).
The 10-day horizon multiplies the 1-day figure by √10 ≈ 3.16, which assumes each day is independent.
Real markets tend to bunch bad days together, so treat 10-day figures as a floor rather than a ceiling.

**Three ways to estimate VaR and CVaR**:
- *Historical:* replays the portfolio's actual past daily returns. No assumptions, but it only knows the
  period it has seen.
- *Parametric (normal):* assumes returns follow a bell curve. Fast, but real markets have more extreme
  days than a bell curve predicts.
- *Monte Carlo (Student-t):* simulates 10,000 days from a "fat-tailed" distribution fitted to the data,
  which allows for more extreme days.

*How to read it:* if the historical and Student-t numbers sit well above the normal one, especially for
CVaR or at 99%, the portfolio has fat tails and the bell curve understates its risk. The fitted
"degrees of freedom" measures tail fatness: about 3–6 is typical for daily stock returns, and 30 is close
to a bell curve.

*Your portfolio:* normal-model VaR {_pct(s.var_normal, 2)}, Student-t VaR {_pct(s.var_t, 2)}, fitted
degrees of freedom {s.df:.1f}.
"""


def _diversification(s: GuideStats) -> str:
    if s.n_assets < 2 or not _ok(s.avg_corr):
        corr_line = "there is only one holding, so there is nothing to compare it with."
    else:
        level = (
            "high: the holdings tend to fall together, so they protect each other less"
            if s.avg_corr > 0.7
            else "moderate"
            if s.avg_corr >= 0.3
            else "low: the holdings often move independently, which is what diversification needs"
        )
        corr_line = f"the average correlation between your holdings is {s.avg_corr:.2f}, which is {level}."
    if _ok(s.top_risk):
        risk_line = (f"{s.top_ticker} is {_pct(s.top_weight, 0)} of the money but carries "
                     f"{_pct(s.top_risk, 0)} of the risk.")
    else:
        risk_line = "risk contributions can't be computed because the portfolio's returns barely varied."
    return f"""
**Correlation**: how closely two holdings move together, from −1 (always opposite) through 0 (unrelated)
to +1 (in lockstep). Diversification only helps when holdings are not highly correlated. In the heatmap,
red means two holdings move together and blue means they move in opposite directions.

*Your portfolio:* {corr_line}

**Risk contribution**: each holding's share of the portfolio's total volatility. It depends on the
holding's weight, how volatile it is, and how it moves with everything else, so it can differ a lot from
its weight. Cash contributes no risk.

*Your portfolio:* {risk_line}
"""


def _benchmark(s: GuideStats) -> str:
    b = s.benchmark
    beta_line = (
        f"{s.beta:.2f}. When {b} moved 1%, this portfolio typically moved about {s.beta:.2f}%."
        if _ok(s.beta) else f"beta can't be computed here because {b} barely moved."
    )
    alpha_line = (
        f"{_pct(s.alpha)} a year. Over a window of {s.years:.1f} years, an alpha of a few percent "
        "either way is easily noise, so read it as a hint rather than proof of skill."
        if _ok(s.alpha) else "alpha can't be computed here because beta can't."
    )
    if _ok(s.tracking_error):
        te_level = (
            "it hugs the benchmark" if s.tracking_error < 0.03
            else "it follows its own path to a moderate degree" if s.tracking_error < 0.08
            else "it follows a very different path from the benchmark"
        )
        te_line = f"{_pct(s.tracking_error)}: {te_level}."
    else:
        te_line = "n/a."
    return f"""
**Beta**: how much the portfolio tends to move when the benchmark ({b}) moves 1%. A beta of 1 moves in
step, 0.6 moves about 0.6% (calmer), and 1.3 amplifies the benchmark's moves.

*Your portfolio:* {beta_line}

**Alpha**: the yearly return left over after accounting for how much benchmark risk (beta) the
portfolio took. Positive means it did better than its beta alone would predict; negative means it lagged.

*Your portfolio:* {alpha_line}

**Tracking error**: how much the portfolio's return differs from the benchmark's, per year. About 1–3%
means it closely follows the benchmark; much higher means it goes its own way.

*Your portfolio:* {te_line}

**Rolling beta**: beta recomputed over every trailing 6-month window. A flat line means the relationship
with the benchmark is stable; big swings mean it changes with market conditions.
"""


def _assumptions(s: GuideStats) -> str:
    cash_line = (
        f"\n\n*Your portfolio:* {_pct(s.cash_weight, 0)} is held as cash, earning the "
        f"{_pct(s.rf, 2)} risk-free rate."
        if s.cash_weight > 0 else ""
    )
    return f"""
**Weights**: weights are rescaled to add up to 100% while keeping their ratios (60 and 20 become 75% and
25%). With "Treat unallocated weight as cash" switched on, they are read as percent of the portfolio
instead, and the unallocated part becomes cash earning the risk-free rate.{cash_line}

**Rebalancing**: the numbers assume the weights are reset every day. A buy-and-hold portfolio would
drift as winners grow.

**History is not a forecast**: every number here describes the chosen past window. Different windows give
different answers, and none of this is investment advice.
"""


def sections(s: GuideStats) -> list[tuple[str, str]]:
    return [
        ("Return and risk", _return_and_risk(s)),
        ("Tail risk: VaR and CVaR", _tail_risk(s)),
        ("Diversification", _diversification(s)),
        ("Benchmark", _benchmark(s)),
        ("Assumptions behind the numbers", _assumptions(s)),
    ]
