"""Encode portfolio holdings into a compact, URL-safe string for shareable links.

Format: "SPY:60,AGG:40" (ticker:weight percent, comma-separated).
"""

from __future__ import annotations

import math
import re

from risk.portfolio import Portfolio

MAX_HOLDINGS = 50
_TICKER = re.compile(r"^[A-Z0-9.^=-]{1,15}$")


def encode_holdings(portfolio: Portfolio) -> str:
    return ",".join(
        f"{ticker}:{round(weight * 100, 2):g}"
        for ticker, weight in zip(portfolio.tickers, portfolio.weights)
    )


def decode_holdings(text: str) -> list[tuple[str, float]]:
    """Parse a link string into (ticker, weight) pairs; raises ValueError if it is malformed."""
    entries = [entry for entry in text.split(",") if entry.strip()]
    if not entries:
        raise ValueError("The link has no holdings.")
    if len(entries) > MAX_HOLDINGS:
        raise ValueError(f"A link can hold at most {MAX_HOLDINGS} holdings.")
    pairs = []
    for entry in entries:
        raw_ticker, sep, raw_weight = entry.partition(":")
        ticker = raw_ticker.strip().upper()
        if not sep or not _TICKER.match(ticker):
            raise ValueError(f"Can't read holding {entry.strip()!r}.")
        try:
            weight = float(raw_weight)
        except ValueError:
            raise ValueError(f"Weight for {ticker} is not a number.") from None
        if not math.isfinite(weight):
            raise ValueError(f"Weight for {ticker} is not a number.")
        pairs.append((ticker, weight))
    return pairs
