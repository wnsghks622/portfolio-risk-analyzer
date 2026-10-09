"""Portfolio definition: tickers with normalized, long-only weights."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import pandas as pd


def _clean_ticker(raw: object) -> str:
    if raw is None or (not isinstance(raw, str) and pd.isna(raw)):
        return ""
    return str(raw).strip().upper()


def _clean_weight(ticker: str, raw: object) -> float:
    if raw is None or (not isinstance(raw, str) and pd.isna(raw)):
        return 0.0
    try:
        return float(raw)
    except (TypeError, ValueError):
        raise ValueError(f"Weight for {ticker} is not a number.") from None


@dataclass(frozen=True)
class Portfolio:
    tickers: tuple[str, ...]
    weights: tuple[float, ...]

    @classmethod
    def from_inputs(cls, holdings: Iterable[tuple[object, object]]) -> Portfolio:
        totals: dict[str, float] = {}
        for raw_ticker, raw_weight in holdings:
            ticker = _clean_ticker(raw_ticker)
            if not ticker:
                continue
            weight = _clean_weight(ticker, raw_weight)
            if weight < 0:
                raise ValueError(
                    f"Negative weight for {ticker}: short positions are not supported."
                )
            totals[ticker] = totals.get(ticker, 0.0) + weight
        return cls._normalized(totals, "Add at least one holding with a positive weight.")

    @classmethod
    def _normalized(cls, totals: dict[str, float], empty_message: str) -> Portfolio:
        positive = {t: w for t, w in totals.items() if w > 0}
        total = sum(positive.values())
        if not positive or total <= 0:
            raise ValueError(empty_message)
        return cls(tuple(positive), tuple(w / total for w in positive.values()))

    def drop(self, tickers: Iterable[str]) -> Portfolio:
        remove = set(tickers)
        kept = {t: w for t, w in zip(self.tickers, self.weights) if t not in remove}
        return self._normalized(kept, "None of the holdings have price data.")

    def weight_series(self) -> pd.Series:
        return pd.Series(self.weights, index=list(self.tickers), name="weight", dtype=float)
