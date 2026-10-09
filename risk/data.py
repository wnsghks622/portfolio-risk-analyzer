"""Price loading: live from Yahoo Finance, falling back to a bundled sample.

This is the only module in `risk/` that performs I/O.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal

import pandas as pd
import yfinance as yf

SAMPLE_PATH = Path(__file__).resolve().parent.parent / "data" / "sample_prices.parquet"
CLIP_TOLERANCE_DAYS = 5


@dataclass(frozen=True)
class PriceResult:
    prices: pd.DataFrame  # adjusted close, columns = valid tickers, aligned dates, no NaN
    source: Literal["live", "sample"]
    as_of: date | None  # last date in prices; None when there is no data
    missing: list[str]  # requested tickers with no data
    clipped_start: date | None  # set when the window was shortened to the shortest history
    limiting_ticker: str | None  # the ticker with the shortest history, when clipped


def _download_live(tickers: list[str], start: date, end: date) -> pd.DataFrame:
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False, timeout=10)
    if raw is None or raw.empty:
        return pd.DataFrame()
    close = raw["Close"]
    if isinstance(close, pd.Series):
        close = close.to_frame(name=tickers[0])
    return close


def load_sample(path: Path = SAMPLE_PATH) -> pd.DataFrame:
    return pd.read_parquet(path)


def available_sample_tickers(path: Path = SAMPLE_PATH) -> list[str]:
    return sorted(load_sample(path).columns)


def _sample_window(tickers: list[str], start: date, end: date, path: Path) -> pd.DataFrame:
    """Same window length as requested, shifted to end at the sample's last date if needed."""
    sample = load_sample(path)
    length = pd.Timestamp(end) - pd.Timestamp(start)
    end_ts = min(pd.Timestamp(end), sample.index.max())
    start_ts = end_ts - length
    present = [t for t in tickers if t in sample.columns]
    return sample.loc[start_ts:end_ts, present]


def _align(raw: pd.DataFrame, tickers: list[str], source: Literal["live", "sample"]) -> PriceResult:
    raw = raw.copy()
    raw.index = pd.DatetimeIndex(raw.index)
    if raw.index.tz is not None:
        raw.index = raw.index.tz_localize(None)
    raw = raw.sort_index()

    missing = [t for t in tickers if t not in raw.columns or raw[t].isna().all()]
    valid = [t for t in tickers if t not in missing]
    if not valid:
        return PriceResult(pd.DataFrame(), source, None, missing, None, None)

    first_valid = {t: raw[t].first_valid_index() for t in valid}
    limiting = max(first_valid, key=first_valid.get)
    prices = raw.loc[first_valid[limiting]:, valid].dropna(how="any")
    if prices.empty:
        return PriceResult(prices, source, None, missing, None, None)

    skipped = int((raw.index < prices.index[0]).sum())
    clipped = skipped > CLIP_TOLERANCE_DAYS
    return PriceResult(
        prices=prices,
        source=source,
        as_of=prices.index[-1].date(),
        missing=missing,
        clipped_start=prices.index[0].date() if clipped else None,
        limiting_ticker=limiting if clipped else None,
    )


def get_prices(
    tickers: list[str], start: date, end: date, sample_path: Path = SAMPLE_PATH
) -> PriceResult:
    wanted = list(dict.fromkeys(t.strip().upper() for t in tickers if t and t.strip()))
    source: Literal["live", "sample"] = "live"
    try:
        raw = _download_live(wanted, start, end)
    except Exception:
        raw = pd.DataFrame()
    if raw.empty or raw.isna().all().all():
        source = "sample"
        raw = _sample_window(wanted, start, end, sample_path)
    return _align(raw, wanted, source)


def to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily simple returns; the first row (no prior price) is dropped."""
    return (prices / prices.shift(1) - 1).iloc[1:]
