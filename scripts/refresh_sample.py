"""Regenerate data/sample_prices.parquet from Yahoo Finance.

Run locally (needs network):  python scripts/refresh_sample.py
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from risk.data import SAMPLE_PATH  # noqa: E402

SAMPLE_TICKERS = [
    "SPY", "AGG", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "VTI", "VXUS", "BND",
    "GLD", "VNQ", "TLT", "QQQ", "IWM", "EFA", "JPM", "XOM", "JNJ", "BRK-B",
]
YEARS = 10


def main() -> None:
    end = date.today()
    start = (pd.Timestamp(end) - pd.DateOffset(years=YEARS)).date()
    raw = yf.download(SAMPLE_TICKERS, start=start, end=end, auto_adjust=True, progress=False)
    close = raw["Close"]
    failed = [t for t in SAMPLE_TICKERS if t not in close.columns or close[t].isna().all()]
    if failed:
        raise SystemExit(f"Download failed for: {', '.join(failed)}. Try again later.")

    prices = close[SAMPLE_TICKERS].dropna(how="all").astype(float)
    prices.index = pd.DatetimeIndex(prices.index)
    if prices.index.tz is not None:
        prices.index = prices.index.tz_localize(None)
    prices.index.name = "Date"
    prices.columns.name = None

    SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(SAMPLE_PATH)
    size_kb = SAMPLE_PATH.stat().st_size / 1024
    print(
        f"Wrote {SAMPLE_PATH}: {prices.shape[1]} tickers, {len(prices)} days, "
        f"{prices.index[0]:%Y-%m-%d} to {prices.index[-1]:%Y-%m-%d}, {size_kb:.0f} KB"
    )


if __name__ == "__main__":
    main()
