from datetime import date

import numpy as np
import pandas as pd
import pytest

from risk import data


def make_prices(tickers, start="2015-01-01", end="2025-06-30", seed=0):
    idx = pd.bdate_range(start, end, name="Date")
    steps = np.random.default_rng(seed).normal(0.0003, 0.01, size=(len(idx), len(tickers)))
    return pd.DataFrame(100 * np.exp(np.cumsum(steps, axis=0)), index=idx, columns=tickers)


@pytest.fixture
def sample_path(tmp_path):
    path = tmp_path / "sample.parquet"
    make_prices(["SPY", "AGG", "AAPL"]).to_parquet(path)
    return path


def offline(*args, **kwargs):
    raise ConnectionError("no network in tests")


def test_live_failure_falls_back_to_sample(monkeypatch, sample_path):
    monkeypatch.setattr(data, "_download_live", offline)
    result = data.get_prices(["SPY", "AGG"], date(2021, 1, 1), date(2024, 1, 1), sample_path=sample_path)
    assert result.source == "sample"
    assert list(result.prices.columns) == ["SPY", "AGG"]
    assert result.missing == []


def test_empty_live_result_falls_back(monkeypatch, sample_path):
    monkeypatch.setattr(data, "_download_live", lambda tickers, start, end: pd.DataFrame())
    result = data.get_prices(["SPY"], date(2021, 1, 1), date(2024, 1, 1), sample_path=sample_path)
    assert result.source == "sample"


def test_all_nan_live_result_falls_back(monkeypatch, sample_path):
    idx = pd.bdate_range("2023-01-02", periods=50)
    nan_frame = pd.DataFrame(np.nan, index=idx, columns=["SPY", "AGG"])
    monkeypatch.setattr(data, "_download_live", lambda tickers, start, end: nan_frame)
    result = data.get_prices(["SPY", "AGG"], date(2021, 1, 1), date(2024, 1, 1), sample_path=sample_path)
    assert result.source == "sample"
    assert result.missing == []


def test_sample_window_keeps_length_but_ends_at_sample_end(monkeypatch, sample_path):
    monkeypatch.setattr(data, "_download_live", offline)
    result = data.get_prices(["SPY"], date(2021, 1, 1), date(2026, 1, 1), sample_path=sample_path)
    assert result.as_of == date(2025, 6, 30)
    first = result.prices.index[0]
    assert pd.Timestamp("2020-06-29") <= first <= pd.Timestamp("2020-07-08")
    assert result.clipped_start is None


def test_ticker_not_in_sample_is_missing(monkeypatch, sample_path):
    monkeypatch.setattr(data, "_download_live", offline)
    result = data.get_prices(["SPY", "ZZZZ"], date(2021, 1, 1), date(2024, 1, 1), sample_path=sample_path)
    assert result.missing == ["ZZZZ"]
    assert list(result.prices.columns) == ["SPY"]


def test_live_ticker_with_no_data_is_missing_but_stays_live(monkeypatch, sample_path):
    live = make_prices(["SPY", "AGG"], start="2023-01-02", end="2023-12-29")
    live["BAD"] = np.nan
    monkeypatch.setattr(data, "_download_live", lambda tickers, start, end: live)
    result = data.get_prices(["SPY", "BAD", "AGG"], date(2023, 1, 1), date(2024, 1, 1), sample_path=sample_path)
    assert result.source == "live"
    assert result.missing == ["BAD"]
    assert list(result.prices.columns) == ["SPY", "AGG"]


def test_tickers_are_deduplicated_and_uppercased(monkeypatch, sample_path):
    seen = {}

    def fake(tickers, start, end):
        seen["tickers"] = tickers
        return make_prices(tickers, start="2023-01-02", end="2023-12-29")

    monkeypatch.setattr(data, "_download_live", fake)
    result = data.get_prices(["spy", "AGG", " SPY "], date(2023, 1, 1), date(2024, 1, 1), sample_path=sample_path)
    assert seen["tickers"] == ["SPY", "AGG"]
    assert list(result.prices.columns) == ["SPY", "AGG"]


def test_window_clips_to_shortest_history(monkeypatch, sample_path):
    live = make_prices(["SPY", "NEW"], start="2023-01-02", end="2023-12-29")
    live.loc[:"2023-03-31", "NEW"] = np.nan
    monkeypatch.setattr(data, "_download_live", lambda tickers, start, end: live)
    result = data.get_prices(["SPY", "NEW"], date(2023, 1, 1), date(2024, 1, 1), sample_path=sample_path)
    assert result.prices.index[0] == pd.Timestamp("2023-04-03")
    assert result.clipped_start == date(2023, 4, 3)
    assert result.limiting_ticker == "NEW"
    assert not result.prices.isna().any().any()


def test_small_gap_is_not_reported_as_clipping(monkeypatch, sample_path):
    live = make_prices(["SPY", "NEW"], start="2023-01-02", end="2023-12-29")
    live.iloc[:2, 1] = np.nan
    monkeypatch.setattr(data, "_download_live", lambda tickers, start, end: live)
    result = data.get_prices(["SPY", "NEW"], date(2023, 1, 1), date(2024, 1, 1), sample_path=sample_path)
    assert result.clipped_start is None
    assert result.limiting_ticker is None
    assert result.prices.index[0] == live.index[2]


def test_timezone_aware_index_is_made_naive(monkeypatch, sample_path):
    live = make_prices(["SPY"], start="2023-01-02", end="2023-12-29")
    live.index = live.index.tz_localize("America/New_York")
    monkeypatch.setattr(data, "_download_live", lambda tickers, start, end: live)
    result = data.get_prices(["SPY"], date(2023, 1, 1), date(2024, 1, 1), sample_path=sample_path)
    assert result.prices.index.tz is None
    assert result.as_of == date(2023, 12, 29)


def test_no_valid_tickers_gives_empty_result(monkeypatch, sample_path):
    monkeypatch.setattr(data, "_download_live", offline)
    result = data.get_prices(["ZZZZ"], date(2021, 1, 1), date(2024, 1, 1), sample_path=sample_path)
    assert result.prices.empty
    assert result.missing == ["ZZZZ"]
    assert result.as_of is None


def test_to_returns():
    prices = pd.DataFrame({"A": [100.0, 110.0, 99.0]}, index=pd.bdate_range("2024-01-01", periods=3))
    returns = data.to_returns(prices)
    assert returns["A"].tolist() == pytest.approx([0.10, -0.10])
    assert returns.index[0] == prices.index[1]


def test_available_sample_tickers(sample_path):
    assert data.available_sample_tickers(sample_path) == ["AAPL", "AGG", "SPY"]
