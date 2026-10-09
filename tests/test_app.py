from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import risk.data
from presets import PRESETS
from risk.data import load_sample

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def offline(*args, **kwargs):
    raise ConnectionError("no network in tests")


@pytest.fixture(autouse=True)
def fresh_cache():
    st.cache_data.clear()
    yield
    st.cache_data.clear()


def new_app():
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    return at


def test_every_preset_ticker_and_default_benchmark_is_in_sample():
    sample = set(load_sample().columns)
    for weights in PRESETS.values():
        assert set(weights) <= sample
    assert "SPY" in sample


@pytest.mark.parametrize("preset", list(PRESETS))
def test_app_renders_each_preset_from_sample_data(monkeypatch, preset):
    monkeypatch.setattr(risk.data, "_download_live", offline)
    at = new_app()
    at.selectbox(key="preset").set_value(preset).run()
    assert not at.exception
    assert not at.error
    assert list(at.session_state["holdings"]["Ticker"]) == list(PRESETS[preset])
    assert any("Live data unavailable" in w.value for w in at.warning)
    assert len(at.tabs) == 4


def test_99_percent_ten_day_path_shows_horizon_note(monkeypatch):
    monkeypatch.setattr(risk.data, "_download_live", offline)
    at = new_app()
    at.radio(key="conf").set_value("99%")
    at.radio(key="horizon").set_value("10-day").run()
    assert not at.exception
    assert any("square-root-of-time" in c.value for c in at.caption)


def test_short_history_is_blocked_with_a_message(monkeypatch):
    def recent_ipo(tickers, start, end):
        idx = pd.bdate_range(end=pd.Timestamp(end) - pd.Timedelta(days=1), periods=40)
        return pd.DataFrame({t: np.linspace(100, 110, 40) for t in tickers}, index=idx)

    monkeypatch.setattr(risk.data, "_download_live", recent_ipo)
    at = new_app()
    assert not at.exception
    assert any("days of returns" in e.value for e in at.error)


def test_all_holdings_missing_shows_error(monkeypatch):
    def only_qqq(tickers, start, end):
        idx = pd.bdate_range(end=pd.Timestamp(end) - pd.Timedelta(days=1), periods=300)
        frame = pd.DataFrame(np.nan, index=idx, columns=tickers, dtype=float)
        if "QQQ" in frame.columns:
            frame["QQQ"] = np.linspace(100, 130, 300)
        return frame

    monkeypatch.setattr(risk.data, "_download_live", only_qqq)
    at = new_app()
    at.text_input(key="benchmark").set_value("qqq").run()
    assert not at.exception
    assert any("None of your holdings have price data" in e.value for e in at.error)
