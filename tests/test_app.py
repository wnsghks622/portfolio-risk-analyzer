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
    assert len(at.tabs) == 5


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
    def only_benchmark(tickers, start, end):
        idx = pd.bdate_range(end=pd.Timestamp(end) - pd.Timedelta(days=1), periods=300)
        frame = pd.DataFrame(np.nan, index=idx, columns=tickers, dtype=float)
        frame["SPY"] = np.linspace(100, 130, 300)
        return frame

    monkeypatch.setattr(risk.data, "_download_live", only_benchmark)
    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["preset"] = "Custom"
    at.session_state["holdings"] = pd.DataFrame({"Ticker": ["ZZZZ", "YYYY"], "Weight %": [50.0, 50.0]})
    at.session_state["editor_version"] = 0
    at.run()
    assert not at.exception
    assert any("None of your holdings have price data (ZZZZ, YYYY)" in e.value for e in at.error)


def test_partial_live_failure_still_renders_from_sample(monkeypatch):
    def spy_throttled(tickers, start, end):
        idx = pd.bdate_range(end=pd.Timestamp(end) - pd.Timedelta(days=1), periods=300)
        frame = pd.DataFrame(np.nan, index=idx, columns=tickers, dtype=float)
        frame["AGG"] = np.linspace(100, 105, 300)
        return frame

    monkeypatch.setattr(risk.data, "_download_live", spy_throttled)
    at = new_app()
    assert not at.exception
    assert not at.error
    assert any("Live data unavailable" in w.value for w in at.warning)
    assert len(at.tabs) == 5


def test_link_with_holdings_loads_them_as_custom(monkeypatch):
    monkeypatch.setattr(risk.data, "_download_live", offline)
    at = AppTest.from_file(APP, default_timeout=60)
    at.query_params["p"] = "QQQ:70,TLT:30"
    at.run()
    assert not at.exception
    assert not at.error
    assert at.selectbox(key="preset").value == "Custom"
    assert list(at.session_state["holdings"]["Ticker"]) == ["QQQ", "TLT"]
    assert len(at.tabs) == 5


def test_unreadable_link_warns_and_uses_default_preset(monkeypatch):
    monkeypatch.setattr(risk.data, "_download_live", offline)
    at = AppTest.from_file(APP, default_timeout=60)
    at.query_params["p"] = "SPY:lots"
    at.run()
    assert not at.exception
    assert at.selectbox(key="preset").value == "60/40 ETF"
    assert any("Couldn't read the saved portfolio in this link" in w.value for w in at.warning)


def test_save_button_writes_holdings_into_the_link(monkeypatch):
    monkeypatch.setattr(risk.data, "_download_live", offline)
    at = new_app()
    at.button(key="save_link").click().run()
    assert not at.exception
    assert at.query_params["p"] == "SPY:60,AGG:40"


def test_cash_link_adds_cash_holding(monkeypatch):
    monkeypatch.setattr(risk.data, "_download_live", offline)
    at = AppTest.from_file(APP, default_timeout=60)
    at.query_params["p"] = "SPY:60,AGG:20"
    at.query_params["cash"] = "1"
    at.run()
    assert not at.exception
    assert not at.error
    assert at.checkbox(key="cash").value is True
    assert any("Cash" in list(df.value.index) for df in at.dataframe)


def test_cash_option_over_100_explains_scaling(monkeypatch):
    monkeypatch.setattr(risk.data, "_download_live", offline)
    at = AppTest.from_file(APP, default_timeout=60)
    at.query_params["p"] = "SPY:90,AGG:30"
    at.query_params["cash"] = "1"
    at.run()
    assert not at.exception
    assert any("scaled to 100%, no cash" in i.value for i in at.info)


def test_save_link_keeps_cash_setting(monkeypatch):
    monkeypatch.setattr(risk.data, "_download_live", offline)
    at = AppTest.from_file(APP, default_timeout=60)
    at.query_params["p"] = "SPY:60,AGG:20"
    at.query_params["cash"] = "1"
    at.run()
    at.button(key="save_link").click().run()
    assert at.query_params["p"] == "SPY:60,AGG:20"
    assert at.query_params["cash"] == "1"


def test_guide_tab_explains_terms_with_live_numbers(monkeypatch):
    monkeypatch.setattr(risk.data, "_download_live", offline)
    at = new_app()
    assert not at.exception
    assert at.tabs[4].label == "Guide"
    guide_text = " ".join(" ".join(m.value for m in at.tabs[4].markdown).split()).replace("\\$", "$")
    assert "Value at Risk" in guide_text
    assert "on a $10,000 portfolio" in guide_text
