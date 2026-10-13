from risk.data import SAMPLE_PATH, load_sample
from scripts.refresh_sample import SAMPLE_TICKERS


def test_sample_tickers_list():
    assert len(SAMPLE_TICKERS) == 21
    assert len(set(SAMPLE_TICKERS)) == 21
    assert "SPY" in SAMPLE_TICKERS


def test_sample_file_has_every_ticker_and_about_ten_years():
    sample = load_sample()
    assert set(SAMPLE_TICKERS) <= set(sample.columns)
    assert sample.index.is_monotonic_increasing
    assert (sample.index[-1] - sample.index[0]).days >= 3_600
    assert sample.isna().mean().max() < 0.01
    assert SAMPLE_PATH.stat().st_size < 1_000_000
