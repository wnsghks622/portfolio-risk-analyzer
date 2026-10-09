import pytest

from risk.portfolio import Portfolio
from portfolio_link import MAX_HOLDINGS, decode_holdings, encode_holdings


def test_encode_uses_normalized_percent_weights():
    p = Portfolio.from_inputs([("SPY", 3), ("AGG", 2)])
    assert encode_holdings(p) == "SPY:60,AGG:40"


def test_encode_rounds_to_two_decimals():
    p = Portfolio.from_inputs([("A", 1), ("B", 1), ("C", 1)])
    assert encode_holdings(p) == "A:33.33,B:33.33,C:33.33"


def test_round_trip_keeps_tickers_and_weights():
    p = Portfolio.from_inputs([("BRK-B", 12.5), ("VTI", 87.5)])
    restored = Portfolio.from_inputs(decode_holdings(encode_holdings(p)))
    assert restored.tickers == ("BRK-B", "VTI")
    assert restored.weights == pytest.approx(p.weights)


def test_decode_uppercases_and_strips():
    assert decode_holdings(" spy : 60 , agg:40 ") == [("SPY", 60.0), ("AGG", 40.0)]


@pytest.mark.parametrize("text", ["", "SPY", "SPY:abc", ":50", "SPY:inf", "SPY:nan", "S P Y:10", "<b>:10"])
def test_decode_rejects_malformed_links(text):
    with pytest.raises(ValueError):
        decode_holdings(text)


def test_decode_rejects_too_many_holdings():
    text = ",".join(f"T{i}:1" for i in range(MAX_HOLDINGS + 1))
    with pytest.raises(ValueError, match="at most 50"):
        decode_holdings(text)
