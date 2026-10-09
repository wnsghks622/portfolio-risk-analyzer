import pytest

from risk.portfolio import Portfolio


def test_weights_are_normalized_to_one():
    p = Portfolio.from_inputs([("SPY", 60), ("AGG", 40)])
    assert p.tickers == ("SPY", "AGG")
    assert p.weights == pytest.approx((0.6, 0.4))


def test_any_weight_scale_is_normalized():
    p = Portfolio.from_inputs([("A", 1), ("B", 3)])
    assert p.weights == pytest.approx((0.25, 0.75))


def test_tickers_are_stripped_and_uppercased():
    p = Portfolio.from_inputs([("  spy ", 1)])
    assert p.tickers == ("SPY",)


def test_duplicate_tickers_are_summed():
    p = Portfolio.from_inputs([("SPY", 30), ("AGG", 40), ("spy", 30)])
    assert p.tickers == ("SPY", "AGG")
    assert p.weights == pytest.approx((0.6, 0.4))


def test_blank_rows_from_the_editor_are_ignored():
    rows = [("SPY", 50), (None, None), ("", 10), (float("nan"), 5), ("AGG", 50)]
    p = Portfolio.from_inputs(rows)
    assert p.tickers == ("SPY", "AGG")
    assert p.weights == pytest.approx((0.5, 0.5))


def test_missing_or_zero_weight_drops_the_ticker():
    p = Portfolio.from_inputs([("SPY", 100), ("AGG", None), ("GLD", 0), ("TLT", float("nan"))])
    assert p.tickers == ("SPY",)
    assert p.weights == (1.0,)


def test_negative_weight_is_rejected():
    with pytest.raises(ValueError, match="Negative weight for AGG"):
        Portfolio.from_inputs([("SPY", 100), ("AGG", -10)])


def test_non_numeric_weight_is_rejected():
    with pytest.raises(ValueError, match="Weight for SPY is not a number"):
        Portfolio.from_inputs([("SPY", "lots")])


def test_zero_total_is_rejected():
    with pytest.raises(ValueError, match="positive weight"):
        Portfolio.from_inputs([("SPY", 0), ("AGG", 0)])


def test_no_holdings_is_rejected():
    with pytest.raises(ValueError, match="positive weight"):
        Portfolio.from_inputs([])


def test_drop_renormalizes_remaining():
    p = Portfolio.from_inputs([("A", 50), ("B", 25), ("C", 25)]).drop(["B"])
    assert p.tickers == ("A", "C")
    assert p.weights == pytest.approx((2 / 3, 1 / 3))


def test_dropping_every_holding_raises():
    with pytest.raises(ValueError, match="None of the holdings have price data"):
        Portfolio.from_inputs([("A", 1)]).drop(["A"])


def test_weight_series():
    s = Portfolio.from_inputs([("A", 1), ("B", 1)]).weight_series()
    assert list(s.index) == ["A", "B"]
    assert s.tolist() == pytest.approx([0.5, 0.5])
    assert s.name == "weight"
