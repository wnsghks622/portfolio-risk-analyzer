"""Preset portfolios shown in the sidebar. Weights are in percent."""

BIG_TECH = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA"]

PRESETS: dict[str, dict[str, float]] = {
    "60/40 ETF": {"SPY": 60.0, "AGG": 40.0},
    "Big Tech": {ticker: 100 / len(BIG_TECH) for ticker in BIG_TECH},
    "Diversified": {"VTI": 30.0, "VXUS": 20.0, "BND": 25.0, "GLD": 10.0, "VNQ": 10.0, "TLT": 5.0},
}
CUSTOM = "Custom"
DEFAULT_PRESET = "60/40 ETF"
