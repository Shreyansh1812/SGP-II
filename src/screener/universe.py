"""
src/screener/universe.py
Universe definition containing 40 liquid US mega-cap stock tickers.
"""

from typing import List

# 40 liquid US mega-cap tickers across key market sectors
MEGA_CAP_UNIVERSE: List[str] = [
    # Technology / Communication
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AVGO", "ORCL", "CSCO",
    # Financial Services
    "BRK-B", "JPM", "V", "MA", "BAC", "WFC", "MS", "GS", "C", "BLK",
    # Healthcare / Pharma
    "UNH", "JNJ", "LLY", "ABBV", "MRK", "PFE", "TMO", "DHR", "AMGN", "ABT",
    # Consumer Staples & Discretionary
    "WMT", "PG", "COST", "HD", "KO", "PEP", "MCD", "DIS", "NKE", "NFLX"
]

def get_universe() -> List[str]:
    """Returns a copy of the mega-cap stock ticker universe."""
    return list(MEGA_CAP_UNIVERSE)
