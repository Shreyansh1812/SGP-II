"""
src/screener module
Fundamental screening and news web scraper components.
"""

from .universe import MEGA_CAP_UNIVERSE
from .news_scraper import CompanyNewsScraper
from .fundamental_screener import FundamentalScreener

__all__ = [
    "MEGA_CAP_UNIVERSE",
    "CompanyNewsScraper",
    "FundamentalScreener",
]
