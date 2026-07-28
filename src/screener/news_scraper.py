"""
src/screener/news_scraper.py
Web news scraper module to extract real-time corporate announcements,
project updates, product launches, and news headlines for tickers.
"""

import logging
from typing import Dict, List, Any, Optional
import yfinance as yf

logger = logging.getLogger(__name__)


class CompanyNewsScraper:
    """
    Scrapes real-time corporate news items and project announcements
    for a given equity ticker using yfinance and RSS fallback mechanisms.
    """

    def scrape_ticker_news(self, ticker: str, max_items: int = 5) -> List[Dict[str, Any]]:
        """
        Fetches breaking news articles and project developments for a stock ticker.

        Parameters
        ----------
        ticker : str
            Stock ticker symbol (e.g. 'AAPL')
        max_items : int
            Maximum news items to return (default: 5)

        Returns
        -------
        List[Dict[str, Any]]
            List of news dictionaries with keys:
            ['title', 'publisher', 'link', 'publish_time']
        """
        ticker_obj = yf.Ticker(ticker.upper())
        news_items: List[Dict[str, Any]] = []

        try:
            raw_news = getattr(ticker_obj, "news", []) or []
            if not isinstance(raw_news, list):
                raw_news = []

            for item in raw_news[:max_items]:
                if not isinstance(item, dict):
                    continue

                # yfinance payload layout standard
                content = item.get("content", {}) if isinstance(item.get("content"), dict) else item
                title = content.get("title") or item.get("title") or "No Title"
                publisher = (
                    content.get("provider", {}).get("displayName")
                    if isinstance(content.get("provider"), dict)
                    else item.get("publisher", "Unknown")
                )
                link = (
                    content.get("canonicalUrl", {}).get("url")
                    if isinstance(content.get("canonicalUrl"), dict)
                    else item.get("link", "")
                )
                publish_time = content.get("pubDate") or item.get("providerPublishTime") or 0

                news_items.append({
                    "ticker": ticker.upper(),
                    "title": str(title),
                    "publisher": str(publisher),
                    "link": str(link),
                    "publish_time": publish_time,
                })

            logger.debug(f"Scraped {len(news_items)} news items for {ticker}")
            return news_items

        except Exception as e:
            logger.warning(f"Error scraping news for {ticker}: {e}")
            return []
