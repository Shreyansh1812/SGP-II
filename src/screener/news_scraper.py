"""
src/screener/news_scraper.py
Web news scraper module to extract real-time corporate announcements,
project updates, product launches, and news headlines for tickers.

Implements a fallback chain:
  yfinance.Ticker.news → Google News RSS → Yahoo Finance RSS
"""

import time
import logging
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import yfinance as yf

from src.sentiment.news_sources import MultiSourceNewsFetcher

logger = logging.getLogger(__name__)


class CompanyNewsScraper:
    """
    Scrapes real-time corporate news items and project announcements
    for a given equity ticker using yfinance and RSS fallback mechanisms.
    Includes in-memory TTL caching and polite multi-threaded universe scraping.
    """

    _news_cache: Dict[str, Dict[str, Any]] = {}
    CACHE_TTL_SECONDS: int = 7200  # 2 hours

    def __init__(self):
        self._rss_fetcher = MultiSourceNewsFetcher()

    def scrape_ticker_news(self, ticker: str, max_items: int = 5) -> List[Dict[str, Any]]:
        """
        Fetches breaking news articles and project developments for a stock ticker.
        Uses in-memory cache to prevent redundant network calls.

        Fallback chain:
        1. yfinance.Ticker.news (primary, may be broken)
        2. Google News RSS + Yahoo Finance RSS (stdlib, no API key)

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
        ticker_clean = ticker.upper()
        cache_key = f"{ticker_clean}_{max_items}"
        now = time.time()

        if cache_key in self._news_cache:
            entry = self._news_cache[cache_key]
            if now - entry["timestamp"] < self.CACHE_TTL_SECONDS:
                logger.debug(f"Returning cached news for {ticker_clean}")
                return entry["items"]

        news_items: List[Dict[str, Any]] = []

        # ── Layer 1: Try yfinance.Ticker.news ──
        try:
            ticker_obj = yf.Ticker(ticker_clean)
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
                    "ticker": ticker_clean,
                    "title": str(title),
                    "publisher": str(publisher),
                    "link": str(link),
                    "publish_time": publish_time,
                })

            logger.debug(f"yfinance returned {len(news_items)} news items for {ticker_clean}")

        except Exception as e:
            logger.warning(f"yfinance news fetch error for {ticker_clean}: {e}")

        # ── Layer 2: Fallback to RSS if yfinance returned nothing ──
        if not news_items:
            logger.info(
                f"yfinance.Ticker.news returned empty for {ticker_clean}. "
                f"Falling back to RSS sources..."
            )
            try:
                rss_items = self._rss_fetcher.fetch_ticker_news(
                    ticker_clean, max_items=max_items
                )
                for item in rss_items:
                    news_items.append({
                        "ticker": ticker_clean,
                        "title": str(item.get("title", "No Title")),
                        "publisher": str(item.get("publisher", "News")),
                        "link": str(item.get("link", "")),
                        "publish_time": item.get("publish_time", 0),
                    })
                if rss_items:
                    logger.info(
                        f"RSS fallback returned {len(rss_items)} items for {ticker_clean}"
                    )
            except Exception as e:
                logger.warning(f"RSS fallback failed for {ticker_clean}: {e}")

        # Cache the result (even if empty, to prevent repeated failures)
        self._news_cache[cache_key] = {"items": news_items, "timestamp": now}
        return news_items

    def fetch_universe_news(
        self,
        tickers: List[str],
        max_workers: int = 5,
        max_items_per_ticker: int = 3
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Concurrently fetches news headlines across multiple tickers using a thread pool.
        Utilizes caching and polite concurrency to bypass Yahoo Finance rate limits.

        Parameters
        ----------
        tickers : List[str]
            List of stock tickers (e.g. MEGA_CAP_UNIVERSE)
        max_workers : int
            Concurrency limit (default 5 to avoid triggering IP rate limits)
        max_items_per_ticker : int
            Maximum news items per ticker (default 3)

        Returns
        -------
        Dict[str, List[Dict[str, Any]]]
            Mapping of uppercase ticker -> list of news dictionaries
        """
        results: Dict[str, List[Dict[str, Any]]] = {}
        if not tickers:
            return results

        logger.info(f"Fetching news for {len(tickers)} tickers with {max_workers} worker threads...")

        def _fetch(t: str) -> tuple:
            try:
                items = self.scrape_ticker_news(t, max_items=max_items_per_ticker)
                return t.upper(), items
            except Exception as e:
                logger.warning(f"Error in threaded news fetch for {t}: {e}")
                return t.upper(), []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_ticker = {executor.submit(_fetch, t): t for t in tickers}
            for future in as_completed(future_to_ticker):
                t_symbol, items = future.result()
                results[t_symbol] = items

        logger.info(f"Completed universe news fetch: {len(results)} tickers processed.")
        return results
