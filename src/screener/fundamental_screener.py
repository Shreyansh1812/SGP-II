"""
src/screener/fundamental_screener.py
Fundamental Screener engine that evaluates equities against health metrics
and persists healthy universe masks to the database.
"""

import time
import logging
from typing import Dict, List, Any, Optional
import yfinance as yf

from src.database.dao import FundamentalsDAO
from src.screener.universe import MEGA_CAP_UNIVERSE, get_universe
from src.screener.news_scraper import CompanyNewsScraper

logger = logging.getLogger(__name__)


class FundamentalScreener:
    """
    Evaluates US equities against fundamental health thresholds:
    1. Debt-to-Equity <= 2.0
    2. Operating Margin > 0.0 (strictly positive)
    """

    def __init__(self, news_scraper: Optional[CompanyNewsScraper] = None):
        self.news_scraper = news_scraper or CompanyNewsScraper()

    def fetch_ticker_metrics(self, ticker: str, max_retries: int = 3) -> Dict[str, Any]:
        """
        Fetches fundamental financial metrics for a ticker via yfinance.
        Implements exponential backoff retries on failure.

        Parameters
        ----------
        ticker : str
            Stock ticker symbol
        max_retries : int
            Maximum network retries (default: 3)

        Returns
        -------
        Dict[str, Any]
            Dictionary containing fundamental ratios:
            ['ticker', 'pe_ratio', 'debt_to_equity', 'operating_margin', 'roe']
        """
        ticker_str = ticker.upper()
        last_exception = None

        for attempt in range(1, max_retries + 1):
            try:
                t_obj = yf.Ticker(ticker_str)
                info = t_obj.info or {}

                # Safely extract ratios
                pe_ratio = self._safe_float(info.get("trailingPE"))
                debt_to_equity = self._safe_float(info.get("debtToEquity"))
                
                # yfinance debtToEquity is often expressed as a percentage (e.g., 145.0 for 1.45)
                # Normalize if debt_to_equity > 20.0
                if debt_to_equity is not None and debt_to_equity > 20.0:
                    debt_to_equity = debt_to_equity / 100.0

                operating_margin = self._safe_float(info.get("operatingMargins"))
                roe = self._safe_float(info.get("returnOnEquity"))

                return {
                    "ticker": ticker_str,
                    "pe_ratio": pe_ratio,
                    "debt_to_equity": debt_to_equity,
                    "operating_margin": operating_margin,
                    "roe": roe,
                }
            except Exception as e:
                last_exception = e
                logger.warning(f"Attempt {attempt}/{max_retries} failed fetching {ticker_str}: {e}")
                if attempt < max_retries:
                    time.sleep(2 ** (attempt - 1))  # Exponential backoff: 1s, 2s, 4s

        logger.error(f"All {max_retries} attempts failed for {ticker_str}: {last_exception}")
        return {
            "ticker": ticker_str,
            "pe_ratio": None,
            "debt_to_equity": None,
            "operating_margin": None,
            "roe": None,
        }

    def evaluate_health(self, metrics: Dict[str, Any]) -> bool:
        """
        Evaluates quantitative fundamental health rules:
        - Debt-to-Equity <= 2.0
        - Operating Margin > 0.0

        Returns True if healthy, False if unhealthy or data missing.
        """
        debt_to_equity = metrics.get("debt_to_equity")
        operating_margin = metrics.get("operating_margin")

        # Missing data fails health check safely
        if debt_to_equity is None or operating_margin is None:
            logger.info(f"Ticker {metrics.get('ticker')} missing required health metrics. Marked UNHEALTHY.")
            return False

        is_debt_ok = debt_to_equity <= 2.0
        is_margin_ok = operating_margin > 0.0

        is_healthy = is_debt_ok and is_margin_ok
        logger.debug(
            f"Health check for {metrics.get('ticker')}: "
            f"Debt/Equity={debt_to_equity} (≤2.0: {is_debt_ok}), "
            f"OpMargin={operating_margin} (>0.0: {is_margin_ok}) -> Healthy: {is_healthy}"
        )
        return is_healthy

    def run_screener_batch(
        self,
        dao: FundamentalsDAO,
        universe: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Executes fundamental screening batch process over candidate universe.
        Scrapes news, updates company_fundamentals database table via DAO.

        Returns summary statistics dictionary.
        """
        start_time = time.time()
        ticker_list = universe or get_universe()
        
        healthy_count = 0
        filtered_count = 0
        total_count = len(ticker_list)

        logger.info(f"Starting fundamental screener batch for {total_count} tickers...")

        for idx, ticker in enumerate(ticker_list, start=1):
            logger.info(f"[{idx}/{total_count}] Screening {ticker}...")
            metrics = self.fetch_ticker_metrics(ticker)
            is_healthy = self.evaluate_health(metrics)
            is_healthy_flag = 1 if is_healthy else 0

            if is_healthy:
                healthy_count += 1
            else:
                filtered_count += 1

            # Scrape recent news headlines for context
            news_items = self.news_scraper.scrape_ticker_news(ticker, max_items=3)

            # Persist to database via DAO
            dao.upsert_fundamental(
                ticker=ticker,
                pe_ratio=metrics.get("pe_ratio"),
                debt_to_equity=metrics.get("debt_to_equity"),
                operating_margin=metrics.get("operating_margin"),
                roe=metrics.get("roe"),
                is_healthy=is_healthy_flag
            )

        elapsed = round(time.time() - start_time, 2)
        summary = {
            "total": total_count,
            "healthy": healthy_count,
            "filtered": filtered_count,
            "elapsed_seconds": elapsed
        }
        logger.info(f"Batch screener complete in {elapsed}s. Summary: {summary}")
        return summary

    @staticmethod
    def _safe_float(val: Any) -> Optional[float]:
        """Safely parses float values handling strings or None."""
        if val is None:
            return None
        try:
            return float(val)
        except (ValueError, TypeError):
            return None
