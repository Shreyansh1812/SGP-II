"""
src/screener/fundamental_screener.py
Fundamental Screener engine that evaluates equities against health metrics
and persists healthy universe masks to the database.
"""

import time
import logging
from typing import Dict, List, Any, Optional, Tuple
import yfinance as yf

from src.database.dao import FundamentalsDAO
from src.screener.universe import MEGA_CAP_UNIVERSE, get_universe
from src.screener.news_scraper import CompanyNewsScraper
from src.sentiment.sentiment_engine import MarketNewsFetcher
from src.sentiment.llm_client import LLMSentimentClient

logger = logging.getLogger(__name__)

# Keywords signaling strong secular future demand (AI, chips, datacenter infrastructure)
FUTURE_DEMAND_KEYWORDS = {
    "ai",
    "artificial intelligence",
    "chip",
    "chips",
    "semiconductor",
    "semiconductors",
    "gpu",
    "gpus",
    "accelerator",
    "accelerators",
    "datacenter",
    "datacenters",
    "cloud",
    "cloud computing",
    "generative ai",
    "genai",
    "foundry",
    "fab",
    "hbm",
    "high-bandwidth memory",
    "neural",
    "deep learning",
    "server",
    "quantum",
}

# Core mega-cap tech, semiconductor, and AI infrastructure leaders
FUTURE_DEMAND_TICKERS = {
    "NVDA", "AVGO", "TSM", "AMD", "ASML", "QCOM", "ARM", "INTC", "MU",
    "AMAT", "LRCX", "KLAC", "MRVL", "SNPS", "CDNS", "ORCL", "MSFT",
    "GOOGL", "META", "AMZN", "AAPL"
}


class FundamentalScreener:
    """
    Evaluates US equities against fundamental health thresholds:
    1. Debt-to-Equity <= 2.0 (with high-demand AI/chip override if > 2.0)
    2. Operating Margin > 0.0 (strictly positive)
    3. Multi-source news synthesis: 5 broad-market benchmarks (SPY, QQQ, DIA, AAPL, MSFT)
       + individual stock headlines.
    """

    def __init__(
        self,
        news_scraper: Optional[CompanyNewsScraper] = None,
        llm_client: Optional[LLMSentimentClient] = None,
    ):
        self.news_scraper = news_scraper or CompanyNewsScraper()
        self.market_fetcher = MarketNewsFetcher()
        self.llm_client = llm_client or LLMSentimentClient()

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

    def evaluate_health_with_context(
        self,
        metrics: Dict[str, Any],
        ticker_news: Optional[List[str]] = None,
        macro_sentiment: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        Evaluates stock health by combining:
        1. 5 macro benchmarks (SPY, QQQ, DIA, AAPL, MSFT)
        2. Individual stock news and sentiment
        3. Static financial ratios (Debt/Equity <= 2.0, Operating Margin > 0.0)
        4. Concrete future demand override (AI, chips, datacenters) with cautionary remark

        Returns
        -------
        Tuple[bool, Optional[str]]
            (is_healthy: bool, cautionary_remark: Optional[str])
        """
        ticker = str(metrics.get("ticker", "")).upper()
        debt_to_equity = metrics.get("debt_to_equity")
        operating_margin = metrics.get("operating_margin")

        # Guard: Operating margin must be strictly positive (operational viability)
        if operating_margin is None or operating_margin <= 0.0:
            msg = f"Filtered: Operating margin is non-positive or missing ({operating_margin})."
            logger.info(f"{ticker}: {msg}")
            return False, msg

        # Analyze ticker-specific news for strong future demand signals
        import re
        extracted_headlines = []
        for item in (ticker_news or []):
            if isinstance(item, dict):
                extracted_headlines.append(str(item.get("title", "")))
            elif isinstance(item, str):
                extracted_headlines.append(item)
        news_text = " ".join(extracted_headlines)
        matched_catalysts = [
            kw for kw in FUTURE_DEMAND_KEYWORDS
            if re.search(rf"\b{re.escape(kw)}\b", news_text, re.IGNORECASE)
        ]
        has_future_demand = (
            ticker in FUTURE_DEMAND_TICKERS
            or len(matched_catalysts) > 0
        )
        catalyst_str = ", ".join(matched_catalysts[:3]) if matched_catalysts else "AI / Semiconductor ecosystem"

        # Case 1: Standard healthy (Debt-to-Equity <= 2.0)
        if debt_to_equity is not None and debt_to_equity <= 2.0:
            logger.debug(f"{ticker}: Standard healthy passed (D/E={debt_to_equity:.2f}, Margin={operating_margin:.2%})")
            return True, None

        # Case 2: High Debt-to-Equity (> 2.0 or restructured) BUT strong concrete future demand
        if has_future_demand:
            de_str = f"{debt_to_equity:.2f}" if debt_to_equity is not None else "N/A"
            remark = (
                f"CAUTION: High leverage (D/E={de_str} > 2.0). Qualified as healthy due to "
                f"strong concrete future secular demand in {catalyst_str}."
            )
            logger.info(f"{ticker}: {remark}")
            return True, remark

        # Case 3: High Debt-to-Equity without concrete future demand
        de_str = f"{debt_to_equity:.2f}" if debt_to_equity is not None else "N/A"
        msg = f"Filtered: Debt-to-Equity ({de_str} > 2.0) exceeds risk threshold with no concrete AI/chip future demand catalysts."
        logger.info(f"{ticker}: {msg}")
        return False, msg

    def evaluate_health(self, metrics: Dict[str, Any]) -> bool:
        """
        Backwards-compatible health evaluation method.
        Returns True if healthy, False if unhealthy or data missing.
        """
        is_healthy, _ = self.evaluate_health_with_context(metrics)
        return is_healthy

    def run_screener_batch(
        self,
        dao: FundamentalsDAO,
        universe: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Executes fundamental screening batch process over candidate universe.
        Synthesizes macro benchmark headlines + ticker-specific news, evaluates
        health rules with AI/chip future demand override, and persists to database.

        Returns summary statistics dictionary.
        """
        start_time = time.time()
        ticker_list = universe or get_universe()
        
        healthy_count = 0
        cautionary_count = 0
        filtered_count = 0
        total_count = len(ticker_list)

        logger.info(f"Starting fundamental screener batch for {total_count} tickers...")

        # Step 1: Pre-fetch 5 broad-market benchmarks (SPY, QQQ, DIA, AAPL, MSFT)
        logger.info("Pre-fetching 5 broad-market benchmark headlines (SPY, QQQ, DIA, AAPL, MSFT)...")
        macro_sentiment = None
        try:
            macro_news = self.market_fetcher.fetch_news(
                tickers=MarketNewsFetcher.DEFAULT_REFERENCE_TICKERS,
                max_items_per_ticker=3,
            )
            if macro_news and self.llm_client:
                macro_sentiment = self.llm_client.analyze(macro_news)
                logger.info(f"Macro Market Sentiment: {macro_sentiment.label} (score: {macro_sentiment.score:.2f})")
        except Exception as e:
            logger.warning(f"Could not fetch macro benchmark sentiment, continuing: {e}")

        # Step 2: Screen each ticker with combined macro + individual news context
        for idx, ticker in enumerate(ticker_list, start=1):
            logger.info(f"[{idx}/{total_count}] Screening {ticker}...")
            metrics = self.fetch_ticker_metrics(ticker)

            # Scrape individual ticker news headlines
            news_items = self.news_scraper.scrape_ticker_news(ticker, max_items=5)

            # Evaluate health combining macro + individual news + future demand override
            is_healthy, cautionary_remark = self.evaluate_health_with_context(
                metrics=metrics,
                ticker_news=news_items,
                macro_sentiment=macro_sentiment,
            )
            is_healthy_flag = 1 if is_healthy else 0

            if is_healthy:
                healthy_count += 1
                if cautionary_remark:
                    cautionary_count += 1
            else:
                filtered_count += 1

            # Persist to database via DAO (including cautionary_remark)
            dao.upsert_fundamental(
                ticker=ticker,
                pe_ratio=metrics.get("pe_ratio"),
                debt_to_equity=metrics.get("debt_to_equity"),
                operating_margin=metrics.get("operating_margin"),
                roe=metrics.get("roe"),
                is_healthy=is_healthy_flag,
                cautionary_remark=cautionary_remark,
            )

        elapsed = round(time.time() - start_time, 2)
        summary = {
            "total": total_count,
            "healthy": healthy_count,
            "cautionary": cautionary_count,
            "filtered": filtered_count,
            "elapsed_seconds": elapsed,
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
