"""
tests/test_screener.py
Unit tests for the SGP-II Fundamental Screener and News Scraper.
Mocks yfinance API calls to guarantee offline, deterministic testing.
"""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from src.database.connection import DatabaseManager, init_db
from src.database.dao import FundamentalsDAO
from src.screener.universe import MEGA_CAP_UNIVERSE
from src.screener.news_scraper import CompanyNewsScraper
from src.screener.fundamental_screener import FundamentalScreener

TEST_DB_URL = "sqlite:///:memory:"
SCHEMA_PATH = Path(__file__).resolve().parent.parent / "database" / "schema.sql"


@pytest.fixture(autouse=True)
def setup_test_db():
    """Initializes in-memory SQLite schema before each test."""
    DatabaseManager._instance = None
    init_db(db_url=TEST_DB_URL, schema_path=SCHEMA_PATH)
    yield
    if DatabaseManager._instance:
        DatabaseManager._instance.close()
        DatabaseManager._instance = None


class TestFundamentalScreenerRules:

    def setup_method(self):
        self.screener = FundamentalScreener()

    def test_evaluate_health_healthy_stock(self):
        metrics = {
            "ticker": "AAPL",
            "debt_to_equity": 1.2,
            "operating_margin": 0.25,
            "pe_ratio": 28.0,
            "roe": 1.4
        }
        assert self.screener.evaluate_health(metrics) is True

    def test_evaluate_health_high_debt_fails(self):
        metrics = {
            "ticker": "HIGHDEBT",
            "debt_to_equity": 2.8,  # > 2.0 limit
            "operating_margin": 0.15,
            "pe_ratio": 15.0,
            "roe": 0.1
        }
        assert self.screener.evaluate_health(metrics) is False

    def test_evaluate_health_negative_margin_fails(self):
        metrics = {
            "ticker": "UNPROFITABLE",
            "debt_to_equity": 0.8,
            "operating_margin": -0.04,  # <= 0.0 limit
            "pe_ratio": None,
            "roe": -0.1
        }
        assert self.screener.evaluate_health(metrics) is False

    def test_evaluate_health_missing_data_fails_safely(self):
        metrics = {
            "ticker": "CORRUPT",
            "debt_to_equity": None,
            "operating_margin": 0.20,
        }
        assert self.screener.evaluate_health(metrics) is False


class TestYFinanceMocking:

    @patch("yfinance.Ticker")
    def test_fetch_ticker_metrics_success(self, mock_ticker_class):
        mock_instance = MagicMock()
        mock_instance.info = {
            "trailingPE": 30.5,
            "debtToEquity": 145.0,  # Percentage format -> 1.45
            "operatingMargins": 0.35,
            "returnOnEquity": 0.85,
        }
        mock_ticker_class.return_value = mock_instance

        screener = FundamentalScreener()
        metrics = screener.fetch_ticker_metrics("MSFT")

        assert metrics["ticker"] == "MSFT"
        assert metrics["pe_ratio"] == 30.5
        assert metrics["debt_to_equity"] == 1.45
        assert metrics["operating_margin"] == 0.35

    @patch("yfinance.Ticker")
    def test_fetch_ticker_metrics_network_failure_fallback(self, mock_ticker_class):
        mock_ticker_class.side_effect = Exception("Network Timeout")

        screener = FundamentalScreener()
        metrics = screener.fetch_ticker_metrics("BADNET", max_retries=2)

        assert metrics["ticker"] == "BADNET"
        assert metrics["debt_to_equity"] is None
        assert metrics["operating_margin"] is None


class TestNewsScraper:

    @patch("yfinance.Ticker")
    def test_scrape_ticker_news(self, mock_ticker_class):
        mock_instance = MagicMock()
        mock_instance.news = [
            {
                "title": "Apple Announces Breakthrough AI Product Line",
                "publisher": "CNBC",
                "link": "https://cnbc.com/apple-ai",
                "providerPublishTime": 1722100000
            }
        ]
        mock_ticker_class.return_value = mock_instance

        scraper = CompanyNewsScraper()
        news = scraper.scrape_ticker_news("AAPL")

        assert len(news) == 1
        assert news[0]["ticker"] == "AAPL"
        assert news[0]["title"] == "Apple Announces Breakthrough AI Product Line"
        assert news[0]["publisher"] == "CNBC"


class TestScreenerBatchIntegration:

    @patch("src.screener.fundamental_screener.FundamentalScreener.fetch_ticker_metrics")
    @patch("src.screener.news_scraper.CompanyNewsScraper.scrape_ticker_news")
    def test_run_screener_batch(self, mock_scrape_news, mock_fetch_metrics):
        # Mock metrics for 2 tickers: 1 healthy, 1 unhealthy
        def mock_metrics_side_effect(ticker):
            if ticker == "AAPL":
                return {"ticker": "AAPL", "debt_to_equity": 1.2, "operating_margin": 0.30, "pe_ratio": 28.0, "roe": 1.5}
            else:
                return {"ticker": ticker, "debt_to_equity": 3.5, "operating_margin": -0.10, "pe_ratio": None, "roe": None}

        mock_fetch_metrics.side_effect = mock_metrics_side_effect
        mock_scrape_news.return_value = []

        dao = FundamentalsDAO(db_url=TEST_DB_URL)
        screener = FundamentalScreener()

        # Run screener on mini universe
        test_universe = ["AAPL", "BADCO"]
        summary = screener.run_screener_batch(dao, universe=test_universe)

        assert summary["total"] == 2
        assert summary["healthy"] == 1
        assert summary["filtered"] == 1

        # Verify DB contents
        healthy_tickers = dao.get_healthy_tickers()
        assert healthy_tickers == ["AAPL"]
