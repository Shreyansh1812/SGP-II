"""
tests/test_dao.py
Unit tests for SGP-II Database Connection and Data Access Objects (DAO).
Uses an in-memory SQLite database ('sqlite:///:memory:') for fast, deterministic execution.
"""

import pytest
from pathlib import Path

from src.database.connection import DatabaseManager, init_db, get_db_session
from src.database.dao import FundamentalsDAO, SentimentDAO, RecommendationsDAO

# Test connection string
TEST_DB_URL = "sqlite:///:memory:"
SCHEMA_PATH = Path(__file__).resolve().parent.parent / "database" / "schema.sql"


@pytest.fixture(autouse=True)
def setup_test_database():
    """Initializes schema in memory before each test and cleans up connection instance."""
    # Reset singleton
    DatabaseManager._instance = None
    init_db(db_url=TEST_DB_URL, schema_path=SCHEMA_PATH)
    yield
    if DatabaseManager._instance:
        DatabaseManager._instance.close()
        DatabaseManager._instance = None


class TestFundamentalsDAO:

    def test_upsert_and_get_healthy_tickers(self):
        dao = FundamentalsDAO(db_url=TEST_DB_URL)

        # Upsert healthy ticker
        dao.upsert_fundamental(
            ticker="AAPL",
            pe_ratio=28.5,
            debt_to_equity=1.2,
            operating_margin=0.30,
            roe=1.5,
            is_healthy=1
        )

        # Upsert another healthy ticker
        dao.upsert_fundamental(
            ticker="MSFT",
            pe_ratio=32.0,
            debt_to_equity=0.8,
            operating_margin=0.42,
            roe=0.35,
            is_healthy=1
        )

        # Upsert unhealthy ticker
        dao.upsert_fundamental(
            ticker="BADCO",
            pe_ratio=150.0,
            debt_to_equity=3.5,
            operating_margin=-0.05,
            roe=-0.2,
            is_healthy=0
        )

        healthy_tickers = dao.get_healthy_tickers()
        assert healthy_tickers == ["AAPL", "MSFT"]
        assert "BADCO" not in healthy_tickers

    def test_upsert_conflict_update(self):
        dao = FundamentalsDAO(db_url=TEST_DB_URL)

        # Initial insert as healthy
        dao.upsert_fundamental("TSLA", 60.0, 0.5, 0.12, 0.25, 1)
        record = dao.get_fundamental("TSLA")
        assert record is not None
        assert record["is_healthy"] == 1
        assert record["pe_ratio"] == 60.0

        # Update to unhealthy
        dao.upsert_fundamental("TSLA", 75.0, 2.5, -0.02, 0.05, 0)
        updated = dao.get_fundamental("TSLA")
        assert updated is not None
        assert updated["is_healthy"] == 0
        assert updated["pe_ratio"] == 75.0


class TestSentimentDAO:

    def test_upsert_and_get_sentiment(self):
        dao = SentimentDAO(db_url=TEST_DB_URL)

        date_str = "2026-07-28"
        dao.upsert_sentiment(
            date_str=date_str,
            score=0.65,
            label="bullish",
            summary="Strong earnings reports across tech sector."
        )

        sentiment = dao.get_sentiment(date_str)
        assert sentiment is not None
        assert sentiment["date"] == date_str
        assert sentiment["sentiment_score"] == 0.65
        assert sentiment["sentiment_label"] == "bullish"
        assert "tech sector" in sentiment["summary"]

    def test_sentiment_upsert_override(self):
        dao = SentimentDAO(db_url=TEST_DB_URL)
        date_str = "2026-07-28"

        dao.upsert_sentiment(date_str, 0.10, "neutral", "Initial report")
        dao.upsert_sentiment(date_str, 0.80, "bullish", "Updated guidance")

        sentiment = dao.get_sentiment(date_str)
        assert sentiment is not None
        assert sentiment["sentiment_score"] == 0.80
        assert sentiment["sentiment_label"] == "bullish"

    def test_nonexistent_sentiment(self):
        dao = SentimentDAO(db_url=TEST_DB_URL)
        assert dao.get_sentiment("1999-01-01") is None


class TestRecommendationsDAO:

    def test_insert_and_get_recommendation(self):
        dao = RecommendationsDAO(db_url=TEST_DB_URL)

        rec = {
            "date": "2026-07-28",
            "ticker": "NVDA",
            "signal_type": "BUY",
            "prob_buy": 0.82,
            "prob_sell": 0.08,
            "prob_hold": 0.10,
            "sentiment_score": 0.45,
            "is_sentiment_passed": 1,
            "execution_price": 125.50,
            "rationale": "High ML probability backed by positive sentiment."
        }

        dao.insert_recommendation(rec)

        latest = dao.get_latest_recommendations(limit=5)
        assert len(latest) == 1
        row = latest[0]
        assert row["ticker"] == "NVDA"
        assert row["signal_type"] == "BUY"
        assert row["prob_buy"] == 0.82
        assert row["is_sentiment_passed"] == 1

    def test_upsert_recommendation_conflict(self):
        dao = RecommendationsDAO(db_url=TEST_DB_URL)

        rec1 = {
            "date": "2026-07-28",
            "ticker": "AAPL",
            "signal_type": "HOLD",
            "prob_buy": 0.55,
            "prob_sell": 0.20,
            "prob_hold": 0.25,
            "sentiment_score": 0.15,
            "is_sentiment_passed": 0,
            "execution_price": 220.00,
            "rationale": "Neutral"
        }

        rec2 = {
            "date": "2026-07-28",
            "ticker": "AAPL",
            "signal_type": "BUY",
            "prob_buy": 0.78,
            "prob_sell": 0.10,
            "prob_hold": 0.12,
            "sentiment_score": 0.35,
            "is_sentiment_passed": 1,
            "execution_price": 221.50,
            "rationale": "Upgraded after late news."
        }

        dao.insert_recommendation(rec1)
        dao.insert_recommendation(rec2)

        latest = dao.get_latest_recommendations(limit=10)
        assert len(latest) == 1
        assert latest[0]["signal_type"] == "BUY"
        assert latest[0]["prob_buy"] == 0.78
        assert latest[0]["is_sentiment_passed"] == 1
