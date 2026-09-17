"""
src/database/dao.py
Data Access Objects (DAO) providing CRUD operations for company fundamentals,
daily sentiment, and daily trade recommendations.
"""

import logging
from typing import Dict, List, Optional, Any
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.database.connection import get_db_session

logger = logging.getLogger(__name__)


class BaseDAO:
    """Base class for Data Access Objects allowing optional custom session injection."""
    def __init__(self, db_url: Optional[str] = None):
        self.db_url = db_url


class FundamentalsDAO(BaseDAO):
    """Data Access Object for the `company_fundamentals` table."""

    def upsert_fundamental(
        self,
        ticker: str,
        pe_ratio: Optional[float],
        debt_to_equity: Optional[float],
        operating_margin: Optional[float],
        roe: Optional[float],
        is_healthy: int,
        cautionary_remark: Optional[str] = None,
        session: Optional[Session] = None
    ) -> None:
        """
        Inserts or updates a fundamental metrics row for a given stock ticker.

        Parameters
        ----------
        ticker : str
            Stock ticker symbol (e.g. 'AAPL')
        pe_ratio : Optional[float]
            Price-to-Earnings ratio
        debt_to_equity : Optional[float]
            Debt-to-Equity ratio
        operating_margin : Optional[float]
            Operating Margin
        roe : Optional[float]
            Return on Equity
        is_healthy : int
            Health status mask (1 for Healthy, 0 for Unhealthy)
        cautionary_remark : Optional[str]
            Optional cautionary note when high debt is qualified by strong future demand.
        session : Optional[Session]
            Optional SQLAlchemy Session. If omitted, uses default session context.
        """
        query = text("""
            INSERT INTO company_fundamentals (
                ticker, pe_ratio, debt_to_equity, operating_margin, roe, is_healthy,
                cautionary_remark, last_updated
            ) VALUES (
                :ticker, :pe_ratio, :debt_to_equity, :operating_margin, :roe, :is_healthy,
                :cautionary_remark, CURRENT_TIMESTAMP
            ) ON CONFLICT(ticker) DO UPDATE SET
                pe_ratio = excluded.pe_ratio,
                debt_to_equity = excluded.debt_to_equity,
                operating_margin = excluded.operating_margin,
                roe = excluded.roe,
                is_healthy = excluded.is_healthy,
                cautionary_remark = excluded.cautionary_remark,
                last_updated = CURRENT_TIMESTAMP;
        """)
        params = {
            "ticker": ticker.upper(),
            "pe_ratio": pe_ratio,
            "debt_to_equity": debt_to_equity,
            "operating_margin": operating_margin,
            "roe": roe,
            "is_healthy": int(is_healthy),
            "cautionary_remark": cautionary_remark,
        }

        def _execute(s: Session):
            s.execute(query, params)
            logger.debug(f"Upserted fundamental data for {ticker}")

        if session:
            _execute(session)
        else:
            with get_db_session(self.db_url) as s:
                _execute(s)

    def get_healthy_tickers(self, session: Optional[Session] = None) -> List[str]:
        """
        Retrieves all stock tickers marked as healthy (is_healthy = 1).

        Returns
        -------
        List[str]
            List of healthy ticker symbols sorted alphabetically.
        """
        query = text("""
            SELECT ticker FROM company_fundamentals
            WHERE is_healthy = 1
            ORDER BY ticker ASC;
        """)

        def _execute(s: Session) -> List[str]:
            result = s.execute(query)
            return [row[0] for row in result.fetchall()]

        if session:
            return _execute(session)
        else:
            with get_db_session(self.db_url) as s:
                return _execute(s)

    def get_fundamental(self, ticker: str, session: Optional[Session] = None) -> Optional[Dict[str, Any]]:
        """Retrieves single fundamental record by ticker."""
        query = text("""
            SELECT ticker, pe_ratio, debt_to_equity, operating_margin, roe, is_healthy,
                   cautionary_remark, last_updated
            FROM company_fundamentals WHERE ticker = :ticker;
        """)

        def _execute(s: Session) -> Optional[Dict[str, Any]]:
            row = s.execute(query, {"ticker": ticker.upper()}).fetchone()
            if row:
                return dict(row._mapping)
            return None

        if session:
            return _execute(session)
        else:
            with get_db_session(self.db_url) as s:
                return _execute(s)


class SentimentDAO(BaseDAO):
    """Data Access Object for the `daily_sentiment` table."""

    def upsert_sentiment(
        self,
        date_str: str,
        score: float,
        label: str,
        summary: str,
        session: Optional[Session] = None
    ) -> None:
        """
        Inserts or updates daily market news sentiment.

        Parameters
        ----------
        date_str : str
            Date formatted as YYYY-MM-DD
        score : float
            Sentiment score between -1.0 and 1.0
        label : str
            Categorical label ('bullish', 'bearish', 'neutral')
        summary : str
            Brief rationale summary generated by LLM
        session : Optional[Session]
            Optional SQLAlchemy Session
        """
        if not (-1.0 <= score <= 1.0):
            logger.warning(f"Sentiment score {score} out of recommended bounds [-1.0, 1.0]")

        query = text("""
            INSERT INTO daily_sentiment (
                date, sentiment_score, sentiment_label, summary, created_at
            ) VALUES (
                :date, :sentiment_score, :sentiment_label, :summary, CURRENT_TIMESTAMP
            ) ON CONFLICT(date) DO UPDATE SET
                sentiment_score = excluded.sentiment_score,
                sentiment_label = excluded.sentiment_label,
                summary = excluded.summary;
        """)
        params = {
            "date": date_str,
            "sentiment_score": float(score),
            "sentiment_label": label,
            "summary": summary,
        }

        def _execute(s: Session):
            s.execute(query, params)
            logger.debug(f"Upserted sentiment for date {date_str}")

        if session:
            _execute(session)
        else:
            with get_db_session(self.db_url) as s:
                _execute(s)

    def get_sentiment(self, date_str: str, session: Optional[Session] = None) -> Optional[Dict[str, Any]]:
        """
        Retrieves daily sentiment record by date.

        Parameters
        ----------
        date_str : str
            Date string YYYY-MM-DD

        Returns
        -------
        Optional[Dict[str, Any]]
            Dictionary containing sentiment attributes or None if not found.
        """
        query = text("""
            SELECT date, sentiment_score, sentiment_label, summary, created_at
            FROM daily_sentiment WHERE date = :date;
        """)

        def _execute(s: Session) -> Optional[Dict[str, Any]]:
            row = s.execute(query, {"date": date_str}).fetchone()
            if row:
                return dict(row._mapping)
            return None

        if session:
            return _execute(session)
        else:
            with get_db_session(self.db_url) as s:
                return _execute(s)


class RecommendationsDAO(BaseDAO):
    """Data Access Object for the `daily_recommendations` table."""

    def insert_recommendation(
        self,
        record: Dict[str, Any],
        session: Optional[Session] = None
    ) -> None:
        """
        Inserts or updates a daily model trade recommendation record.

        Parameters
        ----------
        record : Dict[str, Any]
            Dictionary matching daily_recommendations table schema.
        session : Optional[Session]
            Optional SQLAlchemy Session
        """
        query = text("""
            INSERT INTO daily_recommendations (
                date, ticker, signal_type, prob_buy, prob_sell, prob_hold,
                sentiment_score, is_sentiment_passed, execution_price, rationale
            ) VALUES (
                :date, :ticker, :signal_type, :prob_buy, :prob_sell, :prob_hold,
                :sentiment_score, :is_sentiment_passed, :execution_price, :rationale
            ) ON CONFLICT(date, ticker) DO UPDATE SET
                signal_type = excluded.signal_type,
                prob_buy = excluded.prob_buy,
                prob_sell = excluded.prob_sell,
                prob_hold = excluded.prob_hold,
                sentiment_score = excluded.sentiment_score,
                is_sentiment_passed = excluded.is_sentiment_passed,
                execution_price = excluded.execution_price,
                rationale = excluded.rationale;
        """)
        params = {
            "date": record["date"],
            "ticker": record["ticker"].upper(),
            "signal_type": record.get("signal_type", "HOLD"),
            "prob_buy": record.get("prob_buy"),
            "prob_sell": record.get("prob_sell"),
            "prob_hold": record.get("prob_hold"),
            "sentiment_score": record.get("sentiment_score"),
            "is_sentiment_passed": int(record.get("is_sentiment_passed", 0)),
            "execution_price": record.get("execution_price"),
            "rationale": record.get("rationale", ""),
        }

        def _execute(s: Session):
            s.execute(query, params)
            logger.debug(f"Inserted recommendation for {record['ticker']} on {record['date']}")

        if session:
            _execute(session)
        else:
            with get_db_session(self.db_url) as s:
                _execute(s)

    def get_latest_recommendations(
        self,
        limit: int = 10,
        session: Optional[Session] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieves latest daily recommendations sorted by date descending.

        Parameters
        ----------
        limit : int
            Maximum records to return (default 10)

        Returns
        -------
        List[Dict[str, Any]]
            List of recommendation dictionary records.
        """
        query = text("""
            SELECT date, ticker, signal_type, prob_buy, prob_sell, prob_hold,
                   sentiment_score, is_sentiment_passed, execution_price, rationale
            FROM daily_recommendations
            ORDER BY date DESC, ticker ASC
            LIMIT :limit;
        """)

        def _execute(s: Session) -> List[Dict[str, Any]]:
            result = s.execute(query, {"limit": limit})
            return [dict(row._mapping) for row in result.fetchall()]

        if session:
            return _execute(session)
        else:
            with get_db_session(self.db_url) as s:
                return _execute(s)
