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

    def get_all_fundamentals(self, session: Optional[Session] = None) -> List[Dict[str, Any]]:
        """
        Retrieves all fundamental records across the tracked universe.

        Returns
        -------
        List[Dict[str, Any]]
            List of dictionaries for all tracked companies, sorted alphabetically by ticker.
        """
        query = text("""
            SELECT ticker, pe_ratio, debt_to_equity, operating_margin, roe, is_healthy,
                   cautionary_remark, last_updated
            FROM company_fundamentals
            ORDER BY ticker ASC;
        """)

        def _execute(s: Session) -> List[Dict[str, Any]]:
            result = s.execute(query)
            return [dict(row._mapping) for row in result.fetchall()]

        if session:
            return _execute(session)
        else:
            with get_db_session(self.db_url) as s:
                return _execute(s)


class SentimentDAO(BaseDAO):
    """Data Access Object for the `daily_sentiment` table (both macro and per-ticker)."""

    def upsert_sentiment(
        self,
        date_str: str,
        score: float,
        label: str,
        summary: str,
        ticker: str = "MACRO",
        session: Optional[Session] = None
    ) -> None:
        """
        Inserts or updates daily news sentiment for a specific ticker or 'MACRO'.

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
        ticker : str
            Stock ticker symbol or 'MACRO' for general market sentiment
        session : Optional[Session]
            Optional SQLAlchemy Session
        """
        if not (-1.0 <= score <= 1.0):
            logger.warning(f"Sentiment score {score} out of recommended bounds [-1.0, 1.0]")

        query = text("""
            INSERT INTO daily_sentiment (
                date, ticker, sentiment_score, sentiment_label, summary, created_at
            ) VALUES (
                :date, :ticker, :sentiment_score, :sentiment_label, :summary, CURRENT_TIMESTAMP
            ) ON CONFLICT(date, ticker) DO UPDATE SET
                sentiment_score = excluded.sentiment_score,
                sentiment_label = excluded.sentiment_label,
                summary = excluded.summary;
        """)
        params = {
            "date": date_str,
            "ticker": ticker.upper(),
            "sentiment_score": float(score),
            "sentiment_label": label,
            "summary": summary,
        }

        def _execute(s: Session):
            s.execute(query, params)
            logger.debug(f"Upserted sentiment for date {date_str}, ticker {ticker}")

        if session:
            _execute(session)
        else:
            with get_db_session(self.db_url) as s:
                _execute(s)

    def upsert_batch_sentiments(
        self,
        records: List[Dict[str, Any]],
        session: Optional[Session] = None
    ) -> None:
        """
        Bulk upserts multiple sentiment records in a single database transaction.

        Parameters
        ----------
        records : List[Dict[str, Any]]
            List of dicts with keys: ['date', 'ticker', 'sentiment_score', 'sentiment_label', 'summary']
        """
        if not records:
            return

        query = text("""
            INSERT INTO daily_sentiment (
                date, ticker, sentiment_score, sentiment_label, summary, created_at
            ) VALUES (
                :date, :ticker, :sentiment_score, :sentiment_label, :summary, CURRENT_TIMESTAMP
            ) ON CONFLICT(date, ticker) DO UPDATE SET
                sentiment_score = excluded.sentiment_score,
                sentiment_label = excluded.sentiment_label,
                summary = excluded.summary;
        """)

        clean_params = [
            {
                "date": r["date"],
                "ticker": r.get("ticker", "MACRO").upper(),
                "sentiment_score": float(r["sentiment_score"]),
                "sentiment_label": r["sentiment_label"],
                "summary": r.get("summary", ""),
            }
            for r in records
        ]

        def _execute(s: Session):
            s.execute(query, clean_params)
            logger.debug(f"Batch upserted {len(clean_params)} sentiment records.")

        if session:
            _execute(session)
        else:
            with get_db_session(self.db_url) as s:
                _execute(s)

    def get_sentiment(
        self,
        date_str: str,
        ticker: str = "MACRO",
        session: Optional[Session] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieves daily sentiment record by date and ticker (with fallback to MACRO).

        Parameters
        ----------
        date_str : str
            Date string YYYY-MM-DD
        ticker : str
            Stock ticker or 'MACRO' (default 'MACRO')

        Returns
        -------
        Optional[Dict[str, Any]]
            Dictionary containing sentiment attributes or None if not found.
        """
        query = text("""
            SELECT date, ticker, sentiment_score, sentiment_label, summary, created_at
            FROM daily_sentiment 
            WHERE date = :date AND ticker = :ticker;
        """)

        def _execute(s: Session) -> Optional[Dict[str, Any]]:
            row = s.execute(query, {"date": date_str, "ticker": ticker.upper()}).fetchone()
            if row:
                return dict(row._mapping)
            if ticker.upper() != "MACRO":
                # Fallback to MACRO record if ticker-specific record doesn't exist
                macro_row = s.execute(query, {"date": date_str, "ticker": "MACRO"}).fetchone()
                if macro_row:
                    return dict(macro_row._mapping)
            return None

        if session:
            return _execute(session)
        else:
            with get_db_session(self.db_url) as s:
                return _execute(s)

    def get_all_sentiments_for_date(
        self,
        date_str: str,
        session: Optional[Session] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Retrieves all sentiment records for a given date mapped by ticker.

        Returns
        -------
        Dict[str, Dict[str, Any]]
            Mapping of uppercase ticker -> sentiment record dict.
        """
        query = text("""
            SELECT date, ticker, sentiment_score, sentiment_label, summary, created_at
            FROM daily_sentiment 
            WHERE date = :date;
        """)

        def _execute(s: Session) -> Dict[str, Dict[str, Any]]:
            result = s.execute(query, {"date": date_str}).fetchall()
            return {row._mapping["ticker"]: dict(row._mapping) for row in result}

        if session:
            return _execute(session)
        else:
            with get_db_session(self.db_url) as s:
                return _execute(s)

    def get_sentiment_history(
        self,
        limit: int = 60,
        ticker: str = "MACRO",
        session: Optional[Session] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieves historical sentiment records sorted by date descending for a ticker or MACRO.

        Parameters
        ----------
        limit : int
            Maximum records to return (default 60).
        ticker : str
            Ticker symbol to retrieve history for (default 'MACRO').

        Returns
        -------
        List[Dict[str, Any]]
            List of sentiment record dictionaries.
        """
        query = text("""
            SELECT date, ticker, sentiment_score, sentiment_label, summary, created_at
            FROM daily_sentiment
            WHERE ticker = :ticker
            ORDER BY date DESC
            LIMIT :limit;
        """)

        def _execute(s: Session) -> List[Dict[str, Any]]:
            result = s.execute(query, {"limit": limit, "ticker": ticker.upper()})
            return [dict(row._mapping) for row in result.fetchall()]

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

    def get_recommendations_by_date(
        self,
        date_str: str,
        session: Optional[Session] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieves all model recommendations for a specific date, ordered by prob_buy descending.

        Parameters
        ----------
        date_str : str
            Date formatted as YYYY-MM-DD.

        Returns
        -------
        List[Dict[str, Any]]
            List of recommendation dictionary records for that date.
        """
        query = text("""
            SELECT date, ticker, signal_type, prob_buy, prob_sell, prob_hold,
                   sentiment_score, is_sentiment_passed, execution_price, rationale
            FROM daily_recommendations
            WHERE date = :date
            ORDER BY prob_buy DESC;
        """)

        def _execute(s: Session) -> List[Dict[str, Any]]:
            result = s.execute(query, {"date": date_str})
            return [dict(row._mapping) for row in result.fetchall()]

        if session:
            return _execute(session)
        else:
            with get_db_session(self.db_url) as s:
                return _execute(s)

    def get_available_dates(
        self,
        limit: int = 30,
        session: Optional[Session] = None
    ) -> List[str]:
        """
        Retrieves distinct dates that have recommendation records.

        Parameters
        ----------
        limit : int
            Maximum number of distinct dates to return (default 30).

        Returns
        -------
        List[str]
            List of date strings ordered descending.
        """
        query = text("""
            SELECT DISTINCT date
            FROM daily_recommendations
            ORDER BY date DESC
            LIMIT :limit;
        """)

        def _execute(s: Session) -> List[str]:
            result = s.execute(query, {"limit": limit})
            return [row[0] for row in result.fetchall()]

        if session:
            return _execute(session)
        else:
            with get_db_session(self.db_url) as s:
                return _execute(s)

    def get_recommendations_by_ticker(
        self,
        ticker: str,
        limit: int = 50,
        session: Optional[Session] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieves historical recommendations for an individual ticker.

        Parameters
        ----------
        ticker : str
            Ticker symbol.
        limit : int
            Maximum records to return (default 50).

        Returns
        -------
        List[Dict[str, Any]]
            List of recommendation records ordered by date descending.
        """
        query = text("""
            SELECT date, ticker, signal_type, prob_buy, prob_sell, prob_hold,
                   sentiment_score, is_sentiment_passed, execution_price, rationale
            FROM daily_recommendations
            WHERE ticker = :ticker
            ORDER BY date DESC
            LIMIT :limit;
        """)

        def _execute(s: Session) -> List[Dict[str, Any]]:
            result = s.execute(query, {"ticker": ticker.upper(), "limit": limit})
            return [dict(row._mapping) for row in result.fetchall()]

        if session:
            return _execute(session)
        else:
            with get_db_session(self.db_url) as s:
                return _execute(s)


# =============================================================================
# ENHANCEMENT DAOs (added 2026-09-30)
# =============================================================================

class AuditTrailDAO(BaseDAO):
    """Data Access Object for the `recommendation_audit_trail` table."""

    def insert_audit_record(
        self,
        record: Dict[str, Any],
        session: Optional[Session] = None
    ) -> Optional[int]:
        """
        Inserts a complete recommendation audit snapshot.

        Parameters
        ----------
        record : Dict[str, Any]
            Dictionary matching recommendation_audit_trail schema.

        Returns
        -------
        Optional[int]
            The auto-generated ID of the inserted record, or None on failure.
        """
        query = text("""
            INSERT INTO recommendation_audit_trail (
                ticker, recommendation, recommendation_score, timestamp,
                market_price_at_recommendation, ml_probability,
                technical_signals, fundamental_signals, sentiment_signal,
                risk_factors, supporting_factors, score_explanation,
                model_version, data_timestamp
            ) VALUES (
                :ticker, :recommendation, :recommendation_score, :timestamp,
                :market_price_at_recommendation, :ml_probability,
                :technical_signals, :fundamental_signals, :sentiment_signal,
                :risk_factors, :supporting_factors, :score_explanation,
                :model_version, :data_timestamp
            );
        """)
        params = {
            "ticker": record.get("ticker", "").upper(),
            "recommendation": record.get("recommendation", "HOLD"),
            "recommendation_score": record.get("recommendation_score"),
            "timestamp": record.get("timestamp"),
            "market_price_at_recommendation": record.get("market_price_at_recommendation"),
            "ml_probability": record.get("ml_probability"),
            "technical_signals": record.get("technical_signals"),
            "fundamental_signals": record.get("fundamental_signals"),
            "sentiment_signal": record.get("sentiment_signal"),
            "risk_factors": record.get("risk_factors"),
            "supporting_factors": record.get("supporting_factors"),
            "score_explanation": record.get("score_explanation"),
            "model_version": record.get("model_version"),
            "data_timestamp": record.get("data_timestamp"),
        }

        def _execute(s: Session) -> Optional[int]:
            s.execute(query, params)
            result = s.execute(text("SELECT last_insert_rowid();"))
            row = result.fetchone()
            logger.debug(f"Inserted audit record for {record.get('ticker')}")
            return row[0] if row else None

        if session:
            return _execute(session)
        else:
            with get_db_session(self.db_url) as s:
                return _execute(s)

    def get_audit_by_id(
        self, audit_id: int, session: Optional[Session] = None
    ) -> Optional[Dict[str, Any]]:
        """Retrieves a single audit record by ID."""
        query = text("""
            SELECT * FROM recommendation_audit_trail WHERE id = :id;
        """)

        def _execute(s: Session) -> Optional[Dict[str, Any]]:
            row = s.execute(query, {"id": audit_id}).fetchone()
            return dict(row._mapping) if row else None

        if session:
            return _execute(session)
        else:
            with get_db_session(self.db_url) as s:
                return _execute(s)

    def get_audits_by_ticker(
        self, ticker: str, limit: int = 50, session: Optional[Session] = None
    ) -> List[Dict[str, Any]]:
        """Retrieves audit records for a specific ticker, most recent first."""
        query = text("""
            SELECT * FROM recommendation_audit_trail
            WHERE ticker = :ticker
            ORDER BY timestamp DESC
            LIMIT :limit;
        """)

        def _execute(s: Session) -> List[Dict[str, Any]]:
            result = s.execute(query, {"ticker": ticker.upper(), "limit": limit})
            return [dict(row._mapping) for row in result.fetchall()]

        if session:
            return _execute(session)
        else:
            with get_db_session(self.db_url) as s:
                return _execute(s)

    def get_audits_by_date_range(
        self,
        start_date: str,
        end_date: str,
        recommendation_filter: Optional[str] = None,
        session: Optional[Session] = None
    ) -> List[Dict[str, Any]]:
        """Retrieves audit records within a date range with optional recommendation filter."""
        # Handle both ISO 'T' format and space format for robust date matching
        norm_start = start_date.replace(" ", "T") if " " in start_date else start_date
        norm_end = end_date.replace(" ", "T") if " " in end_date else end_date
        if len(norm_start) == 10:
            norm_start = f"{norm_start}T00:00:00"
        if len(norm_end) == 10:
            norm_end = f"{norm_end}T23:59:59"

        space_start = start_date.replace("T", " ") if "T" in start_date else start_date
        space_end = end_date.replace("T", " ") if "T" in end_date else end_date

        if recommendation_filter:
            query = text("""
                SELECT * FROM recommendation_audit_trail
                WHERE ((timestamp >= :start_date AND timestamp <= :end_date)
                   OR (timestamp >= :space_start AND timestamp <= :space_end))
                AND recommendation = :rec_filter
                ORDER BY timestamp DESC;
            """)
            params = {
                "start_date": norm_start,
                "end_date": norm_end,
                "space_start": space_start,
                "space_end": space_end,
                "rec_filter": recommendation_filter,
            }
        else:
            query = text("""
                SELECT * FROM recommendation_audit_trail
                WHERE ((timestamp >= :start_date AND timestamp <= :end_date)
                   OR (timestamp >= :space_start AND timestamp <= :space_end))
                ORDER BY timestamp DESC;
            """)
            params = {
                "start_date": norm_start,
                "end_date": norm_end,
                "space_start": space_start,
                "space_end": space_end,
            }

        def _execute(s: Session) -> List[Dict[str, Any]]:
            result = s.execute(query, params)
            return [dict(row._mapping) for row in result.fetchall()]

        if session:
            return _execute(session)
        else:
            with get_db_session(self.db_url) as s:
                return _execute(s)

    def get_audits_by_date(
        self, date_str: str, session: Optional[Session] = None
    ) -> List[Dict[str, Any]]:
        """Retrieves all audit records for a specific date (YYYY-MM-DD)."""
        return self.get_audits_by_date_range(date_str, date_str, session=session)

    def get_recent_audits(
        self, limit: int = 100, session: Optional[Session] = None
    ) -> List[Dict[str, Any]]:
        """Retrieves the most recent audit records."""
        query = text("""
            SELECT * FROM recommendation_audit_trail
            ORDER BY timestamp DESC
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


class PortfolioDAO(BaseDAO):
    """Data Access Object for the `portfolio_positions` table."""

    def add_position(
        self,
        record: Dict[str, Any],
        session: Optional[Session] = None
    ) -> Optional[int]:
        """
        Adds a new portfolio position (paper trade).

        Parameters
        ----------
        record : Dict[str, Any]
            Must contain: ticker, quantity, purchase_price, purchase_date.
            Optional: recommendation_audit_id, recommendation_score,
                      recommendation_type, notes.

        Returns
        -------
        Optional[int]
            The auto-generated ID of the new position, or None on failure.
        """
        query = text("""
            INSERT INTO portfolio_positions (
                ticker, quantity, purchase_price, purchase_date,
                recommendation_audit_id, recommendation_score,
                recommendation_type, status, notes
            ) VALUES (
                :ticker, :quantity, :purchase_price, :purchase_date,
                :recommendation_audit_id, :recommendation_score,
                :recommendation_type, 'OPEN', :notes
            );
        """)
        params = {
            "ticker": record["ticker"].upper(),
            "quantity": float(record["quantity"]),
            "purchase_price": float(record["purchase_price"]),
            "purchase_date": record["purchase_date"],
            "recommendation_audit_id": record.get("recommendation_audit_id"),
            "recommendation_score": record.get("recommendation_score"),
            "recommendation_type": record.get("recommendation_type"),
            "notes": record.get("notes"),
        }

        def _execute(s: Session) -> Optional[int]:
            s.execute(query, params)
            result = s.execute(text("SELECT last_insert_rowid();"))
            row = result.fetchone()
            logger.debug(f"Added portfolio position for {record['ticker']}")
            return row[0] if row else None

        if session:
            return _execute(session)
        else:
            with get_db_session(self.db_url) as s:
                return _execute(s)

    def close_position(
        self,
        position_id: int,
        sell_price: float,
        sell_date: str,
        session: Optional[Session] = None
    ) -> bool:
        """Closes an open position by recording sell price and date. Returns True if closed."""
        query = text("""
            UPDATE portfolio_positions
            SET status = 'CLOSED', sell_price = :sell_price, sell_date = :sell_date
            WHERE id = :id AND status = 'OPEN';
        """)

        def _execute(s: Session) -> bool:
            result = s.execute(query, {
                "id": position_id,
                "sell_price": float(sell_price),
                "sell_date": sell_date,
            })
            logger.debug(f"Closed position {position_id}")
            return result.rowcount > 0

        if session:
            return _execute(session)
        else:
            with get_db_session(self.db_url) as s:
                return _execute(s)

    def get_open_positions(
        self, session: Optional[Session] = None
    ) -> List[Dict[str, Any]]:
        """Retrieves all currently open portfolio positions."""
        query = text("""
            SELECT * FROM portfolio_positions
            WHERE status = 'OPEN'
            ORDER BY purchase_date DESC;
        """)

        def _execute(s: Session) -> List[Dict[str, Any]]:
            result = s.execute(query)
            return [dict(row._mapping) for row in result.fetchall()]

        if session:
            return _execute(session)
        else:
            with get_db_session(self.db_url) as s:
                return _execute(s)

    def get_all_positions(
        self, session: Optional[Session] = None
    ) -> List[Dict[str, Any]]:
        """Retrieves all portfolio positions (open and closed)."""
        query = text("""
            SELECT * FROM portfolio_positions
            ORDER BY purchase_date DESC;
        """)

        def _execute(s: Session) -> List[Dict[str, Any]]:
            result = s.execute(query)
            return [dict(row._mapping) for row in result.fetchall()]

        if session:
            return _execute(session)
        else:
            with get_db_session(self.db_url) as s:
                return _execute(s)

    def get_position_by_id(
        self, position_id: int, session: Optional[Session] = None
    ) -> Optional[Dict[str, Any]]:
        """Retrieves a single position by ID."""
        query = text("""
            SELECT * FROM portfolio_positions WHERE id = :id;
        """)

        def _execute(s: Session) -> Optional[Dict[str, Any]]:
            row = s.execute(query, {"id": position_id}).fetchone()
            return dict(row._mapping) if row else None

        if session:
            return _execute(session)
        else:
            with get_db_session(self.db_url) as s:
                return _execute(s)

    def get_portfolio_summary(
        self, session: Optional[Session] = None
    ) -> Dict[str, Any]:
        """
        Retrieves aggregate portfolio summary for open positions.

        Returns
        -------
        Dict[str, Any]
            Contains total_invested, position_count, and positions list.
        """
        query = text("""
            SELECT ticker, quantity, purchase_price, purchase_date,
                   recommendation_score, recommendation_type
            FROM portfolio_positions
            WHERE status = 'OPEN'
            ORDER BY purchase_date DESC;
        """)

        def _execute(s: Session) -> Dict[str, Any]:
            result = s.execute(query)
            positions = [dict(row._mapping) for row in result.fetchall()]
            total_invested = sum(
                p["quantity"] * p["purchase_price"] for p in positions
            )
            return {
                "total_invested": total_invested,
                "position_count": len(positions),
                "positions": positions,
            }

        if session:
            return _execute(session)
        else:
            with get_db_session(self.db_url) as s:
                return _execute(s)


class BenchmarkDAO(BaseDAO):
    """Data Access Object for the `model_benchmark_reports` table."""

    def insert_report(
        self,
        record: Dict[str, Any],
        session: Optional[Session] = None
    ) -> Optional[int]:
        """
        Inserts a model benchmark report.

        Parameters
        ----------
        record : Dict[str, Any]
            Dictionary matching model_benchmark_reports schema.

        Returns
        -------
        Optional[int]
            The auto-generated ID of the inserted report.
        """
        query = text("""
            INSERT INTO model_benchmark_reports (
                model_name, model_version, dataset_period,
                training_period, validation_period, test_period,
                accuracy, precision_score, recall_score, f1_score,
                roc_auc, brier_score, directional_accuracy,
                backtested_return, max_drawdown, sharpe_ratio,
                win_rate, total_trades, training_time_seconds,
                total_samples, class_distribution,
                parameters, feature_importances, tickers_used
            ) VALUES (
                :model_name, :model_version, :dataset_period,
                :training_period, :validation_period, :test_period,
                :accuracy, :precision_score, :recall_score, :f1_score,
                :roc_auc, :brier_score, :directional_accuracy,
                :backtested_return, :max_drawdown, :sharpe_ratio,
                :win_rate, :total_trades, :training_time_seconds,
                :total_samples, :class_distribution,
                :parameters, :feature_importances, :tickers_used
            );
        """)
        params = {
            "model_name": record.get("model_name", "unknown"),
            "model_version": record.get("model_version"),
            "dataset_period": record.get("dataset_period"),
            "training_period": record.get("training_period"),
            "validation_period": record.get("validation_period"),
            "test_period": record.get("test_period"),
            "accuracy": record.get("accuracy"),
            "precision_score": record.get("precision_score"),
            "recall_score": record.get("recall_score"),
            "f1_score": record.get("f1_score"),
            "roc_auc": record.get("roc_auc"),
            "brier_score": record.get("brier_score"),
            "directional_accuracy": record.get("directional_accuracy"),
            "backtested_return": record.get("backtested_return"),
            "max_drawdown": record.get("max_drawdown"),
            "sharpe_ratio": record.get("sharpe_ratio"),
            "win_rate": record.get("win_rate"),
            "total_trades": record.get("total_trades"),
            "training_time_seconds": record.get("training_time_seconds"),
            "total_samples": record.get("total_samples"),
            "class_distribution": record.get("class_distribution"),
            "parameters": record.get("parameters"),
            "feature_importances": record.get("feature_importances"),
            "tickers_used": record.get("tickers_used"),
        }

        def _execute(s: Session) -> Optional[int]:
            s.execute(query, params)
            result = s.execute(text("SELECT last_insert_rowid();"))
            row = result.fetchone()
            logger.debug(f"Inserted benchmark report for {record.get('model_name')}")
            return row[0] if row else None

        if session:
            return _execute(session)
        else:
            with get_db_session(self.db_url) as s:
                return _execute(s)

    def get_latest_reports(
        self, limit: int = 20, session: Optional[Session] = None
    ) -> List[Dict[str, Any]]:
        """Retrieves the most recent benchmark reports."""
        query = text("""
            SELECT * FROM model_benchmark_reports
            ORDER BY created_at DESC
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

    def get_reports_by_model(
        self, model_name: str, session: Optional[Session] = None
    ) -> List[Dict[str, Any]]:
        """Retrieves all benchmark reports for a specific model name."""
        query = text("""
            SELECT * FROM model_benchmark_reports
            WHERE model_name = :model_name
            ORDER BY created_at DESC;
        """)

        def _execute(s: Session) -> List[Dict[str, Any]]:
            result = s.execute(query, {"model_name": model_name})
            return [dict(row._mapping) for row in result.fetchall()]

        if session:
            return _execute(session)
        else:
            with get_db_session(self.db_url) as s:
                return _execute(s)


class RecommendationPerformanceDAO(BaseDAO):
    """Data Access Object for the `recommendation_performance` table."""

    def upsert_performance(
        self,
        record: Dict[str, Any],
        session: Optional[Session] = None
    ) -> None:
        """
        Inserts or updates a recommendation performance evaluation record.

        Parameters
        ----------
        record : Dict[str, Any]
            Dictionary matching recommendation_performance schema.
        """
        query = text("""
            INSERT INTO recommendation_performance (
                recommendation_audit_id, ticker, recommendation_date,
                recommendation_type, entry_price,
                price_1d, price_1w, price_1m, price_3m,
                return_1d, return_1w, return_1m, return_3m,
                max_gain, max_loss,
                is_profitable_1d, is_profitable_1w,
                is_profitable_1m, is_profitable_3m,
                last_evaluated
            ) VALUES (
                :recommendation_audit_id, :ticker, :recommendation_date,
                :recommendation_type, :entry_price,
                :price_1d, :price_1w, :price_1m, :price_3m,
                :return_1d, :return_1w, :return_1m, :return_3m,
                :max_gain, :max_loss,
                :is_profitable_1d, :is_profitable_1w,
                :is_profitable_1m, :is_profitable_3m,
                CURRENT_TIMESTAMP
            ) ON CONFLICT(recommendation_audit_id) DO UPDATE SET
                price_1d = excluded.price_1d,
                price_1w = excluded.price_1w,
                price_1m = excluded.price_1m,
                price_3m = excluded.price_3m,
                return_1d = excluded.return_1d,
                return_1w = excluded.return_1w,
                return_1m = excluded.return_1m,
                return_3m = excluded.return_3m,
                max_gain = excluded.max_gain,
                max_loss = excluded.max_loss,
                is_profitable_1d = excluded.is_profitable_1d,
                is_profitable_1w = excluded.is_profitable_1w,
                is_profitable_1m = excluded.is_profitable_1m,
                is_profitable_3m = excluded.is_profitable_3m,
                last_evaluated = CURRENT_TIMESTAMP;
        """)
        params = {
            "recommendation_audit_id": record["recommendation_audit_id"],
            "ticker": record["ticker"].upper(),
            "recommendation_date": record["recommendation_date"],
            "recommendation_type": record["recommendation_type"],
            "entry_price": record["entry_price"],
            "price_1d": record.get("price_1d"),
            "price_1w": record.get("price_1w"),
            "price_1m": record.get("price_1m"),
            "price_3m": record.get("price_3m"),
            "return_1d": record.get("return_1d"),
            "return_1w": record.get("return_1w"),
            "return_1m": record.get("return_1m"),
            "return_3m": record.get("return_3m"),
            "max_gain": record.get("max_gain"),
            "max_loss": record.get("max_loss"),
            "is_profitable_1d": record.get("is_profitable_1d"),
            "is_profitable_1w": record.get("is_profitable_1w"),
            "is_profitable_1m": record.get("is_profitable_1m"),
            "is_profitable_3m": record.get("is_profitable_3m"),
        }

        def _execute(s: Session):
            s.execute(query, params)
            logger.debug(
                f"Upserted performance for audit_id={record['recommendation_audit_id']}"
            )

        if session:
            _execute(session)
        else:
            with get_db_session(self.db_url) as s:
                _execute(s)

    def get_performance_by_audit_id(
        self, audit_id: int, session: Optional[Session] = None
    ) -> Optional[Dict[str, Any]]:
        """Retrieves performance record for a specific audit trail entry."""
        query = text("""
            SELECT * FROM recommendation_performance
            WHERE recommendation_audit_id = :audit_id;
        """)

        def _execute(s: Session) -> Optional[Dict[str, Any]]:
            row = s.execute(query, {"audit_id": audit_id}).fetchone()
            return dict(row._mapping) if row else None

        if session:
            return _execute(session)
        else:
            with get_db_session(self.db_url) as s:
                return _execute(s)

    def get_all_performance(
        self, limit: int = 200, session: Optional[Session] = None
    ) -> List[Dict[str, Any]]:
        """Retrieves all performance evaluation records."""
        query = text("""
            SELECT * FROM recommendation_performance
            ORDER BY recommendation_date DESC
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

    def get_aggregate_stats(
        self, session: Optional[Session] = None
    ) -> Dict[str, Any]:
        """
        Computes aggregate performance statistics across all evaluated recommendations.

        Returns
        -------
        Dict[str, Any]
            Contains total_evaluated, win_rate, avg_return, avg_gain, avg_loss, etc.
        """
        query = text("""
            SELECT
                COUNT(*) as total_evaluated,
                SUM(CASE WHEN is_profitable_1w = 1 THEN 1 ELSE 0 END) as profitable_1w,
                SUM(CASE WHEN is_profitable_1m = 1 THEN 1 ELSE 0 END) as profitable_1m,
                AVG(return_1w) as avg_return_1w,
                AVG(return_1m) as avg_return_1m,
                AVG(CASE WHEN return_1w > 0 THEN return_1w END) as avg_gain_1w,
                AVG(CASE WHEN return_1w < 0 THEN return_1w END) as avg_loss_1w,
                AVG(CASE WHEN return_1m > 0 THEN return_1m END) as avg_gain_1m,
                AVG(CASE WHEN return_1m < 0 THEN return_1m END) as avg_loss_1m,
                AVG(max_gain) as avg_max_gain,
                AVG(max_loss) as avg_max_loss
            FROM recommendation_performance
            WHERE return_1w IS NOT NULL;
        """)

        def _execute(s: Session) -> Dict[str, Any]:
            row = s.execute(query).fetchone()
            if row is None:
                return {"total_evaluated": 0}
            data = dict(row._mapping)
            total = data.get("total_evaluated", 0)
            if total > 0:
                data["win_rate_1w"] = (data.get("profitable_1w", 0) or 0) / total
                data["win_rate_1m"] = (data.get("profitable_1m", 0) or 0) / total
            else:
                data["win_rate_1w"] = 0.0
                data["win_rate_1m"] = 0.0
            return data

        if session:
            return _execute(session)
        else:
            with get_db_session(self.db_url) as s:
                return _execute(s)

    def get_stats_by_recommendation_type(
        self, session: Optional[Session] = None
    ) -> List[Dict[str, Any]]:
        """Retrieves aggregate stats grouped by recommendation type (BUY/HOLD/SELL)."""
        query = text("""
            SELECT
                recommendation_type,
                COUNT(*) as total,
                SUM(CASE WHEN is_profitable_1w = 1 THEN 1 ELSE 0 END) as profitable_1w,
                SUM(CASE WHEN is_profitable_1m = 1 THEN 1 ELSE 0 END) as profitable_1m,
                AVG(return_1w) as avg_return_1w,
                AVG(return_1m) as avg_return_1m
            FROM recommendation_performance
            WHERE return_1w IS NOT NULL
            GROUP BY recommendation_type;
        """)

        def _execute(s: Session) -> List[Dict[str, Any]]:
            result = s.execute(query)
            return [dict(row._mapping) for row in result.fetchall()]

        if session:
            return _execute(session)
        else:
            with get_db_session(self.db_url) as s:
                return _execute(s)
