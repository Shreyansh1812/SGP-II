"""
src/config.py
=============

Centralized Configuration Module for SGP-II Quantitative Decision-Support System.

Loads all application settings from environment variables and `.env` files using
``pydantic-settings``. This module serves as the single source of truth for:

- Database connection parameters
- API keys for LLM providers (Gemini, Groq)
- Trading strategy parameters
- Signal generator thresholds
- ML model configuration
- Logging levels

Architecture
------------
Settings are loaded lazily via ``get_settings()`` and cached for the process
lifetime. All sensitive values (API keys, DB credentials) are read exclusively
from environment variables — **never hardcoded**.

Usage
-----
>>> from src.config import get_settings
>>> settings = get_settings()
>>> settings.GEMINI_API_KEY  # Read from .env or environment
>>> settings.SENTIMENT_THRESHOLD  # Default: 0.2

Author: Shreyansh Patel
Project: SGP-II — AI-Augmented Decision-Support Sandbox
"""

import os
from pathlib import Path
from typing import Literal, Optional
from functools import lru_cache
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ==============================================================================
# PATH CONFIGURATION
# ==============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent
"""Absolute path to the project root directory (parent of ``src/``)."""


class Settings(BaseSettings):
    """
    Application Settings for SGP-II.

    All settings are loaded from environment variables or the ``.env`` file
    located at the project root. Sensitive credentials are never stored in
    source code.

    Attributes
    ----------
    DB_TYPE : str
        Database dialect — ``'sqlite'`` for local development,
        ``'postgres'`` or ``'postgresql'`` for cloud deployment.
    DB_URL : str
        SQLAlchemy-compatible database connection URL.
    GEMINI_API_KEY : str
        Google Gemini API key for LLM-based sentiment analysis.
    GROQ_API_KEY : str
        Groq API key used as fallback LLM provider.
    SENTIMENT_THRESHOLD : float
        Minimum sentiment score required for BUY signal confirmation.
    PROB_BUY_THRESHOLD : float
        Minimum ML model P(BUY) probability for BUY signal generation.
    MODEL_PATH : str
        Relative path to the serialized ML model artifact.
    LOG_LEVEL : str
        Python logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    """

    # ======================== DATABASE ========================
    DB_TYPE: Literal["sqlite", "postgres", "postgresql"] = Field(
        default="sqlite",
        description="Database dialect: 'sqlite' or 'postgres'."
    )
    DB_URL: str = Field(
        default="sqlite:///data/sgp2.db",
        description="Database connection string URL."
    )

    # ======================== API KEYS ========================
    GEMINI_API_KEY: str = Field(
        default="",
        description="Google Gemini API key for sentiment analysis."
    )
    GROQ_API_KEY: str = Field(
        default="",
        description="Groq API key (fallback LLM provider)."
    )

    # ======================== LLM CONFIGURATION ========================
    LLM_PRIMARY_PROVIDER: Literal["gemini", "groq"] = Field(
        default="gemini",
        description="Primary LLM provider for sentiment analysis."
    )
    GEMINI_MODEL: str = Field(
        default="gemini-2.0-flash",
        description="Gemini model identifier."
    )
    GROQ_MODEL: str = Field(
        default="llama-3.1-8b-instant",
        description="Groq model identifier."
    )

    # ======================== SIGNAL THRESHOLDS ========================
    SENTIMENT_THRESHOLD: float = Field(
        default=0.2,
        description="Minimum sentiment score for BUY signal confirmation."
    )
    PROB_BUY_THRESHOLD: float = Field(
        default=0.65,
        description="Minimum P(BUY) probability for BUY signal generation."
    )
    FORWARD_RETURN_PCT: float = Field(
        default=0.02,
        description="Forward return threshold for target generation (e.g., 0.02 = 2%)."
    )
    FORWARD_RETURN_DAYS: int = Field(
        default=5,
        description="Forward return lookback window in trading days."
    )

    # ======================== MODEL CONFIGURATION ========================
    MODEL_PATH: str = Field(
        default="models/xgboost_signal_v1.pkl",
        description="Relative path to the serialized ML model artifact."
    )

    # ======================== TRADING CONFIGURATION ========================
    TICKER: str = Field(default="AAPL", description="Default stock ticker symbol.")
    START_DATE: str = Field(default="2020-01-01", description="Backtest start date.")
    END_DATE: str = Field(default="2024-01-01", description="Backtest end date.")
    INITIAL_CASH: float = Field(default=100000.0, description="Starting capital.")
    COMMISSION: float = Field(default=0.001, description="Commission rate per trade.")

    # ======================== STRATEGY PARAMETERS ========================
    GOLDEN_CROSS_FAST: int = Field(default=50, description="Golden Cross fast SMA period.")
    GOLDEN_CROSS_SLOW: int = Field(default=200, description="Golden Cross slow SMA period.")
    RSI_PERIOD: int = Field(default=14, description="RSI calculation period.")
    RSI_OVERSOLD: float = Field(default=30.0, description="RSI oversold threshold.")
    RSI_OVERBOUGHT: float = Field(default=70.0, description="RSI overbought threshold.")
    MACD_FAST: int = Field(default=12, description="MACD fast EMA period.")
    MACD_SLOW: int = Field(default=26, description="MACD slow EMA period.")
    MACD_SIGNAL: int = Field(default=9, description="MACD signal line period.")

    # ======================== LOGGING ========================
    LOG_LEVEL: str = Field(
        default="INFO",
        description="Application logging level."
    )

    # ======================== PYDANTIC SETTINGS CONFIG ========================
    model_config = SettingsConfigDict(
        env_file=os.path.join(BASE_DIR, ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
    )

    # ======================== VALIDATORS ========================
    @field_validator("SENTIMENT_THRESHOLD")
    @classmethod
    def validate_sentiment_threshold(cls, v: float) -> float:
        """Ensure sentiment threshold is within valid bounds."""
        if not -1.0 <= v <= 1.0:
            raise ValueError(f"SENTIMENT_THRESHOLD must be between -1.0 and 1.0, got {v}")
        return v

    @field_validator("PROB_BUY_THRESHOLD")
    @classmethod
    def validate_prob_buy_threshold(cls, v: float) -> float:
        """Ensure probability threshold is a valid probability."""
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"PROB_BUY_THRESHOLD must be between 0.0 and 1.0, got {v}")
        return v

    # ======================== COMPUTED PROPERTIES ========================
    def get_resolved_db_url(self) -> str:
        """
        Returns the resolved database connection URL.

        For SQLite file-based databases, ensures the parent directory exists
        and converts relative paths to absolute paths rooted at the project
        directory.

        Returns
        -------
        str
            Fully resolved database connection URL.
        """
        db_url = self.DB_URL
        if db_url.startswith("sqlite:///") and not db_url.startswith("sqlite:///:memory:"):
            file_path_str = db_url.replace("sqlite:///", "")
            if not os.path.isabs(file_path_str):
                abs_path = (BASE_DIR / file_path_str).resolve()
            else:
                abs_path = Path(file_path_str)
            abs_path.parent.mkdir(parents=True, exist_ok=True)
            return f"sqlite:///{abs_path.as_posix()}"
        return db_url

    def get_model_abs_path(self) -> Path:
        """
        Returns the absolute path to the ML model artifact.

        Creates parent directories if they do not exist.

        Returns
        -------
        Path
            Absolute path to the model file.
        """
        model_path = Path(self.MODEL_PATH)
        if not model_path.is_absolute():
            model_path = BASE_DIR / model_path
        model_path.parent.mkdir(parents=True, exist_ok=True)
        return model_path.resolve()

    @property
    def has_gemini_key(self) -> bool:
        """Check if a valid Gemini API key is configured."""
        return bool(self.GEMINI_API_KEY and self.GEMINI_API_KEY.strip())

    @property
    def has_groq_key(self) -> bool:
        """Check if a valid Groq API key is configured."""
        return bool(self.GROQ_API_KEY and self.GROQ_API_KEY.strip())


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Returns a cached Settings instance.

    The settings object is created once and reused for the lifetime of the
    process. Call ``get_settings.cache_clear()`` to force a reload (e.g.,
    after modifying environment variables in tests).

    Returns
    -------
    Settings
        Application configuration loaded from environment.
    """
    return Settings()
