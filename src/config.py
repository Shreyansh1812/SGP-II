"""
src/config.py
Configuration module for SGP-II quantitative decision-support system.
Loads settings from environment variables and .env files using pydantic-settings.
"""

import os
from pathlib import Path
from typing import Literal
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    """
    Application Settings for SGP-II.
    Reads environment variables and .env configuration.
    """
    DB_TYPE: Literal["sqlite", "postgres", "postgresql"] = Field(
        default="sqlite",
        description="Database dialect: 'sqlite' or 'postgres'"
    )
    DB_URL: str = Field(
        default="sqlite:///data/sgp2.db",
        description="Database connection string URL"
    )
    LOG_LEVEL: str = Field(
        default="INFO",
        description="Application logging level"
    )

    model_config = SettingsConfigDict(
        env_file=os.path.join(BASE_DIR, ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    def get_resolved_db_url(self) -> str:
        """
        Returns the resolved database connection URL.
        If SQLite local file path, ensures parent data directory exists.
        """
        db_url = self.DB_URL
        if db_url.startswith("sqlite:///") and not db_url.startswith("sqlite:///:memory:"):
            # Extract file path after sqlite:///
            file_path_str = db_url.replace("sqlite:///", "")
            if not os.path.isabs(file_path_str):
                abs_path = (BASE_DIR / file_path_str).resolve()
            else:
                abs_path = Path(file_path_str)
            abs_path.parent.mkdir(parents=True, exist_ok=True)
            return f"sqlite:///{abs_path.as_posix()}"
        return db_url

def get_settings() -> Settings:
    """Returns a fresh or cached Settings instance."""
    return Settings()
