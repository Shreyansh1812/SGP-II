"""
Configuration Module for Algorithmic Trading Backtester

This module loads configuration from environment variables with sensible defaults.
Environment variables allow flexible deployment across different environments
(development, testing, production) without hardcoding sensitive values.

Usage:
------
1. Create a .env file in project root (ignored by git):
   ```
   TICKER=RELIANCE.NS
   START_DATE=2020-01-01
   END_DATE=2024-01-01
   INITIAL_CASH=100000.0
   COMMISSION=0.001
   ```

2. Or set environment variables directly:
   ```bash
   # Windows PowerShell
   $env:TICKER="TCS.NS"
   
   # Linux/Mac
   export TICKER="TCS.NS"
   ```

3. Import and use:
   ```python
   from config import TICKER, START_DATE, INITIAL_CASH
   ```

Author: Shreyansh Patel
Project: SGP-II - Python-Based Algorithmic Trading Backtester
"""

import os
from pathlib import Path
from typing import Optional

# ==================== PATH CONFIGURATION ====================

# Dynamic Path Determination
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "raw"

# Ensure data directory exists
DATA_PATH.mkdir(parents=True, exist_ok=True)

# ==================== ENVIRONMENT VARIABLE HELPERS ====================

def get_env(key: str, default: Optional[str] = None, required: bool = False) -> str:
    """
    Get environment variable with validation.
    
    Parameters
    ----------
    key : str
        Environment variable name
    default : Optional[str], default=None
        Default value if variable not set
    required : bool, default=False
        If True, raises ValueError when variable missing and no default
        
    Returns
    -------
    str
        Environment variable value or default
        
    Raises
    ------
    ValueError
        If required=True and variable not found and no default
    """
    value = os.getenv(key, default)
    if required and value is None:
        raise ValueError(f"Required environment variable '{key}' is not set")
    return value


def get_env_float(key: str, default: float) -> float:
    """
    Get environment variable as float.
    
    Parameters
    ----------
    key : str
        Environment variable name
    default : float
        Default value if variable not set or invalid
        
    Returns
    -------
    float
        Environment variable value converted to float, or default
    """
    value = os.getenv(key)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        print(f"Warning: Invalid float value for {key}='{value}', using default {default}")
        return default


# ==================== TRADING CONFIGURATION ====================

# Stock Ticker Symbol (NSE format: SYMBOL.NS)
TICKER = get_env("TICKER", default="RELIANCE.NS")

# Backtest Date Range
START_DATE = get_env("START_DATE", default="2020-01-01")
END_DATE = get_env("END_DATE", default="2024-01-01")

# Portfolio Configuration
INITIAL_CASH = get_env_float("INITIAL_CASH", default=100000.0)
COMMISSION = get_env_float("COMMISSION", default=0.001)  # 0.1% per trade

# ==================== STRATEGY PARAMETERS ====================

# Golden Cross Strategy
GOLDEN_CROSS_FAST_PERIOD = int(get_env("GOLDEN_CROSS_FAST", default="50"))
GOLDEN_CROSS_SLOW_PERIOD = int(get_env("GOLDEN_CROSS_SLOW", default="200"))

# RSI Mean Reversion Strategy
RSI_PERIOD = int(get_env("RSI_PERIOD", default="14"))
RSI_OVERSOLD = get_env_float("RSI_OVERSOLD", default=30.0)
RSI_OVERBOUGHT = get_env_float("RSI_OVERBOUGHT", default=70.0)

# MACD Trend Following Strategy
MACD_FAST_PERIOD = int(get_env("MACD_FAST", default="12"))
MACD_SLOW_PERIOD = int(get_env("MACD_SLOW", default="26"))
MACD_SIGNAL_PERIOD = int(get_env("MACD_SIGNAL", default="9"))

# ==================== LOGGING CONFIGURATION ====================

LOG_LEVEL = get_env("LOG_LEVEL", default="INFO")

# ==================== DISPLAY CONFIGURATION ====================

def display_config():
    """Print current configuration for debugging/verification."""
    print("=" * 60)
    print("CONFIGURATION SETTINGS")
    print("=" * 60)
    print(f"Paths:")
    print(f"  BASE_DIR:  {BASE_DIR}")
    print(f"  DATA_PATH: {DATA_PATH}")
    print()
    print(f"Trading:")
    print(f"  TICKER:        {TICKER}")
    print(f"  START_DATE:    {START_DATE}")
    print(f"  END_DATE:      {END_DATE}")
    print(f"  INITIAL_CASH:  ${INITIAL_CASH:,.2f}")
    print(f"  COMMISSION:    {COMMISSION:.4f} ({COMMISSION*100:.2f}%)")
    print()
    print(f"Strategy Parameters:")
    print(f"  Golden Cross:  {GOLDEN_CROSS_FAST_PERIOD}/{GOLDEN_CROSS_SLOW_PERIOD}")
    print(f"  RSI:           Period={RSI_PERIOD}, Oversold={RSI_OVERSOLD}, Overbought={RSI_OVERBOUGHT}")
    print(f"  MACD:          Fast={MACD_FAST_PERIOD}, Slow={MACD_SLOW_PERIOD}, Signal={MACD_SIGNAL_PERIOD}")
    print()
    print(f"Logging:")
    print(f"  LOG_LEVEL:     {LOG_LEVEL}")
    print("=" * 60)


# Display configuration on import (optional, comment out in production)
if __name__ == "__main__":
    display_config()



