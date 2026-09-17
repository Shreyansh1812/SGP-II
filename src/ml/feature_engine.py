"""
src/ml/feature_engine.py
=========================

Zero-Lookahead Technical Feature Engineering Pipeline.

Computes 14+ quantitative technical features from raw OHLCV data for use in
the SGP-II ML signal generator. All features are strictly lagged by one bar
(``shift(1)``) relative to the target variable to mathematically eliminate
lookahead bias.

Feature Set
-----------
Momentum:
    - RSI (14-period Relative Strength Index)
    - MACD Line, Signal, and Histogram (12, 26, 9)

Trend:
    - Bollinger Band %B and Bandwidth (20-period, 2σ)
    - Moving Average Ratios: Close/SMA20, Close/SMA50, SMA50/SMA200
    - ADX (14-period Average Directional Index)

Volatility:
    - ATR (14-period) normalized by Close price
    - 20-day annualized volatility (rolling std of returns × √252)

Volume:
    - On-Balance Volume (OBV) — 20-period Z-score normalized
    - Volume / SMA(Volume, 20) ratio

Target:
    - ``Target_Buy`` = 1 if 5-day forward return > +2.0%, else 0

Zero-Lookahead Guarantee
------------------------
All features at index *t* are derived from data up to *t−1* via
``features.shift(1)``. The target at index *t* uses the forward return
from *t* to *t+5*. This ensures no future information leaks into
the feature matrix.

Author: Shreyansh Patel
Project: SGP-II — AI-Augmented Decision-Support Sandbox
"""

import logging
from typing import Tuple

import numpy as np
import pandas as pd

from src.indicators import (
    calculate_sma,
    calculate_rsi,
    calculate_macd,
    calculate_bollinger_bands,
)

logger = logging.getLogger(__name__)


# ==============================================================================
# HELPER INDICATORS (not available in src/indicators.py)
# ==============================================================================

def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range (ATR) using Wilder's smoothing.

    ATR measures market volatility by decomposing the entire range of a
    security's price for that period.

    Formula (True Range per bar)::

        TR = max(High - Low, |High - Close_prev|, |Low - Close_prev|)
        ATR = EWM(TR, alpha=1/period)

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with 'High', 'Low', 'Close' columns.
    period : int
        Smoothing period (default 14).

    Returns
    -------
    pd.Series
        ATR values with NaN for warm-up period.
    """
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())

    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)

    # Wilder's exponential smoothing
    atr = true_range.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    return atr


def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Average Directional Index (ADX) for trend strength measurement.

    ADX quantifies trend strength regardless of direction. Values above 25
    indicate a strong trend; below 20 indicate a weak or absent trend.

    Formula::

        +DM = High_t - High_{t-1}   (if positive and > -DM, else 0)
        -DM = Low_{t-1} - Low_t     (if positive and > +DM, else 0)
        +DI = 100 × EWM(+DM) / ATR
        -DI = 100 × EWM(-DM) / ATR
        DX  = 100 × |+DI - -DI| / (+DI + -DI)
        ADX = EWM(DX, period)

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with 'High', 'Low', 'Close' columns.
    period : int
        Smoothing period (default 14).

    Returns
    -------
    pd.Series
        ADX values (0–100 scale). NaN during warm-up period.
    """
    high = df['High']
    low = df['Low']

    # Directional Movement
    plus_dm = high.diff()
    minus_dm = -low.diff()

    # Apply DM rules: only keep if positive and greater than the other
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    # Wilder's smoothing for DM and ATR
    atr = compute_atr(df, period)

    # Smoothed directional movement
    smooth_plus_dm = plus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    smooth_minus_dm = minus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    # Directional Indicators
    plus_di = 100.0 * smooth_plus_dm / atr.replace(0, np.nan)
    minus_di = 100.0 * smooth_minus_dm / atr.replace(0, np.nan)

    # Directional Index
    di_sum = plus_di + minus_di
    di_diff = (plus_di - minus_di).abs()
    dx = 100.0 * di_diff / di_sum.replace(0, np.nan)

    # ADX = smoothed DX
    adx = dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    return adx


def compute_obv(df: pd.DataFrame) -> pd.Series:
    """
    Calculate On-Balance Volume (OBV).

    OBV is a cumulative volume indicator that relates volume to price change.
    Rising OBV confirms an uptrend; falling OBV confirms a downtrend.

    Formula::

        If Close_t > Close_{t-1}: OBV_t = OBV_{t-1} + Volume_t
        If Close_t < Close_{t-1}: OBV_t = OBV_{t-1} - Volume_t
        If Close_t = Close_{t-1}: OBV_t = OBV_{t-1}

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with 'Close' and 'Volume' columns.

    Returns
    -------
    pd.Series
        OBV values (cumulative sum).
    """
    close_diff = df['Close'].diff()
    volume_direction = pd.Series(
        np.where(close_diff > 0, df['Volume'],
                 np.where(close_diff < 0, -df['Volume'], 0)),
        index=df.index,
        dtype=float,
    )
    obv = volume_direction.cumsum()
    return obv


# ==============================================================================
# MAIN FEATURE ENGINEERING FUNCTION
# ==============================================================================

def compute_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute technical indicators, lag them by 1 bar to prevent lookahead bias,
    and generate the binary classification target.

    This function is the sole feature engineering entry point for the SGP-II
    ML pipeline. It guarantees that features at time *t* use only data
    available up to *t−1*.

    Parameters
    ----------
    df : pd.DataFrame
        Raw OHLCV DataFrame with columns: Open, High, Low, Close, Volume.
        Index should be a DatetimeIndex.

    Returns
    -------
    pd.DataFrame
        Combined DataFrame containing:
        - Original OHLCV columns
        - 14 shifted (lagged) technical feature columns
        - ``Target_Buy`` binary target column
        - All NaN rows from warm-up periods dropped

    Raises
    ------
    ValueError
        If any required OHLCV column is missing.
    """
    result = df.copy()

    # Validate required columns
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing = [c for c in required_cols if c not in result.columns]
    if missing:
        raise ValueError(f"Missing required OHLCV column(s): {missing}")

    logger.info(f"Computing technical features for {len(result)} bars...")

    # ------------------------------------------------------------------
    # Calculate features on CURRENT data (will be shifted later)
    # ------------------------------------------------------------------
    features = pd.DataFrame(index=result.index)

    # --- Momentum ---
    features['RSI_14'] = calculate_rsi(result, column='Close', period=14)

    macd_line, signal_line, histogram = calculate_macd(
        result, column='Close', fast_period=12, slow_period=26, signal_period=9
    )
    features['MACD_Line'] = macd_line
    features['MACD_Signal'] = signal_line
    features['MACD_Hist'] = histogram

    # --- Trend: Bollinger Bands ---
    upper, middle, lower = calculate_bollinger_bands(
        result, column='Close', period=20, std_multiplier=2.0
    )
    band_width = upper - lower
    band_width = band_width.replace(0, np.nan)
    features['BB_PctB'] = (result['Close'] - lower) / band_width
    features['BB_Bandwidth'] = (upper - lower) / middle

    # --- Volatility: ATR normalized by Close ---
    atr = compute_atr(result, period=14)
    features['ATR_Norm'] = atr / result['Close']

    # --- Trend: Moving Average Ratios ---
    sma_20 = calculate_sma(result, column='Close', period=20)
    sma_50 = calculate_sma(result, column='Close', period=50)
    sma_200 = calculate_sma(result, column='Close', period=200)

    features['Ratio_Close_SMA20'] = result['Close'] / sma_20.replace(0, np.nan)
    features['Ratio_Close_SMA50'] = result['Close'] / sma_50.replace(0, np.nan)
    features['Ratio_SMA50_SMA200'] = sma_50 / sma_200.replace(0, np.nan)

    # --- Volatility: 20-day annualized ---
    features['Volatility_20d'] = (
        result['Close'].pct_change().rolling(20).std() * np.sqrt(252)
    )

    # --- Trend: ADX (14-period) ---
    features['ADX_14'] = compute_adx(result, period=14)

    # --- Volume: On-Balance Volume (20-period Z-score normalized) ---
    obv = compute_obv(result)
    obv_mean = obv.rolling(20).mean()
    obv_std = obv.rolling(20).std().replace(0, np.nan)
    features['OBV_Zscore'] = (obv - obv_mean) / obv_std

    # --- Volume: Volume / SMA(Volume, 20) ratio ---
    vol_sma_20 = result['Volume'].rolling(20).mean().replace(0, np.nan)
    features['Volume_Ratio_SMA20'] = result['Volume'] / vol_sma_20

    # ------------------------------------------------------------------
    # SHIFT ALL FEATURES BY 1 to prevent lookahead bias
    # Features at time t are derived from data up to t-1
    # ------------------------------------------------------------------
    shifted_features = features.shift(1)

    logger.info(f"Computed {len(shifted_features.columns)} technical features.")

    # ------------------------------------------------------------------
    # Target Generation: 5-day forward return > +2.0%
    # ------------------------------------------------------------------
    forward_return_5d = (result['Close'].shift(-5) / result['Close']) - 1.0
    target = (forward_return_5d > 0.02).astype(int)
    target = target.where(forward_return_5d.notna(), np.nan)

    # ------------------------------------------------------------------
    # Combine and clean
    # ------------------------------------------------------------------
    final_df = pd.concat([result, shifted_features], axis=1)
    final_df['Target_Buy'] = target

    # Drop all NaN rows from indicator warm-up and shift operations
    rows_before = len(final_df)
    final_df = final_df.dropna()
    rows_after = len(final_df)

    logger.info(
        f"Feature matrix ready: {rows_after} samples "
        f"(dropped {rows_before - rows_after} warm-up rows)."
    )

    return final_df
