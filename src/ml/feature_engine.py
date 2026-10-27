"""
src/ml/feature_engine.py
=========================

Quantitative Feature Engineering Engine for SGP-II.

Computes stationary, normalized technical and market features from daily OHLCV
data for use in the SGP-II ML stock selection engine.

Key Principles:
1. Stationarity: All features are scaled as returns, ratios, bounded oscillators,
   or z-scores. No non-stationary raw price levels.
2. Zero Lookahead: Features for date t use information strictly up to and including
   Close_t.
3. Cross-Sectional Normalization: Features can be cross-sectionally ranked
   across the equity universe per date, eliminating macro market regime distortions.
4. Backwards Compatibility: Preserves compute_technical_features() for legacy modules.

Author: Shreyansh Patel & Quantitative Engineering Team
Project: SGP-II — AI-Augmented Decision-Support Sandbox
"""

import logging
from typing import Dict, List, Optional, Tuple

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
# HELPER INDICATORS
# ==============================================================================

def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range (ATR) using Wilder's smoothing."""
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())

    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)

    atr = true_range.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    return atr


def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average Directional Index (ADX) for trend strength measurement."""
    high = df['High']
    low = df['Low']

    plus_dm = high.diff()
    minus_dm = -low.diff()

    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    atr = compute_atr(df, period)

    smooth_plus_dm = plus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    smooth_minus_dm = minus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    plus_di = 100.0 * smooth_plus_dm / atr.replace(0, np.nan)
    minus_di = 100.0 * smooth_minus_dm / atr.replace(0, np.nan)

    di_sum = plus_di + minus_di
    di_diff = (plus_di - minus_di).abs()
    dx = 100.0 * di_diff / di_sum.replace(0, np.nan)

    adx = dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    return adx


def compute_obv(df: pd.DataFrame) -> pd.Series:
    """Calculate On-Balance Volume (OBV)."""
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
# EXTENDED STATIONARY FEATURE ENGINE (25 FEATURES)
# ==============================================================================

def compute_extended_features(
    df: pd.DataFrame, spy_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Compute ~25 stationary, normalized quantitative features for a single security.

    Features computed strictly using information available up to and including
    the current bar (Close_t). Does NOT attach future target variables.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with columns: Open, High, Low, Close, Volume.
    spy_df : Optional[pd.DataFrame]
        SPY benchmark OHLCV DataFrame for market-relative features.

    Returns
    -------
    pd.DataFrame
        DataFrame of technical features indexed identically to df.
    """
    result = df.copy()
    features = pd.DataFrame(index=result.index)
    close = result['Close']
    volume = result['Volume']

    # --- 1. Momentum & Relative Strength ---
    features['RSI_7'] = calculate_rsi(result, column='Close', period=7) / 100.0 - 0.5  # centered in [-0.5, 0.5]
    features['RSI_14'] = calculate_rsi(result, column='Close', period=14) / 100.0 - 0.5
    features['RSI_28'] = calculate_rsi(result, column='Close', period=28) / 100.0 - 0.5

    macd_line, signal_line, histogram = calculate_macd(
        result, column='Close', fast_period=12, slow_period=26, signal_period=9
    )
    features['MACD_Hist_Norm'] = histogram / close.replace(0, np.nan)
    features['MACD_Line_Norm'] = macd_line / close.replace(0, np.nan)

    features['Momentum_5d'] = close.pct_change(5)
    features['Momentum_20d'] = close.pct_change(20)
    features['Momentum_60d'] = close.pct_change(60)

    # --- 2. Trend & Moving Average Disparities ---
    sma_20 = calculate_sma(result, column='Close', period=20) if len(result) >= 20 else pd.Series(np.nan, index=result.index)
    sma_50 = calculate_sma(result, column='Close', period=50) if len(result) >= 50 else pd.Series(np.nan, index=result.index)
    sma_200 = calculate_sma(result, column='Close', period=200) if len(result) >= 200 else pd.Series(np.nan, index=result.index)

    features['Ratio_Close_SMA20'] = (close / sma_20.replace(0, np.nan)) - 1.0
    features['Ratio_Close_SMA50'] = (close / sma_50.replace(0, np.nan)) - 1.0
    features['Ratio_Close_SMA200'] = (close / sma_200.replace(0, np.nan)) - 1.0
    features['Ratio_SMA50_SMA200'] = (sma_50 / sma_200.replace(0, np.nan)) - 1.0

    # Bollinger Bands
    upper, middle, lower = calculate_bollinger_bands(
        result, column='Close', period=20, std_multiplier=2.0
    )
    band_width = (upper - lower).replace(0, np.nan)
    features['BB_PctB'] = (close - lower) / band_width - 0.5  # centered in [-0.5, 0.5]
    features['BB_Bandwidth'] = band_width / middle.replace(0, np.nan)

    # ADX Trend Strength
    features['ADX_14'] = compute_adx(result, period=14) / 100.0

    # --- 3. Volatility & Intraday Spreads ---
    atr = compute_atr(result, period=14)
    features['ATR_Norm'] = atr / close.replace(0, np.nan)

    vol_10 = close.pct_change().rolling(10).std() * np.sqrt(252)
    vol_30 = close.pct_change().rolling(30).std() * np.sqrt(252)
    features['Volatility_10d'] = vol_10
    features['Volatility_30d'] = vol_30
    features['Vol_Ratio_10_30'] = (vol_10 / vol_30.replace(0, np.nan)) - 1.0

    features['High_Low_Spread'] = (result['High'] - result['Low']) / close.replace(0, np.nan)

    # --- 4. Volume & Liquidity Dynamics ---
    vol_sma_20 = volume.rolling(20).mean().replace(0, np.nan)
    vol_sma_50 = volume.rolling(50).mean().replace(0, np.nan)
    features['Volume_Ratio_SMA20'] = (volume / vol_sma_20) - 1.0
    features['Volume_Ratio_SMA50'] = (volume / vol_sma_50) - 1.0

    obv = compute_obv(result)
    obv_mean = obv.rolling(20).mean()
    obv_std = obv.rolling(20).std().replace(0, np.nan)
    features['OBV_Zscore'] = (obv - obv_mean) / obv_std

    # Price-Volume 20-day correlation
    ret_1d = close.pct_change()
    vol_chg_1d = volume.pct_change()
    features['Price_Vol_Corr_20d'] = ret_1d.rolling(20).corr(vol_chg_1d)

    # --- 5. Market-Relative Features (if SPY provided) ---
    if spy_df is not None and 'Close' in spy_df.columns:
        spy_close = spy_df['Close'].reindex(result.index)
        spy_ret_20d = spy_close.pct_change(20)
        spy_ret_60d = spy_close.pct_change(60)

        features['SPY_Rel_Strength_20d'] = features['Momentum_20d'] - spy_ret_20d
        features['SPY_Rel_Strength_60d'] = features['Momentum_60d'] - spy_ret_60d

        # Rolling 60-day Beta to SPY
        spy_ret_1d = spy_close.pct_change()
        cov = ret_1d.rolling(60).cov(spy_ret_1d)
        var_spy = spy_ret_1d.rolling(60).var().replace(0, np.nan)
        features['Rolling_Beta_60d'] = (cov / var_spy) - 1.0  # centered: beta 1.0 -> 0.0

    return features


def cross_sectional_rank_normalize(
    features_by_ticker: Dict[str, pd.DataFrame]
) -> Dict[str, pd.DataFrame]:
    """
    Rank-normalize features across the equity universe on each calendar date.

    Transforms each feature X_{i,t} into a uniform percentile rank score:
    Z_{i,t} = (Rank(X_{i,t}) - 1) / (K - 1) - 0.5 in [-0.5, +0.5]
    where K is the number of active tickers on date t.

    Parameters
    ----------
    features_by_ticker : Dict[str, pd.DataFrame]
        Mapping of ticker -> feature DataFrame.

    Returns
    -------
    Dict[str, pd.DataFrame]
        Mapping of ticker -> rank-normalized feature DataFrame.
    """
    if not features_by_ticker:
        return {}

    tickers = list(features_by_ticker.keys())
    sample_df = next(iter(features_by_ticker.values()))
    feature_cols = sample_df.columns.tolist()

    # Pre-allocate output dictionaries
    ranked_by_ticker = {t: pd.DataFrame(index=features_by_ticker[t].index) for t in tickers}

    for col in feature_cols:
        # Build cross-sectional matrix for this feature: Index=Date, Columns=Ticker
        col_matrix = pd.DataFrame({
            t: features_by_ticker[t][col] for t in tickers if col in features_by_ticker[t].columns
        })

        # Rank across axis 1 (across tickers for each date)
        # pct=True produces values in (0, 1]
        ranks = col_matrix.rank(axis=1, pct=True) - 0.5  # centered in [-0.5, +0.5]

        for t in col_matrix.columns:
            ranked_by_ticker[t][col] = ranks[t]

    return ranked_by_ticker


# ==============================================================================
# LEGACY WRAPPER (FOR BACKWARDS COMPATIBILITY)
# ==============================================================================

def compute_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute 14 legacy technical indicators with shift(1) lag and Target_Buy.
    Maintained for complete backwards-compatibility with existing tests and scripts.
    """
    result = df.copy()
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing = [c for c in required_cols if c not in result.columns]
    if missing:
        raise ValueError(f"Missing required OHLCV column(s): {missing}")

    features = pd.DataFrame(index=result.index)

    # Momentum
    features['RSI_14'] = calculate_rsi(result, column='Close', period=14)
    macd_line, signal_line, histogram = calculate_macd(
        result, column='Close', fast_period=12, slow_period=26, signal_period=9
    )
    features['MACD_Line'] = macd_line
    features['MACD_Signal'] = signal_line
    features['MACD_Hist'] = histogram

    # Trend: Bollinger Bands
    upper, middle, lower = calculate_bollinger_bands(
        result, column='Close', period=20, std_multiplier=2.0
    )
    band_width = upper - lower
    band_width = band_width.replace(0, np.nan)
    features['BB_PctB'] = (result['Close'] - lower) / band_width
    features['BB_Bandwidth'] = (upper - lower) / middle

    # Volatility: ATR
    atr = compute_atr(result, period=14)
    features['ATR_Norm'] = atr / result['Close']

    # Trend: Moving Average Ratios
    sma_20 = calculate_sma(result, column='Close', period=20)
    sma_50 = calculate_sma(result, column='Close', period=50)
    sma_200 = calculate_sma(result, column='Close', period=200)

    features['Ratio_Close_SMA20'] = result['Close'] / sma_20.replace(0, np.nan)
    features['Ratio_Close_SMA50'] = result['Close'] / sma_50.replace(0, np.nan)
    features['Ratio_SMA50_SMA200'] = sma_50 / sma_200.replace(0, np.nan)

    # Volatility: 20-day annualized
    features['Volatility_20d'] = (
        result['Close'].pct_change().rolling(20).std() * np.sqrt(252)
    )

    # Trend: ADX
    features['ADX_14'] = compute_adx(result, period=14)

    # Volume: OBV
    obv = compute_obv(result)
    obv_mean = obv.rolling(20).mean()
    obv_std = obv.rolling(20).std().replace(0, np.nan)
    features['OBV_Zscore'] = (obv - obv_mean) / obv_std

    # Volume: Ratio
    vol_sma_20 = result['Volume'].rolling(20).mean().replace(0, np.nan)
    features['Volume_Ratio_SMA20'] = result['Volume'] / vol_sma_20

    shifted_features = features.shift(1)

    forward_return_5d = (result['Close'].shift(-5) / result['Close']) - 1.0
    target = (forward_return_5d > 0.02).astype(int)
    target = target.where(forward_return_5d.notna(), np.nan)

    final_df = pd.concat([result, shifted_features], axis=1)
    final_df['Target_Buy'] = target
    final_df = final_df.dropna()

    return final_df
