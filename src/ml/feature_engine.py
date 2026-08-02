import pandas as pd
import numpy as np
import logging

# Ensure src.indicators can be imported
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.indicators import (
    calculate_sma,
    calculate_rsi,
    calculate_macd,
    calculate_bollinger_bands
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range (ATR)."""
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())

    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)

    # ATR is typically smoothed using Wilder's smoothing (like RSI), but standard simple rolling mean is often used.
    # Wilder's smoothing:
    atr = true_range.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    return atr

def compute_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes technical indicators, lags them by 1 to prevent lookahead bias,
    and generates the target variable.
    """
    # Create a copy to avoid SettingWithCopy warnings and mutating original
    result = df.copy()

    # Require standard OHLCV
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_cols:
        if col not in result.columns:
            raise ValueError(f"Missing required column: {col}")

    # Calculate features on CURRENT data
    features = pd.DataFrame(index=result.index)

    # RSI
    features['RSI_14'] = calculate_rsi(result, column='Close', period=14)

    # MACD
    macd_line, signal_line, histogram = calculate_macd(result, column='Close', fast_period=12, slow_period=26, signal_period=9)
    features['MACD_Line'] = macd_line
    features['MACD_Signal'] = signal_line
    features['MACD_Hist'] = histogram

    # Bollinger Bands
    upper, middle, lower = calculate_bollinger_bands(result, column='Close', period=20, std_multiplier=2.0)
    # %B = (Price - Lower) / (Upper - Lower)
    band_width = upper - lower
    # Avoid division by zero
    band_width = band_width.replace(0, np.nan)
    features['BB_PctB'] = (result['Close'] - lower) / band_width
    features['BB_Bandwidth'] = (upper - lower) / middle

    # ATR normalized by Close
    atr = compute_atr(result, period=14)
    features['ATR_Norm'] = atr / result['Close']

    # Moving Average Ratios
    sma_20 = calculate_sma(result, column='Close', period=20)
    sma_50 = calculate_sma(result, column='Close', period=50)
    sma_200 = calculate_sma(result, column='Close', period=200)

    features['Ratio_Close_SMA20'] = result['Close'] / sma_20.replace(0, np.nan)
    features['Ratio_Close_SMA50'] = result['Close'] / sma_50.replace(0, np.nan)
    features['Ratio_SMA50_SMA200'] = sma_50 / sma_200.replace(0, np.nan)

    # Volatility
    features['Volatility_20d'] = result['Close'].pct_change().rolling(20).std() * np.sqrt(252)

    # SHIFT ALL FEATURES BY 1 to prevent lookahead bias!
    # If today is t, the features we use to predict the target for t must be based on data up to t-1
    # Alternatively, the target for t is the return from t to t+5. We can keep features at t and shift target to t-5 (which means it uses future prices).

    # Let's align features at time t with target based on future.
    # The requirement: "Ensure all features are lagged by 1 bar (shift(1)) relative to target calculation to prevent data leakage!"
    # Target at index `t` should reflect the return from `t` to `t+5`. The features at index `t` should only use data up to `t-1`.
    # Therefore, we calculate all features, then shift them forward by 1.
    shifted_features = features.shift(1)

    # Target Generation: 5-day forward return > +2.0%
    # Forward return from close of day t to close of day t+5
    forward_return_5d = (result['Close'].shift(-5) / result['Close']) - 1.0
    target = (forward_return_5d > 0.02).astype(int)
    # If forward return is NaN (at the end of the series), target should be NaN too before dropping
    target = target.where(forward_return_5d.notna(), np.nan)

    # Combine everything
    final_df = pd.concat([result, shifted_features], axis=1)
    final_df['Target_Buy'] = target

    # Drop all NaNs resulting from indicator warm-up periods and shift operations
    final_df = final_df.dropna()

    return final_df
