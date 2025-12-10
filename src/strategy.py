"""
Trading Strategy Module for Algorithmic Trading Backtester.

This module implements various technical analysis-based trading strategies that generate
buy/sell signals based on technical indicators. Each strategy returns a signal series
where 1 = BUY, -1 = SELL, and 0 = HOLD.

Strategies Implemented:
    1. Golden Cross Strategy - SMA crossover signals
    2. RSI Mean Reversion Strategy - Threshold crossing signals
    3. MACD Trend Following Strategy - MACD/Signal line crossover

Author: Shreyansh Patel
Date: December 10, 2025
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional

# Import technical indicators from indicators module
from indicators import calculate_sma, calculate_rsi, calculate_ema

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def golden_cross_strategy(
    data: pd.DataFrame,
    column: str = 'Close',
    fast_period: int = 50,
    slow_period: int = 200
) -> pd.Series:
    """
    Generate buy/sell signals based on Golden Cross and Death Cross patterns.
    
    The Golden Cross is a bullish signal that occurs when a short-term moving average
    (fast SMA) crosses above a long-term moving average (slow SMA), indicating potential
    upward momentum. Conversely, the Death Cross (fast crosses below slow) signals
    potential downward momentum.
    
    This is one of the most widely-used technical trading patterns, particularly popular
    among institutional investors. The classic implementation uses SMA(50) and SMA(200),
    though these parameters can be adjusted for different trading timeframes.
    
    SIGNAL LOGIC:
        Golden Cross (BUY):
            - Previous day: SMA_fast <= SMA_slow
            - Current day:  SMA_fast > SMA_slow
            → Signal = 1 (Enter long position)
        
        Death Cross (SELL):
            - Previous day: SMA_fast >= SMA_slow
            - Current day:  SMA_fast < SMA_slow
            → Signal = -1 (Exit long position)
        
        Hold:
            - No crossover detected
            → Signal = 0 (Maintain current position)
    
    TRADING INTERPRETATION:
        - Golden Cross: Bullish reversal, momentum shifting upward
          * Fast SMA crossing above slow SMA indicates short-term strength
          * Typically marks the beginning of sustained uptrends
          * Best used in trending markets, not sideways/choppy conditions
        
        - Death Cross: Bearish reversal, momentum shifting downward
          * Fast SMA crossing below slow SMA indicates short-term weakness
          * Often precedes extended downtrends
          * Signal to exit positions or consider shorting
    
    STRENGTHS:
        ✓ Filters out short-term noise (especially with 50/200 periods)
        ✓ Catches major trends lasting months or years
        ✓ Clear, objective entry/exit rules
        ✓ Low false positive rate in trending markets
    
    WEAKNESSES:
        ✗ Lagging indicator - signals after trend already started (10-20% move missed)
        ✗ Poor performance in sideways/ranging markets (whipsaws)
        ✗ Requires substantial historical data (slow_period + buffer)
        ✗ Slow exit signals may give back significant profits
    
    PARAMETER GUIDELINES:
        Classic (Long-term):
            fast_period=50, slow_period=200
            → For swing traders, position traders
            → Captures major multi-month trends
        
        Aggressive (Short-term):
            fast_period=20, slow_period=50
            → For active day traders
            → More signals but higher risk of whipsaws
        
        Conservative (Confirmation):
            fast_period=100, slow_period=200
            → For risk-averse investors
            → Fewer but more reliable signals
    
    Args:
        data (pd.DataFrame): OHLCV DataFrame with DatetimeIndex.
            Required columns depend on 'column' parameter (default: 'Close').
            Must have sufficient rows (>= slow_period) for meaningful signals.
        
        column (str, optional): Price column to analyze. Defaults to 'Close'.
            Common alternatives: 'Open', 'High', 'Low', 'Adj Close'.
        
        fast_period (int, optional): Period for fast (short-term) SMA. Defaults to 50.
            Must be positive and less than slow_period.
            Smaller values = more responsive but noisier signals.
        
        slow_period (int, optional): Period for slow (long-term) SMA. Defaults to 200.
            Must be positive and greater than fast_period.
            Larger values = smoother but more lagging signals.
    
    Returns:
        pd.Series: Signal series with same index as input data.
            - Values: 1 (BUY), -1 (SELL), 0 (HOLD)
            - Name: 'Golden_Cross_Signal_{fast}_{slow}'
            - Index: DatetimeIndex matching input data
            - dtype: int8 (memory efficient)
    
    Raises:
        ValueError: If input validation fails:
            - data is empty or not a DataFrame
            - fast_period >= slow_period (logically invalid)
            - fast_period or slow_period < 1
            - slow_period > length of data (insufficient history)
            - column not found in data
            - column contains non-numeric data
    
    Example:
        >>> from src.data_loader import get_stock_data
        >>> from src.strategy import golden_cross_strategy
        >>> 
        >>> # Get stock data
        >>> df = get_stock_data("RELIANCE.NS", "2022-01-01", "2023-12-31")
        >>> 
        >>> # Generate signals with default parameters (50/200)
        >>> signals = golden_cross_strategy(df)
        >>> 
        >>> # Check for buy signals
        >>> buy_dates = signals[signals == 1].index
        >>> print(f"Golden Crosses detected: {len(buy_dates)}")
        >>> print(buy_dates)
        >>> 
        >>> # Generate signals with aggressive parameters (20/50)
        >>> signals_aggressive = golden_cross_strategy(df, fast_period=20, slow_period=50)
        >>> 
        >>> # Combine with price data for analysis
        >>> df['Signal'] = signals
        >>> df['Position'] = signals.replace(0, np.nan).ffill().fillna(0)
        >>> print(df[df['Signal'] != 0])  # Show only days with signals
    
    Real-World Example:
        Apple Inc. (AAPL) - Golden Cross on April 6, 2020:
            - Context: COVID-19 market recovery beginning
            - SMA(50) crossed above SMA(200) at $254.80
            - Entry price: ~$259.43
            - 3-month return: +50.1% (July 2020: $389.32)
            - Death Cross occurred in March 2022 after major run-up
    
    Performance Considerations:
        - Time Complexity: O(n) where n = len(data)
          * Two SMA calculations: O(n) each
          * Crossover detection: O(n)
        
        - Space Complexity: O(n)
          * Two SMA series stored
          * One signal series returned
        
        - Typical execution time: <0.1s for 1 year of daily data
    
    Notes:
        - First (slow_period - 1) signals will be 0 (insufficient data for SMA)
        - Signals are generated on close prices; execution assumed at next open
        - Strategy assumes long-only positions (no shorting on sell signals)
        - For short positions, interpret -1 as "enter short" rather than "exit long"
        - Combine with volume analysis or other indicators for confirmation
    
    See Also:
        - calculate_sma(): Simple Moving Average calculation
        - rsi_mean_reversion_strategy(): Alternative mean reversion approach
        - macd_trend_strategy(): Trend following with MACD crossovers
    
    References:
        - Murphy, J. (1999). Technical Analysis of the Financial Markets
        - Investopedia: Golden Cross vs Death Cross
        - Classic implementation: 50-day/200-day crossover system
    """
    # =========================================================================
    # PARAMETER VALIDATION
    # =========================================================================
    
    logging.info(f"Generating Golden Cross signals with fast={fast_period}, slow={slow_period}")
    
    # Validate data input
    if not isinstance(data, pd.DataFrame):
        raise ValueError(
            f"Input 'data' must be a pandas DataFrame. Got {type(data)}. "
            f"Ensure you're passing a DataFrame with OHLCV columns."
        )
    
    if data.empty:
        raise ValueError(
            "Input DataFrame is empty. Cannot generate signals on empty data. "
            "Please provide historical price data with at least slow_period rows."
        )
    
    # Validate periods
    if fast_period < 1:
        raise ValueError(
            f"fast_period must be >= 1. Got: {fast_period}. "
            f"Period represents number of days for moving average calculation."
        )
    
    if slow_period < 1:
        raise ValueError(
            f"slow_period must be >= 1. Got: {slow_period}. "
            f"Period represents number of days for moving average calculation."
        )
    
    if fast_period >= slow_period:
        raise ValueError(
            f"fast_period ({fast_period}) must be less than slow_period ({slow_period}). "
            f"Golden Cross requires fast SMA to be shorter than slow SMA. "
            f"Common combinations: (50, 200), (20, 50), (10, 30)."
        )
    
    if slow_period > len(data):
        raise ValueError(
            f"slow_period ({slow_period}) cannot be greater than data length ({len(data)}). "
            f"Need at least {slow_period} rows for {slow_period}-day SMA calculation. "
            f"Consider using shorter periods or fetching more historical data."
        )
    
    # Validate column
    if column not in data.columns:
        raise ValueError(
            f"Column '{column}' not found in DataFrame. "
            f"Available columns: {list(data.columns)}. "
            f"Common price columns: 'Open', 'High', 'Low', 'Close', 'Adj Close'."
        )
    
    if not pd.api.types.is_numeric_dtype(data[column]):
        raise ValueError(
            f"Column '{column}' must contain numeric data for SMA calculation. "
            f"Found dtype: {data[column].dtype}. "
            f"Please ensure price data is float or integer type."
        )
    
    # =========================================================================
    # CALCULATE MOVING AVERAGES
    # =========================================================================
    
    logging.info(f"Calculating SMA({fast_period}) and SMA({slow_period}) on column '{column}'")
    
    # Calculate fast and slow SMAs using indicators module
    sma_fast = calculate_sma(data, column=column, period=fast_period)
    sma_slow = calculate_sma(data, column=column, period=slow_period)
    
    logging.info(f"SMAs calculated: Fast has {sma_fast.notna().sum()} valid values, "
                f"Slow has {sma_slow.notna().sum()} valid values")
    
    # =========================================================================
    # DETECT CROSSOVERS
    # =========================================================================
    
    # Initialize signal series with 0 (HOLD)
    signals = pd.Series(index=data.index, data=0, dtype='int8', name=f'Golden_Cross_Signal_{fast_period}_{slow_period}')
    
    # Get previous day's values for crossover detection
    sma_fast_prev = sma_fast.shift(1)
    sma_slow_prev = sma_slow.shift(1)
    
    # Golden Cross Detection (BUY signal)
    # Condition: Fast was below/equal to slow, now fast is above slow
    golden_cross = (sma_fast_prev <= sma_slow_prev) & (sma_fast > sma_slow)
    signals[golden_cross] = 1
    
    # Death Cross Detection (SELL signal)
    # Condition: Fast was above/equal to slow, now fast is below slow
    death_cross = (sma_fast_prev >= sma_slow_prev) & (sma_fast < sma_slow)
    signals[death_cross] = -1
    
    # =========================================================================
    # LOGGING & STATISTICS
    # =========================================================================
    
    buy_signals = (signals == 1).sum()
    sell_signals = (signals == -1).sum()
    hold_signals = (signals == 0).sum()
    
    logging.info(f"Signal generation complete:")
    logging.info(f"  - Buy signals (Golden Cross): {buy_signals}")
    logging.info(f"  - Sell signals (Death Cross): {sell_signals}")
    logging.info(f"  - Hold signals: {hold_signals}")
    logging.info(f"  - Total signals: {len(signals)}")
    
    if buy_signals > 0:
        buy_dates = signals[signals == 1].index
        logging.info(f"  - Golden Cross dates: {list(buy_dates.strftime('%Y-%m-%d'))}")
    
    if sell_signals > 0:
        sell_dates = signals[signals == -1].index
        logging.info(f"  - Death Cross dates: {list(sell_dates.strftime('%Y-%m-%d'))}")
    
    # Calculate current position (are we in uptrend or downtrend?)
    # Valid only where both SMAs exist
    valid_comparison = sma_fast.notna() & sma_slow.notna()
    if valid_comparison.any():
        last_valid_idx = valid_comparison[::-1].idxmax()  # Last True index
        current_position = "BULLISH (Fast > Slow)" if sma_fast[last_valid_idx] > sma_slow[last_valid_idx] else "BEARISH (Fast < Slow)"
        logging.info(f"  - Current market position: {current_position}")
    
    return signals


def rsi_mean_reversion_strategy(
    data: pd.DataFrame,
    column: str = 'Close',
    rsi_period: int = 14,
    oversold_threshold: float = 30.0,
    overbought_threshold: float = 70.0
) -> pd.Series:
    """
    Generate buy/sell signals based on RSI mean reversion (overbought/oversold).
    
    Mean reversion is a financial theory suggesting that asset prices tend to return
    to their average or mean over time. This strategy exploits RSI extremes, buying
    when prices are oversold (expecting bounce) and selling when overbought (expecting
    pullback).
    
    The Relative Strength Index (RSI) oscillates between 0-100. Traditional thresholds:
    - RSI < 30: Oversold (price fell too much → expect recovery)
    - RSI > 70: Overbought (price rose too much → expect correction)
    
    SIGNAL LOGIC:
        Buy Signal (Oversold Bounce):
            - Previous: RSI >= oversold_threshold (30)
            - Current:  RSI < oversold_threshold
            → Signal = 1 (Enter long, expecting mean reversion up)
        
        Sell Signal (Overbought Correction):
            - Previous: RSI <= overbought_threshold (70)
            - Current:  RSI > overbought_threshold
            → Signal = -1 (Exit long or short, expecting mean reversion down)
        
        Hold:
            - RSI between thresholds (normal range)
            → Signal = 0 (No extreme condition detected)
    
    TRADING INTERPRETATION:
        - Oversold (RSI < 30): Market panic, sellers exhausted
          * Price dropped sharply, RSI in extreme territory
          * Mean reversion theory: Price will bounce back
          * BUY signal: Enter long position
        
        - Overbought (RSI > 70): Market euphoria, buyers exhausted
          * Price rallied sharply, RSI in extreme territory
          * Mean reversion theory: Price will pull back
          * SELL signal: Exit long or enter short
        
        - Normal Range (30-70): No extreme conditions
          * Price movement within typical volatility
          * No action needed, wait for extremes
    
    STRENGTHS:
        ✓ Fast signals - Responds quicker than moving average crossovers
        ✓ Works well in ranging/sideways markets (oscillating prices)
        ✓ Clear numerical thresholds (objective entry/exit)
        ✓ Can catch short-term reversals for quick profits
        ✓ Multiple opportunities per trend cycle
    
    WEAKNESSES:
        ✗ Whipsaws in strong trends - RSI stays oversold in downtrend, overbought in uptrend
        ✗ False signals in trending markets - "overbought can stay overbought"
        ✗ Requires threshold tuning per asset (30/70 not universal)
        ✗ Sensitive to period selection (14 vs 9 vs 21 days)
        ✗ Can miss big moves waiting for mean reversion
    
    PARAMETER GUIDELINES:
        Standard (Balanced):
            oversold=30, overbought=70, rsi_period=14
            → Traditional Wilder parameters
            → Good for most stocks, moderate signal frequency
        
        Conservative (High confidence, fewer signals):
            oversold=20, overbought=80, rsi_period=14
            → Only extreme conditions
            → Lower false positives, may miss moves
        
        Aggressive (More signals, higher risk):
            oversold=40, overbought=60, rsi_period=9
            → Earlier entries, faster RSI response
            → More whipsaws, suitable for day trading
        
        Long-term (Trend confirmation):
            oversold=30, overbought=70, rsi_period=21
            → Smoother RSI, fewer signals
            → Better for position traders
    
    Args:
        data (pd.DataFrame): OHLCV DataFrame with DatetimeIndex.
            Must have sufficient rows (>= rsi_period + 1) for RSI calculation.
        
        column (str, optional): Price column to analyze. Defaults to 'Close'.
            RSI typically calculated on closing prices.
        
        rsi_period (int, optional): RSI calculation period. Defaults to 14.
            Wilder's original RSI used 14 periods.
            Smaller = more sensitive, Larger = smoother.
        
        oversold_threshold (float, optional): RSI level for buy signals. Defaults to 30.0.
            Must be between 0-100 and less than overbought_threshold.
            Lower values = more conservative (fewer buy signals).
        
        overbought_threshold (float, optional): RSI level for sell signals. Defaults to 70.0.
            Must be between 0-100 and greater than oversold_threshold.
            Higher values = more conservative (fewer sell signals).
    
    Returns:
        pd.Series: Signal series with same index as input data.
            - Values: 1 (BUY/oversold), -1 (SELL/overbought), 0 (HOLD/normal)
            - Name: 'RSI_Mean_Reversion_Signal_{oversold}_{overbought}'
            - Index: DatetimeIndex matching input data
            - dtype: int8 (memory efficient)
    
    Raises:
        ValueError: If input validation fails:
            - data is empty or not a DataFrame
            - rsi_period < 2 (insufficient for RSI calculation)
            - oversold_threshold not in range (0, 100)
            - overbought_threshold not in range (0, 100)
            - oversold_threshold >= overbought_threshold (illogical)
            - column not found in data
            - column contains non-numeric data
    
    Example:
        >>> from src.data_loader import get_stock_data
        >>> from src.strategy import rsi_mean_reversion_strategy
        >>> 
        >>> # Get stock data
        >>> df = get_stock_data("RELIANCE.NS", "2023-01-01", "2023-12-31")
        >>> 
        >>> # Generate signals with default parameters (30/70)
        >>> signals = rsi_mean_reversion_strategy(df)
        >>> 
        >>> # Count signals
        >>> buy_signals = (signals == 1).sum()
        >>> sell_signals = (signals == -1).sum()
        >>> print(f"Buy: {buy_signals}, Sell: {sell_signals}")
        >>> 
        >>> # Aggressive parameters for day trading
        >>> signals_aggressive = rsi_mean_reversion_strategy(
        ...     df, oversold=40, overbought=60, rsi_period=9
        ... )
        >>> 
        >>> # Conservative parameters for swing trading
        >>> signals_conservative = rsi_mean_reversion_strategy(
        ...     df, oversold=20, overbought=80
        ... )
    
    Real-World Example:
        Bitcoin (BTC) - High volatility, good for mean reversion:
            - Jan 15: RSI drops to 28 → BUY signal
            - Jan 20: Price bounces +8%, RSI at 45
            - Feb 10: RSI spikes to 72 → SELL signal
            - Feb 15: Price corrects -6%, RSI at 58
            - Result: 2 profitable mean reversion trades
    
    Performance Considerations:
        - Time Complexity: O(n) where n = len(data)
          * One RSI calculation: O(n)
          * Crossover detection: O(n)
        
        - Space Complexity: O(n)
          * One RSI series stored
          * One signal series returned
        
        - Typical execution time: <0.05s for 1 year of daily data
    
    Notes:
        - First (rsi_period) signals will be 0 (insufficient data for RSI)
        - Works best in range-bound, oscillating markets
        - Less effective in strong trends (RSI can stay extreme)
        - Consider combining with trend filter (e.g., only buy if above 200 SMA)
        - RSI can remain overbought/oversold for extended periods in strong trends
        - Default 30/70 thresholds are Wilder's original recommendations
        - Some traders use 25/75 or 20/80 for less frequent signals
    
    See Also:
        - calculate_rsi(): RSI calculation function
        - golden_cross_strategy(): Trend-following alternative
        - macd_trend_strategy(): Momentum-based trend following
    
    References:
        - Wilder, J.W. (1978). New Concepts in Technical Trading Systems
        - Murphy, J. (1999). Technical Analysis of the Financial Markets
        - Investopedia: RSI Mean Reversion Strategy
    """
    # =========================================================================
    # PARAMETER VALIDATION
    # =========================================================================
    
    logging.info(f"Generating RSI Mean Reversion signals with RSI({rsi_period}), "
                f"oversold={oversold_threshold}, overbought={overbought_threshold}")
    
    # Validate data input
    if not isinstance(data, pd.DataFrame):
        raise ValueError(
            f"Input 'data' must be a pandas DataFrame. Got {type(data)}. "
            f"Ensure you're passing a DataFrame with OHLCV columns."
        )
    
    if data.empty:
        raise ValueError(
            "Input DataFrame is empty. Cannot generate signals on empty data. "
            "Please provide historical price data with at least rsi_period rows."
        )
    
    # Validate RSI period
    if rsi_period < 2:
        raise ValueError(
            f"rsi_period must be >= 2 for RSI calculation. Got: {rsi_period}. "
            f"Common values: 9 (fast), 14 (standard), 21 (slow)."
        )
    
    # Validate thresholds
    if not (0 < oversold_threshold < 100):
        raise ValueError(
            f"oversold_threshold must be between 0 and 100. Got: {oversold_threshold}. "
            f"Common values: 20 (conservative), 30 (standard), 40 (aggressive)."
        )
    
    if not (0 < overbought_threshold < 100):
        raise ValueError(
            f"overbought_threshold must be between 0 and 100. Got: {overbought_threshold}. "
            f"Common values: 60 (aggressive), 70 (standard), 80 (conservative)."
        )
    
    if oversold_threshold >= overbought_threshold:
        raise ValueError(
            f"oversold_threshold ({oversold_threshold}) must be less than "
            f"overbought_threshold ({overbought_threshold}). "
            f"Typical configuration: oversold=30, overbought=70."
        )
    
    # Validate column
    if column not in data.columns:
        raise ValueError(
            f"Column '{column}' not found in DataFrame. "
            f"Available columns: {list(data.columns)}. "
            f"RSI is typically calculated on 'Close' prices."
        )
    
    if not pd.api.types.is_numeric_dtype(data[column]):
        raise ValueError(
            f"Column '{column}' must contain numeric data for RSI calculation. "
            f"Found dtype: {data[column].dtype}. "
            f"Please ensure price data is float or integer type."
        )
    
    # =========================================================================
    # CALCULATE RSI
    # =========================================================================
    
    logging.info(f"Calculating RSI({rsi_period}) on column '{column}'")
    
    # Calculate RSI using indicators module
    rsi = calculate_rsi(data, column=column, period=rsi_period)
    
    logging.info(f"RSI calculated: {rsi.notna().sum()} valid values out of {len(data)}")
    
    # =========================================================================
    # DETECT THRESHOLD CROSSINGS
    # =========================================================================
    
    # Initialize signal series with 0 (HOLD)
    signals = pd.Series(
        index=data.index, 
        data=0, 
        dtype='int8', 
        name=f'RSI_Mean_Reversion_Signal_{int(oversold_threshold)}_{int(overbought_threshold)}'
    )
    
    # Get previous day's RSI for crossover detection
    rsi_prev = rsi.shift(1)
    
    # Buy Signal Detection (RSI crosses BELOW oversold threshold)
    # Condition: RSI was above/equal to threshold, now below
    oversold_cross = (rsi_prev >= oversold_threshold) & (rsi < oversold_threshold)
    signals[oversold_cross] = 1
    
    # Sell Signal Detection (RSI crosses ABOVE overbought threshold)
    # Condition: RSI was below/equal to threshold, now above
    overbought_cross = (rsi_prev <= overbought_threshold) & (rsi > overbought_threshold)
    signals[overbought_cross] = -1
    
    # =========================================================================
    # LOGGING & STATISTICS
    # =========================================================================
    
    buy_signals = (signals == 1).sum()
    sell_signals = (signals == -1).sum()
    hold_signals = (signals == 0).sum()
    
    logging.info(f"Signal generation complete:")
    logging.info(f"  - Buy signals (Oversold): {buy_signals}")
    logging.info(f"  - Sell signals (Overbought): {sell_signals}")
    logging.info(f"  - Hold signals: {hold_signals}")
    logging.info(f"  - Total signals: {len(signals)}")
    
    if buy_signals > 0:
        buy_dates = signals[signals == 1].index
        buy_rsi_values = rsi[signals == 1]
        logging.info(f"  - Oversold crossings: {len(buy_dates)} times")
        if len(buy_dates) <= 5:
            for date, rsi_val in zip(buy_dates, buy_rsi_values):
                logging.info(f"    * {date.strftime('%Y-%m-%d')}: RSI={rsi_val:.2f}")
        else:
            logging.info(f"    * First: {buy_dates[0].strftime('%Y-%m-%d')}, "
                        f"Last: {buy_dates[-1].strftime('%Y-%m-%d')}")
    
    if sell_signals > 0:
        sell_dates = signals[signals == -1].index
        sell_rsi_values = rsi[signals == -1]
        logging.info(f"  - Overbought crossings: {len(sell_dates)} times")
        if len(sell_dates) <= 5:
            for date, rsi_val in zip(sell_dates, sell_rsi_values):
                logging.info(f"    * {date.strftime('%Y-%m-%d')}: RSI={rsi_val:.2f}")
        else:
            logging.info(f"    * First: {sell_dates[0].strftime('%Y-%m-%d')}, "
                        f"Last: {sell_dates[-1].strftime('%Y-%m-%d')}")
    
    # Calculate current RSI status
    if rsi.notna().any():
        last_valid_idx = rsi.last_valid_index()
        current_rsi = rsi[last_valid_idx]
        
        if current_rsi < oversold_threshold:
            status = f"OVERSOLD (RSI={current_rsi:.2f} < {oversold_threshold})"
        elif current_rsi > overbought_threshold:
            status = f"OVERBOUGHT (RSI={current_rsi:.2f} > {overbought_threshold})"
        else:
            status = f"NORMAL (RSI={current_rsi:.2f}, range {oversold_threshold}-{overbought_threshold})"
        
        logging.info(f"  - Current RSI status: {status}")
    
    return signals


# =============================================================================
# STRATEGY 3: MACD TREND FOLLOWING STRATEGY
# =============================================================================

def macd_trend_following_strategy(
    data: pd.DataFrame,
    column: str = 'Close',
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
    zero_line_filter: bool = False
) -> pd.Series:
    """
    Generate trading signals based on MACD (Moving Average Convergence Divergence) crossovers.
    
    MACD is a trend-following momentum indicator that shows the relationship between two 
    exponential moving averages (EMAs). It consists of three components:
    
    1. MACD Line = EMA(fast) - EMA(slow)
    2. Signal Line = EMA(MACD Line, signal_period)
    3. MACD Histogram = MACD Line - Signal Line
    
    **Trading Logic:**
    - **BUY Signal (1):**  MACD Line crosses ABOVE Signal Line (bullish crossover)
    - **SELL Signal (-1):** MACD Line crosses BELOW Signal Line (bearish crossover)
    - **HOLD Signal (0):**  No crossover detected
    
    **Zero-Line Filter (Optional):**
    When enabled, only generates signals that align with overall trend:
    - BUY only when MACD > 0 (bullish territory)
    - SELL only when MACD < 0 (bearish territory)
    
    **Mathematical Formula:**
    ```
    EMA_fast = EMA(Close, fast_period)       # Default: 12
    EMA_slow = EMA(Close, slow_period)       # Default: 26
    MACD_line = EMA_fast - EMA_slow
    Signal_line = EMA(MACD_line, signal_period)  # Default: 9
    
    # Detect crossovers using previous day comparison
    MACD_prev = MACD_line.shift(1)
    Signal_prev = Signal_line.shift(1)
    
    Bullish_cross = (MACD_prev <= Signal_prev) & (MACD_line > Signal_line)
    Bearish_cross = (MACD_prev >= Signal_prev) & (MACD_line < Signal_line)
    ```
    
    **Real-World Example:**
    
    Stock: RELIANCE.NS (June 2023)
    
    | Date   | Close   | MACD  | Signal | Crossover | Action |
    |--------|---------|-------|--------|-----------|--------|
    | Jun 5  | ₹1155   | +2.5  | +3.0   | -         | HOLD   |
    | Jun 8  | ₹1148   | +1.8  | +2.2   | -         | HOLD   |
    | Jun 12 | ₹1140   | +0.5  | +1.5   | -         | HOLD   |
    | Jun 15 | ₹1135   | -0.2  | +0.8   | Bearish   | SELL   |
    | Jun 20 | ₹1128   | -1.5  | +0.2   | -         | HOLD   |
    
    On June 15, MACD crossed below Signal line → SELL signal generated
    
    **Use Cases:**
    - **Medium to long-term trading:** Best for swing trading (days to weeks)
    - **Trending markets:** Works well in clear uptrends or downtrends
    - **Momentum confirmation:** Combines trend direction with momentum strength
    - **Divergence detection:** Can identify potential trend reversals
    
    **Advantages:**
    - Combines trend and momentum analysis
    - Widely used by institutional traders
    - Flexible with zero-line filter option
    - Works well in trending markets
    
    **Limitations:**
    - Lagging indicator (uses EMAs, not real-time)
    - Generates whipsaws in sideways/choppy markets
    - Requires sufficient historical data (at least slow_period + signal_period)
    - False signals in ranging markets
    
    **Parameter Guidelines:**
    - **fast_period (12):** Faster response to price changes, more signals
      * Conservative: 15-20 (slower, fewer signals)
      * Aggressive: 8-10 (faster, more signals)
    
    - **slow_period (26):** Defines overall trend sensitivity
      * Conservative: 30-40 (smoother, less noise)
      * Aggressive: 20-24 (more responsive)
    
    - **signal_period (9):** Smoothing for MACD line
      * Conservative: 12-15 (smoother signals)
      * Aggressive: 6-8 (faster signals)
    
    - **zero_line_filter (False):** Filter signals by MACD position
      * True: Only trade with the trend (fewer signals, higher quality)
      * False: Trade all crossovers (more signals, more risk)
    
    **Strategy Variations:**
    1. **Standard MACD (12/26/9):** Classic parameters, balanced approach
    2. **Fast MACD (8/17/9):** More responsive, day trading
    3. **Slow MACD (19/39/9):** Conservative, swing trading
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing OHLCV price data with DatetimeIndex.
        Must have at least (slow_period + signal_period) rows.
    
    column : str, default='Close'
        Name of the price column to use for MACD calculation.
        Common choices: 'Close', 'Adj Close'
    
    fast_period : int, default=12
        Period for fast EMA calculation. Must be >= 2 and < slow_period.
        Typical range: 8-15 days
    
    slow_period : int, default=26
        Period for slow EMA calculation. Must be >= 2 and > fast_period.
        Must be <= len(data) for sufficient data.
        Typical range: 20-40 days
    
    signal_period : int, default=9
        Period for signal line EMA (smoothing MACD). Must be >= 2.
        Typical range: 6-15 days
    
    zero_line_filter : bool, default=False
        If True, only generate BUY when MACD > 0, SELL when MACD < 0.
        Reduces false signals but may miss some opportunities.
    
    Returns:
    --------
    pd.Series
        Signal series with same index as input data:
        - 1: BUY signal (bullish MACD crossover)
        - -1: SELL signal (bearish MACD crossover)
        - 0: HOLD (no crossover)
        dtype: int8 for memory efficiency
        name: 'MACD_Trend_Signal_{fast}_{slow}_{signal}'
    
    Raises:
    -------
    ValueError
        - If data is not a pandas DataFrame
        - If data is empty
        - If fast_period or slow_period or signal_period < 2
        - If fast_period >= slow_period
        - If slow_period + signal_period > len(data)
        - If column not in DataFrame or non-numeric
    
    Examples:
    ---------
    >>> # Standard MACD strategy (12/26/9)
    >>> signals = macd_trend_following_strategy(df)
    >>> 
    >>> # Fast MACD for day trading (8/17/9)
    >>> signals = macd_trend_following_strategy(df, fast_period=8, slow_period=17)
    >>> 
    >>> # With zero-line filter (trade with trend only)
    >>> signals = macd_trend_following_strategy(df, zero_line_filter=True)
    >>> 
    >>> # Count signals
    >>> buy_signals = (signals == 1).sum()
    >>> sell_signals = (signals == -1).sum()
    
    Notes:
    ------
    - MACD works best in trending markets (up or down)
    - In sideways markets, consider using RSI Mean Reversion instead
    - Zero-line filter reduces signals but improves quality
    - Combine with volume analysis for confirmation
    - Divergences (price vs MACD) can signal reversals
    
    References:
    -----------
    - Appel, Gerald (1979). "The Moving Average Convergence-Divergence Trading Method"
    - Murphy, John J. (1999). "Technical Analysis of the Financial Markets"
    
    See Also:
    ---------
    golden_cross_strategy : SMA crossover signals (slower, long-term)
    rsi_mean_reversion_strategy : Threshold crossing signals (mean reversion)
    calculate_ema : Exponential moving average calculation
    """
    
    # =========================================================================
    # PARAMETER VALIDATION
    # =========================================================================
    
    logging.info(f"Generating MACD Trend signals with fast={fast_period}, slow={slow_period}, signal={signal_period}, zero_filter={zero_line_filter}")
    
    # Validate data type
    if not isinstance(data, pd.DataFrame):
        raise ValueError(
            f"Input 'data' must be a pandas DataFrame. "
            f"Got {type(data)}. "
            f"Ensure you pass a DataFrame with OHLCV columns."
        )
    
    # Validate data is not empty
    if data.empty:
        raise ValueError(
            "Input DataFrame is empty. Cannot generate signals on empty data. "
            "Please provide a DataFrame with at least one row of price data."
        )
    
    # Validate fast_period
    if fast_period < 2:
        raise ValueError(
            f"fast_period must be >= 2 for EMA calculation. Got: {fast_period}. "
            f"Common values: 8 (aggressive), 12 (standard), 15 (conservative)."
        )
    
    # Validate slow_period
    if slow_period < 2:
        raise ValueError(
            f"slow_period must be >= 2 for EMA calculation. Got: {slow_period}. "
            f"Common values: 20 (aggressive), 26 (standard), 30 (conservative)."
        )
    
    # Validate signal_period
    if signal_period < 2:
        raise ValueError(
            f"signal_period must be >= 2 for EMA calculation. Got: {signal_period}. "
            f"Common values: 6 (aggressive), 9 (standard), 12 (conservative)."
        )
    
    # Validate fast < slow
    if fast_period >= slow_period:
        raise ValueError(
            f"fast_period ({fast_period}) must be less than slow_period ({slow_period}). "
            f"MACD requires fast EMA to respond quicker than slow EMA. "
            f"Standard ratio: 12/26 or 8/17."
        )
    
    # Validate sufficient data length
    min_length = slow_period + signal_period
    if len(data) < min_length:
        raise ValueError(
            f"Insufficient data for MACD calculation. Need at least {min_length} rows "
            f"(slow_period={slow_period} + signal_period={signal_period}), got {len(data)}. "
            f"MACD requires data for slow EMA calculation plus signal line smoothing."
        )
    
    # Validate column exists
    if column not in data.columns:
        raise ValueError(
            f"Column '{column}' not found in DataFrame. "
            f"Available columns: {list(data.columns)}. "
            f"Common price columns: 'Close', 'Adj Close', 'Open', 'High', 'Low'."
        )
    
    # Validate column is numeric
    if not pd.api.types.is_numeric_dtype(data[column]):
        raise ValueError(
            f"Column '{column}' must contain numeric data for MACD calculation. "
            f"Got dtype: {data[column].dtype}. "
            f"Ensure price data is numeric (float or int)."
        )
    
    # =========================================================================
    # CALCULATE MACD COMPONENTS
    # =========================================================================
    
    logging.info(f"Calculating MACD components on column '{column}'")
    
    # Calculate fast and slow EMAs
    logging.info(f"Calculating EMA({fast_period}) and EMA({slow_period})")
    ema_fast = calculate_ema(data, column=column, period=fast_period)
    ema_slow = calculate_ema(data, column=column, period=slow_period)
    
    # Calculate MACD line (difference between fast and slow EMAs)
    macd_line = ema_fast - ema_slow
    logging.info(f"MACD line calculated: {macd_line.notna().sum()} valid values")
    
    # Calculate Signal line (EMA of MACD line)
    # Create temporary DataFrame for signal line calculation
    temp_df = pd.DataFrame({'MACD': macd_line}, index=data.index)
    signal_line = calculate_ema(temp_df, column='MACD', period=signal_period)
    logging.info(f"Signal line calculated: {signal_line.notna().sum()} valid values")
    
    # Calculate MACD histogram (for reference, not used in crossover detection)
    macd_histogram = macd_line - signal_line
    
    # Log MACD statistics
    valid_macd = macd_line.dropna()
    if len(valid_macd) > 0:
        logging.info(f"MACD statistics: min={valid_macd.min():.2f}, max={valid_macd.max():.2f}, mean={valid_macd.mean():.2f}")
    
    # =========================================================================
    # DETECT MACD CROSSOVERS
    # =========================================================================
    
    # Initialize signal series with 0 (HOLD)
    signals = pd.Series(
        index=data.index,
        data=0,
        dtype='int8',
        name=f'MACD_Trend_Signal_{fast_period}_{slow_period}_{signal_period}'
    )
    
    # Get previous day values for crossover detection
    macd_prev = macd_line.shift(1)
    signal_prev = signal_line.shift(1)
    
    # Detect bullish crossover (MACD crosses ABOVE signal)
    # Condition: previous MACD <= previous Signal AND current MACD > current Signal
    bullish_cross = (macd_prev <= signal_prev) & (macd_line > signal_line)
    
    # Detect bearish crossover (MACD crosses BELOW signal)
    # Condition: previous MACD >= previous Signal AND current MACD < current Signal
    bearish_cross = (macd_prev >= signal_prev) & (macd_line < signal_line)
    
    # Apply zero-line filter if enabled
    if zero_line_filter:
        # Only BUY when MACD > 0 (bullish territory)
        bullish_cross = bullish_cross & (macd_line > 0)
        # Only SELL when MACD < 0 (bearish territory)
        bearish_cross = bearish_cross & (macd_line < 0)
        logging.info("Zero-line filter applied: BUY only when MACD>0, SELL only when MACD<0")
    
    # Assign signals
    signals[bullish_cross] = 1   # BUY signal
    signals[bearish_cross] = -1  # SELL signal
    
    # =========================================================================
    # LOG SIGNAL STATISTICS
    # =========================================================================
    
    buy_count = (signals == 1).sum()
    sell_count = (signals == -1).sum()
    hold_count = (signals == 0).sum()
    
    logging.info(f"Signal generation complete:")
    logging.info(f"  - Buy signals (Bullish crossover): {buy_count}")
    logging.info(f"  - Sell signals (Bearish crossover): {sell_count}")
    logging.info(f"  - Hold signals: {hold_count}")
    logging.info(f"  - Total signals: {len(signals)}")
    
    # Log crossover details (first 5 of each type)
    if buy_count > 0:
        buy_dates = signals[signals == 1].index
        logging.info(f"  - Bullish crossovers: {buy_count} times")
        for i, date in enumerate(buy_dates[:5]):
            macd_val = macd_line.loc[date]
            signal_val = signal_line.loc[date]
            logging.info(f"    * {date.strftime('%Y-%m-%d')}: MACD={macd_val:.2f}, Signal={signal_val:.2f}")
        if buy_count > 5:
            logging.info(f"    * ... and {buy_count - 5} more bullish crossovers")
    
    if sell_count > 0:
        sell_dates = signals[signals == -1].index
        logging.info(f"  - Bearish crossovers: {sell_count} times")
        for i, date in enumerate(sell_dates[:5]):
            macd_val = macd_line.loc[date]
            signal_val = signal_line.loc[date]
            logging.info(f"    * {date.strftime('%Y-%m-%d')}: MACD={macd_val:.2f}, Signal={signal_val:.2f}")
        if sell_count > 5:
            logging.info(f"    * ... and {sell_count - 5} more bearish crossovers")
    
    # Log current MACD position
    if len(macd_line.dropna()) > 0:
        current_macd = macd_line.iloc[-1]
        current_signal = signal_line.iloc[-1]
        
        if pd.notna(current_macd) and pd.notna(current_signal):
            if current_macd > current_signal:
                position = f"BULLISH (MACD={current_macd:.2f} > Signal={current_signal:.2f})"
            else:
                position = f"BEARISH (MACD={current_macd:.2f} < Signal={current_signal:.2f})"
            
            if current_macd > 0:
                territory = "positive territory (above zero line)"
            else:
                territory = "negative territory (below zero line)"
            
            logging.info(f"  - Current position: {position}, {territory}")
    
    return signals