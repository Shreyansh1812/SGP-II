"""
Visualization Module for Algorithmic Trading Backtester

This module provides comprehensive visualization capabilities for backtesting results,
including price charts with technical indicators, trading signals, equity curves, 
drawdown analysis, and performance reports.

All functions use Plotly for interactive, web-based visualizations suitable for
Streamlit dashboards and Jupyter notebooks.

Author: Shreyansh Patel
Project: SGP-II - Python-Based Algorithmic Trading Backtester
Phase: 6 - Visualization Module
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, Dict, List, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Color scheme for consistent visualization
COLORS = {
    'price_up': '#26A69A',      # Green candlesticks (bullish)
    'price_down': '#EF5350',    # Red candlesticks (bearish)
    'buy_signal': '#00E676',    # Bright green markers
    'sell_signal': '#FF1744',   # Bright red markers
    'sma_fast': '#2196F3',      # Blue (faster moving average)
    'sma_slow': '#FF9800',      # Orange (slower moving average)
    'rsi': '#9C27B0',           # Purple (oscillator)
    'macd': '#00BCD4',          # Cyan (MACD line)
    'signal': '#FF5722',        # Deep orange (signal line)
    'histogram': '#78909C',     # Gray (MACD histogram)
    'bb_upper': '#F44336',      # Red (upper band)
    'bb_middle': '#2196F3',     # Blue (middle band)
    'bb_lower': '#4CAF50',      # Green (lower band)
    'equity': '#4CAF50',        # Green (growth)
    'drawdown': '#F44336',      # Red (risk)
    'volume': '#78909C',        # Gray (secondary info)
    'reference': '#9E9E9E'      # Gray (reference lines)
}


def plot_price_with_indicators(
    data: pd.DataFrame,
    sma: Optional[pd.Series] = None,
    ema: Optional[pd.Series] = None,
    rsi: Optional[pd.Series] = None,
    macd_line: Optional[pd.Series] = None,
    signal_line: Optional[pd.Series] = None,
    histogram: Optional[pd.Series] = None,
    bb_upper: Optional[pd.Series] = None,
    bb_middle: Optional[pd.Series] = None,
    bb_lower: Optional[pd.Series] = None,
    title: str = "Stock Price with Technical Indicators",
    height: int = 800
) -> go.Figure:
    """
    Plot stock price with technical indicators on multiple subplots.
    
    Creates an interactive candlestick chart with optional technical indicators:
    - Subplot 1 (60% height): Price candlesticks with SMA, EMA, Bollinger Bands
    - Subplot 2 (20% height): RSI with overbought/oversold threshold lines
    - Subplot 3 (20% height): Volume bars
    
    If MACD indicators are provided instead of RSI, the layout adjusts to show:
    - Subplot 2: MACD line, signal line, and histogram
    
    Parameters
    ----------
    data : pd.DataFrame
        OHLCV data with DatetimeIndex and columns: ['Open', 'High', 'Low', 'Close', 'Volume']
    sma : Optional[pd.Series], default=None
        Simple Moving Average series (typically slower period like 50 or 200)
    ema : Optional[pd.Series], default=None
        Exponential Moving Average series (typically faster period like 12 or 26)
    rsi : Optional[pd.Series], default=None
        Relative Strength Index series (0-100 range)
    macd_line : Optional[pd.Series], default=None
        MACD line (fast EMA - slow EMA)
    signal_line : Optional[pd.Series], default=None
        MACD signal line (EMA of MACD line)
    histogram : Optional[pd.Series], default=None
        MACD histogram (MACD line - signal line)
    bb_upper : Optional[pd.Series], default=None
        Bollinger Bands upper band
    bb_middle : Optional[pd.Series], default=None
        Bollinger Bands middle band (typically 20-period SMA)
    bb_lower : Optional[pd.Series], default=None
        Bollinger Bands lower band
    title : str, default="Stock Price with Technical Indicators"
        Chart title displayed at the top
    height : int, default=800
        Total chart height in pixels
        
    Returns
    -------
    go.Figure
        Plotly Figure object with interactive candlestick chart and indicators.
        Use fig.show() to display in browser or st.plotly_chart(fig) in Streamlit.
        
    Raises
    ------
    TypeError
        If data is not a pandas DataFrame
    ValueError
        If data is empty or missing required OHLCV columns
        If data index is not DatetimeIndex
        
    Examples
    --------
    Basic usage with SMA:
    
    >>> from src.data_loader import get_stock_data
    >>> from src.indicators import calculate_sma
    >>> from src.plotting import plot_price_with_indicators
    >>> 
    >>> data = get_stock_data("RELIANCE.NS", "2023-01-01", "2023-12-31")
    >>> sma_50 = calculate_sma(data, period=50)
    >>> sma_200 = calculate_sma(data, period=200)
    >>> 
    >>> fig = plot_price_with_indicators(data, sma=sma_50)
    >>> fig.show()  # Opens in browser
    
    With multiple indicators:
    
    >>> from src.indicators import calculate_rsi, calculate_bollinger_bands
    >>> 
    >>> rsi_14 = calculate_rsi(data, period=14)
    >>> bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(data, period=20)
    >>> 
    >>> fig = plot_price_with_indicators(
    ...     data, 
    ...     sma=sma_50,
    ...     rsi=rsi_14,
    ...     bb_upper=bb_upper,
    ...     bb_middle=bb_middle,
    ...     bb_lower=bb_lower,
    ...     title="RELIANCE.NS Technical Analysis"
    ... )
    >>> fig.show()
    
    With MACD indicators:
    
    >>> from src.indicators import calculate_macd
    >>> 
    >>> macd, signal, hist = calculate_macd(data)
    >>> 
    >>> fig = plot_price_with_indicators(
    ...     data,
    ...     macd_line=macd,
    ...     signal_line=signal,
    ...     histogram=hist
    ... )
    >>> fig.show()
    
    Notes
    -----
    - Candlestick colors: Green (bullish) when Close > Open, Red (bearish) when Close < Open
    - RSI thresholds: 70 (overbought) and 30 (oversold) marked with dashed lines
    - MACD zero line marked with dashed line
    - All indicators are optional - function adapts to provided indicators
    - Subplot heights: Price (60%), Indicator (20%), Volume (20%)
    - Interactive features: Zoom, pan, hover for exact values, toggle traces
    
    The function automatically validates input data and handles missing values in
    indicator series (common for the first N periods due to calculation requirements).
    """
    logger.info(f"Generating price chart with indicators: {title}")
    
    # ========== INPUT VALIDATION ==========
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"data must be pandas DataFrame, got {type(data).__name__}")
    
    if len(data) == 0:
        raise ValueError("data is empty, cannot plot")
    
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"data missing required columns: {missing_columns}")
    
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("data must have DatetimeIndex")
    
    # Determine subplot configuration based on indicators provided
    has_rsi = rsi is not None
    has_macd = macd_line is not None
    
    if has_rsi or has_macd:
        # 3 subplots: Price, Indicator, Volume
        subplot_titles = (title, "RSI" if has_rsi else "MACD", "Volume")
        row_heights = [0.6, 0.2, 0.2]
        specs = [[{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": False}]]
        rows = 3
    else:
        # 2 subplots: Price, Volume
        subplot_titles = (title, "Volume")
        row_heights = [0.8, 0.2]
        specs = [[{"secondary_y": False}], [{"secondary_y": False}]]
        rows = 2
    
    # Create figure with subplots
    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=subplot_titles,
        row_heights=row_heights,
        specs=specs
    )
    
    # ========== SUBPLOT 1: CANDLESTICK CHART ==========
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Price',
            increasing_line_color=COLORS['price_up'],
            decreasing_line_color=COLORS['price_down']
        ),
        row=1, col=1
    )
    
    # Add SMA if provided
    if sma is not None:
        fig.add_trace(
            go.Scatter(
                x=sma.index,
                y=sma,
                mode='lines',
                name=f'SMA({len(sma.dropna())})',
                line=dict(color=COLORS['sma_slow'], width=2)
            ),
            row=1, col=1
        )
        logger.info(f"Added SMA to chart (length: {len(sma.dropna())})")
    
    # Add EMA if provided
    if ema is not None:
        fig.add_trace(
            go.Scatter(
                x=ema.index,
                y=ema,
                mode='lines',
                name=f'EMA({len(ema.dropna())})',
                line=dict(color=COLORS['sma_fast'], width=2)
            ),
            row=1, col=1
        )
        logger.info(f"Added EMA to chart (length: {len(ema.dropna())})")
    
    # Add Bollinger Bands if provided
    if bb_upper is not None and bb_middle is not None and bb_lower is not None:
        # Upper band
        fig.add_trace(
            go.Scatter(
                x=bb_upper.index,
                y=bb_upper,
                mode='lines',
                name='BB Upper',
                line=dict(color=COLORS['bb_upper'], width=1, dash='dash')
            ),
            row=1, col=1
        )
        
        # Middle band
        fig.add_trace(
            go.Scatter(
                x=bb_middle.index,
                y=bb_middle,
                mode='lines',
                name='BB Middle',
                line=dict(color=COLORS['bb_middle'], width=1)
            ),
            row=1, col=1
        )
        
        # Lower band with fill
        fig.add_trace(
            go.Scatter(
                x=bb_lower.index,
                y=bb_lower,
                mode='lines',
                name='BB Lower',
                line=dict(color=COLORS['bb_lower'], width=1, dash='dash'),
                fill='tonexty',
                fillcolor='rgba(33, 150, 243, 0.1)'
            ),
            row=1, col=1
        )
        logger.info("Added Bollinger Bands to chart")
    
    # ========== SUBPLOT 2: RSI OR MACD ==========
    if has_rsi:
        # Add RSI line
        fig.add_trace(
            go.Scatter(
                x=rsi.index,
                y=rsi,
                mode='lines',
                name='RSI',
                line=dict(color=COLORS['rsi'], width=2)
            ),
            row=2, col=1
        )
        
        # Add overbought line (70)
        fig.add_hline(
            y=70,
            line_dash="dash",
            line_color=COLORS['reference'],
            annotation_text="Overbought (70)",
            annotation_position="right",
            row=2, col=1
        )
        
        # Add oversold line (30)
        fig.add_hline(
            y=30,
            line_dash="dash",
            line_color=COLORS['reference'],
            annotation_text="Oversold (30)",
            annotation_position="right",
            row=2, col=1
        )
        
        # Update RSI subplot y-axis
        fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
        logger.info("Added RSI subplot")
    
    elif has_macd:
        # Add MACD line
        fig.add_trace(
            go.Scatter(
                x=macd_line.index,
                y=macd_line,
                mode='lines',
                name='MACD',
                line=dict(color=COLORS['macd'], width=2)
            ),
            row=2, col=1
        )
        
        # Add signal line
        if signal_line is not None:
            fig.add_trace(
                go.Scatter(
                    x=signal_line.index,
                    y=signal_line,
                    mode='lines',
                    name='Signal',
                    line=dict(color=COLORS['signal'], width=2)
                ),
                row=2, col=1
            )
        
        # Add histogram
        if histogram is not None:
            colors = [COLORS['price_up'] if val >= 0 else COLORS['price_down'] for val in histogram]
            fig.add_trace(
                go.Bar(
                    x=histogram.index,
                    y=histogram,
                    name='Histogram',
                    marker_color=colors
                ),
                row=2, col=1
            )
        
        # Add zero line
        fig.add_hline(
            y=0,
            line_dash="dash",
            line_color=COLORS['reference'],
            row=2, col=1
        )
        
        # Update MACD subplot y-axis
        fig.update_yaxes(title_text="MACD", row=2, col=1)
        logger.info("Added MACD subplot")
    
    # ========== SUBPLOT 3 (or 2): VOLUME ==========
    volume_row = 3 if (has_rsi or has_macd) else 2
    
    # Color volume bars based on price movement
    volume_colors = [
        COLORS['price_up'] if data.loc[idx, 'Close'] >= data.loc[idx, 'Open'] 
        else COLORS['price_down'] 
        for idx in data.index
    ]
    
    fig.add_trace(
        go.Bar(
            x=data.index,
            y=data['Volume'],
            name='Volume',
            marker_color=volume_colors,
            showlegend=False
        ),
        row=volume_row, col=1
    )
    
    # Update volume subplot y-axis
    fig.update_yaxes(title_text="Volume", row=volume_row, col=1)
    logger.info("Added volume subplot")
    
    # ========== LAYOUT CONFIGURATION ==========
    fig.update_layout(
        height=height,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        hovermode='x unified',
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Update x-axis
    fig.update_xaxes(title_text="Date", row=volume_row, col=1)
    
    # Update price subplot y-axis
    fig.update_yaxes(title_text="Price", row=1, col=1)
    
    logger.info(f"Chart generated successfully with {len(data)} data points")
    
    return fig


def plot_signals(
    data: pd.DataFrame,
    signals: pd.Series,
    trades: Optional[List[Dict]] = None,
    title: str = "Trading Signals",
    height: int = 600
) -> go.Figure:
    """
    Plot stock price with BUY and SELL signal markers.
    
    Creates an interactive candlestick chart with:
    - Green upward triangles (▲) marking BUY signals
    - Red downward triangles (▼) marking SELL signals
    - Optional annotations showing entry/exit prices and returns for completed trades
    - Price range highlighting for holding periods
    
    This visualization helps validate trading strategy logic by showing:
    - Timing of entries and exits relative to price action
    - Whether signals occur at logical points (e.g., trend reversals)
    - Signal frequency (clustering indicates overtrading)
    - Trade outcomes (profitable exits marked differently)
    
    Parameters
    ----------
    data : pd.DataFrame
        OHLCV data with DatetimeIndex and columns: ['Open', 'High', 'Low', 'Close', 'Volume']
    signals : pd.Series
        Trading signals with same index as data, values in {-1, 0, 1}:
        - 1: BUY signal
        - -1: SELL signal
        - 0: HOLD (no action)
    trades : Optional[List[Dict]], default=None
        List of completed trade dictionaries from backtester, each containing:
        {
            'entry_date': str or Timestamp,
            'exit_date': str or Timestamp,
            'entry_price': float,
            'exit_price': float,
            'shares': int,
            'return_pct': float,
            'return_abs': float,
            'holding_days': int
        }
        If provided, adds annotations showing trade outcomes
    title : str, default="Trading Signals"
        Chart title displayed at the top
    height : int, default=600
        Chart height in pixels
        
    Returns
    -------
    go.Figure
        Plotly Figure object with candlestick chart and signal markers.
        Use fig.show() to display in browser or st.plotly_chart(fig) in Streamlit.
        
    Raises
    ------
    TypeError
        If data is not pandas DataFrame or signals is not pandas Series
    ValueError
        If data is empty, missing required columns, or index is not DatetimeIndex
        If signals contain values outside {-1, 0, 1}
        If signals length doesn't match data length
        
    Examples
    --------
    Basic usage with signals only:
    
    >>> from src.data_loader import get_stock_data
    >>> from src.strategy import golden_cross_strategy
    >>> from src.plotting import plot_signals
    >>> 
    >>> data = get_stock_data("RELIANCE.NS", "2023-01-01", "2023-12-31")
    >>> signals = golden_cross_strategy(data, fast_period=50, slow_period=200)
    >>> 
    >>> fig = plot_signals(data, signals, title="Golden Cross Signals")
    >>> fig.show()
    
    With completed trades from backtest:
    
    >>> from src.backtester import run_backtest
    >>> 
    >>> results = run_backtest(data, signals, initial_capital=100000)
    >>> 
    >>> fig = plot_signals(
    ...     data, 
    ...     signals, 
    ...     trades=results['trades'],
    ...     title="Golden Cross Strategy - Trades"
    ... )
    >>> fig.show()
    
    Notes
    -----
    - BUY signals: Green upward triangles (▲) positioned below candlesticks
    - SELL signals: Red downward triangles (▼) positioned above candlesticks
    - Trade annotations (if trades provided):
      * Entry: "BUY @ $X"
      * Exit: "SELL @ $Y (+Z%)" for profits or "SELL @ $Y (-Z%)" for losses
    - Holding periods: Shaded vertical regions between entry and exit
    - Signal validation: Function checks that signal values are valid {-1, 0, 1}
    - Index alignment: signals must have same index as data
    
    The function automatically filters out HOLD signals (0) and only displays
    BUY (1) and SELL (-1) markers on the chart.
    """
    logger.info(f"Generating signals chart: {title}")
    
    # ========== INPUT VALIDATION ==========
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"data must be pandas DataFrame, got {type(data).__name__}")
    
    if not isinstance(signals, pd.Series):
        raise TypeError(f"signals must be pandas Series, got {type(signals).__name__}")
    
    if len(data) == 0:
        raise ValueError("data is empty, cannot plot")
    
    required_columns = ['Open', 'High', 'Low', 'Close']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"data missing required columns: {missing_columns}")
    
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("data must have DatetimeIndex")
    
    if len(signals) != len(data):
        raise ValueError(f"signals length ({len(signals)}) must match data length ({len(data)})")
    
    # Check signal values are valid
    unique_signals = signals.dropna().unique()
    invalid_signals = [s for s in unique_signals if s not in [-1, 0, 1]]
    if invalid_signals:
        raise ValueError(f"signals contain invalid values: {invalid_signals}. Must be -1, 0, or 1")
    
    # Create figure
    fig = go.Figure()
    
    # ========== CANDLESTICK CHART ==========
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Price',
            increasing_line_color=COLORS['price_up'],
            decreasing_line_color=COLORS['price_down']
        )
    )
    
    # ========== BUY SIGNALS (1) ==========
    buy_signals = signals[signals == 1]
    if len(buy_signals) > 0:
        buy_prices = data.loc[buy_signals.index, 'Low'] * 0.995  # Position slightly below candle
        
        fig.add_trace(
            go.Scatter(
                x=buy_signals.index,
                y=buy_prices,
                mode='markers',
                name='BUY Signal',
                marker=dict(
                    symbol='triangle-up',
                    size=12,
                    color=COLORS['buy_signal'],
                    line=dict(width=1, color='white')
                )
            )
        )
        logger.info(f"Added {len(buy_signals)} BUY signals")
    
    # ========== SELL SIGNALS (-1) ==========
    sell_signals = signals[signals == -1]
    if len(sell_signals) > 0:
        sell_prices = data.loc[sell_signals.index, 'High'] * 1.005  # Position slightly above candle
        
        fig.add_trace(
            go.Scatter(
                x=sell_signals.index,
                y=sell_prices,
                mode='markers',
                name='SELL Signal',
                marker=dict(
                    symbol='triangle-down',
                    size=12,
                    color=COLORS['sell_signal'],
                    line=dict(width=1, color='white')
                )
            )
        )
        logger.info(f"Added {len(sell_signals)} SELL signals")
    
    # ========== TRADE ANNOTATIONS (if provided) ==========
    if trades is not None and len(trades) > 0:
        for i, trade in enumerate(trades):
            entry_date = pd.to_datetime(trade['entry_date'])
            exit_date = pd.to_datetime(trade['exit_date'])
            entry_price = trade['entry_price']
            exit_price = trade['exit_price']
            return_pct = trade['return_pct']
            
            # Entry annotation
            fig.add_annotation(
                x=entry_date,
                y=entry_price,
                text=f"BUY<br>${entry_price:.2f}",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor=COLORS['buy_signal'],
                ax=0,
                ay=40,
                font=dict(size=10, color=COLORS['buy_signal']),
                bgcolor='rgba(255, 255, 255, 0.8)',
                bordercolor=COLORS['buy_signal'],
                borderwidth=1
            )
            
            # Exit annotation with return
            exit_color = COLORS['price_up'] if return_pct >= 0 else COLORS['price_down']
            return_sign = '+' if return_pct >= 0 else ''
            
            fig.add_annotation(
                x=exit_date,
                y=exit_price,
                text=f"SELL<br>${exit_price:.2f}<br>({return_sign}{return_pct:.1f}%)",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor=exit_color,
                ax=0,
                ay=-40,
                font=dict(size=10, color=exit_color),
                bgcolor='rgba(255, 255, 255, 0.8)',
                bordercolor=exit_color,
                borderwidth=1
            )
            
            # Holding period shading
            fig.add_vrect(
                x0=entry_date,
                x1=exit_date,
                fillcolor=COLORS['buy_signal'] if return_pct >= 0 else COLORS['sell_signal'],
                opacity=0.05,
                layer="below",
                line_width=0
            )
        
        logger.info(f"Added annotations for {len(trades)} completed trades")
    
    # ========== LAYOUT CONFIGURATION ==========
    fig.update_layout(
        title=title,
        height=height,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        hovermode='x unified',
        template='plotly_white',
        xaxis_title="Date",
        yaxis_title="Price",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    logger.info(f"Signals chart generated successfully")
    
    return fig


def plot_equity_curve(
    equity_curve: pd.Series,
    initial_capital: float,
    title: str = "Portfolio Equity Curve",
    height: int = 500
) -> go.Figure:
    """
    Plot portfolio equity (value) over time.
    
    The equity curve is THE MOST IMPORTANT visualization in backtesting, showing:
    - Portfolio value evolution (cash + position value)
    - Whether the strategy made money (upward slope = profit)
    - Return consistency (smooth curve = stable strategy)
    - Risk exposure (volatility = curve fluctuations)
    - Drawdown periods (dips below running maximum)
    
    A good equity curve:
    - Trends upward consistently (positive returns)
    - Has smooth progression (low volatility)
    - Recovers quickly from drawdowns (resilience)
    - Ends significantly above starting capital (profitability)
    
    A problematic equity curve:
    - Flat or downward trend (no profit or losses)
    - High volatility (excessive risk)
    - Prolonged drawdowns (poor recovery)
    - Large gaps between peaks (inconsistent performance)
    
    Parameters
    ----------
    equity_curve : pd.Series
        Portfolio equity value over time with DatetimeIndex.
        Each value represents total portfolio value (cash + position value) at that date.
        Typically obtained from backtester results['equity_curve']
    initial_capital : float
        Starting capital amount for reference line.
        Used to show break-even point and calculate returns.
    title : str, default="Portfolio Equity Curve"
        Chart title displayed at the top
    height : int, default=500
        Chart height in pixels
        
    Returns
    -------
    go.Figure
        Plotly Figure object with equity curve and reference lines.
        Use fig.show() to display in browser or st.plotly_chart(fig) in Streamlit.
        
    Raises
    ------
    TypeError
        If equity_curve is not pandas Series
    ValueError
        If equity_curve is empty or contains negative values
        If equity_curve index is not DatetimeIndex
        If initial_capital is not positive
        
    Examples
    --------
    Basic usage from backtest results:
    
    >>> from src.backtester import run_backtest
    >>> from src.plotting import plot_equity_curve
    >>> 
    >>> results = run_backtest(data, signals, initial_capital=100000)
    >>> 
    >>> fig = plot_equity_curve(
    ...     results['equity_curve'],
    ...     initial_capital=100000,
    ...     title="Golden Cross Strategy - Equity"
    ... )
    >>> fig.show()
    
    Comparing multiple strategies:
    
    >>> results1 = run_backtest(data, signals1, initial_capital=100000)
    >>> results2 = run_backtest(data, signals2, initial_capital=100000)
    >>> 
    >>> fig = go.Figure()
    >>> fig.add_trace(go.Scatter(x=results1['equity_curve'].index, 
    ...                          y=results1['equity_curve'], 
    ...                          name='Strategy 1'))
    >>> fig.add_trace(go.Scatter(x=results2['equity_curve'].index, 
    ...                          y=results2['equity_curve'], 
    ...                          name='Strategy 2'))
    >>> fig.show()
    
    Notes
    -----
    - Green line: Equity curve (portfolio value over time)
    - Gray dashed line: Initial capital (break-even reference)
    - Final value annotation: Shows ending portfolio value and total return
    - Running maximum shading: Area between equity and peak shows drawdowns
    - Slope interpretation: Steeper = higher returns, flatter = lower returns
    - Volatility: Smooth curve = stable, jagged = volatile
    
    The function calculates the running maximum equity to show drawdown periods
    as shaded regions between the equity curve and previous peaks.
    """
    logger.info(f"Generating equity curve chart: {title}")
    
    # ========== INPUT VALIDATION ==========
    if not isinstance(equity_curve, pd.Series):
        raise TypeError(f"equity_curve must be pandas Series, got {type(equity_curve).__name__}")
    
    if len(equity_curve) == 0:
        raise ValueError("equity_curve is empty, cannot plot")
    
    if not isinstance(equity_curve.index, pd.DatetimeIndex):
        raise ValueError("equity_curve must have DatetimeIndex")
    
    if initial_capital <= 0:
        raise ValueError(f"initial_capital must be positive, got {initial_capital}")
    
    if (equity_curve < 0).any():
        raise ValueError("equity_curve contains negative values (invalid portfolio state)")
    
    # Create figure
    fig = go.Figure()
    
    # ========== EQUITY LINE ==========
    fig.add_trace(
        go.Scatter(
            x=equity_curve.index,
            y=equity_curve,
            mode='lines',
            name='Portfolio Equity',
            line=dict(color=COLORS['equity'], width=2),
            fill='tonexty',
            fillcolor='rgba(76, 175, 80, 0.1)'
        )
    )
    
    # ========== INITIAL CAPITAL REFERENCE LINE ==========
    fig.add_hline(
        y=initial_capital,
        line_dash="dash",
        line_color=COLORS['reference'],
        annotation_text=f"Initial Capital: ${initial_capital:,.0f}",
        annotation_position="left"
    )
    
    # ========== FINAL VALUE ANNOTATION ==========
    final_equity = equity_curve.iloc[-1]
    total_return = ((final_equity - initial_capital) / initial_capital) * 100
    return_sign = '+' if total_return >= 0 else ''
    
    fig.add_annotation(
        x=equity_curve.index[-1],
        y=final_equity,
        text=f"Final: ${final_equity:,.0f}<br>({return_sign}{total_return:.2f}%)",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor=COLORS['equity'] if total_return >= 0 else COLORS['drawdown'],
        ax=60,
        ay=-40,
        font=dict(size=12, color=COLORS['equity'] if total_return >= 0 else COLORS['drawdown']),
        bgcolor='rgba(255, 255, 255, 0.9)',
        bordercolor=COLORS['equity'] if total_return >= 0 else COLORS['drawdown'],
        borderwidth=2
    )
    
    # ========== RUNNING MAXIMUM (for drawdown shading) ==========
    running_max = equity_curve.expanding().max()
    
    # Add running maximum line (subtle)
    fig.add_trace(
        go.Scatter(
            x=running_max.index,
            y=running_max,
            mode='lines',
            name='Peak Equity',
            line=dict(color=COLORS['reference'], width=1, dash='dot'),
            showlegend=True
        )
    )
    
    # ========== LAYOUT CONFIGURATION ==========
    fig.update_layout(
        title=title,
        height=height,
        showlegend=True,
        hovermode='x unified',
        template='plotly_white',
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    logger.info(f"Equity curve chart generated: Initial=${initial_capital:,.0f}, Final=${final_equity:,.0f}, Return={return_sign}{total_return:.2f}%")
    
    return fig


def plot_drawdown(
    equity_curve: pd.Series,
    title: str = "Drawdown Analysis",
    height: int = 400
) -> go.Figure:
    """
    Plot drawdown percentage over time.
    
    Drawdown shows the portfolio's decline from previous peak, measuring:
    - Risk exposure (maximum drawdown = worst loss from peak)
    - Recovery ability (time to return to previous peak)
    - Psychological pain (how much loss traders must endure)
    - Strategy robustness (frequent/deep drawdowns = problematic)
    
    Drawdown Formula:
        Drawdown(t) = (Equity(t) - Running_Max(t)) / Running_Max(t) × 100
    
    Drawdown is always negative or zero:
    - 0%: At new peak (no drawdown)
    - -10%: Portfolio down 10% from recent peak
    - -30%: Portfolio down 30% from recent peak (severe)
    
    Interpretation Guidelines:
    - Max Drawdown < -15%: Acceptable risk for most traders
    - Max Drawdown -15% to -30%: Moderate risk, requires risk tolerance
    - Max Drawdown > -30%: High risk, position sizing needs adjustment
    - Drawdown duration: Time underwater (below previous peak)
    
    Parameters
    ----------
    equity_curve : pd.Series
        Portfolio equity value over time with DatetimeIndex.
        Each value represents total portfolio value (cash + position value).
        Typically obtained from backtester results['equity_curve']
    title : str, default="Drawdown Analysis"
        Chart title displayed at the top
    height : int, default=400
        Chart height in pixels
        
    Returns
    -------
    go.Figure
        Plotly Figure object with drawdown chart.
        Use fig.show() to display in browser or st.plotly_chart(fig) in Streamlit.
        
    Raises
    ------
    TypeError
        If equity_curve is not pandas Series
    ValueError
        If equity_curve is empty or contains negative values
        If equity_curve index is not DatetimeIndex
        
    Examples
    --------
    Basic usage from backtest results:
    
    >>> from src.backtester import run_backtest
    >>> from src.plotting import plot_drawdown
    >>> 
    >>> results = run_backtest(data, signals, initial_capital=100000)
    >>> 
    >>> fig = plot_drawdown(
    ...     results['equity_curve'],
    ...     title="Golden Cross Strategy - Drawdown"
    ... )
    >>> fig.show()
    
    Analyzing drawdown statistics:
    
    >>> equity = results['equity_curve']
    >>> running_max = equity.expanding().max()
    >>> drawdown = (equity - running_max) / running_max * 100
    >>> 
    >>> print(f"Max Drawdown: {drawdown.min():.2f}%")
    >>> print(f"Avg Drawdown: {drawdown[drawdown < 0].mean():.2f}%")
    >>> print(f"Days Underwater: {(drawdown < 0).sum()}")
    
    Notes
    -----
    - Red shaded area: Drawdown magnitude (distance below peak)
    - Deeper red = larger drawdown (more risk)
    - Maximum drawdown marker: Worst point in strategy history
    - Zero line: At peak equity (no drawdown)
    - Underwater periods: Continuous time below previous peak
    - Recovery: Return to zero line (new peak achieved)
    
    The function automatically calculates running maximum equity and drawdown
    percentage at each time point, highlighting the maximum drawdown event.
    """
    logger.info(f"Generating drawdown chart: {title}")
    
    # ========== INPUT VALIDATION ==========
    if not isinstance(equity_curve, pd.Series):
        raise TypeError(f"equity_curve must be pandas Series, got {type(equity_curve).__name__}")
    
    if len(equity_curve) == 0:
        raise ValueError("equity_curve is empty, cannot plot")
    
    if not isinstance(equity_curve.index, pd.DatetimeIndex):
        raise ValueError("equity_curve must have DatetimeIndex")
    
    if (equity_curve < 0).any():
        raise ValueError("equity_curve contains negative values (invalid portfolio state)")
    
    # ========== CALCULATE DRAWDOWN ==========
    running_max = equity_curve.expanding().max()
    drawdown = (equity_curve - running_max) / running_max * 100  # Percentage
    
    # Find maximum drawdown point
    max_dd_idx = drawdown.idxmin()
    max_dd_value = drawdown.min()
    
    # Create figure
    fig = go.Figure()
    
    # ========== DRAWDOWN AREA CHART ==========
    fig.add_trace(
        go.Scatter(
            x=drawdown.index,
            y=drawdown,
            mode='lines',
            name='Drawdown',
            line=dict(color=COLORS['drawdown'], width=2),
            fill='tozeroy',
            fillcolor='rgba(244, 67, 54, 0.2)'
        )
    )
    
    # ========== ZERO LINE (at peak) ==========
    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color=COLORS['reference'],
        annotation_text="Peak Equity (0% Drawdown)",
        annotation_position="right"
    )
    
    # ========== MAXIMUM DRAWDOWN MARKER ==========
    fig.add_annotation(
        x=max_dd_idx,
        y=max_dd_value,
        text=f"Max Drawdown<br>{max_dd_value:.2f}%",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor=COLORS['drawdown'],
        ax=0,
        ay=40,
        font=dict(size=12, color=COLORS['drawdown']),
        bgcolor='rgba(255, 255, 255, 0.9)',
        bordercolor=COLORS['drawdown'],
        borderwidth=2
    )
    
    # Add marker point
    fig.add_trace(
        go.Scatter(
            x=[max_dd_idx],
            y=[max_dd_value],
            mode='markers',
            name='Max Drawdown',
            marker=dict(
                size=10,
                color=COLORS['drawdown'],
                line=dict(width=2, color='white')
            ),
            showlegend=False
        )
    )
    
    # ========== LAYOUT CONFIGURATION ==========
    fig.update_layout(
        title=title,
        height=height,
        showlegend=True,
        hovermode='x unified',
        template='plotly_white',
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Set y-axis range to show drawdowns clearly
    y_min = min(max_dd_value * 1.2, -5)  # At least -5% range
    fig.update_yaxes(range=[y_min, 5])  # Slight positive range for visibility
    
    logger.info(f"Drawdown chart generated: Max Drawdown={max_dd_value:.2f}% at {max_dd_idx.date()}")
    
    return fig


def plot_returns_distribution(
    trades: List[Dict],
    title: str = "Trade Returns Distribution",
    height: int = 500
) -> go.Figure:
    """
    Plot histogram of individual trade returns.
    
    Returns distribution reveals strategy characteristics:
    - Symmetric distribution: Balanced wins and losses (mean reversion)
    - Right-skewed: Many small losses, few large wins (trend following)
    - Left-skewed: Many small wins, few large losses (option selling, picking pennies)
    - Fat tails: Occasional extreme returns (lottery tickets or black swans)
    - Tight distribution: Consistent returns (low variance)
    - Wide distribution: Variable returns (high variance)
    
    Statistical Insights:
    - Mean > 0: Profitable strategy on average
    - Median > Mean: Skewed by large losses (few big losers drag average down)
    - Median < Mean: Skewed by large wins (few big winners pull average up)
    - Standard deviation: Return variability (risk per trade)
    
    Parameters
    ----------
    trades : List[Dict]
        List of completed trade dictionaries from backtester, each containing:
        {
            'entry_date': str or Timestamp,
            'exit_date': str or Timestamp,
            'entry_price': float,
            'exit_price': float,
            'shares': int,
            'return_pct': float,  ← Used for distribution
            'return_abs': float,
            'holding_days': int
        }
    title : str, default="Trade Returns Distribution"
        Chart title displayed at the top
    height : int, default=500
        Chart height in pixels
        
    Returns
    -------
    go.Figure
        Plotly Figure object with histogram and statistics.
        Use fig.show() to display in browser or st.plotly_chart(fig) in Streamlit.
        
    Raises
    ------
    TypeError
        If trades is not a list
    ValueError
        If trades list is empty or trades missing 'return_pct' key
        
    Examples
    --------
    Basic usage from backtest results:
    
    >>> from src.backtester import run_backtest
    >>> from src.plotting import plot_returns_distribution
    >>> 
    >>> results = run_backtest(data, signals, initial_capital=100000)
    >>> 
    >>> fig = plot_returns_distribution(
    ...     results['trades'],
    ...     title="Golden Cross Strategy - Returns"
    ... )
    >>> fig.show()
    
    Analyzing distribution statistics:
    
    >>> returns = [trade['return_pct'] for trade in results['trades']]
    >>> print(f"Mean Return: {np.mean(returns):.2f}%")
    >>> print(f"Median Return: {np.median(returns):.2f}%")
    >>> print(f"Std Dev: {np.std(returns):.2f}%")
    >>> print(f"Skewness: {pd.Series(returns).skew():.2f}")
    
    Notes
    -----
    - Histogram bars: Frequency of trades in each return range
    - Green bars (right): Profitable trades (positive returns)
    - Red bars (left): Losing trades (negative returns)
    - Mean line (blue): Average trade return
    - Median line (orange): Middle trade return
    - Bin width: Automatically calculated based on return range
    - Normal curve overlay: Shows expected distribution for comparison
    
    The function separates winning and losing trades with different colors
    and provides statistical annotations showing mean, median, and standard deviation.
    """
    logger.info(f"Generating returns distribution chart: {title}")
    
    # ========== INPUT VALIDATION ==========
    if not isinstance(trades, list):
        raise TypeError(f"trades must be list, got {type(trades).__name__}")
    
    if len(trades) == 0:
        raise ValueError("trades list is empty, cannot plot")
    
    # Extract returns
    try:
        returns = [trade['return_pct'] for trade in trades]
    except KeyError as e:
        raise ValueError(f"trades missing required key: {e}")
    
    returns_array = np.array(returns)
    
    # Calculate statistics
    mean_return = np.mean(returns_array)
    median_return = np.median(returns_array)
    std_return = np.std(returns_array)
    
    # Separate wins and losses
    wins = returns_array[returns_array >= 0]
    losses = returns_array[returns_array < 0]
    
    # Create figure
    fig = go.Figure()
    
    # ========== HISTOGRAM: LOSSES (red) ==========
    if len(losses) > 0:
        fig.add_trace(
            go.Histogram(
                x=losses,
                name='Losses',
                marker_color=COLORS['price_down'],
                opacity=0.7,
                xbins=dict(size=(returns_array.max() - returns_array.min()) / 20)
            )
        )
    
    # ========== HISTOGRAM: WINS (green) ==========
    if len(wins) > 0:
        fig.add_trace(
            go.Histogram(
                x=wins,
                name='Wins',
                marker_color=COLORS['price_up'],
                opacity=0.7,
                xbins=dict(size=(returns_array.max() - returns_array.min()) / 20)
            )
        )
    
    # ========== MEAN LINE ==========
    fig.add_vline(
        x=mean_return,
        line_dash="dash",
        line_color=COLORS['sma_fast'],
        annotation_text=f"Mean: {mean_return:.2f}%",
        annotation_position="top"
    )
    
    # ========== MEDIAN LINE ==========
    fig.add_vline(
        x=median_return,
        line_dash="dot",
        line_color=COLORS['sma_slow'],
        annotation_text=f"Median: {median_return:.2f}%",
        annotation_position="top"
    )
    
    # ========== STATISTICS ANNOTATION ==========
    stats_text = (
        f"<b>Statistics:</b><br>"
        f"Mean: {mean_return:.2f}%<br>"
        f"Median: {median_return:.2f}%<br>"
        f"Std Dev: {std_return:.2f}%<br>"
        f"Total Trades: {len(returns)}<br>"
        f"Wins: {len(wins)} ({len(wins)/len(returns)*100:.1f}%)<br>"
        f"Losses: {len(losses)} ({len(losses)/len(returns)*100:.1f}%)"
    )
    
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.98, y=0.98,
        text=stats_text,
        showarrow=False,
        font=dict(size=10),
        bgcolor='rgba(255, 255, 255, 0.9)',
        bordercolor=COLORS['reference'],
        borderwidth=1,
        xanchor='right',
        yanchor='top'
    )
    
    # ========== LAYOUT CONFIGURATION ==========
    fig.update_layout(
        title=title,
        height=height,
        showlegend=True,
        barmode='overlay',
        template='plotly_white',
        xaxis_title="Return (%)",
        yaxis_title="Frequency (Number of Trades)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    logger.info(f"Returns distribution generated: Mean={mean_return:.2f}%, Median={median_return:.2f}%, Std={std_return:.2f}%, Trades={len(returns)}")
    
    return fig


def plot_monthly_returns(
    equity_curve: pd.Series,
    title: str = "Monthly Returns Heatmap",
    height: int = 400
) -> go.Figure:
    """
    Plot calendar heatmap of monthly returns.
    
    Monthly returns heatmap reveals:
    - Seasonal patterns (e.g., "January effect", "September curse")
    - Strategy consistency (mostly green = reliable)
    - Bad periods (red clusters = regime failures)
    - Year-over-year performance comparison
    - Monthly performance distribution
    
    Color Interpretation:
    - Dark green: Strong positive monthly return (>5%)
    - Light green: Moderate positive return (1-5%)
    - White: Flat month (~0%)
    - Light red: Moderate negative return (-1% to -5%)
    - Dark red: Strong negative return (<-5%)
    
    Pattern Analysis:
    - All green: Robust strategy across market conditions
    - Mixed colors: Normal variance, acceptable
    - Red clusters: Strategy fails in certain regimes
    - Seasonal pattern: Strategy exploits calendar effects
    
    Parameters
    ----------
    equity_curve : pd.Series
        Portfolio equity value over time with DatetimeIndex.
        Monthly returns are calculated from this equity progression.
        Typically obtained from backtester results['equity_curve']
    title : str, default="Monthly Returns Heatmap"
        Chart title displayed at the top
    height : int, default=400
        Chart height in pixels
        
    Returns
    -------
    go.Figure
        Plotly Figure object with heatmap chart.
        Use fig.show() to display in browser or st.plotly_chart(fig) in Streamlit.
        
    Raises
    ------
    TypeError
        If equity_curve is not pandas Series
    ValueError
        If equity_curve is empty or doesn't span multiple months
        If equity_curve index is not DatetimeIndex
        
    Examples
    --------
    Basic usage from backtest results:
    
    >>> from src.backtester import run_backtest
    >>> from src.plotting import plot_monthly_returns
    >>> 
    >>> results = run_backtest(data, signals, initial_capital=100000)
    >>> 
    >>> fig = plot_monthly_returns(
    ...     results['equity_curve'],
    ...     title="Golden Cross Strategy - Monthly Performance"
    ... )
    >>> fig.show()
    
    Analyzing best/worst months:
    
    >>> monthly_returns = equity_curve.resample('ME').last().pct_change() * 100
    >>> print(f"Best Month: {monthly_returns.max():.2f}%")
    >>> print(f"Worst Month: {monthly_returns.min():.2f}%")
    >>> print(f"Avg Monthly Return: {monthly_returns.mean():.2f}%")
    
    Notes
    -----
    - Rows: Years (2023, 2024, etc.)
    - Columns: Months (Jan through Dec)
    - Cell color: Monthly return percentage
    - Rightmost column: Annual return for each year
    - Bottom row: Average return for each month across all years
    - Colorscale: Red (negative) → White (zero) → Green (positive)
    
    The function resamples equity curve to month-end values and calculates
    percentage returns between consecutive months, handling missing data gracefully.
    """
    logger.info(f"Generating monthly returns heatmap: {title}")
    
    # ========== INPUT VALIDATION ==========
    if not isinstance(equity_curve, pd.Series):
        raise TypeError(f"equity_curve must be pandas Series, got {type(equity_curve).__name__}")
    
    if len(equity_curve) == 0:
        raise ValueError("equity_curve is empty, cannot plot")
    
    if not isinstance(equity_curve.index, pd.DatetimeIndex):
        raise ValueError("equity_curve must have DatetimeIndex")
    
    # ========== CALCULATE MONTHLY RETURNS ==========
    # Resample to month-end values
    monthly_equity = equity_curve.resample('ME').last()
    
    if len(monthly_equity) < 2:
        raise ValueError("equity_curve must span at least 2 months")
    
    # Calculate monthly percentage returns
    monthly_returns = monthly_equity.pct_change() * 100
    monthly_returns = monthly_returns.dropna()
    
    # Create DataFrame with Year and Month columns
    returns_df = pd.DataFrame({
        'Year': monthly_returns.index.year,
        'Month': monthly_returns.index.month,
        'Return': monthly_returns.values
    })
    
    # Pivot to create heatmap structure (years as rows, months as columns)
    heatmap_data = returns_df.pivot(index='Year', columns='Month', values='Return')
    
    # Add annual returns column (sum of monthly returns)
    heatmap_data['Annual'] = returns_df.groupby('Year')['Return'].sum()
    
    # Month names for x-axis
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Annual']
    
    # Ensure all months are present (fill missing with NaN)
    for month in range(1, 13):
        if month not in heatmap_data.columns:
            heatmap_data[month] = np.nan
    
    # Reorder columns: 1-12, Annual
    heatmap_data = heatmap_data[[*range(1, 13), 'Annual']]
    
    # Create text labels (formatted percentages)
    text_labels = [[f"{val:.1f}%" if not np.isnan(val) else "" 
                    for val in row] 
                   for row in heatmap_data.values]
    
    # Create figure
    fig = go.Figure()
    
    # ========== HEATMAP ==========
    fig.add_trace(
        go.Heatmap(
            z=heatmap_data.values,
            x=month_names,
            y=[str(year) for year in heatmap_data.index],
            text=text_labels,
            texttemplate='%{text}',
            textfont=dict(size=10),
            colorscale=[
                [0, '#EF5350'],      # Dark red (large losses)
                [0.4, '#FFCDD2'],    # Light red (small losses)
                [0.5, '#FFFFFF'],    # White (break-even)
                [0.6, '#C8E6C9'],    # Light green (small gains)
                [1, '#4CAF50']       # Dark green (large gains)
            ],
            zmid=0,  # Center colorscale at zero
            colorbar=dict(
                title="Return (%)",
                titleside="right"
            ),
            hovertemplate='%{y} %{x}<br>Return: %{z:.2f}%<extra></extra>'
        )
    )
    
    # ========== LAYOUT CONFIGURATION ==========
    fig.update_layout(
        title=title,
        height=height,
        template='plotly_white',
        xaxis_title="Month",
        yaxis_title="Year",
        xaxis=dict(side='bottom'),
        yaxis=dict(autorange='reversed')  # Most recent year at top
    )
    
    logger.info(f"Monthly returns heatmap generated: {len(heatmap_data)} years, {len(month_names)-1} months")
    
    return fig


def create_backtest_report(
    data: pd.DataFrame,
    signals: pd.Series,
    backtest_results: Dict,
    indicators: Optional[Dict[str, pd.Series]] = None,
    strategy_name: str = "Trading Strategy"
) -> Dict[str, go.Figure]:
    """
    Generate complete visual report for backtesting results.
    
    This master function creates all visualization charts in one call, providing:
    1. Price with indicators (if provided)
    2. Trading signals with trade markers
    3. Portfolio equity curve
    4. Drawdown analysis
    5. Trade returns distribution
    6. Monthly returns heatmap
    
    The function orchestrates all individual plotting functions and returns
    a dictionary of Plotly figures that can be displayed, saved, or integrated
    into dashboards (e.g., Streamlit).
    
    Parameters
    ----------
    data : pd.DataFrame
        OHLCV data with DatetimeIndex and columns: ['Open', 'High', 'Low', 'Close', 'Volume']
    signals : pd.Series
        Trading signals with same index as data, values in {-1, 0, 1}:
        - 1: BUY signal
        - -1: SELL signal
        - 0: HOLD (no action)
    backtest_results : Dict
        Dictionary returned by backtester's run_backtest() function containing:
        {
            'trades': List[Dict],            # Completed trades
            'equity_curve': pd.Series,       # Portfolio value over time
            'daily_positions': pd.Series,    # Position state each day
            'metrics': Dict                  # Performance metrics
        }
    indicators : Optional[Dict[str, pd.Series]], default=None
        Dictionary of indicator names to Series for plotting:
        {
            'SMA_50': sma_50_series,
            'RSI_14': rsi_14_series,
            'MACD': macd_line_series,
            'Signal': signal_line_series,
            'Histogram': histogram_series,
            'BB_Upper': bb_upper_series,
            'BB_Middle': bb_middle_series,
            'BB_Lower': bb_lower_series,
            ...
        }
        If None, price chart is generated without indicators
    strategy_name : str, default="Trading Strategy"
        Name of the strategy for chart titles
        
    Returns
    -------
    Dict[str, go.Figure]
        Dictionary of figure names to Plotly Figure objects:
        {
            'price_indicators': go.Figure,    # Price with technical indicators
            'signals': go.Figure,             # Trading signals on price chart
            'equity_curve': go.Figure,        # Portfolio equity progression
            'drawdown': go.Figure,            # Drawdown analysis
            'returns_distribution': go.Figure, # Histogram of trade returns
            'monthly_returns': go.Figure      # Calendar heatmap
        }
        
    Raises
    ------
    TypeError
        If data is not DataFrame, signals is not Series, or backtest_results is not Dict
    ValueError
        If data is empty or missing required columns
        If backtest_results missing required keys
        
    Examples
    --------
    Complete workflow with all visualizations:
    
    >>> from src.data_loader import get_stock_data
    >>> from src.indicators import calculate_sma, calculate_rsi
    >>> from src.strategy import golden_cross_strategy
    >>> from src.backtester import run_backtest
    >>> from src.plotting import create_backtest_report
    >>> 
    >>> # 1. Load data
    >>> data = get_stock_data("RELIANCE.NS", "2023-01-01", "2023-12-31")
    >>> 
    >>> # 2. Calculate indicators
    >>> sma_50 = calculate_sma(data, period=50)
    >>> sma_200 = calculate_sma(data, period=200)
    >>> rsi_14 = calculate_rsi(data, period=14)
    >>> 
    >>> # 3. Generate signals
    >>> signals = golden_cross_strategy(data, fast_period=50, slow_period=200)
    >>> 
    >>> # 4. Run backtest
    >>> results = run_backtest(data, signals, initial_capital=100000)
    >>> 
    >>> # 5. Generate complete visual report
    >>> indicators_dict = {
    ...     'SMA_50': sma_50,
    ...     'SMA_200': sma_200,
    ...     'RSI': rsi_14
    ... }
    >>> 
    >>> figures = create_backtest_report(
    ...     data=data,
    ...     signals=signals,
    ...     backtest_results=results,
    ...     indicators=indicators_dict,
    ...     strategy_name="Golden Cross (50/200)"
    ... )
    >>> 
    >>> # Display all charts
    >>> for chart_name, fig in figures.items():
    ...     print(f"Displaying: {chart_name}")
    ...     fig.show()
    >>> 
    >>> # Or save to files
    >>> figures['equity_curve'].write_html("equity_curve.html")
    >>> figures['signals'].write_image("signals.png")
    
    Streamlit dashboard integration:
    
    >>> import streamlit as st
    >>> 
    >>> st.title(f"{strategy_name} - Backtest Report")
    >>> 
    >>> # Display all charts
    >>> st.subplot("Price & Indicators")
    >>> st.plotly_chart(figures['price_indicators'], use_container_width=True)
    >>> 
    >>> st.subplotlot("Trading Signals")
    >>> st.plotly_chart(figures['signals'], use_container_width=True)
    >>> 
    >>> col1, col2 = st.columns(2)
    >>> with col1:
    ...     st.plotly_chart(figures['equity_curve'], use_container_width=True)
    >>> with col2:
    ...     st.plotly_chart(figures['drawdown'], use_container_width=True)
    >>> 
    >>> st.plotly_chart(figures['returns_distribution'], use_container_width=True)
    >>> st.plotly_chart(figures['monthly_returns'], use_container_width=True)
    
    Notes
    -----
    - All charts are generated with consistent styling and color scheme
    - Charts are independent and can be displayed/saved individually
    - Function validates all inputs before generating any charts
    - Indicators dictionary keys are used as trace names in charts
    - If no trades occurred, distribution and monthly charts may be empty
    - Chart generation is logged for debugging
    
    The function is the recommended way to generate visualizations for backtesting
    results, ensuring consistency and completeness across all charts.
    """
    logger.info(f"Creating complete backtest report for: {strategy_name}")
    
    # ========== INPUT VALIDATION ==========
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"data must be pandas DataFrame, got {type(data).__name__}")
    
    if not isinstance(signals, pd.Series):
        raise TypeError(f"signals must be pandas Series, got {type(signals).__name__}")
    
    if not isinstance(backtest_results, dict):
        raise TypeError(f"backtest_results must be dict, got {type(backtest_results).__name__}")
    
    # Validate backtest_results structure
    required_keys = ['trades', 'equity_curve', 'metrics']
    missing_keys = [key for key in required_keys if key not in backtest_results]
    if missing_keys:
        raise ValueError(f"backtest_results missing required keys: {missing_keys}")
    
    # Extract data from backtest results
    trades = backtest_results['trades']
    equity_curve = backtest_results['equity_curve']
    metrics = backtest_results['metrics']
    initial_capital = metrics['initial_capital']
    
    # Initialize figures dictionary
    figures = {}
    
    # ========== 1. PRICE WITH INDICATORS ==========
    logger.info("Generating price with indicators chart...")
    
    if indicators is not None and len(indicators) > 0:
        # Parse indicators dict into function parameters
        plot_kwargs = {'data': data, 'title': f"{strategy_name} - Price Analysis"}
        
        # Map indicator names to function parameters
        indicator_mapping = {
            'SMA': 'sma',
            'SMA_50': 'sma',
            'SMA_200': 'sma',
            'EMA': 'ema',
            'EMA_12': 'ema',
            'EMA_26': 'ema',
            'RSI': 'rsi',
            'RSI_14': 'rsi',
            'MACD': 'macd_line',
            'Signal': 'signal_line',
            'Histogram': 'histogram',
            'BB_Upper': 'bb_upper',
            'BB_Middle': 'bb_middle',
            'BB_Lower': 'bb_lower'
        }
        
        for ind_name, ind_series in indicators.items():
            # Find matching parameter name
            for pattern, param in indicator_mapping.items():
                if pattern in ind_name:
                    plot_kwargs[param] = ind_series
                    break
        
        figures['price_indicators'] = plot_price_with_indicators(**plot_kwargs)
    else:
        # Generate basic price chart without indicators
        figures['price_indicators'] = plot_price_with_indicators(
            data=data,
            title=f"{strategy_name} - Price Chart"
        )
    
    # ========== 2. TRADING SIGNALS ==========
    logger.info("Generating trading signals chart...")
    figures['signals'] = plot_signals(
        data=data,
        signals=signals,
        trades=trades,
        title=f"{strategy_name} - Trading Signals"
    )
    
    # ========== 3. EQUITY CURVE ==========
    logger.info("Generating equity curve chart...")
    figures['equity_curve'] = plot_equity_curve(
        equity_curve=equity_curve,
        initial_capital=initial_capital,
        title=f"{strategy_name} - Equity Curve"
    )
    
    # ========== 4. DRAWDOWN ANALYSIS ==========
    logger.info("Generating drawdown chart...")
    figures['drawdown'] = plot_drawdown(
        equity_curve=equity_curve,
        title=f"{strategy_name} - Drawdown Analysis"
    )
    
    # ========== 5. RETURNS DISTRIBUTION ==========
    logger.info("Generating returns distribution chart...")
    if len(trades) > 0:
        figures['returns_distribution'] = plot_returns_distribution(
            trades=trades,
            title=f"{strategy_name} - Returns Distribution"
        )
    else:
        logger.warning("No trades to plot returns distribution")
        # Create empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No trades executed in backtest",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20)
        )
        fig.update_layout(title=f"{strategy_name} - Returns Distribution", height=500)
        figures['returns_distribution'] = fig
    
    # ========== 6. MONTHLY RETURNS HEATMAP ==========
    logger.info("Generating monthly returns heatmap...")
    try:
        figures['monthly_returns'] = plot_monthly_returns(
            equity_curve=equity_curve,
            title=f"{strategy_name} - Monthly Returns"
        )
    except ValueError as e:
        logger.warning(f"Could not generate monthly returns heatmap: {e}")
        # Create empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="Insufficient data for monthly returns heatmap",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20)
        )
        fig.update_layout(title=f"{strategy_name} - Monthly Returns", height=400)
        figures['monthly_returns'] = fig
    
    # ========== SUMMARY ==========
    total_return = metrics['total_return']
    num_trades = metrics['total_trades']
    win_rate = metrics['win_rate']
    sharpe_ratio = metrics['sharpe_ratio']
    max_drawdown = metrics['max_drawdown']
    
    logger.info(f"""
    ============================================
    BACKTEST REPORT GENERATED: {strategy_name}
    ============================================
    Charts Created: {len(figures)}
    
    Key Metrics:
    - Total Return: {total_return:.2f}%
    - Number of Trades: {num_trades}
    - Win Rate: {win_rate:.2f}%
    - Sharpe Ratio: {sharpe_ratio:.2f}
    - Max Drawdown: {max_drawdown:.2f}%
    
    Charts Available:
    {chr(10).join(f"  - {name}" for name in figures.keys())}
    ============================================
    """)
    
    return figures