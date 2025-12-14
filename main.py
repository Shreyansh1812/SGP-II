"""
Trading Strategy Backtest Dashboard
====================================

A comprehensive Streamlit-based web application for backtesting trading strategies
on Indian and global stock markets. This dashboard provides an intuitive interface
for non-technical users to:

- Fetch historical stock data
- Apply technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands)
- Execute trading strategies (Golden Cross, RSI, MACD)
- Analyze backtest performance with interactive visualizations
- Export results and trade logs

Features:
---------
- **Zero-Code Interface:** Point-and-click backtesting without writing code
- **Real-Time Data:** Fetches live market data via yfinance
- **Multiple Strategies:** Choose from 3 proven trading strategies
- **Interactive Charts:** Plotly-powered visualizations with zoom/pan/hover
- **Comprehensive Metrics:** Risk-adjusted returns, drawdown, win rate, Sharpe ratio
- **Export Functionality:** Download trade logs as CSV

Architecture:
-------------
    User Input (Sidebar) → Data Loading → Indicator Calculation → 
    Signal Generation → Backtesting → Results Display (Tabs)

Modules Used:
-------------
- data_loader: Fetch and validate stock data
- indicators: Calculate technical indicators (Phase 3)
- strategy: Generate trading signals (Phase 4)
- backtester: Execute backtest simulation (Phase 5)
- plotting: Create interactive visualizations (Phase 6)

Usage:
------
Run the dashboard locally:
    streamlit run main.py

Or deploy to Streamlit Cloud for public access.

Author: Shreyansh
Project: SGP-II (Stock Trading Backtest System)
Phase: 7 - Streamlit Dashboard (Final Phase)
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, Tuple, Optional

# Import project modules
from src.data_loader import (
    get_stock_data,
    validate_data
)
from src.indicators import (
    calculate_sma,
    calculate_ema,
    calculate_rsi,
    calculate_macd,
    calculate_bollinger_bands
)
from src.strategy import (
    golden_cross_strategy,
    rsi_mean_reversion_strategy,
    macd_trend_following_strategy
)
from src.backtester import run_backtest
from src.plotting import create_backtest_report
import config

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Trading Backtest Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/Shreyansh1812/SGP-II',
        'Report a bug': 'https://github.com/Shreyansh1812/SGP-II/issues',
        'About': """
        ## Trading Strategy Backtest System
        **Version:** 1.0.0  
        **Phase 7:** Streamlit Dashboard  
        
        A comprehensive backtesting platform for quantitative trading strategies.
        Built with Python, Streamlit, and industry-standard financial libraries.
        """
    }
)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_stock_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch and cache stock data to avoid repeated downloads.
    
    Parameters
    ----------
    ticker : str
        Stock ticker symbol (e.g., 'RELIANCE.NS', 'AAPL')
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
    
    Returns
    -------
    pd.DataFrame
        Stock data with OHLCV columns
    
    Notes
    -----
    Uses Streamlit's caching to prevent redundant API calls.
    Cache expires after 1 hour to allow fresh data fetching.
    This function queries Yahoo Finance API directly when cache misses.
    """
    logger.info(f"🌐 Fetching LIVE data from Yahoo Finance for {ticker} ({start_date} to {end_date})")
    data = get_stock_data(ticker, start_date, end_date)
    if data is not None and not data.empty:
        logger.info(f"✅ Successfully fetched {len(data)} rows from Yahoo Finance")
    return data


@st.cache_data
def calculate_all_indicators(data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Calculate all technical indicators and cache results.
    
    Parameters
    ----------
    data : pd.DataFrame
        Stock price data with OHLCV columns
    
    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary containing all calculated indicators:
        - 'sma_50': 50-period Simple Moving Average
        - 'sma_200': 200-period Simple Moving Average
        - 'ema_12': 12-period Exponential Moving Average
        - 'ema_26': 26-period Exponential Moving Average
        - 'rsi': Relative Strength Index
        - 'macd': MACD with signal and histogram
        - 'bb': Bollinger Bands (upper, middle, lower)
    """
    logger.info("Calculating technical indicators")
    indicators = {}
    
    # Moving Averages for Golden Cross
    indicators['sma_50'] = calculate_sma(data, period=config.GOLDEN_CROSS_FAST_PERIOD)
    indicators['sma_200'] = calculate_sma(data, period=config.GOLDEN_CROSS_SLOW_PERIOD)
    
    # EMAs for MACD
    indicators['ema_12'] = calculate_ema(data, period=config.MACD_FAST_PERIOD)
    indicators['ema_26'] = calculate_ema(data, period=config.MACD_SLOW_PERIOD)
    
    # RSI
    indicators['rsi'] = calculate_rsi(data, period=config.RSI_PERIOD)
    
    # MACD
    indicators['macd'] = calculate_macd(
        data,
        fast_period=config.MACD_FAST_PERIOD,
        slow_period=config.MACD_SLOW_PERIOD,
        signal_period=config.MACD_SIGNAL_PERIOD
    )
    
    # Bollinger Bands
    indicators['bb'] = calculate_bollinger_bands(data, period=20, std_multiplier=2.0)
    
    return indicators


def generate_trading_signals(
    data: pd.DataFrame,
    strategy_name: str,
    indicators: Dict[str, pd.DataFrame]
) -> pd.Series:
    """
    Generate trading signals based on selected strategy.
    
    Parameters
    ----------
    data : pd.DataFrame
        Stock price data
    strategy_name : str
        Name of strategy: 'Golden Cross', 'RSI', or 'MACD'
    indicators : Dict[str, pd.DataFrame]
        Pre-calculated technical indicators
    
    Returns
    -------
    pd.Series
        Trading signals: 1 (BUY), -1 (SELL), 0 (HOLD)
    """
    logger.info(f"Generating signals for {strategy_name} strategy")
    
    if strategy_name == "Golden Cross":
        signals = golden_cross_strategy(
            data,
            fast_period=config.GOLDEN_CROSS_FAST_PERIOD,
            slow_period=config.GOLDEN_CROSS_SLOW_PERIOD
        )
    elif strategy_name == "RSI":
        signals = rsi_mean_reversion_strategy(
            data,
            rsi_period=config.RSI_PERIOD,
            oversold_threshold=config.RSI_OVERSOLD,
            overbought_threshold=config.RSI_OVERBOUGHT
        )
    elif strategy_name == "MACD":
        signals = macd_trend_following_strategy(
            data,
            fast_period=config.MACD_FAST_PERIOD,
            slow_period=config.MACD_SLOW_PERIOD,
            signal_period=config.MACD_SIGNAL_PERIOD
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")
    
    return signals


def format_currency(value: float) -> str:
    """Format currency values with Indian comma notation."""
    if value >= 10000000:  # 1 Crore
        return f"₹{value/10000000:.2f} Cr"
    elif value >= 100000:  # 1 Lakh
        return f"₹{value/100000:.2f} L"
    else:
        return f"₹{value:,.2f}"


def format_percentage(value: float, decimals: int = 2) -> str:
    """Format percentage values with color coding."""
    return f"{value:.{decimals}f}%"


# =============================================================================
# UI COMPONENT FUNCTIONS
# =============================================================================

def render_sidebar() -> Dict:
    """
    Render sidebar with user configuration inputs.
    
    Returns
    -------
    Dict
        Dictionary containing all user-selected parameters:
        - ticker: Stock symbol
        - start_date: Backtest start date
        - end_date: Backtest end date
        - strategy: Selected strategy name
        - initial_capital: Starting capital amount
        - commission: Transaction commission rate
    """
    st.sidebar.title("⚙️ Configuration")
    st.sidebar.markdown("---")
    
    # Stock Selection
    st.sidebar.subheader("📊 Stock Selection")
    ticker = st.sidebar.text_input(
        "Ticker Symbol",
        value=config.TICKER,
        help="Enter stock symbol (e.g., AAPL for US stocks, TCS.NS for Indian NSE stocks)",
        placeholder="AAPL"
    )
    
    # Ticker examples
    with st.sidebar.expander("💡 Ticker Examples", expanded=False):
        st.caption("""
        **US Stocks (Most Reliable):**
        - AAPL (Apple) ✅ Recommended
        - MSFT (Microsoft)
        - GOOGL (Google)
        - TSLA (Tesla)
        - AMZN (Amazon)
        
        **Indian Stocks (NSE):**
        - TCS.NS (Tata Consultancy Services)
        - INFY.NS (Infosys)
        - HDFCBANK.NS (HDFC Bank)
        - RELIANCE.NS (Reliance Industries)
        
        **Note:** Yahoo Finance may have temporary issues with .NS stocks.
        """)
    
    
    # Date Range
    st.sidebar.subheader("📅 Date Range")
    col1, col2 = st.sidebar.columns(2)
    
    default_start = datetime.strptime(config.START_DATE, "%Y-%m-%d").date()
    default_end = datetime.strptime(config.END_DATE, "%Y-%m-%d").date()
    
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=default_start,
            max_value=datetime.now().date(),
            help="Beginning of backtest period"
        )
    
    with col2:
        end_date = st.date_input(
            "End Date",
            value=default_end,
            max_value=datetime.now().date(),
            help="End of backtest period"
        )
    
    # Strategy Selection
    st.sidebar.subheader("🎯 Trading Strategy")
    strategy = st.sidebar.selectbox(
        "Select Strategy",
        options=["Golden Cross", "RSI", "MACD"],
        help="""
        - **Golden Cross:** 50/200 SMA crossover (trend following)
        - **RSI:** Overbought/oversold momentum (mean reversion)
        - **MACD:** Moving average convergence divergence (momentum)
        """
    )
    
    # Display strategy parameters
    with st.sidebar.expander("📋 Strategy Parameters", expanded=False):
        if strategy == "Golden Cross":
            st.write(f"- Fast SMA: {config.GOLDEN_CROSS_FAST_PERIOD} days")
            st.write(f"- Slow SMA: {config.GOLDEN_CROSS_SLOW_PERIOD} days")
        elif strategy == "RSI":
            st.write(f"- Period: {config.RSI_PERIOD} days")
            st.write(f"- Oversold: {config.RSI_OVERSOLD}")
            st.write(f"- Overbought: {config.RSI_OVERBOUGHT}")
        elif strategy == "MACD":
            st.write(f"- Fast Period: {config.MACD_FAST_PERIOD}")
            st.write(f"- Slow Period: {config.MACD_SLOW_PERIOD}")
            st.write(f"- Signal Period: {config.MACD_SIGNAL_PERIOD}")
    
    # Capital Settings
    st.sidebar.subheader("💰 Capital Settings")
    initial_capital = st.sidebar.slider(
        "Initial Capital (₹)",
        min_value=10000,
        max_value=10000000,
        value=int(config.INITIAL_CASH),
        step=10000,
        format="₹%d",
        help="Starting capital for backtest"
    )
    
    commission = st.sidebar.slider(
        "Commission (%)",
        min_value=0.0,
        max_value=1.0,
        value=config.COMMISSION * 100,
        step=0.05,
        format="%.2f%%",
        help="Brokerage commission per trade"
    )
    
    st.sidebar.markdown("---")
    
    # Disclaimer
    with st.sidebar.expander("⚠️ Disclaimer", expanded=False):
        st.caption("""
        **Important Notes:**
        
        1. **Execution Timing:** This backtest uses closing prices for 
           execution (simplification). Real trading executes next-day open.
        
        2. **Past Performance:** Historical results do not guarantee 
           future performance.
        
        3. **Risk Warning:** Trading involves substantial risk. This is 
           for educational purposes only.
        
        4. **Data Accuracy:** Market data is fetched from Yahoo Finance 
           and may contain errors.
        """)
    
    return {
        'ticker': ticker.strip().upper(),
        'start_date': start_date.strftime("%Y-%m-%d"),
        'end_date': end_date.strftime("%Y-%m-%d"),
        'strategy': strategy,
        'initial_capital': float(initial_capital),
        'commission': commission / 100.0  # Convert to decimal
    }


def render_header():
    """Render main dashboard header."""
    st.title("📈 Trading Strategy Backtest Dashboard")
    st.markdown("""
    <div style='background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin-bottom: 20px;'>
        <p style='margin: 0; font-size: 16px;'>
            <strong>Welcome to the Trading Backtest System!</strong><br>
            Configure your backtest parameters in the sidebar, click <strong>"Run Backtest"</strong>, 
            and analyze comprehensive results across multiple tabs. Supports Indian (NSE/BSE) and 
            global stocks with proven technical strategies.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Data Source Information
    st.info("""
    ℹ️ **Data Source:** Real-time historical data from **Yahoo Finance**
    - Fetches live market data with 3 automatic retries
    - Falls back to cached data if available
    - Supports 100,000+ global stocks (US, India, Europe, Asia)
    - Data updates: Daily after market close
    
    **Note:** If Yahoo Finance is temporarily unavailable, the system will use cached data from previous downloads.
    """)


def render_summary_tab(results: Dict, params: Dict):
    """
    Render executive summary tab with key metrics.
    
    Parameters
    ----------
    results : Dict
        Backtest results from run_backtest()
    params : Dict
        User configuration parameters
    """
    st.header("📊 Executive Summary")
    
    # Extract key metrics
    metrics = results['metrics']
    total_return = metrics['total_return']
    final_equity = results['equity_curve'].iloc[-1]
    total_trades = metrics['total_trades']
    win_rate = metrics['win_rate']
    sharpe_ratio = metrics.get('sharpe_ratio', 0.0)
    max_drawdown = metrics.get('max_drawdown', 0.0)
    
    # Top-level metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Return",
            value=format_percentage(total_return),
            delta=format_currency(final_equity - params['initial_capital'])
        )
    
    with col2:
        st.metric(
            label="Final Equity",
            value=format_currency(final_equity),
            delta=None
        )
    
    with col3:
        st.metric(
            label="Sharpe Ratio",
            value=f"{sharpe_ratio:.2f}",
            delta="Good" if sharpe_ratio > 1 else "Poor",
            delta_color="normal" if sharpe_ratio > 1 else "inverse"
        )
    
    with col4:
        st.metric(
            label="Max Drawdown",
            value=format_percentage(abs(max_drawdown)),
            delta=None
        )
    
    st.markdown("---")
    
    # Trade Statistics
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        st.metric(
            label="Total Trades",
            value=f"{total_trades}",
            delta=None
        )
    
    with col6:
        st.metric(
            label="Win Rate",
            value=format_percentage(win_rate),
            delta="Strong" if win_rate > 60 else "Weak",
            delta_color="normal" if win_rate > 60 else "inverse"
        )
    
    with col7:
        avg_win = metrics.get('avg_win', 0.0)
        st.metric(
            label="Avg Win",
            value=format_percentage(avg_win),
            delta=None
        )
    
    with col8:
        avg_loss = metrics.get('avg_loss', 0.0)
        st.metric(
            label="Avg Loss",
            value=format_percentage(abs(avg_loss)),
            delta=None
        )
    
    st.markdown("---")
    
    # Backtest Configuration Summary
    st.subheader("⚙️ Backtest Configuration")
    config_col1, config_col2, config_col3 = st.columns(3)
    
    with config_col1:
        st.write(f"**Ticker:** {params['ticker']}")
        st.write(f"**Strategy:** {params['strategy']}")
    
    with config_col2:
        st.write(f"**Period:** {params['start_date']} to {params['end_date']}")
        days = (datetime.strptime(params['end_date'], "%Y-%m-%d") - 
                datetime.strptime(params['start_date'], "%Y-%m-%d")).days
        st.write(f"**Duration:** {days} days")
    
    with config_col3:
        st.write(f"**Initial Capital:** {format_currency(params['initial_capital'])}")
        st.write(f"**Commission:** {format_percentage(params['commission'] * 100)}")


def render_charts_tab(figures: Dict, params: Dict):
    """
    Render interactive charts tab.
    
    Parameters
    ----------
    figures : Dict
        Dictionary of Plotly figures from create_backtest_report()
    params : Dict
        User configuration parameters
    """
    st.header("📉 Performance Charts")
    
    # Price chart with indicators
    st.subheader(f"💹 {params['ticker']} Price with Technical Indicators")
    if 'price_indicators' in figures:
        st.plotly_chart(
            figures['price_indicators'],
            use_container_width=True,
            key="price_chart"
        )
    else:
        st.warning("Price chart not available")
    
    st.markdown("---")
    
    # Trading signals
    st.subheader("🎯 Trading Signals (Buy/Sell)")
    if 'signals' in figures:
        st.plotly_chart(
            figures['signals'],
            use_container_width=True,
            key="signals_chart"
        )
    else:
        st.warning("Signals chart not available")
    
    st.markdown("---")
    
    # Equity curve
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💰 Equity Curve")
        if 'equity_curve' in figures:
            st.plotly_chart(
                figures['equity_curve'],
                use_container_width=True,
                key="equity_chart"
            )
        else:
            st.warning("Equity curve not available")
    
    with col2:
        st.subheader("📉 Drawdown Analysis")
        if 'drawdown' in figures:
            st.plotly_chart(
                figures['drawdown'],
                use_container_width=True,
                key="drawdown_chart"
            )
        else:
            st.warning("Drawdown chart not available")
    
    st.markdown("---")
    
    # Returns distribution and monthly returns
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("📊 Returns Distribution")
        if 'returns_distribution' in figures:
            st.plotly_chart(
                figures['returns_distribution'],
                use_container_width=True,
                key="returns_dist_chart"
            )
        else:
            st.warning("Returns distribution not available")
    
    with col4:
        st.subheader("📅 Monthly Returns Heatmap")
        if 'monthly_returns' in figures:
            st.plotly_chart(
                figures['monthly_returns'],
                use_container_width=True,
                key="monthly_returns_chart"
            )
        else:
            st.warning("Monthly returns heatmap not available")


def render_trades_tab(results: Dict):
    """
    Render trade log table.
    
    Parameters
    ----------
    results : Dict
        Backtest results containing trade history
    """
    st.header("📋 Trade Log")
    
    if 'trades' not in results or len(results['trades']) == 0:
        st.warning("⚠️ No trades were executed during this backtest period.")
        st.info("""
        **Possible reasons:**
        - Strategy conditions were never met
        - Date range too short for signals to generate
        - Market conditions didn't trigger entry/exit rules
        
        **Suggestions:**
        - Extend the date range
        - Try a different strategy
        - Adjust strategy parameters (available in config.py)
        """)
        return
    
    # Convert trades to DataFrame
    trades_df = pd.DataFrame(results['trades'])
    
    # Format dates
    trades_df['entry_date'] = pd.to_datetime(trades_df['entry_date']).dt.strftime('%Y-%m-%d')
    trades_df['exit_date'] = pd.to_datetime(trades_df['exit_date']).dt.strftime('%Y-%m-%d')
    
    # Format numeric columns (holding_days already exists in trades)
    trades_df['entry_price'] = trades_df['entry_price'].apply(lambda x: f"₹{x:.2f}")
    trades_df['exit_price'] = trades_df['exit_price'].apply(lambda x: f"₹{x:.2f}")
    trades_df['return_pct'] = trades_df['return_pct'].apply(lambda x: f"{x:.2f}%")
    
    # Rename columns for display
    display_df = trades_df[[
        'entry_date', 'exit_date', 'entry_price', 'exit_price', 
        'return_pct', 'holding_days'
    ]].copy()
    
    display_df.columns = [
        'Entry Date', 'Exit Date', 'Entry Price', 'Exit Price',
        'Return %', 'Holding Days'
    ]
    
    # Display statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Trades", len(trades_df))
    with col2:
        winning_trades = len([t for t in results['trades'] if t['return_pct'] > 0])
        st.metric("Winning Trades", winning_trades)
    with col3:
        losing_trades = len([t for t in results['trades'] if t['return_pct'] < 0])
        st.metric("Losing Trades", losing_trades)
    with col4:
        avg_holding = np.mean([t['holding_days'] for t in results['trades']])
        st.metric("Avg Holding", f"{avg_holding:.1f} days")
    
    st.markdown("---")
    
    # Display table
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )
    
    # Download button
    csv = trades_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Trade Log (CSV)",
        data=csv,
        file_name=f"trade_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )


def render_metrics_tab(results: Dict):
    """
    Render detailed metrics tab.
    
    Parameters
    ----------
    results : Dict
        Backtest results with comprehensive metrics
    """
    st.header("📈 Detailed Metrics")
    
    metrics = results['metrics']
    
    # Performance Metrics
    st.subheader("🎯 Performance Metrics")
    perf_col1, perf_col2, perf_col3 = st.columns(3)
    
    with perf_col1:
        st.metric("Total Return", format_percentage(metrics['total_return']))
        st.metric("Annualized Return", format_percentage(metrics.get('total_return', 0)))
        st.metric("CAGR", format_percentage(metrics.get('cagr', 0)))
    
    with perf_col2:
        st.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.3f}")
        st.metric("Sortino Ratio", f"{metrics.get('sortino_ratio', 0):.3f}")
        st.metric("Calmar Ratio", f"{metrics.get('calmar_ratio', 0):.3f}")
    
    with perf_col3:
        st.metric("Max Drawdown", format_percentage(abs(metrics.get('max_drawdown', 0))))
        st.metric("Volatility (Annual)", format_percentage(metrics.get('volatility', 0)))
        st.metric("Win Rate", format_percentage(metrics['win_rate']))
    
    st.markdown("---")
    
    # Trade Statistics
    st.subheader("📊 Trade Statistics")
    trade_col1, trade_col2, trade_col3 = st.columns(3)
    
    with trade_col1:
        st.metric("Total Trades", f"{metrics['total_trades']}")
        # Calculate winning/losing trades from results
        winning_trades = len([t for t in results['trades'] if t['return_pct'] > 0])
        losing_trades = len([t for t in results['trades'] if t['return_pct'] < 0])
        st.metric("Winning Trades", f"{winning_trades}")
        st.metric("Losing Trades", f"{losing_trades}")
    
    with trade_col2:
        st.metric("Average Win", format_percentage(metrics.get('avg_win', 0)))
        st.metric("Average Loss", format_percentage(abs(metrics.get('avg_loss', 0))))
        # Calculate largest win from trades
        if len(results['trades']) > 0:
            largest_win = max([t['return_pct'] for t in results['trades']])
        else:
            largest_win = 0
        st.metric("Largest Win", format_percentage(largest_win))
    
    with trade_col3:
        # Calculate largest loss from trades
        if len(results['trades']) > 0:
            largest_loss = min([t['return_pct'] for t in results['trades']])
        else:
            largest_loss = 0
        st.metric("Largest Loss", format_percentage(abs(largest_loss)))
        st.metric("Profit Factor", f"{metrics.get('profit_factor', 0):.2f}")
        st.metric("Avg Holding Period", f"{metrics.get('avg_holding_days', 0):.1f} days")
    
    st.markdown("---")
    
    # Risk Metrics
    st.subheader("⚠️ Risk Metrics")
    risk_col1, risk_col2, risk_col3 = st.columns(3)
    
    with risk_col1:
        st.metric("Max Drawdown", format_percentage(abs(metrics.get('max_drawdown', 0))))
        st.metric("Volatility", format_percentage(metrics.get('volatility', 0)))
    
    with risk_col2:
        st.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.3f}")
        st.metric("Exposure Time", format_percentage(metrics.get('exposure_time', 0)))
    
    with risk_col3:
        st.metric("Initial Capital", format_currency(metrics.get('initial_capital', 0)))
        st.metric("Final Equity", format_currency(metrics.get('final_equity', 0)))


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application entry point."""
    
    # Render header
    render_header()
    
    # Render sidebar and get parameters
    params = render_sidebar()
    
    # Cache Control Buttons
    st.sidebar.markdown("---")
    cache_col1, cache_col2 = st.sidebar.columns(2)
    
    with cache_col1:
        if st.button("🔄 Clear Cache", use_container_width=True, help="Clear cached data and fetch fresh from Yahoo Finance"):
            st.cache_data.clear()
            st.sidebar.success("✅ Cache cleared!")
            st.rerun()
    
    with cache_col2:
        if st.button("ℹ️ Cache Info", use_container_width=True, help="View cache status"):
            st.sidebar.info(f"""
            **Cache Settings:**
            - Data cache TTL: 1 hour
            - Indicators: Cached until cleared
            - Yahoo Finance is queried on first load or after cache expiry
            """)
    
    # Run Backtest Button
    st.sidebar.markdown("---")
    run_button = st.sidebar.button(
        "🚀 Run Backtest",
        type="primary",
        use_container_width=True
    )
    
    if run_button:
        # Validation
        if params['start_date'] >= params['end_date']:
            st.error("❌ **Error:** Start date must be before end date!")
            st.stop()
        
        if params['initial_capital'] < 1000:
            st.error("❌ **Error:** Initial capital must be at least ₹1,000!")
            st.stop()
        
        # Processing Pipeline
        try:
            # Step 1: Fetch Data
            with st.spinner(f"📡 Fetching data for {params['ticker']} from Yahoo Finance..."):
                # Check if data is in cache
                cache_key = f"{params['ticker']}_{params['start_date']}_{params['end_date']}"
                
                data = fetch_stock_data(
                    params['ticker'],
                    params['start_date'],
                    params['end_date']
                )
                
                # Validate data
                if data is None or data.empty:
                    st.error(f"❌ **Error:** No data found for {params['ticker']} in the specified date range.")
                    st.info("""
                    **Possible Reasons:**
                    - Invalid ticker symbol (use format: TICKER.NS for NSE, TICKER.BO for BSE)
                    - Network connectivity issues
                    - Yahoo Finance API temporarily unavailable
                    - Symbol may be delisted
                    
                    **Suggestions:**
                    - For Indian stocks, try: RELIANCE.NS, TCS.NS, INFY.NS, HDFCBANK.NS
                    - For US stocks, try: AAPL, MSFT, GOOGL, TSLA
                    - Check your internet connection
                    - Try a different date range
                    - Wait a few minutes and try again
                    - Click "🔄 Clear Cache" button to force fresh download
                    """)
                    st.stop()
                
                # Show data source info
                st.success(f"✅ Loaded {len(data)} days of data from Yahoo Finance")
                st.caption(f"💾 Data cached for 1 hour | Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Step 2: Calculate Indicators
            with st.spinner("📊 Calculating technical indicators..."):
                indicators = calculate_all_indicators(data)
                st.success("✅ Indicators calculated")
            
            # Step 3: Generate Signals
            with st.spinner(f"🎯 Generating {params['strategy']} signals..."):
                signals = generate_trading_signals(data, params['strategy'], indicators)
                signal_count = (signals != 0).sum()
                st.success(f"✅ Generated {signal_count} trading signals")
            
            # Step 4: Run Backtest
            with st.spinner("⚙️ Running backtest simulation..."):
                results = run_backtest(
                    data=data,
                    signals=signals,
                    initial_capital=params['initial_capital'],
                    commission=params['commission']
                )
                st.success("✅ Backtest completed")
            
            # Step 5: Create Visualizations
            with st.spinner("📈 Generating interactive charts..."):
                figures = create_backtest_report(
                    data=data,
                    signals=signals,
                    backtest_results=results,
                    indicators=indicators,
                    strategy_name=params['strategy']
                )
                st.success("✅ Visualizations ready")
            
            # Display Results in Tabs
            st.markdown("---")
            tab1, tab2, tab3, tab4 = st.tabs([
                "📊 Summary",
                "📈 Charts",
                "📋 Trades",
                "📉 Metrics"
            ])
            
            with tab1:
                render_summary_tab(results, params)
            
            with tab2:
                render_charts_tab(figures, params)
            
            with tab3:
                render_trades_tab(results)
            
            with tab4:
                render_metrics_tab(results)
            
            # Success message
            st.sidebar.success("✅ Backtest completed successfully!")
            
        except Exception as e:
            st.error(f"❌ **Error during backtest:** {str(e)}")
            logger.error(f"Backtest failed: {str(e)}", exc_info=True)
            
            with st.expander("🐛 Debug Information"):
                st.code(f"""
Error Type: {type(e).__name__}
Error Message: {str(e)}

Parameters:
- Ticker: {params['ticker']}
- Start Date: {params['start_date']}
- End Date: {params['end_date']}
- Strategy: {params['strategy']}
- Initial Capital: {params['initial_capital']}
- Commission: {params['commission']}
                """)
    
    else:
        # Initial state - show instructions
        st.info("""
        ### 👈 Get Started
        
        1. **Configure Settings:** Adjust parameters in the sidebar
        2. **Click "Run Backtest":** Execute the backtest simulation
        3. **Analyze Results:** Explore charts, metrics, and trade logs
        4. **Iterate:** Modify settings and re-run for comparison
        
        **Tip:** Start with default settings to see how the system works!
        """)
        
        # Display sample metrics as placeholder
        st.subheader("Sample Backtest Results")
        st.caption("_These are example metrics. Run a backtest to see real results._")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Return", "45.23%", "₹45,230")
        with col2:
            st.metric("Sharpe Ratio", "1.85", "Good")
        with col3:
            st.metric("Win Rate", "62.50%", "Strong")
        with col4:
            st.metric("Max Drawdown", "12.34%")


if __name__ == "__main__":
    main()