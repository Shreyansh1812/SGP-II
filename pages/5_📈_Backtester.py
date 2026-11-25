"""
Page 5: Strategy Backtester
=============================

Migrated from the original single-file ``main.py``. Provides a comprehensive
backtesting interface for technical trading strategies with interactive charts,
trade logs, and performance metrics.

Strategies:
- Golden Cross (SMA 50/200 crossover)
- RSI Mean Reversion (overbought/oversold)
- MACD Trend Following (signal line crossover)

Author: Shreyansh Patel
Project: SGP-II — AI-Augmented Decision-Support Sandbox
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict

from src.data_loader import get_stock_data, validate_data
from src.indicators import (
    calculate_sma, calculate_ema, calculate_rsi,
    calculate_macd, calculate_bollinger_bands,
)
from src.strategy import (
    golden_cross_strategy,
    rsi_mean_reversion_strategy,
    macd_trend_following_strategy,
)
from src.backtester import run_backtest
from src.plotting import create_backtest_report
from src.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


@st.cache_data(ttl=3600)
def fetch_stock_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch and cache stock data from Yahoo Finance.

    Uses Streamlit's built-in caching (TTL=1 hour) to avoid redundant API calls.
    """
    logger.info(f"🌐 Fetching data for {ticker} ({start_date} to {end_date})")
    data = get_stock_data(ticker, start_date, end_date)
    if data is not None and not data.empty:
        logger.info(f"✅ Fetched {len(data)} rows from Yahoo Finance")
    return data


@st.cache_data
def calculate_all_indicators(data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Calculate all technical indicators and cache results."""
    indicators = {}
    indicators["sma_50"] = calculate_sma(data, period=settings.GOLDEN_CROSS_FAST)
    indicators["sma_200"] = calculate_sma(data, period=settings.GOLDEN_CROSS_SLOW)
    indicators["ema_12"] = calculate_ema(data, period=settings.MACD_FAST)
    indicators["ema_26"] = calculate_ema(data, period=settings.MACD_SLOW)
    indicators["rsi"] = calculate_rsi(data, period=settings.RSI_PERIOD)
    indicators["macd"] = calculate_macd(
        data,
        fast_period=settings.MACD_FAST,
        slow_period=settings.MACD_SLOW,
        signal_period=settings.MACD_SIGNAL,
    )
    indicators["bb"] = calculate_bollinger_bands(data, period=20, std_multiplier=2.0)
    return indicators


def generate_signals(data, strategy_name, indicators):
    """Generate trading signals based on selected strategy."""
    if strategy_name == "Golden Cross":
        return golden_cross_strategy(
            data, fast_period=settings.GOLDEN_CROSS_FAST, slow_period=settings.GOLDEN_CROSS_SLOW
        )
    elif strategy_name == "RSI":
        return rsi_mean_reversion_strategy(
            data, rsi_period=settings.RSI_PERIOD,
            oversold_threshold=settings.RSI_OVERSOLD,
            overbought_threshold=settings.RSI_OVERBOUGHT,
        )
    elif strategy_name == "MACD":
        return macd_trend_following_strategy(
            data, fast_period=settings.MACD_FAST,
            slow_period=settings.MACD_SLOW,
            signal_period=settings.MACD_SIGNAL,
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")


def format_currency(value: float) -> str:
    """Format currency in USD notation."""
    if value >= 1_000_000:
        return f"${value / 1_000_000:.2f}M"
    elif value >= 1_000:
        return f"${value / 1_000:.2f}K"
    return f"${value:,.2f}"


def format_pct(value: float, decimals: int = 2) -> str:
    return f"{value:.{decimals}f}%"


# =============================================================================
# PAGE CONTENT
# =============================================================================

st.title("📈 Strategy Backtester")
st.markdown(
    """
    <div style='background: linear-gradient(135deg, #141e30, #243b55);
                padding: 20px; border-radius: 12px; margin-bottom: 20px;
                color: white;'>
        <h4 style='margin: 0;'>⚡ Technical Strategy Backtesting</h4>
        <p style='margin: 5px 0 0 0; font-size: 14px; opacity: 0.85;'>
            Fetch live market data, apply proven technical strategies, and analyze
            comprehensive backtest results with interactive Plotly charts.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# SIDEBAR CONTROLS
# --------------------------------------------------------------------------

with st.sidebar:
    st.subheader("⚙️ Backtest Settings")

    ticker = st.text_input("Ticker Symbol", value=settings.TICKER, placeholder="AAPL")

    with st.expander("💡 Ticker Examples", expanded=False):
        st.caption(
            "**US:** AAPL, MSFT, GOOGL, TSLA, AMZN\n"
            "**India (NSE):** TCS.NS, INFY.NS, RELIANCE.NS"
        )

    col_d1, col_d2 = st.columns(2)
    with col_d1:
        start_date = st.date_input(
            "Start Date",
            value=datetime.strptime(settings.START_DATE, "%Y-%m-%d").date(),
            max_value=datetime.now().date(),
        )
    with col_d2:
        end_date = st.date_input(
            "End Date",
            value=datetime.strptime(settings.END_DATE, "%Y-%m-%d").date(),
            max_value=datetime.now().date(),
        )

    strategy = st.selectbox(
        "🎯 Strategy",
        options=["Golden Cross", "RSI", "MACD"],
        help=(
            "**Golden Cross:** 50/200 SMA crossover\n"
            "**RSI:** Overbought/Oversold momentum\n"
            "**MACD:** Signal line crossover"
        ),
    )

    with st.expander("📋 Strategy Parameters"):
        if strategy == "Golden Cross":
            st.write(f"Fast SMA: {settings.GOLDEN_CROSS_FAST} days")
            st.write(f"Slow SMA: {settings.GOLDEN_CROSS_SLOW} days")
        elif strategy == "RSI":
            st.write(f"Period: {settings.RSI_PERIOD} days")
            st.write(f"Oversold: {settings.RSI_OVERSOLD}")
            st.write(f"Overbought: {settings.RSI_OVERBOUGHT}")
        elif strategy == "MACD":
            st.write(f"Fast: {settings.MACD_FAST}")
            st.write(f"Slow: {settings.MACD_SLOW}")
            st.write(f"Signal: {settings.MACD_SIGNAL}")

    initial_capital = st.slider(
        "Initial Capital ($)", 1_000, 1_000_000,
        int(settings.INITIAL_CASH), step=5_000, format="$%d",
    )
    commission = st.slider(
        "Commission (%)", 0.0, 1.0,
        settings.COMMISSION * 100, step=0.05, format="%.2f%%",
    )

    st.divider()

    run_button = st.button("🚀 Run Backtest", type="primary", use_container_width=True)

    if st.button("🔄 Clear Cache", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# --------------------------------------------------------------------------
# BACKTEST EXECUTION
# --------------------------------------------------------------------------

if run_button:
    params = {
        "ticker": ticker.strip().upper(),
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "strategy": strategy,
        "initial_capital": float(initial_capital),
        "commission": commission / 100.0,
    }

    if params["start_date"] >= params["end_date"]:
        st.error("❌ Start date must be before end date!")
        st.stop()

    try:
        # Step 1: Fetch data
        with st.spinner(f"📡 Fetching data for {params['ticker']}..."):
            data = fetch_stock_data(params["ticker"], params["start_date"], params["end_date"])
            if data is None or data.empty:
                st.error(f"❌ No data found for {params['ticker']}.")
                st.stop()
            st.success(f"✅ Loaded {len(data)} trading days")

        # Step 2: Calculate indicators
        with st.spinner("📊 Calculating technical indicators..."):
            indicators = calculate_all_indicators(data)

        # Step 3: Generate signals
        with st.spinner(f"🎯 Generating {strategy} signals..."):
            signals = generate_signals(data, strategy, indicators)
            signal_count = (signals != 0).sum()
            st.success(f"✅ Generated {signal_count} signals")

        # Step 4: Run backtest
        with st.spinner("⚙️ Running backtest..."):
            results = run_backtest(
                data=data, signals=signals,
                initial_capital=params["initial_capital"],
                commission=params["commission"],
            )

        # Step 5: Create charts
        with st.spinner("📈 Generating charts..."):
            figures = create_backtest_report(
                data=data, signals=signals, backtest_results=results,
                indicators=indicators, strategy_name=strategy,
            )

        # ===== RESULTS =====
        st.divider()

        tab1, tab2, tab3, tab4 = st.tabs(["📊 Summary", "📈 Charts", "📋 Trades", "📉 Metrics"])

        # --- Tab 1: Summary ---
        with tab1:
            metrics = results["metrics"]
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Total Return", format_pct(metrics["total_return"]),
                          format_currency(results["equity_curve"].iloc[-1] - params["initial_capital"]))
            with c2:
                st.metric("Final Equity", format_currency(results["equity_curve"].iloc[-1]))
            with c3:
                sr = metrics.get("sharpe_ratio", 0)
                st.metric("Sharpe Ratio", f"{sr:.2f}",
                          "Good" if sr > 1 else "Poor",
                          delta_color="normal" if sr > 1 else "inverse")
            with c4:
                st.metric("Max Drawdown", format_pct(abs(metrics.get("max_drawdown", 0))))

            st.divider()
            c5, c6, c7, c8 = st.columns(4)
            with c5:
                st.metric("Total Trades", metrics["total_trades"])
            with c6:
                wr = metrics["win_rate"]
                st.metric("Win Rate", format_pct(wr),
                          "Strong" if wr > 60 else "Weak",
                          delta_color="normal" if wr > 60 else "inverse")
            with c7:
                st.metric("Avg Win", format_pct(metrics.get("avg_win", 0)))
            with c8:
                st.metric("Avg Loss", format_pct(abs(metrics.get("avg_loss", 0))))

        # --- Tab 2: Charts ---
        with tab2:
            if "price_indicators" in figures:
                st.subheader(f"💹 {params['ticker']} — Price & Indicators")
                st.plotly_chart(figures["price_indicators"], use_container_width=True)
            if "signals" in figures:
                st.subheader("🎯 Buy/Sell Signals")
                st.plotly_chart(figures["signals"], use_container_width=True)

            ch1, ch2 = st.columns(2)
            with ch1:
                if "equity_curve" in figures:
                    st.subheader("💰 Equity Curve")
                    st.plotly_chart(figures["equity_curve"], use_container_width=True)
            with ch2:
                if "drawdown" in figures:
                    st.subheader("📉 Drawdown")
                    st.plotly_chart(figures["drawdown"], use_container_width=True)

            ch3, ch4 = st.columns(2)
            with ch3:
                if "returns_distribution" in figures:
                    st.subheader("📊 Returns Distribution")
                    st.plotly_chart(figures["returns_distribution"], use_container_width=True)
            with ch4:
                if "monthly_returns" in figures:
                    st.subheader("📅 Monthly Returns")
                    st.plotly_chart(figures["monthly_returns"], use_container_width=True)

        # --- Tab 3: Trades ---
        with tab3:
            if results.get("trades"):
                trades_df = pd.DataFrame(results["trades"])
                trades_df["entry_date"] = pd.to_datetime(trades_df["entry_date"]).dt.strftime("%Y-%m-%d")
                trades_df["exit_date"] = pd.to_datetime(trades_df["exit_date"]).dt.strftime("%Y-%m-%d")

                tc1, tc2, tc3, tc4 = st.columns(4)
                with tc1:
                    st.metric("Total", len(trades_df))
                with tc2:
                    st.metric("Winners", len([t for t in results["trades"] if t["return_pct"] > 0]))
                with tc3:
                    st.metric("Losers", len([t for t in results["trades"] if t["return_pct"] < 0]))
                with tc4:
                    avg_hold = np.mean([t["holding_days"] for t in results["trades"]])
                    st.metric("Avg Hold", f"{avg_hold:.1f}d")

                disp = trades_df[["entry_date", "exit_date", "entry_price", "exit_price", "return_pct", "holding_days"]].copy()
                disp["entry_price"] = disp["entry_price"].apply(lambda x: f"${x:.2f}")
                disp["exit_price"] = disp["exit_price"].apply(lambda x: f"${x:.2f}")
                disp["return_pct"] = disp["return_pct"].apply(lambda x: f"{x:.2f}%")
                disp.columns = ["Entry", "Exit", "Buy Price", "Sell Price", "Return %", "Days"]
                st.dataframe(disp, use_container_width=True, hide_index=True)

                st.download_button("📥 Download Trades (CSV)", trades_df.to_csv(index=False),
                                   f"trades_{datetime.now():%Y%m%d}.csv", "text/csv")
            else:
                st.warning("No trades executed during this period.")

        # --- Tab 4: Metrics ---
        with tab4:
            metrics = results["metrics"]
            mc1, mc2, mc3 = st.columns(3)
            with mc1:
                st.metric("Total Return", format_pct(metrics["total_return"]))
                st.metric("CAGR", format_pct(metrics.get("cagr", 0)))
                st.metric("Sharpe", f"{metrics.get('sharpe_ratio', 0):.3f}")
            with mc2:
                st.metric("Sortino", f"{metrics.get('sortino_ratio', 0):.3f}")
                st.metric("Calmar", f"{metrics.get('calmar_ratio', 0):.3f}")
                st.metric("Profit Factor", f"{metrics.get('profit_factor', 0):.2f}")
            with mc3:
                st.metric("Max Drawdown", format_pct(abs(metrics.get("max_drawdown", 0))))
                st.metric("Volatility", format_pct(metrics.get("volatility", 0)))
                st.metric("Exposure Time", format_pct(metrics.get("exposure_time", 0)))

    except Exception as e:
        st.error(f"❌ Backtest failed: {e}")
        logger.error(f"Backtest error: {e}", exc_info=True)
        with st.expander("🐛 Debug"):
            st.code(f"Error: {type(e).__name__}: {e}\nTicker: {params['ticker']}")

else:
    st.info(
        "### 👈 Configure & Run\n\n"
        "1. Set ticker, dates, and strategy in the sidebar\n"
        "2. Click **🚀 Run Backtest** to analyze\n"
        "3. Explore results across the four tabs"
    )
