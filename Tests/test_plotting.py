"""
Comprehensive Test Suite for Plotting Module

This test suite validates all visualization functions in src/plotting.py
Tests cover:
1. Input validation and error handling
2. Figure structure and trace validation
3. Layout configuration verification
4. Data integrity checks
5. Edge cases and special scenarios
6. Integration with real market data

Author: Shreyansh Patel
Project: SGP-II - Python-Based Algorithmic Trading Backtester
Phase: 6 - Visualization Module Testing
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.plotting import (
    plot_price_with_indicators,
    plot_signals,
    plot_equity_curve,
    plot_drawdown,
    plot_returns_distribution,
    plot_monthly_returns,
    create_backtest_report
)


def create_sample_data(days=100):
    """Create sample OHLCV data for testing"""
    dates = pd.date_range('2023-01-01', periods=days, freq='D')
    np.random.seed(42)
    
    close = 100 + np.cumsum(np.random.randn(days) * 2)
    high = close + np.random.rand(days) * 2
    low = close - np.random.rand(days) * 2
    open_price = close + np.random.randn(days) * 1
    volume = np.random.randint(1000000, 5000000, days)
    
    data = pd.DataFrame({
        'Open': open_price,
        'High': high,
        'Low': low,
        'Close': close,
        'Volume': volume
    }, index=dates)
    
    return data


def create_sample_signals(length=100):
    """Create sample trading signals"""
    dates = pd.date_range('2023-01-01', periods=length, freq='D')
    signals = pd.Series(0, index=dates)  # Initialize with scalar 0, not list
    
    # Add some BUY and SELL signals (only if length allows)
    if length > 10:
        signals.iloc[10] = 1   # BUY
    if length > 20:
        signals.iloc[20] = -1  # SELL
    if length > 30:
        signals.iloc[30] = 1   # BUY
    if length > 50:
        signals.iloc[50] = -1  # SELL
    if length > 70:
        signals.iloc[70] = 1   # BUY
    if length > 90:
        signals.iloc[90] = -1  # SELL
    
    return signals


def create_sample_trades():
    """Create sample trade list"""
    trades = [
        {
            'entry_date': '2023-01-11',
            'exit_date': '2023-01-21',
            'entry_price': 100.0,
            'exit_price': 105.0,
            'shares': 100,
            'return_pct': 5.0,
            'return_abs': 500.0,
            'holding_days': 10
        },
        {
            'entry_date': '2023-01-31',
            'exit_date': '2023-02-20',
            'entry_price': 110.0,
            'exit_price': 108.0,
            'shares': 90,
            'return_pct': -1.82,
            'return_abs': -180.0,
            'holding_days': 20
        },
        {
            'entry_date': '2023-03-12',
            'exit_date': '2023-04-01',
            'entry_price': 115.0,
            'exit_price': 120.0,
            'shares': 86,
            'return_pct': 4.35,
            'return_abs': 430.0,
            'holding_days': 20
        }
    ]
    return trades


def create_sample_equity_curve(days=100, initial_capital=10000):
    """Create sample equity curve"""
    dates = pd.date_range('2023-01-01', periods=days, freq='D')
    np.random.seed(42)
    
    # Simulate growing equity with some volatility
    returns = np.random.randn(days) * 0.02 + 0.001  # 0.1% daily drift
    equity = initial_capital * (1 + returns).cumprod()
    
    return pd.Series(equity, index=dates)


# ==================== TEST FUNCTIONS ====================

def test_01_invalid_data_type():
    """TEST 1: Non-DataFrame input should raise TypeError"""
    import pytest
    with pytest.raises(TypeError):
        fig = plot_price_with_indicators(data="not a dataframe")
def test_02_basic_price_chart():
    """TEST 2: Verify basic price chart generates correctly"""
    print("\nTEST 2: Basic price chart generation")
    
    data = create_sample_data(50)
    fig = plot_price_with_indicators(data=data)
    
    assert isinstance(fig, go.Figure)
    
    assert len(fig.data) > 0
    
    assert isinstance(fig.data[0], go.Candlestick)
    
    print("[PASS] TEST 2 PASSED: Price chart structure correct")


def test_03_signals_chart():
    """TEST 3: Verify signals chart with markers"""
    print("\nTEST 3: Signals chart with BUY/SELL markers")
    
    data = create_sample_data(50)
    signals = create_sample_signals(50)
    
    fig = plot_signals(data=data, signals=signals)
    
    assert isinstance(fig, go.Figure)
    
    marker_traces = [trace for trace in fig.data if isinstance(trace, go.Scatter) and trace.mode == 'markers']
    
    assert len(marker_traces) > 0
    
    print("[PASS] TEST 3 PASSED: Signal markers added correctly")


def test_04_equity_curve():
    """TEST 4: Verify equity curve chart"""
    print("\nTEST 4: Equity curve generation")
    
    equity = create_sample_equity_curve(100, 10000)
    fig = plot_equity_curve(equity_curve=equity, initial_capital=10000)
    
    assert isinstance(fig, go.Figure)
    
    assert len(fig.data) > 0
    
    assert isinstance(fig.data[0], go.Scatter)
    
    print("[PASS] TEST 4 PASSED: Equity curve structure correct")


def test_05_drawdown_chart():
    """TEST 5: Verify drawdown analysis chart"""
    print("\nTEST 5: Drawdown chart generation")
    
    equity = create_sample_equity_curve(100, 10000)
    fig = plot_drawdown(equity_curve=equity)
    
    assert isinstance(fig, go.Figure)
    
    assert len(fig.data) > 0
    
    assert fig.data[0].fill == 'tozeroy'
    
    print("[PASS] TEST 5 PASSED: Drawdown structure correct")


def test_06_returns_distribution():
    """TEST 6: Verify returns distribution histogram"""
    print("\nTEST 6: Returns distribution histogram")
    
    trades = create_sample_trades()
    fig = plot_returns_distribution(trades=trades)
    
    assert isinstance(fig, go.Figure)
    
    histogram_found = any(isinstance(trace, go.Histogram) for trace in fig.data)
    assert histogram_found
    
    print("[PASS] TEST 6 PASSED: Returns histogram structure correct")


def test_07_monthly_heatmap():
    """TEST 7: Verify monthly returns heatmap"""
    print("\nTEST 7: Monthly returns heatmap")
    
    equity = create_sample_equity_curve(365, 10000)  # 1 year
    fig = plot_monthly_returns(equity_curve=equity)
    
    assert isinstance(fig, go.Figure)
    
    assert isinstance(fig.data[0], go.Heatmap)
    
    print("[PASS] TEST 7 PASSED: Monthly heatmap structure correct")


def test_08_complete_report():
    """TEST 8: Verify complete backtest report generation"""
    print("\nTEST 8: Complete backtest report")
    
    data = create_sample_data(100)
    signals = create_sample_signals(100)
    equity = create_sample_equity_curve(100, 10000)
    trades = create_sample_trades()
    
    backtest_results = {
        'trades': trades,
        'equity_curve': equity,
        'daily_positions': pd.Series([0, 1, 1, 0] * 25, index=data.index),
        'metrics': {
            'initial_capital': 10000,
            'final_equity': equity.iloc[-1],
            'total_return': (equity.iloc[-1] - 10000) / 10000 * 100,
            'total_trades': len(trades),
            'win_rate': 66.67,
            'sharpe_ratio': 1.5,
            'max_drawdown': -15.0
        }
    }
    
    figures = create_backtest_report(
        data=data,
        signals=signals,
        backtest_results=backtest_results,
        strategy_name="Test Strategy"
    )
    
    assert isinstance(figures, dict)
    
    expected_keys = ['price_indicators', 'signals', 'equity_curve', 'drawdown', 'returns_distribution', 'monthly_returns']
    missing_keys = [key for key in expected_keys if key not in figures]
    
    assert not missing_keys
    
    for key, fig in figures.items():
        assert isinstance(fig, go.Figure)
    
    print("[PASS] TEST 8 PASSED: Complete report structure correct")


# ==================== TEST RUNNER ====================

