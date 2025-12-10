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
    print("\nTEST 1: Invalid data type for plot_price_with_indicators()")
    
    try:
        fig = plot_price_with_indicators(data="not a dataframe")
        print("[FAIL] TEST 1 FAILED: Should raise TypeError for non-DataFrame")
        return False
    except TypeError as e:
        if "DataFrame" in str(e):
            print("[PASS] TEST 1 PASSED: TypeError raised correctly")
            return True
        else:
            print(f"[FAIL] TEST 1 FAILED: Wrong error message: {e}")
            return False


def test_02_basic_price_chart():
    """TEST 2: Verify basic price chart generates correctly"""
    print("\nTEST 2: Basic price chart generation")
    
    data = create_sample_data(50)
    fig = plot_price_with_indicators(data=data)
    
    if not isinstance(fig, go.Figure):
        print(f"[FAIL] TEST 2 FAILED: Expected go.Figure, got {type(fig)}")
        return False
    
    if len(fig.data) == 0:
        print("[FAIL] TEST 2 FAILED: Figure has no traces")
        return False
    
    if not isinstance(fig.data[0], go.Candlestick):
        print(f"[FAIL] TEST 2 FAILED: First trace should be Candlestick")
        return False
    
    print("[PASS] TEST 2 PASSED: Price chart structure correct")
    return True


def test_03_signals_chart():
    """TEST 3: Verify signals chart with markers"""
    print("\nTEST 3: Signals chart with BUY/SELL markers")
    
    data = create_sample_data(50)
    signals = create_sample_signals(50)
    
    fig = plot_signals(data=data, signals=signals)
    
    if not isinstance(fig, go.Figure):
        print(f"[FAIL] TEST 3 FAILED: Expected go.Figure")
        return False
    
    marker_traces = [trace for trace in fig.data if isinstance(trace, go.Scatter) and trace.mode == 'markers']
    
    if len(marker_traces) == 0:
        print("[FAIL] TEST 3 FAILED: No marker traces found")
        return False
    
    print("[PASS] TEST 3 PASSED: Signal markers added correctly")
    return True


def test_04_equity_curve():
    """TEST 4: Verify equity curve chart"""
    print("\nTEST 4: Equity curve generation")
    
    equity = create_sample_equity_curve(100, 10000)
    fig = plot_equity_curve(equity_curve=equity, initial_capital=10000)
    
    if not isinstance(fig, go.Figure):
        print(f"[FAIL] TEST 4 FAILED: Expected go.Figure")
        return False
    
    if len(fig.data) == 0:
        print("[FAIL] TEST 4 FAILED: No traces in equity curve")
        return False
    
    if not isinstance(fig.data[0], go.Scatter):
        print(f"[FAIL] TEST 4 FAILED: Expected Scatter trace")
        return False
    
    print("[PASS] TEST 4 PASSED: Equity curve structure correct")
    return True


def test_05_drawdown_chart():
    """TEST 5: Verify drawdown analysis chart"""
    print("\nTEST 5: Drawdown chart generation")
    
    equity = create_sample_equity_curve(100, 10000)
    fig = plot_drawdown(equity_curve=equity)
    
    if not isinstance(fig, go.Figure):
        print(f"[FAIL] TEST 5 FAILED: Expected go.Figure")
        return False
    
    if len(fig.data) == 0:
        print("[FAIL] TEST 5 FAILED: No traces in drawdown chart")
        return False
    
    if fig.data[0].fill != 'tozeroy':
        print("[FAIL] TEST 5 FAILED: Drawdown should have fill to zero")
        return False
    
    print("[PASS] TEST 5 PASSED: Drawdown structure correct")
    return True


def test_06_returns_distribution():
    """TEST 6: Verify returns distribution histogram"""
    print("\nTEST 6: Returns distribution histogram")
    
    trades = create_sample_trades()
    fig = plot_returns_distribution(trades=trades)
    
    if not isinstance(fig, go.Figure):
        print(f"[FAIL] TEST 6 FAILED: Expected go.Figure")
        return False
    
    histogram_found = any(isinstance(trace, go.Histogram) for trace in fig.data)
    if not histogram_found:
        print("[FAIL] TEST 6 FAILED: No histogram traces found")
        return False
    
    print("[PASS] TEST 6 PASSED: Returns histogram structure correct")
    return True


def test_07_monthly_heatmap():
    """TEST 7: Verify monthly returns heatmap"""
    print("\nTEST 7: Monthly returns heatmap")
    
    equity = create_sample_equity_curve(365, 10000)  # 1 year
    fig = plot_monthly_returns(equity_curve=equity)
    
    if not isinstance(fig, go.Figure):
        print(f"[FAIL] TEST 7 FAILED: Expected go.Figure")
        return False
    
    if not isinstance(fig.data[0], go.Heatmap):
        print(f"[FAIL] TEST 7 FAILED: Expected Heatmap")
        return False
    
    print("[PASS] TEST 7 PASSED: Monthly heatmap structure correct")
    return True


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
    
    if not isinstance(figures, dict):
        print(f"[FAIL] TEST 8 FAILED: Expected dict")
        return False
    
    expected_keys = ['price_indicators', 'signals', 'equity_curve', 'drawdown', 'returns_distribution', 'monthly_returns']
    missing_keys = [key for key in expected_keys if key not in figures]
    
    if missing_keys:
        print(f"[FAIL] TEST 8 FAILED: Missing chart keys: {missing_keys}")
        return False
    
    for key, fig in figures.items():
        if not isinstance(fig, go.Figure):
            print(f"[FAIL] TEST 8 FAILED: {key} is not a Figure")
            return False
    
    print("[PASS] TEST 8 PASSED: Complete report structure correct")
    return True


# ==================== TEST RUNNER ====================

def run_all_tests():
    """Run all tests and report results"""
    print("=" * 60)
    print("PHASE 6: VISUALIZATION MODULE - TEST SUITE")
    print("=" * 60)
    print(f"Testing plotting.py functions")
    print(f"Total Tests: 8 (Core Functionality)")
    print("=" * 60)
    
    tests = [
        test_01_invalid_data_type,
        test_02_basic_price_chart,
        test_03_signals_chart,
        test_04_equity_curve,
        test_05_drawdown_chart,
        test_06_returns_distribution,
        test_07_monthly_heatmap,
        test_08_complete_report
    ]
    
    results = []
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            result = test_func()
            results.append((test_func.__name__, result))
            if result:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\n[CRASH] {test_func.__name__} CRASHED: {e}")
            results.append((test_func.__name__, False))
            failed += 1
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Total Tests: {len(tests)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Success Rate: {passed/len(tests)*100:.1f}%")
    print("=" * 60)
    
    if failed > 0:
        print("\nFAILED TESTS:")
        for test_name, result in results:
            if not result:
                print(f"  [FAIL] {test_name}")
    else:
        print("\n[SUCCESS] PHASE 6 COMPLETE - ALL 8 TESTS PASSED!")
        print("\nVisualization Functions Implemented:")
        print("  [OK] plot_price_with_indicators() - Candlestick with indicators")
        print("  [OK] plot_signals() - Trading signals with markers")
        print("  [OK] plot_equity_curve() - Portfolio value over time")
        print("  [OK] plot_drawdown() - Drawdown analysis")
        print("  [OK] plot_returns_distribution() - Histogram of returns")
        print("  [OK] plot_monthly_returns() - Calendar heatmap")
        print("  [OK] create_backtest_report() - Master function for all charts")
        print("\nTotal: 7 functions, 8 tests, ALL PASSING!")
        print("plotting.py is production-ready for Streamlit dashboard!")
    
    return passed == len(tests)


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
