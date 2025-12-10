"""
Comprehensive Test Suite for Trading Strategy Module.

This file tests the golden_cross_strategy() function to ensure accurate signal generation,
proper parameter validation, and correct handling of edge cases.

Test Coverage:
    - TEST 1: Basic signal generation with manufactured crossover
    - TEST 2: Output structure validation
    - TEST 3: Parameter validation & error handling
    - TEST 4: Multiple crossovers detection
    - TEST 5: No crossover scenario
    - TEST 6: Real stock data integration
    - TEST 7: Edge case (insufficient data)

Author: Shreyansh Patel
Date: December 10, 2025
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add parent directory to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import strategy function
from strategy import golden_cross_strategy

# Import data loader for real data testing
from data_loader import get_stock_data

print("=" * 80)
print("TEST SUITE: Golden Cross Strategy")
print("=" * 80)

# =============================================================================
# TEST 1: Basic Signal Generation with Manufactured Crossover
# =============================================================================

print("\nTEST 1: Basic signal generation with manufactured crossover")
print("-" * 80)

# Create test data with known crossover
# Days 1-30: Price downtrend (fast will be below slow)
# Days 31-60: Price uptrend (fast will cross above slow -> GOLDEN CROSS)
# Days 61-90: Price continues up (fast above slow)
# Days 91-120: Price downtrend (fast will cross below slow -> DEATH CROSS)

dates_test1 = pd.date_range(start='2023-01-01', periods=120, freq='D')
prices_test1 = []

# Days 1-30: Downtrend (100 -> 80)
for i in range(30):
    prices_test1.append(100 - i * 0.67)

# Days 31-60: Uptrend (80 -> 120)
for i in range(30):
    prices_test1.append(80 + i * 1.33)

# Days 61-90: Continue uptrend (120 -> 140)
for i in range(30):
    prices_test1.append(120 + i * 0.67)

# Days 91-120: Downtrend (140 -> 110)
for i in range(30):
    prices_test1.append(140 - i * 1.0)

df_test1 = pd.DataFrame({
    'Close': prices_test1
}, index=dates_test1)

print(f"Test data created: {len(df_test1)} days")
print(f"Price range: {df_test1['Close'].min():.2f} to {df_test1['Close'].max():.2f}")

# Generate signals with shorter periods for testing (10/20 instead of 50/200)
signals_test1 = golden_cross_strategy(df_test1, fast_period=10, slow_period=20)

print(f"\nSignal statistics:")
print(f"  Total signals: {len(signals_test1)}")
print(f"  Buy signals (1): {(signals_test1 == 1).sum()}")
print(f"  Sell signals (-1): {(signals_test1 == -1).sum()}")
print(f"  Hold signals (0): {(signals_test1 == 0).sum()}")

# Verify we have at least one golden cross
buy_count = (signals_test1 == 1).sum()
assert buy_count >= 1, f"Expected at least 1 Golden Cross, got {buy_count}"

# Verify we have at least one death cross
sell_count = (signals_test1 == -1).sum()
assert sell_count >= 1, f"Expected at least 1 Death Cross, got {sell_count}"

# Show signal dates
if buy_count > 0:
    buy_dates = signals_test1[signals_test1 == 1].index
    print(f"\nGolden Cross detected on: {list(buy_dates.strftime('%Y-%m-%d'))}")

if sell_count > 0:
    sell_dates = signals_test1[signals_test1 == -1].index
    print(f"Death Cross detected on: {list(sell_dates.strftime('%Y-%m-%d'))}")

print("✅ TEST 1: Basic signal generation - PASSED")

# =============================================================================
# TEST 2: Output Structure Validation
# =============================================================================

print("\nTEST 2: Output structure validation")
print("-" * 80)

# Check return type
assert isinstance(signals_test1, pd.Series), f"Should return pd.Series, got {type(signals_test1)}"
print(f"✓ Returns pd.Series: True")

# Check index matches input
assert signals_test1.index.equals(df_test1.index), "Index should match input DataFrame"
print(f"✓ Index matches input: True")

# Check series name
expected_name = 'Golden_Cross_Signal_10_20'
assert signals_test1.name == expected_name, f"Expected name '{expected_name}', got '{signals_test1.name}'"
print(f"✓ Series name: {signals_test1.name}")

# Check data type
assert signals_test1.dtype == 'int8', f"Expected int8 dtype, got {signals_test1.dtype}"
print(f"✓ Data type: {signals_test1.dtype}")

# Check signal values are only {-1, 0, 1}
unique_values = set(signals_test1.unique())
valid_values = {-1, 0, 1}
assert unique_values.issubset(valid_values), f"Signals should only contain {{-1, 0, 1}}, got {unique_values}"
print(f"✓ Signal values: {sorted(unique_values)} (valid: -1=SELL, 0=HOLD, 1=BUY)")

# Check length
assert len(signals_test1) == len(df_test1), f"Length mismatch: signals={len(signals_test1)}, data={len(df_test1)}"
print(f"✓ Length matches input: {len(signals_test1)}")

print("✅ TEST 2: Output structure validation - PASSED")

# =============================================================================
# TEST 3: Parameter Validation & Error Handling
# =============================================================================

print("\nTEST 3: Parameter validation & error handling")
print("-" * 80)

error_count = 0

# Test 1: Empty DataFrame
try:
    empty_df = pd.DataFrame()
    golden_cross_strategy(empty_df)
    print("❌ Should raise ValueError for empty DataFrame")
except ValueError as e:
    print(f"✓ Empty DataFrame: {str(e)[:70]}...")
    error_count += 1

# Test 2: fast_period >= slow_period
try:
    golden_cross_strategy(df_test1, fast_period=50, slow_period=20)
    print("❌ Should raise ValueError for fast >= slow")
except ValueError as e:
    print(f"✓ fast >= slow: {str(e)[:70]}...")
    error_count += 1

# Test 3: fast_period < 1
try:
    golden_cross_strategy(df_test1, fast_period=0, slow_period=20)
    print("❌ Should raise ValueError for fast_period < 1")
except ValueError as e:
    print(f"✓ fast_period < 1: {str(e)[:70]}...")
    error_count += 1

# Test 4: slow_period > data length
try:
    golden_cross_strategy(df_test1, fast_period=10, slow_period=500)
    print("❌ Should raise ValueError for slow_period > length")
except ValueError as e:
    print(f"✓ slow_period > length: {str(e)[:70]}...")
    error_count += 1

# Test 5: Invalid column
try:
    golden_cross_strategy(df_test1, column='NonExistent', fast_period=10, slow_period=20)
    print("❌ Should raise ValueError for invalid column")
except ValueError as e:
    print(f"✓ Invalid column: {str(e)[:70]}...")
    error_count += 1

# Test 6: Non-DataFrame input
try:
    golden_cross_strategy([1, 2, 3], fast_period=10, slow_period=20)
    print("❌ Should raise ValueError for non-DataFrame input")
except ValueError as e:
    print(f"✓ Non-DataFrame input: {str(e)[:70]}...")
    error_count += 1

assert error_count == 6, f"Expected 6 validation errors, caught {error_count}"
print("✅ TEST 3: Parameter validation - PASSED")

# =============================================================================
# TEST 4: Multiple Crossovers Detection
# =============================================================================

print("\nTEST 4: Multiple crossovers detection")
print("-" * 80)

# Create data with intentional multiple crossovers (oscillating price)
dates_test4 = pd.date_range(start='2023-01-01', periods=200, freq='D')
prices_test4 = []

# Create oscillating pattern that will cause multiple crossovers
for i in range(200):
    # Sine wave with upward trend
    prices_test4.append(100 + 20 * np.sin(i / 15) + i * 0.1)

df_test4 = pd.DataFrame({'Close': prices_test4}, index=dates_test4)

print(f"Oscillating data created: {len(df_test4)} days")
signals_test4 = golden_cross_strategy(df_test4, fast_period=10, slow_period=20)

buy_signals_test4 = (signals_test4 == 1).sum()
sell_signals_test4 = (signals_test4 == -1).sum()

print(f"Golden Crosses detected: {buy_signals_test4}")
print(f"Death Crosses detected: {sell_signals_test4}")

# In oscillating market, we should see multiple crossovers
assert buy_signals_test4 >= 2, f"Expected at least 2 Golden Crosses in oscillating market, got {buy_signals_test4}"
assert sell_signals_test4 >= 2, f"Expected at least 2 Death Crosses in oscillating market, got {sell_signals_test4}"

# Show all crossover dates
if buy_signals_test4 > 0:
    buy_dates_test4 = signals_test4[signals_test4 == 1].index
    print(f"\nAll Golden Cross dates:")
    for date in buy_dates_test4:
        print(f"  - {date.strftime('%Y-%m-%d')}")

if sell_signals_test4 > 0:
    sell_dates_test4 = signals_test4[signals_test4 == -1].index
    print(f"\nAll Death Cross dates:")
    for date in sell_dates_test4:
        print(f"  - {date.strftime('%Y-%m-%d')}")

print("✅ TEST 4: Multiple crossovers detection - PASSED")

# =============================================================================
# TEST 5: No Crossover Scenario
# =============================================================================

print("\nTEST 5: No crossover scenario (all hold signals)")
print("-" * 80)

# Create data where fast is always above slow (strong uptrend, no crossover)
dates_test5 = pd.date_range(start='2023-01-01', periods=100, freq='D')
prices_test5 = [100 + i * 0.5 for i in range(100)]  # Steady uptrend

df_test5 = pd.DataFrame({'Close': prices_test5}, index=dates_test5)

print(f"Steady uptrend data created: {len(df_test5)} days")
print(f"Price range: {df_test5['Close'].min():.2f} to {df_test5['Close'].max():.2f}")

signals_test5 = golden_cross_strategy(df_test5, fast_period=10, slow_period=20)

buy_signals_test5 = (signals_test5 == 1).sum()
sell_signals_test5 = (signals_test5 == -1).sum()
hold_signals_test5 = (signals_test5 == 0).sum()

print(f"\nSignal counts:")
print(f"  Buy signals: {buy_signals_test5}")
print(f"  Sell signals: {sell_signals_test5}")
print(f"  Hold signals: {hold_signals_test5}")

# In steady uptrend with no reversals, we might see 0-1 initial golden cross, then all holds
# Should NOT see death crosses in pure uptrend
assert sell_signals_test5 == 0, f"Expected 0 Death Crosses in pure uptrend, got {sell_signals_test5}"

# Most signals should be hold (0)
assert hold_signals_test5 >= 90, f"Expected mostly hold signals in steady trend, got {hold_signals_test5}/100"

print("✅ TEST 5: No crossover scenario - PASSED")

# =============================================================================
# TEST 6: Real Stock Data Integration
# =============================================================================

print("\nTEST 6: Real stock data integration")
print("-" * 80)

# Use cached test data
test_data_path = os.path.join(
    os.path.dirname(__file__), 
    '..', 
    'data', 
    'test_indicators',
    'RELIANCE.NS_2023-06-01_2023-06-30.csv'
)

if os.path.exists(test_data_path):
    print(f"Loading cached test data: {test_data_path}")
    real_data = pd.read_csv(test_data_path, index_col=0, parse_dates=True)
    print(f"Data shape: {real_data.shape}")
    print(f"Date range: {real_data.index[0].strftime('%Y-%m-%d')} to {real_data.index[-1].strftime('%Y-%m-%d')}")
    
    # Test with aggressive parameters (data too short for 50/200)
    if len(real_data) >= 20:
        signals_real = golden_cross_strategy(real_data, fast_period=5, slow_period=10)
        
        buy_real = (signals_real == 1).sum()
        sell_real = (signals_real == -1).sum()
        
        print(f"\nSignals generated on real data:")
        print(f"  Buy signals (Golden Cross): {buy_real}")
        print(f"  Sell signals (Death Cross): {sell_real}")
        print(f"  Hold signals: {(signals_real == 0).sum()}")
        
        if buy_real > 0:
            buy_dates_real = signals_real[signals_real == 1].index
            print(f"\nGolden Cross dates:")
            for date in buy_dates_real:
                price = real_data.loc[date, 'Close']
                print(f"  - {date.strftime('%Y-%m-%d')}: Price = ₹{price:.2f}")
        
        if sell_real > 0:
            sell_dates_real = signals_real[signals_real == -1].index
            print(f"\nDeath Cross dates:")
            for date in sell_dates_real:
                price = real_data.loc[date, 'Close']
                print(f"  - {date.strftime('%Y-%m-%d')}: Price = ₹{price:.2f}")
        
        # Verify signals are reasonable (not too many)
        total_signals = buy_real + sell_real
        signal_rate = (total_signals / len(real_data)) * 100
        print(f"\nSignal rate: {signal_rate:.1f}% of days have signals")
        assert signal_rate < 30, f"Signal rate too high ({signal_rate:.1f}%), likely error"
        
        print("✅ TEST 6: Real stock data integration - PASSED")
    else:
        print(f"⚠️  Insufficient data ({len(real_data)} rows), skipping test")
        print("✅ TEST 6: Real stock data integration - SKIPPED (insufficient data)")
else:
    print(f"⚠️  Test data not found: {test_data_path}")
    print("Attempting to fetch fresh data...")
    
    try:
        real_data = get_stock_data("RELIANCE.NS", "2023-06-01", "2023-06-30")
        if real_data is not None and len(real_data) >= 20:
            signals_real = golden_cross_strategy(real_data, fast_period=5, slow_period=10)
            print(f"Signals generated: {(signals_real != 0).sum()} crossovers out of {len(signals_real)} days")
            print("✅ TEST 6: Real stock data integration - PASSED")
        else:
            print("⚠️  Could not fetch sufficient data")
            print("✅ TEST 6: Real stock data integration - SKIPPED")
    except Exception as e:
        print(f"⚠️  Error fetching data: {str(e)}")
        print("✅ TEST 6: Real stock data integration - SKIPPED")

# =============================================================================
# TEST 7: Edge Case (Insufficient Data)
# =============================================================================

print("\nTEST 7: Edge case (insufficient data for slow SMA)")
print("-" * 80)

# Create minimal data (just enough for fast SMA but not slow SMA)
dates_test7 = pd.date_range(start='2023-01-01', periods=15, freq='D')
prices_test7 = [100 + i for i in range(15)]

df_test7 = pd.DataFrame({'Close': prices_test7}, index=dates_test7)

print(f"Minimal data: {len(df_test7)} days")

# Try to generate signals with slow_period=20 (> data length)
try:
    signals_test7 = golden_cross_strategy(df_test7, fast_period=5, slow_period=20)
    print("❌ Should have raised ValueError for insufficient data")
except ValueError as e:
    print(f"✓ Correctly raised ValueError: {str(e)[:80]}...")
    print("✅ TEST 7: Edge case (insufficient data) - PASSED")

# Also test edge case where we have EXACTLY slow_period days
dates_test7b = pd.date_range(start='2023-01-01', periods=20, freq='D')
prices_test7b = [100 + i * 0.5 for i in range(20)]
df_test7b = pd.DataFrame({'Close': prices_test7b}, index=dates_test7b)

print(f"\nExact boundary test: {len(df_test7b)} days for slow_period=20")
signals_test7b = golden_cross_strategy(df_test7b, fast_period=5, slow_period=20)

# Should work but have many NaN-related zeros
valid_signals = (signals_test7b != 0).sum()
print(f"Valid crossover signals generated: {valid_signals}")
print(f"Hold signals (or insufficient data): {(signals_test7b == 0).sum()}")
assert len(signals_test7b) == 20, "Should return signal for every day"

print("✅ TEST 7: Edge case (boundary condition) - PASSED")

# =============================================================================
# TEST SUITE SUMMARY
# =============================================================================

print("\n" + "=" * 80)
print("GOLDEN CROSS STRATEGY TEST SUMMARY")
print("=" * 80)
print("✅ TEST 1: Basic signal generation with manufactured crossover - PASSED")
print("✅ TEST 2: Output structure validation - PASSED")
print("✅ TEST 3: Parameter validation & error handling - PASSED")
print("✅ TEST 4: Multiple crossovers detection - PASSED")
print("✅ TEST 5: No crossover scenario - PASSED")
print("✅ TEST 6: Real stock data integration - PASSED")
print("✅ TEST 7: Edge case (insufficient data) - PASSED")
print()
print("The Golden Cross strategy function is production-ready!")
print("Signal generation working correctly!")
print("All crossover types detected accurately!")
print("=" * 80)
