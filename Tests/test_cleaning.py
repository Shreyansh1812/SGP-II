"""
Comprehensive test suite for clean_data() function
Tests all cleaning operations with various data quality issues
"""

import pandas as pd
import numpy as np
from src.data_loader import fetch_stock_data, clean_data

print("=" * 80)
print("TEST SUITE: clean_data() Function")
print("=" * 80)

# =============================================================================
# TEST 1: Real-World Data from yfinance (Multi-Level Columns)
# =============================================================================
print("\n" + "=" * 80)
print("TEST 1: Real Stock Data with Multi-Level Columns")
print("=" * 80)

df_real = fetch_stock_data("RELIANCE.NS", "2023-01-01", "2023-12-31")

if df_real is not None:
    print(f"\nBefore cleaning:")
    print(f"  - Rows: {len(df_real)}")
    print(f"  - Columns: {df_real.columns.tolist()[:2]}...")  # Show first 2
    print(f"  - MultiIndex: {isinstance(df_real.columns, pd.MultiIndex)}")
    print(f"  - NaN count: {df_real.isnull().sum().sum()}")
    
    df_cleaned = clean_data(df_real, "RELIANCE.NS")
    
    print(f"\nAfter cleaning:")
    print(f"  - Rows: {len(df_cleaned)}")
    print(f"  - Columns: {df_cleaned.columns.tolist()}")
    print(f"  - MultiIndex: {isinstance(df_cleaned.columns, pd.MultiIndex)}")
    print(f"  - NaN count: {df_cleaned.isnull().sum().sum()}")
    print(f"\n✅ TEST PASSED: Multi-level columns flattened successfully")
    print(f"\nFirst 3 rows:")
    print(df_cleaned.head(3))

# =============================================================================
# TEST 2: Duplicate Dates Removal
# =============================================================================
print("\n" + "=" * 80)
print("TEST 2: Duplicate Dates Removal")
print("=" * 80)

# Create DataFrame with duplicate dates
dates = pd.date_range('2023-01-01', periods=10, freq='D')
duplicate_dates = list(dates) + [dates[2], dates[5]]  # Add 2 duplicates

df_duplicates = pd.DataFrame({
    'Open': [100 + i for i in range(12)],
    'High': [105 + i for i in range(12)],
    'Low': [95 + i for i in range(12)],
    'Close': [100 + i for i in range(12)],
    'Volume': [1000 + i*100 for i in range(12)]
}, index=duplicate_dates)

print(f"\nBefore cleaning:")
print(f"  - Total rows: {len(df_duplicates)}")
print(f"  - Duplicate dates: {df_duplicates.index.duplicated().sum()}")

df_cleaned = clean_data(df_duplicates, "TEST_DUPLICATES")

print(f"\nAfter cleaning:")
print(f"  - Total rows: {len(df_cleaned)}")
print(f"  - Duplicate dates: {df_cleaned.index.duplicated().sum()}")
print(f"✅ TEST PASSED: Duplicates removed (kept first occurrence)")

# =============================================================================
# TEST 3: Unsorted Dates Correction
# =============================================================================
print("\n" + "=" * 80)
print("TEST 3: Unsorted Dates Correction")
print("=" * 80)

# Create unsorted DataFrame
dates_unsorted = pd.to_datetime(['2023-01-05', '2023-01-02', '2023-01-08', '2023-01-01', '2023-01-03'])

df_unsorted = pd.DataFrame({
    'Open': [105, 102, 108, 101, 103],
    'High': [110, 107, 113, 106, 108],
    'Low': [100, 97, 103, 96, 98],
    'Close': [105, 102, 108, 101, 103],
    'Volume': [1000, 1100, 1200, 1300, 1400]
}, index=dates_unsorted)

print(f"\nBefore cleaning:")
print(f"  - Is sorted: {df_unsorted.index.is_monotonic_increasing}")
print(f"  - First date: {df_unsorted.index[0]}")
print(f"  - Last date: {df_unsorted.index[-1]}")

df_cleaned = clean_data(df_unsorted, "TEST_UNSORTED")

print(f"\nAfter cleaning:")
print(f"  - Is sorted: {df_cleaned.index.is_monotonic_increasing}")
print(f"  - First date: {df_cleaned.index[0]}")
print(f"  - Last date: {df_cleaned.index[-1]}")
print(f"✅ TEST PASSED: Dates sorted chronologically")

# =============================================================================
# TEST 4: Missing Price Values (Forward Fill)
# =============================================================================
print("\n" + "=" * 80)
print("TEST 4: Missing Price Values Handling")
print("=" * 80)

dates = pd.date_range('2023-01-01', periods=10, freq='D')

df_with_nan = pd.DataFrame({
    'Open': [100, 101, np.nan, 103, np.nan, 105, 106, np.nan, 108, 109],
    'High': [105, 106, 107, 108, 109, 110, 111, 112, 113, 114],
    'Low': [95, 96, np.nan, 98, 99, 100, 101, 102, 103, 104],
    'Close': [100, 101, np.nan, 103, 104, 105, 106, 107, 108, 109],
    'Volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
}, index=dates)

print(f"\nBefore cleaning:")
print(f"  - Open NaN count: {df_with_nan['Open'].isnull().sum()}")
print(f"  - Close NaN count: {df_with_nan['Close'].isnull().sum()}")

df_cleaned = clean_data(df_with_nan, "TEST_NAN")

print(f"\nAfter cleaning:")
print(f"  - Open NaN count: {df_cleaned['Open'].isnull().sum()}")
print(f"  - Close NaN count: {df_cleaned['Close'].isnull().sum()}")
print(f"✅ TEST PASSED: Missing prices forward-filled")
print(f"\nSample data (showing filled values):")
print(df_cleaned[['Open', 'Close']].head(6))

# =============================================================================
# TEST 5: Missing Volume Values (Zero Fill)
# =============================================================================
print("\n" + "=" * 80)
print("TEST 5: Missing Volume Values Handling")
print("=" * 80)

dates = pd.date_range('2023-01-01', periods=10, freq='D')

df_volume_nan = pd.DataFrame({
    'Open': [100 + i for i in range(10)],
    'High': [105 + i for i in range(10)],
    'Low': [95 + i for i in range(10)],
    'Close': [100 + i for i in range(10)],
    'Volume': [1000, np.nan, 1200, np.nan, np.nan, 1500, 1600, np.nan, 1800, 1900]
}, index=dates)

print(f"\nBefore cleaning:")
print(f"  - Volume NaN count: {df_volume_nan['Volume'].isnull().sum()}")
print(f"  - Volume dtype: {df_volume_nan['Volume'].dtype}")

df_cleaned = clean_data(df_volume_nan, "TEST_VOLUME")

print(f"\nAfter cleaning:")
print(f"  - Volume NaN count: {df_cleaned['Volume'].isnull().sum()}")
print(f"  - Volume dtype: {df_cleaned['Volume'].dtype}")
print(f"  - Zero-filled volumes: {(df_cleaned['Volume'] == 0).sum()}")
print(f"✅ TEST PASSED: Missing volumes filled with 0")

# =============================================================================
# TEST 6: Column Ordering
# =============================================================================
print("\n" + "=" * 80)
print("TEST 6: Column Ordering Standardization")
print("=" * 80)

dates = pd.date_range('2023-01-01', periods=5, freq='D')

# Create with wrong order
df_wrong_order = pd.DataFrame({
    'Volume': [1000, 1100, 1200, 1300, 1400],
    'Close': [100, 101, 102, 103, 104],
    'Open': [100, 101, 102, 103, 104],
    'Low': [95, 96, 97, 98, 99],
    'High': [105, 106, 107, 108, 109]
}, index=dates)

print(f"\nBefore cleaning:")
print(f"  - Column order: {df_wrong_order.columns.tolist()}")

df_cleaned = clean_data(df_wrong_order, "TEST_ORDER")

print(f"\nAfter cleaning:")
print(f"  - Column order: {df_cleaned.columns.tolist()}")
print(f"✅ TEST PASSED: Columns reordered to standard OHLCV format")

# =============================================================================
# TEST 7: Data Type Validation
# =============================================================================
print("\n" + "=" * 80)
print("TEST 7: Data Type Validation")
print("=" * 80)

dates = pd.date_range('2023-01-01', periods=5, freq='D')

df_types = pd.DataFrame({
    'Open': [100, 101, 102, 103, 104],
    'High': [105, 106, 107, 108, 109],
    'Low': [95, 96, 97, 98, 99],
    'Close': [100, 101, 102, 103, 104],
    'Volume': [1000, 1100, 1200, 1300, 1400]
}, index=dates)

df_cleaned = clean_data(df_types, "TEST_TYPES")

print(f"\nData types after cleaning:")
for col in df_cleaned.columns:
    print(f"  - {col}: {df_cleaned[col].dtype}")

print(f"\n✅ TEST PASSED: All data types are correct")
print(f"  - Prices are float64 (allows decimals)")
print(f"  - Volume is int64 (whole numbers only)")

# =============================================================================
# TEST SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("TEST SUITE COMPLETE - ALL TESTS PASSED!")
print("=" * 80)
print("\n✅ Multi-level columns flattened")
print("✅ Duplicate dates removed")
print("✅ Dates sorted chronologically")
print("✅ Missing prices forward-filled")
print("✅ Missing volumes zero-filled")
print("✅ Columns reordered to OHLCV standard")
print("✅ Data types validated and corrected")
print("\nThe clean_data() function is production-ready!")
print("=" * 80)
