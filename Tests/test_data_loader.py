"""
Quick test script to verify data_loader.py functions
"""

from src.data_loader import fetch_stock_data

# Test 1: Valid ticker (Reliance - Indian stock)
print("=" * 60)
print("TEST 1: Fetching RELIANCE.NS data")
print("=" * 60)
df = fetch_stock_data("RELIANCE.NS", "2023-01-01", "2023-12-31")

if df is not None:
    print(f"\n✅ SUCCESS! Fetched {len(df)} rows")
    print(f"\nFirst 5 rows:")
    print(df.head())
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nData types:\n{df.dtypes}")
else:
    print("\n❌ FAILED: No data retrieved")

# Test 2: Invalid ticker
print("\n" + "=" * 60)
print("TEST 2: Testing invalid ticker (should fail gracefully)")
print("=" * 60)
df_invalid = fetch_stock_data("INVALID.NS", "2025-01-01", "2025-12-03")

if df_invalid is None:
    print("✅ Correctly handled invalid ticker (returned None)")
else:
    print("⚠️ Unexpected: Got data for invalid ticker")

print("\n" + "=" * 60)
print("Testing complete!")
print("=" * 60)
