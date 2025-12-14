"""
Test Yahoo Finance API Connection (UPDATED)
===========================================
Updated script to verify yfinance connection and data integrity.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

print("="*60)
print("YAHOO FINANCE API TEST (UPDATED)")
print("="*60)
print(f"yfinance version: {yf.__version__}")
print()

# Test configuration
test_ticker = "AAPL"
end_date = datetime.now()
start_date = end_date - timedelta(days=30)  # Last 30 days

print(f"Test Parameters:")
print(f"  Ticker: {test_ticker}")
print(f"  Start Date: {start_date.strftime('%Y-%m-%d')}")
print(f"  End Date: {end_date.strftime('%Y-%m-%d')}")
print()

# ---------------------------------------------------------
# Test 1: Using yf.download() with MultiIndex Handling
# ---------------------------------------------------------
print("-" * 60)
print("TEST 1: Using yf.download() (Bulk Download)")
print("-" * 60)
try:
    print("Attempting to download data...")
    # Explicitly set auto_adjust=True to silence warnings and get usable price
    data1 = yf.download(
        test_ticker,
        start=start_date.strftime('%Y-%m-%d'),
        end=end_date.strftime('%Y-%m-%d'),
        progress=False,
        auto_adjust=True
    )
    
    if not data1.empty:
        # FIX: Handle MultiIndex columns (e.g., ('Close', 'AAPL') -> 'Close')
        if isinstance(data1.columns, pd.MultiIndex):
            print("   ℹ️  Note: Flattening MultiIndex columns...")
            data1.columns = data1.columns.get_level_values(0)

        print(f"✅ SUCCESS! Downloaded {len(data1)} rows")
        print(f"\nFirst 5 rows (Cleaned):")
        print(data1.head())
        print(f"\nColumns: {list(data1.columns)}")
    else:
        print("❌ FAILED: No data returned (empty DataFrame)")
except Exception as e:
    print(f"❌ ERROR: {str(e)}")

print()

# ---------------------------------------------------------
# Test 2: Using Ticker.history()
# ---------------------------------------------------------
print("-" * 60)
print("TEST 2: Using Ticker.history() (Object Oriented)")
print("-" * 60)
try:
    print("Creating Ticker object...")
    stock = yf.Ticker(test_ticker)
    
    print("Fetching history...")
    data2 = stock.history(
        start=start_date.strftime('%Y-%m-%d'),
        end=end_date.strftime('%Y-%m-%d'),
        auto_adjust=True
    )
    
    if not data2.empty:
        print(f"✅ SUCCESS! Downloaded {len(data2)} rows")
        print(f"\nFirst 5 rows:")
        print(data2.head())
    else:
        print("❌ FAILED: No data returned (empty DataFrame)")
except Exception as e:
    print(f"❌ ERROR: {str(e)}")

print()

# ---------------------------------------------------------
# Test 3: Production Integrity Check
# ---------------------------------------------------------
print("-" * 60)
print("TEST 3: Data Integrity & Validation (Production Check)")
print("-" * 60)
try:
    # We use the data from Test 2 (stock object) to validte logic
    if data2.empty:
        print("⚠️ SKIPPING: Cannot validate data because Test 2 failed.")
    else:
        print("Running logic checks on fetched data...")
        
        # 1. Price Reality Check
        latest_price = data2['Close'].iloc[-1]
        print(f"   Latest Close Price: ${latest_price:.2f}")
        
        if latest_price < 1 or latest_price > 10000:
            print("   ⚠️ WARNING: Price looks suspicious for AAPL.")
        else:
            print("   ✅ Price range seems reasonable.")
            
        # 2. Volume Reality Check
        avg_volume = data2['Volume'].mean()
        print(f"   Average Volume: {avg_volume:,.0f}")
        
        if avg_volume < 1000:
             print("   ⚠️ WARNING: Volume is dangerously low (illiquid?)")
        else:
             print("   ✅ Volume indicates a liquid market.")

        # 3. Date Check
        last_date = data2.index[-1]
        print(f"   Last Data Point: {last_date}")

except Exception as e:
    print(f"❌ ERROR: {str(e)}")

print()
print("="*60)
print("TEST SUMMARY")
print("="*60)
print("If Test 1 & 2 passed, you are ready for SGP-II.")
print("Note: Custom Session logic was removed as it conflicts with yfinance v0.2.50+")
print("="*60)