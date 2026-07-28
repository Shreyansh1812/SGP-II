"""
Tests/test_database.py
Unit and Integration tests for the SGP-II database schema.
Verifies table creation, data insertion, ON CONFLICT constraints (UPSERT),
data type affinity, and clean teardown.
"""

import sqlite3
import os
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ensure base path is added
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SCHEMA_PATH = os.path.join(BASE_DIR, 'database', 'schema.sql')
TEST_DB_PATH = os.path.join(BASE_DIR, 'data', 'test_sandbox.db')

print("=" * 80)
print("TEST SUITE: Database Migration & Verification (Step 1)")
print("=" * 80)

def init_test_db(db_path):
    """Initializes the database schema using schema.sql."""
    # Ensure directory exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    # Connect and run schema
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    with open(SCHEMA_PATH, 'r') as f:
        schema_sql = f.read()
        
    cursor.executescript(schema_sql)
    conn.commit()
    conn.close()
    logging.info(f"Test database initialized at {db_path}")

def run_tests():
    # Remove existing test DB if any
    if os.path.exists(TEST_DB_PATH):
        try:
            os.remove(TEST_DB_PATH)
        except OSError as e:
            logging.warning(f"Could not remove existing test DB: {e}")

    conn = None
    try:
        # Initialize
        init_test_db(TEST_DB_PATH)
        
        # Connect to run test queries
        conn = sqlite3.connect(TEST_DB_PATH)
        cursor = conn.cursor()
        
        # ---------------------------------------------------------
        # TEST 1: Table Creation Check
        # ---------------------------------------------------------
        print("\nTEST 1: Table Creation Check")
        print("-" * 60)
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        print(f"Detected tables in DB: {tables}")
        
        required_tables = ['company_fundamentals', 'daily_sentiment', 'daily_recommendations']
        for table in required_tables:
            assert table in tables, f"Table '{table}' was not created!"
        print("PASS: All tables verified successfully.")

        # ---------------------------------------------------------
        # TEST 2: Data Insertion & Data Type Integrity
        # ---------------------------------------------------------
        print("\nTEST 2: Data Insertion & Type Integrity")
        print("-" * 60)
        
        # Insert into company_fundamentals
        cursor.execute("""
            INSERT INTO company_fundamentals (ticker, pe_ratio, debt_to_equity, operating_margin, roe, is_healthy)
            VALUES (?, ?, ?, ?, ?, ?)
        """, ('AAPL', 28.50, 1.45, 0.302, 1.54, 1))
        
        # Insert into daily_sentiment
        cursor.execute("""
            INSERT INTO daily_sentiment (date, sentiment_score, sentiment_label, summary)
            VALUES (?, ?, ?, ?)
        """, ('2026-07-18', 0.450, 'bullish', 'Strong tech earnings guidance and macro trends.'))
        
        # Insert into daily_recommendations
        cursor.execute("""
            INSERT INTO daily_recommendations (date, ticker, signal_type, prob_buy, prob_sell, prob_hold, sentiment_score, is_sentiment_passed, execution_price, rationale)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, ('2026-07-18', 'AAPL', 'BUY', 0.742, 0.103, 0.155, 0.450, 1, 214.30, 'Strong metrics aligned with positive sentiment.'))
        
        conn.commit()
        
        # Retrieve and verify types
        cursor.execute("SELECT pe_ratio, debt_to_equity, is_healthy FROM company_fundamentals WHERE ticker='AAPL'")
        pe, debt, healthy = cursor.fetchone()
        assert isinstance(pe, float), f"Expected float for pe_ratio, got {type(pe)}"
        assert isinstance(debt, float), f"Expected float for debt_to_equity, got {type(debt)}"
        assert isinstance(healthy, int), f"Expected int for is_healthy, got {type(healthy)}"
        assert healthy == 1, "Expected is_healthy to be 1"
        
        cursor.execute("SELECT sentiment_score FROM daily_sentiment WHERE date='2026-07-18'")
        score = cursor.fetchone()[0]
        assert isinstance(score, float), f"Expected float for sentiment_score, got {type(score)}"
        
        cursor.execute("SELECT prob_buy, execution_price FROM daily_recommendations WHERE date='2026-07-18' AND ticker='AAPL'")
        prob_buy, price = cursor.fetchone()
        assert isinstance(prob_buy, float), f"Expected float for prob_buy, got {type(prob_buy)}"
        assert isinstance(price, float), f"Expected float for execution_price, got {type(price)}"
        
        print("PASS: Data insertion successful and field types conform to schema.")

        # ---------------------------------------------------------
        # TEST 3: Conflict/Upsert Validation
        # ---------------------------------------------------------
        print("\nTEST 3: ON CONFLICT Upsert Handling")
        print("-" * 60)
        
        # Test company_fundamentals UPSERT (ON CONFLICT on PRIMARY KEY ticker)
        cursor.execute("""
            INSERT INTO company_fundamentals (ticker, pe_ratio, debt_to_equity, operating_margin, roe, is_healthy)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(ticker) DO UPDATE SET
                pe_ratio = excluded.pe_ratio,
                is_healthy = excluded.is_healthy
        """, ('AAPL', 30.10, 1.45, 0.302, 1.54, 0)) # PE updated to 30.10, healthy updated to 0
        
        cursor.execute("SELECT pe_ratio, is_healthy FROM company_fundamentals WHERE ticker='AAPL'")
        pe, healthy = cursor.fetchone()
        assert pe == 30.10, f"Expected PE to be updated to 30.10, got {pe}"
        assert healthy == 0, f"Expected is_healthy to be updated to 0, got {healthy}"
        
        # Test daily_sentiment UPSERT (ON CONFLICT on PRIMARY KEY date)
        cursor.execute("""
            INSERT INTO daily_sentiment (date, sentiment_score, sentiment_label, summary)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(date) DO UPDATE SET
                sentiment_score = excluded.sentiment_score,
                sentiment_label = excluded.sentiment_label
        """, ('2026-07-18', -0.150, 'neutral', 'Updated to neutral guidance.'))
        
        cursor.execute("SELECT sentiment_score, sentiment_label FROM daily_sentiment WHERE date='2026-07-18'")
        score, label = cursor.fetchone()
        assert score == -0.150, f"Expected sentiment_score to be -0.150, got {score}"
        assert label == 'neutral', f"Expected label to be 'neutral', got {label}"
        
        # Test daily_recommendations UPSERT (ON CONFLICT on PRIMARY KEY (date, ticker))
        cursor.execute("""
            INSERT INTO daily_recommendations (date, ticker, signal_type, prob_buy, prob_sell, prob_hold, sentiment_score, is_sentiment_passed, execution_price, rationale)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(date, ticker) DO UPDATE SET
                signal_type = excluded.signal_type,
                prob_buy = excluded.prob_buy
        """, ('2026-07-18', 'AAPL', 'HOLD', 0.450, 0.103, 0.155, -0.150, 0, 214.30, 'Downgraded based on neutral sentiment.'))
        
        cursor.execute("SELECT signal_type, prob_buy FROM daily_recommendations WHERE date='2026-07-18' AND ticker='AAPL'")
        signal, prob_buy = cursor.fetchone()
        assert signal == 'HOLD', f"Expected signal to be HOLD, got {signal}"
        assert prob_buy == 0.450, f"Expected prob_buy to be 0.450, got {prob_buy}"
        
        print("PASS: All ON CONFLICT / Upsert constraints verified successfully.")
        
    except Exception as e:
        print(f"FAIL: TEST SUITE FAILED: {str(e)}")
        raise e
    finally:
        # Ensure database connection is closed before attempting file deletion
        if conn:
            conn.close()
            logging.info("Test database connection closed.")
            
        # ---------------------------------------------------------
        # TEST 4: Clean Teardown
        # ---------------------------------------------------------
        print("\nTEST 4: Clean Teardown Check")
        print("-" * 60)
        if os.path.exists(TEST_DB_PATH):
            try:
                os.remove(TEST_DB_PATH)
                print("PASS: Temporary test database cleaned up.")
            except OSError as e:
                print(f"FAIL: Clean Teardown failed: {e}")
        else:
            print("PASS: Clean teardown (no database file left).")

if __name__ == "__main__":
    run_tests()
