"""
Generate Sample Stock Data for Testing

Creates synthetic stock data when Yahoo Finance API is unavailable.
Useful for testing and demonstrating the backtesting system offline.

Usage:
------
python generate_sample_data.py

This will create sample CSV files in data/raw/ for common stocks.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

def generate_stock_data(
    ticker: str,
    start_date: str,
    end_date: str,
    initial_price: float = 100.0,
    volatility: float = 0.02,
    trend: float = 0.0002
) -> pd.DataFrame:
    """
    Generate synthetic stock data with realistic characteristics.
    
    Parameters
    ----------
    ticker : str
        Stock ticker symbol
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
    initial_price : float, default=100.0
        Starting price
    volatility : float, default=0.02
        Daily volatility (standard deviation of returns)
    trend : float, default=0.0002
        Daily drift (average return per day)
    
    Returns
    -------
    pd.DataFrame
        Stock data with OHLCV columns
    """
    # Generate date range
    dates = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days
    n_days = len(dates)
    
    # Generate random returns with trend
    np.random.seed(42)  # For reproducibility
    returns = np.random.normal(trend, volatility, n_days)
    
    # Calculate prices from returns
    price_multipliers = np.exp(returns)
    prices = initial_price * np.cumprod(price_multipliers)
    
    # Generate OHLCV data
    data = {
        'Open': [],
        'High': [],
        'Low': [],
        'Close': [],
        'Volume': []
    }
    
    for i, close_price in enumerate(prices):
        # Open is previous close with small gap
        if i == 0:
            open_price = initial_price
        else:
            open_price = prices[i-1] * (1 + np.random.normal(0, 0.002))
        
        # High and Low based on volatility
        intraday_range = close_price * volatility * np.random.uniform(0.5, 2.0)
        high = max(open_price, close_price) + abs(np.random.normal(0, intraday_range))
        low = min(open_price, close_price) - abs(np.random.normal(0, intraday_range))
        
        # Volume with some randomness
        volume = int(np.random.uniform(1000000, 10000000))
        
        data['Open'].append(open_price)
        data['High'].append(high)
        data['Low'].append(low)
        data['Close'].append(close_price)
        data['Volume'].append(volume)
    
    # Create DataFrame
    df = pd.DataFrame(data, index=dates)
    df.index.name = 'Date'
    
    return df


def save_sample_data():
    """Generate and save sample data for common stocks."""
    
    # Create data directory
    data_dir = Path(__file__).parent / "data" / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Stock configurations
    stocks = [
        {
            'ticker': 'AAPL',
            'start': '2020-01-01',
            'end': '2024-01-01',
            'initial': 75.0,
            'volatility': 0.018,
            'trend': 0.0005
        },
        {
            'ticker': 'MSFT',
            'start': '2020-01-01',
            'end': '2024-01-01',
            'initial': 160.0,
            'volatility': 0.016,
            'trend': 0.0004
        },
        {
            'ticker': 'GOOGL',
            'start': '2020-01-01',
            'end': '2024-01-01',
            'initial': 68.0,
            'volatility': 0.017,
            'trend': 0.0003
        },
        {
            'ticker': 'TSLA',
            'start': '2020-01-01',
            'end': '2024-01-01',
            'initial': 84.0,
            'volatility': 0.035,
            'trend': 0.0008
        },
        {
            'ticker': 'TCS.NS',
            'start': '2020-01-01',
            'end': '2024-01-01',
            'initial': 2000.0,
            'volatility': 0.020,
            'trend': 0.0003
        },
        {
            'ticker': 'INFY.NS',
            'start': '2020-01-01',
            'end': '2024-01-01',
            'initial': 750.0,
            'volatility': 0.019,
            'trend': 0.0004
        },
        {
            'ticker': 'RELIANCE.NS',
            'start': '2020-01-01',
            'end': '2024-01-01',
            'initial': 1500.0,
            'volatility': 0.022,
            'trend': 0.0002
        }
    ]
    
    print("Generating sample stock data...\n")
    
    for stock in stocks:
        print(f"Generating {stock['ticker']}...", end=' ')
        
        df = generate_stock_data(
            ticker=stock['ticker'],
            start_date=stock['start'],
            end_date=stock['end'],
            initial_price=stock['initial'],
            volatility=stock['volatility'],
            trend=stock['trend']
        )
        
        # Save to CSV
        filename = f"{stock['ticker']}_{stock['start']}_{stock['end']}.csv"
        filepath = data_dir / filename
        df.to_csv(filepath)
        
        print(f"✓ Saved {len(df)} days to {filename}")
    
    print(f"\n✅ All sample data generated successfully!")
    print(f"📁 Location: {data_dir.absolute()}")
    print("\n🚀 You can now run the Streamlit dashboard with these tickers:")
    print("   - AAPL, MSFT, GOOGL, TSLA (US stocks)")
    print("   - TCS.NS, INFY.NS, RELIANCE.NS (Indian stocks)")


if __name__ == "__main__":
    save_sample_data()
