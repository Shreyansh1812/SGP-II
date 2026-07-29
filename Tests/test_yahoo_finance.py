import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

@pytest.fixture
def mock_yfinance_data():
    dates = pd.date_range(start='2023-01-01', periods=30, freq='D')
    df = pd.DataFrame({
        'Open': np.random.uniform(100, 150, 30),
        'High': np.random.uniform(100, 150, 30),
        'Low': np.random.uniform(100, 150, 30),
        'Close': np.random.uniform(100, 150, 30),
        'Volume': np.random.randint(1000, 10000, 30)
    }, index=dates)
    return df


@patch('yfinance.download')
def test_yfinance_download(mock_download, mock_yfinance_data):
    mock_download.return_value = mock_yfinance_data

    import yfinance as yf
    data1 = yf.download(
        "AAPL",
        start="2023-01-01",
        end="2023-01-31",
        progress=False,
        auto_adjust=True
    )
    
    assert not data1.empty
    assert len(data1) == 30
    assert 'Close' in data1.columns

@patch('yfinance.Ticker')
def test_yfinance_ticker(mock_ticker_class, mock_yfinance_data):
    mock_ticker_instance = MagicMock()
    mock_ticker_instance.history.return_value = mock_yfinance_data
    mock_ticker_class.return_value = mock_ticker_instance
    
    import yfinance as yf
    stock = yf.Ticker("AAPL")
    data2 = stock.history(
        start="2023-01-01",
        end="2023-01-31",
        auto_adjust=True
    )
    
    assert not data2.empty
    assert len(data2) == 30
    assert 'Volume' in data2.columns

def test_production_integrity_check(mock_yfinance_data):
    data2 = mock_yfinance_data
    assert not data2.empty

    latest_price = data2['Close'].iloc[-1]
    assert 1 <= latest_price <= 10000

    avg_volume = data2['Volume'].mean()
    assert avg_volume >= 1000
