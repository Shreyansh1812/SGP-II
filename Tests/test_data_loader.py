"""
Unit tests for data_loader.py with mocked yfinance calls
"""

import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime
from src.data_loader import fetch_stock_data


class TestDataLoader(unittest.TestCase):
    """Test suite for data_loader module"""
    
    def setUp(self):
        """Create sample data for mocking"""
        dates = pd.date_range('2023-01-01', '2023-01-10', freq='D')
        self.mock_data = pd.DataFrame({
            'Open': np.random.uniform(100, 110, len(dates)),
            'High': np.random.uniform(110, 120, len(dates)),
            'Low': np.random.uniform(90, 100, len(dates)),
            'Close': np.random.uniform(100, 110, len(dates)),
            'Volume': np.random.randint(1000000, 2000000, len(dates))
        }, index=dates)
        self.mock_data.index.name = 'Date'
    
    @patch('yfinance.Ticker')
    def test_fetch_valid_ticker(self, mock_ticker_class):
        """Test fetching data for valid ticker"""
        # Setup mock
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = self.mock_data
        mock_ticker_class.return_value = mock_ticker
        
        # Execute
        result = fetch_stock_data("RELIANCE.NS", "2023-01-01", "2023-12-31")
        
        # Assert
        self.assertIsNotNone(result)
        self.assertEqual(len(result), len(self.mock_data))
        self.assertIn('Close', result.columns)
        mock_ticker_class.assert_called_once_with("RELIANCE.NS")
    
    @patch('yfinance.Ticker')
    def test_fetch_invalid_ticker(self, mock_ticker_class):
        """Test fetching data for invalid ticker"""
        # Setup mock to return empty DataFrame
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame()
        mock_ticker_class.return_value = mock_ticker
        
        # Execute
        result = fetch_stock_data("INVALID.NS", "2023-01-01", "2023-12-31")
        
        # Assert
        self.assertIsNone(result)
    
    @patch('yfinance.download')
    @patch('yfinance.Ticker')
    def test_fetch_with_api_error(self, mock_ticker_class, mock_download):
        """Test handling of API errors"""
        # Setup mock to raise exception
        mock_ticker = MagicMock()
        mock_ticker.history.side_effect = Exception("API Error")
        mock_ticker_class.return_value = mock_ticker
        
        # Also mock download to fail
        mock_download.return_value = pd.DataFrame()
        
        # Execute
        result = fetch_stock_data("AAPL", "2023-01-01", "2023-12-31")
        
        # Assert - should handle error gracefully
        self.assertIsNone(result)


if __name__ == '__main__':
    print("=" * 60)
    print("Running data_loader tests with mocked yfinance")
    print("=" * 60)
    unittest.main(verbosity=2)

