"""
Data Provider Abstraction Layer

This module provides an abstract interface for stock data providers,
allowing easy swapping between Yahoo Finance, Alpha Vantage, IEX Cloud, etc.
"""

from abc import ABC, abstractmethod
import pandas as pd
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class DataProvider(ABC):
    """
    Abstract base class for stock data providers.
    
    Subclasses must implement the fetch() method to retrieve historical data
    from their specific data source (Yahoo Finance, Alpha Vantage, etc.)
    """
    
    @abstractmethod
    def fetch(self, ticker: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        Fetch historical stock data.
        
        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL', 'RELIANCE.NS')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            
        Returns:
            DataFrame with DatetimeIndex and OHLCV columns, or None if failed
        """
        pass
    
    @abstractmethod
    def validate_ticker(self, ticker: str) -> bool:
        """
        Check if ticker symbol is valid for this provider.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            True if ticker is valid, False otherwise
        """
        pass


class YahooFinanceProvider(DataProvider):
    """
    Yahoo Finance data provider implementation.
    
    Uses yfinance library to fetch free historical stock data.
    """
    
    def __init__(self):
        """Initialize Yahoo Finance provider."""
        import yfinance as yf
        self.yf = yf
        logger.info("Initialized Yahoo Finance data provider")
    
    def fetch(self, ticker: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        Fetch data from Yahoo Finance API.
        
        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL', 'RELIANCE.NS')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            
        Returns:
            DataFrame with OHLCV data, or None if failed
        """
        import time
        
        max_retries = 2
        base_delay = 2
        
        # Method 1: Try Ticker.history()
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    delay = base_delay * attempt
                    logger.info(f"Waiting {delay}s before retry...")
                    time.sleep(delay)
                
                logger.info(f"[Yahoo Finance] Fetching {ticker} from {start_date} to {end_date} (attempt {attempt + 1}/{max_retries})")
                
                stock = self.yf.Ticker(ticker)
                data = stock.history(
                    start=start_date,
                    end=end_date,
                    auto_adjust=False,
                    actions=False
                )
                
                if not data.empty:
                    logger.info(f"✅ [Yahoo Finance] Fetched {len(data)} rows for {ticker}")
                    return data
                
                logger.warning(f"[Yahoo Finance] No data for {ticker} (attempt {attempt + 1})")
            
            except Exception as e:
                logger.warning(f"[Yahoo Finance] Error: {str(e)}")
                if attempt == max_retries - 1:
                    logger.error(f"[Yahoo Finance] Failed after {max_retries} attempts")
        
        # Method 2: Fallback to yf.download()
        try:
            logger.info(f"[Yahoo Finance] Using fallback download for {ticker}...")
            
            data = self.yf.download(
                ticker,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=False,
                actions=False
            )
            
            if not data.empty:
                # Flatten MultiIndex columns
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.droplevel(1)
                
                logger.info(f"✅ [Yahoo Finance] Fallback successful: {len(data)} rows")
                return data
        
        except Exception as e:
            logger.error(f"[Yahoo Finance] Fallback failed: {str(e)}")
        
        return None
    
    def validate_ticker(self, ticker: str) -> bool:
        """
        Validate ticker by attempting to fetch info.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            True if ticker exists, False otherwise
        """
        try:
            stock = self.yf.Ticker(ticker)
            info = stock.info
            return 'symbol' in info or 'shortName' in info
        except:
            return False


class AlphaVantageProvider(DataProvider):
    """
    Alpha Vantage data provider (placeholder for future implementation).
    
    Requires API key: https://www.alphavantage.co/support/#api-key
    """
    
    def __init__(self, api_key: str):
        """
        Initialize Alpha Vantage provider.
        
        Args:
            api_key: Alpha Vantage API key
        """
        self.api_key = api_key
        logger.info("Initialized Alpha Vantage data provider")
    
    def fetch(self, ticker: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Not implemented - placeholder for future enhancement."""
        raise NotImplementedError("Alpha Vantage provider not yet implemented")
    
    def validate_ticker(self, ticker: str) -> bool:
        """Not implemented - placeholder for future enhancement."""
        raise NotImplementedError("Alpha Vantage provider not yet implemented")


class IEXCloudProvider(DataProvider):
    """
    IEX Cloud data provider (placeholder for future implementation).
    
    Requires API token: https://iexcloud.io/
    """
    
    def __init__(self, api_token: str):
        """
        Initialize IEX Cloud provider.
        
        Args:
            api_token: IEX Cloud API token
        """
        self.api_token = api_token
        logger.info("Initialized IEX Cloud data provider")
    
    def fetch(self, ticker: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Not implemented - placeholder for future enhancement."""
        raise NotImplementedError("IEX Cloud provider not yet implemented")
    
    def validate_ticker(self, ticker: str) -> bool:
        """Not implemented - placeholder for future enhancement."""
        raise NotImplementedError("IEX Cloud provider not yet implemented")


# Factory function to create providers
def create_provider(provider_name: str = "yahoo", **kwargs) -> DataProvider:
    """
    Factory function to create data provider instances.
    
    Args:
        provider_name: Name of provider ('yahoo', 'alphavantage', 'iexcloud')
        **kwargs: Provider-specific arguments (e.g., api_key, api_token)
        
    Returns:
        DataProvider instance
        
    Examples:
        >>> provider = create_provider("yahoo")
        >>> data = provider.fetch("AAPL", "2024-01-01", "2024-12-01")
        
        >>> provider = create_provider("alphavantage", api_key="YOUR_KEY")
        >>> data = provider.fetch("MSFT", "2024-01-01", "2024-12-01")
    """
    providers = {
        "yahoo": YahooFinanceProvider,
        "alphavantage": AlphaVantageProvider,
        "iexcloud": IEXCloudProvider
    }
    
    provider_class = providers.get(provider_name.lower())
    if not provider_class:
        raise ValueError(f"Unknown provider: {provider_name}. Available: {list(providers.keys())}")
    
    # YahooFinanceProvider doesn't need kwargs
    if provider_name.lower() == "yahoo":
        return provider_class()
    else:
        return provider_class(**kwargs)
