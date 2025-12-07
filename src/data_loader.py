"""
Data Loader Module

This module handles fetching, validating, cleaning, and caching stock data
from Yahoo Finance for backtesting purposes.
"""

import yfinance as yf
import pandas as pd
import logging
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def fetch_stock_data(ticker: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """
    Fetch historical stock data from Yahoo Finance.
    
    Args:
        ticker (str): Stock ticker symbol (e.g., 'RELIANCE.NS' for NSE stocks)
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
    
    Returns:
        pd.DataFrame: Historical OHLCV data with columns [Open, High, Low, Close, Adj Close, Volume]
        None: If download fails
    
    Raises:
        Exception: If API call fails or network error occurs
    
    Example:
        >>> df = fetch_stock_data("RELIANCE.NS", "2020-01-01", "2024-01-01")
        >>> print(df.head())
    """
    try:
        logging.info(f"Fetching data for {ticker} from {start_date} to {end_date}")
        
        # Download data using yfinance
        data = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            progress=False  # Suppress progress bar for cleaner logs
        )
        
        # Check if data was successfully retrieved
        if data.empty:
            logging.warning(f"No data retrieved for {ticker}. Check ticker symbol or date range.")
            return None
        
        logging.info(f"Successfully fetched {len(data)} rows of data for {ticker}")
        return data
    
    except Exception as e:
        logging.error(f"Error fetching data for {ticker}: {str(e)}")
        return None


def validate_data(df: pd.DataFrame, ticker: str, min_rows: int = 10) -> bool:
    """
    Validate the quality and completeness of fetched stock data.
    
    Performs comprehensive checks to ensure data is suitable for backtesting:
    - Verifies DataFrame is not empty
    - Checks for required OHLCV columns
    - Validates data types are numeric
    - Ensures sufficient data volume (minimum rows)
    - Checks for excessive missing values (>30% threshold)
    - Detects invalid price values (negative or zero)
    
    Args:
        df (pd.DataFrame): DataFrame containing stock data to validate
        ticker (str): Stock ticker symbol (used in error messages for context)
        min_rows (int, optional): Minimum number of rows required. Defaults to 10.
                                   Backtesting needs sufficient data for meaningful results.
    
    Returns:
        bool: True if all validation checks pass
    
    Raises:
        ValueError: If any validation check fails, with descriptive error message
                   explaining what failed and why
    
    Example:
        >>> df = fetch_stock_data("RELIANCE.NS", "2023-01-01", "2023-12-31")
        >>> if validate_data(df, "RELIANCE.NS"):
        ...     print("Data is valid and ready for backtesting")
    
    Notes:
        - This is a critical quality gate before backtesting
        - Catching data issues here prevents crashes in the backtester
        - 30% missing data threshold is industry standard for time series
    """
    # Define required columns for OHLCV (Open, High, Low, Close, Volume) data
    # These are essential for technical analysis and backtesting
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    # =============================================================================
    # CHECK 1: Empty DataFrame Validation
    # =============================================================================
    # An empty DataFrame means no data was fetched (API error, invalid dates, etc.)
    # Cannot proceed with backtesting without any data
    if df is None or df.empty:
        error_msg = f"Validation failed for {ticker}: DataFrame is empty"
        logging.error(error_msg)
        raise ValueError(error_msg)
    
    # =============================================================================
    # CHECK 2: Minimum Data Volume
    # =============================================================================
    # Backtesting requires sufficient historical data to be meaningful
    # Too few rows (e.g., <10) won't provide reliable strategy performance
    # Example: SMA(50) needs at least 50 rows to calculate properly
    if len(df) < min_rows:
        error_msg = (
            f"Validation failed for {ticker}: Insufficient data rows. "
            f"Found {len(df)} rows, minimum required is {min_rows}. "
            f"Consider expanding the date range."
        )
        logging.error(error_msg)
        raise ValueError(error_msg)
    
    # =============================================================================
    # CHECK 3: Required Columns Presence
    # =============================================================================
    # Flatten multi-level columns if present (yfinance sometimes returns nested structure)
    # Example: [('Close', 'RELIANCE.NS')] becomes ['Close']
    if isinstance(df.columns, pd.MultiIndex):
        actual_columns = df.columns.get_level_values(0).tolist()
    else:
        actual_columns = df.columns.tolist()
    
    # Check if all required OHLCV columns are present
    missing_columns = [col for col in required_columns if col not in actual_columns]
    
    if missing_columns:
        error_msg = (
            f"Validation failed for {ticker}: Missing required columns {missing_columns}. "
            f"Found columns: {actual_columns}. "
            f"OHLCV data is essential for backtesting."
        )
        logging.error(error_msg)
        raise ValueError(error_msg)
    
    # =============================================================================
    # CHECK 4: Data Type Validation
    # =============================================================================
    # Price columns (OHLC) must be numeric (float/int) for calculations
    # Non-numeric data (strings, objects) will cause math operations to fail
    price_columns = ['Open', 'High', 'Low', 'Close']
    
    for col in price_columns:
        # Access column properly based on whether it's MultiIndex or not
        if isinstance(df.columns, pd.MultiIndex):
            col_data = df[col].iloc[:, 0]  # Get first level of MultiIndex
        else:
            col_data = df[col]
        
        # Check if column dtype is numeric (float64, int64, etc.)
        if not pd.api.types.is_numeric_dtype(col_data):
            error_msg = (
                f"Validation failed for {ticker}: Column '{col}' has non-numeric data type "
                f"{col_data.dtype}. Expected float or int for price data."
            )
            logging.error(error_msg)
            raise ValueError(error_msg)
    
    # =============================================================================
    # CHECK 5: Missing Values Analysis
    # =============================================================================
    # Calculate percentage of missing values for each required column
    # Too many NaN values (>30%) indicate poor data quality
    # Forward-fill can handle some gaps, but not excessive missing data
    missing_percentage_threshold = 30.0  # Industry standard threshold
    
    for col in required_columns:
        # Access column properly based on MultiIndex structure
        if isinstance(df.columns, pd.MultiIndex):
            col_data = df[col].iloc[:, 0]
        else:
            col_data = df[col]
        
        # Calculate percentage of NaN values
        missing_count = col_data.isnull().sum()
        missing_percentage = (missing_count / len(df)) * 100
        
        if missing_percentage > missing_percentage_threshold:
            error_msg = (
                f"Validation failed for {ticker}: Column '{col}' has {missing_percentage:.2f}% "
                f"missing values (threshold: {missing_percentage_threshold}%). "
                f"Data quality is insufficient for reliable backtesting."
            )
            logging.error(error_msg)
            raise ValueError(error_msg)
        
        # Log warning for moderate missing values (10-30%)
        if missing_percentage > 10:
            logging.warning(
                f"{ticker}: Column '{col}' has {missing_percentage:.2f}% missing values. "
                f"Will be handled during cleaning phase."
            )
    
    # =============================================================================
    # CHECK 6: Invalid Price Values
    # =============================================================================
    # Stock prices must be positive (negative or zero prices are invalid)
    # This catches data corruption or API errors
    for col in price_columns:
        # Access column properly
        if isinstance(df.columns, pd.MultiIndex):
            col_data = df[col].iloc[:, 0]
        else:
            col_data = df[col]
        
        # Check for negative or zero prices (excluding NaN which are handled separately)
        invalid_prices = col_data[col_data <= 0].dropna()
        
        if len(invalid_prices) > 0:
            error_msg = (
                f"Validation failed for {ticker}: Column '{col}' contains "
                f"{len(invalid_prices)} invalid values (negative or zero). "
                f"First few invalid values: {invalid_prices.head().tolist()}"
            )
            logging.error(error_msg)
            raise ValueError(error_msg)
    
    # =============================================================================
    # All Checks Passed
    # =============================================================================
    logging.info(
        f"Validation successful for {ticker}: {len(df)} rows, "
        f"all required columns present, data quality acceptable"
    )
    return True


def clean_data(df: pd.DataFrame, ticker: str = "UNKNOWN") -> pd.DataFrame:
    """
    Clean and standardize stock data for backtesting compatibility.
    
    Transforms raw yfinance data into a clean, standardized format that
    Backtrader and other backtesting engines can process reliably. Handles
    common data quality issues including multi-level columns, missing values,
    duplicates, and improper sorting.
    
    Operations Performed (in order):
        1. Flatten multi-level columns from yfinance nested structure
        2. Ensure index is proper DatetimeIndex
        3. Remove duplicate date entries (keeps first occurrence)
        4. Sort data chronologically (ascending date order)
        5. Forward-fill missing price values (OHLC columns)
        6. Zero-fill missing volume values
        7. Standardize column order (OHLCV)
        8. Remove any remaining unfixable rows
        9. Ensure proper data types (numeric)
    
    Args:
        df (pd.DataFrame): Raw DataFrame from yfinance with potential quality issues
        ticker (str, optional): Stock ticker symbol for logging context. Defaults to "UNKNOWN".
    
    Returns:
        pd.DataFrame: Cleaned DataFrame with:
                     - Simple column names ['Open', 'High', 'Low', 'Close', 'Volume']
                     - DatetimeIndex in chronological order
                     - No missing values
                     - No duplicate dates
                     - Proper data types (float64 for prices, int64 for volume)
    
    Raises:
        ValueError: If DataFrame becomes empty after cleaning
                   (indicates severe data quality issues)
    
    Example:
        >>> raw_df = fetch_stock_data("RELIANCE.NS", "2023-01-01", "2023-12-31")
        >>> clean_df = clean_data(raw_df, "RELIANCE.NS")
        >>> print(clean_df.head())
                    Open     High      Low    Close    Volume
        Date                                                  
        2023-01-02  1500.0  1520.0  1490.0  1510.0   1000000
    
    Notes:
        - Forward-fill is used for prices (conservative assumption: price unchanged)
        - Zero-fill is used for volume (no trades = zero volume)
        - Creates a copy of input DataFrame (non-destructive operation)
        - Logs warnings for significant data modifications
    """
    # Create a copy to avoid modifying the original DataFrame
    # This is a defensive programming practice - caller's data remains unchanged
    df_clean = df.copy()
    
    # Track original row count for logging data loss
    original_rows = len(df_clean)
    
    logging.info(f"Starting data cleaning for {ticker}: {original_rows} rows")
    
    # =============================================================================
    # STEP 1: Flatten Multi-Level Columns
    # =============================================================================
    # yfinance returns MultiIndex columns like [('Close', 'RELIANCE.NS'), ...]
    # Backtrader expects simple columns like ['Close', 'High', ...]
    # This step extracts only the first level (the actual column name)
    
    if isinstance(df_clean.columns, pd.MultiIndex):
        logging.info(f"{ticker}: Flattening multi-level columns")
        
        # get_level_values(0) extracts the first level: 'Close' from ('Close', 'RELIANCE.NS')
        df_clean.columns = df_clean.columns.get_level_values(0)
        
        logging.info(f"{ticker}: Columns after flattening: {df_clean.columns.tolist()}")
    
    # =============================================================================
    # STEP 2: Ensure Proper DatetimeIndex
    # =============================================================================
    # The index must be a DatetimeIndex for time-based operations
    # Some data sources return string dates which need conversion
    
    if not isinstance(df_clean.index, pd.DatetimeIndex):
        logging.info(f"{ticker}: Converting index to DatetimeIndex")
        df_clean.index = pd.to_datetime(df_clean.index)
    
    # Name the index for clarity (some operations expect this)
    df_clean.index.name = 'Date'
    
    # =============================================================================
    # STEP 3: Remove Duplicate Dates
    # =============================================================================
    # Duplicate dates can occur from:
    # - Data adjustments (splits, dividends)
    # - API errors
    # - Multiple data sources merged incorrectly
    # Keeping duplicates would cause strategies to execute multiple times per day
    
    duplicate_count = df_clean.index.duplicated().sum()
    
    if duplicate_count > 0:
        logging.warning(
            f"{ticker}: Found {duplicate_count} duplicate dates. "
            f"Keeping first occurrence only."
        )
        
        # ~df_clean.index.duplicated(keep='first') creates a boolean mask
        # False for duplicates (except first), True for unique dates
        df_clean = df_clean[~df_clean.index.duplicated(keep='first')]
        
        logging.info(f"{ticker}: Removed {duplicate_count} duplicate rows")
    
    # =============================================================================
    # STEP 4: Sort by Date (Chronological Order)
    # =============================================================================
    # Technical indicators require chronological order (SMA, EMA, RSI, etc.)
    # Backtrader processes data sequentially and expects ascending dates
    
    if not df_clean.index.is_monotonic_increasing:
        logging.info(f"{ticker}: Sorting data by date (ascending)")
        df_clean = df_clean.sort_index()
    
    # =============================================================================
    # STEP 5: Handle Missing Price Values (Forward Fill)
    # =============================================================================
    # Missing prices occur when:
    # - Exchange closed (holidays, weekends)
    # - API data gaps
    # - Data transmission errors
    #
    # Forward fill (ffill) strategy:
    # - Assumes price didn't change from previous day (conservative)
    # - Better than interpolation (which might create artificial trends)
    # - Better than dropping rows (loses valuable data points)
    
    price_columns = ['Open', 'High', 'Low', 'Close']
    
    for col in price_columns:
        if col in df_clean.columns:
            missing_count = df_clean[col].isnull().sum()
            
            if missing_count > 0:
                logging.info(
                    f"{ticker}: Forward-filling {missing_count} missing values in '{col}' column"
                )
                
                # ffill = forward fill: propagate last valid observation forward
                df_clean[col] = df_clean[col].fillna(method='ffill')
                
                # If first rows are NaN (no previous value to fill), use backward fill
                if df_clean[col].isnull().any():
                    df_clean[col] = df_clean[col].fillna(method='bfill')
    
    # =============================================================================
    # STEP 6: Handle Missing Volume Values (Fill with Zero)
    # =============================================================================
    # Missing volume means no trading activity occurred
    # Zero is the appropriate value (unlike prices which shouldn't be zero)
    
    if 'Volume' in df_clean.columns:
        volume_missing = df_clean['Volume'].isnull().sum()
        
        if volume_missing > 0:
            logging.info(
                f"{ticker}: Filling {volume_missing} missing volume values with 0 "
                f"(no trading activity)"
            )
            df_clean['Volume'] = df_clean['Volume'].fillna(0)
    
    # =============================================================================
    # STEP 7: Standardize Column Order
    # =============================================================================
    # Ensure consistent column order across all datasets
    # Makes debugging easier and meets Backtrader expectations
    
    desired_order = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    # Only reorder columns that exist (some data might not have all columns)
    available_columns = [col for col in desired_order if col in df_clean.columns]
    
    if available_columns:
        df_clean = df_clean[available_columns]
        logging.info(f"{ticker}: Reordered columns to: {available_columns}")
    
    # =============================================================================
    # STEP 8: Remove Remaining Unfixable Rows
    # =============================================================================
    # After forward/backward fill, any remaining NaN indicates unfixable data
    # These rows must be dropped to prevent calculation errors
    
    rows_with_nan = df_clean.isnull().any(axis=1).sum()
    
    if rows_with_nan > 0:
        logging.warning(
            f"{ticker}: Dropping {rows_with_nan} rows with remaining NaN values "
            f"(unfixable data quality issues)"
        )
        df_clean = df_clean.dropna()
    
    # =============================================================================
    # STEP 9: Ensure Proper Data Types
    # =============================================================================
    # Convert columns to appropriate numeric types
    # This prevents type-related errors during backtesting
    
    for col in price_columns:
        if col in df_clean.columns:
            # Convert to float64 (standard for prices)
            # errors='coerce' converts invalid values to NaN instead of raising error
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    if 'Volume' in df_clean.columns:
        # Volume should be integer (can't trade fractional shares)
        # Convert to int64 after ensuring no NaN values
        df_clean['Volume'] = pd.to_numeric(df_clean['Volume'], errors='coerce').fillna(0).astype('int64')
    
    # =============================================================================
    # Final Validation: Check if DataFrame is Empty
    # =============================================================================
    # If all rows were dropped during cleaning, the data is unusable
    final_rows = len(df_clean)
    
    if final_rows == 0:
        error_msg = (
            f"Cleaning failed for {ticker}: All rows were removed during cleaning. "
            f"Original rows: {original_rows}. This indicates severe data quality issues."
        )
        logging.error(error_msg)
        raise ValueError(error_msg)
    
    # =============================================================================
    # Log Summary
    # =============================================================================
    rows_removed = original_rows - final_rows
    removal_percentage = (rows_removed / original_rows) * 100 if original_rows > 0 else 0
    
    logging.info(
        f"Cleaning complete for {ticker}: "
        f"{final_rows} rows remaining (removed {rows_removed} rows, {removal_percentage:.1f}%)"
    )
    
    # Warn if significant data loss occurred
    if removal_percentage > 5:
        logging.warning(
            f"{ticker}: Significant data loss during cleaning ({removal_percentage:.1f}%). "
            f"Consider investigating data quality issues."
        )
    
    return df_clean

