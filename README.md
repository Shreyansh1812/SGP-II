# SGP-II: Python-Based Algorithmic Trading Backtester

## 📊 Project Overview

A comprehensive Python-based algorithmic trading backtesting system designed for Indian stock markets (NSE). This project implements industry-standard practices for data acquisition, validation, caching, and backtesting using event-driven architecture.

**Developer:** Shreyansh Patel  
**Language:** Python 3.10+  
**Primary Market:** Indian Stock Market (NSE)

---

## 🎯 Project Objectives

- Build a production-ready backtesting engine for algorithmic trading strategies
- Implement data acquisition with intelligent caching (10-20x performance improvement)
- Develop technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands)
- Create multiple trading strategies (Golden Cross, RSI Mean Reversion, MACD)
- Visualize backtesting results with interactive charts
- Deploy via Streamlit dashboard for user interaction

---

## 🏗️ Architecture

```
User Input (Ticker/Dates)
    ↓
Data Loader (yfinance + caching) ✅ COMPLETED
    ↓
Backtester (Cerebro engine)
    ↓
Strategy (Buy/Sell logic)
    ↓
Results (Portfolio value, trades, metrics)
    ↓
Visualization (Plotly charts)
```

---

## 📁 Project Structure

```
SGP-II/
│
├── config.py                      # Central configuration (ticker, dates, paths)
├── main.py                        # Main entry point
├── requirements.txt               # Python dependencies
│
├── data/
│   ├── raw/                       # Cached stock data (CSV files)
│   └── processed/                 # Processed data for backtesting
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py            # ✅ Data acquisition & caching (COMPLETE)
│   ├── indicators.py             # Technical indicators (TODO)
│   ├── strategy.py               # Trading strategies (TODO)
│   ├── backtester.py             # Backtrader integration (TODO)
│   └── plotting.py               # Visualization (TODO)
│
├── Tests/
│   ├── test_data_loader.py       # Basic data loader tests
│   ├── test_validation.py        # Validation function tests (7 tests)
│   ├── test_cleaning.py          # Cleaning function tests (7 tests)
│   ├── test_caching.py           # Caching function tests (8 tests)
│   └── test_orchestrator.py      # Orchestrator tests (7 tests)
│
├── notebooks/
│   ├── 01_data_analysis.ipynb    # Data exploration
│   └── 02_data_analysis.ipynb    # Advanced analysis
│
└── models/                        # Saved models/configurations
```

---

## 🚀 Setup & Installation

### Prerequisites
- Python 3.10 or higher
- pip (Python package manager)
- Git

### Installation Steps

1. **Clone the repository:**
```bash
git clone https://github.com/Shreyansh1812/SGP-II.git
cd SGP-II
```

2. **Create virtual environment:**
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1  # Windows PowerShell
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Verify installation:**
```bash
python -c "import yfinance, pandas, backtrader, streamlit; print('All packages installed!')"
```

---

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| yfinance | 0.2.37 | Yahoo Finance API for stock data |
| pandas | 2.2.1 | Data manipulation and analysis |
| backtrader | 1.9.78.123 | Event-driven backtesting engine |
| streamlit | 1.32.0 | Web dashboard interface |
| plotly | 5.19.0 | Interactive visualizations |
| matplotlib | 3.8.3 | Static plotting |
| numpy | 1.26.4 | Numerical computations |

---

## ✅ Phase 2: Data Loader Module (COMPLETED)

### Overview
The data loader module provides complete data acquisition, validation, cleaning, and caching functionality with industry-standard error handling and performance optimization.

### Key Features
- ✅ **Intelligent Caching**: 12-16x performance improvement
- ✅ **Data Validation**: 6 comprehensive quality checks
- ✅ **Data Cleaning**: 9-step standardization pipeline
- ✅ **Error Handling**: Graceful handling of invalid tickers, network errors
- ✅ **Cache-First Strategy**: Automatic cache hit/miss detection
- ✅ **Full Documentation**: Google-style docstrings with type hints
- ✅ **Comprehensive Testing**: 29 tests across 5 test files

### Functions

#### 1. `get_stock_data()` - Main Orchestrator (PRIMARY INTERFACE)

**The single function you need for all data operations!**

```python
from src.data_loader import get_stock_data

# Simple usage - handles everything automatically
df = get_stock_data("RELIANCE.NS", "2023-01-01", "2023-12-31")

# With custom cache path
df = get_stock_data("TCS.NS", "2023-01-01", "2023-12-31", path="data/custom/")
```

**What it does:**
1. Checks cache first (0.03s) ⚡
2. If cache miss → Fetches from API (0.5s)
3. Validates data quality (6 checks)
4. Cleans and standardizes data (9 steps)
5. Saves to cache for future use
6. Returns clean DataFrame

**Returns:**
- `pd.DataFrame`: Clean OHLCV data with DatetimeIndex
- `None`: If ticker invalid or data fetch fails

**Example Output:**
```
                  Open    High     Low   Close    Volume
Date
2023-01-02  2550.0  2580.5  2540.0  2575.3  12345678
2023-01-03  2570.0  2590.0  2565.0  2585.0  10234567
```

---

#### 2. `fetch_stock_data()` - Download from Yahoo Finance

```python
from src.data_loader import fetch_stock_data

df = fetch_stock_data("INFY.NS", "2023-01-01", "2023-12-31")
```

**Purpose:** Download raw stock data from Yahoo Finance API  
**Returns:** Raw DataFrame with MultiIndex columns or None if failed

---

#### 3. `validate_data()` - Data Quality Checks

```python
from src.data_loader import validate_data

is_valid = validate_data(df, "RELIANCE.NS", min_rows=10)
```

**6 Validation Checks:**
1. Not empty DataFrame
2. Minimum rows threshold (default: 10)
3. Required OHLCV columns present
4. Data types are numeric
5. Missing values < 30% threshold
6. No negative prices

**Raises:** `ValueError` if validation fails with detailed error message

---

#### 4. `clean_data()` - 9-Step Cleaning Pipeline

```python
from src.data_loader import clean_data

df_clean = clean_data(df_raw, "TCS.NS")
```

**9 Cleaning Steps:**
1. Flatten MultiIndex columns from yfinance
2. Convert index to DatetimeIndex
3. Remove duplicate rows
4. Sort by date ascending
5. Forward-fill missing prices (conservative)
6. Zero-fill missing volume
7. Reorder columns to OHLCV format
8. Drop remaining NaN rows
9. Ensure correct data types (float64/int64)

---

#### 5. `save_data()` - Cache Storage

```python
from src.data_loader import save_data

filepath = save_data(df, "HDFCBANK.NS", "2023-01-01", "2023-12-31", "data/raw/")
```

**Features:**
- Naming convention: `{TICKER}_{START_DATE}_{END_DATE}.csv`
- Automatic directory creation
- File size logging
- Overwrite warning

---

#### 6. `load_data()` - Cache Retrieval

```python
from src.data_loader import load_data

df = load_data("ITC.NS", "2023-01-01", "2023-12-31", "data/raw/")
# Returns None if cache miss
```

**Features:**
- Automatic DatetimeIndex parsing
- Column validation
- Empty file detection
- Returns None on cache miss (not an error)

---

## 🧪 Testing

### Run All Tests
```bash
# Run specific test file
python Tests/test_orchestrator.py
python Tests/test_caching.py
python Tests/test_validation.py
python Tests/test_cleaning.py

# Or run all tests
python -m pytest Tests/
```

### Test Coverage

| Test File | Tests | Coverage |
|-----------|-------|----------|
| test_orchestrator.py | 7 | Cache hit/miss, invalid ticker, multi-ticker |
| test_caching.py | 8 | Save/load, integrity, performance |
| test_validation.py | 7 | All 6 validation checks + edge cases |
| test_cleaning.py | 7 | All 9 cleaning steps |
| test_data_loader.py | Basic | Fetch functionality |
| **TOTAL** | **29+** | **Full coverage** |

---

## 📊 Performance Benchmarks

### Cache Performance

| Operation | Time | Notes |
|-----------|------|-------|
| API Fetch (cache miss) | ~0.4-0.5s | Network dependent |
| Cache Load (cache hit) | ~0.03s | Local disk read |
| **Speedup** | **12-16x** | Typical improvement |

**Example from test run:**
```
API Fetch Time:   0.405 seconds (cache miss)
Cache Load Time:  0.033 seconds (cache hit)
Speedup:          12.3x faster! ⚡
```

---

## 🔧 Configuration

### config.py

```python
# Stock selection
TICKER = "RELIANCE.NS"

# Date range
START_DATE = "2023-01-01"
END_DATE = "2023-12-31"

# Cache directory
DATA_PATH = "C:/Users/shrey/Downloads/SGP-II/data/raw"

# Backtesting parameters
INITIAL_CASH = 100000.0
COMMISSION = 0.001  # 0.1%
```

---

## 💡 Usage Examples

### Example 1: Basic Data Fetch
```python
from src.data_loader import get_stock_data

# Fetch RELIANCE stock data
df = get_stock_data("RELIANCE.NS", "2023-01-01", "2023-12-31")

if df is not None:
    print(f"Fetched {len(df)} rows")
    print(df.head())
else:
    print("Failed to fetch data")
```

### Example 2: Multiple Tickers
```python
from src.data_loader import get_stock_data

tickers = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS"]
data = {}

for ticker in tickers:
    df = get_stock_data(ticker, "2023-01-01", "2023-12-31")
    if df is not None:
        data[ticker] = df
        print(f"✓ {ticker}: {len(df)} rows")
    else:
        print(f"✗ {ticker}: Failed")
```

### Example 3: Error Handling
```python
from src.data_loader import get_stock_data

def safe_fetch(ticker, start, end):
    """Safely fetch data with error handling"""
    df = get_stock_data(ticker, start, end)
    
    if df is None:
        print(f"Error: Could not fetch {ticker}")
        return None
    
    if len(df) < 100:
        print(f"Warning: Only {len(df)} rows for {ticker}")
    
    return df

# Usage
df = safe_fetch("INVALID.NS", "2023-01-01", "2023-12-31")
```

---

## 📈 Next Steps (Upcoming Phases)

### Phase 3: Technical Indicators (TODO)
- [ ] Simple Moving Average (SMA)
- [ ] Exponential Moving Average (EMA)
- [ ] Relative Strength Index (RSI)
- [ ] MACD (Moving Average Convergence Divergence)
- [ ] Bollinger Bands

### Phase 4: Trading Strategies (TODO)
- [ ] Golden Cross Strategy (SMA crossover)
- [ ] RSI Mean Reversion Strategy
- [ ] MACD Trend Following Strategy

### Phase 5: Backtesting Engine (TODO)
- [ ] Backtrader Cerebro integration
- [ ] Portfolio management
- [ ] Performance metrics calculation

### Phase 6: Visualization (TODO)
- [ ] Price & indicator charts
- [ ] Trade markers
- [ ] Equity curve
- [ ] Drawdown analysis

### Phase 7: Streamlit Dashboard (TODO)
- [ ] User input interface
- [ ] Real-time backtesting
- [ ] Interactive charts
- [ ] Results display

---

## 🤝 Contributing

This is a solo university project by Shreyansh1812. For educational purposes only.

---

## 📝 License

Educational project - Not for commercial use.

---

## 📞 Contact

**GitHub:** [Shreyansh1812](https://github.com/Shreyansh1812)  
**Repository:** [SGP-II](https://github.com/Shreyansh1812/SGP-II)

---

## 📚 References

- [yfinance Documentation](https://pypi.org/project/yfinance/)
- [Backtrader Documentation](https://www.backtrader.com/docu/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

**Last Updated:** December 9, 2025  
**Status:** Phase 2 Complete ✅ | Phase 3-7 In Progress 🚧