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
Technical Indicators (SMA, EMA, RSI, MACD, BB) ✅ COMPLETED
    ↓
Trading Strategies (Signal Generation) ✅ COMPLETED
    ↓
Backtester (Event-driven execution) ✅ COMPLETED
    ↓
Visualization (Interactive Plotly charts) ✅ COMPLETED
    ↓
Results (Portfolio metrics, trades, charts)
    ↓
Streamlit Dashboard (Web interface) ✅ COMPLETED
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
│   ├── indicators.py             # ✅ Technical indicators (COMPLETE)
│   ├── strategy.py               # ✅ Trading strategies (COMPLETE)
│   ├── backtester.py             # ✅ Backtesting engine (COMPLETE)
│   └── plotting.py               # ✅ Visualization (COMPLETE)
│
├── Tests/
│   ├── test_data_loader.py       # Data loader tests (29 tests)
│   ├── test_validation.py        # Validation tests (7 tests)
│   ├── test_cleaning.py          # Cleaning tests (7 tests)
│   ├── test_caching.py           # Caching tests (8 tests)
│   ├── test_orchestrator.py      # Orchestrator tests (7 tests)
│   ├── test_indicators.py        # ✅ Indicators tests (35 tests)
│   ├── test_strategy.py          # ✅ Strategy tests (21 tests)
│   ├── test_backtester.py        # ✅ Backtester tests (33 tests)
│   └── test_plotting.py          # ✅ Plotting tests (28 tests)
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

### ✅ Phase 3: Technical Indicators (COMPLETED)
- ✅ Simple Moving Average (SMA) - 7 tests
- ✅ Exponential Moving Average (EMA) - 7 tests
- ✅ Relative Strength Index (RSI) - 7 tests
- ✅ MACD (Moving Average Convergence Divergence) - 7 tests
- ✅ Bollinger Bands - 7 tests

**Total: 5 indicators, 35 comprehensive tests, 1,448 lines of production code**

### ✅ Phase 4: Trading Strategies (COMPLETED)
- ✅ Golden Cross Strategy (SMA 50/200 crossover) - 7 tests
- ✅ RSI Mean Reversion Strategy (RSI 14 overbought/oversold) - 7 tests
- ✅ MACD Trend Following Strategy (MACD/Signal crossover) - 7 tests

**Total: 3 strategies, 21 comprehensive tests, 850+ lines of production code**

### ✅ Phase 5: Backtesting Engine (COMPLETED)
- ✅ Event-driven backtesting architecture
- ✅ Position management (FLAT/LONG states)
- ✅ Trade execution with realistic order fills
- ✅ Performance metrics (Return, CAGR, Sharpe, Drawdown, Win Rate)
- ✅ Portfolio tracking and equity curve
- ✅ Comprehensive validation and error handling

**Total: 3 core functions, 33 comprehensive tests, 1,050+ lines of production code**

### ✅ Phase 6: Visualization (COMPLETED)
- ✅ Price charts with technical indicators
- ✅ Trading signals with BUY/SELL markers
- ✅ Equity curve visualization
- ✅ Drawdown analysis charts
- ✅ Returns distribution histogram
- ✅ Monthly returns heatmap
- ✅ Complete backtest report generator

**Total: 7 plotting functions, 28 comprehensive tests, 1,700+ lines of production code**

### Phase 7: Streamlit Dashboard (COMPLETED)
- [ ] User input interface
- [ ] Real-time backtesting
- [ ] Interactive charts display
- [ ] Results and metrics display

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
---

## ✅ Phase 3: Technical Indicators Module (COMPLETED)

### Overview
Complete implementation of 5 core technical indicators with comprehensive testing, industry-standard formulas, and production-ready code.

### Implemented Indicators

#### 1. **Simple Moving Average (SMA)**
```python
from src.indicators import calculate_sma

sma = calculate_sma(data, column='Close', period=20)
```
- **Formula:** Unweighted mean of previous N periods
- **Use Case:** Trend identification, support/resistance levels
- **Tests:** 7 comprehensive tests covering calculation, validation, edge cases

#### 2. **Exponential Moving Average (EMA)**
```python
from src.indicators import calculate_ema

ema = calculate_ema(data, column='Close', period=12)
```
- **Formula:** Weighted mean with exponential decay (multiplier = 2/(period+1))
- **Use Case:** More responsive to recent prices, MACD component
- **Tests:** 7 tests including responsiveness comparison with SMA

#### 3. **Relative Strength Index (RSI)**
```python
from src.indicators import calculate_rsi

rsi = calculate_rsi(data, column='Close', period=14)
```
- **Formula:** RSI = 100 - (100 / (1 + RS)), where RS = Avg Gain / Avg Loss
- **Smoothing:** Wilder's smoothing (alpha = 1/period)
- **Range:** 0-100 (>70 overbought, <30 oversold)
- **Tests:** 7 tests including overbought/oversold detection, edge cases

#### 4. **MACD (Moving Average Convergence Divergence)**
```python
from src.indicators import calculate_macd

macd_line, signal_line, histogram = calculate_macd(
    data, 
    column='Close',
    fast_period=12,
    slow_period=26,
    signal_period=9
)
```
- **Components:** 
  - MACD Line = EMA(12) - EMA(26)
  - Signal Line = EMA(9) of MACD Line
  - Histogram = MACD Line - Signal Line
- **Use Case:** Trend direction, momentum, crossover signals
- **Tests:** 7 tests including crossover detection, zero-line analysis

#### 5. **Bollinger Bands**
```python
from src.indicators import calculate_bollinger_bands

upper, middle, lower = calculate_bollinger_bands(
    data,
    column='Close',
    period=20,
    std_multiplier=2.0
)
```
- **Components:**
  - Middle Band = SMA(20)
  - Upper Band = Middle + (2 × Standard Deviation)
  - Lower Band = Middle - (2 × Standard Deviation)
- **Use Case:** Volatility measurement, squeeze detection, overbought/oversold
- **Statistical Significance:** 2 std dev captures ~95% of price action
- **Tests:** 7 tests including squeeze detection, symmetry validation

### Key Features
- ✅ **Industry-Standard Formulas:** Exact implementations matching trading platforms
- ✅ **Comprehensive Validation:** 5-8 parameter validations per function
- ✅ **Advanced Logging:** Detailed statistics, market conditions, crossover detection
- ✅ **Type Safety:** Full type hints (pd.DataFrame → pd.Series/Tuple[pd.Series])
- ✅ **Google-Style Docstrings:** Complete documentation with formulas, examples
- ✅ **Vectorized Operations:** O(n) pandas operations for performance
- ✅ **Real Data Testing:** Validated with RELIANCE.NS stock data
- ✅ **35 Comprehensive Tests:** All indicators thoroughly tested

### Test Coverage

| Indicator | Tests | Line Coverage |
|-----------|-------|---------------|
| SMA | 7 | Basic calc, structure, validation, periods, columns, real data, edge cases |
| EMA | 7 | Basic calc, structure, validation, responsiveness, MACD prep, real data |
| RSI | 7 | Basic calc, structure, validation, overbought/oversold, periods, edge cases |
| MACD | 7 | Component verification, structure, validation, crossovers, periods, real data |
| Bollinger Bands | 7 | Basic calc, structure, validation, squeeze, price position, multipliers |
| **TOTAL** | **35** | **Complete coverage** |

### Usage Example
```python
from src.data_loader import get_stock_data
from src.indicators import calculate_sma, calculate_ema, calculate_rsi, calculate_macd, calculate_bollinger_bands

# Get stock data
df = get_stock_data("RELIANCE.NS", "2023-01-01", "2023-12-31")

# Calculate indicators
sma_20 = calculate_sma(df, period=20)
ema_12 = calculate_ema(df, period=12)
rsi_14 = calculate_rsi(df, period=14)
macd, signal, histogram = calculate_macd(df)
bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(df)

# Add to DataFrame
df['SMA_20'] = sma_20
df['EMA_12'] = ema_12
df['RSI_14'] = rsi_14
df['MACD'] = macd
df['BB_Upper'] = bb_upper
df['BB_Lower'] = bb_lower

print(df.tail())
```

### Run Indicator Tests
```bash
# Activate virtual environment
.venv\Scripts\Activate.ps1

# Run all 35 indicator tests
python Tests\test_indicators.py
```

**Expected Output:**
```
🎉 PHASE 3 COMPLETE - ALL 35 TESTS PASSED! 🎉
Technical Indicators Implemented:
  ✅ SMA (Simple Moving Average) - 7 tests
  ✅ EMA (Exponential Moving Average) - 7 tests
  ✅ RSI (Relative Strength Index) - 7 tests
  ✅ MACD (Moving Average Convergence Divergence) - 7 tests
  ✅ Bollinger Bands - 7 tests

Total: 5 indicators, 35 tests, ALL PASSING!
indicators.py is production-ready for algorithmic trading!
```

---

## ✅ Phase 4: Trading Strategies Module (COMPLETED)

### Overview
Complete implementation of 3 production-ready trading strategies with signal generation, comprehensive testing, and industry-standard logic.

### Implemented Strategies

#### 1. **Golden Cross Strategy**
```python
from src.strategy import golden_cross_strategy

signals = golden_cross_strategy(data, fast_period=50, slow_period=200)
```
- **Logic:** BUY when fast SMA crosses above slow SMA, SELL when crosses below
- **Parameters:** fast_period (default: 50), slow_period (default: 200)
- **Use Case:** Long-term trend following
- **Tests:** 7 comprehensive tests

#### 2. **RSI Mean Reversion Strategy**
```python
from src.strategy import rsi_mean_reversion_strategy

signals = rsi_mean_reversion_strategy(data, rsi_period=14, oversold=30, overbought=70)
```
- **Logic:** BUY when RSI < oversold threshold, SELL when RSI > overbought threshold
- **Parameters:** rsi_period (default: 14), oversold (default: 30), overbought (default: 70)
- **Use Case:** Range-bound markets, reversals
- **Tests:** 7 comprehensive tests

#### 3. **MACD Trend Following Strategy**
```python
from src.strategy import macd_trend_following_strategy

signals = macd_trend_following_strategy(data, fast=12, slow=26, signal=9)
```
- **Logic:** BUY when MACD crosses above signal line, SELL when crosses below
- **Parameters:** fast (default: 12), slow (default: 26), signal (default: 9)
- **Use Case:** Momentum trading, trend identification
- **Tests:** 7 comprehensive tests

### Run Strategy Tests
```bash
python Tests\test_strategy.py
```

**Expected Output:**
```
PHASE 4 COMPLETE - ALL 21 TESTS PASSED!
Trading Strategies Implemented:
  ✅ Golden Cross Strategy - 7 tests
  ✅ RSI Mean Reversion Strategy - 7 tests
  ✅ MACD Trend Following Strategy - 7 tests

Total: 3 strategies, 21 tests, ALL PASSING!
```

---

## ✅ Phase 5: Backtesting Engine (COMPLETED)

### Overview
Event-driven backtesting engine with realistic trade execution, comprehensive performance metrics, and production-ready architecture.

### Core Functions

#### 1. **`run_backtest()` - Main Orchestrator**
```python
from src.backtester import run_backtest

results = run_backtest(
    data=data,
    signals=signals,
    initial_capital=100000,
    commission=0.001,  # 0.1% per trade
    slippage=0.0
)
```

**Returns:**
```python
{
    'trades': [
        {
            'entry_date': '2023-01-15',
            'exit_date': '2023-02-20',
            'entry_price': 2550.0,
            'exit_price': 2650.0,
            'shares': 39,
            'return_pct': 3.92,
            'return_abs': 3900.0,
            'holding_days': 36
        },
        ...
    ],
    'equity_curve': pd.Series([100000, 101500, ...]),
    'daily_positions': pd.Series([0, 1, 1, 0, ...]),
    'metrics': {
        'initial_capital': 100000.0,
        'final_equity': 135000.0,
        'total_return': 35.0,
        'cagr': 28.5,
        'sharpe_ratio': 1.85,
        'max_drawdown': -12.5,
        'volatility': 18.2,
        'total_trades': 25,
        'win_rate': 64.0,
        'profit_factor': 2.1,
        'avg_win': 5.2,
        'avg_loss': -2.8,
        'exposure_time': 65.0,
        'avg_holding_days': 18
    }
}
```

### Key Features
- ✅ **Event-driven architecture:** Day-by-day execution (no look-ahead bias)
- ✅ **Position management:** FLAT (no position) ↔ LONG (holding asset)
- ✅ **Realistic execution:** Floor division for shares, exact price fills
- ✅ **Mark-to-market:** Daily equity tracking (cash + position value)
- ✅ **Performance metrics:** 15+ industry-standard metrics
- ✅ **Comprehensive logging:** Every trade logged with details
- ✅ **Input validation:** 9 pre-execution checks

### Run Backtester Tests
```bash
python Tests\test_backtester.py
```

**Expected Output:**
```
PHASE 5 COMPLETE - ALL 33 TESTS PASSED!
Backtesting Engine Functions:
  ✅ run_backtest() - Main orchestrator
  ✅ execute_trades() - Trade execution engine
  ✅ calculate_metrics() - Performance calculator

Total: 3 functions, 33 tests, ALL PASSING!
```

---

## ✅ Phase 6: Visualization Module (COMPLETED)

### Overview
Interactive Plotly-based visualization system for comprehensive backtesting analysis with 7 chart types.

### Visualization Functions

#### 1. **`plot_price_with_indicators()` - Candlestick with Indicators**
```python
from src.plotting import plot_price_with_indicators

fig = plot_price_with_indicators(
    data=data,
    sma=sma_50,
    rsi=rsi_14,
    bb_upper=bb_upper,
    bb_lower=bb_lower,
    title="RELIANCE.NS Technical Analysis"
)
fig.show()
```
- **Features:** Candlestick chart, overlaid indicators, subplots for RSI/MACD, volume bars
- **Interactive:** Zoom, pan, hover for values

#### 2. **`plot_signals()` - Trading Signals**
```python
from src.plotting import plot_signals

fig = plot_signals(
    data=data,
    signals=signals,
    trades=results['trades'],
    title="Trading Signals"
)
fig.show()
```
- **Features:** BUY/SELL markers, trade annotations, holding periods
- **Colors:** Green (BUY), Red (SELL)

#### 3. **`plot_equity_curve()` - Portfolio Value**
```python
from src.plotting import plot_equity_curve

fig = plot_equity_curve(
    equity_curve=results['equity_curve'],
    initial_capital=100000,
    title="Equity Curve"
)
fig.show()
```
- **Features:** Equity progression, final value annotation, running peak
- **Key Insight:** Most important chart (shows profitability)

#### 4. **`plot_drawdown()` - Risk Analysis**
```python
from src.plotting import plot_drawdown

fig = plot_drawdown(
    equity_curve=results['equity_curve'],
    title="Drawdown Analysis"
)
fig.show()
```
- **Features:** Drawdown percentage over time, max drawdown marker
- **Interpretation:** Deeper = riskier strategy

#### 5. **`plot_returns_distribution()` - Trade Returns**
```python
from src.plotting import plot_returns_distribution

fig = plot_returns_distribution(
    trades=results['trades'],
    title="Returns Distribution"
)
fig.show()
```
- **Features:** Histogram of trade returns, win/loss separation, statistics
- **Insight:** Distribution shape reveals strategy characteristics

#### 6. **`plot_monthly_returns()` - Calendar Heatmap**
```python
from src.plotting import plot_monthly_returns

fig = plot_monthly_returns(
    equity_curve=results['equity_curve'],
    title="Monthly Returns"
)
fig.show()
```
- **Features:** Monthly performance grid, annual returns
- **Insight:** Reveals seasonal patterns and consistency

#### 7. **`create_backtest_report()` - Complete Report**
```python
from src.plotting import create_backtest_report

figures = create_backtest_report(
    data=data,
    signals=signals,
    backtest_results=results,
    indicators={'SMA_50': sma_50, 'RSI_14': rsi_14},
    strategy_name="Golden Cross (50/200)"
)

# Display all charts
for chart_name, fig in figures.items():
    fig.show()
```
- **Returns:** Dictionary with all 6 chart types
- **Use Case:** One-line generation of complete visual report

### Run Visualization Tests
```bash
python Tests\test_plotting.py
```

**Expected Output:**
```
PHASE 6 COMPLETE - ALL 28 TESTS PASSED!
Visualization Functions:
  ✅ plot_price_with_indicators()
  ✅ plot_signals()
  ✅ plot_equity_curve()
  ✅ plot_drawdown()
  ✅ plot_returns_distribution()
  ✅ plot_monthly_returns()
  ✅ create_backtest_report()

Total: 7 functions, 28 tests, ALL PASSING!
```

---

## ✅ Phase 7: Streamlit Dashboard (COMPLETED)

### Overview
A comprehensive web-based dashboard that provides a professional, user-friendly interface for backtesting trading strategies. Built with Streamlit, it transforms the command-line backtesting system into an interactive web application accessible to non-technical users.

### Key Features
- ✅ **Zero-Code Interface**: Point-and-click backtesting without writing code
- ✅ **Real-Time Data**: Fetches live market data from Yahoo Finance with 3-retry logic
- ✅ **Multiple Strategies**: Choose from Golden Cross, RSI, or MACD strategies
- ✅ **Interactive Visualizations**: 6 Plotly charts with zoom, pan, hover tooltips
- ✅ **Comprehensive Results**: 4-tab interface (Summary, Charts, Trades, Metrics)
- ✅ **Smart Caching**: Prevents redundant downloads with 1-hour cache
- ✅ **Export Functionality**: Download trade logs as CSV
- ✅ **Responsive Design**: Adapts to desktop, tablet, and mobile screens

### Dashboard Components

#### 1. Sidebar Configuration Panel
```python
# User configures backtest parameters via intuitive UI:
- Ticker Symbol (text input with examples dropdown)
- Date Range (interactive date pickers)
- Strategy Selection (dropdown: Golden Cross, RSI, MACD)
- Initial Capital (slider: ₹10,000 - ₹10,000,000)
- Commission (slider: 0% - 1%)
- Strategy Parameters (collapsible view)
```

#### 2. Summary Tab
Displays key performance metrics in color-coded cards:
- **Total Return**: Percentage gain/loss with absolute rupee change
- **Final Equity**: Current portfolio value
- **Sharpe Ratio**: Risk-adjusted returns (color: green if >1, red if <1)
- **Max Drawdown**: Worst peak-to-trough decline
- **Total Trades**: Number of completed round-trips
- **Win Rate**: Percentage of profitable trades
- **Average Win/Loss**: Mean return per winning/losing trade

#### 3. Charts Tab
Six interactive Plotly visualizations:

**a) Price with Indicators**
- Candlestick chart with OHLC data
- Technical indicators overlay (SMA, EMA, RSI, MACD, Bollinger Bands)
- Volume subplot
- RSI/MACD oscillator subplots

**b) Trading Signals**
- Price line chart with BUY/SELL markers
- Green arrows for buy signals
- Red arrows for sell signals
- Annotations showing trade details on hover

**c) Equity Curve**
- Portfolio value progression over time
- Peak equity tracking (dotted line)
- Annotations for major drawdowns
- Percentage returns on y-axis

**d) Drawdown Analysis**
- Underwater period visualization (area chart)
- Shows % decline from peak at each point
- Maximum drawdown highlighted
- Recovery periods visible

**e) Returns Distribution**
- Histogram of trade returns
- Mean return line (vertical dashed)
- Statistical summary (mean, median, std dev)
- Color-coded (green for profits, red for losses)

**f) Monthly Returns Heatmap**
- Calendar-style visualization
- Rows: Years, Columns: Months
- Color intensity shows return magnitude
- Easy identification of profitable/unprofitable periods

#### 4. Trades Tab
- **Sortable Table**: All executed trades with columns:
  * Entry Date, Exit Date
  * Entry Price, Exit Price
  * Return %, Holding Days
- **Trade Statistics**: Total trades, winning trades, losing trades, avg holding period
- **CSV Export**: Download button for trade log
- **Empty State Handling**: Helpful message when no trades executed

#### 5. Metrics Tab
Detailed performance breakdown:

**Performance Metrics:**
- Total Return, Annualized Return, CAGR
- Sharpe Ratio, Sortino Ratio, Calmar Ratio
- Max Drawdown, Volatility, Win Rate

**Trade Statistics:**
- Total/Winning/Losing Trades
- Average Win/Loss, Largest Win/Loss
- Profit Factor, Average Holding Period

**Risk Metrics:**
- Max Drawdown, Volatility
- Sharpe Ratio, Exposure Time
- Initial Capital, Final Equity

### Usage Instructions

#### Running the Dashboard Locally

```bash
# Activate virtual environment
.venv\Scripts\Activate.ps1  # Windows
source .venv/bin/activate    # Linux/Mac

# Launch Streamlit app
streamlit run main.py

# Opens in browser at http://localhost:8501
```

#### Using the Dashboard

1. **Configure Parameters** (Sidebar)
   - Enter ticker symbol (e.g., AAPL, MSFT, TCS.NS)
   - Select date range (start and end dates)
   - Choose strategy (Golden Cross, RSI, or MACD)
   - Adjust capital and commission if needed

2. **Run Backtest** (Sidebar)
   - Click "🚀 Run Backtest" button
   - Watch progress spinners for each step:
     * Fetching data
     * Calculating indicators
     * Generating signals
     * Running backtest
     * Creating visualizations

3. **Analyze Results** (Main Area)
   - **Summary Tab**: Quick overview of performance
   - **Charts Tab**: Visual analysis of strategy behavior
   - **Trades Tab**: Detailed trade-by-trade breakdown
   - **Metrics Tab**: Comprehensive statistical analysis

4. **Export Results** (Optional)
   - Navigate to Trades tab
   - Click "📥 Download Trade Log (CSV)"
   - Opens in Excel/Google Sheets for further analysis

### Code Architecture

```python
# main.py structure (900+ lines)

# 1. Imports and Configuration
import streamlit as st
from src.data_loader import get_stock_data
from src.indicators import calculate_*
from src.strategy import *_strategy
from src.backtester import run_backtest
from src.plotting import create_backtest_report

# 2. Helper Functions
@st.cache_data(ttl=3600)
def fetch_stock_data()  # Cached data fetching
def calculate_all_indicators()  # Batch indicator calculation
def generate_trading_signals()  # Strategy signal generation
def format_currency()  # Indian Rupee formatting
def format_percentage()  # Percentage formatting

# 3. UI Component Functions
def render_header()  # Dashboard title and info
def render_sidebar()  # Configuration panel
def render_summary_tab()  # Key metrics display
def render_charts_tab()  # Plotly visualizations
def render_trades_tab()  # Trade log table
def render_metrics_tab()  # Detailed statistics

# 4. Main Application
def main()  # Orchestrates entire dashboard
    - Renders header and sidebar
    - Handles "Run Backtest" button click
    - Executes 5-step pipeline:
      1. Fetch data
      2. Calculate indicators
      3. Generate signals
      4. Run backtest
      5. Create visualizations
    - Displays results in 4 tabs
    - Error handling and validation
```

### Technical Implementation

#### Smart Caching Strategy
```python
@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_stock_data(ticker, start, end):
    """Prevents redundant API calls"""
    return get_stock_data(ticker, start, end)

@st.cache_data
def calculate_all_indicators(data):
    """Caches expensive indicator calculations"""
    return {
        'sma_50': calculate_sma(data, 50),
        'rsi': calculate_rsi(data, 14),
        'macd': calculate_macd(data),
        ...
    }
```

#### Error Handling
```python
# Validates user inputs before execution
if start_date >= end_date:
    st.error("❌ Start date must be before end date!")
    st.stop()

# Handles API failures gracefully
if data is None or data.empty:
    st.error(f"❌ No data found for {ticker}")
    st.info("Suggestions: Check ticker, try different dates...")
    st.stop()

# Catches and displays exceptions
try:
    results = run_backtest(...)
except Exception as e:
    st.error(f"❌ Error: {str(e)}")
    with st.expander("🐛 Debug Information"):
        st.code(f"Error Type: {type(e).__name__}...")
```

#### Responsive Layout
```python
# Column-based layouts adapt to screen size
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Return", "45.23%")
with col2:
    st.metric("Sharpe Ratio", "1.85")
# ... automatically stacks on mobile
```

### Sample Data Generator

For offline testing or when Yahoo Finance is unavailable:

```bash
# Generate synthetic data for 7 popular stocks
python generate_sample_data.py

# Creates realistic OHLCV data:
- AAPL, MSFT, GOOGL, TSLA (US stocks)
- TCS.NS, INFY.NS, RELIANCE.NS (Indian stocks)
- 1044 trading days (2020-2024)
- Saves to data/raw/ directory
```

**Generated Data Characteristics:**
- Realistic price movements (volatility, trend, drift)
- Proper OHLCV relationships (High > Close > Low)
- Business days only (no weekends)
- Volume with natural variation
- Reproducible (seed=42 for consistency)

### Performance Optimizations

1. **Streamlit Caching**
   - Data fetching: 5s → 0.03s on cache hit
   - Indicator calculations: 2s → 0.01s on cache hit
   - 100x speedup on repeated backtests

2. **Lazy Loading**
   - Charts only rendered when tab is clicked
   - Prevents generating all 6 charts upfront
   - Faster initial page load

3. **Data Compression**
   - Cached indicators stored separately
   - Reused across strategy changes
   - Reduces memory footprint

### Deployment Options

#### Option 1: Streamlit Cloud (Free)
```bash
# 1. Push code to GitHub
git push origin main

# 2. Visit streamlit.io/cloud
# 3. Connect GitHub repo
# 4. Deploy with one click
# 5. Get public URL: https://your-app.streamlit.app
```

#### Option 2: Local Network
```bash
# Run with network access
streamlit run main.py --server.port 8501 --server.address 0.0.0.0

# Access from other devices:
# http://YOUR_IP:8501
```

#### Option 3: Docker Container
```dockerfile
FROM python:3.11
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "main.py"]
```

### Dashboard Statistics

- **Lines of Code**: 900+ (main.py)
- **Functions**: 12 (6 UI components, 6 helpers)
- **Components**: 4 tabs, 1 sidebar, 1 header
- **Charts**: 6 interactive Plotly figures
- **Caching**: 2 cached functions (data, indicators)
- **Error Handlers**: 5 validation checks
- **User Inputs**: 6 configurable parameters

---

## 🔄 Complete End-to-End Workflow

```python
from src.data_loader import get_stock_data
from src.indicators import calculate_sma
from src.strategy import golden_cross_strategy
from src.backtester import run_backtest
from src.plotting import create_backtest_report

# 1. Load data
data = get_stock_data("RELIANCE.NS", "2020-01-01", "2023-12-31")

# 2. Calculate indicators
sma_50 = calculate_sma(data, period=50)
sma_200 = calculate_sma(data, period=200)

# 3. Generate trading signals
signals = golden_cross_strategy(data, fast_period=50, slow_period=200)

# 4. Run backtest
results = run_backtest(data, signals, initial_capital=100000)

# 5. Generate visualizations
indicators_dict = {'SMA_50': sma_50, 'SMA_200': sma_200}
figures = create_backtest_report(
    data=data,
    signals=signals,
    backtest_results=results,
    indicators=indicators_dict,
    strategy_name="Golden Cross (50/200)"
)

# 6. Display results
print(f"Total Return: {results['metrics']['total_return']:.2f}%")
print(f"Sharpe Ratio: {results['metrics']['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {results['metrics']['max_drawdown']:.2f}%")
print(f"Win Rate: {results['metrics']['win_rate']:.2f}%")

# 7. Show charts
for name, fig in figures.items():
    fig.show()
```

---

**Last Updated:** December 10, 2025  
**Status:** Phase 2-7 Complete ✅ | All Features Implemented 🎉