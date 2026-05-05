# SGP-II: Algorithmic Trading Backtester

Production-oriented Python backtesting system with a Streamlit dashboard, modular data providers, realistic next-day execution, technical indicators, strategy modules, and interactive reporting.

## What Is Included

- Historical data pipeline with validation, cleaning, and CSV caching
- Data provider abstraction layer (Yahoo Finance implemented, others scaffolded)
- Technical indicators:
  - SMA
  - EMA
  - RSI
  - MACD
  - Bollinger Bands
- Trading strategies:
  - Golden Cross (trend following)
  - RSI Mean Reversion
  - MACD Trend Following
- Backtesting engine with realistic execution model:
  - Signal on day T
  - Execution at next trading day open (T+1)
- Interactive Plotly visualizations:
  - Price + indicators
  - Buy/sell signal chart
  - Equity curve
  - Drawdown chart
  - Returns distribution
  - Monthly returns heatmap
- Streamlit dashboard for end-to-end backtest workflow
- Comprehensive test suites under Tests

## Current Status

All core phases are implemented and integrated:

- Data loading and caching
- Indicators
- Strategies
- Backtester
- Plotting
- Streamlit UI
- Provider abstraction and mocked test setup improvements

This repository is suitable for research, learning, and iterative strategy prototyping.

## Project Structure

```text
SGP-II/
  config.py
  main.py
  requirements.txt
  generate_sample_data.py

  data/
    raw/
    processed/
    test_indicators/

  src/
    __init__.py
    data_loader.py
    data_provider.py
    indicators.py
    strategy.py
    backtester.py
    plotting.py

  Tests/
    test_data_loader.py
    test_validation.py
    test_cleaning.py
    test_caching.py
    test_indicators.py
    test_strategy.py
    test_backtester.py
    test_plotting.py
    test_orchestrator.py
    test_yahoo_finance.py

  notebooks/
    01_data_analysis.ipynb
    02_data_analysis.ipynb
```

## Architecture

```text
UI / Script Input
  -> Data Loader (provider + validation + cleaning + cache)
  -> Indicator Calculation
  -> Strategy Signal Generation
  -> Backtester (next-day open execution)
  -> Metrics + Trade Log
  -> Plotly Report + Streamlit Dashboard
```

## Installation

### Prerequisites

- Python 3.10+
- pip
- Windows PowerShell (or your preferred shell)

### Setup

```powershell
cd C:\Users\shrey\Downloads\SGP-II
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run The Dashboard

```powershell
streamlit run main.py
```

Open the local URL shown by Streamlit (usually http://localhost:8501).

## Programmatic Usage

### 1) Load data

```python
from src.data_loader import get_stock_data

df = get_stock_data("AAPL", "2020-01-01", "2024-01-01")
```

### 2) Generate signals

```python
from src.strategy import golden_cross_strategy

signals = golden_cross_strategy(df, fast_period=50, slow_period=200)
```

### 3) Run backtest

```python
from src.backtester import run_backtest

results = run_backtest(
    data=df,
    signals=signals,
    initial_capital=100000,
    commission=0.001,
)
```

### 4) Build report charts

```python
from src.plotting import create_backtest_report

figures = create_backtest_report(
    data=df,
    signals=signals,
    backtest_results=results,
    strategy_name="Golden Cross"
)
```

## Data Provider Abstraction

The provider layer in src/data_provider.py enables swapping data backends.

Implemented:

- YahooFinanceProvider

Scaffolded placeholders:

- AlphaVantageProvider
- IEXCloudProvider

Example provider usage:

```python
from src.data_provider import create_provider
from src.data_loader import set_data_provider

provider = create_provider("yahoo")
set_data_provider(provider)
```

## Caching Behavior

- Raw market data is cached as CSV in data/raw
- Loader checks cache before API fetch
- Streamlit also applies in-memory cache decorators for dashboard responsiveness

## Execution Model

The backtester uses realistic timing:

- Signals are computed from current bar information
- Trades execute at next bar open

This reduces look-ahead bias compared to same-bar close execution.

## Configuration

Configuration is centralized in config.py and can be overridden with environment variables.

Key variables include:

- TICKER
- START_DATE
- END_DATE
- INITIAL_CASH
- COMMISSION
- GOLDEN_CROSS_FAST
- GOLDEN_CROSS_SLOW
- RSI_PERIOD
- RSI_OVERSOLD
- RSI_OVERBOUGHT
- MACD_FAST
- MACD_SLOW
- MACD_SIGNAL
- LOG_LEVEL

Example:

```powershell
$env:TICKER="AAPL"
$env:START_DATE="2020-01-01"
$env:END_DATE="2024-01-01"
streamlit run main.py
```

## Testing

### Recommended

```powershell
python -m pytest Tests
```

### Individual suites

```powershell
python Tests\test_data_loader.py
python Tests\test_validation.py
python Tests\test_cleaning.py
python Tests\test_caching.py
python Tests\test_indicators.py
python Tests\test_strategy.py
python Tests\test_backtester.py
python Tests\test_plotting.py
python Tests\test_orchestrator.py
python Tests\test_yahoo_finance.py
```

## Notes On Reliability

- Tests were improved with mocked market-data calls for better CI stability
- The architecture now supports provider substitution without changing strategy/backtester code
- Core simulation logic and visualization stack are modular and independently testable

## Known Limitations

- Only Yahoo provider is fully implemented today
- Backtester is long-only (no short-selling workflow)
- Commission is supported; slippage parameter exists but is not fully modeled in all paths
- Not intended for direct live-trading execution

## Troubleshooting

- If PowerShell blocks venv activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

- If ticker fetch fails:
  - Try a US symbol first, for example AAPL or MSFT
  - For NSE symbols, use suffix .NS (example: RELIANCE.NS)
  - Verify internet connectivity and retry

- If chart rendering is slow:
  - Use shorter date ranges first
  - Clear Streamlit cache from sidebar and rerun

## References

- yfinance documentation
- pandas documentation
- Plotly documentation
- Streamlit documentation

## Disclaimer

This project is for education, research, and backtesting experiments. It is not financial advice. Past performance does not guarantee future results.