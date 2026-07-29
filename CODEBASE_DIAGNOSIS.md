# SGP-II Codebase Diagnosis Report

## Executive Summary

The SGP-II algorithmic trading backtester codebase is generally well-structured, modular, and heavily documented. Core components such as `data_loader.py` and `indicators.py` demonstrate robust error handling, detailed logging, and a solid understanding of financial data processing. The backtesting engine (`src/backtester.py`) successfully mitigates look-ahead bias by implementing a realistic T+1 execution model.

However, there are notable deviations from standard testing practices within the `Tests/` directory. Several test files are written as standalone scripts rather than utilizing the `pytest` framework correctly.

## 1. Backtesting Engine Analysis (`src/backtester.py`)

The backtester is the core engine of the repository and demonstrates high quality, though with a few areas for improvement.

### Strengths

*   **Realistic Execution Model (No Look-Ahead Bias):**
    The engine correctly models real-world trading by processing a signal on day `T` and executing the trade at the `Open` price on day `T+1`.

    ```python
    # src/backtester.py:651
    if position == 'FLAT' and signal == 1:
        # Enter LONG position (BUY) at next day's open
        if i + 1 < len(data) and cash > 0:
            next_date = data.index[i + 1]
            next_row = data.iloc[i + 1]
            execution_price = next_row['Open']
    ```
*   **Performance Optimization:** Iteration over the DataFrame is done using `itertuples()`, which is significantly faster than `iterrows()`.
*   **Extensive Metric Calculation:** Industry-standard metrics (Sharpe Ratio, Max Drawdown, CAGR, Profit Factor) are calculated gracefully, handling edge cases such as zero volatility or division by zero.

### Areas for Improvement / Flaws

*   **Unimplemented Commission and Slippage:** While `commission` and `slippage` are accepted as parameters in `run_backtest()`, they are not actually applied during the trade execution in `execute_trades()`. This can lead to overly optimistic backtest results if users expect these parameters to be functional.

    ```python
    # src/backtester.py:61
    def run_backtest(
        data: pd.DataFrame,
        signals: pd.Series,
        initial_capital: float = 10000.0,
        commission: float = 0.0,
        slippage: float = 0.0
    ) -> Dict:
    # ...
    # src/backtester.py:660
    cost = shares * execution_price
    cash -= cost # Commission and slippage are NOT applied here
    ```

*   **Fractional Shares:** The logic avoids fractional shares via floor division (`int(cash // execution_price)`). However, for high-priced assets or low initial capital, this might lead to uninvested cash or 0 shares purchased.
*   **Long-Only Limitation:** The state machine strictly limits the strategy to `FLAT` or `LONG`. Adding short selling capabilities would require a restructuring of the state machine logic.

## 2. Testing Suite Analysis (`Tests/`)

The most significant issue in the codebase lies in the execution and structure of the testing suite.

### Flawed Test Structure

Files like `test_backtester.py`, `test_strategy.py`, and `test_plotting.py` do not conform to `pytest` conventions. They are written as imperative scripts that print output to standard out and use raw `assert` statements (or `if` statements returning booleans) instead of encapsulating tests within `def test_...()` functions.

Because of this, running `pytest` directly completely skips collecting these tests, leading to false impressions of coverage.

**Example from `test_plotting.py` (Raises `PytestReturnNotNoneWarning`):**
```python
# Tests/test_plotting.py:126
def test_01_invalid_data_type():
    """TEST 1: Non-DataFrame input should raise TypeError"""
    print("\nTEST 1: Invalid data type for plot_price_with_indicators()")

    try:
        fig = plot_price_with_indicators(data="not a dataframe")
        print("[FAIL] TEST 1 FAILED: Should raise TypeError for non-DataFrame")
        return False # <-- Anti-pattern for pytest
    except TypeError as e:
        if "DataFrame" in str(e):
            print("[PASS] TEST 1 PASSED: TypeError raised correctly")
            return True # <-- Anti-pattern for pytest
        else:
            print(f"[FAIL] TEST 1 FAILED: Wrong error message: {e}")
            return False # <-- Anti-pattern for pytest
```

**Example from `test_strategy.py` (Script execution instead of pytest structure):**
```python
# Tests/test_strategy.py:38
print("\nTEST 1: Basic signal generation with manufactured crossover")
print("-" * 80)
# ... code to generate test data ...
buy_count = (signals_test1 == 1).sum()
assert buy_count >= 1, f"Expected at least 1 Golden Cross, got {buy_count}"
```

### Coverage

When running the tests, the overall coverage is ~83%.

*   `src/backtester.py`: 98% coverage
*   `src/strategy.py`: 93% coverage
*   `src/indicators.py`: 96% coverage
*   `src/cli.py`: 0% coverage (CLI tests are missing)

### Mocking

Mocking is used appropriately in certain files (e.g., `test_data_loader.py` and `test_orchestrator.py`) to simulate network calls to `yfinance`. However, `test_yahoo_finance.py` appears to perform live API calls, which is an anti-pattern for unit tests as it makes the suite brittle and dependent on network connectivity.

## 3. General Code Quality

*   **Documentation:** Exceptional. Docstrings are detailed, follow standard conventions, and include mathematical formulas and examples.
*   **Logging:** The usage of `logging` across the modules is thorough, making debugging and tracing system behavior very straightforward.
*   **Deprecation Warnings:** Running the codebase throws future deprecation warnings from pandas.
    ```
    src/data_loader.py:414: FutureWarning: Series.fillna with 'method' is deprecated and will raise in a future version. Use obj.ffill() or obj.bfill() instead.
      df_clean[col] = df_clean[col].fillna(method='ffill')
    ```

## Recommendations

1.  **Refactor Test Suite:** Rewrite `test_backtester.py`, `test_strategy.py`, and `test_plotting.py` to use proper `pytest` functions, leveraging `pytest.raises` for exception handling and asserting conditions without returning boolean values.
2.  **Implement Commission and Slippage:** Integrate the `commission` and `slippage` parameters into the `execute_trades` logic in `src/backtester.py` to provide a fully realistic backtest environment.
3.  **Fix Pandas Deprecations:** Update `.fillna(method='ffill')` to `.ffill()` in `src/data_loader.py`.
4.  **Isolate Network Calls:** Ensure all tests that hit external APIs (like Yahoo Finance) are properly mocked.
