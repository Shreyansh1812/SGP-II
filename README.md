# SGP-II: AI-Augmented Quantitative Equity Decision-Support & Trading Sandbox

An end-to-end, production-grade quantitative trading platform combining technical analysis, machine learning models, LLM-powered market sentiment, fundamental screening, paper trading, model benchmarking, and closed-loop performance tracking for equity markets.

---

## 🌟 Key Features & Capabilities

### 1. 📊 Interactive Multi-Page Streamlit Dashboard
Structured via `st.navigation()` across three dedicated functional domains:
- **Decision Support:**
  - **1. Dashboard (`pages/1_📊_Dashboard.py`):** Real-time recommendation matrix, composite 0–100 confidence scoring, factor explainability breakdown, and 1-click "Quick-Add to Paper Portfolio".
  - **2. Fundamentals (`pages/2_🏥_Fundamentals.py`):** Multi-factor screener covering a 40 mega-cap ticker universe with customizable health toggles (P/E, Debt-to-Equity, Profit Margins, etc.).
  - **3. Sentiment (`pages/3_📰_Sentiment.py`):** Real-time financial news aggregator with LLM-generated sentiment scores (Groq / Gemini) and multi-source RSS ingestion.
- **Engine Room:**
  - **4. ML Engine (`pages/4_🤖_ML_Engine.py`):** Model training center with `TimeSeriesSplit` cross-validation, automated benchmark generation, and side-by-side model comparison tools.
  - **5. Backtester (`pages/5_📈_Backtester.py`):** Realistic next-day open execution engine testing trend-following and mean-reversion strategies with interactive Plotly visual reports.
- **Portfolio & Performance:**
  - **6. Portfolio Manager (`pages/6_💼_Portfolio.py`):** Live paper trading module with full lifecycle tracking (open/close positions) and live real-time P&L calculation via `yfinance`.
  - **7. Performance Tracker (`pages/7_📉_Performance.py`):** Closed-loop evaluation measuring realized forward returns (1D, 1W, 1M, 3M), win rates, and strategy benchmarks against historical audit trails.

### 2. 🤖 Machine Learning & Benchmark Validation
- **Feature Engineering:** Computes 14 lagged technical features, return momentum, volatility metrics, and binary targets (`Target_Buy`).
- **Model Training:** Supports XGBoost and Random Forest classifiers with walk-forward validation (`TimeSeriesSplit`).
- **Holdout Benchmarking:** Automated 60/20/20 chronological train/val/test splits, financial simulation metrics (directional accuracy, simulated win rate, maximum drawdown), and automated model versioning (`v{timestamp}`).

### 3. 🎯 Composite Recommendation & Explainability
- **Scoring Engine (0–100):** Weighted multi-factor score:
  - **ML Probability:** 35%
  - **Technical Indicators:** 25%
  - **Fundamental Health:** 20%
  - **LLM Sentiment:** 15%
  - **Risk/Volatility Penalty:** -5%
- **Explainability:** Clear factor contribution breakdowns and transparent rationale without black-box hallucination.
- **Sentiment Gating:** Automatically flags signals as `"held pending sentiment"` when negative sentiment contradicts technical or ML signals.

### 4. 📰 Resilient Sentiment & News Ingestion
- **RSS Ingestion:** RSS-based financial news scraping to overcome web rate limits.
- **Dual LLM Client:** Integration with **Groq** (ultra-fast inference) and **Google Gemini**, with automatic fallbacks and JSON-structured outputs.
- **Network Resilience:** Exponential backoff retry decorators (`@resilient`, `safe_fetch`, `graceful_api_call`) for network requests and market data feeds.

### 5. 🗄️ Relational Database & Immutable Audit Trail
Managed SQLite database (`data/sgp2.db`) with SQLAlchemy connection pooling and dedicated Data Access Objects (DAOs):
1. `company_fundamentals`: Ticker fundamental metrics and health flags.
2. `daily_sentiment`: Daily sentiment scores, confidence, and source summaries.
3. `daily_recommendations`: Raw computed daily trade recommendations.
4. `recommendation_audit_trail`: Append-only, immutable record capturing full factor snapshots and scores at recommendation time.
5. `portfolio_positions`: Active and historical paper-trading positions linked to audit records.
6. `model_benchmark_reports`: Stored model evaluation metrics on unseen holdout test sets.
7. `recommendation_performance`: Forward return evaluation (1D, 1W, 1M, 3M) with unique constraint on recommendation IDs.

### 6. 📈 Execution & Realistic Backtesting Engine
- **Look-Ahead Bias Elimination:** Signals generated on bar $T$ execute on bar $T+1$ open.
- **Transaction Costs:** Configurable commissions and slippage modeling.
- **Visual Analytics:** Interactive Plotly charts including equity curves, underwater drawdown profiles, returns distribution, and monthly returns heatmaps.

---

## 🏗️ Architecture & Project Structure

```text
SGP-II/
├── main.py                          # Streamlit application entry point (st.navigation)
├── config.py                        # Legacy config bridge
├── requirements.txt                 # Project dependencies
├── generate_sample_data.py          # Sample data generator for development
│
├── pages/                           # Streamlit multi-page interface
│   ├── 1_Dashboard.py            # Daily recommendation matrix & quick-add
│   ├── 2_Fundamentals.py         # 40-ticker fundamental universe screener
│   ├── 3_Sentiment.py            # RSS & LLM news sentiment feed
│   ├── 4_ML_Engine.py            # Model training & benchmark comparisons
│   ├── 5_Backtester.py           # Technical strategy backtesting
│   ├── 6_Portfolio.py            # Paper trading & live P&L tracking
│   └── 7_Performance.py          # Forward return & recommendation evaluation
│
├── src/                             # Core system modules
│   ├── __init__.py
│   ├── cli.py                       # Unified command-line interface (11 commands)
│   ├── config.py                    # Pydantic Settings & environment config
│   ├── data_loader.py               # Data loading, validation, and CSV cache
│   ├── data_provider.py             # Data provider abstraction (Yahoo, etc.)
│   ├── indicators.py                # Technical indicators (SMA, EMA, RSI, MACD, BB)
│   ├── strategy.py                  # Rule-based strategies (Golden Cross, RSI, MACD)
│   ├── backtester.py                # Backtesting engine & risk metrics
│   ├── plotting.py                  # Plotly interactive chart generation
│   │
│   ├── database/                    # Persistence layer
│   │   ├── connection.py            # SQLAlchemy pooling & engine management
│   │   ├── dao.py                   # DAOs for all 7 database tables
│   │   └── seeder.py                # Synthetic demo data generation
│   │
│   ├── ml/                          # Machine learning pipeline
│   │   ├── feature_engine.py        # 14 technical features & target generation
│   │   ├── trainer.py               # XGBoost/RF trainer with holdout benchmark
│   │   ├── splitter.py              # TimeSeriesSplit & chronological splitters
│   │   ├── baselines.py             # Baseline model benchmarks
│   │   └── evaluator.py             # Metrics & evaluation helpers
│   │
│   ├── sentiment/                   # Sentiment analysis engine
│   │   ├── llm_client.py            # Groq & Gemini LLM connectors
│   │   ├── news_sources.py          # Multi-feed RSS scraper
│   │   └── sentiment_engine.py      # Sentiment scoring orchestrator
│   │
│   ├── screener/                    # Universe screening
│   │   ├── universe.py              # 40 mega-cap ticker definitions
│   │   ├── fundamental_screener.py  # Fundamental metrics & health masks
│   │   └── news_scraper.py          # Company news fetcher
│   │
│   ├── scoring/                     # Recommendation scoring
│   │   └── recommendation_scorer.py # 0–100 multi-factor composite scorer
│   │
│   ├── signals/                     # Signal generation
│   │   └── generator.py             # Multi-source signal synthesizer & audit logger
│   │
│   ├── portfolio/                   # Paper trading & validation
│   │   ├── portfolio_manager.py     # Position lifecycle & live pricing
│   │   └── performance_tracker.py   # Forward return tracker (1D, 1W, 1M, 3M)
│   │
│   └── utils/                       # Shared utilities
│       └── resilience.py            # Exponential backoff retry utilities
│
├── Tests/                           # Comprehensive test suite (26 test files)
│   ├── test_backtester.py
│   ├── test_dao.py
│   ├── test_database.py
│   ├── test_e2e_pipeline.py
│   ├── test_indicators.py
│   ├── test_llm_client.py
│   ├── test_ml.py
│   ├── test_ml_benchmark.py
│   ├── test_new_cli.py
│   ├── test_new_daos.py
│   ├── test_performance_tracker.py
│   ├── test_portfolio_manager.py
│   ├── test_recommendation_scorer.py
│   ├── test_resilience.py
│   ├── test_screener.py
│   ├── test_sentiment_api.py
│   ├── test_signal_generator.py
│   └── ...
│
└── data/                            # Persistent data store (SQLite db, cache)
    ├── sgp2.db
    ├── raw/
    └── processed/
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- Virtual environment tool (`venv`)
- Git

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd SGP-II
   ```

2. **Set up a virtual environment:**
   ```powershell
   # Windows PowerShell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```
   *(For Linux/macOS: `source venv/bin/activate`)*

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables:**
   Copy the example environment file and add your API keys:
   ```powershell
   cp .env.example .env
   ```
   Edit `.env` to configure:
   ```ini
   GROQ_API_KEY=your_groq_api_key_here
   GEMINI_API_KEY=your_gemini_api_key_here
   DATABASE_URL=sqlite:///data/sgp2.db
   LOG_LEVEL=INFO
   ```

5. **Initialize and seed the database:**
   ```bash
   python -m src.cli init-db
   python -m src.cli seed-db --days 30
   ```

---

## 💻 Running the Application

### 1. Launch the Streamlit Dashboard
```bash
streamlit run main.py
```
Open your browser at `http://localhost:8501`. Navigate through the sidebar across all 7 pages.

---

## 🛠️ CLI Command Reference

The platform includes a unified command-line interface (`src/cli.py`) for pipeline execution, data processing, model training, and performance tracking:

### Core Pipeline & Database
```bash
# Initialize SQLite database schema
python -m src.cli init-db

# Seed database with synthetic demo data
python -m src.cli seed-db --days 30

# Run fundamental screening across the universe
python -m src.cli run-screener

# Run news scraping and LLM sentiment scoring
python -m src.cli run-sentiment --date 2026-08-30

# Generate daily signals and log to audit trail
python -m src.cli run-daily-signals --date 2026-08-30

# Execute end-to-end daily pipeline
python -m src.cli run-pipeline --date 2026-08-30
```

### Model Training & Benchmarking
```bash
# Standard ML model training
python -m src.cli train-model --ticker AAPL --years 6

# Train ML model with full holdout benchmark report
python -m src.cli train-model-benchmark --ticker AAPL --years 6

# Compare recent model benchmark results
python -m src.cli compare-models --limit 5
```

### Portfolio & Forward Evaluation
```bash
# Display summary of active paper trading positions & live P&L
python -m src.cli portfolio-summary

# Evaluate recommendation forward returns (1D, 1W, 1M, 3M)
python -m src.cli evaluate-recommendations --limit 50
```

---

## 🧪 Running Tests

The test suite includes over 100 unit, integration, and end-to-end tests across 26 test files:

```bash
# Run the complete test suite
python -m pytest Tests

# Run specific functional test suites
python -m pytest Tests/test_portfolio_manager.py
python -m pytest Tests/test_performance_tracker.py
python -m pytest Tests/test_ml_benchmark.py
python -m pytest Tests/test_recommendation_scorer.py
python -m pytest Tests/test_e2e_pipeline.py
```

---

## ⚙️ Core Configuration Options

Centralized configuration is managed via Pydantic in `src/config.py` and `.env`:

| Parameter | Default | Description |
|---|---|---|
| `DATABASE_URL` | `sqlite:///data/sgp2.db` | SQLAlchemy connection string |
| `GROQ_API_KEY` | `""` | API key for Groq LLM inference |
| `GEMINI_API_KEY` | `""` | API key for Google Gemini LLM |
| `PRIMARY_LLM` | `groq` | Primary LLM provider (`groq` or `gemini`) |
| `UNIVERSE_SIZE` | `40` | Default mega-cap equity universe size |
| `INITIAL_CAPITAL`| `100000.0` | Default paper portfolio / backtest capital |
| `DEFAULT_COMMISSION`| `0.001` | Simulated commission rate per trade |

---

## 🛡️ License & Disclaimer

This project is developed for educational, academic, and quantitative research purposes under the IT452 Minor Project course. It is not financial or investment advice. Always perform thorough risk evaluation and due diligence before committing real capital to financial markets.
