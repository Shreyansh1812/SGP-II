"""
src/cli.py
===========

Unified Command-Line Interface (CLI) for SGP-II Quantitative Trading Engine.

Provides all pipeline operations as modular CLI subcommands, enabling
headless execution, cron scheduling, and CI/CD integration.

Commands
--------
- ``init-db``           — Initialize database schema
- ``run-screener``      — Execute weekly fundamental screening
- ``run-sentiment``     — Run daily LLM sentiment analysis
- ``train-model``       — Train ML model on historical data
- ``run-daily-signals`` — Generate daily trading signals
- ``run-pipeline``      — Execute full end-to-end daily pipeline

Usage
-----
::

    python -m src.cli init-db
    python -m src.cli run-screener
    python -m src.cli run-sentiment --date 2026-08-30
    python -m src.cli train-model --ticker AAPL --years 6
    python -m src.cli run-daily-signals --date 2026-08-30
    python -m src.cli run-pipeline --date 2026-08-30

Author: Shreyansh Patel
Project: SGP-II — AI-Augmented Decision-Support Sandbox
"""

import sys
import logging
import argparse
from datetime import datetime

from src.config import get_settings

# Ensure UTF-8 output encoding on Windows consoles
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("sgp2.cli")


# ==============================================================================
# COMMAND: init-db
# ==============================================================================

def cmd_init_db(args) -> None:
    """Initialize the database schema from schema.sql."""
    from src.database.connection import init_db

    logger.info("Initializing database schema...")
    init_db()
    print("✅ Database schema initialized successfully.")


# ==============================================================================
# COMMAND: run-screener
# ==============================================================================

def cmd_run_screener(args) -> None:
    """Execute the weekly fundamental screener batch process."""
    from src.database.connection import init_db
    from src.database.dao import FundamentalsDAO
    from src.screener.fundamental_screener import FundamentalScreener

    logger.info("Initializing database...")
    init_db()

    dao = FundamentalsDAO()
    screener = FundamentalScreener()

    logger.info("Executing Fundamental Screener Batch...")
    summary = screener.run_screener_batch(dao)

    print("\n" + "=" * 60)
    print("SGP-II FUNDAMENTAL SCREENER SUMMARY")
    print("=" * 60)
    print(f"  Total Tickers Analyzed : {summary['total']}")
    print(f"  Healthy Tickers        : {summary['healthy']}")
    print(f"  Filtered (Unhealthy)   : {summary['filtered']}")
    print(f"  Execution Time         : {summary['elapsed_seconds']}s")
    print("=" * 60 + "\n")


# ==============================================================================
# COMMAND: run-sentiment
# ==============================================================================

def cmd_run_sentiment(args) -> None:
    """Run daily LLM-powered sentiment analysis pipeline."""
    from src.database.connection import init_db
    from src.sentiment.sentiment_engine import SentimentEngine

    date_str = args.date or datetime.now().strftime("%Y-%m-%d")

    logger.info(f"Initializing database...")
    init_db()

    logger.info(f"Running sentiment analysis for {date_str}...")
    engine = SentimentEngine()
    result = engine.run_daily_sentiment(date_str)

    print("\n" + "=" * 60)
    print("SGP-II SENTIMENT ANALYSIS RESULT")
    print("=" * 60)
    print(f"  Date     : {result['date']}")
    print(f"  Status   : {result['status']}")
    print(f"  Score    : {result['score']:.3f}")
    print(f"  Label    : {result['label'].title()}")
    print(f"  Provider : {result['provider'].title()}")
    print(f"  Summary  : {result['summary']}")
    if result.get("error"):
        print(f"  Error    : {result['error']}")
    print("=" * 60 + "\n")


# ==============================================================================
# COMMAND: train-model
# ==============================================================================

def cmd_train_model(args) -> None:
    """Train XGBoost/RandomForest ML model on historical OHLCV data."""
    import yfinance as yf
    import pandas as pd
    from src.ml.feature_engine import compute_technical_features
    from src.ml.trainer import MLModelTrainer

    years = args.years or 6
    settings = get_settings()
    model_path = str(settings.get_model_abs_path())
    multi_ticker = getattr(args, "multi_ticker", False)

    trainer = MLModelTrainer()

    if multi_ticker:
        # --- Multi-Ticker Universal Training Mode ---
        from src.screener.universe import get_universe

        tickers = get_universe()
        print(f"\n🌍 Multi-Ticker Training Mode: {len(tickers)} tickers × {years}y data.")
        print("This may take several minutes...\n")

        result = trainer.train_multi_ticker_model(
            tickers=tickers,
            years=years,
            model_output_path=model_path,
        )
        metrics = result["metrics"]

        print("\n" + "=" * 60)
        print("SGP-II ML MODEL TRAINING RESULT (UNIVERSAL)")
        print("=" * 60)
        print(f"  Model Path     : {result['model_path']}")
        print(f"  Tickers Trained: {len(result['tickers_trained'])}")
        print(f"  Tickers Failed : {len(result['tickers_failed'])}")
        print(f"  Total Samples  : {result['total_samples']}")
        print(f"  Class 0 (HOLD) : {result['samples_per_class']['class_0']}")
        print(f"  Class 1 (BUY)  : {result['samples_per_class']['class_1']}")
        print(f"  Features       : {len(result['features'])}")
        print(f"  ROC-AUC        : {metrics['roc_auc']:.4f}")
        print(f"  Precision      : {metrics['precision']:.4f}")
        print(f"  Recall         : {metrics['recall']:.4f}")
        print(f"  Brier Score    : {metrics['brier_score']:.4f}")
        if result.get("feature_importances"):
            print("  Top 5 Features :")
            for fi in result["feature_importances"][:5]:
                print(f"    • {fi['feature']}: {fi['importance']:.4f}")
        print("=" * 60 + "\n")

    else:
        # --- Single-Ticker Training Mode ---
        ticker = args.ticker or "AAPL"

        logger.info(f"Downloading {years}y of OHLCV data for {ticker}...")
        df = yf.download(
            ticker.upper(),
            period=f"{years}y",
            interval="1d",
            progress=True,
            auto_adjust=True,
        )

        if df is None or df.empty:
            print(f"❌ No data found for {ticker}. Exiting.")
            sys.exit(1)

        # Handle multi-level columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        print(f"📊 Downloaded {len(df)} bars of data for {ticker}.")

        logger.info("Computing technical features...")
        feature_df = compute_technical_features(df)
        print(f"🔧 Feature matrix: {len(feature_df)} samples × {len(feature_df.columns)} columns.")

        logger.info("Training ML model with TimeSeriesSplit CV...")
        result = trainer.train_xgboost_model(feature_df, model_path)
        metrics = result["metrics"]

        print("\n" + "=" * 60)
        print("SGP-II ML MODEL TRAINING RESULT")
        print("=" * 60)
        print(f"  Model Path : {result['model_path']}")
        print(f"  Features   : {len(result['features'])}")
        print(f"  ROC-AUC    : {metrics['roc_auc']:.4f}")
        print(f"  Precision  : {metrics['precision']:.4f}")
        print(f"  Recall     : {metrics['recall']:.4f}")
        print(f"  Brier Score: {metrics['brier_score']:.4f}")
        print("=" * 60 + "\n")


# ==============================================================================
# COMMAND: run-daily-signals
# ==============================================================================

def cmd_run_daily_signals(args) -> None:
    """Generate daily ML inference signals with sentiment gating."""
    from src.database.connection import init_db
    from src.signals.generator import DailySignalGenerator

    date_str = args.date or datetime.now().strftime("%Y-%m-%d")

    logger.info("Initializing database...")
    init_db()

    logger.info(f"Generating daily signals for {date_str}...")
    generator = DailySignalGenerator()
    signals = generator.generate_daily_signals(date_str)

    buy_count = sum(1 for s in signals if s["signal_type"] == "BUY")
    hold_count = sum(1 for s in signals if s["signal_type"] == "HOLD")

    print("\n" + "=" * 60)
    print("SGP-II DAILY SIGNAL GENERATION RESULT")
    print("=" * 60)
    print(f"  Date         : {date_str}")
    print(f"  Total Signals: {len(signals)}")
    print(f"  BUY Signals  : {buy_count}")
    print(f"  HOLD Signals : {hold_count}")
    print("-" * 60)

    for sig in signals:
        emoji = "🟢" if sig["signal_type"] == "BUY" else "🟡"
        prob = sig.get("prob_buy", 0)
        sent = sig.get("sentiment_score")
        sent_str = f"{sent:.2f}" if sent is not None else "N/A"
        print(f"  {emoji} {sig['ticker']:6s} | {sig['signal_type']:4s} | P(BUY)={prob:.2f} | Sentiment={sent_str}")

    print("=" * 60 + "\n")


# ==============================================================================
# COMMAND: run-pipeline
# ==============================================================================

def cmd_run_pipeline(args) -> None:
    """Execute the full end-to-end daily pipeline."""
    date_str = args.date or datetime.now().strftime("%Y-%m-%d")

    print("\n" + "=" * 60)
    print(f"SGP-II FULL PIPELINE — {date_str}")
    print("=" * 60)

    # Step 1: Init DB
    print("\n📦 Step 1/4: Initializing database...")
    cmd_init_db(args)

    # Step 2: Run Screener
    print("\n🔍 Step 2/4: Running fundamental screener...")
    cmd_run_screener(args)

    # Step 3: Run Sentiment
    print("\n📰 Step 3/4: Running sentiment analysis...")
    # Create a mock args with date
    class SentimentArgs:
        pass
    sent_args = SentimentArgs()
    sent_args.date = date_str
    cmd_run_sentiment(sent_args)

    # Step 4: Generate Signals
    print("\n🎯 Step 4/4: Generating daily signals...")
    class SignalArgs:
        pass
    sig_args = SignalArgs()
    sig_args.date = date_str
    cmd_run_daily_signals(sig_args)

    print("\n" + "=" * 60)
    print("✅ FULL PIPELINE COMPLETE")
    print("=" * 60)
    print(f"\nView results: streamlit run main.py")
    print(f"Date: {date_str}\n")


# ==============================================================================
# MAIN CLI PARSER
# ==============================================================================

def main():
    """Main CLI entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="SGP-II Quantitative Trading Engine CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.cli init-db
  python -m src.cli run-screener
  python -m src.cli run-sentiment --date 2026-08-30
  python -m src.cli train-model --ticker AAPL --years 6
  python -m src.cli run-daily-signals --date 2026-08-30
  python -m src.cli run-pipeline --date 2026-08-30
        """,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # init-db
    sub_init = subparsers.add_parser("init-db", help="Initialize database schema")
    sub_init.set_defaults(func=cmd_init_db)

    # run-screener
    sub_screener = subparsers.add_parser("run-screener", help="Run weekly fundamental screener")
    sub_screener.set_defaults(func=cmd_run_screener)

    # run-sentiment
    sub_sentiment = subparsers.add_parser("run-sentiment", help="Run daily sentiment analysis")
    sub_sentiment.add_argument("--date", type=str, default=None, help="Target date (YYYY-MM-DD)")
    sub_sentiment.set_defaults(func=cmd_run_sentiment)

    # train-model
    sub_train = subparsers.add_parser("train-model", help="Train ML model")
    sub_train.add_argument("--ticker", type=str, default="AAPL", help="Training ticker (default: AAPL)")
    sub_train.add_argument("--years", type=int, default=6, help="Years of history (default: 6)")
    sub_train.add_argument(
        "--multi-ticker",
        action="store_true",
        default=False,
        help="Train universal model on all 40 mega-cap tickers (production mode).",
    )
    sub_train.set_defaults(func=cmd_train_model)

    # run-daily-signals
    sub_signals = subparsers.add_parser("run-daily-signals", help="Generate daily signals")
    sub_signals.add_argument("--date", type=str, default=None, help="Target date (YYYY-MM-DD)")
    sub_signals.set_defaults(func=cmd_run_daily_signals)

    # run-pipeline
    sub_pipeline = subparsers.add_parser("run-pipeline", help="Run full E2E pipeline")
    sub_pipeline.add_argument("--date", type=str, default=None, help="Target date (YYYY-MM-DD)")
    sub_pipeline.set_defaults(func=cmd_run_pipeline)

    args = parser.parse_args()

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
