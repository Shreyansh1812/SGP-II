"""
src/cli.py
===========

Unified Command-Line Interface (CLI) for SGP-II Quantitative Trading Engine.

Provides all pipeline operations as modular CLI subcommands, enabling
headless execution, cron scheduling, and CI/CD integration.

Commands
--------
- ``init-db``                   — Initialize database schema
- ``run-screener``              — Execute weekly fundamental screening
- ``run-sentiment``             — Run daily LLM sentiment analysis
- ``train-model``               — Train ML model on historical data
- ``train-model-benchmark``     — Train ML model with holdout test & benchmark report
- ``run-daily-signals``         — Generate daily trading signals
- ``run-pipeline``              — Execute full end-to-end daily pipeline
- ``portfolio-summary``         — Display paper trading positions & live P&L
- ``evaluate-recommendations``   — Realized return tracking of audit recommendations
- ``compare-models``            — Side-by-side benchmark comparison of model versions

Usage
-----
::

    python -m src.cli init-db
    python -m src.cli run-screener
    python -m src.cli run-sentiment --date 2026-08-30
    python -m src.cli train-model --ticker AAPL --years 6
    python -m src.cli train-model-benchmark --ticker AAPL --years 6
    python -m src.cli run-daily-signals --date 2026-08-30
    python -m src.cli run-pipeline --date 2026-08-30
    python -m src.cli portfolio-summary
    python -m src.cli evaluate-recommendations --limit 50
    python -m src.cli compare-models --limit 5

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
# COMMAND: seed-db
# ==============================================================================

def cmd_seed_db(args) -> None:
    """Seed database with historical sentiment and recommendation records."""
    from src.database.seeder import DatabaseSeeder

    days = getattr(args, "days", 30) or 30
    print(f"\n🌱 Seeding database with {days} days of historical demo records...")

    seeder = DatabaseSeeder()
    summary = seeder.seed_database(days=days)

    print("\n" + "=" * 60)
    print("SGP-II DATABASE SEEDING SUMMARY")
    print("=" * 60)
    print(f"  Days Seeded          : {summary['days']}")
    print(f"  Date Range           : {summary['date_range'][0]} to {summary['date_range'][1]}")
    print(f"  Tickers Processed    : {summary['tickers_seeded']}")
    print(f"  Sentiment Records    : {summary['sentiment_records']}")
    print(f"  Total Recommendations: {summary['recommendation_records']}")
    print(f"  BUY Signals          : {summary['buy_signals']}")
    print(f"  HOLD Signals         : {summary['hold_signals']}")
    print("=" * 60 + "\n")


# ==============================================================================
# COMMAND: train-model-benchmark
# ==============================================================================

def cmd_train_model_benchmark(args) -> None:
    """Train ML model with proper train/val/test split and generate benchmark report."""
    import yfinance as yf
    import pandas as pd
    from src.ml.feature_engine import compute_technical_features
    from src.ml.trainer import MLModelTrainer

    ticker = getattr(args, "ticker", "AAPL") or "AAPL"
    years = getattr(args, "years", 6) or 6
    model_name = getattr(args, "model_name", "XGBoost_Signal") or "XGBoost_Signal"
    save_path = getattr(args, "save_path", None)

    settings = get_settings()
    model_path = save_path or str(settings.get_model_abs_path())

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

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    print(f"📊 Downloaded {len(df)} bars of data for {ticker}.")

    logger.info("Computing technical features...")
    feature_df = compute_technical_features(df)
    print(f"🔧 Feature matrix: {len(feature_df)} samples × {len(feature_df.columns)} columns.")

    logger.info("Training ML model with holdout test benchmark...")
    trainer = MLModelTrainer()
    result = trainer.train_with_benchmark(
        historical_df=feature_df,
        model_output_path=model_path,
        model_name=model_name,
    )
    test_metrics = result.get("test_metrics", {})
    report_id = result.get("benchmark_report_id")

    print("\n" + "=" * 60)
    print("SGP-II MODEL BENCHMARK TRAINING REPORT")
    print("=" * 60)
    print(f"  Model Name          : {result.get('model_name')}")
    print(f"  Model Version       : {result.get('model_version')}")
    print(f"  Saved Path          : {result.get('model_path')}")
    print(f"  Database Report ID  : {report_id or 'N/A'}")
    print(f"  Training Time       : {result.get('training_time_seconds', 0):.2f}s")
    print(f"  Dataset Period      : {result.get('dataset_period')}")
    print("  --- Predictive Metrics (Held-Out Test Set) ---")
    print(f"  ROC-AUC             : {test_metrics.get('roc_auc', 0):.4f}")
    print(f"  F1 Score            : {test_metrics.get('f1_score', 0):.4f}")
    print(f"  Precision           : {test_metrics.get('precision', 0):.4f}")
    print(f"  Recall              : {test_metrics.get('recall', 0):.4f}")
    print(f"  Accuracy            : {test_metrics.get('accuracy', 0):.4f}")
    print(f"  Directional Accuracy: {test_metrics.get('directional_accuracy', 0):.4f}")
    print("  --- Financial Backtest Metrics ---")
    print(f"  Backtested Return   : {test_metrics.get('backtested_return', 0) * 100:.2f}%")
    print(f"  Sharpe Ratio        : {test_metrics.get('sharpe_ratio', 0):.4f}")
    print(f"  Max Drawdown        : {test_metrics.get('max_drawdown', 0) * 100:.2f}%")
    print(f"  Win Rate            : {test_metrics.get('win_rate', 0) * 100:.2f}%")
    print(f"  Total Trades        : {test_metrics.get('total_trades', 0)}")
    print("=" * 60 + "\n")


# ==============================================================================
# COMMAND: evaluate-recommendations
# ==============================================================================

def cmd_evaluate_recommendations(args) -> None:
    """Evaluate realized performance of past recommendations in audit trail."""
    from src.database.connection import init_db
    from src.portfolio.performance_tracker import PerformanceTracker

    init_db()
    limit = getattr(args, "limit", 100) or 100

    print(f"\n🎯 Evaluating up to {limit} recent recommendations from audit trail...")
    tracker = PerformanceTracker()
    summary = tracker.evaluate_all_recommendations(limit=limit)

    agg = summary.get("aggregate_stats") or {}
    print("\n" + "=" * 60)
    print("SGP-II RECOMMENDATION EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Evaluated Positions : {summary.get('total_evaluated', 0)}")
    print(f"  Evaluation Failures : {summary.get('total_failed', 0)}")
    print("  --- Realized Return Performance ---")
    print(f"  Average Return 1D   : {(agg.get('avg_return_1d') or 0.0) * 100:.2f}%")
    print(f"  Average Return 1W   : {(agg.get('avg_return_1w') or 0.0) * 100:.2f}%")
    print(f"  Average Return 1M   : {(agg.get('avg_return_1m') or 0.0) * 100:.2f}%")
    print(f"  Average Return 3M   : {(agg.get('avg_return_3m') or 0.0) * 100:.2f}%")
    print(f"  Win Rate (1W)       : {(agg.get('win_rate_1w') or 0.0) * 100:.2f}%")
    print(f"  Win Rate (1M)       : {(agg.get('win_rate_1m') or 0.0) * 100:.2f}%")
    print("=" * 60 + "\n")


# ==============================================================================
# COMMAND: portfolio-summary
# ==============================================================================

def cmd_portfolio_summary(args) -> None:
    """Display current paper trading portfolio positions and live P&L."""
    from src.database.connection import init_db
    from src.portfolio.portfolio_manager import PortfolioManager

    init_db()
    pm = PortfolioManager()
    portfolio = pm.get_portfolio_with_live_prices()

    positions = portfolio.get("positions", [])

    print("\n" + "=" * 60)
    print("SGP-II PAPER TRADING PORTFOLIO SUMMARY")
    print("=" * 60)
    print(f"  Total Invested      : ${portfolio.get('total_invested', 0):,.2f}")
    print(f"  Current Value       : ${portfolio.get('current_value', 0):,.2f}")
    print(f"  Total P&L ($)       : ${portfolio.get('total_pnl', 0):+,.2f}")
    print(f"  Total Return (%)    : {portfolio.get('total_return_pct', 0):+.2f}%")
    print(f"  Open Positions Count: {portfolio.get('position_count', 0)}")
    print("=" * 60)

    if positions:
        print("\nOpen Positions:")
        fmt = "{:<8} {:>8} {:>12} {:>12} {:>12} {:>10} {:>8}"
        print(fmt.format("Ticker", "Shares", "Buy Price", "Cur Price", "P&L ($)", "Return", "Score"))
        print("-" * 75)
        for p in positions:
            cur_p = f"${p.get('current_price', 0):.2f}" if p.get('current_price') else "N/A"
            pnl_s = f"${p.get('pnl', 0):+,.2f}" if p.get('pnl') is not None else "N/A"
            ret_s = f"{p.get('pnl_pct', 0):+.2f}%" if p.get('pnl_pct') is not None else "N/A"
            score_s = f"{p.get('recommendation_score') or 'N/A'}"
            print(fmt.format(
                p.get("ticker", ""),
                f"{p.get('quantity', 0):.1f}",
                f"${p.get('purchase_price', 0):.2f}",
                cur_p,
                pnl_s,
                ret_s,
                str(score_s),
            ))
        print("-" * 75 + "\n")
    else:
        print("\nNo open positions in portfolio. Add positions in UI (pages/6_💼_Portfolio.py).\n")


# ==============================================================================
# COMMAND: compare-models
# ==============================================================================

def cmd_compare_models(args) -> None:
    """Display comparison table of historical model benchmark reports."""
    from src.database.connection import init_db
    from src.database.dao import BenchmarkDAO

    init_db()
    limit = getattr(args, "limit", 5) or 5
    model_name = getattr(args, "model_name", None)

    dao = BenchmarkDAO()
    if model_name:
        reports = dao.get_reports_by_model(model_name=model_name)[:limit]
    else:
        reports = dao.get_latest_reports(limit=limit)

    print("\n" + "=" * 85)
    print("SGP-II MODEL BENCHMARK COMPARISON TABLE")
    print("=" * 85)

    if not reports:
        print("  No benchmark reports found in database.")
        print("  Train a benchmarked model using: python -m src.cli train-model-benchmark")
        print("=" * 85 + "\n")
        return

    fmt = "{:<16} {:<18} {:>8} {:>8} {:>9} {:>9} {:>8} {:>9}"
    print(fmt.format("Model", "Version", "ROC-AUC", "F1", "Dir Acc", "Return", "Sharpe", "Max DD"))
    print("-" * 92)
    for r in reports:
        ret_s = f"{r.get('backtested_return', 0) * 100:.1f}%" if r.get('backtested_return') is not None else "N/A"
        dd_s = f"{r.get('max_drawdown', 0) * 100:.1f}%" if r.get('max_drawdown') is not None else "N/A"
        auc_s = f"{r.get('roc_auc', 0):.3f}" if r.get('roc_auc') is not None else "N/A"
        f1_s = f"{r.get('f1_score', 0):.3f}" if r.get('f1_score') is not None else "N/A"
        dir_s = f"{r.get('directional_accuracy', 0):.3f}" if r.get('directional_accuracy') is not None else "N/A"
        sharpe_s = f"{r.get('sharpe_ratio', 0):.2f}" if r.get('sharpe_ratio') is not None else "N/A"

        print(fmt.format(
            str(r.get("model_name", ""))[:15],
            str(r.get("model_version", ""))[:17],
            auc_s,
            f1_s,
            dir_s,
            ret_s,
            sharpe_s,
            dd_s,
        ))
    print("-" * 92 + "\n")


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
  python -m src.cli seed-db --days 30
  python -m src.cli run-sentiment --date 2026-08-30
  python -m src.cli train-model --ticker AAPL --years 6
  python -m src.cli train-model-benchmark --ticker AAPL --years 6
  python -m src.cli run-daily-signals --date 2026-08-30
  python -m src.cli run-pipeline --date 2026-08-30
  python -m src.cli portfolio-summary
  python -m src.cli evaluate-recommendations --limit 50
  python -m src.cli compare-models --limit 5
        """,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # init-db
    sub_init = subparsers.add_parser("init-db", help="Initialize database schema")
    sub_init.set_defaults(func=cmd_init_db)

    # seed-db
    sub_seed = subparsers.add_parser("seed-db", help="Seed database with historical sentiment and recommendations")
    sub_seed.add_argument("--days", type=int, default=30, help="Number of historical business days (default: 30)")
    sub_seed.set_defaults(func=cmd_seed_db)

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

    # train-model-benchmark
    sub_benchmark = subparsers.add_parser(
        "train-model-benchmark",
        help="Train ML model with train/val/test split and generate benchmark report",
    )
    sub_benchmark.add_argument("--ticker", type=str, default="AAPL", help="Training ticker (default: AAPL)")
    sub_benchmark.add_argument("--years", type=int, default=6, help="Years of history (default: 6)")
    sub_benchmark.add_argument("--model-name", type=str, default="XGBoost_Signal", help="Report model name")
    sub_benchmark.add_argument("--save-path", type=str, default=None, help="Custom output model path")
    sub_benchmark.set_defaults(func=cmd_train_model_benchmark)

    # run-daily-signals
    sub_signals = subparsers.add_parser("run-daily-signals", help="Generate daily signals")
    sub_signals.add_argument("--date", type=str, default=None, help="Target date (YYYY-MM-DD)")
    sub_signals.set_defaults(func=cmd_run_daily_signals)

    # run-pipeline
    sub_pipeline = subparsers.add_parser("run-pipeline", help="Run full E2E pipeline")
    sub_pipeline.add_argument("--date", type=str, default=None, help="Target date (YYYY-MM-DD)")
    sub_pipeline.set_defaults(func=cmd_run_pipeline)

    # portfolio-summary
    sub_port = subparsers.add_parser(
        "portfolio-summary",
        help="Display paper trading portfolio positions and current P&L",
    )
    sub_port.set_defaults(func=cmd_portfolio_summary)

    # evaluate-recommendations
    sub_eval = subparsers.add_parser(
        "evaluate-recommendations",
        help="Evaluate realized performance of past recommendations from audit trail",
    )
    sub_eval.add_argument("--limit", type=int, default=100, help="Max audit records to evaluate (default: 100)")
    sub_eval.set_defaults(func=cmd_evaluate_recommendations)

    # compare-models
    sub_cmp = subparsers.add_parser(
        "compare-models",
        help="Display comparison table of historical model benchmark reports",
    )
    sub_cmp.add_argument("--limit", type=int, default=5, help="Number of benchmark reports to compare (default: 5)")
    sub_cmp.add_argument("--model-name", type=str, default=None, help="Filter by model name")
    sub_cmp.set_defaults(func=cmd_compare_models)

    args = parser.parse_args()

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
