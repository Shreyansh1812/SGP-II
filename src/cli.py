"""
src/cli.py
Command-Line Interface (CLI) runner for SGP-II quantitative tasks.
"""

import sys
import logging
import argparse

from src.config import get_settings
from src.database.connection import init_db
from src.database.dao import FundamentalsDAO
from src.screener.fundamental_screener import FundamentalScreener

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("sgp2.cli")


def run_screener_command(args) -> None:
    """Executes the weekly fundamental screener batch task."""
    logger.info("Initializing database schema...")
    init_db()

    dao = FundamentalsDAO()
    screener = FundamentalScreener()

    logger.info("Executing Fundamental Screener Batch...")
    summary = screener.run_screener_batch(dao)

    print("\n" + "=" * 60)
    print("SGP-II FUNDAMENTAL SCREENER SUMMARY")
    print("=" * 60)
    print(f"Total Tickers Analyzed : {summary['total']}")
    print(f"Healthy Tickers        : {summary['healthy']}")
    print(f"Filtered (Unhealthy)   : {summary['filtered']}")
    print(f"Execution Time         : {summary['elapsed_seconds']}s")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="SGP-II Quantitative Trading Engine CLI"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Command: run-screener
    screener_parser = subparsers.add_parser(
        "run-screener",
        help="Run weekly fundamental stock health screener"
    )
    screener_parser.set_defaults(func=run_screener_command)

    args = parser.parse_args()

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
