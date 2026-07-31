"""CLI for Phase 1 ingestion.

python scripts/ingest.py --tickers AAPL MSFT NVDA AMZN GOOGL --limit 4
"""

from __future__ import annotations

import argparse
import logging
import sys

from edgar_rag.ingest.pipeline import ingest
from edgar_rag.models import FormType

DEFAULT_TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest SEC EDGAR filings")
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    parser.add_argument(
        "--limit", type=int, default=4, help="max filings per ticker (most recent first)"
    )
    parser.add_argument(
        "--forms",
        nargs="+",
        default=["10-K", "10-Q"],
        choices=[f.value for f in FormType],
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    report = ingest(
        args.tickers,
        form_types=[FormType(f) for f in args.forms],
        limit_per_ticker=args.limit,
    )

    print(f"\n{report.summary()}")
    for filing, error in report.failed:
        print(f"  FAILED {filing.filing_id}: {error}")
    return 1 if report.failed else 0


if __name__ == "__main__":
    sys.exit(main())
