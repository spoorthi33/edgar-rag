"""CLI for Phase 2 parsing.

python scripts/parse.py                      # summarise every filing
python scripts/parse.py --ticker AAPL --item 1A --show
"""

from __future__ import annotations

import argparse
import logging
import sys

from edgar_rag.config import get_settings
from edgar_rag.ingest.manifest import Manifest
from edgar_rag.parsing.pipeline import parse_all


def main() -> int:
    parser = argparse.ArgumentParser(description="Parse stored filings into sections")
    parser.add_argument("--ticker", help="limit to one ticker")
    parser.add_argument("--item", help="show only this item, e.g. 1A")
    parser.add_argument("--show", action="store_true", help="print section text")
    parser.add_argument("--chars", type=int, default=1500, help="characters to print")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    settings = get_settings()
    manifest = Manifest.load(settings.local_storage_path / "manifest.json")
    parsed = parse_all(settings=settings, tickers=[args.ticker] if args.ticker else None)

    if not parsed:
        print("no filings parsed; run scripts/ingest.py first")
        return 1

    filings = [manifest.filings[fid] for fid in parsed]
    for filing in sorted(filings, key=lambda f: (f.ticker or "", -f.fiscal_year)):
        sections = parsed[filing.filing_id]
        if args.item:
            sections = [s for s in sections if s.item.upper() == args.item.upper()]
            if not sections:
                continue

        label = f"{filing.ticker} {filing.form_type.value} FY{filing.fiscal_year}"
        print(f"\n=== {label} ({len(sections)} sections) ===")
        for section in sections:
            part = f"Part {section.part}" if section.part else "-"
            title = section.title[:60]
            print(f"  {part:8} Item {section.item:4} {len(section.text):8,} chars  {title}")
            if args.show:
                print("\n" + section.text[: args.chars] + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
