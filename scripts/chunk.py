"""CLI for Phase 3 chunking.

python scripts/chunk.py --stats
python scripts/chunk.py --ticker AAPL --item 1A --show 3
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter

from edgar_rag.chunking.pipeline import chunk_filing
from edgar_rag.chunking.splitter import FixedSplitter, SemanticSplitter
from edgar_rag.config import get_settings
from edgar_rag.embeddings.sentence_transformer import SentenceTransformerEmbedder
from edgar_rag.ingest.manifest import Manifest
from edgar_rag.parsing.pipeline import parse_all


def main() -> int:
    parser = argparse.ArgumentParser(description="Chunk parsed filings")
    parser.add_argument("--ticker", help="limit to one ticker")
    parser.add_argument("--item", help="limit to one item, e.g. 1A")
    parser.add_argument("--show", type=int, default=0, help="print this many chunks")
    parser.add_argument("--stats", action="store_true", help="print corpus statistics")
    parser.add_argument("--fixed", action="store_true", help="use fixed splitting, no embeddings")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    settings = get_settings()

    splitter = (
        FixedSplitter(settings.chunk_target_tokens, settings.chunk_overlap_tokens)
        if args.fixed
        else SemanticSplitter(
            embedder=SentenceTransformerEmbedder(settings=settings),
            target_tokens=settings.chunk_target_tokens,
            overlap_tokens=settings.chunk_overlap_tokens,
            breakpoint_percentile=settings.semantic_breakpoint_percentile,
        )
    )

    manifest = Manifest.load(settings.local_storage_path / "manifest.json")
    parsed = parse_all(settings=settings, tickers=[args.ticker] if args.ticker else None)
    if not parsed:
        print("no filings parsed; run scripts/ingest.py first")
        return 1

    all_chunks = []
    for filing_id, sections in parsed.items():
        if args.item:
            sections = [s for s in sections if s.item.upper() == args.item.upper()]
        all_chunks.extend(chunk_filing(manifest.filings[filing_id], sections, splitter))

    print(f"\n{len(all_chunks):,} chunks from {len(parsed)} filings")

    if args.stats and all_chunks:
        tokens = [c.token_count or 0 for c in all_chunks]
        tokens.sort()
        print(f"  tokens  min={tokens[0]}  median={tokens[len(tokens) // 2]}  max={tokens[-1]}")
        print(f"  mean    {sum(tokens) / len(tokens):.0f}")
        by_item = Counter(c.metadata.section_label for c in all_chunks)
        print("  by section:")
        for label, count in by_item.most_common(12):
            print(f"    {label:8} {count:6,}")

    for chunk in all_chunks[: args.show]:
        meta = chunk.metadata
        print(
            f"\n--- {meta.ticker} {meta.form_type.value} FY{meta.fiscal_year} "
            f"[{meta.citation}] {chunk.token_count} tokens ---"
        )
        print(chunk.text[:600])

    return 0


if __name__ == "__main__":
    sys.exit(main())
