"""CLI for Phase 4 indexing and search.

python scripts/index.py build
python scripts/index.py search "what are the supply chain risks?"
python scripts/index.py search "revenue growth" --ticker AAPL --year 2025
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

from edgar_rag.config import get_settings
from edgar_rag.embeddings.sentence_transformer import SentenceTransformerEmbedder
from edgar_rag.index.builder import build_index, load_index
from edgar_rag.models import SearchFilter


def main() -> int:
    parser = argparse.ArgumentParser(description="Build and query the vector index")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="chunk, embed and index the corpus")
    build.add_argument("--ticker", help="limit to one ticker")

    settings = get_settings()

    search = sub.add_parser("search", help="query the index")
    search.add_argument("query")
    search.add_argument("--top-k", type=int, default=settings.retrieval_top_k)
    search.add_argument("--ticker", action="append", help="repeatable")
    search.add_argument("--year", type=int, action="append", help="repeatable")
    search.add_argument("--item", action="append", help="repeatable, e.g. 1A")
    search.add_argument("--chars", type=int, default=400)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.command == "build":
        started = time.perf_counter()
        index = build_index(settings=settings, tickers=[args.ticker] if args.ticker else None)
        elapsed = time.perf_counter() - started
        print(f"\nindexed {index.size:,} chunks in {elapsed:.1f}s -> {settings.index_path}")
        return 0

    embedder = SentenceTransformerEmbedder(settings=settings)
    index = load_index(settings=settings, embedder=embedder)

    filters = SearchFilter(
        tickers=args.ticker,
        fiscal_years=args.year,
        items=args.item,
    )

    started = time.perf_counter()
    results = index.search(embedder.embed_query(args.query), args.top_k, filters)
    elapsed_ms = (time.perf_counter() - started) * 1000

    print(f'\n"{args.query}" -> {len(results)} results in {elapsed_ms:.1f}ms')
    for result in results:
        meta = result.chunk.metadata
        print(
            f"\n[{result.dense_rank}] score={result.score:.3f}  "
            f"{meta.ticker} {meta.form_type.value} FY{meta.fiscal_year} "
            f"Item {meta.section_label}"
        )
        print(f"    {result.chunk.text[: args.chars]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
