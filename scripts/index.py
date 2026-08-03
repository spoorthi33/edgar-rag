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
from edgar_rag.index.builder import build_index, load_retrievers
from edgar_rag.models import SearchFilter
from edgar_rag.retrieval.hybrid import HybridRetriever


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
    search.add_argument(
        "--mode",
        choices=["hybrid", "dense", "sparse"],
        default="hybrid",
        help="retrieval strategy (default: hybrid)",
    )
    search.add_argument("--compare", action="store_true", help="show all three modes side by side")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.command == "build":
        started = time.perf_counter()
        index = build_index(settings=settings, tickers=[args.ticker] if args.ticker else None)
        elapsed = time.perf_counter() - started
        print(f"\nindexed {index.size:,} chunks in {elapsed:.1f}s -> {settings.index_path}")
        return 0

    embedder = SentenceTransformerEmbedder(settings=settings)
    index, sparse = load_retrievers(settings=settings, embedder=embedder)
    retriever = HybridRetriever(index=index, embedder=embedder, sparse=sparse, settings=settings)

    filters = SearchFilter(
        tickers=args.ticker,
        fiscal_years=args.year,
        items=args.item,
    )

    modes = ["dense", "sparse", "hybrid"] if args.compare else [args.mode]
    for mode in modes:
        started = time.perf_counter()
        results = retriever.retrieve(args.query, args.top_k, filters, mode=mode)
        elapsed_ms = (time.perf_counter() - started) * 1000

        print(f'\n=== {mode} === "{args.query}" -> {len(results)} in {elapsed_ms:.0f}ms')
        for rank, result in enumerate(results, start=1):
            meta = result.chunk.metadata
            origin = _origin(result)
            print(
                f"\n[{rank}] score={result.score:.4f}{origin}  "
                f"{meta.ticker} {meta.form_type.value} FY{meta.fiscal_year} "
                f"Item {meta.section_label}"
            )
            print(f"    {result.chunk.text[: args.chars]}")
    return 0


def _origin(result) -> str:
    """Which retrievers found this chunk, and at what rank."""
    parts = []
    if result.dense_rank:
        parts.append(f"d{result.dense_rank}")
    if result.sparse_rank:
        parts.append(f"s{result.sparse_rank}")
    return f" ({'+'.join(parts)})" if parts else ""


if __name__ == "__main__":
    sys.exit(main())
