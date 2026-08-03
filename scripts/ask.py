"""CLI for Phase 6 question answering.

python scripts/ask.py "what are Apple's supply chain risks?"
python scripts/ask.py "how much did Apple spend on R&D?" --ticker AAPL
python scripts/ask.py "..." --show-context --mode dense
"""

from __future__ import annotations

import argparse
import logging
import sys

from edgar_rag.config import get_settings
from edgar_rag.embeddings.sentence_transformer import SentenceTransformerEmbedder
from edgar_rag.generation.budget import BudgetExceeded
from edgar_rag.generation.pipeline import AnswerPipeline, get_llm_client
from edgar_rag.index.builder import load_retrievers
from edgar_rag.models import SearchFilter
from edgar_rag.retrieval.hybrid import HybridRetriever


def main() -> int:
    parser = argparse.ArgumentParser(description="Ask a question about the filings")
    parser.add_argument("question")
    parser.add_argument("--ticker", action="append", help="repeatable")
    parser.add_argument("--year", type=int, action="append", help="repeatable")
    parser.add_argument("--item", action="append", help="repeatable, e.g. 1A")
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--mode", choices=["hybrid", "dense", "sparse"], default="hybrid")
    parser.add_argument("--show-context", action="store_true", help="print retrieved passages")
    parser.add_argument("--no-cache", action="store_true", help="bypass the response cache")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    settings = get_settings()
    if args.no_cache:
        settings = settings.model_copy(update={"llm_cache_enabled": False})

    embedder = SentenceTransformerEmbedder(settings=settings)
    index, sparse = load_retrievers(settings=settings, embedder=embedder)
    pipeline = AnswerPipeline(
        retriever=HybridRetriever(index=index, embedder=embedder, sparse=sparse, settings=settings),
        llm=get_llm_client(settings),
        settings=settings,
    )

    try:
        answer = pipeline.answer(
            args.question,
            top_k=args.top_k,
            filters=SearchFilter(tickers=args.ticker, fiscal_years=args.year, items=args.item),
            mode=args.mode,
        )
    except BudgetExceeded as exc:
        print(f"\nstopped: {exc}")
        return 1

    print(f"\nQ: {answer.question}\n")
    print(answer.answer)

    if answer.citations:
        print("\nSources:")
        for citation in answer.citations:
            print(f"  [{citation.citation}] {citation.excerpt[:110].strip()}...")

    if answer.ungrounded_figures:
        print(f"\nUNGROUNDED FIGURES: {', '.join(answer.ungrounded_figures)}")
        print("  (these do not appear in the retrieved passages)")
    else:
        print("\nAll figures traced to retrieved passages.")

    if args.show_context:
        print("\n--- retrieved passages ---")
        for rank, result in enumerate(answer.retrieved, start=1):
            meta = result.chunk.metadata
            print(f"\n[{rank}] {meta.ticker} FY{meta.fiscal_year} Item {meta.section_label}")
            print(f"    {result.chunk.text[:300]}")

    budget = getattr(pipeline.llm, "budget", None)
    if budget is not None:
        print(f"\n{answer.latency_ms:.0f}ms | {budget.summary()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
