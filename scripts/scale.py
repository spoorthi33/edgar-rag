"""Grows the corpus one batch at a time.

Reaching ten thousand filings is roughly a day of continuous work, which is
not something a laptop can be asked to do in one stretch. This runs it as a
sequence of independent batches of a couple of hours each: every batch
ingests, chunks, embeds and appends to the existing index, and leaves the
corpus complete and queryable when it finishes.

    python scripts/scale.py status
    python scripts/scale.py batch --filings 900
    python scripts/scale.py reindex          # once, after the last batch

Batches are resumable at two levels. Chunking checkpoints as it runs, so an
interruption inside a batch resumes within minutes of where it stopped; and
a batch that never completes leaves the index untouched, so the worst case
is repeating one batch rather than corrupting the corpus.

`reindex` is not optional. IVF centroids are fitted when the index is first
trained, and chunks appended later are filed against those original
centroids -- so a corpus grown in batches ends up partitioned for its first
batch alone. That costs recall silently. Retraining at the end fits the
centroids to the whole corpus, and needs no re-embedding.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

from edgar_rag.config import Settings, get_settings
from edgar_rag.embeddings.sentence_transformer import SentenceTransformerEmbedder
from edgar_rag.index.builder import build_index, load_index
from edgar_rag.ingest.client import EdgarClient
from edgar_rag.ingest.manifest import Manifest
from edgar_rag.ingest.pipeline import ingest
from edgar_rag.models import FormType
from edgar_rag.retrieval.bm25 import BM25Retriever

logger = logging.getLogger("scale")

# Ingesting is by ticker, but the batch is measured in filings, so tickers
# are drawn in waves until enough filings are pending.
TICKER_WAVE = 40


#: Filing ids already indexed, read once per command. The index is untouched
#: until the append at the very end, so this cannot go stale mid-run -- and
#: re-reading it would mean loading a million vectors several times over just
#: to ask the same question.
_indexed: set[str] | None = None


def _indexed_filings(settings: Settings) -> set[str]:
    """Filing ids already in the persisted index.

    Read from the index rather than tracked in a file of its own: the index
    is the thing that must not be double-counted, so it should be the thing
    that is asked. Only the ids are kept, so the index itself is released
    rather than held alongside the one the append will build.
    """
    global _indexed
    if _indexed is None:
        try:
            _indexed = set(load_index(settings=settings).store.filing_ids)
        except (FileNotFoundError, OSError):
            _indexed = set()
    return _indexed


def _pending(settings: Settings) -> tuple[Manifest, list[str]]:
    """The manifest, and the filings in it that are not yet indexed."""
    manifest = Manifest.load(settings.local_storage_path / "manifest.json")
    indexed = _indexed_filings(settings)
    return manifest, [f for f in manifest.filings if f not in indexed]


def _ingest_until(settings: Settings, target: int, forms: list[FormType], per_ticker: int) -> int:
    """Ingest whole tickers until at least `target` filings are pending."""
    manifest, pending = _pending(settings)
    if len(pending) >= target:
        return len(pending)

    seen = {(f.ticker or "").upper() for f in manifest.filings.values()}
    with EdgarClient(settings) as client:
        universe = [t.upper() for t in client.ticker_map()]

    # A stable order over a stable list, so successive batches keep walking
    # forward through the same universe instead of revisiting companies.
    candidates = [t for t in universe if t not in seen]
    if not candidates:
        logger.warning("every known ticker has been ingested; nothing left to add")
        return len(pending)

    while len(pending) < target and candidates:
        wave, candidates = candidates[:TICKER_WAVE], candidates[TICKER_WAVE:]
        logger.info("ingesting %d tickers (%d/%d filings pending)", len(wave), len(pending), target)
        ingest(wave, form_types=forms, limit_per_ticker=per_ticker, settings=settings)
        _, pending = _pending(settings)

    return len(pending)


def _run_batch(settings: Settings, args: argparse.Namespace) -> int:
    forms = [FormType(f) for f in args.forms]
    started = time.perf_counter()

    available = _ingest_until(settings, args.filings, forms, args.per_ticker)
    if not available:
        print("nothing left to index")
        return 0

    _, pending = _pending(settings)
    batch = sorted(pending)[: args.filings]
    logger.info("batch of %d filings (%d pending in total)", len(batch), len(pending))

    # With no index yet there is nothing to append to, and loading one would
    # fail; the first batch creates it and every batch after extends it.
    append = bool(_indexed_filings(settings))
    index = build_index(settings=settings, filing_ids=batch, append=append)

    minutes = (time.perf_counter() - started) / 60
    print(
        f"\nbatch complete: {len(batch)} filings in {minutes:.0f} min\n"
        f"index now holds {index.size:,} chunks from "
        f"{len(set(index.store.filing_ids)):,} filings"
    )
    if minutes > args.max_minutes:
        print(
            f"note: this batch took longer than --max-minutes ({args.max_minutes:.0f}); "
            f"use --filings {int(args.filings * args.max_minutes / minutes)} next time"
        )
    return 0


def _run_reindex(settings: Settings) -> int:
    embedder = SentenceTransformerEmbedder(settings=settings)
    index = load_index(settings=settings, embedder=embedder)

    started = time.perf_counter()
    if not index.retrain():
        print(f"index is {index.index_type}, which has no centroids to retrain; nothing to do")
        return 0
    index.save(settings.index_path)

    # Postings are rebuilt too, so a corpus that was appended to without a
    # BM25 rebuild cannot be left behind by this command.
    sparse = BM25Retriever()
    sparse.rebuild_from(index.store)
    sparse.save(settings.index_path)

    minutes = (time.perf_counter() - started) / 60
    print(f"retrained over {index.size:,} vectors in {minutes:.1f} min")
    return 0


def _run_status(settings: Settings) -> int:
    manifest, pending = _pending(settings)
    indexed = len(manifest.filings) - len(pending)
    tickers = {(f.ticker or "").upper() for f in manifest.filings.values()}

    print(f"ingested filings : {len(manifest.filings):,} across {len(tickers):,} tickers")
    print(f"indexed filings  : {indexed:,}")
    print(f"pending          : {len(pending):,}")
    print(f"remaining to 10k : {max(0, 10_000 - len(manifest.filings)):,}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Grow the corpus in resumable batches")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="what is ingested, indexed and pending")

    batch = sub.add_parser("batch", help="ingest, chunk, embed and append one batch")
    batch.add_argument(
        "--filings",
        type=int,
        default=900,
        help="filings per batch; sized so a batch runs well under two hours (default: 900)",
    )
    batch.add_argument("--per-ticker", type=int, default=8, help="filings per company")
    batch.add_argument(
        "--forms", nargs="+", default=["10-K", "10-Q"], choices=[f.value for f in FormType]
    )
    batch.add_argument(
        "--max-minutes",
        type=float,
        default=100.0,
        help="advisory: suggests a smaller batch if this is exceeded",
    )

    sub.add_parser("reindex", help="retrain IVF centroids over the whole corpus")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    settings = get_settings()

    if args.command == "status":
        return _run_status(settings)
    if args.command == "reindex":
        return _run_reindex(settings)
    return _run_batch(settings, args)


if __name__ == "__main__":
    sys.exit(main())
