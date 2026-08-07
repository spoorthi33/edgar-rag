"""Builds the vector index from stored filings.

The build is three stages — parse, chunk, embed — and on a large corpus
each takes hours. Both slow stages checkpoint to disk and report progress,
because a build that is silent for hours and loses everything on
interruption is one nobody can run to completion. A laptop stopped 2.5
hours into chunking should resume, not start over.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from edgar_rag.chunking.pipeline import chunk_filing
from edgar_rag.chunking.splitter import SemanticSplitter, Splitter
from edgar_rag.config import Settings, get_settings
from edgar_rag.embeddings.base import Embedder
from edgar_rag.embeddings.sentence_transformer import SentenceTransformerEmbedder
from edgar_rag.index.checkpoint import (
    ChunkCheckpoint,
    ChunkFingerprint,
    EmbeddingCheckpoint,
)
from edgar_rag.index.faiss_index import FaissIndex
from edgar_rag.ingest.manifest import Manifest
from edgar_rag.models import Chunk
from edgar_rag.parsing.pipeline import parse_all
from edgar_rag.retrieval.bm25 import BM25Retriever

logger = logging.getLogger(__name__)

# How often the slow stages report progress and save. Small enough that an
# interruption costs minutes rather than hours, large enough that saving is
# not itself the bottleneck.
CHUNK_LOG_EVERY = 50
EMBED_BATCH = 2_000


def _fingerprint(settings: Settings, splitter: Splitter, filing_count: int) -> ChunkFingerprint:
    """What a set of chunks depends on, so an inapplicable one is rejected."""
    return ChunkFingerprint(
        filing_count=filing_count,
        splitter=type(splitter).__name__,
        target_tokens=getattr(splitter, "target_tokens", 0),
        overlap_tokens=getattr(splitter, "overlap_tokens", 0),
        breakpoint_percentile=getattr(splitter, "breakpoint_percentile", 0),
        embedding_model=settings.embedding_model,
    )


def build_chunks(
    settings: Settings | None = None,
    splitter: Splitter | None = None,
    tickers: list[str] | None = None,
    use_checkpoint: bool = True,
) -> list[Chunk]:
    """Parse and chunk every stored filing, resuming from a checkpoint."""
    settings = settings or get_settings()
    splitter = splitter or SemanticSplitter(
        embedder=SentenceTransformerEmbedder(settings=settings),
        overlap_tokens=settings.chunk_overlap_tokens,
        breakpoint_percentile=settings.semantic_breakpoint_percentile,
    )

    manifest = Manifest.load(settings.local_storage_path / "manifest.json")

    # The checkpoint is consulted before parsing, not after: when it applies
    # there is nothing to parse, and re-parsing 1,800 filings to discover
    # that would cost a quarter of an hour for no reason. The filing count
    # therefore comes from the manifest rather than the parse result.
    wanted = {t.upper() for t in tickers} if tickers else None
    filing_count = sum(
        1
        for filing in manifest.filings.values()
        if not wanted or (filing.ticker or "").upper() in wanted
    )

    checkpoint = ChunkCheckpoint(settings.index_path)
    fingerprint = _fingerprint(settings, splitter, filing_count)

    if use_checkpoint:
        saved = checkpoint.load(fingerprint)
        if saved is not None:
            return saved

    parsed = parse_all(settings=settings, tickers=tickers)
    chunks: list[Chunk] = []
    started = time.perf_counter()
    total = len(parsed)

    for index, (filing_id, sections) in enumerate(parsed.items(), start=1):
        chunks.extend(chunk_filing(manifest.filings[filing_id], sections, splitter))

        if index % CHUNK_LOG_EVERY == 0 or index == total:
            elapsed = time.perf_counter() - started
            rate = index / elapsed if elapsed else 0
            logger.info(
                "chunked %d/%d filings — %d chunks, %.1f filings/s, ~%.0f min left",
                index,
                total,
                len(chunks),
                rate,
                ((total - index) / rate / 60) if rate else 0,
            )

    if use_checkpoint:
        checkpoint.save(chunks, fingerprint)
    return chunks


def _embed_with_checkpoint(
    chunks: list[Chunk],
    embedder: Embedder,
    settings: Settings,
    use_checkpoint: bool,
) -> np.ndarray:
    """Embed every chunk, saving partial progress as batches complete.

    Embedding is the one stage that can resume part-way through itself:
    the output is a plain array, so however many rows are finished can be
    written out and a restart continues from that row.
    """
    checkpoint = EmbeddingCheckpoint(settings.index_path, len(chunks), embedder.dimension)
    done = checkpoint.load() if use_checkpoint else None

    completed: list[np.ndarray] = [done] if done is not None else []
    start = len(done) if done is not None else 0
    if start >= len(chunks):
        return np.vstack(completed)[: len(chunks)]

    began = time.perf_counter()
    for offset in range(start, len(chunks), EMBED_BATCH):
        batch = chunks[offset : offset + EMBED_BATCH]
        vectors = embedder.embed_documents([chunk.text for chunk in batch])
        completed.append(vectors)

        so_far = offset + len(batch)
        elapsed = time.perf_counter() - began
        rate = (so_far - start) / elapsed if elapsed else 0
        logger.info(
            "embedded %d/%d chunks — %.0f chunks/s, ~%.0f min left",
            so_far,
            len(chunks),
            rate,
            ((len(chunks) - so_far) / rate / 60) if rate else 0,
        )
        if use_checkpoint:
            # Only this batch is written. Re-saving the whole array here
            # made checkpointing cost more than the embedding itself.
            checkpoint.append(vectors)

    return np.vstack(completed)


def build_index(
    settings: Settings | None = None,
    embedder: Embedder | None = None,
    chunks: list[Chunk] | None = None,
    tickers: list[str] | None = None,
    use_checkpoint: bool = True,
) -> FaissIndex:
    """Chunk, embed and index the corpus, then persist it."""
    settings = settings or get_settings()
    embedder = embedder or SentenceTransformerEmbedder(settings=settings)

    if chunks is None:
        splitter = SemanticSplitter(
            embedder=embedder,
            overlap_tokens=settings.chunk_overlap_tokens,
            breakpoint_percentile=settings.semantic_breakpoint_percentile,
        )
        chunks = build_chunks(
            settings=settings,
            splitter=splitter,
            tickers=tickers,
            use_checkpoint=use_checkpoint,
        )

    if not chunks:
        raise ValueError("no chunks to index; run scripts/ingest.py first")

    logger.info("embedding %d chunks with %s", len(chunks), embedder.model_name)
    vectors = _embed_with_checkpoint(chunks, embedder, settings, use_checkpoint)

    index = FaissIndex(
        dimension=embedder.dimension,
        index_type=settings.faiss_index_type,
        nlist=settings.faiss_ivf_nlist,
        model_name=embedder.model_name,
    )
    index.add(chunks, vectors)
    index.save(settings.index_path)

    # The sparse index is persisted alongside so query processes do not pay
    # to re-tokenize the corpus before their first search.
    BM25Retriever(chunks).save(settings.index_path)

    if use_checkpoint:
        # The index is written, so the checkpoints have served their purpose
        # and would otherwise keep a second copy of the corpus on disk.
        ChunkCheckpoint(settings.index_path).clear()
    return index


def load_index(
    settings: Settings | None = None,
    embedder: Embedder | None = None,
) -> FaissIndex:
    """Load the persisted index, checking it matches the configured model."""
    settings = settings or get_settings()
    embedder = embedder or SentenceTransformerEmbedder(settings=settings)

    index = FaissIndex(dimension=embedder.dimension, model_name=embedder.model_name)
    index.load(settings.index_path)
    return index


def load_retrievers(
    settings: Settings | None = None,
    embedder: Embedder | None = None,
) -> tuple[FaissIndex, BM25Retriever]:
    """Load both retrievers over the same chunks."""
    settings = settings or get_settings()
    embedder = embedder or SentenceTransformerEmbedder(settings=settings)

    index = load_index(settings=settings, embedder=embedder)
    sparse = BM25Retriever()
    sparse.load(settings.index_path, index.store)
    return index, sparse
