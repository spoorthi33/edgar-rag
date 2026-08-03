"""Builds the vector index from stored filings."""

from __future__ import annotations

import logging

from edgar_rag.chunking.pipeline import chunk_filing
from edgar_rag.chunking.splitter import SemanticSplitter, Splitter
from edgar_rag.config import Settings, get_settings
from edgar_rag.embeddings.base import Embedder
from edgar_rag.embeddings.sentence_transformer import SentenceTransformerEmbedder
from edgar_rag.index.faiss_index import FaissIndex
from edgar_rag.ingest.manifest import Manifest
from edgar_rag.models import Chunk
from edgar_rag.parsing.pipeline import parse_all
from edgar_rag.retrieval.bm25 import BM25Retriever

logger = logging.getLogger(__name__)


def build_chunks(
    settings: Settings | None = None,
    splitter: Splitter | None = None,
    tickers: list[str] | None = None,
) -> list[Chunk]:
    """Parse and chunk every stored filing."""
    settings = settings or get_settings()
    splitter = splitter or SemanticSplitter(
        embedder=SentenceTransformerEmbedder(settings=settings),
        overlap_tokens=settings.chunk_overlap_tokens,
        breakpoint_percentile=settings.semantic_breakpoint_percentile,
    )

    manifest = Manifest.load(settings.local_storage_path / "manifest.json")
    parsed = parse_all(settings=settings, tickers=tickers)

    chunks: list[Chunk] = []
    for filing_id, sections in parsed.items():
        chunks.extend(chunk_filing(manifest.filings[filing_id], sections, splitter))
    return chunks


def build_index(
    settings: Settings | None = None,
    embedder: Embedder | None = None,
    chunks: list[Chunk] | None = None,
    tickers: list[str] | None = None,
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
        chunks = build_chunks(settings=settings, splitter=splitter, tickers=tickers)

    if not chunks:
        raise ValueError("no chunks to index; run scripts/ingest.py first")

    logger.info("embedding %d chunks with %s", len(chunks), embedder.model_name)
    vectors = embedder.embed_documents([chunk.text for chunk in chunks])

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
    sparse.load(settings.index_path, index.chunks)
    return index, sparse
