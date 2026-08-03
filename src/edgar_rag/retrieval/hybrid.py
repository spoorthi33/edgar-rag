"""Hybrid retrieval: dense meaning plus sparse exact matching."""

from __future__ import annotations

import logging

from edgar_rag.config import Settings, get_settings
from edgar_rag.embeddings.base import Embedder
from edgar_rag.index.base import VectorIndex
from edgar_rag.models import RetrievedChunk, SearchFilter
from edgar_rag.retrieval.bm25 import BM25Retriever
from edgar_rag.retrieval.fusion import reciprocal_rank_fusion

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Runs both retrievers and fuses their rankings.

    Each retriever is asked for `candidate_k` results rather than `top_k`:
    fusion can only promote what it is given, so a chunk ranked eighth by
    one retriever and second by the other never surfaces if both lists stop
    at five.
    """

    def __init__(
        self,
        index: VectorIndex,
        embedder: Embedder,
        sparse: BM25Retriever,
        settings: Settings | None = None,
    ) -> None:
        settings = settings or get_settings()
        self.index = index
        self.embedder = embedder
        self.sparse = sparse
        self.candidate_k = settings.retrieval_candidate_k
        self.rrf_k = settings.rrf_k

    def retrieve(
        self,
        query: str,
        top_k: int,
        filters: SearchFilter | None = None,
        mode: str = "hybrid",
    ) -> list[RetrievedChunk]:
        """Retrieve for `query`. `mode` is hybrid, dense or sparse."""
        if mode == "dense":
            return self._dense(query, top_k, filters)
        if mode == "sparse":
            return self.sparse.search(query, top_k, filters)

        dense = self._dense(query, self.candidate_k, filters)
        sparse = self.sparse.search(query, self.candidate_k, filters)
        return reciprocal_rank_fusion(dense, sparse, top_k=top_k, k=self.rrf_k)

    def _dense(self, query: str, top_k: int, filters: SearchFilter | None) -> list[RetrievedChunk]:
        return self.index.search(self.embedder.embed_query(query), top_k, filters)
