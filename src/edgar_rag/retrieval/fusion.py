"""Reciprocal Rank Fusion.

Dense and sparse scores are not comparable: cosine similarity lands around
0.8 while BM25 might return 14.7, and both shift with the query. Any attempt
to normalise them needs tuning that has to be redone whenever the model or
corpus changes.

RRF sidesteps that by discarding the scores entirely and using only each
result's *position* in its list. A chunk scores `1 / (k + rank)` in each
list it appears in, and the contributions add, so anything both retrievers
rank highly rises above anything only one of them liked. There is nothing
to tune but `k`, which merely damps the advantage of the very top ranks.
"""

from __future__ import annotations

from collections.abc import Sequence

from edgar_rag.models import RetrievedChunk

# Smaller than RRF's published default of 60: see config.rrf_k for the
# measured sweep behind this value.
DEFAULT_RRF_K = 10


def reciprocal_rank_fusion(
    dense: Sequence[RetrievedChunk],
    sparse: Sequence[RetrievedChunk],
    top_k: int,
    k: int = DEFAULT_RRF_K,
) -> list[RetrievedChunk]:
    """Merge two ranked lists into one, best first."""
    scores: dict[str, float] = {}
    merged: dict[str, RetrievedChunk] = {}

    for results in (dense, sparse):
        for position, result in enumerate(results):
            # Prefer the rank the retriever reported; fall back to position
            # for callers that build lists by hand.
            rank = result.dense_rank or result.sparse_rank or position + 1
            chunk_id = result.chunk.chunk_id
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)

            if chunk_id in merged:
                # Keep both ranks so it is visible which retrievers found it.
                existing = merged[chunk_id]
                merged[chunk_id] = existing.model_copy(
                    update={
                        "dense_rank": existing.dense_rank or result.dense_rank,
                        "sparse_rank": existing.sparse_rank or result.sparse_rank,
                    }
                )
            else:
                merged[chunk_id] = result

    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [
        merged[chunk_id].model_copy(update={"score": score}) for chunk_id, score in ordered[:top_k]
    ]
