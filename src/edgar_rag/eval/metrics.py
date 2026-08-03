"""Retrieval and answer metrics.

Retrieval and generation are measured separately and deliberately so. When
an answer is wrong, the pair of numbers says which half broke: poor recall
means the evidence was never retrieved and the fix is chunking or search;
good recall with poor faithfulness means the evidence was there and the
fix is the prompt. A single end-to-end score would hide that.
"""

from __future__ import annotations

from dataclasses import dataclass

from edgar_rag.models import RetrievedChunk


def recall_at_k(retrieved: list[RetrievedChunk], relevant_ids: set[str], k: int) -> float:
    """Fraction of relevant chunks that appear in the top `k`.

    The headline retrieval number: did the evidence reach the model at all?
    Nothing downstream can recover from a miss here.
    """
    if not relevant_ids:
        return 0.0
    found = {r.chunk.chunk_id for r in retrieved[:k]} & relevant_ids
    return len(found) / len(relevant_ids)


def achievable_recall_at_k(
    retrieved: list[RetrievedChunk], relevant_ids: set[str], k: int
) -> float:
    """Recall as a fraction of what `k` slots could possibly capture.

    Raw Recall@k is bounded by k/|relevant|, so when a question has 100
    relevant chunks and k is 5, perfect retrieval still scores 0.05. That
    number reads as a failure and isn't comparable across questions.
    Dividing by min(k, |relevant|) asks the answerable question instead:
    of the slots available, how many held evidence?
    """
    if not relevant_ids:
        return 0.0
    found = {r.chunk.chunk_id for r in retrieved[:k]} & relevant_ids
    return len(found) / min(k, len(relevant_ids))


def precision_at_k(retrieved: list[RetrievedChunk], relevant_ids: set[str], k: int) -> float:
    """Fraction of the top `k` that are relevant.

    Measures how much noise surrounds the evidence in the prompt — every
    irrelevant passage is context the model must read past.
    """
    top = retrieved[:k]
    if not top:
        return 0.0
    return len([r for r in top if r.chunk.chunk_id in relevant_ids]) / len(top)


def reciprocal_rank(retrieved: list[RetrievedChunk], relevant_ids: set[str]) -> float:
    """1/rank of the first relevant chunk, or 0 if none was retrieved.

    Rewards putting the evidence first rather than merely somewhere in the
    list, which matters because attention degrades down a long context.
    """
    for rank, result in enumerate(retrieved, start=1):
        if result.chunk.chunk_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def hit_at_k(retrieved: list[RetrievedChunk], relevant_ids: set[str], k: int) -> float:
    """1.0 if any relevant chunk is in the top `k`."""
    return float(bool({r.chunk.chunk_id for r in retrieved[:k]} & relevant_ids))


@dataclass
class RetrievalMetrics:
    """Per-question retrieval scores."""

    recall_at_k: float
    achievable_recall_at_k: float
    precision_at_k: float
    reciprocal_rank: float
    hit_at_k: float
    retrieved_count: int
    relevant_count: int

    @classmethod
    def compute(
        cls, retrieved: list[RetrievedChunk], relevant_ids: set[str], k: int
    ) -> RetrievalMetrics:
        return cls(
            recall_at_k=recall_at_k(retrieved, relevant_ids, k),
            achievable_recall_at_k=achievable_recall_at_k(retrieved, relevant_ids, k),
            precision_at_k=precision_at_k(retrieved, relevant_ids, k),
            reciprocal_rank=reciprocal_rank(retrieved, relevant_ids),
            hit_at_k=hit_at_k(retrieved, relevant_ids, k),
            retrieved_count=len(retrieved),
            relevant_count=len(relevant_ids),
        )


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
