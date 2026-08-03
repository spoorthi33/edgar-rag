"""Request and response models for the HTTP API."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from edgar_rag.models import Answer


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    tickers: list[str] | None = None
    fiscal_years: list[int] | None = None
    items: list[str] | None = None
    top_k: int | None = Field(default=None, ge=1, le=20)
    mode: str = Field(default="hybrid", pattern="^(hybrid|dense|sparse)$")
    include_passages: bool = False


class CitationResponse(BaseModel):
    citation: str
    chunk_id: str
    excerpt: str


class PassageResponse(BaseModel):
    citation: str
    ticker: str | None
    fiscal_year: int
    item: str | None
    score: float
    dense_rank: int | None
    sparse_rank: int | None
    text: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    citations: list[CitationResponse]
    # Surfaced rather than hidden: a caller that displays a figure should
    # know when it could not be traced to a retrieved passage.
    is_grounded: bool
    ungrounded_figures: list[str]
    model: str | None
    latency_ms: float | None
    passages: list[PassageResponse] | None = None

    @classmethod
    def from_answer(cls, answer: Answer, include_passages: bool = False) -> QueryResponse:
        return cls(
            question=answer.question,
            answer=answer.answer,
            citations=[
                CitationResponse(citation=c.citation, chunk_id=c.chunk_id, excerpt=c.excerpt)
                for c in answer.citations
            ],
            is_grounded=answer.is_grounded,
            ungrounded_figures=answer.ungrounded_figures,
            model=answer.model,
            latency_ms=answer.latency_ms,
            passages=(
                [
                    PassageResponse(
                        citation=r.chunk.metadata.citation,
                        ticker=r.chunk.metadata.ticker,
                        fiscal_year=r.chunk.metadata.fiscal_year,
                        item=r.chunk.metadata.item,
                        score=r.score,
                        dense_rank=r.dense_rank,
                        sparse_rank=r.sparse_rank,
                        text=r.chunk.text,
                    )
                    for r in answer.retrieved
                ]
                if include_passages
                else None
            ),
        )


class HealthResponse(BaseModel):
    status: str
    index_size: int
    database: str
    model: str
    total_cost_usd: float = 0.0
    total_calls: int = 0


class QueryHistoryItem(BaseModel):
    id: int
    question: str
    answer: str
    is_grounded: bool
    model: str | None
    latency_ms: float | None
    created_at: datetime


class StatsResponse(BaseModel):
    total_queries: int
    grounded: int
    ungrounded: int
    grounded_rate: float
