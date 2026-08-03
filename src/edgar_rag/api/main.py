"""FastAPI service.

The index and embedding model are loaded once at startup rather than per
request: loading the model takes seconds and the index holds every chunk in
memory, so doing it per request would dominate latency.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy import text
from sqlalchemy.orm import Session

from edgar_rag.api.schemas import (
    HealthResponse,
    QueryHistoryItem,
    QueryRequest,
    QueryResponse,
    StatsResponse,
)
from edgar_rag.config import Settings, get_settings
from edgar_rag.db.repository import grounding_stats, recent_queries, record_answer
from edgar_rag.db.session import create_tables, get_session_factory
from edgar_rag.embeddings.sentence_transformer import SentenceTransformerEmbedder
from edgar_rag.generation.budget import BudgetExceeded, CumulativeSpend
from edgar_rag.generation.pipeline import AnswerPipeline, get_llm_client
from edgar_rag.index.builder import load_retrievers
from edgar_rag.models import SearchFilter
from edgar_rag.retrieval.hybrid import HybridRetriever

logger = logging.getLogger(__name__)


@dataclass
class ServiceState:
    """Everything loaded once and shared across requests."""

    pipeline: AnswerPipeline
    index_size: int
    settings: Settings
    spend: CumulativeSpend = field(default_factory=CumulativeSpend)

    def start_request(self) -> None:
        """Give the request a fresh budget.

        The ceilings exist to stop a runaway loop inside one request. Left
        cumulative in a long-lived service they would instead reject every
        request once enough legitimate ones had been served.
        """
        budget = getattr(self.pipeline.llm, "budget", None)
        if budget is not None:
            budget.reset()

    def finish_request(self) -> None:
        """Bank the request's spend into the running total.

        Done when the request ends rather than when the next one starts,
        so the reported total includes the most recent request instead of
        always lagging one behind.
        """
        budget = getattr(self.pipeline.llm, "budget", None)
        if budget is not None:
            self.spend.absorb(budget)


state: ServiceState | None = None


def build_state(settings: Settings | None = None) -> ServiceState:
    settings = settings or get_settings()

    embedder = SentenceTransformerEmbedder(settings=settings)
    index, sparse = load_retrievers(settings=settings, embedder=embedder)
    pipeline = AnswerPipeline(
        retriever=HybridRetriever(index=index, embedder=embedder, sparse=sparse, settings=settings),
        llm=get_llm_client(settings),
        settings=settings,
    )
    logger.info("loaded index with %d chunks", index.size)
    return ServiceState(pipeline=pipeline, index_size=index.size, settings=settings)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global state
    settings = get_settings()
    create_tables()
    state = build_state(settings)
    yield
    state = None


app = FastAPI(
    title="edgar-rag",
    description="Retrieval-augmented question answering over SEC EDGAR filings",
    version="0.1.0",
    lifespan=lifespan,
)


def get_state() -> ServiceState:
    if state is None:
        raise HTTPException(status_code=503, detail="service is still starting")
    return state


def get_db() -> Session:
    session = get_session_factory()()
    try:
        yield session
    finally:
        session.close()


@app.get("/health", response_model=HealthResponse)
def health(service: ServiceState = Depends(get_state), db: Session = Depends(get_db)):
    try:
        db.execute(text("SELECT 1"))
        database = "ok"
    except Exception as exc:  # noqa: BLE001 - report rather than fail the check
        logger.error("database health check failed: %s", exc)
        database = "unavailable"

    return HealthResponse(
        status="ok",
        index_size=service.index_size,
        database=database,
        model=service.pipeline.llm.model,
        total_cost_usd=round(service.spend.cost_usd, 4),
        total_calls=service.spend.calls,
    )


@app.post("/query", response_model=QueryResponse)
def query(
    request: QueryRequest,
    service: ServiceState = Depends(get_state),
    db: Session = Depends(get_db),
):
    service.start_request()

    try:
        answer = service.pipeline.answer(
            request.question,
            top_k=request.top_k,
            filters=SearchFilter(
                tickers=request.tickers,
                fiscal_years=request.fiscal_years,
                items=request.items,
            ),
            mode=request.mode,
        )
    except BudgetExceeded as exc:
        # 429 rather than 500: the request was well-formed and the caller
        # can retry once the ceiling is raised.
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    finally:
        # Banked even on failure: a request that hit the ceiling still spent
        # money getting there.
        service.finish_request()

    try:
        record_answer(
            db,
            answer,
            mode=request.mode,
            top_k=request.top_k or service.settings.retrieval_top_k,
            filters=SearchFilter(
                tickers=request.tickers,
                fiscal_years=request.fiscal_years,
                items=request.items,
            ),
        )
        db.commit()
    except Exception as exc:  # noqa: BLE001 - the answer is still valid
        db.rollback()
        logger.error("could not record query: %s", exc)

    return QueryResponse.from_answer(answer, include_passages=request.include_passages)


@app.get("/queries", response_model=list[QueryHistoryItem])
def query_history(limit: int = 20, db: Session = Depends(get_db)):
    return [
        QueryHistoryItem(
            id=record.id,
            question=record.question,
            answer=record.answer,
            is_grounded=record.is_grounded,
            model=record.model,
            latency_ms=record.latency_ms,
            created_at=record.created_at,
        )
        for record in recent_queries(db, limit=limit)
    ]


@app.get("/stats", response_model=StatsResponse)
def stats(db: Session = Depends(get_db)):
    return StatsResponse(**grounding_stats(db))
