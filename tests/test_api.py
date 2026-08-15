"""Phase 7 tests.

The database is SQLite in a temp file so the suite needs no Docker; the
schema uses portable column types so the same models run on Postgres.
Retrieval and generation are stubbed — this covers the service and
persistence layers, not the pipeline beneath them.
"""

from __future__ import annotations

from datetime import date

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from edgar_rag.api import main as api_main
from edgar_rag.api.main import ServiceState, app, get_db
from edgar_rag.config import Settings
from edgar_rag.db.models import QueryRecord
from edgar_rag.db.repository import grounding_stats, record_answer, upsert_filing
from edgar_rag.db.session import create_tables, get_engine, get_session_factory
from edgar_rag.generation.budget import BudgetExceeded
from edgar_rag.models import (
    Answer,
    Chunk,
    ChunkMetadata,
    Citation,
    Filing,
    FormType,
    RetrievedChunk,
    SearchFilter,
)


def _filing(accession: str = "0000320193-25-000079") -> Filing:
    return Filing(
        cik="0000320193",
        ticker="AAPL",
        company_name="Apple Inc.",
        form_type=FormType.TEN_K,
        accession_number=accession,
        filing_date=date(2025, 10, 31),
        fiscal_year=2025,
        source_url="https://www.sec.gov/Archives/example.htm",
        storage_key="key.html",
    )


def _retrieved() -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(
            chunk_id="c1",
            filing_id="0000320193/acc",
            text="Research and development expense was $10,887 million.",
            order=0,
            metadata=ChunkMetadata(
                cik="0000320193",
                ticker="AAPL",
                company_name="Apple Inc.",
                form_type=FormType.TEN_K,
                fiscal_year=2026,
                item="1",
                part="I",
                filing_date=date(2025, 10, 31),
                accession_number="acc",
            ),
        ),
        score=0.87,
        dense_rank=1,
        sparse_rank=2,
    )


def _answer(grounded: bool = True) -> Answer:
    return Answer(
        question="How much did Apple spend on R&D?",
        answer="R&D was $10,887 million [0000320193:2026:I-1].",
        citations=[
            Citation(chunk_id="c1", citation="0000320193:2026:I-1", excerpt="R&D expense...")
        ],
        retrieved=[_retrieved()],
        ungrounded_figures=[] if grounded else ["99.9"],
        model="claude-sonnet-5",
        latency_ms=1234.5,
    )


@pytest.fixture
def db(tmp_path) -> Session:
    engine = get_engine(f"sqlite:///{tmp_path}/test.db")
    create_tables(engine)
    session = get_session_factory(engine)()
    yield session
    session.close()


class StubPipeline:
    """Stands in for retrieval + generation."""

    def __init__(self, answer: Answer | None = None, raises: Exception | None = None) -> None:
        self._answer = answer or _answer()
        self._raises = raises
        self.calls: list[tuple] = []
        self.llm = type("LLM", (), {"model": "stub-model"})()

    def answer(self, question, top_k=None, filters=None, mode="hybrid"):
        self.calls.append((question, top_k, filters, mode))
        if self._raises:
            raise self._raises
        return self._answer


@pytest.fixture
def client(db: Session, monkeypatch):
    """App wired to the SQLite session, with the heavy startup path stubbed.

    Startup normally loads the embedding model and index and checks the
    schema against Postgres; the `db` fixture has already built it here,
    and the real startup path is verified separately against a live
    database and a real migration.
    """
    pipeline = StubPipeline()
    monkeypatch.setattr(api_main, "_warn_if_unmigrated", lambda: None)
    monkeypatch.setattr(
        api_main,
        "build_state",
        lambda settings=None: ServiceState(
            pipeline=pipeline, index_size=2105, settings=Settings(_env_file=None)
        ),
    )
    app.dependency_overrides[get_db] = lambda: db

    with TestClient(app) as test_client:
        test_client.pipeline = pipeline  # type: ignore[attr-defined]
        yield test_client

    app.dependency_overrides.clear()
    api_main.state = None


# --- Persistence ---------------------------------------------------------


def test_answer_is_recorded_with_citations(db: Session) -> None:
    record = record_answer(db, _answer())
    db.commit()

    stored = db.get(QueryRecord, record.id)
    assert stored is not None
    assert stored.model == "claude-sonnet-5"
    assert stored.is_grounded is True
    assert len(stored.citations) == 1
    assert stored.citations[0].citation == "0000320193:2026:I-1"


def test_ungrounded_answer_is_flagged_in_the_row(db: Session) -> None:
    record = record_answer(db, _answer(grounded=False))
    db.commit()

    stored = db.get(QueryRecord, record.id)
    assert stored.is_grounded is False
    assert stored.ungrounded_figures == ["99.9"]


def test_only_active_filters_are_stored(db: Session) -> None:
    """Storing the empty criteria would make every row look filtered."""
    record = record_answer(db, _answer(), filters=SearchFilter(tickers=["AAPL"]))
    db.commit()

    assert db.get(QueryRecord, record.id).filters == {"tickers": ["AAPL"]}


def test_no_filters_stores_null(db: Session) -> None:
    record = record_answer(db, _answer(), filters=SearchFilter())
    db.commit()
    assert db.get(QueryRecord, record.id).filters is None


def test_deleting_a_query_removes_its_citations(db: Session) -> None:
    record = record_answer(db, _answer())
    db.commit()
    db.delete(record)
    db.commit()

    from edgar_rag.db.models import CitationRecord

    assert db.query(CitationRecord).count() == 0


def test_reingesting_a_filing_updates_rather_than_duplicates(db: Session) -> None:
    """Accession numbers are unique, so a re-run must not add a second row."""
    upsert_filing(db, _filing(), chunk_count=10)
    db.commit()

    updated = _filing()
    updated.company_name = "Apple Inc. (renamed)"
    upsert_filing(db, updated, chunk_count=25)
    db.commit()

    from edgar_rag.db.models import FilingRecord

    rows = db.query(FilingRecord).all()
    assert len(rows) == 1
    assert rows[0].company_name == "Apple Inc. (renamed)"
    assert rows[0].chunk_count == 25


def test_grounding_stats_are_computed(db: Session) -> None:
    record_answer(db, _answer(grounded=True))
    record_answer(db, _answer(grounded=True))
    record_answer(db, _answer(grounded=False))
    db.commit()

    stats = grounding_stats(db)
    assert stats["total_queries"] == 3
    assert stats["grounded"] == 2
    assert stats["ungrounded"] == 1
    assert stats["grounded_rate"] == pytest.approx(2 / 3)


def test_grounding_stats_on_an_empty_database(db: Session) -> None:
    assert grounding_stats(db)["grounded_rate"] == 0.0


# --- API -----------------------------------------------------------------


def test_health_reports_index_and_database(client) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["index_size"] == 2105
    assert body["database"] == "ok"


def test_query_returns_answer_and_citations(client) -> None:
    response = client.post("/query", json={"question": "How much on R&D?"})

    assert response.status_code == 200
    body = response.json()
    assert "10,887" in body["answer"]
    assert body["citations"][0]["citation"] == "0000320193:2026:I-1"
    assert body["is_grounded"] is True


def test_query_persists_the_answer(client, db: Session) -> None:
    client.post("/query", json={"question": "How much on R&D?"})
    assert db.query(QueryRecord).count() == 1


def test_ungrounded_figures_are_surfaced_to_the_caller(client) -> None:
    """A caller displaying a figure must know it could not be traced."""
    client.pipeline._answer = _answer(grounded=False)

    body = client.post("/query", json={"question": "q"}).json()
    assert body["is_grounded"] is False
    assert body["ungrounded_figures"] == ["99.9"]


def test_passages_are_omitted_by_default(client) -> None:
    assert client.post("/query", json={"question": "q"}).json()["passages"] is None


def test_passages_are_returned_on_request(client) -> None:
    body = client.post("/query", json={"question": "q", "include_passages": True}).json()

    assert len(body["passages"]) == 1
    assert body["passages"][0]["dense_rank"] == 1
    assert body["passages"][0]["sparse_rank"] == 2


def test_filters_reach_the_pipeline(client) -> None:
    client.post(
        "/query",
        json={"question": "q", "tickers": ["AAPL"], "fiscal_years": [2025], "mode": "dense"},
    )

    _, _, filters, mode = client.pipeline.calls[0]
    assert filters.tickers == ["AAPL"]
    assert filters.fiscal_years == [2025]
    assert mode == "dense"


def test_budget_exceeded_returns_429(client) -> None:
    """Well-formed request, retryable condition — not a server error."""
    client.pipeline._raises = BudgetExceeded("call ceiling reached")

    response = client.post("/query", json={"question": "q"})
    assert response.status_code == 429
    assert "ceiling" in response.json()["detail"]


@pytest.mark.parametrize(
    "payload",
    [
        {"question": ""},
        {"question": "q", "mode": "magic"},
        {"question": "q", "top_k": 0},
        {"question": "q", "top_k": 100},
        {},
    ],
)
def test_invalid_requests_are_rejected(client, payload: dict) -> None:
    assert client.post("/query", json=payload).status_code == 422


def test_the_ceiling_is_per_request_not_per_process(client) -> None:
    """Shared across requests, the loop guard becomes a permanent outage:
    the service would 429 forever once enough requests had been served."""
    from edgar_rag.generation.budget import Budget

    budget = Budget(max_calls=2)
    client.pipeline.llm.budget = budget  # type: ignore[attr-defined]

    def spend_then_answer(question, top_k=None, filters=None, mode="hybrid"):
        budget.record("claude-sonnet-5", 1000, 100)  # this request's usage
        return _answer()

    client.pipeline.answer = spend_then_answer  # type: ignore[method-assign]
    for _ in range(5):
        assert client.post("/query", json={"question": "q"}).status_code == 200


def test_spend_accumulates_across_requests_for_reporting(client) -> None:
    from edgar_rag.generation.budget import Budget

    budget = Budget(max_calls=100)
    client.pipeline.llm.budget = budget  # type: ignore[attr-defined]

    def spend_then_answer(question, top_k=None, filters=None, mode="hybrid"):
        budget.record("claude-sonnet-5", 1_000_000, 0)  # $2 of usage
        return _answer()

    client.pipeline.answer = spend_then_answer  # type: ignore[method-assign]
    for _ in range(3):
        client.post("/query", json={"question": "q"})

    body = client.get("/health").json()
    assert body["total_calls"] == 3
    assert body["total_cost_usd"] == pytest.approx(6.00)


def test_token_counts_are_persisted(client, db: Session) -> None:
    """Otherwise the cost of a query is recoverable only from logs."""
    answer = _answer()
    answer.input_tokens = 2663
    answer.output_tokens = 57
    client.pipeline._answer = answer

    client.post("/query", json={"question": "q"})

    stored = db.query(QueryRecord).one()
    assert stored.input_tokens == 2663
    assert stored.output_tokens == 57


def test_history_returns_recent_queries(client) -> None:
    client.post("/query", json={"question": "first"})
    client.post("/query", json={"question": "second"})

    history = client.get("/queries").json()
    assert len(history) == 2


def test_stats_endpoint_reports_grounding_rate(client) -> None:
    client.post("/query", json={"question": "q"})

    body = client.get("/stats").json()
    assert body["total_queries"] == 1
    assert body["grounded_rate"] == 1.0
