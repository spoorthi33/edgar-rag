"""Persistence for answers and ingested filings."""

from __future__ import annotations

import logging

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from edgar_rag.db.models import CitationRecord, FilingRecord, QueryRecord
from edgar_rag.models import Answer, Filing, SearchFilter

logger = logging.getLogger(__name__)


def record_answer(
    session: Session,
    answer: Answer,
    mode: str = "hybrid",
    top_k: int = 5,
    filters: SearchFilter | None = None,
) -> QueryRecord:
    """Persist an answer with its citations."""
    record = QueryRecord(
        question=answer.question,
        answer=answer.answer,
        model=answer.model,
        retrieval_mode=mode,
        top_k=top_k,
        # Only the criteria actually set. The trailing `or None` matters: an
        # all-empty filter would otherwise store `{}`, which reads as
        # "filtered with no criteria" rather than "unfiltered".
        filters=({k: v for k, v in filters.model_dump().items() if v} or None if filters else None),
        is_grounded=answer.is_grounded,
        ungrounded_figures=answer.ungrounded_figures or None,
        latency_ms=answer.latency_ms,
        input_tokens=answer.input_tokens,
        output_tokens=answer.output_tokens,
    )
    record.citations = [
        CitationRecord(
            chunk_id=citation.chunk_id,
            citation=citation.citation,
            excerpt=citation.excerpt,
            rank=rank,
        )
        for rank, citation in enumerate(answer.citations, start=1)
    ]
    session.add(record)
    session.flush()  # assign the id without ending the caller's transaction
    return record


def upsert_filing(session: Session, filing: Filing, chunk_count: int = 0) -> FilingRecord:
    """Insert or update a filing by accession number.

    Accession numbers are unique per filing, so re-running ingestion
    updates the existing row rather than duplicating it.
    """
    existing = session.scalar(
        select(FilingRecord).where(FilingRecord.accession_number == filing.accession_number)
    )
    record = existing or FilingRecord(accession_number=filing.accession_number)

    record.cik = filing.cik
    record.ticker = filing.ticker
    record.company_name = filing.company_name
    record.form_type = filing.form_type.value
    record.filing_date = filing.filing_date
    record.fiscal_year = filing.fiscal_year
    record.fiscal_period = filing.fiscal_period
    record.source_url = filing.source_url
    record.storage_key = filing.storage_key
    if chunk_count:
        record.chunk_count = chunk_count

    if existing is None:
        session.add(record)
    session.flush()
    return record


def recent_queries(session: Session, limit: int = 20) -> list[QueryRecord]:
    return list(
        session.scalars(select(QueryRecord).order_by(QueryRecord.created_at.desc()).limit(limit))
    )


def grounding_stats(session: Session) -> dict[str, int | float]:
    """How often answers were fully traceable to retrieved passages."""
    total = session.scalar(select(func.count()).select_from(QueryRecord)) or 0
    grounded = (
        session.scalar(select(func.count()).select_from(QueryRecord).where(QueryRecord.is_grounded))
        or 0
    )
    return {
        "total_queries": total,
        "grounded": grounded,
        "ungrounded": total - grounded,
        "grounded_rate": grounded / total if total else 0.0,
    }
