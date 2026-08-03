"""Database schema.

Postgres holds what is queryable — which filings are ingested, what was
asked, what was answered, and whether each answer was grounded. Chunk text
and vectors stay in the object store and FAISS: rows here are for auditing
and for the evaluation harness, not for retrieval.

Column types are kept portable so the suite can run against SQLite while
the service runs against Postgres.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

from sqlalchemy import (
    JSON,
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class FilingRecord(Base):
    """A filing that has been ingested."""

    __tablename__ = "filings"
    __table_args__ = (UniqueConstraint("accession_number", name="uq_filings_accession"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    cik: Mapped[str] = mapped_column(String(10), index=True)
    ticker: Mapped[str | None] = mapped_column(String(10), index=True)
    company_name: Mapped[str] = mapped_column(String(255))
    form_type: Mapped[str] = mapped_column(String(10), index=True)
    accession_number: Mapped[str] = mapped_column(String(25))
    filing_date: Mapped[date] = mapped_column(Date)
    fiscal_year: Mapped[int] = mapped_column(Integer, index=True)
    fiscal_period: Mapped[str | None] = mapped_column(String(4))
    source_url: Mapped[str] = mapped_column(Text)
    storage_key: Mapped[str | None] = mapped_column(Text)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class QueryRecord(Base):
    """One question and the answer given, with its grounding verdict."""

    __tablename__ = "queries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    question: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text)
    model: Mapped[str | None] = mapped_column(String(64))
    retrieval_mode: Mapped[str] = mapped_column(String(16), default="hybrid")
    top_k: Mapped[int] = mapped_column(Integer, default=5)
    # Stored so a stale answer can be traced back to the filter that
    # produced it — the filter is as much a part of the result as the
    # question is.
    filters: Mapped[dict | None] = mapped_column(JSON)
    is_grounded: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    ungrounded_figures: Mapped[list | None] = mapped_column(JSON)
    latency_ms: Mapped[float | None] = mapped_column(Float)
    input_tokens: Mapped[int | None] = mapped_column(Integer)
    output_tokens: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, index=True
    )

    citations: Mapped[list[CitationRecord]] = relationship(
        back_populates="query", cascade="all, delete-orphan"
    )


class CitationRecord(Base):
    """A passage an answer cited, kept for audit."""

    __tablename__ = "citations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    query_id: Mapped[int] = mapped_column(ForeignKey("queries.id", ondelete="CASCADE"), index=True)
    chunk_id: Mapped[str] = mapped_column(String(128))
    citation: Mapped[str] = mapped_column(String(64), index=True)
    excerpt: Mapped[str] = mapped_column(Text)
    rank: Mapped[int] = mapped_column(Integer, default=0)

    query: Mapped[QueryRecord] = relationship(back_populates="citations")
