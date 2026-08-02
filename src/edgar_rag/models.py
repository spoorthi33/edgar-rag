"""Core data types shared across the pipeline.

These are the contracts between stages: ingest produces `Filing`, parsing
produces `Section`, chunking produces `Chunk`, retrieval produces
`RetrievedChunk`, generation produces `Answer`.
"""

from __future__ import annotations

from datetime import date
from enum import StrEnum

from pydantic import BaseModel, Field


class FormType(StrEnum):
    TEN_K = "10-K"
    TEN_Q = "10-Q"


class Filing(BaseModel):
    """One filing as retrieved from EDGAR."""

    cik: str
    ticker: str | None = None
    company_name: str
    form_type: FormType
    accession_number: str
    filing_date: date
    fiscal_year: int
    fiscal_period: str | None = None  # e.g. "Q3"; None for 10-K
    source_url: str
    storage_key: str | None = None  # where the raw HTML landed in the object store

    @property
    def filing_id(self) -> str:
        """Stable identifier used as the object-store key prefix."""
        return f"{self.cik}/{self.accession_number}"


class Section(BaseModel):
    """A structural section of a filing (Item 1A, Item 7, ...)."""

    filing_id: str
    item: str  # "1A", "7", "7A", "8", ...
    title: str
    text: str
    order: int
    # 10-Qs reuse item numbers across parts: Part I Item 1 is the financial
    # statements, Part II Item 1 is legal proceedings.
    part: str | None = None


class ChunkMetadata(BaseModel):
    """Provenance carried by every chunk.

    This is what makes metadata pre-filtering possible, which is the main
    defense against answering with the right topic from the wrong company
    or the wrong year.
    """

    cik: str
    ticker: str | None = None
    company_name: str
    form_type: FormType
    fiscal_year: int
    fiscal_period: str | None = None
    item: str | None = None
    # 10-Qs reuse item numbers across parts, so the part is required to
    # identify a section: Part I Item 1 is the financial statements, Part II
    # Item 1 is legal proceedings.
    part: str | None = None
    filing_date: date
    accession_number: str

    @property
    def section_label(self) -> str:
        """Item qualified by part where one applies, e.g. "II-1" or "7A"."""
        if not self.item:
            return "-"
        return f"{self.part}-{self.item}" if self.part else self.item

    @property
    def citation(self) -> str:
        """Citation token the LLM is instructed to emit, e.g. [320193:2023:7]."""
        return f"{self.cik}:{self.fiscal_year}:{self.section_label}"


class Chunk(BaseModel):
    """An embeddable unit of text plus its provenance."""

    chunk_id: str
    filing_id: str
    text: str
    metadata: ChunkMetadata
    token_count: int | None = None
    order: int = 0


class RetrievedChunk(BaseModel):
    """A chunk returned by retrieval, with the scores that got it there."""

    chunk: Chunk
    score: float
    dense_rank: int | None = None
    sparse_rank: int | None = None


class Citation(BaseModel):
    chunk_id: str
    citation: str
    excerpt: str


class Answer(BaseModel):
    """Final response, plus the evidence and grounding checks behind it."""

    question: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    retrieved: list[RetrievedChunk] = Field(default_factory=list)
    ungrounded_figures: list[str] = Field(default_factory=list)
    model: str | None = None
    latency_ms: float | None = None

    @property
    def is_grounded(self) -> bool:
        """True when every numeric figure in the answer traces to retrieved text."""
        return not self.ungrounded_figures


class SearchFilter(BaseModel):
    """Metadata pre-filter applied before similarity ranking."""

    ciks: list[str] | None = None
    tickers: list[str] | None = None
    form_types: list[FormType] | None = None
    fiscal_years: list[int] | None = None
    items: list[str] | None = None
    parts: list[str] | None = None
