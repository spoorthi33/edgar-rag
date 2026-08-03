"""Section to chunks, carrying provenance.

Every chunk records the filing, fiscal year and item it came from. That
metadata is what lets retrieval filter to the right company and period
before ranking by similarity, which is the main defence against answering
from a semantically perfect passage in the wrong filing.
"""

from __future__ import annotations

import logging

from edgar_rag.chunking.splitter import Splitter
from edgar_rag.models import Chunk, ChunkMetadata, Filing, Section

logger = logging.getLogger(__name__)

# Sections shorter than this carry no usable content: quarterly Item 1A is
# often just "no material changes from our Annual Report", and [Reserved]
# items are a single line.
MIN_CHUNK_CHARS = 200


def chunk_section(
    filing: Filing,
    section: Section,
    splitter: Splitter,
) -> list[Chunk]:
    """Split one section into chunks tagged with the filing's provenance."""
    body = _strip_heading(section)
    if len(body) < MIN_CHUNK_CHARS:
        return []

    metadata = ChunkMetadata(
        cik=filing.cik,
        ticker=filing.ticker,
        company_name=filing.company_name,
        form_type=filing.form_type,
        fiscal_year=filing.fiscal_year,
        fiscal_period=filing.fiscal_period,
        item=section.item or None,
        part=section.part,
        filing_date=filing.filing_date,
        accession_number=filing.accession_number,
    )

    chunks: list[Chunk] = []
    for order, text in enumerate(splitter.split(body)):
        if len(text) < MIN_CHUNK_CHARS:
            continue
        chunks.append(
            Chunk(
                chunk_id=_chunk_id(filing, section, order),
                filing_id=filing.filing_id,
                text=text,
                metadata=metadata,
                token_count=splitter.count_tokens(text),
                order=order,
            )
        )
    return chunks


def chunk_filing(
    filing: Filing,
    sections: list[Section],
    splitter: Splitter,
) -> list[Chunk]:
    """Chunk every section of one filing."""
    chunks: list[Chunk] = []
    for section in sections:
        chunks.extend(chunk_section(filing, section, splitter))
    return chunks


def _strip_heading(section: Section) -> str:
    """Drop the repeated "Item 1A. Risk Factors" line from the body.

    The heading is already captured in metadata; leaving it in the first
    chunk of every section gives all of them a spurious similarity to any
    query mentioning the item.
    """
    lines = section.text.split("\n", 1)
    if len(lines) == 2 and lines[0].lower().startswith("item "):
        return lines[1].strip()
    return section.text.strip()


def _chunk_id(filing: Filing, section: Section, order: int) -> str:
    """Stable identifier: same filing and section always yield the same ids.

    Accession number, part, item and order already identify a chunk
    uniquely, so no hash is needed and the id stays readable in logs.
    """
    label = f"{section.part or '-'}-{section.item or '-'}"
    return f"{filing.accession_number}-{label}-{order}"
