"""Verification that an answer's figures come from the retrieved context.

The prompt asks the model to quote figures exactly; this checks that it
did. Every number in the answer must appear in the passages the model was
given — otherwise it was computed, recalled, or invented, and on a
financial question those are all failures.

This is the mechanism behind the claim that the system reduces
hallucination on financial figures: not a hope that the prompt worked, but
a test that runs on every answer.
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

from edgar_rag.models import Citation, RetrievedChunk

# A number, with the punctuation filings use: "$383.3", "8.1%", "1,234".
NUMBER_PATTERN = re.compile(r"\d[\d,]*(?:\.\d+)?")

# Citation tags carry digits (CIK, year) that are not claims about figures.
CITATION_PATTERN = re.compile(r"\[([^\]]+)\]")

# Small integers are usually prose ("the top 3 risks", "2 segments") rather
# than reported figures, and flagging them buries the real findings. The
# threshold sits below any dollar or percentage figure worth checking.
MIN_CHECKED_VALUE = Decimal("10")


def extract_numbers(text: str) -> list[str]:
    """Numeric literals in `text`, excluding those inside citation tags."""
    without_citations = CITATION_PATTERN.sub(" ", text)
    return NUMBER_PATTERN.findall(without_citations)


def _to_decimal(literal: str) -> Decimal | None:
    try:
        return Decimal(literal.replace(",", ""))
    except InvalidOperation:
        return None


def find_ungrounded_figures(answer: str, chunks: list[RetrievedChunk]) -> list[str]:
    """Figures in `answer` that do not appear in the retrieved passages.

    Comparison is numeric rather than textual, so "8,866" in a filing table
    matches "$8,866 million" in the answer, and 29.9 matches 29.90.
    """
    context = " ".join(result.chunk.text for result in chunks)
    grounded = {
        value
        for literal in NUMBER_PATTERN.findall(context)
        if (value := _to_decimal(literal)) is not None
    }

    ungrounded: list[str] = []
    for literal in extract_numbers(answer):
        value = _to_decimal(literal)
        if value is None or value < MIN_CHECKED_VALUE:
            continue
        if value not in grounded and literal not in ungrounded:
            ungrounded.append(literal)
    return ungrounded


def extract_citations(answer: str, chunks: list[RetrievedChunk]) -> list[Citation]:
    """Citations in `answer` that resolve to a retrieved passage.

    A tag naming a passage that was not retrieved is dropped rather than
    returned: it cannot be verified, so presenting it as a source would be
    misleading.
    """
    by_tag = {result.chunk.metadata.citation: result.chunk for result in chunks}

    citations: list[Citation] = []
    seen: set[str] = set()
    for tag in CITATION_PATTERN.findall(answer):
        tag = tag.strip()
        chunk = by_tag.get(tag)
        if chunk is None or tag in seen:
            continue
        seen.add(tag)
        citations.append(
            Citation(
                chunk_id=chunk.chunk_id,
                citation=tag,
                excerpt=chunk.text[:300],
            )
        )
    return citations


def has_declined(answer: str) -> bool:
    """True when the model reported that the context lacks the answer."""
    lowered = answer.lower()
    return "i don't know" in lowered or "i do not know" in lowered
