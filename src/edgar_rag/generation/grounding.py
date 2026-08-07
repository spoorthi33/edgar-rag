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
# A comma only counts as a thousands separator when three digits follow, so
# sentence punctuation is left out of the literal — "FY2026, revenue" once
# yielded the literal "2026," which then failed every digit check applied
# to it.
# Two patterns, deliberately asymmetric: strict about what counts as a
# *claim*, permissive about what counts as *evidence*.
#
# In an answer, digits attached to letters belong to a name — a corpus of
# 1,800 filers contains "Data443 Risk Mitigation", whose 443 was reported
# as an invented figure. In a filing, the same shape is ordinary: table
# extraction runs cells together, so "June 28, 2025AmericasEurope" is how a
# real figure appears. Applying the strict pattern to the context stopped
# it finding numbers that were genuinely there, and four correctly grounded
# answers were then reported as unfaithful.
CLAIM_PATTERN = re.compile(r"(?<![A-Za-z\d])\d+(?:,\d{3})*(?:\.\d+)?(?![A-Za-z])")
NUMBER_PATTERN = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?")

# Citation tags carry digits (CIK, year) that are not claims about figures.
CITATION_PATTERN = re.compile(r"\[([^\]]+)\]")

# A citation the model never finished, because the answer hit the output
# cap mid-tag. The pattern above needs the closing bracket, so without this
# a truncated "[0001500412:2025:I" spills its CIK into the figure check and
# a correctly grounded answer is reported as having invented a number.
UNCLOSED_CITATION_PATTERN = re.compile(r"\[[^\]]*$")

# Dates are periods, not reported figures: "adopted 12/18/2025" would
# otherwise contribute 12 and 18 as claims to verify.
DATE_PATTERN = re.compile(
    r"\d{1,2}/\d{1,2}/\d{2,4}"
    r"|\d{4}-\d{2}-\d{2}"
    r"|(?:January|February|March|April|May|June|July|August|September"
    r"|October|November|December)\s+\d{1,2},?\s+\d{4}",
    re.IGNORECASE,
)

# Small integers are usually prose ("the top 3 risks", "2 segments") rather
# than reported figures, and flagging them buries the real findings. The
# threshold sits below any dollar or percentage figure worth checking.
MIN_CHECKED_VALUE = Decimal("10")

# Bare four-digit years are periods the model states ("Q1 FY2026"), not
# figures quoted from a passage, and the fiscal year often differs from any
# date printed in the text. Checking them produced false positives on
# answers that were otherwise entirely grounded. The cost is that a genuine
# figure that happens to equal a year is not checked.
YEAR_RANGE = (Decimal("1900"), Decimal("2100"))

# Filings tabulate in millions while answers often restate in billions.
# "$62,184 million ($62.184 billion)" is one grounded figure written twice,
# not an invented second one.
SCALE_FACTORS = (Decimal("1000"), Decimal("0.001"))


def extract_numbers(text: str) -> list[str]:
    """Numeric literals in `text` that are claims about figures.

    Citation tags, unfinished citation tags, and dates are removed first —
    each carries digits the model is not asserting as a quantity, and each
    produced a false "invented figure" on the 1,823-filing corpus.
    """
    stripped = CITATION_PATTERN.sub(" ", text)
    stripped = UNCLOSED_CITATION_PATTERN.sub(" ", stripped)
    stripped = DATE_PATTERN.sub(" ", stripped)
    return CLAIM_PATTERN.findall(stripped)


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
        if value is None or value < MIN_CHECKED_VALUE or _is_year(literal, value):
            continue
        if _is_grounded(value, grounded) or literal in ungrounded:
            continue
        ungrounded.append(literal)
    return ungrounded


def _is_year(literal: str, value: Decimal) -> bool:
    """A bare four-digit integer in the plausible year range."""
    return literal.isdigit() and len(literal) == 4 and YEAR_RANGE[0] <= value <= YEAR_RANGE[1]


def _is_grounded(value: Decimal, grounded: set[Decimal]) -> bool:
    """Whether `value` appears in the context, allowing a scale restatement."""
    if value in grounded:
        return True
    return any(value * factor in grounded for factor in SCALE_FACTORS)


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
