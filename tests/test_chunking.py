"""Phase 3 tests.

Semantic splitting is exercised with a stub embedder so the suite stays
offline and deterministic; the real model is validated by running
scripts/chunk.py against the corpus.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from edgar_rag.chunking.pipeline import chunk_filing, chunk_section
from edgar_rag.chunking.sentences import split_sentences
from edgar_rag.chunking.splitter import (
    FixedSplitter,
    SemanticSplitter,
    estimate_tokens,
)
from edgar_rag.embeddings.base import Embedder
from edgar_rag.models import Filing, FormType, Section


class StubEmbedder(Embedder):
    """Embeds by topic marker: sentences sharing a marker are similar.

    A sentence containing "TOPICB" points one way, everything else points
    another, so the similarity drop lands exactly at the topic change.
    """

    @property
    def dimension(self) -> int:
        return 2

    @property
    def model_name(self) -> str:
        return "stub"

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.array(
            [[0.0, 1.0] if "TOPICB" in t else [1.0, 0.0] for t in texts],
            dtype=np.float32,
        )

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]

    @property
    def max_tokens(self) -> int:
        return 512

    def count_tokens(self, text: str) -> int:
        """Stands in for a real tokenizer: one token per whitespace word."""
        return len(text.split())


def _filing() -> Filing:
    return Filing(
        cik="0000320193",
        ticker="AAPL",
        company_name="Apple Inc.",
        form_type=FormType.TEN_K,
        accession_number="0000320193-25-000079",
        filing_date=date(2025, 10, 31),
        fiscal_year=2025,
        source_url="https://www.sec.gov/Archives/example.htm",
    )


def _section(text: str, item: str = "1A", part: str | None = "I") -> Section:
    return Section(
        filing_id=_filing().filing_id,
        item=item,
        title="Risk Factors",
        text=text,
        order=0,
        part=part,
    )


def _sentences(count: int, marker: str = "") -> str:
    return " ".join(
        f"Sentence number {i} carries enough words to matter {marker}." for i in range(count)
    )


# --- Sentence splitting --------------------------------------------------


def test_splits_on_sentence_end() -> None:
    assert split_sentences("Revenue rose. Costs fell.") == ["Revenue rose.", "Costs fell."]


def test_does_not_split_inside_a_decimal() -> None:
    text = "Net sales were $383.3 billion for the year."
    assert split_sentences(text) == [text]


def test_does_not_split_after_an_abbreviation() -> None:
    text = "Apple Inc. and its subsidiaries design products."
    assert split_sentences(text) == [text]


def test_does_not_split_after_vs() -> None:
    text = "Sales grew 8.1% vs. fiscal 2022."
    assert split_sentences(text) == [text]


def test_does_not_split_between_initials() -> None:
    assert split_sentences("See J. P. Morgan analysis. It was positive.") == [
        "See J. P. Morgan analysis.",
        "It was positive.",
    ]


def test_splits_after_a_year() -> None:
    """A trailing year ends a sentence; it is not a decimal."""
    assert split_sentences("Results improved in fiscal 2022. The Company grew.") == [
        "Results improved in fiscal 2022.",
        "The Company grew.",
    ]


def test_closing_quote_stays_with_its_sentence() -> None:
    """Chunk text must match the filing verbatim for later grounding checks."""
    parts = split_sentences('Under the heading "Risk Factors." Except as noted, none.')
    assert parts[0] == 'Under the heading "Risk Factors."'


def test_lines_are_split_independently() -> None:
    assert split_sentences("First line\nSecond line") == ["First line", "Second line"]


def test_empty_text_yields_nothing() -> None:
    assert split_sentences("   \n  ") == []


# --- Fixed splitting -----------------------------------------------------


def test_respects_the_token_budget() -> None:
    splitter = FixedSplitter(target_tokens=100, overlap_tokens=0)
    chunks = splitter.split(_sentences(60))
    assert all(estimate_tokens(c) <= 130 for c in chunks)  # budget plus one sentence


def test_never_cuts_mid_sentence() -> None:
    splitter = FixedSplitter(target_tokens=60, overlap_tokens=0)
    for chunk in splitter.split(_sentences(40)):
        assert chunk.endswith(".")


def test_overlap_repeats_trailing_sentences() -> None:
    """A fact near a boundary must appear whole in at least one chunk."""
    splitter = FixedSplitter(target_tokens=100, overlap_tokens=40)
    chunks = splitter.split(_sentences(60))

    assert len(chunks) > 1
    tail = split_sentences(chunks[0])[-1]
    assert tail in chunks[1]


def test_no_overlap_when_disabled() -> None:
    splitter = FixedSplitter(target_tokens=100, overlap_tokens=0)
    chunks = splitter.split(_sentences(60))
    assert split_sentences(chunks[0])[-1] not in chunks[1]


def test_short_text_becomes_one_chunk() -> None:
    assert FixedSplitter().split("A single short sentence.") == ["A single short sentence."]


def test_empty_text_yields_no_chunks() -> None:
    assert FixedSplitter().split("") == []


def test_overlap_must_be_smaller_than_target() -> None:
    with pytest.raises(ValueError, match="smaller"):
        FixedSplitter(target_tokens=50, overlap_tokens=50)


# --- Semantic splitting --------------------------------------------------


def test_breaks_where_the_topic_changes() -> None:
    text = _sentences(8) + " " + _sentences(8, marker="TOPICB")
    # The budget must leave room for the break to fire: a topic shift only
    # closes a chunk once it is at least half full.
    splitter = SemanticSplitter(
        embedder=StubEmbedder(), target_tokens=100, overlap_tokens=0, min_sentences=4
    )
    chunks = splitter.split(text)

    assert len(chunks) > 1
    # The break lands at the topic change, not mid-topic.
    assert "TOPICB" not in chunks[0]


def test_topic_break_waits_until_the_chunk_is_half_full() -> None:
    """Otherwise a run of topic shifts yields chunks too small to carry context."""
    text = _sentences(4) + " " + _sentences(4, marker="TOPICB")
    splitter = SemanticSplitter(
        embedder=StubEmbedder(), target_tokens=400, overlap_tokens=0, min_sentences=4
    )
    assert len(splitter.split(text)) == 1


def test_falls_back_to_packing_for_short_text() -> None:
    """Too few sentences to measure a similarity drop."""
    splitter = SemanticSplitter(embedder=StubEmbedder(), min_sentences=10)
    chunks = splitter.split("One sentence. Two sentences. Three sentences.")
    assert len(chunks) == 1


def test_token_budget_still_applies_within_one_topic() -> None:
    """A long single-topic run is broken by size even with no topic change."""
    splitter = SemanticSplitter(
        embedder=StubEmbedder(), target_tokens=100, overlap_tokens=0, min_sentences=4
    )
    assert len(splitter.split(_sentences(80))) > 1


def test_no_chunk_exceeds_the_model_limit() -> None:
    """Overflow is discarded silently at embedding time, so it must not happen."""
    embedder = StubEmbedder()
    splitter = SemanticSplitter(embedder=embedder, overlap_tokens=32, min_sentences=4)

    chunks = splitter.split(_sentences(400) + " " + _sentences(400, marker="TOPICB"))

    assert chunks
    assert all(embedder.count_tokens(c) <= embedder.max_tokens for c in chunks)


def test_target_defaults_to_the_model_limit() -> None:
    assert SemanticSplitter(embedder=StubEmbedder()).target_tokens == 512


def test_target_above_the_model_limit_is_rejected() -> None:
    with pytest.raises(ValueError, match="silently discarded"):
        SemanticSplitter(embedder=StubEmbedder(), target_tokens=1024)


def test_splitter_uses_the_model_tokenizer_not_the_estimate() -> None:
    """The character estimate ran up to 1.81x off on real filing text."""
    embedder = StubEmbedder()
    text = _sentences(60)
    splitter = SemanticSplitter(embedder=embedder, target_tokens=100, overlap_tokens=0)

    assert splitter.count_tokens(text) == embedder.count_tokens(text)
    assert splitter.count_tokens(text) != estimate_tokens(text)


def test_zero_vectors_do_not_produce_nan() -> None:
    class ZeroEmbedder(StubEmbedder):
        def embed_documents(self, texts: list[str]) -> np.ndarray:
            return np.zeros((len(texts), 2), dtype=np.float32)

    splitter = SemanticSplitter(embedder=ZeroEmbedder(), target_tokens=200, min_sentences=2)
    assert splitter.split(_sentences(20))


# --- Provenance ----------------------------------------------------------


def test_chunks_carry_filing_provenance() -> None:
    chunks = chunk_section(_filing(), _section(_sentences(40)), FixedSplitter())
    meta = chunks[0].metadata

    assert meta.cik == "0000320193"
    assert meta.ticker == "AAPL"
    assert meta.fiscal_year == 2025
    assert meta.item == "1A"
    assert meta.part == "I"


def test_citation_distinguishes_parts() -> None:
    """Part I Item 1 and Part II Item 1 of a 10-Q are different sections."""
    filing = _filing()
    financials = chunk_section(filing, _section(_sentences(40), "1", "I"), FixedSplitter())
    legal = chunk_section(filing, _section(_sentences(40), "1", "II"), FixedSplitter())

    assert financials[0].metadata.citation != legal[0].metadata.citation
    assert financials[0].metadata.citation.endswith("I-1")
    assert legal[0].metadata.citation.endswith("II-1")


def test_heading_line_is_stripped_from_the_body() -> None:
    """Left in, it gives every section's first chunk a spurious match."""
    section = _section("Item 1A. Risk Factors\n" + _sentences(40))
    chunks = chunk_section(_filing(), section, FixedSplitter())
    assert "Item 1A. Risk Factors" not in chunks[0].text


def test_boilerplate_section_is_dropped() -> None:
    """Quarterly Item 1A is often just "no material changes"."""
    section = _section("There have been no material changes to our risk factors.")
    assert chunk_section(_filing(), section, FixedSplitter()) == []


def test_chunk_ids_are_stable_across_runs() -> None:
    section = _section(_sentences(40))
    first = chunk_section(_filing(), section, FixedSplitter())
    second = chunk_section(_filing(), section, FixedSplitter())
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]


def test_chunk_ids_are_unique_within_a_filing() -> None:
    sections = [_section(_sentences(40), item) for item in ["1A", "7", "7A"]]
    chunks = chunk_filing(_filing(), sections, FixedSplitter())
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_order_is_preserved() -> None:
    chunks = chunk_section(_filing(), _section(_sentences(60)), FixedSplitter())
    assert [c.order for c in chunks] == list(range(len(chunks)))


def test_token_count_is_recorded() -> None:
    chunks = chunk_section(_filing(), _section(_sentences(40)), FixedSplitter())
    assert all(c.token_count and c.token_count > 0 for c in chunks)
