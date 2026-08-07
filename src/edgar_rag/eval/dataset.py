"""The labelled question set.

Marking which of 2,105 chunks are relevant by hand is impractical, so
relevance is expressed as a predicate over chunk text and metadata and
resolved against the corpus at run time. That keeps the labels stable when
chunking changes — a hand-listed chunk id would go stale the moment the
splitter is retuned, which is exactly when the metrics matter most.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel, Field

from edgar_rag.models import Chunk, SearchFilter


class RelevanceRule(BaseModel):
    """Which chunks count as containing the answer."""

    tickers: list[str] | None = None
    fiscal_years: list[int] | None = None
    items: list[str] | None = None
    # Every phrase must appear (case-insensitive). Requiring all of them
    # keeps the label tight: a chunk merely on the topic is not evidence.
    must_contain: list[str] = Field(default_factory=list)

    def matches(self, chunk: Chunk) -> bool:
        meta = chunk.metadata
        if self.tickers and (meta.ticker or "").upper() not in {t.upper() for t in self.tickers}:
            return False
        if self.fiscal_years and meta.fiscal_year not in self.fiscal_years:
            return False
        if self.items and (meta.item or "") not in self.items:
            return False

        text = chunk.text.lower()
        return all(phrase.lower() in text for phrase in self.must_contain)


class EvalQuestion(BaseModel):
    """One labelled question."""

    id: str
    question: str
    # Applied at query time, exactly as a caller would.
    filters: SearchFilter | None = None
    relevance: RelevanceRule | None = None
    # Figures the answer should quote, checked literally against the text.
    expected_figures: list[str] = Field(default_factory=list)
    # False for questions the corpus cannot answer. Declining is the
    # correct behaviour and is scored as such, so the harness measures
    # restraint as well as recall.
    answerable: bool = True
    category: str = "general"

    def relevant_chunk_ids(self, chunks: Iterable[Chunk]) -> set[str]:
        if self.relevance is None:
            return set()
        return {chunk.chunk_id for chunk in chunks if self.relevance.matches(chunk)}


class EvalDataset(BaseModel):
    questions: list[EvalQuestion]

    def __len__(self) -> int:
        return len(self.questions)

    @classmethod
    def load(cls, path: Path) -> EvalDataset:
        questions = [
            EvalQuestion.model_validate_json(line)
            for line in Path(path).read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("//")
        ]
        return cls(questions=questions)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "\n".join(q.model_dump_json(exclude_none=True) for q in self.questions) + "\n"
        )

    def answerable(self) -> list[EvalQuestion]:
        return [q for q in self.questions if q.answerable]

    def unanswerable(self) -> list[EvalQuestion]:
        return [q for q in self.questions if not q.answerable]


def resolve_relevance(dataset: EvalDataset, chunks: Iterable[Chunk]) -> dict[str, set[str]]:
    """Relevant chunk ids per question, in a single pass over the corpus.

    One pass rather than one per question: at a million chunks, scanning
    the corpus 52 times would dominate the run.
    """
    matches: dict[str, set[str]] = {q.id: set() for q in dataset.questions}
    rules = [(q.id, q.relevance) for q in dataset.questions if q.relevance is not None]

    for chunk in chunks:
        for question_id, rule in rules:
            if rule.matches(chunk):
                matches[question_id].add(chunk.chunk_id)
    return matches


def validate_against_corpus(
    dataset: EvalDataset, relevance: dict[str, set[str]]
) -> dict[str, list[str]]:
    """Find labels the corpus cannot support.

    A question marked answerable whose relevance rule matches nothing is a
    broken label, not a retrieval failure — scoring it would silently
    depress recall and send the next fix in the wrong direction.
    """
    problems: dict[str, list[str]] = {"no_matching_chunks": [], "unanswerable_with_matches": []}

    for question in dataset.questions:
        matches = relevance.get(question.id, set())
        if question.answerable and question.relevance and not matches:
            problems["no_matching_chunks"].append(question.id)
        if not question.answerable and matches:
            problems["unanswerable_with_matches"].append(question.id)

    return {key: ids for key, ids in problems.items() if ids}
