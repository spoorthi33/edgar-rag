"""Phase 8 tests: metrics, labels, judge parsing, and the harness."""

from __future__ import annotations

from datetime import date

import pytest

from edgar_rag.eval.dataset import (
    EvalDataset,
    EvalQuestion,
    RelevanceRule,
    validate_against_corpus,
)
from edgar_rag.eval.harness import EvalHarness, _quoted_all
from edgar_rag.eval.judge import _parse_verdict, judge_answer
from edgar_rag.eval.metrics import (
    RetrievalMetrics,
    achievable_recall_at_k,
    hit_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from edgar_rag.generation.base import LLMClient, LLMResponse
from edgar_rag.models import Answer, Chunk, ChunkMetadata, FormType, RetrievedChunk


def _chunk(chunk_id: str, text: str = "text", *, ticker: str = "AAPL", item: str = "1A") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        filing_id="cik/acc",
        text=text,
        order=0,
        metadata=ChunkMetadata(
            cik="0000320193",
            ticker=ticker,
            company_name="Apple Inc.",
            form_type=FormType.TEN_K,
            fiscal_year=2025,
            item=item,
            part="I",
            filing_date=date(2025, 10, 31),
            accession_number="acc",
        ),
    )


def _retrieved(*chunk_ids: str) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(chunk=_chunk(cid), score=1.0 - i / 10, dense_rank=i + 1)
        for i, cid in enumerate(chunk_ids)
    ]


# --- Metrics -------------------------------------------------------------


def test_recall_counts_relevant_chunks_found() -> None:
    retrieved = _retrieved("a", "b", "c", "d", "e")
    assert recall_at_k(retrieved, {"a", "c"}, 5) == 1.0
    assert recall_at_k(retrieved, {"a", "z"}, 5) == 0.5


def test_recall_is_bounded_when_relevant_exceeds_k() -> None:
    """The reason achievable_recall exists: perfect retrieval scores 0.05."""
    retrieved = _retrieved("a", "b", "c", "d", "e")
    relevant = {f"chunk{i}" for i in range(100)} | {"a", "b", "c", "d", "e"}

    assert recall_at_k(retrieved, relevant, 5) == pytest.approx(5 / 105)
    assert achievable_recall_at_k(retrieved, relevant, 5) == 1.0


def test_achievable_recall_matches_recall_when_relevant_is_small() -> None:
    retrieved = _retrieved("a", "b", "c")
    assert achievable_recall_at_k(retrieved, {"a", "b"}, 5) == recall_at_k(retrieved, {"a", "b"}, 5)


def test_precision_measures_noise_in_the_prompt() -> None:
    retrieved = _retrieved("a", "b", "c", "d", "e")
    assert precision_at_k(retrieved, {"a", "b"}, 5) == 0.4


def test_reciprocal_rank_rewards_ranking_evidence_first() -> None:
    retrieved = _retrieved("a", "b", "c")
    assert reciprocal_rank(retrieved, {"a"}) == 1.0
    assert reciprocal_rank(retrieved, {"c"}) == pytest.approx(1 / 3)


def test_reciprocal_rank_is_zero_when_nothing_relevant_retrieved() -> None:
    assert reciprocal_rank(_retrieved("a", "b"), {"z"}) == 0.0


def test_hit_is_binary() -> None:
    retrieved = _retrieved("a", "b")
    assert hit_at_k(retrieved, {"b"}, 5) == 1.0
    assert hit_at_k(retrieved, {"z"}, 5) == 0.0


def test_metrics_on_empty_retrieval() -> None:
    metrics = RetrievalMetrics.compute([], {"a"}, 5)
    assert (metrics.recall_at_k, metrics.precision_at_k, metrics.hit_at_k) == (0.0, 0.0, 0.0)


def test_metrics_with_no_relevant_chunks() -> None:
    """Unanswerable questions have no evidence; scores must not divide by zero."""
    metrics = RetrievalMetrics.compute(_retrieved("a"), set(), 5)
    assert metrics.recall_at_k == 0.0
    assert metrics.achievable_recall_at_k == 0.0


# --- Relevance rules -----------------------------------------------------


def test_rule_requires_every_phrase() -> None:
    """Requiring all phrases keeps the label tight: on-topic is not evidence."""
    rule = RelevanceRule(must_contain=["research and development", "expense"])

    assert rule.matches(_chunk("a", "research and development expense was $10"))
    assert not rule.matches(_chunk("b", "research and development activities"))


def test_rule_matching_is_case_insensitive() -> None:
    assert RelevanceRule(must_contain=["Mine Safety"]).matches(_chunk("a", "mine safety is n/a"))


def test_rule_filters_on_metadata() -> None:
    rule = RelevanceRule(tickers=["MSFT"], items=["1A"])
    assert not rule.matches(_chunk("a", "text", ticker="AAPL", item="1A"))
    assert rule.matches(_chunk("b", "text", ticker="MSFT", item="1A"))


def test_relevant_ids_resolve_against_the_corpus() -> None:
    question = EvalQuestion(
        id="q1", question="?", relevance=RelevanceRule(must_contain=["goodwill"])
    )
    corpus = [_chunk("a", "goodwill impairment"), _chunk("b", "unrelated")]

    assert question.relevant_chunk_ids(corpus) == {"a"}


def test_a_rule_matching_nothing_is_reported_as_a_broken_label() -> None:
    """Scoring it as a miss would blame retrieval and misdirect the next fix."""
    dataset = EvalDataset(
        questions=[
            EvalQuestion(
                id="q1", question="?", relevance=RelevanceRule(must_contain=["nonexistent"])
            )
        ]
    )
    problems = validate_against_corpus(dataset, [_chunk("a", "something else")])

    assert problems["no_matching_chunks"] == ["q1"]


def test_unanswerable_question_with_matches_is_reported() -> None:
    dataset = EvalDataset(
        questions=[
            EvalQuestion(
                id="q1",
                question="?",
                answerable=False,
                relevance=RelevanceRule(must_contain=["goodwill"]),
            )
        ]
    )
    problems = validate_against_corpus(dataset, [_chunk("a", "goodwill")])

    assert problems["unanswerable_with_matches"] == ["q1"]


def test_clean_labels_report_no_problems() -> None:
    dataset = EvalDataset(
        questions=[EvalQuestion(id="q1", question="?", relevance=RelevanceRule(must_contain=["x"]))]
    )
    assert validate_against_corpus(dataset, [_chunk("a", "x marks it")]) == {}


def test_dataset_round_trips_through_disk(tmp_path) -> None:
    dataset = EvalDataset(
        questions=[
            EvalQuestion(id="q1", question="What?", category="figure"),
            EvalQuestion(id="q2", question="Unknowable?", answerable=False),
        ]
    )
    path = tmp_path / "questions.jsonl"
    dataset.save(path)

    restored = EvalDataset.load(path)
    assert len(restored) == 2
    assert len(restored.answerable()) == 1
    assert len(restored.unanswerable()) == 1


# --- Judge ---------------------------------------------------------------


def test_unsupported_is_not_read_as_supported() -> None:
    """ "UNSUPPORTED" contains "SUPPORTED"; order of checks decides correctness."""
    verdict, _ = _parse_verdict("UNSUPPORTED - the figure is absent from the context.")
    assert verdict is False


def test_supported_is_parsed() -> None:
    verdict, reason = _parse_verdict("SUPPORTED - every claim appears in passage 2.")
    assert verdict is True
    assert "passage 2" in reason


def test_unparseable_verdict_is_none_rather_than_a_guess() -> None:
    verdict, _ = _parse_verdict("I am not sure about this one.")
    assert verdict is None


class StubJudge(LLMClient):
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls = 0

    @property
    def model(self) -> str:
        return "stub-judge"

    def complete(self, prompt, system=None, max_tokens=None, temperature=None) -> LLMResponse:
        self.calls += 1
        return LLMResponse(text=self.text, model=self.model)


def _answer(text: str, ungrounded: list[str] | None = None) -> Answer:
    return Answer(
        question="q",
        answer=text,
        retrieved=_retrieved("a"),
        ungrounded_figures=ungrounded or [],
    )


def test_invented_figure_makes_an_answer_unfaithful() -> None:
    verdict = judge_answer(_answer("It was $99.9bn.", ungrounded=["99.9"]), use_llm=False)

    assert verdict.numerically_grounded is False
    assert verdict.is_faithful is False


def test_declining_counts_as_faithful() -> None:
    """Reporting that the filings lack the answer is correct behaviour."""
    judge = StubJudge("UNSUPPORTED")
    verdict = judge_answer(_answer("I don't know based on the filings provided."), llm=judge)

    assert verdict.declined is True
    assert verdict.is_faithful is True
    assert judge.calls == 0  # nothing asserted, so nothing to pay to judge


def test_judge_verdict_can_fail_a_numerically_clean_answer() -> None:
    """Catches unsupported claims that contain no invented number."""
    verdict = judge_answer(_answer("Apple leads the market."), llm=StubJudge("UNSUPPORTED"))

    assert verdict.numerically_grounded is True
    assert verdict.is_faithful is False


def test_unparseable_judge_does_not_fail_the_answer() -> None:
    verdict = judge_answer(_answer("A claim."), llm=StubJudge("hmm"))
    assert verdict.judge_supported is None
    assert verdict.is_faithful is True


# --- Expected figures ----------------------------------------------------


def test_expected_figures_match_ignoring_commas() -> None:
    assert _quoted_all("R&D was $8,866 million", ["8866"])
    assert _quoted_all("R&D was $8866 million", ["8,866"])


def test_missing_expected_figure_is_detected() -> None:
    assert not _quoted_all("R&D was $1,000 million", ["8,866"])


def test_no_expected_figures_passes() -> None:
    assert _quoted_all("anything", [])


# --- Harness -------------------------------------------------------------


class StubPipeline:
    def __init__(self, retrieved: list[RetrievedChunk]) -> None:
        self.retrieved = retrieved
        self.llm = type("LLM", (), {"model": "stub"})()
        self.answer_calls = 0
        self.retriever = self

    def answer(self, question, top_k=None, filters=None, mode="hybrid"):
        self.answer_calls += 1
        return Answer(question=question, answer="An answer.", retrieved=self.retrieved)

    def retrieve(self, query, top_k, filters=None, mode="hybrid"):
        return self.retrieved


def test_retrieval_only_makes_no_generation_calls() -> None:
    """The tuning loop must be free, or it will not be run often enough."""
    pipeline = StubPipeline(_retrieved("a", "b"))
    harness = EvalHarness(pipeline=pipeline, chunks=[_chunk("a", "goodwill")])
    dataset = EvalDataset(
        questions=[
            EvalQuestion(id="q1", question="?", relevance=RelevanceRule(must_contain=["goodwill"]))
        ]
    )

    report = harness.run(dataset, retrieval_only=True)

    assert pipeline.answer_calls == 0
    assert report.results[0].retrieval.hit_at_k == 1.0


def test_retrieval_only_reports_no_faithfulness() -> None:
    """A faithfulness score over empty answers would be a lie."""
    harness = EvalHarness(pipeline=StubPipeline(_retrieved("a")), chunks=[_chunk("a")])
    dataset = EvalDataset(questions=[EvalQuestion(id="q1", question="?")])

    summary = harness.run(dataset, retrieval_only=True).summary()

    assert summary["faithfulness"] is None
    assert summary["correct_declines"] is None


def test_summary_excludes_unanswerable_from_retrieval_metrics() -> None:
    """They have no relevant chunk; scoring them would blame retrieval."""
    harness = EvalHarness(pipeline=StubPipeline([]), chunks=[_chunk("a", "goodwill")])
    dataset = EvalDataset(
        questions=[
            EvalQuestion(id="q1", question="?", relevance=RelevanceRule(must_contain=["goodwill"])),
            EvalQuestion(id="q2", question="?", answerable=False),
        ]
    )

    summary = harness.run(dataset, retrieval_only=True).summary()

    assert summary["questions"] == 2
    assert summary["answerable"] == 1


def test_quoted_expected_is_none_when_nothing_is_labelled() -> None:
    """An empty mean would read as total failure."""
    harness = EvalHarness(pipeline=StubPipeline(_retrieved("a")), chunks=[_chunk("a")])
    dataset = EvalDataset(questions=[EvalQuestion(id="q1", question="?")])

    assert harness.run(dataset, retrieval_only=True).summary()["quoted_expected"] is None


def test_report_frame_has_a_row_per_question() -> None:
    harness = EvalHarness(pipeline=StubPipeline(_retrieved("a")), chunks=[_chunk("a")])
    dataset = EvalDataset(questions=[EvalQuestion(id=f"q{i}", question="?") for i in range(3)])

    frame = harness.run(dataset, retrieval_only=True).frame()

    assert len(frame) == 3
    assert set(frame["id"]) == {"q0", "q1", "q2"}
