"""Runs the labelled question set and reports the metrics."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import pandas as pd

from edgar_rag.config import Settings, get_settings
from edgar_rag.eval.dataset import EvalDataset, EvalQuestion
from edgar_rag.eval.judge import FaithfulnessVerdict, judge_answer
from edgar_rag.eval.metrics import RetrievalMetrics, mean
from edgar_rag.generation.base import LLMClient
from edgar_rag.generation.pipeline import AnswerPipeline
from edgar_rag.models import Answer, Chunk

logger = logging.getLogger(__name__)


@dataclass
class QuestionResult:
    """Everything measured for one question."""

    question: EvalQuestion
    answer: Answer
    retrieval: RetrievalMetrics
    faithfulness: FaithfulnessVerdict
    quoted_expected_figures: bool
    latency_ms: float

    def as_row(self) -> dict:
        return {
            "id": self.question.id,
            "category": self.question.category,
            "answerable": self.question.answerable,
            "recall@k": self.retrieval.recall_at_k,
            "achievable_recall@k": self.retrieval.achievable_recall_at_k,
            "precision@k": self.retrieval.precision_at_k,
            "mrr": self.retrieval.reciprocal_rank,
            "hit@k": self.retrieval.hit_at_k,
            "relevant_chunks": self.retrieval.relevant_count,
            "faithful": self.faithfulness.is_faithful,
            "declined": self.faithfulness.declined,
            "numerically_grounded": self.faithfulness.numerically_grounded,
            "judge_supported": self.faithfulness.judge_supported,
            "quoted_expected": self.quoted_expected_figures,
            "latency_ms": self.latency_ms,
            "input_tokens": self.answer.input_tokens,
            "output_tokens": self.answer.output_tokens,
        }


@dataclass
class EvalReport:
    """Aggregate results for one run."""

    results: list[QuestionResult] = field(default_factory=list)
    mode: str = "hybrid"
    top_k: int = 5
    model: str | None = None
    elapsed_s: float = 0.0
    cost_usd: float = 0.0
    retrieval_only: bool = False

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame([r.as_row() for r in self.results])

    def summary(self) -> dict[str, float | int | str]:
        """Headline numbers.

        Retrieval metrics cover answerable questions only: a question the
        corpus cannot answer has no relevant chunk, so scoring its recall
        as zero would blame retrieval for a question with no right answer.
        """
        answerable = [r for r in self.results if r.question.answerable]
        unanswerable = [r for r in self.results if not r.question.answerable]

        return {
            "mode": self.mode,
            "top_k": self.top_k,
            "questions": len(self.results),
            "answerable": len(answerable),
            f"recall@{self.top_k}": mean([r.retrieval.recall_at_k for r in answerable]),
            f"achievable_recall@{self.top_k}": mean(
                [r.retrieval.achievable_recall_at_k for r in answerable]
            ),
            f"precision@{self.top_k}": mean([r.retrieval.precision_at_k for r in answerable]),
            "mrr": mean([r.retrieval.reciprocal_rank for r in answerable]),
            f"hit@{self.top_k}": mean([r.retrieval.hit_at_k for r in answerable]),
            # None when no answers were generated: a faithfulness score of
            # 1.000 computed over empty answers would be a lie.
            "faithfulness": None
            if self.retrieval_only
            else mean([float(r.faithfulness.is_faithful) for r in self.results]),
            "numeric_grounding": None
            if self.retrieval_only
            else mean([float(r.faithfulness.numerically_grounded) for r in self.results]),
            # Restraint is measured too: on questions the corpus cannot
            # answer, declining is the correct behaviour.
            "correct_declines": None
            if self.retrieval_only
            else mean([float(r.faithfulness.declined) for r in unanswerable]),
            # None rather than 0.0 when no question carries expected
            # figures: an empty mean reads as total failure.
            "quoted_expected": (
                mean(scored)
                if (
                    scored := [
                        float(r.quoted_expected_figures)
                        for r in answerable
                        if r.question.expected_figures
                    ]
                )
                else None
            ),
            "median_latency_ms": float(pd.Series([r.latency_ms for r in self.results]).median())
            if self.results
            else 0.0,
            "cost_usd": self.cost_usd,
            "elapsed_s": self.elapsed_s,
        }


class EvalHarness:
    """Runs every question and scores retrieval and faithfulness."""

    def __init__(
        self,
        pipeline: AnswerPipeline,
        chunks: list[Chunk],
        judge_llm: LLMClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.chunks = chunks
        self.judge_llm = judge_llm
        self.settings = settings or get_settings()

    def run(
        self,
        dataset: EvalDataset,
        mode: str = "hybrid",
        top_k: int | None = None,
        use_judge: bool = True,
        retrieval_only: bool = False,
    ) -> EvalReport:
        top_k = top_k or self.settings.retrieval_top_k
        started = time.perf_counter()
        report = EvalReport(
            mode=mode,
            top_k=top_k,
            model=self.pipeline.llm.model,
            retrieval_only=retrieval_only,
        )

        for index, question in enumerate(dataset.questions, start=1):
            logger.info("[%d/%d] %s", index, len(dataset), question.id)
            report.results.append(
                self._run_one(
                    question,
                    mode=mode,
                    top_k=top_k,
                    use_judge=use_judge,
                    retrieval_only=retrieval_only,
                )
            )

        report.elapsed_s = time.perf_counter() - started
        report.cost_usd = self._spent()
        return report

    def _run_one(
        self,
        question: EvalQuestion,
        mode: str,
        top_k: int,
        use_judge: bool,
        retrieval_only: bool = False,
    ) -> QuestionResult:
        started = time.perf_counter()
        if retrieval_only:
            # Tuning retrieval does not need answers, and skipping them
            # makes the loop free — the difference between iterating in
            # seconds and paying a dollar a sweep.
            answer = self._retrieve_only(question, top_k=top_k, mode=mode)
        else:
            answer = self.pipeline.answer(
                question.question, top_k=top_k, filters=question.filters, mode=mode
            )
        latency_ms = (time.perf_counter() - started) * 1000

        relevant = question.relevant_chunk_ids(self.chunks)
        retrieval = RetrievalMetrics.compute(answer.retrieved, relevant, top_k)

        faithfulness = judge_answer(
            answer,
            llm=self.judge_llm,
            settings=self.settings,
            use_llm=use_judge and self.judge_llm is not None,
        )

        return QuestionResult(
            question=question,
            answer=answer,
            retrieval=retrieval,
            faithfulness=faithfulness,
            quoted_expected_figures=_quoted_all(answer.answer, question.expected_figures),
            latency_ms=latency_ms,
        )

    def _retrieve_only(self, question: EvalQuestion, top_k: int, mode: str) -> Answer:
        retrieved = self.pipeline.retriever.retrieve(
            question.question, top_k, question.filters, mode=mode
        )
        return Answer(question=question.question, answer="", retrieved=retrieved)

    def _spent(self) -> float:
        total = 0.0
        for client in (self.pipeline.llm, self.judge_llm):
            budget = getattr(client, "budget", None)
            if budget is not None:
                total += budget.cost_usd
        return total


def _quoted_all(answer: str, expected: list[str]) -> bool:
    """Whether the answer contains every expected figure.

    Compared with commas stripped so "8,866" matches "8866", but not
    numerically: this checks that the model *quoted* the figure rather
    than merely landed on the same value by another route.
    """
    if not expected:
        return True
    normalised = answer.replace(",", "")
    return all(figure.replace(",", "") in normalised for figure in expected)
