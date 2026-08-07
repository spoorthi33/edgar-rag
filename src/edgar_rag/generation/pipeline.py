"""Question to grounded answer."""

from __future__ import annotations

import logging
import time

from edgar_rag.config import LLMProvider, Settings, get_settings
from edgar_rag.generation.base import LLMClient
from edgar_rag.generation.grounding import (
    extract_citations,
    find_ungrounded_figures,
)
from edgar_rag.generation.prompts import SYSTEM_PROMPT, build_prompt
from edgar_rag.models import Answer, SearchFilter
from edgar_rag.retrieval.hybrid import HybridRetriever

logger = logging.getLogger(__name__)


def get_llm_client(settings: Settings | None = None) -> LLMClient:
    """Build the configured LLM client."""
    settings = settings or get_settings()

    if settings.llm_provider is LLMProvider.OPENAI:
        from edgar_rag.generation.openai_client import OpenAIClient

        return OpenAIClient(settings=settings)

    from edgar_rag.generation.anthropic_client import AnthropicClient

    return AnthropicClient(settings=settings)


class AnswerPipeline:
    """Retrieves passages, generates an answer, then verifies it."""

    def __init__(
        self,
        retriever: HybridRetriever,
        llm: LLMClient,
        settings: Settings | None = None,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.settings = settings or get_settings()

    def answer(
        self,
        question: str,
        top_k: int | None = None,
        filters: SearchFilter | None = None,
        mode: str = "hybrid",
    ) -> Answer:
        started = time.perf_counter()
        top_k = top_k or self.settings.retrieval_top_k

        retrieved = self.retriever.retrieve(question, top_k, filters, mode=mode)

        if not retrieved:
            # Answering with no passages would be answering from the model's
            # own knowledge, which is the failure this system exists to avoid.
            return Answer(
                question=question,
                answer="I don't know based on the filings provided.",
                retrieved=[],
                model=self.llm.model,
                latency_ms=(time.perf_counter() - started) * 1000,
            )

        response = self.llm.complete(
            prompt=build_prompt(question, retrieved),
            system=SYSTEM_PROMPT,
            max_tokens=self.settings.llm_max_tokens,
            temperature=self.settings.llm_temperature,
        )

        if response.truncated:
            logger.warning("answer for %r was truncated at the output cap", question)

        # A truncated answer is cut mid-sentence and often mid-citation, so
        # its trailing text is not a claim the model finished making.
        # Checking it reported invented figures that were really the digits
        # of an unclosed citation tag.
        ungrounded = [] if response.truncated else find_ungrounded_figures(response.text, retrieved)
        if ungrounded:
            logger.warning(
                "answer contains figures absent from the retrieved passages: %s",
                ", ".join(ungrounded),
            )

        return Answer(
            question=question,
            answer=response.text,
            citations=extract_citations(response.text, retrieved),
            retrieved=retrieved,
            ungrounded_figures=ungrounded,
            model=response.model,
            latency_ms=(time.perf_counter() - started) * 1000,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            truncated=response.truncated,
        )
