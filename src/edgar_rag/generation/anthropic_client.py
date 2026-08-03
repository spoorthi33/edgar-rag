"""Anthropic-backed LLM client.

Every call passes through the budget guard and the response cache before
reaching the API, so a runaway loop stops at the ceiling and a repeated
evaluation run costs nothing.
"""

from __future__ import annotations

import logging
from typing import Any

from edgar_rag.config import Settings, get_settings
from edgar_rag.generation.base import LLMClient, LLMResponse
from edgar_rag.generation.budget import Budget
from edgar_rag.generation.cache import ResponseCache

logger = logging.getLogger(__name__)


class AnthropicClient(LLMClient):
    """Completions from the Anthropic Messages API."""

    def __init__(
        self,
        model: str | None = None,
        settings: Settings | None = None,
        budget: Budget | None = None,
        cache: ResponseCache | None = None,
        client: Any | None = None,
    ) -> None:
        settings = settings or get_settings()
        self.settings = settings
        self._model = model or settings.anthropic_model
        self.budget = budget or Budget(
            max_calls=settings.max_api_calls_per_run,
            max_cost_usd=settings.max_cost_usd_per_run,
        )
        self.cache = cache or ResponseCache(settings.llm_cache_path, settings.llm_cache_enabled)
        self._client = client

    @property
    def model(self) -> str:
        return self._model

    @property
    def client(self) -> Any:
        if self._client is None:
            import anthropic

            if not self.settings.anthropic_api_key:
                raise ValueError(
                    "ANTHROPIC_API_KEY is not set. Generation requires an API key "
                    "from console.anthropic.com — a Claude Pro subscription does "
                    "not include API access."
                )
            self._client = anthropic.Anthropic(api_key=self.settings.anthropic_api_key)
        return self._client

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        max_tokens = max_tokens if max_tokens is not None else self.settings.llm_max_tokens

        key = ResponseCache.key(self._model, prompt, system, max_tokens, temperature)
        cached = self.cache.get(key)
        if cached is not None:
            self.budget.record(self._model, 0, 0, cached=True)
            return cached

        # Checked immediately before the call, so a loop stops at the
        # ceiling rather than after it.
        self.budget.check()

        request: dict[str, Any] = {
            "model": self._model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            request["system"] = system

        message = self.client.messages.create(**request)

        text = "".join(block.text for block in message.content if block.type == "text")
        truncated = message.stop_reason == "max_tokens"
        if truncated:
            logger.warning(
                "answer hit the %d-token output cap and is incomplete; "
                "raise LLM_MAX_TOKENS or tighten the prompt",
                max_tokens,
            )
        response = LLMResponse(
            text=text,
            model=message.model,
            input_tokens=message.usage.input_tokens,
            output_tokens=message.usage.output_tokens,
            truncated=truncated,
        )

        cost = self.budget.record(
            response.model, response.input_tokens or 0, response.output_tokens or 0
        )
        logger.debug(
            "anthropic call: %d in / %d out, $%.4f (run total %s)",
            response.input_tokens or 0,
            response.output_tokens or 0,
            cost,
            self.budget.summary(),
        )

        self.cache.put(key, response)
        return response
