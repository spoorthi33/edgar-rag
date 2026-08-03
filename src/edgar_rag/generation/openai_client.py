"""OpenAI-backed LLM client.

Kept behind the same interface as the Anthropic client so the provider is
a config value rather than a code change, and so provider comparison
becomes an axis the evaluation harness can measure.
"""

from __future__ import annotations

import logging
from typing import Any

from edgar_rag.config import Settings, get_settings
from edgar_rag.generation.base import LLMClient, LLMResponse
from edgar_rag.generation.budget import Budget
from edgar_rag.generation.cache import ResponseCache

logger = logging.getLogger(__name__)


class OpenAIClient(LLMClient):
    """Completions from the OpenAI chat completions API."""

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
        self._model = model or settings.openai_model
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
            import openai

            if not self.settings.openai_api_key:
                raise ValueError("OPENAI_API_KEY is not set")
            self._client = openai.OpenAI(api_key=self.settings.openai_api_key)
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

        self.budget.check()

        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        request: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "max_completion_tokens": max_tokens,
        }
        if temperature is not None:
            request["temperature"] = temperature

        completion = self.client.chat.completions.create(**request)
        choice = completion.choices[0]

        # Mirrors the Anthropic client: without this, switching provider
        # silently loses truncation detection.
        truncated = choice.finish_reason == "length"
        if truncated:
            logger.warning(
                "answer hit the %d-token output cap and is incomplete; "
                "raise LLM_MAX_TOKENS or tighten the prompt",
                max_tokens,
            )

        response = LLMResponse(
            text=choice.message.content or "",
            model=completion.model,
            input_tokens=completion.usage.prompt_tokens if completion.usage else None,
            output_tokens=completion.usage.completion_tokens if completion.usage else None,
            truncated=truncated,
        )

        self.budget.record(response.model, response.input_tokens or 0, response.output_tokens or 0)
        self.cache.put(key, response)
        return response
