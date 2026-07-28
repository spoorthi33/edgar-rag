"""LLM client contract.

Implemented for Anthropic (default) and OpenAI. Kept deliberately small —
the interesting work lives in prompt construction and grounding checks,
not in provider plumbing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cached: bool = False


class LLMClient(ABC):
    """Single-turn completion, provider-agnostic."""

    @property
    @abstractmethod
    def model(self) -> str:
        """Model identifier, recorded on every answer and eval run."""

    @abstractmethod
    def complete(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        """Generate a completion for `prompt`."""
