"""On-disk cache for LLM responses, keyed by request content.

The evaluation harness re-runs the same questions after every retrieval or
prompt change. Without a cache each re-run re-bills every question; with
one, only the questions whose context actually changed cost anything.

The key must cover everything that changes the response — model, system
prompt, user prompt, and sampling settings. A key that misses one of those
silently serves a stale answer; a key that includes something volatile
(a timestamp, an unordered dict) never hits and quietly re-bills.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from edgar_rag.generation.base import LLMResponse

logger = logging.getLogger(__name__)


class ResponseCache:
    """Content-addressed cache of completions."""

    def __init__(self, path: Path, enabled: bool = True) -> None:
        self.path = Path(path)
        self.enabled = enabled
        self.hits = 0
        self.misses = 0
        if self.enabled:
            self.path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key(
        model: str,
        prompt: str,
        system: str | None,
        max_tokens: int | None,
        temperature: float | None,
    ) -> str:
        """Stable hash of everything that determines the response.

        `sort_keys` matters: without it, dict ordering could vary between
        runs and every lookup would miss while still writing an entry.
        """
        payload = json.dumps(
            {
                "model": model,
                "system": system,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def get(self, key: str) -> LLMResponse | None:
        if not self.enabled:
            return None
        entry = self.path / f"{key}.json"
        if not entry.is_file():
            self.misses += 1
            return None

        try:
            data = json.loads(entry.read_text())
        except (OSError, json.JSONDecodeError):
            # A truncated write from an interrupted run: treat as a miss
            # rather than failing the whole evaluation.
            logger.warning("discarding unreadable cache entry %s", entry.name)
            self.misses += 1
            return None

        self.hits += 1
        return LLMResponse(
            text=data["text"],
            model=data["model"],
            input_tokens=data.get("input_tokens"),
            output_tokens=data.get("output_tokens"),
            # Must survive the round trip: an incomplete answer served from
            # cache would otherwise look complete, and on evaluation re-runs
            # every answer comes from cache.
            truncated=data.get("truncated", False),
            cached=True,
        )

    def put(self, key: str, response: LLMResponse) -> None:
        if not self.enabled:
            return
        entry = self.path / f"{key}.json"
        tmp = entry.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(
                {
                    "text": response.text,
                    "model": response.model,
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "truncated": response.truncated,
                },
                indent=2,
            )
        )
        tmp.replace(entry)

    def summary(self) -> str:
        total = self.hits + self.misses
        rate = self.hits / total if total else 0.0
        return f"cache {self.hits}/{total} hits ({rate:.0%})"
