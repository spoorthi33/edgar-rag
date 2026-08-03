"""Embedding model contract.

Queries and documents are embedded through separate methods because some
models (including the BGE family) expect an instruction prefix on the query
side only.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Embedder(ABC):
    """Turns text into vectors that place similar meanings near each other."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Vector length; must match the index this feeds."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Model identifier, recorded alongside the index for reproducibility."""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embed chunk texts. Returns shape (len(texts), dimension)."""

    @abstractmethod
    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single user question. Returns shape (dimension,)."""

    @property
    @abstractmethod
    def max_tokens(self) -> int:
        """Longest input the model accepts before it truncates."""

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Token count as the model itself measures it.

        Chunking must size against this rather than a character estimate:
        filings are dense with figures and tickers that fragment into far
        more tokens than prose, and anything past `max_tokens` is discarded
        silently at embedding time.
        """
