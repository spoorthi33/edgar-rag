"""Vector index contract.

Backed by FAISS: exact `flat` search while the pipeline is still being
debugged, `ivf` once the corpus is large enough that exhaustive search is
too slow.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from edgar_rag.models import Chunk, RetrievedChunk, SearchFilter


class VectorIndex(ABC):
    """Nearest-neighbour search over chunk embeddings."""

    @abstractmethod
    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        """Index `chunks` with their corresponding row vectors."""

    @abstractmethod
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int,
        filters: SearchFilter | None = None,
    ) -> list[RetrievedChunk]:
        """Return the `top_k` nearest chunks.

        `filters` is applied *before* ranking: restricting to the right
        company and year is what stops a semantically perfect match from
        the wrong filing being returned.
        """

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist index and chunk metadata to `path`."""

    @abstractmethod
    def load(self, path: Path) -> None:
        """Restore a previously saved index from `path`."""

    @property
    @abstractmethod
    def size(self) -> int:
        """Number of indexed vectors."""
