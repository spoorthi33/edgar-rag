"""Local sentence-transformer embeddings.

Runs on the machine rather than through an API: embedding a full corpus is
hundreds of thousands of calls, which would dominate both cost and latency
if it were remote.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from edgar_rag.config import Settings, get_settings
from edgar_rag.embeddings.base import Embedder

logger = logging.getLogger(__name__)

# BGE models are trained with an instruction prefix on the query side only;
# adding it to documents as well degrades retrieval.
QUERY_PREFIXES = {
    "bge": "Represent this sentence for searching relevant passages: ",
}


class SentenceTransformerEmbedder(Embedder):
    """Embedder backed by a HuggingFace sentence-transformers model."""

    def __init__(
        self,
        model_name: str | None = None,
        batch_size: int | None = None,
        settings: Settings | None = None,
        model: Any | None = None,
    ) -> None:
        settings = settings or get_settings()
        self._model_name = model_name or settings.embedding_model
        self.batch_size = batch_size if batch_size is not None else settings.embedding_batch_size
        self._expected_dimension = settings.embedding_dim
        self._model = model  # loaded lazily; importing torch is slow

    @property
    def model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("loading embedding model %s", self._model_name)
            self._model = SentenceTransformer(self._model_name)
            self._check_dimension()
        return self._model

    def _check_dimension(self) -> None:
        """Fail fast when the configured dimension does not match the model.

        A mismatch would otherwise surface as a malformed vector index: the
        index is built to the configured width, not the model's.
        """
        actual = int(self._model.get_sentence_embedding_dimension())
        if actual != self._expected_dimension:
            raise ValueError(
                f"{self._model_name} produces {actual}-dimensional vectors but "
                f"EMBEDDING_DIM is set to {self._expected_dimension}. "
                "Update EMBEDDING_DIM to match the model."
            )

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            # Normalised so cosine similarity is a plain dot product, which
            # is what the FAISS inner-product index expects.
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([self._query_prefix() + text])[0]

    @property
    def max_tokens(self) -> int:
        return int(self.model.max_seq_length)

    def count_tokens(self, text: str) -> int:
        """Exact count from the model's own tokenizer.

        Chunking measures candidate joins that it then rejects for being too
        long, so over-limit input is expected here. The tokenizer's warning
        about it is suppressed to keep it from dominating the logs; nothing
        is embedded at this point.
        """
        from transformers import logging as hf_logging

        tokenizer = self.model.tokenizer
        verbosity = hf_logging.get_verbosity()
        hf_logging.set_verbosity_error()
        try:
            return len(tokenizer.encode(text, add_special_tokens=True))
        finally:
            hf_logging.set_verbosity(verbosity)

    def _query_prefix(self) -> str:
        lowered = self._model_name.lower()
        for family, prefix in QUERY_PREFIXES.items():
            if family in lowered:
                return prefix
        return ""
