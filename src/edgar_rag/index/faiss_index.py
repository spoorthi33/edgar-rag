"""FAISS-backed vector index.

Vectors are L2-normalised by the embedder, so inner product equals cosine
similarity and `IndexFlatIP` gives exact cosine ranking.

Two index types, chosen by config:

  - `flat` compares against every vector. Exact, and the right choice while
    retrieval quality is still being debugged: a miss is then certainly a
    chunking or embedding problem, never the index approximating.
  - `ivf` clusters vectors and searches only the nearest cells. Much faster
    at corpus scale, at the cost of occasionally missing a true neighbour.

Metadata filtering is applied *before* ranking. Restricting to the right
company and fiscal year is what stops a semantically perfect passage from
the wrong filing being returned, which similarity alone cannot prevent.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import faiss
import numpy as np

from edgar_rag.index.base import VectorIndex
from edgar_rag.models import Chunk, RetrievedChunk, SearchFilter

logger = logging.getLogger(__name__)

INDEX_FILENAME = "faiss.index"
CHUNKS_FILENAME = "chunks.jsonl"
MANIFEST_FILENAME = "index.json"

# Below this, IVF has too few vectors per cluster to train usefully.
MIN_VECTORS_FOR_IVF = 1000

# SearchFilter field -> the ChunkMetadata attribute it constrains. Values are
# normalised to strings so tickers can match case-insensitively and years can
# be given as either int or str.
FILTERABLE_FIELDS = {
    "ciks": "cik",
    "tickers": "ticker",
    "form_types": "form_type",
    "fiscal_years": "fiscal_year",
    "items": "item",
    "parts": "part",
}


def _key(value: object) -> str:
    return str(value).upper() if value is not None else ""


class FaissIndex(VectorIndex):
    """Dense similarity search over chunk embeddings."""

    def __init__(
        self,
        dimension: int,
        index_type: str = "flat",
        nlist: int = 256,
        nprobe: int = 16,
        model_name: str | None = None,
    ) -> None:
        self.dimension = dimension
        self.index_type = index_type.lower()
        # Kept so a small first batch that forces flat can be revisited once
        # enough vectors have accumulated to train IVF.
        self._requested_type = self.index_type
        self.nlist = nlist
        self.nprobe = nprobe
        self.model_name = model_name
        self._index: faiss.Index | None = None
        self._chunks: list[Chunk] = []
        # Inverted maps from metadata value to row id. Built as chunks are
        # added so a filtered query intersects small id sets instead of
        # scanning the corpus: the scan cost 80ms at 200k chunks and grows
        # linearly, on the path every filtered query takes.
        self._by_field: dict[str, dict[str, set[int]]] = {field: {} for field in FILTERABLE_FIELDS}

    # --- Building ------------------------------------------------------

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise ValueError(f"got {len(chunks)} chunks but {len(vectors)} vectors")
        if not chunks:
            return

        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"vectors are {vectors.shape[1]}-dimensional but the index is "
                f"{self.dimension}-dimensional"
            )

        if self._index is None:
            self._index = self._build_index(vectors)
        elif self._should_upgrade_to_ivf(len(vectors)):
            # An early small batch must not permanently downgrade the index:
            # rebuild as IVF now that there are enough vectors to train on.
            self._rebuild_as_ivf(vectors)

        first_id = len(self._chunks)
        self._index.add(vectors)
        self._chunks.extend(chunks)
        self._index_metadata(chunks, first_id)

    def _index_metadata(self, chunks: list[Chunk], first_id: int) -> None:
        for offset, chunk in enumerate(chunks):
            for field, attribute in FILTERABLE_FIELDS.items():
                key = _key(getattr(chunk.metadata, attribute))
                self._by_field[field].setdefault(key, set()).add(first_id + offset)

    def _should_upgrade_to_ivf(self, incoming: int) -> bool:
        """True when IVF was requested but an earlier small batch forced flat."""
        return (
            self._requested_type == "ivf"
            and self.index_type == "flat"
            and len(self._chunks) + incoming >= MIN_VECTORS_FOR_IVF
        )

    def _rebuild_as_ivf(self, incoming: np.ndarray) -> None:
        """Retrain as IVF and re-add everything indexed so far."""
        existing = self._index.reconstruct_n(0, self._index.ntotal)
        self.index_type = "ivf"
        self._index = self._build_index(np.vstack([existing, incoming]))
        self._index.add(np.ascontiguousarray(existing, dtype=np.float32))
        logger.info("rebuilt index as IVF after reaching %d vectors", len(self._chunks))

    def _build_index(self, vectors: np.ndarray) -> faiss.Index:
        """Create the underlying index, training it first if IVF."""
        if self.index_type == "ivf":
            if len(vectors) < MIN_VECTORS_FOR_IVF:
                logger.warning(
                    "only %d vectors; using flat for now (IVF needs ~%d to train)",
                    len(vectors),
                    MIN_VECTORS_FOR_IVF,
                )
                self.index_type = "flat"
            else:
                # FAISS wants many vectors per centroid; cap nlist so a small
                # corpus does not produce near-empty cells.
                nlist = min(self.nlist, max(1, len(vectors) // 39))
                quantizer = faiss.IndexFlatIP(self.dimension)
                index = faiss.IndexIVFFlat(
                    quantizer, self.dimension, nlist, faiss.METRIC_INNER_PRODUCT
                )
                index.train(vectors)
                index.nprobe = min(self.nprobe, nlist)
                logger.info("trained IVF index with nlist=%d nprobe=%d", nlist, index.nprobe)
                return index

        return faiss.IndexFlatIP(self.dimension)

    # --- Searching -----------------------------------------------------

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int,
        filters: SearchFilter | None = None,
    ) -> list[RetrievedChunk]:
        if self._index is None or not self._chunks:
            return []

        query = np.ascontiguousarray(query_vector, dtype=np.float32).reshape(1, -1)
        allowed = self._allowed_ids(filters)
        if allowed is not None and not len(allowed):
            return []

        if allowed is None:
            scores, ids = self._index.search(query, min(top_k, len(self._chunks)))
        else:
            # An ID selector keeps filtering inside FAISS, so the requested
            # top_k is filled from matching chunks rather than being thinned
            # by discarding results after ranking.
            selector = faiss.IDSelectorArray(allowed)
            params = self._search_params(selector)
            scores, ids = self._index.search(query, min(top_k, len(allowed)), params=params)

        results: list[RetrievedChunk] = []
        for rank, (score, chunk_id) in enumerate(zip(scores[0], ids[0], strict=True)):
            if chunk_id < 0:  # FAISS pads short result sets with -1
                continue
            results.append(
                RetrievedChunk(
                    chunk=self._chunks[int(chunk_id)],
                    score=float(score),
                    dense_rank=rank + 1,
                )
            )
        return results

    def _search_params(self, selector: faiss.IDSelector) -> faiss.SearchParameters:
        if self.index_type == "ivf":
            return faiss.SearchParametersIVF(sel=selector, nprobe=self._index.nprobe)
        return faiss.SearchParameters(sel=selector)

    def _allowed_ids(self, filters: SearchFilter | None) -> np.ndarray | None:
        """Row ids matching the metadata filter, or None when unfiltered.

        Each criterion contributes the union of its values' id sets, and the
        criteria intersect. Cost scales with the number of matching chunks
        rather than the size of the corpus.
        """
        if filters is None:
            return None

        matching: set[int] | None = None
        for field, values in filters.model_dump().items():
            if not values or field not in self._by_field:
                continue
            index = self._by_field[field]
            candidates: set[int] = set()
            for value in values:
                candidates |= index.get(_key(value), set())
            matching = candidates if matching is None else matching & candidates
            if not matching:
                break

        if matching is None:  # no criteria set
            return None
        return np.fromiter(sorted(matching), dtype=np.int64, count=len(matching))

    @property
    def size(self) -> int:
        return len(self._chunks)

    # --- Persistence ---------------------------------------------------

    def save(self, path: Path) -> None:
        """Write index, chunks and build settings to `path`."""
        if self._index is None:
            raise ValueError("cannot save an empty index")

        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(path / INDEX_FILENAME))

        with (path / CHUNKS_FILENAME).open("w") as handle:
            for chunk in self._chunks:
                handle.write(chunk.model_dump_json() + "\n")

        (path / MANIFEST_FILENAME).write_text(
            json.dumps(
                {
                    "dimension": self.dimension,
                    "index_type": self.index_type,
                    "nlist": self.nlist,
                    "nprobe": self.nprobe,
                    # Recorded so a query is never embedded with a different
                    # model than the one that built the index.
                    "model_name": self.model_name,
                    "size": self.size,
                },
                indent=2,
            )
        )
        logger.info("saved %d vectors to %s", self.size, path)

    def load(self, path: Path) -> None:
        """Restore a previously saved index."""
        manifest = json.loads((path / MANIFEST_FILENAME).read_text())

        if manifest["dimension"] != self.dimension:
            raise ValueError(
                f"index at {path} is {manifest['dimension']}-dimensional but "
                f"{self.dimension} was configured"
            )
        if self.model_name and manifest.get("model_name") not in (None, self.model_name):
            raise ValueError(
                f"index at {path} was built with {manifest['model_name']} but "
                f"{self.model_name} is configured; queries would be embedded "
                "into a different space than the chunks"
            )

        self.index_type = manifest["index_type"]
        self._requested_type = self.index_type
        self.nlist = manifest.get("nlist", self.nlist)
        self.nprobe = manifest.get("nprobe", self.nprobe)
        self._index = faiss.read_index(str(path / INDEX_FILENAME))
        if self.index_type == "ivf":
            self._index.nprobe = self.nprobe

        with (path / CHUNKS_FILENAME).open() as handle:
            self._chunks = [Chunk.model_validate_json(line) for line in handle if line.strip()]

        self._by_field = {field: {} for field in FILTERABLE_FIELDS}
        self._index_metadata(self._chunks, first_id=0)

        logger.info("loaded %d vectors from %s", self.size, path)
