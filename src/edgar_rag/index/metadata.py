"""Metadata filtering shared by the dense and sparse retrievers.

Both must apply the same filter, and both must apply it *before* ranking:
restricting to the right company and fiscal year is what stops a passage
that is a perfect match on wording from being returned out of the wrong
filing, which relevance scoring alone cannot prevent.

Filtering is done through inverted maps rather than by scanning the corpus,
because it sits on the path of every query. At 200k chunks a scan measured
80ms; the lookup measures 3.5ms and scales with matches, not corpus size.
"""

from __future__ import annotations

from edgar_rag.models import Chunk, SearchFilter

# SearchFilter field -> the ChunkMetadata attribute it constrains. Values are
# normalised to upper-case strings so tickers match case-insensitively and
# years can be supplied as either int or str.
FILTERABLE_FIELDS = {
    "ciks": "cik",
    "tickers": "ticker",
    "form_types": "form_type",
    "fiscal_years": "fiscal_year",
    "items": "item",
    "parts": "part",
}


def normalise(value: object) -> str:
    return str(value).upper() if value is not None else ""


class MetadataFilterIndex:
    """Inverted maps from metadata value to chunk position."""

    def __init__(self) -> None:
        self._by_field: dict[str, dict[str, set[int]]] = {field: {} for field in FILTERABLE_FIELDS}

    def add(self, chunks: list[Chunk], first_id: int = 0) -> None:
        """Record `chunks`, numbered from `first_id`."""
        for offset, chunk in enumerate(chunks):
            position = first_id + offset
            for field, attribute in FILTERABLE_FIELDS.items():
                key = normalise(getattr(chunk.metadata, attribute))
                self._by_field[field].setdefault(key, set()).add(position)

    def rebuild(self, chunks: list[Chunk]) -> None:
        """Discard and rebuild from `chunks`, used after loading from disk."""
        self._by_field = {field: {} for field in FILTERABLE_FIELDS}
        self.add(chunks)

    def add_from_store(self, store, first_id: int = 0, count: int | None = None) -> None:
        """Build from a `ChunkStore`, reading interned codes directly.

        Materialising a `ChunkMetadata` per chunk purely to index it would
        cost more than the store saves.
        """
        self._by_field = {field: {} for field in FILTERABLE_FIELDS}
        total = count if count is not None else len(store)
        for position in range(first_id, first_id + total):
            for field, attribute in FILTERABLE_FIELDS.items():
                key = normalise(store.field(attribute, position))
                self._by_field[field].setdefault(key, set()).add(position)

    def allowed(self, filters: SearchFilter | None) -> set[int] | None:
        """Positions matching `filters`, or None when nothing is constrained.

        Values within one criterion are a union; separate criteria intersect.
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
                candidates |= index.get(normalise(value), set())
            matching = candidates if matching is None else matching & candidates
            if not matching:
                break
        return matching
