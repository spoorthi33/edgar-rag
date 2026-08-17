"""Compact storage for chunk text and metadata.

Holding every chunk as a pydantic `Chunk` measured 7.6 KB each, of which
only about 1.9 KB is the text itself — the rest is Python object overhead,
two model instances and a dozen attribute slots per chunk. At a million
chunks that is 7.6 GB of overhead to store 1.9 GB of text.

Here the text lives on disk and is read only for the handful of chunks a
query actually returns, while metadata lives in parallel arrays with the
repeated values (ticker, company name, form type) interned once rather
than per chunk. Filtering and relevance checks read the arrays directly;
a `Chunk` object is materialised only for retrieved results.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from datetime import date
from pathlib import Path

import numpy as np

from edgar_rag.models import Chunk, ChunkMetadata, FormType

logger = logging.getLogger(__name__)

TEXT_FILENAME = "chunk_text.jsonl"
META_FILENAME = "chunk_meta.npz"


class _Interner:
    """Maps repeated strings to small integer codes."""

    def __init__(self) -> None:
        self.values: list[str] = []
        self._codes: dict[str, int] = {}

    def code(self, value: str | None) -> int:
        key = value or ""
        code = self._codes.get(key)
        if code is None:
            code = len(self.values)
            self._codes[key] = code
            self.values.append(key)
        return code

    def value(self, code: int) -> str:
        return self.values[code]


class ChunkStore:
    """Chunk metadata in arrays, text on disk."""

    def __init__(self) -> None:
        self.chunk_ids: list[str] = []
        self.filing_ids: list[str] = []
        self._offsets: list[int] = []
        self._text_path: Path | None = None

        # One interner per repeated field. A 10,000-filing corpus has a few
        # thousand distinct tickers and company names but a million chunks.
        self._tickers = _Interner()
        self._companies = _Interner()
        self._ciks = _Interner()
        self._items = _Interner()
        self._parts = _Interner()
        self._periods = _Interner()
        self._accessions = _Interner()
        self._forms = _Interner()

        self._ticker_codes: list[int] = []
        self._company_codes: list[int] = []
        self._cik_codes: list[int] = []
        self._item_codes: list[int] = []
        self._part_codes: list[int] = []
        self._period_codes: list[int] = []
        self._accession_codes: list[int] = []
        self._form_codes: list[int] = []
        self._years: list[int] = []
        self._filing_dates: list[str] = []
        self._orders: list[int] = []
        self._token_counts: list[int] = []

        # In-memory text, used before anything is written to disk.
        self._pending_text: list[str] = []

    def __len__(self) -> int:
        return len(self.chunk_ids)

    def add(self, chunks: list[Chunk]) -> None:
        for chunk in chunks:
            meta = chunk.metadata
            self.chunk_ids.append(chunk.chunk_id)
            self.filing_ids.append(chunk.filing_id)
            self._ticker_codes.append(self._tickers.code(meta.ticker))
            self._company_codes.append(self._companies.code(meta.company_name))
            self._cik_codes.append(self._ciks.code(meta.cik))
            self._item_codes.append(self._items.code(meta.item))
            self._part_codes.append(self._parts.code(meta.part))
            self._period_codes.append(self._periods.code(meta.fiscal_period))
            self._accession_codes.append(self._accessions.code(meta.accession_number))
            self._form_codes.append(self._forms.code(meta.form_type.value))
            self._years.append(meta.fiscal_year)
            self._filing_dates.append(meta.filing_date.isoformat())
            self._orders.append(chunk.order)
            self._token_counts.append(chunk.token_count or 0)
            self._pending_text.append(chunk.text)

    # --- Reading -------------------------------------------------------

    def metadata(self, position: int) -> ChunkMetadata:
        """Metadata for one chunk, rebuilt from the interned codes."""
        return ChunkMetadata(
            cik=self._ciks.value(self._cik_codes[position]),
            ticker=self._tickers.value(self._ticker_codes[position]) or None,
            company_name=self._companies.value(self._company_codes[position]),
            form_type=FormType(self._forms.value(self._form_codes[position])),
            fiscal_year=self._years[position],
            fiscal_period=self._periods.value(self._period_codes[position]) or None,
            item=self._items.value(self._item_codes[position]) or None,
            part=self._parts.value(self._part_codes[position]) or None,
            filing_date=date.fromisoformat(self._filing_dates[position]),
            accession_number=self._accessions.value(self._accession_codes[position]),
        )

    def text(self, position: int) -> str:
        """Chunk text, read from disk unless still buffered in memory.

        A store can be both at once: loading an index and appending to it
        leaves the original chunks on disk and the new ones in memory until
        the next save. `_offsets` counts the chunks the file actually holds,
        so it -- not the presence of a file -- decides where a chunk lives.
        """
        saved = len(self._offsets)
        if position >= saved:
            return self._pending_text[position - saved]
        with self._text_path.open("rb") as handle:  # type: ignore[union-attr]
            handle.seek(self._offsets[position])
            return json.loads(handle.readline())

    def chunk(self, position: int) -> Chunk:
        """Materialise a full `Chunk` — done only for retrieved results."""
        return Chunk(
            chunk_id=self.chunk_ids[position],
            filing_id=self.filing_ids[position],
            text=self.text(position),
            metadata=self.metadata(position),
            token_count=self._token_counts[position] or None,
            order=self._orders[position],
        )

    def iter_chunks(self) -> Iterator[Chunk]:
        """Stream every chunk, for full scans such as evaluation labelling.

        Reads the text file sequentially rather than seeking per chunk,
        which matters when the caller wants all of them.
        """
        saved = len(self._offsets)
        if self._text_path is not None:
            with self._text_path.open() as handle:
                for position, line in enumerate(handle):
                    if position >= saved:
                        break
                    yield Chunk(
                        chunk_id=self.chunk_ids[position],
                        filing_id=self.filing_ids[position],
                        text=json.loads(line),
                        metadata=self.metadata(position),
                        token_count=self._token_counts[position] or None,
                        order=self._orders[position],
                    )

        # Anything appended since the last save is still in memory.
        for position in range(saved, len(self)):
            yield self.chunk(position)

    def field(self, name: str, position: int) -> str | int:
        """Interned metadata value, without building a `ChunkMetadata`.

        Used by the metadata filter, which touches every chunk at build
        time and must not pay for object construction.
        """
        match name:
            case "cik":
                return self._ciks.value(self._cik_codes[position])
            case "ticker":
                return self._tickers.value(self._ticker_codes[position])
            case "form_type":
                return self._forms.value(self._form_codes[position])
            case "fiscal_year":
                return self._years[position]
            case "item":
                return self._items.value(self._item_codes[position])
            case "part":
                return self._parts.value(self._part_codes[position])
        raise KeyError(name)

    # --- Persistence ---------------------------------------------------

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        text_path = path / TEXT_FILENAME

        # Written to a temporary file and renamed rather than in place. Once
        # a store has been saved its text lives on disk, so `text()` reads
        # from exactly the file being written -- opening it "wb" truncates
        # the source and every chunk saved so far comes back empty. That is
        # what happens when an existing index is loaded and appended to.
        # The rename also makes the save atomic: an interruption leaves the
        # previous text file intact instead of a half-written one.
        tmp_path = path / (TEXT_FILENAME + ".tmp")
        offsets: list[int] = []
        with tmp_path.open("wb") as handle:
            for position in range(len(self)):
                offsets.append(handle.tell())
                handle.write((json.dumps(self.text(position)) + "\n").encode())
        tmp_path.replace(text_path)

        self._offsets = offsets
        self._text_path = text_path
        self._pending_text = []  # now backed by the file

        np.savez_compressed(
            path / META_FILENAME,
            chunk_ids=np.array(self.chunk_ids, dtype=object),
            filing_ids=np.array(self.filing_ids, dtype=object),
            offsets=np.asarray(offsets, dtype=np.int64),
            ticker_codes=np.asarray(self._ticker_codes, dtype=np.int32),
            company_codes=np.asarray(self._company_codes, dtype=np.int32),
            cik_codes=np.asarray(self._cik_codes, dtype=np.int32),
            item_codes=np.asarray(self._item_codes, dtype=np.int32),
            part_codes=np.asarray(self._part_codes, dtype=np.int32),
            period_codes=np.asarray(self._period_codes, dtype=np.int32),
            accession_codes=np.asarray(self._accession_codes, dtype=np.int32),
            form_codes=np.asarray(self._form_codes, dtype=np.int32),
            years=np.asarray(self._years, dtype=np.int32),
            filing_dates=np.array(self._filing_dates, dtype=object),
            orders=np.asarray(self._orders, dtype=np.int32),
            token_counts=np.asarray(self._token_counts, dtype=np.int32),
            tickers=np.array(self._tickers.values, dtype=object),
            companies=np.array(self._companies.values, dtype=object),
            ciks=np.array(self._ciks.values, dtype=object),
            items=np.array(self._items.values, dtype=object),
            parts=np.array(self._parts.values, dtype=object),
            periods=np.array(self._periods.values, dtype=object),
            accessions=np.array(self._accessions.values, dtype=object),
            forms=np.array(self._forms.values, dtype=object),
        )

    def load(self, path: Path) -> None:
        data = np.load(path / META_FILENAME, allow_pickle=True)

        self.chunk_ids = data["chunk_ids"].tolist()
        self.filing_ids = data["filing_ids"].tolist()
        self._offsets = data["offsets"].tolist()
        self._ticker_codes = data["ticker_codes"].tolist()
        self._company_codes = data["company_codes"].tolist()
        self._cik_codes = data["cik_codes"].tolist()
        self._item_codes = data["item_codes"].tolist()
        self._part_codes = data["part_codes"].tolist()
        self._period_codes = data["period_codes"].tolist()
        self._accession_codes = data["accession_codes"].tolist()
        self._form_codes = data["form_codes"].tolist()
        self._years = data["years"].tolist()
        self._filing_dates = data["filing_dates"].tolist()
        self._orders = data["orders"].tolist()
        self._token_counts = data["token_counts"].tolist()

        for interner, key in (
            (self._tickers, "tickers"),
            (self._companies, "companies"),
            (self._ciks, "ciks"),
            (self._items, "items"),
            (self._parts, "parts"),
            (self._periods, "periods"),
            (self._accessions, "accessions"),
            (self._forms, "forms"),
        ):
            interner.values = data[key].tolist()
            interner._codes = {value: code for code, value in enumerate(interner.values)}

        self._text_path = path / TEXT_FILENAME
        self._pending_text = []
