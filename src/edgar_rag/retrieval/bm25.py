"""BM25 sparse retrieval.

Dense embeddings match meaning but blur exact strings, and filings turn on
exact strings: a ticker, "Item 7A", a specific dollar figure. Asked "how
much did the company spend on research and development", dense search
returned segment tables that merely looked like financial statements and
ranked the passage actually containing "Research and Development" third.
BM25 scores literal term overlap and recovers precisely those cases.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from edgar_rag.index.chunk_store import ChunkStore
from edgar_rag.index.metadata import MetadataFilterIndex
from edgar_rag.models import Chunk, RetrievedChunk, SearchFilter
from edgar_rag.retrieval.postings import BM25Postings

logger = logging.getLogger(__name__)

POSTINGS_FILENAME = "bm25_postings.npz"

# Tokens worth keeping from filing text: words, and figures with their
# punctuation intact so "$383.3" and "8.1%" survive as single terms rather
# than fragmenting into digits that match everything.
TOKEN_PATTERN = re.compile(r"[a-z]+(?:'[a-z]+)?|\$?\d[\d,]*(?:\.\d+)?%?")

# Function words plus the interrogatives that natural-language questions are
# built from. Both matter here: users ask "how much did the company spend on
# research and development", and without this the query's content terms are
# outweighed by its grammar.
#
# "company" is included deliberately. It looks like a content word, but each
# filer refers to itself consistently — Apple writes "the Company" where
# others write "we" — so it behaves as a filer fingerprint and pulls in that
# company's chunks whatever the topic. It carries no relevance signal in a
# corpus made entirely of company filings.
_STOPWORD_TEXT = """
    a about above after again against all am an and any are as at
    be because been before being below between both but by
    can could did do does doing down during
    each few for from further
    had has have having he her here hers him his how however
    i if in into is it its itself
    just me more most much must my
    no nor not now of off on once only or other others otherwise our ours out
    over own
    same shall she should so some such than that the their theirs them then
    there these they this those through to too
    under until up us very was we were what when where whether which while who
    whom why will with would you your yours
    company companies
"""

STOPWORDS = frozenset(_STOPWORD_TEXT.split())


def tokenize(text: str) -> list[str]:
    """Lower-case, keep figures whole, drop stopwords."""
    return [token for token in TOKEN_PATTERN.findall(text.lower()) if token not in STOPWORDS]


class BM25Retriever:
    """Lexical search over chunk text, with the same metadata filtering."""

    def __init__(self, chunks: list[Chunk] | None = None) -> None:
        self._chunks: list[Chunk] = []
        self._store: ChunkStore | None = None
        self._filters = MetadataFilterIndex()
        self._bm25: BM25Postings | None = None
        self._tokenized: list[list[str]] = []
        self._stale = False
        if chunks:
            self.add(chunks)

    # --- Persistence ---------------------------------------------------

    def save(self, path: Path) -> None:
        """Write the built postings next to the vector index.

        Every query process — each CLI call, and every service start —
        needs term statistics before its first search. Saving the postings
        rather than the tokenized corpus means loading is a few array reads
        instead of re-tokenizing and re-counting 231k documents, which
        measured 10.5s.
        """
        if self._bm25 is None:
            self.finalize()
        if self._bm25 is None:
            return
        path.mkdir(parents=True, exist_ok=True)
        self._bm25.save(path / POSTINGS_FILENAME)

    def load(self, path: Path, store: ChunkStore) -> None:
        """Restore from `path`, pairing postings with the index's chunk store."""
        self._store = store
        self._filters.add_from_store(store)

        postings_file = path / POSTINGS_FILENAME
        if postings_file.is_file():
            postings = BM25Postings()
            postings.load(postings_file)
            if postings.size == len(store):
                self._bm25 = postings
                self._stale = False
                return
            # Saved postings that cover a different number of documents
            # would score the wrong chunks: the row index is the chunk id.
            logger.warning(
                "saved BM25 postings cover %d chunks but the index has %d; rebuilding",
                postings.size,
                len(store),
            )
        else:
            logger.info("no saved BM25 postings at %s; building from the store", path)

        self._rebuild_from(store)

    def _rebuild_from(self, store: ChunkStore) -> None:
        """Tokenize the corpus and build postings, streaming a chunk at a time."""
        # Streamed a chunk at a time: the tokenized corpus is the largest
        # single allocation in this class and never needs to exist whole.
        postings = BM25Postings()
        postings.build(tokenize(chunk.text) for chunk in store.iter_chunks())
        self._bm25 = postings
        self._stale = False

    def add(self, chunks: list[Chunk], defer: bool = False) -> None:
        """Index `chunks`.

        BM25 scores against corpus-wide document frequencies, so adding
        documents invalidates the statistics and they must be recomputed
        rather than appended to. That makes a loop of small adds quadratic
        in corpus size. Pass `defer=True` while loading many batches and
        call `finalize()` once at the end.
        """
        if not chunks:
            return
        first_id = len(self._chunks)
        self._chunks.extend(chunks)
        self._filters.add(chunks, first_id)
        self._tokenized.extend(tokenize(chunk.text) for chunk in chunks)
        self._stale = True
        if not defer:
            self.finalize()

    def finalize(self, release_tokens: bool = False) -> None:
        """Recompute term statistics after deferred adds.

        `release_tokens` frees the tokenized corpus, but only the caller
        knows whether more chunks are coming: BM25 statistics are
        corpus-wide, so a later `add()` rebuilds from every token, and
        dropping them mid-build would score the new chunks against an empty
        corpus. The load path never populates them in the first place.
        """
        if self._stale and self._tokenized:
            postings = BM25Postings()
            postings.build(self._tokenized)
            self._bm25 = postings
            self._stale = False
        if release_tokens:
            self._tokenized = []

    def search(
        self,
        query: str,
        top_k: int,
        filters: SearchFilter | None = None,
    ) -> list[RetrievedChunk]:
        """Return the `top_k` best lexical matches for `query`."""
        if self._bm25 is None or not self.size:
            return []

        tokens = tokenize(query)
        if not tokens:
            return []

        allowed = self._filters.allowed(filters)
        if allowed is not None and not allowed:
            return []

        scores = self._bm25.scores(tokens)
        candidates = allowed if allowed is not None else range(self.size)

        # A zero score means no query term appears; such a chunk is not a
        # match at all and would otherwise pad the results with noise that
        # then enters the fusion.
        ranked = sorted(
            (position for position in candidates if scores[position] > 0),
            key=lambda position: scores[position],
            reverse=True,
        )[:top_k]

        return [
            RetrievedChunk(
                chunk=self._chunk_at(position),
                score=float(scores[position]),
                sparse_rank=rank + 1,
            )
            for rank, position in enumerate(ranked)
        ]

    @property
    def size(self) -> int:
        return len(self._store) if self._store is not None else len(self._chunks)

    def _chunk_at(self, position: int) -> Chunk:
        """One chunk, from the store when loaded and the list when built."""
        if self._store is not None:
            return self._store.chunk(position)
        return self._chunks[position]
