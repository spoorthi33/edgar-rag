"""BM25 sparse retrieval.

Dense embeddings match meaning but blur exact strings, and filings turn on
exact strings: a ticker, "Item 7A", a specific dollar figure. Asked "how
much did the company spend on research and development", dense search
returned segment tables that merely looked like financial statements and
ranked the passage actually containing "Research and Development" third.
BM25 scores literal term overlap and recovers precisely those cases.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from edgar_rag.index.metadata import MetadataFilterIndex
from edgar_rag.models import Chunk, RetrievedChunk, SearchFilter

logger = logging.getLogger(__name__)

TOKENS_FILENAME = "bm25_tokens.jsonl"

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
        self._filters = MetadataFilterIndex()
        self._bm25: BM25Okapi | None = None
        self._tokenized: list[list[str]] = []
        self._stale = False
        if chunks:
            self.add(chunks)

    # --- Persistence ---------------------------------------------------

    def save(self, path: Path) -> None:
        """Write the tokenized corpus next to the vector index.

        Every query process — each CLI call, and every service start in
        Phase 7 — must have the term statistics before its first search.
        Reading saved tokens measured 0.089s against 0.190s to tokenize from
        scratch at 2,105 chunks: a little over twice as fast, and both scale
        linearly, so the gap is tens of seconds at Phase 9's corpus size.
        """
        path.mkdir(parents=True, exist_ok=True)
        with (path / TOKENS_FILENAME).open("w") as handle:
            for tokens in self._tokenized:
                handle.write(json.dumps(tokens) + "\n")

    def load(self, path: Path, chunks: list[Chunk]) -> None:
        """Restore from `path`, pairing tokens with the index's chunks."""
        tokens_file = path / TOKENS_FILENAME
        if not tokens_file.is_file():
            logger.info("no saved BM25 tokens at %s; tokenizing from scratch", path)
            self.add(chunks)
            return

        with tokens_file.open() as handle:
            tokenized = [json.loads(line) for line in handle if line.strip()]

        if len(tokenized) != len(chunks):
            # The corpus changed since the tokens were written; trusting them
            # would pair scores with the wrong chunks.
            logger.warning(
                "saved BM25 tokens cover %d chunks but the index has %d; rebuilding",
                len(tokenized),
                len(chunks),
            )
            self.add(chunks)
            return

        self._chunks = list(chunks)
        self._tokenized = tokenized
        self._filters.rebuild(self._chunks)
        self._bm25 = BM25Okapi(self._tokenized)
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

    def finalize(self) -> None:
        """Recompute term statistics after deferred adds."""
        if self._stale and self._tokenized:
            self._bm25 = BM25Okapi(self._tokenized)
            self._stale = False

    def search(
        self,
        query: str,
        top_k: int,
        filters: SearchFilter | None = None,
    ) -> list[RetrievedChunk]:
        """Return the `top_k` best lexical matches for `query`."""
        if self._bm25 is None or not self._chunks:
            return []

        tokens = tokenize(query)
        if not tokens:
            return []

        allowed = self._filters.allowed(filters)
        if allowed is not None and not allowed:
            return []

        scores = self._bm25.get_scores(tokens)
        candidates = allowed if allowed is not None else range(len(self._chunks))

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
                chunk=self._chunks[position],
                score=float(scores[position]),
                sparse_rank=rank + 1,
            )
            for rank, position in enumerate(ranked)
        ]

    @property
    def size(self) -> int:
        return len(self._chunks)
