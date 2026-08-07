"""Compact BM25 over a numpy postings list.

`rank_bm25` keeps a Python dict of term frequencies per document plus the
tokenized corpus, which measured at 15.7 KB per chunk — 7.9 GB at 500k
chunks, and the single largest reason the corpus would not fit in memory.

The same index as CSR-style numpy arrays costs roughly 1 KB per chunk:
term ids and frequencies are int32/float32 in three flat arrays, and the
vocabulary is one dict for the whole corpus rather than one per document.

Scoring is unchanged — this is the standard Okapi BM25 with the same
defaults — so ranking is identical to the previous implementation.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Okapi BM25 defaults, matching rank_bm25's BM25Okapi.
K1 = 1.5
B = 0.75
EPSILON = 0.25


class BM25Postings:
    """Term-frequency postings with BM25 scoring.

    Layout is per-document CSR: `doc_start[i]:doc_start[i+1]` slices
    `term_ids` and `term_freqs` for document `i`. Flat arrays rather than
    per-document containers is the whole point — Python object overhead
    dominated the previous implementation.
    """

    def __init__(self) -> None:
        self.vocabulary: dict[str, int] = {}
        self.term_ids = np.zeros(0, dtype=np.int32)
        self.term_freqs = np.zeros(0, dtype=np.float32)
        self.doc_start = np.zeros(1, dtype=np.int64)
        self.doc_len = np.zeros(0, dtype=np.int32)
        self.idf = np.zeros(0, dtype=np.float32)
        self.avgdl = 0.0

    @property
    def size(self) -> int:
        return len(self.doc_len)

    def build(self, tokenized: Iterable[list[str]]) -> None:
        """Index a tokenized corpus.

        Takes an iterable, not a list, and holds only one document's tokens
        at a time. Materialising the whole corpus first was the actual peak
        allocation — freeing it afterwards does not help, because the peak
        is what exhausts memory and the allocator does not return the arena
        to the OS regardless.
        """
        vocabulary: dict[str, int] = {}
        term_ids: list[int] = []
        term_freqs: list[float] = []
        doc_start = [0]
        doc_len = []

        document_frequency: dict[int, int] = {}

        for tokens in tokenized:
            counts: dict[int, int] = {}
            for token in tokens:
                term_id = vocabulary.get(token)
                if term_id is None:
                    term_id = len(vocabulary)
                    vocabulary[token] = term_id
                counts[term_id] = counts.get(term_id, 0) + 1

            for term_id, count in counts.items():
                term_ids.append(term_id)
                term_freqs.append(count)
                document_frequency[term_id] = document_frequency.get(term_id, 0) + 1

            doc_len.append(len(tokens))
            doc_start.append(len(term_ids))

        if not doc_len:
            return

        self.vocabulary = vocabulary
        self.term_ids = np.asarray(term_ids, dtype=np.int32)
        self.term_freqs = np.asarray(term_freqs, dtype=np.float32)
        self.doc_start = np.asarray(doc_start, dtype=np.int64)
        self.doc_len = np.asarray(doc_len, dtype=np.int32)
        self.avgdl = float(self.doc_len.mean())
        self.idf = self._compute_idf(document_frequency)

        logger.info(
            "built BM25 postings: %d docs, %d terms, %d postings",
            self.size,
            len(vocabulary),
            len(self.term_ids),
        )

    def _compute_idf(self, document_frequency: dict[int, int]) -> np.ndarray:
        """Okapi IDF, matching rank_bm25's BM25Okapi exactly.

        Two details decide whether rankings agree, and both were wrong in
        the first version here:

          - The formula is `log(N - df + 0.5) - log(df + 0.5)`, with no `+1`
            smoothing inside the log. Adding it shifted every score and
            reordered five of eight probe queries.
          - The floor applied to negative IDFs is `epsilon x mean(all idf)`,
            averaged over every term including the negative ones — not over
            the positive ones only.

        A term appearing in more than half the corpus gets a negative raw
        IDF; left unclamped it would subtract from a document's score for
        containing a query term.
        """
        n = float(self.size)
        frequencies = np.zeros(len(self.vocabulary), dtype=np.float64)
        for term_id, freq in document_frequency.items():
            frequencies[term_id] = freq

        idf = np.log(n - frequencies + 0.5) - np.log(frequencies + 0.5)

        floor = EPSILON * float(idf.mean())
        return np.where(idf < 0, floor, idf).astype(np.float32)

    def scores(self, query_tokens: list[str]) -> np.ndarray:
        """BM25 score for every document against `query_tokens`."""
        scores = np.zeros(self.size, dtype=np.float32)
        query_ids = {
            term_id for token in query_tokens if (term_id := self.vocabulary.get(token)) is not None
        }
        if not query_ids:
            return scores

        # One vectorised pass over the postings: mask the entries whose term
        # is in the query, then accumulate per document.
        matching = np.isin(self.term_ids, np.fromiter(query_ids, dtype=np.int32))
        if not matching.any():
            return scores

        positions = np.flatnonzero(matching)
        # Which document each matching posting belongs to.
        doc_ids = np.searchsorted(self.doc_start, positions, side="right") - 1

        freqs = self.term_freqs[positions]
        lengths = self.doc_len[doc_ids].astype(np.float32)
        idf = self.idf[self.term_ids[positions]]

        denominator = freqs + K1 * (1 - B + B * lengths / self.avgdl)
        np.add.at(scores, doc_ids, idf * freqs * (K1 + 1) / denominator)
        return scores

    # --- Persistence ---------------------------------------------------

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path,
            term_ids=self.term_ids,
            term_freqs=self.term_freqs,
            doc_start=self.doc_start,
            doc_len=self.doc_len,
            idf=self.idf,
            avgdl=np.float32(self.avgdl),
            vocabulary_terms=np.array(list(self.vocabulary), dtype=object),
            vocabulary_ids=np.fromiter(self.vocabulary.values(), dtype=np.int32),
        )

    def load(self, path: Path) -> None:
        data = np.load(path, allow_pickle=True)
        self.term_ids = data["term_ids"]
        self.term_freqs = data["term_freqs"]
        self.doc_start = data["doc_start"]
        self.doc_len = data["doc_len"]
        self.idf = data["idf"]
        self.avgdl = float(data["avgdl"])
        self.vocabulary = dict(
            zip(data["vocabulary_terms"], data["vocabulary_ids"].tolist(), strict=True)
        )
