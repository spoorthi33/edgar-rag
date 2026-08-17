"""Chunk assembly.

Two strategies share one packing routine:

  - `FixedSplitter` fills chunks to a token budget, breaking only between
    sentences. Deterministic and free.
  - `SemanticSplitter` additionally prefers boundaries where the topic
    changes, measured as a drop in similarity between neighbouring
    sentences, so a chunk is less likely to straddle two subjects.

Both emit overlapping chunks: a fact sitting near a boundary would
otherwise be split across two chunks and retrieved well by neither.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np

from edgar_rag.chunking.sentences import split_sentences
from edgar_rag.embeddings.base import Embedder

# Rough fallback for when no model is available. Only safe for tests and the
# fixed splitter: measured against a real tokenizer on filing text it runs
# between 0.64x and 1.81x the true count, because dollar figures, percentages
# and tickers fragment into many more tokens than prose. Chunks sized by this
# estimate overran a 512-token model on a fifth of the corpus.
CHARS_PER_TOKEN = 4

TokenCounter = Callable[[str], int]


def estimate_tokens(text: str) -> int:
    return max(1, round(len(text) / CHARS_PER_TOKEN))


class Splitter(ABC):
    """Turns one section's text into chunk-sized pieces."""

    #: Set by subclasses; the model's tokenizer where one is available.
    _count_tokens: TokenCounter = staticmethod(estimate_tokens)

    @abstractmethod
    def split(self, text: str) -> list[str]:
        """Return the chunk texts for `text`, in document order."""

    def split_many(self, texts: list[str]) -> list[list[str]]:
        """Split several texts at once.

        The default just loops; `SemanticSplitter` overrides it to embed
        every text's sentences in one batch, which is the difference
        between a few dozen sentences per model call and several thousand.
        """
        return [self.split(text) for text in texts]

    def count_tokens(self, text: str) -> int:
        """Token count under whichever counter this splitter was given."""
        return self._count_tokens(text)


class FixedSplitter(Splitter):
    """Packs sentences up to a token budget, with sentence-level overlap."""

    def __init__(
        self,
        target_tokens: int = 512,
        overlap_tokens: int = 64,
        token_counter: TokenCounter | None = None,
    ) -> None:
        if overlap_tokens >= target_tokens:
            raise ValueError("overlap_tokens must be smaller than target_tokens")
        self.target_tokens = target_tokens
        self.overlap_tokens = overlap_tokens
        self._count_tokens = token_counter or estimate_tokens

    def split(self, text: str) -> list[str]:
        sentences = split_sentences(text)
        if not sentences:
            return []
        return _pack(
            sentences,
            self.target_tokens,
            self.overlap_tokens,
            breakpoints=set(),
            count_tokens=self._count_tokens,
        )


class SemanticSplitter(Splitter):
    """Packs sentences, preferring boundaries where the topic shifts.

    Consecutive sentences are embedded and compared; the least similar
    junctions become preferred break points. Packing still respects the
    token budget, so a long stretch on one topic is broken anyway.
    """

    def __init__(
        self,
        embedder: Embedder,
        target_tokens: int | None = None,
        overlap_tokens: int = 64,
        breakpoint_percentile: int = 95,
        min_sentences: int = 4,
    ) -> None:
        # Default to what the model actually accepts, so chunks are never
        # truncated at embedding time.
        target_tokens = target_tokens or embedder.max_tokens
        if overlap_tokens >= target_tokens:
            raise ValueError("overlap_tokens must be smaller than target_tokens")
        if target_tokens > embedder.max_tokens:
            raise ValueError(
                f"target_tokens={target_tokens} exceeds the model limit of "
                f"{embedder.max_tokens}; the overflow would be silently discarded"
            )
        self.embedder = embedder
        self.target_tokens = target_tokens
        self.overlap_tokens = overlap_tokens
        self.breakpoint_percentile = breakpoint_percentile
        self.min_sentences = min_sentences
        # Counted by the model's own tokenizer, not estimated from characters.
        self._count_tokens = embedder.count_tokens

    def split(self, text: str) -> list[str]:
        sentences = split_sentences(text)
        if not sentences:
            return []
        breakpoints = (
            self._find_breakpoints(sentences) if len(sentences) >= self.min_sentences else set()
        )
        return _pack(
            sentences,
            self.target_tokens,
            self.overlap_tokens,
            breakpoints,
            count_tokens=self._count_tokens,
        )

    def split_many(self, texts: list[str]) -> list[list[str]]:
        """Split several texts, embedding all their sentences in one call.

        Embedding per text was the dominant cost of a large build: a
        section yields a few dozen sentences, so the model was invoked
        hundreds of thousands of times at a batch size far below what the
        hardware wants, and per-call overhead swamped the actual work.
        Batching across texts leaves the arithmetic identical — each text
        still gets breakpoints from its own sentences — while turning
        thousands of tiny calls into a handful of large ones.
        """
        per_text = [split_sentences(text) for text in texts]

        # One flat batch, remembering where each text's sentences begin.
        flat: list[str] = []
        spans: list[tuple[int, int]] = []
        for sentences in per_text:
            start = len(flat)
            if len(sentences) >= self.min_sentences:
                flat.extend(sentences)
            spans.append((start, len(flat)))

        vectors = self.embedder.embed_documents(flat) if flat else None

        results: list[list[str]] = []
        for sentences, (start, end) in zip(per_text, spans, strict=True):
            if not sentences:
                results.append([])
                continue
            breakpoints = self._breakpoints_from(vectors[start:end]) if end > start else set()
            results.append(
                _pack(
                    sentences,
                    self.target_tokens,
                    self.overlap_tokens,
                    breakpoints,
                    count_tokens=self._count_tokens,
                )
            )
        return results

    def _find_breakpoints(self, sentences: list[str]) -> set[int]:
        """Indices after which the topic changes most sharply."""
        return self._breakpoints_from(self.embedder.embed_documents(sentences))

    def _breakpoints_from(self, vectors: np.ndarray) -> set[int]:
        """Breakpoints from already-computed sentence vectors."""
        if len(vectors) < 2:
            return set()
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        # A degenerate sentence can embed to the zero vector; avoid dividing
        # by zero rather than propagating NaNs into the percentile.
        unit = vectors / np.where(norms == 0, 1, norms)
        similarities = np.sum(unit[:-1] * unit[1:], axis=1)

        threshold = np.percentile(similarities, 100 - self.breakpoint_percentile)
        return {i for i, score in enumerate(similarities) if score <= threshold}


def _pack(
    sentences: list[str],
    target_tokens: int,
    overlap_tokens: int,
    breakpoints: set[int],
    count_tokens: TokenCounter,
) -> list[str]:
    """Group sentences into chunks under the token budget.

    A chunk closes when adding the next sentence would exceed the budget, or
    at a preferred breakpoint once the chunk is at least half full — the
    half-full guard stops a run of topic shifts from producing chunks too
    small to carry context.

    The budget is checked against the joined text, not a running total:
    tokenizing sentences separately counts each one's special tokens, and
    that difference is what pushes a chunk over a hard model limit.

    Measuring the joined text on every sentence is quadratic, though — a
    40-sentence chunk re-tokenized 40 progressively longer strings, and
    that alone was 42% of a full build. Summing the per-sentence counts is
    an upper bound on the joined count (verified on the corpus: the joined
    text is always at least 2 tokens shorter, being one pair of special
    tokens rather than many), so while the bound stays under budget the
    chunk provably does too and the exact count can be skipped. It is only
    consulted once the bound crosses the limit, near a chunk boundary. Both
    the decision and the resulting chunks are unchanged.
    """
    chunks: list[str] = []
    current: list[str] = []
    sizes: list[int] = []
    bound = 0
    half_budget = target_tokens // 2

    for index, sentence in enumerate(sentences):
        size = count_tokens(sentence)

        if (
            current
            and bound + size > target_tokens
            and count_tokens(" ".join([*current, sentence])) > target_tokens
        ):
            chunks.append(" ".join(current))
            current, sizes = _carry_overlap(current, sizes, overlap_tokens, count_tokens)
            bound = sum(sizes)

        current.append(sentence)
        sizes.append(size)
        bound += size

        # The bound can only prove the chunk is *under* half budget, which
        # is the case that skips the call; reaching it still needs the
        # exact count.
        if (
            index in breakpoints
            and bound >= half_budget
            and count_tokens(" ".join(current)) >= half_budget
        ):
            chunks.append(" ".join(current))
            current, sizes = _carry_overlap(current, sizes, overlap_tokens, count_tokens)
            bound = sum(sizes)

    if current:
        chunks.append(" ".join(current))
    return chunks


def _carry_overlap(
    sentences: list[str],
    sizes: list[int],
    overlap_tokens: int,
    count_tokens: TokenCounter,
) -> tuple[list[str], list[int]]:
    """Take trailing sentences from a closed chunk to start the next one.

    Returns the carried sentences with their token counts, so the caller
    does not re-count what has already been measured.
    """
    if overlap_tokens <= 0:
        return [], []

    carried: list[str] = []
    carried_sizes: list[int] = []
    bound = 0
    for sentence, size in zip(reversed(sentences), reversed(sizes), strict=True):
        if (
            carried
            and bound + size > overlap_tokens
            and count_tokens(" ".join([sentence, *carried])) > overlap_tokens
        ):
            break
        carried.insert(0, sentence)
        carried_sizes.insert(0, size)
        bound += size
    return carried, carried_sizes
