"""Phase checkpoints for the index build.

Building the index is parse, then chunk, then embed. On a large corpus
each stage takes hours, and without checkpoints a stop at any point
discards everything: a laptop that overheats 2.5 hours into chunking
starts again from the beginning.

Each stage writes its output here and skips itself when a valid
checkpoint already exists, so a rebuild resumes at the last completed
stage rather than the start.

A checkpoint records what produced it. Reusing chunks built by a
different chunker or embedding model would silently mix incompatible
data — worse than rebuilding, because nothing would look wrong.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from edgar_rag.models import Chunk

logger = logging.getLogger(__name__)

CHUNKS_FILENAME = "chunks.checkpoint.jsonl"
MANIFEST_FILENAME = "chunks.checkpoint.json"
EMBEDDINGS_FILENAME = "embeddings.checkpoint.f32"


@dataclass(frozen=True)
class ChunkFingerprint:
    """The settings a set of chunks depends on.

    A checkpoint is only reusable when every one of these matches: change
    the splitter or the chunk budget and the chunks are different objects
    that happen to share a filename.
    """

    filing_count: int
    splitter: str
    target_tokens: int
    overlap_tokens: int
    breakpoint_percentile: int
    embedding_model: str

    def matches(self, other: ChunkFingerprint) -> tuple[bool, str]:
        """Whether `other` may be reused, and why not when it may not."""
        for field, mine in asdict(self).items():
            theirs = asdict(other)[field]
            if mine != theirs:
                return False, f"{field} changed ({theirs} -> {mine})"
        return True, ""


class ChunkCheckpoint:
    """Chunks written to disk between the chunking and embedding stages."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    @property
    def exists(self) -> bool:
        return (self.path / MANIFEST_FILENAME).is_file() and (self.path / CHUNKS_FILENAME).is_file()

    def save(self, chunks: list[Chunk], fingerprint: ChunkFingerprint) -> None:
        self.path.mkdir(parents=True, exist_ok=True)

        # Written to a temporary file and renamed, so an interrupted save
        # never leaves a partial checkpoint that looks complete.
        tmp = self.path / (CHUNKS_FILENAME + ".tmp")
        with tmp.open("w") as handle:
            for chunk in chunks:
                handle.write(chunk.model_dump_json() + "\n")
        tmp.replace(self.path / CHUNKS_FILENAME)

        (self.path / MANIFEST_FILENAME).write_text(json.dumps(asdict(fingerprint), indent=2))
        logger.info("checkpointed %d chunks to %s", len(chunks), self.path)

    def load(self, fingerprint: ChunkFingerprint) -> list[Chunk] | None:
        """Reusable chunks, or None when the checkpoint does not apply."""
        if not self.exists:
            return None

        try:
            saved = ChunkFingerprint(**json.loads((self.path / MANIFEST_FILENAME).read_text()))
        except (OSError, TypeError, ValueError) as exc:
            logger.warning("unreadable chunk checkpoint (%s); rebuilding", exc)
            return None

        reusable, reason = fingerprint.matches(saved)
        if not reusable:
            logger.info("chunk checkpoint does not apply: %s; rebuilding", reason)
            return None

        with (self.path / CHUNKS_FILENAME).open() as handle:
            chunks = [Chunk.model_validate_json(line) for line in handle if line.strip()]

        logger.info("resuming from %d checkpointed chunks", len(chunks))
        return chunks

    def clear(self) -> None:
        for name in (
            CHUNKS_FILENAME,
            MANIFEST_FILENAME,
            EMBEDDINGS_FILENAME,
            EMBEDDINGS_FILENAME + ".json",
        ):
            (self.path / name).unlink(missing_ok=True)


class EmbeddingCheckpoint:
    """Partially completed embeddings, appended as the batches finish.

    Embedding is the one stage that can checkpoint *within* itself: the
    vectors are a plain array, so however many rows are done can be written
    out and a restart continues from that row.

    Batches are **appended** to a flat float32 file rather than the whole
    array being rewritten each time. Rewriting made checkpointing quadratic:
    on a 231k-chunk corpus every save would eventually rewrite 340 MB —
    about 39 GB of writes across the run — and it measured 17-24 chunks/s
    against the 59 chunks/s the same model and batch size reach with no
    checkpointing at all. Appending writes each vector exactly once.
    """

    def __init__(self, path: Path, chunk_count: int, dimension: int) -> None:
        self.file = Path(path) / EMBEDDINGS_FILENAME
        # A flat binary carries no metadata of its own, so the dimension is
        # recorded beside it. Without this, vectors from a different model
        # would be reshaped into plausible-looking rows instead of rejected.
        self.meta_file = Path(path) / (EMBEDDINGS_FILENAME + ".json")
        self.chunk_count = chunk_count
        self.dimension = dimension

    def load(self) -> np.ndarray | None:
        """Vectors completed so far, or None when there are none to reuse."""
        if not self.file.is_file():
            return None

        try:
            saved_dimension = json.loads(self.meta_file.read_text())["dimension"]
        except (OSError, KeyError, ValueError):
            logger.warning("embedding checkpoint has no readable metadata; restarting")
            return None

        if saved_dimension != self.dimension:
            logger.warning(
                "embedding checkpoint is %d-dimensional but %d was configured; restarting",
                saved_dimension,
                self.dimension,
            )
            return None

        try:
            flat = np.fromfile(self.file, dtype=np.float32)
        except OSError as exc:
            logger.warning("unreadable embedding checkpoint (%s); restarting", exc)
            return None

        # A trailing partial row means the process died mid-write; drop it
        # rather than reshaping into misaligned vectors.
        rows = len(flat) // self.dimension
        if rows == 0:
            return None
        vectors = flat[: rows * self.dimension].reshape(rows, self.dimension)
        if len(vectors) > self.chunk_count:
            # More vectors than chunks means they belong to a different run.
            logger.warning("embedding checkpoint is longer than the corpus; restarting")
            return None

        logger.info("resuming embedding at %d of %d chunks", len(vectors), self.chunk_count)
        return vectors

    def append(self, vectors: np.ndarray) -> None:
        """Add one batch of vectors to the end of the checkpoint."""
        self.file.parent.mkdir(parents=True, exist_ok=True)
        if not self.meta_file.is_file():
            self.meta_file.write_text(json.dumps({"dimension": self.dimension}))
        with self.file.open("ab") as handle:
            handle.write(np.ascontiguousarray(vectors, dtype=np.float32).tobytes())

    def reset(self) -> None:
        """Discard the checkpoint, for a run starting from scratch."""
        self.file.unlink(missing_ok=True)
        self.meta_file.unlink(missing_ok=True)
