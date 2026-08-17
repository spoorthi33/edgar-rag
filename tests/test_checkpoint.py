"""Phase 9 tests: build checkpoints.

The property under test is that an interrupted build resumes instead of
starting over, and — just as important — that it refuses to resume from a
checkpoint that no longer applies.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from edgar_rag.index.checkpoint import (
    ChunkCheckpoint,
    ChunkFingerprint,
    EmbeddingCheckpoint,
    filings_digest,
)
from edgar_rag.models import Chunk, ChunkMetadata, FormType


def _fingerprint(**overrides) -> ChunkFingerprint:
    defaults = dict(
        filings=filings_digest(["a", "b"]),
        splitter="SemanticSplitter",
        target_tokens=512,
        overlap_tokens=64,
        breakpoint_percentile=95,
        embedding_model="BAAI/bge-small-en-v1.5",
    )
    return ChunkFingerprint(**{**defaults, **overrides})


def _chunks(count: int = 3) -> list[Chunk]:
    return [
        Chunk(
            chunk_id=f"c{i}",
            filing_id="cik/acc",
            text=f"Chunk {i} text about goodwill and revenue.",
            order=i,
            token_count=460,
            metadata=ChunkMetadata(
                cik="0000320193",
                ticker="AAPL",
                company_name="Apple Inc.",
                form_type=FormType.TEN_K,
                fiscal_year=2025,
                item="1A",
                part="I",
                filing_date=date(2025, 10, 31),
                accession_number="acc",
            ),
        )
        for i in range(count)
    ]


# --- Chunk checkpoint ----------------------------------------------------


def test_chunks_survive_a_round_trip(tmp_path) -> None:
    """The whole point: 2.5 hours of chunking is not lost to a stop."""
    checkpoint = ChunkCheckpoint(tmp_path)
    original = _chunks(5)
    checkpoint.save(original, _fingerprint())

    restored = checkpoint.load(_fingerprint())

    assert restored is not None
    assert len(restored) == 5
    assert [c.chunk_id for c in restored] == [c.chunk_id for c in original]
    assert restored[0].text == original[0].text
    assert restored[0].metadata.citation == original[0].metadata.citation


def test_no_checkpoint_returns_none(tmp_path) -> None:
    assert ChunkCheckpoint(tmp_path).load(_fingerprint()) is None


@pytest.mark.parametrize(
    "changed",
    [
        {"splitter": "FixedSplitter"},
        {"target_tokens": 256},
        {"overlap_tokens": 0},
        {"breakpoint_percentile": 80},
        {"embedding_model": "some/other-model"},
        {"filings": filings_digest(["a", "b", "c"])},
    ],
)
def test_checkpoint_is_rejected_when_the_build_changed(tmp_path, changed: dict) -> None:
    """Reusing chunks built under different settings would silently mix
    incompatible data — worse than rebuilding, because nothing looks wrong."""
    checkpoint = ChunkCheckpoint(tmp_path)
    checkpoint.save(_chunks(), _fingerprint())

    assert checkpoint.load(_fingerprint(**changed)) is None


def test_corrupt_manifest_is_treated_as_no_checkpoint(tmp_path) -> None:
    checkpoint = ChunkCheckpoint(tmp_path)
    checkpoint.save(_chunks(), _fingerprint())
    (tmp_path / "chunks.checkpoint.json").write_text("{ truncated")

    assert checkpoint.load(_fingerprint()) is None


def test_clear_removes_the_checkpoint(tmp_path) -> None:
    checkpoint = ChunkCheckpoint(tmp_path)
    checkpoint.save(_chunks(), _fingerprint())
    checkpoint.clear()

    assert not checkpoint.exists
    assert checkpoint.load(_fingerprint()) is None


# --- Partial chunking ----------------------------------------------------


def test_a_partial_checkpoint_is_not_mistaken_for_a_complete_one(tmp_path) -> None:
    """`load` answers "is this the whole job", and a partial one is not.

    Returning partial chunks here would index a fraction of the batch and
    then clear the checkpoint, silently dropping the rest.
    """
    checkpoint = ChunkCheckpoint(tmp_path)
    checkpoint.save(_chunks(3), _fingerprint(), completed=["a"])

    assert checkpoint.load(_fingerprint()) is None


def test_partial_progress_reports_which_filings_are_done(tmp_path) -> None:
    """Chunking saves as it runs, so an interruption costs minutes."""
    checkpoint = ChunkCheckpoint(tmp_path)
    checkpoint.save(_chunks(3), _fingerprint(), completed=["a"])

    loaded = checkpoint.load_partial(_fingerprint())

    assert loaded is not None
    chunks, completed = loaded
    assert len(chunks) == 3
    assert completed == ["a"]


def test_a_completed_checkpoint_reports_no_outstanding_filings(tmp_path) -> None:
    checkpoint = ChunkCheckpoint(tmp_path)
    checkpoint.save(_chunks(3), _fingerprint())

    loaded = checkpoint.load_partial(_fingerprint())

    assert loaded is not None
    assert loaded[1] is None


def test_partial_progress_is_discarded_when_the_batch_changes(tmp_path) -> None:
    """Resuming one batch's partial work into another would mix corpora."""
    checkpoint = ChunkCheckpoint(tmp_path)
    checkpoint.save(_chunks(3), _fingerprint(), completed=["a"])

    other = _fingerprint(filings=filings_digest(["x", "y"]))
    assert checkpoint.load_partial(other) is None


def test_clearing_embeddings_keeps_chunk_progress(tmp_path) -> None:
    """Re-chunking must invalidate vectors without discarding chunk work.

    An embedding checkpoint shorter than the new chunk list is accepted as
    finished work, so its vectors would be paired with different chunks.
    """
    checkpoint = ChunkCheckpoint(tmp_path)
    checkpoint.save(_chunks(3), _fingerprint(), completed=["a"])
    EmbeddingCheckpoint(tmp_path, 3, 4).append(np.ones((2, 4), dtype=np.float32))

    checkpoint.clear_embeddings()

    assert EmbeddingCheckpoint(tmp_path, 3, 4).load() is None
    assert checkpoint.load_partial(_fingerprint()) is not None


# --- Filing digests ------------------------------------------------------


def test_digest_ignores_ordering(tmp_path) -> None:
    assert filings_digest(["b", "a"]) == filings_digest(["a", "b"])


def test_digest_separates_sets_of_the_same_size(tmp_path) -> None:
    """A count could not: this is why the fingerprint stopped using one.

    Two different batches of equal size sharing a fingerprint would let one
    batch's chunks be reused as the other's.
    """
    assert filings_digest(["a", "b"]) != filings_digest(["a", "c"])


def test_digest_is_not_confused_by_delimiters(tmp_path) -> None:
    """Concatenation alone would make ["ab","c"] and ["a","bc"] identical."""
    assert filings_digest(["ab", "c"]) != filings_digest(["a", "bc"])


# --- Embedding checkpoint ------------------------------------------------


def test_partial_embeddings_resume_from_the_right_row(tmp_path) -> None:
    checkpoint = EmbeddingCheckpoint(tmp_path, chunk_count=100, dimension=4)
    partial = np.ones((40, 4), dtype=np.float32)
    checkpoint.append(partial)

    restored = checkpoint.load()

    assert restored is not None
    assert restored.shape == (40, 4)  # a restart continues at row 40


def test_no_embedding_checkpoint_returns_none(tmp_path) -> None:
    assert EmbeddingCheckpoint(tmp_path, 100, 4).load() is None


def test_wrong_dimension_is_rejected(tmp_path) -> None:
    """A flat binary carries no metadata, so the dimension is recorded
    beside it — otherwise vectors from a different model would reshape into
    plausible-looking rows instead of being refused."""
    EmbeddingCheckpoint(tmp_path, 100, 8).append(np.ones((40, 8), dtype=np.float32))

    assert EmbeddingCheckpoint(tmp_path, 100, 4).load() is None


def test_missing_metadata_is_treated_as_no_checkpoint(tmp_path) -> None:
    checkpoint = EmbeddingCheckpoint(tmp_path, 100, 4)
    checkpoint.append(np.ones((10, 4), dtype=np.float32))
    checkpoint.meta_file.unlink()

    assert checkpoint.load() is None


def test_checkpoint_longer_than_the_corpus_is_rejected(tmp_path) -> None:
    """More vectors than chunks means they belong to a different run."""
    EmbeddingCheckpoint(tmp_path, 100, 4).append(np.ones((150, 4), dtype=np.float32))

    assert EmbeddingCheckpoint(tmp_path, 100, 4).load() is None


def test_a_partial_row_from_a_killed_process_is_dropped(tmp_path) -> None:
    """A process killed mid-write leaves an incomplete vector; reshaping it
    would misalign every row after it."""
    checkpoint = EmbeddingCheckpoint(tmp_path, 100, 4)
    checkpoint.append(np.ones((3, 4), dtype=np.float32))
    with checkpoint.file.open("ab") as handle:  # half a row
        handle.write(np.ones(2, dtype=np.float32).tobytes())

    restored = checkpoint.load()
    assert restored is not None
    assert restored.shape == (3, 4)


def test_batches_accumulate_across_appends(tmp_path) -> None:
    """The resumed run continues the file rather than replacing it."""
    checkpoint = EmbeddingCheckpoint(tmp_path, 100, 4)
    checkpoint.append(np.zeros((10, 4), dtype=np.float32))
    checkpoint.append(np.ones((10, 4), dtype=np.float32))

    restored = checkpoint.load()
    assert restored.shape == (20, 4)
    assert restored[0].sum() == 0 and restored[10].sum() == 4


def test_reset_discards_the_checkpoint(tmp_path) -> None:
    checkpoint = EmbeddingCheckpoint(tmp_path, 100, 4)
    checkpoint.append(np.ones((5, 4), dtype=np.float32))
    checkpoint.reset()

    assert checkpoint.load() is None
