"""Phase 4 tests.

Vectors are constructed by hand so similarity ordering is known in advance;
no model is loaded.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from edgar_rag.index.faiss_index import FaissIndex
from edgar_rag.models import Chunk, ChunkMetadata, FormType, SearchFilter

DIM = 4


def _unit(*values: float) -> np.ndarray:
    vector = np.array(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


def _chunk(
    chunk_id: str,
    *,
    cik: str = "0000320193",
    ticker: str = "AAPL",
    year: int = 2025,
    item: str = "1A",
    part: str | None = "I",
    form: FormType = FormType.TEN_K,
    text: str = "chunk text",
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        filing_id=f"{cik}/acc-{chunk_id}",
        text=text,
        order=0,
        metadata=ChunkMetadata(
            cik=cik,
            ticker=ticker,
            company_name="Apple Inc.",
            form_type=form,
            fiscal_year=year,
            item=item,
            part=part,
            filing_date=date(2025, 10, 31),
            accession_number=f"acc-{chunk_id}",
        ),
    )


@pytest.fixture
def index() -> FaissIndex:
    """Four chunks pointing along distinct axes, so ranking is predictable."""
    index = FaissIndex(dimension=DIM, model_name="stub-model")
    chunks = [
        _chunk("a", ticker="AAPL", year=2025, item="1A"),
        _chunk("b", ticker="MSFT", cik="0000789019", year=2025, item="1A"),
        _chunk("c", ticker="AAPL", year=2024, item="7"),
        _chunk("d", ticker="MSFT", cik="0000789019", year=2024, item="7", part="II"),
    ]
    vectors = np.stack([_unit(1, 0, 0, 0), _unit(0, 1, 0, 0), _unit(0, 0, 1, 0), _unit(0, 0, 0, 1)])
    index.add(chunks, vectors)
    return index


# --- Building ------------------------------------------------------------


def test_size_reflects_added_chunks(index: FaissIndex) -> None:
    assert index.size == 4


def test_mismatched_lengths_are_rejected() -> None:
    idx = FaissIndex(dimension=DIM)
    with pytest.raises(ValueError, match="2 chunks but 1 vectors"):
        idx.add([_chunk("a"), _chunk("b")], np.zeros((1, DIM), dtype=np.float32))


def test_wrong_dimension_is_rejected() -> None:
    idx = FaissIndex(dimension=DIM)
    with pytest.raises(ValueError, match="dimensional"):
        idx.add([_chunk("a")], np.zeros((1, DIM + 1), dtype=np.float32))


def test_adding_nothing_is_a_noop() -> None:
    idx = FaissIndex(dimension=DIM)
    idx.add([], np.zeros((0, DIM), dtype=np.float32))
    assert idx.size == 0


def test_ivf_falls_back_to_flat_on_a_small_corpus() -> None:
    """IVF cannot train on a handful of vectors."""
    idx = FaissIndex(dimension=DIM, index_type="ivf")
    idx.add([_chunk("a")], _unit(1, 0, 0, 0).reshape(1, DIM))
    assert idx.index_type == "flat"


def _random_corpus(count: int, dim: int) -> tuple[list[Chunk], np.ndarray]:
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(count, dim)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return [_chunk(f"c{i}") for i in range(count)], vectors


def test_ivf_trains_on_a_large_enough_corpus() -> None:
    chunks, vectors = _random_corpus(3000, 32)
    idx = FaissIndex(dimension=32, index_type="ivf", nlist=64)
    idx.add(chunks, vectors)

    assert idx.index_type == "ivf"
    assert idx.size == 3000


def test_ivf_approximates_exact_search_closely() -> None:
    """IVF trades some recall for speed; flat stays the default until scale."""
    chunks, vectors = _random_corpus(3000, 32)
    exact = FaissIndex(dimension=32, index_type="flat")
    exact.add(chunks, vectors)
    approx = FaissIndex(dimension=32, index_type="ivf", nlist=64)
    approx.add(chunks, vectors)

    overlap = 0
    for query in vectors[:50]:
        found = {r.chunk.chunk_id for r in exact.search(query, 5)}
        guessed = {r.chunk.chunk_id for r in approx.search(query, 5)}
        overlap += len(found & guessed)

    assert overlap / 250 > 0.7


def test_small_first_batch_does_not_latch_the_index_to_flat() -> None:
    """Incremental ingestion must still reach IVF once there are enough vectors."""
    chunks, vectors = _random_corpus(3000, 32)
    idx = FaissIndex(dimension=32, index_type="ivf", nlist=64)

    idx.add(chunks[:100], vectors[:100])  # too few to train
    assert idx.index_type == "flat"

    idx.add(chunks[100:], vectors[100:])
    assert idx.index_type == "ivf"
    assert idx.size == 3000
    assert idx.search(vectors[0], 1)  # still searchable after the rebuild


def test_filtering_cost_does_not_scale_with_the_corpus() -> None:
    """The filter must not scan every chunk: at 500k it would dominate latency."""
    chunks, vectors = _random_corpus(20000, 8)
    for i, chunk in enumerate(chunks):
        chunk.metadata.ticker = "AAPL" if i == 0 else "MSFT"
    idx = FaissIndex(dimension=8)
    idx.add(chunks, vectors)

    ids = idx._allowed_ids(SearchFilter(tickers=["AAPL"]))
    assert len(ids) == 1  # found by lookup, not by inspecting 20k chunks


def test_ivf_supports_metadata_filtering() -> None:
    chunks, vectors = _random_corpus(3000, 32)
    idx = FaissIndex(dimension=32, index_type="ivf", nlist=64)
    idx.add(chunks, vectors)

    assert len(idx.search(vectors[0], 5, SearchFilter(tickers=["AAPL"]))) == 5
    assert idx.search(vectors[0], 5, SearchFilter(tickers=["TSLA"])) == []


# --- Searching -----------------------------------------------------------


def test_returns_the_nearest_chunk_first(index: FaissIndex) -> None:
    results = index.search(_unit(1, 0, 0, 0), top_k=2)
    assert results[0].chunk.chunk_id == "a"
    assert results[0].score > results[1].score


def test_scores_are_cosine_similarity(index: FaissIndex) -> None:
    """Vectors are normalised, so inner product is cosine."""
    results = index.search(_unit(1, 0, 0, 0), top_k=1)
    assert results[0].score == pytest.approx(1.0, abs=1e-5)


def test_dense_rank_is_recorded(index: FaissIndex) -> None:
    results = index.search(_unit(1, 1, 0, 0), top_k=3)
    assert [r.dense_rank for r in results] == [1, 2, 3]


def test_top_k_larger_than_the_corpus_is_safe(index: FaissIndex) -> None:
    assert len(index.search(_unit(1, 0, 0, 0), top_k=99)) == 4


def test_searching_an_empty_index_returns_nothing() -> None:
    assert FaissIndex(dimension=DIM).search(_unit(1, 0, 0, 0), top_k=5) == []


# --- Metadata filtering --------------------------------------------------


def test_ticker_filter_excludes_other_companies(index: FaissIndex) -> None:
    """The core defence: a perfect match in the wrong filing must not return."""
    results = index.search(_unit(0, 1, 0, 0), top_k=5, filters=SearchFilter(tickers=["AAPL"]))
    assert {r.chunk.metadata.ticker for r in results} == {"AAPL"}


def test_ticker_filter_is_case_insensitive(index: FaissIndex) -> None:
    results = index.search(_unit(1, 0, 0, 0), top_k=5, filters=SearchFilter(tickers=["aapl"]))
    assert results


def test_fiscal_year_filter(index: FaissIndex) -> None:
    results = index.search(_unit(1, 1, 1, 1), top_k=5, filters=SearchFilter(fiscal_years=[2024]))
    assert {r.chunk.metadata.fiscal_year for r in results} == {2024}


def test_item_filter(index: FaissIndex) -> None:
    results = index.search(_unit(1, 1, 1, 1), top_k=5, filters=SearchFilter(items=["7"]))
    assert {r.chunk.metadata.item for r in results} == {"7"}


def test_part_filter(index: FaissIndex) -> None:
    results = index.search(_unit(1, 1, 1, 1), top_k=5, filters=SearchFilter(parts=["II"]))
    assert [r.chunk.chunk_id for r in results] == ["d"]


def test_filters_combine_as_and(index: FaissIndex) -> None:
    results = index.search(
        _unit(1, 1, 1, 1),
        top_k=5,
        filters=SearchFilter(tickers=["AAPL"], fiscal_years=[2024]),
    )
    assert [r.chunk.chunk_id for r in results] == ["c"]


def test_filter_matching_nothing_returns_nothing(index: FaissIndex) -> None:
    results = index.search(_unit(1, 0, 0, 0), top_k=5, filters=SearchFilter(tickers=["TSLA"]))
    assert results == []


def test_empty_filter_is_ignored(index: FaissIndex) -> None:
    assert len(index.search(_unit(1, 0, 0, 0), top_k=5, filters=SearchFilter())) == 4


def test_filtering_fills_top_k_from_matching_chunks(index: FaissIndex) -> None:
    """Filtering happens inside the search, not by discarding results after."""
    results = index.search(
        _unit(0, 1, 0, 0),  # nearest chunk is MSFT, which the filter excludes
        top_k=2,
        filters=SearchFilter(tickers=["AAPL"]),
    )
    assert len(results) == 2


# --- Persistence ---------------------------------------------------------


def test_roundtrip_preserves_chunks_and_ranking(index: FaissIndex, tmp_path) -> None:
    index.save(tmp_path)

    restored = FaissIndex(dimension=DIM, model_name="stub-model")
    restored.load(tmp_path)

    assert restored.size == index.size
    before = index.search(_unit(1, 0, 0, 0), top_k=3)
    after = restored.search(_unit(1, 0, 0, 0), top_k=3)
    assert [r.chunk.chunk_id for r in before] == [r.chunk.chunk_id for r in after]


def test_roundtrip_preserves_metadata(index: FaissIndex, tmp_path) -> None:
    index.save(tmp_path)
    restored = FaissIndex(dimension=DIM)
    restored.load(tmp_path)

    meta = restored.search(_unit(1, 0, 0, 0), top_k=1)[0].chunk.metadata
    assert meta.ticker == "AAPL"
    assert meta.section_label == "I-1A"


def test_saving_an_empty_index_is_rejected(tmp_path) -> None:
    with pytest.raises(ValueError, match="empty index"):
        FaissIndex(dimension=DIM).save(tmp_path)


def test_loading_with_a_different_dimension_is_rejected(index: FaissIndex, tmp_path) -> None:
    index.save(tmp_path)
    with pytest.raises(ValueError, match="dimensional"):
        FaissIndex(dimension=DIM + 1).load(tmp_path)


def test_roundtrip_preserves_filtering(index: FaissIndex, tmp_path) -> None:
    """The inverted maps are rebuilt on load, not persisted."""
    index.save(tmp_path)
    restored = FaissIndex(dimension=DIM)
    restored.load(tmp_path)

    results = restored.search(_unit(1, 1, 1, 1), top_k=5, filters=SearchFilter(tickers=["AAPL"]))
    assert {r.chunk.metadata.ticker for r in results} == {"AAPL"}


def test_roundtrip_preserves_build_settings(index: FaissIndex, tmp_path) -> None:
    index.nlist = 128
    index.save(tmp_path)

    restored = FaissIndex(dimension=DIM)
    restored.load(tmp_path)
    assert restored.nlist == 128


def test_loading_with_a_different_model_is_rejected(index: FaissIndex, tmp_path) -> None:
    """Queries would be embedded into a different space than the chunks."""
    index.save(tmp_path)
    with pytest.raises(ValueError, match="different space"):
        FaissIndex(dimension=DIM, model_name="other-model").load(tmp_path)
