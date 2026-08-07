"""Phase 5 tests: sparse retrieval, RRF fusion and the hybrid retriever."""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from edgar_rag.config import Settings
from edgar_rag.embeddings.base import Embedder
from edgar_rag.index.chunk_store import ChunkStore
from edgar_rag.index.faiss_index import FaissIndex
from edgar_rag.models import Chunk, ChunkMetadata, FormType, RetrievedChunk, SearchFilter
from edgar_rag.retrieval.bm25 import BM25Retriever, tokenize
from edgar_rag.retrieval.fusion import reciprocal_rank_fusion
from edgar_rag.retrieval.hybrid import HybridRetriever

DIM = 4


def _chunk(chunk_id: str, text: str, *, ticker: str = "AAPL", year: int = 2025) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        filing_id=f"cik/{chunk_id}",
        text=text,
        order=0,
        metadata=ChunkMetadata(
            cik="0000320193",
            ticker=ticker,
            company_name="Apple Inc.",
            form_type=FormType.TEN_K,
            fiscal_year=year,
            item="1A",
            part="I",
            filing_date=date(2025, 10, 31),
            accession_number=f"acc-{chunk_id}",
        ),
    )


def _retrieved(chunk_id: str, *, dense: int | None = None, sparse: int | None = None):
    return RetrievedChunk(
        chunk=_chunk(chunk_id, "text"),
        score=0.0,
        dense_rank=dense,
        sparse_rank=sparse,
    )


# --- Tokenisation --------------------------------------------------------


def test_lowercases_and_splits_words() -> None:
    assert tokenize("Revenue Grew") == ["revenue", "grew"]


def test_keeps_dollar_amounts_whole() -> None:
    """Splitting "$383.3" into digits would match almost every filing."""
    assert "$383.3" in tokenize("Net sales were $383.3 billion")


def test_keeps_percentages_whole() -> None:
    assert "8.1%" in tokenize("margin rose 8.1% this year")


def test_drops_function_words() -> None:
    assert tokenize("the risks of a disruption") == ["risks", "disruption"]


def test_drops_question_words() -> None:
    """Users ask questions; the grammar must not outweigh the content terms."""
    assert tokenize("how much did they spend on research?") == ["spend", "research"]


def test_drops_company_as_a_filer_fingerprint() -> None:
    """Apple writes "the Company" where others write "we"; it signals the
    filer rather than the topic, and pulls in that filer's chunks regardless."""
    assert "company" not in tokenize("the Company reported strong revenue")


# --- BM25 ----------------------------------------------------------------


@pytest.fixture
def sparse() -> BM25Retriever:
    return BM25Retriever(
        [
            _chunk("a", "Mine safety disclosures are not applicable to our operations."),
            _chunk("b", "Research and development expense was $8,866 million for the year."),
            _chunk("c", "Our supply chain depends on single source component suppliers."),
            _chunk("d", "Total net sales increased across all reportable segments."),
        ]
    )


def test_finds_the_exact_phrase(sparse: BM25Retriever) -> None:
    assert sparse.search("mine safety disclosures", top_k=1)[0].chunk.chunk_id == "a"


def test_matches_a_literal_figure(sparse: BM25Retriever) -> None:
    """Exact figures are what dense embeddings blur; this is BM25's job."""
    assert sparse.search("$8,866", top_k=1)[0].chunk.chunk_id == "b"


def test_records_sparse_rank(sparse: BM25Retriever) -> None:
    results = sparse.search("research development expense", top_k=2)
    assert results[0].sparse_rank == 1


def test_returns_nothing_when_no_term_matches(sparse: BM25Retriever) -> None:
    """A zero-score chunk is not a match and would only add noise to fusion."""
    assert sparse.search("cryptocurrency mining rigs", top_k=5) == []


def test_query_of_only_stopwords_returns_nothing(sparse: BM25Retriever) -> None:
    assert sparse.search("what is the", top_k=5) == []


def test_empty_retriever_returns_nothing() -> None:
    assert BM25Retriever().search("anything", top_k=5) == []


def test_respects_metadata_filters() -> None:
    retriever = BM25Retriever(
        [
            _chunk("a", "Our supply chain depends on single source suppliers.", ticker="AAPL"),
            _chunk("b", "Supply chain disruption could delay product shipments.", ticker="MSFT"),
            _chunk("c", "Revenue grew across every reportable segment.", ticker="MSFT"),
        ]
    )
    results = retriever.search("supply chain", 5, SearchFilter(tickers=["MSFT"]))
    assert [r.chunk.chunk_id for r in results] == ["b"]


def test_filter_matching_nothing_returns_nothing(sparse: BM25Retriever) -> None:
    assert sparse.search("mine safety", 5, SearchFilter(tickers=["TSLA"])) == []


def test_adding_chunks_updates_the_corpus(sparse: BM25Retriever) -> None:
    sparse.add([_chunk("e", "Goodwill impairment testing is performed annually.")])
    assert sparse.size == 5
    assert sparse.search("goodwill impairment", top_k=1)[0].chunk.chunk_id == "e"


def test_deferred_adds_recompute_only_once(sparse: BM25Retriever) -> None:
    """Term statistics are corpus-wide, so a loop of eager adds is quadratic."""
    sparse.add([_chunk("e", "Goodwill impairment testing is performed annually.")], defer=True)
    sparse.add([_chunk("f", "Deferred revenue is recognised over the contract term.")], defer=True)
    sparse.finalize()

    assert sparse.size == 6
    assert sparse.search("goodwill impairment", top_k=1)[0].chunk.chunk_id == "e"


# --- Sparse persistence --------------------------------------------------


def _store_of(chunks, tmp_path) -> ChunkStore:
    store = ChunkStore()
    store.add(chunks)
    store.save(tmp_path)
    return store


def test_saved_tokens_are_reused(sparse: BM25Retriever, tmp_path) -> None:
    """Otherwise every query process re-tokenizes: ~46s at 500k chunks."""
    sparse.save(tmp_path)
    store = _store_of(sparse._chunks, tmp_path)

    restored = BM25Retriever()
    restored.load(tmp_path, store)

    assert restored.size == sparse.size
    assert restored.search("mine safety", top_k=1)[0].chunk.chunk_id == "a"


def test_missing_token_file_falls_back_to_tokenizing(sparse: BM25Retriever, tmp_path) -> None:
    store = _store_of(sparse._chunks, tmp_path)
    restored = BM25Retriever()
    restored.load(tmp_path, store)  # no tokens saved here
    assert restored.search("mine safety", top_k=1)[0].chunk.chunk_id == "a"


def test_stale_token_file_is_discarded(sparse: BM25Retriever, tmp_path) -> None:
    """Pairing saved tokens with a changed corpus would score the wrong chunks."""
    sparse.save(tmp_path)
    grown = [*sparse._chunks, _chunk("z", "Newly ingested filing text about goodwill.")]
    store = _store_of(grown, tmp_path / "grown")

    restored = BM25Retriever()
    restored.load(tmp_path, store)

    assert restored.size == len(grown)
    assert restored.search("goodwill", top_k=1)[0].chunk.chunk_id == "z"


# --- Fusion --------------------------------------------------------------


def test_chunks_found_by_both_outrank_chunks_found_by_one() -> None:
    fused = reciprocal_rank_fusion(
        dense=[_retrieved("shared", dense=2), _retrieved("dense-only", dense=1)],
        sparse=[_retrieved("shared", sparse=2), _retrieved("sparse-only", sparse=1)],
        top_k=3,
        k=60,
    )
    assert fused[0].chunk.chunk_id == "shared"


def test_small_k_lets_a_rank_one_exclusive_hit_beat_deep_agreement() -> None:
    """Why rrf_k is 5 rather than the published 60.

    Some answers only BM25 can find: "mine safety disclosures" is its top
    hit and absent from dense results entirely. At k=60 the top ranks are
    weighted so gently that agreement far down both lists outweighs a
    rank-1 exclusive hit, and the answer falls out of the top 5.
    """
    dense = [_retrieved(f"agreed{i}", dense=15 + i) for i in range(3)]
    sparse = [_retrieved("exclusive", sparse=1)] + [
        _retrieved(f"agreed{i}", sparse=15 + i) for i in range(3)
    ]

    at_60 = reciprocal_rank_fusion(dense, sparse, top_k=1, k=60)
    at_5 = reciprocal_rank_fusion(dense, sparse, top_k=1, k=5)

    assert at_60[0].chunk.chunk_id != "exclusive"
    assert at_5[0].chunk.chunk_id == "exclusive"


def test_both_ranks_are_preserved() -> None:
    fused = reciprocal_rank_fusion(
        dense=[_retrieved("x", dense=3)],
        sparse=[_retrieved("x", sparse=7)],
        top_k=1,
    )
    assert fused[0].dense_rank == 3
    assert fused[0].sparse_rank == 7


def test_score_is_the_fused_score_not_the_original() -> None:
    fused = reciprocal_rank_fusion([_retrieved("x", dense=1)], [], top_k=1, k=5)
    assert fused[0].score == pytest.approx(1 / 6)


def test_fusing_with_an_empty_list_keeps_the_other() -> None:
    fused = reciprocal_rank_fusion([_retrieved("x", dense=1)], [], top_k=5)
    assert [r.chunk.chunk_id for r in fused] == ["x"]


def test_fusing_two_empty_lists_returns_nothing() -> None:
    assert reciprocal_rank_fusion([], [], top_k=5) == []


def test_top_k_is_respected() -> None:
    dense = [_retrieved(f"c{i}", dense=i) for i in range(1, 11)]
    assert len(reciprocal_rank_fusion(dense, [], top_k=3)) == 3


def test_no_chunk_appears_twice() -> None:
    fused = reciprocal_rank_fusion([_retrieved("x", dense=1)], [_retrieved("x", sparse=1)], top_k=5)
    assert len(fused) == 1


# --- Hybrid --------------------------------------------------------------


class StubEmbedder(Embedder):
    """Embeds by keyword so dense ranking is predictable without a model."""

    @property
    def dimension(self) -> int:
        return DIM

    @property
    def model_name(self) -> str:
        return "stub"

    @property
    def max_tokens(self) -> int:
        return 512

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            lowered = text.lower()
            vectors.append(
                [
                    1.0 if "risk" in lowered else 0.0,
                    1.0 if "revenue" in lowered else 0.0,
                    1.0 if "safety" in lowered else 0.0,
                    0.1,
                ]
            )
        array = np.array(vectors, dtype=np.float32)
        return array / np.linalg.norm(array, axis=1, keepdims=True)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]


@pytest.fixture
def hybrid() -> HybridRetriever:
    chunks = [
        _chunk("risk", "Our operations face significant risk from tariffs."),
        _chunk("revenue", "Revenue increased in every reportable segment."),
        _chunk("safety", "Mine safety disclosures are not applicable."),
    ]
    embedder = StubEmbedder()
    index = FaissIndex(dimension=DIM)
    index.add(chunks, embedder.embed_documents([c.text for c in chunks]))
    return HybridRetriever(
        index=index,
        embedder=embedder,
        sparse=BM25Retriever(chunks),
        settings=Settings(_env_file=None),
    )


def test_hybrid_mode_returns_results(hybrid: HybridRetriever) -> None:
    assert hybrid.retrieve("risk", top_k=2)


def test_dense_mode_uses_only_the_vector_index(hybrid: HybridRetriever) -> None:
    results = hybrid.retrieve("revenue", top_k=1, mode="dense")
    assert results[0].chunk.chunk_id == "revenue"
    assert results[0].sparse_rank is None


def test_sparse_mode_uses_only_bm25(hybrid: HybridRetriever) -> None:
    results = hybrid.retrieve("mine safety disclosures", top_k=1, mode="sparse")
    assert results[0].chunk.chunk_id == "safety"
    assert results[0].dense_rank is None


def test_hybrid_finds_what_dense_alone_misses(hybrid: HybridRetriever) -> None:
    """The motivating case: an exact phrase the embedding model blurs."""
    found = {r.chunk.chunk_id for r in hybrid.retrieve("mine safety disclosures", top_k=2)}
    assert "safety" in found


def test_filters_apply_to_both_retrievers(hybrid: HybridRetriever) -> None:
    assert hybrid.retrieve("risk", top_k=5, filters=SearchFilter(tickers=["TSLA"])) == []


def test_candidates_are_fetched_deeper_than_top_k(hybrid: HybridRetriever) -> None:
    """Fusion can only promote what it is given."""
    assert hybrid.candidate_k > 5
