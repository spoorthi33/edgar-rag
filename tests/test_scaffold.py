"""Phase 0 smoke tests: the package imports and the contracts hold."""

from datetime import date

import pytest

from edgar_rag.config import LLMProvider, Settings, StorageBackend
from edgar_rag.embeddings import Embedder
from edgar_rag.generation import LLMClient
from edgar_rag.index import VectorIndex
from edgar_rag.models import Chunk, ChunkMetadata, Filing, FormType
from edgar_rag.storage import ObjectStore


def _metadata(**overrides) -> ChunkMetadata:
    defaults = dict(
        cik="320193",
        ticker="AAPL",
        company_name="Apple Inc.",
        form_type=FormType.TEN_K,
        fiscal_year=2023,
        item="7",
        filing_date=date(2023, 11, 3),
        accession_number="0000320193-23-000106",
    )
    return ChunkMetadata(**{**defaults, **overrides})


def test_filing_id_is_stable() -> None:
    filing = Filing(
        cik="320193",
        ticker="AAPL",
        company_name="Apple Inc.",
        form_type=FormType.TEN_K,
        accession_number="0000320193-23-000106",
        filing_date=date(2023, 11, 3),
        fiscal_year=2023,
        source_url="https://www.sec.gov/Archives/example.htm",
    )
    assert filing.filing_id == "320193/0000320193-23-000106"


def test_citation_token_format() -> None:
    assert _metadata().citation == "320193:2023:7"


def test_citation_handles_missing_item() -> None:
    assert _metadata(item=None).citation == "320193:2023:-"


def test_chunk_carries_provenance() -> None:
    chunk = Chunk(
        chunk_id="c1",
        filing_id="320193/0000320193-23-000106",
        text="Research and development expense was $29.9 billion.",
        metadata=_metadata(),
    )
    assert chunk.metadata.ticker == "AAPL"
    assert chunk.metadata.fiscal_year == 2023


def test_settings_defaults() -> None:
    settings = Settings(_env_file=None)
    assert settings.storage_backend is StorageBackend.LOCAL
    assert settings.llm_provider is LLMProvider.ANTHROPIC
    assert settings.embedding_dim == 384
    assert settings.edgar_requests_per_second <= 10  # SEC hard cap


@pytest.mark.parametrize("interface", [ObjectStore, Embedder, VectorIndex, LLMClient])
def test_interfaces_are_abstract(interface: type) -> None:
    with pytest.raises(TypeError):
        interface()  # type: ignore[abstract]
