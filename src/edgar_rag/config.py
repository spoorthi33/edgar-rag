"""Application settings, loaded from environment / .env.

Every swappable choice in the pipeline (storage backend, embedding model,
LLM provider) is a value here rather than an import, so that changing one
is a config edit instead of a code change.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class StorageBackend(StrEnum):
    LOCAL = "local"
    S3 = "s3"


class LLMProvider(StrEnum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    # --- EDGAR ---------------------------------------------------------
    # The SEC rejects requests without a descriptive User-Agent carrying a
    # real email, and rate-limits to 10 req/s per IP.
    edgar_user_agent: str = Field(
        default="edgar-rag/0.1 (set EDGAR_USER_AGENT to 'Name email@example.com')"
    )
    edgar_base_url: str = "https://data.sec.gov"
    edgar_archives_url: str = "https://www.sec.gov/Archives"
    edgar_requests_per_second: float = 8.0  # headroom under the 10/s cap
    edgar_max_retries: int = 5

    # --- Storage -------------------------------------------------------
    storage_backend: StorageBackend = StorageBackend.LOCAL
    local_storage_path: Path = PROJECT_ROOT / "data"
    s3_bucket: str | None = None
    s3_prefix: str = "edgar"
    aws_region: str = "us-east-1"

    # --- Chunking ------------------------------------------------------
    chunk_target_tokens: int = 512
    chunk_overlap_tokens: int = 64
    semantic_breakpoint_percentile: int = 95

    # --- Embeddings ----------------------------------------------------
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    embedding_batch_size: int = 64

    # --- Index / retrieval ---------------------------------------------
    index_path: Path = PROJECT_ROOT / "data" / "index"
    faiss_index_type: str = "flat"  # "flat" while developing, "ivf" at scale
    faiss_ivf_nlist: int = 256
    retrieval_top_k: int = 5
    retrieval_candidate_k: int = 20  # per-retriever depth before RRF fusion
    # RRF's published default is 60, which weights the top ranks gently and so
    # favours chunks both retrievers found. That loses answers only one of them
    # can see: "mine safety disclosures" is BM25's top hit and absent from dense
    # results entirely, and at k=60 it fell out of the fused top 5. A smaller k
    # lets a rank-1 exclusive hit outweigh a mid-ranked consensus pair.
    # Measured on a 12-query probe; Phase 8's labelled set should confirm it.
    rrf_k: int = 5

    # --- Generation ----------------------------------------------------
    llm_provider: LLMProvider = LLMProvider.ANTHROPIC
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    anthropic_model: str = "claude-sonnet-5"
    openai_model: str = "gpt-4o-mini"
    llm_max_tokens: int = 1024
    llm_temperature: float = 0.0

    # --- Database ------------------------------------------------------
    database_url: str = "postgresql+psycopg://edgar:edgar@localhost:5432/edgar_rag"

    # --- API -----------------------------------------------------------
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # --- Eval ----------------------------------------------------------
    eval_dataset_path: Path = PROJECT_ROOT / "eval" / "questions.jsonl"
    eval_cache_path: Path = PROJECT_ROOT / "data" / ".llm_cache"
    eval_judge_model: str = "claude-sonnet-5"


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings()
