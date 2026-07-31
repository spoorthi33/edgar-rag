"""Builds the configured object store."""

from __future__ import annotations

from edgar_rag.config import Settings, StorageBackend, get_settings
from edgar_rag.storage.base import ObjectStore
from edgar_rag.storage.local import LocalObjectStore


def get_object_store(settings: Settings | None = None) -> ObjectStore:
    settings = settings or get_settings()

    if settings.storage_backend is StorageBackend.S3:
        if not settings.s3_bucket:
            raise ValueError("STORAGE_BACKEND=s3 requires S3_BUCKET to be set")
        from edgar_rag.storage.s3 import S3ObjectStore

        return S3ObjectStore(
            bucket=settings.s3_bucket,
            prefix=settings.s3_prefix,
            region=settings.aws_region,
        )

    return LocalObjectStore(settings.local_storage_path / "raw")
