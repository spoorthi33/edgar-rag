"""Object storage contract.

Implemented twice: local filesystem for development, S3 for the real
deployment. The rest of the pipeline never learns which one it has.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator


class ObjectStore(ABC):
    """Key/value store for raw filings and parsed artifacts."""

    @abstractmethod
    def put_bytes(self, key: str, data: bytes, content_type: str | None = None) -> str:
        """Store `data` at `key`. Returns the resolved URI."""

    @abstractmethod
    def get_bytes(self, key: str) -> bytes:
        """Fetch the object at `key`. Raises KeyError if absent."""

    @abstractmethod
    def exists(self, key: str) -> bool:
        """True when `key` is present — used to skip re-downloading filings."""

    @abstractmethod
    def list_keys(self, prefix: str = "") -> Iterator[str]:
        """Yield keys under `prefix`."""

    @abstractmethod
    def delete(self, key: str) -> None:
        """Remove `key`; a no-op when it does not exist."""

    def put_text(self, key: str, text: str, content_type: str = "text/plain") -> str:
        return self.put_bytes(key, text.encode("utf-8"), content_type)

    def get_text(self, key: str) -> str:
        return self.get_bytes(key).decode("utf-8")
