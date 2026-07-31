"""Filesystem-backed object store, used for local development."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from edgar_rag.storage.base import ObjectStore


class LocalObjectStore(ObjectStore):
    """Stores objects as files under `root`, mirroring key paths on disk."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        # Keys are relative by contract; guard against escaping the root.
        path = (self.root / key).resolve()
        if not path.is_relative_to(self.root.resolve()):
            raise ValueError(f"key escapes storage root: {key!r}")
        return path

    def put_bytes(self, key: str, data: bytes, content_type: str | None = None) -> str:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write to a temp sibling then rename, so an interrupted run never
        # leaves a half-written filing that `exists()` would treat as done.
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(path)
        return str(path)

    def get_bytes(self, key: str) -> bytes:
        path = self._path(key)
        if not path.is_file():
            raise KeyError(key)
        return path.read_bytes()

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()

    def list_keys(self, prefix: str = "") -> Iterator[str]:
        base = self._path(prefix) if prefix else self.root
        if base.is_file():
            yield str(base.relative_to(self.root))
            return
        if not base.is_dir():
            return
        for path in sorted(base.rglob("*")):
            if path.is_file() and not path.name.endswith(".tmp"):
                yield str(path.relative_to(self.root))

    def delete(self, key: str) -> None:
        self._path(key).unlink(missing_ok=True)
