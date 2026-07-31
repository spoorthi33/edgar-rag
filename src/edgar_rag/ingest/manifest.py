"""Record of what has been ingested, so re-runs are cheap and resumable.

Filings are immutable once accepted, so anything already recorded here is
never re-downloaded. This matters because EDGAR is rate-limited and later
phases will reprocess the corpus repeatedly.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from edgar_rag.models import Filing


class Manifest(BaseModel):
    """Filings on hand, keyed by `filing_id`."""

    filings: dict[str, Filing] = Field(default_factory=dict)

    def has(self, filing: Filing) -> bool:
        return filing.filing_id in self.filings

    def add(self, filing: Filing) -> None:
        self.filings[filing.filing_id] = filing

    def __len__(self) -> int:
        return len(self.filings)

    @classmethod
    def load(cls, path: Path) -> Manifest:
        if not path.is_file():
            return cls()
        return cls.model_validate_json(path.read_text())

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.model_dump(mode="json"), indent=2))
        tmp.replace(path)
