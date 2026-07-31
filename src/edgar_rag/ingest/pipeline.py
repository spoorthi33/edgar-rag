"""Downloads filings into the object store and records them in the manifest."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from edgar_rag.config import Settings, get_settings
from edgar_rag.ingest.client import EdgarClient
from edgar_rag.ingest.manifest import Manifest
from edgar_rag.models import Filing, FormType
from edgar_rag.storage.base import ObjectStore
from edgar_rag.storage.factory import get_object_store

logger = logging.getLogger(__name__)


@dataclass
class IngestReport:
    downloaded: list[Filing] = field(default_factory=list)
    skipped: list[Filing] = field(default_factory=list)
    failed: list[tuple[Filing, str]] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"downloaded={len(self.downloaded)} "
            f"skipped={len(self.skipped)} failed={len(self.failed)}"
        )


def storage_key(filing: Filing) -> str:
    """Object-store key for a filing's raw document."""
    return f"{filing.filing_id}/{filing.form_type.value}-{filing.fiscal_year}.html"


def ingest(
    tickers: list[str],
    *,
    form_types: list[FormType] | None = None,
    limit_per_ticker: int = 4,
    settings: Settings | None = None,
    client: EdgarClient | None = None,
    store: ObjectStore | None = None,
) -> IngestReport:
    """Fetch filings for `tickers`, skipping anything already ingested."""
    settings = settings or get_settings()
    store = store or get_object_store(settings)
    manifest_path = settings.local_storage_path / "manifest.json"
    manifest = Manifest.load(manifest_path)
    report = IngestReport()

    owns_client = client is None
    client = client or EdgarClient(settings)

    try:
        for ticker in tickers:
            try:
                filings = client.list_filings(ticker, form_types=form_types, limit=limit_per_ticker)
            except KeyError:
                logger.warning("unknown ticker, skipping: %s", ticker)
                continue

            logger.info("%s: %d filings to consider", ticker, len(filings))

            for filing in filings:
                key = storage_key(filing)
                # Filings are immutable once accepted, so presence in both the
                # manifest and the store means there is nothing to do.
                if manifest.has(filing) and store.exists(key):
                    report.skipped.append(filing)
                    continue

                try:
                    content = client.fetch_document(filing.source_url)
                except Exception as exc:  # noqa: BLE001 - one bad filing must not stop the run
                    logger.error("failed %s: %s", filing.filing_id, exc)
                    report.failed.append((filing, str(exc)))
                    continue

                store.put_bytes(key, content, content_type="text/html")
                filing.storage_key = key
                manifest.add(filing)
                # Saved per filing so an interrupted run keeps its progress.
                manifest.save(manifest_path)
                report.downloaded.append(filing)
                logger.info(
                    "stored %s %s FY%s (%d KB)",
                    filing.ticker,
                    filing.form_type.value,
                    filing.fiscal_year,
                    len(content) // 1024,
                )
    finally:
        if owns_client:
            client.close()

    logger.info("ingest complete: %s", report.summary())
    return report
