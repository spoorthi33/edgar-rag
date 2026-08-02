"""Turns stored raw filings into sections."""

from __future__ import annotations

import logging

from edgar_rag.config import Settings, get_settings
from edgar_rag.ingest.manifest import Manifest
from edgar_rag.models import Filing, Section
from edgar_rag.parsing.html import html_to_text
from edgar_rag.parsing.sections import find_item_headings
from edgar_rag.storage.base import ObjectStore
from edgar_rag.storage.factory import get_object_store

logger = logging.getLogger(__name__)


def parse_filing(filing: Filing, html: str | bytes) -> list[Section]:
    """Split one filing's HTML into its item sections."""
    text = html_to_text(html)
    headings = find_item_headings(text)

    if not headings:
        # Better a single unsegmented section than nothing: retrieval still
        # works, only the item-level metadata is unavailable.
        logger.warning("no item headings found in %s; keeping whole document", filing.filing_id)
        return [
            Section(
                filing_id=filing.filing_id,
                item="",
                title="",
                text=text,
                order=0,
                part=None,
            )
        ]

    sections: list[Section] = []
    for order, heading in enumerate(headings):
        end = headings[order + 1].char_offset if order + 1 < len(headings) else len(text)
        body = text[heading.char_offset : end].strip()
        sections.append(
            Section(
                filing_id=filing.filing_id,
                item=heading.item,
                title=heading.title,
                text=body,
                order=order,
                part=heading.part,
            )
        )
    return sections


def parse_all(
    settings: Settings | None = None,
    store: ObjectStore | None = None,
    tickers: list[str] | None = None,
) -> dict[str, list[Section]]:
    """Parse filings from the manifest. Returns sections by filing_id.

    A filing whose raw document is missing is logged and skipped rather than
    aborting the run, matching the ingestion pipeline's behaviour.
    """
    settings = settings or get_settings()
    store = store or get_object_store(settings)
    manifest = Manifest.load(settings.local_storage_path / "manifest.json")
    wanted = {t.upper() for t in tickers} if tickers else None

    results: dict[str, list[Section]] = {}
    for filing in manifest.filings.values():
        if wanted and (filing.ticker or "").upper() not in wanted:
            continue
        if not filing.storage_key:
            logger.error("no stored document recorded for %s", filing.filing_id)
            continue
        try:
            html = store.get_bytes(filing.storage_key)
        except KeyError:
            logger.error(
                "raw document missing from store for %s (%s)",
                filing.filing_id,
                filing.storage_key,
            )
            continue

        sections = parse_filing(filing, html)
        results[filing.filing_id] = sections
        logger.info(
            "%s %s FY%s: %d sections",
            filing.ticker,
            filing.form_type.value,
            filing.fiscal_year,
            len(sections),
        )
    return results
