"""Phase 1 tests. No network: EDGAR responses are served by a mock transport."""

from __future__ import annotations

import json
import time
from datetime import date, timedelta

import httpx
import pytest

from edgar_rag.config import Settings
from edgar_rag.ingest.client import (
    EdgarClient,
    EdgarError,
    _fiscal_year_end_month,
    fiscal_quarter,
    fiscal_year,
)
from edgar_rag.ingest.manifest import Manifest
from edgar_rag.ingest.pipeline import ingest, storage_key
from edgar_rag.ingest.rate_limit import RateLimiter
from edgar_rag.models import FormType
from edgar_rag.storage.local import LocalObjectStore

TICKERS_PAYLOAD = {
    "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
    "1": {"cik_str": 789019, "ticker": "MSFT", "title": "Microsoft Corporation"},
}

SUBMISSIONS_PAYLOAD = {
    "fiscalYearEnd": "0928",  # Apple's fiscal year ends in September
    "filings": {
        "recent": {
            "form": ["10-K", "10-Q", "8-K", "10-Q"],
            "accessionNumber": [
                "0000320193-23-000106",
                "0000320193-23-000077",
                "0000320193-23-000050",
                "0000320193-23-000064",
            ],
            "filingDate": ["2023-11-03", "2023-08-04", "2023-05-05", "2023-05-05"],
            "reportDate": ["2023-09-30", "2023-07-01", "2023-05-01", "2023-04-01"],
            "primaryDocument": ["aapl-10k.htm", "aapl-10q.htm", "aapl-8k.htm", "aapl-10q2.htm"],
        }
    },
}


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        _env_file=None,
        edgar_user_agent="Test Runner test@example.org",
        edgar_requests_per_second=1000.0,  # keep tests fast
        local_storage_path=tmp_path,
    )


@pytest.fixture
def mock_client() -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "company_tickers.json" in url:
            return httpx.Response(200, json=TICKERS_PAYLOAD)
        if "submissions/CIK" in url:
            return httpx.Response(200, json=SUBMISSIONS_PAYLOAD)
        if "/Archives/" in url:
            return httpx.Response(200, content=b"<html><body>filing text</body></html>")
        return httpx.Response(404)

    return httpx.Client(transport=httpx.MockTransport(handler))


@pytest.fixture
def client(settings, mock_client) -> EdgarClient:
    return EdgarClient(settings=settings, client=mock_client)


# --- User-Agent enforcement ---------------------------------------------


def test_rejects_placeholder_user_agent(tmp_path) -> None:
    bad = Settings(_env_file=None, local_storage_path=tmp_path)  # default is a placeholder
    with pytest.raises(ValueError, match="real email"):
        EdgarClient(settings=bad)


def test_rejects_user_agent_without_email(tmp_path) -> None:
    bad = Settings(_env_file=None, edgar_user_agent="edgar-rag", local_storage_path=tmp_path)
    with pytest.raises(ValueError, match="real email"):
        EdgarClient(settings=bad)


def test_sends_user_agent_header(settings) -> None:
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["ua"] = request.headers["User-Agent"]
        return httpx.Response(200, json=TICKERS_PAYLOAD)

    edgar = EdgarClient(
        settings=settings, client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    edgar.ticker_map()
    assert seen["ua"] == "Test Runner test@example.org"


def test_403_explains_the_cause(settings) -> None:
    edgar = EdgarClient(
        settings=settings,
        client=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(403))),
    )
    with pytest.raises(EdgarError, match="User-Agent"):
        edgar.ticker_map()


# --- Rate limiting -------------------------------------------------------


def test_rate_limiter_spaces_requests() -> None:
    limiter = RateLimiter(requests_per_second=20.0)  # 50ms apart
    start = time.monotonic()
    for _ in range(3):
        limiter.acquire()
    assert time.monotonic() - start >= 0.09  # two gaps of 50ms


def test_default_rate_is_under_sec_cap(settings) -> None:
    assert Settings(_env_file=None).edgar_requests_per_second <= 10


def test_retries_on_429(settings) -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429)
        return httpx.Response(200, json=TICKERS_PAYLOAD)

    settings = settings.model_copy(update={"edgar_max_retries": 3})
    edgar = EdgarClient(
        settings=settings, client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    assert edgar.ticker_map()["AAPL"]["cik"] == "0000320193"
    assert calls["n"] == 2


def test_retry_count_comes_from_settings(settings) -> None:
    """Against an endpoint that always fails, stop after the configured tries."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(503)

    settings = settings.model_copy(update={"edgar_max_retries": 2})
    edgar = EdgarClient(
        settings=settings, client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    with pytest.raises(EdgarError):
        edgar.ticker_map()
    assert calls["n"] == 2


# --- Ticker resolution ---------------------------------------------------


def test_resolve_ticker_zero_pads_cik(client) -> None:
    cik, name = client.resolve_ticker("aapl")
    assert cik == "0000320193"  # submissions API needs 10 digits
    assert name == "Apple Inc."


def test_unknown_ticker_raises(client) -> None:
    with pytest.raises(KeyError):
        client.resolve_ticker("NOTATICKER")


# --- Filing listing ------------------------------------------------------


def test_lists_only_requested_forms(client) -> None:
    filings = client.list_filings("AAPL")
    assert {f.form_type for f in filings} == {FormType.TEN_K, FormType.TEN_Q}
    assert len(filings) == 3  # the 8-K is excluded


def test_fiscal_year_comes_from_report_date(client) -> None:
    tenk = next(f for f in client.list_filings("AAPL") if f.form_type is FormType.TEN_K)
    # Filed 2023-11-03 for the period ending 2023-09-30.
    assert tenk.fiscal_year == 2023
    assert tenk.fiscal_period is None


def test_quarter_derived_for_10q(client) -> None:
    tenq = next(f for f in client.list_filings("AAPL") if f.form_type is FormType.TEN_Q)
    # Period ends 2023-07-01; Apple's fiscal year starts in October, so this
    # is their Q3, not calendar Q3 by coincidence.
    assert tenq.fiscal_period == "Q3"


# --- Fiscal calendars ----------------------------------------------------
# Companies label periods by their own fiscal year, not the calendar. Getting
# this wrong mislabels chunks and defeats metadata filtering downstream.


@pytest.mark.parametrize(
    ("period_end", "fy_end_month", "expected"),
    [
        # Apple: FY ends September, named for the year it ends in.
        (date(2025, 9, 27), 9, 2025),
        (date(2025, 12, 27), 9, 2026),  # calendar 2025 but Apple's FY2026
        # NVIDIA: FY ends January.
        (date(2026, 1, 25), 1, 2026),
        (date(2025, 7, 27), 1, 2026),
        # Microsoft: FY ends June.
        (date(2026, 6, 30), 6, 2026),
        (date(2025, 9, 30), 6, 2026),
        # Alphabet: calendar-aligned.
        (date(2025, 12, 31), 12, 2025),
        (date(2026, 6, 30), 12, 2026),
    ],
)
def test_fiscal_year_derivation(period_end, fy_end_month, expected) -> None:
    assert fiscal_year(period_end, fy_end_month) == expected


@pytest.mark.parametrize(
    ("period_end", "fy_end_month", "expected"),
    [
        # Apple, FY ends late September. Its 52/53-week calendar drifts these
        # period ends across month boundaries.
        (date(2022, 12, 31), 9, "Q1"),
        (date(2023, 4, 1), 9, "Q2"),
        (date(2023, 7, 1), 9, "Q3"),  # lands in July but is fiscal Q3
        (date(2025, 12, 27), 9, "Q1"),
        # NVIDIA, FY ends January.
        (date(2025, 7, 27), 1, "Q2"),
        (date(2025, 10, 26), 1, "Q3"),
        # Microsoft, FY ends June.
        (date(2025, 9, 30), 6, "Q1"),
        (date(2025, 12, 31), 6, "Q2"),
        # Alphabet, calendar-aligned.
        (date(2026, 3, 31), 12, "Q1"),
        (date(2026, 6, 30), 12, "Q2"),
        (date(2025, 9, 30), 12, "Q3"),
    ],
)
def test_fiscal_quarter_derivation(period_end, fy_end_month, expected) -> None:
    assert fiscal_quarter(period_end, fy_end_month) == expected


@pytest.mark.parametrize("fy_end_month", range(1, 13))
def test_quarter_always_in_range(fy_end_month: int) -> None:
    """No period may fall outside Q1-Q4, whatever the fiscal calendar."""
    for day_offset in range(0, 365, 7):
        period_end = date(2025, 1, 1) + timedelta(days=day_offset)
        assert fiscal_quarter(period_end, fy_end_month) in {"Q1", "Q2", "Q3", "Q4"}


def test_missing_fiscal_year_end_defaults_to_december() -> None:
    assert _fiscal_year_end_month(None) == 12
    assert _fiscal_year_end_month("") == 12
    assert _fiscal_year_end_month("--0630") == 12  # unparseable
    assert _fiscal_year_end_month("0630") == 6


def test_limit_is_respected(client) -> None:
    assert len(client.list_filings("AAPL", limit=2)) == 2


def test_document_url_strips_dashes_and_pads(client) -> None:
    url = client.document_url("0000320193", "0000320193-23-000106", "aapl-10k.htm")
    assert url.endswith("/edgar/data/320193/000032019323000106/aapl-10k.htm")


# --- Storage -------------------------------------------------------------


def test_local_store_roundtrip(tmp_path) -> None:
    store = LocalObjectStore(tmp_path)
    store.put_text("a/b.txt", "hello")
    assert store.get_text("a/b.txt") == "hello"
    assert store.exists("a/b.txt")
    assert list(store.list_keys()) == ["a/b.txt"]
    store.delete("a/b.txt")
    assert not store.exists("a/b.txt")


def test_local_store_missing_key_raises(tmp_path) -> None:
    with pytest.raises(KeyError):
        LocalObjectStore(tmp_path).get_bytes("nope.txt")


def test_local_store_blocks_path_escape(tmp_path) -> None:
    with pytest.raises(ValueError, match="escapes"):
        LocalObjectStore(tmp_path).put_text("../escaped.txt", "x")


# --- Pipeline ------------------------------------------------------------


def test_ingest_downloads_and_records(settings, client, tmp_path) -> None:
    store = LocalObjectStore(tmp_path / "raw")
    report = ingest(["AAPL"], settings=settings, client=client, store=store)

    assert len(report.downloaded) == 3
    assert not report.failed

    manifest = Manifest.load(tmp_path / "manifest.json")
    assert len(manifest) == 3

    filing = report.downloaded[0]
    assert store.exists(storage_key(filing))
    assert filing.storage_key == storage_key(filing)


def test_ingest_is_idempotent(settings, client, tmp_path) -> None:
    """The second run must not re-download: EDGAR is rate-limited."""
    store = LocalObjectStore(tmp_path / "raw")
    ingest(["AAPL"], settings=settings, client=client, store=store)
    second = ingest(["AAPL"], settings=settings, client=client, store=store)

    assert not second.downloaded
    assert len(second.skipped) == 3


def test_ingest_redownloads_if_file_missing(settings, client, tmp_path) -> None:
    """Manifest alone is not enough — the object must actually be present."""
    store = LocalObjectStore(tmp_path / "raw")
    first = ingest(["AAPL"], settings=settings, client=client, store=store)
    store.delete(storage_key(first.downloaded[0]))

    second = ingest(["AAPL"], settings=settings, client=client, store=store)
    assert len(second.downloaded) == 1


def test_unknown_ticker_does_not_abort_run(settings, client, tmp_path) -> None:
    store = LocalObjectStore(tmp_path / "raw")
    report = ingest(["NOPE", "AAPL"], settings=settings, client=client, store=store)
    assert len(report.downloaded) == 3


def test_failed_download_is_recorded_not_raised(settings, tmp_path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "company_tickers.json" in url:
            return httpx.Response(200, json=TICKERS_PAYLOAD)
        if "submissions/CIK" in url:
            return httpx.Response(200, json=SUBMISSIONS_PAYLOAD)
        return httpx.Response(404)  # every document fetch fails

    edgar = EdgarClient(
        settings=settings, client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    report = ingest(
        ["AAPL"], settings=settings, client=edgar, store=LocalObjectStore(tmp_path / "raw")
    )
    assert not report.downloaded
    assert len(report.failed) == 3


def test_manifest_survives_reload(tmp_path, settings, client) -> None:
    ingest(["AAPL"], settings=settings, client=client, store=LocalObjectStore(tmp_path / "raw"))
    raw = json.loads((tmp_path / "manifest.json").read_text())
    assert len(raw["filings"]) == 3
    assert Manifest.load(tmp_path / "manifest.json").filings
