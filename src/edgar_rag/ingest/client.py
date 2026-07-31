"""HTTP client for the SEC EDGAR APIs.

Endpoints used:
  - https://www.sec.gov/files/company_tickers.json   ticker -> CIK map
  - https://data.sec.gov/submissions/CIK##########.json  a company's filings
  - https://www.sec.gov/Archives/edgar/data/...      the filing documents

Two SEC rules are enforced here because violating either breaks ingestion:
a descriptive User-Agent carrying a real email (absent it, 403), and the
10 requests/second per-IP cap (exceed it, 429 and a temporary block).
"""

from __future__ import annotations

import calendar
import logging
from collections.abc import Iterator
from datetime import date, datetime

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from edgar_rag.config import Settings, get_settings
from edgar_rag.ingest.rate_limit import RateLimiter
from edgar_rag.models import Filing, FormType

logger = logging.getLogger(__name__)

TICKER_URL = "https://www.sec.gov/files/company_tickers.json"

# Retried: transient rate limiting and SEC-side unavailability.
RETRY_STATUS = {429, 500, 502, 503, 504}


class EdgarError(RuntimeError):
    pass


class RetryableStatus(EdgarError):
    """Raised for statuses worth retrying, so tenacity can back off."""


class EdgarClient:
    """Rate-limited, retrying client for EDGAR."""

    def __init__(
        self, settings: Settings | None = None, client: httpx.Client | None = None
    ) -> None:
        self.settings = settings or get_settings()
        self._validate_user_agent(self.settings.edgar_user_agent)
        self.limiter = RateLimiter(self.settings.edgar_requests_per_second)
        self._client = client or httpx.Client(timeout=30.0, follow_redirects=True)
        # Applied here rather than at construction so an injected client is
        # also guaranteed to carry them.
        self._client.headers.update(
            {
                "User-Agent": self.settings.edgar_user_agent,
                "Accept-Encoding": "gzip, deflate",
            }
        )
        self._ticker_map: dict[str, dict[str, str]] | None = None

    @staticmethod
    def _validate_user_agent(user_agent: str) -> None:
        """Fail loudly at construction rather than with a confusing 403 later."""
        if "@" not in user_agent or "example.com" in user_agent:
            raise ValueError(
                "EDGAR_USER_AGENT must identify you with a real email address, "
                'e.g. "Jane Doe jane@company.com". The SEC returns 403 otherwise.'
            )

    # --- HTTP ----------------------------------------------------------

    @retry(
        retry=retry_if_exception_type((RetryableStatus, httpx.TransportError)),
        wait=wait_exponential(multiplier=5, min=5, max=60),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def _get(self, url: str) -> httpx.Response:
        self.limiter.acquire()
        response = self._client.get(url)
        if response.status_code in RETRY_STATUS:
            raise RetryableStatus(f"{response.status_code} from {url}")
        if response.status_code == 403:
            raise EdgarError(
                f"403 from {url}. The SEC rejects requests without a descriptive "
                "User-Agent containing a real email address."
            )
        response.raise_for_status()
        return response

    # --- Ticker / CIK --------------------------------------------------

    def ticker_map(self) -> dict[str, dict[str, str]]:
        """Ticker -> {cik, name}, fetched once per client."""
        if self._ticker_map is None:
            payload = self._get(TICKER_URL).json()
            self._ticker_map = {
                entry["ticker"].upper(): {
                    # CIKs are zero-padded to 10 digits in the submissions API.
                    "cik": str(entry["cik_str"]).zfill(10),
                    "name": entry["title"],
                }
                for entry in payload.values()
            }
            logger.info("loaded %d ticker mappings", len(self._ticker_map))
        return self._ticker_map

    def resolve_ticker(self, ticker: str) -> tuple[str, str]:
        """Returns (cik, company_name). Raises KeyError for unknown tickers."""
        entry = self.ticker_map().get(ticker.upper())
        if entry is None:
            raise KeyError(f"unknown ticker: {ticker}")
        return entry["cik"], entry["name"]

    # --- Filings -------------------------------------------------------

    def list_filings(
        self,
        ticker: str,
        form_types: list[FormType] | None = None,
        limit: int | None = None,
        since: date | None = None,
    ) -> list[Filing]:
        """List a company's filings, most recent first.

        Only the `recent` block is read (roughly the last 1000 filings),
        which covers many years of 10-Ks and 10-Qs.
        """
        forms = {f.value for f in (form_types or [FormType.TEN_K, FormType.TEN_Q])}
        cik, company_name = self.resolve_ticker(ticker)

        url = f"{self.settings.edgar_base_url}/submissions/CIK{cik}.json"
        payload = self._get(url).json()
        recent = payload.get("filings", {}).get("recent", {})
        # "MMDD", e.g. "0928" for Apple. Companies whose fiscal year is not
        # calendar-aligned need this to label periods the way they do.
        fy_end_month = _fiscal_year_end_month(payload.get("fiscalYearEnd"))

        filings: list[Filing] = []
        for row in _rows(recent):
            if row["form"] not in forms:
                continue
            filing_date = datetime.strptime(row["filingDate"], "%Y-%m-%d").date()
            if since and filing_date < since:
                continue
            if not row.get("primaryDocument"):
                continue

            report_date = (
                datetime.strptime(row["reportDate"], "%Y-%m-%d").date()
                if row.get("reportDate")
                else filing_date
            )
            form_type = FormType(row["form"])

            filings.append(
                Filing(
                    cik=cik,
                    ticker=ticker.upper(),
                    company_name=company_name,
                    form_type=form_type,
                    accession_number=row["accessionNumber"],
                    filing_date=filing_date,
                    # Derived from the period covered, not the filing date: a
                    # 10-K filed in Nov 2023 for FY2023 and one filed in Jan
                    # 2024 for FY2023 must agree.
                    fiscal_year=fiscal_year(report_date, fy_end_month),
                    fiscal_period=(
                        fiscal_quarter(report_date, fy_end_month)
                        if form_type is FormType.TEN_Q
                        else None
                    ),
                    source_url=self.document_url(
                        cik, row["accessionNumber"], row["primaryDocument"]
                    ),
                )
            )
            if limit and len(filings) >= limit:
                break
        return filings

    def document_url(self, cik: str, accession_number: str, document: str) -> str:
        """Archive URL for a filing's primary document."""
        return (
            f"{self.settings.edgar_archives_url}/edgar/data/"
            f"{int(cik)}/{accession_number.replace('-', '')}/{document}"
        )

    def fetch_document(self, url: str) -> bytes:
        return self._get(url).content

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> EdgarClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _fiscal_year_end_month(fiscal_year_end: str | None) -> int:
    """Parse EDGAR's "MMDD" fiscal-year-end field. Defaults to December."""
    if not fiscal_year_end or len(fiscal_year_end) != 4 or not fiscal_year_end.isdigit():
        return 12
    month = int(fiscal_year_end[:2])
    return month if 1 <= month <= 12 else 12


def fiscal_year(report_date: date, fy_end_month: int) -> int:
    """Fiscal year a period belongs to, labelled as the company labels it.

    Convention: a fiscal year is named for the calendar year it *ends* in.
    Apple's year ends in September, so a quarter ending December 2025 falls
    in Apple's FY2026 — not FY2025 as a calendar reading would suggest.
    """
    return report_date.year if report_date.month <= fy_end_month else report_date.year + 1


def fiscal_quarter(report_date: date, fy_end_month: int) -> str:
    """Quarter within the fiscal year, measured back from the year end.

    Counting months forward from the fiscal year start is unreliable: many
    filers use a 52/53-week calendar, so period ends drift across month
    boundaries (Apple's fiscal Q3 can end on July 1, which naive month
    arithmetic reads as Q4). Measuring the distance to year end instead
    tolerates that drift.
    """
    fy = fiscal_year(report_date, fy_end_month)
    fy_end = _last_day_of_month(fy, fy_end_month)
    quarters_remaining = round((fy_end - report_date).days / 91.25)
    return f"Q{min(4, max(1, 4 - quarters_remaining))}"


def _last_day_of_month(year: int, month: int) -> date:
    return date(year, month, calendar.monthrange(year, month)[1])


def _rows(recent: dict[str, list]) -> Iterator[dict]:
    """EDGAR returns parallel arrays; yield them as per-filing dicts."""
    keys = list(recent)
    if not keys:
        return
    for values in zip(*(recent[k] for k in keys), strict=False):
        yield dict(zip(keys, values, strict=False))
