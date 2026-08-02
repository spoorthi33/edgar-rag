"""Filing HTML to clean text.

Filers wrap headings in nested inline tags, so a naive `get_text("\\n")`
shatters "ITEM 1A. RISK FACTORS" into fragments across several lines and
nothing downstream can match it. Newlines are therefore inserted only at
block-level boundaries, keeping each visual line intact.
"""

from __future__ import annotations

import re
import warnings

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

# Modern filings are XHTML; the warning is noise for our purposes.
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

BLOCK_TAGS = (
    "p", "div", "br", "tr", "table", "li", "ul", "ol",
    "h1", "h2", "h3", "h4", "h5", "h6", "section", "article", "hr",
)  # fmt: skip

DROP_TAGS = ("script", "style", "head", "meta", "link")

# Inline XBRL wraps values in tags that carry no visible text of their own.
_WHITESPACE = re.compile(r"[ \t\xa0​]+")
_BLANK_LINES = re.compile(r"\n\s*\n+")


def html_to_text(html: str | bytes) -> str:
    """Extract visible text, one line per visual block."""
    if isinstance(html, bytes):
        html = html.decode("utf-8", errors="replace")

    soup = BeautifulSoup(html, "lxml")

    for tag in soup(DROP_TAGS):
        tag.decompose()

    # Mark block boundaries before flattening, so inline spans inside a
    # heading stay on one line while separate blocks do not run together.
    for tag in soup.find_all(BLOCK_TAGS):
        tag.insert_before("\n")
        tag.append("\n")

    text = soup.get_text("")
    return normalize_text(text)


def normalize_text(text: str) -> str:
    """Collapse whitespace and non-breaking spaces, drop blank lines."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Typographic characters that break naive matching downstream.
    text = text.translate(str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"'}))
    text = _WHITESPACE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = _BLANK_LINES.sub("\n", text)
    return text.strip()
