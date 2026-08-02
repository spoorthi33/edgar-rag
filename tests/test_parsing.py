"""Phase 2 tests.

The synthetic filings below reproduce formatting quirks observed in the
real 10-Ks of Apple, Microsoft, Amazon, Alphabet and NVIDIA.
"""

from __future__ import annotations

from datetime import date

import pytest

from edgar_rag.models import Filing, FormType
from edgar_rag.parsing.html import html_to_text, normalize_text
from edgar_rag.parsing.pipeline import parse_filing
from edgar_rag.parsing.sections import find_item_headings

BODY = "Filler sentence providing enough body text to constitute a section. " * 12


def _filing() -> Filing:
    return Filing(
        cik="0000320193",
        ticker="AAPL",
        company_name="Apple Inc.",
        form_type=FormType.TEN_K,
        accession_number="0000320193-25-000079",
        filing_date=date(2025, 10, 31),
        fiscal_year=2025,
        source_url="https://www.sec.gov/Archives/example.htm",
    )


# --- HTML to text --------------------------------------------------------


def test_inline_tags_do_not_split_a_heading() -> None:
    """Microsoft wraps headings in nested spans; naive extraction shatters them."""
    html = "<div><span><b>ITEM 1A.</b></span><span> RISK <i>FACTORS</i></span></div>"
    assert "ITEM 1A. RISK FACTORS" in html_to_text(html)


def test_block_tags_become_separate_lines() -> None:
    text = html_to_text("<p>First block</p><p>Second block</p>")
    assert text.split("\n") == ["First block", "Second block"]


def test_script_and_style_are_dropped() -> None:
    html = "<style>.a{color:red}</style><script>var x=1</script><p>Visible</p>"
    assert html_to_text(html) == "Visible"


def test_nbsp_and_smart_quotes_normalized() -> None:
    text = normalize_text("Registrant’s  Common   Equity")
    assert text == "Registrant's Common Equity"


def test_accepts_bytes() -> None:
    assert html_to_text(b"<p>Bytes input</p>") == "Bytes input"


# --- Heading detection ---------------------------------------------------


def test_finds_headings_with_standard_spacing() -> None:
    text = f"Item 1A. Risk Factors\n{BODY}\nItem 7. MD and A\n{BODY}"
    assert [h.item for h in find_item_headings(text)] == ["1A", "7"]


def test_finds_headings_without_space_after_period() -> None:
    """Amazon and Alphabet emit "Item 1A.Risk Factors" with no space."""
    text = f"Item 1A.Risk Factors\n{BODY}\nItem 8.Financial Statements\n{BODY}"
    assert [h.item for h in find_item_headings(text)] == ["1A", "8"]


def test_finds_headings_without_punctuation() -> None:
    text = f"Item 1A Risk Factors\n{BODY}\nItem 7 Management Discussion\n{BODY}"
    assert [h.item for h in find_item_headings(text)] == ["1A", "7"]


def test_heading_match_is_case_insensitive() -> None:
    text = f"ITEM 1A. RISK FACTORS\n{BODY}"
    headings = find_item_headings(text)
    assert headings[0].item == "1A"
    assert headings[0].title == "RISK FACTORS"


def test_bare_item_line_is_not_a_heading() -> None:
    """Running page headers carry the item number but no title."""
    text = f"Item 8\n{BODY}\nItem 1A. Risk Factors\n{BODY}"
    assert [h.item for h in find_item_headings(text)] == ["1A"]


def test_table_of_contents_is_skipped() -> None:
    toc = "\n".join(
        f"Item {n}. {title}"
        for n, title in [
            ("1", "Business"),
            ("1A", "Risk Factors"),
            ("1B", "Unresolved Staff Comments"),
            ("2", "Properties"),
            ("3", "Legal Proceedings"),
            ("7", "Management Discussion"),
            ("8", "Financial Statements"),
        ]
    )
    text = f"Table of Contents\n{toc}\nItem 1A. Risk Factors\n{BODY}\nItem 7. MD and A\n{BODY}"

    headings = find_item_headings(text)
    assert [h.item for h in headings] == ["1A", "7"]
    # The surviving heading is the real one, far past the contents block.
    assert headings[0].line_index > 7


def test_repeated_running_headers_collapse_to_the_real_section() -> None:
    """Microsoft repeats "Item 8. Financial Statements" atop each page."""
    text = (
        f"Item 8. Financial Statements\nshort\n"
        f"Item 8. Financial Statements\nalso short\n"
        f"Item 8. Financial Statements\n{BODY * 3}"
    )
    headings = find_item_headings(text)
    assert len(headings) == 1
    assert headings[0].line_index == 4  # the occurrence with real body text


def test_short_section_kept_when_it_appears_once() -> None:
    """NVIDIA's Item 8 defers to statements filed later and is ~200 chars."""
    text = f"Item 7A. Market Risk\n{BODY}\nItem 8. Financial Statements\nSee Part IV.\n"
    assert [h.item for h in find_item_headings(text)] == ["7A", "8"]


def test_page_number_is_not_a_title() -> None:
    text = f"Item 1A. 42\n{BODY}\nItem 7. Management Discussion\n{BODY}"
    assert [h.item for h in find_item_headings(text)] == ["7"]


def test_part_is_tracked() -> None:
    text = (
        f"PART I\nItem 1. Financial Statements\n{BODY}\nPART II\nItem 1. Legal Proceedings\n{BODY}"
    )
    headings = find_item_headings(text)
    assert [(h.part, h.item) for h in headings] == [("I", "1"), ("II", "1")]


def test_same_item_in_different_parts_is_not_deduplicated() -> None:
    """A 10-Q has Item 1 in both parts; they are different sections."""
    text = (
        f"PART I\nItem 1. Financial Statements\n{BODY}\nPART II\nItem 1. Legal Proceedings\n{BODY}"
    )
    titles = [h.title for h in find_item_headings(text)]
    assert titles == ["Financial Statements", "Legal Proceedings"]


def test_no_headings_returns_empty() -> None:
    assert find_item_headings("Just prose with no item headings at all.") == []


# --- Sectioning ----------------------------------------------------------


def test_sections_span_to_the_next_heading() -> None:
    text = (
        "<p>Item 1A. Risk Factors</p><p>RISK BODY</p>"
        "<p>Item 7. Management Discussion</p><p>MDA BODY</p>"
    )
    padded = text.replace("RISK BODY", BODY).replace("MDA BODY", BODY)
    sections = parse_filing(_filing(), padded)

    assert [s.item for s in sections] == ["1A", "7"]
    assert "Risk Factors" in sections[0].text
    assert "Management Discussion" not in sections[0].text


def test_section_carries_filing_id_and_order() -> None:
    html = f"<p>Item 1A. Risk Factors</p><p>{BODY}</p><p>Item 7. MD and A</p><p>{BODY}</p>"
    sections = parse_filing(_filing(), html)

    assert all(s.filing_id == "0000320193/0000320193-25-000079" for s in sections)
    assert [s.order for s in sections] == [0, 1]


def test_unparseable_filing_falls_back_to_one_section() -> None:
    """Losing item metadata is acceptable; losing the document is not."""
    sections = parse_filing(_filing(), "<p>Prose with no headings whatsoever.</p>")

    assert len(sections) == 1
    assert sections[0].item == ""
    assert "Prose with no headings" in sections[0].text


@pytest.mark.parametrize("item", ["1A", "7", "7A", "8"])
def test_key_items_are_extractable(item: str) -> None:
    """These four carry most of the answerable content in a 10-K."""
    parts = [f"Item {i}. Section {i} Heading\n{BODY}" for i in ["1", "1A", "7", "7A", "8"]]
    sections = parse_filing(_filing(), "\n".join(parts))
    assert item in {s.item for s in sections}
