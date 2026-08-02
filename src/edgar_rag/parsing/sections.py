"""Item boundary detection.

Finding "Item 1A" in a filing is easy; finding the one occurrence that
actually starts the Risk Factors section is not. Every filing contains
decoys:

  - a table of contents listing every item in order,
  - running page headers repeating the current item on each page
    (Microsoft's 10-K carries a bare "Item 8" roughly forty times),
  - cross-references in body text ("see Item 7A below").

Three rules separate real headings from decoys, in order of application:
a heading must carry a title, must not sit inside the table of contents,
and must be followed by enough body text to be a section.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# "Item 7A. Quantitative and Qualitative Disclosures" — the title is required,
# which alone eliminates bare running headers and bare TOC entries.
#
# The separator is either punctuation or whitespace: Amazon and Alphabet emit
# "Item 1A.Risk Factors" with no space at all, while others use "Item 1A Risk
# Factors" with no punctuation.
ITEM_PATTERN = re.compile(
    r"^item\s+(?P<item>\d{1,2}[A-Z]?)\s*(?:[.:—–-]\s*|\s+)(?P<title>.{3,})$",
    re.IGNORECASE,
)

PART_PATTERN = re.compile(r"^part\s+(?P<part>[IV]{1,3})\b", re.IGNORECASE)

# A real section carries at least this much text. Tuned to sit above TOC
# rows and running headers but below genuinely short sections such as
# "Item 6. [Reserved]".
MIN_SECTION_CHARS = 400

# A table of contents is a run of item headings packed close together.
TOC_MIN_ENTRIES = 5
TOC_MAX_GAP_CHARS = 400


@dataclass(frozen=True)
class ItemHeading:
    item: str
    title: str
    part: str | None
    line_index: int
    char_offset: int


def find_item_headings(text: str) -> list[ItemHeading]:
    """Locate real item headings in parsed filing text, in document order."""
    candidates = _candidate_headings(text)
    candidates = _drop_table_of_contents(candidates)
    return _drop_undersized(candidates, len(text))


def _candidate_headings(text: str) -> list[ItemHeading]:
    headings: list[ItemHeading] = []
    part: str | None = None
    offset = 0

    for index, line in enumerate(text.split("\n")):
        part_match = PART_PATTERN.match(line)
        if part_match:
            part = part_match.group("part").upper()

        match = ITEM_PATTERN.match(line)
        if match and _looks_like_title(match.group("title")):
            headings.append(
                ItemHeading(
                    item=match.group("item").upper(),
                    title=match.group("title").strip(),
                    part=part,
                    line_index=index,
                    char_offset=offset,
                )
            )
        offset += len(line) + 1

    return headings


def _item_key(item: str) -> tuple[int, str]:
    """Sort key for item labels: "7A" sorts after "7" and before "8"."""
    match = re.match(r"(\d+)([A-Z]*)", item.upper())
    if not match:
        return (0, item.upper())
    return (int(match.group(1)), match.group(2))


def _looks_like_title(title: str) -> bool:
    """Reject page numbers and dotted leaders left over from contents rows."""
    stripped = title.strip(" .…")
    if not stripped or stripped.isdigit():
        return False
    # Needs real words, not just punctuation or a stray figure.
    return bool(re.search(r"[A-Za-z]{3,}", stripped))


def _drop_table_of_contents(headings: list[ItemHeading]) -> list[ItemHeading]:
    """Remove the contents block: many item headings packed tightly together.

    Detected structurally rather than by looking for the words "table of
    contents", which filers style inconsistently or omit.
    """
    if len(headings) < TOC_MIN_ENTRIES:
        return headings

    runs: list[list[int]] = []
    current = [0]
    for i in range(1, len(headings)):
        gap = headings[i].char_offset - headings[i - 1].char_offset
        # A run also ends where item numbering restarts. Contents rows ascend
        # (1, 1A, 2, ... 8); a drop back to a lower item means the body has
        # begun, which matters when the first real heading follows the
        # contents block closely.
        ascends = _item_key(headings[i].item) > _item_key(headings[i - 1].item)
        if gap <= TOC_MAX_GAP_CHARS and ascends:
            current.append(i)
        else:
            runs.append(current)
            current = [i]
    runs.append(current)

    dense = {i for run in runs if len(run) >= TOC_MIN_ENTRIES for i in run}
    survivors = [h for i, h in enumerate(headings) if i not in dense]

    # If everything looked like a contents block, the heuristic misfired;
    # keeping the original list is safer than returning nothing.
    return survivors or headings


def _drop_undersized(headings: list[ItemHeading], text_length: int) -> list[ItemHeading]:
    """Collapse repeated headings, keeping the one that starts the real section.

    The size test only breaks ties between repeats, because the decoys it
    targets — contents rows, running headers, cross-references — are always
    repeated. An item appearing exactly once is kept however short it is:
    NVIDIA's Item 8 is barely 200 characters because it defers to statements
    filed later, and dropping it would silently fold that section into
    Item 7A.
    """
    groups: dict[tuple[str | None, str], list[tuple[ItemHeading, int]]] = {}
    for i, heading in enumerate(headings):
        end = headings[i + 1].char_offset if i + 1 < len(headings) else text_length
        groups.setdefault((heading.part, heading.item), []).append(
            (heading, end - heading.char_offset)
        )

    kept: list[ItemHeading] = []
    for occurrences in groups.values():
        if len(occurrences) == 1:
            kept.append(occurrences[0][0])
            continue
        heading, size = max(occurrences, key=lambda pair: pair[1])
        if size >= MIN_SECTION_CHARS:
            kept.append(heading)

    return sorted(kept, key=lambda h: h.char_offset)
