"""Sentence segmentation for filing prose.

Chunk boundaries are placed between sentences, never inside one. Filings
are dense with abbreviations and figures that trip naive splitting on ".",
so the common offenders are handled explicitly:

    "...net sales of $383.3 billion, up 8.1% vs. fiscal 2022."
    "Apple Inc. and its subsidiaries (the Company)..."
"""

from __future__ import annotations

import re

# Abbreviations that end in a period but do not end a sentence.
ABBREVIATIONS = {
    "inc", "corp", "co", "ltd", "llc", "lp", "plc", "n.v", "s.a",
    "mr", "mrs", "ms", "dr", "prof", "jr", "sr", "st",
    "vs", "etc", "e.g", "i.e", "approx", "est", "no", "nos",
    "u.s", "u.k", "fig", "figs", "ref", "cf", "al",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
}  # fmt: skip

# Whitespace is the only thing ever consumed at a boundary, so closing
# quotes and brackets stay attached to the sentence they belong to. Chunk
# text must match the filing verbatim: Phase 6 verifies figures in an answer
# by looking for them literally in the retrieved text.
_GAP = re.compile(r"\s+")

# The sentence ending just before a gap: .!? plus any closing punctuation.
_ENDS_SENTENCE = re.compile(r"[.!?][\"')\]]*$")

# What a new sentence may start with.
_STARTS_SENTENCE = re.compile(r"[\"'(\[]*[A-Z0-9]")

# Trailing token before a candidate boundary, lowercased and stripped.
_LAST_TOKEN = re.compile(r"([A-Za-z.]+)\.[\"')\]]*\s*$")

# A single capital letter before the period is an initial ("J. P. Morgan").
_INITIAL = re.compile(r"\b[A-Z]\.[\"')\]]*\s*$")

# Decimals such as "$383.3" need no guard here: they carry no space after the
# period, so they never present as a boundary. Guarding on a trailing digit
# would instead suppress real breaks after a year ("...fiscal 2022. The").


def split_sentences(text: str) -> list[str]:
    """Split `text` into sentences, keeping list items and headings intact."""
    sentences: list[str] = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        sentences.extend(_split_line(line))
    return sentences


def _split_line(line: str) -> list[str]:
    pieces: list[str] = []
    start = 0
    for match in _GAP.finditer(line):
        preceding = line[start : match.start()]
        if not _ENDS_SENTENCE.search(preceding):
            continue
        if not _STARTS_SENTENCE.match(line[match.end() :]):
            continue
        if _is_false_boundary(preceding):
            continue

        piece = preceding.strip()
        if piece:
            pieces.append(piece)
        start = match.end()

    tail = line[start:].strip()
    if tail:
        pieces.append(tail)
    return pieces


def _is_false_boundary(preceding: str) -> bool:
    """True when the period before this gap does not end a sentence."""
    if _INITIAL.search(preceding):
        return True
    match = _LAST_TOKEN.search(preceding)
    if not match:
        return False
    return match.group(1).lower().strip(".") in ABBREVIATIONS
