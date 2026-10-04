"""Candidate section headings, rebuilt from glyph positions on right-to-left lines.

In the main PDF a heading such as «۱۹-۱-۱ ارزیابی چرخه عمر ساختمان» is extracted as
several fragments («19», «\u20111\u20111 ارزیابی …») and some hyphens are not in the text layer at
all. Reading the digit groups by their x position from right to left recovers the
section number regardless of fragment order or missing hyphens.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from mabhas19.ingest.audit.models import Confidence, HeadingCandidate
from mabhas19.text.charclass import HYPHEN_VARIANTS, ZWNJ, digits_to_int, is_digit
from mabhas19.text.section_numbers import PART_NUMBER, canonical_from_components

# Glyphs closer than this (points) belong to the same digit group.
DIGIT_GROUP_GAP = 1.5
# Fragments whose baselines differ by at most this (points) form one visual line.
BASELINE_TOLERANCE = 3.5
TOC_LEADER = "....."
_PREFIX_FILLER = frozenset({" ", ZWNJ, "\u200f", "\u200e"})


@dataclass(frozen=True)
class Glyph:
    c: str
    x0: float
    x1: float


@dataclass(frozen=True)
class Fragment:
    """One extracted text line/span group, in extraction order."""

    text: str
    glyphs: tuple[Glyph, ...]
    baseline: float
    bold: bool


def group_visual_lines(fragments: list[Fragment]) -> list[list[Fragment]]:
    """Cluster fragments sharing a baseline; keeps extraction order within a line."""
    lines: list[tuple[float, list[Fragment]]] = []
    for frag in fragments:
        for baseline, members in lines:
            if abs(baseline - frag.baseline) <= BASELINE_TOLERANCE:
                members.append(frag)
                break
        else:
            lines.append((frag.baseline, [frag]))
    lines.sort(key=lambda item: item[0])
    return [members for _, members in lines]


def _number_prefix(glyphs: list[Glyph]) -> list[Glyph]:
    """Digits/hyphens at the right edge of a line (the start of an RTL line)."""
    prefix: list[Glyph] = []
    for g in sorted(glyphs, key=lambda g: -g.x1):
        if g.c in _PREFIX_FILLER:
            if prefix:
                prefix.append(g)
            continue
        if is_digit(g.c) or g.c in HYPHEN_VARIANTS:
            prefix.append(g)
        else:
            break
    return [g for g in prefix if g.c not in _PREFIX_FILLER]


def digit_groups_rtl(glyphs: list[Glyph]) -> list[str]:
    """Digit groups ordered right-to-left; digits inside a group read left-to-right."""
    groups: list[list[Glyph]] = []
    for g in sorted(glyphs, key=lambda g: g.x0):
        if not is_digit(g.c):
            groups.append([])
            continue
        if groups and groups[-1] and g.x0 - groups[-1][-1].x1 <= DIGIT_GROUP_GAP:
            groups[-1].append(g)
        else:
            groups.append([g])
    texts = ["".join(g.c for g in grp) for grp in groups if grp]
    return texts[::-1]


def _leftmost_number(glyphs: list[Glyph]) -> str | None:
    digits = digit_groups_rtl([g for g in glyphs if is_digit(g.c)])
    return str(digits_to_int(digits[-1])) if digits else None


def heading_from_line(doc_id: str, pdf_page: int, line: list[Fragment]) -> HeadingCandidate | None:
    glyphs = [g for frag in line for g in frag.glyphs]
    prefix = _number_prefix(glyphs)
    groups = digit_groups_rtl(prefix)
    values = [digits_to_int(g) for g in groups]
    if len(values) < 2 or values[0] != PART_NUMBER or any(v is None for v in values):
        return None
    canonical = canonical_from_components([v for v in values if v is not None])
    if canonical is None:
        return None

    raw = "".join(frag.text for frag in line)
    is_toc = TOC_LEADER in raw or "…" in raw
    bold_chars = sum(len(f.text) for f in line if f.bold)
    bold = bold_chars * 2 >= sum(len(f.text) for f in line)
    kind = "toc_entry" if is_toc else ("heading" if bold else "clause")
    hyphen_glyphs = [g for g in prefix if g.c in HYPHEN_VARIANTS]
    hyphens = sorted({f"U+{ord(g.c):04X}" for g in hyphen_glyphs})
    notes = []
    missing = len(groups) - 1 - len(hyphen_glyphs)
    if missing > 0:
        notes.append(
            f"{missing} of {len(groups) - 1} separator hyphen(s) absent from the text layer; "
            "number rebuilt from glyph positions"
        )

    title = raw.lstrip("".join(HYPHEN_VARIANTS) + "0123456789٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹ \u200c")
    return HeadingCandidate(
        doc_id=doc_id,
        pdf_page=pdf_page,
        raw_heading=raw,
        raw_fragments=[f.text for f in line],
        normalized_candidate_heading=canonical,
        title_raw=title.strip()[:160],
        kind=kind,
        hyphen_codepoints=hyphens,
        toc_printed_page=_leftmost_number(glyphs) if is_toc else None,
        confidence=Confidence.MEDIUM if is_toc else (Confidence.HIGH if bold else Confidence.LOW),
        notes=notes,
    )


def _key(number: str) -> tuple[int, ...]:
    return tuple(int(p) for p in number.split("-"))


def apply_sequence_check(candidates: list[HeadingCandidate]) -> list[HeadingCandidate]:
    """Downgrade body headings that do not increase monotonically in document order."""
    checked: list[HeadingCandidate] = []
    previous: tuple[int, ...] | None = None
    for c in candidates:
        if c.kind != "heading":
            checked.append(c)
            continue
        key = _key(c.normalized_candidate_heading)
        if previous is not None and key <= previous:
            note = "not greater than the previous heading in document order"
            checked.append(replace(c, confidence=Confidence.MEDIUM, notes=[*c.notes, note]))
        else:
            previous = key
            checked.append(c)
    return checked
