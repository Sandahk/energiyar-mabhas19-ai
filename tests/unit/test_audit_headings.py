"""Heading reconstruction from glyph positions on right-to-left lines.

Fragment layouts below mirror what PyMuPDF extracts from the main Mabhas 19 PDF.
"""

from __future__ import annotations

from mabhas19.ingest.audit.headings import (
    Fragment,
    Glyph,
    apply_sequence_check,
    digit_groups_rtl,
    group_visual_lines,
    heading_from_line,
)
from mabhas19.ingest.audit.models import Confidence

NBH = "\u2011"


def frag(chars: list[tuple[str, float]], baseline: float = 100.0, bold: bool = True) -> Fragment:
    """Fragment from (char, x0) pairs in extraction order; glyphs are ~4pt wide."""
    glyphs = tuple(Glyph(c, x, x + 4.0) for c, x in chars)
    return Fragment("".join(c for c, _ in chars), glyphs, baseline, bold)


def title(text: str, x_right: float) -> list[tuple[str, float]]:
    return [(c, x_right - 5.0 * i) for i, c in enumerate(text)]


# «۱۹-۱-۱ ارزیابی» : '19' at the right edge, then '-1-1' (U+2011 hyphens) and the title leftwards.
HEADING_19_1_1 = [
    frag([("1", 388), ("9", 392)]),
    frag([(NBH, 384), ("1", 380), (NBH, 375), ("1", 371), (" ", 367), *title("ارزیابی", 362)]),
]
# «۱۹-۲-۱ دامنه» with the hyphen between 19 and 2 missing from the text layer.
HEADING_19_2_1_MISSING_HYPHEN = [
    frag([("1", 388), ("9", 392)]),
    frag([("2", 378)]),
    frag([(NBH, 374), ("1", 370), (" ", 366), *title("دامنه", 361)]),
]


def test_digit_groups_are_read_right_to_left() -> None:
    glyphs = [g for f in HEADING_19_1_1 for g in f.glyphs if g.c.isdigit()]
    assert digit_groups_rtl(glyphs) == ["19", "1", "1"]


def test_heading_with_non_breaking_hyphens() -> None:
    h = heading_from_line("d", 19, HEADING_19_1_1)
    assert h is not None
    assert h.normalized_candidate_heading == "19-1-1"
    assert h.kind == "heading"
    assert h.hyphen_codepoints == ["U+2011"]
    assert h.raw_heading == f"19{NBH}1{NBH}1 ارزیابی"
    assert h.title_raw == "ارزیابی"
    assert h.confidence == Confidence.HIGH


def test_heading_with_missing_hyphen_is_rebuilt_and_noted() -> None:
    h = heading_from_line("d", 55, HEADING_19_2_1_MISSING_HYPHEN)
    assert h is not None
    assert h.normalized_candidate_heading == "19-2-1"
    assert h.raw_fragments == ["19", "2", f"{NBH}1 دامنه"]
    assert any("absent from the text layer" in n for n in h.notes)


def test_clause_extracted_in_visual_order() -> None:
    chars = [(c, 370.0 + 4.2 * i) for i, c in enumerate("1-4-2-19")]
    line = [frag(chars, bold=False), frag(title("بازدهی", 360), bold=False)]
    h = heading_from_line("d", 62, line)
    assert h is not None
    assert h.normalized_candidate_heading == "19-2-4-1"
    assert h.kind == "clause"
    assert h.hyphen_codepoints == ["U+002D"]
    assert h.confidence == Confidence.LOW


def test_toc_entry_records_printed_page() -> None:
    line = [
        frag([("1", 388), ("9", 392)]),
        frag([(NBH, 384), ("1", 380), (" ", 376), *title("تعاریف", 372)]),
        frag([*[(".", 300.0 - 3 * i) for i in range(10)], ("1", 60)]),
    ]
    h = heading_from_line("d", 15, line)
    assert h is not None
    assert h.kind == "toc_entry"
    assert h.toc_printed_page == "1"
    assert h.confidence == Confidence.MEDIUM


def test_lines_not_starting_with_part_19_are_ignored() -> None:
    year = [frag([(c, 380.0 + 4.2 * i) for i, c in enumerate("1404")]), frag(title("سال", 370))]
    reference = [
        frag(title("جدول", 400)),
        frag([(c, 300.0 + 4.2 * i) for i, c in enumerate("19-3-1")]),
    ]
    lone = [frag([("1", 388), ("9", 392)]), frag(title("درصد", 380))]
    assert heading_from_line("d", 1, year) is None
    assert heading_from_line("d", 1, reference) is None
    assert heading_from_line("d", 1, lone) is None


def test_fragments_are_grouped_by_baseline() -> None:
    a, b, c = frag([("1", 1)], 100.0), frag([("2", 1)], 102.5), frag([("3", 1)], 130.0)
    assert group_visual_lines([c, a, b]) == [[a, b], [c]]


def test_out_of_sequence_heading_is_downgraded() -> None:
    first = heading_from_line("d", 1, HEADING_19_2_1_MISSING_HYPHEN)
    second = heading_from_line("d", 2, HEADING_19_1_1)
    assert first is not None
    assert second is not None
    checked = apply_sequence_check([first, second])
    assert checked[0].confidence == Confidence.HIGH
    assert checked[1].confidence == Confidence.MEDIUM
    assert "previous heading" in checked[1].notes[-1]
