"""Printed-page marker classification and candidate scoring."""

from __future__ import annotations

import pytest

from mabhas19.ingest.audit.models import Confidence
from mabhas19.ingest.audit.page_numbers import (
    METHOD_NONE,
    METHOD_TEXT_FOOTER,
    METHOD_VECTOR_GLYPH,
    FooterObservation,
    build_page_map_candidates,
    classify_marker,
    letter_index,
    offset_runs,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("12", ("number", "12")),
        ("۱۲", ("number", "12")),
        ("١٢", ("number", "12")),
        ("\u200c3\u200c", ("number", "3")),
        ("ب", ("letter", "ب")),
        ("أ", ("letter", "أ")),
        ("ض", ("letter", "ض")),
    ],
)
def test_classify_marker(raw: str, expected: tuple[str, str]) -> None:
    assert classify_marker(raw) == expected


@pytest.mark.parametrize("raw", ["", "\u200c\u200c", "پ", "ab", "صفحه", "12a"])
def test_classify_marker_rejects_non_markers(raw: str) -> None:
    assert classify_marker(raw) is None


@pytest.mark.parametrize(
    ("letter", "index"), [("ا", 1), ("أ", 1), ("ب", 2), ("ت", 3), ("ض", 15), ("ظ", 17)]
)
def test_front_matter_letters_follow_hijai_order(letter: str, index: int) -> None:
    assert letter_index(letter) == index


def obs(
    page: int, *tokens: str, glyphs: int = 0, hidden: tuple[str, ...] = ()
) -> FooterObservation:
    return FooterObservation(page, tuple(tokens), glyphs, hidden)


def test_consistent_sequence_is_high_confidence_and_breaks_are_medium() -> None:
    cands = build_page_map_candidates(
        "d", [obs(1, "ب"), obs(2, "ت"), obs(3, "1"), obs(4, "2"), obs(5, "9"), obs(6, "4")]
    )
    by_page = {c.pdf_page: c for c in cands}
    assert by_page[1].marker_kind == "letter"
    assert by_page[1].confidence == Confidence.HIGH  # ب (2nd) on page 1, ت (3rd) on page 2
    assert by_page[3].printed_page_candidate == "1"
    assert by_page[3].confidence == Confidence.HIGH
    assert by_page[5].confidence == Confidence.MEDIUM
    assert all(c.detection_method == METHOD_TEXT_FOOTER for c in cands)
    assert not any(c.verified for c in cands)


def test_hidden_or_missing_markers_are_absent() -> None:
    cands = build_page_map_candidates("d", [obs(1, "5"), obs(2, hidden=("110",)), obs(3)])
    assert [c.marker_kind for c in cands] == ["number", "absent", "absent"]
    assert cands[1].confidence == Confidence.NONE
    assert cands[1].detection_method == METHOD_NONE
    assert "110" in cands[1].notes[-1]


def test_vector_glyph_markers_are_inferred_with_low_confidence() -> None:
    cands = build_page_map_candidates("d", [obs(1), obs(2, glyphs=1), obs(3, glyphs=1)])
    assert [c.printed_page_candidate for c in cands] == [None, "1", "2"]
    assert cands[1].confidence == Confidence.LOW
    assert cands[1].detection_method == METHOD_VECTOR_GLYPH
    assert cands[1].raw_marker is None


def test_glyph_count_mismatch_is_noted() -> None:
    cands = build_page_map_candidates("d", [obs(1, glyphs=3)])
    assert "3 glyph cluster(s)" in cands[0].notes[-1]


def test_offset_runs_summarise_page_ranges() -> None:
    cands = build_page_map_candidates(
        "d", [obs(1), obs(2, "1"), obs(3, "2"), obs(4), obs(5, "3"), obs(6, "4")]
    )
    runs = [
        (r.marker_kind, r.offset_printed_minus_pdf, r.first_pdf_page, r.last_pdf_page)
        for r in offset_runs(cands)
    ]
    assert runs == [
        ("absent", None, 1, 1),
        ("number", -1, 2, 3),
        ("absent", None, 4, 4),
        ("number", -2, 5, 6),
    ]
