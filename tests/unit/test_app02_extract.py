"""Appendix 2 candidates: climate codes are never remapped, reads stay separate, flags."""

from __future__ import annotations

import pytest

from mabhas19.tables import app02 as a2


@pytest.mark.parametrize(
    ("raw", "value"), [("4B  ", "4B"), (" 0B", "0B"), ("5C", "5C"), ("۳a", "3A"), ("2 A", "2A")]
)
def test_codes_are_only_cleaned(raw: str, value: str) -> None:
    code = a2.normalize_code(raw)
    assert (code.value, code.valid, code.raw) == (value, True, raw)


@pytest.mark.parametrize("raw", ["", None, "5D", "9B", "B4", "4BB", "4-B"])
def test_invalid_codes_are_rejected_not_guessed(raw: str | None) -> None:
    code = a2.normalize_code(raw)
    assert code.value is None
    assert not code.valid


def row(code: str = "5C", number: str = "5", glyphs: int = 1) -> a2.TextRow:
    return a2.TextRow(2, 5, number, code, 223.0, (200.0, 215.0, 300.0, 231.0), glyphs)


def test_5c_is_kept_and_linked_to_source_issues() -> None:
    rec = a2.build_record(row(), "۱", a2.VisualRead("اردبیل", "5C", "5"),
                          a2.VisualRead("اردبیل"), "d")  # fmt: skip
    assert (rec["climate_code_raw"], rec["climate_code"]) == ("5C", "5C")
    assert rec["source_issue_refs"] == ["SI-0001", "SI-0002"]
    assert rec["suspicious"] == []  # a source issue, not an extraction error
    assert rec["provenance"]["climate_code"] == "pdf_text_layer"
    assert rec["verification_status"] == "machine_extracted_pending_review"


def test_text_layer_code_wins_and_disagreement_is_flagged() -> None:
    rec = a2.build_record(row("4B"), "۱", a2.VisualRead("اراک", "4A", "5"), None, "d")
    assert rec["climate_code"] == "4B"
    codes = {f["code"] for f in rec["suspicious"]}
    assert {"code_reads_disagree", "city_uncorroborated"} <= codes


def test_city_reads_compare_but_never_replace() -> None:
    rec = a2.build_record(row("4B"), "۱", a2.VisualRead("كرج", "4B", "5"),
                          a2.VisualRead("کرچ"), "d")  # fmt: skip
    assert rec["city_raw"] == "كرج"  # raw kept exactly (Arabic kaf)
    assert rec["city"] == "کرج"  # display: Arabic kaf -> Persian kaf only
    assert rec["city_normalized"] == "کرج"
    assert rec["city_second_read_raw"] == "کرچ"
    assert any(f["code"] == "city_reads_disagree" for f in rec["suspicious"])


def test_missing_city_and_empty_cell_and_alignment() -> None:
    rec = a2.build_record(row("4B", glyphs=0), "۱", None, None, "d")
    assert {"city_missing", "city_cell_empty"} <= {f["code"] for f in rec["suspicious"]}
    rec = a2.build_record(row("4B"), "۱", a2.VisualRead("اراک", "4B", "6"), a2.VisualRead("اراک"),
                          "d")  # fmt: skip
    assert [f["code"] for f in rec["suspicious"]] == ["row_alignment"]


def test_duplicate_city_names_flag_both_rows() -> None:
    a = a2.build_record(row("4B", "1"), "۱", a2.VisualRead("کرج"), a2.VisualRead("کرج"), "d")
    b = a2.build_record(row("3B", "2"), "۱", a2.VisualRead("كرج"), a2.VisualRead("کرج"), "d")
    b["candidate_id"] = "other"
    assert a2.flag_duplicates([a, b]) == [[a["candidate_id"], "other"]]
    assert all(any(f["code"] == "duplicate_city" for f in r["suspicious"]) for r in (a, b))


def test_station_names_in_parentheses_are_valid() -> None:
    rec = a2.build_record(row("3B", "29"), "۲", a2.VisualRead("تهران (مهرآباد)", "3B", "29"),
                          a2.VisualRead("تهران (مهرآباد)"), "d")  # fmt: skip
    assert rec["suspicious"] == []
