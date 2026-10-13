"""Curated appendix 7 records: shape, status, and the validation that blocks promotion."""

from __future__ import annotations

from typing import Any

from mabhas19.tables import app07_decisions as dec
from mabhas19.tables import app07_promote as pr


def cand(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "candidate_id": "app07-p004-r05", "doc_id": "d", "pdf_page": 4, "printed_page": "۳",
        "source_row_order": 5, "row_y": 333.0, "category": "۲. بتن", "category_footnote_refs": [],
        "subgroup": "بتن سبک‌دانه", "subgroup_footnote_refs": [], "material": "با پوکه",
        "material_inherited_from": None, "condition": "با ذرات ریز",
        "condition_parts": [{"text": "با ذرات ریز", "role": "leaf", "source": "inherited",
                             "inherited_from": "app07-p004-r04"}],
        "footnote_refs": ["1"], "display_name": "بتن سبک‌دانه با پوکه",
        "raw_density_text": "۱۲۰۰ تا ۱۴۰۰", "raw_conductivity_text": "۰٫۴۴",
        "parsed_density_kind": "range", "parsed_density_qualifier": None,
        "parsed_density_min": "1200", "parsed_density_max": "1400", "parsed_conductivity": "0.44",
        "units": {"density": "kg/m3", "thermal_conductivity": "W/m.K"},
        "raw_category": None, "raw_subgroup": None, "raw_material": None, "raw_condition": None,
        "review": {"status": "reviewed", "basis": [dec.OWNER], "primary_basis": dec.OWNER,
                   "decisions": ["A-p004-r05"], "changed_fields": [], "open_questions": []},
        "extraction_method": "gemini_whole_page", "extraction_model": "m",
        "extraction_timestamp": "t", "extraction_timestamp_source": "s",
        "raw_ref": {"request_id": "app07-p004.page", "row_index": 4},
    }  # fmt: skip
    return {**base, **over}


def test_record_shape_and_status() -> None:
    r = pr.curated_record(cand())
    assert r["verification"]["status"] == dec.OWNER
    assert r["category"]["number"] == 2
    assert r["density"] == {
        "raw": "۱۲۰۰ تا ۱۴۰۰",
        "kind": "range",
        "qualifier": None,
        "min": "1200",
        "max": "1400",
        "unit": "kg/m3",
    }
    assert r["thermal_conductivity"]["value"] == "0.44"
    assert r["condition"]["parts"][0]["inherited_from"] == "app07-p004-r04"


def test_validation_blocks_numbers_that_do_not_reparse() -> None:
    good = pr.curated_record(cand())
    bad = pr.curated_record(cand(candidate_id="x", parsed_conductivity="0.4"))
    problems = pr.validate([good, bad])
    assert any("207" in p for p in problems)
    assert any(p.startswith("x:") and "re-parse" in p for p in problems)
    assert not any(p.startswith("app07-p004-r05") for p in problems)


def test_validation_blocks_unreviewed_rows() -> None:
    review = {**cand()["review"], "status": "not_reviewed", "basis": [], "primary_basis": None}
    problems = pr.validate([pr.curated_record(cand(review=review))])
    assert any("verification status" in p for p in problems)
