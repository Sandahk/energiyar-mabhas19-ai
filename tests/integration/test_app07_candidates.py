"""Phase 2C on the real appendix 7 PDF and the human-verified reference rows (offline)."""

from __future__ import annotations

import copy
from typing import Any

import pymupdf
import pytest

from mabhas19.benchmark import app07_scoring as sc
from mabhas19.benchmark.references import load_references
from mabhas19.core.paths import EVAL_DIR
from mabhas19.tables import app07 as a7
from mabhas19.tables import app07_build as build


@pytest.fixture(scope="module")
def pdf() -> pymupdf.Document:
    return pymupdf.open(build.PDF)


def test_value_row_count_matches_verified_pages(pdf: pymupdf.Document) -> None:
    refs = load_references(EVAL_DIR / "extraction" / "reference")
    for page_id in build.CROSSCHECK_PAGES:
        pno = int(page_id.rsplit("p", 1)[1])
        assert len(build.value_row_centres(pdf[pno - 1])) == len(sc.verified_rows(refs[page_id]))


def test_value_rows_on_headerless_and_split_pages(pdf: pymupdf.Document) -> None:
    assert len(build.value_row_centres(pdf[4])) == 1  # p5: one row, no column header
    assert len(build.value_row_centres(pdf[8])) == 27  # p9: header repeated mid-page


def _perfect_candidates() -> list[dict[str, Any]]:
    """Candidates that say exactly what the verified references say."""
    refs = load_references(EVAL_DIR / "extraction" / "reference")
    out = []
    for page_id in build.CROSSCHECK_PAGES:
        for ref in sc.verified_rows(refs[page_id]):
            f = ref.expected_fields
            q = a7.QUALIFIERS.get(f.get("density_qualifier") or "")
            out.append({
                "candidate_id": ref.id, "pdf_page": ref.pdf_page,
                "raw_density_text": ref.expected_text, "raw_conductivity_text": None,
                "category": f.get("category"), "subgroup": f.get("subgroup"),
                "material": f.get("material"), "condition": f.get("condition"),
                "footnote_refs": f.get("footnote_refs", []),
                "subgroup_footnote_refs": f.get("subgroup_footnote_refs", []),
                "category_footnote_refs": f.get("category_footnote_refs", []),
                "parsed_density_qualifier": q, "parsed_density_min": f.get("density_min"),
                "parsed_density_max": f.get("density_max"),
                "parsed_conductivity": f.get("conductivity"),
            })  # fmt: skip
    return out


def test_crosscheck_reports_disagreements_without_replacing_either_side() -> None:
    perfect = _perfect_candidates()
    check = build.crosscheck(perfect)
    assert check["reference_rows"] == 56
    assert check["disagreements"] == []
    assert check["unmatched_reference_rows"] == []

    changed = copy.deepcopy(perfect)
    changed[0]["parsed_conductivity"] = "1.8"  # same value, lost trailing zero
    changed[1]["material"] = "something else"
    check = build.crosscheck(changed)
    assert [(d["field"], d["reference"], d["candidate"]) for d in check["disagreements"]] == [
        ("conductivity", "1.80", "1.8"),
        ("material", None, "something else"),
    ]
    assert changed[0]["parsed_conductivity"] == "1.8"  # the candidate is not overwritten
