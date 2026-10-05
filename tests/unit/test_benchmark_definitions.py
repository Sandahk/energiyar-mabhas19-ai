"""Benchmark page set, reference transcriptions, metrics and source-issue file."""

from __future__ import annotations

from typing import Any

import pytest
import yaml

from mabhas19.benchmark.metrics import label_match, score_text
from mabhas19.benchmark.references import (
    ReferenceError,
    load_benchmark_pages,
    load_references,
    parse_reference,
)
from mabhas19.core.manifest import load_manifest
from mabhas19.core.paths import CONFIGS_DIR, CURATED_DIR, EVAL_DIR, MANIFEST_PATH

PAGES = load_benchmark_pages(CONFIGS_DIR / "extraction-benchmark.yaml")[1]
REFS = load_references(EVAL_DIR / "extraction" / "reference")
MANIFEST = load_manifest(MANIFEST_PATH)


def test_benchmark_page_set_is_small_and_representative() -> None:
    assert 15 <= len(PAGES) <= 20
    docs = {p.doc_id for p in PAGES}
    for required in ("main", "app01", "app02", "app03", "app04", "app07", "app09"):
        assert f"m19-ed5-1404-{required}" in docs
    assert sum(p.doc_id.endswith("app07") for p in PAGES) >= 2


def test_every_page_has_a_reference_with_items() -> None:
    assert {p.id for p in PAGES} == set(REFS)
    for p in PAGES:
        ref = REFS[p.id]
        assert (ref.doc_id, ref.pdf_page) == (p.doc_id, p.pdf_page)
        assert ref.items


def test_reference_pages_exist_in_sources() -> None:
    for ref in REFS.values():
        assert 1 <= ref.pdf_page <= MANIFEST.get(ref.doc_id).current.page_count


def test_structured_references_carry_expected_fields() -> None:
    app03 = REFS["app03-p002"]
    for item in app03.items:
        assert set(item.expected_fields["values"]) == set(app03.extra["columns"])
    for page_id in ("app07-p003", "app07-p006", "app07-p013"):
        for item in REFS[page_id].items:
            if item.type == "material_row":
                assert item.bbox is not None
                assert "conductivity" in item.expected_fields


def test_numeric_reference_values_are_ascii_decimal_strings() -> None:
    for ref in REFS.values():
        for item in ref.items:
            values = item.expected_fields.get("values", {})
            for v in [*values.values(), item.expected_fields.get("conductivity")]:
                if v is not None:
                    assert isinstance(v, str)
                    float(v)


def _minimal() -> dict[str, Any]:
    return {"page_id": "x", "doc_id": "d", "pdf_page": 1, "status": "human_verified",
            "items": [{"id": "x-1", "doc_id": "d", "pdf_page": 1, "type": "prose",
                       "expected_text": "متن"}]}  # fmt: skip


def test_reference_schema_rejects_missing_fields_and_bad_types() -> None:
    assert parse_reference(_minimal()).items[0].expected_text == "متن"
    for key, value in (("type", "poem"), ("pdf_page", 2)):
        raw = _minimal()
        raw["items"][0][key] = value
        with pytest.raises(ReferenceError):
            parse_reference(raw)
    raw = _minimal()
    del raw["items"][0]["expected_text"]
    with pytest.raises(ReferenceError, match="missing"):
        parse_reference(raw)


def test_score_text_locates_item_and_ignores_superficial_differences() -> None:
    s = score_text("انرژی نهفته مصالح", "۱۹-۱-۱۱ انرژی نهفته مصالح\nMaterial Embodied Energy")
    assert s.cer == 0.0
    s = score_text("ساختمان\u200cها ۱۹", "متن ساختمان ها 19 پایان")
    assert s.cer > 0
    assert s.cer_normalized == 0.0


def test_label_match_is_normalized_exact_match() -> None:
    assert label_match("مسکونی", "مسكوني")
    assert not label_match("مسکونی", "مسکن")


ISSUE_STATES = {"unresolved", "resolved", "wont_fix"}


def test_source_issues_file() -> None:
    raw = yaml.safe_load((CURATED_DIR / "source_issues.yaml").read_text(encoding="utf-8"))
    issues = raw["issues"]
    ids = [i["id"] for i in issues]
    assert len(ids) == len(set(ids))
    for issue in issues:
        assert issue["status"] in ISSUE_STATES
        for loc in issue["locations"]:
            count = MANIFEST.get(loc["doc_id"]).current.page_count
            assert all(1 <= p <= count for p in loc["pdf_pages"])
        for rel in issue.get("related", []):
            assert rel in ids
    by_id = {i["id"]: i for i in issues}
    assert by_id["SI-0001"]["status"] == "unresolved"
    assert "5C" in by_id["SI-0001"]["title"]
    assert "m19-ed5-1404-app08" in {loc["doc_id"] for loc in by_id["SI-0004"]["locations"]}
