"""Validation of human-verification and structural metadata in benchmark references."""

from __future__ import annotations

import copy
from datetime import date
from typing import Any

import pytest

from mabhas19.benchmark.references import (
    VERIFIED,
    ReferenceError,
    load_references,
    parse_reference,
)
from mabhas19.core.paths import EVAL_DIR

VERIFICATION = {
    "status": VERIFIED,
    "verified_by": "project_owner",
    "verified_on": date(2026, 10, 6),
}


def page(**page_fields: Any) -> dict[str, Any]:
    item = {"id": "p-1", "doc_id": "d", "pdf_page": 1, "type": "material_row",
            "expected_text": "x", "expected_fields": {"conductivity": "0.25"}}  # fmt: skip
    return {"page_id": "p", "doc_id": "d", "pdf_page": 1,
            "status": "draft_pending_human_review", "items": [item], **page_fields}  # fmt: skip


def with_item(raw: dict[str, Any], **fields: Any) -> dict[str, Any]:
    raw = copy.deepcopy(raw)
    raw["items"][0].update(fields)
    return raw


def test_verified_item_is_loaded_with_its_metadata() -> None:
    ref = parse_reference(with_item(page(), **VERIFICATION))
    item = ref.items[0]
    assert (item.status, item.verified_by, item.verified_on) == (
        VERIFIED, "project_owner", date(2026, 10, 6),
    )  # fmt: skip


def test_quoted_iso_date_is_accepted() -> None:
    item = parse_reference(
        with_item(page(), **{**VERIFICATION, "verified_on": "2026-10-06"})
    ).items[0]
    assert item.verified_on == date(2026, 10, 6)


@pytest.mark.parametrize("status", ["verified", "approved", "Human_Verified", ""])
def test_status_values_are_a_closed_set(status: str) -> None:
    with pytest.raises(ReferenceError, match="unknown status"):
        parse_reference(with_item(page(), status=status))
    with pytest.raises(ReferenceError, match="unknown status"):
        parse_reference(page(status=status))


@pytest.mark.parametrize("missing", ["verified_by", "verified_on"])
def test_human_verified_requires_who_and_when(missing: str) -> None:
    fields = {k: v for k, v in VERIFICATION.items() if k != missing}
    with pytest.raises(ReferenceError, match=f"requires {missing}"):
        parse_reference(with_item(page(), **fields))


def test_blank_verified_by_is_rejected() -> None:
    with pytest.raises(ReferenceError, match="requires verified_by"):
        parse_reference(with_item(page(), **{**VERIFICATION, "verified_by": "  "}))


@pytest.mark.parametrize("value", ["06/10/2026", "2026-13-01", "yesterday"])
def test_verified_on_must_be_an_iso_date(value: str) -> None:
    with pytest.raises(ReferenceError, match="not an ISO date"):
        parse_reference(with_item(page(), **{**VERIFICATION, "verified_on": value}))


def test_verification_fields_without_verified_status_are_rejected() -> None:
    with pytest.raises(ReferenceError, match="status is"):
        parse_reference(with_item(page(), verified_by="project_owner"))


def test_verified_page_requires_verified_items_and_metadata() -> None:
    with pytest.raises(ReferenceError, match="items are not"):
        parse_reference(page(**VERIFICATION))
    with pytest.raises(ReferenceError, match="requires verified_by"):
        parse_reference(with_item(page(status=VERIFIED), **VERIFICATION))
    ref = parse_reference(with_item(page(**VERIFICATION), **VERIFICATION))
    assert (ref.status, ref.verified_by) == (VERIFIED, "project_owner")


@pytest.mark.parametrize(
    "key", ["footnote_refs", "category_footnote_refs", "subgroup_footnote_refs"]
)
@pytest.mark.parametrize("bad", ["1", [1], ["1", None], {"1": "AAC"}])
def test_footnote_reference_fields_are_lists_of_strings(key: str, bad: Any) -> None:
    raw = with_item(page(), expected_fields={"conductivity": "0.25", key: bad})
    with pytest.raises(ReferenceError, match="list of strings"):
        parse_reference(raw)
    ok = with_item(page(), expected_fields={"conductivity": "0.25", key: ["1", "2"]})
    assert parse_reference(ok).items[0].expected_fields[key] == ["1", "2"]


def test_subgroup_must_be_a_string_or_null() -> None:
    with pytest.raises(ReferenceError, match="subgroup"):
        parse_reference(with_item(page(), expected_fields={"subgroup": ["a"]}))
    assert parse_reference(with_item(page(), expected_fields={"subgroup": None})).items
    item = parse_reference(with_item(page(), expected_fields={"subgroup": "بتن"})).items[0]
    assert item.expected_fields["subgroup"] == "بتن"


def test_unknown_item_fields_are_preserved_not_discarded() -> None:
    item = parse_reference(with_item(page(), reviewer_comment="checked twice")).items[0]
    assert item.extra == {"reviewer_comment": "checked twice"}


def test_appendix7_references_are_fully_human_verified() -> None:
    refs = load_references(EVAL_DIR / "extraction" / "reference")
    for page_id in ("app07-p003", "app07-p006", "app07-p013"):
        ref = refs[page_id]
        assert (ref.status, ref.verified_by, ref.verified_on) == (
            VERIFIED, "project_owner", date(2026, 10, 6),
        )  # fmt: skip
        for item in ref.items:
            assert (item.status, item.verified_by, item.verified_on) == (
                VERIFIED, "project_owner", date(2026, 10, 6),
            )  # fmt: skip
    others = [r for pid, r in refs.items() if not pid.startswith("app07")]
    assert all(r.status == "draft_pending_human_review" for r in others)
