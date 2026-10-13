"""Review preparation for suspicious appendix 7 rows: grouping, inheritance scope, decisions."""

from __future__ import annotations

from typing import Any

from mabhas19.tables import app07 as a7
from mabhas19.tables import app07_review as rv


def page(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return a7.normalize_page(
        a7.PageInput("d", 4, "۳", "app07-p004.page", "m", "t", "test", rows),
        a7.Context(category=a7.Slot("۲. بتن", origin="app07-p003-r09")),
    )


def row(mat: str | None, cond: str | None = None, sub: str | None = "- با رس:") -> dict[str, Any]:
    return {
        "category_raw": "بتن سبک:",
        "subgroup_raw": sub,
        "material_raw": mat,
        "condition_raw": cond,
        "density_raw": "۱۰۰ تا ۲۰۰",
        "conductivity_raw": "۰٫۵۰",
    }


def test_inherited_fields_are_listed_with_rule_and_origin() -> None:
    first, second = page([row("- بدون ماسه"), row(None)])
    rules = {i["rule"]: i for i in rv.inherited_fields(second)}
    assert rules["leaf_label"]["from"] == first["candidate_id"]
    assert rules["cross_page_category"]["from"] == "app07-p003-r09"
    # an unambiguous numbered category carried over is listed but not questioned
    assert [i["rule"] for i in rv.decision_inheritance(second)] == ["leaf_label"]


def test_inheritance_scope_lists_unflagged_rows_too() -> None:
    rows = page([row("- بدون ماسه"), row(None)])
    scope = rv.inheritance_scope(rows)
    [leaf] = scope["leaf_label"]
    assert leaf["row_id"] == rows[1]["candidate_id"]
    assert leaf["flagged"] is False


def test_entries_need_decisions_and_are_never_verified() -> None:
    rows = page([row("الیافی", sub=None), row("- بدون ماسه", sub="الیافی")])
    rows[0]["suspicious"].append(
        {
            "code": "heading_scope_ambiguous",
            "detail": "the model gave this item label as a subgroup heading",
        }
    )
    raw = {
        (c["raw_ref"]["request_id"], c["raw_ref"]["row_index"]): {"material_raw": c["raw_material"]}
        for c in rows
    }
    entries = rv.build_entries(rows, raw, {"disagreements": []})
    assert entries
    for e in entries:
        assert e["review_status"] == "pending_human_review"
        assert e["decision"] is None
        assert e["verification_status"] == "machine_extracted_pending_review"
        assert e["decisions_needed"]
    assert "heading" in entries[0]["groups"]
