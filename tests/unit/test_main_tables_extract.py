from __future__ import annotations

import hashlib
from typing import Any

import pytest

from mabhas19.tables.main_audit import TABLES
from mabhas19.tables.main_extract import (
    ExtractionError,
    Grid,
    _metric,
    _standard,
    build_cells,
    climate_group,
    label,
    parse_condition,
    t19_3_3,
    value,
)
from mabhas19.tables.main_validate import EXPECTED, condition_bounds, validate_table


def _cell(row: int, col: int, text: str, kind: str = "label", **kw: Any) -> dict[str, Any]:
    rowspan, colspan = kw.get("rowspan", 1), kw.get("colspan", 1)
    parsed, table = kw.get("parsed"), kw.get("table", "19-3-3")
    return {
        "cell_id": f"main-t{table}-r{row:02d}c{col:02d}",
        "row": row,
        "col": col,
        "rowspan": rowspan,
        "colspan": colspan,
        "text": text,
        "text_raw": text,
        "value_class": kind,
        "parsed": parsed,
        "bbox": [0, 0, 1, 1],
    }


def test_expected_shapes_cover_all_tables() -> None:
    assert set(EXPECTED) == {t.number for t in TABLES}
    assert sum(n for _, _, n in EXPECTED.values()) == 674


def test_parse_condition_keeps_printed_operators() -> None:
    c = parse_condition("5000 <CDD10°C≤6000")
    assert c["variable"] == "CDD10°C"
    assert (c["lower"]["value"], c["lower"]["operator"]) == ("5000", "<")
    assert (c["upper"]["value"], c["upper"]["operator"]) == ("6000", "≤")
    only_upper = parse_condition("R ≤ 1")
    assert only_upper["lower"] is None
    assert only_upper["upper"]["operator"] == "≤"
    with pytest.raises(ExtractionError):
        parse_condition("R")


def test_climate_group_lists_classes_and_links_si_0002_for_5c() -> None:
    g = climate_group(_cell(0, 0, "5B, 5C", "climate_classes"))
    assert g["applies_to"] == ["5B", "5C"]
    assert g["source_issue_refs"] == ["SI-0002"]
    assert g["evidence_cell"] == "main-t19-3-3-r00c00"
    assert climate_group(_cell(0, 1, "2A , 2B", "climate_classes"))["source_issue_refs"] == []


def test_label_strips_known_footnote_marker() -> None:
    c = _cell(1, 0, "گروه‌الف", table="19-2-1")
    assert label(c) == {"text": "گروه", "cell": "main-t19-2-1-r01c00", "footnote_marker": "الف"}


def test_dash_is_no_printed_value_never_zero() -> None:
    v = value(_cell(1, 1, "-", "dash"))
    assert v["value"] is None
    assert v["status"] == "no_printed_value"


def test_percent_value_keeps_token_and_sign() -> None:
    c = _cell(1, 3, "٪۹۲", "percent", parsed={"token": "۹۲", "value": "92", "percent_sign": "٪"})
    v = value(c)
    assert (v["printed"], v["token"], v["value"], v["unit"]) == ("٪۹۲", "۹۲", "92", "%")


def test_metric_only_from_printed_evidence() -> None:
    assert _metric(_cell(1, 3, "۶", "number"), "بر اساس ضریب عملکرد (COP)")[0] == "COP"
    assert _metric(_cell(1, 3, "۶", "number"), "جریان مختلط")[0] is None
    assert _metric(_cell(1, 3, "IE4", "category"), "سرعت متغیر")[0] == "IE_efficiency_class"


def test_standard_number_order_is_not_inferred() -> None:
    s = _standard(_cell(1, 4, "۲-۷۸۱۷", "identifier"))
    assert s["displayed"] == "۲-۷۸۱۷"
    assert s["part_order"] == "ambiguous_bidi"


def test_grid_rejects_overlapping_cells() -> None:
    with pytest.raises(ExtractionError):
        Grid([_cell(0, 0, "a", colspan=2), _cell(0, 1, "b")])


def _audit_cell(row: int, col: int, text: str) -> dict[str, Any]:
    return {
        "row": row,
        "col": col,
        "rowspan": 1,
        "colspan": 1,
        "bbox": [0, 0, 1, 1],
        "text_raw": text,
        "text_repair_v1": text,
        "text": text,
        "value_class": "label",
        "flags": [],
    }


def test_corrections_are_guarded_and_keep_the_pre_correction_text() -> None:
    spec = next(t for t in TABLES if t.number == "19-5-7")
    before = "عموم ی"
    good = {
        "main-t19-5-7-r02c00": {
            "id": "L1",
            "kind": "split_word",
            "basis": "assistant_reviewed",
            "expect_text_sha256": hashlib.sha256(before.encode()).hexdigest(),
            "replacements": [{"from": "عموم ی", "to": "عمومی", "count": 1}],
        }
    }
    cells = build_cells(spec, [_audit_cell(2, 0, before)], "97", good)
    assert (cells[0]["text_before_correction"], cells[0]["text"]) == (before, "عمومی")
    assert cells[0]["verification"]["status"] == "assistant_reviewed"
    stale = {"main-t19-5-7-r02c00": {**good["main-t19-5-7-r02c00"], "expect_text_sha256": "0"}}
    with pytest.raises(ExtractionError):
        build_cells(spec, [_audit_cell(2, 0, before)], "97", stale)


def test_19_3_3_records_link_the_formula_and_validate() -> None:
    rows = [
        ("نسبت R", "رده بازدهی انرژی ساختمان"),
        ("R ≤ 1", "A"),
        ("1 < R ≤ 2", "B"),
        ("2 < R ≤ 3", "C"),
        ("3 < R ≤ 4", "D"),
    ]
    cells = [
        _cell(r, k, text, "expression" if r and k == 0 else "label")
        for r, pair in enumerate(rows)
        for k, text in enumerate(pair)
    ]
    for c in cells:
        c.update(visual_ltr=c["text"], symbols=None)
    ctx = {
        "blocks": {"main-p073-f01": {"text": "R = E_actual / E_ideal", "formula": {"latex": "x"}}}
    }
    records, meta = t19_3_3(Grid(cells), ctx)
    assert [r["energy_rating"]["code"] for r in records] == ["A", "B", "C", "D"]
    assert all(r["formula_ref"] == "main-p073-f01" for r in records)
    assert len(list(condition_bounds(records[1]))) == 2
    meta["grid"] = {"rows": 5, "cols": 2}
    assert validate_table("19-3-3", cells, records, meta) == []
    records[3]["r_range"]["upper"]["value"] = "5"  # an invented bound is caught
    assert validate_table("19-3-3", cells, records, meta)
