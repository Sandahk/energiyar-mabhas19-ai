"""Appendix 3 candidates: text-layer numbers, column assignment, no 5C, label reads."""

from __future__ import annotations

from typing import Any

import pytest

from mabhas19.tables import app03 as a3


@pytest.mark.parametrize(("raw", "value"), [("151 ", "151"), ("1030", "1030"), ("۹۵", "95")])
def test_eui_is_parsed_as_printed(raw: str, value: str) -> None:
    assert a3.parse_eui(raw).value == value


@pytest.mark.parametrize("raw", ["", None, "12.5", "1,030", "0", "15a", "1 030"])
def test_bad_eui_is_rejected_not_repaired(raw: str | None) -> None:
    assert a3.parse_eui(raw).value is None


def row(values: dict[str, str] | None = None, order: int = 3, glyphs: int = 1) -> a3.TableRow:
    values = values or {"5B": "130", "4B ": "116", "0B": "90"}
    cols = {k: i for i, k in enumerate(values)}
    boxes = {k: (float(i), 0.0, float(i) + 1, 1.0) for i, k in enumerate(values)}
    return a3.TableRow("D", 2, order, 100.0, values, boxes, cols, order + 1,
                       (10.0, 0.0, 20.0, 1.0), (20.0, 0.0, 22.0, 1.0), glyphs)  # fmt: skip


def gem(label: str = "ویلایی", number: str = "۳", values: dict[str, str] | None = None
        ) -> a3.LabelRead:  # fmt: skip
    return a3.LabelRead(label, number, values or {"5B": "130", "4B": "116", "0B": "90"})


def test_values_follow_their_own_column_header_and_no_5c_cell() -> None:
    r = row()
    rec = a3.row_record(r, "۱", gem(), a3.LabelRead("ویلایی"), "d")
    cells = a3.cell_records(r, rec, gem())
    by = {c["climate_class"]: c for c in cells}
    assert (by["5B"]["eui"], by["4B"]["eui"], by["0B"]["eui"]) == ("130", "116", "90")
    assert by["4B"]["climate_class_raw"] == "4B "
    assert "5C" not in by
    assert all(c["provenance"]["eui"] == "pdf_text_layer" for c in cells)
    assert all(not c["suspicious"] for c in cells)
    assert rec["suspicious"] == []


def test_visual_number_disagreement_flags_but_keeps_text_layer() -> None:
    r = row()
    rec = a3.row_record(r, "۱", gem(), None, "d")
    cells = a3.cell_records(r, rec, gem(values={"5B": "131", "4B": "116", "0B": "90"}))
    bad = [c for c in cells if c["suspicious"]]
    assert [(c["climate_class"], c["eui"]) for c in bad] == [("5B", "130")]


def test_label_flags() -> None:
    rec = a3.row_record(row(glyphs=0), "۱", None, None, "d")
    assert {"label_cell_empty", "label_missing"} <= {f["code"] for f in rec["suspicious"]}
    rec = a3.row_record(row(), "۱", gem(number="۴"), a3.LabelRead("ویلائی"), "d")
    assert {"row_alignment", "label_reads_disagree"} <= {f["code"] for f in rec["suspicious"]}
    rec = a3.row_record(row(), "۱", gem(), a3.LabelRead(None), "d")
    assert [f["code"] for f in rec["suspicious"]] == ["label_uncorroborated"]


def test_operator_rows_carry_the_source_issue() -> None:
    rec = a3.row_record(row(order=1), "۱", gem("مسکونی ≤۳۰۰۰ مترمربع", "۱"),
                        a3.LabelRead("مسکونی ≤۳۰۰۰ مترمربع"), "d")  # fmt: skip
    assert rec["source_issue_refs"] == ["SI-0003"]
    assert rec["suspicious"] == []


def test_near_duplicates() -> None:
    found = a3.near_duplicates([("a", "اداری عمومی"), ("b", "اداری خصوصی"), ("c", "بیمارستان"),
                                ("d", "بیمارستان ")])  # fmt: skip
    kinds = {(f["a"], f["b"]): f["kind"] for f in found}
    assert kinds[("c", "d")] == "duplicate"
    assert ("a", "c") not in kinds


def _rows(labels: dict[str, list[str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for rating, names in labels.items():
        for i, name in enumerate(names, 1):
            issues = ["SI-0003"] if i == 1 else []
            out.append({"energy_rating": rating, "row_order": i, "occupancy": name,
                        "suspicious": [], "source_issue_refs": issues})  # fmt: skip
    return out


def test_cross_rating_order_flags_differing_positions_and_operator_rows() -> None:
    from mabhas19.tables import app03_build as b  # noqa: PLC0415

    rows = _rows({"D": ["مسکونی ≥ ۳۰۰۰", "هتل"], "C": ["مسکونی ۳۰۰۰", "هتل"],
                  "B": ["مسکونی ≥ ۳۰۰۰", "هتل"], "A": ["مسکونی ≥ ۳۰۰۰", "هتل"]})  # fmt: skip
    order = b.cross_rating_order(rows)
    assert order["same_row_count"]
    assert order["identical_positions"] == 1
    assert not order["all_identical_in_order"]
    b.flag_cross_rating(rows, order)
    codes = {f["code"] for r in rows if r["row_order"] == 1 for f in r["suspicious"]}
    assert codes == {"cross_rating_label_differs", "operator_ambiguous"}
    assert all(not r["suspicious"] for r in rows if r["row_order"] == 2)
