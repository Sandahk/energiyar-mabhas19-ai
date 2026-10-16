"""Appendix 3 label decisions and promotion validation."""

from __future__ import annotations

import copy
from typing import Any

import pytest

from mabhas19.tables import app03 as a3
from mabhas19.tables import app03_decisions as dec
from mabhas19.tables import app03_promote as pr
from mabhas19.text.normalize import to_persian_digits


def make(
    rating: str, order: int, label: str, *, agree: bool = True
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    values = {
        "5B": str(100 * (5 - a3.RATINGS.index(rating))),
        "0B": str(90 - a3.RATINGS.index(rating)),
    }
    row = a3.TableRow(rating, 2, order, 100.0, values,
                      {k: (0.0, 0.0, 1.0, 1.0) for k in values}, {"5B": 0, "0B": 1}, order + 1,
                      (0.0, 0.0, 1.0, 1.0), (1.0, 0.0, 2.0, 1.0), 1)  # fmt: skip
    gem = a3.LabelRead(label, str(order), {"5B": values["5B"], "0B": values["0B"]})
    rec = a3.row_record(row, "۱", gem, a3.LabelRead(label if agree else None), "d")
    return rec, a3.cell_records(row, rec, gem)


def test_set_label_keeps_raw_and_machine_and_refuses_stale() -> None:
    rec, _ = make("C", 1, "مسکونی ۳۰۰۰ مترمربع")
    resolution = {"SI-0003": {"status": "resolved_by_assistant_visual_review"}}
    d = {"id": "A3-r01", "rows": [rec["row_id"]], "decision": "set_label",
         "basis": "assistant_reviewed", "label": "مسکونی ≥ ۳۰۰۰ مترمربع",
         "was": {rec["row_id"]: "مسکونی ۳۰۰۰ مترمربع"},
         "source_issue_resolution": resolution}  # fmt: skip
    [out] = dec.apply_decisions([rec], [d])
    assert out["occupancy"] == "مسکونی ≥ ۳۰۰۰ مترمربع"
    assert (out["occupancy_raw"], out["occupancy_machine"]) == ("مسکونی ۳۰۰۰ مترمربع",) * 2
    assert out["source_issue_refs"] == ["SI-0003"]  # the issue reference stays
    assert (
        out["source_issue_resolution"]["SI-0003"]["status"] == "resolved_by_assistant_visual_review"
    )
    stale = {**d, "was": {rec["row_id"]: "something else"}}
    with pytest.raises(dec.DecisionError, match="stale"):
        dec.apply_decisions([rec], [stale])


def table() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows, cells = [], []
    for k in a3.RATINGS:
        for n in range(1, pr.ROWS_PER_TABLE + 1):
            rec, cs = make(k, n, f"کاربری {to_persian_digits(str(n))}")
            rows.append(rec)
            cells += cs
    return dec.apply_decisions(rows, []), cells


def test_valid_table_only_lacks_si0003_resolution_and_full_columns() -> None:
    rows, cells = table()
    types = pr.occupancy_types(rows)
    eui = pr.eui_records(cells, types)
    assert {t["verification"]["status"] for t in types} == {"machine_corroborated"}
    assert all("strict_D>C>B>A" in e["verification"]["value_corroboration"] for e in eui)
    problems = pr.validate(types, eui, cells, rows)
    # rows 1-2 carry SI-0003 without a recorded resolution; the test table has 2 of 9 columns
    assert {p.split(": ", 1)[-1] for p in problems} == {
        "SI-0003 row without a recorded resolution",
        "272 EUI records",
        "a rating/occupancy pair does not have exactly 9 climate values",
    }


def test_validation_blocks_changed_values_and_5c() -> None:
    rows, cells = table()
    types = pr.occupancy_types(rows)
    eui = copy.deepcopy(pr.eui_records(cells, types))
    eui[0]["max_eui"]["value"] = "999"
    eui[1]["climate_class"] = "5C"
    problems = pr.validate(types, eui, cells, rows)
    assert any("differs from the text-layer candidate" in p for p in problems)
    assert any("5C never filled" in p for p in problems)


def test_unreviewed_flagged_position_has_no_status() -> None:
    rows, cells = table()
    rows[0]["review"]["corroboration"] = None
    types = pr.occupancy_types(rows)
    assert types[0]["verification"]["status"] is None
    assert any(
        "no verification status" in p
        for p in pr.validate(types, pr.eui_records(cells, types), cells, rows)
    )
