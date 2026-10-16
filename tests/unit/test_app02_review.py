"""Appendix 2 review decisions and promotion validation."""

from __future__ import annotations

import copy
from typing import Any

import pytest

from mabhas19.tables import app02 as a2
from mabhas19.tables import app02_decisions as dec
from mabhas19.tables import app02_promote as pr


def rec(number: int, city: str, code: str = "0B", second: str | None = None) -> dict[str, Any]:
    row = a2.TextRow(2, number, str(number), code, 100.0 + number, (1.0, 2.0, 3.0, 4.0), 1)
    gem = a2.VisualRead(city, code, str(number), {"method": "gemini_whole_page"})
    ocr = a2.VisualRead(second if second is not None else city, None, None, {"method": "t"})
    return a2.build_record(row, "۱", gem, ocr, "d")


def test_set_changes_city_only_and_keeps_raw_and_machine_value() -> None:
    machine = [rec(3, "آغاجاری", second="آقاجاری")]
    d = {"id": "A2-r03", "rows": [machine[0]["candidate_id"]], "decision": "set",
         "basis": "assistant_reviewed",
         "set": [{"field": "city", "was": "آغاجاری", "now": "آقاجاری"}]}  # fmt: skip
    [out] = dec.apply_decisions(machine, [d])
    assert (out["city"], out["city_raw"], out["city_machine"]) == ("آقاجاری", "آغاجاری", "آغاجاری")
    assert out["climate_code"] == "0B"
    assert out["review"]["basis"] == ["assistant_reviewed"]
    assert machine[0]["city"] == "آغاجاری"


def test_stale_and_code_decisions_are_refused(tmp_path: Any) -> None:
    machine = [rec(3, "اراک")]
    change = {"field": "city", "was": "x", "now": "y"}
    d = {"id": "X", "rows": [machine[0]["candidate_id"]], "decision": "set",
         "basis": "assistant_reviewed", "set": [change]}  # fmt: skip
    with pytest.raises(dec.DecisionError, match="stale"):
        dec.apply_decisions(machine, [d])
    path = tmp_path / "d.yaml"
    path.write_text("basis_default: assistant_reviewed\ndecisions:\n- {id: X, rows: [r], "
                    "decision: set, set: [{field: climate_code, was: 5C, now: 5B}]}\n",
                    encoding="utf-8")  # fmt: skip
    with pytest.raises(dec.DecisionError, match="cannot be changed"):
        dec.load_all((path,))


def table(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    reviewed = dec.apply_decisions(records, [])
    return [pr.curated_record(r) for r in reviewed], records


def test_corroborated_rows_get_machine_status_and_5c_needs_source_issues() -> None:
    curated, machine = table([rec(1, "آبادان"), rec(2, "اردبیل", "5C ")])
    assert [c["verification"]["status"] for c in curated] == ["machine_corroborated"] * 2
    problems = pr.validate(curated, machine)
    assert not any("5C" in p or "status" in p for p in problems)
    broken = copy.deepcopy(curated)
    broken[1]["source_issue_refs"] = []
    assert any("SI-0001" in p for p in pr.validate(broken, machine))


def test_validation_blocks_code_drift_unreviewed_flags_and_duplicates() -> None:
    curated, machine = table([rec(1, "کرج"), rec(2, "كرج"), rec(3, "اراک", second="ارات")])
    curated[0]["climate_class"]["raw"] = "0A"
    problems = pr.validate(curated, machine)
    assert any("differs from the text layer" in p for p in problems)
    assert any("duplicated" in p for p in problems)
    assert any("without a review decision" in p for p in problems)
    assert any("76" in p for p in problems)
