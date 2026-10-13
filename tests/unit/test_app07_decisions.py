"""Owner-review decisions: applied on top of the machine layer, guarded, labels only."""

from __future__ import annotations

from typing import Any

import pytest

from mabhas19.tables import app07_decisions as dec


def rec(rid: str, **fields: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "candidate_id": rid,
        "category": "۱۰. صفحات سیمانی",
        "subgroup": None,
        "material": "الیافی",
        "condition": None,
        "condition_parts": [],
        "suspicious": [],
        "subgroup_footnote_refs": [],
        "parsed_conductivity": "0.95",
    }
    return {**base, **fields}


def doc(*decisions: dict[str, Any]) -> dict[str, Any]:
    return {"decisions": [{"basis": dec.OWNER, **d} for d in decisions]}


def test_set_replaces_label_keeps_machine_value_and_input_untouched() -> None:
    machine = [rec("r1", condition="ضخامت بیش از ۶۰")]
    d = {
        "id": "C09",
        "rows": ["r1"],
        "decision": "set",
        "set": [
            {
                "field": "condition",
                "was": "ضخامت بیش از ۶۰",
                "now": "هوا؛ ضخامت بیش از ۶۰",
                "parts": [
                    {"text": "هوا", "role": "sub_heading"},
                    {"text": "ضخامت بیش از ۶۰", "role": "leaf"},
                ],
            }
        ],
    }
    [out] = dec.apply_decisions(machine, doc(d))
    assert out["condition"] == "هوا؛ ضخامت بیش از ۶۰"
    assert out["condition_machine"] == "ضخامت بیش از ۶۰"
    assert out["condition_source"] == dec.OWNER
    assert out["review"]["basis"] == [dec.OWNER]
    assert out["review"]["status"] == dec.REVIEWED
    assert machine[0]["condition"] == "ضخامت بیش از ۶۰"  # machine layer not modified
    assert out["parsed_conductivity"] == "0.95"


def test_stale_decision_is_refused() -> None:
    d = {
        "id": "C06",
        "rows": ["r1"],
        "decision": "set",
        "set": [{"field": "subgroup", "was": "something else", "now": None}],
    }
    with pytest.raises(dec.DecisionError, match="stale"):
        dec.apply_decisions([rec("r1")], doc(d))


def test_numeric_fields_cannot_be_decided(tmp_path: Any) -> None:
    path = tmp_path / "d.yaml"
    path.write_text(
        "basis_default: project_owner_reviewed\n"
        "decisions:\n- {id: X, rows: [r1], decision: set, set: "
        "[{field: parsed_conductivity, was: '0.95', now: '0.9'}]}\n",
        encoding="utf-8",
    )
    with pytest.raises(dec.DecisionError, match="cannot be changed"):
        dec.load_decisions(path)


def test_unresolved_wins_and_unknown_rows_fail() -> None:
    ds = doc(
        {"id": "C", "rows": ["r1"], "decision": "confirm"},
        {"id": "Q", "rows": ["r1"], "decision": "unresolved", "question": "subgroup?"},
    )
    [out] = dec.apply_decisions([rec("r1")], ds)
    assert out["review"]["status"] == dec.UNRESOLVED
    assert out["review"]["open_questions"] == ["subgroup?"]
    with pytest.raises(dec.DecisionError, match="unknown rows"):
        dec.apply_decisions(
            [rec("r1")], doc({"id": "C", "rows": ["r9"], "decision": "confirm", "basis": dec.OWNER})
        )


def test_display_name_is_derived_without_touching_hierarchy() -> None:
    [cat] = dec.apply_decisions([rec("r1")], doc())
    assert cat["display_name"] == "صفحات سیمانی الیافی"
    assert (cat["category"], cat["material"]) == ("۱۰. صفحات سیمانی", "الیافی")
    assert cat["review"]["status"] == dec.NOT_REVIEWED
    [sub] = dec.apply_decisions([rec("r1", subgroup="بتن سبک", material="متراکم")], doc())
    assert sub["display_name"] == "بتن سبک متراکم"


def test_committed_decision_files_keep_their_bases_apart() -> None:
    owner, assistant = (dec.load_decisions(p) for p in dec.DECISION_FILES)
    assert owner["reviewed_by"] == "project_owner"
    assert {d["basis"] for d in owner["decisions"]} == {dec.OWNER, dec.MODEL}
    assert {d["basis"] for d in assistant["decisions"]} == {dec.ASSISTANT}
    assert all(d["decision"] == "confirm" for d in assistant["decisions"])
    assert sum(len(d["rows"]) for d in assistant["decisions"]) == 76


def test_bases_are_kept_side_by_side_with_precedence() -> None:
    ds = doc(
        {"id": "M", "rows": ["r1"], "decision": "confirm", "basis": dec.MODEL},
        {"id": "O", "rows": ["r1"], "decision": "confirm", "basis": dec.OWNER},
    )
    [out] = dec.apply_decisions([rec("r1")], ds)
    dec.add_benchmark_basis([out], {"r1"})
    assert out["review"]["basis"] == [dec.BENCHMARK, dec.OWNER, dec.MODEL]
    assert out["review"]["primary_basis"] == dec.BENCHMARK


def test_unknown_basis_is_rejected(tmp_path: Any) -> None:
    path = tmp_path / "d.yaml"
    path.write_text("decisions:\n- {id: X, rows: [r1], decision: confirm}\n", encoding="utf-8")
    with pytest.raises(dec.DecisionError, match="basis"):
        dec.load_decisions(path)
