"""Final-5 authoring validation: an excluded true record is never a no-record expectation."""

from __future__ import annotations

from functools import cache
from typing import Any

import pytest

from mabhas19.cli.eval_retrieval import EARLIER_SETS, SETS
from mabhas19.core.paths import CURATED_DIR
from mabhas19.retrieval.eval import load_cases
from mabhas19.retrieval.final_validation import (
    VALIDATED_SETS,
    excluded_records_of,
    issue_records,
    validate_final_cases,
)
from mabhas19.retrieval.structured import StructuredStore


@cache
def _store() -> StructuredStore:
    return StructuredStore(CURATED_DIR)


def _records() -> set[str]:
    s = _store()
    return {r["id"] for r in s.app02 + s.app03 + s.app07} | {
        r["record_id"] for t in s.main.values() for r in t.records
    }


def _check(cases: list[dict[str, Any]], excluded: set[str] | None = None) -> list[str]:
    return validate_final_cases(
        cases, _records(), issue_records(_store().issues), excluded or set()
    )


def test_the_final4_artifacts_would_be_rejected_and_nothing_else() -> None:
    # Final-4 stays frozen; the rules are only applied here to show what they catch
    earlier = [
        load_cases(SETS / f"{s}.yaml") for s in ("dev", "heldout", "final", "final2", "final3")
    ]
    problems = _check(load_cases(SETS / "final4.yaml"), excluded_records_of(earlier))
    assert {p.split(":")[0] for p in problems} == {"L338", "L342"}


def test_historical_sets_are_never_validated_with_the_final5_rules() -> None:
    assert VALIDATED_SETS == ("final5", "final6", "final7", "final8")
    assert "final4" not in EARLIER_SETS
    assert set(EARLIER_SETS["final5"]) == {"dev", "heldout", "final", "final2", "final3", "final4"}
    assert set(EARLIER_SETS["final6"]) == {*EARLIER_SETS["final5"], "final5"}
    assert set(EARLIER_SETS["final7"]) == {*EARLIER_SETS["final6"], "final6"}
    assert set(EARLIER_SETS["final8"]) == {*EARLIER_SETS["final7"], "final7"}


def _case(**evidence: Any) -> dict[str, Any]:
    return {"id": "X1", "expected_route": "structured", "query": "q", "evidence": evidence}


@pytest.mark.parametrize(
    "case",
    [
        _case(records=["main-t19-3-2-r21"]),
        _case(no_records=True, no_match_outcomes=["unknown_entity"], tables=["t"]),
        _case(tables=["m19-ed5-1404-main.t19-5-1"]),  # a table-level question
        {"id": "P1", "expected_route": "prose", "query": "q", "evidence": {"blocks": ["b"]}},
    ],
)
def test_valid_cases_pass(case: dict[str, Any]) -> None:
    assert _check([case]) == []


def test_a_named_true_record_cannot_become_a_no_record_case() -> None:
    case = _case(source_issues=["SI-0010"])
    case["notes"] = "The Kerman row (main-t19-3-2-r21) is excluded"
    problems = _check([case])
    assert any("names the curated record" in p for p in problems)
    assert any("located on record" in p for p in problems)


def test_a_record_level_issue_without_its_record_is_rejected() -> None:
    assert any("located on record" in p for p in _check([_case(conflict="SI-0009")]))


def test_no_records_needs_a_typed_outcome() -> None:
    assert any("typed" in p for p in _check([_case(no_records=True)]))


def test_an_excluded_record_may_be_the_answer_when_it_is_flagged() -> None:
    excluded = {"main-t19-3-2-r21"}
    case = _case(records=["main-t19-3-2-r21"])
    assert any("uses_excluded_records" in p for p in _check([case], excluded))
    case["uses_excluded_records"] = True
    assert _check([case], excluded) == []
    clean = _case(records=["main-t19-3-2-r04"])
    clean["uses_excluded_records"] = True
    assert any("but no excluded record" in p for p in _check([clean], excluded))
