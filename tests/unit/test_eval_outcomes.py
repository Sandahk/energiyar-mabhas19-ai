"""Evaluator corrected-3: optional typed-outcome, calculation-refusal and enumeration scoring, and
the consistency checks a final set must pass before it is frozen."""

from __future__ import annotations

from typing import Any

from mabhas19.retrieval.eval import (
    CaseResult,
    _typed_outcomes,
    check_expectations,
    dropped_constraints,
)
from mabhas19.retrieval.models import (
    RetrievalResult,
    RouteDecision,
    StructuredHit,
    StructuredNoMatch,
)
from mabhas19.retrieval.plan import OUTCOMES

APP03 = "m19-ed5-1404-app03.max_energy_use_intensity"


def _result(
    records: list[str] | None = None,
    outcomes: list[str] | None = None,
    enumerations: list[dict[str, Any]] | None = None,
) -> RetrievalResult:
    hits = [StructuredHit(APP03, "d", i, 1, "1", [], {}, {}, [], "x") for i in records or []]
    nm = [StructuredNoMatch(APP03, "r", "", {}, None, outcome=o) for o in outcomes or []]
    r = RetrievalResult("q", "q", RouteDecision("structured", []), structured=hits, no_match=nm)
    r.enumerations = enumerations or []
    return r


def _case(record_exact: bool | None = None) -> CaseResult:
    res = CaseResult("C", "c", "q", "structured", "structured", [], True)
    res.record_exact = record_exact
    return res


def test_typed_outcome_and_calculation_refusal() -> None:
    ev = {"no_records": True, "no_match_outcomes": ["unsupported_calculation"]}
    ok = _case()
    _typed_outcomes(_result(outcomes=["unsupported_calculation"]), ev, ok)
    assert (ok.outcome_ok, ok.calculation_refusal) == (True, True)
    wrong_code = _case()
    _typed_outcomes(_result(outcomes=["missing_required_dimension"]), ev, wrong_code)
    assert (wrong_code.outcome_ok, wrong_code.calculation_refusal) == (False, False)
    with_rows = _case()
    _typed_outcomes(_result(records=["a"], outcomes=["unsupported_calculation"]), ev, with_rows)
    assert with_rows.outcome_ok is False


def test_outcome_scoring_only_where_stated() -> None:
    res = _case()
    _typed_outcomes(_result(outcomes=["unknown_entity"]), {"no_records": True}, res)
    assert (res.outcome_ok, res.calculation_refusal, res.enumeration_ok) == (None, None, None)


def test_enumeration_needs_exact_records_and_explicit_metadata() -> None:
    ev = {"records": ["a", "b"], "enumeration": {"dimension": "climate_class"}}
    meta = [{"table": APP03, "dimension": "climate_class", "basis": "explicit_request"}]
    ok = _case(record_exact=True)
    _typed_outcomes(_result(["a", "b"], enumerations=meta), ev, ok)
    assert ok.enumeration_ok is True
    no_meta = _case(record_exact=True)
    _typed_outcomes(_result(["a", "b"]), ev, no_meta)
    assert no_meta.enumeration_ok is False
    inexact = _case(record_exact=False)
    _typed_outcomes(_result(["a"], enumerations=meta), ev, inexact)
    assert inexact.enumeration_ok is False


def test_expectations_are_checked_before_freezing() -> None:
    cases = [
        {"id": "A", "expected_route": "structured", "evidence": {"records": ["x"]}},
        {"id": "B", "expected_route": "answer", "evidence": {}},
        {
            "id": "C",
            "expected_route": "structured",
            "evidence": {"no_records": True, "no_match_outcomes": ["made_up"]},
        },
        {
            "id": "D",
            "expected_route": "structured",
            "evidence": {"records": ["app03-A-r01-1B"], "enumeration": {"dimension": "size"}},
        },
        {"id": "E", "expected_route": "prose", "evidence": {"source_issues": ["SI-9999"]}},
        {
            "id": "F",
            "expected_route": "structured",
            "evidence": {"records": ["x"], "no_match_outcomes": ["unknown_entity"]},
        },
    ]
    enumerable = {APP03: {"climate_class", "rating", "occupancy", "area"}}
    problems = check_expectations(cases, {"SI-0001"}, enumerable, OUTCOMES)
    flagged = {p.split(":")[0] for p in problems}
    assert flagged == {"B", "C", "D", "E", "F"}


def _row(rid: str, climate: str, rating: str) -> StructuredHit:
    values = {"climate_class": climate, "energy_rating": rating}
    return StructuredHit(APP03, "d", rid, 1, "1", [], values, {}, [], "x")


def test_a_coordinated_members_constraint_binds_only_its_own_rows() -> None:  # corrected-6
    a, b = _row("a", "2B", "D"), _row("b", "1B", "C")
    r = RetrievalResult("q", "q", RouteDecision("structured", []), structured=[a, b])

    def entry(dim: str, value: str, rows: list[str] | None) -> dict[str, Any]:
        e = {"table": APP03, "constraint": dim, "dimension": dim, "value": [value]}
        return {**e, "state": "applied", **({"rows": rows} if rows is not None else {})}

    r.constraints = [entry("climate_class", "2B", ["a"]), entry("climate_class", "1B", ["b"])]
    assert dropped_constraints(r) == []
    # without member rows every returned row of the table must satisfy the constraint
    r.constraints = [entry("climate_class", "2B", None)]
    assert dropped_constraints(r) == ["climate_class: b contradicts ['2B']"]
