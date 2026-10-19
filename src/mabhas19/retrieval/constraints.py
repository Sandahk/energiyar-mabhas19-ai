"""Constraint ledger: every constraint extracted from a structured query ends in one state.

For each queried table and each extracted constraint the ledger records exactly one of:

* ``applied`` - the table prints that dimension and the lookup filtered by it;
* ``invalid`` - the value is outside the table's printed domain (a typed no-match says so);
* ``ambiguous`` - several printed rows/conditions fit (all are returned, with a note);
* ``unsupported_by_table`` - the table has no such dimension (it did not filter);
* ``unsupported_calculation`` - the request asks for a value computed from printed values.

No constraint may disappear silently; tests check the ledger on every development set and that
no returned record violates a constraint recorded as applied.
"""

from __future__ import annotations

from typing import Any

from mabhas19.retrieval.answerability import has_words
from mabhas19.retrieval.dimensions import (
    area_requested,
    density_requested,
    groups_requested,
    ratings_requested,
    years_requested,
)
from mabhas19.retrieval.models import RetrievalResult
from mabhas19.retrieval.plan import LEDGER_STATES
from mabhas19.retrieval.structured import (
    APP02,
    APP03,
    APP07,
    MAIN,
    climate_codes,
    degree_days_in,
    r_value_in,
    rating_code,
)

STATES = LEDGER_STATES  # the plan ledger's states (retrieval.plan)
# printed dimensions per dataset
CAPABILITIES: dict[str, frozenset[str]] = {
    APP02: frozenset(),
    APP03: frozenset({"climate_class", "rating", "area"}),
    APP07: frozenset({"density"}),
    MAIN + "19-2-1": frozenset({"group", "year"}),
    MAIN + "19-3-1": frozenset({"climate_class", "degree_days"}),
    MAIN + "19-3-2": frozenset(),
    MAIN + "19-3-3": frozenset({"rating", "r_value"}),
    MAIN + "19-5-1": frozenset({"climate_class"}),
    MAIN + "19-5-2": frozenset({"climate_class"}),
    MAIN + "19-5-3": frozenset({"rating"}),
    MAIN + "19-5-4": frozenset({"rating"}),
    MAIN + "19-5-5": frozenset(),
    MAIN + "19-5-6": frozenset({"area"}),
    MAIN + "19-5-7": frozenset(),
}
# typed no-match reasons that mark a constraint invalid
INVALID_REASONS: dict[str, frozenset[str]] = {
    "climate_class": frozenset({"climate_class_not_in_table", "missing_dimension"}),
    "rating": frozenset({"rating_not_in_table"}),
    "group": frozenset({"group_not_in_table"}),
    "year": frozenset({"year_not_in_table"}),
    "r_value": frozenset({"no_applicable_category"}),
    "degree_days": frozenset({"no_printed_condition"}),
    "area": frozenset({"no_matching_row"}),
    "density": frozenset({"no_matching_row"}),
}
AMBIGUITY_MARKERS = ("spans several", "size not stated", "several printed rows")


def extract_constraints(norm: str) -> dict[str, Any]:
    """Every constraint value stated in a structured query (before any table applies it)."""
    found: dict[str, Any] = {}
    if climate_codes(norm):
        found["climate_class"] = climate_codes(norm)
    if ratings_requested(norm):
        found["rating"] = ratings_requested(norm)
    if groups_requested(norm):
        found["group"] = groups_requested(norm)
    if years_requested(norm):
        found["year"] = years_requested(norm)
    area = area_requested(norm)
    if area is not None:
        found["area"] = area.describe()
    density = density_requested(norm)
    if density is not None:
        found["density"] = density.describe()
    r = r_value_in(norm)
    if r is not None:
        found["r_value"] = str(r)
    dd = degree_days_in(norm)
    if dd is not None:
        found["degree_days"] = f"{dd[0]} {dd[1]}"
    return found


def constraint_ledger(
    norm: str, tables: list[str], result: RetrievalResult, derivation: tuple[str, ...]
) -> list[dict[str, Any]]:
    """One entry per (queried table, extracted constraint) with exactly one state."""
    found = extract_constraints(norm)
    computed = bool(has_words(norm, derivation))
    ledger = []
    for table in tables:
        caps = CAPABILITIES.get(table, frozenset())
        reasons = {n.reason for n in result.no_match if n.dataset == table}
        ambiguous = any(m in note for note in result.notes for m in AMBIGUITY_MARKERS)
        for name, value in found.items():
            if computed:
                state = "unsupported_calculation"
            elif name not in caps:
                state = "unsupported_by_table"
            elif reasons & INVALID_REASONS.get(name, frozenset()):
                state = "invalid"
            elif ambiguous and name in ("area", "density"):
                state = "ambiguous"
            else:
                state = "applied"
            ledger.append({"table": table, "constraint": name, "value": value, "state": state})
    return ledger


def violates(record: dict[str, Any], name: str, value: Any) -> bool:
    """True when a returned record contradicts an applied constraint (used by the tests)."""
    if name == "climate_class":
        printed = record.get("climate_class")
        if isinstance(printed, str):
            return printed not in value
        groups = [
            v["applies_to"] for v in record.values() if isinstance(v, dict) and "applies_to" in v
        ]
        return bool(groups) and not any(c in g for g in groups for c in value)
    if name == "rating":
        code = rating_code(record)
        return code is not None and code not in value
    if name == "group" and "building_group" in record:
        return str(record["building_group"]["text"]) not in {str(v) for v in value}
    if name == "year" and "year" in record:
        return str(record["year"]["index"]) not in {str(v) for v in value}
    return False
