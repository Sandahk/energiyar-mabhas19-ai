"""Final-test authoring validation (Final-5 onward).

The exclusion pool (every evidence ID of the earlier sets) exists for authoring independence: an
excluded record or section may not *seed* a new case. It never changes what a case's correct
answer is. Final-4 broke that: two cases whose true answers were excluded records were frozen as
structured cases with no expected records (L338, L342), so the true records scored as wrong
values. From Final-5 on, when independent authoring produces a question whose answer is an
excluded record, the case lists that record (and sets ``uses_excluded_records: true``) or is
dropped before freeze; it is never turned into a no-record expectation.

The checks are deterministic and read only the case and the curated source - never the
retrieval candidate under test, whose answers must not shape the benchmark:

* a structured or mixed case with no expected records states why: either a typed no-match
  (``no_records`` with ``no_match_outcomes``) or a table-level question (``tables``);
* such a case never names a curated record ID (in its query, notes, anchor or evidence);
* nor cites a source issue or conflict that the curated issue register locates on records
  (a conflict is between records; a record-level anomaly is exposed on its record);
* a ``no_records`` case states its typed outcome;
* a case using an excluded record says so (``uses_excluded_records: true``), and only then.

Historical sets (dev … Final-4) are never validated with these rules and stay unchanged.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

VERSION = "final-validation-1"
# the sets these rules apply to (later final sets are added here when they are created)
VALIDATED_SETS = ("final5", "final6", "final7")
_RECORD_ID = re.compile(r"(?:app0[237]-[a-z0-9-]+|main-t19-\d+-\d+-[a-z0-9-]+)", re.IGNORECASE)
STRUCTURED_ROUTES = ("structured", "mixed")


def _text_fields(case: dict[str, Any]) -> Iterable[str]:
    for key in ("query", "notes", "anchor"):
        value = case.get(key)
        if isinstance(value, str):
            yield value
    stack: list[Any] = [case.get("evidence") or {}]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)
        elif isinstance(item, str):
            yield item


def issue_records(issues: dict[str, Any]) -> dict[str, set[str]]:
    """Source issue ID -> the records the curated issue register locates it on."""
    out: dict[str, set[str]] = {}
    for iid, issue in issues.items():
        locs = issue.get("locations", []) if isinstance(issue, dict) else []
        out[iid] = {
            str(loc["record"]) for loc in locs if isinstance(loc, dict) and loc.get("record")
        }
    return out


def validate_final_cases(
    cases: list[dict[str, Any]],
    record_ids: set[str],
    issue_locations: dict[str, set[str]],
    excluded_records: set[str],
) -> list[str]:
    """Problems that block freezing a Final-5+ set (empty = valid)."""
    problems: list[str] = []
    for c in cases:
        cid = c.get("id", "?")
        ev = c.get("evidence") or {}
        records = set(ev.get("records", [])) | set(ev.get("acceptable_records", []))
        routes = {c.get("expected_route"), *c.get("acceptable_routes", [])}
        structured = bool(routes & set(STRUCTURED_ROUTES))
        if ev.get("no_records") and not ev.get("no_match_outcomes"):
            problems.append(f"{cid}: no_records without a typed no_match_outcomes")
        if structured and not records:
            if not ev.get("no_records") and not ev.get("tables"):
                problems.append(
                    f"{cid}: structured case with no expected records states neither a typed "
                    "no-match (no_records + no_match_outcomes) nor a table-level question"
                )
            named = sorted(
                {m for t in _text_fields(c) for m in _RECORD_ID.findall(t) if m in record_ids}
            )
            if named:
                problems.append(
                    f"{cid}: names the curated record(s) {named} but expects no records - list "
                    "the true record(s) or drop the case (an excluded record is never turned "
                    "into a no-record expectation)"
                )
            cited = sorted({i for i in [*ev.get("source_issues", []), ev.get("conflict")] if i})
            located = sorted({r for i in cited for r in issue_locations.get(i, set())})
            if located:
                problems.append(
                    f"{cid}: cites {cited}, located on record(s) {located}, but expects no "
                    "records - a true structured record exists; list it or drop the case"
                )
        uses = sorted(records & excluded_records)
        flagged = bool(c.get("uses_excluded_records"))
        if uses and not flagged:
            problems.append(
                f"{cid}: expects excluded record(s) {uses}; mark uses_excluded_records: true "
                "(allowed only when independent authoring led to them, never as a seed)"
            )
        if flagged and not uses:
            problems.append(f"{cid}: uses_excluded_records set but no excluded record expected")
    return problems


def excluded_records_of(sets: Iterable[list[dict[str, Any]]]) -> set[str]:
    """Every record ID used as evidence by earlier sets (the authoring-independence pool)."""
    out: set[str] = set()
    for cases in sets:
        for c in cases:
            ev = c.get("evidence") or {}
            out |= set(ev.get("records", [])) | set(ev.get("acceptable_records", []))
    return out
