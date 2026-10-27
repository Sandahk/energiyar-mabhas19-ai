"""Wrong-row attribution for the safety headline (evaluator side, independent of the runtime).

Every returned structured row a case does not allow (``eval.wrong_values``) is attributed to
exactly one class, using only the curated records and the case's own labels - never the plan,
the admission, the coverage ledger or the certification of the run being scored:

* **foreign_table_rows** - a row of a table the case does not name (its expected records,
  acceptable records and ``tables``), or any row at all when the case expects no structured
  answer: a second table leaked into the answer (unowned table admission);
* **refusal_bypassed[kind]** - a row returned where the case expects no record
  (``no_records``); ``kind`` comes from the case's ``refusal_kind``, else from its allowed
  ``no_match_outcomes`` refined by its ``category`` (corrected-6): quantity, unit, calculation,
  scope, family, reverse, entity, missing_dimension, constraint, no_records;
* **dimension_leak[dim]** - a row of an expected table that has the identity of an expected row
  but another value of a dimension (rating, climate class, group, year, density…): a constraint
  value the lookup did not own leaked into it;
* **identity_substitution** - a row of an expected table whose identity (occupancy, material,
  printed label, city…) is not the identity of any expected row: a dropped head noun or
  modifier selected another printed row.

Diagnostics (not wrong rows, corrected-6):

* **member_loss** - a case expecting rows of two or more distinct identities (coordinated
  members) gets rows of some of them and none of another: the missing member's expected rows.

The per-table identity and dimension fields below are declared here, from the curated record
layout, on purpose: the evaluator must not share the runtime's notion of identity. This module
reads curated JSONL directly and imports no retrieval runtime module
(``tests/unit/test_eval_isolation.py``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

APP02 = "m19-ed5-1404-app02.city_climate_classes"
APP03 = "m19-ed5-1404-app03.max_energy_use_intensity"
APP07 = "m19-ed5-1404-app07.material_thermal_properties"
MAIN = "m19-ed5-1404-main.t"

# dataset -> (identity field paths, {dimension: field path})
SPEC: dict[str, tuple[tuple[str, ...], dict[str, str]]] = {
    APP02: (("city.text",), {}),
    APP03: (("occupancy_id",), {"climate_class": "climate_class", "rating": "energy_rating"}),
    APP07: (
        ("category.text", "subgroup.text", "material.text"),
        {"condition": "condition.text", "density": "density.raw"},
    ),
    MAIN + "19-2-1": (
        ("requirement.text",),
        {"group": "building_group.text", "year": "year.index"},
    ),
    MAIN + "19-3-1": ((), {"climate_class": "climate_classes.raw"}),
    MAIN + "19-3-2": (("city.text",), {}),
    MAIN + "19-3-3": ((), {"rating": "energy_rating.code"}),
    MAIN + "19-5-1": (("element.text",), {"climate_class": "climate_group.raw"}),
    MAIN + "19-5-2": (("section.text", "row.text"), {"climate_class": "climate_group.raw"}),
    MAIN + "19-5-3": (("equipment.text", "specification.text", "sub_row"), {}),
    MAIN + "19-5-4": (("equipment.text", "specification.text", "sub_row"), {}),
    MAIN + "19-5-5": (("place_of_use.text",), {}),
    MAIN + "19-5-6": (("use.text",), {}),
    MAIN + "19-5-7": (("space.text",), {}),
}

# allowed no-match outcome -> refusal kind (first match in this order wins)
_REFUSAL_KINDS = (
    ("unsupported_output", "quantity"),
    ("unsupported_calculation", "calculation"),
    ("unsupported_reverse_lookup", "reverse"),
    ("invalid_constraint", "scope"),
    ("unknown_entity", "entity"),
    ("ambiguous_entity", "entity"),
    ("missing_required_dimension", "missing_dimension"),
    ("unresolved_constraint", "constraint"),
    ("no_printed_category", "constraint"),
)
# (kind from the outcome, case category) -> the finer kind the case's own label states
_CATEGORY_KINDS = {
    ("calculation", "unit_request"): "unit",
    ("entity", "family_wording"): "family",
    ("entity", "polarity_qualifier"): "family",
}


def dataset_of(record_id: str) -> str:
    """The curated dataset of a record ID (by its printed prefix)."""
    if record_id.startswith("app02-"):
        return APP02
    if record_id.startswith("app03-"):
        return APP03
    if record_id.startswith("app07-"):
        return APP07
    if record_id.startswith("main-t"):
        table = record_id.removeprefix("main-t")
        return MAIN + table.rsplit("-", 1)[0]
    return ""


@cache
def load_records(curated_dir: Path) -> dict[str, dict[str, Any]]:
    """Every curated table record by ID, read directly from ``curated/tables``."""
    out: dict[str, dict[str, Any]] = {}
    for path in sorted((curated_dir / "tables").glob("*/**/*.jsonl")):
        if path.name == "cells.jsonl":
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            rid = r.get("id") or r.get("record_id")
            if rid:
                out[str(rid)] = r
    return out


def _get(r: dict[str, Any], path: str) -> Any:
    cur: Any = r
    for part in path.split("."):
        cur = cur.get(part) if isinstance(cur, dict) else None
    return cur


def _text(x: Any) -> str:
    if isinstance(x, dict):
        x = x.get("code") or x.get("text") or x.get("printed")
    return " ".join(str(x).split()) if x is not None else ""


def identity(ds: str, record_id: str, r: dict[str, Any]) -> tuple[str, ...]:
    """The record's identity (what it is about), independent of its dimension values."""
    spec = SPEC.get(ds)
    if spec is None:
        return (record_id,)
    return tuple(_text(_get(r, p)) for p in spec[0])


def dimensions(ds: str, r: dict[str, Any]) -> dict[str, str]:
    spec = SPEC.get(ds)
    return {d: _text(_get(r, p)) for d, p in spec[1].items()} if spec else {}


def allowed_tables(ev: dict[str, Any]) -> set[str]:
    out = set(ev.get("tables", []))
    for rid in [*ev.get("records", []), *ev.get("acceptable_records", [])]:
        out.add(dataset_of(str(rid)))
    return out - {""}


def refusal_kind(ev: dict[str, Any], category: str = "") -> str:
    explicit = ev.get("refusal_kind")
    if explicit:
        return str(explicit)
    allowed = set(ev.get("no_match_outcomes") or [])
    kind = next((kind for code, kind in _REFUSAL_KINDS if code in allowed), "no_records")
    return _CATEGORY_KINDS.get((kind, category), kind)


@dataclass
class Attribution:
    foreign_table_rows: list[str] = field(default_factory=list)
    refusal_bypassed: dict[str, list[str]] = field(default_factory=dict)
    dimension_leaks: dict[str, list[str]] = field(default_factory=dict)
    identity_substitutions: list[str] = field(default_factory=list)


def attribute(
    ev: dict[str, Any],
    wrong: list[str],
    records: dict[str, dict[str, Any]],
    category: str = "",
) -> Attribution:
    """Attribute each wrong row of a case to exactly one class (see the module docstring)."""
    out = Attribution()
    tables = allowed_tables(ev)
    expected = [str(x) for x in [*ev.get("records", []), *ev.get("acceptable_records", [])]]
    for rid in wrong:
        ds = dataset_of(rid)
        if ds not in tables:
            out.foreign_table_rows.append(rid)
            continue
        if ev.get("no_records"):
            out.refusal_bypassed.setdefault(refusal_kind(ev, category), []).append(rid)
            continue
        r = records.get(rid, {})
        mine = identity(ds, rid, r)
        same = [
            e
            for e in expected
            if dataset_of(e) == ds and identity(ds, e, records.get(e, {})) == mine
        ]
        if not same:
            out.identity_substitutions.append(rid)
            continue
        dims = dimensions(ds, r)
        differing = [
            d
            for d, v in dims.items()
            if all(dimensions(ds, records.get(e, {})).get(d) != v for e in same)
        ] or ["other"]
        for d in differing:
            out.dimension_leaks.setdefault(d, []).append(rid)
    return out


def member_loss(
    ev: dict[str, Any], returned: list[str], records: dict[str, dict[str, Any]]
) -> list[str]:
    """Expected rows of a coordinated member (an identity of the expected rows) none of which is
    returned while another member's rows are: a member lost from a several-entity answer."""
    members: dict[tuple[str, tuple[str, ...]], list[str]] = {}
    for rid in (str(x) for x in ev.get("records", [])):
        ds = dataset_of(rid)
        members.setdefault((ds, identity(ds, rid, records.get(rid, {}))), []).append(rid)
    if len(members) < 2:
        return []
    got = set(returned)
    if not any(set(ids) & got for ids in members.values()):
        return []
    return sorted(rid for ids in members.values() if not set(ids) & got for rid in ids)


__all__ = [
    "Attribution",
    "allowed_tables",
    "attribute",
    "dataset_of",
    "load_records",
    "member_loss",
    "refusal_kind",
]
