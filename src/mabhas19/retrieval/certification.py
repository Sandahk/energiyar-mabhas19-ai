"""Result certification: after execution, independently prove every returned row answers the
admitted plan before it is passed on (defense in depth).

The check never calls the executor's matching function: it reads the printed row fields itself
(by the schema paths), compares them with the admission certificate
(:class:`~mabhas19.retrieval.admission.PlanAdmission`) using its own comparisons, and fails
closed on anything it cannot verify. A row is certified only when

* its table is the admitted table, and an explicitly named table (``explicit_targets``) is never
  left: a row of another table is a hard ``explicit_target_leak`` failure;
* its printed label carries every admitted entity word and no excluded phrase;
* every admitted constraint holds on the printed field (categorical equality, exact Decimal
  equality for exact values, containment for printed intervals);
* reverse lookups return rows whose printed output value satisfies the stated value under the
  output's ``value_semantics`` (``exact``: equality; ``interval``: containment);
* **I9** the admitted output (named, or the schema default) is printed on the row - a dash or
  empty cell is not a value, and another output never stands in for it unless the schema
  declares it an alternative of the same requirement (the returned field then names it);
  its printed unit is the admitted unit, and a unit the question asks for is that unit;
* **I5** when other identity words of the question name rows of a printed family, the row
  shares a printed level with them (re-derived here from the row's printed level texts);
* **I5** the row prints every printed condition phrase the question uses positively
  («بدون دودکش»);
* **I6** the row prints no category value a correction of the question rejected;
* **I10** every stated value of a regulatory dimension the table does not print lies inside the
  table's declared scope (:mod:`mabhas19.retrieval.table_scope`);
* no calculation was requested (no row answers a refused operation);
* provenance (source cells / source location) is present.

Any failure removes every row of that plan; the failures are surfaced as typed safety failures
in the result metadata and counted by the evaluator.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import TYPE_CHECKING, Any

from mabhas19.retrieval import table_scope, units
from mabhas19.retrieval.entities import FUNCTION_WORDS, keys
from mabhas19.retrieval.identity import query_tokens, row_label
from mabhas19.retrieval.numeric import (
    CLOSED,
    OPEN,
    Bound,
    NumInterval,
    find_quantities,
    interval_from_operators,
    printed_interval,
)
from mabhas19.retrieval.schema import APP03, APP07, MAIN, Output, TableSchema
from mabhas19.text.normalize import normalize_for_search

if TYPE_CHECKING:
    from mabhas19.retrieval.admission import PlanAdmission
    from mabhas19.retrieval.plan import Constraint
    from mabhas19.retrieval.structured import StructuredStore

_DASHES = {"-", "–", "—", "_", ""}


@dataclass
class CertificationFailure:
    record: str
    table: str
    check: str
    detail: str

    def to_dict(self) -> dict[str, str]:
        return {
            "record": self.record,
            "table": self.table,
            "check": self.check,
            "detail": self.detail,
        }


def _get(r: dict[str, Any], path: str) -> Any:
    cur: Any = r
    for part in path.split("."):
        cur = cur.get(part) if isinstance(cur, dict) else None
    return cur


def _dec(x: Any) -> Decimal | None:
    try:
        return Decimal(str(x).strip()) if x not in (None, "") else None
    except (InvalidOperation, ValueError):
        return None


def _bound_ok(x: Decimal, b: Bound | None, *, lower: bool) -> bool:
    if b is None:
        return True
    if x == b.value:
        return b.kind != OPEN
    return x > b.value if lower else x < b.value


def _in(x: Decimal, iv: NumInterval) -> bool:
    """x lies in iv (closed and unspecified ends include the bound, open ones exclude it)."""
    return _bound_ok(x, iv.lo, lower=True) and _bound_ok(x, iv.hi, lower=False)


def _meets(a: NumInterval, b: NumInterval) -> bool:
    """The two intervals share at least one value."""
    lo = a.lo if b.lo is None or (a.lo is not None and a.lo.value >= b.lo.value) else b.lo
    hi = a.hi if b.hi is None or (a.hi is not None and a.hi.value <= b.hi.value) else b.hi
    if lo is None or hi is None:
        return True
    if lo.value < hi.value:
        return True
    return lo.value == hi.value and OPEN not in (lo.kind, hi.kind)


def _point(iv: NumInterval) -> Decimal | None:
    if iv.lo is not None and iv.hi is not None and iv.lo.value == iv.hi.value:
        return iv.lo.value
    return None


def certify(  # noqa: PLR0912, PLR0915
    store: StructuredStore,
    schemas: dict[str, TableSchema],
    admission: PlanAdmission,
    dataset: str,
    record: dict[str, Any],
) -> list[CertificationFailure]:
    rid = str(record.get("id") or record.get("record_id"))
    fails: list[CertificationFailure] = []

    def fail(check: str, detail: str) -> None:
        fails.append(CertificationFailure(rid, dataset, check, detail))

    if admission.explicit_targets and dataset not in admission.explicit_targets:
        fail("explicit_target_leak", f"the question names {admission.explicit_targets}")
    if dataset != admission.target:
        fail("table", f"admitted {admission.target}")
    if not admission.admitted:
        fail("not_admitted", "a row of a plan that was never admitted")
    if admission.calculation:
        fail("calculation", f"a refused {admission.calculation} has no regulatory row")
    schema = schemas[dataset]
    # entity and exclusions against the printed label
    label = row_label(store, dataset, record) if schema.family != "place" else ""
    label_keys = {k for w in query_tokens(label) for k in keys(w)}
    if schema.family == "place":
        place = normalize_for_search(str(_get(record, "city.text") or ""))
        joined = place.replace(" ", "")
        for w in admission.entity_words:
            if w.replace(" ", "") not in joined and joined not in w.replace(" ", ""):
                fail("entity", f"row prints «{place}», admitted «{w}»")
    else:
        missing = [w for w in admission.entity_words if not keys(w) & label_keys]
        if missing:
            fail("entity", f"label «{label}» lacks admitted words {missing}")
    for phrase in admission.exclusions:
        content = [w for w in phrase if w not in FUNCTION_WORDS]
        if content and all(keys(w) & label_keys for w in content):
            fail("exclusion", f"label «{label}» prints the excluded «{' '.join(phrase)}»")
    # I5: every printed identity phrase the question uses positively is the row's own
    for phrase in admission.required_phrases:
        content = [w for w in phrase if w not in FUNCTION_WORDS]
        if content and not all(keys(w) & label_keys for w in content):
            fail("identity_phrase", f"label «{label}» does not print «{' '.join(phrase)}»")
    # constraints, independently re-checked on the printed fields
    for c in admission.constraints:
        if c.state != "applied":
            continue
        ok = _constraint_holds(store, schema, dataset, record, c)
        if ok is None:
            fail("uncheckable", f"{c.describe()} cannot be verified on the printed row")
        elif not ok:
            fail("constraint", f"{c.describe()} does not hold on the printed row")
    # I9: exactly the admitted output is printed (or a declared alternative of it), in its unit
    asked = admission.output or admission.requested_field
    if asked is not None and schema.output(asked) is not None:
        printed = certified_output(schema, asked, record)
        if printed is None:
            fail("no_printed_value", f"the row prints no {asked} (nor a declared alternative)")
        else:
            name, unit = printed
            admitted = units.parse(admission.unit) if admission.unit else None
            row_unit = units.parse(unit) if unit else None
            if name == asked and admitted and row_unit and not units.same(admitted, row_unit):
                fail("unit", f"the row prints {name} in «{unit}», admitted «{admission.unit}»")
            requested = units.parse(admission.requested_unit) if admission.requested_unit else None
            if requested is not None and not (
                (row_unit or admitted) and units.compatible(requested, row_unit or admitted)  # type: ignore[arg-type]
            ):
                fail("unit", f"«{admission.requested_unit}» was asked; the row prints «{unit}»")
    # I5: the printed family the question's other identity words name
    if admission.family_levels:
        mine = set(printed_levels(store, dataset, record))
        if not mine & set(admission.family_levels):
            fail(
                "identity_family",
                f"the row's printed levels {sorted(mine)[:3]} share none with "
                f"{admission.family_levels[:3]}",
            )
    # I6: no rejected value; I10: stated values inside the declared scope
    for dim, values in admission.rejected_values.items():
        if _prints_value(schema, record, dim, values):
            fail("rejected_value", f"the row prints {dim} {values}, which the question rejects")
    for dim, values in admission.scope_values.items():
        v = table_scope.verdict(schema, dim, values)
        if v.blocks:
            fail("scope", v.detail)
    # provenance
    if not (record.get("source_cells") or record.get("source") or record.get("provenance")):
        fail("provenance", "no source cells or source location")
    return fails


def _prints_value(schema: TableSchema, record: dict[str, Any], dim: str, values: list[str]) -> bool:
    """The row prints one of ``values`` in its own column of ``dim`` (none: it cannot)."""
    d = schema.dim(dim)
    printed = _get(record, d.path) if d is not None else None
    if isinstance(printed, dict):
        printed = printed.get("code") or printed.get("printed")
    if printed is None:
        return False
    got = [str(x) for x in printed] if isinstance(printed, list) else [str(printed)]
    norm = {normalize_for_search(x).replace(" ", "").upper() for x in got}
    return bool(norm & {normalize_for_search(v).replace(" ", "").upper() for v in values})


def _printed(record: dict[str, Any], o: Output) -> bool:
    return str(_get(record, o.path) or "").strip() not in _DASHES


def _printed_unit(record: dict[str, Any], o: Output) -> str | None:
    """The unit printed with an output's cell (falls back to the table's declared unit)."""
    cell = record.get(o.path.split(".")[0])
    unit = cell.get("unit") if isinstance(cell, dict) else None
    return str(unit) if unit else o.unit


def certified_output(
    schema: TableSchema, asked: str, record: dict[str, Any]
) -> tuple[str, str | None] | None:
    """(the output the row prints for an admitted output, its printed unit): the output itself,
    else a declared alternative; None when the row prints neither."""
    o = schema.output(asked)
    if o is None:
        return None
    for x in (o, *(schema.output(a) for a in o.alternatives)):
        if x is not None and _printed(record, x):
            return x.name, _printed_unit(record, x)
    return None


def printed_levels(store: StructuredStore, ds: str, record: dict[str, Any]) -> list[str]:
    """The printed name levels of a row (category, subgroup, material; an occupancy label; a
    main table's label fields), normalized, list numbering removed."""
    from mabhas19.retrieval.structured import LABEL_FIELDS, _field  # noqa: PLC0415

    if ds == APP07:
        texts = [
            str((record.get(k) or {}).get("text") or "")
            for k in ("category", "subgroup", "material")
        ]
    elif ds == APP03:
        label = record.get("label")
        if label is None:
            label = store.occupancies.get(str(record.get("occupancy_id")), {}).get("label", "")
        texts = [str(label)]
    else:
        fields = LABEL_FIELDS.get(ds.removeprefix(MAIN), ())
        texts = [x for path, _ in fields if (x := _field(record, path))]
    out = []
    for t in texts:
        norm = re.sub(r"^\s*\d+\s*[.٫]\s*", "", normalize_for_search(t)).strip()
        if norm:
            out.append(norm)
    return out


def _constraint_holds(  # noqa: PLR0911, PLR0912
    store: StructuredStore,
    schema: TableSchema,
    ds: str,
    r: dict[str, Any],
    c: Constraint,
) -> bool | None:
    """Does a constraint hold on the printed row? None when it cannot be checked."""
    dim = c.dimension
    d = schema.dim(dim)
    if dim in ("climate_class", "rating", "group", "year", "wmo_code", "province"):
        if d is None:
            return None
        printed = _get(r, d.path)
        if dim == "rating" and isinstance(printed, dict):
            printed = printed.get("code") or printed.get("printed")
        if printed is None:
            return dim == "climate_class" and ds not in (APP03,)  # tables without the column
        got = {str(x) for x in printed} if isinstance(printed, list) else {str(printed)}
        norm = {normalize_for_search(x).replace(" ", "").upper() for x in got}
        want = {normalize_for_search(str(v)).replace(" ", "").upper() for v in c.values}
        return bool(norm & want)
    iv = c.interval
    if iv is None and c.ratio is None:
        return None
    if ds == APP07 and dim == "density":
        printed_iv = printed_interval(normalize_for_search(str(r["density"]["raw"])))
        return printed_iv is not None and iv is not None and _meets(printed_iv, iv)
    if dim.startswith("condition:") and ds == APP07:
        name = dim.split(":", 1)[1]
        from mabhas19.retrieval.planner import condition_quantities  # noqa: PLC0415

        named = [q for n, q in condition_quantities(r) if n == name]
        if not named:
            return False
        if c.ratio is not None:
            return any(q.ratio and (str(q.ratio[0]), str(q.ratio[1])) == c.ratio for q in named)
        return iv is not None and all(_meets(q.interval, iv) for q in named)
    if dim == "r_ratio":
        return iv is not None and _covers(interval_from_operators(r["r_range"]), iv)
    if dim.startswith("degree_days:"):
        var = dim.split(":", 1)[1]
        conds = [x for x in r["criterion"]["conditions"] if x["variable"] == var]
        return (
            bool(conds)
            and iv is not None
            and all(_covers(interval_from_operators(x), iv) for x in conds)
        )
    if dim == "area" and ds == APP03:
        label = normalize_for_search(str(store.occupancies.get(r["occupancy_id"], {}).get("label")))
        printed_iv = printed_interval(label)
        return printed_iv is None or (iv is not None and _meets(printed_iv, iv))
    if dim.startswith("label:"):
        unit = dim.split(":", 1)[1]
        label = row_label(store, ds, r)
        printed = [q for q in find_quantities(label, printed=True) if q.unit == unit]
        if not printed or iv is None:
            return not printed
        x = _point(iv)
        plain = [q for q in printed if not q.negated]
        neg = [q for q in printed if q.negated]
        if x is not None:
            return (not plain or any(_in(x, q.interval) for q in plain)) and not any(
                _in(x, q.interval) for q in neg
            )
        return not plain or any(_meets(q.interval, iv) for q in plain)
    out = schema.output(dim)
    if out is not None and iv is not None:
        value = _dec(_get(r, out.path))
        if value is None:
            return False
        if out.value_semantics == "exact":
            return _in(value, iv)  # a stated point: exact Decimal equality, never «nearby»
        return None
    return None


def _covers(printed: NumInterval, stated: NumInterval) -> bool:
    """A stated value (point or range) lies within a printed interval, every bound compared with
    its inclusivity: «3000 < x ≤ 4000» stated lies within the printed «3000 < HDD ≤ 4000»."""
    x = _point(stated)
    if x is not None:
        return _in(x, printed)
    return _bound_within(stated.lo, printed.lo, lower=True) and _bound_within(
        stated.hi, printed.hi, lower=False
    )


def _bound_within(stated: Bound | None, printed: Bound | None, *, lower: bool) -> bool:
    """A stated interval end is not looser than the printed one."""
    if printed is None:
        return True
    if stated is None:
        return False
    if stated.value != printed.value:
        return stated.value > printed.value if lower else stated.value < printed.value
    return printed.kind != OPEN or stated.kind == OPEN


__all__ = ["CLOSED", "CertificationFailure", "certified_output", "certify", "printed_levels"]
