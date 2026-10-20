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
* the requested output is printed on the row (a dash or empty cell is not a value);
* no calculation was requested (no row answers a refused operation);
* provenance (source cells / source location) is present.

Any failure removes every row of that plan; the failures are surfaced as typed safety failures
in the result metadata and counted by the evaluator.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import TYPE_CHECKING, Any

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
from mabhas19.retrieval.schema import APP03, APP07, TableSchema
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


def certify(  # noqa: PLR0912
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
    # constraints, independently re-checked on the printed fields
    for c in admission.constraints:
        if c.state != "applied":
            continue
        ok = _constraint_holds(store, schema, dataset, record, c)
        if ok is None:
            fail("uncheckable", f"{c.describe()} cannot be verified on the printed row")
        elif not ok:
            fail("constraint", f"{c.describe()} does not hold on the printed row")
    # the requested output is printed (or, where rows print one of several values, another one)
    values = [
        o
        for o in schema.outputs
        if o.kind != "text" and str(_get(record, o.path) or "").strip() not in _DASHES
    ]
    asked = schema.output(admission.requested_field or "")
    if asked is not None and asked not in values and not values:
        fail("no_printed_value", f"the row prints no {asked.name}")
    # provenance
    if not (record.get("source_cells") or record.get("source") or record.get("provenance")):
        fail("provenance", "no source cells or source location")
    return fails


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
    """A stated value (point or range) lies within a printed interval."""
    x = _point(stated)
    if x is not None:
        return _in(x, printed)
    lo_ok = (stated.lo is not None and _in(stated.lo.value, printed)) or (
        stated.lo is None and printed.lo is None
    )
    hi_ok = (stated.hi is not None and _in(stated.hi.value, printed)) or (
        stated.hi is None and printed.hi is None
    )
    return bool(lo_ok and hi_ok)


__all__ = ["CLOSED", "CertificationFailure", "certify"]
