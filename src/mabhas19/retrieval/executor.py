"""Execution of validated structured query plans (fail closed, never broadened).

:func:`Executor.run` refuses any plan the gate rejects, selects rows only through the plan's
interpreted constraints (category membership and :mod:`~mabhas19.retrieval.numeric` interval
relations) and finally re-checks every returned record against every applied constraint with an
independent evaluation: a contradiction returns zero records (``contradiction``), so a planner
error can never surface a wrong regulatory value.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING, Any

from mabhas19.retrieval.models import StructuredHit, StructuredNoMatch
from mabhas19.retrieval.numeric import (
    NumInterval,
    interval_from_operators,
    printed_interval,
)
from mabhas19.retrieval.plan import Constraint, PlanOutcome, StructuredQueryPlan, validate
from mabhas19.retrieval.schema import APP02, APP03, APP07, MAIN, TableSchema
from mabhas19.text.normalize import normalize_for_search

if TYPE_CHECKING:
    from mabhas19.retrieval.structured import StructuredStore

MAX_HITS = 60
_NO_PRINTED_REASON = {
    "rating_range": "no_applicable_category",
    "criterion": "no_printed_condition",
}


def _get(r: dict[str, Any], path: str) -> Any:
    cur: Any = r
    for part in path.split("."):
        cur = cur.get(part) if isinstance(cur, dict) else None
    return cur


def climate_values(ds: str, r: dict[str, Any]) -> list[str] | None:
    """Printed climate classes of a record (None: the record has no climate dimension)."""
    if ds in (APP02, MAIN + "19-3-2"):
        return [str(r["climate_class"]["code"])]
    if ds == APP03:
        return [str(r["climate_class"])]
    groups = [v["applies_to"] for v in r.values() if isinstance(v, dict) and "applies_to" in v]
    return [c for g in groups for c in g] if groups else None


def rating_value(ds: str, r: dict[str, Any]) -> str | None:
    rating = r.get("energy_rating")
    if isinstance(rating, str):
        return rating
    if isinstance(rating, dict):
        if rating.get("code"):
            return str(rating["code"])
        if rating.get("value_class") == "category" and rating.get("printed"):
            return str(rating["printed"])
    del ds
    return None


def density_interval(r: dict[str, Any]) -> NumInterval | None:
    return printed_interval(normalize_for_search(r["density"]["raw"]))


def condition_quantities(r: dict[str, Any]) -> list[tuple[str, Any]]:
    """Named numbers of an Appendix 7 condition (shared with the planner)."""
    from mabhas19.retrieval.planner import condition_quantities as named  # noqa: PLC0415

    return named(r)


def enumeration_value(  # noqa: PLR0911
    store: StructuredStore, ds: str, r: dict[str, Any], dimension: str
) -> str:
    """The value a record prints for an enumerated dimension (its distinguishing label)."""
    if dimension == "climate_class":
        return ", ".join(climate_values(ds, r) or [])
    if dimension == "rating":
        return rating_value(ds, r) or ""
    if dimension == "group":
        return str(_get(r, "building_group.text") or "")
    if dimension == "year":
        return str(_get(r, "year.text") or "")
    if dimension in ("occupancy", "area") and ds == APP03:
        return str(store.occupancies.get(r["occupancy_id"], {}).get("label", ""))
    if dimension == "density" and ds == APP07:
        return str(r["density"]["raw"])
    if dimension == "material" and ds == APP07:
        cond = (r.get("condition") or {}).get("text") or ""
        return f"{r['display_name']} {cond}".strip()
    if dimension == "row_condition":
        from mabhas19.retrieval.planner import _label_texts  # noqa: PLC0415

        t = store.main.get(ds)
        return " ".join(_label_texts(t, r)) if t else ""
    return ""


class Executor:
    def __init__(self, store: StructuredStore, schemas: dict[str, TableSchema]) -> None:
        self.store = store
        self.schemas = schemas

    # ------------------------------------------------------------------ entry
    def run(
        self, plan: StructuredQueryPlan
    ) -> tuple[list[StructuredHit], list[StructuredNoMatch], list[str]]:
        schema = self.schemas[plan.target]
        notes = list(plan.notes)
        outcome = validate(plan, schema)
        if outcome is not None:
            cc = plan.constraint("climate_class")
            if plan.target == APP03 and cc is not None and "5C" in cc.values:
                from mabhas19.retrieval.structured import NO_5C_NOTE  # noqa: PLC0415

                notes.append(NO_5C_NOTE)
            return [], [self._no_match(plan, outcome)], notes
        records = self._candidates(plan, schema)
        records, extra = self._filter(plan, schema, records)
        notes += extra
        # a row printing no value (a dash or an empty cell) answers nothing: typed, never returned
        printed = [r for r in records if _prints_a_value(schema, r)]
        if records and not printed:
            dashes = ", ".join(_rid(r) for r in records)
            return (
                [],
                [
                    self._no_match(
                        plan,
                        PlanOutcome(
                            "no_printed_category",
                            f"the source prints no value (a dash or empty cell) for {dashes}",
                            "no_printed_value",
                        ),
                    )
                ],
                notes,
            )
        records = printed
        if not records:
            reason = _NO_PRINTED_REASON.get(schema.family, "no_matching_row")
            detail = "no printed row satisfies every interpreted constraint: " + "; ".join(
                c.describe() for c in plan.constraints if c.state == "applied"
            )
            return (
                [],
                [self._no_match(plan, PlanOutcome("no_printed_category", detail, reason))],
                notes,
            )
        bad = [
            (rid, c.describe())
            for r in records
            for c in plan.constraints
            if c.state == "applied" and (rid := _rid(r)) and not self.satisfies(plan.target, r, c)
        ]
        # an enumeration returns only rows of its resolved scope, each carrying its own value of
        # the enumerated dimension
        for e in plan.enumerations:
            for r in records:
                value = enumeration_value(self.store, plan.target, r, e.dimension)
                if not value or (e.members and value not in e.members):
                    bad.append((_rid(r), f"enumeration {e.dimension}: {value!r}"))
            notes.append(
                f"enumeration of printed alternatives ({e.dimension}; asked by «{e.request}»"
                + (f"; within «{e.parent}»" if e.parent else "")
                + f"): {len(e.members)} printed values, each result carries its own"
            )
        if bad:
            outcome = PlanOutcome(
                "contradiction",
                f"returned rows contradict applied constraints {bad}; nothing is returned",
                "contradiction",
            )
            return [], [self._no_match(plan, outcome)], notes
        why = f"{plan.operation}: " + "; ".join(
            [c.describe() for c in plan.constraints if c.state == "applied"]
            + [e.text for e in plan.entities if e.status == "resolved" and e.text]
        )
        hits = [self._hit(plan, r, why) for r in records[:MAX_HITS]]
        return hits, [], notes

    # ------------------------------------------------------------------ rows
    def _candidates(self, plan: StructuredQueryPlan, schema: TableSchema) -> list[dict[str, Any]]:
        s = self.store
        ds = plan.target
        if schema.family == "place":
            places = next(
                (e.ids for e in plan.entities if e.kind == "place" and e.status == "resolved"), []
            )
            if places:
                return [r for p in places for d, r in s.places[p] if d == ds]
            return list(s.app02 if ds == APP02 else s.main[ds].records)
        if ds == APP03:
            occ = set(plan.row_ids or [])
            return [r for r in s.app03 if r["occupancy_id"] in occ]
        if ds == APP07:
            ids = set(plan.row_ids or [])
            return [r for r in s.app07 if r["id"] in ids]
        t = s.main[ds]
        if plan.row_ids is None:
            return (
                list(t.records)
                if plan.operation in ("whole_table", "reverse", "existence")
                or schema.family in ("criterion", "rating_range")
                else []
            )
        ids = set(plan.row_ids)
        return [r for r in t.records if r["record_id"] in ids]

    def _filter(
        self, plan: StructuredQueryPlan, schema: TableSchema, records: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], list[str]]:
        notes: list[str] = []
        ds = plan.target
        for c in plan.constraints:
            if c.state != "applied":
                continue
            if ds == APP07 and c.dimension == "density" and c.interval is not None:
                rels = {
                    _rid(r): (density_interval(r) or NumInterval(None, None)).relation(c.interval)
                    for r in records
                }
                inside = [r for r in records if rels[_rid(r)] in ("within", "boundary")]
                if any(rels[_rid(r)] == "boundary" for r in inside):
                    notes.append(
                        "density equals a printed bound whose inclusivity the source does not "
                        "state: each row touching that bound is returned"
                    )
                if not inside and not c.interval.is_point:
                    inside = [r for r in records if rels[_rid(r)] == "overlap"]
                    if inside:
                        notes.append(
                            f"stated density {c.interval.describe()} spans several printed ranges; "
                            "each overlapping printed row is returned, none is interpolated"
                        )
                records = inside
                continue
            records = [r for r in records if self.satisfies(ds, r, c)]
        return records, notes

    # ------------------------------------------------------------------ independent checks
    def satisfies(self, ds: str, r: dict[str, Any], c: Constraint) -> bool:  # noqa: PLR0911, PLR0912
        dim = c.dimension
        if dim == "climate_class":
            printed = climate_values(ds, r)
            return printed is None or bool(set(printed) & set(c.values))
        if dim == "rating":
            v = rating_value(ds, r)
            return v is not None and v in c.values
        if dim == "group":
            return str(_get(r, "building_group.text")) in c.values
        if dim == "year":
            return str(_get(r, "year.index")) in c.values
        if dim == "wmo_code":
            return str(_get(r, "wmo_code.ascii")) in c.values
        if dim == "province":
            return normalize_for_search(str(_get(r, "province.text"))) in c.values
        if c.interval is None and c.ratio is None:
            return False  # fail closed: a constraint this check cannot evaluate never passes
        stated = c.interval
        if dim == "density" and ds == APP07:
            iv = density_interval(r)
            return (
                stated is not None
                and iv is not None
                and iv.relation(stated) in ("within", "boundary", "overlap")
            )
        if dim == "thermal_conductivity":
            return _point_in(_dec(_get(r, "thermal_conductivity.value")), c.interval)
        if dim.startswith("condition:"):
            name = dim.split(":", 1)[1]
            named = [q for n, q in condition_quantities(r) if n == name]
            if not named:
                return False
            if c.ratio is not None:
                return any(q.ratio and (str(q.ratio[0]), str(q.ratio[1])) == c.ratio for q in named)
            return stated is not None and all(
                q.interval.relation(stated) in ("within", "boundary") for q in named
            )
        if dim == "r_ratio":
            assert c.interval is not None
            return interval_from_operators(r["r_range"]).relation(c.interval) == "within"
        if dim.startswith("degree_days:"):
            var = dim.split(":", 1)[1]
            conds = [x for x in r["criterion"]["conditions"] if x["variable"] == var]
            return (
                stated is not None
                and bool(conds)
                and all(interval_from_operators(x).relation(stated) == "within" for x in conds)
            )
        if dim.startswith("label:"):
            unit = dim.split(":", 1)[1]

            t = self.store.main.get(ds)
            if t is None:
                return False
            from mabhas19.retrieval.numeric import Quantity  # noqa: PLC0415
            from mabhas19.retrieval.planner import (  # noqa: PLC0415
                label_condition_ok,
                label_quantities,
            )

            if stated is None:
                return False
            claim = Quantity(stated, unit, 0, 0, c.source, ())
            return label_condition_ok(label_quantities(t, r), claim)
        if dim == "area" and ds == APP03:
            occ = self.store.occupancies.get(r["occupancy_id"], {})
            iv = printed_interval(normalize_for_search(str(occ.get("label", ""))))
            return iv is None or iv.relation(c.interval) in ("within", "boundary")  # type: ignore[arg-type]
        if dim == "max_eui":
            return _point_in(_dec(_get(r, "max_eui.value")), c.interval)
        schema = self.schemas[ds]
        out = schema.output(dim)
        if out is not None and out.kind == "number" and out.value_semantics == "exact":
            # an exact printed scalar: a stated point matches by Decimal equality only (a stated
            # range by containment) - never the nearest printed value
            return _point_in(_dec(_get(r, out.path)), c.interval)
        return False

    # ------------------------------------------------------------------ evidence
    def _hit(self, plan: StructuredQueryPlan, r: dict[str, Any], why: str) -> StructuredHit:
        hit = self.store._hit(plan.target, r, why)
        issues = [i for d in plan.derivations for i in d.issues]
        if issues:
            hit.source_issues = list(dict.fromkeys([*hit.source_issues, *issues]))
        return hit

    def _no_match(self, plan: StructuredQueryPlan, outcome: PlanOutcome) -> StructuredNoMatch:
        issues: list[str] = [i for d in plan.derivations for i in d.issues]
        if plan.target == APP03:
            cc = plan.constraint("climate_class")
            if cc is not None and cc.state == "invalid":
                for c in cc.values:
                    issues += self.store.app03_missing_issues.get(c, [])
            occ = next(
                (e.ids for e in plan.entities if e.kind == "occupancy" and e.status == "resolved"),
                [],
            )
            for o in occ:
                issues += list(self.store.occupancies.get(o, {}).get("source_issue_refs") or [])
        nm = self.store._no_match(
            plan.target,
            outcome.reason,
            outcome.detail,
            {"plan": _brief(plan)},
            list(dict.fromkeys(issues)),
        )
        nm.outcome = outcome.code
        return nm


def _prints_a_value(schema: TableSchema, r: dict[str, Any]) -> bool:
    """The row prints at least one value output (not a dash or an empty cell)."""
    for o in schema.outputs:
        if o.kind == "text":
            continue
        v = _get(r, o.path)
        if v is not None and str(v).strip() not in ("", "-", "–", "—"):
            return True
    return False


def _rid(r: dict[str, Any]) -> str:
    return str(r.get("id") or r.get("record_id"))


def _dec(x: Any) -> Decimal | None:
    try:
        return Decimal(str(x)) if x is not None else None
    except Exception:
        return None


def _point_in(x: Decimal | None, q: NumInterval | None) -> bool:
    if x is None or q is None:
        return False
    return q.contains(x) in ("yes", "boundary")


def _brief(plan: StructuredQueryPlan) -> dict[str, Any]:
    return {
        "operation": plan.operation,
        "requested_field": plan.requested_field,
        "constraints": [f"{c.describe()} [{c.state}]" for c in plan.constraints],
        "qualifiers": [f"{q.polarity}:{q.word} [{q.state}]" for q in plan.qualifiers],
        "entities": [f"{e.kind}:{e.text} [{e.status}]" for e in plan.entities],
        "unresolved": list(plan.unresolved),
        "computation": plan.computation,
        "enumerations": [f"{e.dimension} <- «{e.request}»" for e in plan.enumerations],
        "missing": list(plan.missing),
    }
