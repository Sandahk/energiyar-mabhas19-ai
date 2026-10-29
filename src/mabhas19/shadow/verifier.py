"""Field-by-field deterministic verification of a proposal (independent of model quality).

A proposal is never evidence: each field is checked against the candidate context and the
deterministic facts of the production run, and every check is a deterministic function of
those. Three kinds of failure, in this order:

* **integrity** (status ``invalid``) - a span whose offsets or text do not match its context
  item (invented offsets, a value reference pointing at another span), an index out of range, a
  duplicate, an identifier outside every supplied list (candidate row, table, output), an
  invented coordination member. A proposal that fabricates is not salvaged: nothing of it is
  used;
* **required field** (status ``rejected``) - clause roles, entity uses, value roles (I11
  completeness: every stated value has a role), polarity, the operation and coordination. A
  required field that fails rejects the whole semantic plan: the homograph rule (a grammatical
  homograph alone never names a row), I6 ownership (a value / exclusion belongs to the clause
  that holds it; an exclusion or rejected reading is kept and its values are never lookup
  constraints), I10 scope (a lookup constraint some requested table can carry), the calculation
  rule (a detected calculation is never removed), I8 (every deterministic member is listed);
* **optional field** (status ``partial``) - a candidate row, a request's output or unit. A
  failing optional field is *replaced* by ``ambiguous``, which the translation turns into a
  refusal of the tables it concerns (fail closed); the model's raw value is never used.

Accepted fields are only labels: the deterministic pipeline still decides every row.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from mabhas19.retrieval import table_scope, units
from mabhas19.retrieval.plan import concept
from mabhas19.shadow.proposal import (
    AMBIGUOUS,
    CALCULATIONS,
    NONE,
    EntityLabel,
    Proposal,
    Request,
)

if TYPE_CHECKING:
    from mabhas19.retrieval.schema import TableSchema
    from mabhas19.shadow.context import CandidateContext, DeterministicFacts

VERIFIER_VERSION = "sh0-verifier-1"
REGULATORY = frozenset(table_scope.REGULATORY)
INVALID, REJECTED, PARTIAL, VERIFIED = "invalid", "rejected", "partial", "verified"


@dataclass(frozen=True)
class Decision:
    field: str  # clauses | entities | values | polarity | operation | requests | coordination
    item: str  # item index / table ID ("" for the field as a whole)
    outcome: str  # accepted | rejected | replaced | invalid
    check: str  # the deterministic check that decided
    reason: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "field": self.field,
            "item": self.item,
            "outcome": self.outcome,
            "check": self.check,
            "reason": self.reason,
        }


@dataclass
class Verification:
    status: str  # verified | partial | rejected | invalid
    decisions: list[Decision] = field(default_factory=list)
    # the proposal as verified (optional rejected fields replaced by «ambiguous»); None unless
    # verified or partial
    proposal: Proposal | None = None
    calculation_added: bool = False
    opportunities: list[dict[str, str]] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        return self.status in (VERIFIED, PARTIAL)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": VERIFIER_VERSION,
            "status": self.status,
            "decisions": [d.to_dict() for d in self.decisions],
            "calculation_added": self.calculation_added,
            "opportunities": list(self.opportunities),
        }


def _concept(name: str) -> str:
    return {"energy_rating": "rating"}.get(name, concept(name))


class _Run:
    def __init__(self, ctx: CandidateContext, facts: DeterministicFacts) -> None:
        self.ctx, self.facts = ctx, facts
        self.decisions: list[Decision] = []

    def add(self, fld: str, item: object, outcome: str, check: str, reason: str = "") -> None:
        self.decisions.append(Decision(fld, str(item), outcome, check, reason))

    def bad(self, outcome: str) -> bool:
        return any(d.outcome == outcome for d in self.decisions)


def verify(
    proposal: Proposal,
    ctx: CandidateContext,
    facts: DeterministicFacts,
    schemas: dict[str, TableSchema],
) -> Verification:
    run = _Run(ctx, facts)
    _integrity(run, proposal)
    if run.bad("invalid"):
        return Verification(INVALID, run.decisions)
    _completeness(run, proposal)
    _clauses(run, proposal)
    _values(run, proposal, schemas)
    _polarity(run, proposal)
    calc_added = _operation(run, proposal)
    _coordination(run, proposal)
    entities, opportunities = _entities(run, proposal)
    requests, more = _requests(run, proposal, schemas)
    opportunities += more
    if run.bad("rejected"):
        return Verification(REJECTED, run.decisions, opportunities=opportunities)
    verified = replace(proposal, entities=entities, requests=requests)
    status = PARTIAL if run.bad("replaced") else VERIFIED
    return Verification(status, run.decisions, verified, calc_added, opportunities)


# ---------------------------------------------------------------- integrity


def _integrity(run: _Run, p: Proposal) -> None:  # noqa: PLR0912
    ctx = run.ctx
    q = ctx.query

    def span_ok(  # noqa: PLR0913, PLR0917
        fld: str, i: int, items: tuple[Any, ...], s: int, e: int, text: str | None
    ) -> None:
        if i >= len(items):
            run.add(fld, i, "invalid", "index", "no such context item")
            return
        it = items[i]
        if (it.start, it.end) != (s, e):
            run.add(
                fld, i, "invalid", "offsets", f"[{s},{e}) is not the item's [{it.start},{it.end})"
            )
        elif text is not None and text != q[s:e]:
            run.add(fld, i, "invalid", "text", "the echoed text is not the query text at the span")

    def unique(fld: str, idx: list[int | str]) -> None:
        dup = sorted({str(x) for x in idx if idx.count(x) > 1})
        if dup:
            run.add(fld, ",".join(dup), "invalid", "duplicate", "an item is labelled twice")

    for c in p.clauses:
        span_ok("clauses", c.index, ctx.clauses, c.start, c.end, None)
    unique("clauses", [c.index for c in p.clauses])
    known = ctx.candidate_ids()
    for e in p.entities:
        span_ok("entities", e.index, ctx.entities, e.start, e.end, e.text)
        if e.candidate not in (NONE, AMBIGUOUS) and e.candidate not in known:
            run.add("entities", e.index, "invalid", "candidate_list", "candidate not supplied")
    unique("entities", [e.index for e in p.entities])
    for v in p.values:
        span_ok("values", v.index, ctx.values, v.start, v.end, v.text)
    unique("values", [v.index for v in p.values])
    for x in p.polarity:
        span_ok("polarity", x.index, ctx.polarity, x.start, x.end, None)
    unique("polarity", [x.index for x in p.polarity])
    clause_refs = [v.clause for v in p.values] + [x.clause for x in p.polarity]
    if any(c is not None and c >= len(ctx.clauses) for c in clause_refs):
        run.add("values", "", "invalid", "index", "a clause reference outside the partition")
    if any(r >= len(ctx.operations) for r in p.operation.refs):
        run.add("operation", "", "invalid", "index", "an operation reference outside the context")
    for r in p.requests:
        t = ctx.table(r.table)
        if t is None:
            run.add("requests", r.table, "invalid", "table_list", "table not supplied")
            continue
        if r.output not in (NONE, AMBIGUOUS) and r.output not in {o.name for o in t.outputs}:
            run.add("requests", r.table, "invalid", "output_list", f"no output {r.output!r}")
        if isinstance(r.unit, int) and r.unit >= len(ctx.units):
            run.add("requests", r.table, "invalid", "index", "no such unit expression")
    unique("requests", [r.table for r in p.requests])
    supplied = {c.table: {(m.start, m.end, m.text) for m in c.members} for c in ctx.coordination}
    for co in p.coordination:
        if co.table not in supplied:
            run.add("coordination", co.table, "invalid", "coordination_list", "none supplied")
            continue
        if any((m.start, m.end, m.text) not in supplied[co.table] for m in co.members):
            run.add("coordination", co.table, "invalid", "offsets", "a member is not supplied")
    unique("coordination", [co.table for co in p.coordination])


# ---------------------------------------------------------------- required fields


def _completeness(run: _Run, p: Proposal) -> None:
    ctx = run.ctx
    for fld, given, total in (
        ("clauses", {c.index for c in p.clauses}, len(ctx.clauses)),
        ("entities", {e.index for e in p.entities}, len(ctx.entities)),
        ("values", {v.index for v in p.values}, len(ctx.values)),
        ("polarity", {x.index for x in p.polarity}, len(ctx.polarity)),
    ):
        missing = sorted(set(range(total)) - given)
        if missing:
            check = "I11_completeness" if fld == "values" else "completeness"
            run.add(fld, ",".join(map(str, missing)), "rejected", check, "items left unlabelled")
        else:
            run.add(fld, "", "accepted", "completeness")


def _clauses(run: _Run, p: Proposal) -> None:
    # roles are semantic labels: the partition (offsets, completeness) is the check
    for c in p.clauses:
        run.add("clauses", c.index, "accepted", "partition")


def _containing(items: tuple[Any, ...], start: int, end: int, kinds: tuple[str, ...]) -> bool:
    return any(x.kind in kinds and x.start <= start and end <= x.end for x in items)


def _values(run: _Run, p: Proposal, schemas: dict[str, TableSchema]) -> None:
    ctx = run.ctx
    requested = [r.table for r in p.requests]
    for v in p.values:
        item = ctx.values[v.index]
        if v.clause != ctx.clause_of(v.start, v.end):
            run.add("values", v.index, "rejected", "I6_ownership", "owned by another clause")
            continue
        # an exclusion and a correction may print the same span («…، نه رده D»): either role
        rejected = _containing(ctx.polarity, v.start, v.end, ("exclusion", "correction_rejected"))
        if (v.role in ("excluded", "corrected_away")) != rejected:
            run.add(
                "values", v.index, "rejected", "I6_polarity", "disagrees with the polarity parse"
            )
            continue
        if v.role == "lookup_constraint" and item.dimension in REGULATORY:
            carried = [
                t
                for t in requested
                if not table_scope.verdict(schemas[t], item.dimension, [item.value]).blocks
            ]
            if not carried:
                run.add("values", v.index, "rejected", "I10_scope", "no requested table carries it")
                continue
        run.add("values", v.index, "accepted", "I6_I10")


def _polarity(run: _Run, p: Proposal) -> None:
    for x in p.polarity:
        if not x.kept:
            run.add("polarity", x.index, "rejected", "I6_polarity", "a parsed exclusion is removed")
        elif x.clause != run.ctx.clause_of(x.start, x.end):
            run.add("polarity", x.index, "rejected", "I6_ownership", "owned by another clause")
        else:
            run.add("polarity", x.index, "accepted", "I6_polarity")


def _operation(run: _Run, p: Proposal) -> bool:
    detected = run.ctx.operations
    kind = p.operation.kind
    if detected and kind not in (*CALCULATIONS, AMBIGUOUS):
        run.add("operation", kind, "rejected", "calculation", "a detected calculation is removed")
        return False
    added = kind in CALCULATIONS and not detected
    note = "calculation added" if added else ""
    if detected and kind in CALCULATIONS and kind not in {o.kind for o in detected}:
        note = "another calculation kind than the detector's (refused either way)"
    run.add("operation", kind, "accepted", "calculation", note)
    return added


def _coordination(run: _Run, p: Proposal) -> None:
    given = {c.table: {(m.start, m.end) for m in c.members} for c in p.coordination}
    for c in run.ctx.coordination:
        want = {(m.start, m.end) for m in c.members}
        if given.get(c.table) != want:
            run.add("coordination", c.table, "rejected", "I8", "a member is missing")
        else:
            run.add("coordination", c.table, "accepted", "I8")


# ---------------------------------------------------------------- entities (use required,
# candidate optional)


def _entities(run: _Run, p: Proposal) -> tuple[tuple[EntityLabel, ...], list[dict[str, str]]]:
    ctx, facts = run.ctx, run.facts
    out: list[EntityLabel] = []
    opportunities: list[dict[str, str]] = []
    for e in p.entities:
        span = ctx.entities[e.index]
        mine = {c.id: c for c in span.candidates}
        chosen = mine.get(e.candidate)
        exact = [c for c in span.candidates if c.status == "exact"]
        if e.use == "identity" and (
            (span.homograph and (not exact or (chosen is not None and chosen.status != "exact")))
            or all(c.status == "homograph_only" for c in span.candidates)
        ):
            run.add("entities", e.index, "rejected", "homograph", "a homograph alone names no row")
            out.append(e)
            continue
        run.add("entities", e.index, "accepted", "use")
        if e.candidate in (NONE, AMBIGUOUS):
            out.append(e)
            continue
        why = ""
        if chosen is None:
            why, check = "the candidate belongs to another span", "span_candidates"
        elif e.use != "identity":
            why, check = f"a candidate chosen for a {e.use} span", "use"
        elif chosen.status != "exact":
            why, check = f"the span names the row only as {chosen.status}", "grounding"
            if chosen.status == "head_only":
                opportunities.append({"kind": "would_disambiguate", "item": str(e.index)})
        else:
            tf = facts.per_table.get(chosen.table)
            rows = {c.row for s in ctx.entities for c in s.candidates if c.table == chosen.table}
            overlaps = tf is not None and any(a < e.end and e.start < b for a, b in tf.entity_spans)
            if (
                tf is not None
                and overlaps
                and tf.entity_ids & rows
                and chosen.row not in tf.entity_ids
            ):
                why, check = "the pipeline resolves the span to other rows", "I5_family"
        if why:
            run.add("entities", e.index, "replaced", check, why)
            out.append(replace(e, candidate=AMBIGUOUS))
        else:
            run.add("entities", e.index, "accepted", "grounding")
            out.append(e)
    return tuple(out), opportunities


# ---------------------------------------------------------------- requests (optional)


def _requests(  # noqa: PLR0912
    run: _Run, p: Proposal, schemas: dict[str, TableSchema]
) -> tuple[tuple[Request, ...], list[dict[str, str]]]:
    ctx = run.ctx
    found = units.find(run.facts.query or ctx.query)  # offsets equal; never redacted
    stated = {
        _concept(ctx.values[v.index].dimension) for v in p.values if v.role == "lookup_constraint"
    }
    out: list[Request] = []
    opportunities = []
    for r in p.requests:
        t = ctx.table(r.table)
        if t is not None and not t.selected:  # requested, but the router did not select it
            opportunities.append({"kind": "would_add_table", "item": r.table})
    for r in p.requests:
        req = r
        schema = schemas[r.table]
        if r.output not in (NONE, AMBIGUOUS):
            if r.intent == "lookup" and _concept(r.output) in stated:
                run.add("requests", r.table, "replaced", "I2", "asks for a quantity it states")
                req = replace(req, output=AMBIGUOUS)
            else:
                run.add("requests", r.table, "accepted", "output_schema")
        if isinstance(r.unit, int):
            expr = found[r.unit]
            o = schema.output(req.output) if req.output not in (NONE, AMBIGUOUS) else None
            printed = units.parse(o.unit) if o is not None and o.unit else None
            if expr.attached:
                why = "the unit of a stated number, not a request"
            elif printed is None:
                why = "no requested output with a printed unit"
            elif units.verdict(expr.unit, printed) != units.GROUND:
                why = f"{units.verdict(expr.unit, printed)}: not the printed unit"
            else:
                why = ""
            if why:
                run.add("requests", r.table, "replaced", "I4_unit", why)
                req = replace(req, unit=AMBIGUOUS)
            else:
                run.add("requests", r.table, "accepted", "I4_unit")
        out.append(req)
    return tuple(out), opportunities
