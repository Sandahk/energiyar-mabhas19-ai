"""Verified proposal -> restriction-only shadow plan -> counterfactual through the real pipeline.

Only verified components enter the :class:`ShadowPlan`, and in SH0 every lever it has can only
*restrict* what the deterministic pipeline does:

* **mask** - an entity span the proposal labels ``grammatical`` / ``contextual`` that the
  production run grounded as an entity, or that names a row exactly in the identity index, is
  removed from the query text (the identity reading is withdrawn; another grounded entity still
  names its row);
* **drop** - a table that is not requested, whose anchors all lie in clauses labelled
  ``background`` / ``explanation`` / ``unsupported``, or whose plan checks a stated claim the
  proposal labels as background or lookup constraint (not ``verify_request``) is removed from
  the router's decision;
* **refuse** - after the run, a table is refused (fail closed) when the proposal leaves
  something about it ambiguous (operation, intent, output, unit, entity), adds a calculation,
  disagrees with its plan's intent or output, or labels a value it applies as background or
  excluded.

The counterfactual runs the masked query through the *unchanged* production pipeline (router,
planner, admission, I1-I11 plan coverage, execution, certification) with the dropped tables
filtered from the routing decision - no stage is skipped for a narrowing proposal. A final
check refuses every row if the run returns one the production result does not contain (a
restriction must never add a row). Levers that could add rows (choosing among ambiguous
candidates, adding a table, dismissing context words) are recorded as opportunities only.

:func:`fallback_eligible` is the S1 architectural invariant: a future verified fallback may act
only where the deterministic structured path failed closed; it never replaces, adds to,
removes from or reinterprets a production answer.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, cast

from mabhas19.shadow.context import CandidateContext, DeterministicFacts, Pair
from mabhas19.shadow.context import _facts as facts_of
from mabhas19.shadow.proposal import AMBIGUOUS, NONE, REQUESTED_CLAUSES
from mabhas19.text.normalize import normalize_for_search

if TYPE_CHECKING:
    from mabhas19.retrieval.hybrid import Retriever
    from mabhas19.retrieval.models import RetrievalResult, RouteDecision
    from mabhas19.retrieval.router import Router
    from mabhas19.retrieval.schema import TableSchema
    from mabhas19.shadow.verifier import Verification

TRANSLATION_VERSION = "sh0-translate-1"
_INTENT = {"direct": "lookup", "whole_table": "lookup", "reverse": "reverse_lookup",
           "existence": "verify"}  # fmt: skip
_BACKGROUND_ROLES = frozenset({"background", "excluded", "corrected_away"})


@dataclass
class ShadowPlan:
    """The verified semantic components as restrictions on the deterministic pipeline."""

    masked: list[dict[str, Any]] = field(default_factory=list)  # start, end, text, reason
    dropped: dict[str, str] = field(default_factory=dict)  # table -> reason (before routing)
    refused: dict[str, str] = field(default_factory=dict)  # table -> reason (after the run)
    requested: list[str] = field(default_factory=list)  # tables the proposal requests
    value_roles: dict[int, str] = field(default_factory=dict)  # value index -> role
    calculation_added: bool = False
    all_refused: str = ""  # reason that refuses every table
    opportunities: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": TRANSLATION_VERSION,
            "masked": list(self.masked),
            "dropped": dict(sorted(self.dropped.items())),
            "refused": dict(sorted(self.refused.items())),
            "requested": list(self.requested),
            "value_roles": {str(k): v for k, v in sorted(self.value_roles.items())},
            "calculation_added": self.calculation_added,
            "all_refused": self.all_refused,
            "opportunities": list(self.opportunities),
        }


@dataclass
class Counterfactual:
    query: str  # the (masked) query that was run
    route: str
    tables: list[str]  # the routing decision after the drops
    rows: list[str]  # the rows the counterfactual would return
    outcomes: list[tuple[str, str, str]]  # (table, outcome, reason) incl. shadow refusals
    refused: dict[str, str]  # tables refused by the shadow plan after the run
    added_rows: list[str]  # rows the run returned that production does not (before the guard)
    guard: str  # ok | restriction_added_rows | mask_alignment
    certification_failures: int
    diff: dict[str, bool]

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "route": self.route,
            "tables": list(self.tables),
            "rows": list(self.rows),
            "outcomes": [list(x) for x in self.outcomes],
            "refused": dict(sorted(self.refused.items())),
            "added_rows": list(self.added_rows),
            "guard": self.guard,
            "certification_failures": self.certification_failures,
            "diff": dict(self.diff),
        }


def _overlaps(a: Pair, b: Pair) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def _inside(inner: Pair, outer: Pair) -> bool:
    return outer[0] <= inner[0] and inner[1] <= outer[1]


# ---------------------------------------------------------------- translation


def translate(  # noqa: PLR0912
    verification: Verification,
    ctx: CandidateContext,
    facts: DeterministicFacts,
    schemas: dict[str, TableSchema],
) -> ShadowPlan | None:
    """The shadow plan of a verified (or partially verified) proposal; None otherwise."""
    p = verification.proposal
    if p is None or not verification.usable:
        return None
    plan = ShadowPlan(
        calculation_added=verification.calculation_added,
        opportunities=list(verification.opportunities),
    )
    roles = {c.index: c.role for c in p.clauses}
    plan.value_roles = {v.index: v.role for v in p.values}
    requests = {r.table: r for r in p.requests}
    plan.requested = list(requests)
    # mask: an identity reading the pipeline made (a grounded entity span) or would make (an
    # exact identity-index candidate) and the proposal withdraws
    entity_spans = [s for f in facts.per_table.values() for s in f.entity_spans]
    identities = [(e.start, e.end) for e in p.entities if e.use == "identity"]
    for e in p.entities:
        exact = any(c.status == "exact" for c in ctx.entities[e.index].candidates)
        nested = any(_inside((e.start, e.end), s) and s != (e.start, e.end) for s in identities)
        if (
            e.use in ("grammatical", "contextual")
            and not nested  # a word inside a longer name the proposal keeps as identity
            and (exact or any(_overlaps((e.start, e.end), s) for s in entity_spans))
        ):
            plan.masked.append({"start": e.start, "end": e.end, "text": e.text, "reason": e.use})
    # drop: tables that are not requested
    for t, f in facts.per_table.items():
        if t not in requests:
            plan.dropped[t] = "not_requested"
            continue
        clauses = {ctx.clause_of(a, a + 1) for a, _ in f.anchors}
        if f.anchors and not any(
            c is not None and roles.get(c) in REQUESTED_CLAUSES for c in clauses
        ):
            plan.dropped[t] = "anchored_outside_requested_clauses"
            continue
        claimed = [v for v in ctx.values if f.checks(v)]
        if "existence" in f.operations and any(
            plan.value_roles.get(v.index) != "verify_request" for v in claimed
        ):
            plan.dropped[t] = "claim_not_requested"
    # refuse: everything the proposal leaves ambiguous or contradicts
    if p.operation.kind == AMBIGUOUS:
        plan.all_refused = "operation_ambiguous"
    elif plan.calculation_added:
        plan.all_refused = "unsupported_calculation"
    for t, r in requests.items():
        tf = facts.per_table.get(t)
        intents = {_INTENT[o] for o in (tf.operations if tf else ()) if o in _INTENT}
        if r.intent == AMBIGUOUS:
            plan.refused[t] = "intent_ambiguous"
        elif intents and intents != {r.intent}:
            plan.refused[t] = "intent_mismatch"
        elif r.output == AMBIGUOUS:
            plan.refused[t] = "output_ambiguous"
        elif r.unit == AMBIGUOUS:
            plan.refused[t] = "unit_ambiguous"
        elif r.output != NONE and tf is not None and not _output_agrees(schemas[t], r.output, tf):
            plan.refused[t] = "output_mismatch"
    for e in p.entities:
        if AMBIGUOUS in (e.use, e.candidate):
            for t, f in facts.per_table.items():
                if any(_overlaps((e.start, e.end), s) for s in f.entity_spans):
                    plan.refused.setdefault(t, "entity_ambiguous")
    return plan


def _output_agrees(schema: TableSchema, output: str, f: Any) -> bool:
    asked = {x for x in f.requested if x}
    if not asked:
        return True
    o = schema.output(output)
    allowed = {output, *(o.alternatives if o else ())}
    return bool(asked & allowed)


# ---------------------------------------------------------------- counterfactual run


class _RouterView:
    """The production router with tables removed from its decision (nothing else changes)."""

    def __init__(self, router: Router, keep: frozenset[str]) -> None:
        self._router, self._keep = router, keep

    def route(self, query: str) -> RouteDecision:
        decision = self._router.route(query)
        return replace(decision, tables=[t for t in decision.tables if t in self._keep])

    def __getattr__(self, name: str) -> Any:
        return getattr(self._router, name)


def mask(query: str, spans: list[dict[str, Any]]) -> tuple[str, list[int]] | None:
    """The query without the masked spans (spacing collapsed) and, per character, its offset in
    the original query. None when the result is not stable under normalization."""
    drop = [False] * len(query)
    for s in spans:
        for i in range(int(s["start"]), int(s["end"])):
            drop[i] = True
    chars: list[tuple[str, int]] = [(ch, i) for i, ch in enumerate(query) if not drop[i]]
    out: list[tuple[str, int]] = []
    for ch, i in chars:
        if ch.isspace() and (not out or out[-1][0].isspace()):
            continue
        out.append((ch, i))
    while out and out[-1][0].isspace():
        out.pop()
    text = "".join(ch for ch, _ in out)
    if normalize_for_search(text) != text:
        return None
    return text, [i for _, i in out]


def _back(span: Pair, index: list[int]) -> Pair:
    return index[span[0]], index[span[1] - 1] + 1


def counterfactual(
    retriever: Retriever,
    ctx: CandidateContext,
    facts: DeterministicFacts,
    plan: ShadowPlan,
    production: RetrievalResult,
) -> Counterfactual:
    """Run the shadow plan through the unchanged pipeline; production is only read."""
    masked = mask(ctx.query, plan.masked)
    prod_rows = [h.record_id for h in production.structured]
    if masked is None:
        return _closed(ctx.query, production, "mask_alignment")
    text, index = masked
    keep = frozenset(plan.requested) - set(plan.dropped)
    view = copy.copy(retriever)
    view.router = cast("Router", _RouterView(retriever.router, keep))
    res = view.retrieve(text)
    cf = facts_of(res)
    refused = dict(plan.refused)
    for t in cf.per_table:
        if plan.all_refused:
            refused[t] = plan.all_refused
            continue
        tf = cf.per_table[t]
        # a value read back in the original query's offsets
        shifted = replace(tf, applied=tuple(_back(s, index) for s in tf.applied))
        if any(
            plan.value_roles.get(v.index) in _BACKGROUND_ROLES and shifted.applies(v)
            for v in ctx.values
        ):
            refused.setdefault(t, "value_role_conflict")
    rows = [h.record_id for h in res.structured if h.dataset not in refused]
    outcomes = [(n.dataset, str(n.outcome or ""), n.reason) for n in res.no_match]
    # a refusal changes the visible outcome only where the run returned rows for the table
    outcomes += [
        (t, "shadow_refused", why)
        for t, why in sorted(refused.items())
        if any(h.dataset == t for h in res.structured)
    ]
    added = [r for r in rows if r not in prod_rows]
    guard = "ok"
    if added:  # a restriction never adds a row: refuse them all
        guard, rows = "restriction_added_rows", []
    out = Counterfactual(
        query=text,
        route=res.route.route,
        tables=list(res.route.tables),
        rows=rows,
        outcomes=sorted(set(outcomes)),
        refused={t: why for t, why in refused.items() if t in cf.per_table},
        added_rows=added,
        guard=guard,
        certification_failures=len(res.certification_failures),
        diff={},
    )
    out.diff = diff(production, out)
    return out


def _closed(query: str, production: RetrievalResult, guard: str) -> Counterfactual:
    tables = sorted({h.dataset for h in production.structured} | set(production.route.tables))
    out = Counterfactual(
        query=query,
        route=production.route.route,
        tables=list(production.route.tables),
        rows=[],
        outcomes=[(t, "shadow_refused", guard) for t in tables],
        refused=dict.fromkeys(tables, guard),
        added_rows=[],
        guard=guard,
        certification_failures=0,
        diff={},
    )
    out.diff = diff(production, out)
    return out


def diff(production: RetrievalResult, cf: Counterfactual) -> dict[str, bool]:
    """How the counterfactual differs from production (structured part; prose is not
    compared)."""
    prod_rows = {h.record_id for h in production.structured}
    rows = set(cf.rows)
    prod_out = sorted({(n.dataset, str(n.outcome or ""), n.reason) for n in production.no_match})
    route_changed = cf.route != production.route.route or cf.tables != list(production.route.tables)
    out = {
        "adds_rows": bool(cf.added_rows),
        "removes_rows": bool(prod_rows - rows),
        "route_changed": route_changed,
        "no_match_changed": [tuple(x) for x in cf.outcomes] != prod_out,
    }
    out["identical"] = not any(out.values())
    return out


# ---------------------------------------------------------------- S1 invariant


def fallback_eligible(production: RetrievalResult) -> bool:
    """The only situation a future verified fallback (S1) may act in: the deterministic
    structured path ran and failed closed - no structured row, and a typed refusal. A
    production answer is never replaced, added to, narrowed or reinterpreted; a prose or
    out-of-scope route is not a failed structured lookup (a broader intervention needs its own
    approved stage)."""
    if production.structured:
        return False
    if production.route.route not in ("structured", "mixed", "unsupported"):
        return False
    return any(n.outcome for n in production.no_match)
