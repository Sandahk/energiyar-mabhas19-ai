"""One SH0 shadow run: production first and unchanged, then the proposal beside it.

:func:`shadow` runs the production pipeline, builds the candidate context, parses and verifies
the proposal, translates it and runs the counterfactual. The production result it returns is
the object the pipeline produced; nothing in the shadow path writes to it, and its
counterfactual never replaces it (``fallback_eligible`` only records whether a future S1 stage
could have acted at all).

:func:`echo_proposal` is the deterministic reading written as a proposal (every label agrees
with the production run): a negative control whose counterfactual must equal production, used
by the tests and the pilot harness.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from mabhas19.shadow.context import CandidateContext, DeterministicFacts, build
from mabhas19.shadow.proposal import AMBIGUOUS, CALCULATIONS, NONE, PROPOSAL_SCHEMA, Proposal, parse
from mabhas19.shadow.replay import ReplayKey, ReplayRecord
from mabhas19.shadow.translate import (
    _INTENT,
    Counterfactual,
    ShadowPlan,
    counterfactual,
    fallback_eligible,
    translate,
)
from mabhas19.shadow.verifier import Verification, verify

if TYPE_CHECKING:
    from mabhas19.retrieval.hybrid import Retriever
    from mabhas19.retrieval.models import RetrievalResult


@dataclass
class ShadowRun:
    production: RetrievalResult
    context: CandidateContext
    facts: DeterministicFacts
    proposal: Proposal | None
    schema_errors: list[str]
    verification: Verification | None
    plan: ShadowPlan | None
    counterfactual: Counterfactual | None
    fallback_eligible: bool

    def record(
        self, key: ReplayKey, raw: str | None, timestamp: str, usage: dict[str, Any]
    ) -> ReplayRecord:
        return ReplayRecord(
            key=key,
            timestamp=timestamp,
            raw_response=raw,
            parsed_proposal=self.proposal.to_dict() if self.proposal else None,
            schema_errors=list(self.schema_errors),
            verifier=self.verification.to_dict() if self.verification else {},
            translated_plan=self.plan.to_dict() if self.plan else None,
            counterfactual=self.counterfactual.to_dict() if self.counterfactual else None,
            usage=dict(usage),
        )


def shadow(
    retriever: Retriever,
    query: str,
    raw: str | bytes | dict[str, Any] | None,
    production: RetrievalResult | None = None,
) -> ShadowRun:
    """Production (unchanged), then the proposal's verification and counterfactual. ``raw`` is
    the provider's response (None: no proposal)."""
    prod = production if production is not None else retriever.retrieve(query)
    ctx, facts = build(retriever, prod)
    proposal, errors = parse(raw) if raw is not None else (None, ["no proposal"])
    run = ShadowRun(prod, ctx, facts, proposal, errors, None, None, None, fallback_eligible(prod))
    if proposal is None:
        return run
    schemas = retriever.store.schemas
    run.verification = verify(proposal, ctx, facts, schemas)
    run.plan = translate(run.verification, ctx, facts, schemas)
    if run.plan is not None:
        run.counterfactual = counterfactual(retriever, ctx, facts, run.plan, prod)
    return run


def echo_proposal(ctx: CandidateContext, facts: DeterministicFacts) -> dict[str, Any]:
    """The production run's own reading as a proposal (a negative control)."""
    used = [s for f in facts.per_table.values() for s in f.entity_spans]
    resolved = {i for f in facts.per_table.values() for i in f.entity_ids}
    ambiguous_spans = [
        s
        for f in facts.per_table.values()
        if "ambiguous" in f.entity_states
        for s in f.entity_spans
    ]

    def inside(a: int, b: int, spans: list[tuple[int, int]]) -> bool:
        return any(s <= a and b <= e for s, e in spans)

    def overlaps(a: int, b: int, spans: list[tuple[int, int]]) -> bool:
        return any(a < e and s < b for s, e in spans)

    entities = []
    for e in ctx.entities:
        exact = [c for c in e.candidates if c.status == "exact"]
        if all(c.status == "homograph_only" for c in e.candidates):
            use, cand = "contextual", NONE  # names its own candidates only as a homograph
        elif not overlaps(e.start, e.end, used):
            use, cand = ("identity" if exact else "contextual"), NONE
        elif overlaps(e.start, e.end, ambiguous_spans):
            use, cand = "identity", NONE
        else:
            mine = [c for c in exact if c.row in resolved]
            use, cand = "identity", (mine[0].id if len(mine) == 1 else NONE)
        entities.append(
            {
                "index": e.index,
                "start": e.start,
                "end": e.end,
                "text": e.text,
                "use": use,
                "candidate": cand,
            }
        )
    values = []
    for v in ctx.values:
        kinds = {p.kind for p in ctx.polarity if p.start <= v.start and v.end <= p.end}
        if "exclusion" in kinds:
            role = "excluded"
        elif "correction_rejected" in kinds:
            role = "corrected_away"
        elif any(f.checks(v) for f in facts.per_table.values()):
            role = "verify_request"
        elif any(f.applies(v) for f in facts.per_table.values()):
            role = "lookup_constraint"
        else:
            role = "background"  # the pipeline used it for no row
        values.append(
            {
                "index": v.index,
                "start": v.start,
                "end": v.end,
                "text": v.text,
                "role": role,
                "clause": ctx.clause_of(v.start, v.end),
            }
        )
    requests = []
    for t in facts.tables:
        f = facts.per_table.get(t)
        intents = sorted({_INTENT[o] for o in (f.operations if f else ()) if o in _INTENT})
        table = ctx.table(t)
        outputs = {o.name for o in table.outputs} if table is not None else set()
        asked = sorted({x for x in (f.requested if f else ()) if x in outputs})
        requests.append(
            {
                "table": t,
                "intent": intents[0]
                if len(intents) == 1
                else ("lookup" if not intents else AMBIGUOUS),
                "output": asked[0] if len(asked) == 1 else NONE,
                "unit": NONE,
            }
        )
    detected = [o.kind for o in ctx.operations]
    return {
        "schema": PROPOSAL_SCHEMA,
        "clauses": [
            {"index": c.index, "start": c.start, "end": c.end, "role": "lookup"}
            for c in ctx.clauses
        ],
        "entities": entities,
        "values": values,
        "polarity": [
            {
                "index": p.index,
                "start": p.start,
                "end": p.end,
                "kept": True,
                "clause": ctx.clause_of(p.start, p.end),
            }
            for p in ctx.polarity
        ],
        "operation": {
            "kind": detected[0] if detected and detected[0] in CALCULATIONS else NONE,
            "refs": [o.index for o in ctx.operations[:1]],
        },
        "requests": requests,
        "coordination": [
            {
                "table": c.table,
                "members": [{"start": m.start, "end": m.end, "text": m.text} for m in c.members],
            }
            for c in ctx.coordination
        ],
        "notes": "",
    }
