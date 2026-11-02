"""Candidate S1-V policy ``s1v-grammatical-veto-1`` and its veto-benchmark metrics (measurement
only: nothing here changes a production result).

**Authority.** The policy has exactly one lever: a verifier-accepted ``entity.use =
grammatical`` label masks that exact query span, and only when the span participates in the
production run's deterministic entity grounding (it overlaps a grounded entity span). Every
other proposal field is non-authoritative - logged and scored, never acting
(:data:`NON_AUTHORITATIVE`). Eligibility never depends on the deterministic homograph lexicon:
whether the model's judgment is trustworthy is what the benchmark measures.

Field-level eligibility: the proposal must parse and pass the verifier's integrity checks (a
proposal that fabricates spans or identifiers is not used at all); the label's own entity
decisions must all be accepted; a rejected unrelated field does not block it. One abstention
rule: a span nested inside a longer span the same proposal labels ``identity`` is not masked (a
self-contradicting proposal never vetoes; it can only remove a veto, never add one).

**Counterfactual.** The masked query runs through the unchanged pipeline (router restricted to
the production tables, planner, admission, I1-I11 coverage, execution, certification); no table
is dropped or refused by the model. A run that returns a row production does not is refused
whole by the add-row guard, so the result is always a subset of the production rows. A mask
that cannot be applied stably (normalization) is not applied: production stands.

**Metrics** (veto benchmark gold: per production row ``correct`` / ``wrong`` / ``uncertain``,
per grounded span ``identity`` / ``grammatical`` / ``contextual`` / ``ambiguous``):

* a veto *acts* when it removes a production row; it is **correct** when it removes only wrong
  rows, **harmful** when it removes a correct row, **uncertain** when it removes only rows the
  gold leaves uncertain; an applied mask that removes nothing is **neutral**;
* **opportunity recall** - among cases whose gold has a grounded grammatical span that removes
  a wrong row when masked: correct vetoes / opportunities (the rest are missed);
* **precision** - correct / acting vetoes; **harmful-veto rate** on the genuine-identity
  controls and on every case whose production rows are all correct;
* **span accuracy** - gold use x proposed use on grounded spans only;
* **safety** - rows outside production, guard refusals, certification failures.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from mabhas19.shadow.translate import Counterfactual, ShadowPlan, counterfactual

if TYPE_CHECKING:
    from mabhas19.retrieval.hybrid import Retriever
    from mabhas19.retrieval.models import RetrievalResult
    from mabhas19.shadow.context import CandidateContext, DeterministicFacts
    from mabhas19.shadow.run import ShadowRun

POLICY_VERSION = "s1v-grammatical-veto-1"
VETO_METRICS_VERSION = "s1v-veto-metrics-1"
GRAMMATICAL = "grammatical"
NON_AUTHORITATIVE = (
    "ambiguous",
    "none",
    "contextual",
    "entity candidate",
    "clause role",
    "value role",
    "request intent",
    "output",
    "unit",
    "operation",
    "coordination",
    "background",
)
SPAN_USES = ("identity", "grammatical", "contextual", "ambiguous")
ROW_LABELS = ("correct", "wrong", "uncertain")
# benchmark groups: A positive opportunities, B / C genuine-identity controls, D no safe decision
GROUPS = ("A_grammatical_misread", "B_identity_lookalike", "C_plain_identity", "D_ambiguous")
CONTROLS = frozenset({"B_identity_lookalike", "C_plain_identity"})
# the proposed first promotion criteria for V1 on the unseen veto benchmark
V1_CRITERIA = {
    "harmful_vetoes": 0,
    "veto_precision": 1.0,
    "opportunity_recall": 0.40,
    "rows_outside_production": 0,
    "certification_failures": 0,
    "replay_determinism": 1.0,
    "schema_valid_rate": 0.98,
}


@dataclass
class VetoDecision:
    policy: str
    masked: list[dict[str, Any]]  # the eligible spans (entity index, offsets, text)
    applied: bool  # a mask was applied and the counterfactual ran
    reason: str  # no_proposal | integrity | not_eligible | applied | mask_alignment
    rows: list[str]  # the rows under the policy (production when not applied)
    removed: list[str]
    counterfactual: Counterfactual | None = None
    guard: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "masked": list(self.masked),
            "applied": self.applied,
            "reason": self.reason,
            "rows": list(self.rows),
            "removed": list(self.removed),
            "guard": self.guard,
            "certification_failures": (
                self.counterfactual.certification_failures if self.counterfactual else 0
            ),
        }


def _overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def grounded_spans(facts: DeterministicFacts) -> list[tuple[int, int]]:
    return [s for f in facts.per_table.values() for s in f.entity_spans]


def grounded_entities(ctx: CandidateContext, facts: DeterministicFacts) -> list[int]:
    """Context entity spans that participate in the production run's entity grounding."""
    spans = grounded_spans(facts)
    return [e.index for e in ctx.entities if any(_overlaps((e.start, e.end), s) for s in spans)]


def eligible(run: ShadowRun) -> tuple[str, list[dict[str, Any]]]:
    """(reason, spans): the spans V1 may mask (empty: the policy does not act)."""
    p, v = run.proposal, run.verification
    if p is None or v is None:
        return "no_proposal", []
    if v.status == "invalid":
        return "integrity", []
    decided: dict[str, set[str]] = {}
    for d in v.decisions:
        if d.field == "entities":
            decided.setdefault(d.item, set()).add(d.outcome)
    grounded = set(grounded_entities(run.context, run.facts))
    identity = [(e.start, e.end) for e in p.entities if e.use == "identity"]
    out = []
    for e in p.entities:
        span = (e.start, e.end)
        if (
            e.use == GRAMMATICAL
            and decided.get(str(e.index)) == {"accepted"}
            and e.index in grounded
            and not any(s[0] <= span[0] and span[1] <= s[1] and s != span for s in identity)
        ):
            out.append({"index": e.index, "start": e.start, "end": e.end, "text": e.text})
    return ("applied" if out else "not_eligible"), out


def mask_rows(
    retriever: Retriever,
    ctx: CandidateContext,
    facts: DeterministicFacts,
    production: RetrievalResult,
    spans: list[dict[str, Any]],
) -> Counterfactual:
    """The unchanged pipeline on the query without ``spans`` (no table dropped or refused)."""
    plan = ShadowPlan(
        masked=[{**s, "reason": GRAMMATICAL} for s in spans],
        requested=list(dict.fromkeys([*production.route.tables, *facts.per_table])),
    )
    return counterfactual(retriever, ctx, facts, plan, production)


def decide(retriever: Retriever, run: ShadowRun) -> VetoDecision:
    """V1 on one shadow run."""
    prod = [h.record_id for h in run.production.structured]
    reason, spans = eligible(run)
    if not spans:
        return VetoDecision(POLICY_VERSION, [], False, reason, prod, [])
    cf = mask_rows(retriever, run.context, run.facts, run.production, spans)
    if cf.guard == "mask_alignment":
        return VetoDecision(POLICY_VERSION, spans, False, "mask_alignment", prod, [], cf, cf.guard)
    rows = [r for r in cf.rows if r in prod]  # a subset by construction (add-row guard)
    removed = [r for r in prod if r not in rows]
    return VetoDecision(POLICY_VERSION, spans, True, "applied", rows, removed, cf, cf.guard)


# ---------------------------------------------------------------- benchmark gold and metrics


@dataclass(frozen=True)
class VetoGold:
    group: str
    rows: Mapping[str, str] = field(default_factory=dict)  # production row -> ROW_LABELS
    spans: Mapping[int, str] = field(default_factory=dict)  # grounded entity index -> SPAN_USES
    opportunity: bool = False  # masking a gold-grammatical grounded span removes a wrong row

    @property
    def production_correct(self) -> bool:
        return bool(self.rows) and all(v == "correct" for v in self.rows.values())


def classify(removed: Iterable[str], applied: bool, gold: VetoGold) -> str:
    gone = list(removed)
    if not applied:
        return "none"
    if not gone:
        return "neutral"
    labels = {gold.rows.get(r, "uncertain") for r in gone}
    if "correct" in labels:
        return "harmful"
    if labels == {"wrong"}:
        return "correct"
    return "uncertain"


def case_metrics(run: ShadowRun, decision: VetoDecision, gold: VetoGold) -> dict[str, Any]:
    prod = {h.record_id for h in run.production.structured}
    veto = classify(decision.removed, decision.applied, gold)
    proposed = {e.index: e.use for e in run.proposal.entities} if run.proposal else {}
    cf = decision.counterfactual
    return {
        "group": gold.group,
        "schema_valid": run.proposal is not None,
        "status": run.verification.status if run.verification else "unparsed",
        "reason": decision.reason,
        "veto": veto,
        "masked": [s["text"] for s in decision.masked],
        "removed": list(decision.removed),
        "opportunity": gold.opportunity,
        "production_correct": gold.production_correct,
        "spans": [(gold.spans[i], proposed.get(i, "")) for i in sorted(gold.spans)],
        "rows_outside_production": sorted(set(decision.rows) - prod),
        "guard_refused": decision.guard == "restriction_added_rows",
        "certification_failures": cf.certification_failures if cf else 0,
    }


def aggregate(cases: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(cases)
    n = len(rows)
    veto = Counter(m["veto"] for m in rows)
    acting = veto["correct"] + veto["harmful"] + veto["uncertain"]
    opp = [m for m in rows if m["opportunity"]]
    controls = [m for m in rows if m["group"] in CONTROLS]
    correct_prod = [m for m in rows if m["production_correct"]]
    confusion: dict[str, Counter[str]] = {}
    for m in rows:
        for g, p in m["spans"]:
            confusion.setdefault(g, Counter())[p or "unlabelled"] += 1
    return {
        "version": VETO_METRICS_VERSION,
        "policy": POLICY_VERSION,
        "cases": n,
        "schema_valid_rate": sum(m["schema_valid"] for m in rows) / n if n else 0.0,
        "verifier_status": dict(Counter(m["status"] for m in rows)),
        "veto": dict(veto),
        "veto_by_group": {
            g: dict(Counter(m["veto"] for m in rows if m["group"] == g)) for g in GROUPS
        },
        "opportunities": len(opp),
        "opportunity_correct": sum(m["veto"] == "correct" for m in opp),
        "opportunity_recall": (sum(m["veto"] == "correct" for m in opp) / len(opp) if opp else 0.0),
        "opportunity_missed": sum(m["veto"] != "correct" for m in opp),
        "acting_vetoes": acting,
        "veto_precision": veto["correct"] / acting if acting else 1.0,
        "harmful_vetoes": veto["harmful"],
        "harmful_on_controls": sum(m["veto"] == "harmful" for m in controls),
        "controls": len(controls),
        "harmful_on_correct_production": sum(m["veto"] == "harmful" for m in correct_prod),
        "correct_production_cases": len(correct_prod),
        "span_accuracy": {
            g: {
                **dict(c),
                "accuracy": c[g] / sum(c.values()) if sum(c.values()) else 0.0,
            }
            for g, c in sorted(confusion.items())
        },
        "rows_outside_production": sum(len(m["rows_outside_production"]) for m in rows),
        "guard_refusals": sum(m["guard_refused"] for m in rows),
        "certification_failures": sum(m["certification_failures"] for m in rows),
    }


def criteria(report: Mapping[str, Any], extra: Mapping[str, float]) -> dict[str, dict[str, Any]]:
    """The proposed V1 promotion criteria (``extra``: replay determinism)."""
    values = {
        "harmful_vetoes": report["harmful_vetoes"],
        "veto_precision": report["veto_precision"],
        "opportunity_recall": report["opportunity_recall"],
        "rows_outside_production": report["rows_outside_production"],
        "certification_failures": report["certification_failures"],
        "replay_determinism": extra.get("replay_determinism", float("nan")),
        "schema_valid_rate": report["schema_valid_rate"],
    }
    at_least = {"veto_precision", "opportunity_recall", "replay_determinism", "schema_valid_rate"}
    out = {}
    for name, threshold in V1_CRITERIA.items():
        v = values[name]
        ok = v >= threshold if name in at_least else v <= threshold
        out[name] = {"value": v, "threshold": threshold, "pass": bool(ok)}
    return out


# ---------------------------------------------------------------- gold-derived expectations


def span_effect(  # noqa: PLR0913, PLR0917
    retriever: Retriever,
    ctx: CandidateContext,
    facts: DeterministicFacts,
    production: RetrievalResult,
    index: int,
    rows: Mapping[str, str],
) -> tuple[str, list[str]]:
    """What masking one context entity span alone does to the production rows, judged by the
    gold row labels: removes_wrong | removes_correct | removes_uncertain | no_effect |
    not_applicable (mask not stable)."""
    e = ctx.entities[index]
    cf = mask_rows(
        retriever, ctx, facts, production,
        [{"index": e.index, "start": e.start, "end": e.end, "text": e.text}],
    )  # fmt: skip
    if cf.guard == "mask_alignment":
        return "not_applicable", []
    prod = [h.record_id for h in production.structured]
    removed = [r for r in prod if r not in cf.rows]
    if not removed:
        return "no_effect", []
    labels = {rows.get(r, "uncertain") for r in removed}
    if "correct" in labels:
        return "removes_correct", removed
    if labels == {"wrong"}:
        return "removes_wrong", removed
    return "removes_uncertain", removed


def oracle_rows(
    retriever: Retriever,
    ctx: CandidateContext,
    facts: DeterministicFacts,
    production: RetrievalResult,
    spans: Mapping[int, str],
) -> list[str]:
    """The rows V1 would leave with a perfect proposal (every gold-grammatical grounded span
    masked); production's rows when there is none."""
    grounded = set(grounded_entities(ctx, facts))
    chosen = [
        {"index": e.index, "start": e.start, "end": e.end, "text": e.text}
        for e in ctx.entities
        if spans.get(e.index) == GRAMMATICAL and e.index in grounded
    ]
    prod = [h.record_id for h in production.structured]
    if not chosen:
        return prod
    cf = mask_rows(retriever, ctx, facts, production, chosen)
    if cf.guard == "mask_alignment":
        return prod
    return [r for r in cf.rows if r in prod]


def gold_of(case: Mapping[str, Any]) -> VetoGold:
    g = case.get("gold") or {}
    return VetoGold(
        group=str(case["stratum"]),
        rows={str(k): str(v) for k, v in (g.get("rows") or {}).items()},
        spans={int(k): str(v) for k, v in (g.get("spans") or {}).items()},
        opportunity=bool(g.get("opportunity")),
    )
