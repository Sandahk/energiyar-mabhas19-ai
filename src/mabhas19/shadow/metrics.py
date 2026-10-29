"""S1-V / S1-R diagnostics of shadow runs (measurement only; nothing here acts on a result).

**Veto (S1-V)** - what the verified proposal would remove: rows, tables, an unasked claim, a
false identity reading, or the whole answer (forced fail-closed). Against the gold expected rows
each veto is *correct* (it removes only rows that are not expected - wrong or unasked),
*harmful* (it removes an expected row) or *neutral* (it removes nothing production returned).

**Recovery (S1-R, blocked)** - rows the pipeline run would add, split by cause: after a veto
(a withdrawn reading lets the pipeline ground another entity on its own) or from a positive
model choice (an entity / table the model selected; SH0 has no such lever, so it is always
false), plus the opportunities recorded by the verifier and whether the unguarded run would be
record-exact.

**Preservation** - on currently correct cases: unchanged, expected rows removed, tables removed,
typed outcome changed, proposal rejected.

**Label overuse** - per semantic role, how often the proposal uses it where the gold label is
another (``grammatical``, ``background``, ``contextual``, ``ambiguous`` …).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mabhas19.shadow.run import ShadowRun

METRICS_VERSION = "sh0-metrics-1"
# the first S1-V promotion decision (no harmful-veto allowance)
S1V_CRITERIA = {
    "unsafe_new_rows": 0,
    "malicious_verifier_escapes": 0,
    "replay_determinism": 1.0,
    "provider_schema_valid_rate": 0.98,
    "targeted_correct_veto_rate": 0.75,
    "harmful_vetoes_dev": 0,
    "harmful_vetoes_unseen": 0,
}
TARGETED = ("homograph_possessive", "background_claim_request")


@dataclass(frozen=True)
class Gold:
    """A frozen pilot case's gold: expected rows/outcomes and semantic labels."""

    expected_rows: frozenset[str] = frozenset()
    acceptable_rows: frozenset[str] = frozenset()
    expected_outcomes: frozenset[str] = frozenset()
    currently_correct: bool = False
    stratum: str = ""
    clauses: Mapping[int, str] = field(default_factory=dict)
    entities: Mapping[int, str] = field(default_factory=dict)
    values: Mapping[int, str] = field(default_factory=dict)

    @property
    def allowed(self) -> frozenset[str]:
        return self.expected_rows | self.acceptable_rows


def case_metrics(run: ShadowRun, gold: Gold) -> dict[str, Any]:
    prod = {h.record_id for h in run.production.structured}
    by_table: dict[str, set[str]] = {}
    for h in run.production.structured:
        by_table.setdefault(h.dataset, set()).add(h.record_id)
    status = run.verification.status if run.verification else "unparsed"
    cf, plan = run.counterfactual, run.plan
    out: dict[str, Any] = {
        "status": status,
        "schema_valid": run.proposal is not None,
        "verifier_accepted": status in ("verified", "partial"),
    }
    # ---- veto
    removed = prod - set(cf.rows) if cf else set()
    vetoed_tables = set(plan.dropped) | set(cf.refused if cf else {}) if plan else set()
    tables_removed = sorted(t for t, rows in by_table.items() if cf and not rows & set(cf.rows))
    claim_tables = {
        t
        for t, f in run.facts.per_table.items()
        if "existence" in f.operations and plan is not None and t in plan.dropped
    }
    out |= {
        "would_remove_rows": sorted(removed),
        "would_remove_tables": tables_removed,
        "would_drop_unasked_claim": bool(claim_tables & set(by_table)),
        "would_mask_false_identity": bool(plan and plan.masked),
        "would_force_fail_closed": bool(prod) and cf is not None and not cf.rows,
    }
    if cf is None or (not removed and not vetoed_tables and not (plan and plan.masked)):
        veto = "none"
    elif removed & gold.allowed:
        veto = "harmful"
    elif removed:
        veto = "correct"
    else:
        veto = "neutral"
    out["veto"] = veto
    # ---- recovery (blocked in SH0)
    added = list(cf.added_rows) if cf else []
    positive = False  # SH0 applies no positive model choice (no lever exists)
    opportunities = Counter(
        o["kind"] for o in (run.verification.opportunities if run.verification else [])
    )
    run_rows = set(cf.run_rows) if cf else set()
    out |= {
        "would_add_rows": added,
        "adds_rows_after_veto": bool(added) and not positive,
        "adds_rows_from_positive_llm_choice": bool(added) and positive,
        "would_disambiguate": opportunities["would_disambiguate"],
        "would_add_table": opportunities["would_add_table"],
        "counterfactual_record_exact": bool(gold.expected_rows)
        and run_rows <= gold.allowed
        and gold.expected_rows <= run_rows,
    }
    # ---- preservation (currently correct cases)
    if gold.currently_correct:
        rejected = status in ("invalid", "rejected", "unparsed")
        unchanged = cf is not None and cf.diff.get("identical", False)
        out["preservation"] = {
            "unchanged": unchanged and not rejected,
            "expected_rows_removed": bool(removed & gold.allowed),
            "tables_removed": bool(tables_removed),
            "typed_outcome_changed": bool(cf and cf.diff.get("no_match_changed")),
            "rejected_proposal": rejected,
        }
    # ---- labels
    out["labels"] = _labels(run, gold)
    out["unsafe_new_rows"] = sorted(set(cf.rows) - prod) if cf else []
    return out


def _labels(run: ShadowRun, gold: Gold) -> dict[str, dict[str, int]]:
    """Per role: proposed, gold, agreed, overused (proposed where gold differs)."""
    out: dict[str, Counter[str]] = {}
    p = run.proposal
    if p is None:
        return {}
    pairs: list[tuple[str, str, str]] = []
    for kind, given, labelled in (
        ("clause", {c.index: c.role for c in p.clauses}, gold.clauses),
        ("entity", {e.index: e.use for e in p.entities}, gold.entities),
        ("value", {v.index: v.role for v in p.values}, gold.values),
    ):
        for i, role in labelled.items():
            pairs.append((kind, given.get(i, ""), role))
    for kind, proposed, truth in pairs:
        if proposed:
            out.setdefault(f"{kind}:{proposed}", Counter())["proposed"] += 1
        out.setdefault(f"{kind}:{truth}", Counter())["gold"] += 1
        if proposed == truth:
            out[f"{kind}:{truth}"]["agreed"] += 1
        elif proposed:
            out[f"{kind}:{proposed}"]["overused"] += 1
    return {k: dict(v) for k, v in sorted(out.items())}


def aggregate(cases: Iterable[tuple[dict[str, Any], Gold]]) -> dict[str, Any]:
    rows = list(cases)
    n = len(rows)
    targeted = [m for m, g in rows if g.stratum in TARGETED]
    correct = [m for m, g in rows if g.currently_correct]
    labels: dict[str, Counter[str]] = {}
    for m, _ in rows:
        for role, counts in m["labels"].items():
            labels.setdefault(role, Counter()).update(counts)
    veto = Counter(m["veto"] for m, _ in rows)
    out = {
        "version": METRICS_VERSION,
        "cases": n,
        "provider_schema_valid_rate": sum(m["schema_valid"] for m, _ in rows) / n if n else 0.0,
        "verifier_status": dict(Counter(m["status"] for m, _ in rows)),
        "unsafe_new_rows": sum(len(m["unsafe_new_rows"]) for m, _ in rows),
        "veto": dict(veto),
        "harmful_vetoes": veto["harmful"],
        "harmful_vetoes_on_currently_correct": sum(m["veto"] == "harmful" for m in correct),
        "targeted_correct_veto_rate": (
            sum(m["veto"] == "correct" for m in targeted) / len(targeted) if targeted else 0.0
        ),
        "would_force_fail_closed": sum(m["would_force_fail_closed"] for m, _ in rows),
        "recovery_blocked": {
            "would_add_rows": sum(bool(m["would_add_rows"]) for m, _ in rows),
            "adds_rows_after_veto": sum(m["adds_rows_after_veto"] for m, _ in rows),
            "adds_rows_from_positive_llm_choice": sum(
                m["adds_rows_from_positive_llm_choice"] for m, _ in rows
            ),
            "would_disambiguate": sum(m["would_disambiguate"] for m, _ in rows),
            "would_add_table": sum(m["would_add_table"] for m, _ in rows),
            "counterfactual_record_exact": sum(m["counterfactual_record_exact"] for m, _ in rows),
        },
        "preservation": {
            k: sum(m["preservation"][k] for m in correct)
            for k in (
                "unchanged",
                "expected_rows_removed",
                "tables_removed",
                "typed_outcome_changed",
                "rejected_proposal",
            )
        },
        "label_overuse": {
            role: {
                **dict(c),
                "overuse_rate": c["overused"] / c["proposed"] if c["proposed"] else 0.0,
            }
            for role, c in sorted(labels.items())
        },
    }
    return out


def s1v_check(report: Mapping[str, Any], extra: Mapping[str, float]) -> dict[str, dict[str, Any]]:
    """The first S1-V promotion criteria (``extra`` supplies the values measured elsewhere:
    malicious-suite escapes, replay determinism, harmful vetoes on the unseen sample)."""
    values = {
        "unsafe_new_rows": report["unsafe_new_rows"],
        "malicious_verifier_escapes": extra.get("malicious_verifier_escapes", float("nan")),
        "replay_determinism": extra.get("replay_determinism", float("nan")),
        "provider_schema_valid_rate": report["provider_schema_valid_rate"],
        "targeted_correct_veto_rate": report["targeted_correct_veto_rate"],
        "harmful_vetoes_dev": report["harmful_vetoes_on_currently_correct"],
        "harmful_vetoes_unseen": extra.get("harmful_vetoes_unseen", float("nan")),
    }
    at_least = {"replay_determinism", "provider_schema_valid_rate", "targeted_correct_veto_rate"}
    out = {}
    for name, threshold in S1V_CRITERIA.items():
        v = values[name]
        ok = v >= threshold if name in at_least else v <= threshold
        out[name] = {"value": v, "threshold": threshold, "pass": bool(ok)}
    return out
