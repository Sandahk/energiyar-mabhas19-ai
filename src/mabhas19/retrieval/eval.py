"""Offline retrieval evaluation (evidence-based; no generated answers are scored).

Metrics:

* prose Recall@1/3/5 and MRR, both *intrinsic* (BM25 run for every case with prose
  evidence) and *end-to-end* (only when the router chose a prose path);
* route accuracy (``acceptable_routes`` honored);
* structured evidence, scored by the most specific expectation and reported separately:

  - **record exact** (cases with ``records``): every required record returned and nothing
    outside required + acceptable; a typed no-match or table evidence never satisfies it;
  - **typed no-match** (``no_records`` cases): no record returned and a typed no-match for an
    expected table (or any typed no-match when no table is named);
  - **table match** (cases with ``tables`` only, no records): hits non-empty and all from the
    expected tables, or table-level evidence / no-match for an expected table;

* unsupported precision/recall with raw confusion counts, honoring ``acceptable_routes`` (a
  route the case explicitly allows is neither a false positive nor a false negative);
* source-issue exposure and conflict exposure.

The original (v1/v2 and first final benchmark) scorer is kept as ``structured_exact_original``
and ``unsupported_original`` so historical numbers stay reproducible: it scored any case with
``tables`` at table level even when records were expected, and used strict route labels for
unsupported scoring.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from mabhas19.retrieval.hybrid import Retriever
from mabhas19.retrieval.models import ProseHit, RetrievalResult

KS = (1, 3, 5)


@dataclass
class CaseResult:
    id: str
    category: str
    query: str
    expected_route: str
    route: str
    route_reasons: list[str]
    route_ok: bool
    acceptable_routes: list[str] = field(default_factory=list)
    prose_rank: int | None = None  # end-to-end (router-dependent)
    bm25_rank: int | None = None  # intrinsic
    structured_exact_original: bool | None = None  # historical scorer (see module docstring)
    record_exact: bool | None = None
    typed_no_match: bool | None = None
    table_match: bool | None = None
    structured_missing: list[str] = field(default_factory=list)
    structured_extra: list[str] = field(default_factory=list)
    issues_ok: bool | None = None
    conflict_ok: bool | None = None
    returned_records: list[str] = field(default_factory=list)
    top_chunks: list[str] = field(default_factory=list)
    has_prose_evidence: bool = False
    top_evidence: list[dict[str, Any]] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    structured_query: str | None = None
    prose_query: str | None = None
    no_match: list[str] = field(default_factory=list)


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = yaml.safe_load(path.read_text(encoding="utf-8"))["cases"]
    return cases


def check_evidence_ids(
    cases: list[dict[str, Any]],
    block_ids: set[str],
    sections: set[str],
    record_ids: set[str],
    tables: set[str],
) -> list[str]:
    problems = []
    for c in cases:
        ev = c.get("evidence") or {}
        for key, known in (
            ("blocks", block_ids),
            ("sections", sections),
            ("tables", tables),
            ("records", record_ids),
            ("acceptable_records", record_ids),
        ):
            unknown = [x for x in ev.get(key, []) if str(x) not in known]
            if unknown:
                problems.append(f"{c['id']}: unknown {key} {unknown}")
    return problems


def _prose_rank(
    hits: list[ProseHit], ev: dict[str, Any], chunk_sections: dict[str, str | None]
) -> int | None:
    blocks, sections = set(ev.get("blocks", [])), {str(s) for s in ev.get("sections", [])}
    for h in hits:
        if blocks & set(h.block_ids) or chunk_sections.get(h.chunk_id) in sections:
            return h.rank
    return None


def _structured(r: RetrievalResult, ev: dict[str, Any], res: CaseResult) -> None:
    got = [h.record_id for h in r.structured]
    res.returned_records = got
    res.structured_exact_original = _original_structured_exact(r, ev)
    expected_tables = set(ev.get("tables", []))
    if ev.get("records"):
        required = set(ev["records"])
        ok = required | set(ev.get("acceptable_records", []))
        res.structured_missing = sorted(required - set(got))
        res.structured_extra = sorted(set(got) - ok)
        res.record_exact = not res.structured_missing and not res.structured_extra
    elif ev.get("no_records"):
        res.structured_extra = got
        typed = {n.dataset for n in r.no_match}
        res.typed_no_match = not got and bool(typed & expected_tables if expected_tables else typed)
    elif expected_tables:
        hit_tables = {h.dataset for h in r.structured}
        table_level = {n.dataset for n in r.no_match} | {t.dataset for t in r.tables}
        res.table_match = hit_tables <= expected_tables and bool(
            (hit_tables | table_level) & expected_tables
        )


def _original_structured_exact(r: RetrievalResult, ev: dict[str, Any]) -> bool:
    """Historical scorer: ``tables`` took precedence over ``records`` (kept for reproducibility
    of the immutable first final result; do not use for new decisions)."""
    got = [h.record_id for h in r.structured]
    if ev.get("no_records"):
        return not got
    if ev.get("tables"):
        expected = set(ev["tables"])
        hit_tables = {h.dataset for h in r.structured}
        table_level = {n.dataset for n in r.no_match} | {t.dataset for t in r.tables}
        return hit_tables <= expected and bool((hit_tables | table_level) & expected)
    required, ok = set(ev["records"]), set(ev["records"]) | set(ev.get("acceptable_records", []))
    return required <= set(got) and set(got) <= ok


def evaluate(
    retriever: Retriever, cases: list[dict[str, Any]], k: int = 10
) -> tuple[list[CaseResult], dict[str, Any]]:
    chunk_sections = {c.chunk_id: c.section_number for c in retriever.lexical.chunks}
    results = []
    for c in cases:
        ev = c.get("evidence") or {}
        r = retriever.retrieve(c["query"], k)
        res = CaseResult(
            c["id"],
            c["category"],
            c["query"],
            c["expected_route"],
            r.route.route,
            r.route.reasons,
            r.route.route in {c["expected_route"], *c.get("acceptable_routes", [])},
            list(c.get("acceptable_routes", [])),
        )
        res.top_chunks = [h.chunk_id for h in r.prose[:5]]
        res.top_evidence = [
            {
                "chunk": h.chunk_id,
                "section": chunk_sections.get(h.chunk_id),
                "blocks": list(h.block_ids[:4]),
            }
            for h in r.prose[:3]
        ]
        res.conflicts = [x.issue_id or "-" for x in r.conflicts]
        res.notes = list(r.notes)
        res.structured_query = r.route.structured_query
        res.prose_query = r.route.prose_query
        res.no_match = [f"{n.dataset}: {n.reason}" for n in r.no_match]
        if ev.get("blocks") or ev.get("sections"):
            res.has_prose_evidence = True
            res.prose_rank = _prose_rank(r.prose, ev, chunk_sections)
            res.bm25_rank = _prose_rank(retriever.rank_prose(c["query"], k), ev, chunk_sections)
        if ev.get("records") or ev.get("tables") or ev.get("no_records"):
            _structured(r, ev, res)
        if ev.get("source_issues"):
            exposed = {i for h in r.structured for i in h.source_issues}
            exposed |= {x.issue_id for x in r.conflicts if x.issue_id}
            exposed |= {i for n in r.no_match for i in n.source_issues}
            exposed |= {i for h in r.prose for i in h.source_issues}
            exposed |= {i for n in r.notes for i in ev["source_issues"] if i in n}
            res.issues_ok = set(ev["source_issues"]) <= exposed
        if ev.get("conflict"):
            res.conflict_ok = any(x.issue_id == ev["conflict"] for x in r.conflicts)
        results.append(res)
    return results, metrics(results)


def _recall(ranks: list[int | None]) -> dict[str, float]:
    n = len(ranks) or 1
    out = {
        f"recall@{k}": round(sum(1 for r in ranks if r is not None and r <= k) / n, 3) for k in KS
    }
    out["mrr"] = round(sum(1 / r for r in ranks if r) / n, 3)
    out["n"] = len(ranks)
    return out


def _rate(values: list[bool | None]) -> dict[str, Any]:
    vals = [v for v in values if v is not None]
    return {"rate": round(sum(vals) / len(vals), 3) if vals else None, "n": len(vals)}


def metrics(results: list[CaseResult]) -> dict[str, Any]:
    prose = [r for r in results if r.has_prose_evidence and r.expected_route in ("prose", "mixed")]
    by_cat: dict[str, list[CaseResult]] = defaultdict(list)
    for r in results:
        by_cat[r.category].append(r)
    return {
        "cases": len(results),
        "composition": dict(Counter(r.category for r in results)),
        "bm25_intrinsic": _recall([r.bm25_rank for r in prose]),
        "prose_end_to_end": _recall([r.prose_rank for r in prose]),
        "route_accuracy": _rate([r.route_ok for r in results]),
        "route_confusion": dict(
            Counter(f"{r.expected_route}->{r.route}" for r in results if not r.route_ok)
        ),
        "route_matrix": route_matrix(results),
        "unsupported": unsupported_scores(results),
        "unsupported_original": unsupported_scores_strict(results),
        "record_exact": _rate([r.record_exact for r in results]),
        "typed_no_match": _rate([r.typed_no_match for r in results]),
        "table_match": _rate([r.table_match for r in results]),
        "structured_exact_original": _rate([r.structured_exact_original for r in results]),
        "source_issue_exposure": _rate([r.issues_ok for r in results]),
        "conflict_exposure": _rate([r.conflict_ok for r in results]),
        "by_category": {
            cat: {
                "n": len(rs),
                "route": _rate([r.route_ok for r in rs])["rate"],
                "record_exact": _rate([r.record_exact for r in rs])["rate"],
                "typed_no_match": _rate([r.typed_no_match for r in rs])["rate"],
                "bm25_recall@5": _recall([r.bm25_rank for r in rs if r.has_prose_evidence])[
                    "recall@5"
                ]
                if any(r.has_prose_evidence for r in rs)
                else None,
            }
            for cat, rs in sorted(by_cat.items())
        },
    }


ROUTES = ("structured", "prose", "mixed", "unsupported")


def route_matrix(results: list[CaseResult]) -> dict[str, dict[str, int]]:
    """Expected route (rows) x actual route (columns), strict expected labels."""
    return {
        e: {a: sum(1 for r in results if r.expected_route == e and r.route == a) for a in ROUTES}
        for e in ROUTES
    }


def unsupported_scores(results: list[CaseResult]) -> dict[str, Any]:
    """Unsupported detection honoring acceptable routes, with raw confusion counts.

    tp: routed unsupported where the case allows it; fp: routed unsupported where it does not;
    fn: expected unsupported but routed to a route the case does not allow; ``allowed_other``:
    expected unsupported, routed to an explicitly acceptable other route (neither tp nor fn).
    """
    tp = fp = fn = allowed_other = 0
    for r in results:
        allowed = {r.expected_route, *r.acceptable_routes}
        if r.route == "unsupported":
            if "unsupported" in allowed:
                tp += 1
            else:
                fp += 1
        elif r.expected_route == "unsupported":
            if r.route in allowed:
                allowed_other += 1
            else:
                fn += 1
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "allowed_other": allowed_other,
        "precision": round(tp / (tp + fp), 3) if tp + fp else None,
        "recall": round(tp / (tp + fn), 3) if tp + fn else None,
    }


def unsupported_scores_strict(results: list[CaseResult]) -> dict[str, Any]:
    """Historical scorer: strict expected labels, acceptable routes ignored."""
    tp = sum(1 for r in results if r.expected_route == "unsupported" and r.route == "unsupported")
    fp = sum(1 for r in results if r.expected_route != "unsupported" and r.route == "unsupported")
    fn = sum(1 for r in results if r.expected_route == "unsupported" and r.route != "unsupported")
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(tp / (tp + fp), 3) if tp + fp else None,
        "recall": round(tp / (tp + fn), 3) if tp + fn else None,
    }


def as_rows(results: list[CaseResult]) -> list[dict[str, Any]]:
    return [asdict(r) for r in results]
