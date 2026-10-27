"""Retrieval pipeline: route, then prose (BM25) and/or structured lookup; rank fusion.

Mixed queries are decomposed by the router: the structured part drives the lookups, the prose
part drives BM25 (both are recorded on the route). The offline baseline uses BM25 only. When a
vector retriever is added (an :class:`~mabhas19.providers.embeddings.Embedder` over the same
chunks), prose rankings are combined with reciprocal rank fusion, which uses ranks only, so
BM25 and cosine scores are never added. Exact section-number matches already join BM25 this
way (RRF k=60, the common default; nothing tuned).

Structured outcomes are typed: records, :class:`StructuredNoMatch` (known table without an
applicable printed value, or an unknown entity) and :class:`TableEvidence` for direct table
references. Captions are table-level evidence, never value records.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from mabhas19.indexing.chunking import build_chunks
from mabhas19.retrieval.admission import PlanAdmission, admit
from mabhas19.retrieval.certification import certified_output, certify
from mabhas19.retrieval.coverage import Candidate, clauses_of, split_coordinated
from mabhas19.retrieval.coverage import evaluate as coverage_evaluate
from mabhas19.retrieval.executor import Executor
from mabhas19.retrieval.lexical import LexicalRetriever
from mabhas19.retrieval.models import (
    Chunk,
    Conflict,
    ProseHit,
    RetrievalResult,
    RouteDecision,
    StructuredHit,
    StructuredNoMatch,
)
from mabhas19.retrieval.normalization import normalize_query
from mabhas19.retrieval.plan import PlanOutcome, StructuredQueryPlan, validate
from mabhas19.retrieval.planner import Planner
from mabhas19.retrieval.router import APPENDIX_TABLES, Router
from mabhas19.retrieval.structured import MAIN, StructuredStore
from mabhas19.text.normalize import normalize_for_search

RRF_K = 60


def rrf(rankings: Sequence[Sequence[str]], k: int = RRF_K) -> list[tuple[str, float]]:
    """Reciprocal rank fusion of ranked ID lists (rank positions only)."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, 1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])


_LINE_NUMBER = re.compile(r"^(19(?:-\d+){1,4})(?![\d-])")


def numbered_lines(chunks: list[Chunk]) -> dict[str, list[str]]:
    """Section/clause number -> chunks containing a line (heading or clause) that starts with it."""
    out: dict[str, list[str]] = {}
    for c in chunks:
        for line in c.text.split("\n"):
            m = _LINE_NUMBER.match(normalize_for_search(line))
            if m and c.chunk_id not in out.get(m[1], []):
                out.setdefault(m[1], []).append(c.chunk_id)
    return out


def _referenced_tables(refs: list[str], store: StructuredStore) -> list[str]:
    out = []
    for raw in refs:
        ref = raw.replace(" ", "")
        ds = APPENDIX_TABLES.get(ref.split("-")[0]) if ref.startswith("پ") else MAIN + ref
        if ds and (ds in store.main or ds in store.metas):
            out.append(ds)
    return out


def _ledger(cand: Candidate, hits: list[StructuredHit]) -> list[dict[str, Any]]:
    """The plan's constraint ledger; a coordinated member's values bind its own rows only."""
    entries = cand.plan.ledger()
    if cand.member is None:
        return entries
    rows = [h.record_id for h in hits]
    return [{**e, "member": cand.member, "rows": rows} for e in entries]


class Retriever:
    def __init__(self, chunks: list[Chunk], store: StructuredStore) -> None:
        self.lexical = LexicalRetriever(chunks)
        self.store = store
        self.numbered = numbered_lines(chunks)
        self.router = Router(store, self.lexical.index.vocabulary())
        self.planner: Planner = self.router.planner  # the planner the router dry-runs
        self.executor = Executor(store, store.schemas)
        self._records: dict[str, dict[str, Any]] | None = None

    @classmethod
    def from_curated(cls, curated: Path, blocks: list[dict[str, object]], doc_id: str) -> Retriever:
        return cls(build_chunks(blocks, doc_id), StructuredStore(curated))

    def _appendix_issue_notes(self, decision: RouteDecision) -> list[str]:
        """Registered source issues of an appendix the query names explicitly, when that
        appendix has no curated dataset (curated appendices link issues per record): the issue's
        locations name the appendix document itself (e.g. SI-0007, a title printed differently
        on the cover)."""
        out = []
        for ref in decision.signals.get("references", []) or []:
            if ref.get("kind") != "appendix" or ref.get("status") != "valid":
                continue
            n = int(str(ref["ref"]).split()[-1])
            appendix = self.router.scope.appendices.get(n) or {}
            if appendix.get("curated_dataset"):
                continue
            doc = f"{MAIN.split('-main')[0]}-app{n:02d}"
            for issue in self.store.issues.values():
                if any(loc.get("doc_id") == doc for loc in issue.get("locations") or []):
                    out.append(f"{issue['id']} (appendix {n}): {issue.get('title', '')}")
        return out

    def retrieve(self, query: str, k: int = 10) -> RetrievalResult:
        norm = normalize_query(query)
        decision = self.router.route(query)
        result = RetrievalResult(query=query, normalized_query=norm, route=decision)
        refs = [str(r) for r in decision.signals.get("table_ref", []) or []]
        for ds in _referenced_tables(refs, self.store):
            result.tables.append(self.store.table_evidence(ds, "direct table reference"))
        if decision.route in ("prose", "mixed"):
            sections = [str(r) for r in decision.signals.get("section_ref", []) or []]
            result.prose = self._prose(decision.prose_query or norm, sections, k)
            self._attach_block_issues(result.prose)
            result.notes += self._appendix_issue_notes(decision)
        if decision.route in ("structured", "mixed"):
            self._structured(decision.structured_query or norm, decision, refs, result)
        for u in decision.signals.get("unknown_entities", []) or []:
            result.no_match.append(
                StructuredNoMatch(
                    u["dataset"],
                    f"unknown_{u['type']}",
                    f"no curated {u['type']} matches {u['tokens']}",
                    {"tokens": u["tokens"]},
                    self.store.table_evidence(u["dataset"], f"unknown {u['type']}"),
                    outcome="unknown_entity",  # the planner's dry-run outcome for this table
                )
            )
        return result

    def lookup(
        self, dataset: str, query: str, *, explicit_reference: bool = False
    ) -> tuple[list[StructuredHit], list[StructuredNoMatch], list[str], StructuredQueryPlan]:
        """One dataset, bypassing routing: plan -> validation gate -> admission -> execution
        -> result certification."""
        norm = normalize_query(query)
        plan = self.planner.plan(dataset, norm, explicit_reference=explicit_reference)
        hits, no_match, notes, _admission, _failures = self.execute(
            plan, norm, norm, [dataset] if explicit_reference else []
        )
        return hits, no_match, notes, plan

    def execute(
        self, plan: StructuredQueryPlan, text: str, full: str, explicit_targets: list[str]
    ) -> tuple[
        list[StructuredHit], list[StructuredNoMatch], list[str], PlanAdmission, list[dict[str, str]]
    ]:
        """Admit one plan, pass it through the coverage stage alone, execute and certify it."""
        cand = self.admit_plan(plan, text, full, explicit_targets)
        coverage_evaluate(self.planner, self.store, [cand], full, clauses_of(full, []))
        hits, no_match, notes, failures = self.run_candidate(cand)
        return hits, no_match, notes, cand.admission, failures

    def admit_plan(
        self,
        plan: StructuredQueryPlan,
        text: str,
        full: str,
        explicit_targets: list[str],
        member: str | None = None,
    ) -> Candidate:
        """Admission and the validation gate of one plan (nothing is executed)."""
        schema = self.store.schemas[plan.target]
        admission = admit(
            self.planner,
            plan,
            schema,
            text,
            full,
            explicit_targets=explicit_targets,
            output_text=self.router._bilingual(full),
        )
        gate = validate(plan, schema)
        if gate is not None and admission.admitted:
            # the validation gate already refuses the plan: never admitted
            admission.admitted, admission.outcome = False, gate
        return Candidate(plan, admission, gate, text, member)

    def run_candidate(
        self, cand: Candidate
    ) -> tuple[list[StructuredHit], list[StructuredNoMatch], list[str], list[dict[str, str]]]:
        """Execute and certify a candidate the coverage stage left executable. A plan the gate
        rejects keeps its typed outcome; a valid plan that admission or coverage refuses never
        executes (an unowned one is not what the question asked: no no-match either); a row that
        fails certification removes every row of the plan (fail closed) and is reported as a
        safety failure."""
        plan, admission = cand.plan, cand.admission
        if cand.gate is None and not admission.admitted:
            assert admission.outcome is not None
            return [], [self.executor._no_match(plan, admission.outcome)], list(plan.notes), []
        if cand.gate is None and not cand.owned:
            return [], [], list(plan.notes), []
        if cand.gate is None and cand.outcome is not None:
            return [], [self.executor._no_match(plan, cand.outcome)], list(plan.notes), []
        hits, no_match, notes = self.executor.run(plan)
        schema = self.store.schemas[plan.target]
        failures: list[dict[str, str]] = []
        for h in hits:
            record = self._record(h)
            failures += [
                f.to_dict()
                for f in certify(self.store, self.store.schemas, admission, h.dataset, record)
            ]
            printed = (
                certified_output(schema, admission.output, record) if admission.output else None
            )
            if printed is not None:
                h.output, h.unit = printed
                if h.output != admission.output:
                    notes.append(
                        f"{h.record_id} prints its requirement as {h.output}, the declared "
                        f"alternative of {admission.output}"
                    )
        if failures:
            outcome = PlanOutcome(
                "certification_failed",
                f"returned rows failed independent certification: {failures[:3]}; nothing is "
                "returned",
                "certification_failed",
            )
            return [], [self.executor._no_match(plan, outcome)], notes, failures
        return hits, no_match, notes, []

    def _check_coordination(
        self, members: dict[str, dict[str, list[str]]], result: RetrievalResult
    ) -> None:
        """I8 parent check: every row of coordinated child plans answers exactly one member
        (each child's rows are already certified against that member). A row claimed by two
        members removes every row of that table (fail closed)."""
        for ds, rows in members.items():
            owners = Counter(rid for ids in rows.values() for rid in ids)
            shared = sorted(rid for rid, n in owners.items() if n > 1)
            if not shared:
                continue
            result.structured = [h for h in result.structured if h.dataset != ds]
            result.certification_failures += [
                {
                    "record": rid,
                    "table": ds,
                    "check": "coordination",
                    "detail": "the row maps to several coordinated members",
                }
                for rid in shared
            ]

    def _record(self, hit: StructuredHit) -> dict[str, Any]:
        """The printed record behind a hit (certification reads the row itself)."""
        if self._records is None:
            s = self.store
            rows = [*s.app02, *s.app03, *s.app07, *(r for t in s.main.values() for r in t.records)]
            self._records = {str(r.get("id") or r.get("record_id")): r for r in rows}
        return self._records[hit.record_id]

    def rank_prose(self, query: str, k: int) -> list[ProseHit]:
        """The prose ranking used for a query (BM25 in the baseline)."""
        return self.lexical.search(query, k)

    def _prose(self, query: str, sections: list[str], k: int) -> list[ProseHit]:
        if not sections:
            return self.rank_prose(query, k)
        return self._fuse_exact(query, sections, k)

    def _attach_block_issues(self, hits: list[ProseHit]) -> None:
        """Provenance only, after ranking: each hit inherits the deduplicated source issues of
        its own blocks, with the blocks each issue came from (no ranking or text change)."""
        for h in hits:
            origins: dict[str, list[str]] = {}
            for bid in h.block_ids:
                for issue in self.store.block_issues.get(bid, []):
                    origins.setdefault(issue, []).append(bid)
            h.source_issues = list(origins)
            h.issue_origins = origins

    def _fuse_exact(self, query: str, refs: list[str], k: int) -> list[ProseHit]:
        """RRF of the exact section-number list and the prose ranking (ranks only)."""
        exact = [cid for ref in refs for cid in self.numbered.get(ref, [])]
        bm25 = self.rank_prose(query, max(k, 50))
        by_id = {h.chunk_id: h for h in bm25}
        chunks = {c.chunk_id: c for c in self.lexical.chunks}
        out = []
        for rank, (cid, score) in enumerate(rrf([exact, [h.chunk_id for h in bm25]])[:k], 1):
            h = by_id.get(cid)
            c = chunks[cid]
            out.append(
                ProseHit(
                    chunk_id=cid,
                    score=round(score, 5),
                    rank=rank,
                    block_ids=c.block_ids,
                    section_path=c.section_path,
                    pdf_pages=c.pdf_pages,
                    printed_pages=c.printed_pages,
                    matched_terms=(h.matched_terms if h else [])
                    + (["exact section number"] if cid in exact else []),
                    text=c.text,
                )
            )
        return out

    def _structured(  # noqa: PLR0912
        self, norm: str, decision: RouteDecision, refs: list[str], result: RetrievalResult
    ) -> None:
        """Every candidate dataset: plan -> validation gate -> execution (fail closed). The
        computation check reads the whole query, so a calculation asked in another clause is
        never answered with its operands."""
        whole = {r.replace(" ", "") for r in refs}
        full = result.normalized_query
        explicit_targets = _referenced_tables(refs, self.store)
        seen: set[str] = set()
        by_place: dict[str, list[StructuredHit]] = {}
        cands: list[Candidate] = []
        clauses = clauses_of(full, list(decision.signals.get("clauses") or []))
        for ds in decision.tables:
            explicit = ds.removeprefix(MAIN) in whole or any(
                APPENDIX_TABLES.get(r.split("-")[0]) == ds for r in whole if r.startswith("پ")
            )
            plan = self.planner.plan(ds, norm, explicit_reference=explicit, full_norm=full)
            # I8: coordinated entities run as one child plan per member
            schema = self.store.schemas[ds]
            for child, member in split_coordinated(plan, schema, self.planner, full, clauses):
                # a coordinated row member is admitted on its own conjunct (the other members'
                # words are not its modifiers); a place member on the whole structured text
                text = member if member and schema.family != "place" and member in full else norm
                cands.append(self.admit_plan(child, text, full, explicit_targets, member))
        # the query-level coverage stage sees every candidate (refused ones included)
        ledger = coverage_evaluate(self.planner, self.store, cands, full, clauses)
        result.coverage = ledger.to_dict()
        members: dict[str, dict[str, list[str]]] = {}
        for cand, verdict in zip(cands, ledger.verdicts, strict=True):
            plan, admission = cand.plan, cand.admission
            hits, no_match, notes, failures = self.run_candidate(cand)
            if cand.member is not None:
                members.setdefault(plan.target, {})[cand.member] = [h.record_id for h in hits]
            result.plans.append(plan.to_dict())
            result.admissions.append({**admission.to_dict(), "coverage": verdict})
            result.certification_failures += failures
            if hits:
                result.enumerations += [{"table": ds, **asdict(e)} for e in plan.enumerations]
            result.constraints += _ledger(cand, hits)
            result.no_match += no_match
            result.notes += [n for n in notes if n not in result.notes]
            if plan.computation:
                note = (
                    "interpolation, extrapolation, averages, totals and ratios are never "
                    "performed: only printed values exist"
                )
                if note not in result.notes:
                    result.notes.append(note)
            for h in hits:
                if h.record_id in seen:
                    continue
                seen.add(h.record_id)
                result.structured.append(h)
                entity = next(
                    (e for e in plan.entities if e.kind == "place" and e.status == "resolved"), None
                )
                if entity is not None:
                    by_place.setdefault(entity.text, []).append(h)
        self._check_coordination(members, result)
        for place, group in by_place.items():
            codes = {str(h.values["climate_class"]["code"]) for h in group}
            if len(codes) > 1:
                issues = sorted({i for h in group for i in h.source_issues})
                result.conflicts.append(
                    Conflict(
                        issue_id=issues[0] if issues else None,
                        field="climate_class",
                        hits=[h.record_id for h in group],
                        values=sorted(codes),
                        note=f"curated sources disagree for «{place}»; both are returned, "
                        "neither is preferred",
                    )
                )
            for p, stations in self.store.ambiguous_places(place.split(", ")).items():
                note = f"«{p}» resolves to several stations: {stations}"
                if note not in result.notes:
                    result.notes.append(note)
