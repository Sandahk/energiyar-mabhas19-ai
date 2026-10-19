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
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from mabhas19.indexing.chunking import build_chunks
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
from mabhas19.retrieval.plan import StructuredQueryPlan
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


class Retriever:
    def __init__(self, chunks: list[Chunk], store: StructuredStore) -> None:
        self.lexical = LexicalRetriever(chunks)
        self.store = store
        self.numbered = numbered_lines(chunks)
        self.router = Router(store, self.lexical.index.vocabulary())
        self.planner = Planner(store, store.schemas)
        self.executor = Executor(store, store.schemas)

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
                )
            )
        return result

    def lookup(
        self, dataset: str, query: str, *, explicit_reference: bool = False
    ) -> tuple[list[StructuredHit], list[StructuredNoMatch], list[str], StructuredQueryPlan]:
        """One dataset, bypassing routing: plan -> validation gate -> execution."""
        plan = self.planner.plan(
            dataset, normalize_query(query), explicit_reference=explicit_reference
        )
        hits, no_match, notes = self.executor.run(plan)
        return hits, no_match, notes, plan

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

    def _structured(
        self, norm: str, decision: RouteDecision, refs: list[str], result: RetrievalResult
    ) -> None:
        """Every candidate dataset: plan -> validation gate -> execution (fail closed). The
        computation check reads the whole query, so a calculation asked in another clause is
        never answered with its operands."""
        whole = {r.replace(" ", "") for r in refs}
        full = result.normalized_query
        seen: set[str] = set()
        by_place: dict[str, list[StructuredHit]] = {}
        for ds in decision.tables:
            explicit = ds.removeprefix(MAIN) in whole or any(
                APPENDIX_TABLES.get(r.split("-")[0]) == ds for r in whole if r.startswith("پ")
            )
            plan = self.planner.plan(ds, norm, explicit_reference=explicit, full_norm=full)
            hits, no_match, notes = self.executor.run(plan)
            result.plans.append(plan.to_dict())
            if hits:
                result.enumerations += [{"table": ds, **asdict(e)} for e in plan.enumerations]
            result.constraints += plan.ledger()
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
