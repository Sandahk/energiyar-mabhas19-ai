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
from pathlib import Path

from mabhas19.indexing.chunking import build_chunks
from mabhas19.retrieval.answerability import has_words
from mabhas19.retrieval.lexical import LexicalRetriever
from mabhas19.retrieval.models import (
    Chunk,
    ProseHit,
    RetrievalResult,
    RouteDecision,
    StructuredHit,
    StructuredNoMatch,
)
from mabhas19.retrieval.normalization import normalize_query
from mabhas19.retrieval.router import APPENDIX_TABLES, DERIVATION_REQUEST, Router
from mabhas19.retrieval.structured import APP02, APP03, APP07, MAIN, MAIN_3_2, StructuredStore
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
        sections = {p.number for c in chunks for p in c.section_path if p.number}
        self.router = Router(
            store, self.lexical.index.vocabulary(), {str(x) for x in sections} | set(self.numbered)
        )

    @classmethod
    def from_curated(cls, curated: Path, blocks: list[dict[str, object]], doc_id: str) -> Retriever:
        return cls(build_chunks(blocks, doc_id), StructuredStore(curated))

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
        if decision.route in ("structured", "mixed"):
            self._structured(decision.structured_query or norm, decision, refs, result)
            if has_words(norm, DERIVATION_REQUEST):
                result.notes.append(
                    "interpolation/extrapolation is never performed: only printed values are "
                    "returned"
                )
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

    def rank_prose(self, query: str, k: int) -> list[ProseHit]:
        """The prose ranking used for a query (BM25 in the baseline)."""
        return self.lexical.search(query, k)

    def _prose(self, query: str, sections: list[str], k: int) -> list[ProseHit]:
        if not sections:
            return self.rank_prose(query, k)
        return self._fuse_exact(query, sections, k)

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
        tables = decision.tables
        whole = {r.replace(" ", "") for r in refs}
        hits: list[StructuredHit] = []
        if APP02 in tables:
            climate, conflicts = self.store.climate(norm)
            hits += climate
            result.conflicts += conflicts
            ambiguous = self.store.ambiguous_places(self.store.find_places(norm))
            result.notes += [
                f"«{p}» resolves to several stations: {s}" for p, s in ambiguous.items()
            ]
        if MAIN_3_2 in tables and re.search(r"(?<!\d)\d{6}(?!\d)", norm):
            hits += self.store.wmo(norm)
        if APP03 in tables:
            eui, notes, no_match = self.store.eui(norm)
            hits += eui
            result.notes += notes
            result.no_match += no_match
        if APP07 in tables:
            mats, notes, no_match = self.store.materials(norm)
            hits += mats
            result.notes += notes
            result.no_match += no_match
        for t in tables:
            if t.startswith(MAIN) and not (t == MAIN_3_2 and APP02 in tables):
                found, no_match, notes = self.store.main_table(
                    t, norm, whole_table=t.removeprefix(MAIN) in whole
                )
                hits += found
                result.no_match += no_match
                result.notes += notes
        seen: set[str] = set()
        for h in hits:
            if h.record_id not in seen:
                seen.add(h.record_id)
                result.structured.append(h)
