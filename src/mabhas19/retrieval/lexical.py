"""Offline BM25 baseline over the section-aware prose chunks.

Indexed per chunk: the section-path titles, the heading text and the final corrected block text,
all through the search normalization (``index_terms``: normalized tokens plus adjacent
concatenations). Standard Okapi BM25 parameters (k1=1.5, b=0.75); nothing is tuned.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from mabhas19.retrieval.models import Chunk, ProseHit, SectionRef
from mabhas19.retrieval.normalization import index_terms

K1 = 1.5
B = 0.75


def chunk_search_text(c: Chunk) -> str:
    path = " ".join(f"{p.number or ''} {p.title}" for p in c.section_path)
    return f"{path}\n{c.heading_text or ''}\n{c.text}"


class BM25:
    def __init__(self, docs: Sequence[Sequence[str]], k1: float = K1, b: float = B) -> None:
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.len = [len(d) for d in docs]
        self.avg = sum(self.len) / max(1, len(self.len))
        df: Counter[str] = Counter()
        for t in self.tf:
            df.update(t.keys())
        n = len(docs)
        self.idf = {w: math.log(1 + (n - f + 0.5) / (f + 0.5)) for w, f in df.items()}

    def scores(self, query: Sequence[str]) -> list[tuple[int, float, list[str]]]:
        out = []
        terms = list(dict.fromkeys(query))
        for i, tf in enumerate(self.tf):
            s, matched = 0.0, []
            norm = self.k1 * (1 - self.b + self.b * self.len[i] / self.avg)
            for w in terms:
                f = tf.get(w)
                if f:
                    s += self.idf[w] * f * (self.k1 + 1) / (f + norm)
                    matched.append(w)
            if s > 0:
                out.append((i, s, matched))
        return sorted(out, key=lambda x: -x[1])

    def vocabulary(self) -> set[str]:
        return set(self.idf)


class LexicalRetriever:
    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self.index = BM25([index_terms(chunk_search_text(c)) for c in chunks])

    def search(self, query: str, k: int = 10) -> list[ProseHit]:
        hits = []
        for rank, (i, score, matched) in enumerate(self.index.scores(index_terms(query))[:k], 1):
            c = self.chunks[i]
            hits.append(
                ProseHit(
                    chunk_id=c.chunk_id,
                    score=round(score, 4),
                    rank=rank,
                    block_ids=c.block_ids,
                    section_path=c.section_path,
                    pdf_pages=c.pdf_pages,
                    printed_pages=c.printed_pages,
                    matched_terms=matched,
                    text=c.text,
                )
            )
        return hits


def save_chunks(chunks: list[Chunk], path: Path, meta: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps({"meta": meta}, ensure_ascii=False) + "\n")
        for c in chunks:
            fh.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")


def load_chunks(path: Path) -> tuple[list[Chunk], dict[str, Any]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    meta = json.loads(lines[0])["meta"]
    chunks = []
    for line in lines[1:]:
        d = json.loads(line)
        d["section_path"] = tuple(SectionRef(**p) for p in d["section_path"])
        for k in ("block_ids", "block_kinds", "pdf_pages", "printed_pages"):
            d[k] = tuple(d[k])
        d["block_pages"] = tuple(tuple(x) for x in d["block_pages"])
        chunks.append(Chunk(**d))
    return chunks, meta
