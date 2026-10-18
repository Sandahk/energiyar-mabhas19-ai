"""Dense prose retrieval over the frozen chunks (embedding experiment; prose ranking only).

The chunk input is the same text BM25 indexes (:func:`chunk_search_text`: section path, heading,
block text); the embedder adds its own model-input prefix. Vectors are L2-normalized, so the
ranking score is the dot product (= cosine). Vectors are stored as little-endian float32 under
``data/`` with a JSON header recording the model, its revision and the chunk fingerprint; a
stored index is reused only when both match.

:class:`EmbeddingRetriever` keeps every other part of :class:`Retriever` (router, gate,
structured lookup, decomposition, exact section-number fusion) and replaces only the prose
ranking: ``vector`` (cosine only) or ``hybrid`` (RRF of BM25 and cosine, k=60, ranks only).
Embeddings never produce regulatory values.
"""

from __future__ import annotations

import hashlib
import json
import sys
from array import array
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

from mabhas19.providers.embeddings import Embedder
from mabhas19.retrieval.hybrid import RRF_K, Retriever, rrf
from mabhas19.retrieval.lexical import chunk_search_text
from mabhas19.retrieval.models import Chunk, ProseHit
from mabhas19.retrieval.structured import StructuredStore

Mode = Literal["lexical", "vector", "hybrid"]
DEPTH = 50  # candidates per ranking before fusion


def chunks_fingerprint(chunks: Sequence[Chunk]) -> str:
    h = hashlib.sha256()
    for c in chunks:
        h.update(f"{c.chunk_id}\x1f{chunk_search_text(c)}\x1e".encode())
    return h.hexdigest()[:16]


class VectorIndex:
    def __init__(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("one vector per chunk required")
        self.chunks = chunks
        self.vectors = vectors

    @classmethod
    def build(cls, chunks: list[Chunk], embedder: Embedder) -> VectorIndex:
        return cls(chunks, embedder.embed_documents([chunk_search_text(c) for c in chunks]))

    def scores(self, query_vector: Sequence[float]) -> list[tuple[int, float]]:
        out = [
            (i, sum(a * b for a, b in zip(v, query_vector, strict=True)))
            for i, v in enumerate(self.vectors)
        ]
        return sorted(out, key=lambda x: (-x[1], x[0]))

    def save(self, directory: Path, header: dict[str, Any]) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        flat = array("f", (x for v in self.vectors for x in v))
        if sys.byteorder != "little":
            flat.byteswap()
        (directory / "vectors.f32").write_bytes(flat.tobytes())
        meta = {
            **header,
            "chunks": [c.chunk_id for c in self.chunks],
            "dimensions": len(self.vectors[0]) if self.vectors else 0,
        }
        (directory / "index.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8"
        )

    @classmethod
    def load(
        cls, directory: Path, chunks: list[Chunk], expect: dict[str, Any]
    ) -> VectorIndex | None:
        """The stored index, or None when missing or built for other chunks/model."""
        try:
            meta = json.loads((directory / "index.json").read_text(encoding="utf-8"))
            raw = (directory / "vectors.f32").read_bytes()
        except FileNotFoundError:
            return None
        if any(meta.get(k) != v for k, v in expect.items()):
            return None
        if meta["chunks"] != [c.chunk_id for c in chunks]:
            return None
        flat = array("f")
        flat.frombytes(raw)
        if sys.byteorder != "little":
            flat.byteswap()
        d = meta["dimensions"]
        return cls(chunks, [list(flat[i * d : (i + 1) * d]) for i in range(len(chunks))])


class EmbeddingRetriever(Retriever):
    """:class:`Retriever` with the prose ranking replaced by vector or hybrid ranking."""

    def __init__(
        self,
        chunks: list[Chunk],
        store: StructuredStore,
        embedder: Embedder,
        index: VectorIndex,
        mode: Mode = "hybrid",
    ) -> None:
        super().__init__(chunks, store)
        self.embedder = embedder
        self.index = index
        self.mode = mode
        self._query_cache: dict[str, list[float]] = {}

    def query_vector(self, query: str) -> list[float]:
        if query not in self._query_cache:
            self._query_cache[query] = self.embedder.embed_query(query)
        return self._query_cache[query]

    def vector_search(self, query: str, k: int) -> list[ProseHit]:
        ranked = self.index.scores(self.query_vector(query))[:k]
        return [
            self._hit(self.index.chunks[i], score, rank, ["vector"])
            for rank, (i, score) in enumerate(ranked, 1)
        ]

    def rank_prose(self, query: str, k: int) -> list[ProseHit]:
        if self.mode == "lexical":
            return self.lexical.search(query, k)
        if self.mode == "vector":
            return self.vector_search(query, k)
        bm25 = self.lexical.search(query, max(k, DEPTH))
        vec = self.vector_search(query, max(k, DEPTH))
        terms = {h.chunk_id: h.matched_terms for h in bm25}
        chunks = {c.chunk_id: c for c in self.lexical.chunks}
        fused = rrf([[h.chunk_id for h in bm25], [h.chunk_id for h in vec]], RRF_K)[:k]
        return [
            self._hit(chunks[cid], score, rank, terms.get(cid, []))
            for rank, (cid, score) in enumerate(fused, 1)
        ]

    @staticmethod
    def _hit(c: Chunk, score: float, rank: int, matched: list[str]) -> ProseHit:
        return ProseHit(
            chunk_id=c.chunk_id,
            score=round(score, 5),
            rank=rank,
            block_ids=c.block_ids,
            section_path=c.section_path,
            pdf_pages=c.pdf_pages,
            printed_pages=c.printed_pages,
            matched_terms=matched,
            text=c.text,
        )
