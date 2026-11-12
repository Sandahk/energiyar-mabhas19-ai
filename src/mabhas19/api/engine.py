"""The frozen retrieval engine behind the pilot chatbot, and the answer pipeline.

``hybrid`` (the default) is the frozen Final-8 candidate: BM25 + multilingual-e5-small + RRF over
the stored vector index, with the deterministic router and structured lookup. Before serving,
:func:`pin_drift` compares the live engine with the Final-8 candidate manifest - retrieval module
hashes, curated tree, schemas, scope map, chunks, BM25/RRF and E5 pins. The whole-``src`` git
tree hash is not compared: adding the answer layer changes it without touching retrieval.

``lexical`` (BM25 only) exists for tests and machines without the embedding model; it is not the
frozen prose ranking and :attr:`Engine.info` says so.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

import yaml

from mabhas19.core.paths import CURATED_DIR, DATA_DIR
from mabhas19.generation.composer import COMPOSER_VERSION, compose
from mabhas19.generation.contract import CONTRACT_VERSION, Answer
from mabhas19.indexing.chunking import build_chunks
from mabhas19.retrieval.hybrid import Retriever
from mabhas19.retrieval.models import RetrievalResult
from mabhas19.retrieval.structured import StructuredStore

DOC_ID = "m19-ed5-1404-main"
Mode = Literal["hybrid", "lexical"]
# Final-8 pins that may differ: the whole-src tree and the clean-tree flag change with any new
# module (the answer layer); the per-module hashes of retrieval stay compared
_UNPINNED = ("code.src_tree", "code.src_clean")


def pin_drift() -> list[str]:
    """Differences between the live retrieval engine and the frozen Final-8 candidate."""
    from mabhas19.retrieval.candidate import CURRENT, MANIFESTS, live_candidate  # noqa: PLC0415

    frozen = yaml.safe_load(MANIFESTS[CURRENT].read_text(encoding="utf-8"))["candidate"]
    live = live_candidate()
    out = []
    for section, values in frozen.items():
        if not isinstance(values, dict):
            if live.get(section) != values:
                out.append(f"{section}: frozen {values!r} != live {live.get(section)!r}")
            continue
        for key, value in values.items():
            if f"{section}.{key}" in _UNPINNED:
                continue
            got = (live.get(section) or {}).get(key)
            if got != value:
                out.append(f"{section}.{key}: frozen {str(value)[:60]!r} != live {str(got)[:60]!r}")
    return out


@dataclass
class Engine:
    retriever: Retriever
    store: StructuredStore
    mode: Mode
    info: dict[str, Any] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def ask(self, question: str) -> tuple[Answer, RetrievalResult, float]:
        """(answer, the retrieval result it was composed from, latency in ms)."""
        request_id = uuid.uuid4().hex
        start = time.perf_counter()
        with self._lock:  # the retriever keeps per-query caches; one question at a time
            result = self.retriever.retrieve(question)
            answer = compose(result, self.store, request_id)
        return answer, result, round((time.perf_counter() - start) * 1000, 1)


def load_engine(mode: Mode = "hybrid", threads: int = 4) -> Engine:
    path = CURATED_DIR / "text" / DOC_ID / "blocks.jsonl"
    blocks = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    chunks = build_chunks(blocks, DOC_ID)
    store = StructuredStore(CURATED_DIR)
    info: dict[str, Any] = {
        "mode": mode,
        "composer": COMPOSER_VERSION,
        "contract": CONTRACT_VERSION,
        "chunks": len(chunks),
    }
    if mode == "lexical":
        info["frozen_prose_ranking"] = False
        return Engine(Retriever(chunks, store), store, mode, info)

    from mabhas19.cli.embedding_experiment import CANDIDATE, check_candidate  # noqa: PLC0415
    from mabhas19.providers.local_embeddings import SentenceTransformerEmbedder  # noqa: PLC0415
    from mabhas19.retrieval.vector import (  # noqa: PLC0415
        EmbeddingRetriever,
        VectorIndex,
        chunks_fingerprint,
    )

    candidate = yaml.safe_load(CANDIDATE.read_text(encoding="utf-8"))
    embedder = SentenceTransformerEmbedder(candidate["embedding"]["model_id"], threads=threads)
    prov = embedder.provenance()
    problems = check_candidate(candidate, prov, chunks)
    if problems:
        raise RuntimeError(
            "the embedding model differs from the frozen candidate: " + "; ".join(problems)
        )
    slug = str(prov["model_id"]).replace("/", "__")
    fingerprint = chunks_fingerprint(chunks)
    index = VectorIndex.load(
        DATA_DIR / "retrieval" / DOC_ID / "embeddings" / slug,
        chunks,
        {"revision": prov["revision"], "chunks_fingerprint": fingerprint},
    )
    if index is None:
        raise RuntimeError("the stored vector index is missing or was built for other chunks")
    info.update(
        {
            "frozen_prose_ranking": True,
            "architecture": candidate["architecture"],
            "model_id": prov["model_id"],
            "revision": prov["revision"],
            "chunks_fingerprint": fingerprint,
        }
    )
    retriever = EmbeddingRetriever(chunks, store, embedder, index, "hybrid")
    return Engine(retriever, store, mode, info)
