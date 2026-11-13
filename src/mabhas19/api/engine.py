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
from dataclasses import dataclass, field, replace
from typing import Any, Literal

import yaml

from mabhas19.api.orchestration import respond
from mabhas19.core.paths import CURATED_DIR, DATA_DIR, EVAL_DIR
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


def prepilot_pins(frozen: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Final-8 pins with the approved pre-pilot module hashes applied (eval/retrieval/
    m2_prepilot_pins.yaml), and the entries that do not start from the Final-8 hash."""
    path = EVAL_DIR / "retrieval" / "m2_prepilot_pins.yaml"
    if not path.exists():
        return frozen, []
    record = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    modules = dict(frozen["code"]["modules_sha256"])
    bad = []
    for module, pin in (record.get("modules_sha256") or {}).items():
        if modules.get(module) != pin["from"]:
            bad.append(f"pre-pilot pin {module}: 'from' is not the Final-8 hash")
            continue
        modules[module] = pin["to"]
    return {**frozen, "code": {**frozen["code"], "modules_sha256": modules}}, bad


def pin_drift() -> list[str]:
    """Differences between the live retrieval engine and the frozen Final-8 candidate, with the
    approved pre-pilot routing patch (one listed module hash) applied."""
    from mabhas19.retrieval.candidate import CURRENT, MANIFESTS, live_candidate  # noqa: PLC0415

    final8 = yaml.safe_load(MANIFESTS[CURRENT].read_text(encoding="utf-8"))["candidate"]
    frozen, out = prepilot_pins(final8)
    live = live_candidate()
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

    def ask(
        self,
        question: str,
        *,
        follow_up: dict[str, Any] | None = None,
        base: str | None = None,
    ) -> tuple[Answer, RetrievalResult, float]:
        """(answer, the main retrieval result it was composed from, latency in ms).

        ``follow_up``/``base``: a clarification continued (the question before the choice was
        added); a typed city for a climate question is looked up first (api.orchestration)."""
        request_id = uuid.uuid4().hex
        start = time.perf_counter()
        with self._lock:  # the retriever keeps per-query caches; one question at a time
            reply = respond(
                self._run, self.store, question, request_id, follow_up=follow_up, base=base
            )
        return reply.answer, reply.result, round((time.perf_counter() - start) * 1000, 1)

    def _run(self, question: str, request_id: str) -> tuple[Answer, RetrievalResult]:
        """One frozen engine call: retrieve, compose (provenance-checked), useful choices."""
        result = self.retriever.retrieve(question)
        answer = compose(result, self.store, request_id)
        return self._useful_choices(answer, result, request_id), result

    def _leads_on(self, question: str, choice: str, field: str) -> bool:
        """Whether choosing ``choice`` moves the question forward: the frozen engine, asked
        the question with the choice added, answers from certified rows or asks for a different
        item (not a refusal, and not the same question again)."""
        follow = compose(self.retriever.retrieve(f"{question} ({choice})"), self.store, "probe")
        if follow.response_type == "structured_value":
            return True
        return follow.response_type == "clarification" and follow.clarifications[0].field != field

    def _useful_choices(self, answer: Answer, result: RetrievalResult, request_id: str) -> Answer:
        """Keep only the choices of an ambiguity or quantity clarification that lead on; with
        none left, the engine's refusal is shown as it was. (Printed category values and the
        rows of a missing row label always constrain the same lookup and are not re-asked.)"""
        if answer.response_type != "clarification":
            return answer
        c = answer.clarifications[0]
        if c.outcome == "missing_required_dimension":
            return answer
        kept = tuple(o for o in c.options if self._leads_on(result.query, o.text, c.field))
        if not kept:
            return compose(result, self.store, request_id, clarify=False)
        if len(kept) == len(c.options):
            return answer
        return replace(answer, clarifications=[replace(c, options=kept)])


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
