"""Embedding experiment plumbing, with a deterministic toy embedder (no model, no network).

The real E5 comparison runs through ``mabhas19.cli.embedding_experiment``; these tests pin the
guarantees it relies on: the lexical mode is the frozen baseline, and switching the prose
ranking to vector or hybrid changes nothing outside prose ranking (routes, structured results,
unsupported decisions, notes, conflicts, source-issue exposure).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from functools import cache
from pathlib import Path

import pytest
import yaml

from mabhas19.cli.embedding_experiment import (
    CANDIDATE,
    INVARIANT,
    check_candidate,
    compare,
    regression,
    token_audit,
)
from mabhas19.core.paths import CURATED_DIR, EVAL_DIR
from mabhas19.indexing.chunking import build_chunks
from mabhas19.retrieval.eval import as_rows, evaluate, load_cases
from mabhas19.retrieval.hybrid import Retriever
from mabhas19.retrieval.models import Chunk
from mabhas19.retrieval.normalization import tokens
from mabhas19.retrieval.structured import StructuredStore
from mabhas19.retrieval.vector import EmbeddingRetriever, VectorIndex, chunks_fingerprint

DIM = 64


class ToyEmbedder:
    """Hashed bag of normalized tokens, L2-normalized (deterministic, dependency-free)."""

    name = "toy-hash"
    dimensions = DIM

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * DIM
        for t in tokens(text):
            v[int(hashlib.md5(t.encode()).hexdigest(), 16) % DIM] += 1.0
        n = sum(x * x for x in v) ** 0.5 or 1.0
        return [x / n for x in v]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


@cache
def _setup() -> tuple[list[Chunk], StructuredStore, VectorIndex]:
    path = CURATED_DIR / "text" / "m19-ed5-1404-main" / "blocks.jsonl"
    blocks = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
    chunks = build_chunks(blocks, "m19-ed5-1404-main")
    return chunks, StructuredStore(CURATED_DIR), VectorIndex.build(chunks, ToyEmbedder())


def test_vector_index_round_trip(tmp_path: Path) -> None:
    chunks, _, index = _setup()
    header = {"model_id": "toy", "chunks_fingerprint": chunks_fingerprint(chunks)}
    index.save(tmp_path, header)
    loaded = VectorIndex.load(tmp_path, chunks, header)
    assert loaded is not None
    assert all(
        abs(a - b) < 1e-6
        for u, v in zip(index.vectors, loaded.vectors, strict=True)
        for a, b in zip(u, v, strict=True)
    )
    assert VectorIndex.load(tmp_path, chunks, {"chunks_fingerprint": "other"}) is None
    assert VectorIndex.load(tmp_path, chunks[:-1], header) is None


def test_lexical_mode_is_the_frozen_baseline() -> None:
    chunks, store, index = _setup()
    cases = load_cases(EVAL_DIR / "retrieval" / "dev.yaml")
    plain, _ = evaluate(Retriever(chunks, store), cases)
    lexical, _ = evaluate(EmbeddingRetriever(chunks, store, ToyEmbedder(), index, "lexical"), cases)
    assert as_rows(plain) == as_rows(lexical)


@pytest.mark.parametrize("set_name", ["dev", "heldout"])
def test_prose_ranking_mode_changes_nothing_but_prose(set_name: str) -> None:
    chunks, store, index = _setup()
    cases = load_cases(EVAL_DIR / "retrieval" / f"{set_name}.yaml")
    runs = {
        mode: evaluate(EmbeddingRetriever(chunks, store, ToyEmbedder(), index, mode), cases)[0]
        for mode in ("lexical", "vector", "hybrid")
    }
    report, rows, violations = compare(runs, cases, set_name)  # type: ignore[arg-type]
    assert violations == []
    for key in (
        "route_accuracy",
        "record_exact",
        "typed_no_match",
        "unsupported",
        "source_issue_exposure",
        "conflict_exposure",
    ):
        assert report["vector"][key] == report["lexical"][key] == report["hybrid"][key]
    assert rows
    assert {r["id"] for r in rows} == {
        r.id for r in runs["lexical"] if r.has_prose_evidence and r.expected_route != "structured"
    }
    assert "route" in INVARIANT
    assert "returned_records" in INVARIANT


def test_vector_and_hybrid_rankings_use_ranks_only() -> None:
    chunks, store, index = _setup()
    vec = EmbeddingRetriever(chunks, store, ToyEmbedder(), index, "vector")
    hyb = EmbeddingRetriever(chunks, store, ToyEmbedder(), index, "hybrid")
    q = "تعریف ساختمان کم مصرف"
    v, h = vec.rank_prose(q, 10), hyb.rank_prose(q, 10)
    assert [x.rank for x in v] == list(range(1, 11))
    assert all(x.score <= 2 / 61 for x in h)  # RRF scores (k=60) of two rankings
    assert [x.score for x in h] == sorted((x.score for x in h), reverse=True)


def test_token_audit_reports_truncation() -> None:
    chunks, _, _ = _setup()
    audit = token_audit([10, 600, 512, 30], chunks[:4], 512)
    assert audit["truncated"] == 1
    assert audit["truncated_chunks"] == [{"chunk": chunks[1].chunk_id, "tokens": 600, "cut": 88}]
    assert audit["max_cut_tokens"] == 88


# ---------------------------------------------------------------- frozen candidate


def test_frozen_candidate_matches_code_and_chunks() -> None:
    from mabhas19.providers import local_embeddings as le  # noqa: PLC0415

    chunks, _, _ = _setup()
    candidate = yaml.safe_load(CANDIDATE.read_text(encoding="utf-8"))
    assert check_candidate(candidate, None, chunks) == []
    emb = candidate["embedding"]
    assert (emb["model_id"], emb["revision"]) == (le.E5_SMALL, le.E5_SMALL_REVISION)
    assert (emb["query_prefix"], emb["passage_prefix"]) == (le.QUERY_PREFIX, le.PASSAGE_PREFIX)
    assert emb["token_audit"]["truncated"] == 0
    assert emb["token_audit"]["max"] < emb["max_seq_length"]


def test_candidate_check_rejects_another_model() -> None:
    chunks, _, _ = _setup()
    candidate = yaml.safe_load(CANDIDATE.read_text(encoding="utf-8"))
    emb = candidate["embedding"]
    prov = {
        **{k: emb[k] for k in ("model_id", "revision", "dimensions", "max_seq_length")},
        **{k: emb[k] for k in ("query_prefix", "passage_prefix", "normalized")},
        "weights_in_snapshot": [{"file": "model.safetensors", "sha256": emb["weights_sha256"]}],
    }
    assert check_candidate(candidate, prov, chunks) == []
    other = {**prov, "revision": "0" * 40, "query_prefix": ""}
    assert len(check_candidate(candidate, other, chunks)) == 2
    pickle = {**prov, "weights_in_snapshot": [{"file": "pytorch_model.bin", "sha256": "x"}]}
    assert check_candidate(candidate, pickle, chunks)


def test_regression_flags() -> None:
    assert regression(1, 1, ["a"], ["a"]) == []
    assert regression(1, 7, ["a", "b"], ["c"]) == [
        "bm25_top1_to_hybrid_beyond5",
        "displaced_by_semantic_only_evidence",
    ]
    assert regression(4, None, ["a"], ["a"]) == ["bm25_top5_to_hybrid_miss"]
    assert regression(3, 2, ["a"], ["z"]) == []  # improvement is never a regression
