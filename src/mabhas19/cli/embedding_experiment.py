"""Embedding retrieval experiment: BM25 vs local E5 vector vs BM25+E5 (RRF k=60).

Usage (needs the optional dependency group ``embeddings``):
    uv run --group embeddings python -m mabhas19.cli.embedding_experiment            # dev+heldout
    uv run --group embeddings python -m mabhas19.cli.embedding_experiment --set final

The configuration must match the frozen candidate in eval/retrieval/hybrid_candidate.yaml (model
revision, weights SHA-256, prefixes, dimensions, chunks, BM25 and RRF parameters), and every set
must match its recorded SHA-256 (``final`` must be frozen). Every non-prose component is
unchanged across modes. The run audits chunk token lengths with the model's tokenizer, measures
load/embedding/query latency on CPU, and checks that routes, structured results, unsupported
decisions, notes, conflicts and source-issue exposure are identical across the three modes.
Writes vectors under
``data/retrieval/<doc>/embeddings/`` and the report under ``data/retrieval/embedding_runs/``.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

import yaml

from mabhas19.cli.eval_retrieval import DOC_ID, OUT, SETS, _jsonl
from mabhas19.core.manifest import sha256_file
from mabhas19.core.paths import CURATED_DIR
from mabhas19.indexing.chunking import build_chunks
from mabhas19.retrieval.eval import CaseResult, as_rows, evaluate, load_cases, metrics
from mabhas19.retrieval.hybrid import RRF_K, Retriever
from mabhas19.retrieval.lexical import K1, B, chunk_search_text
from mabhas19.retrieval.models import Chunk
from mabhas19.retrieval.structured import StructuredStore
from mabhas19.retrieval.vector import (
    DEPTH,
    EmbeddingRetriever,
    Mode,
    VectorIndex,
    chunks_fingerprint,
)

MODES: tuple[Mode, ...] = ("lexical", "vector", "hybrid")
SET_NAMES = ("dev", "heldout")
CANDIDATE = SETS / "hybrid_candidate.yaml"
PROBES = ("H022", "H073", "H079", "H112", "R015", "R052", "R053", "H070")
GROUPS = {
    "conceptual": "conceptual",
    "section_lookup": "section_clause",
    "mixed": "mixed_prose",
    "spelling_variant": "spelling",
    "conceptual_colloquial": "conceptual",
}
# fields that must not depend on the prose ranking
INVARIANT = (
    "route",
    "route_reasons",
    "route_ok",
    "structured_exact_original",
    "record_exact",
    "typed_no_match",
    "table_match",
    "structured_missing",
    "structured_extra",
    "issues_ok",
    "conflict_ok",
    "returned_records",
    "conflicts",
    "notes",
    "structured_query",
    "prose_query",
    "no_match",
)
TRUNCATION_STOP = 0.25  # stop before embedding when more chunks than this would be truncated


def _peak_rss_mb() -> float | None:
    if sys.platform != "win32":
        import resource  # noqa: PLC0415

        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    import ctypes  # noqa: PLC0415
    from ctypes import wintypes  # noqa: PLC0415

    class Counters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    c = Counters()
    c.cb = ctypes.sizeof(c)
    psapi = ctypes.WinDLL("psapi")
    kernel = ctypes.WinDLL("kernel32")
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
    ok = psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(c), c.cb)
    return round(c.PeakWorkingSetSize / 2**20, 1) if ok else None


def _pct(values: list[float], q: float) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, round(q * (len(s) - 1)))]


def token_audit(lengths: list[int], chunks: list[Chunk], limit: int) -> dict[str, Any]:
    over = [(c.chunk_id, n) for c, n in zip(chunks, lengths, strict=True) if n > limit]
    return {
        "max_seq_length": limit,
        "count": len(lengths),
        "min": min(lengths),
        "median": statistics.median(lengths),
        "p75": _pct([float(x) for x in lengths], 0.75),
        "p95": _pct([float(x) for x in lengths], 0.95),
        "max": max(lengths),
        "truncated": len(over),
        "truncated_pct": round(100 * len(over) / len(lengths), 2),
        "truncated_chunks": [{"chunk": cid, "tokens": n, "cut": n - limit} for cid, n in over],
        "max_cut_tokens": max((n - limit for _, n in over), default=0),
    }


def check_candidate(
    candidate: dict[str, Any], prov: dict[str, Any] | None, chunks: Sequence[Chunk]
) -> list[str]:
    """Differences between the live configuration and the frozen candidate (empty = frozen)."""
    emb = candidate["embedding"]
    live: dict[str, Any] = {
        "chunks.count": len(chunks),
        "chunks.fingerprint": chunks_fingerprint(chunks),
        "bm25.k1": K1,
        "bm25.b": B,
        "fusion.rrf_k": RRF_K,
        "fusion.depth": DEPTH,
    }
    frozen: dict[str, Any] = {
        "chunks.count": candidate["chunks"]["count"],
        "chunks.fingerprint": candidate["chunks"]["fingerprint"],
        "bm25.k1": candidate["bm25"]["k1"],
        "bm25.b": candidate["bm25"]["b"],
        "fusion.rrf_k": candidate["fusion"]["rrf_k"],
        "fusion.depth": candidate["fusion"]["depth"],
    }
    if prov is not None:
        weights = {w["file"]: w for w in prov["weights_in_snapshot"]}
        keys: tuple[str, ...] = ("model_id", "revision", "dimensions", "max_seq_length")
        keys += ("query_prefix", "passage_prefix", "normalized")
        for key in keys:
            live[f"embedding.{key}"], frozen[f"embedding.{key}"] = prov[key], emb[key]
        live["embedding.weights_sha256"] = weights.get(emb["weights_file"], {}).get("sha256")
        frozen["embedding.weights_sha256"] = emb["weights_sha256"]
        live["embedding.weights_files"] = sorted(weights)
        frozen["embedding.weights_files"] = [emb["weights_file"]]
    return [
        f"{k}: live {live[k]!r} != frozen {frozen[k]!r}" for k in frozen if live[k] != frozen[k]
    ]


def _verdict(base: int | None, other: int | None) -> str:
    b, o = base or 10**6, other or 10**6
    return "tied" if b == o else ("improved" if o < b else "worsened")


def _group(r: CaseResult) -> str:
    return GROUPS.get(r.category, "other_prose")


def _prose_cases(results: list[CaseResult]) -> list[CaseResult]:
    return [r for r in results if r.has_prose_evidence and r.expected_route in ("prose", "mixed")]


def compare(
    runs: dict[Mode, list[CaseResult]], cases: list[dict[str, Any]], set_name: str
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    """Per-mode metrics, per-group breakdown, per-case rows and invariance violations."""
    by_id = {c["id"]: c for c in cases}
    out: dict[str, Any] = {}
    for mode, results in runs.items():
        m = metrics(results)
        prose = _prose_cases(results)
        groups: dict[str, list[CaseResult]] = {}
        for r in prose:
            groups.setdefault(_group(r), []).append(r)
        out[mode] = {
            "intrinsic": m["bm25_intrinsic"],
            "end_to_end": m["prose_end_to_end"],
            "route_accuracy": m["route_accuracy"],
            "structured_exact_original": m["structured_exact_original"],
            "scope_classes": m["scope_classes"],
            "scope_confusion": m["scope_confusion"],
            "structured_no_match_original": _no_match_rate(results, by_id),
            "record_exact": m["record_exact"],
            "typed_no_match": m["typed_no_match"],
            "table_match": m["table_match"],
            "unsupported": m["unsupported"],
            "unsupported_original": m["unsupported_original"],
            "source_issue_exposure": m["source_issue_exposure"],
            "conflict_exposure": m["conflict_exposure"],
            "by_group": {
                g: {
                    "intrinsic": _recall_of([r.bm25_rank for r in rs]),
                    "end_to_end": _recall_of([r.prose_rank for r in rs]),
                }
                for g, rs in sorted(groups.items())
            },
        }
    violations = []
    base = runs["lexical"]
    for mode in MODES[1:]:
        for a, b in zip(base, runs[mode], strict=True):
            for f in INVARIANT:
                if getattr(a, f) != getattr(b, f):
                    violations.append(f"{set_name} {a.id} {mode}: {f} differs")
    rows = []
    for i, r in enumerate(base):
        if not (r.has_prose_evidence and r.expected_route in ("prose", "mixed")):
            continue
        ev = by_id[r.id].get("evidence") or {}
        ranks = {m: (runs[m][i].bm25_rank, runs[m][i].prose_rank) for m in MODES}
        rows.append(
            {
                "set": set_name,
                "id": r.id,
                "category": r.category,
                "group": _group(r),
                "query": r.query,
                "prose_query": r.prose_query,
                "route": r.route,
                "expected": {"blocks": ev.get("blocks", []), "sections": ev.get("sections", [])},
                "intrinsic": {m: ranks[m][0] for m in MODES},
                "end_to_end": {m: ranks[m][1] for m in MODES},
                "top3": {m: runs[m][i].top_evidence for m in MODES},
                "vector_vs_bm25": _verdict(ranks["lexical"][1], ranks["vector"][1]),
                "hybrid_vs_bm25": _verdict(ranks["lexical"][1], ranks["hybrid"][1]),
                "vector_vs_bm25_intrinsic": _verdict(ranks["lexical"][0], ranks["vector"][0]),
                "hybrid_vs_bm25_intrinsic": _verdict(ranks["lexical"][0], ranks["hybrid"][0]),
                "hybrid_regression": regression(
                    ranks["lexical"][1],
                    ranks["hybrid"][1],
                    r.top_chunks,
                    runs["hybrid"][i].top_chunks,
                ),
            }
        )
    return out, rows, violations


def regression(
    bm25: int | None, hybrid: int | None, bm25_top: list[str], hybrid_top: list[str]
) -> list[str]:
    """Material end-to-end regressions of hybrid against BM25 (empty when none)."""
    flags = []
    if bm25 == 1 and (hybrid is None or hybrid > 5):
        flags.append("bm25_top1_to_hybrid_beyond5")
    if bm25 is not None and bm25 <= 5 and hybrid is None:
        flags.append("bm25_top5_to_hybrid_miss")
    if (bm25 or 10**6) < (hybrid or 10**6) and hybrid_top and hybrid_top[0] not in bm25_top:
        # correct evidence pushed down by a chunk that BM25 did not rank in its top 5
        flags.append("displaced_by_semantic_only_evidence")
    return flags


def _no_match_rate(results: list[CaseResult], by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Historical no-match score of the first final benchmark (original scorer)."""
    vals = [
        r.structured_exact_original
        for r in results
        if (by_id[r.id].get("evidence") or {}).get("no_records")
        and r.structured_exact_original is not None
    ]
    return {"rate": round(sum(vals) / len(vals), 3) if vals else None, "n": len(vals)}


def _recall_of(ranks: list[int | None]) -> dict[str, float]:
    n = len(ranks) or 1
    out: dict[str, float] = {
        f"recall@{k}": round(sum(1 for r in ranks if r is not None and r <= k) / n, 3)
        for k in (1, 3, 5)
    }
    out["mrr"] = round(sum(1 / r for r in ranks if r) / n, 3)
    out["n"] = len(ranks)
    return out


def _latency(fn: Any, inputs: list[str]) -> dict[str, float]:
    fn(inputs[0])  # warm-up
    times = []
    for q in inputs:
        t = time.perf_counter()
        fn(q)
        times.append((time.perf_counter() - t) * 1000)
    return {
        "n": len(times),
        "mean_ms": round(statistics.fmean(times), 2),
        "p50_ms": round(statistics.median(times), 2),
        "p95_ms": round(_pct(times, 0.95), 2),
        "max_ms": round(max(times), 2),
    }


def main(argv: list[str] | None = None) -> int:  # noqa: PLR0911, PLR0912, PLR0915
    from mabhas19.providers.local_embeddings import SentenceTransformerEmbedder  # noqa: PLC0415

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--set", dest="sets", action="append", choices=[*SET_NAMES, "final", "final2"]
    )
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args(argv)
    set_names: list[str] = args.sets or list(SET_NAMES)
    candidate = yaml.safe_load(CANDIDATE.read_text(encoding="utf-8"))

    blocks = _jsonl(CURATED_DIR / "text" / DOC_ID / "blocks.jsonl")
    chunks = build_chunks(blocks, DOC_ID)
    store = StructuredStore(CURATED_DIR)
    sets: dict[str, list[dict[str, Any]]] = {}
    if "final2" in set_names:
        from mabhas19.retrieval.candidate import check_manifest  # noqa: PLC0415

        manifest_drift = check_manifest()
        if manifest_drift:
            print("candidate differs from final2_candidate.yaml:\n  " + "\n  ".join(manifest_drift))
            return 2
    problems = check_candidate(candidate, None, chunks)
    if problems:
        print("configuration differs from the frozen candidate:\n  " + "\n  ".join(problems))
        return 2
    for s in set_names:
        path, fp = SETS / f"{s}.yaml", SETS / f"{s}.sha256"
        if s in ("final", "final2") and not fp.is_file():
            print("the final set must be frozen (--freeze) before it is run", file=sys.stderr)
            return 2
        if fp.is_file() and fp.read_text(encoding="utf-8").split()[0] != sha256_file(path):
            print(f"{path.name} changed after it was frozen; refusing to run", file=sys.stderr)
            return 2
        sets[s] = load_cases(path)

    rss0 = _peak_rss_mb()
    t = time.perf_counter()
    embedder = SentenceTransformerEmbedder(candidate["embedding"]["model_id"], threads=args.threads)
    load_s = time.perf_counter() - t
    prov = embedder.provenance()
    problems = check_candidate(candidate, prov, chunks)
    if problems:
        print("model differs from the frozen candidate:\n  " + "\n  ".join(problems))
        return 2
    rss_loaded = _peak_rss_mb()

    lengths = embedder.token_lengths([chunk_search_text(c) for c in chunks])
    audit = token_audit(lengths, chunks, embedder.max_seq_length)
    print(json.dumps({"provenance": prov, "token_audit": audit}, ensure_ascii=False, indent=1))
    if audit["truncated"] / len(chunks) > TRUNCATION_STOP:
        print("too many chunks would be truncated; stopping before embedding", file=sys.stderr)
        return 3

    fingerprint = chunks_fingerprint(chunks)
    slug = str(prov["model_id"]).replace("/", "__")
    vec_dir = OUT / DOC_ID / "embeddings" / slug
    embedder.embed_documents([chunk_search_text(chunks[0])])  # warm-up
    t = time.perf_counter()
    index = VectorIndex.build(chunks, embedder)
    embed_s = time.perf_counter() - t
    again = VectorIndex.build(chunks[:40], embedder)
    drift = max(
        abs(a - b)
        for u, v in zip(index.vectors, again.vectors, strict=False)
        for a, b in zip(u, v, strict=True)
    )
    header = {
        "model_id": prov["model_id"],
        "revision": prov["revision"],
        "chunks_fingerprint": fingerprint,
        "input": "passage prefix + chunk_search_text",
        "normalized": True,
        "dtype": "float32-le",
    }
    index.save(vec_dir, header)
    stored = VectorIndex.load(
        vec_dir, chunks, {"revision": prov["revision"], "chunks_fingerprint": fingerprint}
    )
    assert stored is not None
    index_bytes = sum(f.stat().st_size for f in vec_dir.iterdir())

    runs: dict[str, dict[Mode, list[CaseResult]]] = {}
    baseline_identical = {}
    for s, cases in sets.items():
        runs[s] = {}
        plain, _ = evaluate(Retriever(chunks, store), cases)
        for mode in MODES:
            r = EmbeddingRetriever(chunks, store, embedder, stored, mode)
            runs[s][mode], _ = evaluate(r, cases)
        baseline_identical[s] = as_rows(plain) == as_rows(runs[s]["lexical"])

    report: dict[str, Any] = {"sets": {}, "violations": []}
    all_rows = []
    for s, cases in sets.items():
        comp, rows, violations = compare(runs[s], cases, s)
        report["sets"][s] = comp
        report["violations"] += violations
        all_rows += rows

    # latency on all distinct prose queries of both sets (fresh retriever, empty query cache)
    queries = sorted(
        {r.prose_query or r.query for s in runs for r in _prose_cases(runs[s]["lexical"])}
    )
    probe = EmbeddingRetriever(chunks, store, embedder, stored, "vector")
    qvecs = {q: embedder.embed_query(q) for q in queries}
    lat = {
        "query_embedding": _latency(embedder.embed_query, queries),
        "vector_scoring_only": _latency(lambda q: stored.scores(qvecs[q]), queries),
        "bm25_search": _latency(lambda q: probe.lexical.search(q, 10), queries),
    }
    probe._query_cache = dict(qvecs)
    lat["vector_search_cached_query"] = _latency(lambda q: probe.vector_search(q, 10), queries)
    probe.mode = "hybrid"
    lat["hybrid_ranking_cached_query"] = _latency(lambda q: probe.rank_prose(q, 10), queries)
    fresh = EmbeddingRetriever(chunks, store, embedder, stored, "hybrid")

    def uncached(q: str) -> None:
        fresh._query_cache.clear()
        fresh.retrieve(q, 10)

    lat["hybrid_full_retrieve_uncached"] = _latency(uncached, queries)
    lexical_full = Retriever(chunks, store)
    lat["lexical_full_retrieve"] = _latency(lambda q: lexical_full.retrieve(q, 10), queries)

    probes = [r for r in all_rows if r["id"] in PROBES]
    report.update(
        {
            "created": datetime.now(UTC).isoformat(timespec="seconds"),
            "candidate": "eval/retrieval/hybrid_candidate.yaml",
            "cases_sha256": {s: sha256_file(SETS / f"{s}.yaml") for s in set_names},
            "provenance": prov,
            "token_audit": audit,
            "chunks": {"count": len(chunks), "fingerprint": fingerprint},
            "lexical_mode_identical_to_baseline": baseline_identical,
            "determinism_max_abs_diff_rebuild": drift,
            "performance": {
                "threads": embedder.threads,
                "model_load_s": round(load_s, 2),
                "corpus_embedding_s": round(embed_s, 2),
                "corpus_embedding_ms_per_chunk": round(1000 * embed_s / len(chunks), 1),
                "vector_index_bytes": index_bytes,
                "latency": lat,
                "peak_rss_mb": {
                    "before_load": rss0,
                    "after_load": rss_loaded,
                    "end": _peak_rss_mb(),
                },
            },
            "probes": probes,
            "verdict_counts": {
                s: {
                    key: {
                        v: sum(1 for r in all_rows if r["set"] == s and r[key] == v)
                        for v in ("improved", "worsened", "tied")
                    }
                    for key in (
                        "vector_vs_bm25",
                        "hybrid_vs_bm25",
                        "vector_vs_bm25_intrinsic",
                        "hybrid_vs_bm25_intrinsic",
                    )
                }
                for s in set_names
            },
        }
    )
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run = OUT / "embedding_runs" / f"{stamp}-{'-'.join(set_names)}"
    run.mkdir(parents=True, exist_ok=True)
    (run / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    with (run / "per_case.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for row in all_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    for s in set_names:
        for mode in MODES:
            with (run / f"results-{s}-{mode}.jsonl").open(
                "w", encoding="utf-8", newline="\n"
            ) as fh:
                for row in as_rows(runs[s][mode]):
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        k: report[k]
        for k in (
            "sets",
            "violations",
            "lexical_mode_identical_to_baseline",
            "determinism_max_abs_diff_rebuild",
            "performance",
            "verdict_counts",
        )
    }
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print(f"run: {run}")
    return 1 if report["violations"] or not all(baseline_identical.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
