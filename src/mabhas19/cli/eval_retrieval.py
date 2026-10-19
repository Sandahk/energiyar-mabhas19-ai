"""Build the offline retrieval artifacts and run a retrieval benchmark set.

Usage:
    uv run python -m mabhas19.cli.eval_retrieval --set dev
    uv run python -m mabhas19.cli.eval_retrieval --set heldout --freeze  # validate, fingerprint
    uv run python -m mabhas19.cli.eval_retrieval --set heldout  # run (fingerprint checked)

Sets live in eval/retrieval/<set>.yaml. ``--freeze`` validates every evidence ID against
curated/ and writes the SHA-256 of the set to eval/retrieval/<set>.sha256 without running any
retrieval. A run refuses to start if a recorded fingerprint no longer matches the file, and the
fingerprint is written into the run's metrics.json. Reads curated/ only; writes
data/retrieval/. No external API or model is used.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mabhas19.core.manifest import sha256_file
from mabhas19.core.paths import CURATED_DIR, DATA_DIR, EVAL_DIR
from mabhas19.indexing.chunking import MAX_WORDS, build_chunks, chunk_stats
from mabhas19.retrieval.eval import (
    as_rows,
    check_evidence_ids,
    check_expectations,
    evaluate,
    load_cases,
)
from mabhas19.retrieval.hybrid import Retriever
from mabhas19.retrieval.lexical import K1, B, save_chunks
from mabhas19.retrieval.models import Chunk
from mabhas19.retrieval.plan import OUTCOMES
from mabhas19.retrieval.structured import StructuredStore

DOC_ID = "m19-ed5-1404-main"
OUT = DATA_DIR / "retrieval"
SETS = EVAL_DIR / "retrieval"


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _validate(
    cases: list[dict[str, Any]],
    blocks: list[dict[str, Any]],
    chunks: list[Chunk],
    store: StructuredStore,
) -> list[str]:
    records = {r["id"] for r in store.app02 + store.app03 + store.app07}
    records |= {r["record_id"] for t in store.main.values() for r in t.records}
    problems = check_evidence_ids(
        cases,
        {b["block_id"] for b in blocks},
        {str(c.section_number) for c in chunks},
        records,
        set(store.main) | set(store.metas),
    )
    enumerable = {
        ds: {d.name for d in schema.dimensions if d.enumerable}
        for ds, schema in store.schemas.items()
    }
    problems += check_expectations(cases, set(store.issues), enumerable, OUTCOMES)
    ids = [c["id"] for c in cases]
    if len(ids) != len(set(ids)):
        problems.append("duplicate case IDs")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--set", default="dev")
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args(argv)
    cases_path = SETS / f"{args.set}.yaml"
    fingerprint_path = SETS / f"{args.set}.sha256"
    sha = sha256_file(cases_path)
    blocks_path = CURATED_DIR / "text" / DOC_ID / "blocks.jsonl"
    blocks = _jsonl(blocks_path)
    chunks = build_chunks(blocks, DOC_ID)
    store = StructuredStore(CURATED_DIR)
    cases = load_cases(cases_path)
    problems = _validate(cases, blocks, chunks, store)
    if problems:
        print("evaluation set has unknown evidence:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 2
    if args.freeze:
        fingerprint_path.write_text(f"{sha}  {cases_path.name}\n", encoding="utf-8", newline="\n")
        print(f"validated {len(cases)} cases; fingerprint {sha} -> {fingerprint_path}")
        return 0
    if (
        fingerprint_path.is_file()
        and fingerprint_path.read_text(encoding="utf-8").split()[0] != sha
    ):
        print(f"{cases_path.name} changed after it was frozen; refusing to run", file=sys.stderr)
        return 2
    stats = chunk_stats(chunks)
    save_chunks(
        chunks,
        OUT / DOC_ID / "chunks.jsonl",
        {
            "source": str(blocks_path.relative_to(CURATED_DIR.parent)),
            "source_sha256": sha256_file(blocks_path),
            "max_words": MAX_WORDS,
            "bm25": {"k1": K1, "b": B},
            "stats": stats,
        },
    )
    results, metrics = evaluate(Retriever(chunks, store), cases)
    run = OUT / "runs" / f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{args.set}"
    run.mkdir(parents=True, exist_ok=True)
    with (run / "results.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for row in as_rows(results):
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {
        "set": args.set,
        "cases_file": f"eval/retrieval/{cases_path.name}",
        "cases_sha256": sha,
        "frozen": fingerprint_path.is_file(),
        "chunks": stats,
        "metrics": metrics,
    }
    (run / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=1))
    print(f"run: {run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
