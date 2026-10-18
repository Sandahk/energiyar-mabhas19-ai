"""Post-hoc corrected evaluation: score frozen sets with the original and the corrected scorer.

Usage:
    uv run python -m mabhas19.cli.rescore_retrieval                 # dev, heldout, final

Runs the deterministic pipeline (routing, structured lookup, BM25 prose; routing and structured
results are identical in every prose mode) on frozen sets and reports, side by side, the
historical scorer (``structured_exact_original``, ``unsupported_original``) and the corrected
scorer (record exact, typed no-match, table match, unsupported honoring acceptable routes).
This is a re-scoring, labelled "post-hoc corrected evaluation"; it never replaces the immutable
first final result recorded in eval/retrieval/final_baseline.yaml. Writes
data/retrieval/rescoring/<timestamp>/.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from typing import Any

from mabhas19.cli.eval_retrieval import DOC_ID, OUT, SETS, _jsonl
from mabhas19.core.manifest import sha256_file
from mabhas19.core.paths import CURATED_DIR
from mabhas19.indexing.chunking import build_chunks
from mabhas19.retrieval.eval import as_rows, evaluate, load_cases
from mabhas19.retrieval.hybrid import Retriever
from mabhas19.retrieval.structured import StructuredStore

LABEL = "post-hoc corrected evaluation"
ORIGINAL = (
    "route_accuracy",
    "unsupported_original",
    "structured_exact_original",
    "source_issue_exposure",
    "conflict_exposure",
)
CORRECTED = (
    "route_accuracy",
    "unsupported",
    "record_exact",
    "typed_no_match",
    "table_match",
    "source_issue_exposure",
    "conflict_exposure",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--set", dest="sets", action="append")
    args = parser.parse_args(argv)
    names: list[str] = args.sets or ["dev", "heldout", "final"]
    chunks = build_chunks(_jsonl(CURATED_DIR / "text" / DOC_ID / "blocks.jsonl"), DOC_ID)
    retriever = Retriever(chunks, StructuredStore(CURATED_DIR))
    run = OUT / "rescoring" / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"label": LABEL, "sets": {}}
    for name in names:
        path, fp = SETS / f"{name}.yaml", SETS / f"{name}.sha256"
        sha = sha256_file(path)
        if fp.is_file() and fp.read_text(encoding="utf-8").split()[0] != sha:
            print(f"{path.name} changed after it was frozen; refusing", file=sys.stderr)
            return 2
        results, m = evaluate(retriever, load_cases(path))
        report["sets"][name] = {
            "cases_sha256": sha,
            "original_scorer": {k: m[k] for k in ORIGINAL},
            LABEL: {k: m[k] for k in CORRECTED},
            "prose_bm25": {"intrinsic": m["bm25_intrinsic"], "end_to_end": m["prose_end_to_end"]},
        }
        with (run / f"results-{name}.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
            for row in as_rows(results):
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    (run / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=1))
    print(f"run: {run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
