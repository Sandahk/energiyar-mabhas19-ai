"""Audit every page of every source PDF listed in sources/manifest.yaml.

Usage:
    uv run python -m mabhas19.cli.audit_sources [--doc DOC_ID ...] [--no-render] [--no-tables]

Outputs go to data/interim/<doc_id>/<sha256[:8]>/ plus data/interim/audit-summary.json.
Everything written is a machine suggestion; nothing is marked verified.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from mabhas19.core.manifest import load_manifest, verify_sources
from mabhas19.core.paths import CONFIGS_DIR, DATA_DIR, MANIFEST_PATH, REPO_ROOT, SOURCES_DIR
from mabhas19.ingest.audit.runner import (
    AuditOptions,
    audit_version,
    load_manual_tables,
    load_samples,
)

SAMPLES_CONFIG = CONFIGS_DIR / "audit-samples.yaml"
MANUAL_TABLES = REPO_ROOT / "docs" / "audit" / "manual-table-observations.yaml"
SUMMARY_FILE = "audit-summary.json"


def _matrix(summaries: list[dict[str, Any]]) -> str:
    header = (
        f"{'doc_id':<20} {'pages':>5} {'GOOD':>5} {'DMG':>5} {'LAT':>5} {'NONE':>5} "
        f"{'vect':>5} {'bkmk':>5} {'lbls':>5} {'tbl_a':>5} {'tbl_m':>5} {'pm_hi':>5}"
    )
    rows = [header, "-" * len(header)]
    for s in summaries:
        q = s["quality_counts"]
        rows.append(
            f"{s['doc_id']:<20} {s['page_count']:>5} {q['GOOD']:>5} {q['DAMAGED']:>5} "
            f"{q['LATIN_DIGITS_ONLY']:>5} {q['NO_TEXT']:>5} "
            f"{len(s['vector_text_suspected_pages']):>5} {len(s['pdf_bookmarks']):>5} "
            f"{len(s['pdf_page_labels']):>5} {len(s['table_pages_auto']):>5} "
            f"{len(s['table_pages_manual']):>5} {s['page_map_confidence']['high']:>5}"
        )
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--doc", action="append", help="limit to these doc_ids")
    parser.add_argument("--no-render", action="store_true", help="skip sample rendering")
    parser.add_argument("--no-tables", action="store_true", help="skip table detection")
    parser.add_argument("--out", type=Path, default=DATA_DIR / "interim")
    args = parser.parse_args(argv)

    manifest = load_manifest(MANIFEST_PATH)
    problems = verify_sources(manifest, SOURCES_DIR)
    if problems:
        print("Source integrity check failed:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 1

    options = AuditOptions(
        samples=tuple(load_samples(SAMPLES_CONFIG)),
        manual_tables=tuple(load_manual_tables(MANUAL_TABLES)),
        render_samples=not args.no_render,
        detect_tables=not args.no_tables,
    )
    summaries = []
    for doc in manifest.documents:
        if args.doc and doc.doc_id not in args.doc:
            continue
        print(f"auditing {doc.doc_id}@{doc.current_version} ...", flush=True)
        summaries.append(audit_version(doc, doc.current, SOURCES_DIR, args.out, options))

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / SUMMARY_FILE).write_text(
        json.dumps({"documents": summaries}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print()
    print(_matrix(summaries))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
