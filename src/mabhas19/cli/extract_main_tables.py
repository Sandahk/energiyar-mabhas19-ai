"""Candidate structured extraction of the 11 main-document tables (offline).

Usage:
    uv run python -m mabhas19.cli.audit_main_tables     # first: the table audit
    uv run python -m mabhas19.cli.extract_main_tables

Writes data/processed/main_tables/candidate/: per table ``<number>/cells.jsonl``,
``records.jsonl`` and ``table.yaml``; ``label_corrections_review.md`` with a crop per corrected
label cell; ``summary.json``. Refuses to write if any validator fails. No external API is
called and nothing is written to curated/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pymupdf
import yaml

from mabhas19.core.manifest import load_manifest, sha256_file
from mabhas19.core.paths import CURATED_DIR, DATA_DIR, MANIFEST_PATH, REPO_ROOT, SOURCES_DIR
from mabhas19.tables.main_audit import TABLES
from mabhas19.tables.main_extract import (
    BUILDERS,
    CORRECTIONS_PATH,
    DOC_ID,
    Grid,
    build_cells,
    load_corrections,
    table_meta,
)
from mabhas19.tables.main_validate import (
    condition_bounds,
    validate_prose_isolation,
    validate_symbol_placements,
    validate_table,
    value_objects,
)

BASE = DATA_DIR / "processed" / "main_tables"
OUT = BASE / "candidate"
ROW_NUMBER_FIELDS = ("row_number", "item_number")


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def review_list(pdf: pymupdf.Document, cells: list[dict[str, Any]], crops: Path) -> list[str]:
    lines = []
    for c in cells:
        if not c["correction"] and not c["applied_rules"]:
            continue
        name = f"{c['cell_id']}.png"
        pdf[c["pdf_page"] - 1].get_pixmap(dpi=300, clip=pymupdf.Rect(c["bbox"])).save(crops / name)
        how = (
            c["correction"]["kind"]
            if c["correction"]
            else "rule " + "+".join(r["rule"] for r in c["applied_rules"])
        )
        lines.append(
            f"| {c['cell_id']} | {how} | {c['text_before_correction']!r} | "
            f"{c['text']!r} | crops/{name} |"
        )
    return lines


def main() -> int:
    audit = json.loads((BASE / "audit" / "audit.json").read_text(encoding="utf-8"))
    version = load_manifest(MANIFEST_PATH).get(DOC_ID).current
    if audit["source_sha256"] != version.sha256:
        print("audit is stale: source sha256 differs; re-run audit_main_tables", file=sys.stderr)
        return 2
    corpus = _jsonl(CURATED_DIR / "text" / DOC_ID / "blocks.jsonl")
    blocks = {b["block_id"]: b for b in corpus}
    corrections = load_corrections()
    review = {
        "file": CORRECTIONS_PATH.relative_to(REPO_ROOT).as_posix(),
        "sha256": sha256_file(CORRECTIONS_PATH),
        "basis": "assistant_reviewed",
    }
    source = {
        "path": f"sources/{version.path}",
        "version": version.version,
        "sha256": version.sha256,
    }
    problems: list[str] = []
    built: dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]] = {}
    review_rows: list[str] = []
    (OUT / "crops").mkdir(parents=True, exist_ok=True)
    with pymupdf.open(SOURCES_DIR / version.path) as pdf:
        for spec in TABLES:
            entry = audit["tables"][spec.number]
            cells = build_cells(
                spec,
                _jsonl(BASE / "audit" / spec.number / "cells.jsonl"),
                entry["printed_page"],
                corrections,
            )
            grid = Grid(cells)
            ctx = {
                "blocks": blocks,
                "printed_page": entry["printed_page"],
                "bbox": entry["bbox"],
                "source": source,
                "review": review,
            }
            records, meta = BUILDERS[spec.number](grid, ctx)
            meta = table_meta(spec, grid, records, meta, ctx)
            problems += validate_table(spec.number, cells, records, meta)
            problems += validate_prose_isolation(entry["bbox"], spec.pdf_page, corpus)
            if spec.number == "19-2-1":
                problems += validate_symbol_placements(pdf[spec.pdf_page - 1], cells)
            built[spec.number] = (cells, records, meta)
            review_rows += review_list(pdf, cells, OUT / "crops")
    used = {c["correction"]["id"] for cs, _, _ in built.values() for c in cs if c["correction"]}
    unused = sorted({d["id"] for d in corrections.values()} - used)
    if unused:
        problems.append(f"corrections not applied: {unused}")
    if problems:
        print("extraction blocked:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 2
    summary: dict[str, Any] = {"source_sha256": version.sha256, "tables": {}}
    for number, (cells, records, meta) in built.items():
        d = OUT / number
        d.mkdir(parents=True, exist_ok=True)
        _write_jsonl(d / "cells.jsonl", cells)
        _write_jsonl(d / "records.jsonl", records)
        (d / "table.yaml").write_text(
            yaml.safe_dump(meta, allow_unicode=True, sort_keys=False, width=100),
            encoding="utf-8",
            newline="\n",
        )
        values = [
            v
            for r in records
            for k, f in r.items()
            if k not in ROW_NUMBER_FIELDS
            for v in value_objects(f)
        ]
        bounds = [b for r in records for _, b in condition_bounds(r)]
        summary["tables"][number] = {
            "records": len(records),
            "cells": len(cells),
            "numeric_values": sum(1 for v in values if "token" in v),
            "inequality_bounds": len(bounds),
            "row_numbers": sum(1 for r in records for k in ROW_NUMBER_FIELDS if k in r),
            "value_fields": len(values),
            "no_printed_value": sum(1 for v in values if v.get("status") == "no_printed_value"),
            "symbols": sum(1 for r in records if "printed_symbol" in r),
            "corrected_label_cells": meta["counts"]["corrected_label_cells"],
            "rule_corrected_cells": sum(1 for c in cells if c["applied_rules"]),
            "source_issues": sorted(meta["source_issues"]),
            "validation": "passed",
        }
    (OUT / "label_corrections_review.md").write_text(
        "# Corrected label cells (assistant_reviewed, pending human review)\n\n"
        "| cell | correction | before | after | crop |\n|---|---|---|---|---|\n"
        + "\n".join(review_rows)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(json.dumps(summary["tables"], ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
