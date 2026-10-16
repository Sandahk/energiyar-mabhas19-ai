"""Audit the 11 ruled tables of the main Mabhas 19 text (offline, candidate data only).

Usage:
    uv run python -m mabhas19.cli.audit_main_tables

Needs the Phase 1 audit output and the curated main corpus (for captions). Writes
data/processed/main_tables/audit/: per table ``<number>/cells.jsonl`` and ``crop.png``, plus
``audit.json`` (per-table summaries and cross-checks) and ``symbol_masks.txt``. No external
API is called and nothing is written to curated/.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pymupdf

from mabhas19.core.manifest import load_manifest
from mabhas19.core.paths import CURATED_DIR, DATA_DIR, MANIFEST_PATH, SOURCES_DIR
from mabhas19.ingest.main_corpus import DOC_ID, Context, audit_dir
from mabhas19.tables.main_audit import (
    SYMBOL_MASKS,
    TABLES,
    audit_table,
    mask_ascii,
    mask_digest,
    page_symbols,
    summarize,
    year_monotonic_violations,
)
from mabhas19.text.normalize import normalize_for_search

OUT = DATA_DIR / "processed" / "main_tables" / "audit"
APP02 = CURATED_DIR / "tables" / "m19-ed5-1404-app02" / "city_climate_classes.jsonl"


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def placement_check(page: pymupdf.Page) -> dict[str, Any]:
    """Image rects from the content-stream parser against PyMuPDF's own image boxes."""
    mine = sorted(tuple(round(v, 1) for v in s["rect"]) for s in page_symbols(page))
    theirs = sorted(tuple(round(v, 1) for v in d["bbox"]) for d in page.get_image_info()
                    if d["width"] == 2 and d["height"] == 2)  # fmt: skip
    return {"parsed": len(mine), "pymupdf": len(theirs), "identical_rects": mine == theirs}


def app02_check(audit: dict[str, Any]) -> dict[str, Any]:
    """Table 19-3-2 against curated Appendix 2 (comparison only, the datasets stay separate)."""
    app02 = {normalize_for_search(r["city"]["text"]): r for r in _jsonl(APP02)}
    rows: dict[int, dict[int, str]] = {}
    for c in audit["cells"]:
        rows.setdefault(c["row"], {})[c["col"]] = c["text"]
    same, differ, absent = [], [], []
    for r, cols in sorted(rows.items()):
        if r == 0:
            continue
        city, cls = cols[2], cols[4]
        hit = app02.get(normalize_for_search(city))
        if hit is None:
            absent.append(city)
        elif hit["climate_class"]["code"] == cls:
            same.append(city)
        else:
            differ.append({"row": r, "city": city, "table_19_3_2": cls, "app02_id": hit["id"],
                           "app02": hit["climate_class"]["code"]})  # fmt: skip
    return {"same_class": len(same), "different_class": differ, "not_found_by_exact_name": absent}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    blocks = {b["block_id"]: b for b in _jsonl(CURATED_DIR / "text" / DOC_ID / "blocks.jsonl")}
    version = load_manifest(MANIFEST_PATH).get(DOC_ID).current
    report: dict[str, Any] = {"source_sha256": version.sha256, "tables": {}}
    with pymupdf.open(SOURCES_DIR / version.path) as pdf:
        ctx = Context(pdf, audit_dir(), None)
        for spec in TABLES:
            near = pymupdf.Rect(blocks[spec.table_block]["bbox"])
            audit = audit_table(ctx, spec, near)
            d = OUT / spec.number
            d.mkdir(exist_ok=True)
            with (d / "cells.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
                for c in audit["cells"]:
                    fh.write(json.dumps(c, ensure_ascii=False) + "\n")
            page = pdf[spec.pdf_page - 1]
            x0, y0, x1, y1 = audit["bbox"]
            clip = pymupdf.Rect(x0 - 4, y0 - 30, x1 + 4, y1 + 4)
            page.get_pixmap(dpi=200, clip=clip).save(str(d / "crop.png"))
            caption = blocks[spec.caption_block]
            entry = {"pdf_page": spec.pdf_page, "printed_page": caption["printed_page"],
                     "caption": caption["text"], "caption_block": spec.caption_block,
                     "table_block": spec.table_block, "bbox": audit["bbox"],
                     **summarize(audit)}  # fmt: skip
            if spec.number == "19-2-1":
                entry["image_placements"] = placement_check(page)
                entry["year_monotonic_violations"] = year_monotonic_violations(audit, 2, 4, 5)
                masks = sorted({i[1] for i in page.get_images(full=True) if i[1] and i[2] == 2})
                art = [f"mask xref {m}: {SYMBOL_MASKS.get(mask_digest(pdf, m), 'unidentified')} "
                       f"({mask_digest(pdf, m)})\n{mask_ascii(pdf, m)}" for m in masks]  # fmt: skip
                (OUT / "symbol_masks.txt").write_text("\n\n".join(art), encoding="utf-8")
            if spec.number == "19-3-2":
                entry["app02_comparison"] = app02_check(audit)
            if spec.number == "19-3-3":
                f = blocks.get("main-p073-f01", {})
                latex = (f.get("formula") or {}).get("latex")
                entry["formula_link"] = {"block": "main-p073-f01", "plain": f.get("text"),
                                         "latex": latex}  # fmt: skip
            report["tables"][spec.number] = entry
    (OUT / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=1),
                                    encoding="utf-8")  # fmt: skip
    for number, e in report["tables"].items():
        print(number, e["pdf_page"], e["grid"], e["value_classes"], e["flags"], e["symbols"],
              e["glyph_rules"], e["parse_errors"])  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
