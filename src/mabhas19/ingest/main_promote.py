"""Promotion of the reviewed main-text corpus to curated files (preview by default).

Proposed curated layout (paths relative to the curated root):

* ``page_maps/m19-ed5-1404-main.jsonl``: 146 pages (role, printed page, methods);
* ``structure/m19-ed5-1404-main/sections.jsonl`` and ``toc.jsonl``: heading hierarchy, TOC;
* ``text/m19-ed5-1404-main/blocks.jsonl``: every block with raw, repair_v1 and corrected text,
  rules, decisions, verification status and retrieval eligibility;
* ``text/m19-ed5-1404-main/corpus.yaml``: source, methods, review files, rule counts, counts.

Ruled tables are kept as references only (bbox, size, caption); their cells stay in
data/processed until the structured table extraction. The validation refuses the promotion
if any check fails.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pymupdf
import yaml

from mabhas19.core.manifest import load_manifest, sha256_file
from mabhas19.core.paths import MANIFEST_PATH, REPO_ROOT
from mabhas19.ingest.main_decisions import (
    DECISIONS_PATH,
    ERRATA_PATH,
    STRUCTURE_PATH,
    load_decisions,
    load_errata,
    load_structure,
)
from mabhas19.tables.main_audit import TABLES
from mabhas19.tables.main_extract import caption

DOC_ID = "m19-ed5-1404-main"
EXPECTED_PAGES = 146
EXPECTED_RULES = {
    "M1": 383,
    "M1-exception": 2,
    "G1": 49,
    "G2": 23,
    "B1": 5,
    "C1": 2,
    "K1": 1,
    # main-corpus errata (2026-10-17): occurrences of the reviewed cases only
    "B2": 106,
    "L1": 62,
    "T1": 10,
}
EXPECTED_ERRATA = {"blocks": 178, "candidates": 259, "B2": 106, "L1": 60, "T1": 10, "per_block": 83}
BLOCK_FIELDS = (
    "block_id",
    "doc_id",
    "pdf_page",
    "printed_page",
    "printed_page_confidence",
    "page_role",
    "kind",
    "structural_class",
    "section_number",
    "heading_level",
    "section_path",
    "bbox",
    "extraction_method",
    "text_raw",
    "text_repair_v1",
    "text_before_review",
    "text",
    "text_normalized",
    "applied_rules",
    "decisions",
    "verification",
    "retrieval",
    "resolved_flags",
    "open_flags",
    "confidence",
    "font_size",
    "line_count",
    "formula",
    "fragments",
    "merged_into",
    "continues_previous_block",
    "retrieval_exclusion",
    "ocr_mean_confidence",
    "rows",
    "cols",
    "table_index",
    "figure_index",
    "embedded_symbol_images",
    "text_before_errata",
    "caption_block",
    "errata",
    "reclassification",
)


def _load(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def curated_block(b: dict[str, Any]) -> dict[str, Any]:
    rec = {k: b[k] for k in BLOCK_FIELDS if k in b}
    if b["kind"] == "table":
        rec["table_cells"] = "not promoted: pending structured extraction (data/processed)"
    return rec


def _validate_blocks(blocks: list[dict[str, Any]]) -> list[str]:
    problems: list[str] = []
    ids = [b["block_id"] for b in blocks]
    if len(ids) != len(set(ids)):
        problems.append("duplicate block ids")
    open_flags = [(b["block_id"], b["open_flags"]) for b in blocks if b.get("open_flags")]
    if open_flags:
        problems.append(f"open review flags: {open_flags[:5]}")
    for b in blocks:
        if not b.get("verification", {}).get("status"):
            problems.append(f"{b['block_id']}: no verification status")
        if b.get("text") is not None and b.get("text_raw") is None:
            problems.append(f"{b['block_id']}: repaired text without raw text")
        furniture = b["kind"] in ("running_header", "page_number")
        if b["page_role"] == "body" and not furniture and not b.get("section_path"):
            problems.append(f"{b['block_id']}: body block without section path")
    for t in (b for b in blocks if b["kind"] == "table"):
        tr = pymupdf.Rect(t["bbox"])
        leaked = [
            b["block_id"]
            for b in blocks
            if b["pdf_page"] == t["pdf_page"]
            and b.get("text")
            and b["kind"] not in ("table_caption", "running_header", "page_number")
            and tr.contains(pymupdf.Rect(b["bbox"]))
        ]
        if leaked:
            problems.append(f"{t['block_id']}: prose blocks inside the table {leaked}")
    return problems


def _validate_corpus(
    blocks: list[dict[str, Any]], pages: list[dict[str, Any]], summary: dict[str, Any]
) -> list[str]:
    problems: list[str] = []
    if len(pages) != EXPECTED_PAGES:
        problems.append(f"{len(pages)} pages, expected {EXPECTED_PAGES}")
    rules = Counter(
        r["rule"] for b in blocks for r in b.get("applied_rules", []) for _ in range(r["count"])
    )
    if dict(rules) != EXPECTED_RULES:
        problems.append(f"rule applications {dict(rules)} differ from {EXPECTED_RULES}")
    decided = {d for b in blocks for d in b["verification"]["decisions"]}
    errata = load_errata()["decisions"]
    reviewed = {
        d["id"]
        for d in errata
        if "reclassify" in d or any(not r["rule"] for r in d["replacements"])
    }
    expected = {d["id"] for d in load_decisions()["decisions"]} | reviewed
    if decided != expected:
        problems.append(f"decisions not applied: {sorted(expected ^ decided)}")
    applied = {b["errata"]["id"] for b in blocks if b.get("errata")}
    if applied != {d["id"] for d in errata}:
        problems.append("errata decisions not all applied")
    problems += _validate_errata(errata)
    problems += _validate_structure(blocks)
    checks = summary["heading_checks"]
    if checks["audit_headings_found"] != checks["audit_headings"]:
        problems.append("audit headings not all found")
    if checks["toc_matched_to_body_heading"] != checks["toc_numbered"]:
        problems.append("TOC entries without a body heading")
    return problems


def _validate_errata(errata: list[dict[str, Any]]) -> list[str]:
    reps = [r for d in errata for r in d["replacements"]]
    cands: dict[str, set[str]] = {}
    for r in reps:
        cands.setdefault(r["rule"] or "per_block", set()).update(r["candidates"])
    cands["per_block"] = cands.get("per_block", set()) | {
        c for d in errata for c in d.get("reclassify", {}).get("candidates", [])
    }
    got = {k: len(v) for k, v in cands.items()}
    got |= {"blocks": len(errata), "candidates": sum(got.values())}
    if got != EXPECTED_ERRATA:
        return [f"errata counts {got} != {EXPECTED_ERRATA}"]
    return []


EXPECTED_RELABELS = 136


def _validate_structure(blocks: list[dict[str, Any]]) -> list[str]:
    """Every audited relabel applied: footnote -> paragraph + definition_body, nothing else."""
    problems = []
    expected = {d["id"]: d["block"] for d in load_structure()["decisions"]}
    applied = {r["id"]: b for b in blocks for r in b.get("reclassification", [])}
    if set(applied) != set(expected) or len(expected) != EXPECTED_RELABELS:
        problems.append(
            f"definition-body relabels: {len(applied)} applied, {len(expected)} decided"
        )
    for rid, b in applied.items():
        if (b["block_id"], b["kind"], b.get("structural_class")) != (
            expected.get(rid),
            "paragraph",
            "definition_body",
        ):
            problems.append(f"{rid}: not relabelled as paragraph/definition_body")
    return problems


def validate(
    blocks: list[dict[str, Any]], pages: list[dict[str, Any]], summary: dict[str, Any]
) -> list[str]:
    """Problems that block the promotion (empty list = promotable)."""
    return _validate_blocks(blocks) + _validate_corpus(blocks, pages, summary)


def corpus_metadata(blocks: list[dict[str, Any]], summary: dict[str, Any]) -> dict[str, Any]:
    version = load_manifest(MANIFEST_PATH).get(DOC_ID).current
    return {
        "schema_version": 1,
        "doc_id": DOC_ID,
        "title": "Mabhas 19 (5th edition, 1404) – main text, candidate-reviewed corpus",
        "source": {
            "path": f"sources/{version.path}",
            "version": version.version,
            "sha256": version.sha256,
            "pages": EXPECTED_PAGES,
        },
        "methods": {
            "text": "repair_v1 (glyph-evidence text-layer repair)",
            "glyph_rules": "G1/G2: BNazanin glyphs 95/112 drawn with ink but mapped to "
            "U+0020 -> «پ»/«چ»",
            "text_rules": "M1 list markers, B1 Latin brackets, C1 caption numbers",
            "ocr": "Tesseract on the one scanned region (PDF page 5), excluded from retrieval",
            "tables": "ruled-table references only (cells pending)",
        },
        "review": {
            "file": DECISIONS_PATH.relative_to(REPO_ROOT).as_posix(),
            "sha256": sha256_file(DECISIONS_PATH),
            "basis": "assistant_reviewed",
            "errata": {
                "file": ERRATA_PATH.relative_to(REPO_ROOT).as_posix(),
                "sha256": sha256_file(ERRATA_PATH),
                "approved_on": "2026-10-17",
                "counts": EXPECTED_ERRATA,
            },
            "structure": {
                "file": STRUCTURE_PATH.relative_to(REPO_ROOT).as_posix(),
                "sha256": sha256_file(STRUCTURE_PATH),
                "approved_on": "2026-10-18",
                "counts": {"definition_body_relabels": EXPECTED_RELABELS},
            },
        },
        "counts": {
            "pages_by_role": summary["pages_by_role"],
            "blocks": len(blocks),
            "blocks_by_kind": dict(Counter(b["kind"] for b in blocks)),
            "verification_status": summary["verification_status"],
            "retrieval_included": summary["retrieval"]["included"],
            "rule_applications": summary["rule_applications"],
            "headings": summary["headings"],
            "sections": summary["sections"],
        },
        "text_layers": {
            "text_raw": "text-layer characters, content-stream order (damaged)",
            "text_repair_v1": "machine repair before any approved rule",
            "text_before_review": "after approved rules, before a per-block decision",
            "text_before_errata": "final text before the 2026-10-17 errata",
            "text": "final reviewed text",
        },
    }


def build_preview(
    corpus_dir: Path,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any], list[str]]:
    blocks = _load(corpus_dir / "blocks.jsonl")
    pages = _load(corpus_dir / "pages.jsonl")
    summary = json.loads((corpus_dir / "summary.json").read_text(encoding="utf-8"))
    files = {
        f"page_maps/{DOC_ID}.jsonl": pages,
        f"structure/{DOC_ID}/sections.jsonl": _load(corpus_dir / "sections.jsonl"),
        f"structure/{DOC_ID}/toc.jsonl": _load(corpus_dir / "toc.jsonl"),
        f"text/{DOC_ID}/blocks.jsonl": [curated_block(b) for b in blocks],
    }
    return files, corpus_metadata(blocks, summary), validate(blocks, pages, summary)


def write_files(
    files: dict[str, list[dict[str, Any]]], meta: dict[str, Any], root: Path
) -> list[Path]:
    written = []
    for rel, rows in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="\n") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        written.append(path)
    meta_path = root / f"text/{DOC_ID}/corpus.yaml"
    meta_path.write_text(
        yaml.safe_dump(meta, allow_unicode=True, sort_keys=False, width=100),
        encoding="utf-8",
        newline="\n",
    )
    return [*written, meta_path]


# --------------------------------------------------------------------------- table captions

TABLES_DIR = Path("tables") / DOC_ID
CAPTION_SYNC = frozenset({"19-2-1", "19-3-1", "19-3-2", "19-5-2", "19-5-7"})


def table_caption_updates(blocks: list[dict[str, Any]], curated: Path) -> dict[str, dict[str, Any]]:
    """table.yaml of the structured main tables whose caption follows the corrected corpus
    (caption text changed, or a caption continuation line is linked). Only ``caption`` changes;
    cells.jsonl and records.jsonl are never touched."""
    by_id = {b["block_id"]: b for b in blocks}
    out = {}
    for spec in TABLES:
        rel = (TABLES_DIR / f"table-{spec.number}" / "table.yaml").as_posix()
        meta = yaml.safe_load((curated / rel).read_text(encoding="utf-8"))
        cap = caption(spec, by_id)
        if cap["text"] != meta["caption"]["text"] or "parts" in cap:
            out[rel] = {**meta, "caption": cap}
    return out


def validate_captions(
    updates: dict[str, dict[str, Any]], blocks: list[dict[str, Any]], curated: Path
) -> list[str]:
    problems = []
    by_id = {b["block_id"]: b for b in blocks}
    changed = set()
    for spec in TABLES:
        rel = (TABLES_DIR / f"table-{spec.number}" / "table.yaml").as_posix()
        old: dict[str, Any] = yaml.safe_load((curated / rel).read_text(encoding="utf-8"))
        new: dict[str, Any] = updates.get(rel, old)
        if {k: v for k, v in new.items() if k != "caption"} != {
            k: v for k, v in old.items() if k != "caption"
        }:
            problems.append(f"{rel}: changes outside the caption")
        if new["caption"]["text"] != old["caption"]["text"]:
            changed.add(spec.number)
        if new["caption"]["text"] != caption(spec, by_id)["text"]:
            problems.append(f"{spec.number}: structured caption differs from the corpus")
    # only the errata-synchronized captions may ever differ from the curated table.yaml
    # (they changed once, on 2026-10-17; after that every promotion must leave them equal)
    if changed - CAPTION_SYNC:
        problems.append(f"caption text changed for {sorted(changed - CAPTION_SYNC)}")
    cap51 = updates.get((TABLES_DIR / "table-19-5-1" / "table.yaml").as_posix(), {}).get("caption")
    if not cap51 or [p["kind"] for p in cap51.get("parts", [])] != [
        "table_caption",
        "caption_continuation",
    ]:
        problems.append("19-5-1 caption must link main-p084-b004 as caption_continuation")
    return problems


def write_table_yaml(updates: dict[str, dict[str, Any]], root: Path) -> list[Path]:
    written = []
    for rel, meta in updates.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.safe_dump(meta, allow_unicode=True, sort_keys=False, width=100),
            encoding="utf-8",
            newline="\n",
        )
        written.append(path)
    return written
