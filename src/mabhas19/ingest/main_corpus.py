"""Candidate text corpus of the main Mabhas 19 document (no chunking, no indexing).

For every page: its role (blank, scanned, toc, front_matter, body), printed page (from the
Phase 1 audit page map), and the blocks found on it. Text-layer pages are repaired with
``repair_v1`` line by line; the only scanned page is read with Tesseract into separate
``ocr_text`` blocks, never merged with text-layer text. Ruled tables become ``table`` region
blocks (cells kept as data, not prose); figures become ``figure`` region blocks.

Outputs go to data/processed/main/<doc_id>/ (git-ignored); nothing is written to curated/.
"""

from __future__ import annotations

import base64
import html
import json
import tempfile
from collections import Counter
from collections.abc import Iterable
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pymupdf

from mabhas19.benchmark.extractors import render_clean
from mabhas19.core.manifest import load_manifest
from mabhas19.core.paths import DATA_DIR, MANIFEST_PATH, SOURCES_DIR
from mabhas19.ingest.audit.pdf import watermark_xrefs
from mabhas19.ingest.main_blocks import (
    STATUS,
    Hierarchy,
    Layout,
    Line,
    apply_text_rules,
    finish_block,
    group_blocks,
    page_layout,
    section_level,
    toc_entry,
)
from mabhas19.ingest.main_decisions import (
    apply_decisions,
    load_decisions,
    load_errata,
    load_provenance,
    load_structure,
)
from mabhas19.ingest.repair.glyphs import (
    Cluster,
    detect_kashida_glyphs,
    kashida_keys,
    page_clusters,
)
from mabhas19.ingest.repair.pipeline import FinalFormStats, _lines, learn_final_forms, repair_line
from mabhas19.providers.ocr import OcrEngine
from mabhas19.text.normalize import normalize_for_search

DOC_ID = "m19-ed5-1404-main"
METHOD_TEXT = "repair_v1"
METHOD_OCR = "tesseract_page"
MIN_FIGURE_SIDE = 25.0  # smaller images are symbols (e.g. check marks in table cells)
SCAN_SHARE = 0.25  # an image covering this share of the page may be a scanned region
SCAN_MAX_TEXT_GLYPHS = 80  # ...if the rest of the page has almost no text layer
HEADER_BAND = 0.085  # share of page height holding the running header
FOOTER_BAND = 0.085
CROP_DPI = 110


def audit_dir() -> Path:
    version = load_manifest(MANIFEST_PATH).get(DOC_ID).current
    return DATA_DIR / "interim" / DOC_ID / version.sha256[:8]


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- lines


def _share(clusters: list[Cluster], pred: Any) -> float:
    return sum(1 for c in clusters if pred(c)) / len(clusters) if clusters else 0.0


# Approved glyph rules (assistant review 2026-10-16): BNazanin glyphs drawn with ink but
# mapped to U+0020 by the font's ToUnicode table.
GLYPH_RULES = {95: ("G1", "پ"), 112: ("G2", "چ")}
GLYPH_MIN_WIDTH_RATIO = 0.4  # width/size; these letters are ~0.56 (a space glyph is narrower)
INK_THRESHOLD = 100  # grey level counted as ink when the glyph box is rendered
INK_MIN_PIXELS = 15


def _has_ink(page: pymupdf.Page, c: Cluster) -> bool:
    clip = pymupdf.Rect(c.x0 + 1, c.top + 2, c.x1 - 1, c.bottom - 2)
    if clip.is_empty:
        return False
    pix = page.get_pixmap(dpi=200, clip=clip, colorspace=pymupdf.csGRAY)
    return sum(1 for v in pix.samples if v < INK_THRESHOLD) > INK_MIN_PIXELS


def fix_glyphs(page: pymupdf.Page, clusters: list[Cluster]) -> tuple[list[Cluster], dict[int, str]]:
    """G1/G2 from glyph and ink evidence. Returns the fixed clusters and, per fixed cluster
    (by id), the rule applied; the original clusters are kept for the raw/machine text."""
    out: list[Cluster] = []
    applied: dict[int, str] = {}
    for c in clusters:
        rule = GLYPH_RULES.get(c.gid)
        wide = c.x1 - c.x0 > GLYPH_MIN_WIDTH_RATIO * c.size
        if rule and c.font.endswith("BNazanin") and c.text == " " and wide and _has_ink(page, c):
            fixed = replace(c, text=rule[1])
            applied[id(fixed)] = rule[0]
            out.append(fixed)
        else:
            out.append(c)
    return out, applied


def make_line(pdf_page: int, clusters: list[Cluster], kashida: frozenset[tuple[str, int]],
              stats: FinalFormStats,
              glyphs: tuple[dict[int, str], dict[int, Cluster]] | None = None) -> Line:  # fmt: skip
    """One repaired line. ``glyphs`` = (rule per fixed cluster id, original per fixed id)."""
    glyph_fixes, originals = glyphs if glyphs else (None, None)
    ops: Counter[str] = Counter()
    text = repair_line(clusters, kashida, stats, ops=ops)
    fixed = [c for c in clusters if glyph_fixes and id(c) in glyph_fixes]
    machine = None
    if fixed and originals is not None:  # repair_v1 of the unmodified glyphs, kept separately
        machine = repair_line([originals.get(id(c), c) for c in clusters], kashida, stats)
    rules = Counter(glyph_fixes[id(c)] for c in fixed) if glyph_fixes else Counter()
    clusters = [originals.get(id(c), c) for c in clusters] if originals else clusters
    sizes = Counter(round(c.size, 1) for c in clusters)
    letters = [c for c in clusters if c.text.strip() and c.text[0].isalpha()]
    return Line(
        pdf_page,
        (min(c.x0 for c in clusters), min(c.top for c in clusters),
         max(c.x1 for c in clusters), max(c.bottom for c in clusters)),
        "".join(c.text for c in clusters), text, sizes.most_common(1)[0][0],
        _share(clusters, lambda c: "Bold" in c.font),
        _share(clusters, lambda c: "Math" in c.font or "Symbol" in c.font),
        _share(letters, lambda c: c.text[0].isascii()), dict(ops), machine, dict(rules),
    )  # fmt: skip


def _inside(c: Cluster, rects: list[pymupdf.Rect]) -> bool:
    p = pymupdf.Point((c.x0 + c.x1) / 2, (c.top + c.bottom) / 2)
    return any(r.contains(p) for r in rects)


# --------------------------------------------------------------------------- pages


def image_rects(page: pymupdf.Page, wm: frozenset[int]) -> list[pymupdf.Rect]:
    return [pymupdf.Rect(r) for i in page.get_images() if i[0] not in wm
            for r in page.get_image_rects(i[0])]  # fmt: skip


def scan_regions(page: pymupdf.Page, wm: frozenset[int]) -> list[pymupdf.Rect]:
    """Large raster images on a page with (almost) no text layer: scanned content to OCR.

    A large image on a page with a normal text layer is a figure (e.g. the climate map),
    never OCR'd into prose.
    """
    area = page.rect.width * page.rect.height
    large = [r for r in image_rects(page, wm) if r.width * r.height >= SCAN_SHARE * area]
    if not large:
        return []
    words = sum(len(c.text.strip()) > 0 for c in page_clusters(page) if not any(
        r.contains(pymupdf.Point(c.x0, c.top)) for r in large))  # fmt: skip
    return large if words <= SCAN_MAX_TEXT_GLYPHS else []


def _content(clusters: list[Cluster]) -> bool:
    return bool("".join(c.text for c in clusters).replace("‌", "").strip())


def page_role(page: pymupdf.Page, clusters: list[Cluster], toc_pages: set[int],
              first_body_page: int, wm: frozenset[int]) -> str:  # fmt: skip
    """blank: only furniture (page marker) and no images; scanned: a large raster region."""
    pno = page.number + 1
    if scan_regions(page, wm):
        return "scanned"
    h = page.rect.height
    content = [c for c in clusters if h * HEADER_BAND < c.top and c.bottom < h * (1 - FOOTER_BAND)]
    if not _content(content) and not image_rects(page, wm):
        return "blank"
    if pno in toc_pages:
        return "toc"
    return "front_matter" if pno < first_body_page else "body"


def table_regions(page: pymupdf.Page) -> list[pymupdf.Rect]:
    return [pymupdf.Rect(t.bbox) for t in page.find_tables().tables if t.row_count >= 2]


def table_block(page: pymupdf.Page, rect: pymupdf.Rect, kashida: frozenset[tuple[str, int]],
                stats: FinalFormStats, index: int) -> dict[str, Any]:  # fmt: skip
    """A ruled table kept as data: cell texts (raw and repaired) by row and column."""
    table = next(t for t in page.find_tables().tables if pymupdf.Rect(t.bbox) == rect)
    clusters = page_clusters(page)
    cells: list[list[dict[str, Any] | None]] = []
    for row in table.rows:
        out_row: list[dict[str, Any] | None] = []
        for bbox in row.cells:
            if bbox is None:
                out_row.append(None)
                continue
            r = pymupdf.Rect(bbox)
            inside = [c for c in clusters if _inside(c, [r])]
            lines = [make_line(page.number + 1, ln, kashida, stats) for ln in _lines(inside)]
            out_row.append({"bbox": [round(v, 1) for v in r],
                            "text_raw": "\n".join(ln.raw for ln in lines),
                            "text": "\n".join(ln.text for ln in lines)})  # fmt: skip
        cells.append(out_row)
    return {"kind": "table", "table_index": index, "bbox": [round(v, 1) for v in rect],
            "rows": table.row_count, "cols": table.col_count, "cells": cells,
            "flags": ["table_pending_structured_extraction"]}  # fmt: skip


def furniture(line: Line, height: float) -> str | None:
    if line.bbox[3] <= height * HEADER_BAND and line.size <= 10.5:
        return "running_header"
    if line.bbox[1] >= height * (1 - FOOTER_BAND) and len(line.text.split()) <= 2 and len(
            line.text) <= 6:  # fmt: skip
        return "page_number"
    return None


# --------------------------------------------------------------------------- build


class Context:
    def __init__(self, pdf: pymupdf.Document, audit: Path, engine: OcrEngine | None) -> None:
        self.pdf = pdf
        self.kashida = kashida_keys(detect_kashida_glyphs(pdf, list(range(18, pdf.page_count))))
        self.stats = learn_final_forms(pdf, self.kashida)
        self.wm = watermark_xrefs(pdf)
        doc = json.loads((audit / "document.json").read_text(encoding="utf-8"))
        self.page_map = {p["pdf_page"]: p for p in _jsonl(audit / "page-map-candidates.jsonl")}
        self.headings = _jsonl(audit / "heading-candidates.jsonl")
        self.table_pages = set(doc["table_pages_manual"]) | set(doc["table_pages_auto"])
        self.toc_pages = {h["pdf_page"] for h in self.headings if h["kind"] == "toc_entry"} | set(
            doc["printed_toc_pages"])  # fmt: skip
        self.first_body_page = min(h["pdf_page"] for h in self.headings if h["kind"] == "heading")
        self.engine = engine
        self.hierarchy = Hierarchy()


def _base(ctx: Context, pno: int, k: int, method: str) -> dict[str, Any]:
    pm = ctx.page_map.get(pno, {})
    return {"block_id": f"main-p{pno:03d}-b{k:03d}", "doc_id": DOC_ID, "pdf_page": pno,
            "printed_page": pm.get("printed_page_candidate"),
            "printed_page_confidence": pm.get("confidence"), "extraction_method": method,
            "review_status": STATUS}  # fmt: skip


def _confidence(block: dict[str, Any]) -> str:
    if block["kind"] == "figure":
        return "not_applicable"
    if block["extraction_method"] == METHOD_OCR or block["kind"] in ("formula", "table"):
        return "low"
    if any(f in block["flags"] for f in ("repair_empty", "unmapped_glyphs")):
        return "low"
    return "medium" if block["flags"] else "high"


def page_blocks(ctx: Context, pno: int, role: str, tmp: Path) -> list[dict[str, Any]]:
    page = ctx.pdf[pno - 1]
    if role == "blank":
        return []
    clusters = page_clusters(page)
    tables = table_regions(page) if pno in ctx.table_pages else []
    scans = scan_regions(page, ctx.wm)
    images = [r for r in image_rects(page, ctx.wm) if r not in scans]
    in_table = [r for r in images if any(t.contains(r) for t in tables)]
    figures = [r for r in images if r not in in_table and min(r.width, r.height) >= MIN_FIGURE_SIDE]
    fixed, glyph_fixes = fix_glyphs(page, [c for c in clusters if not _inside(c, tables)])
    originals = {id(f): o for f, o in zip(fixed, [c for c in clusters if not _inside(c, tables)],
                                          strict=True) if f is not o}  # fmt: skip
    lines = [make_line(pno, ln, ctx.kashida, ctx.stats, (glyph_fixes, originals))
             for ln in _lines(fixed) if _content(ln)]  # fmt: skip
    body, furn = [], []
    for ln in lines:
        kind = furniture(ln, page.rect.height)
        (furn.append((kind, ln)) if kind else body.append(ln))
    layout: Layout = page_layout(body)
    blocks: list[dict[str, Any]] = []
    for kind, ln in furn:
        blocks.append({"kind": kind, "lines": [ln]})
    blocks += group_blocks(body, layout)
    out: list[dict[str, Any]] = []
    for blk in blocks:
        rec = {**_base(ctx, pno, len(out) + 1, METHOD_TEXT), **finish_block(blk)}
        out.append(apply_text_rules(rec, role))
    for i, r in enumerate(tables, 1):
        rec = {
            **_base(ctx, pno, len(out) + 1, METHOD_TEXT + "+ruled_grid"),
            **table_block(page, r, ctx.kashida, ctx.stats, i),
        }
        rec.update(
            text=None,
            text_raw=None,
            text_normalized=None,
            section_number=None,
            embedded_symbol_images=sum(r.contains(x) for x in in_table),
        )
        out.append(rec)
    for r in scans:
        out.append(_ocr_block(ctx, page, r, len(out) + 1, tmp))
    for i, r in enumerate(figures, 1):
        out.append({**_base(ctx, pno, len(out) + 1, "image_region"), "kind": "figure",
                    "figure_index": i, "bbox": [round(v, 1) for v in r], "text": None,
                    "text_raw": None, "text_normalized": None, "section_number": None,
                    "flags": ["figure_not_transcribed"]})  # fmt: skip
    return out


def _ocr_block(ctx: Context, page: pymupdf.Page, region: pymupdf.Rect, k: int,
               tmp: Path) -> dict[str, Any]:  # fmt: skip
    """OCR of one scanned image region: a separate block, never merged with text-layer text."""
    rec = {**_base(ctx, page.number + 1, k, METHOD_OCR), "kind": "ocr_text",
           "bbox": [round(v, 1) for v in region], "section_number": None,
           "flags": ["ocr_scanned_region"]}  # fmt: skip
    if ctx.engine is None:
        rec.update(text=None, text_raw=None, text_normalized=None)
        rec["flags"].append("ocr_not_run")
        return rec
    img = render_clean(page, tmp / "ocr.png", clip=region, dpi=300)
    res = ctx.engine.recognize(img, ("fas",), psm=6)
    conf = [w.confidence for w in res.words]
    rec.update(text_raw=res.text, text=res.text, text_normalized=normalize_for_search(res.text),
               ocr_mean_confidence=round(sum(conf) / len(conf), 1) if conf else None)  # fmt: skip
    return rec


def assign_hierarchy(ctx: Context, blocks: list[dict[str, Any]]) -> None:
    """Section path for every block, from printed heading numbers in reading order."""
    for b in blocks:
        role = b["page_role"]
        if role in ("front_matter", "toc", "scanned"):
            b["section_path"] = [{"number": None, "title": role}]
            continue
        if b["kind"] == "heading" and b["section_number"]:
            title = b["text"].split(maxsplit=1)[1] if " " in b["text"] else ""
            problems = ctx.hierarchy.enter(b["section_number"], title, b["block_id"])
            b["flags"] += [f"hierarchy: {p}" for p in problems]
            b["heading_level"] = section_level(b["section_number"])
        elif b["kind"] == "clause" and b["section_number"]:
            ctx.hierarchy.note_clause(b["section_number"])
        b["section_path"] = ctx.hierarchy.path()


def build(out: Path, engine: OcrEngine | None) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    version = load_manifest(MANIFEST_PATH).get(DOC_ID).current
    pdf_path = SOURCES_DIR / version.path
    pages: list[dict[str, Any]] = []
    blocks: list[dict[str, Any]] = []
    with pymupdf.open(pdf_path) as pdf, tempfile.TemporaryDirectory() as td:
        ctx = Context(pdf, audit_dir(), engine)
        for page in pdf:
            pno = page.number + 1
            role = page_role(page, page_clusters(page), ctx.toc_pages, ctx.first_body_page,
                             ctx.wm)  # fmt: skip
            recs = page_blocks(ctx, pno, role, Path(td))
            for r in recs:
                r["page_role"] = role
                r["confidence"] = _confidence(r)
            blocks += recs
            methods = sorted({r["extraction_method"] for r in recs})
            pm = ctx.page_map.get(pno, {})
            pages.append({"doc_id": DOC_ID, "pdf_page": pno, "role": role,
                          "printed_page": pm.get("printed_page_candidate"),
                          "printed_page_confidence": pm.get("confidence"),
                          "methods": methods, "blocks": len(recs),
                          "table_page": pno in ctx.table_pages,
                          "low_confidence_blocks": sum(r["confidence"] == "low"
                                                       for r in recs)})  # fmt: skip
        assign_hierarchy(ctx, blocks)
        machine_layer = blocks
        blocks = apply_decisions(
            blocks, load_decisions(), load_errata(), load_structure(), load_provenance()
        )
        refresh_section_titles(blocks)
        toc = toc_entries(blocks)
        checks = heading_checks(ctx, blocks, toc)
    sections = section_index(blocks)
    _write_jsonl(out / "pages.jsonl", pages)
    _write_jsonl(out / "blocks_before_review.jsonl", machine_layer)
    _write_jsonl(out / "blocks.jsonl", blocks)
    _write_jsonl(out / "sections.jsonl", sections)
    _write_jsonl(out / "toc.jsonl", toc)
    summary = _summary(
        version.sha256, pages, blocks, {"sections": sections, "toc": toc, "checks": checks}
    )
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n",
                                      encoding="utf-8", newline="\n")  # fmt: skip
    (out / "review.html").write_text(review_html(pdf_path, blocks), encoding="utf-8",
                                     newline="\n")  # fmt: skip
    return summary


def toc_entries(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"block_id": b["block_id"], "pdf_page": b["pdf_page"], **toc_entry(b["text"])}
            for b in blocks if b["kind"] == "toc_entry" and b["page_role"] == "toc"]  # fmt: skip


def heading_checks(ctx: Context, blocks: list[dict[str, Any]], toc: list[dict[str, Any]]
                   ) -> dict[str, Any]:  # fmt: skip
    """Our numbered headings/clauses against the Phase 1 audit candidates and the TOC."""
    ours = {(b["pdf_page"], b["section_number"]): b["kind"] for b in blocks
            if b["kind"] in ("heading", "clause") and b["section_number"]}  # fmt: skip
    out: dict[str, Any] = {}
    for kind in ("heading", "clause"):
        audit = {(h["pdf_page"], h["normalized_candidate_heading"]) for h in ctx.headings
                 if h["kind"] == kind}  # fmt: skip
        found = {k for k in audit if k in ours}
        out[f"audit_{kind}s"] = len(audit)
        out[f"audit_{kind}s_found"] = len(found)
        out[f"audit_{kind}s_missing"] = sorted(audit - found)
        out[f"audit_{kind}s_kind_differs"] = sorted(k for k in found if ours[k] != kind)
    heading_numbers = {b["section_number"] for b in blocks if b["kind"] == "heading"}
    toc_numbers = [t["section_number"] for t in toc if t["section_number"]]
    out["toc_entries"] = len(toc)
    out["toc_numbered"] = len(toc_numbers)
    out["toc_matched_to_body_heading"] = sum(n in heading_numbers for n in toc_numbers)
    out["toc_missing_in_body"] = sorted(n for n in toc_numbers if n not in heading_numbers)
    return out


def refresh_section_titles(blocks: list[dict[str, Any]]) -> None:
    """Section titles follow the final (reviewed) heading text; numbers never change."""
    titles = {b["section_number"]: b["text"].split(maxsplit=1)[1] if " " in b["text"] else ""
              for b in blocks if b["kind"] == "heading" and b["section_number"]}  # fmt: skip
    for b in blocks:
        for part in b.get("section_path") or []:
            if part["number"] in titles:
                part["title"] = titles[part["number"]]


def section_index(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One entry per printed heading: number, title, level, page span, block counts."""
    out: list[dict[str, Any]] = []
    for b in blocks:
        if b["kind"] == "heading" and b["section_number"]:
            title = b["section_path"][-1]["title"]
            out.append({"section_number": b["section_number"], "title": title,
                        "level": b.get("heading_level"), "heading_block": b["block_id"],
                        "first_pdf_page": b["pdf_page"], "last_pdf_page": b["pdf_page"],
                        "blocks": 0})  # fmt: skip
        elif (
            out and b["page_role"] == "body" and b["kind"] not in ("running_header", "page_number")
        ):
            path = [p["number"] for p in b.get("section_path", [])]
            for s in out:
                if s["section_number"] in path:
                    s["last_pdf_page"] = max(s["last_pdf_page"], b["pdf_page"])
                    s["blocks"] += 1
    return out


def _summary(sha: str, pages: list[dict[str, Any]], blocks: list[dict[str, Any]],
             parts: dict[str, Any]) -> dict[str, Any]:  # fmt: skip
    sections, toc, checks = parts["sections"], parts["toc"], parts["checks"]
    corpus = [b for b in blocks if b["kind"] not in ("running_header", "page_number")]
    in_body = [b for b in corpus if b["page_role"] == "body"]
    return {
        "doc_id": DOC_ID, "source_sha256": sha,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "pages": len(pages), "pages_by_role": dict(Counter(p["role"] for p in pages)),
        "pages_by_method": dict(Counter("+".join(p["methods"]) or "none" for p in pages)),
        "ocr_pages": [p["pdf_page"] for p in pages if "tesseract_page" in p["methods"]],
        "table_pages": [p["pdf_page"] for p in pages if p["table_page"]],
        "blocks_total": len(blocks), "blocks_by_kind": dict(Counter(b["kind"] for b in blocks)),
        "headings": sum(b["kind"] == "heading" for b in blocks),
        "headings_by_level": dict(Counter(b.get("heading_level") for b in blocks
                                          if b["kind"] == "heading")),
        "clauses": sum(b["kind"] == "clause" for b in blocks),
        "sections": len(sections),
        "body_blocks_with_section": sum(bool(b["section_path"]) for b in in_body),
        "body_blocks": len(in_body),
        "confidence": dict(Counter(b["confidence"] for b in corpus)),
        "flags": dict(Counter(f.split(":")[0] for b in corpus for f in b["flags"])),
        "open_review_flags": dict(Counter(f for b in corpus for f in b.get("open_flags", []))),
        "verification_status": dict(Counter(b["verification"]["status"] for b in blocks)),
        "retrieval": {"included": sum(b["retrieval"]["include"] for b in blocks),
                      "excluded_by_reason": dict(Counter(b["retrieval"]["reason"] for b in blocks
                                                         if not b["retrieval"]["include"]))},
        "rule_applications": dict(Counter(
            r["rule"] for b in blocks for r in b.get("applied_rules", [])
            for _ in range(r["count"]))),
        "decisions_applied": sorted({d for b in blocks for d in b["verification"]["decisions"]}),
        "low_confidence_pages": sorted({b["pdf_page"] for b in corpus if b["confidence"] == "low"}),
        "heading_checks": checks, "toc_entries": len(toc),
    }  # fmt: skip


# --------------------------------------------------------------------------- review page

_CSS = """
:root{--bg:#f7f6f2;--fg:#1d1d1b;--muted:#66655f;--line:#d9d5ca;--card:#fff}
@media (prefers-color-scheme:dark){:root{--bg:#191917;--fg:#ecebe6;--muted:#a3a29b;
--line:#3a3934;--card:#23231f}}
body{background:var(--bg);color:var(--fg);margin:0;font:14px/1.6 Vazirmatn,Tahoma,sans-serif}
main{max-width:1100px;margin:auto;padding:16px}section{background:var(--card);border:1px
solid var(--line);border-radius:6px;margin:12px 0;padding:10px}img{max-width:100%;
background:#fff;border:1px solid var(--line)}.m{color:var(--muted);font-size:12px;direction:ltr;
text-align:left}pre{white-space:pre-wrap;font:inherit;margin:4px 0}
"""


def review_html(pdf_path: Path, blocks: list[dict[str, Any]]) -> str:
    """Low- and medium-confidence blocks with their source crops (first 150)."""
    todo = [b for b in blocks if b.get("confidence") in ("low", "medium")][:150]
    parts = [f"<!doctype html><html lang='fa' dir='rtl'><head><meta charset='utf-8'><meta "
             f"name='viewport' content='width=device-width,initial-scale=1'><title>Main text "
             f"review</title><style>{_CSS}</style></head><body><main><h1>مبحث ۱۹ — بلوک‌های "
             f"نیازمند بازبینی ({len(todo)})</h1>"]  # fmt: skip
    with pymupdf.open(pdf_path) as pdf:
        for b in todo:
            r = pymupdf.Rect(b["bbox"]) & pdf[b["pdf_page"] - 1].rect
            png = pdf[b["pdf_page"] - 1].get_pixmap(dpi=CROP_DPI, clip=r).tobytes("png")
            parts.append(
                f"<section><div class='m'>{b['block_id']} · {b['kind']} · {b['confidence']} · "
                f"{html.escape(', '.join(b['flags']))}</div><img alt='{b['block_id']}' "
                f"src='data:image/png;base64,{base64.b64encode(png).decode()}'>"
                f"<pre>{html.escape(b.get('text') or '')}</pre></section>")  # fmt: skip
    parts.append("</main></body></html>\n")
    return "".join(parts)
