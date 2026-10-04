"""Runs the audit for one source version and writes its outputs under data/interim/."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pymupdf
import yaml

from mabhas19.core.manifest import SourceDocument, SourceVersion
from mabhas19.ingest.audit.headings import (
    apply_sequence_check,
    group_visual_lines,
    heading_from_line,
)
from mabhas19.ingest.audit.models import (
    AUDIT_SCHEMA_VERSION,
    HeadingCandidate,
    PageAudit,
    PageMapCandidate,
    TableInventoryEntry,
    TextQuality,
)
from mabhas19.ingest.audit.page_numbers import (
    build_page_map_candidates,
    confidence_counts,
    offset_runs,
)
from mabhas19.ingest.audit.pdf import (
    DocumentContext,
    audit_page,
    auto_table_entry,
    body_fragments,
    document_structure,
    footer_observation,
    render_page,
    watermark_xrefs,
)

PAGES_FILE = "pages.jsonl"
PAGE_MAP_FILE = "page-map-candidates.jsonl"
HEADINGS_FILE = "heading-candidates.jsonl"
TABLES_FILE = "table-inventory.jsonl"
DOCUMENT_FILE = "document.json"
SAMPLES_DIR = "audit-samples"

# A page with this many vector paths but no Persian text most likely has its Persian
# text drawn as outlines (as produced by "Microsoft: Print To PDF").
VECTOR_TEXT_MIN_DRAWINGS = 20
PRINTED_TOC_MARKER = "فهرست"
PRINTED_TOC_MIN_LEADER_LINES = 5


@dataclass(frozen=True)
class SamplePage:
    doc_id: str
    pdf_page: int
    reason: str


@dataclass(frozen=True)
class AuditOptions:
    samples: tuple[SamplePage, ...] = ()  # pages to render for visual comparison
    manual_tables: tuple[TableInventoryEntry, ...] = ()  # visual-review observations
    render_samples: bool = True
    detect_tables: bool = True
    dpi: int = 110


def load_samples(path: Path) -> list[SamplePage]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return [
        SamplePage(doc_id, int(s["pdf_page"]), str(s["reason"]))
        for doc_id, items in (raw.get("samples") or {}).items()
        for s in items
    ]


def load_manual_tables(path: Path) -> list[TableInventoryEntry]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    entries = []
    for obs in raw.get("observations") or []:
        for page in obs["pdf_pages"]:
            entries.append(
                TableInventoryEntry(
                    doc_id=str(obs["doc_id"]),
                    pdf_page=int(page),
                    probable_table=bool(obs.get("probable_table", True)),
                    approximate_table_count=int(obs.get("tables_per_page", 1)),
                    source="manual_visual_review",
                    notes=[str(obs["notes"])] if obs.get("notes") else [],
                )
            )
    return entries


def write_jsonl(path: Path, records: Iterable[Any]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for rec in records:
            fh.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")


def _vector_text_pages(pages: list[PageAudit]) -> list[int]:
    return [
        p.pdf_page
        for p in pages
        if p.persian_char_count == 0 and p.vector_drawing_count >= VECTOR_TEXT_MIN_DRAWINGS
    ]


def _printed_toc_pages(pdf: pymupdf.Document) -> list[int]:
    found = []
    for page in pdf:
        text = page.get_text()
        leaders = sum(1 for line in text.splitlines() if "....." in line)
        if PRINTED_TOC_MARKER in text and leaders >= PRINTED_TOC_MIN_LEADER_LINES:
            found.append(page.number + 1)
    # The printed TOC continues on following pages without repeating its title.
    if found:
        page_no = found[-1] + 1
        while page_no <= pdf.page_count:
            text = pdf[page_no - 1].get_text()
            if sum(1 for line in text.splitlines() if "....." in line) < (
                PRINTED_TOC_MIN_LEADER_LINES
            ):
                break
            found.append(page_no)
            page_no += 1
    return found


def toc_consistency(
    headings: list[HeadingCandidate], page_map: list[PageMapCandidate]
) -> dict[str, Any]:
    """Compare printed-TOC page numbers with the printed page found at each body heading."""
    printed = {c.pdf_page: c.printed_page_candidate for c in page_map}
    body: dict[str, int] = {}
    for h in headings:
        if h.kind == "heading":
            body.setdefault(h.normalized_candidate_heading, h.pdf_page)
    matched, mismatched, missing = 0, [], []
    for h in headings:
        if h.kind != "toc_entry" or h.toc_printed_page is None:
            continue
        number = h.normalized_candidate_heading
        if number not in body:
            missing.append(number)
            continue
        found = printed.get(body[number])
        if found == h.toc_printed_page:
            matched += 1
        else:
            mismatched.append(
                {
                    "section": number,
                    "toc_printed_page": h.toc_printed_page,
                    "body_pdf_page": body[number],
                    "printed_page_candidate_at_body": found,
                }
            )
    return {"matched": matched, "mismatched": mismatched, "missing_in_body": missing}


def audit_version(
    doc: SourceDocument,
    version: SourceVersion,
    sources_dir: Path,
    out_root: Path,
    options: AuditOptions,
) -> dict[str, Any]:
    out_dir = out_root / doc.doc_id / version.sha256[:8]
    out_dir.mkdir(parents=True, exist_ok=True)

    with pymupdf.open(sources_dir / version.path) as pdf:
        ctx = DocumentContext(doc.doc_id, watermark_xrefs(pdf))
        pages = [audit_page(ctx, page) for page in pdf]
        page_map = build_page_map_candidates(doc.doc_id, [footer_observation(p) for p in pdf])

        headings: list[HeadingCandidate] = []
        if doc.kind == "main":
            for page in pdf:
                for line in group_visual_lines(body_fragments(page)):
                    h = heading_from_line(doc.doc_id, page.number + 1, line)
                    if h is not None:
                        headings.append(h)
            headings = apply_sequence_check(headings)

        vector_pages = set(_vector_text_pages(pages))
        tables: list[TableInventoryEntry] = []
        if options.detect_tables:
            for page in pdf:
                entry = auto_table_entry(doc.doc_id, page, page.number + 1 in vector_pages)
                if entry.probable_table or entry.caption_candidates:
                    tables.append(entry)
        tables += [m for m in options.manual_tables if m.doc_id == doc.doc_id]
        tables.sort(key=lambda t: (t.pdf_page, t.source))

        rendered = []
        if options.render_samples:
            sample_dir = out_dir / SAMPLES_DIR
            sample_dir.mkdir(exist_ok=True)
            for s in options.samples:
                if s.doc_id == doc.doc_id and 1 <= s.pdf_page <= pdf.page_count:
                    name = f"p{s.pdf_page:03d}.png"
                    render_page(pdf[s.pdf_page - 1], str(sample_dir / name), options.dpi)
                    rendered.append({"pdf_page": s.pdf_page, "file": name, "reason": s.reason})

        structure = document_structure(pdf)
        printed_toc = _printed_toc_pages(pdf)

    write_jsonl(out_dir / PAGES_FILE, pages)
    write_jsonl(out_dir / PAGE_MAP_FILE, page_map)
    if doc.kind == "main":
        write_jsonl(out_dir / HEADINGS_FILE, headings)
    write_jsonl(out_dir / TABLES_FILE, tables)

    quality = Counter(p.likely_text_layer_quality for p in pages)
    summary: dict[str, Any] = {
        "audit_schema_version": AUDIT_SCHEMA_VERSION,
        "doc_id": doc.doc_id,
        "kind": doc.kind,
        "appendix_number": doc.appendix_number,
        "version": version.version,
        "source_sha256": version.sha256,
        "canonical_filename": version.canonical_filename,
        "output_dir": out_dir.relative_to(out_root).as_posix(),
        **structure,
        "page_count_matches_manifest": structure["page_count"] == version.page_count,
        "has_pdf_bookmarks": bool(structure["pdf_bookmarks"]),
        "has_pdf_page_labels": bool(structure["pdf_page_labels"]),
        "printed_toc_pages": printed_toc,
        "pages_with_selectable_text": sum(1 for p in pages if p.text_char_count > 0),
        "quality_counts": {q.value: quality.get(q, 0) for q in TextQuality},
        "vector_text_suspected_pages": sorted(vector_pages),
        "pages_with_content_raster_images": [p.pdf_page for p in pages if p.content_image_count],
        "full_page_raster_pages": [p.pdf_page for p in pages if p.has_full_page_raster],
        "watermark_image_count": len(ctx.watermark_xrefs),
        "page_marker_runs": [vars(r) for r in offset_runs(page_map)],
        "page_map_confidence": confidence_counts(page_map),
        "heading_candidates": dict(Counter(h.kind for h in headings)),
        "toc_vs_body": toc_consistency(headings, page_map) if headings else None,
        "table_pages_auto": sorted(
            {t.pdf_page for t in tables if t.source == "auto" and t.probable_table}
        ),
        "table_reference_only_pages_auto": sorted(
            {t.pdf_page for t in tables if t.source == "auto" and not t.probable_table}
        ),
        "table_pages_manual": sorted(
            {t.pdf_page for t in tables if t.source != "auto" and t.probable_table}
        ),
        "rendered_samples": rendered,
    }
    (out_dir / DOCUMENT_FILE).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary
