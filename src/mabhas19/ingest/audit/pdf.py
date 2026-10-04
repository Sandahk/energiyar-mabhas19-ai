"""PyMuPDF adapter: turns PDF pages into audit records. Reads sources, never writes them."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import pymupdf

from mabhas19.ingest.audit.headings import Fragment, Glyph
from mabhas19.ingest.audit.models import PageAudit, TableInventoryEntry
from mabhas19.ingest.audit.page_numbers import FooterObservation, classify_marker
from mabhas19.ingest.audit.quality import classify_text_quality, quality_signals
from mabhas19.text.charclass import char_profile

HEADER_BAND_BOTTOM = 0.10  # fraction of page height
FOOTER_BAND_TOP = 0.88
FOOTER_CENTER_TOLERANCE = 0.10  # fraction of page width either side of centre
FOOTER_GLYPH_CENTER_TOLERANCE = 0.06
FOOTER_GLYPH_MAX_HEIGHT = 15.0
FOOTER_GLYPH_GAP = 1.5
WATERMARK_MIN_PAGE_SHARE = 0.5
FULL_PAGE_RASTER_AREA = 0.5
SAMPLE_CHARS = 240
VISIBILITY_DPI = 144
VISIBLE_DARKNESS = 180  # grey level (0-255) below which a pixel counts as ink (any colour;
# the light watermark renders at ~209)
TABLE_CAPTION_WORD = "جدول"


@dataclass(frozen=True)
class DocumentContext:
    """Document-wide facts needed while auditing individual pages."""

    doc_id: str
    watermark_xrefs: frozenset[int]


def watermark_xrefs(doc: pymupdf.Document) -> frozenset[int]:
    """Image xrefs placed on at least half of the pages (e.g. the «inbr» watermark)."""
    if doc.page_count < 4:
        return frozenset()
    seen: Counter[int] = Counter()
    for page in doc:
        seen.update({img[0] for img in page.get_images()})
    return frozenset(x for x, n in seen.items() if n >= WATERMARK_MIN_PAGE_SHARE * doc.page_count)


def audit_page(ctx: DocumentContext, page: pymupdf.Page) -> PageAudit:
    text = page.get_text()
    profile = char_profile(text)
    quality, reasons = classify_text_quality(profile)
    images = page.get_images()
    content_xrefs = [img[0] for img in images if img[0] not in ctx.watermark_xrefs]
    page_area = page.rect.width * page.rect.height
    full_page_raster = any(
        r.width * r.height >= FULL_PAGE_RASTER_AREA * page_area
        for xref in content_xrefs
        for r in page.get_image_rects(xref)
    )
    return PageAudit(
        doc_id=ctx.doc_id,
        pdf_page=page.number + 1,
        text_char_count=profile.total_chars,
        persian_char_count=profile.persian_letters,
        latin_char_count=profile.latin_letters,
        digit_count=profile.digits,
        image_count=len(images),
        content_image_count=len(content_xrefs),
        vector_drawing_count=len(page.get_drawings()),
        page_width=round(page.rect.width, 2),
        page_height=round(page.rect.height, 2),
        extracted_text_sample=" ".join(text.split())[:SAMPLE_CHARS],
        likely_text_layer_quality=quality,
        quality_reasons=reasons,
        quality_signals=quality_signals(profile),
        has_full_page_raster=full_page_raster,
    )


def is_rendered_visibly(page: pymupdf.Page, bbox: pymupdf.Rect) -> bool:
    """True if the area actually renders dark pixels (text may be covered by an image)."""
    pix = page.get_pixmap(dpi=VISIBILITY_DPI, clip=bbox, colorspace=pymupdf.csGRAY)
    return any(b < VISIBLE_DARKNESS for b in pix.samples)


def footer_observation(page: pymupdf.Page) -> FooterObservation:
    width, height = page.rect.width, page.rect.height
    centre = width / 2
    visible: list[str] = []
    hidden: list[str] = []
    for w in page.get_text("words"):
        if w[1] < FOOTER_BAND_TOP * height:
            continue
        if abs((w[0] + w[2]) / 2 - centre) > FOOTER_CENTER_TOLERANCE * width:
            continue
        if classify_marker(w[4]) is not None and not is_rendered_visibly(page, pymupdf.Rect(w[:4])):
            hidden.append(w[4])
        else:
            visible.append(w[4])
    glyph_rects = sorted(
        (
            d["rect"]
            for d in page.get_drawings()
            if d["rect"].y0 >= FOOTER_BAND_TOP * height
            and d["rect"].height <= FOOTER_GLYPH_MAX_HEIGHT
            and abs((d["rect"].x0 + d["rect"].x1) / 2 - centre)
            <= FOOTER_GLYPH_CENTER_TOLERANCE * width
        ),
        key=lambda r: r.x0,
    )
    clusters = 0
    right_edge = float("-inf")
    for r in glyph_rects:
        if r.x0 - right_edge > FOOTER_GLYPH_GAP:
            clusters += 1
        right_edge = max(right_edge, r.x1)
    return FooterObservation(page.number + 1, tuple(visible), clusters, tuple(hidden))


def body_fragments(page: pymupdf.Page) -> list[Fragment]:
    """Text lines between the running header and the footer band, with glyph positions."""
    height = page.rect.height
    fragments: list[Fragment] = []
    for block in page.get_text("rawdict")["blocks"]:
        for line in block.get("lines", []):
            spans: list[dict[str, Any]] = line["spans"]
            chars = [ch for s in spans for ch in s["chars"]]
            if not chars:
                continue
            baseline = float(chars[0]["origin"][1])
            if not HEADER_BAND_BOTTOM * height < baseline < FOOTER_BAND_TOP * height:
                continue
            bold_len = sum(len(s["chars"]) for s in spans if "Bold" in s["font"])
            fragments.append(
                Fragment(
                    text="".join(ch["c"] for ch in chars),
                    glyphs=tuple(Glyph(ch["c"], ch["bbox"][0], ch["bbox"][2]) for ch in chars),
                    baseline=baseline,
                    bold=bold_len * 2 >= len(chars),
                )
            )
    return fragments


def auto_table_entry(doc_id: str, page: pymupdf.Page, vector_text: bool) -> TableInventoryEntry:
    tables = page.find_tables().tables
    captions = [
        " ".join(line.split())[:160]
        for line in page.get_text().splitlines()
        if TABLE_CAPTION_WORD in line
    ]
    notes = ["PyMuPDF find_tables on ruling lines"]
    if captions and not tables:
        notes.append("mentions «جدول» in text but no ruled table detected (likely a reference)")
    if vector_text:
        notes.append("page text is drawn as vector outlines: detection unreliable")
    return TableInventoryEntry(
        doc_id=doc_id,
        pdf_page=page.number + 1,
        probable_table=bool(tables),
        approximate_table_count=len(tables),
        source="auto",
        notes=notes,
        detected_shapes=[[t.row_count, t.col_count] for t in tables],
        caption_candidates=captions,
    )


def document_structure(doc: pymupdf.Document) -> dict[str, Any]:
    labels = doc.get_page_labels()
    return {
        "page_count": doc.page_count,
        "metadata": {k: v for k, v in (doc.metadata or {}).items() if v},
        "pdf_bookmarks": [list(entry) for entry in doc.get_toc(simple=True)],
        "pdf_page_labels": [dict(label) for label in labels],
        "is_encrypted": doc.is_encrypted,
    }


def render_page(page: pymupdf.Page, path: str, dpi: int) -> None:
    page.get_pixmap(dpi=dpi).save(path)
