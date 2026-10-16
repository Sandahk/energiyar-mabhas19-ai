"""Audit of the 11 ruled tables of the main Mabhas 19 text (candidate data, never curated).

For each table the audit reads the *logical* grid (``find_tables(strategy="lines_strict")``:
stroked rules only, so the shaded row backgrounds do not split cells) and records, per cell:

* row/column position, row/column span and bbox;
* ``text_raw`` (text-layer characters, content-stream order), ``text_repair_v1`` (machine
  repair of the unmodified glyphs) and ``text`` (with the approved G1/G2 glyph rules);
* the value class of the repaired text (number, percent, dash, climate codes, expression,
  identifier, category, label) and, for numbers, a Decimal-safe parse with the exact token;
* symbols drawn as images inside the cell, identified by the sha256 of their soft mask
  (the base images are identical 2x2 bitmaps, so only the mask carries the shape);
* review flags (split words, mirrored brackets, superscript markers, multi-line labels).

Nothing is interpreted: symbols keep their shape identity, numbers keep their raw token,
and no value is completed or corrected. Outputs go to data/processed/main_tables/audit/.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

import pymupdf

from mabhas19.ingest.main_blocks import quality_flags
from mabhas19.ingest.main_corpus import Context, _inside, fix_glyphs, make_line
from mabhas19.ingest.repair.glyphs import Cluster, page_clusters
from mabhas19.ingest.repair.pipeline import _lines
from mabhas19.text.numbers import parse_numeric


@dataclass(frozen=True)
class TableSpec:
    number: str  # printed table number, ASCII digits (e.g. "19-2-1")
    pdf_page: int
    caption_block: str  # block ids in curated/text/m19-ed5-1404-main/blocks.jsonl
    table_block: str


TABLES = (
    TableSpec("19-2-1", 64, "main-p064-b002", "main-p064-b004"),
    TableSpec("19-3-1", 67, "main-p067-b003", "main-p067-b006"),
    TableSpec("19-3-2", 68, "main-p068-b003", "main-p068-b004"),
    TableSpec("19-3-3", 74, "main-p074-b003", "main-p074-b008"),
    TableSpec("19-5-1", 84, "main-p084-b003", "main-p084-b008"),
    TableSpec("19-5-2", 90, "main-p090-b005", "main-p090-b010"),
    TableSpec("19-5-3", 99, "main-p099-b004", "main-p099-b005"),
    TableSpec("19-5-4", 100, "main-p100-b004", "main-p100-b011"),
    TableSpec("19-5-5", 107, "main-p107-b005", "main-p107-b012"),
    TableSpec("19-5-6", 115, "main-p115-b007", "main-p115-b010"),
    TableSpec("19-5-7", 116, "main-p116-b007", "main-p116-b010"),
)

# Soft masks of the symbol images in table 19-2-1, identified by assistant visual inspection
# of the mask bitmaps (ASCII renderings are written to the audit). Pending review: the
# *shape* is identified here, its regulatory meaning is not.
SYMBOL_MASKS = {
    "d537a6ab54": "cross_mark",  # ✗ (Image395, Image410)
    "d338749e34": "check_mark",  # ✓ (Image408, Image412)
}
EDGE_TOLERANCE = 1.0  # points: grid edges closer than this are one edge
SUPERSCRIPT_RATIO = 0.8  # glyphs smaller than this share of the cell's main size

_PERCENT = "%٪"
_DIGITS = "0-9۰-۹٠-٩"
_NUMBER = re.compile(rf"^[{_DIGITS}]+(?:[/٫.][{_DIGITS}]+)?$")
_CLIMATE = re.compile(r"^\d[ABC](?:\s*,\s*\d[ABC])*$")
_IDENTIFIER = re.compile(rf"^[{_DIGITS}]+(?:\s*[-‑]\s*[{_DIGITS}]+)+(?:\s+A\d)?$")
_CATEGORY = re.compile(r"^(?:IE\d|[A-D]\+*)$")
_EXPRESSION = re.compile(r"[<>≤≥]")
_MIRRORED = re.compile(r"^[^(]*\)")  # a closing bracket before any opening one


# --------------------------------------------------------------------------- grid


def _edges(values: list[float]) -> list[float]:
    out: list[float] = []
    for v in sorted(values):
        if not out or v - out[-1] > EDGE_TOLERANCE:
            out.append(v)
    return out


def _span(lo: float, hi: float, edges: list[float]) -> tuple[int, int]:
    """Index of the first grid line at ``lo`` and the number of grid intervals to ``hi``."""
    first = min(range(len(edges)), key=lambda i: abs(edges[i] - lo))
    last = min(range(len(edges)), key=lambda i: abs(edges[i] - hi))
    return first, max(1, last - first)


def grid_cells(bboxes: list[tuple[float, float, float, float]]) -> list[dict[str, Any]]:
    """Logical cells (row, col, rowspan, colspan) of a ruled grid from its cell bboxes.

    Columns are numbered right to left (column 0 is the rightmost, as the table is read).
    """
    xs = _edges([v for b in bboxes for v in (b[0], b[2])])
    ys = _edges([v for b in bboxes for v in (b[1], b[3])])
    rx = [-x for x in reversed(xs)]
    out = []
    for b in bboxes:
        row, rowspan = _span(b[1], b[3], ys)
        col, colspan = _span(-b[2], -b[0], rx)
        out.append({"row": row, "col": col, "rowspan": rowspan, "colspan": colspan,
                    "bbox": [round(v, 1) for v in b]})  # fmt: skip
    return sorted(out, key=lambda c: (c["row"], c["col"]))


def logical_table(page: pymupdf.Page, near: pymupdf.Rect) -> Any:
    tables = [t for t in page.find_tables(strategy="lines_strict").tables if t.row_count >= 2]
    return max(tables, key=lambda t: (pymupdf.Rect(t.bbox) & near).get_area())


# --------------------------------------------------------------------------- images


def _skip_string(data: bytes, i: int) -> int:
    """Index after the literal string that opens at ``i`` (nested brackets, escapes)."""
    depth, i = 1, i + 1
    while i < len(data) and depth:
        c = data[i : i + 1]
        if c == b"\\":
            i += 1
        elif c == b"(":
            depth += 1
        elif c == b")":
            depth -= 1
        i += 1
    return i


def _tokens(data: bytes) -> Iterator[bytes]:
    """Content-stream tokens; strings, dictionaries and arrays are skipped as a whole."""
    i, n = 0, len(data)
    while i < n:
        ch = data[i : i + 1]
        if ch.isspace():
            i += 1
        elif ch == b"%":
            while i < n and data[i : i + 1] not in (b"\n", b"\r"):
                i += 1
        elif ch == b"(":
            i = _skip_string(data, i)
        elif data[i : i + 2] in (b"<<", b">>"):
            i += 2
        elif ch in b"<>[]{}":
            if ch == b"<":
                i = data.index(b">", i)
            i += 1
        else:
            j = i + 1
            while j < n and not data[j : j + 1].isspace() and data[j : j + 1] not in b"()<>[]{}/%":
                j += 1
            yield data[i:j]
            i = j


def image_placements(page: pymupdf.Page) -> list[tuple[str, pymupdf.Rect]]:
    """(resource name, page rect) of every image drawn directly by the page content."""
    doc = page.parent
    data = b"".join(doc.xref_stream(x) or b"" for x in page.get_contents())
    if re.search(rb"(?<![A-Za-z])BI(?![A-Za-z])", data):
        raise ValueError(f"page {page.number + 1}: inline images are not supported")
    stack: list[pymupdf.Matrix] = []
    ctm = pymupdf.Matrix(1, 0, 0, 1, 0, 0)
    operands: list[bytes] = []
    out = []
    for tok in _tokens(data):
        if tok == b"q":
            stack.append(pymupdf.Matrix(ctm))
        elif tok == b"Q":
            ctm = stack.pop()
        elif tok == b"cm":
            ctm = pymupdf.Matrix(*(float(v) for v in operands[-6:])) * ctm
        elif tok == b"Do":
            r = pymupdf.Rect(0, 0, 1, 1) * ctm
            h = page.mediabox.height
            out.append((operands[-1][1:].decode(), pymupdf.Rect(r.x0, h - r.y1, r.x1, h - r.y0)))
        if tok[:1].isalpha() or tok in (b"'", b'"'):
            operands = []
        else:
            operands.append(tok)
    return out


def mask_digest(doc: pymupdf.Document, smask: int) -> str:
    return hashlib.sha256(pymupdf.Pixmap(doc, smask).samples).hexdigest()[:10]


def ink_box(doc: pymupdf.Document, smask: int) -> tuple[float, float, float, float]:
    """Opaque part of a soft mask as fractions of the image (images carry transparent margins)."""
    pix = pymupdf.Pixmap(doc, smask)
    s, w, h = pix.samples, pix.width, pix.height
    pts = [(x, y) for y in range(h) for x in range(w) if s[(y * w + x) * pix.n] > 128]
    return (min(x for x, _ in pts) / w, min(y for _, y in pts) / h,
            (max(x for x, _ in pts) + 1) / w, (max(y for _, y in pts) + 1) / h)  # fmt: skip


def mask_ascii(doc: pymupdf.Document, smask: int, step: int = 3) -> str:
    pix = pymupdf.Pixmap(doc, smask)
    s, w = pix.samples, pix.width
    rows = range(0, pix.height, 2 * step)
    return "\n".join("".join("#" if s[(y * w + x) * pix.n] > 128 else "."
                             for x in range(0, w, step)) for y in rows)  # fmt: skip


def page_symbols(page: pymupdf.Page) -> list[dict[str, Any]]:
    """Images on the page with their mask identity (only images with a soft mask)."""
    doc = page.parent
    by_name = {i[7]: i for i in page.get_images(full=True) if i[9] == 0}
    out = []
    boxes: dict[int, tuple[float, float, float, float]] = {}
    for name, rect in image_placements(page):
        info = by_name.get(name)
        if info is None or not info[1]:
            continue
        digest = mask_digest(doc, info[1])
        fx0, fy0, fx1, fy1 = boxes.setdefault(info[1], ink_box(doc, info[1]))
        ink = pymupdf.Rect(rect.x0 + fx0 * rect.width, rect.y0 + fy0 * rect.height,
                           rect.x0 + fx1 * rect.width, rect.y0 + fy1 * rect.height)  # fmt: skip
        out.append({"name": name, "xref": info[0], "mask_digest": digest,
                    "symbol": SYMBOL_MASKS.get(digest, "unidentified"), "rect": rect,
                    "ink": ink})  # fmt: skip
    return out


# --------------------------------------------------------------------------- values


def _is_percent(t: str) -> bool:
    bare = t.strip(_PERCENT).strip()
    return bare != t and bool(_NUMBER.match(bare)) and len(t) - len(t.strip(_PERCENT)) == 1


VALUE_CLASSES: tuple[tuple[str, Any], ...] = (  # first match wins
    ("empty", lambda t: not t),
    ("dash", lambda t: t in ("-", "–", "‑")),
    ("number", _NUMBER.match),
    ("percent", _is_percent),
    ("climate_classes", _CLIMATE.match),
    ("identifier", _IDENTIFIER.match),
    ("category", _CATEGORY.match),
    ("expression", _EXPRESSION.search),
)


def classify_value(text: str) -> str:
    t = text.strip()
    return next((name for name, test in VALUE_CLASSES if test(t)), "label")


def parse_value(text: str) -> dict[str, Any] | None:
    """Decimal-safe parse of a number or percent cell (exact token kept, nothing rounded)."""
    kind = classify_value(text)
    if kind not in ("number", "percent"):
        return None
    token = text.strip().strip(_PERCENT).strip()
    v = parse_numeric(token, slash_is_decimal=True)
    if v.normalized_value is None:
        return {"token": token, "value": None, "error": "; ".join(v.issues)}
    Decimal(v.normalized_value)  # must be a valid Decimal literal
    return {"token": token, "value": v.normalized_value, "unit": "%" if kind == "percent" else None,
            "decimal_separator": "/" if "/" in token else None,
            "percent_sign": next((c for c in text if c in _PERCENT), None),
            "confidence": v.confidence}  # fmt: skip


def cell_flags(text: str, raw: str, clusters: list[Cluster], lines: int) -> list[str]:
    flags = [f for f in quality_flags(text, raw) if f != "list_marker_paren_mirrored"]
    if _MIRRORED.search(text):
        flags.append("mirrored_brackets")
    if lines > 1:
        flags.append("multi_line")
    if superscripts(clusters):
        flags.append("superscript_marker")
    return flags


def superscripts(clusters: list[Cluster]) -> str:
    """Small glyphs raised above the baseline of the line they sit on (footnote markers)."""
    out = []
    for c in clusters:
        line = [o for o in clusters if o.text.strip() and abs(o.top - c.top) < 2 * c.size]
        main = Counter(round(o.size, 1) for o in line).most_common(1)
        if not c.text.strip() or not main or c.size >= SUPERSCRIPT_RATIO * main[0][0]:
            continue
        base = Counter(round(o.baseline, 1) for o in line if round(o.size, 1) == main[0][0])
        if base and c.baseline < base.most_common(1)[0][0] - 0.25 * main[0][0]:
            out.append(c)
    return "".join(c.text for c in sorted(out, key=lambda c: -c.x0))


def visual_ltr(clusters: list[Cluster]) -> str:
    """Glyphs left to right as drawn (bidi-independent evidence for identifiers/expressions)."""
    return "".join(c.text for c in sorted(clusters, key=lambda c: c.x0)).strip()


# --------------------------------------------------------------------------- audit


def audit_table(ctx: Context, spec: TableSpec, near: pymupdf.Rect) -> dict[str, Any]:
    page = ctx.pdf[spec.pdf_page - 1]
    table = logical_table(page, near)
    rect = pymupdf.Rect(table.bbox)
    bboxes = [tuple(c) for row in table.rows for c in row.cells if c is not None]
    cells = grid_cells(list(dict.fromkeys(bboxes)))
    clusters = [c for c in page_clusters(page) if _inside(c, [rect])]
    fixed, glyph_fixes = fix_glyphs(page, clusters)
    originals = {id(f): o for f, o in zip(fixed, clusters, strict=True)}
    symbols = [s for s in page_symbols(page) if rect.contains(s["rect"])]
    for cell in cells:
        r = pymupdf.Rect(cell["bbox"])
        inside = [f for f in fixed if _inside(f, [r])]
        lines = [make_line(spec.pdf_page, ln, ctx.kashida, ctx.stats, (glyph_fixes, originals))
                 for ln in _lines(inside)]  # fmt: skip
        text = "\n".join(ln.text for ln in lines)
        raw = "\n".join(ln.raw for ln in lines)
        kind = classify_value(text.replace("\n", " "))
        cell.update(text_raw=raw, text_repair_v1="\n".join(ln.machine or ln.text for ln in lines),
                    text=text, value_class=kind,
                    glyph_rules=dict(sum((Counter(ln.glyph_rules) for ln in lines), Counter())),
                    flags=cell_flags(text, raw, [originals.get(id(f), f) for f in inside],
                                     len(lines)))  # fmt: skip
        parsed = parse_value(text)
        if parsed:
            cell["parsed"] = parsed
        if kind in ("identifier", "expression", "number", "percent"):
            cell["visual_ltr"] = visual_ltr(inside)
        sup = superscripts([originals.get(id(f), f) for f in inside])
        if sup:
            cell["superscript_text"] = sup
        here = [s for s in symbols if r.contains((s["ink"].tl + s["ink"].br) / 2)]
        for s in here:
            s["ink_inside_cell"] = r.contains(s["ink"])
        if here:
            cell["value_class"] = "symbol" if not text.strip() else "label_with_symbol"
            keys = ("symbol", "mask_digest", "name", "ink_inside_cell")
            cell["symbols"] = [{k: s[k] for k in keys} for s in here]
    return {"table": spec.number, "pdf_page": spec.pdf_page, "bbox": [round(v, 1) for v in rect],
            "rows": table.row_count, "cols": table.col_count, "cells": cells,
            "symbols_on_table": len(symbols)}  # fmt: skip


def summarize(audit: dict[str, Any]) -> dict[str, Any]:
    cells = audit["cells"]
    merged = [c for c in cells if c["rowspan"] > 1 or c["colspan"] > 1]
    symbols = Counter(s["symbol"] for c in cells for s in c.get("symbols", []))
    return {
        "grid": f"{audit['rows']} x {audit['cols']}", "cells": len(cells),
        "merged_cells": [{"row": c["row"], "col": c["col"], "rowspan": c["rowspan"],
                          "colspan": c["colspan"], "text": c["text"][:60]} for c in merged],
        "value_classes": dict(Counter(c["value_class"] for c in cells)),
        "flags": dict(Counter(f for c in cells for f in c["flags"])),
        "glyph_rules": dict(sum((Counter(c["glyph_rules"]) for c in cells), Counter())),
        "symbols": dict(symbols),
        "symbols_ink_crossing_cell_border": sum(not s["ink_inside_cell"] for c in cells
                                                for s in c.get("symbols", [])),
        "cells_with_more_than_one_symbol": sum(len(c.get("symbols", [])) > 1 for c in cells),
        "parse_errors": [c["text"] for c in cells if c.get("parsed", {}).get("error")],
        "decimal_values": sum(1 for c in cells if c.get("parsed", {}).get("value")),
    }  # fmt: skip


def symbol_matrix(audit: dict[str, Any]) -> dict[tuple[int, int], str]:
    return {(c["row"], c["col"]): c["symbols"][0]["symbol"] for c in audit["cells"]
            if len(c.get("symbols", [])) == 1}  # fmt: skip


def year_monotonic_violations(audit: dict[str, Any], first_row: int, groups: int,
                              years: int) -> list[dict[str, Any]]:  # fmt: skip
    """Table 19-2-1: a requirement marked ✓ in one year and ✗ in a later year of the same group
    (an observation for review, not a rule of the source)."""
    m = symbol_matrix(audit)
    cols = sorted({c for (_, c) in m})
    out = []
    for g in range(groups):
        rows = range(first_row + g * years, first_row + (g + 1) * years)
        for col in cols:
            seq = [m.get((r, col)) for r in rows]
            for i in range(1, len(seq)):
                if seq[i - 1] == "check_mark" and seq[i] == "cross_mark":
                    out.append({"group_index": g, "col": col, "year_index": i})
    return out
