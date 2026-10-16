"""Review sheet (PDF) for the flagged appendix 3 occupancy labels (offline, no decisions).

The flagged rows are deduplicated by occupancy position (row order); each position shows the
four tables' crops (D, C, B, A) side by side, every Gemini and Tesseract reading, the current
candidate label and all flags. Rows 1-2 (SI-0003) get large crops, a tight crop around the
comparison operator and «۳۰۰۰», and the drawing order of the vector outlines in that region:
the PDF holds outlines only, so no character (>, <, ≥, ≤) is derived from it. Nothing is
preselected, recorded or changed.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import pymupdf

from mabhas19.tables.app03 import RATINGS
from mabhas19.tables.app03_build import PDF
from mabhas19.text.normalize import to_ascii_digits

OPTIONS = ("ACCEPT CURRENT LABEL", "CORRECT TO: ______________", "SOURCE AMBIGUOUS")
OPERATOR_ROWS = (1, 2)
CROP_DPI = 300
TIGHT_WINDOW = (0.30, 0.70)  # share of the label's drawn width holding «۳۰۰۰» + operator
_CSS = ("body{font-family:sans-serif;font-size:8.5px;direction:ltr}h1{font-size:12px;margin:0}"
        "h2{font-size:11px;margin:0}table{border-collapse:collapse;width:100%}"
        "td,th{border:0.5px solid #aaa;padding:1px 3px;vertical-align:top;width:22%}"
        "th{color:#555}th:first-child{width:12%}"
        ".fa{direction:rtl;text-align:right;font-size:10.5px}.o{margin-right:14px}"
        ".si{color:#a33;font-weight:bold}")  # fmt: skip


def positions(rows: list[dict[str, Any]]) -> dict[int, dict[str, dict[str, Any]]]:
    """Flagged row positions -> rating -> row (all four ratings kept for context)."""
    flagged = sorted({r["row_order"] for r in rows if r["suspicious"]})
    return {n: {r["energy_rating"]: r for r in rows if r["row_order"] == n} for n in flagged}


def outlines(page: pymupdf.Page, box: pymupdf.Rect) -> list[dict[str, Any]]:
    """Vector outlines inside a cell, in drawing (content-stream) order, cell frame excluded."""
    return [{"seq": d["seqno"], "x0": round(d["rect"].x0, 1), "x1": round(d["rect"].x1, 1),
             "segments": len(d["items"])} for d in page.get_drawings()
            if box.contains(d["rect"]) and d["rect"].width < box.width * 0.9]  # fmt: skip


def tight_region(
    page: pymupdf.Page, box: pymupdf.Rect
) -> tuple[pymupdf.Rect, list[dict[str, Any]]]:
    """The middle part of the drawn label (rows 1-2: «۳۰۰۰» and the operator).

    A fixed window over the horizontal extent of the outlines, not a reading of the glyphs:
    word gaps are too uneven in these outlines to segment words reliably.
    """
    items = outlines(page, box)
    if not items:
        return box, []
    left, right = min(o["x0"] for o in items), max(o["x1"] for o in items)
    x0, x1 = left + TIGHT_WINDOW[0] * (right - left), left + TIGHT_WINDOW[1] * (right - left)
    inside = [o for o in items if x0 <= (o["x0"] + o["x1"]) / 2 <= x1]
    return pymupdf.Rect(x0, box.y0, x1, box.y1), sorted(inside, key=lambda o: o["seq"])


def _crop(src: pymupdf.Document, r: dict[str, Any], clip: pymupdf.Rect) -> bytes:
    return bytes(src[r["pdf_page"] - 1].get_pixmap(dpi=CROP_DPI, clip=clip).tobytes("png"))


def _info(rows: dict[str, dict[str, Any]]) -> str:
    def e(v: Any) -> str:
        return "—" if v in (None, "") else html.escape(str(v))

    head = "<tr><th></th>" + "".join(f"<th>{k}</th>" for k in RATINGS) + "</tr>"
    lines = [("Gemini (raw)", "occupancy_raw"), ("Current candidate", "occupancy"),
             ("Normalized", "occupancy_normalized"),
             ("Tesseract", "occupancy_second_read_raw")]  # fmt: skip
    body = "".join(
        f"<tr><th>{name}</th>" + "".join(f"<td class='fa'>{e(rows[k].get(key))}</td>"
                                          for k in RATINGS) + "</tr>"
        for name, key in lines)  # fmt: skip
    flags = "".join(f"<td>{e('; '.join(f['code'] for f in rows[k]['suspicious']))}</td>"
                    for k in RATINGS)  # fmt: skip
    return f"<table>{head}{body}<tr><th>Flags</th>{flags}</tr></table>"


def _options(n: int) -> str:
    boxes = "".join(f"<span class='o'>☐ {html.escape(o)}</span>" for o in OPTIONS)
    return f"<div><b>Row {n} decision:</b> {boxes}</div>"


def _new_page(doc: pymupdf.Document, archive: pymupdf.Archive | None, title: str) -> pymupdf.Page:
    page = doc.new_page(width=842, height=595)  # A4 landscape: four crops side by side
    page.insert_htmlbox(pymupdf.Rect(24, 10, 818, 28), f"<h1>{html.escape(title)}</h1>",
                        css=_CSS, archive=archive)  # fmt: skip
    return page


def _four_crops(page: pymupdf.Page, src: pymupdf.Document, rows: dict[str, dict[str, Any]],
                area: tuple[float, float], *, tight: bool = False) -> float:  # fmt: skip
    """Crops for D, C, B, A from left to right (labels above each); returns the bottom.

    ``area`` is (top, maximum crop height) in points.
    """
    top, height = area
    width = (794 - 3 * 8) / 4
    bottom = top
    for i, k in enumerate(RATINGS):
        r = rows[k]
        box = pymupdf.Rect(r["label_bbox"])
        clip = tight_region(src[r["pdf_page"] - 1], box)[0] if tight else pymupdf.Rect(
            box.x0, box.y0, box.x1 + 22, box.y1)  # fmt: skip
        x = 24 + i * (width + 8)
        h = min(height, width * clip.height / clip.width)
        printed = to_ascii_digits(r["printed_page"])
        page.insert_text((x, top + 8), f"{k}  (PDF p{r['pdf_page']}, printed page {printed})",
                         fontsize=7)  # fmt: skip
        page.insert_image(pymupdf.Rect(x, top + 11, x + width, top + 11 + h),
                          stream=_crop(src, r, clip), keep_proportion=True)  # fmt: skip
        bottom = max(bottom, top + 11 + h)
    return bottom


def _operator_page(doc: pymupdf.Document, src: pymupdf.Document, archive: pymupdf.Archive | None,
                   n: int, rows: dict[str, dict[str, Any]]) -> None:  # fmt: skip
    page = _new_page(doc, archive, f"Occupancy row {n} – SI-0003 (comparison operator) – "
                     "four tables D, C, B, A")  # fmt: skip
    y = _four_crops(page, src, rows, (32, 60)) + 6
    page.insert_text((24, y + 8), "Tight crops: middle 30-70% of the drawn label (the digits and "
                     "the operator), left to right exactly as printed", fontsize=7)  # fmt: skip
    y = _four_crops(page, src, rows, (y + 10, 70), tight=True) + 6
    order = []
    for k in RATINGS:
        r = rows[k]
        _, mid = tight_region(src[r["pdf_page"] - 1], pymupdf.Rect(r["label_bbox"]))
        seq = ", ".join(f"#{o['seq']} x{o['x0']}–{o['x1']} ({o['segments']} seg)" for o in mid)
        order.append(f"<tr><th>{k}</th><td>{html.escape(seq) or '—'}</td></tr>")
    note = ("<p class='si'>SI-0003: the operator direction is not settled. The PDF has no text "
            "layer here: the label is vector outlines only, so no character code exists. Below: "
            "the outlines of the tight region in drawing order (seq number, x range in points, "
            "path segments). Drawing order is not character identity; nothing is inferred.</p>"
            f"<table>{''.join(order)}</table>")  # fmt: skip
    page.insert_htmlbox(pymupdf.Rect(24, y, 818, 585), note + _info(rows) + _options(n),
                        css=_CSS, archive=archive, scale_low=0)  # fmt: skip


def _row_block(ctx: tuple[pymupdf.Page, pymupdf.Document, pymupdf.Archive | None], n: int,
               rows: dict[str, dict[str, Any]], top: float) -> None:  # fmt: skip
    page, src, archive = ctx
    page.insert_htmlbox(pymupdf.Rect(24, top, 818, top + 14), f"<h2>Occupancy row {n}</h2>",
                        css=_CSS, archive=archive)  # fmt: skip
    y = _four_crops(page, src, rows, (top + 13, 34)) + 3
    note = ""
    if n == 23:
        note = (
            "<p>Check the C crop: in table C, Gemini and Tesseract disagree on the word "
            "«سوپر» (Tesseract: «سویر»).</p>"
        )
    if n == 28:
        note = (
            "<p>Confirm the printed spacing and punctuation of «کلانتری/ آتش نشانی» in all "
            "four crops (Gemini spacing around «/» differs between tables).</p>"
        )
    page.insert_htmlbox(pymupdf.Rect(24, y, 818, top + 172), note + _info(rows) + _options(n),
                        css=_CSS, archive=archive, scale_low=0)  # fmt: skip


def write_pdf(cand_dir: Path, out: Path) -> dict[str, Any]:
    lines = (cand_dir / "occupancy_rows.jsonl").read_text(encoding="utf-8").splitlines()
    rows = [json.loads(x) for x in lines if x.strip()]
    pos = positions(rows)
    fonts = Path("C:/Windows/Fonts")
    archive = pymupdf.Archive(str(fonts)) if fonts.is_dir() else None
    doc = pymupdf.open()
    with pymupdf.open(PDF) as src:
        for n in (p for p in OPERATOR_ROWS if p in pos):
            _operator_page(doc, src, archive, n, pos[n])
        rest = [n for n in pos if n not in OPERATOR_ROWS]
        for i in range(0, len(rest), 3):
            page = _new_page(doc, archive, "Appendix 3 – occupancy labels to review "
                             f"(rows {', '.join(map(str, rest[i:i + 3]))})")  # fmt: skip
            for k, n in enumerate(rest[i : i + 3]):
                _row_block((page, src, archive), n, pos[n], 32 + k * 186)
    doc.subset_fonts()
    doc.save(out, garbage=4, deflate=True)
    pages = int(doc.page_count)
    doc.close()
    return {"positions": list(pos), "count": len(pos), "pages": pages, "pdf": str(out)}
