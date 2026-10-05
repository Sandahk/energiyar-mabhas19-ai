"""Candidate extraction methods compared by the Phase 2A benchmark.

Page -> plain text:   text_layer_raw, repair_v1 (main only), tesseract_page
Region -> structured: grid_textlayer (+ OCR fallback for vector cells), grid_ocr,
                      app07_rows_ocr (rows located from vector digit glyphs)
These are benchmark candidates, not the production pipeline.
"""

from __future__ import annotations

import re
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

import pymupdf

from mabhas19.ingest.repair.pipeline import FinalFormStats, repair_page
from mabhas19.providers.ocr import OcrEngine
from mabhas19.tables.schemas import (
    CityClimateRow,
    DensitySpec,
    EnergyLimitCell,
    MaterialPropertyRow,
    NumericField,
    SourceRef,
    normalized_label,
)
from mabhas19.text.charclass import digits_to_int
from mabhas19.text.normalize import fold_hyphens, to_ascii_digits
from mabhas19.text.numbers import HIGH, LOW, MEDIUM, parse_numeric

OCR_DPI = 300
WATERMARK_CUTOFF = 180  # grey levels above this (light «inbr» watermark, paper) -> white
LANGS = ("fas", "eng")
_WHITE = bytes(b if b <= WATERMARK_CUTOFF else 255 for b in range(256))


def render_clean(page: pymupdf.Page, out: Path, clip: pymupdf.Rect | None = None,
                 dpi: int = OCR_DPI) -> Path:  # fmt: skip
    """Greyscale render with light pixels (the pink watermark) pushed to white."""
    pix = page.get_pixmap(dpi=dpi, clip=clip, colorspace=pymupdf.csGRAY)
    clean = pymupdf.Pixmap(pymupdf.csGRAY, pix.width, pix.height,
                           bytes(pix.samples).translate(_WHITE), False)  # fmt: skip
    clean.save(out)
    return out


def ocr_confidence(conf: float) -> str:
    """Map engine confidence to ours. OCR output is never ``high``: in the Phase 2A run
    Tesseract reported >= 90 for «5» where the source prints «۶۲۵», so OCR-derived values
    always require verification."""
    return MEDIUM if conf >= 70 else LOW


# --------------------------------------------------------------------------- page text


def text_layer_raw(page: pymupdf.Page) -> str:
    return str(page.get_text())


def repair_v1(page: pymupdf.Page, kashida: frozenset[tuple[str, int]],
              stats: FinalFormStats | None) -> str:  # fmt: skip
    return repair_page(page, kashida, stats).repaired_display_text


def tesseract_page(page: pymupdf.Page, engine: OcrEngine, workdir: Path) -> str:
    img = render_clean(page, workdir / f"p{page.number + 1}.png")
    return engine.recognize(img, LANGS, psm=6).text


# --------------------------------------------------------------------------- grid tables


@dataclass(frozen=True)
class Cell:
    x: float  # centre
    text: str
    confidence: str


@dataclass(frozen=True)
class GridRow:
    y: float  # centre
    cells: list[Cell]  # left to right


def _ocr_cell(page: pymupdf.Page, rect: pymupdf.Rect, engine: OcrEngine, tmp: Path) -> Cell:
    inner = pymupdf.Rect(rect.x0 + 1, rect.y0 + 1, rect.x1 - 1, rect.y1 - 1)
    img = render_clean(page, tmp / "cell.png", clip=inner)
    res = engine.recognize(img, LANGS, psm=7)
    conf = min((w.confidence for w in res.words), default=0.0)
    return Cell((rect.x0 + rect.x1) / 2, " ".join(res.text.split()), ocr_confidence(conf))


def grid_rows(page: pymupdf.Page, mode: str, engine: OcrEngine | None,
              kashida: frozenset[tuple[str, int]]) -> list[GridRow]:  # fmt: skip
    """Rows of the largest ruled table. mode: 'textlayer' (OCR only empty cells) or 'ocr'."""
    tables = page.find_tables().tables
    if not tables:
        return []
    table = max(tables, key=lambda t: t.row_count * t.col_count)
    rows: list[GridRow] = []
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for row in table.rows:
            cells = []
            for bbox in row.cells:
                if bbox is None:
                    continue
                rect = pymupdf.Rect(bbox)
                if rect.width < 8:
                    continue  # spacer columns produced by thick rules
                text = ""
                if mode == "textlayer":
                    text = repair_page(page, kashida, None, clip=rect).repaired_display_text
                    text = " ".join(text.split())
                if text:
                    cells.append(Cell((rect.x0 + rect.x1) / 2, text, HIGH))
                elif engine is not None:
                    cells.append(_ocr_cell(page, rect, engine, tmp))
                else:  # keep empty cells so column positions stay aligned
                    cells.append(Cell((rect.x0 + rect.x1) / 2, "", LOW))
            y = (row.bbox[1] + row.bbox[3]) / 2
            rows.append(GridRow(y, sorted(cells, key=lambda c: c.x)))
    return rows


_CLIMATE = re.compile(r"^[0-9][A-Za-z]$")


def city_climate_rows(rows: list[GridRow], src: SourceRef) -> list[CityClimateRow]:
    """Appendix 2 grid: columns (left to right) climate class, city, row number."""
    out = []
    for r in rows:
        if len(r.cells) < 2:
            continue
        climate = r.cells[0].text.replace(" ", "")
        if not _CLIMATE.match(to_ascii_digits(climate)):
            continue
        number = digits_to_int(to_ascii_digits(r.cells[-1].text)) if len(r.cells) >= 3 else None
        city = (r.cells[1].text or None) if len(r.cells) >= 3 else None
        out.append(CityClimateRow(replace(src, row_y=r.y), number, city,
                                  normalized_label(city), climate, climate.upper()))  # fmt: skip
    return out


def energy_limit_cells(rows: list[GridRow], src: SourceRef, rating: str,
                       columns: list[str]) -> list[EnergyLimitCell]:  # fmt: skip
    """Appendix 3 grid: 9 numeric climate columns, then use type, then row number."""
    out: list[EnergyLimitCell] = []
    order = 0
    for r in rows:
        if len(r.cells) < len(columns):
            continue
        values = r.cells[: len(columns)]
        if not all(digits_to_int(to_ascii_digits(c.text.replace(" ", ""))) for c in values):
            continue
        order += 1
        label = (r.cells[len(columns)].text or None) if len(r.cells) > len(columns) else None
        for climate, cell in zip(columns, values, strict=True):
            v = parse_numeric(cell.text, source_confidence=cell.confidence)
            out.append(EnergyLimitCell(replace(src, row_y=r.y), None, order,
                                       label, normalized_label(label), rating, climate,
                                       NumericField.from_value(v)))  # fmt: skip
    return out


def value_rows(rows: list[GridRow]) -> list[tuple[float, list[Cell]]]:
    """Main table 19-5-1: rows holding numeric cells (left to right)."""
    rows_cells = [(r.y, [c for c in r.cells if c.text]) for r in rows]
    return [(y, cells) for y, cells in rows_cells if len(cells) >= 4]


# --------------------------------------------------------------------------- appendix 7

APP07_COLUMNS = {"conductivity": (121.0, 185.0), "density": (185.0, 263.0),
                 "material": (263.0, 542.0)}  # fmt: skip
APP07_HEADER_BOTTOM = 166.0
APP07_UNITS = {"density": "kg/m3", "thermal_conductivity": "W/m.K"}
_RANGE = re.compile(r"^(?P<a>[\d./]+)\s*تا\s*(?P<b>[\d./]+)$")
_QUALIFIED = re.compile(r"^(?P<q>بیش از|کمتر از)\s*(?P<a>[\d./]+)$")
_SINGLE = re.compile(r"^(?P<a>[\d./]+)$")
_CATEGORY = re.compile(r"^\s*\d{1,2}\s*[.]")


def glyph_row_centres(page: pymupdf.Page, x0: float, x1: float, top: float) -> list[float]:
    """Vertical centres of vector-drawn glyph lines inside a column (OCR-independent)."""
    ys = sorted(
        (d["rect"].y0 + d["rect"].y1) / 2
        for d in page.get_drawings()
        if x0 + 1 < d["rect"].x0 and d["rect"].x1 < x1 - 1 and d["rect"].height < 15
        and d["rect"].y0 > top and d["rect"].width < 40
    )  # fmt: skip
    rows: list[list[float]] = []
    for y in ys:
        if rows and y - rows[-1][-1] <= 4:
            rows[-1].append(y)
        else:
            rows.append([y])
    return [sum(r) / len(r) for r in rows if len(r) >= 2]


def parse_density(text: str, confidence: str) -> DensitySpec | None:
    t = fold_hyphens(to_ascii_digits(" ".join(text.split())))
    if not t:
        return None

    def num(s: str) -> NumericField:
        return NumericField.from_value(parse_numeric(s, source_confidence=confidence))

    if m := _RANGE.match(t):
        a, b = num(m["a"]), num(m["b"])
        return DensitySpec("range", text, a, b)
    if m := _QUALIFIED.match(t):
        v = num(m["a"])
        return DensitySpec("more_than" if m["q"] == "بیش از" else "less_than", text,
                           v if m["q"] == "بیش از" else None,
                           v if m["q"] == "کمتر از" else None)  # fmt: skip
    if m := _SINGLE.match(t):
        v = num(m["a"])
        return DensitySpec("single", text, v, v)
    return DensitySpec("range", text, None, None)  # unparsed: both bounds missing


def app07_rows_ocr(page: pymupdf.Page, engine: OcrEngine, doc_id: str,
                   method: str = "app07_rows_ocr") -> list[MaterialPropertyRow]:  # fmt: skip
    pdf_page = page.number + 1
    cx0, cx1 = APP07_COLUMNS["conductivity"]
    dx0, dx1 = APP07_COLUMNS["density"]
    mx0, mx1 = APP07_COLUMNS["material"]
    rows: list[MaterialPropertyRow] = []
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        # Material column: OCR once, keep line positions to attach labels by height.
        mat_rect = pymupdf.Rect(mx0, APP07_HEADER_BOTTOM, mx1, page.rect.height * 0.86)
        mat_img = render_clean(page, tmp / "mat.png", clip=mat_rect)
        mat = engine.recognize(mat_img, LANGS, psm=6)
        scale = 72 / OCR_DPI
        lines: dict[float, list[str]] = {}
        for w in mat.words:
            y = mat_rect.y0 + (w.y0 + w.y1) / 2 * scale
            key = next((k for k in lines if abs(k - y) <= 5), y)
            lines.setdefault(key, []).append(w.text)
        label_lines = sorted((y, " ".join(reversed(ws))) for y, ws in lines.items())
        category: str | None = None
        material: str | None = None
        for y in glyph_row_centres(page, cx0, cx1, APP07_HEADER_BOTTOM):
            for ly, text in label_lines:
                if ly <= y + 5 and _CATEGORY.match(to_ascii_digits(text)):
                    category = text
            same_line = [t for ly, t in label_lines if abs(ly - y) <= 5]
            inherited = not same_line
            if same_line:
                material = same_line[0]
            cond = _ocr_cell(page, pymupdf.Rect(cx0, y - 8, cx1, y + 8), engine, tmp)
            dens = _ocr_cell(page, pymupdf.Rect(dx0, y - 8, dx1, y + 8), engine, tmp)
            k = parse_numeric(cond.text, slash_is_decimal=True, source_confidence=cond.confidence)
            rows.append(MaterialPropertyRow(
                SourceRef(doc_id, pdf_page, method, y), category, material, inherited, None,
                parse_density(dens.text, dens.confidence), NumericField.from_value(k),
                dict(APP07_UNITS),
            ))  # fmt: skip
    return rows
