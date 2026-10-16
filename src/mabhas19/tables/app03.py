"""Appendix 3 candidates: occupancy x climate class x energy rating -> maximum EUI (Phase 2E).

Field sources (never mixed):

* EUI values, climate-column labels and the energy rating: the PDF text layer, exactly as
  printed; each value is assigned to the climate label printed at the head of its own column;
* occupancy labels and row numbers: drawn as vector outlines (no text layer), read visually
  (Gemini whole page); Tesseract on the cell image is only a corroboration signal.

The tables have no 5C column. No value is ever produced for 5C: the gap is recorded as the
existing source issue (SI-0002, linked to SI-0001), not as an extraction error.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from decimal import Decimal
from itertools import combinations
from typing import Any

from rapidfuzz import fuzz

from mabhas19.tables.app02 import normalize_code
from mabhas19.text.normalize import normalize_for_comparison, normalize_for_search, to_ascii_digits
from mabhas19.text.numbers import HIGH, parse_numeric

STATUS = "machine_extracted_pending_review"
UNIT = "kWh/m2.yr"
RATINGS = ("D", "C", "B", "A")
MISSING_CLIMATE = {"5C": ["SI-0002", "SI-0001"]}  # column absent; never filled
OPERATOR_ROWS = {1: ["SI-0003"], 2: ["SI-0003"]}  # «مسکونی ≤/> ۳۰۰۰ مترمربع»
NEAR_DUPLICATE = 85.0  # rapidfuzz ratio (0-100) on comparison-normalized labels
_DISPLAY_FOLDS = {"ك": "ک", "ي": "ی"}
_SPLIT_NUMBER = re.compile(r"[0-9۰-۹٠-٩]\s+[0-9۰-۹٠-٩]")
_LABEL_CHARS = re.compile(r"[؀-ۿ‌ ()/،\-–≤≥<>]+")


@dataclass(frozen=True)
class Eui:
    raw: str
    value: str | None
    error: str | None = None


def parse_eui(raw: str | None) -> Eui:
    """One explicit positive integer as printed; anything else is rejected, never repaired."""
    text = (raw or "").strip()
    if not text:
        return Eui(raw or "", None, "missing")
    if _SPLIT_NUMBER.search(text):  # «1 030»: two cells or a separator, never joined
        return Eui(raw or "", None, "whitespace inside the number")
    v = parse_numeric(text, source_confidence=HIGH)
    if v.normalized_value is None or v.confidence != HIGH or v.issues:
        return Eui(raw or "", None, "; ".join(v.issues) or "unparseable")
    if "." in v.normalized_value or Decimal(v.normalized_value) <= 0:
        return Eui(raw or "", None, "not a positive integer")
    return Eui(raw or "", v.normalized_value)


def label_display(raw: str | None) -> str | None:
    if raw is None or not raw.strip():
        return None
    text = unicodedata.normalize("NFC", raw)
    return " ".join("".join(_DISPLAY_FOLDS.get(c, c) for c in text).split())


def label_normalized(raw: str | None) -> str | None:
    return None if raw is None or not raw.strip() else normalize_for_search(raw)


def same_label(a: str | None, b: str | None) -> bool:
    return bool(a and b) and normalize_for_comparison(a or "") == normalize_for_comparison(b or "")


@dataclass
class TableRow:
    """One occupancy row of one table, from the text layer and the cell geometry."""

    rating: str
    pdf_page: int
    row_order: int
    y: float
    values: dict[str, str]  # climate label (as printed) -> raw cell text
    value_bboxes: dict[str, tuple[float, float, float, float]]
    columns: dict[str, int]  # climate label -> table column index
    table_row_index: int
    label_bbox: tuple[float, float, float, float]
    number_bbox: tuple[float, float, float, float]
    label_glyph_lines: int


@dataclass
class LabelRead:
    label_raw: str | None
    row_number_raw: str | None = None
    values: dict[str, str] = field(default_factory=dict)
    source: dict[str, Any] = field(default_factory=dict)


def _flag(rec: dict[str, Any], code: str, detail: str) -> None:
    rec["suspicious"].append({"code": code, "detail": detail})


def row_record(row: TableRow, printed_page: str, gemini: LabelRead | None,
               ocr: LabelRead | None, doc_id: str) -> dict[str, Any]:  # fmt: skip
    raw = gemini.label_raw if gemini else None
    digits = to_ascii_digits((gemini.row_number_raw if gemini else "") or "").strip()
    rec: dict[str, Any] = {
        "row_id": f"app03-{row.rating}-r{row.row_order:02d}", "doc_id": doc_id,
        "energy_rating": row.rating, "pdf_page": row.pdf_page, "printed_page": printed_page,
        "row_order": row.row_order, "table_row_index": row.table_row_index,
        "row_number_raw": gemini.row_number_raw if gemini else None,
        "row_number": int(digits) if digits.isascii() and digits.isdigit() else None,
        "row_y": round(row.y, 1), "label_bbox": [round(v, 1) for v in row.label_bbox],
        "occupancy_raw": raw, "occupancy": label_display(raw),
        "occupancy_normalized": label_normalized(raw),
        "occupancy_second_read_raw": ocr.label_raw if ocr else None,
        "occupancy_reads_agree": same_label(raw, ocr.label_raw if ocr else None),
        "source_issue_refs": OPERATOR_ROWS.get(row.row_order, []),
        "provenance": {"occupancy": gemini.source if gemini else None,
                       "occupancy_second_read": ocr.source if ocr else None,
                       "row_number": "visual read (Gemini); vector-drawn, no text layer"},
        "suspicious": [], "verification_status": STATUS,
    }  # fmt: skip
    if row.label_glyph_lines == 0:
        _flag(rec, "label_cell_empty", "no printed glyphs in the occupancy cell")
    if not raw:
        _flag(rec, "label_missing", "no visual read for this row")
    elif not _LABEL_CHARS.fullmatch(raw.strip()):
        _flag(rec, "label_unexpected_characters", f"{raw!r}")
    if rec["row_number"] != row.row_order:
        _flag(rec, "row_alignment", f"visual row number {rec['row_number_raw']!r}, position "
              f"{row.row_order}")  # fmt: skip
    if raw and (ocr is None or not ocr.label_raw):
        _flag(rec, "label_uncorroborated", "the second read (Tesseract) returned nothing")
    elif raw and not rec["occupancy_reads_agree"]:
        _flag(rec, "label_reads_disagree", f"Gemini {raw!r} vs Tesseract {ocr.label_raw!r}")  # type: ignore[union-attr]
    return rec


def cell_records(row: TableRow, rec: dict[str, Any], gemini: LabelRead | None
                 ) -> list[dict[str, Any]]:  # fmt: skip
    out = []
    for label, raw in row.values.items():
        code = normalize_code(label)
        eui = parse_eui(raw)
        visual = gemini.values.get(code.value or label) if gemini else None
        visual_value = parse_eui(visual).value if visual is not None else None
        cell: dict[str, Any] = {
            "cell_id": f"{rec['row_id']}-{code.value or label.strip()}", "row_id": rec["row_id"],
            "doc_id": rec["doc_id"], "energy_rating": row.rating, "pdf_page": row.pdf_page,
            "printed_page": rec["printed_page"], "row_order": row.row_order,
            "climate_class_raw": label, "climate_class": code.value,
            "eui_raw": raw, "eui": eui.value, "unit": UNIT,
            "position": {"table_row_index": row.table_row_index,
                         "table_col_index": row.columns[label],
                         "bbox": [round(v, 1) for v in row.value_bboxes[label]]},
            "eui_visual_read": visual,
            "provenance": {"eui": "pdf_text_layer", "climate_class": "pdf_text_layer header",
                           "energy_rating": "pdf_text_layer caption"},
            "suspicious": [], "verification_status": STATUS,
        }  # fmt: skip
        if not code.valid:
            _flag(cell, "climate_label_invalid", code.note or "invalid")
        if eui.error:
            _flag(cell, "eui_unparseable", eui.error)
        if visual is not None and visual_value != eui.value:
            _flag(cell, "eui_reads_disagree", f"visual {visual!r} vs text layer {raw!r} "
                  "(text layer kept)")  # fmt: skip
        out.append(cell)
    return out


def near_duplicates(labels: list[tuple[str, str]]) -> list[dict[str, Any]]:
    """Pairs of distinct labels (id, text) that are identical or very similar."""
    out = []
    for (ida, a), (idb, b) in combinations(labels, 2):
        na, nb = normalize_for_comparison(a), normalize_for_comparison(b)
        score = fuzz.ratio(na, nb)
        if na == nb or score >= NEAR_DUPLICATE:
            out.append({"a": ida, "b": idb, "label_a": a, "label_b": b,
                        "kind": "duplicate" if na == nb else "near_duplicate",
                        "similarity": round(score, 1)})  # fmt: skip
    return out
