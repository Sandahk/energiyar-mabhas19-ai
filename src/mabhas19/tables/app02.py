"""Appendix 2 candidate records: city -> climate class (Phase 2D).

Field sources (never mixed):

* row number and climate code: the PDF text layer of the table cell, exactly as printed;
* city name: a visual read (Gemini whole page), because the Persian city names are drawn as
  vector outlines without a text layer; a second, independent read (Tesseract on the cell
  image) is kept only to flag disagreements, never to choose a value.

The climate code is regulatory data: it is normalized only by removing whitespace and
folding digits/letter case, validated against the ASHRAE 169 pattern, and never remapped.
«5C» stays «5C» and is linked to the known source issues instead of being treated as an
extraction error.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

from mabhas19.text.normalize import normalize_for_comparison, normalize_for_search, to_ascii_digits

STATUS = "machine_extracted_pending_review"
CLIMATE_CODE = re.compile(r"^[0-8][ABC]$")  # ASHRAE 169 climate zone + moisture regime
SOURCE_ISSUES_BY_CODE = {"5C": ["SI-0001", "SI-0002"]}  # see curated/source_issues.yaml
# Backlinks to unresolved cross-document inconsistencies (no value is changed by them):
# SI-0009 = Isfahan is 4B here but 3B in main table 19-3-2.
SOURCE_ISSUES_BY_ROW = {"app02-p002-r07": ["SI-0009"]}
_PERSIAN_LETTERS = re.compile(r"[؀-ۿ‌ ()]+")  # e.g. «تهران (مهرآباد)»
_DISPLAY_FOLDS = {"ك": "ک", "ي": "ی"}  # Arabic kaf/yeh -> Persian (display-safe)


@dataclass(frozen=True)
class Code:
    raw: str | None
    value: str | None
    valid: bool
    note: str | None = None


def normalize_code(raw: str | None) -> Code:
    """Whitespace removed, digits/letters folded; anything else is rejected, never guessed."""
    if raw is None or not raw.strip():
        return Code(raw, None, False, "missing")
    value = to_ascii_digits("".join(raw.split())).upper()
    if not CLIMATE_CODE.match(value):
        return Code(raw, None, False, f"not a climate code: {raw!r}")
    return Code(raw, value, True)


def city_display(raw: str | None) -> str | None:
    """Printed spelling kept: NFC, single spaces, Arabic kaf/yeh to Persian only."""
    if raw is None or not raw.strip():
        return None
    text = unicodedata.normalize("NFC", raw)
    text = "".join(_DISPLAY_FOLDS.get(c, c) for c in text)
    return " ".join(text.split())


def city_normalized(raw: str | None) -> str | None:
    return None if raw is None or not raw.strip() else normalize_for_search(raw)


def same_city(a: str | None, b: str | None) -> bool:
    if not a or not b:
        return False
    return normalize_for_comparison(a) == normalize_for_comparison(b)


@dataclass
class TextRow:
    """One table row from the PDF text layer (and its geometry)."""

    pdf_page: int
    row_order: int
    row_number_raw: str
    code_raw: str
    y: float
    city_bbox: tuple[float, float, float, float]
    city_glyph_lines: int


@dataclass
class VisualRead:
    city_raw: str | None
    code_raw: str | None = None
    row_number_raw: str | None = None
    source: dict[str, Any] = field(default_factory=dict)


def _flag(rec: dict[str, Any], code: str, detail: str) -> None:
    rec["suspicious"].append({"code": code, "detail": detail})


def build_record(row: TextRow, printed_page: str, gemini: VisualRead | None,
                 ocr: VisualRead | None, doc_id: str) -> dict[str, Any]:  # fmt: skip
    digits = to_ascii_digits(row.row_number_raw.strip())
    number = int(digits) if digits.isascii() and digits.isdigit() else None
    code = normalize_code(row.code_raw)
    city_raw = gemini.city_raw if gemini else None
    rec: dict[str, Any] = {
        "candidate_id": f"app02-p{row.pdf_page:03d}-r{row.row_order:02d}",
        "doc_id": doc_id, "pdf_page": row.pdf_page, "printed_page": printed_page,
        "row_order": row.row_order, "row_number_raw": row.row_number_raw,
        "row_number": number, "row_y": round(row.y, 1),
        "city_bbox": [round(v, 1) for v in row.city_bbox],
        "city_raw": city_raw, "city": city_display(city_raw),
        "city_normalized": city_normalized(city_raw),
        "city_second_read_raw": ocr.city_raw if ocr else None,
        "city_reads_agree": same_city(city_raw, ocr.city_raw if ocr else None),
        "climate_code_raw": row.code_raw, "climate_code": code.value,
        "climate_code_valid": code.valid,
        "climate_code_visual_read": gemini.code_raw if gemini else None,
        "source_issue_refs": SOURCE_ISSUES_BY_CODE.get(code.value or "", [])
        + SOURCE_ISSUES_BY_ROW.get(f"app02-p{row.pdf_page:03d}-r{row.row_order:02d}", []),
        "provenance": {
            "row_number": "pdf_text_layer", "climate_code": "pdf_text_layer",
            "city": gemini.source if gemini else None,
            "city_second_read": ocr.source if ocr else None,
        },
        "suspicious": [], "verification_status": STATUS,
    }  # fmt: skip
    _check(rec, row, gemini, ocr, code)
    return rec


def _check(rec: dict[str, Any], row: TextRow, gemini: VisualRead | None,
           ocr: VisualRead | None, code: Code) -> None:  # fmt: skip
    if rec["row_number"] is None:
        _flag(rec, "row_number_unparsed", f"row number text {row.row_number_raw!r}")
    if not code.valid:
        _flag(rec, "climate_code_invalid", code.note or "invalid")
    if row.city_glyph_lines == 0:
        _flag(rec, "city_cell_empty", "no printed glyphs in the city cell")
    city = rec["city_raw"]
    if gemini is None or not city:
        _flag(rec, "city_missing", "no visual read for this row")
        return
    if not _PERSIAN_LETTERS.fullmatch(city.strip()):
        _flag(rec, "city_unexpected_characters", f"{city!r} has non-Persian characters")
    if gemini.code_raw is not None and normalize_code(gemini.code_raw).value != code.value:
        _flag(rec, "code_reads_disagree", f"visual read {gemini.code_raw!r} vs text layer "
              f"{row.code_raw!r} (text layer kept)")  # fmt: skip
    visual_number = to_ascii_digits((gemini.row_number_raw or "").strip())
    if gemini.row_number_raw is not None and visual_number != to_ascii_digits(
            row.row_number_raw.strip()):  # fmt: skip
        _flag(rec, "row_alignment", f"visual row number {gemini.row_number_raw!r}")
    if ocr is None or not ocr.city_raw:
        _flag(rec, "city_uncorroborated", "the second read (Tesseract) returned nothing")
    elif not rec["city_reads_agree"]:
        _flag(rec, "city_reads_disagree", f"Gemini {city!r} vs Tesseract {ocr.city_raw!r}")


def flag_duplicates(records: list[dict[str, Any]]) -> list[list[str]]:
    """Rows sharing a normalized city name (flagged; both rows are kept)."""
    by_name: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        if r["city_normalized"]:
            by_name.setdefault(r["city_normalized"], []).append(r)
    groups = []
    for name, rows in by_name.items():
        if len(rows) > 1:
            ids = [r["candidate_id"] for r in rows]
            groups.append(ids)
            for r in rows:
                _flag(
                    r,
                    "duplicate_city",
                    f"«{name}» also on {[i for i in ids if i != r['candidate_id']]}",
                )
    return groups
