"""Phase 2E outputs for appendix 3: raw layer, occupancy rows, EUI cells, checks, review.

Everything is written under data/processed/app03/<model>/ (git-ignored). No external API is
called here (Gemini answers are read from the cache) and nothing is written to curated/.
"""

from __future__ import annotations

import base64
import html
import json
import re
import shutil
import tempfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pymupdf

from mabhas19.benchmark import phase2b as pb
from mabhas19.benchmark.extractors import render_clean
from mabhas19.benchmark.references import load_references
from mabhas19.core.paths import DATA_DIR, EVAL_DIR, SOURCES_DIR
from mabhas19.providers.ocr import OcrEngine
from mabhas19.tables import app03 as a3
from mabhas19.tables import app03_decisions as dec
from mabhas19.tables.app02 import normalize_code
from mabhas19.tables.app02_build import _glyph_lines
from mabhas19.text.normalize import normalize_for_comparison, to_ascii_digits, to_persian_digits

DOC_ID = "m19-ed5-1404-app03"
PDF = SOURCES_DIR / "mabhas19" / "ed5-1404" / "appendices" / f"{DOC_ID}-v1.pdf"
TABLE_PAGES = (2, 3, 4, 5)
PRINTED_PAGE_OFFSET = 1  # printed page = PDF page - 1 (p2 «۱»)
OCR_LANGS = ("fas",)
CROP_DPI = 150
_RATING_LINE = re.compile(r"^\s*([ABCD])\s*$")


def request_for(pdf_page: int) -> pb.Request:
    page_id = f"app03-p{pdf_page:03d}"
    return pb.Request(f"{page_id}.app03", "app03", page_id, DOC_ID, pdf_page, pb.APP03_PROMPT,
                      True)  # fmt: skip


# --------------------------------------------------------------------------- text layer


def _text(page: pymupdf.Page, rect: pymupdf.Rect) -> str:
    return str(page.get_text("text", clip=rect)).replace("\n", "")


def page_rating(page: pymupdf.Page) -> list[str]:
    """Stand-alone rating letters printed on the page (the caption's «... انرژی D ...»)."""
    return [m[1] for line in str(page.get_text()).splitlines() if (m := _RATING_LINE.match(line))]


def table_rows(page: pymupdf.Page, rating: str) -> tuple[list[a3.TableRow], dict[str, Any]]:
    tables = page.find_tables().tables
    if not tables:
        return [], {"table_found": False}
    table = max(tables, key=lambda t: t.row_count * t.col_count)
    header: dict[int, str] = {}
    header_index = None
    for k, row in enumerate(table.rows):
        texts = {i: _text(page, pymupdf.Rect(b)) for i, b in enumerate(row.cells) if b is not None}
        codes = {i: t for i, t in texts.items() if normalize_code(t).valid}
        if len(codes) >= 5:
            header, header_index = codes, k
            break
    rows: list[a3.TableRow] = []
    if header_index is None:
        return rows, {"table_found": True, "header_found": False}
    for k, row in enumerate(table.rows[header_index + 1 :], start=header_index + 1):
        cells = row.cells
        values = {header[i]: _text(page, pymupdf.Rect(cells[i])) for i in header
                  if cells[i] is not None}  # fmt: skip
        if len(values) != len(header) or not any(v.strip() for v in values.values()):
            continue
        others = sorted((pymupdf.Rect(b) for i, b in enumerate(cells)
                         if b is not None and i not in header), key=lambda r: r.x0)  # fmt: skip
        if len(others) != 2:
            continue
        label_box, number_box = (others if others[0].width > others[1].width
                                 else (others[1], others[0]))  # fmt: skip
        rows.append(a3.TableRow(
            rating, page.number + 1, len(rows) + 1, (label_box.y0 + label_box.y1) / 2, values,
            {header[i]: tuple(pymupdf.Rect(cells[i])) for i in header},
            {label: i for i, label in header.items()}, k, tuple(label_box), tuple(number_box),
            _glyph_lines(page, label_box)))  # fmt: skip
    info = {"table_found": True, "header_found": True, "ruled_rows": table.row_count,
            "ruled_cols": table.col_count, "climate_columns": [header[i] for i in sorted(header)],
            "value_rows": len(rows),
            "rows_with_label_glyphs": sum(r.label_glyph_lines > 0 for r in rows),
            "rows_with_number_glyphs": sum(_glyph_lines(page, pymupdf.Rect(r.number_bbox)) > 0
                                           for r in rows)}  # fmt: skip
    return rows, info


# --------------------------------------------------------------------------- reads


def gemini_reads(responses: Path, pdf_page: int) -> tuple[list[a3.LabelRead], dict[str, Any]]:
    req = request_for(pdf_page)
    path = responses / f"{req.id}.json"
    if not path.is_file():
        return [], {"request_id": req.id, "status": "missing"}
    data = json.loads(path.read_text(encoding="utf-8"))
    parsed = pb.parse_json(data.get("text")) if data.get("status") == "ok" else None
    ok = pb.schema_ok("app03", parsed)
    rows = parsed["rows"] if ok and parsed else []
    reads = []
    for i, r in enumerate(rows):
        values = {normalize_code(str(k)).value or str(k): str(v)
                  for k, v in (r.get("values_raw") or {}).items()}  # fmt: skip
        label, number = pb._s(r.get("occupancy_type_raw")), pb._s(r.get("row_number_raw"))
        reads.append(a3.LabelRead(label, number, values, {"method": "gemini_whole_page",
                                           "model": data.get("model_version"),
                                           "request_id": req.id, "row_index": i,
                                           "received_at": data.get("received_at")}))  # fmt: skip
    info = {"request_id": req.id, "status": data.get("status"), "json_schema_ok": ok,
            "rows_returned": len(rows), "usage": data.get("usage"),
            "energy_rating_raw": (parsed or {}).get("energy_rating_raw"),
            "caption_raw": (parsed or {}).get("table_caption_raw"),
            "origin": "phase2e_call" if data.get("received_at") else "phase2b_cache"}  # fmt: skip
    return reads, info


def _match(reads: list[a3.LabelRead], order: int) -> a3.LabelRead | None:
    """The read whose printed row number equals the position, else the read at that index."""
    for r in reads:
        if to_ascii_digits((r.row_number_raw or "").strip()) == str(order):
            return r
    return reads[order - 1] if order - 1 < len(reads) else None


def ocr_read(page: pymupdf.Page, row: a3.TableRow, engine: OcrEngine, tmp: Path) -> a3.LabelRead:
    r = pymupdf.Rect(row.label_bbox)
    img = render_clean(page, tmp / "label.png", clip=pymupdf.Rect(r.x0 + 1, r.y0 + 1, r.x1 - 1,
                                                                  r.y1 - 1))  # fmt: skip
    res = engine.recognize(img, OCR_LANGS, psm=7)
    conf = min((w.confidence for w in res.words), default=None)
    return a3.LabelRead(" ".join(res.text.split()) or None, None, {},
                        {"method": "tesseract_cell", "languages": OCR_LANGS,
                         "min_confidence": conf})  # fmt: skip


# --------------------------------------------------------------------------- build


def build(model: str, responses: Path, out: Path, engine: OcrEngine | None) -> dict[str, Any]:
    (out / "raw").mkdir(parents=True, exist_ok=True)
    rows_out: list[dict[str, Any]] = []
    cells_out: list[dict[str, Any]] = []
    pages: dict[int, dict[str, Any]] = {}
    crops: dict[str, bytes] = {}
    with pymupdf.open(PDF) as pdf, tempfile.TemporaryDirectory() as td:
        for p in TABLE_PAGES:
            page = pdf[p - 1]
            ratings = page_rating(page)
            rating = ratings[0] if len(ratings) == 1 else "?"
            rows, info = table_rows(page, rating)
            reads, g_info = gemini_reads(responses, p)
            src = responses / f"{g_info['request_id']}.json"
            if src.is_file():
                shutil.copy2(src, out / "raw")
            printed = to_persian_digits(str(p - PRINTED_PAGE_OFFSET))
            for row in rows:
                gem = _match(reads, row.row_order)
                ocr = ocr_read(page, row, engine, Path(td)) if engine else None
                rec = a3.row_record(row, printed, gem, ocr, DOC_ID)
                rows_out.append(rec)
                cells_out += a3.cell_records(row, rec, gem)
                band = pymupdf.Rect(row.label_bbox) | pymupdf.Rect(row.number_bbox)
                crops[rec["row_id"]] = page.get_pixmap(dpi=CROP_DPI, clip=band).tobytes("png")
            pages[p] = {"energy_rating": rating, "rating_letters_in_text_layer": ratings,
                        "printed_page": printed, **info, "gemini": g_info}  # fmt: skip
    for r in rows_out:
        r["extraction_model"] = model
    order = cross_rating_order(rows_out)
    flag_cross_rating(rows_out, order)
    dups = a3.near_duplicates([(r["row_id"], r["occupancy"]) for r in rows_out
                               if r["energy_rating"] == "D" and r["occupancy"]])  # fmt: skip
    check = draft_reference_check(rows_out, cells_out)
    out.mkdir(parents=True, exist_ok=True)
    _jsonl(out / "occupancy_rows_machine.jsonl", rows_out)  # before review decisions
    rows_out = dec.apply_decisions(rows_out, dec.load_all())
    _jsonl(out / "occupancy_rows.jsonl", rows_out)
    reviewed_order = cross_rating_order(rows_out)
    _jsonl(out / "cells.jsonl", cells_out)
    summary = _summary(
        model, pages, rows_out, cells_out, {"order": order, "duplicates": dups, "draft": check}
    )
    summary["reviewed_cross_rating_order"] = {
        k: v for k, v in reviewed_order.items() if k != "differing_positions"
    }
    summary["review"] = {
        "reviewed_rows": sum(r["review"]["status"] == "reviewed" for r in rows_out),
        "by_basis": dict(Counter(b for r in rows_out for b in r["review"]["basis"])),
        "corroborated_not_reviewed": sum(r["review"]["status"] == "not_reviewed"
                                         and r["review"]["corroboration"] is not None
                                         for r in rows_out),
        "unresolved_flagged_rows": [r["row_id"] for r in rows_out if r["suspicious"]
                                    and r["review"]["status"] != "reviewed"],
        "labels_changed": [{"row_id": r["row_id"], "machine": r["occupancy_machine"],
                            "reviewed": r["occupancy"]} for r in rows_out
                           if "occupancy_machine" in r],
        "source_issue_resolutions": {r["row_id"]: r["source_issue_resolution"]["SI-0003"]["status"]
                                     for r in rows_out if r.get("source_issue_resolution")},
    }  # fmt: skip
    for name, obj in (("summary.json", summary), ("cross_rating_order.json", order),
                      ("draft_reference_check.json", check)):  # fmt: skip
        (out / name).write_text(json.dumps(obj, ensure_ascii=False, indent=1) + "\n",
                                encoding="utf-8", newline="\n")  # fmt: skip
    (out / "review.html").write_text(review_html(rows_out, crops, order, summary),
                                     encoding="utf-8", newline="\n")  # fmt: skip
    return summary


def _jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def flag_cross_rating(rows: list[dict[str, Any]], order: dict[str, Any]) -> None:
    """Rows whose label differs between the four independently read tables, and the
    operator rows of SI-0003, always go to review (the source text is not settled)."""
    differing = {p["row_order"]: p["labels"] for p in order["differing_positions"]}
    for r in rows:
        if r["row_order"] in differing:
            labels = differing[r["row_order"]]
            r["suspicious"].append({"code": "cross_rating_label_differs", "detail": "; ".join(
                f"{k}: {v}" for k, v in labels.items())})  # fmt: skip
        if "SI-0003" in r["source_issue_refs"]:
            r["suspicious"].append({"code": "operator_ambiguous", "detail": "comparison "
                                    "operator direction not settled (SI-0003)"})  # fmt: skip


def cross_rating_order(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Are the occupancy labels identical and in the same order in tables D, C, B, A?"""
    by_rating = {k: [r for r in rows if r["energy_rating"] == k] for k in a3.RATINGS}
    counts = {k: len(v) for k, v in by_rating.items()}
    n = max(counts.values(), default=0)
    positions = []
    for i in range(n):
        labels = {k: (v[i]["occupancy"] if i < len(v) else None) for k, v in by_rating.items()}
        keys = {normalize_for_comparison(x) if x else None for x in labels.values()}
        positions.append({"row_order": i + 1, "labels": labels, "identical": len(keys) == 1
                          and None not in keys})  # fmt: skip
    return {"rows_per_rating": counts, "same_row_count": len(set(counts.values())) == 1,
            "identical_positions": sum(p["identical"] for p in positions),
            "differing_positions": [p for p in positions if not p["identical"]],
            "all_identical_in_order": all(p["identical"] for p in positions)}  # fmt: skip


def draft_reference_check(
    rows: list[dict[str, Any]], cells: list[dict[str, Any]]
) -> dict[str, Any]:
    """Agreement with the page-2 draft reference (Claude transcription, NOT human-verified)."""
    ref = load_references(EVAL_DIR / "extraction" / "reference").get("app03-p002")
    if ref is None:
        return {"status": "no reference"}
    by_row = {r["row_order"]: r for r in rows if r["energy_rating"] == "D"}
    by_cell = {(c["row_order"], c["climate_class"]): c for c in cells if c["energy_rating"] == "D"}
    diffs, compared = [], 0
    for item in (i for i in ref.items if i.type == "table_row"):
        f = item.expected_fields
        for code, value in f["values"].items():
            compared += 1
            cell = by_cell.get((f["row_number"], code))
            if cell is None or cell["eui"] != value:
                diffs.append({"item": item.id, "climate": code, "reference": value,
                              "candidate": cell and cell["eui"]})  # fmt: skip
        rec = by_row.get(f["row_number"])
        label = rec["occupancy"] if rec else None
        if f.get("label_check", True) and not a3.same_label(f["occupancy_type"], label):
            diffs.append({"item": item.id, "field": "occupancy", "reference": f["occupancy_type"],
                          "candidate": label})  # fmt: skip
    return {"reference": "app03-p002", "reference_status": ref.status, "values_compared": compared,
            "differences": diffs}  # fmt: skip


def _summary(model: str, pages: dict[int, dict[str, Any]], rows: list[dict[str, Any]],
             cells: list[dict[str, Any]], checks: dict[str, Any]) -> dict[str, Any]:  # fmt: skip
    order, dups, check = checks["order"], checks["duplicates"], checks["draft"]
    flags = Counter(f["code"] for r in rows for f in r["suspicious"])
    cell_flags = Counter(f["code"] for c in cells for f in c["suspicious"])
    return {
        "model": model, "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "pages": {str(k): v for k, v in pages.items()},
        "tables": len(pages), "ratings": [v["energy_rating"] for v in pages.values()],
        "occupancy_rows": len(rows), "numeric_cells": len(cells),
        "cells_parsed": sum(c["eui"] is not None for c in cells),
        "climate_classes": sorted({c["climate_class"] for c in cells if c["climate_class"]}),
        "missing_climate_columns": {k: v for k, v in a3.MISSING_CLIMATE.items()
                                    if k not in {c["climate_class"] for c in cells}},
        "eui_reads_agree": sum(c["eui_visual_read"] is not None and not c["suspicious"]
                               for c in cells),
        "cells_with_visual_read": sum(c["eui_visual_read"] is not None for c in cells),
        "occupancy_reads_agree": sum(r["occupancy_reads_agree"] for r in rows),
        "cross_rating_order": {k: v for k, v in order.items() if k != "differing_positions"},
        "near_duplicate_labels": dups,
        "suspicious_rows": sum(bool(r["suspicious"]) for r in rows),
        "suspicious_by_code": dict(flags), "cell_flags": dict(cell_flags),
        "draft_reference_differences": len(check.get("differences", [])),
    }  # fmt: skip


# --------------------------------------------------------------------------- review page

_CSS = """
:root{--bg:#f7f6f2;--fg:#1d1d1b;--muted:#66655f;--line:#d9d5ca;--card:#fff;--warn:#fff1c7}
@media (prefers-color-scheme:dark){:root{--bg:#191917;--fg:#ecebe6;--muted:#a3a29b;
--line:#3a3934;--card:#23231f;--warn:#4a3f17}}
body{background:var(--bg);color:var(--fg);margin:0;font:14px/1.5 Vazirmatn,Tahoma,sans-serif}
main{max-width:1200px;margin:auto;padding:16px}h1{font-size:18px}table{border-collapse:
collapse;width:100%;background:var(--card)}td,th{border:1px solid var(--line);padding:3px 6px;
text-align:right;vertical-align:middle}th{color:var(--muted);font-weight:normal}
tr.flag td{background:var(--warn)}img{height:26px;background:#fff}.muted{color:var(--muted);
font-size:12px}.ltr{direction:ltr;text-align:left;font-family:monospace}
"""


def review_html(rows: list[dict[str, Any]], crops: dict[str, bytes], order: dict[str, Any],
                summary: dict[str, Any]) -> str:  # fmt: skip
    def e(v: Any) -> str:
        return "—" if v is None else html.escape(str(v))

    by = {(r["energy_rating"], r["row_order"]): r for r in rows}
    n = max(order["rows_per_rating"].values(), default=0)
    body = []
    for i in range(1, n + 1):
        rs = [by.get((k, i)) for k in a3.RATINGS]
        flagged = any(r and r["suspicious"] for r in rs)
        d = by.get(("D", i))
        img = (f"<img alt='row {i}' src='data:image/png;base64,"
               f"{base64.b64encode(crops[d['row_id']]).decode()}'>" if d else "")  # fmt: skip
        reads = "".join(f"<td>{e(r and r['occupancy'])}<div class='muted'>"
                        f"{e(r and r['occupancy_second_read_raw'])}</div></td>"
                        for r in rs)  # fmt: skip
        flags = "<br>".join(e(f"{r['energy_rating']}: {f['code']}") for r in rs if r
                            for f in r["suspicious"])  # fmt: skip
        body.append(f"<tr{' class=flag' if flagged else ''}><td class='ltr'>{i}</td><td>{img}</td>"
                    f"{reads}<td>{flags}</td></tr>")  # fmt: skip
    head = "".join(f"<th>{k} (Gemini / Tesseract)</th>" for k in a3.RATINGS)
    return (
        "<!doctype html><html lang='fa' dir='rtl'><head><meta charset='utf-8'><meta "
        "name='viewport' content='width=device-width,initial-scale=1'><title>Appendix 3 "
        f"candidates</title><style>{_CSS}</style></head><body><main><h1>پیوست ۳ — "
        f"{summary['occupancy_rows']} occupancy rows, {summary['numeric_cells']} EUI cells "
        "(candidate, pending review)</h1><p class='muted'>Crop from table D. EUI values and "
        "climate labels come from the PDF text layer and are not shown here. Highlighted rows "
        f"have an occupancy-label flag in any table.</p><table><tr><th>#</th><th>source crop "
        f"(D)</th>{head}<th>flags</th></tr>{''.join(body)}</table></main></body></html>\n"
    )  # fmt: skip


def default_responses(model: str) -> Path:
    return DATA_DIR / "interim" / "gemini-phase2b" / model / "responses"
