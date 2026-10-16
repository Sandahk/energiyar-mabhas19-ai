"""Phase 2D outputs for appendix 2: raw layer, candidates, checks and a review page.

Everything is written under data/processed/app02/<model>/ (git-ignored). No external API is
called here (Gemini answers are read from the cache) and nothing is written to curated/.
"""

from __future__ import annotations

import base64
import html
import json
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
from mabhas19.tables import app02 as a2
from mabhas19.tables import app02_decisions as dec
from mabhas19.text.normalize import to_ascii_digits, to_persian_digits

DOC_ID = "m19-ed5-1404-app02"
PDF = SOURCES_DIR / "mabhas19" / "ed5-1404" / "appendices" / f"{DOC_ID}-v1.pdf"
TABLE_PAGES = (2, 3, 4)
PRINTED_PAGE_OFFSET = 1  # printed page = PDF page - 1 (p2 «۱», p4 «۳»)
OCR_LANGS = ("fas",)
CROP_DPI = 160


def request_for(pdf_page: int) -> pb.Request:
    page_id = f"app02-p{pdf_page:03d}"
    return pb.Request(f"{page_id}.app02", "app02", page_id, DOC_ID, pdf_page, pb.APP02_PROMPT,
                      True)  # fmt: skip


# --------------------------------------------------------------------------- text layer


def _cell_text(page: pymupdf.Page, rect: pymupdf.Rect) -> str:
    return str(page.get_text("text", clip=rect)).replace("\n", "")


def _glyph_lines(page: pymupdf.Page, rect: pymupdf.Rect) -> int:
    """Vector-drawn glyph lines inside a cell (the city names have no text layer)."""
    ys = sorted((d["rect"].y0 + d["rect"].y1) / 2 for d in page.get_drawings()
                if rect.contains(d["rect"]) and d["rect"].height < 15
                and d["rect"].width < 60)  # fmt: skip
    lines: list[float] = []
    for y in ys:
        if not lines or y - lines[-1] > 4:
            lines.append(y)
    return len(lines)


def text_rows(page: pymupdf.Page) -> tuple[list[a2.TextRow], dict[str, Any]]:
    """Data rows of the ruled table: (climate code, city, row number) from left to right."""
    tables = page.find_tables().tables
    if not tables:
        return [], {"table_found": False}
    table = max(tables, key=lambda t: t.row_count * t.col_count)
    out: list[a2.TextRow] = []
    skipped = 0
    for row in table.rows:
        rects = sorted((pymupdf.Rect(b) for b in row.cells if b is not None),
                       key=lambda r: r.x0)  # fmt: skip
        rects = [r for r in rects if r.width >= 8]
        if len(rects) != 3:
            skipped += 1
            continue
        code, number = _cell_text(page, rects[0]), _cell_text(page, rects[2])
        if not code.strip() and not number.strip():
            skipped += 1  # the column header (drawn text only)
            continue
        out.append(a2.TextRow(page.number + 1, len(out) + 1, number, code,
                              (rects[1].y0 + rects[1].y1) / 2, tuple(rects[1]),
                              _glyph_lines(page, rects[1])))  # fmt: skip
    return out, {"table_found": True, "ruled_rows": table.row_count, "skipped_rows": skipped}


# --------------------------------------------------------------------------- reads


def gemini_reads(responses: Path, pdf_page: int) -> tuple[dict[str, a2.VisualRead], dict[str, Any]]:
    req = request_for(pdf_page)
    path = responses / f"{req.id}.json"
    if not path.is_file():
        return {}, {"request_id": req.id, "status": "missing"}
    data = json.loads(path.read_text(encoding="utf-8"))
    parsed = pb.parse_json(data.get("text")) if data.get("status") == "ok" else None
    ok = pb.schema_ok("app02", parsed)
    rows = parsed["rows"] if ok and parsed else []
    reads: dict[str, a2.VisualRead] = {}
    duplicates = []
    for i, r in enumerate(rows):
        key = to_ascii_digits(str(r.get("row_number_raw") or "").strip())
        src = {"method": "gemini_whole_page", "model": data.get("model_version"),
               "request_id": req.id, "row_index": i,
               "received_at": data.get("received_at")}  # fmt: skip
        if key in reads:
            duplicates.append(key)
        reads[key] = a2.VisualRead(pb._s(r.get("city_raw")), pb._s(r.get("climate_class_raw")),
                                   pb._s(r.get("row_number_raw")), src)  # fmt: skip
    info = {"request_id": req.id, "status": data.get("status"), "json_schema_ok": ok,
            "rows_returned": len(rows), "duplicate_row_numbers": duplicates,
            "usage": data.get("usage"), "latency_s": data.get("latency_s"),
            "origin": "phase2d_call" if data.get("received_at") else "phase2b_cache"}  # fmt: skip
    return reads, info


def ocr_read(page: pymupdf.Page, row: a2.TextRow, engine: OcrEngine, tmp: Path) -> a2.VisualRead:
    r = pymupdf.Rect(row.city_bbox)
    img = render_clean(page, tmp / "city.png", clip=pymupdf.Rect(r.x0 + 1, r.y0 + 1, r.x1 - 1,
                                                                 r.y1 - 1))  # fmt: skip
    res = engine.recognize(img, OCR_LANGS, psm=7)
    text = " ".join(res.text.split()) or None
    conf = min((w.confidence for w in res.words), default=None)
    return a2.VisualRead(text, None, None, {"method": "tesseract_cell", "languages": OCR_LANGS,
                                            "min_confidence": conf})  # fmt: skip


# --------------------------------------------------------------------------- build


def build(model: str, responses: Path, out: Path, engine: OcrEngine | None) -> dict[str, Any]:
    (out / "raw").mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    pages: dict[int, dict[str, Any]] = {}
    crops: dict[str, bytes] = {}
    raw_rows: list[dict[str, Any]] = []
    with pymupdf.open(PDF) as pdf, tempfile.TemporaryDirectory() as td:
        for p in TABLE_PAGES:
            page = pdf[p - 1]
            rows, table_info = text_rows(page)
            reads, g_info = gemini_reads(responses, p)
            src = responses / f"{g_info['request_id']}.json"
            if src.is_file():
                shutil.copy2(src, out / "raw")
            raw_rows += [{"pdf_page": p, "row_order": r.row_order,
                          "row_number_raw": r.row_number_raw, "climate_code_raw": r.code_raw,
                          "city_bbox": r.city_bbox} for r in rows]  # fmt: skip
            printed = to_persian_digits(str(p - PRINTED_PAGE_OFFSET))
            text_numbers = {to_ascii_digits(r.row_number_raw.strip()) for r in rows}
            for row in rows:
                key = to_ascii_digits(row.row_number_raw.strip())
                ocr = ocr_read(page, row, engine, Path(td)) if engine else None
                rec = a2.build_record(row, printed, reads.get(key), ocr, DOC_ID)
                records.append(rec)
                cell = pymupdf.Rect(row.city_bbox)
                band = pymupdf.Rect(cell.x0 - 90, cell.y0 - 1, cell.x1 + 50, cell.y1 + 1)
                crops[rec["candidate_id"]] = page.get_pixmap(dpi=CROP_DPI, clip=band).tobytes("png")
            pages[p] = {**table_info, "text_layer_rows": len(rows),
                        "rows_with_city_glyphs": sum(r.city_glyph_lines > 0 for r in rows),
                        "gemini": g_info,
                        "gemini_rows_not_in_text_layer": sorted(set(reads) - text_numbers),
                        "printed_page": printed}  # fmt: skip
    duplicates = a2.flag_duplicates(records)
    for r in records:
        r["extraction_model"] = model
    out.mkdir(parents=True, exist_ok=True)
    _jsonl(out / "raw_rows.jsonl", raw_rows)
    _jsonl(out / "candidates_machine.jsonl", records)  # before review decisions
    records = dec.apply_decisions(records, dec.load_all())
    _jsonl(out / "candidates.jsonl", records)
    check = draft_reference_check(records)
    summary = _summary(model, pages, records, duplicates, check)
    for name, obj in (("summary.json", summary), ("draft_reference_check.json", check)):
        (out / name).write_text(json.dumps(obj, ensure_ascii=False, indent=1) + "\n",
                                encoding="utf-8", newline="\n")  # fmt: skip
    (out / "review.html").write_text(review_html(records, crops, summary), encoding="utf-8",
                                     newline="\n")  # fmt: skip
    return summary


def _jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def draft_reference_check(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Agreement with the page-2 draft reference (Claude transcription, NOT human-verified)."""
    refs = load_references(EVAL_DIR / "extraction" / "reference").get("app02-p002")
    if refs is None:
        return {"status": "no reference"}
    by_number = {r["row_number"]: r for r in records}
    rows = [i for i in refs.items if i.type == "table_row"]
    diffs = []
    for item in rows:
        f = item.expected_fields
        rec = by_number.get(f["row_number"])
        if rec is None:
            diffs.append({"item": item.id, "problem": "row missing"})
            continue
        if f["climate_class"] != rec["climate_code"]:
            diffs.append({"item": item.id, "field": "climate_code", "reference": f["climate_class"],
                          "candidate": rec["climate_code"]})  # fmt: skip
        if not a2.same_city(f["city"], rec["city"]):
            diffs.append({"item": item.id, "field": "city", "reference": f["city"],
                          "candidate": rec["city"]})  # fmt: skip
    return {"reference": "app02-p002", "reference_status": refs.status, "rows": len(rows),
            "differences": diffs}  # fmt: skip


def _summary(model: str, pages: dict[int, dict[str, Any]], records: list[dict[str, Any]],
             duplicates: list[list[str]], check: dict[str, Any]) -> dict[str, Any]:  # fmt: skip
    numbers = [r["row_number"] for r in records]
    flags = Counter(f["code"] for r in records for f in r["suspicious"])
    return {
        "model": model, "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "pages": {str(k): v for k, v in pages.items()},
        "rows": len(records),
        "row_numbers_contiguous": numbers == list(range(1, len(records) + 1)),
        "climate_codes": dict(sorted(Counter(r["climate_code"] for r in records).items(),
                                     key=lambda kv: str(kv[0]))),
        "climate_codes_valid": sum(r["climate_code_valid"] for r in records),
        "code_reads_agree": sum(not any(f["code"] == "code_reads_disagree" for f in r["suspicious"])
                                and r["climate_code_visual_read"] is not None for r in records),
        "city_reads_agree": sum(r["city_reads_agree"] for r in records),
        "rows_5c": [{k: r[k] for k in ("candidate_id", "row_number", "city_raw", "climate_code_raw",
                                       "pdf_page", "printed_page", "source_issue_refs")}
                    for r in records if r["climate_code"] == "5C"],
        "duplicate_cities": duplicates,
        "suspicious_rows": sum(bool(r["suspicious"]) for r in records),
        "unresolved_suspicious_rows": [r["row_number"] for r in records if r["suspicious"]
                                       and r["review"]["status"] != "reviewed"],
        "review": {
            "reviewed": sum(r["review"]["status"] == "reviewed" for r in records),
            "by_basis": dict(Counter(b for r in records for b in r["review"]["basis"])),
            "corroborated_not_reviewed": sum(r["review"]["status"] == "not_reviewed"
                                             and r["review"]["corroboration"] is not None
                                             for r in records),
            "fields_changed": [{"row_number": r["row_number"], "city_machine": r["city_machine"],
                                "city": r["city"]} for r in records if "city_machine" in r],
        },
        "suspicious_by_code": dict(flags),
        "draft_reference_differences": len(check.get("differences", [])),
    }  # fmt: skip


# --------------------------------------------------------------------------- review page

_CSS = """
:root{--bg:#f7f6f2;--fg:#1d1d1b;--muted:#66655f;--line:#d9d5ca;--card:#fff;--warn:#fff1c7}
@media (prefers-color-scheme:dark){:root{--bg:#191917;--fg:#ecebe6;--muted:#a3a29b;
--line:#3a3934;--card:#23231f;--warn:#4a3f17}}
body{background:var(--bg);color:var(--fg);margin:0;font:14px/1.5 Vazirmatn,Tahoma,sans-serif}
main{max-width:1100px;margin:auto;padding:16px}h1{font-size:18px}
table{border-collapse:collapse;width:100%;background:var(--card)}td,th{border:1px solid
var(--line);padding:3px 6px;text-align:right;vertical-align:middle}th{color:var(--muted);
font-weight:normal}tr.flag td{background:var(--warn)}img{height:28px;background:#fff}
.ltr{direction:ltr;text-align:left;font-family:monospace}.muted{color:var(--muted);font-size:12px}
@media (max-width:700px){img{height:22px}td,th{font-size:12px}}
"""


def _open(r: dict[str, Any]) -> bool:
    return bool(r["suspicious"]) and r["review"]["status"] != "reviewed"


def review_html(records: list[dict[str, Any]], crops: dict[str, bytes],
                summary: dict[str, Any]) -> str:  # fmt: skip
    def e(v: Any) -> str:
        return "—" if v is None else html.escape(str(v))

    rows = "".join(
        f"<tr{' class=flag' if _open(r) else ''}><td class='ltr'>{e(r['candidate_id'][6:])}"
        f"</td><td>{e(r['row_number'])}</td><td><img alt='row {e(r['row_number'])}' "
        f"src='data:image/png;base64,{base64.b64encode(crops[r['candidate_id']]).decode()}'></td>"
        f"<td>{e(r['city'])}<div class='muted'>{e(r['review']['status'])} "
        f"{e(', '.join(r['review']['basis']))}</div></td><td>{e(r['city_second_read_raw'])}</td>"
        f"<td class='ltr'>{e(r['climate_code_raw'])} → {e(r['climate_code'])}</td>"
        f"<td>{e(', '.join(r['source_issue_refs']))}</td>"
        f"<td>{'<br>'.join(e(f['code'] + ': ' + f['detail']) for f in r['suspicious'])}</td></tr>"
        for r in records)  # fmt: skip
    return (
        "<!doctype html><html lang='fa' dir='rtl'><head><meta charset='utf-8'><meta "
        "name='viewport' content='width=device-width,initial-scale=1'><title>Appendix 2 "
        f"candidates</title><style>{_CSS}</style></head><body><main><h1>پیوست ۲ — "
        f"{summary['rows']} ردیف (نامزد، در انتظار بازبینی)</h1><p class='muted'>Code and row "
        "number: PDF text layer. City: Gemini whole-page read; second read: Tesseract (flags "
        "only). Highlighted rows are suspicious. Nothing here is verified.</p><table><tr><th>id"
        "</th><th>ردیف</th><th>source crop</th><th>city (Gemini)</th><th>second read</th><th>"
        f"code</th><th>source issue</th><th>flags</th></tr>{rows}</table></main></body></html>\n"
    )  # fmt: skip


def default_responses(model: str) -> Path:
    return DATA_DIR / "interim" / "gemini-phase2b" / model / "responses"
