"""Review sheet (PDF) for the suspicious appendix 2 city names (offline, no decisions).

One block per flagged row: a large crop of the printed row (row number, city, climate code),
the Gemini reading, the Tesseract second read, the draft-reference reading where one exists,
and empty option boxes. Nothing is preselected, recorded or changed.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import pymupdf

from mabhas19.benchmark.references import load_references
from mabhas19.core.paths import EVAL_DIR
from mabhas19.tables.app02_build import PDF

OPTIONS = ("ACCEPT Gemini reading", "CORRECT (write the printed name)", "UNCLEAR")
ROWS_PER_PAGE = 6
CROP_DPI = 220
GROUPS = (
    ("city_reads_disagree", "Gemini and Tesseract disagree"),
    ("city_uncorroborated", "Gemini reading not corroborated (Tesseract returned nothing)"),
)
_CSS = ("body{font-family:sans-serif;font-size:9px;direction:ltr}h1{font-size:13px;margin:0}"
        "h2{font-size:10.5px;margin:0}.fa{direction:rtl;text-align:right;font-size:12px}"
        "table{border-collapse:collapse}td{padding:1px 4px;vertical-align:top}"
        "td.k{color:#555;white-space:nowrap}.o{margin-right:12px}")  # fmt: skip


def suspicious_rows(candidates: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    out = []
    for code, title in GROUPS:
        out += [(title, c) for c in candidates if any(f["code"] == code for f in c["suspicious"])]
    return out


def draft_readings() -> dict[int, str]:
    ref = load_references(EVAL_DIR / "extraction" / "reference").get("app02-p002")
    if ref is None:
        return {}
    return {i.expected_fields["row_number"]: i.expected_fields["city"]
            for i in ref.items if i.type == "table_row"}  # fmt: skip


def _block(c: dict[str, Any], group: str, draft: dict[int, str]) -> str:
    def e(v: Any) -> str:
        return "—" if v in (None, "") else html.escape(str(v))

    rows = [("Gemini (candidate)", f"<span class='fa'>{e(c['city_raw'])}</span>"),
            ("Tesseract", f"<span class='fa'>{e(c['city_second_read_raw'])}</span>"),
            ("Draft reference (not verified)",
             f"<span class='fa'>{e(draft.get(c['row_number']))}</span>"),
            ("Climate code (text layer)", f"{e(c['climate_code'])}"),
            ("Flag", e("; ".join(f["detail"] for f in c["suspicious"])))]  # fmt: skip
    table = "".join(f"<tr><td class='k'>{k}</td><td>{v}</td></tr>" for k, v in rows)
    boxes = "".join(f"<span class='o'>☐ {html.escape(o)}</span>" for o in OPTIONS)
    return (f"<h2>Row {c['row_number']} · {c['candidate_id']} · PDF p{c['pdf_page']} · printed "
            f"«{e(c['printed_page'])}» · {html.escape(group)}</h2><table>{table}</table>"
            f"<div>{boxes}</div><div>Correct name: ____________________</div>")  # fmt: skip


def write_pdf(cand_dir: Path, out: Path) -> dict[str, Any]:
    lines = (cand_dir / "candidates.jsonl").read_text(encoding="utf-8").splitlines()
    candidates = [json.loads(x) for x in lines if x.strip()]
    items = suspicious_rows(candidates)
    draft = draft_readings()
    fonts = Path("C:/Windows/Fonts")
    archive = pymupdf.Archive(str(fonts)) if fonts.is_dir() else None
    doc = pymupdf.open()
    with pymupdf.open(PDF) as src:
        for start in range(0, len(items), ROWS_PER_PAGE):
            page = doc.new_page(width=595, height=842)
            page.insert_htmlbox(pymupdf.Rect(30, 14, 565, 34),
                                f"<h1>Appendix 2 – suspicious city names ({len(items)} rows) – "
                                f"sheet {start // ROWS_PER_PAGE + 1}</h1>", css=_CSS,
                                archive=archive)  # fmt: skip
            for k, (group, c) in enumerate(items[start : start + ROWS_PER_PAGE]):
                top = 40 + k * 132
                cell = pymupdf.Rect(c["city_bbox"])
                clip = pymupdf.Rect(cell.x0 - 90, cell.y0 - 1, cell.x1 + 50, cell.y1 + 1)
                pix = src[c["pdf_page"] - 1].get_pixmap(dpi=CROP_DPI, clip=clip)
                w = 535.0
                h = w * clip.height / clip.width
                page.insert_image(pymupdf.Rect(30, top, 30 + w, top + h), stream=pix.tobytes("png"))
                page.insert_htmlbox(pymupdf.Rect(30, top + h + 3, 565, top + 128),
                                    _block(c, group, draft), css=_CSS, archive=archive,
                                    scale_low=0)  # fmt: skip
                page.draw_line((30, top + 129), (565, top + 129), color=(0.8, 0.8, 0.8))
    doc.subset_fonts()
    doc.save(out, garbage=4, deflate=True)
    pages = int(doc.page_count)
    doc.close()
    return {"rows": len(items), "pages": pages, "pdf": str(out)}
