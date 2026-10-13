"""Review package for the appendix 7 rows nobody has reviewed yet (offline, no decisions).

Selects the reviewed candidate layer's rows with ``review.status ==
not_reviewed`` outside the human-verified benchmark pages, groups them into compact blocks
(same page, category and subgroup; at most ``MAX_ROWS`` rows per block) and writes a
standalone HTML file (images embedded, answers only exported as JSON, never stored) and a
PDF review sheet. Each row states whether the secondary workbook independently confirms its
numbers, which decides the question asked. Nothing is changed, recorded or promoted.
"""

from __future__ import annotations

import base64
import html
import itertools
import json
from pathlib import Path
from typing import Any

import pymupdf

from mabhas19.tables import app07_owner_review as ow
from mabhas19.tables.app07_build import PDF, TABLE_X
from mabhas19.tables.app07_review import load_candidates

MAX_ROWS = 10
EXCLUDED_PAGES = (3, 6, 13)  # covered by the human-verified references
CONFIRMED_OPTIONS = ("ACCEPT hierarchy/name", "CORRECT", "UNCLEAR")
UNCONFIRMED_OPTIONS = ("ACCEPT hierarchy + numbers", "CORRECT", "UNCLEAR")


def workbook_confirmations(crosscheck: dict[str, Any]) -> dict[str, list[int]]:
    """Candidate id -> workbook rows whose density and λ match it exactly."""
    out: dict[str, list[int]] = {}
    for row in crosscheck.get("rows", []):
        for m in row["matching_candidates"]:
            out.setdefault(m["id"], []).append(row["wb_row"])
    return out


def remaining_rows(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [c for c in candidates if c["review"]["status"] == "not_reviewed"
            and c["pdf_page"] not in EXCLUDED_PAGES]  # fmt: skip


def blocks(rows: list[dict[str, Any]]) -> list[ow.Block]:
    out: list[ow.Block] = []
    key = lambda c: (c["pdf_page"], c["category"], c["subgroup"])  # noqa: E731
    for (page, cat, sub), group in itertools.groupby(rows, key=key):
        ids = [c["candidate_id"] for c in group]
        chunks = [ids[i : i + MAX_ROWS] for i in range(0, len(ids), MAX_ROWS)]
        for chunk in chunks:
            span = chunk[0][6:] if len(chunk) == 1 else f"{chunk[0][6:]}–{chunk[-1][6:]}"
            context = f"{cat or '—'} · {sub or 'بدون زیرگروه'}"
            title = f"PDF p{page} ({span}, {len(chunk)} rows)"
            out.append(ow.Block(f"R{len(out) + 1:02d}", "R", title,
                                [(cid, "target") for cid in chunk], [], [context]))  # fmt: skip
    return out


def row_info(c: dict[str, Any], wb: dict[str, list[int]]) -> dict[str, Any]:
    confirmed = c["candidate_id"] in wb
    return {
        "id": c["candidate_id"], "category": c["category"], "subgroup": c["subgroup"],
        "material": c["material"], "material_inherited": c["material_source"] == "inherited",
        "condition": c["condition"], "display_name": c["display_name"],
        "density": c["raw_density_text"], "conductivity": c["raw_conductivity_text"],
        "workbook_rows": wb.get(c["candidate_id"], []), "confirmed": confirmed,
        "options": CONFIRMED_OPTIONS if confirmed else UNCONFIRMED_OPTIONS,
    }  # fmt: skip


def _e(v: Any) -> str:
    return "—" if v is None or v == "" else html.escape(str(v))


def _wb(r: dict[str, Any]) -> str:
    if r["confirmed"]:
        return "✔ yes (row " + ", ".join(str(x) for x in r["workbook_rows"]) + ")"
    return "✘ no"


_CSS = """
:root{--bg:#f7f6f2;--fg:#1d1d1b;--muted:#66655f;--line:#d9d5ca;--card:#fff;--acc:#2456a6;
--ok:#2d7a46;--no:#b5651d}
@media (prefers-color-scheme:dark){:root{--bg:#191917;--fg:#ecebe6;--muted:#a3a29b;
--line:#3a3934;--card:#23231f;--acc:#7aa7ef;--ok:#6cc08a;--no:#e0a060}}
*{box-sizing:border-box}body{background:var(--bg);color:var(--fg);margin:0;
font:14px/1.55 Vazirmatn,Tahoma,"Segoe UI",sans-serif}
header{position:sticky;top:0;z-index:2;background:var(--card);border-bottom:1px solid
var(--line);padding:10px 16px;display:flex;flex-wrap:wrap;gap:10px;align-items:center}
header h1{font-size:17px;margin:0;flex:1 1 auto}main{max-width:1150px;margin:auto;padding:16px}
section{background:var(--card);border:1px solid var(--line);border-radius:8px;margin:16px 0;
padding:12px}h2{font-size:15px;margin:0 0 6px;direction:ltr;text-align:left}
img{width:100%;max-width:820px;display:block;margin:6px auto;border:1px solid var(--line);
background:#fff}table{border-collapse:collapse;width:100%;font-size:13px}
td,th{border:1px solid var(--line);padding:3px 6px;text-align:right;vertical-align:top}
th{color:var(--muted);font-weight:normal;white-space:nowrap}.id{direction:ltr;white-space:nowrap;
font-family:monospace}.inh{color:var(--acc)}.ok{color:var(--ok)}.no{color:var(--no)}
.dn{font-weight:600}label{display:block;white-space:nowrap;cursor:pointer}
textarea{width:100%;min-height:30px;font:inherit;background:var(--bg);color:var(--fg);
border:1px solid var(--line)}button{font:inherit;padding:6px 12px;border-radius:6px;border:0;
background:var(--acc);color:#fff;cursor:pointer}#out{width:100%;min-height:120px;display:none;
direction:ltr}.muted{color:var(--muted);font-size:13px}#prog{color:var(--muted)}
@media (max-width:700px){main{padding:8px}td,th{font-size:12px}}
"""

_JS = """
const rows=[...document.querySelectorAll('tr[data-row]')];
function prog(){document.getElementById('prog').textContent=rows.filter(r=>
r.querySelector('input:checked')).length+' / '+rows.length+' rows answered';}
document.addEventListener('change',prog);prog();
function exportJson(){const out={exported_at:new Date().toISOString(),answers:rows.map(r=>{
const c=r.querySelector('input:checked');return {row:r.dataset.row,answer:c?c.value:null,
note:r.querySelector('textarea').value};})};const t=document.getElementById('out');
t.style.display='block';t.value=JSON.stringify(out,null,1);t.select();}
"""

_HEAD = (
    "<tr><th>row</th><th>category</th><th>subgroup</th><th>material</th><th>condition</th>"
    "<th>display_name</th><th>density</th><th>λ</th><th>workbook</th><th>decision</th></tr>"
)


def _row_html(r: dict[str, Any]) -> str:
    mat = _e(r["material"]) + (" <span class='inh'>(inherited)</span>" if r["material_inherited"]
                                else "")  # fmt: skip
    opts = "".join(f"<label><input type='radio' name='{r['id']}' value='{html.escape(o)}'> "
                   f"{html.escape(o)}</label>" for o in r["options"])  # fmt: skip
    cls = "ok" if r["confirmed"] else "no"
    return (f"<tr data-row='{r['id']}'><td class='id'>{r['id'][6:]}</td><td>{_e(r['category'])}"
            f"</td><td>{_e(r['subgroup'])}</td><td>{mat}</td><td>{_e(r['condition'])}</td>"
            f"<td class='dn'>{_e(r['display_name'])}</td><td>{_e(r['density'])}</td>"
            f"<td>{_e(r['conductivity'])}</td><td class='{cls}'>{_wb(r)}</td><td>{opts}"
            "<textarea placeholder='correction / note'></textarea></td></tr>")  # fmt: skip


def page_html(items: list[tuple[ow.Block, list[ow.Crop], list[dict[str, Any]]]]) -> str:
    total = sum(len(rows) for _, _, rows in items)
    confirmed = sum(r["confirmed"] for _, _, rows in items for r in rows)
    parts = [
        "<!doctype html><html lang='fa' dir='rtl'><head><meta charset='utf-8'><meta name='viewport'"
        " content='width=device-width,initial-scale=1'><title>Appendix 7 remaining rows</title>"
        f"<style>{_CSS}</style></head><body><header><h1>پیوست ۷ — {total} ردیف بازبینی‌نشده</h1>"
        "<span id='prog'></span><button onclick='exportJson()'>Export answers (JSON)</button>"
        "<textarea id='out' readonly></textarea></header><main><p class='muted'>"
        f"{len(items)} blocks · {confirmed} rows numerically confirmed by the workbook "
        f"(question: {' / '.join(CONFIRMED_OPTIONS)}) · {total - confirmed} not confirmed "
        f"(question: {' / '.join(UNCONFIRMED_OPTIONS)}). Red boxes in the crops mark the rows "
        "under review; unboxed rows were reviewed earlier. Nothing is preselected or saved: "
        "export the answers as JSON and send them.</p>"
    ]  # fmt: skip
    for b, crops, rows in items:
        imgs = "".join(f"<img alt='{b.id} crop, PDF page {c.pdf_page}' src='data:image/png;base64,"
                       f"{base64.b64encode(c.png).decode()}'>" for c in crops)  # fmt: skip
        body = "".join(_row_html(r) for r in rows)
        ctx = html.escape(b.notes[0]) if b.notes else ""
        parts.append(f"<section id='{b.id}'><h2>{b.id} — {html.escape(b.title)}</h2>"
                     f"<div class='dn'>{ctx}</div>{imgs}<table>{_HEAD}{body}</table>"
                     "</section>")  # fmt: skip
    parts.append(f"</main><script>{_JS}</script></body></html>\n")
    return "".join(parts)


_SHEET_CSS = ("body{font-family:sans-serif;font-size:7.5px;direction:ltr}"
              "table{border-collapse:collapse;width:100%}td,th{border:0.5px solid #999;"
              "padding:1px 2px;text-align:right;direction:rtl}th{color:#555}"
              "h1{font-size:11px;margin:0}.o{white-space:nowrap}")  # fmt: skip


def sheets_pdf(items: list[tuple[ow.Block, list[ow.Crop], list[dict[str, Any]]]],
               out: Path) -> int:  # fmt: skip
    doc = pymupdf.open()
    fonts = Path("C:/Windows/Fonts")
    archive = pymupdf.Archive(str(fonts)) if fonts.is_dir() else None
    for b, crops, rows in items:
        page = doc.new_page(width=842, height=595)  # A4 landscape: wide table
        left, width, top = 24.0, 794.0, 30.0
        page.insert_htmlbox(pymupdf.Rect(left, 10, left + 320, 28),
                            f"<h1>{b.id} — {html.escape(b.title)}</h1>", css=_SHEET_CSS,
                            archive=archive)  # fmt: skip
        page.insert_htmlbox(pymupdf.Rect(left + 330, 10, left + width, 28),
                            f"<div dir='rtl' style='text-align:right;font-size:10px'>"
                            f"{html.escape(b.notes[0]) if b.notes else ''}</div>", css=_SHEET_CSS,
                            archive=archive)  # fmt: skip
        budget = 300.0 / max(len(crops), 1)
        for c in crops:
            ratio = (c.y1 - c.y0) / (TABLE_X[1] - TABLE_X[0] + 10)
            h = min(520 * ratio, budget)
            w = h / ratio
            page.insert_image(pymupdf.Rect(left + width - w, top, left + width, top + h),
                              stream=c.png)  # fmt: skip
            top += h + 4
        body = "".join(
            f"<tr><td>{r['id'][6:]}</td><td>{_e(r['category'])}</td><td>{_e(r['subgroup'])}</td>"
            f"<td>{_e(r['material'])}{' (inherited)' if r['material_inherited'] else ''}</td>"
            f"<td>{_e(r['condition'])}</td><td><b>{_e(r['display_name'])}</b></td>"
            f"<td>{_e(r['density'])}</td><td>{_e(r['conductivity'])}</td><td>{_wb(r)}</td><td>"
            + " ".join(f"<span class='o'>☐ {html.escape(o)}</span>" for o in r["options"])
            + "</td></tr>" for r in rows)  # fmt: skip
        page.insert_htmlbox(pymupdf.Rect(left, top + 4, left + width, 585),
                            f"<table>{_HEAD}{body}</table>", css=_SHEET_CSS, archive=archive,
                            scale_low=0)  # fmt: skip
    doc.subset_fonts()
    doc.save(out, garbage=4, deflate=True)
    count = int(doc.page_count)
    doc.close()
    return count


def write_remaining_review(cand_dir: Path, html_out: Path, pdf_out: Path) -> dict[str, Any]:
    candidates = load_candidates(cand_dir / "candidates.jsonl")
    wb_path = cand_dir / "workbook_crosscheck.json"
    wb = workbook_confirmations(json.loads(wb_path.read_text(encoding="utf-8"))
                                if wb_path.is_file() else {})  # fmt: skip
    idx = {c["candidate_id"]: c for c in candidates}
    rows = remaining_rows(candidates)
    items = []
    with pymupdf.open(PDF) as doc:
        for b in blocks(rows):
            crops = ow.render_block(doc, b, candidates)
            items.append((b, crops, [row_info(idx[cid], wb) for cid, _ in b.rows]))
    html_out.parent.mkdir(parents=True, exist_ok=True)
    html_out.write_text(page_html(items), encoding="utf-8", newline="\n")
    sheets = sheets_pdf(items, pdf_out)
    return {"rows": len(rows), "workbook_confirmed": sum(r["confirmed"] for _, _, rs in items
                                                         for r in rs),
            "blocks": len(items), "pdf_pages": sheets,
            "html": str(html_out), "pdf": str(pdf_out)}  # fmt: skip
