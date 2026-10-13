"""Project-owner decision package for appendix 7 (Phase 2C review).

Three sections, each a list of review blocks with a large source crop and plain questions:

* A - the rows that use the leaf-label inheritance rule;
* B - material-continuation runs outside the human-verified pages (consecutive rows that
  inherit the same material form one block);
* C - the suspicious rows, consolidated by visual issue.

Outputs (under ``<candidates dir>/owner_review/``): a standalone ``review.html`` (images
embedded, no dependencies; answers stay in the page until exported as JSON), a
``review_sheets.pdf`` with one numbered sheet per block and the same sheets as PNG files.

Nothing is decided or recorded here: no option is preselected, the candidate data is not
modified, no external API is called and nothing is written to curated/.
"""

from __future__ import annotations

import base64
import html
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pymupdf

from mabhas19.tables.app07_build import PDF, TABLE_X
from mabhas19.tables.app07_review import inherited_fields, load_candidates

CROP_DPI = 150
SHEET_DPI = 110
TABLE_TOP = 66.0  # below the running page header
BAND = 8.0  # half height of a highlighted row band, points
VERIFIED_PAGES = (3, 6, 13)

ROLE_COLOURS = {  # stroke colour, label
    "target": ((0.85, 0.15, 0.1), "target row"),
    "inherits": ((0.1, 0.35, 0.85), "row under review"),
    "source": ((0.1, 0.55, 0.2), "row that prints the label"),
    "context": ((0.55, 0.55, 0.55), "context / boundary"),
}

INHERIT_OPTIONS = ("YES — inherit previous row label", "NO — do not inherit", "UNCLEAR")
APPLY_OPTIONS = ("YES", "NO", "PARTIAL", "UNCLEAR")


@dataclass
class Question:
    id: str
    text: str
    options: tuple[str, ...]


@dataclass
class Block:
    id: str
    section: str
    title: str
    rows: list[tuple[str, str]]  # (candidate id, role)
    questions: list[Question]
    notes: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- blocks


def _index(candidates: list[dict[str, Any]]) -> dict[str, int]:
    return {c["candidate_id"]: i for i, c in enumerate(candidates)}


def leaf_blocks(candidates: list[dict[str, Any]]) -> list[Block]:
    idx = _index(candidates)
    out = []
    for c in candidates:
        leaf = [i for i in inherited_fields(c) if i["rule"] == "leaf_label"]
        if not leaf:
            continue
        i = idx[c["candidate_id"]]
        start = i - 1  # first row of the same material on this page: its headings are above it
        while (start > 0 and candidates[start - 1]["pdf_page"] == c["pdf_page"]
               and candidates[start - 1]["material"] == c["material"]):  # fmt: skip
            start -= 1
        rows = [(x["candidate_id"], "context") for x in candidates[start : i - 1]]
        rows.append((candidates[i - 1]["candidate_id"], "source"))
        rows.append((c["candidate_id"], "target"))
        if i + 1 < len(candidates) and candidates[i + 1]["pdf_page"] == c["pdf_page"]:
            rows.append((candidates[i + 1]["candidate_id"], "context"))
        out.append(Block(
            f"A-{c['candidate_id'][6:]}", "A", f"Leaf-label inheritance: {c['candidate_id']}",
            rows, [Question("inherit", "Does the target row clearly continue the previous row's "
                            "label?", INHERIT_OPTIONS)],
            [f"Currently inherited: «{leaf[0]['value']}» from {leaf[0]['from']}."],
        ))  # fmt: skip
    return out


def material_blocks(candidates: list[dict[str, Any]]) -> list[Block]:
    """Runs of rows inheriting one material, outside the verified pages."""
    idx = _index(candidates)
    runs: dict[str, list[str]] = {}
    for c in candidates:
        if c["material_source"] == "inherited" and c["pdf_page"] not in VERIFIED_PAGES:
            runs.setdefault(c["material_inherited_from"], []).append(c["candidate_id"])
    out = []
    for n, (origin, ids) in enumerate(runs.items(), 1):
        rows = [(origin, "source")] + [(x, "inherits") for x in ids]
        after = idx[ids[-1]] + 1
        if after < len(candidates):
            rows.append((candidates[after]["candidate_id"], "context"))
        mat = candidates[idx[origin]]["material"]
        span = ids[0][6:] if len(ids) == 1 else f"{ids[0][6:]}–{ids[-1][6:]}"
        out.append(Block(
            f"B{n:02d}", "B", f"Material continuation: {span} ({len(ids)} row(s))", rows,
            [Question("applies", f"Does this material heading «{mat}» apply to all "
                      "highlighted rows?", APPLY_OPTIONS)],
            [f"Printed on {origin}; inherited by {', '.join(ids)}. If PARTIAL, list the "
             "rows it does not apply to in the note."],
        ))  # fmt: skip
    return out


_YNU = ("YES", "NO", "UNCLEAR")


def suspicious_blocks() -> list[Block]:
    """The 36 suspicious rows consolidated by visual issue (questions: structure only)."""

    def rows(page: int, first: int, last: int, role: str = "target") -> list[tuple[str, str]]:
        return [(f"app07-p{page:03d}-r{r:02d}", role) for r in range(first, last + 1)]

    return [
        Block(
            "C01",
            "C",
            "Category heading copied into the material field",
            rows(3, 1, 1),
            [
                Question(
                    "drop",
                    "Row r01 has no material of its own; the category heading "
                    "«۱. اندود و ملات آهکی یا سیمانی» was copied into the material field. "
                    "Keep material empty (as in the verified reference)?",
                    _YNU,
                )
            ],
        ),
        Block(
            "C02",
            "C",
            "Footnote ¹ on the density/cement sub-headings",
            rows(4, 9, 15),
            [
                Question(
                    "footnote",
                    "In «... عیار سیمان بیش از ۳۰۰¹:» and «... کمتر از ۲۵۰¹:» "
                    "is the trailing 1 a footnote marker (not part of 300 / 250)?",
                    ("YES — footnote 1", "NO — part of the text", "UNCLEAR"),
                ),
                Question(
                    "placement",
                    "Placement: material «با رس منبسط یا شیست منبسط», "
                    "condition = density/cement sub-heading + the row's own item. Correct?",
                    ("CORRECT", "WRONG", "UNCLEAR"),
                ),
            ],
            ["r15's own label is reviewed in section A."],
        ),
        Block(
            "C03",
            "C",
            "Three printed headings joined in one field",
            rows(5, 1, 1),
            [
                Question(
                    "split",
                    "Split read as subgroup «بتن با سنگدانه بسیار سبک», material "
                    "«متشکل از پرلیت یا ورمیکولیت (از ۳ تا ۶ میلیمتر) اجرای درجا», condition "
                    "«نسبت: ۱ به ۳». Correct?",
                    ("CORRECT", "WRONG", "UNCLEAR"),
                )
            ],
        ),
        Block(
            "C04",
            "C",
            "Category 3 heading: lost «ی» before the footnote",
            rows(6, 15, 21),
            [
                Question(
                    "category",
                    "Gemini read «...گرماشکن ۲»; the verified reference has "
                    "«۳. بتونهٔ درزها، مواد آب‌بندی و گرماشکنی» with category footnote 2. Use "
                    "the verified text for r15–r21?",
                    _YNU,
                )
            ],
        ),
        Block(
            "C05",
            "C",
            "Wood: Latin/Persian order and heading structure",
            rows(8, 1, 6),
            [
                Question(
                    "order",
                    "The condition lines mix «kg/m3» with Persian numbers. Proposed "
                    "logical readings (Claude, from the crop): r01 «چگالی نرمال متوسط ۶۵۰ تا "
                    "۸۰۰ kg/m3 و رطوبت ۱۵ درصد», r02 «... ۵۰۰ تا ۶۵۰ kg/m3 ...», r03 «چگالی "
                    "طبیعی بیش از ۷۰۰ kg/m3», r04 «چگالی طبیعی ۵۰۰ تا ۶۰۰ kg/m3», r05/r06 "
                    "«چگالی طبیعی ۳۵۰ تا ۵۰۰ kg/m3». Accept?",
                    ("ACCEPT", "REJECT — see note", "UNCLEAR"),
                ),
                Question(
                    "structure",
                    "r01–r02 sit under the heading «بلوط ... درختان میوه‌دار:», "
                    "r03 under «چوب درخت‌های صمغی بسیار سنگین (برگ ریز):»; r04–r06 are "
                    "separate materials («کاج نقره‌ای...», «کاج یا صنوبر...», «تبریزی...») "
                    "with their own density line. Correct?",
                    ("CORRECT", "WRONG — see note", "UNCLEAR"),
                ),
            ],
        ),
        Block(
            "C06",
            "C",
            "«سنگ مصنوعی»: inside the last subgroup or not?",
            [*rows(10, 19, 19, "context"), *rows(10, 20, 20)],
            [
                Question(
                    "boundary",
                    "«سنگ مصنوعی» is undashed after the dashed items of "
                    "«سنگ‌های چخماق (فلینت) و سنگ‌های ساینده و پومیس». Where does it belong?",
                    ("In that subgroup (current)", "Category-level, no subgroup", "UNCLEAR"),
                )
            ],
        ),
        Block(
            "C07",
            "C",
            "«الیافی» / «سلولزی» under category 10",
            rows(11, 5, 8),
            [
                Question(
                    "kind",
                    "Gemini gave «الیافی» and «سلولزی» as subgroups; they print "
                    "undashed without ':' beside two value rows each. Treat them as:",
                    ("Materials (current)", "Subgroups", "UNCLEAR"),
                ),
                Question("continue", "Do r06 and r08 continue «الیافی» and «سلولزی»?", _YNU),
            ],
        ),
        Block(
            "C08",
            "C",
            "Material label wrapped over three lines",
            rows(11, 9, 10),
            [
                Question(
                    "wrap",
                    "Is «پلی‌استایرن برش خورده در بلوک‌های قالبی تولیدشده به‌صورت "
                    "منقطع، یا قالب‌گیری شدهٔ ممتد بدون پوستهٔ سطحی» one label for r09?",
                    _YNU,
                )
            ],
            ["r10–r16 inheriting it are reviewed in section B."],
        ),
        Block(
            "C09",
            "C",
            "Extruded polystyrene: headings, HCFC/CFC",
            [*rows(11, 16, 16, "context"), *rows(11, 17, 21)],
            [
                Question(
                    "subgroup",
                    "«- پلی‌استایرن اکسترود شده با حفره‌های پر از:» is currently "
                    "a material with no subgroup. It is:",
                    (
                        "Material, no subgroup (current)",
                        "Under «پلی‌استایرن منبسط»",
                        "A subgroup itself",
                        "UNCLEAR",
                    ),
                ),
                Question(
                    "r18",
                    "Does the sub-heading «هوا یا گاز کربنیک» (above r17) also apply "
                    "to r18 «ضخامت بیش از ۶۰ میلی‌متر»?",
                    _YNU,
                ),
                Question(
                    "r19",
                    "Is r19's label «- HCFC» (a sibling of «هوا یا گاز کربنیک»), with "
                    "«- CFC:» a separate heading printed below it?",
                    _YNU,
                ),
                Question(
                    "r20r21",
                    "Do r20 «بدون پوسته سطحی» and r21 «با پوسته سطحی» sit under «CFC:»?",
                    _YNU,
                ),
            ],
        ),
        Block(
            "C10",
            "C",
            "Line merged into the previous row",
            rows(13, 18, 19),
            [
                Question(
                    "reference",
                    "The verified reference has r18 material «گچ قطعات "
                    "پیش‌ساختهٔ گچی با روکش مقوایی» and r19 subgroup «گچ با سبک‌دانه یا با الیاف "
                    "معدنی». Apply the verified structure?",
                    _YNU,
                )
            ],
        ),
    ]


def all_blocks(candidates: list[dict[str, Any]]) -> list[Block]:
    return leaf_blocks(candidates) + material_blocks(candidates) + suspicious_blocks()


# --------------------------------------------------------------------------- crops


@dataclass
class Crop:
    pdf_page: int
    y0: float
    y1: float
    png: bytes


def _y(c: dict[str, Any]) -> float:
    return float(c["row_y"])


def crop_spans(block: Block, candidates: list[dict[str, Any]]) -> list[tuple[int, float, float]]:
    """Per page: from just below the value row before the block (so the headings printed
    above the first row are visible) to just below the last row of the block."""
    idx = _index(candidates)
    pages: dict[int, list[int]] = {}
    for cid, _ in block.rows:
        pages.setdefault(candidates[idx[cid]]["pdf_page"], []).append(idx[cid])
    spans = []
    for page, ids in pages.items():
        first, last = min(ids), max(ids)
        before = candidates[first - 1] if first > 0 else None
        top = _y(before) + 6 if before and before["pdf_page"] == page else TABLE_TOP
        nxt = candidates[last + 1] if last + 1 < len(candidates) else None
        bottom = _y(candidates[last]) + 14
        if nxt is None or nxt["pdf_page"] != page:
            bottom += 16  # include the table's bottom edge
        spans.append((page, max(top, TABLE_TOP), bottom))
    return spans


def render_block(doc: pymupdf.Document, block: Block, candidates: list[dict[str, Any]]
                 ) -> list[Crop]:  # fmt: skip
    """Render each crop and draw the row bands on a scratch page (the source is untouched)."""
    idx = _index(candidates)
    out = []
    for page_no, y0, y1 in crop_spans(block, candidates):
        clip = pymupdf.Rect(TABLE_X[0] - 4, y0, TABLE_X[1] + 6, y1)
        pix = doc[page_no - 1].get_pixmap(dpi=CROP_DPI, clip=clip)
        scratch = pymupdf.open()
        sheet = scratch.new_page(width=clip.width, height=clip.height)
        sheet.insert_image(sheet.rect, pixmap=pix)
        for cid, role in block.rows:
            c = candidates[idx[cid]]
            if c["pdf_page"] != page_no:
                continue
            colour, _ = ROLE_COLOURS[role]
            y = _y(c) - y0
            rect = pymupdf.Rect(1, y - BAND, clip.width - 1, y + BAND)
            sheet.draw_rect(rect, color=colour, width=1.6 if role != "context" else 0.8,
                            dashes="[3] 0" if role == "context" else None)  # fmt: skip
            sheet.insert_text((3, y + 3), cid[-3:], fontsize=7, color=colour)
        png = sheet.get_pixmap(dpi=CROP_DPI).tobytes("png")
        scratch.close()
        out.append(Crop(page_no, y0, y1, png))
    return out


# --------------------------------------------------------------------------- row facts


def row_facts(c: dict[str, Any], role: str) -> dict[str, Any]:
    inherited = [f"{i['field']} ← «{i['value']}» ({i['from']})" for i in inherited_fields(c)
                 if i["rule"] != "cross_page_category"]  # fmt: skip
    visible = " / ".join(v for v in (c["raw_material"], c["raw_condition"]) if v)
    return {
        "id": c["candidate_id"], "role": ROLE_COLOURS[role][1],
        "printed_page": c["printed_page"], "visible_label": visible or None,
        "subgroup": c["subgroup"], "material": c["material"], "condition": c["condition"],
        "inherited": inherited, "density": c["raw_density_text"],
        "conductivity": c["raw_conductivity_text"],
    }  # fmt: skip


# --------------------------------------------------------------------------- html

_CSS = """
:root{--bg:#f7f6f2;--fg:#1d1d1b;--muted:#66655f;--line:#d9d5ca;--card:#fff;--acc:#2456a6}
@media (prefers-color-scheme:dark){:root{--bg:#191917;--fg:#ecebe6;--muted:#a3a29b;
--line:#3a3934;--card:#23231f;--acc:#7aa7ef}}
*{box-sizing:border-box}body{background:var(--bg);color:var(--fg);margin:0;
font:15px/1.6 Vazirmatn,Tahoma,"Segoe UI",sans-serif}
header{position:sticky;top:0;z-index:2;background:var(--card);border-bottom:1px solid var(--line);
padding:10px 16px;display:flex;flex-wrap:wrap;gap:10px;align-items:center}
header h1{font-size:17px;margin:0;flex:1 1 auto}main{max-width:1000px;margin:auto;padding:16px}
h2{font-size:18px;margin:28px 0 8px}section.block{background:var(--card);border:1px solid
var(--line);border-radius:8px;margin:16px 0;padding:14px}
.bh{display:flex;gap:10px;align-items:baseline}.num{font:600 13px monospace;color:var(--acc)}
.bh h3{font-size:16px;margin:0}img{width:100%;border:1px solid var(--line);background:#fff;
margin:6px 0}.legend span{margin-left:12px;font-size:12px}.sw{display:inline-block;width:12px;
height:8px;border:2px solid;margin-left:4px;vertical-align:middle}
table{border-collapse:collapse;width:100%;font-size:13px;margin:6px 0}td,th{border:1px solid
var(--line);padding:3px 6px;text-align:right;vertical-align:top}th{color:var(--muted);
font-weight:normal}.ltr{direction:ltr;text-align:left;white-space:nowrap}
fieldset{border:1px solid var(--line);border-radius:6px;margin:10px 0;padding:8px 10px}
legend{font-weight:600;padding:0 4px}label.opt{display:block;padding:3px 0;cursor:pointer}
textarea{width:100%;min-height:42px;font:inherit;background:var(--bg);color:var(--fg);
border:1px solid var(--line);border-radius:4px}.note{color:var(--muted);font-size:13px}
button{font:inherit;padding:6px 12px;border-radius:6px;border:1px solid var(--acc);
background:var(--acc);color:#fff;cursor:pointer}#out{width:100%;min-height:120px;display:none;
direction:ltr}#prog{color:var(--muted);font-size:13px}
@media (max-width:640px){main{padding:10px}td,th{font-size:12px}}
"""

_JS = """
const qs=[...document.querySelectorAll('fieldset[data-q]')];
function prog(){const n=qs.filter(f=>f.querySelector('input:checked')).length;
document.getElementById('prog').textContent=n+' / '+qs.length+' questions answered';}
document.addEventListener('change',prog);prog();
function exportJson(){const out={exported_at:new Date().toISOString(),decisions:[]};
qs.forEach(f=>{const c=f.querySelector('input:checked');out.decisions.push({block:f.dataset.block,
question:f.dataset.q,answer:c?c.value:null,
note:(f.querySelector('textarea')||{}).value||''});});
const t=document.getElementById('out');t.style.display='block';
t.value=JSON.stringify(out,null,1);t.select();}
"""


def _e(v: Any) -> str:
    return (
        "—"
        if v is None or v == []
        else html.escape(", ".join(v) if isinstance(v, list) else str(v))
    )


def _rows_table(facts: list[dict[str, Any]]) -> str:
    head = ("<tr><th>row</th><th>role</th><th>label as transcribed</th><th>subgroup</th>"
            "<th>material</th><th>condition</th><th>inherited now</th><th>density</th>"
            "<th>λ</th></tr>")  # fmt: skip
    body = "".join(
        f"<tr><td class='ltr'>{_e(f['id'][6:])}</td><td>{_e(f['role'])}</td>"
        f"<td>{_e(f['visible_label'])}</td><td>{_e(f['subgroup'])}</td>"
        f"<td>{_e(f['material'])}</td><td>{_e(f['condition'])}</td>"
        f"<td>{_e(f['inherited'])}</td><td>{_e(f['density'])}</td>"
        f"<td>{_e(f['conductivity'])}</td></tr>" for f in facts)  # fmt: skip
    return f"<table>{head}{body}</table>"


def _block_html(n: int, b: Block, crops: list[Crop], facts: list[dict[str, Any]]) -> str:
    imgs = "".join(
        f"<img alt='PDF page {c.pdf_page}, y {c.y0:.0f}–{c.y1:.0f}' src='data:image/png;base64,"
        f"{base64.b64encode(c.png).decode()}'>" for c in crops)  # fmt: skip
    pages = ", ".join(f"PDF p{c.pdf_page}" for c in crops)
    qs = "".join(
        f"<fieldset data-block='{b.id}' data-q='{q.id}'><legend>{html.escape(q.text)}</legend>"
        + "".join(f"<label class='opt'><input type='radio' name='{b.id}-{q.id}' "
                  f"value='{html.escape(o)}'> {html.escape(o)}</label>" for o in q.options)
        + "<textarea placeholder='note (optional)'></textarea></fieldset>"
        for q in b.questions)  # fmt: skip
    notes = "".join(f"<p class='note'>{html.escape(x)}</p>" for x in b.notes)
    return (f"<section class='block' id='{b.id}'><div class='bh'><span class='num'>#{n:02d} "
            f"{b.id}</span><h3>{html.escape(b.title)}</h3></div><div class='note'>{pages}</div>"
            f"{imgs}{_rows_table(facts)}{notes}{qs}</section>")  # fmt: skip


def review_html(items: list[tuple[Block, list[Crop], list[dict[str, Any]]]]) -> str:
    titles = {"A": "A. Leaf-label inheritance (2 rows)",
              "B": "B. Material continuation outside the verified pages",
              "C": "C. Suspicious rows, grouped by visual issue"}  # fmt: skip
    legend = "".join(f"<span><i class='sw' style='border-color:rgb"
                     f"{tuple(int(v * 255) for v in col)}'></i>{lab}</span>"
                     for col, lab in ROLE_COLOURS.values())  # fmt: skip
    parts = [
        "<!doctype html><html lang='fa' dir='rtl'><head><meta charset='utf-8'><meta "
        "name='viewport' content='width=device-width,initial-scale=1'><title>Appendix 7 "
        f"owner review</title><style>{_CSS}</style></head><body><header><h1>پیوست ۷ — "
        "بازبینی مالک پروژه</h1><span id='prog'></span><button onclick='exportJson()'>Export "
        "answers (JSON)</button><textarea id='out' readonly></textarea></header><main>"
        "<p class='note'>No answer is preselected and nothing is saved by this page: answers "
        "exist only until it is closed. Use “Export answers” and send the JSON to record them. "
        "Numbers are shown for orientation only and are not under review.</p>"
        f"<p class='legend'>{legend}</p>"
    ]  # fmt: skip
    section = ""
    for n, (b, crops, facts) in enumerate(items, 1):
        if b.section != section:
            section = b.section
            parts.append(f"<h2>{titles[section]}</h2>")
        parts.append(_block_html(n, b, crops, facts))
    parts.append(f"</main><script>{_JS}</script></body></html>\n")
    return "".join(parts)


# --------------------------------------------------------------------------- pdf sheets

_FONT_DIR = Path("C:/Windows/Fonts")
_SHEET_CSS = ("body{font-family:sans-serif;font-size:8.5px;direction:ltr}"
              "table{border-collapse:collapse;width:100%}td,th{border:0.5px solid #999;"
              "padding:1px 3px;text-align:right;direction:rtl}th{color:#555}"
              "h1{font-size:12px;margin:0}"
              ".q{margin-top:5px;font-size:9px}.o{margin-right:10px}")  # fmt: skip


def sheets_pdf(items: list[tuple[Block, list[Crop], list[dict[str, Any]]]], out: Path) -> int:
    """One A4 sheet per block: crop(s) on top, row facts and empty option boxes below."""
    doc = pymupdf.open()
    archive = pymupdf.Archive(str(_FONT_DIR)) if _FONT_DIR.is_dir() else None
    for n, (b, crops, facts) in enumerate(items, 1):
        page = doc.new_page(width=595, height=842)
        top, left, width = 30.0, 30.0, 535.0
        page.insert_htmlbox(pymupdf.Rect(left, 14, left + width, 30),
                            f"<h1>#{n:02d} {b.id} — {html.escape(b.title)}</h1>", css=_SHEET_CSS,
                            archive=archive)  # fmt: skip
        budget = 480.0 / max(len(crops), 1)
        for c in crops:
            ratio = (c.y1 - c.y0) / (TABLE_X[1] - TABLE_X[0] + 10)
            h = min(width * ratio, budget)
            w = h / ratio
            page.insert_image(pymupdf.Rect(left + width - w, top, left + width, top + h),
                              stream=c.png)  # fmt: skip
            top += h + 6
        text = _rows_table(facts) + "".join(f"<p>{html.escape(x)}</p>" for x in b.notes)
        text += "".join(
            f"<div class='q'><b>{html.escape(q.text)}</b><br>"
            + "".join(f"<span class='o'>☐ {html.escape(o)}</span>" for o in q.options)
            + "</div>" for q in b.questions)  # fmt: skip
        page.insert_htmlbox(pymupdf.Rect(left, top + 4, left + width, 828), text,
                            css=_SHEET_CSS, archive=archive, scale_low=0)  # fmt: skip
    doc.subset_fonts()
    doc.save(out, garbage=4, deflate=True)
    count = int(doc.page_count)
    doc.close()
    return count


# --------------------------------------------------------------------------- build


def write_owner_review(cand_dir: Path) -> dict[str, Any]:
    out = cand_dir / "owner_review"
    (out / "sheets").mkdir(parents=True, exist_ok=True)
    candidates = load_candidates(cand_dir / "candidates.jsonl")
    idx = _index(candidates)
    blocks = all_blocks(candidates)
    items = []
    with pymupdf.open(PDF) as doc:
        for b in blocks:
            crops = render_block(doc, b, candidates)
            facts = [row_facts(candidates[idx[cid]], role) for cid, role in b.rows]
            items.append((b, crops, facts))
    (out / "review.html").write_text(review_html(items), encoding="utf-8", newline="\n")
    pages = sheets_pdf(items, out / "review_sheets.pdf")
    with pymupdf.open(out / "review_sheets.pdf") as sheets:
        for n, page in enumerate(sheets, 1):
            page.get_pixmap(dpi=SHEET_DPI).save(out / "sheets" / f"{n:02d}-{blocks[n - 1].id}.png")
    flagged = {c["candidate_id"] for c in candidates if c["suspicious"]}
    in_c = {cid for b in blocks if b.section == "C" for cid, role in b.rows if role == "target"}
    summary = {
        "blocks": {s: sum(b.section == s for b in blocks) for s in "ABC"},
        "questions": sum(len(b.questions) for b in blocks),
        "material_continuation_rows": sum(role == "inherits" for b in blocks if b.section == "B"
                                          for _, role in b.rows),
        "suspicious_rows_covered": f"{len(in_c & flagged)}/{len(flagged)}",
        "suspicious_rows_missing": sorted(flagged - in_c),
        "sheets": pages, "output": str(out),
    }  # fmt: skip
    (out / "blocks.json").write_text(json.dumps(
        [{"id": b.id, "section": b.section, "title": b.title, "rows": b.rows,
          "questions": [q.__dict__ for q in b.questions], "notes": b.notes, "answer": None}
         for b in blocks], ensure_ascii=False, indent=1) + "\n", encoding="utf-8")  # fmt: skip
    return summary
