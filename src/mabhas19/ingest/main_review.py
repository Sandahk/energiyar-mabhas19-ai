"""Targeted review package for the main-text candidate corpus (offline, nothing changed).

Groups the unresolved / low-confidence cases by issue, deduplicates repetitive patterns,
attaches a source crop, raw text, repaired text, kind, section path and flags to every item,
and states a proposed rule or classification. Writes a standalone HTML file, a PDF review
sheet and a JSON decision summary. The corpus itself is never modified.
"""

from __future__ import annotations

import base64
import html
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pymupdf

from mabhas19.ingest.repair.glyphs import page_clusters  # noqa: F401  (documentation link)

CROP_DPI = 150
_MARKER = re.compile(r"^\s*([آ-ی]{1,3})\(")
_LIST_HEADING = re.compile(r"^\s*[آ-ی]{1,3}[()]")  # before or after M1
_ALPHABET_MARKERS = frozenset(
    [
        "الف",
        "ب",
        "پ",
        "ت",
        "ث",
        "ج",
        "چ",
        "ح",
        "خ",
        "د",
        "ذ",
        "ر",
        "ز",
        "ژ",
        "س",
        "ش",
        "ص",
        "ض",
        "ط",
        "ظ",
        "ع",
        "غ",
        "ف",
        "ق",
        "ک",
        "گ",
        "ل",
        "م",
        "ن",
        "و",
        "ه",
        "ی",
    ]
)
_MIRRORED_LATIN = re.compile(r"\)([A-Za-z0-9][^()]{0,40})\(")
_CAPTION_NUMBER = re.compile(r"جدول\s+([۰-۹\s-]+?)\s*:")
UNMAPPED_GLYPHS = {95: "پ", 112: "چ"}  # BNazanin gids drawn with ink but mapped to U+0020


@dataclass
class Item:
    item_id: str
    block: dict[str, Any] | None
    title: str
    proposal: str
    decision: str  # "rule" (one decision covers the group) or "per_block"
    options: tuple[str, ...]
    extra: dict[str, Any] = field(default_factory=dict)
    crop: pymupdf.Rect | None = None
    pdf_page: int | None = None


@dataclass
class Group:
    key: str
    title: str
    summary: str
    items: list[Item]
    decisions: int
    rule: str | None = None


# --------------------------------------------------------------------------- analyses


def _path(b: dict[str, Any]) -> str:
    return " › ".join(p["number"] or p["title"] for p in b.get("section_path") or []) or "—"


def mirrored_markers(blocks: list[dict[str, Any]]) -> Group:
    flagged = [b for b in blocks if "list_marker_paren_mirrored" in b["flags"]]
    real = [b for b in flagged if _MARKER.match(b["text"])[1] in _ALPHABET_MARKERS]  # type: ignore[index]
    false = [b for b in flagged if b not in real]
    seen: set[str] = set()
    sample = []
    for b in real:  # one per section, spread over the document
        sec = _path(b)
        if sec not in seen:
            seen.add(sec)
            sample.append(b)
    sample = sample[:: max(1, len(sample) // 14)][:14]
    items = [
        Item(
            b["block_id"],
            b,
            "list marker",
            "Rule M1: «X(» at the start of a list item "
            "becomes «X)» (marker letter X from الف…ی); raw text unchanged.",
            "rule",
            ("ACCEPT RULE M1", "KEEP «X(» AS PRINTED", "UNCLEAR"),
        )
        for b in sample
    ]
    items += [
        Item(
            b["block_id"],
            b,
            "not a list marker",
            "Exception: a wrapped parenthetical "
            "(«…بی‌سیم( و…», «…شود(. …») misread as a list item: reclassify as paragraph "
            "continuation; M1 must not apply.",
            "rule",
            ("CONFIRM EXCEPTION", "IT IS A LIST ITEM", "UNCLEAR"),
        )
        for b in false
    ]
    markers = Counter(_MARKER.match(b["text"])[1] for b in real)  # type: ignore[index]
    return Group(
        "M1",
        "1. Mirrored list-marker parenthesis «الف(»",
        f"{len(flagged)} blocks on {len({b['pdf_page'] for b in flagged})} pages: "
        f"{len(real)} real markers ({len(markers)} distinct letters) shown by a sample of "
        f"{len(sample)} across sections, and {len(false)} false positives. Raw text holds "
        "the visual order «)فلا»; repair_v1 mirrors the bracket, giving «الف(».",
        items,
        decisions=2,
        rule="M1: ^([آ-ی]{1,3})\\( -> \\1) for alphabet marker letters only",
    )


def unmapped_glyphs(pdf: pymupdf.Document, blocks: list[dict[str, Any]]) -> Group:
    """Glyph-evidence cases: wide, inked BNazanin glyphs whose ToUnicode entry is a space."""
    occ: list[dict[str, Any]] = []
    for page in pdf:
        for span in page.get_texttrace():
            if span.get("type") == 3 or not span["font"].endswith("BNazanin"):
                continue
            for u, gid, _origin, bbox in span["chars"]:
                if gid in UNMAPPED_GLYPHS and u == 0x20 and bbox[2] - bbox[0] > 5:
                    r = pymupdf.Rect(bbox)
                    blk = next(
                        (
                            b
                            for b in blocks
                            if b["pdf_page"] == page.number + 1
                            and b.get("text")
                            and pymupdf.Rect(b["bbox"]).intersects(r)
                        ),
                        None,
                    )
                    occ.append({"gid": gid, "pdf_page": page.number + 1, "rect": r, "block": blk})

    def pattern(o: dict[str, Any]) -> str:
        t = (o["block"] or {}).get("text") or ""
        if o["gid"] == 112:
            return "final چ in a word (هیچ…)"
        if "list_marker_letter_missing" in (o["block"] or {}).get("flags", []):
            return "list marker «پ)»"
        return "«جدول پ۳-…» appendix-table reference" if "جدول" in t else "پ inside a word (چاپ…)"

    groups: dict[str, list[dict[str, Any]]] = {}
    for o in occ:
        groups.setdefault(pattern(o), []).append(o)
    items = []
    for name, os_ in groups.items():
        for o in os_[:2]:
            r = o["rect"]
            items.append(
                Item(
                    f"{o['block']['block_id'] if o['block'] else 'p' + str(o['pdf_page'])}"
                    f"-g{o['gid']}",
                    o["block"],
                    f"{name} ({len(os_)} occurrences)",
                    f"Rule G{1 if o['gid'] == 95 else 2}: BNazanin glyph {o['gid']} (inked, "
                    f"{r.width:.1f} pt wide, mapped to U+0020) is «{UNMAPPED_GLYPHS[o['gid']]}».",
                    "rule",
                    ("ACCEPT RULE", "REJECT", "UNCLEAR"),
                    {"gid": o["gid"], "occurrences_in_pattern": len(os_)},
                    crop=pymupdf.Rect(r.x0 - 60, r.y0 - 2, r.x1 + 60, r.y1 + 2),
                    pdf_page=o["pdf_page"],
                )
            )
    by_gid = Counter(o["gid"] for o in occ)
    return Group(
        "G",
        "2. Letters missing from the text layer («پ», «چ» mapped to a space)",
        f"{len(occ)} occurrences: glyph 95 «پ» ×{by_gid[95]} (incl. all 39 "
        f"list_marker_letter_missing blocks), glyph 112 «چ» ×{by_gid[112]}. Every occurrence "
        "renders ink and is 6-8 pt wide (a real space glyph is narrow). Patterns: "
        + ", ".join(f"{k} ×{len(v)}" for k, v in groups.items())
        + ".",
        items,
        decisions=2,
        rule="G1: gid 95 -> پ; G2: gid 112 -> چ (BNazanin, ink "
        "evidence, like the Phase 1 kashida/space-glyph rules)",
    )


def per_block(
    blocks: list[dict[str, Any]], flag: str, spec: tuple[str, str, str], options: tuple[str, ...]
) -> Group:
    """``spec`` is (group key, title, proposal)."""
    key, title, proposal = spec
    xs = [b for b in blocks if flag in b["flags"]]
    items = [Item(b["block_id"], b, flag, proposal, "per_block", options) for b in xs]
    return Group(key, title, f"{len(xs)} blocks.", items, decisions=len(xs))


def bracket_group(blocks: list[dict[str, Any]]) -> Group:
    xs = [b for b in blocks if "bracket_mirroring_suspect" in b["flags"]]
    items = []
    for b in xs:
        m = _MIRRORED_LATIN.search(b["text"])
        fix = f"«){m[1]}(» -> «({m[1]})»" if m else "?"
        items.append(
            Item(
                b["block_id"],
                b,
                "Latin run in mirrored brackets",
                f"Rule B1: {fix}",
                "rule",
                ("ACCEPT RULE B1", "REJECT", "UNCLEAR"),
            )
        )
    return Group(
        "B1",
        "4. Mirrored brackets around Latin runs",
        f"{len(xs)} blocks; 137 other blocks have correctly ordered brackets around Latin text.",
        items,
        decisions=1,
        rule="B1: \\)(Latin run)\\( -> (\\1) inside right-to-left text",
    )


def caption_numbers(blocks: list[dict[str, Any]]) -> Group:
    items = []
    for b in blocks:
        if b["kind"] != "table_caption":
            continue
        m = _CAPTION_NUMBER.search(b["text"])
        if not m:
            continue
        parts = [p for p in re.split(r"\s*-\s*", m[1].strip()) if p]
        if parts and parts[0] != "۱۹":
            proposal = (
                "-".join(reversed(parts))
                if parts[-1] == "۱۹"
                else "-".join([parts[1], parts[2], parts[0]])
                if len(parts) == 3
                else "?"
            )
            items.append(
                Item(
                    b["block_id"],
                    b,
                    f"caption number «{m[1].strip()}»",
                    f"Rule C1: read as «{proposal}» (section order, matching the table "
                    "sequence 19-2-1, 19-3-1…19-5-7).",
                    "rule",
                    ("ACCEPT RULE C1", "REJECT", "UNCLEAR"),
                )
            )
    return Group(
        "C1",
        "5. Table-caption numbers out of order",
        f"{len(items)} of 11 captions.",
        items,
        decisions=1 if items else 0,
        rule="C1: caption number parts reordered to start with ۱۹",
    )


BOLD_CLASSES = (
    "english_gloss",
    "appendix_list",
    "front_matter_heading",
    "list_heading",
    "signature",
    "toc_continuation",
    "caption_continuation",
    "emphasis",
)
_FRONT_HEADINGS = ("پیشگفتار", "مقدمه", "فهرست مطالب", "ابلاغیه", "هیأت", "عنوان صفحه")


def _latin_share(t: str) -> float:
    letters = [c for c in t if c.isalpha()]
    return sum(c.isascii() for c in letters) / len(letters) if letters else 0.0


def bold_class(b: dict[str, Any], prev: dict[str, Any] | None) -> tuple[str, bool]:
    """(proposed class, needs a human decision); the first matching rule wins."""
    t = b["text"] or ""
    after_caption = (
        prev is not None and prev["kind"] == "table_caption" and prev["pdf_page"] == b["pdf_page"]
    )
    rules: list[tuple[bool, str, bool]] = [
        (_latin_share(t) >= 0.6, "english_gloss", False),
        (b["pdf_page"] == 146 or t.startswith("پیوست"), "appendix_list", False),
        (b["page_role"] == "toc" and t.startswith(_FRONT_HEADINGS), "front_matter_heading", False),
        (b["page_role"] == "toc", "toc_continuation", False),
        (bool(_LIST_HEADING.match(t)), "list_heading", False),
        (after_caption, "caption_continuation", False),
        (t.startswith(_FRONT_HEADINGS) or b["font_size"] >= 14, "front_matter_heading", False),
        (len(t.split()) <= 4, "signature", True),
    ]
    return next(((cls, ask) for hit, cls, ask in rules if hit), ("emphasis", True))


def bold_group(blocks: list[dict[str, Any]]) -> Group:
    items = []
    for i, b in enumerate(blocks):
        if "unnumbered_bold_line" not in b["flags"]:
            continue
        prev = next((blocks[j] for j in range(i - 1, -1, -1) if blocks[j].get("text")), None)
        cls, ask = bold_class(b, prev)
        items.append(
            Item(
                b["block_id"],
                b,
                f"proposed: {cls}",
                f"Classify as {cls}.",
                "per_block" if ask else "rule",
                ("ACCEPT CLASS", "OTHER CLASS: ______", "UNCLEAR"),
                {"class": cls},
            )
        )
    asks = sum(i.decision == "per_block" for i in items)
    counts = Counter(i.extra["class"] for i in items)
    return Group(
        "U",
        "8. Unnumbered bold lines",
        f"{len(items)} lines: "
        + ", ".join(f"{k} ×{v}" for k, v in counts.items())
        + f". {asks} need a per-line decision; the rest follow from position and form.",
        items,
        decisions=asks + 1,
    )


def table_checks(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for t in (b for b in blocks if b["kind"] == "table"):
        tr = pymupdf.Rect(t["bbox"])
        caps = [
            b for b in blocks if b["pdf_page"] == t["pdf_page"] and b["kind"] == "table_caption"
        ]
        leaked = [
            b["block_id"]
            for b in blocks
            if b["pdf_page"] == t["pdf_page"]
            and b.get("text")
            and b["kind"] not in ("table_caption", "running_header", "page_number")
            and tr.contains(pymupdf.Rect(b["bbox"]))
        ]
        out.append(
            {
                "pdf_page": t["pdf_page"],
                "table": t["block_id"],
                "size": f"{t['rows']}x{t['cols']}",
                "captions": [c["text"].split("\n")[0] for c in caps],
                "leaked_prose_blocks": leaked,
                "status": "table_pending_structured_extraction",
            }
        )
    return out


def build_groups(pdf: pymupdf.Document, blocks: list[dict[str, Any]]) -> list[Group]:
    text_blocks = [b for b in blocks if b.get("text") is not None]
    ocr = next(b for b in blocks if b["kind"] == "ocr_text")
    formulas = [b for b in blocks if b["kind"] == "formula"]
    union = pymupdf.Rect(formulas[0]["bbox"])
    for f in formulas[1:]:
        union |= pymupdf.Rect(f["bbox"])
    return [
        mirrored_markers(text_blocks),
        unmapped_glyphs(pdf, text_blocks),
        per_block(
            text_blocks,
            "spacing_suspect",
            (
                "S",
                "3. Split words (spacing_suspect)",
                "Join the split word fragments shown in the crop (per block); letter-spaced "
                "display text (p2 bibliographic heading) may be intentional.",
            ),
            ("JOIN AS PROPOSED", "KEEP AS IS", "CORRECT TO: ______"),
        ),
        bracket_group(text_blocks),
        caption_numbers(text_blocks),
        per_block(
            text_blocks,
            "missing_spaces_suspect",
            (
                "N",
                "6. Missing spaces (single case)",
                "False positive: the long token is the TOC leader dots; the real defect is the "
                "split «اجر ا» -> «اجرا».",
            ),
            ("ACCEPT", "CORRECT TO: ______", "UNCLEAR"),
        ),
        Group(
            "O",
            "7. Page 5 OCR (scanned notification letter)",
            "The only scanned page: an "
            "administrative notification (ابلاغیه) of the 5th edition, not regulation text.",
            [
                Item(
                    ocr["block_id"],
                    ocr,
                    "OCR of the scan",
                    "Choose how to keep it: keep as "
                    "low-confidence OCR, re-read with Gemini, or exclude from the regulation "
                    "corpus (keep only as provenance).",
                    "per_block",
                    ("KEEP LOW-CONFIDENCE OCR", "RE-READ WITH GEMINI", "EXCLUDE FROM CORPUS"),
                )
            ],
            decisions=1,
        ),
        Group(
            "F",
            "9. Formula (page 73)",
            "Three formula blocks are the parts of one fraction.",
            [
                Item(
                    "main-p073-formula",
                    formulas[0],
                    "three blocks: " + " | ".join(repr(f["text"]) for f in formulas),
                    "Represent as one formula: R = E_actual / E_ideal (check the crop).",
                    "per_block",
                    ("ACCEPT", "CORRECT TO: ______", "UNCLEAR"),
                    crop=pymupdf.Rect(union.x0 - 20, union.y0 - 10, union.x1 + 20, union.y1 + 10),
                    pdf_page=formulas[0]["pdf_page"],
                )
            ],
            decisions=1,
        ),
        bold_group(text_blocks),
    ]


# --------------------------------------------------------------------------- rendering


def _crop(pdf: pymupdf.Document, item: Item) -> bytes:
    pno = item.pdf_page or item.block["pdf_page"]  # type: ignore[index]
    page = pdf[pno - 1]
    if item.crop is not None:
        rect = pymupdf.Rect(item.crop)
    else:
        bb = pymupdf.Rect(item.block["bbox"])  # type: ignore[index]
        rect = pymupdf.Rect(bb.x0 - 6, bb.y0 - 4, bb.x1 + 6, bb.y1 + 4)
    rect &= page.rect
    if rect.height > 260:  # long blocks: show the top, where the issue usually is
        rect = pymupdf.Rect(rect.x0, rect.y0, rect.x1, rect.y0 + 260)
    return bytes(page.get_pixmap(dpi=CROP_DPI, clip=rect).tobytes("png"))


LTR_FIELDS = frozenset({"PDF page / printed", "block", "section path", "flags"})


def _facts(item: Item) -> list[tuple[str, str]]:
    b = item.block or {}
    return [
        ("PDF page / printed", f"{item.pdf_page or b.get('pdf_page')} / {b.get('printed_page')}"),
        ("block", f"{b.get('block_id')} · {b.get('kind')}"),
        ("section path", _path(b)),
        ("flags", ", ".join(b.get("flags", []))),
        ("raw text", (b.get("text_raw") or "")[:400]),
        ("repaired text", (b.get("text") or "")[:400]),
        ("proposal", item.proposal),
    ]


_CSS = """
:root{--bg:#f7f6f2;--fg:#1d1d1b;--muted:#66655f;--line:#d9d5ca;--card:#fff;--acc:#2456a6}
@media (prefers-color-scheme:dark){:root{--bg:#191917;--fg:#ecebe6;--muted:#a3a29b;
--line:#3a3934;--card:#23231f;--acc:#7aa7ef}}
body{background:var(--bg);color:var(--fg);margin:0;font:14px/1.55 Vazirmatn,Tahoma,sans-serif}
main{max-width:1050px;margin:auto;padding:16px}h2{font-size:17px;margin-top:28px}
.sum{color:var(--muted);font-size:13px}section{background:var(--card);border:1px solid
var(--line);border-radius:6px;margin:10px 0;padding:10px}img{max-width:100%;background:#fff;
border:1px solid var(--line)}table{border-collapse:collapse;width:100%;font-size:13px}
td{border:1px solid var(--line);padding:3px 6px;vertical-align:top}td.k{color:var(--muted);
white-space:nowrap;width:120px}pre{white-space:pre-wrap;margin:0;font:inherit}
.rule{direction:ltr;text-align:left;font-family:monospace;font-size:12px;color:var(--acc)}
label{margin-left:14px}
"""


def _cards(g: Group) -> list[Item]:
    """Items shown as full cards; the bold lines classified by rule are listed compactly."""
    return [i for i in g.items if g.key != "U" or i.decision == "per_block"]


def _compact(g: Group) -> list[Item]:
    return [i for i in g.items if g.key == "U" and i.decision == "rule"]


def review_html(
    pdf: pymupdf.Document,
    groups: list[Group],
    tables: list[dict[str, Any]],
    summary: dict[str, Any],
) -> str:
    parts = [
        f"<!doctype html><html lang='fa' dir='rtl'><head><meta charset='utf-8'><meta "
        f"name='viewport' content='width=device-width,initial-scale=1'><title>Main corpus "
        f"review</title><style>{_CSS}</style></head><body><main><h1>مبحث ۱۹ — بازبینی "
        f"هدفمند متن اصلی</h1><p class='sum'>{summary['human_decisions']} human decisions "
        f"after deduplication ({summary['rule_decisions']} rule decisions, "
        f"{summary['per_block_decisions']} per-block). Nothing is preselected or saved; the "
        "corpus is unchanged.</p>"
    ]
    for g in groups:
        parts.append(f"<h2>{html.escape(g.title)}</h2><p class='sum'>{html.escape(g.summary)}</p>")
        if g.rule:
            parts.append(f"<p class='rule'>{html.escape(g.rule)}</p>")
        if _compact(g):
            listing = "".join(
                f"<tr><td dir='ltr'>{i.block['block_id']}</td><td>{html.escape(i.extra['class'])}"  # type: ignore[index]
                f"</td><td>{html.escape((i.block or {}).get('text') or '')[:90]}</td></tr>"
                for i in _compact(g)
            )
            parts.append(
                "<p class='sum'>Classified by position and form (one confirmation):"
                f"</p><table>{listing}</table>"
            )
        for it in _cards(g):
            img = base64.b64encode(_crop(pdf, it)).decode()
            rows = "".join(
                f"<tr><td class='k'>{k}</td><td{' dir=ltr' if k in LTR_FIELDS else ''}>"
                f"<pre>{html.escape(str(v))}</pre></td></tr>"
                for k, v in _facts(it)
            )
            opts = "".join(
                f"<label><input type='radio' name='{it.item_id}'> {html.escape(o)}</label>"
                for o in it.options
            )
            parts.append(
                f"<section><b>{html.escape(it.title)}</b> <span class='sum'>"
                f"({it.decision})</span><br><img alt='{it.item_id}' src='data:image/png;"
                f"base64,{img}'><table>{rows}</table><div>{opts}</div></section>"
            )
    rows = "".join(
        f"<tr><td>{t['pdf_page']}</td><td>{t['table']} ({t['size']})</td><td>"
        f"{html.escape(' | '.join(t['captions']))}</td><td>{len(t['leaked_prose_blocks'])}"
        "</td></tr>"
        for t in tables
    )
    parts.append(
        "<h2>10. Ruled tables (no cell review; pending structured extraction)</h2>"
        f"<table><tr><td>PDF page</td><td>table</td><td>caption</td><td>prose blocks "
        f"inside the table</td></tr>{rows}</table></main></body></html>\n"
    )
    return "".join(parts)


def review_pdf(
    pdf: pymupdf.Document, groups: list[Group], tables: list[dict[str, Any]], out: Path
) -> int:
    css = (
        "body{font-family:sans-serif;font-size:7.5px;direction:ltr}td{border:0.4px solid #aaa;"
        "padding:1px 3px;vertical-align:top}table{border-collapse:collapse;width:100%}"
        "h1{font-size:12px;margin:0}.fa{direction:rtl;text-align:right;font-size:8.5px}"
        ".o{margin-right:10px}"
    )
    fonts = Path("C:/Windows/Fonts")
    archive = pymupdf.Archive(str(fonts)) if fonts.is_dir() else None
    doc = pymupdf.open()
    for g in groups:
        page, y = None, 0.0
        if _compact(g):
            page = doc.new_page(width=595, height=842)
            listing = "".join(
                f"<tr><td>{i.block['block_id']}</td><td>{i.extra['class']}</td>"  # type: ignore[index]
                f"<td class='fa'>{html.escape((i.block or {}).get('text') or '')[:80]}"
                "</td></tr>"
                for i in _compact(g)
            )
            page.insert_htmlbox(
                pymupdf.Rect(28, 14, 567, 828),
                f"<h1>{html.escape(g.title)}</h1>"
                f"<p>{html.escape(g.summary)}</p><p>Classified by position and form "
                "(one confirmation: ☐ ACCEPT ALL ☐ SEE NOTES):</p>"
                f"<table>{listing}</table>",
                css=css,
                archive=archive,
                scale_low=0,
            )
            page = None
        for it in _cards(g):
            if page is None or y > 600:
                page = doc.new_page(width=595, height=842)
                page.insert_htmlbox(
                    pymupdf.Rect(28, 14, 567, 40),
                    f"<h1>{html.escape(g.title)}</h1>",
                    css=css,
                    archive=archive,
                )
                y = 40.0
            png = _crop(pdf, it)
            pix = pymupdf.Pixmap(png)
            w = min(539.0, pix.width * 72 / CROP_DPI)
            h = min(110.0, w * pix.height / pix.width)
            page.insert_image(
                pymupdf.Rect(567 - w, y, 567, y + h), stream=png, keep_proportion=True
            )
            facts = "".join(
                f"<tr><td>{k}</td><td{'' if k in LTR_FIELDS else ' class=fa'}>"
                f"{html.escape(str(v)[:220])}</td></tr>"
                for k, v in _facts(it)
            )
            opts = "".join(f"<span class='o'>☐ {html.escape(o)}</span>" for o in it.options)
            page.insert_htmlbox(
                pymupdf.Rect(28, y + h + 2, 567, y + h + 120),
                f"<b>{html.escape(it.title)}</b> ({it.decision})<table>{facts}</table>{opts}",
                css=css,
                archive=archive,
                scale_low=0,
            )
            y += h + 126
    page = doc.new_page(width=595, height=842)
    rows = "".join(
        f"<tr><td>{t['pdf_page']}</td><td>{t['size']}</td><td class='fa'>"
        f"{html.escape(' | '.join(t['captions']))}</td><td>{len(t['leaked_prose_blocks'])}"
        "</td></tr>"
        for t in tables
    )
    page.insert_htmlbox(
        pymupdf.Rect(28, 20, 567, 820),
        "<h1>10. Ruled tables – caption linkage "
        "and leak check</h1><table><tr><td>PDF page</td><td>size</td><td>caption"
        f"</td><td>prose inside</td></tr>{rows}</table>",
        css=css,
        archive=archive,
    )
    doc.subset_fonts()
    doc.save(out, garbage=4, deflate=True)
    n = int(doc.page_count)
    doc.close()
    return n


def write_review(corpus_dir: Path, pdf_path: Path, html_out: Path, pdf_out: Path) -> dict[str, Any]:
    lines = (corpus_dir / "blocks.jsonl").read_text(encoding="utf-8").splitlines()
    blocks = [json.loads(x) for x in lines if x.strip()]
    with pymupdf.open(pdf_path) as pdf:
        groups = build_groups(pdf, blocks)
        tables = table_checks(blocks)
        rule = sum(g.decisions for g in groups if g.key in ("M1", "G", "B1", "C1"))
        per = sum(g.decisions for g in groups if g.key not in ("M1", "G", "B1", "C1"))
        summary = {
            "groups": {
                g.key: {
                    "title": g.title,
                    "items": len(g.items),
                    "decisions": g.decisions,
                    "rule": g.rule,
                }
                for g in groups
            },
            "human_decisions": rule + per,
            "rule_decisions": rule,
            "per_block_decisions": per,
            "tables": tables,
            "tables_ok": all(
                len(t["captions"]) == 1 and not t["leaked_prose_blocks"] for t in tables
            ),
        }
        html_out.write_text(
            review_html(pdf, groups, tables, summary), encoding="utf-8", newline="\n"
        )
        summary["pdf_pages"] = review_pdf(pdf, groups, tables, pdf_out)
    return summary
