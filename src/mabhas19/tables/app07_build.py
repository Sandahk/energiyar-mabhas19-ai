"""Phase 2C outputs for appendix 7: raw layer, candidate layer, flags, cross-check, review.

Everything is written under data/processed/app07/<model>/ (git-ignored). The raw Gemini
responses are copied byte for byte and never rewritten; candidates reference them by
request id and row index. Nothing here calls an external API or writes to curated/.
"""

from __future__ import annotations

import html
import json
import shutil
from collections import Counter
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

import pymupdf

from mabhas19.benchmark import app07_scoring as sc
from mabhas19.benchmark import phase2b as pb
from mabhas19.benchmark.extractors import glyph_row_centres
from mabhas19.benchmark.metrics import label_match
from mabhas19.benchmark.references import ReferenceItem, load_references
from mabhas19.core.paths import EVAL_DIR, SOURCES_DIR
from mabhas19.tables import app07 as a7
from mabhas19.tables import app07_decisions as dec
from mabhas19.text.normalize import to_persian_digits

DOC_ID = "m19-ed5-1404-app07"
PDF = SOURCES_DIR / "mabhas19" / "ed5-1404" / "appendices" / f"{DOC_ID}-v1.pdf"
PRINTED_PAGE_OFFSET = 1  # printed page = PDF page - 1 (p3 «۲», p6 «۵», p13 «۱۲»)
COND_COLUMN = (121.0, 185.0)
MATERIAL_COLUMN = (263.0, 542.0)
TABLE_X = (122.0, 540.0)
PAGE_HEADER_RULE_Y = 66.0
REVIEW_DPI = 110
CROSSCHECK_PAGES = ("app07-p003", "app07-p006", "app07-p013")
LABEL_CHECKS = ("category", "subgroup", "material", "condition")
FOOTNOTE_CHECKS = ("footnote_refs", "subgroup_footnote_refs", "category_footnote_refs")


# --------------------------------------------------------------------------- row count


def _rules(page: pymupdf.Page) -> list[tuple[float, float, float]]:
    return [(d["rect"].y0, d["rect"].x0, d["rect"].x1) for d in page.get_drawings()
            if d["rect"].height < 1.5 and d["rect"].width > 100]  # fmt: skip


def value_row_centres(page: pymupdf.Page) -> list[float]:
    """Vertical centres of printed conductivity values (vector glyphs, no OCR).

    Glyph lines in the conductivity column between the page header rule and the footnote
    separator, minus repeated table-header bands (a ruled band of >= 50 pt whose material
    column holds one centred line, «مصالح»).
    """
    rules = _rules(page)
    full = sorted(y for y, x0, x1 in rules if x0 <= TABLE_X[0] and x1 >= TABLE_X[1])
    foot = [y for y, x0, x1 in rules if x0 <= TABLE_X[0] and x1 < MATERIAL_COLUMN[0] + 40]
    bottom = min(foot) if foot else page.rect.height - 50
    cond = [y for y in glyph_row_centres(page, *COND_COLUMN, PAGE_HEADER_RULE_Y) if y < bottom]
    mat = [y for y in glyph_row_centres(page, *MATERIAL_COLUMN, PAGE_HEADER_RULE_Y) if y < bottom]

    def header(a: float, b: float) -> bool:
        inside = [y for y in mat if a < y < b]
        return (b - a >= 50 and sum(a < y < b for y in cond) >= 3 and len(inside) == 1
                and abs(inside[0] - (a + b) / 2) < (b - a) / 6)  # fmt: skip

    bands = [(a, b) for a, b in pairwise(full) if header(a, b)]
    return [round(y, 1) for y in cond if not any(a < y < b for a, b in bands)]


# --------------------------------------------------------------------------- inputs


def load_page(responses: Path, pdf: pymupdf.Document, pdf_page: int) -> tuple[
        a7.PageInput, dict[str, Any]]:  # fmt: skip
    request_id = f"app07-p{pdf_page:03d}.page"
    path = responses / f"{request_id}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    parsed = pb.parse_json(data.get("text")) if data.get("status") == "ok" else None
    schema = pb.schema_ok("app07_page", parsed)
    rows = [r for r in parsed["rows"] if isinstance(r, dict)] if schema and parsed else []
    if data.get("received_at"):
        stamp, source = data["received_at"], "response_received_at"
    else:  # Phase 2B responses predate the timestamp field
        stamp = datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat(timespec="seconds")
        source = "cache_file_mtime"
    ys = value_row_centres(pdf[pdf_page - 1])
    page = a7.PageInput(DOC_ID, pdf_page, to_persian_digits(str(pdf_page - PRINTED_PAGE_OFFSET)),
                        request_id, data.get("model_version"), stamp, source, rows,
                        expected_rows=len(ys), row_ys=ys)  # fmt: skip
    info = {"request_id": request_id, "status": data.get("status"), "json_schema_ok": schema,
            "rows_returned": len(rows), "value_rows_on_page": len(ys),
            "usage": data.get("usage"), "latency_s": data.get("latency_s"),
            "origin": "phase2c_call" if data.get("received_at") else "phase2b_cache",
            "cost_usd": round(sc.request_cost(*(data.get("usage", {}).get(k) or 0 for k in (
                "prompt_tokens", "output_tokens", "thinking_tokens"))), 4)}  # fmt: skip
    return page, info


# --------------------------------------------------------------------------- cross-check


def _ref_value(ref: ReferenceItem, name: str) -> Any:
    f = ref.expected_fields
    if name in FOOTNOTE_CHECKS:
        return list(f.get(name) or [])
    if name == "density_qualifier":
        return a7.QUALIFIERS.get(f.get("density_qualifier") or "", None)
    return f.get(name)


def crosscheck(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare candidates with the human-verified reference rows. Nothing is replaced."""
    refs = load_references(EVAL_DIR / "extraction" / "reference")
    by_page: dict[int, list[dict[str, Any]]] = {}
    for c in candidates:
        by_page.setdefault(c["pdf_page"], []).append(c)
    disagreements: list[dict[str, Any]] = []
    compared = Counter[str]()
    unmatched: list[str] = []
    for page_id in CROSSCHECK_PAGES:
        rows = sc.verified_rows(refs[page_id])
        cands = by_page.get(int(page_id.rsplit("p", 1)[1]), [])
        if len(cands) == len(rows):
            pairs = dict(zip([r.id for r in rows], cands, strict=True))
        else:
            extracted = [pb.ExtractedRow(c["raw_density_text"], c["raw_conductivity_text"])
                         for c in cands]  # fmt: skip
            aligned = pb.align_rows(rows, extracted)
            pairs = {rid: cands[extracted.index(e)] for rid, e in aligned.items()}
        for ref in rows:
            cand = pairs.get(ref.id)
            if cand is None:
                unmatched.append(ref.id)
                continue
            for name, got, ok in _compare(ref, cand):
                compared[name] += 1
                if not ok:
                    disagreements.append({"reference_item": ref.id,
                                          "candidate_id": cand["candidate_id"], "field": name,
                                          "reference": _ref_value(ref, name),
                                          "candidate": got})  # fmt: skip
    rows_total = sum(len(sc.verified_rows(refs[p])) for p in CROSSCHECK_PAGES)
    rows_with_disagreement = len({d["reference_item"] for d in disagreements})
    return {"reference_rows": rows_total, "unmatched_reference_rows": unmatched,
            "fields_compared": dict(compared), "rows_with_disagreement": rows_with_disagreement,
            "by_field": dict(Counter(d["field"] for d in disagreements)),
            "disagreements": disagreements}  # fmt: skip


def _compare(ref: ReferenceItem, cand: dict[str, Any]) -> list[tuple[str, Any, bool]]:
    out: list[tuple[str, Any, bool]] = []
    for name in LABEL_CHECKS:
        got = cand.get(name)
        out.append((name, got, label_match(_ref_value(ref, name), got)))
    for name in FOOTNOTE_CHECKS:
        got = cand.get(name) or []
        out.append((name, got, sorted(got) == sorted(_ref_value(ref, name))))
    numeric = {"density_qualifier": cand["parsed_density_qualifier"],
               "density_min": cand["parsed_density_min"],
               "density_max": cand["parsed_density_max"],
               "conductivity": cand["parsed_conductivity"]}  # fmt: skip
    for name, got in numeric.items():
        out.append((name, got, got == _ref_value(ref, name)))  # exact text: precision kept
    return out


# --------------------------------------------------------------------------- build


def build(plan: dict[str, Any], *, model: str, responses: Path, out: Path) -> dict[str, Any]:
    raw_dir, review_dir = out / "raw", out / "review"
    for d in (raw_dir, review_dir / "pages"):
        d.mkdir(parents=True, exist_ok=True)
    ctx = a7.Context()
    pages: dict[int, dict[str, Any]] = {}
    candidates: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    with pymupdf.open(PDF) as pdf:
        for p in plan["pages"]:
            page, info = load_page(responses, pdf, p)
            shutil.copy2(responses / f"{page.request_id}.json", raw_dir)  # immutable copy
            raw_rows += [{"request_id": page.request_id, "pdf_page": p, "row_index": i,
                          "row": r} for i, r in enumerate(page.rows)]  # fmt: skip
            recs = a7.normalize_page(page, ctx)
            candidates += recs
            pages[p] = info
            pdf[p - 1].get_pixmap(dpi=REVIEW_DPI).save(review_dir / "pages" / f"p{p:03d}.png")
    _write_jsonl(out / "raw_rows.jsonl", raw_rows)
    _write_jsonl(out / "candidates_machine.jsonl", candidates)  # before owner decisions
    machine_check = crosscheck(candidates)
    decisions = dec.load_decisions() if dec.DECISIONS_PATH.is_file() else {"decisions": []}
    candidates = dec.apply_decisions(candidates, decisions)
    _write_jsonl(out / "candidates.jsonl", candidates)
    check = crosscheck(candidates)
    suspicious = [{"candidate_id": c["candidate_id"], "pdf_page": c["pdf_page"],
                   "row_y": c["row_y"], "flags": c["suspicious"],
                   "owner_review": c["owner_review"]}
                  for c in candidates if c["suspicious"]]  # fmt: skip
    summary = _summary(model, plan, pages, candidates, check)
    summary["crosscheck_machine_layer"] = {k: v for k, v in machine_check.items()
                                           if k != "disagreements"}  # fmt: skip
    summary["owner_review"] = dec.review_summary(candidates)
    for name, obj in (("suspicious_rows.json", suspicious), ("crosscheck.json", check),
                      ("crosscheck_machine.json", machine_check),
                      ("summary.json", summary)):  # fmt: skip
        (out / name).write_text(json.dumps(obj, ensure_ascii=False, indent=1) + "\n",
                                encoding="utf-8", newline="\n")  # fmt: skip
    (review_dir / "index.html").write_text(review_html(candidates, pages, check),
                                           encoding="utf-8", newline="\n")  # fmt: skip
    print(json.dumps({k: v for k, v in summary.items() if k != "pages"}, ensure_ascii=False,
                     indent=1))  # fmt: skip
    print(f"outputs: {out}")
    return summary


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def _summary(model: str, plan: dict[str, Any], pages: dict[int, dict[str, Any]],
             candidates: list[dict[str, Any]], check: dict[str, Any]
             ) -> dict[str, Any]:  # fmt: skip
    suspicious = [c for c in candidates if c["suspicious"]]
    parsed_ok = [c for c in candidates if c["parsed_conductivity"] is not None
                 and c["parsed_density_kind"] is not None]  # fmt: skip
    usage_keys = ("prompt_tokens", "output_tokens", "thinking_tokens")
    by_origin: dict[str, dict[str, Any]] = {}
    for origin in ("phase2b_cache", "phase2c_call"):
        sel = [i for i in pages.values() if i["origin"] == origin]
        by_origin[origin] = {"calls": len(sel), **{k: sum((i["usage"] or {}).get(k) or 0
                             for i in sel) for k in usage_keys},
                             "cost_usd": round(sum(i["cost_usd"] for i in sel), 4)}  # fmt: skip
    structure: dict[str, dict[str, Any]] = {}
    for c in candidates:
        cat = structure.setdefault(c["category"] or "(none)", {"rows": 0, "subgroups": {},
                                                               "pages": []})  # fmt: skip
        cat["rows"] += 1
        if c["pdf_page"] not in cat["pages"]:
            cat["pages"].append(c["pdf_page"])
        sub = cat["subgroups"].setdefault(c["subgroup"] or "(no subgroup)", set())
        sub.add(c["material"] or "(no material)")
    for cat in structure.values():
        cat["subgroups"] = {k: sorted(v) for k, v in cat["subgroups"].items()}
    return {
        "model": model, "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "verification_status": a7.STATUS, "cost_plan": plan, "api_usage": by_origin,
        "pages": {str(k): v for k, v in pages.items()},
        "candidate_rows": len(candidates), "rows_fully_parsed": len(parsed_ok),
        "value_rows_on_pages": sum(i["value_rows_on_page"] for i in pages.values()),
        "row_count_mismatch_pages": [p for p, i in pages.items()
                                     if i["rows_returned"] != i["value_rows_on_page"]],
        "suspicious_rows": len(suspicious),
        "suspicious_by_code": dict(Counter(f["code"] for c in suspicious for f in c["suspicious"])),
        "crosscheck": {k: v for k, v in check.items() if k != "disagreements"},
        "structure": structure,
    }  # fmt: skip


# --------------------------------------------------------------------------- review page

_CSS = """
:root{--bg:#fafaf7;--fg:#1d1d1b;--muted:#6b6b66;--line:#ddd9cf;--warn:#fff1c7;
--bad:#ffd9d4;--inh:#e8f0fb;--card:#fff}
@media (prefers-color-scheme:dark){:root{--bg:#1b1b19;--fg:#ecebe6;--muted:#a3a29b;
--line:#3a3934;--warn:#4a3f17;--bad:#5a2721;--inh:#1f2e44;--card:#24241f}}
body{background:var(--bg);color:var(--fg);font:14px/1.5 Vazirmatn,Tahoma,sans-serif;margin:0}
header{padding:16px;border-bottom:1px solid var(--line)}
h1{font-size:18px;margin:0 0 4px}h2{font-size:16px;margin:0 0 8px}
.muted{color:var(--muted)}nav a{margin-left:10px}
section{display:grid;grid-template-columns:minmax(280px,520px) 1fr;gap:16px;padding:16px;
border-bottom:1px solid var(--line)}
.img{position:sticky;top:8px;align-self:start}.img div{position:relative}
.img img{width:100%;border:1px solid var(--line);background:#fff}
.mk{position:absolute;left:2px;font:10px monospace;background:#c0392b;color:#fff;
padding:0 3px;border-radius:3px;transform:translateY(-50%)}
.mk.ok{background:#2d7a46}
table{border-collapse:collapse;width:100%;background:var(--card)}
td,th{border:1px solid var(--line);padding:4px 6px;vertical-align:top;text-align:right}
th{font-size:12px;color:var(--muted)}tr.flag td{background:var(--warn)}
.inh{background:var(--inh);border-radius:3px;padding:0 3px}
.raw{font-size:11px;color:var(--muted);direction:rtl}
.num{direction:ltr;text-align:left;font-family:monospace;white-space:nowrap}
.fl{color:#b03a2e;font-size:12px}.dn{font-size:11px;color:#2d7a46}.bad{background:var(--bad)}
@media (max-width:800px){section{grid-template-columns:1fr}.img{position:static}}
"""


_HEAD = "".join(f"<th>{h}</th>" for h in (
    "#", "category", "subgroup", "material", "condition", "density kg/m3", "λ W/m.K",
    "footnotes", "flags / notes"))  # fmt: skip


def _e(value: Any) -> str:
    return "" if value is None else html.escape(str(value))


def _label(c: dict[str, Any], name: str) -> str:
    value = c.get(name)
    src = c.get(f"{name}_source")
    raw = c.get(f"{name}_source_text")
    if value is None:
        return "<span class='muted'>—</span>"
    cls = " class='inh'" if src and src.startswith("inherited") else ""
    tip = f" title='{_e(src)} ← {_e(raw)}'" if src else ""
    return f"<span{cls}{tip}>{_e(value)}</span>"


def _row_html(c: dict[str, Any], bad: set[str]) -> str:
    flags = "<br>".join(f"<span class='fl'>⚑ {_e(f['code'])}: {_e(f['detail'])}</span>"
                        for f in c["suspicious"])  # fmt: skip
    notes = "<br>".join(_e(n) for n in c["normalization_notes"])
    review = c.get("owner_review") or {}
    if review.get("decisions"):
        flags += (f"<br><span class='dn'>owner: {_e(review['status'])} "
                  f"({_e(', '.join(review['decisions']))})</span>")  # fmt: skip
    fn = {k: c.get(k) for k in a7_footnote_keys() if c.get(k)}
    dens = (
        f"{_e(c['parsed_density_qualifier'] or '')} {_e(c['parsed_density_min'])}"
        f"{' – ' + _e(c['parsed_density_max']) if c['parsed_density_kind'] == 'range' else ''}"
    )
    open_flag = c["suspicious"] and review.get("status") != "owner_reviewed"
    cls = " class='flag'" if open_flag else ""
    mark = " bad" if c["candidate_id"] in bad else ""
    return (f"<tr{cls} id='{c['candidate_id']}'><td class='num{mark}'>{c['source_row_order']}</td>"
            f"<td>{_label(c, 'category')}<div class='raw'>{_e(c['raw_category'])}</div></td>"
            f"<td>{_label(c, 'subgroup')}<div class='raw'>{_e(c['raw_subgroup'])}</div></td>"
            f"<td>{_label(c, 'material')}<div class='raw'>{_e(c['raw_material'])}</div>"
            f"<div class='dn'>{_e(c.get('display_name'))}</div></td>"
            f"<td>{_e(c['condition'])}<div class='raw'>{_e(c['raw_condition'])}</div></td>"
            f"<td>{_e(c['raw_density_text'])}<div class='num'>{dens}</div></td>"
            f"<td>{_e(c['raw_conductivity_text'])}<div class='num'>"
            f"{_e(c['parsed_conductivity'])}</div></td>"
            f"<td class='num'>{_e(json.dumps(fn) if fn else '')}</td>"
            f"<td>{flags}<div class='raw'>{notes}</div></td></tr>")  # fmt: skip


def a7_footnote_keys() -> tuple[str, ...]:
    return FOOTNOTE_CHECKS


def review_html(candidates: list[dict[str, Any]], pages: dict[int, dict[str, Any]],
                check: dict[str, Any]) -> str:  # fmt: skip
    bad = {d["candidate_id"] for d in check["disagreements"]}
    scale = REVIEW_DPI / 72
    nav = " ".join(f"<a href='#p{p}'>p{p}</a>" for p in pages)
    parts = [f"<!doctype html><html lang='fa' dir='rtl'><head><meta charset='utf-8'>"
             f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
             f"<title>Appendix 7 review</title><style>{_CSS}</style></head><body><header>"
             f"<h1>پیوست ۷ — نامزدهای استخراج ماشینی (در انتظار بازبینی انسانی)</h1>"
             f"<div class='muted'>{len(candidates)} rows · status {a7.STATUS} · "
             f"⚑ = suspicious · <span class='inh'>blue</span> = inherited · grey = raw model "
             f"text · red row number = disagrees with a verified reference row</div>"
             f"<nav>{nav}</nav></header>"]  # fmt: skip
    for p, info in pages.items():
        recs = [c for c in candidates if c["pdf_page"] == p]
        marks = "".join(
            f"<span class='mk{'' if c['suspicious'] else ' ok'}' style='top:"
            f"{c['row_y'] * scale / (792 * scale) * 100:.2f}%'>{c['source_row_order']}</span>"
            for c in recs if c["row_y"] is not None)  # fmt: skip
        rows = "".join(_row_html(c, bad) for c in recs)
        printed = _e(recs[0]["printed_page"]) if recs else ""
        parts.append(
            f"<section id='p{p}'><div class='img'><h2>PDF p{p} · printed «{printed}»</h2>"
            f"<div class='muted'>{info['rows_returned']} rows returned · "
            f"{info['value_rows_on_page']} value rows on page · {info['origin']}</div>"
            f"<div><img src='pages/p{p:03d}.png' alt='PDF page {p}'>{marks}</div></div>"
            f"<div><table><thead><tr>{_HEAD}</tr></thead><tbody>{rows}</tbody></table></div>"
            "</section>")  # fmt: skip
    parts.append("</body></html>\n")
    return "".join(parts)
