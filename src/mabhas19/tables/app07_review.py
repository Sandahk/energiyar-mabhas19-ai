"""Human-review preparation for the suspicious appendix 7 candidate rows (Phase 2C).

Reads the candidate layer written by :mod:`mabhas19.tables.app07_build` and writes, under
``<candidates dir>/review_suspicious/``:

* ``entries.json``: one entry per suspicious row (source, crop, raw model fields,
  normalized fields, inherited fields, numbers, flags, decisions needed, neighbours);
* ``inheritance_rules.json``: every row - flagged or not - that relies on an inheritance
  rule, grouped by rule, so the scope of each rule can be confirmed;
* ``index.html``: the same material grouped by issue type, with row crops;
* ``crops/``: renders of the source PDF around each row (the PDF is never modified).

Nothing is verified here: every entry is ``pending_human_review`` with an empty decision.
No external API is called and nothing is written to curated/.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import pymupdf

from mabhas19.tables import app07 as a7
from mabhas19.tables.app07_build import PDF, TABLE_X

CROP_DPI = 170
CROP_PAD = 14.0  # points above the previous / below the next row centre
REVIEW_STATUS = "pending_human_review"

GROUPS: tuple[tuple[str, str, set[str]], ...] = (
    ("footnote", "1. Footnote-marker ambiguity", {"footnote_heading_ambiguity"}),
    ("heading", "2. Unclear heading / subgroup", {"heading_scope_ambiguous"}),
    ("merged", "3. Merged or duplicated labels", {"duplicate_or_merged_label"}),
    ("mixed", "4. Latin/Persian mixed-order ambiguity", {"mixed_direction_label"}),
    ("structure", "5. Inconsistent structure", {"structure_inconsistent"}),
    ("inheritance", "6. Inheritance cases", set()),
)

# What was seen on the page renders while preparing this review (to be confirmed).
OBSERVATIONS = {
    "app07-p004-r09": "Render: the sub-heading ends «... بیش از ۳۰۰¹:» with a superscript 1.",
    "app07-p004-r13": "Render: the sub-heading ends «... کمتر از ۲۵۰¹:» with a superscript 1.",
    "app07-p004-r15": "Render: no label on this value row («کمتر از ۶۰۰»); the nearest label "
                      "above is «- بدون ماسه و با عیار سیمان کم» (row r14).",
    "app07-p011-r05": "Render: «الیافی» is undashed, without ':', beside two value rows.",
    "app07-p011-r07": "Render: «سلولزی» is undashed, without ':', beside two value rows.",
    "app07-p011-r19": "Render: two printed lines «- HCFC» and «- CFC:»; this row's values "
                      "are on the «HCFC» line; r20/r21 follow under «CFC:».",
}  # fmt: skip


# --------------------------------------------------------------------------- inputs


def load_candidates(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def load_raw_rows(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    out = {}
    for x in path.read_text(encoding="utf-8").splitlines():
        if x.strip():
            r = json.loads(x)
            out[(r["request_id"], r["row_index"])] = r["row"]
    return out


# --------------------------------------------------------------------------- fields


def inherited_fields(c: dict[str, Any]) -> list[dict[str, Any]]:
    """Every normalized value that did not come from the row's own printed text."""
    out = []
    for name in ("category", "subgroup", "material"):
        src = c.get(f"{name}_source") or ""
        if src.startswith("inherited") and c.get(name) is not None:
            out.append({"field": name, "value": c[name], "rule": _rule(name, src),
                        "from": c.get(f"{name}_inherited_from")})  # fmt: skip
    for part in c.get("condition_parts", []):
        if part.get("source") == "inherited":
            out.append({"field": "condition", "value": part["text"], "rule": "leaf_label",
                        "from": part.get("inherited_from")})  # fmt: skip
    return out


def _rule(name: str, source: str) -> str:
    if name == "material":
        return "material_continuation"
    if source == "inherited_previous_page":
        return "cross_page_subgroup"
    return "cross_page_category" if name == "category" else source


def numbers(c: dict[str, Any]) -> dict[str, Any]:
    return {
        "density_raw": c["raw_density_text"], "conductivity_raw": c["raw_conductivity_text"],
        "density_kind": c["parsed_density_kind"],
        "density_qualifier": c["parsed_density_qualifier"],
        "density_min": c["parsed_density_min"], "density_max": c["parsed_density_max"],
        "conductivity": c["parsed_conductivity"], "units": c["units"],
        "rejections": c["numeric_rejections"],
    }  # fmt: skip


def normalized(c: dict[str, Any]) -> dict[str, Any]:
    keys = ("category", "category_footnote_refs", "subgroup", "subgroup_footnote_refs",
            "material", "footnote_refs", "condition", "condition_parts")  # fmt: skip
    return {k: c.get(k) for k in keys}


def brief(c: dict[str, Any] | None) -> dict[str, Any] | None:
    """A neighbouring row, for judging inheritance."""
    if c is None:
        return None
    return {"id": c["candidate_id"], "raw_material": c["raw_material"],
            "raw_condition": c["raw_condition"], "material": c["material"],
            "material_source": c["material_source"], "condition": c["condition"],
            "subgroup": c["subgroup"], "density": c["raw_density_text"],
            "conductivity": c["raw_conductivity_text"]}  # fmt: skip


def groups_of(c: dict[str, Any]) -> list[str]:
    codes = {f["code"] for f in c["suspicious"]}
    out = [key for key, _, flags in GROUPS if flags & codes]
    if decision_inheritance(c):
        out.append("inheritance")
    return out


def decision_inheritance(c: dict[str, Any]) -> list[dict[str, Any]]:
    """Inherited values that need a judgement (a category carried over from the previous
    page is listed but not questioned: the numbered heading is unambiguous)."""
    return [i for i in inherited_fields(c) if i["rule"] != "cross_page_category"]


# --------------------------------------------------------------------------- decisions


def _ambiguous_footnotes(raw: dict[str, Any]) -> list[a7.Line]:
    return [ln for ln in a7.row_lines(raw) if ln.label.footnote_ambiguous]


def decisions(c: dict[str, Any], raw: dict[str, Any], prev: dict[str, Any] | None,
              earlier_sub: str | None) -> list[str]:  # fmt: skip
    """The exact structural questions a reviewer must answer for this row."""
    out: list[str] = []
    for flag in c["suspicious"]:
        out += _decision_for(flag, c, raw, prev, earlier_sub)
    for inh in decision_inheritance(c):
        out.append(f"{inh['field']}: the row prints no own {inh['field']} text; the "
                   f"candidate inherits «{inh['value']}» from {inh['from']} "
                   f"(rule {inh['rule']}). Is this inheritance valid for this row?")  # fmt: skip
    return list(dict.fromkeys(out))


def _decision_for(flag: dict[str, str], c: dict[str, Any], raw: dict[str, Any],
                  prev: dict[str, Any] | None, earlier_sub: str | None) -> list[str]:  # fmt: skip
    code, detail = flag["code"], flag["detail"]
    if code == "footnote_heading_ambiguity":
        return [f"In «{ln.label.raw}» ({ln.field}) is the trailing «{ln.label.footnotes[0]}» a "
                f"footnote marker (candidate: footnote {ln.label.footnotes[0]}, text "
                f"«{ln.label.text}») or part of the printed text? Is any text lost before it?"
                for ln in _ambiguous_footnotes(raw)]  # fmt: skip
    if code == "heading_scope_ambiguous":
        return [_heading_question(detail, c, earlier_sub)]
    if code == "duplicate_or_merged_label":
        return [_merged_question(detail, c, prev)]
    if code == "mixed_direction_label":
        field = detail.split(":", 1)[0]
        return [f"Confirm the reading order of {field} «{raw.get(field)}» against the crop. "
                f"Candidate condition: «{c['condition']}». Numbers inside labels are not "
                "parsed; state the correct text."]  # fmt: skip
    if code == "structure_inconsistent":
        return [f"{detail}: material «{c['material']}» has no subgroup in the candidate. Which "
                "heading/subgroup does this row belong to?"]  # fmt: skip
    return [f"{code}: {detail}"]


def _heading_question(detail: str, c: dict[str, Any], earlier_sub: str | None) -> str:
    if detail.startswith("the model gave this item label as a subgroup"):
        return (f"Gemini returned «{c['raw_subgroup']}» as a subgroup; by its printed form "
                f"(undashed, no ':') the candidate treats it as material «{c['material']}». "
                "Material or subgroup?")  # fmt: skip
    if detail.startswith("undashed item after dashed items"):
        return (
            f"«{c['material']}» is undashed after dashed items. Does it belong to subgroup "
            f"«{c['subgroup']}» (candidate) or does that subgroup end before it?"
        )
    if detail.startswith("a sub-heading on an earlier row"):
        sub = earlier_sub or "(none found)"
        return (
            f"Does the earlier sub-heading {sub} also apply to this row? Candidate "
            f"condition (sub-heading not inherited): «{c['condition']}»."
        )
    return detail


def earlier_subheading(candidates: list[dict[str, Any]], i: int) -> str | None:
    """The nearest sub-heading on an earlier row of the same page and material."""
    c = candidates[i]
    for prev in reversed(candidates[:i]):
        if prev["pdf_page"] != c["pdf_page"] or prev["material"] != c["material"]:
            return None
        subs = [p["text"] for p in prev["condition_parts"] if p["role"] == "sub_heading"]
        if subs:
            return f"«{subs[-1]}» (on {prev['candidate_id']})"
    return None


def _merged_question(detail: str, c: dict[str, Any], prev: dict[str, Any] | None) -> str:
    if "repeats the category heading" in detail:
        return (
            f"Gemini copied the category heading into the material field "
            f"(«{c['raw_material']}»); the candidate drops it (material = none). Confirm."
        )
    if detail.startswith("the previous row's label"):
        last = (prev or {}).get("raw_material") or (prev or {}).get("raw_condition") or ""
        tail = last.split("\n")[-1] if last else ""
        return (
            f"Does the last printed line of the previous row's label («{tail}») belong to "
            f"this row (e.g. as its subgroup)? Candidate subgroup: «{c['subgroup'] or '(none)'}»."
        )
    if "joins several printed lines" in detail:
        field = detail.split(" ", 1)[0]
        return (f"{field} «{c.get('raw_' + field.removesuffix('_raw'))}» spans several printed "
                "lines and was joined/split by layout rules into material "
                f"«{c['material']}», subgroup «{c['subgroup'] or '(none)'}», condition "
                f"«{c['condition']}». "
                "Is the split correct?")  # fmt: skip
    return detail


# --------------------------------------------------------------------------- crops


def render_crop(doc: pymupdf.Document, c: dict[str, Any], prev: dict[str, Any] | None,
                nxt: dict[str, Any] | None, out: Path) -> dict[str, Any]:  # fmt: skip
    """Crop from the previous to the next row (same page) with the target band located."""
    page = doc[c["pdf_page"] - 1]
    y = c["row_y"]
    if y is None:
        page.get_pixmap(dpi=90).save(out)
        return {"path": out.name, "target_top_pct": None, "target_height_pct": None}
    y0 = min(y - 40, _neighbour_y(prev, c, y - 40)) - CROP_PAD
    y1 = max(y + 12, _neighbour_y(nxt, c, y + 12)) + CROP_PAD
    clip = pymupdf.Rect(TABLE_X[0] - 4, max(y0, 60), TABLE_X[1] + 6, min(y1, page.rect.height))
    page.get_pixmap(dpi=CROP_DPI, clip=clip).save(out)
    h = clip.height
    return {"path": out.name, "clip_pt": [round(v, 1) for v in clip],
            "target_top_pct": round((y - 8 - clip.y0) / h * 100, 2),
            "target_height_pct": round(16 / h * 100, 2)}  # fmt: skip


def _neighbour_y(other: dict[str, Any] | None, c: dict[str, Any], default: float) -> float:
    if other is None or other["pdf_page"] != c["pdf_page"] or other["row_y"] is None:
        return default
    return float(other["row_y"])


# --------------------------------------------------------------------------- build


def build_entries(candidates: list[dict[str, Any]], raw_rows: dict[tuple[str, int], Any],
                  crosscheck: dict[str, Any]) -> list[dict[str, Any]]:  # fmt: skip
    disagreements: dict[str, list[dict[str, Any]]] = {}
    for d in crosscheck.get("disagreements", []):
        disagreements.setdefault(d["candidate_id"], []).append(d)
    entries = []
    for i, c in enumerate(candidates):
        if not c["suspicious"] or c.get("review", {}).get("status") == "reviewed":
            continue  # only flagged rows nobody has resolved
        prev = candidates[i - 1] if i > 0 else None
        nxt = candidates[i + 1] if i + 1 < len(candidates) else None
        raw = raw_rows[(c["raw_ref"]["request_id"], c["raw_ref"]["row_index"])]
        entries.append({
            "row_id": c["candidate_id"], "doc_id": c["doc_id"], "pdf_page": c["pdf_page"],
            "printed_page": c["printed_page"], "source_row_order": c["source_row_order"],
            "row_y": c["row_y"], "groups": groups_of(c), "raw_gemini": raw,
            "normalized": normalized(c), "inherited": inherited_fields(c),
            "numbers": numbers(c), "flags": c["suspicious"],
            "normalization_notes": c["normalization_notes"],
            "reference_disagreements": disagreements.get(c["candidate_id"], []),
            "observation": OBSERVATIONS.get(c["candidate_id"]),
            "decisions_needed": decisions(c, raw, prev, earlier_subheading(candidates, i)),
            "context": {"previous": brief(prev), "next": brief(nxt)},
            "verification_status": c["verification_status"],
            "review_status": REVIEW_STATUS, "decision": None,
        })  # fmt: skip
    return entries


def inheritance_scope(candidates: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """All rows (flagged or not) per inheritance rule."""
    by_rule: dict[str, list[dict[str, Any]]] = {}
    for i, c in enumerate(candidates):
        for inh in inherited_fields(c):
            prev = candidates[i - 1] if i > 0 else None
            by_rule.setdefault(inh["rule"], []).append({
                "row_id": c["candidate_id"], "field": inh["field"], "value": inh["value"],
                "from": inh["from"], "flagged": bool(c["suspicious"]),
                "density": c["raw_density_text"], "conductivity": c["raw_conductivity_text"],
                "previous_row": brief(prev),
            })  # fmt: skip
    return by_rule


def write_review(cand_dir: Path) -> dict[str, Any]:
    out = cand_dir / "review_suspicious"
    (out / "crops").mkdir(parents=True, exist_ok=True)
    candidates = load_candidates(cand_dir / "candidates.jsonl")
    raw_rows = load_raw_rows(cand_dir / "raw_rows.jsonl")
    check = json.loads((cand_dir / "crosscheck.json").read_text(encoding="utf-8"))
    entries = build_entries(candidates, raw_rows, check)
    by_id = {c["candidate_id"]: i for i, c in enumerate(candidates)}
    with pymupdf.open(PDF) as doc:
        for e in entries:
            i = by_id[e["row_id"]]
            prev = candidates[i - 1] if i > 0 else None
            nxt = candidates[i + 1] if i + 1 < len(candidates) else None
            e["crop"] = render_crop(doc, candidates[i], prev, nxt,
                                    out / "crops" / f"{e['row_id']}.png")  # fmt: skip
            e["crop"]["path"] = f"crops/{e['crop']['path']}"
    scope = inheritance_scope(candidates)
    _dump(out / "entries.json", entries)
    _dump(out / "inheritance_rules.json", scope)
    (out / "index.html").write_text(review_page(entries, scope), encoding="utf-8", newline="\n")
    summary = {
        "suspicious_rows": len(entries),
        "by_group": {key: sum(key in e["groups"] for e in entries) for key, _, _ in GROUPS},
        "inheritance_rule_rows": {rule: len(rows) for rule, rows in scope.items()},
        "output": str(out),
    }
    _dump(out / "summary.json", summary)
    return summary


def _dump(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1) + "\n", encoding="utf-8",
                    newline="\n")  # fmt: skip


# --------------------------------------------------------------------------- html

_CSS = """
:root{--bg:#fafaf7;--fg:#1d1d1b;--muted:#6b6b66;--line:#ddd9cf;--card:#fff;--hi:#e0453a;
--inh:#e8f0fb;--warn:#fff1c7}
@media (prefers-color-scheme:dark){:root{--bg:#1b1b19;--fg:#ecebe6;--muted:#a3a29b;
--line:#3a3934;--card:#24241f;--inh:#1f2e44;--warn:#4a3f17}}
body{background:var(--bg);color:var(--fg);font:14px/1.55 Vazirmatn,Tahoma,sans-serif;margin:0}
header,main{padding:16px;max-width:1200px;margin:auto}h1{font-size:19px;margin:0 0 6px}
h2{font-size:17px;border-bottom:2px solid var(--line);padding-bottom:4px;margin-top:28px}
.muted{color:var(--muted)}nav a{margin-left:12px}
article{background:var(--card);border:1px solid var(--line);border-radius:6px;margin:14px 0;
padding:12px;display:grid;grid-template-columns:minmax(260px,460px) 1fr;gap:14px}
.crop{position:relative;align-self:start}.crop img{width:100%;border:1px solid var(--line);
background:#fff}.crop .t{position:absolute;left:0;right:0;border:2px solid var(--hi)}
h3{font-size:15px;margin:0 0 6px;direction:ltr;text-align:left}
table{border-collapse:collapse;width:100%;font-size:13px}td,th{border:1px solid var(--line);
padding:3px 6px;vertical-align:top;text-align:right}th{color:var(--muted);font-weight:normal;
white-space:nowrap}.inh{background:var(--inh)}.q{background:var(--warn);padding:6px 8px;
border-radius:4px;margin:6px 0}.ltr{direction:ltr;text-align:left;font-family:monospace}
pre{white-space:pre-wrap;margin:0;font:12px/1.4 monospace;direction:rtl;text-align:right}
@media (max-width:820px){article{grid-template-columns:1fr}}
"""


def _e(v: Any) -> str:
    if v is None:
        return "<span class='muted'>—</span>"
    if isinstance(v, (list, dict)):
        v = json.dumps(v, ensure_ascii=False)
    return html.escape(str(v))


def _entry_html(e: dict[str, Any]) -> str:
    crop = e["crop"]
    box = ""
    if crop.get("target_top_pct") is not None:
        box = (f"<div class='t' style='top:{crop['target_top_pct']}%;"
               f"height:{crop['target_height_pct']}%'></div>")  # fmt: skip
    inh = {i["field"] for i in e["inherited"]}
    n = e["normalized"]
    norm_rows = "".join(
        f"<tr{' class=inh' if k.split('_')[0] in inh else ''}><th>{k}</th><td>{_e(n[k])}</td></tr>"
        for k in ("category", "category_footnote_refs", "subgroup", "subgroup_footnote_refs",
                  "material", "footnote_refs", "condition"))  # fmt: skip
    raw_rows = "".join(f"<tr><th>{k}</th><td><pre>{_e(v)}</pre></td></tr>"
                       for k, v in e["raw_gemini"].items())  # fmt: skip
    num = e["numbers"]
    nums = (f"<span class='ltr'>{_e(num['density_qualifier'] or '')} {_e(num['density_min'])} – "
            f"{_e(num['density_max'])} kg/m3 · λ {_e(num['conductivity'])} W/m.K</span> "
            f"(raw «{_e(num['density_raw'])}» / «{_e(num['conductivity_raw'])}»)")  # fmt: skip
    inh_html = "".join(f"<li class='inh'>{_e(i['field'])} ← «{_e(i['value'])}» from "
                       f"{_e(i['from'])} <span class='muted'>({_e(i['rule'])})</span></li>"
                       for i in e["inherited"]) or "<li class='muted'>none</li>"  # fmt: skip
    flags = "".join(f"<li>{_e(f['code'])}: {_e(f['detail'])}</li>" for f in e["flags"])
    qs = "".join(f"<div class='q'>❓ {_e(q)}</div>" for q in e["decisions_needed"])
    ref = "".join(f"<li>{_e(d['field'])}: reference «{_e(d['reference'])}» vs candidate "
                  f"«{_e(d['candidate'])}»</li>" for d in e["reference_disagreements"])  # fmt: skip
    obs = f"<p class='muted'>👁 {_e(e['observation'])}</p>" if e["observation"] else ""
    ctx = "".join(
        f"<tr><th>{label}</th><td>{_e(r and r['id'])}</td><td>{_e(r and r['raw_material'])}</td>"
        f"<td>{_e(r and r['raw_condition'])}</td><td>{_e(r and r['material'])}</td>"
        f"<td>{_e(r and r['condition'])}</td><td class='ltr'>{_e(r and r['density'])} / "
        f"{_e(r and r['conductivity'])}</td></tr>"
        for label, r in (("previous", e["context"]["previous"]),
                         ("next", e["context"]["next"])))  # fmt: skip
    return (
        f"<article id='{e['row_id']}'><div class='crop'><img src='{crop['path']}' "
        f"alt='crop of {e['row_id']}'>{box}</div><div>"
        f"<h3>{e['row_id']} · {e['doc_id']} · PDF p{e['pdf_page']} · printed «"
        f"{_e(e['printed_page'])}» · row {e['source_row_order']}</h3>{qs}{obs}"
        f"<p><b>Numbers</b> (unchanged): {nums}</p><p><b>Flags</b></p><ul>{flags}</ul>"
        f"{'<p><b>Disagrees with verified reference</b></p><ul>' + ref + '</ul>' if ref else ''}"
        f"<p><b>Inherited fields</b></p><ul>{inh_html}</ul>"
        f"<table><caption class='muted'>normalized candidate (blue = inherited)</caption>"
        f"{norm_rows}</table><br><table><caption class='muted'>raw Gemini row (unchanged)"
        f"</caption>{raw_rows}</table><br><table><caption class='muted'>neighbouring rows"
        f"</caption><tr><th></th><th>id</th><th>raw material</th><th>raw condition</th>"
        f"<th>material</th><th>condition</th><th>density / λ</th></tr>{ctx}</table>"
        f"<p class='muted'>status: {e['verification_status']} · review: {e['review_status']}"
        "</p></div></article>"
    )  # fmt: skip


def _scope_html(scope: dict[str, list[dict[str, Any]]]) -> str:
    titles = {
        "leaf_label": "NEW generic rule: a value row with no label inherits the leaf label "
                      "(condition) of the row above",
        "material_continuation": "Material continuation (approved on pages 3/6/13; applied "
                                 "here to all pages)",
        "cross_page_subgroup": "Subgroup carried over from the previous page",
        "cross_page_category": "Category carried over from the previous page",
    }  # fmt: skip
    parts = ["<h2 id='scope'>Rows affected by inheritance rules (all rows, flagged or not)</h2>"]
    for rule in ("leaf_label", "material_continuation", "cross_page_subgroup",
                 "cross_page_category"):  # fmt: skip
        rows = scope.get(rule, [])
        body = "".join(
            f"<tr><td class='ltr'>{_e(r['row_id'])}{' ⚑' if r['flagged'] else ''}</td>"
            f"<td>{_e(r['value'])}</td><td class='ltr'>{_e(r['from'])}</td>"
            f"<td class='ltr'>{_e(r['density'])} / {_e(r['conductivity'])}</td></tr>"
            for r in rows)  # fmt: skip
        parts.append(f"<h3 style='direction:rtl;text-align:right'>{titles[rule]} — {len(rows)} "
                     f"rows</h3><table><tr><th>row (⚑ = flagged)</th><th>inherited value</th>"
                     f"<th>from</th><th>density / λ</th></tr>{body}</table>")  # fmt: skip
    return "".join(parts)


def review_page(entries: list[dict[str, Any]], scope: dict[str, list[dict[str, Any]]]) -> str:
    nav = " ".join(f"<a href='#g-{k}'>{t.split('. ', 1)[1]} "
                   f"({sum(k in e['groups'] for e in entries)})</a>"
                   for k, t, _ in GROUPS) + " <a href='#scope'>inheritance scope</a>"  # fmt: skip
    parts = [
        "<!doctype html><html lang='fa' dir='rtl'><head><meta charset='utf-8'><meta "
        "name='viewport' content='width=device-width,initial-scale=1'><title>Appendix 7 "
        f"suspicious rows</title><style>{_CSS}</style></head><body><header><h1>پیوست ۷ — "
        f"بازبینی {len(entries)} ردیف مشکوک</h1><p class='muted'>Nothing here is verified. "
        "Each entry needs a human decision; numbers are shown as extracted and must not "
        "change without explicit approval. A row can appear in several groups. Red box = "
        f"the row in the crop.</p><nav>{nav}</nav></header><main>"
    ]  # fmt: skip
    for key, title, _ in GROUPS:
        group = [e for e in entries if key in e["groups"]]
        parts.append(f"<h2 id='g-{key}'>{title} — {len(group)} rows</h2>")
        parts += [_entry_html(e) for e in group]
    parts.append(_scope_html(scope))
    parts.append("</main></body></html>\n")
    return "".join(parts)
