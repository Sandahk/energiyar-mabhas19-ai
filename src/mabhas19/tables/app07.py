"""Appendix 7 candidate records from whole-page Gemini rows (Phase 2C).

Pure, deterministic post-processing: no model is asked to "fix" anything. The raw model
row is never modified; every normalized field keeps the raw text it came from and every
rule that changed something leaves a note.

The model puts the same printed line in different fields from page to page, so each label
line is classified by its printed *form*, not by the field it arrived in (layout rules
from the human-verified pages 3, 6 and 13, checked against the other table pages):

* ``category``: numbered heading «N. ...»;
* ``subgroup``: undashed heading ending in ':' (an unnumbered heading in the category
  field is a subgroup too);
* ``heading``: dashed line ending in ':' - the first one on a row is the material, further
  ones are sub-headings and go to the condition;
* ``kv``: a «key: value» line (nominal density, ratio, rebar %) or a mix-ratio line
  «... حجم ... برای یک حجم ...» - a condition;
* ``item``: any other label - the material when the row has no heading, otherwise the
  leaf condition under that heading.

Footnote markers are separated from labels (superscript digit; a single digit right after
the heading colon «مسلح:۱», right before it «... شده ۱:», or after a closing parenthesis
«(...) ۱»); a bare trailing digit on a heading is also separated but flagged as
ambiguous because text may have been lost («گرماشکن ۲» for «گرماشکنی²»).

Inheritance follows the printed structure only: a row without its own material inherits
the previous material inside the same category and subgroup; a row without its own leaf
label inherits the previous leaf under the same material (a value row continuing the
label above); rows at the top of a page before any heading inherit category and subgroup
from the previous page. Sub-headings are never inherited (their scope is not printed);
rows where one might apply are flagged instead.

Numbers are parsed only from the raw cell text with the project's non-guessing parser;
anything else is rejected, never repaired.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from mabhas19.text.normalize import normalize_for_comparison, to_ascii_digits
from mabhas19.text.numbers import HIGH, parse_numeric

STATUS = "machine_extracted_pending_review"
UNITS = {"density": "kg/m3", "thermal_conductivity": "W/m.K"}
RAW_FIELDS = ("category_raw", "subgroup_raw", "material_raw", "condition_raw",
              "density_raw", "conductivity_raw")  # fmt: skip
LABEL_FIELDS = ("category_raw", "subgroup_raw", "material_raw", "condition_raw")
MODEL_PARSED_FIELDS = ("density_min_parsed", "density_max_parsed", "conductivity_parsed")
CONDITION_JOIN = "؛ "

CATEGORY, SUBGROUP, HEADING, KV, ITEM = "category", "subgroup", "heading", "kv", "item"

_D = "0-9۰-۹٠-٩"  # ASCII, Persian and Arabic-Indic digits
_SUPERSCRIPT = {"¹": "1", "²": "2", "³": "3", "⁴": "4", "⁵": "5", "⁶": "6", "⁷": "7",
                "⁸": "8", "⁹": "9"}  # fmt: skip
_DASH = re.compile(r"^[-‐-―−]\s*")
_NUMBERED = re.compile(rf"^([{_D}]{{1,2}})\s*[.]\s*\S")
_FN_AFTER_COLON = re.compile(rf"\s*:([{_D}])$")  # «مسلح:۱» (no space: «نسبت: ۳» is a value)
_FN_BEFORE_COLON = re.compile(rf"\s+([{_D}])\s*:$")  # «اتوکلاو شده ۱:»
_FN_AFTER_PAREN = re.compile(rf"(?<=\))\s*([{_D}])$")  # «(... ۶۵۰) ۱»
_FN_BARE = re.compile(rf"\s+([{_D}])$")  # «گرماشکن ۲» (headings only, ambiguous)
_NOT_BEFORE_FOOTNOTE = frozenset({"به", "تا", "از", "و"})  # «۱ به ۳:» is a value
_CONDITION_KV = re.compile(r"^[^:]+:\s*\S.*$")
_MIX_RATIO = re.compile(r"حجم\s.+\sبرای\s+یک\s+حجم")
_LATIN = re.compile(r"[A-Za-z]")
_DIGIT = re.compile(rf"[{_D}]")
_NUM = rf"[{_D}]+(?:[.٫/,،][{_D}]+)?"
_RANGE = re.compile(rf"^({_NUM})\s*تا\s*({_NUM})$")
_QUALIFIED = re.compile(rf"^(بیش از|کمتر از)\s*({_NUM})$")
_SINGLE = re.compile(rf"^({_NUM})$")
QUALIFIERS = {"بیش از": ">", "کمتر از": "<"}


# --------------------------------------------------------------------------- labels


@dataclass
class Label:
    raw: str
    text: str | None
    dashed: bool = False
    heading_colon: bool = False
    footnotes: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    footnote_ambiguous: bool = False


def _cut_footnote(text: str, lbl: Label) -> str:
    """Separate one footnote marker written as a plain digit next to the heading colon."""
    for pattern, where in ((_FN_AFTER_COLON, "after the heading colon"),
                           (_FN_BEFORE_COLON, "before the heading colon"),
                           (_FN_AFTER_PAREN, "after a closing parenthesis")):  # fmt: skip
        m = pattern.search(text)
        if m is None:
            continue
        before = text[: m.start()].split()
        if before and before[-1] in _NOT_BEFORE_FOOTNOTE:
            continue
        lbl.footnotes.append(to_ascii_digits(m[1]))
        lbl.heading_colon = lbl.heading_colon or pattern is not _FN_AFTER_PAREN
        lbl.notes.append(f"footnote marker «{m[1]}» {where} separated")
        if before and _SINGLE.match(before[-1]):  # «... ۳۰۰ ۱:»: a number may have lost a digit
            lbl.footnote_ambiguous = True
            lbl.notes.append("the footnote digit follows a number (ambiguous)")
        return text[: m.start()].rstrip()
    return text


def clean_label(raw: str, *, heading: bool) -> Label:
    """Strip list dash, heading colon and footnote markers; record what was removed."""
    lbl = Label(raw, None)
    text = " ".join(raw.split())
    if m := _DASH.match(text):
        lbl.dashed = True
        text = text[m.end() :]
        lbl.notes.append("leading list dash removed")
    supers = [c for c in text if c in _SUPERSCRIPT]
    if supers:
        lbl.footnotes += [_SUPERSCRIPT[c] for c in supers]
        text = "".join(c for c in text if c not in _SUPERSCRIPT).rstrip()
        lbl.notes.append(f"superscript footnote marker(s) {supers} separated")
    else:
        text = _cut_footnote(text, lbl)
    if text.endswith(":"):
        lbl.heading_colon = True
        text = text.rstrip(":").rstrip()
        lbl.notes.append("trailing heading colon removed")
    if heading and not lbl.footnotes and (m := _FN_BARE.search(text)):
        lbl.footnotes.append(to_ascii_digits(m[1]))
        lbl.footnote_ambiguous = True
        text = text[: m.start()].rstrip()
        lbl.notes.append(f"bare trailing digit «{m[1]}» read as a footnote marker (ambiguous)")
    lbl.text = text or None
    return lbl


def logical_lines(raw: str) -> list[str]:
    """Join wrapped printed lines: a new label starts with a dash or after a heading colon."""
    out: list[str] = []
    for line in (x.strip() for x in raw.split("\n")):
        if not line:
            continue
        prev = out[-1] if out else ""
        prev_closed = prev.endswith(":") or bool(_FN_AFTER_COLON.search(prev))
        if out and not _DASH.match(line) and not prev_closed:
            out[-1] = f"{out[-1]} {line}"
        else:
            out.append(line)
    return out


def is_condition(text: str) -> bool:
    """A «key: value» line or a mix-ratio line (never a heading ending in ':')."""
    return bool(_CONDITION_KV.match(text) or _MIX_RATIO.search(text))


def category_number(text: str) -> int | None:
    m = _NUMBERED.match(text)
    return int(to_ascii_digits(m[1])) if m else None


def same_label(a: str | None, b: str | None) -> bool:
    if a is None or b is None:
        return False
    return normalize_for_comparison(a) == normalize_for_comparison(b)


def mixed_direction(raw: str) -> bool:
    """Latin text mixed with digits or stray edge punctuation: the model's line order of
    left-to-right runs inside right-to-left text is unreliable («۵۰۰kg/m3 تا ۶۰۰»)."""
    if not _LATIN.search(raw):
        return False
    return bool(_DIGIT.search(raw)) or any(
        ln.strip().startswith(":") or ln.strip().endswith(" -") for ln in raw.split("\n")
    )


@dataclass
class Line:
    field: str  # model field the line arrived in
    label: Label
    kind: str
    multi: bool = False  # joined from several printed lines


def classify(field_name: str, lbl: Label) -> str:
    text = lbl.text or ""
    if category_number(text) is not None:
        return CATEGORY
    if field_name == "category_raw":
        return SUBGROUP
    if is_condition(text):
        return KV
    if lbl.dashed and lbl.heading_colon:
        return HEADING
    if lbl.heading_colon:
        return HEADING if field_name == "condition_raw" else SUBGROUP
    return ITEM


def row_lines(raw: dict[str, Any]) -> list[Line]:
    out: list[Line] = []
    for name in LABEL_FIELDS:
        value = _s(raw.get(name))
        if value is None:
            continue
        parts = logical_lines(value)
        for part in parts:
            lbl = clean_label(part, heading=name in ("category_raw", "subgroup_raw"))
            if lbl.text is not None:
                multi = len(parts) < len([x for x in value.split("\n") if x.strip()])
                out.append(Line(name, lbl, classify(name, lbl), multi))
    return out


# --------------------------------------------------------------------------- numbers


@dataclass
class ParsedNumber:
    value: str | None
    error: str | None = None


def strict_number(token: str) -> ParsedNumber:
    """One explicit visible number; anything below full confidence is rejected."""
    v = parse_numeric(token, source_confidence=HIGH)
    if v.normalized_value is None:
        return ParsedNumber(None, "; ".join(v.issues) or "unparseable")
    if v.confidence != HIGH or v.issues:
        return ParsedNumber(None, f"not certain: {'; '.join(v.issues)}")
    if Decimal(v.normalized_value) <= 0:
        return ParsedNumber(None, "non-positive value")
    return ParsedNumber(v.normalized_value)


@dataclass
class ParsedDensity:
    kind: str | None  # range | single | more_than | less_than | None (rejected/missing)
    qualifier: str | None  # ">" | "<" | None
    min: str | None
    max: str | None
    error: str | None = None


def _rejected(error: str | None) -> ParsedDensity:
    return ParsedDensity(None, None, None, None, error or "unparseable")


def _density_range(a: ParsedNumber, b: ParsedNumber) -> ParsedDensity:
    if a.value is None or b.value is None:
        return _rejected(a.error or b.error)
    if Decimal(a.value) > Decimal(b.value):
        return _rejected("range lower bound exceeds upper bound")
    return ParsedDensity("range", None, a.value, b.value)


def _density_qualified(q: str, v: ParsedNumber) -> ParsedDensity:
    if v.value is None:
        return _rejected(v.error)
    if q == ">":
        return ParsedDensity("more_than", q, v.value, None)
    return ParsedDensity("less_than", q, None, v.value)


def parse_density(raw: str | None) -> ParsedDensity:
    if raw is None or not raw.strip():
        return _rejected("missing")
    t = " ".join(raw.split())
    if m := _RANGE.match(t):
        return _density_range(strict_number(m[1]), strict_number(m[2]))
    if m := _QUALIFIED.match(t):
        return _density_qualified(QUALIFIERS[m[1]], strict_number(m[2]))
    if m := _SINGLE.match(t):
        v = strict_number(m[1])
        return ParsedDensity("single", None, v.value, v.value) if v.value else _rejected(v.error)
    return _rejected("density text does not match a known form")


def parse_conductivity(raw: str | None) -> ParsedNumber:
    if raw is None or not raw.strip():
        return ParsedNumber(None, "missing")
    t = " ".join(raw.split())
    if not _SINGLE.match(t):
        return ParsedNumber(None, "conductivity is not a single number")
    return strict_number(t)


# --------------------------------------------------------------------------- records


@dataclass
class Slot:
    """One inheritable label: normalized text, footnotes, raw source text, origin row."""

    text: str | None = None
    raw: str | None = None
    footnotes: list[str] = field(default_factory=list)
    origin: str | None = None


@dataclass
class Context:
    """Structure carried from row to row and from page to page."""

    category: Slot = field(default_factory=Slot)
    subgroup: Slot = field(default_factory=Slot)
    material: Slot = field(default_factory=Slot)
    leaf: list[Slot] = field(default_factory=list)  # leaf condition label(s) of the last row
    had_subheading: bool = False
    item_dashed: bool | None = None  # list style of the last material item in the subgroup
    last_category_number: int | None = None

    def reset_material(self) -> None:
        self.material, self.leaf, self.had_subheading = Slot(), [], False

    def reset_subgroup(self, subgroup: Slot) -> None:
        self.subgroup, self.item_dashed = subgroup, None
        self.reset_material()


@dataclass
class PageInput:
    doc_id: str
    pdf_page: int
    printed_page: str
    request_id: str
    model: str | None
    timestamp: str | None
    timestamp_source: str
    rows: list[dict[str, Any]]
    expected_rows: int | None = None  # rows with a conductivity value, from vector glyphs
    row_ys: list[float] | None = None


def _s(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text.strip() else None


def _slot(line: Line, rid: str) -> Slot:
    return Slot(line.label.text, line.label.raw, list(line.label.footnotes), rid)


def _flag(rec: dict[str, Any], code: str, detail: str) -> None:
    if not any(f["code"] == code and f["detail"] == detail for f in rec["suspicious"]):
        rec["suspicious"].append({"code": code, "detail": detail})


def _note(rec: dict[str, Any], line: Line, role: str) -> None:
    rec["normalization_notes"] += [f"{role} ({line.field}): {n}" for n in line.label.notes]


def _put(rec: dict[str, Any], name: str, slot: Slot, source: str | None) -> None:
    rec[name] = slot.text
    rec[f"{name}_source"] = source if slot.text else None
    rec[f"{name}_source_text"] = slot.raw
    rec[f"{name}_inherited_from"] = slot.origin if source and "inherited" in source else None


def _resolve_category(rec: dict[str, Any], ctx: Context, lines: list[Line], rid: str) -> None:
    cats = [ln for ln in lines if ln.kind == CATEGORY and ln.field == "category_raw"]
    if not cats:
        if ctx.category.text is None:
            _flag(rec, "structure_inconsistent", "no category on the row or before it")
        _put(rec, "category", ctx.category, "inherited")
        rec["category_footnote_refs"] = list(ctx.category.footnotes)
        return
    line = cats[-1]
    number = category_number(line.label.text or "")
    if not same_label(line.label.text, ctx.category.text):
        expected = None if ctx.last_category_number is None else ctx.last_category_number + 1
        if number is not None and expected is not None and number != expected:
            _flag(rec, "structure_inconsistent",
                  f"category number {number} follows {ctx.last_category_number}")  # fmt: skip
        ctx.category = _slot(line, rid)
        ctx.reset_subgroup(Slot())
        ctx.last_category_number = number
    _put(rec, "category", ctx.category, "page_heading")
    rec["category_source_text"] = line.label.raw
    rec["category_footnote_refs"] = list(ctx.category.footnotes)
    _note(rec, line, "category")


def _resolve_subgroup(rec: dict[str, Any], ctx: Context, lines: list[Line], rid: str,
                      continuation: bool) -> None:  # fmt: skip
    subs = [ln for ln in lines if ln.kind == SUBGROUP]
    if subs:
        line = subs[-1]
        if line.field == "category_raw":
            rec["normalization_notes"].append("unnumbered heading in the category field "
                                              "read as a subgroup")  # fmt: skip
            if not line.label.heading_colon:
                _flag(rec, "heading_scope_ambiguous", "unnumbered heading without ':' in "
                      "the category field")  # fmt: skip
        if not same_label(line.label.text, ctx.subgroup.text):
            ctx.reset_subgroup(_slot(line, rid))
        _put(rec, "subgroup", ctx.subgroup, "page_heading")
        rec["subgroup_source_text"] = line.label.raw
        _note(rec, line, "subgroup")
    elif continuation:
        _put(rec, "subgroup", ctx.subgroup, "inherited_previous_page")
    else:
        if ctx.subgroup.text is not None:
            rec["normalization_notes"].append(f"subgroup «{ctx.subgroup.text}» ends before "
                                              "this row")  # fmt: skip
            ctx.reset_subgroup(Slot())
        _put(rec, "subgroup", Slot(), None)
    rec["subgroup_footnote_refs"] = list(ctx.subgroup.footnotes) if rec["subgroup"] else []


def _drop_repeated_headings(rec: dict[str, Any], lines: list[Line]) -> list[Line]:
    """A numbered heading copied into the material field is not a material."""
    keep: list[Line] = []
    for ln in lines:
        if ln.kind == CATEGORY and ln.field != "category_raw":
            dup = any(same_label(ln.label.text, o.label.text) for o in lines
                      if o is not ln and o.field == "category_raw")  # fmt: skip
            if dup:
                rec["normalization_notes"].append(f"{ln.field} repeats the category heading: "
                                                  "not a material")  # fmt: skip
                _flag(rec, "duplicate_or_merged_label", f"{ln.field} repeats the category "
                      "heading")  # fmt: skip
            else:
                _flag(rec, "structure_inconsistent", f"numbered heading in {ln.field}")
            continue
        keep.append(ln)
    return keep


def _split_row_labels(lines: list[Line]) -> tuple[Line | None, list[Line], list[Line], bool]:
    """(material line, sub-headings, leaf lines, several item labels) for one row."""
    heads = [ln for ln in lines if ln.kind == HEADING]
    items = [ln for ln in lines if ln.kind == ITEM]
    kvs = [ln for ln in lines if ln.kind == KV]
    if heads:
        return heads[0], heads[1:], items + kvs, False
    own = next((ln for ln in items if ln.field != "condition_raw"), None)
    rest = [ln for ln in items if ln is not own]
    several = any(ln.field != "condition_raw" for ln in rest)
    return own, [], rest + kvs, several


def _apply_own_material(rec: dict[str, Any], ctx: Context, own: Line, rid: str) -> None:
    if own.field == "subgroup_raw" and own.kind == ITEM:
        _flag(rec, "heading_scope_ambiguous", "the model gave this item label as a subgroup "
              "heading")  # fmt: skip
    if not same_label(own.label.text, ctx.material.text):
        ctx.material, ctx.leaf, ctx.had_subheading = _slot(own, rid), [], False
    _put(rec, "material", _slot(own, rid), "row")
    _note(rec, own, "material")
    if own.label.dashed and rec["subgroup"] is None:
        _flag(rec, "structure_inconsistent", "dashed item without a subgroup heading")
    if own.kind == ITEM:
        if not own.label.dashed and ctx.item_dashed and rec["subgroup"] is not None:
            _flag(rec, "heading_scope_ambiguous", "undashed item after dashed items: the "
                  "subgroup may have ended")  # fmt: skip
        ctx.item_dashed = own.label.dashed


def _resolve_material(rec: dict[str, Any], ctx: Context, lines: list[Line], rid: str) -> None:
    own, subheads, leaves, several = _split_row_labels(lines)
    same_material = own is None or same_label(own.label.text, ctx.material.text)
    if own is not None:
        _apply_own_material(rec, ctx, own, rid)
    else:
        _put(rec, "material", ctx.material, "inherited")
    if several:
        _flag(rec, "duplicate_or_merged_label", "more than one item label on the row")
    rec["footnote_refs"] = list(ctx.material.footnotes) if rec["material"] else []
    parts: list[dict[str, Any]] = [
        {"text": ln.label.text, "role": "sub_heading", "source": "row", "raw": ln.label.raw}
        for ln in subheads
    ]
    if leaves:
        ctx.leaf = [_slot(ln, rid) for ln in leaves]
        parts += [{"text": ln.label.text, "role": "leaf", "source": "row", "raw": ln.label.raw}
                  for ln in leaves]  # fmt: skip
        if not subheads and ctx.had_subheading and same_material:
            _flag(rec, "heading_scope_ambiguous", "a sub-heading on an earlier row may apply "
                  "to this row (not inherited)")  # fmt: skip
    elif ctx.leaf:
        parts += [{"text": sl.text, "role": "leaf", "source": "inherited",
                   "inherited_from": sl.origin, "raw": sl.raw} for sl in ctx.leaf]  # fmt: skip
        rec["normalization_notes"].append("continuation row: leaf label inherited from "
                                          f"{ctx.leaf[0].origin}")  # fmt: skip
    for ln in subheads + leaves:
        _note(rec, ln, "condition")
    ctx.had_subheading = ctx.had_subheading or bool(subheads)
    rec["condition_parts"] = parts
    rec["condition"] = CONDITION_JOIN.join(p["text"] for p in parts) or None
    fns = [f for ln in subheads + leaves for f in ln.label.footnotes]
    fns += [f for sl in ([] if leaves else ctx.leaf) for f in sl.footnotes]
    rec["footnote_refs"] += [f for f in fns if f not in rec["footnote_refs"]]


def _check_labels(rec: dict[str, Any], lines: list[Line], raw: dict[str, Any]) -> None:
    if any(ln.label.footnote_ambiguous for ln in lines):
        _flag(rec, "footnote_heading_ambiguity", "footnote marker taken from a bare digit")
    for ln in lines:
        if ln.multi:
            _flag(rec, "duplicate_or_merged_label", f"{ln.field} joins several printed lines")
    for name in LABEL_FIELDS:
        value = _s(raw.get(name))
        if value is not None and mixed_direction(value):
            _flag(rec, "mixed_direction_label", f"{name}: Latin text mixed with digits or "
                  "misplaced punctuation; word order may be wrong")  # fmt: skip


def _numbers(rec: dict[str, Any], raw: dict[str, Any]) -> None:
    dens = parse_density(_s(raw.get("density_raw")))
    cond = parse_conductivity(_s(raw.get("conductivity_raw")))
    rec.update(parsed_density_kind=dens.kind, parsed_density_qualifier=dens.qualifier,
               parsed_density_min=dens.min, parsed_density_max=dens.max,
               parsed_conductivity=cond.value, numeric_rejections={})  # fmt: skip
    for name, err in (("density", dens.error), ("conductivity", cond.error)):
        if err == "missing":
            _flag(rec, "missing_numeric", f"no {name} text")
        elif err:
            rec["numeric_rejections"][name] = err
            _flag(rec, "unparseable_numeric", f"{name}: {err}")
    model = {k: _s(raw.get(k)) for k in MODEL_PARSED_FIELDS}
    ours = {"density_min_parsed": dens.min, "density_max_parsed": dens.max,
            "conductivity_parsed": cond.value}  # fmt: skip
    differs = [k for k in MODEL_PARSED_FIELDS if ours[k] is not None and model[k] != ours[k]]
    if differs:
        rec["normalization_notes"].append(f"model's own parse differs on {differs} (ignored)")


def _new_record(page: PageInput, i: int, raw: dict[str, Any], count_ok: bool) -> dict[str, Any]:
    return {
        "candidate_id": f"app07-p{page.pdf_page:03d}-r{i:02d}", "doc_id": page.doc_id,
        "pdf_page": page.pdf_page, "printed_page": page.printed_page, "source_row_order": i,
        "row_y": page.row_ys[i - 1] if page.row_ys and count_ok else None,
        "raw_ref": {"request_id": page.request_id, "row_index": i - 1},
        "raw_category": _s(raw.get("category_raw")), "raw_subgroup": _s(raw.get("subgroup_raw")),
        "raw_material": _s(raw.get("material_raw")), "raw_condition": _s(raw.get("condition_raw")),
        "raw_density_text": _s(raw.get("density_raw")),
        "raw_conductivity_text": _s(raw.get("conductivity_raw")),
        "normalization_notes": [], "suspicious": [],
    }  # fmt: skip


def normalize_page(page: PageInput, ctx: Context) -> list[dict[str, Any]]:
    """Candidate records for one page; ``ctx`` carries structure across pages."""
    out: list[dict[str, Any]] = []
    seen_heading = False
    merged_before = False
    count_ok = page.expected_rows is None or page.expected_rows == len(page.rows)
    for i, raw in enumerate(page.rows, 1):
        rec = _new_record(page, i, raw, count_ok)
        rid = rec["candidate_id"]
        lines = _drop_repeated_headings(rec, row_lines(raw))
        continuation = not seen_heading and not any(
            ln.kind in (CATEGORY, SUBGROUP, HEADING) for ln in lines
        )
        seen_heading = not continuation
        _resolve_category(rec, ctx, lines, rid)
        _resolve_subgroup(rec, ctx, lines, rid, continuation)
        _resolve_material(rec, ctx, lines, rid)
        _check_labels(rec, lines, raw)
        if merged_before:
            _flag(rec, "duplicate_or_merged_label", "the previous row's label joins several "
                  "printed lines: its last line may belong to this row")  # fmt: skip
        merged_before = any(ln.multi for ln in lines)
        _numbers(rec, raw)
        if not count_ok:
            _flag(rec, "unexpected_row_count", f"model returned {len(page.rows)} rows, the "
                  f"page has {page.expected_rows} value rows")  # fmt: skip
        rec.update(units=dict(UNITS), extraction_method="gemini_whole_page",
                   extraction_model=page.model, extraction_timestamp=page.timestamp,
                   extraction_timestamp_source=page.timestamp_source,
                   verification_status=STATUS)  # fmt: skip
        out.append(rec)
    return out
