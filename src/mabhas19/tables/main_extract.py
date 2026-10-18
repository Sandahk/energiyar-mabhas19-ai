"""Candidate structured extraction of the 11 main-document tables (never curated).

Input: the table audit (``data/processed/main_tables/audit/<table>/cells.jsonl``, produced by
``mabhas19.cli.audit_main_tables`` from the PDF text layer and drawing/image data), the curated
main corpus (captions, footnotes, body statements) and the guarded label corrections
(``eval/extraction/review/main-tables-assistant-review.yaml``).

Per table:

* ``cells.jsonl``: every logical cell once (merged cells are one cell with a span), with its
  stable id, page, position, span, bbox, raw / repair_v1 / pre-correction / final text,
  value class, Decimal-safe parse, unit, symbol identity, flags and provenance;
* ``records.jsonl``: one record per lookup key; every field points back to cell ids, values
  keep the printed token next to the parsed value, a dash is ``no_printed_value``;
* ``table.yaml``: caption, grid, units, footnotes, source issues and table-level notes.

Nothing is inferred: no symbol meaning, no boundary inclusivity, no standard-number part order,
no value for a dash, no value outside the printed grid.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml

from mabhas19.core.paths import EVAL_DIR
from mabhas19.ingest.main_blocks import rule_b1
from mabhas19.tables.main_audit import TABLES, TableSpec, cell_flags, parse_value
from mabhas19.text.normalize import normalize_for_search, to_ascii_digits

DOC_ID = "m19-ed5-1404-main"
CORRECTIONS_PATH = EVAL_DIR / "extraction" / "review" / "main-tables-assistant-review.yaml"
FORMULA_BLOCK = "main-p073-f01"
ZWNJ = "‌"
LABEL_CLASSES = frozenset({"label", "label_with_symbol"})
SYMBOL_SEMANTICS = "unresolved_in_source"

# footnote markers printed at the end of header cells (cell id -> marker as it appears in text)
MARKERS = {
    "main-t19-2-1-r01c00": "الف",
    "main-t19-3-1-r00c00": "۱",
    "main-t19-5-2-r01c00": "3",
    "main-t19-5-2-r05c00": "الف",
}
YEARS = {"سال اول": 1, "سال دوم": 2, "سال سوم": 3, "سال چهارم": 4, "سال پنجم": 5}
ISSUE_ISFAHAN = "SI-0009"  # proposed as "SI-0004 (candidate)"; SI-0004 was already taken
ISSUE_KERMAN = "SI-0010"  # proposed as "SI-0005 (candidate)"; SI-0005 was already taken
TABLE_ISSUES = {  # 19-3-2, by normalized city name
    normalize_for_search("اصفهان"): [ISSUE_ISFAHAN],
    normalize_for_search("کرمان"): [ISSUE_KERMAN],
}
INFO_FLAGS = frozenset({"multi_line", "superscript_marker"})  # layout facts, nothing to review
ACCEPTED_FLAGS = {  # reviewed and kept as printed (project-owner decision, 2026-10-16)
    "percent_sign_order_bidi": "percent sign kept in text-layer order (e.g. «%۲۰»); any "
    "representation change belongs only to search-normalized text",
}
SOURCE_ISSUES = {
    "SI-0002": "Appendix 3 has no 5C column; main tables print a combined «5B, 5C» group "
    "(applicability recorded as printed, not extended to appendix 3)",
    ISSUE_ISFAHAN: "Isfahan: table 19-3-2 = 3B, Appendix 2 (app02-p002-r07) = 4B; unresolved, "
    "both preserved, neither declared correct",
    ISSUE_KERMAN: "Kerman WMO code printed «۴۸۸۴۱۰»; anomalous relative to the pattern of the "
    "other codes; preserved exactly as printed",
}


class ExtractionError(ValueError):
    """A correction is stale or the printed table does not have the expected structure."""


def cell_id(table: str, row: int, col: int) -> str:
    return f"main-t{table}-r{row:02d}c{col:02d}"


def one_line(text: str) -> str:
    return " ".join(x.strip() for x in text.split("\n") if x.strip())


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- cells


def load_corrections(path: Path = CORRECTIONS_PATH) -> dict[str, dict[str, Any]]:
    doc = yaml.safe_load(path.read_text(encoding="utf-8")) if path.is_file() else {}
    return {d["cell"]: {**d, "basis": doc.get("basis_default")} for d in doc.get("decisions", [])}


def _correct(cid: str, text: str, d: dict[str, Any]) -> str:
    if _sha(text) != d["expect_text_sha256"]:
        raise ExtractionError(f"{d['id']}: {cid} text changed since the review (stale)")
    for r in d["replacements"]:
        if text.count(r["from"]) != r["count"]:
            raise ExtractionError(f"{d['id']}: «{r['from']}» found {text.count(r['from'])} times")
        text = text.replace(r["from"], r["to"])
    return text


def final_flags(a: dict[str, Any], text: str) -> dict[str, list[str]]:
    """Flags of the final cell text: layout facts kept, text checks re-run after correction."""
    flags = [f for f in a["flags"] if f in INFO_FLAGS]
    if a["value_class"] in LABEL_CLASSES:
        flags += cell_flags(text, a["text_raw"], [], 1)
    else:
        flags += [f for f in a["flags"] if f not in INFO_FLAGS]
    if re.search(r"[%٪][0-9۰-۹]", text) and a["value_class"] in LABEL_CLASSES:
        flags.append("percent_sign_order_bidi")
    return {
        "flags": flags,
        "resolved_flags": [f for f in a["flags"] if f not in flags],
        "accepted_flags": [f for f in flags if f in ACCEPTED_FLAGS],
        "open_flags": [f for f in flags if f not in INFO_FLAGS and f not in ACCEPTED_FLAGS],
    }


def build_cells(
    spec: TableSpec,
    audit_cells: list[dict[str, Any]],
    printed_page: str,
    corrections: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    out = []
    for a in audit_cells:
        cid = cell_id(spec.number, a["row"], a["col"])
        text, rules, decided = a["text"], [], None
        if a["value_class"] in LABEL_CLASSES:
            text, n = rule_b1(text)
            if n:
                rules.append({"rule": "B1", "count": n})
        before = text
        if cid in corrections:
            decided = corrections[cid]
            text = _correct(cid, text, decided)
        status = (
            "assistant_reviewed" if decided else "rule_corrected" if rules else "machine_extracted"
        )
        cell = {
            "cell_id": cid,
            "doc_id": DOC_ID,
            "table": spec.number,
            "pdf_page": spec.pdf_page,
            "printed_page": printed_page,
            "row": a["row"],
            "col": a["col"],
            "rowspan": a["rowspan"],
            "colspan": a["colspan"],
            "bbox": a["bbox"],
            "text_raw": a["text_raw"],
            "text_repair_v1": a["text_repair_v1"],
            "text_before_correction": a["text"] if (rules or decided) else None,
            "text": text,
            "value_class": a["value_class"],
            "parsed": a.get("parsed"),
            "visual_ltr": a.get("visual_ltr"),
            "unit": None,
            "symbols": a.get("symbols"),
            "superscript_text": a.get("superscript_text"),
            **final_flags(a, text),
            "applied_rules": rules,
            "correction": None
            if not decided
            else {
                "id": decided["id"],
                "kind": decided["kind"],
                "basis": decided["basis"],
                "replacements": decided["replacements"],
            },
            "verification": {"status": status},
            "provenance": {
                "grid": "pymupdf find_tables(strategy=lines_strict)",
                "text": "pdf_text_layer (repair_v1 + approved rules)",
                **({"symbols": "pdf image soft-mask identity"} if a.get("symbols") else {}),
            },
        }
        if before != text or rules:
            cell["text_rules"] = before
        out.append(cell)
    return out


class Grid:
    """Cells of one table by position (a merged cell covers every position of its span)."""

    def __init__(self, cells: list[dict[str, Any]]) -> None:
        self.cells = cells
        self.cover: dict[tuple[int, int], dict[str, Any]] = {}
        for c in cells:
            for r in range(c["row"], c["row"] + c["rowspan"]):
                for k in range(c["col"], c["col"] + c["colspan"]):
                    if (r, k) in self.cover:
                        raise ExtractionError(
                            f"{c['cell_id']} overlaps {self.cover[(r, k)]['cell_id']}"
                        )
                    self.cover[(r, k)] = c
        self.rows = 1 + max(r for r, _ in self.cover)
        self.cols = 1 + max(k for _, k in self.cover)

    def at(self, row: int, col: int) -> dict[str, Any]:
        return self.cover[(row, col)]

    def full_row(self, row: int) -> bool:
        return bool(self.at(row, 0)["colspan"] == self.cols)


# --------------------------------------------------------------------------- fields


def label(c: dict[str, Any]) -> dict[str, Any]:
    text = one_line(c["text"])
    marker = MARKERS.get(c["cell_id"])
    out: dict[str, Any] = {"text": text, "cell": c["cell_id"]}
    if marker:
        stripped = text.rstrip(ZWNJ + " ")
        if not stripped.endswith(marker):
            raise ExtractionError(f"{c['cell_id']}: footnote marker «{marker}» not found")
        out["text"] = stripped[: -len(marker)].rstrip(ZWNJ + " ")
        out["footnote_marker"] = marker
    return out


def value(
    c: dict[str, Any], unit: str | None = None, unit_source: str | None = None
) -> dict[str, Any]:
    """A printed value: exact token, Decimal-safe value (numbers only), explicit unit."""
    printed = one_line(c["text"])
    out: dict[str, Any] = {
        "cell": c["cell_id"],
        "printed": printed,
        "value_class": c["value_class"],
    }
    if c["value_class"] == "dash":
        out.update(value=None, status="no_printed_value")
        return out
    p = c.get("parsed")
    if p:
        out.update(token=p["token"], value=p["value"])
        if p.get("percent_sign"):
            out["percent_sign"] = p["percent_sign"]
            unit = unit or "%"
    else:
        out["value"] = None
    if unit is not None:
        out.update(unit=unit, unit_source=unit_source or c["cell_id"])
        c["unit"] = unit
    return out


def climate_group(c: dict[str, Any]) -> dict[str, Any]:
    raw = one_line(c["text"])
    classes = re.findall(r"\d[ABC]", raw)
    if not classes:
        raise ExtractionError(f"{c['cell_id']}: no climate class in «{raw}»")
    return {
        "raw": raw,
        "applies_to": classes,
        "evidence_cell": c["cell_id"],
        "source_issue_refs": ["SI-0002"] if "5C" in classes else [],
    }


_BOUND = r"[0-9۰-۹]+(?:[/.][0-9۰-۹]+)?"
_EXPR = re.compile(
    rf"^(?:(?P<lo>{_BOUND})\s*(?P<lop><|≤)\s*)?(?P<var>[A-Za-z][A-Za-z0-9°]*)"
    rf"\s*(?:(?P<uop><|≤)\s*(?P<hi>{_BOUND}))?$"
)


def _bound(printed: str | None, op: str | None) -> dict[str, Any] | None:
    if printed is None:
        return None
    p = parse_value(printed)
    return {"printed": printed, "value": p["value"] if p else None, "operator": op}


def parse_condition(text: str) -> dict[str, Any]:
    """«a < X ≤ b», «X ≤ b», «a < X»: operators kept as printed (no inclusivity inferred)."""
    m = _EXPR.match(text.strip())
    if not m or not (m["lo"] or m["hi"]):
        raise ExtractionError(f"unparsed condition «{text}»")
    return {
        "printed": text.strip(),
        "variable": m["var"],
        "lower": _bound(m["lo"], m["lop"]),
        "upper": _bound(m["hi"], m["uop"]),
    }


def numbers_in(text: str) -> list[dict[str, Any]]:
    out = []
    for tok in re.findall(r"[0-9۰-۹]+(?:/[0-9۰-۹]+)?", text):
        p = parse_value(tok)
        out.append({"printed": tok, "value": p["value"] if p else None})
    return out


def record(table: str, rid: str, anchor: dict[str, Any], **fields: Any) -> dict[str, Any]:
    refs = sorted({v for v in _cell_refs(fields)})
    return {
        "record_id": f"main-t{table}-{rid}",
        "doc_id": DOC_ID,
        "table": table,
        **fields,
        "anchor_cell": anchor["cell_id"],
        "source_cells": refs,
        "source_issue_refs": sorted({i for i in _issue_refs(fields)}),
    }


def _cell_refs(obj: Any) -> list[str]:
    if isinstance(obj, dict):
        out = [obj[k] for k in ("cell", "evidence_cell") if isinstance(obj.get(k), str)]
        return out + [r for k, v in obj.items() if k != "source_cells" for r in _cell_refs(v)]
    if isinstance(obj, list):
        return [r for v in obj for r in _cell_refs(v)]
    return []


def _issue_refs(obj: Any) -> list[str]:
    if isinstance(obj, dict):
        return list(obj.get("source_issue_refs", [])) + [
            r for v in obj.values() if isinstance(v, dict | list) for r in _issue_refs(v)
        ]
    if isinstance(obj, list):
        return [r for v in obj for r in _issue_refs(v)]
    return []


# --------------------------------------------------------------------------- tables

Meta = dict[str, Any]
Built = tuple[list[dict[str, Any]], Meta]


def t19_2_1(g: Grid, ctx: dict[str, Any]) -> Built:
    t = "19-2-1"
    records = []
    for r in range(2, 22):
        group, year = g.at(r, 0), g.at(r, 1)
        for c in range(2, g.cols):
            sym = g.at(r, c)
            if len(sym.get("symbols") or []) != 1:
                raise ExtractionError(f"{sym['cell_id']}: expected exactly one symbol")
            req = g.at(1, c)
            topic = g.at(0, c)
            records.append(
                record(
                    t,
                    f"r{r:02d}c{c:02d}",
                    sym,
                    building_group={**label(group)},
                    year={**label(year), "index": YEARS[one_line(year["text"])]},
                    requirement={**label(req), "topic": None if topic is req else label(topic)},
                    printed_symbol={
                        "symbol": sym["symbols"][0]["symbol"],
                        "cell": sym["cell_id"],
                        "mask_digest": sym["symbols"][0]["mask_digest"],
                    },
                )
            )
    meta = {
        "symbol_semantics": SYMBOL_SEMANTICS,
        "symbol_values": ["check_mark", "cross_mark"],
        "symbol_note": "Symbols are recorded by printed shape only. Neither the table nor the "
        "surrounding text defines them; no boolean or requirement status is derived.",
        "footnotes": [{"marker": "الف", "on_cell": "main-t19-2-1-r01c00", **label(g.at(22, 0))}],
        "header_cells": [g.at(0, 0)["cell_id"], g.at(1, 0)["cell_id"], g.at(1, 1)["cell_id"]],
        "year_note": "years are counted from the notification of this edition (no calendar year)",
    }
    return records, meta


def t19_3_1(g: Grid, ctx: dict[str, Any]) -> Built:
    t = "19-3-1"
    records = []
    for r in range(1, g.rows):
        cls, name, crit = g.at(r, 0), g.at(r, 1), g.at(r, 2)
        group = climate_group(cls)
        names = [
            {"class": m[2], "name": m[1].strip(" ،")}
            for m in re.finditer(r"([^،()]+?)\s*\((\d[ABC])\)", one_line(name["text"]))
        ]
        if [n["class"] for n in names] != group["applies_to"]:
            raise ExtractionError(f"{name['cell_id']}: names do not match {group['applies_to']}")
        printed = one_line(crit["text"])
        parts = [p.strip() for p in printed.split(" AND ")]
        records.append(
            record(
                t,
                f"r{r:02d}",
                cls,
                climate_classes=group,
                climate_names={
                    "cell": name["cell_id"],
                    "text": one_line(name["text"]),
                    "by_class": names,
                },
                criterion={
                    "cell": crit["cell_id"],
                    "printed": printed,
                    "conjunction": "AND" if len(parts) > 1 else None,
                    "conditions": [parse_condition(p) for p in parts],
                },
            )
        )
    meta = {
        "header_cells": [g.at(0, k)["cell_id"] for k in range(3)],
        "footnotes": [
            {
                "marker": "۱",
                "on_cell": "main-t19-3-1-r00c00",
                "page_footnote_block": "main-p067-b005",
                "text": ctx["blocks"]["main-p067-b005"]["text"],
            }
        ],
        "notes": [
            "Operators are kept as printed; no class is defined outside the printed "
            "conditions (e.g. HDD18°C > 4000) and none is inferred.",
            "The A/B/C (humidity) split is not part of this table.",
            "Conceptually overlaps Appendix 2 (city classes): datasets are kept separate.",
        ],
    }
    return records, meta


def t19_3_2(g: Grid, ctx: dict[str, Any]) -> Built:
    t = "19-3-2"
    records = []
    for r in range(1, g.rows):
        no, prov, city, wmo, cls = (g.at(r, k) for k in range(5))
        city_text = one_line(city["text"])
        norm = normalize_for_search(city_text)
        records.append(
            record(
                t,
                f"r{r:02d}",
                no,
                row_number={**value(no), "value": int(no["parsed"]["value"])},
                province={
                    "text": one_line(prov["text"]),
                    "normalized": normalize_for_search(one_line(prov["text"])),
                    "cell": prov["cell_id"],
                },
                city={"text": city_text, "normalized": norm, "cell": city["cell_id"]},
                wmo_code={
                    "printed": one_line(wmo["text"]),
                    "ascii": to_ascii_digits(one_line(wmo["text"])),
                    "value_type": "identifier_text",
                    "cell": wmo["cell_id"],
                },
                climate_class={"code": one_line(cls["text"]), "cell": cls["cell_id"]},
                source_issue_refs=TABLE_ISSUES.get(norm, []),
            )
        )
    meta = {
        "header_cells": [g.at(0, k)["cell_id"] for k in range(5)],
        "lookup": {
            "by_city_normalized": {x["city"]["normalized"]: x["record_id"] for x in records},
            "by_wmo_code": {x["wmo_code"]["ascii"]: x["record_id"] for x in records},
        },
        "notes": [
            "Station codes are identifiers (text), never numbers.",
            f"Not merged with Appendix 2; see {ISSUE_ISFAHAN} for the Isfahan conflict.",
        ],
    }
    return records, meta


def t19_3_3(g: Grid, ctx: dict[str, Any]) -> Built:
    t = "19-3-3"
    records = [
        record(
            t,
            f"r{r:02d}",
            g.at(r, 1),
            energy_rating={"code": one_line(g.at(r, 1)["text"]), "cell": g.at(r, 1)["cell_id"]},
            r_range={"cell": g.at(r, 0)["cell_id"], **parse_condition(g.at(r, 0)["text"])},
            formula_ref=FORMULA_BLOCK,
        )
        for r in range(1, g.rows)
    ]
    f = ctx["blocks"][FORMULA_BLOCK]
    meta = {
        "header_cells": [g.at(0, 0)["cell_id"], g.at(0, 1)["cell_id"]],
        "formula": {"block": FORMULA_BLOCK, "plain": f["text"], "latex": f["formula"]["latex"]},
        "not_defined": "R > 4: no rating is printed (p74 text: rating D is at most 4x the ideal "
        "intensity). No record exists and none may be inferred.",
    }
    return records, meta


def t19_5_1(g: Grid, ctx: dict[str, Any]) -> Built:
    t = "19-5-1"
    unit_block = ctx["blocks"]["main-p084-b004"]["text"]
    unit = re.search(r"\(([^()]*)\)\s*$", unit_block)
    if not unit:
        raise ExtractionError("19-5-1: unit not found in the caption continuation")
    groups = [climate_group(g.at(0, k)) for k in range(g.cols)]
    records, element = [], None
    for r in range(1, g.rows):
        if g.full_row(r):
            element = g.at(r, 0)
            continue
        assert element is not None
        for k in range(g.cols):
            records.append(
                record(
                    t,
                    f"r{r:02d}c{k:02d}",
                    g.at(r, k),
                    element=label(element),
                    climate_group=groups[k],
                    min_r_value=value(g.at(r, k), unit[1], "main-p084-b004"),
                )
            )
    meta = {
        "unit": {
            "printed": unit[1],
            "source_block": "main-p084-b004",
            "note": "unit printed on the caption's second line",
        },
        "header_cells": [g.at(0, k)["cell_id"] for k in range(g.cols)],
    }
    return records, meta


def _metric_unit(text: str) -> tuple[str | None, str | None]:
    metric = "U-Value" if "U-Value" in text else "SHGC" if "SHGC" in text else None
    unit = re.search(r"U-Value\s*\(([^()]*)\)", text)
    return metric, unit[1] if unit else None


def t19_5_2(g: Grid, ctx: dict[str, Any]) -> Built:
    t = "19-5-2"
    groups = {k: climate_group(g.at(0, k)) for k in range(1, g.cols)}
    records: list[dict[str, Any]] = []
    section: dict[str, Any] | None = None
    row_dimension: dict[str, Any] | None = None
    footnotes = []
    for r in range(1, g.rows):
        first = g.at(r, 0)
        if g.full_row(r):
            if one_line(first["text"]).startswith("الف"):
                footnotes.append(
                    {"marker": "الف", "on_cell": "main-t19-5-2-r05c00", **label(first)}
                )
            else:
                section, row_dimension = first, None
            continue
        if g.at(r, 1)["colspan"] == g.cols - 1:  # sub-header: row dimension + metric header
            section, row_dimension = g.at(r, 1), first
            continue
        assert section is not None
        metric, unit = _metric_unit(one_line(first["text"]))
        unit_cell = first
        if metric is None:
            metric, unit = _metric_unit(one_line(section["text"]))
            unit_cell = section
        row = label(first)
        if row_dimension is not None:
            row = {
                **row,
                "dimension": label(row_dimension),
                "numbers_printed": numbers_in(row["text"]),
                "boundary_inclusivity": "not_stated_in_source",
            }
        for k in range(1, g.cols):
            records.append(
                record(
                    t,
                    f"r{r:02d}c{k:02d}",
                    g.at(r, k),
                    section=label(section),
                    row=row,
                    metric=metric,
                    climate_group=groups[k],
                    max_value=value(g.at(r, k), unit, unit_cell["cell_id"] if unit else None),
                )
            )
    meta = {
        "header_cells": [g.at(0, 0)["cell_id"]] + [groups[k]["evidence_cell"] for k in groups],
        "footnotes": [
            *footnotes,
            {
                "marker": "3",
                "on_cell": "main-t19-5-2-r01c00",
                "page_footnote_block": "main-p090-b009",
                "text": ctx["blocks"]["main-p090-b009"]["text"],
            },
        ],
        "units": {
            "U-Value": "W/m2.k (printed in the section/row label)",
            "SHGC": "no unit printed",
        },
        "notes": [
            "PF band labels are kept as printed (percent sign order as in the text layer); "
            "boundary inclusivity is not stated and not inferred."
        ],
    }
    return records, meta


def _metric(c: dict[str, Any], spec: str) -> tuple[str | None, str]:
    kind = c["value_class"]
    if kind == "percent":
        return "percent", "percent sign printed in the value cell"
    if kind == "category" and one_line(c["text"]).startswith("IE"):
        return "IE_efficiency_class", "IE class printed in the value cell"
    if kind == "number":
        named = [m for m in ("COP", "EEI", "SEER") if m in spec]
        if len(named) == 1:
            return named[0], f"specification text names {named[0]}"
        return None, "metric not printed"
    return None, "no printed value" if kind == "dash" else "not a metric value"


def _standard(c: dict[str, Any]) -> dict[str, Any]:
    printed = one_line(c["text"])
    if c["value_class"] == "dash":
        return {
            "cell": c["cell_id"],
            "printed": printed,
            "value": None,
            "status": "no_printed_value",
        }
    return {
        "cell": c["cell_id"],
        "displayed": printed,
        "text_raw": c["text_raw"].strip(),
        "part_order": "ambiguous_bidi" if "-" in printed else "single_part",
        "note": "kept exactly as displayed; component order not inferred",
    }


def _equipment_tables(t: str) -> Callable[[Grid, dict[str, Any]], Built]:
    def build(g: Grid, ctx: dict[str, Any]) -> Built:
        records = []
        for r in range(1, g.rows):
            no = g.at(r, 0)
            sub = r - no["row"] + 1 if no["rowspan"] > 1 else None
            spec_cell, eff = g.at(r, 2), g.at(r, 3)
            spec = one_line(spec_cell["text"])
            metric, basis = _metric(eff, spec)
            ambiguities = []
            if re.search(r"(^|\s)یا(\s|$)", spec) and numbers_in(spec):
                ambiguities.append(
                    "alternative criterion in the specification («یا»); the printed "
                    "minimum is not assigned to either criterion beyond the wording"
                )
            rid = f"item{int(no['parsed']['value']):02d}" + (f"-s{sub}" if sub else "")
            records.append(
                record(
                    t,
                    rid,
                    eff,
                    item_number={**value(no), "value": int(no["parsed"]["value"])},
                    sub_row=sub,
                    equipment=label(g.at(r, 1)),
                    specification={
                        "cell": spec_cell["cell_id"],
                        "text": spec,
                        "numbers_printed": numbers_in(spec),
                        **(
                            {"status": "no_printed_value"}
                            if spec_cell["value_class"] == "dash"
                            else {}
                        ),
                    },
                    metric=metric,
                    metric_basis=basis,
                    threshold=value(eff),
                    standard=_standard(g.at(r, 4)),
                    energy_rating=value(g.at(r, 5)),
                    ambiguities=ambiguities,
                )
            )
        meta = {
            "header_cells": [g.at(0, k)["cell_id"] for k in range(g.cols)],
            "notes": [
                "Standard numbers are kept exactly as displayed; the order of their "
                "components is bidi-ambiguous and is not inferred.",
                "Dash = no printed value (never zero).",
                "metric is taken only from the printed % sign, IE class or a metric "
                "named in the specification text (COP/EEI/SEER).",
            ],
        }
        return records, meta

    return build


def _header_unit(g: Grid, col: int, printed: str) -> str:
    text = one_line(g.at(0, col)["text"])
    if printed not in text:
        raise ExtractionError(f"unit «{printed}» not in header «{text}»")
    return printed


def t19_5_5(g: Grid, ctx: dict[str, Any]) -> Built:
    t = "19-5-5"
    unit = _header_unit(g, 2, "درجه سانتیگراد")
    hid = g.at(0, 2)["cell_id"]
    records = [
        record(
            t,
            f"r{r:02d}",
            g.at(r, 2),
            row_number={**value(g.at(r, 0)), "value": int(g.at(r, 0)["parsed"]["value"])},
            place_of_use=label(g.at(r, 1)),
            max_temperature=value(g.at(r, 2), unit, hid),
        )
        for r in range(1, g.rows)
    ]
    return records, {
        "header_cells": [g.at(0, k)["cell_id"] for k in range(3)],
        "unit": {"printed": unit, "source_cell": hid},
    }


_AREA = re.compile(r"(کوچک‌تر|بزرگتر|بزرگ‌تر) از ([۰-۹]+) (متر مربع)")


def t19_5_6(g: Grid, ctx: dict[str, Any]) -> Built:
    t = "19-5-6"
    unit = _header_unit(g, 1, "در صد")
    hid = g.at(0, 1)["cell_id"]
    records = []
    for r in range(1, g.rows):
        use = label(g.at(r, 0))
        m = _AREA.search(use["text"])
        if m:
            use["area_condition"] = {
                "comparison_printed": m[1],
                "threshold": numbers_in(m[2])[0],
                "unit_printed": m[3],
            }
        records.append(
            record(
                t, f"r{r:02d}", g.at(r, 1), use=use, min_daylight_area=value(g.at(r, 1), unit, hid)
            )
        )
    meta = {
        "header_cells": [g.at(0, 0)["cell_id"], hid],
        "unit": {"printed": unit, "source_cell": hid},
        "notes": [
            "An area of exactly 2000 m2 is covered by neither printed label; not inferred.",
            "Dash = no printed value (never zero).",
        ],
    }
    return records, meta


def t19_5_7(g: Grid, ctx: dict[str, Any]) -> Built:
    t = "19-5-7"
    records = [
        record(
            t, f"r{r:02d}", g.at(r, 1), space=label(g.at(r, 0)), min_uniformity=value(g.at(r, 1))
        )
        for r in range(1, g.rows)
    ]
    b = ctx["blocks"]
    meta = {
        "header_cells": [g.at(0, 0)["cell_id"], g.at(0, 1)["cell_id"]],
        "unit": None,
        "source_statements": [
            {
                "source": "table caption",
                "block": "main-p116-b007",
                "text": b["main-p116-b007"]["text"],
            },
            {
                "source": "body text (item ذ)",
                "block": "main-p116-b006",
                "text": b["main-p116-b006"]["text"],
            },
        ],
        "ambiguity": "The caption prints «U=Emax (Emin) / Eavg»; the body text says the ratio of "
        "the brightest OR dimmest point to the average. Not resolved; no formula is "
        "chosen.",
    }
    return records, meta


BUILDERS: dict[str, Callable[[Grid, dict[str, Any]], Built]] = {
    "19-2-1": t19_2_1,
    "19-3-1": t19_3_1,
    "19-3-2": t19_3_2,
    "19-3-3": t19_3_3,
    "19-5-1": t19_5_1,
    "19-5-2": t19_5_2,
    "19-5-3": _equipment_tables("19-5-3"),
    "19-5-4": _equipment_tables("19-5-4"),
    "19-5-5": t19_5_5,
    "19-5-6": t19_5_6,
    "19-5-7": t19_5_7,
}
HEADER_ROWS = {"19-2-1": 2}
ROW_LABEL_COLS = {
    "19-2-1": [0, 1],
    "19-3-2": [0],
    "19-5-1": [],
    "19-5-2": [0],
    "19-5-3": [0, 1],
    "19-5-4": [0, 1],
    "19-5-5": [0],
    "19-3-1": [0],
    "19-3-3": [1],
    "19-5-6": [0],
    "19-5-7": [0],
}


CAPTION_CONTINUATIONS = {"19-5-1": ["main-p084-b004"]}  # second caption line (unit)
CAPTION_NOTE = "curated main-corpus caption text (main-corpus errata of 2026-10-17 applied)"


def caption(spec: TableSpec, blocks: dict[str, dict[str, Any]]) -> dict[str, Any]:
    parts = [spec.caption_block, *CAPTION_CONTINUATIONS.get(spec.number, [])]
    out: dict[str, Any] = {"text": " ".join(blocks[p]["text"] for p in parts), "blocks": parts}
    if len(parts) > 1:
        out["parts"] = [{"block": p, "kind": blocks[p]["kind"]} for p in parts]
    out["note"] = CAPTION_NOTE
    return out


def table_meta(
    spec: TableSpec, grid: Grid, records: list[dict[str, Any]], meta: Meta, ctx: dict[str, Any]
) -> dict[str, Any]:
    cells = grid.cells
    issues = sorted({i for r in records for i in r["source_issue_refs"]})
    statuses: dict[str, int] = {}
    for c in cells:
        s = c["verification"]["status"]
        statuses[s] = statuses.get(s, 0) + 1
    return {
        "schema_version": 1,
        "status": "candidate",
        "table_id": f"{DOC_ID}.t{spec.number}",
        "doc_id": DOC_ID,
        "table_number": spec.number,
        "caption": caption(spec, ctx["blocks"]),
        "pdf_page": spec.pdf_page,
        "printed_page": ctx["printed_page"],
        "corpus_blocks": {"caption": spec.caption_block, "table": spec.table_block},
        "bbox": ctx["bbox"],
        "grid": {
            "rows": grid.rows,
            "cols": grid.cols,
            "column_order": "right_to_left (col 0 = rightmost)",
            "header_rows": HEADER_ROWS.get(spec.number, 1),
            "row_label_cols": ROW_LABEL_COLS[spec.number],
        },
        "source": ctx["source"],
        "methods": {
            "grid": "pymupdf find_tables(strategy=lines_strict)",
            "text": "PDF text layer, repair_v1 (+ B1); labels: guarded assistant "
            "corrections from source crops",
            "numbers": "Decimal-safe parse of the printed token ('/' decimal separator "
            "kept in the printed text)",
            "symbols": "PDF image soft-mask identity" if spec.number == "19-2-1" else None,
        },
        "review": ctx["review"],
        "counts": {
            "cells": len(cells),
            "records": len(records),
            "cell_verification_status": statuses,
            "corrected_label_cells": sum(c["correction"] is not None for c in cells),
        },
        "source_issues": {i: SOURCE_ISSUES[i] for i in issues},
        **meta,
    }


def decimal_ok(token: str, value: str) -> bool:
    return Decimal(to_ascii_digits(token).replace("/", ".")) == Decimal(value)


TABLE_SPECS = {t.number: t for t in TABLES}
