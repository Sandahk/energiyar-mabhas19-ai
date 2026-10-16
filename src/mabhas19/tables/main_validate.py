"""Validators of the candidate main-table datasets (empty problem list = valid)."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterator
from decimal import Decimal
from typing import Any

import pymupdf

from mabhas19.tables.main_audit import page_symbols
from mabhas19.tables.main_extract import ISSUE_ISFAHAN, ISSUE_KERMAN, decimal_ok, one_line
from mabhas19.text.normalize import normalize_for_search, to_ascii_digits

# table -> (grid rows, grid cols, records)
EXPECTED = {
    "19-2-1": (23, 26, 480),
    "19-3-1": (7, 3, 6),
    "19-3-2": (32, 5, 31),
    "19-3-3": (5, 2, 4),
    "19-5-1": (21, 6, 60),
    "19-5-2": (14, 6, 45),
    "19-5-3": (20, 6, 19),
    "19-5-4": (12, 6, 11),
    "19-5-5": (5, 3, 4),
    "19-5-6": (8, 2, 7),
    "19-5-7": (8, 2, 7),
}
EXPECTED_UNITS: dict[str, set[str]] = {
    "19-5-1": {"m2.k/W"},
    "19-5-2": {"W/m2.k"},
    "19-5-3": {"%"},
    "19-5-4": {"%"},
    "19-5-5": {"درجه سانتیگراد"},
    "19-5-6": {"در صد"},
}
EXPECTED_SYMBOLS = {"check_mark": 299, "cross_mark": 181}


def value_objects(obj: Any) -> Iterator[dict[str, Any]]:
    if isinstance(obj, dict):
        if "printed" in obj and "value_class" in obj and "cell" in obj:
            yield obj
        for v in obj.values():
            yield from value_objects(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from value_objects(v)


def condition_bounds(record: dict[str, Any]) -> Iterator[tuple[str, dict[str, Any]]]:
    """(cell id, bound) of every printed inequality bound (19-3-1 criteria, 19-3-3 R ranges)."""
    holders = [record.get("r_range"), record.get("criterion")]
    for h in (x for x in holders if x):
        for cond in h.get("conditions", [h]):
            for side in ("lower", "upper"):
                if cond.get(side):
                    yield h["cell"], cond[side]


def _digits(text: str) -> str:
    return re.sub(r"\D", "", to_ascii_digits(text))


def _check_number(rid: str, c: dict[str, Any], v: dict[str, Any]) -> list[str]:
    problems = []
    if v["token"] not in c["text"] or not decimal_ok(v["token"], v["value"]):
        problems.append(f"{rid}: value {v['value']} != «{v['token']}»")
    # content-stream order is bidi-scrambled: the raw text must hold the same digits, and the
    # glyphs read left to right must give exactly the printed token
    if sorted(_digits(c["text_raw"])) != sorted(_digits(v["token"])):
        problems.append(f"{rid}: raw digits «{c['text_raw']}» != token")
    if _digits(c.get("visual_ltr") or "") != _digits(v["token"]):
        problems.append(f"{rid}: glyph order «{c.get('visual_ltr')}» != token")
    return problems


def _check_value(rid: str, c: dict[str, Any], v: dict[str, Any]) -> list[str]:
    if v["printed"] != one_line(c["text"]):
        return [f"{rid}: printed «{v['printed']}» is not the cell text"]
    if v["value_class"] == "dash":
        ok = v["value"] is None and v.get("status") == "no_printed_value"
        return [] if ok else [f"{rid}: dash with a value"]
    if "token" in v:
        return _check_number(rid, c, v)
    if v["value"] is not None and not isinstance(v["value"], int):
        return [f"{rid}: value without a printed token"]
    return []


def _check_bound(rid: str, c: dict[str, Any], bound: dict[str, Any]) -> list[str]:
    ok = (
        bound["printed"] in c["text"]
        and bound["value"] is not None
        and decimal_ok(bound["printed"], bound["value"])
        and bound["operator"] in ("<", "≤")
    )
    return [] if ok else [f"{rid}: bound «{bound['printed']}» does not round-trip"]


def _check_values(
    cells: dict[str, dict[str, Any]], records: list[dict[str, Any]], table: str
) -> list[str]:
    problems: list[str] = []
    units: set[str] = set()
    for r in records:
        for v in value_objects(r):
            problems += _check_value(r["record_id"], cells[v["cell"]], v)
            if v.get("unit"):
                units.add(v["unit"])
        for cid, bound in condition_bounds(r):
            problems += _check_bound(r["record_id"], cells[cid], bound)
    if units != EXPECTED_UNITS.get(table, set()):
        problems.append(f"units {sorted(units)} != {sorted(EXPECTED_UNITS.get(table, set()))}")
    return problems


def _check_refs(cells: dict[str, dict[str, Any]], records: list[dict[str, Any]]) -> list[str]:
    problems = []
    for r in records:
        anchor = cells.get(r["anchor_cell"])
        if anchor is None:
            problems.append(f"{r['record_id']}: unknown anchor")
            continue
        for ref in r["source_cells"]:
            c = cells.get(ref)
            if c is None:
                problems.append(f"{r['record_id']}: unknown cell {ref}")
                continue
            rows = range(c["row"], c["row"] + c["rowspan"])
            cols = range(c["col"], c["col"] + c["colspan"])
            if c["rowspan"] > 1 and anchor["row"] not in rows and rows.stop > anchor["row"]:
                problems.append(f"{r['record_id']}: merged {ref} does not cover the record row")
            if c["colspan"] > 1 and anchor["col"] not in cols:
                problems.append(f"{r['record_id']}: merged {ref} does not cover the record column")
    return problems


def _check_coverage(
    cells: dict[str, dict[str, Any]], records: list[dict[str, Any]], meta: dict[str, Any]
) -> list[str]:
    used = {ref for r in records for ref in r["source_cells"]}
    used |= set(meta.get("header_cells", []))
    used |= {f["cell"] for f in meta.get("footnotes", []) if "cell" in f}
    missing = [
        cid
        for cid, c in cells.items()
        if (c["text"].strip() or c.get("symbols")) and cid not in used
    ]
    return [f"cells not covered by any record or metadata: {missing}"] if missing else []


def _t19_2_1(records: list[dict[str, Any]], meta: dict[str, Any]) -> list[str]:
    problems = []
    got = Counter(r["printed_symbol"]["symbol"] for r in records)
    if dict(got) != EXPECTED_SYMBOLS:
        problems.append(f"symbols {dict(got)} != {EXPECTED_SYMBOLS}")
    if meta.get("symbol_semantics") != "unresolved_in_source":
        problems.append("symbol_semantics must be unresolved_in_source")
    if any(k in r for r in records for k in ("required", "active", "in_force", "boolean")):
        problems.append("symbol mapped to a boolean")
    return problems


def _t19_3_1(records: list[dict[str, Any]], meta: dict[str, Any]) -> list[str]:
    multi = [r for r in records if r["criterion"]["conjunction"] == "AND"]
    if len(multi) != 1 or len(multi[0]["criterion"]["conditions"]) != 2:
        return ["the AND row must keep two explicit conditions"]
    return []


def _issue_kept(rec: dict[str, Any], issue: str) -> bool:
    return issue in rec.get("source_issue_refs", [])


def _t19_3_2(records: list[dict[str, Any]], meta: dict[str, Any]) -> list[str]:
    problems = []
    by_city = {r["city"]["normalized"]: r for r in records}
    isf = by_city.get(normalize_for_search("اصفهان"), {})
    ker = by_city.get(normalize_for_search("کرمان"), {})
    if isf.get("climate_class", {}).get("code") != "3B" or not _issue_kept(isf, ISSUE_ISFAHAN):
        problems.append(f"Isfahan must stay 3B with {ISSUE_ISFAHAN}")
    if ker.get("wmo_code", {}).get("printed") != "۴۸۸۴۱۰" or not _issue_kept(ker, ISSUE_KERMAN):
        problems.append(f"Kerman code must stay ۴۸۸۴۱۰ with {ISSUE_KERMAN}")
    for key in ("by_city_normalized", "by_wmo_code"):
        if len(meta["lookup"][key]) != len(records):
            problems.append(f"lookup {key} is not unique")
    return problems


def _t19_3_3(records: list[dict[str, Any]], meta: dict[str, Any]) -> list[str]:
    problems = []
    if any(r["formula_ref"] != "main-p073-f01" for r in records):
        problems.append("R records must link main-p073-f01")
    uppers = [r["r_range"]["upper"] for r in records]
    if any(u is None for u in uppers) or max(Decimal(u["value"]) for u in uppers) != 4:
        problems.append("no rating may exist beyond the printed R ≤ 4")
    return problems


def _t19_5_7(records: list[dict[str, Any]], meta: dict[str, Any]) -> list[str]:
    if len(meta.get("source_statements", [])) != 2:
        return ["caption formula and body wording must both be kept"]
    return []


SPECIFIC = {"19-2-1": _t19_2_1, "19-3-1": _t19_3_1, "19-3-2": _t19_3_2, "19-3-3": _t19_3_3,
            "19-5-7": _t19_5_7}  # fmt: skip


def _check_specific(table: str, records: list[dict[str, Any]], meta: dict[str, Any]) -> list[str]:
    problems = SPECIFIC[table](records, meta) if table in SPECIFIC else []
    for r in records:
        for grp in [v for k, v in r.items() if k in {"climate_group", "climate_classes"}]:
            if "5C" in grp["applies_to"] and "SI-0002" not in grp["source_issue_refs"]:
                problems.append(f"{r['record_id']}: 5C applicability without SI-0002")
    return problems


def validate_table(
    table: str, cells: list[dict[str, Any]], records: list[dict[str, Any]], meta: dict[str, Any]
) -> list[str]:
    rows, cols, n = EXPECTED[table]
    by_id = {c["cell_id"]: c for c in cells}
    problems = []
    if (meta["grid"]["rows"], meta["grid"]["cols"]) != (rows, cols):
        problems.append(f"grid {meta['grid']['rows']}x{meta['grid']['cols']} != {rows}x{cols}")
    if len(records) != n:
        problems.append(f"{len(records)} records, expected {n}")
    if len(by_id) != len(cells):
        problems.append("duplicate cell ids")
    if len({r["record_id"] for r in records}) != len(records):
        problems.append("duplicate record ids")
    problems += _check_refs(by_id, records)
    problems += _check_values(by_id, records, table)
    problems += _check_coverage(by_id, records, meta)
    problems += _check_specific(table, records, meta)
    return [f"{table}: {p}" for p in problems]


def validate_symbol_placements(page: pymupdf.Page, cells: list[dict[str, Any]]) -> list[str]:
    """Table 19-2-1: every symbol placement (content stream) against PyMuPDF's image boxes and
    against the symbol recorded in the cell that contains its ink."""
    symbols = page_symbols(page)
    mine = sorted(tuple(round(v, 1) for v in s["rect"]) for s in symbols)
    theirs = sorted(
        tuple(round(v, 1) for v in d["bbox"])
        for d in page.get_image_info()
        if d["width"] == 2 and d["height"] == 2
    )
    problems = []
    if len(mine) != 480 or mine != theirs:
        problems.append(f"19-2-1: placements {len(mine)} do not match PyMuPDF image boxes")
    for s in symbols:
        centre = (s["ink"].tl + s["ink"].br) / 2
        hit = [c for c in cells if pymupdf.Rect(c["bbox"]).contains(centre)]
        if len(hit) != 1 or [x["symbol"] for x in hit[0].get("symbols") or []] != [s["symbol"]]:
            problems.append(f"19-2-1: placement at {s['rect']} not matched to one cell symbol")
    return problems


def validate_prose_isolation(
    table_bbox: list[float], pdf_page: int, blocks: list[dict[str, Any]]
) -> list[str]:
    """No retrieval-eligible corpus block may sit inside the table region."""
    rect = pymupdf.Rect(table_bbox)
    leaked = []
    for b in blocks:
        if b["pdf_page"] != pdf_page or not b["retrieval"]["include"]:
            continue
        r = pymupdf.Rect(b["bbox"])
        if rect.contains((r.tl + r.br) / 2):
            leaked.append(b["block_id"])
    return [f"page {pdf_page}: retrievable blocks inside the table {leaked}"] if leaked else []
