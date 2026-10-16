"""Promotion of the 11 main-document tables to curated/ (preview by default).

Layout (relative to the curated root)::

    tables/m19-ed5-1404-main/tables.yaml                 index of the 11 tables
    tables/m19-ed5-1404-main/table-<number>/cells.jsonl  every logical source cell
    tables/m19-ed5-1404-main/table-<number>/records.jsonl
    tables/m19-ed5-1404-main/table-<number>/table.yaml
    source_issues.yaml                                   existing file + SI-0009, SI-0010

The candidate datasets are re-validated as a whole before anything is written; any problem
blocks the promotion.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from mabhas19.tables.app02_promote import APP02_SI_0009
from mabhas19.tables.main_audit import TABLES
from mabhas19.tables.main_extract import (
    CORRECTIONS_PATH,
    DOC_ID,
    ISSUE_ISFAHAN,
    ISSUE_KERMAN,
    Grid,
    one_line,
)
from mabhas19.tables.main_validate import (
    condition_bounds,
    validate_prose_isolation,
    validate_table,
    value_objects,
)
from mabhas19.text.normalize import normalize_for_search

TOTALS = {
    "tables": 11,
    "cells": 1106,
    "records": 674,
    "numeric_values": 140,
    "inequality_bounds": 18,
    "check_mark": 299,
    "cross_mark": 181,
    "dashes": 11,
    "corrected_cells": 30,
}
ROW_NUMBER_FIELDS = ("row_number", "item_number")
NEW_ISSUES = f"""
  - id: {ISSUE_ISFAHAN}
    status: unresolved
    title: "Isfahan climate class differs between main table 19-3-2 and Appendix 2"
    locations:
      - {{doc_id: {DOC_ID}, pdf_pages: [68], table: "19-3-2", record: main-t19-3-2-r04}}
      - {{doc_id: m19-ed5-1404-app02, pdf_pages: [2], record: app02-p002-r07}}
    description: >-
      Main table 19-3-2 (printed page ۴۹), row 4 «اصفهان» (WMO code ۴۰۸۰۰۰), gives climate
      class «3B». Appendix 2, row 7 «اصفهان», gives climate class «4B». Both are read from the
      PDF text layer and confirmed visually. Both values are preserved; neither is declared
      correct and neither overrides the other.
    do_not: ["declare either class correct", "overwrite either record with the other",
             "answer an Isfahan climate-class question without citing both sources"]
    raised_on: 2026-10-16
    raised_by: "Main-table audit (Claude); confirmed by the project owner"
    former_candidate_id: "SI-0004 (candidate)"

  - id: {ISSUE_KERMAN}
    status: unresolved
    title: "Main table 19-3-2: Kerman station code is anomalous relative to the other codes"
    locations:
      - {{doc_id: {DOC_ID}, pdf_pages: [68], table: "19-3-2", record: main-t19-3-2-r21}}
    description: >-
      Row 21 «کرمان» prints the WMO station code «۴۸۸۴۱۰». The other 30 codes in the column
      begin «۴۰». The code is preserved exactly as printed.
    do_not: ["replace or correct ۴۸۸۴۱۰", "derive a code from the pattern of the other rows",
             "use the station code as a numeric value"]
    raised_on: 2026-10-16
    raised_by: "Main-table audit (Claude); confirmed by the project owner"
    former_candidate_id: "SI-0005 (candidate)"
"""


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def load_candidate(
    root: Path,
) -> dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]]:
    out = {}
    for spec in TABLES:
        d = root / spec.number
        meta = yaml.safe_load((d / "table.yaml").read_text(encoding="utf-8"))
        out[spec.number] = (_jsonl(d / "cells.jsonl"), _jsonl(d / "records.jsonl"), meta)
    return out


def totals(built: dict[str, Any]) -> dict[str, int]:
    t: Counter[str] = Counter(tables=len(built))
    for cells, records, _ in built.values():
        t["cells"] += len(cells)
        t["records"] += len(records)
        t["corrected_cells"] += sum(1 for c in cells if c["correction"] or c["applied_rules"])
        for r in records:
            vals = [v for k, f in r.items() if k not in ROW_NUMBER_FIELDS for v in value_objects(f)]
            t["numeric_values"] += sum(1 for v in vals if "token" in v)
            t["dashes"] += sum(1 for v in vals if v.get("status") == "no_printed_value")
            t["inequality_bounds"] += len(list(condition_bounds(r)))
            if "printed_symbol" in r:
                t[r["printed_symbol"]["symbol"]] += 1
    return dict(t)


def _check_corrections(built: dict[str, Any]) -> list[str]:
    """Every corrected cell: guard fingerprint, exact counts, and the approved result."""
    doc = yaml.safe_load(CORRECTIONS_PATH.read_text(encoding="utf-8"))
    decisions = {d["cell"]: d for d in doc["decisions"]}
    problems = []
    seen = set()
    for cells, _, _ in built.values():
        for c in cells:
            if not c["correction"]:
                continue
            d = decisions.get(c["cell_id"])
            before = c.get("text_rules") or c["text_before_correction"]
            if d is None or hashlib.sha256(before.encode()).hexdigest() != d["expect_text_sha256"]:
                problems.append(f"{c['cell_id']}: correction guard failed")
                continue
            text = before
            for r in d["replacements"]:
                if text.count(r["from"]) != r["count"]:
                    problems.append(f"{c['cell_id']}: replacement count changed")
                text = text.replace(r["from"], r["to"])
            if text != c["text"] or c["verification"]["status"] != "assistant_reviewed":
                problems.append(f"{c['cell_id']}: corrected text/status differs")
            seen.add(c["cell_id"])
    if seen != set(decisions):
        problems.append(f"corrections not applied: {sorted(set(decisions) - seen)}")
    return problems


def _check_no_inference(built: dict[str, Any]) -> list[str]:
    problems = []
    _, r31, _ = built["19-3-1"]
    for r in r31:
        for cond in r["criterion"]["conditions"]:
            if (
                cond["variable"].startswith("HDD")
                and cond["lower"]
                and float(cond["lower"]["value"]) >= 4000
            ):
                problems.append("an HDD > 4000 class exists")
    _, r52, _ = built["19-5-2"]
    for r in r52:
        row = r["row"]
        if "dimension" in row and (
            row.get("boundary_inclusivity") != "not_stated_in_source"
            or "operator" in json.dumps(row)
        ):
            problems.append(f"{r['record_id']}: PF band inclusivity invented")
    _, r56, _ = built["19-5-6"]
    for r in r56:
        cond = r["use"].get("area_condition")
        if cond and set(cond) != {"comparison_printed", "threshold", "unit_printed"}:
            problems.append(f"{r['record_id']}: 2000 m2 boundary inclusivity invented")
    _, _, m57 = built["19-5-7"]
    if "ambiguity" not in m57 or "formula" in m57:
        problems.append("19-5-7: the Emax/Emin ambiguity must stay unresolved")
    for t in ("19-5-3", "19-5-4"):
        cells = {c["cell_id"]: c for c in built[t][0]}
        for r in built[t][1]:
            s = r["standard"]
            if "displayed" in s and s["displayed"] != one_line(cells[s["cell"]]["text"]):
                problems.append(f"{r['record_id']}: standard number not as displayed")
    return problems


def _check_merged(built: dict[str, Any]) -> list[str]:
    problems = []
    for number, (cells, _, meta) in built.items():
        grid = Grid(cells)  # raises on any position covered twice
        if (grid.rows, grid.cols) != (meta["grid"]["rows"], meta["grid"]["cols"]):
            problems.append(f"{number}: grid does not tile the table")
        if len(grid.cover) != grid.rows * grid.cols:
            problems.append(f"{number}: grid has holes")
    return problems


def _check_issues(built: dict[str, Any], issues_text: str) -> list[str]:
    problems = []
    issues = {i["id"]: i for i in yaml.safe_load(issues_text)["issues"]}
    for iid in ("SI-0001", "SI-0002", ISSUE_ISFAHAN, ISSUE_KERMAN):
        if issues.get(iid, {}).get("status") != "unresolved":
            problems.append(f"{iid} must be unresolved")
    if issues.get("SI-0003", {}).get("status") != "resolved":
        problems.append("SI-0003 must stay resolved")
    if "typo" in issues[ISSUE_KERMAN]["description"].lower():
        problems.append(f"{ISSUE_KERMAN} must not call the code a typo")
    refs = {i for _, recs, _ in built.values() for r in recs for i in r["source_issue_refs"]}
    if not refs <= set(issues):
        problems.append(f"unknown source issues referenced: {sorted(refs - set(issues))}")
    return problems


def validate_promotion(
    built: dict[str, Any], corpus: list[dict[str, Any]], issues_text: str
) -> list[str]:
    problems = []
    got = totals(built)
    if got != TOTALS:
        problems.append(f"totals {got} != {TOTALS}")
    blocks = {b["block_id"]: b for b in corpus}
    for spec in TABLES:
        cells, records, meta = built[spec.number]
        problems += validate_table(spec.number, cells, records, meta)
        problems += validate_prose_isolation(meta["bbox"], spec.pdf_page, corpus)
        block = blocks[spec.table_block]
        if block["kind"] != "table" or block["retrieval"]["include"]:
            problems.append(f"{spec.table_block}: table block must stay out of prose retrieval")
        if meta.get("symbol_semantics", "unresolved_in_source") != "unresolved_in_source":
            problems.append(f"{spec.number}: symbol semantics must stay unresolved")
    open_flags = [
        (c["cell_id"], c["open_flags"])
        for cs, _, _ in built.values()
        for c in cs
        if c["open_flags"]
    ]
    if open_flags:
        problems.append(f"open review flags: {open_flags}")
    return (
        problems
        + _check_corrections(built)
        + _check_no_inference(built)
        + _check_merged(built)
        + _check_issues(built, issues_text)
    )


def curated_meta(meta: dict[str, Any]) -> dict[str, Any]:
    return {**meta, "status": "curated"}


# --------------------------------------------------------------------------- appendix 2 backlink

APP02_DIR = Path("tables") / "m19-ed5-1404-app02"
APP02_RECORDS = APP02_DIR / "city_climate_classes.jsonl"
APP02_META = APP02_DIR / "table.yaml"
APP02_ISFAHAN = "app02-p002-r07"


def app02_with_backlink(records_text: str, meta_text: str) -> tuple[str, str]:
    """Add the SI-0009 backlink to the Isfahan record and the table's issue list (only)."""
    lines = []
    for line in records_text.splitlines():
        rec = json.loads(line)
        if rec["id"] == APP02_ISFAHAN and ISSUE_ISFAHAN not in rec["source_issue_refs"]:
            rec["source_issue_refs"] = [*rec["source_issue_refs"], ISSUE_ISFAHAN]
            lines.append(json.dumps(rec, ensure_ascii=False))
        else:
            lines.append(line)
    meta = yaml.safe_load(meta_text)
    if APP02_SI_0009 not in meta["source_issues"]:
        meta["source_issues"].append(APP02_SI_0009)
    dumped = yaml.safe_dump(meta, allow_unicode=True, sort_keys=False, width=100)
    return "\n".join(lines) + "\n", dumped


def validate_isfahan_backlink(
    built: dict[str, Any], app02_before: str, app02_after: str, issues_text: str
) -> list[str]:
    """SI-0009 is linked from both conflicting records; values and statuses are untouched."""
    problems = []
    isfahan = normalize_for_search("اصفهان")
    main: dict[str, Any] = next(
        (r for r in built["19-3-2"][1] if r["city"]["normalized"] == isfahan), {}
    )
    if main.get("climate_class", {}).get("code") != "3B" or ISSUE_ISFAHAN not in main.get(
        "source_issue_refs", []
    ):
        problems.append("main table 19-3-2 Isfahan must be 3B and reference SI-0009")
    before = {r["id"]: r for r in map(json.loads, app02_before.splitlines())}
    after = {r["id"]: r for r in map(json.loads, app02_after.splitlines())}
    isf = after.get(APP02_ISFAHAN, {})
    if isf.get("climate_class", {}).get("code") != "4B" or ISSUE_ISFAHAN not in isf.get(
        "source_issue_refs", []
    ):
        problems.append("Appendix 2 Isfahan must stay 4B and reference SI-0009")
    for rid, old in before.items():
        new = dict(after.get(rid, {}))
        if rid == APP02_ISFAHAN:
            new["source_issue_refs"] = old["source_issue_refs"]
        if new != old:
            problems.append(f"Appendix 2 {rid} changed beyond the SI-0009 backlink")
    issues = {i["id"]: i for i in yaml.safe_load(issues_text)["issues"]}
    if issues.get(ISSUE_ISFAHAN, {}).get("status") != "unresolved":
        problems.append("SI-0009 must stay unresolved")
    return problems


def write_preview(
    built: dict[str, Any], issues_text: str, root: Path, app02: tuple[str, str]
) -> list[Path]:
    (root / APP02_DIR).mkdir(parents=True, exist_ok=True)
    (root / APP02_RECORDS).write_text(app02[0], encoding="utf-8", newline="\n")
    (root / APP02_META).write_text(app02[1], encoding="utf-8", newline="\n")
    base = root / "tables" / DOC_ID
    written = []
    index = []
    for number, (cells, records, meta) in built.items():
        d = base / f"table-{number}"
        d.mkdir(parents=True, exist_ok=True)
        for name, rows in (("cells.jsonl", cells), ("records.jsonl", records)):
            with (d / name).open("w", encoding="utf-8", newline="\n") as fh:
                for r in rows:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            written.append(d / name)
        (d / "table.yaml").write_text(
            yaml.safe_dump(curated_meta(meta), allow_unicode=True, sort_keys=False, width=100),
            encoding="utf-8",
            newline="\n",
        )
        written.append(d / "table.yaml")
        index.append(
            {
                "table": number,
                "dir": f"table-{number}",
                "pdf_page": meta["pdf_page"],
                "printed_page": meta["printed_page"],
                "records": len(records),
                "cells": len(cells),
                "source_issues": sorted(meta["source_issues"]),
            }
        )
    (base / "tables.yaml").write_text(
        yaml.safe_dump(
            {"schema_version": 1, "doc_id": DOC_ID, "tables": index, "totals": totals(built)},
            allow_unicode=True,
            sort_keys=False,
            width=100,
        ),
        encoding="utf-8",
        newline="\n",
    )
    (root / "source_issues.yaml").write_text(issues_text, encoding="utf-8", newline="\n")
    return [
        *written,
        base / "tables.yaml",
        root / "source_issues.yaml",
        root / APP02_RECORDS,
        root / APP02_META,
    ]


def issues_with_new(existing: str) -> str:
    if f"id: {ISSUE_ISFAHAN}" in existing:
        return existing
    return existing.rstrip("\n") + "\n" + NEW_ISSUES
