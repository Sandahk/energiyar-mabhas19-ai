"""Promotion of the reviewed appendix 3 table to curated files (Phase 2E).

Writes three files:

* ``occupancy_types.jsonl``: the 34 occupancy types (one per printed row position), with the
  reading and provenance from each of the four tables and the label review;
* ``max_eui.jsonl``: one record per printed EUI cell (rating x occupancy x climate class);
* ``table.yaml``: source, tables, columns, missing 5C, source issues, review, counts.

Occupancy verification status: ``assistant_reviewed`` (the position was reviewed) or
``machine_corroborated`` (not reviewed; Gemini and Tesseract agree in all four tables).
EUI values are exact PDF text-layer values; their corroboration (independent Gemini read,
strict D > C > B > A ordering) is recorded but never used to change a value.
"""

from __future__ import annotations

import json
from collections import Counter
from itertools import pairwise
from pathlib import Path
from typing import Any

import yaml

from mabhas19.core.manifest import load_manifest, sha256_file
from mabhas19.core.paths import MANIFEST_PATH, REPO_ROOT
from mabhas19.tables import app03 as a3
from mabhas19.tables import app03_decisions as dec

DOC_ID = "m19-ed5-1404-app03"
TABLE_ID = "m19-ed5-1404-app03.max_energy_use_intensity"
FILES = ("occupancy_types.jsonl", "max_eui.jsonl", "table.yaml")
CLIMATES = ("0B", "1B", "2A", "2B", "3A", "3B", "4A", "4B", "5B")
ROWS_PER_TABLE = 34


def _load(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def occupancy_status(rows: list[dict[str, Any]]) -> str | None:
    if all(r["review"]["status"] == "reviewed" for r in rows):
        bases = {b for r in rows for b in r["review"]["basis"]}
        return (
            "project_owner_reviewed" if "project_owner_reviewed" in bases else "assistant_reviewed"
        )
    if all(r["review"]["corroboration"] == "two_independent_reads_agree" for r in rows):
        return "machine_corroborated"
    return None


def occupancy_types(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for n in sorted({r["row_order"] for r in rows}):
        group = sorted((r for r in rows if r["row_order"] == n),
                       key=lambda r: a3.RATINGS.index(r["energy_rating"]))  # fmt: skip
        first = group[0]
        resolution = first.get("source_issue_resolution")
        out.append({
            "id": f"app03-occ-{n:02d}", "row_number": n,
            "label": first["occupancy"], "normalized": first["occupancy_normalized"],
            "source_issue_refs": first["source_issue_refs"],
            "source_issue_resolution": resolution,
            "readings": {r["energy_rating"]: {
                "row_id": r["row_id"], "pdf_page": r["pdf_page"],
                "printed_page": r["printed_page"], "label_bbox": r["label_bbox"],
                "transcription": r["occupancy_raw"], "second_read": r["occupancy_second_read_raw"],
                "machine_label": r.get("occupancy_machine", r["occupancy"]),
                "flags": [f["code"] for f in r["suspicious"]],
                "provenance": r["provenance"]} for r in group},
            "verification": {
                "status": occupancy_status(group),
                "basis": sorted({b for r in group for b in r["review"]["basis"]}),
                "decisions": sorted({d for r in group for d in r["review"]["decisions"]}),
                "changed_fields": [{**c, "rating": r["energy_rating"],
                                    "machine_value": r.get("occupancy_machine")}
                                   for r in group for c in r["review"]["changed_fields"]],
            },
        })  # fmt: skip
    return out


def eui_records(cells: list[dict[str, Any]], types: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_n = {t["row_number"]: t for t in types}
    values = {(c["row_order"], c["climate_class"], c["energy_rating"]): int(c["eui"])
              for c in cells}  # fmt: skip
    out = []
    for c in cells:
        t = by_n[c["row_order"]]
        chain = [values[(c["row_order"], c["climate_class"], k)] for k in a3.RATINGS]
        corroboration = []
        if c["eui_visual_read"] is not None and not c["suspicious"]:
            corroboration.append("independent_gemini_read_agrees")
        if all(x > y for x, y in pairwise(chain)):
            corroboration.append("strict_D>C>B>A")
        out.append({
            "id": c["cell_id"], "doc_id": c["doc_id"], "energy_rating": c["energy_rating"],
            "occupancy_id": t["id"], "occupancy_row_number": c["row_order"],
            "occupancy_label": t["label"], "climate_class": c["climate_class"],
            "climate_class_raw": c["climate_class_raw"],
            "max_eui": {"value": c["eui"], "raw": c["eui_raw"], "unit": c["unit"]},
            "source": {"pdf_page": c["pdf_page"], "printed_page": c["printed_page"],
                       "table_row_index": c["position"]["table_row_index"],
                       "table_col_index": c["position"]["table_col_index"],
                       "bbox": c["position"]["bbox"]},
            "verification": {"value_source": "pdf_text_layer", "value_corroboration": corroboration,
                             "occupancy_status": t["verification"]["status"]},
            "provenance": c["provenance"],
        })  # fmt: skip
    return out


def _validate_types(types: list[dict[str, Any]], rows: list[dict[str, Any]]) -> list[str]:
    problems: list[str] = []
    ratings = Counter(r["energy_rating"] for r in rows)
    if dict(ratings) != dict.fromkeys(a3.RATINGS, ROWS_PER_TABLE):
        problems.append(f"rows per rating {dict(ratings)}")
    if [t["row_number"] for t in types] != list(range(1, ROWS_PER_TABLE + 1)):
        problems.append("occupancy row numbers are not 1..34")
    for t in types:
        if t["verification"]["status"] is None:
            problems.append(f"{t['id']}: no verification status")
        if not t["label"]:
            problems.append(f"{t['id']}: empty label")
        if any(r["flags"] for r in t["readings"].values()) and not t["verification"]["decisions"]:
            problems.append(f"{t['id']}: flagged without a review decision")
        if "SI-0003" in t["source_issue_refs"] and not t["source_issue_resolution"]:
            problems.append(f"{t['id']}: SI-0003 row without a recorded resolution")
    labels_at: dict[int, set[str]] = {}
    for r in rows:
        labels_at.setdefault(r["row_order"], set()).add(r["occupancy"])
    problems += [
        f"row {n}: labels differ between ratings" for n, s in labels_at.items() if len(s) != 1
    ]
    return problems


def _validate_eui(eui: list[dict[str, Any]], machine_cells: list[dict[str, Any]]) -> list[str]:
    problems: list[str] = []
    if len(eui) != len(a3.RATINGS) * ROWS_PER_TABLE * len(CLIMATES):
        problems.append(f"{len(eui)} EUI records")
    machine = {c["cell_id"]: c for c in machine_cells}
    for e in eui:
        m = machine.get(e["id"])
        if m is None or (e["max_eui"]["raw"], e["max_eui"]["value"]) != (m["eui_raw"], m["eui"]):
            problems.append(f"{e['id']}: EUI differs from the text-layer candidate")
        if a3.parse_eui(e["max_eui"]["raw"]).value != e["max_eui"]["value"]:
            problems.append(f"{e['id']}: EUI does not re-parse from its raw text")
        if e["climate_class"] not in CLIMATES:
            problems.append(f"{e['id']}: climate class {e['climate_class']!r} (5C never filled)")
    combos = Counter((e["energy_rating"], e["occupancy_row_number"]) for e in eui)
    if any(v != len(CLIMATES) for v in combos.values()):
        problems.append("a rating/occupancy pair does not have exactly 9 climate values")
    return problems


def validate(types: list[dict[str, Any]], eui: list[dict[str, Any]],
             machine_cells: list[dict[str, Any]], rows: list[dict[str, Any]]
             ) -> list[str]:  # fmt: skip
    """Problems that block the promotion (empty list = promotable)."""
    return _validate_types(types, rows) + _validate_eui(eui, machine_cells)


def table_metadata(types: list[dict[str, Any]], eui: list[dict[str, Any]], model: str,
                   pages: dict[str, Any]) -> dict[str, Any]:  # fmt: skip
    version = load_manifest(MANIFEST_PATH).get(DOC_ID).current
    files = [p for p in dec.DECISION_FILES if p.is_file()]
    return {
        "schema_version": 1, "table_id": TABLE_ID,
        "title": "Appendix 3 – maximum energy-use intensity by occupancy, climate class and "
                 "energy rating", "doc_id": DOC_ID,
        "source": {"path": f"sources/{version.path}", "version": version.version,
                   "sha256": version.sha256, "pdf_pages": [2, 5], "printed_pages": [1, 4],
                   "authority": "official PDF"},
        "tables": {v["energy_rating"]: {"pdf_page": int(p), "printed_page": v["printed_page"]}
                   for p, v in pages.items()},
        "unit": a3.UNIT, "climate_classes": list(CLIMATES),
        "missing_climate_classes": {"5C": {"source_issue_refs": ["SI-0002", "SI-0001"],
                                           "note": "no 5C column in any table; no value exists "
                                                   "for 5C and none is produced"}},
        "extraction": {"max_eui": "PDF text layer (exact); each value assigned to the climate "
                                  "label printed at the head of its own column",
                       "climate_class": "PDF text layer header", "energy_rating":
                       "PDF text layer caption", "occupancy": f"Gemini whole-page read ({model}) "
                       "per table; Tesseract cell read as corroboration only"},
        "review": {"files": [{"path": p.relative_to(REPO_ROOT).as_posix(),
                              "sha256": sha256_file(p)} for p in files],
                   "SI-0003": "operator rows 1-2 resolved by assistant visual review: canonical "
                              "readings «≥» (row 1) and «<» (row 2); raw transcriptions and vector "
                              "evidence kept"},
        "counts": {
            "occupancy_types": len(types), "max_eui_records": len(eui),
            "occupancy_verification_status": dict(Counter(t["verification"]["status"]
                                                          for t in types)),
            "eui_value_corroboration": dict(Counter(
                c for e in eui for c in e["verification"]["value_corroboration"])),
            "occupancy_labels_changed_by_review": sum(len(t["verification"]["changed_fields"])
                                                      for t in types),
        },
        "notes": ["Each table caption carries footnote 1 stating that the values are to be "
                  "revised and added to this appendix (paraphrased, not transcribed)."],
    }  # fmt: skip


def build_table(cand_dir: Path, model: str) -> tuple[dict[str, Any], list[str]]:
    rows = _load(cand_dir / "occupancy_rows.jsonl")
    cells = _load(cand_dir / "cells.jsonl")
    summary = json.loads((cand_dir / "summary.json").read_text(encoding="utf-8"))
    types = occupancy_types(rows)
    eui = eui_records(cells, types)
    meta = table_metadata(types, eui, model, summary["pages"])
    return {"occupancy_types": types, "max_eui": eui, "meta": meta}, validate(types, eui, cells,
                                                                              rows)  # fmt: skip


def write_table(table: dict[str, Any], out: Path) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, key in (("occupancy_types.jsonl", "occupancy_types"), ("max_eui.jsonl", "max_eui")):
        path = out / name
        with path.open("w", encoding="utf-8", newline="\n") as fh:
            for r in table[key]:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        paths.append(path)
    meta = out / "table.yaml"
    meta.write_text(yaml.safe_dump(table["meta"], allow_unicode=True, sort_keys=False, width=100),
                    encoding="utf-8", newline="\n")  # fmt: skip
    return [*paths, meta]
