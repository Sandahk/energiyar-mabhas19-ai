"""Promotion of the reviewed appendix 7 candidates to a curated table (Phase 2C).

Builds one curated record per candidate row from ``candidates.jsonl`` (the reviewed layer),
validates the whole table, and writes:

* ``material_thermal_properties.jsonl``: one record per printed value row, in page order;
* ``table.yaml``: source, extraction and review provenance, units, counts and rules.

The verification status of each record is its strongest review basis
(``benchmark_human_verified`` > ``project_owner_reviewed`` > ``assistant_reviewed`` >
``model_inferred``); every basis that applies is kept in ``verification.basis`` so the kinds of
review are never merged. Numbers are copied from the candidate layer after re-parsing the raw
cell text; a mismatch blocks the promotion.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from mabhas19.core.manifest import load_manifest, sha256_file
from mabhas19.core.paths import MANIFEST_PATH, REPO_ROOT
from mabhas19.tables import app07 as a7
from mabhas19.tables import app07_decisions as dec

DOC_ID = "m19-ed5-1404-app07"
TABLE_ID = "m19-ed5-1404-app07.material_thermal_properties"
RECORDS_FILE = "material_thermal_properties.jsonl"
META_FILE = "table.yaml"
SCHEMA_VERSION = 1
EXPECTED_ROWS = 207
STATUSES = frozenset(dec.BASES)


def curated_record(c: dict[str, Any]) -> dict[str, Any]:
    review = c["review"]
    changed = [{"field": ch["field"], "machine_value": c.get(f"{ch['field']}_machine"),
                "decision": ch["decision"], "basis": ch["basis"]}
               for ch in review["changed_fields"]]  # fmt: skip
    return {
        "id": c["candidate_id"],
        "doc_id": c["doc_id"],
        "source": {"pdf_page": c["pdf_page"], "printed_page": c["printed_page"],
                   "row_order": c["source_row_order"], "row_y": c["row_y"]},
        "category": {"text": c["category"],
                     "number": a7.category_number(c["category"] or ""),
                     "footnote_refs": c["category_footnote_refs"]},
        "subgroup": None if c["subgroup"] is None else {
            "text": c["subgroup"], "footnote_refs": c["subgroup_footnote_refs"]},
        "material": None if c["material"] is None else {
            "text": c["material"], "inherited_from": c["material_inherited_from"]},
        "condition": None if c["condition"] is None else {
            "text": c["condition"],
            "parts": [{"text": p["text"], "role": p["role"],
                       **({"inherited_from": p["inherited_from"]} if p.get("inherited_from")
                          else {})} for p in c["condition_parts"]]},
        "footnote_refs": c["footnote_refs"],
        "display_name": c["display_name"],
        "density": {"raw": c["raw_density_text"], "kind": c["parsed_density_kind"],
                    "qualifier": c["parsed_density_qualifier"], "min": c["parsed_density_min"],
                    "max": c["parsed_density_max"], "unit": c["units"]["density"]},
        "thermal_conductivity": {"raw": c["raw_conductivity_text"],
                                 "value": c["parsed_conductivity"],
                                 "unit": c["units"]["thermal_conductivity"]},
        "transcription": {"category": c["raw_category"], "subgroup": c["raw_subgroup"],
                          "material": c["raw_material"], "condition": c["raw_condition"]},
        "verification": {"status": review["primary_basis"], "basis": review["basis"],
                         "decisions": review["decisions"], "changed_fields": changed},
        "provenance": {"method": c["extraction_method"], "model": c["extraction_model"],
                       "extracted_at": c["extraction_timestamp"],
                       "extracted_at_source": c["extraction_timestamp_source"],
                       "raw_ref": c["raw_ref"]},
    }  # fmt: skip


def validate(records: list[dict[str, Any]]) -> list[str]:
    """Problems that block the promotion (empty list = promotable)."""
    problems: list[str] = []
    if len(records) != EXPECTED_ROWS:
        problems.append(f"{len(records)} records, expected {EXPECTED_ROWS}")
    ids = [r["id"] for r in records]
    if len(set(ids)) != len(ids):
        problems.append("duplicate record ids")
    for r in records:
        rid, v = r["id"], r["verification"]
        if v["status"] not in STATUSES:
            problems.append(f"{rid}: verification status {v['status']!r}")
        if r["category"]["number"] is None:
            problems.append(f"{rid}: category without a number")
        d, k = (
            a7.parse_density(r["density"]["raw"]),
            a7.parse_conductivity(r["thermal_conductivity"]["raw"]),
        )
        stored = (r["density"]["kind"], r["density"]["qualifier"], r["density"]["min"],
                  r["density"]["max"], r["thermal_conductivity"]["value"])  # fmt: skip
        if d.error or k.error or (d.kind, d.qualifier, d.min, d.max, k.value) != stored:
            problems.append(f"{rid}: numeric fields do not re-parse from the raw text")
    return problems


def table_metadata(records: list[dict[str, Any]], model: str) -> dict[str, Any]:
    version = load_manifest(MANIFEST_PATH).get(DOC_ID).current
    review_files = [p for p in dec.DECISION_FILES if p.is_file()]
    return {
        "schema_version": SCHEMA_VERSION,
        "table_id": TABLE_ID,
        "title": "Appendix 7 – thermal properties of common building materials",
        "doc_id": DOC_ID,
        "source": {"path": f"sources/{version.path}", "version": version.version,
                   "sha256": version.sha256, "pdf_pages": [3, 13], "printed_pages": [2, 12],
                   "authority": "official PDF; the M19_Bavaghar workbook was a secondary "
                                "cross-check only"},
        "columns": {"density": "dry density, kg/m3", "thermal_conductivity":
                    "effective thermal conductivity, W/m.K"},
        "extraction": {"method": "gemini_whole_page", "model": model,
                       "post_processing": "mabhas19.tables.app07 (deterministic)",
                       "numbers": "parsed from the raw transcription with the non-guessing "
                                  "parser; never inferred or repaired"},
        "review": {
            "files": [{"path": p.relative_to(REPO_ROOT).as_posix(), "sha256": sha256_file(p)}
                      for p in review_files],
            "benchmark_references": "eval/extraction/reference/app07-p003|p006|p013.yaml",
            "status_precedence": list(dec.BASES),
        },
        "counts": {
            "records": len(records),
            "verification_status": dict(Counter(r["verification"]["status"] for r in records)),
            "rows_with_basis": {b: sum(b in r["verification"]["basis"] for r in records)
                                for b in dec.BASES},
            "fields_changed_by_review": sum(len(r["verification"]["changed_fields"])
                                            for r in records),
        },
        "rules": {
            "display_name": "derived: (subgroup, or category name without its number) + "
                            "material; never replaces the stored hierarchy",
            "inheritance": "material continues within the same category and subgroup; a value "
                           "row without its own label continues the leaf label above (2 rows, "
                           "owner-confirmed)",
        },
        "known_limitations": [
            "footnote texts are not included; only footnote reference numbers",
            "display_name is verbose for self-contained names (e.g. metals, polymers)",
            "category 14 is kept as printed («فلزت»), see curated/source_issues.yaml SI-0006",
        ],
    }  # fmt: skip


def build_table(cand_dir: Path, model: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    lines = (cand_dir / "candidates.jsonl").read_text(encoding="utf-8").splitlines()
    records = [curated_record(json.loads(x)) for x in lines if x.strip()]
    return records, table_metadata(records, model)


def write_table(records: list[dict[str, Any]], meta: dict[str, Any], out: Path) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    rec_path, meta_path = out / RECORDS_FILE, out / META_FILE
    with rec_path.open("w", encoding="utf-8", newline="\n") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    meta_path.write_text(yaml.safe_dump(meta, allow_unicode=True, sort_keys=False, width=100),
                         encoding="utf-8", newline="\n")  # fmt: skip
    return [rec_path, meta_path]
