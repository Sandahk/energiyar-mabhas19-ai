"""Promotion of the reviewed appendix 2 candidates to a curated table (Phase 2D).

Writes ``city_climate_classes.jsonl`` (one record per printed row) and ``table.yaml``
(source, extraction and review provenance, counts, source issues). Validation blocks the
promotion if a row is missing, a climate code differs from the PDF text layer, a code is
not valid, a flagged row is unreviewed, a normalized city name is duplicated, or a row has
no verification status.

Verification status per record:

* ``assistant_reviewed``: the city name was checked against the source crop in review;
* ``machine_corroborated``: not reviewed, but two independent visual reads of the city name
  (Gemini, Tesseract) agree exactly and nothing was flagged.

Climate codes and row numbers always come from the PDF text layer (exact source characters).
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from mabhas19.core.manifest import load_manifest, sha256_file
from mabhas19.core.paths import MANIFEST_PATH, REPO_ROOT
from mabhas19.tables import app02 as a2
from mabhas19.tables import app02_decisions as dec

DOC_ID = "m19-ed5-1404-app02"
TABLE_ID = "m19-ed5-1404-app02.city_climate_classes"
RECORDS_FILE = "city_climate_classes.jsonl"
META_FILE = "table.yaml"
EXPECTED_ROWS = 76
STATUSES = frozenset({"assistant_reviewed", "project_owner_reviewed", "machine_corroborated"})


def status_of(c: dict[str, Any]) -> str | None:
    basis = c["review"]["basis"]
    for b in ("project_owner_reviewed", "assistant_reviewed"):
        if b in basis:
            return b
    if c["review"]["corroboration"] == "two_independent_reads_agree":
        return "machine_corroborated"
    return None


def curated_record(c: dict[str, Any]) -> dict[str, Any]:
    changed = [{"field": ch["field"], "machine_value": c.get("city_machine"),
                "decision": ch["decision"], "basis": ch["basis"]}
               for ch in c["review"]["changed_fields"]]  # fmt: skip
    return {
        "id": c["candidate_id"], "doc_id": c["doc_id"],
        "source": {"pdf_page": c["pdf_page"], "printed_page": c["printed_page"],
                   "row_order": c["row_order"], "row_y": c["row_y"]},
        "row_number": c["row_number"],
        "city": {"text": c["city"], "normalized": c["city_normalized"],
                 "transcription": c["city_raw"], "second_read": c["city_second_read_raw"]},
        "climate_class": {"code": c["climate_code"], "raw": c["climate_code_raw"],
                          "source": "pdf_text_layer"},
        "source_issue_refs": c["source_issue_refs"],
        "verification": {"status": status_of(c), "basis": c["review"]["basis"],
                         "corroboration": c["review"]["corroboration"],
                         "decisions": c["review"]["decisions"], "changed_fields": changed},
        "provenance": {"row_number": "pdf_text_layer", "climate_class": "pdf_text_layer",
                       "city": c["provenance"]["city"],
                       "city_second_read": c["provenance"]["city_second_read"]},
    }  # fmt: skip


def validate(records: list[dict[str, Any]], machine: list[dict[str, Any]]) -> list[str]:
    problems: list[str] = []
    if len(records) != EXPECTED_ROWS:
        problems.append(f"{len(records)} records, expected {EXPECTED_ROWS}")
    if [r["row_number"] for r in records] != list(range(1, len(records) + 1)):
        problems.append("row numbers are not 1..N in order")
    by_id = {m["candidate_id"]: m for m in machine}
    names = Counter(r["city"]["normalized"] for r in records)
    for r in records:
        rid, m = r["id"], by_id.get(r["id"])
        code = a2.normalize_code(r["climate_class"]["raw"])
        if m is None or r["climate_class"]["raw"] != m["climate_code_raw"]:
            problems.append(f"{rid}: climate code raw text differs from the text layer")
        if not code.valid or code.value != r["climate_class"]["code"]:
            problems.append(f"{rid}: climate code invalid or not the cleaned raw text")
        if r["verification"]["status"] not in STATUSES:
            problems.append(f"{rid}: no valid verification status")
        if m is not None and m["suspicious"] and not r["verification"]["decisions"]:
            problems.append(f"{rid}: flagged row without a review decision")
        if not r["city"]["text"] or names[r["city"]["normalized"]] > 1:
            problems.append(f"{rid}: empty or duplicated city name")
        if r["climate_class"]["code"] == "5C" and r["source_issue_refs"] != ["SI-0001", "SI-0002"]:
            problems.append(f"{rid}: 5C row must carry SI-0001 and SI-0002")
    return problems


def table_metadata(records: list[dict[str, Any]], model: str) -> dict[str, Any]:
    version = load_manifest(MANIFEST_PATH).get(DOC_ID).current
    files = [p for p in dec.DECISION_FILES if p.is_file()]
    return {
        "schema_version": 1, "table_id": TABLE_ID,
        "title": "Appendix 2 – climate classification of Iranian cities", "doc_id": DOC_ID,
        "source": {"path": f"sources/{version.path}", "version": version.version,
                   "sha256": version.sha256, "pdf_pages": [2, 4], "printed_pages": [1, 3],
                   "authority": "official PDF"},
        "columns": {"row_number": "printed row number (ردیف)", "city": "city name (نام شهر)",
                    "climate_class": "climate class (رده اقلیمی), ASHRAE 169 style code"},
        "extraction": {"row_number": "PDF text layer", "climate_class":
                       "PDF text layer; whitespace removed, never remapped",
                       "city": f"Gemini whole-page read ({model}); Tesseract cell read as an "
                               "independent second read"},
        "review": {"files": [{"path": p.relative_to(REPO_ROOT).as_posix(),
                              "sha256": sha256_file(p)} for p in files]},
        "counts": {
            "records": len(records),
            "verification_status": dict(Counter(r["verification"]["status"] for r in records)),
            "climate_class": dict(sorted(Counter(r["climate_class"]["code"]
                                                 for r in records).items())),
            "city_names_changed_by_review": sum(bool(r["verification"]["changed_fields"])
                                                for r in records),
        },
        "source_issues": ["SI-0001: row 5 «اردبیل» is classed 5C, kept as printed",
                          "SI-0002: appendix 3 has no 5C column"],
        "notes": ["PDF p4 prints a note under the table (paraphrased, not transcribed): "
                  "classes for many more weather stations are being analysed and will be "
                  "added per ANSI/ASHRAE 169-2020."],
    }  # fmt: skip


def build_table(cand_dir: Path, model: str) -> tuple[list[dict[str, Any]], dict[str, Any],
                                                     list[dict[str, Any]]]:  # fmt: skip
    def load(name: str) -> list[dict[str, Any]]:
        text = (cand_dir / name).read_text(encoding="utf-8")
        return [json.loads(x) for x in text.splitlines() if x.strip()]

    records = [curated_record(c) for c in load("candidates.jsonl")]
    return records, table_metadata(records, model), load("candidates_machine.jsonl")


def write_table(records: list[dict[str, Any]], meta: dict[str, Any], out: Path) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    rec_path, meta_path = out / RECORDS_FILE, out / META_FILE
    with rec_path.open("w", encoding="utf-8", newline="\n") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    meta_path.write_text(yaml.safe_dump(meta, allow_unicode=True, sort_keys=False, width=100),
                         encoding="utf-8", newline="\n")  # fmt: skip
    return [rec_path, meta_path]
