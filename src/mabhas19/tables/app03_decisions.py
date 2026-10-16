"""Apply review decisions to appendix 3 occupancy rows (labels only).

A ``set_label`` decision gives the reviewed label for one occupancy position and lists, per
row (one per rating), the exact machine label it expects (``was``); a mismatch refuses the
decision (stale). Rows whose machine label differs from the reviewed label keep it as
``occupancy_machine``. The raw Gemini/Tesseract transcriptions are never touched, and EUI
values cannot be changed here.

Source-issue resolutions (SI-0003) are recorded on the rows as ``source_issue_resolution``;
the source-issue reference itself stays on the row.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from mabhas19.core.paths import EVAL_DIR
from mabhas19.tables import app03 as a3

DECISION_FILES = (EVAL_DIR / "extraction" / "review" / "app03-assistant-review.yaml",)
BASES = frozenset({"project_owner_reviewed", "assistant_reviewed", "model_inferred"})
KINDS = frozenset({"set_label"})


class DecisionError(ValueError):
    """A decision is malformed, targets an unknown row, or no longer matches the data."""


def load_all(paths: tuple[Path, ...] = DECISION_FILES) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path in paths:
        if not path.is_file():
            continue
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
        for d in doc.get("decisions", []):
            d.setdefault("basis", doc.get("basis_default"))
            if d.get("decision") not in KINDS or d["basis"] not in BASES:
                raise DecisionError(f"{d.get('id')}: bad decision kind or basis")
            if not d.get("label") or set(d.get("was", {})) != set(d["rows"]):
                raise DecisionError(f"{d['id']}: needs a label and one expected value per row")
            out.append(d)
    return out


def apply_decisions(rows: list[dict[str, Any]], decisions: list[dict[str, Any]]
                    ) -> list[dict[str, Any]]:  # fmt: skip
    out = [copy.deepcopy(r) for r in rows]
    by_id = {r["row_id"]: r for r in out}
    for r in out:
        r["review"] = {"status": "not_reviewed", "basis": [], "decisions": [],
                       "changed_fields": []}  # fmt: skip
    for d in decisions:
        unknown = [x for x in d["rows"] if x not in by_id]
        if unknown:
            raise DecisionError(f"{d['id']}: unknown rows {unknown}")
        for rid in d["rows"]:
            r = by_id[rid]
            if r["occupancy"] != d["was"][rid]:
                raise DecisionError(f"{d['id']} on {rid}: label is {r['occupancy']!r}, expected "
                                    f"{d['was'][rid]!r} (stale decision)")  # fmt: skip
            if r["occupancy"] != d["label"]:
                r["occupancy_machine"] = r["occupancy"]
                r["occupancy"] = a3.label_display(d["label"])
                r["occupancy_normalized"] = a3.label_normalized(d["label"])
                r["occupancy_source"] = d["basis"]
                r["review"]["changed_fields"].append({"field": "occupancy", "decision": d["id"],
                                                      "basis": d["basis"]})  # fmt: skip
            r["review"]["status"] = "reviewed"
            r["review"]["decisions"].append(d["id"])
            if d["basis"] not in r["review"]["basis"]:
                r["review"]["basis"].append(d["basis"])
            if d.get("source_issue_resolution"):
                r["source_issue_resolution"] = {
                    **d["source_issue_resolution"],
                    "decision": d["id"],
                    "basis": d["basis"],
                }
    for r in out:
        corroborated = r["occupancy_reads_agree"] and not r["suspicious"]
        r["review"]["corroboration"] = "two_independent_reads_agree" if corroborated else None
    return out
