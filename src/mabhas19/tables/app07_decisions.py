"""Apply project-owner review decisions to appendix 7 candidate records.

Decisions live in a versioned YAML file (``eval/extraction/review/app07-owner-decisions.yaml``)
and are applied deterministically on top of the machine candidate layer:

* ``confirm``: the machine structure of the listed rows is accepted as is;
* ``set``: a label field is replaced; the decision states the machine value it expects
  (``was``) and the application refuses to run if the machine value differs (stale
  decision). The replaced value is kept as ``<field>_machine``;
* ``unresolved``: an open question; the rows keep their machine values.

Only label fields can be changed: numeric values are never touched here. Every record gets
``owner_review`` (status, decision ids, open questions) and a derived ``display_name`` that
never replaces the stored hierarchy.
"""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

import yaml

from mabhas19.core.paths import EVAL_DIR
from mabhas19.tables.app07 import CONDITION_JOIN

DECISIONS_PATH = EVAL_DIR / "extraction" / "review" / "app07-owner-decisions.yaml"
LABEL_FIELDS = frozenset({"category", "subgroup", "material", "condition"})
KINDS = frozenset({"confirm", "set", "unresolved"})
REVIEWED, UNRESOLVED, NOT_REVIEWED = "owner_reviewed", "unresolved", "not_reviewed"
_CATEGORY_NUMBER = re.compile(r"^\s*[0-9۰-۹٠-٩]{1,2}\s*[.]\s*")


class DecisionError(ValueError):
    """A decision is malformed, targets an unknown row, or no longer matches the data."""


def load_decisions(path: Path = DECISIONS_PATH) -> dict[str, Any]:
    doc: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    for d in doc.get("decisions", []):
        if d.get("decision") not in KINDS:
            raise DecisionError(f"{d.get('id')}: unknown decision kind {d.get('decision')!r}")
        for s in d.get("set", []):
            if s["field"] not in LABEL_FIELDS:
                raise DecisionError(f"{d['id']}: field {s['field']!r} cannot be changed here")
    return doc


def display_name(rec: dict[str, Any]) -> str | None:
    """(subgroup, or the category name without its number) + material, space-joined."""
    base = rec.get("subgroup")
    if base is None and rec.get("category"):
        base = _CATEGORY_NUMBER.sub("", rec["category"]) or None
    parts = [p for p in (base, rec.get("material")) if p]
    return " ".join(parts) or None


def _apply_set(rec: dict[str, Any], did: str, change: dict[str, Any]) -> None:
    name = change["field"]
    if rec.get(name) != change["was"]:
        raise DecisionError(f"{did} on {rec['candidate_id']}: {name} is {rec.get(name)!r}, the "
                            f"decision expects {change['was']!r} (stale decision)")  # fmt: skip
    rec[f"{name}_machine"] = rec.get(name)
    rec[name] = change["now"]
    rec[f"{name}_source"] = "owner_decision" if change["now"] is not None else None
    if name == "condition":
        rec["condition_parts_machine"] = rec.get("condition_parts")
        parts = change.get("parts") or ([{"text": change["now"], "role": "leaf"}]
                                        if change["now"] else [])  # fmt: skip
        rec["condition_parts"] = [{**p, "source": "owner_decision"} for p in parts]
        if change["now"] != (CONDITION_JOIN.join(p["text"] for p in parts) or None):
            raise DecisionError(f"{did}: condition text and parts disagree")
    if name == "subgroup" and change["now"] is None:
        rec["subgroup_footnote_refs"] = []


def apply_decisions(candidates: list[dict[str, Any]], doc: dict[str, Any]
                    ) -> list[dict[str, Any]]:  # fmt: skip
    """Return new records with the decisions applied; the input is not modified."""
    out = [copy.deepcopy(c) for c in candidates]
    by_id = {c["candidate_id"]: c for c in out}
    for c in out:
        c["owner_review"] = {"status": NOT_REVIEWED, "decisions": [], "open_questions": []}
    for d in doc.get("decisions", []):
        unknown = [r for r in d["rows"] if r not in by_id]
        if unknown:
            raise DecisionError(f"{d['id']}: unknown rows {unknown}")
        for rid in d["rows"]:
            rec = by_id[rid]
            review = rec["owner_review"]
            review["decisions"].append(d["id"])
            if d["decision"] == "unresolved":
                review["open_questions"].append(d["question"])
                review["status"] = UNRESOLVED
            elif review["status"] != UNRESOLVED:
                review["status"] = REVIEWED
            for change in d.get("set", []):
                _apply_set(rec, d["id"], change)
    for c in out:
        c["display_name"] = display_name(c)
        c["owner_review"]["decisions"] = list(dict.fromkeys(c["owner_review"]["decisions"]))
    return out


def review_summary(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    status: dict[str, int] = {}
    for c in candidates:
        s = c["owner_review"]["status"]
        status[s] = status.get(s, 0) + 1
    flagged = [c for c in candidates if c["suspicious"]]
    return {
        "status": status,
        "suspicious_rows": len(flagged),
        "suspicious_resolved": sum(c["owner_review"]["status"] == REVIEWED for c in flagged),
        "unresolved_rows": [{"candidate_id": c["candidate_id"],
                             "open_questions": c["owner_review"]["open_questions"]}
                            for c in candidates if c["owner_review"]["status"] == UNRESOLVED],
        "flagged_not_reviewed": [c["candidate_id"] for c in flagged
                                 if c["owner_review"]["status"] == NOT_REVIEWED],
        "fields_changed_by_owner": sum(
            1 for c in candidates for k in c
            if k.endswith("_machine") and k != "condition_parts_machine"),
    }  # fmt: skip
