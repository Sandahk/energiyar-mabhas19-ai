"""Apply review decisions to appendix 7 candidate records.

Decisions live in versioned YAML files under ``eval/extraction/review/`` and are applied
deterministically on top of the machine candidate layer:

* ``confirm``: the machine structure of the listed rows is accepted as is;
* ``set``: a label field is replaced; the decision states the machine value it expects
  (``was``) and the application refuses to run if the machine value differs (stale
  decision). The replaced value is kept as ``<field>_machine``;
* ``unresolved``: an open question; the rows keep their machine values.

Every decision has a *basis* (who decided), taken from the decision or its file:
``project_owner_reviewed``, ``assistant_reviewed`` or ``model_inferred``. Rows matching a
human-verified benchmark reference get ``benchmark_human_verified`` from the build. The
bases are kept side by side in ``review.basis`` and never merged into one another.

Only label fields can be changed: numeric values are never touched here. Every record gets
``review`` (status, basis, decision ids, open questions) and a derived ``display_name`` that
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

REVIEW_DIR = EVAL_DIR / "extraction" / "review"
DECISION_FILES = (REVIEW_DIR / "app07-owner-decisions.yaml",
                  REVIEW_DIR / "app07-assistant-review.yaml")  # fmt: skip
LABEL_FIELDS = frozenset({"category", "subgroup", "material", "condition"})
KINDS = frozenset({"confirm", "set", "unresolved"})
BENCHMARK = "benchmark_human_verified"
OWNER, ASSISTANT, MODEL = "project_owner_reviewed", "assistant_reviewed", "model_inferred"
BASES = (BENCHMARK, OWNER, ASSISTANT, MODEL)  # precedence for ``review.primary_basis``
REVIEWED, UNRESOLVED, NOT_REVIEWED = "reviewed", "unresolved", "not_reviewed"
_CATEGORY_NUMBER = re.compile(r"^\s*[0-9۰-۹٠-٩]{1,2}\s*[.]\s*")


class DecisionError(ValueError):
    """A decision is malformed, targets an unknown row, or no longer matches the data."""


def load_decisions(path: Path) -> dict[str, Any]:
    doc: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    default = doc.get("basis_default")
    for d in doc.get("decisions", []):
        if d.get("decision") not in KINDS:
            raise DecisionError(f"{d.get('id')}: unknown decision kind {d.get('decision')!r}")
        d.setdefault("basis", default)
        if d["basis"] not in BASES[1:]:
            raise DecisionError(f"{d['id']}: unknown or missing basis {d['basis']!r}")
        for s in d.get("set", []):
            if s["field"] not in LABEL_FIELDS:
                raise DecisionError(f"{d['id']}: field {s['field']!r} cannot be changed here")
    return doc


def load_all(paths: tuple[Path, ...] = DECISION_FILES) -> dict[str, Any]:
    """All decision files that exist, in order, as one decision list."""
    decisions: list[dict[str, Any]] = []
    for p in paths:
        if p.is_file():
            decisions += load_decisions(p)["decisions"]
    return {"decisions": decisions}


def display_name(rec: dict[str, Any]) -> str | None:
    """(subgroup, or the category name without its number) + material, space-joined."""
    base = rec.get("subgroup")
    if base is None and rec.get("category"):
        base = _CATEGORY_NUMBER.sub("", rec["category"]) or None
    parts = [p for p in (base, rec.get("material")) if p]
    return " ".join(parts) or None


def _apply_set(rec: dict[str, Any], d: dict[str, Any], change: dict[str, Any]) -> None:
    name = change["field"]
    if rec.get(name) != change["was"]:
        raise DecisionError(f"{d['id']} on {rec['candidate_id']}: {name} is {rec.get(name)!r}, "
                            f"the decision expects {change['was']!r} (stale decision)")  # fmt: skip
    rec[f"{name}_machine"] = rec.get(name)
    rec[name] = change["now"]
    rec[f"{name}_source"] = d["basis"] if change["now"] is not None else None
    if name == "condition":
        rec["condition_parts_machine"] = rec.get("condition_parts")
        parts = change.get("parts") or ([{"text": change["now"], "role": "leaf"}]
                                        if change["now"] else [])  # fmt: skip
        rec["condition_parts"] = [{**p, "source": d["basis"]} for p in parts]
        if change["now"] != (CONDITION_JOIN.join(p["text"] for p in parts) or None):
            raise DecisionError(f"{d['id']}: condition text and parts disagree")
    if name == "subgroup" and change["now"] is None:
        rec["subgroup_footnote_refs"] = []


def _record(rec: dict[str, Any], d: dict[str, Any]) -> None:
    review = rec["review"]
    review["decisions"].append(d["id"])
    review["basis"].append(d["basis"])
    if d["decision"] == "unresolved":
        review["open_questions"].append(d["question"])
        review["status"] = UNRESOLVED
    elif review["status"] != UNRESOLVED:
        review["status"] = REVIEWED
    for change in d.get("set", []):
        _apply_set(rec, d, change)
        review["changed_fields"].append({"field": change["field"], "decision": d["id"],
                                         "basis": d["basis"]})  # fmt: skip


def apply_decisions(candidates: list[dict[str, Any]], doc: dict[str, Any]
                    ) -> list[dict[str, Any]]:  # fmt: skip
    """Return new records with the decisions applied; the input is not modified."""
    out = [copy.deepcopy(c) for c in candidates]
    by_id = {c["candidate_id"]: c for c in out}
    for c in out:
        c["review"] = {"status": NOT_REVIEWED, "basis": [], "primary_basis": None,
                       "decisions": [], "changed_fields": [], "open_questions": []}  # fmt: skip
    for d in doc.get("decisions", []):
        unknown = [r for r in d["rows"] if r not in by_id]
        if unknown:
            raise DecisionError(f"{d['id']}: unknown rows {unknown}")
        for rid in d["rows"]:
            _record(by_id[rid], d)
    for c in out:
        c["display_name"] = display_name(c)
        finish_review(c)
    return out


def add_benchmark_basis(candidates: list[dict[str, Any]], verified_ids: set[str]) -> None:
    """Mark rows that match a human-verified benchmark reference row exactly."""
    for c in candidates:
        if c["candidate_id"] in verified_ids:
            c["review"]["basis"].append(BENCHMARK)
            if c["review"]["status"] == NOT_REVIEWED:
                c["review"]["status"] = REVIEWED
            finish_review(c)


def finish_review(c: dict[str, Any]) -> None:
    review = c["review"]
    review["basis"] = [b for b in BASES if b in review["basis"]]
    review["decisions"] = list(dict.fromkeys(review["decisions"]))
    review["primary_basis"] = review["basis"][0] if review["basis"] else None


def review_summary(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    def count(key: str) -> dict[str, int]:
        out: dict[str, int] = {}
        for c in candidates:
            v = c["review"][key] or "none"
            out[v] = out.get(v, 0) + 1
        return out

    flagged = [c for c in candidates if c["suspicious"]]
    return {
        "status": count("status"),
        "primary_basis": count("primary_basis"),
        "rows_with_basis": {b: sum(b in c["review"]["basis"] for c in candidates) for b in BASES},
        "suspicious_rows": len(flagged),
        "suspicious_resolved": sum(c["review"]["status"] == REVIEWED for c in flagged),
        "unresolved_rows": [{"candidate_id": c["candidate_id"],
                             "open_questions": c["review"]["open_questions"]}
                            for c in candidates if c["review"]["status"] == UNRESOLVED],
        "not_reviewed_rows": [c["candidate_id"] for c in candidates
                              if c["review"]["status"] == NOT_REVIEWED],
        "fields_changed": sum(len(c["review"]["changed_fields"]) for c in candidates),
    }  # fmt: skip
