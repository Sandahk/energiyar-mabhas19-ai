"""Apply review decisions to appendix 2 candidates (city names only).

Same model as appendix 7: ``confirm`` accepts the machine value, ``set`` replaces the city
name after checking the machine value it expects (stale decisions are refused), and the
replaced value is kept as ``city_machine``. Climate codes, row numbers and the raw Gemini
transcription can never be changed here. Each record gets ``review`` with its basis.

Rows nobody reviewed keep ``review.status == not_reviewed``; when both independent reads of
the city agree and nothing was flagged they are marked ``corroboration:
two_independent_reads_agree`` (a machine check, not a review).
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from mabhas19.core.paths import EVAL_DIR
from mabhas19.tables import app02 as a2

DECISION_FILES = (EVAL_DIR / "extraction" / "review" / "app02-assistant-review.yaml",)
FIELDS = frozenset({"city"})
BASES = frozenset({"project_owner_reviewed", "assistant_reviewed", "model_inferred"})
KINDS = frozenset({"confirm", "set"})


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
            for s in d.get("set", []):
                if s["field"] not in FIELDS:
                    raise DecisionError(f"{d['id']}: field {s['field']!r} cannot be changed")
            out.append(d)
    return out


def apply_decisions(records: list[dict[str, Any]], decisions: list[dict[str, Any]]
                    ) -> list[dict[str, Any]]:  # fmt: skip
    out = [copy.deepcopy(r) for r in records]
    by_id = {r["candidate_id"]: r for r in out}
    for r in out:
        r["review"] = {"status": "not_reviewed", "basis": [], "decisions": [],
                       "changed_fields": []}  # fmt: skip
    for d in decisions:
        unknown = [x for x in d["rows"] if x not in by_id]
        if unknown:
            raise DecisionError(f"{d['id']}: unknown rows {unknown}")
        for rid in d["rows"]:
            r = by_id[rid]
            for s in d.get("set", []):
                if r[s["field"]] != s["was"]:
                    raise DecisionError(f"{d['id']} on {rid}: {s['field']} is {r[s['field']]!r}, "
                                        f"expected {s['was']!r} (stale decision)")  # fmt: skip
                r["city_machine"] = r["city"]
                r["city"] = a2.city_display(s["now"])
                r["city_normalized"] = a2.city_normalized(s["now"])
                r["city_source"] = d["basis"]
                r["review"]["changed_fields"].append({"field": "city", "decision": d["id"],
                                                      "basis": d["basis"]})  # fmt: skip
            r["review"]["status"] = "reviewed"
            r["review"]["decisions"].append(d["id"])
            if d["basis"] not in r["review"]["basis"]:
                r["review"]["basis"].append(d["basis"])
    for r in out:
        corroborated = r["city_reads_agree"] and not r["suspicious"]
        r["review"]["corroboration"] = "two_independent_reads_agree" if corroborated else None
    return out
