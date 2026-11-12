"""Local JSONL logs of the closed pilot (``data/chat/``), append-only.

* ``requests.jsonl`` - one immutable event per question: participant, question, the engine's
  route and typed outcomes, the deterministic interpretation, certified rows, retrieved chunks,
  response type, sources, provenance violations and latency; ``auto_hint`` is a deterministic
  pointer from the engine's own outcome, never a judgement of correctness;
* ``feedback.jsonl`` - one event per rating (participant, request, up/down, optional note);
* ``reviews.jsonl`` - manual review, one event per review, keyed by request ID. The request
  log is never rewritten: a later review of the same request supersedes an earlier one.

A participant is a pseudonymous pilot ID (``engineer_01``, ``pilot-3fa9c1``), never a name,
email or password. Nothing secret is logged: no environment, credential, header or address.
"""

from __future__ import annotations

import json
import re
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mabhas19.core.paths import DATA_DIR
from mabhas19.generation.contract import Answer
from mabhas19.retrieval.models import RetrievalResult

LOG_DIR = DATA_DIR / "chat"
LOG_SCHEMA = "m2-chat-log-1"
ANONYMOUS = "anonymous"
PARTICIPANT = re.compile(r"^[A-Za-z0-9_-]{1,40}$")
# the manual review labels one of these per request
TRIAGE_CLASSES = (
    "correct_answer",
    "routing",
    "deterministic_recall",
    "semantic_interpretation",
    "prose",
    "out_of_scope",
    "presentation",
)
JUDGEMENTS = ("yes", "no", "uncertain", "not_applicable")


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def participant_id(raw: object) -> str | None:
    """The pilot ID to log (``anonymous`` when absent), or None when it is not a valid ID."""
    text = str(raw or "").strip()
    if not text:
        return ANONYMOUS
    return text if PARTICIPANT.match(text) else None


def auto_hint(answer: Answer) -> str:
    """A first, deterministic pointer for triage (from the engine's outcome only)."""
    if answer.response_type == "structured_value":
        return "answered:certified_rows"
    if answer.response_type == "prose":
        return "answered:excerpts"
    if answer.response_type == "out_of_scope":
        return f"out_of_scope:{answer.refusals[0].reason if answer.refusals else ''}"
    outcomes = sorted({r.outcome for r in answer.refusals})
    return "refused:" + ",".join(outcomes)


def request_record(
    answer: Answer,
    result: RetrievalResult,
    latency_ms: float,
    engine: dict[str, Any],
    participant: str = ANONYMOUS,
) -> dict[str, Any]:
    return {
        "schema": LOG_SCHEMA,
        "kind": "request",
        "timestamp": _now(),
        "request_id": answer.request_id,
        "participant": participant,
        "question": result.query,
        "normalized_question": result.normalized_query,
        "route": result.route.route,
        "route_reasons": result.route.reasons,
        "scope": result.route.signals.get("scope"),
        "interpretation": [
            {"key": i.key, "label": i.label, "value": i.value, "status": i.status}
            for i in answer.interpretation
        ],
        "plans": [
            {
                "table": p.get("target"),
                "operation": p.get("operation"),
                "requested_field": p.get("requested_field"),
                "entities": [
                    {"kind": e.get("kind"), "text": e.get("text"), "status": e.get("status")}
                    for e in p.get("entities") or []
                ],
                "constraints": [
                    {
                        "dimension": c.get("dimension"),
                        "values": c.get("values"),
                        "interval": c.get("interval"),
                        "state": c.get("state"),
                    }
                    for c in p.get("constraints") or []
                ],
                "missing": p.get("missing"),
            }
            for p in result.plans
        ],
        "certified_rows": [
            {"table": h.dataset, "record": h.record_id, "output": h.output}
            for h in result.structured
        ],
        "typed_outcomes": [
            {"table": n.dataset, "outcome": n.outcome, "reason": n.reason} for n in result.no_match
        ],
        "certification_failures": len(result.certification_failures),
        "prose_chunks": [h.chunk_id for h in result.prose],
        "response_type": answer.response_type,
        "structured_answer": answer.response_type == "structured_value",
        "headline": answer.headline,
        "refusals": [{"outcome": r.outcome, "reason": r.reason} for r in answer.refusals],
        "sources": [
            {"label": s.label, "pages": list(s.printed_pages), "ids": list(s.ids)}
            for s in answer.sources
        ],
        "provenance_violations": answer.safety.get("provenance_violations", []),
        "latency_ms": latency_ms,
        "engine": engine,
        "auto_hint": auto_hint(answer),
    }


def review_problems(review: dict[str, Any]) -> list[str]:
    """Why a manual review entry is not valid (empty = valid)."""
    out = []
    if not re.fullmatch(r"[0-9a-f]{32}", str(review.get("request_id") or "")):
        out.append("request_id must be the 32-character ID of a logged request")
    for key in ("answer_correct", "answerable"):
        if review.get(key) not in JUDGEMENTS:
            out.append(f"{key} must be one of {JUDGEMENTS}")
    if review.get("triage") not in TRIAGE_CLASSES:
        out.append(f"triage must be one of {TRIAGE_CLASSES}")
    for key in ("wrong_regulatory_value", "unsupported_numeric_claim"):
        if not isinstance(review.get(key), bool):
            out.append(f"{key} must be true or false")
    if review.get("wrong_regulatory_value") and review.get("answer_correct") == "yes":
        out.append("an answer with a wrong regulatory value cannot be correct")
    return out


class ChatLog:
    def __init__(self, directory: Path = LOG_DIR) -> None:
        self.directory = directory
        self._lock = threading.Lock()

    def _append(self, name: str, record: dict[str, Any]) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False, default=str)
        with self._lock, (self.directory / name).open("a", encoding="utf-8", newline="\n") as f:
            f.write(line + "\n")

    def request(self, record: dict[str, Any]) -> None:
        self._append("requests.jsonl", record)

    def error(self, request_id: str, question: str, error: str, participant: str) -> None:
        self._append(
            "requests.jsonl",
            {
                "schema": LOG_SCHEMA,
                "kind": "error",
                "timestamp": _now(),
                "request_id": request_id,
                "participant": participant,
                "question": question,
                "error": error,
            },
        )

    def feedback(self, request_id: str, rating: str, note: str, participant: str) -> None:
        self._append(
            "feedback.jsonl",
            {
                "schema": LOG_SCHEMA,
                "kind": "feedback",
                "timestamp": _now(),
                "request_id": request_id,
                "participant": participant,
                "rating": rating,
                "note": note,
            },
        )

    def review(self, review: dict[str, Any], reviewer: str = "") -> dict[str, Any]:
        """Append a manual review (validated); the request log itself is never changed."""
        problems = review_problems(review)
        if problems:
            raise ValueError("; ".join(problems))
        record = {
            "schema": LOG_SCHEMA,
            "kind": "review",
            "timestamp": _now(),
            "reviewed": True,
            "reviewer": reviewer,
            **{k: review[k] for k in ("request_id", "answer_correct", "answerable", "triage")},
            "wrong_regulatory_value": review["wrong_regulatory_value"],
            "unsupported_numeric_claim": review["unsupported_numeric_claim"],
            "note": str(review.get("note") or ""),
        }
        self._append("reviews.jsonl", record)
        return record


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
