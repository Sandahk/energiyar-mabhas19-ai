"""Local JSONL logs of the pilot: one line per request, one per feedback (``data/chat/``).

Request lines hold everything M2 needs to classify a real failure without rerunning anything:
the question, the engine's route and typed outcomes, the deterministic interpretation, the
certified rows and retrieved chunks, the response type, the sources shown and the latency.
``triage`` is left empty for the pilot review; ``auto_hint`` is a deterministic first guess
from the engine's own outcome, never a judgement of correctness.

Nothing secret is logged: no environment, credential or client address.
"""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mabhas19.core.paths import DATA_DIR
from mabhas19.generation.contract import Answer
from mabhas19.retrieval.models import RetrievalResult

LOG_DIR = DATA_DIR / "chat"
LOG_SCHEMA = "m1-chat-log-1"
# the M2 review labels one of these per request (``triage``)
TRIAGE_CLASSES = (
    "correct_answer",
    "routing",
    "deterministic_recall",
    "semantic_interpretation",
    "prose",
    "out_of_scope",
    "presentation",
)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


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
    answer: Answer, result: RetrievalResult, latency_ms: float, engine: dict[str, Any]
) -> dict[str, Any]:
    return {
        "schema": LOG_SCHEMA,
        "kind": "request",
        "timestamp": _now(),
        "request_id": answer.request_id,
        "question": result.query,
        "normalized_question": result.normalized_query,
        "route": result.route.route,
        "route_reasons": result.route.reasons,
        "scope": result.route.signals.get("scope"),
        "interpretation": [
            {"key": i.key, "value": i.value, "status": i.status} for i in answer.interpretation
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
        "triage": None,
    }


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

    def error(self, request_id: str, question: str, error: str) -> None:
        self._append(
            "requests.jsonl",
            {
                "schema": LOG_SCHEMA,
                "kind": "error",
                "timestamp": _now(),
                "request_id": request_id,
                "question": question,
                "error": error,
            },
        )

    def feedback(self, request_id: str, rating: str, note: str) -> None:
        self._append(
            "feedback.jsonl",
            {
                "schema": LOG_SCHEMA,
                "kind": "feedback",
                "timestamp": _now(),
                "request_id": request_id,
                "rating": rating,
                "note": note,
            },
        )
