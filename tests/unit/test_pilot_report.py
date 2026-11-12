"""M2 pilot review log and metrics (logs only; no engine)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from mabhas19.api.chatlog import ChatLog, read_jsonl, review_problems
from mabhas19.api.pilot_metrics import markdown, metrics
from mabhas19.cli.pilot_report import main


def _request(rid: str, rtype: str, participant: str, **kw: Any) -> dict[str, Any]:
    return {
        "kind": "request",
        "timestamp": kw.get("ts", "2026-11-12T10:00:00"),
        "request_id": rid,
        "participant": participant,
        "question": f"q-{rid[:4]}",
        "route": "structured",
        "response_type": rtype,
        "structured_answer": rtype == "structured_value",
        "latency_ms": kw.get("latency", 300.0),
        "provenance_violations": kw.get("violations", []),
    }


def _review(rid: str, **kw: Any) -> dict[str, Any]:
    return {
        "request_id": rid,
        "answer_correct": kw.get("correct", "yes"),
        "answerable": kw.get("answerable", "not_applicable"),
        "triage": kw.get("triage", "correct_answer"),
        "wrong_regulatory_value": kw.get("wrong", False),
        "unsupported_numeric_claim": kw.get("unsupported", False),
        "note": kw.get("note", ""),
    }


A, B, C, D, E = ("a" * 32, "b" * 32, "c" * 32, "d" * 32, "e" * 32)


@pytest.fixture
def logs(tmp_path: Path) -> Path:
    log = ChatLog(tmp_path)
    log.request(_request(A, "structured_value", "eng_01"))
    log.request(_request(B, "prose", "eng_01", latency=900.0))
    log.request(_request(C, "refusal", "stu_02"))
    log.request(_request(D, "out_of_scope", "stu_02"))
    log.request(_request(E, "structured_value", "eng_03", violations=["numeric: 9"]))
    log.error("f" * 32, "boom", "RuntimeError", "eng_03")
    log.feedback(A, "up", "", "eng_01")
    log.feedback(C, "down", "باید جواب می‌داد", "stu_02")
    log.review(_review(A))
    log.review(_review(B, correct="no", triage="prose"))
    log.review(_review(C, correct="not_applicable", answerable="yes", triage="routing"))
    log.review(_review(D, correct="not_applicable", answerable="no", triage="out_of_scope"))
    return tmp_path


def _metrics(directory: Path) -> dict[str, Any]:
    return metrics(
        read_jsonl(directory / "requests.jsonl"),
        read_jsonl(directory / "feedback.jsonl"),
        read_jsonl(directory / "reviews.jsonl"),
    )


def test_usage_answer_rate_and_feedback(logs: Path) -> None:
    m = _metrics(logs)
    assert m["usage"]["participants"] == 3
    assert m["usage"]["questions"] == 5
    assert m["usage"]["errors"] == 1
    assert m["usage"]["questions_per_participant"] == {"eng_01": 2, "stu_02": 2, "eng_03": 1}
    assert m["answer_rate"]["structured_value"]["count"] == 2
    assert m["answer_rate"]["out_of_scope"]["share"] == 0.2
    assert m["feedback"]["feedback_rate"] == 0.4
    assert m["feedback"]["negative_rate"] == 0.5
    assert m["feedback"]["with_note"] == 1


def test_reviewed_quality_taxonomy_and_safety(logs: Path) -> None:
    m = _metrics(logs)
    q = m["reviewed_quality"]
    assert (q["reviewed"], q["correct"], q["incorrect"]) == (4, 1, 1)
    assert (q["unanswered_but_answerable"], q["genuinely_unsupported"]) == (1, 1)
    assert m["failure_taxonomy"]["routing"] == 1
    assert m["failure_taxonomy"]["prose"] == 1
    assert m["safety"]["wrong_regulatory_values"] == 0
    assert m["safety"]["provenance_violations"] == 1
    assert m["safety"]["provenance_violation_requests"] == [E]
    assert m["unreviewed"] == [E]
    assert "Safety" in markdown(m)
    assert metrics([], [], [])["safety"]["hard_criterion"] == "nothing reviewed yet"


def test_a_later_review_supersedes_and_a_wrong_value_is_flagged(logs: Path) -> None:
    log = ChatLog(logs)
    log.review(_review(A, correct="no", wrong=True, triage="semantic_interpretation"))
    m = _metrics(logs)
    assert m["safety"]["wrong_regulatory_values"] == 1
    assert m["safety"]["wrong_regulatory_value_requests"] == [A]
    assert m["safety"]["hard_criterion"] == "investigate now"
    assert m["reviewed_quality"]["correct"] == 0
    assert len(read_jsonl(logs / "reviews.jsonl")) == 5  # appended, nothing rewritten


def test_request_log_is_never_rewritten_by_review(logs: Path) -> None:
    before = (logs / "requests.jsonl").read_bytes()
    ChatLog(logs).review(_review(B, correct="uncertain", triage="presentation"))
    assert (logs / "requests.jsonl").read_bytes() == before


@pytest.mark.parametrize(
    "bad",
    [
        {"request_id": "x"},
        {"answer_correct": "maybe"},
        {"triage": "llm_said_so"},
        {"wrong_regulatory_value": "no"},
        {"answer_correct": "yes", "wrong_regulatory_value": True},
    ],
)
def test_invalid_reviews_are_refused(bad: dict[str, Any]) -> None:
    assert review_problems({**_review(A), **bad})


def test_cli_review_pending_and_report(logs: Path, capsys: pytest.CaptureFixture[str]) -> None:
    base = ["--log-dir", str(logs)]
    assert main([*base, "pending"]) == 0
    assert E in capsys.readouterr().out
    args = ["review", "--request", E, "--correct", "yes", "--answerable", "not_applicable"]
    assert main([*base, *args, "--triage", "correct_answer", "--reviewer", "r1"]) == 0
    assert (
        main(
            [
                *base,
                "review",
                "--request",
                "0" * 32,
                "--correct",
                "yes",
                "--answerable",
                "no",
                "--triage",
                "routing",
            ]
        )
        == 2
    )
    assert main([*base, "report"]) == 0
    [report] = list((logs / "reports").glob("m2_report_*.json"))
    assert json.loads(report.read_text(encoding="utf-8"))["unreviewed"] == []
