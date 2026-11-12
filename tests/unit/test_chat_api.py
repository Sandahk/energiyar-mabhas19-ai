"""Pilot server: endpoints, validation, logging and feedback (stub engine, no network)."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from mabhas19.api.chatlog import TRIAGE_CLASSES, ChatLog, auto_hint
from mabhas19.api.server import make_handler
from mabhas19.generation.contract import Answer, Refusal
from mabhas19.retrieval.models import RetrievalResult, RouteDecision


class StubEngine:
    def __init__(self) -> None:
        self.info: dict[str, Any] = {"mode": "stub", "composer": "test"}
        self.questions: list[str] = []

    def ask(self, question: str) -> tuple[Answer, RetrievalResult, float]:
        self.questions.append(question)
        if question == "boom":
            raise RuntimeError("engine failure")
        result = RetrievalResult(question, question, RouteDecision("unsupported", ["gate"]))
        answer = Answer(
            request_id="a" * 32,
            question=question,
            response_type="out_of_scope",
            headline="خارج از محدوده",
            refusals=[Refusal("out_of_scope", "external_domain", "خارج از محدوده")],
            safety={"provenance_violations": []},
        )
        return answer, result, 1.5


@pytest.fixture
def server(tmp_path: Path) -> Iterator[tuple[str, StubEngine, Path]]:
    from http.server import ThreadingHTTPServer  # noqa: PLC0415

    engine = StubEngine()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(engine, ChatLog(tmp_path)))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", engine, tmp_path
    httpd.shutdown()
    httpd.server_close()


def _call(url: str, payload: dict[str, Any] | None = None) -> tuple[int, dict[str, Any] | str]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            body = r.read().decode("utf-8")
            status = r.status
    except urllib.error.HTTPError as e:
        body, status = e.read().decode("utf-8"), e.code
    try:
        return status, json.loads(body)
    except json.JSONDecodeError:
        return status, body


def _lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]


def test_page_and_health(server: tuple[str, StubEngine, Path]) -> None:
    base, _, _ = server
    status, page = _call(base + "/")
    assert status == 200
    assert isinstance(page, str)
    assert 'dir="rtl"' in page
    assert "/api/ask" in page
    status, health = _call(base + "/api/health")
    assert status == 200
    assert isinstance(health, dict)
    assert health["status"] == "ok"
    assert health["engine"]["mode"] == "stub"
    assert health["warning"]


def test_ask_returns_the_contract_and_logs_the_request(
    server: tuple[str, StubEngine, Path],
) -> None:
    base, engine, logs = server
    status, body = _call(base + "/api/ask", {"question": " قیمت آجر؟ "})
    assert status == 200
    assert isinstance(body, dict)
    assert body["answer"]["response_type"] == "out_of_scope"
    assert body["answer"]["contract"]
    assert engine.questions == ["قیمت آجر؟"]
    [line] = _lines(logs / "requests.jsonl")
    assert line["request_id"] == "a" * 32
    assert line["question"] == "قیمت آجر؟"
    assert line["response_type"] == "out_of_scope"
    assert line["latency_ms"] == 1.5
    assert line["auto_hint"].startswith("out_of_scope")
    assert line["triage"] is None
    assert not any("KEY" in k.upper() or "SECRET" in k.upper() for k in line)


@pytest.mark.parametrize("payload", [{}, {"question": ""}, {"question": "x" * 1001}, {"q": "y"}])
def test_invalid_questions_are_rejected(
    server: tuple[str, StubEngine, Path], payload: dict[str, Any]
) -> None:
    base, engine, _ = server
    status, _ = _call(base + "/api/ask", payload)
    assert status == 400
    assert engine.questions == []


def test_engine_failure_is_logged_and_never_shown_raw(server: tuple[str, StubEngine, Path]) -> None:
    base, _, logs = server
    status, body = _call(base + "/api/ask", {"question": "boom"})
    assert status == 500
    assert isinstance(body, dict)
    assert body["error"] == "internal_error"
    assert "engine failure" not in json.dumps(body)
    [line] = _lines(logs / "requests.jsonl")
    assert line["kind"] == "error"
    assert "engine failure" in line["error"]


def test_feedback_is_validated_and_logged(server: tuple[str, StubEngine, Path]) -> None:
    base, _, logs = server
    ok = {"request_id": "b" * 32, "rating": "down", "note": "ردیف اشتباه"}
    assert _call(base + "/api/feedback", ok)[0] == 200
    for bad in ({**ok, "rating": "meh"}, {**ok, "request_id": "../x"}, {}):
        assert _call(base + "/api/feedback", bad)[0] == 400
    [line] = _lines(logs / "feedback.jsonl")
    assert (line["request_id"], line["rating"], line["note"]) == ("b" * 32, "down", "ردیف اشتباه")


def test_unknown_paths_are_404(server: tuple[str, StubEngine, Path]) -> None:
    base, _, _ = server
    assert _call(base + "/api/nothing")[0] == 404
    assert _call(base + "/api/nothing", {"a": 1})[0] == 404


def test_auto_hint_and_triage_classes() -> None:
    a = Answer(
        "c" * 32,
        "q",
        "refusal",
        "h",
        refusals=[Refusal("ambiguous_entity", "ambiguous_occupancy", "m")],
    )
    assert auto_hint(a) == "refused:ambiguous_entity"
    assert {
        "routing",
        "deterministic_recall",
        "semantic_interpretation",
        "prose",
        "out_of_scope",
        "correct_answer",
    } <= set(TRIAGE_CLASSES)
