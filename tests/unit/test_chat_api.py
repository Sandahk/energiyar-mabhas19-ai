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
        self.follow_ups: list[tuple[dict[str, Any] | None, str | None]] = []

    def ask(
        self,
        question: str,
        *,
        follow_up: dict[str, Any] | None = None,
        base: str | None = None,
    ) -> tuple[Answer, RetrievalResult, float]:
        self.questions.append(question)
        self.follow_ups.append((follow_up, base))
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
    assert "ذخیره" in health["privacy"]
    assert "status" in health
    assert 'id="privacy"' in page
    assert 'id="participant"' in page


def test_ask_returns_the_contract_and_logs_the_request(
    server: tuple[str, StubEngine, Path],
) -> None:
    base, engine, logs = server
    status, body = _call(base + "/api/ask", {"question": " قیمت آجر؟ ", "participant": "eng_01"})
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
    assert line["participant"] == "eng_01"
    assert line["structured_answer"] is False
    assert "triage" not in line  # review lives in reviews.jsonl; this log is never rewritten
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
    ok = {"request_id": "b" * 32, "rating": "down", "note": "ردیف اشتباه", "participant": "s_03"}
    assert _call(base + "/api/feedback", ok)[0] == 200
    bad_cases: tuple[dict[str, Any], ...] = (
        {**ok, "rating": "meh"},
        {**ok, "request_id": "../x"},
        {**ok, "participant": "علی رضایی"},
        {},
    )
    for bad in bad_cases:
        assert _call(base + "/api/feedback", bad)[0] == 400
    [line] = _lines(logs / "feedback.jsonl")
    assert (line["request_id"], line["rating"], line["note"]) == ("b" * 32, "down", "ردیف اشتباه")
    assert line["participant"] == "s_03"


def test_participant_defaults_to_anonymous_and_rejects_names_or_emails(
    server: tuple[str, StubEngine, Path],
) -> None:
    base, engine, logs = server
    assert _call(base + "/api/ask", {"question": "سؤال"})[0] == 200
    for bad in ("ali@example.com", "علی", "a" * 41, "x y"):
        assert _call(base + "/api/ask", {"question": "سؤال", "participant": bad})[0] == 400
    [line] = _lines(logs / "requests.jsonl")
    assert line["participant"] == "anonymous"
    assert engine.questions == ["سؤال"]


def test_server_refuses_a_network_interface_without_explicit_permission() -> None:
    from mabhas19.api.server import main  # noqa: PLC0415

    assert main(["--host", "0.0.0.0", "--mode", "lexical"]) == 2


def test_only_the_listed_site_assets_are_served(server: tuple[str, StubEngine, Path]) -> None:
    base, _, _ = server
    for path, kind in (
        ("/static/logo-icon.png", "image/png"),
        ("/static/footer-logo.png", "image/png"),
        ("/static/Vazirmatn-Variable.woff2", "font/woff2"),
    ):
        with urllib.request.urlopen(base + path, timeout=10) as r:
            assert r.status == 200
            assert r.headers["Content-Type"] == kind
            assert r.read()
    for path in ("/static/index.html", "/static/../server.py", "/static/", "/static/x.png"):
        assert _call(base + path)[0] == 404


def test_an_unknown_post_route_with_a_body_reliably_gets_its_404(
    server: tuple[str, StubEngine, Path],
) -> None:
    """Regression: the body is read before the 404, so the reply is never lost to a reset."""
    base, engine, _ = server
    for i in range(60):
        assert _call(base + "/api/nothing", {"question": "x" * (i * 50), "i": i}) == (
            404,
            {"error": "not_found"},
        )
    assert engine.questions == []  # the engine is never reached


def _raw_post(url: str, body: bytes) -> int:
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return int(r.status)
    except urllib.error.HTTPError as e:
        return int(e.code)


def _declared_only(url: str, length: int) -> int:
    """A POST that declares ``length`` body bytes but sends none (the server must refuse by the
    declared length, without reading; no unread bytes are left to race the reply)."""
    import http.client  # noqa: PLC0415
    from urllib.parse import urlparse  # noqa: PLC0415

    u = urlparse(url)
    conn = http.client.HTTPConnection(u.hostname or "", u.port, timeout=10)
    conn.putrequest("POST", u.path)
    conn.putheader("Content-Type", "application/json")
    conn.putheader("Content-Length", str(length))
    conn.endheaders()
    status = conn.getresponse().status
    conn.close()
    return status


def test_body_protections_of_the_api_are_unchanged(server: tuple[str, StubEngine, Path]) -> None:
    base, engine, _ = server
    assert _raw_post(base + "/api/ask", b"{not json") == 400
    assert _declared_only(base + "/api/ask", 20_000) == 400  # over MAX_BODY: never read
    assert _raw_post(base + "/api/feedback", b"[1, 2]") == 400
    assert engine.questions == []
    assert _call(base + "/api/ask", {"question": "سؤال", "participant": "t1"})[0] == 200


def test_unknown_paths_are_404(server: tuple[str, StubEngine, Path]) -> None:
    base, _, _ = server
    assert _call(base + "/api/nothing")[0] == 404
    assert _call(base + "/api/nothing", {"a": 1})[0] == 404


def test_a_clarification_choice_re_asks_the_question_with_only_that_item_added(
    server: tuple[str, StubEngine, Path],
) -> None:
    url, engine, logs = server
    clarify = {"parent": "b" * 32, "field": "climate_class", "kind": "option", "text": "اقلیم 3B"}
    status, _ = _call(
        url + "/api/ask",
        {"question": "حداقل مقاومت حرارتی دیوار", "participant": "t1", "clarify": clarify},
    )
    assert status == 200
    assert engine.questions[-1] == "حداقل مقاومت حرارتی دیوار (اقلیم 3B)"
    rec = _lines(logs / "requests.jsonl")[-1]
    assert rec["follow_up"] == {
        "parent_request_id": "b" * 32,
        "field": "climate_class",
        "kind": "option",
        "text": "اقلیم 3B",
    }
    assert rec["clarifications"] == []
    assert engine.follow_ups[-1] == (rec["follow_up"], "حداقل مقاومت حرارتی دیوار")


def test_a_typed_city_reaches_the_engine_with_the_question_before_it(
    server: tuple[str, StubEngine, Path],
) -> None:
    """The engine receives the original question as ``base`` so it can look the city up first."""
    url, engine, logs = server
    clarify = {"parent": "b" * 32, "field": "climate_class", "kind": "free", "text": "شهر تهران"}
    payload = {"question": "حداقل مقاومت حرارتی دیوار", "participant": "t1", "clarify": clarify}
    status, _ = _call(url + "/api/ask", payload)
    assert status == 200
    follow_up, base = engine.follow_ups[-1]
    assert base == "حداقل مقاومت حرارتی دیوار"
    assert follow_up is not None
    assert follow_up["text"] == "شهر تهران"
    rec = _lines(logs / "requests.jsonl")[-1]
    assert rec["composite"] is None
    assert rec["orchestration_rejected"] == []


@pytest.mark.parametrize(
    "clarify",
    [
        {"parent": "nope", "field": "climate_class", "kind": "option", "text": "اقلیم 3B"},
        {"parent": "b" * 32, "field": "Climate Class", "kind": "option", "text": "اقلیم 3B"},
        {"parent": "b" * 32, "field": "climate_class", "kind": "answer", "text": "اقلیم 3B"},
        {"parent": "b" * 32, "field": "climate_class", "kind": "free", "text": ""},
        {"parent": "b" * 32, "field": "area", "kind": "free", "text": "x" * 200},
        "اقلیم 3B",
    ],
)
def test_invalid_clarifications_are_rejected(
    server: tuple[str, StubEngine, Path], clarify: Any
) -> None:
    url, engine, _ = server
    status, body = _call(
        url + "/api/ask", {"question": "سؤال", "participant": "t1", "clarify": clarify}
    )
    assert status == 400
    assert isinstance(body, dict)
    assert body["error"] == "invalid_clarification"
    assert not engine.questions


def test_typed_parentheses_cannot_break_out_of_the_added_item(
    server: tuple[str, StubEngine, Path],
) -> None:
    url, engine, _ = server
    clarify = {"parent": "b" * 32, "field": "area", "kind": "free", "text": "مساحت 300) (اقلیم 5C"}
    status, _ = _call(
        url + "/api/ask", {"question": "سؤال", "participant": "t1", "clarify": clarify}
    )
    assert status == 200
    assert engine.questions[-1] == "سؤال (مساحت 300 اقلیم 5C)"


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
