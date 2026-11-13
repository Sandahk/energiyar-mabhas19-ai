"""Pilot chatbot server (standard library only; local, no accounts).

``python -m mabhas19.api.server [--mode hybrid|lexical] [--host 127.0.0.1] [--port 8019]``

* ``GET /`` - the Persian chat page;
* ``POST /api/ask`` ``{"question", "participant", "clarify"?}`` -> the typed answer contract
  (:class:`~mabhas19.generation.contract.Answer`); ``clarify`` continues a clarification (see
  :func:`clarify_request`);
* ``GET /api/health`` -> status, versions and the engine's pin check;
* ``POST /api/feedback`` ``{"request_id", "rating": "up"|"down", "note", "participant"}``.

Every question is answered by the frozen engine and the deterministic composer; there is no
model call. Every request and feedback is logged to ``data/chat/``. The server refuses to start
when the live engine differs from the frozen Final-8 candidate (``--allow-unpinned`` is for
development only and is reported by ``/api/health``). It listens on the loopback interface
only unless ``--allow-network`` is given: it has no authentication, so remote testers reach it
through a private tunnel or VPN, never directly.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import re
import sys
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Protocol

from mabhas19.api.chatlog import ChatLog, participant_id, request_record
from mabhas19.generation.contract import Answer
from mabhas19.generation.labels import PILOT_PRIVACY, PILOT_WARNING
from mabhas19.retrieval.models import RetrievalResult

STATIC = Path(__file__).parent / "static"
# the energiyar.com font and logos, served from here so the page makes no third-party request
ASSETS = {
    "/static/Vazirmatn-Variable.woff2": ("Vazirmatn-Variable.woff2", "font/woff2"),
    "/static/logo-icon.png": ("logo-icon.png", "image/png"),
    "/static/footer-logo.png": ("footer-logo.png", "image/png"),
}
MAX_QUESTION = 1000
MAX_NOTE = 1000
MAX_BODY = 16_384
_REQUEST_ID = re.compile(r"^[0-9a-f]{32}$")
LOOPBACK = ("127.0.0.1", "localhost", "::1")


MAX_CLARIFY_TEXT = 120
_FIELD = re.compile(r"^[a-z_]{1,40}$")


def clarify_request(raw: Any) -> dict[str, Any] | None:
    """A validated clarification follow-up, or ``None``.

    ``{"parent": <request ID>, "field": <field>, "kind": "option"|"free", "text": <words>}``:
    ``text`` (a printed option or the user's typed value) is added to the original question in
    parentheses and the whole question goes through the frozen engine again; nothing is answered
    from the client's choice.
    """
    if not isinstance(raw, dict):
        return None
    parent = str(raw.get("parent") or "")
    field = str(raw.get("field") or "")
    kind = str(raw.get("kind") or "")
    text = " ".join(str(raw.get("text") or "").replace("(", " ").replace(")", " ").split())
    if not (_REQUEST_ID.match(parent) and _FIELD.match(field) and kind in ("option", "free")):
        return None
    if not text or len(text) > MAX_CLARIFY_TEXT:
        return None
    return {"parent_request_id": parent, "field": field, "kind": kind, "text": text}


class AnswerEngine(Protocol):
    info: dict[str, Any]

    def ask(
        self,
        question: str,
        *,
        follow_up: dict[str, Any] | None = None,
        base: str | None = None,
    ) -> tuple[Answer, RetrievalResult, float]: ...


def make_handler(engine: AnswerEngine, log: ChatLog) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "mabhas19-pilot"

        def log_message(self, format: str, *args: Any) -> None:
            pass  # requests are logged to data/chat/, not to the console

        def _send(
            self, status: int, body: bytes, content_type: str, cache: str = "no-store"
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", cache)
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self._send(status, body, "application/json; charset=utf-8")

        def _body(self) -> dict[str, Any] | None:
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0 or length > MAX_BODY:
                return None
            try:
                data = json.loads(self.rfile.read(length).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                return None
            return data if isinstance(data, dict) else None

        def do_GET(self) -> None:
            if self.path in ("/", "/index.html"):
                page = (STATIC / "index.html").read_bytes()
                self._send(HTTPStatus.OK, page, "text/html; charset=utf-8")
            elif self.path in ASSETS:  # a fixed list of the site's own files; nothing else
                name, kind = ASSETS[self.path]
                body = (STATIC / name).read_bytes()
                self._send(HTTPStatus.OK, body, kind, "public, max-age=86400")
            elif self.path == "/api/health":
                self._json(
                    HTTPStatus.OK,
                    {
                        "status": "ok",
                        "warning": PILOT_WARNING,
                        "privacy": PILOT_PRIVACY,
                        "engine": engine.info,
                    },
                )
            else:
                self._json(HTTPStatus.NOT_FOUND, {"error": "not_found"})

        def do_POST(self) -> None:
            if self.path == "/api/ask":
                self._ask()
            elif self.path == "/api/feedback":
                self._feedback()
            else:
                self._discard_body()
                self._json(HTTPStatus.NOT_FOUND, {"error": "not_found"})

        def _discard_body(self) -> None:
            """Read and drop the request body of an unknown route, so the 404 reaches the client
            (a socket closed with unread data can be reset before the reply is read, notably on
            Windows). Only a body within ``MAX_BODY`` is read; any other Content-Length closes the
            connection after the reply instead."""
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                length = -1
            if 0 < length <= MAX_BODY:
                self.rfile.read(length)
            elif length:
                self.close_connection = True

        def _ask(self) -> None:
            data = self._body()
            question = str((data or {}).get("question") or "").strip()
            participant = participant_id((data or {}).get("participant"))
            if participant is None:
                self._json(HTTPStatus.BAD_REQUEST, {"error": "invalid_participant"})
                return
            follow_up = None
            base = question
            if (data or {}).get("clarify") is not None:
                follow_up = clarify_request((data or {}).get("clarify"))
                if follow_up is None:
                    self._json(HTTPStatus.BAD_REQUEST, {"error": "invalid_clarification"})
                    return
                # the original question with only the chosen item added; the engine runs again
                question = f"{question} ({follow_up['text']})"
            if not question or len(question) > MAX_QUESTION:
                self._json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": "invalid_question", "max_length": MAX_QUESTION},
                )
                return
            try:
                if follow_up is None:
                    answer, result, latency = engine.ask(question)
                else:
                    answer, result, latency = engine.ask(question, follow_up=follow_up, base=base)
            except Exception as exc:
                request_id = uuid.uuid4().hex
                log.error(request_id, question, f"{type(exc).__name__}: {exc}", participant)
                self._json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"error": "internal_error", "request_id": request_id},
                )
                return
            log.request(
                request_record(
                    answer, result, latency, engine.info, participant, follow_up=follow_up
                )
            )
            self._json(HTTPStatus.OK, {"answer": answer.to_dict(), "latency_ms": latency})

        def _feedback(self) -> None:
            data = self._body() or {}
            request_id = str(data.get("request_id") or "")
            rating = str(data.get("rating") or "")
            note = str(data.get("note") or "").strip()[:MAX_NOTE]
            participant = participant_id(data.get("participant"))
            valid = _REQUEST_ID.match(request_id) and rating in ("up", "down")
            if not valid or participant is None:
                self._json(HTTPStatus.BAD_REQUEST, {"error": "invalid_feedback"})
                return
            log.feedback(request_id, rating, note, participant)
            self._json(HTTPStatus.OK, {"status": "recorded"})

    return Handler


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=["hybrid", "lexical"], default="hybrid")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8019)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--allow-unpinned", action="store_true")
    parser.add_argument("--allow-network", action="store_true")
    args = parser.parse_args(argv)
    if args.host not in LOOPBACK and not args.allow_network:
        print(
            "the pilot server has no authentication: it listens on the loopback interface only "
            "(use a private tunnel or VPN; --allow-network for a trusted private network)"
        )
        return 2
    from mabhas19.api.engine import load_engine, pin_drift  # noqa: PLC0415

    drift = pin_drift()
    if drift and not args.allow_unpinned:
        print("the live engine differs from the frozen Final-8 candidate:", *drift, sep="\n  ")
        return 2
    engine = load_engine(args.mode, args.threads)
    engine.info["pins"] = "final8" if not drift else {"drift": drift}
    server = ThreadingHTTPServer((args.host, args.port), make_handler(engine, ChatLog()))
    print(f"Mabhas 19 pilot on http://{args.host}:{args.port}/ ({args.mode})")
    with contextlib.suppress(KeyboardInterrupt):
        server.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
