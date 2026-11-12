"""Pilot chatbot server (standard library only; local, no accounts).

``python -m mabhas19.api.server [--mode hybrid|lexical] [--host 127.0.0.1] [--port 8019]``

* ``GET /`` - the Persian chat page;
* ``POST /api/ask`` ``{"question": "..."}`` -> the typed answer contract
  (:class:`~mabhas19.generation.contract.Answer`);
* ``GET /api/health`` -> status, versions and the engine's pin check;
* ``POST /api/feedback`` ``{"request_id", "rating": "up"|"down", "note"}``.

Every question is answered by the frozen engine and the deterministic composer; there is no
model call. Every request and feedback is logged to ``data/chat/``. The server refuses to start
when the live engine differs from the frozen Final-8 candidate (``--allow-unpinned`` is for
development only and is reported by ``/api/health``).
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

from mabhas19.api.chatlog import ChatLog, request_record
from mabhas19.generation.contract import Answer
from mabhas19.generation.labels import PILOT_WARNING
from mabhas19.retrieval.models import RetrievalResult

STATIC = Path(__file__).parent / "static"
MAX_QUESTION = 1000
MAX_NOTE = 1000
MAX_BODY = 16_384
_REQUEST_ID = re.compile(r"^[0-9a-f]{32}$")


class AnswerEngine(Protocol):
    info: dict[str, Any]

    def ask(self, question: str) -> tuple[Answer, RetrievalResult, float]: ...


def make_handler(engine: AnswerEngine, log: ChatLog) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "mabhas19-pilot"

        def log_message(self, format: str, *args: Any) -> None:
            pass  # requests are logged to data/chat/, not to the console

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
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
            elif self.path == "/api/health":
                self._json(
                    HTTPStatus.OK, {"status": "ok", "warning": PILOT_WARNING, "engine": engine.info}
                )
            else:
                self._json(HTTPStatus.NOT_FOUND, {"error": "not_found"})

        def do_POST(self) -> None:
            if self.path == "/api/ask":
                self._ask()
            elif self.path == "/api/feedback":
                self._feedback()
            else:
                self._json(HTTPStatus.NOT_FOUND, {"error": "not_found"})

        def _ask(self) -> None:
            data = self._body()
            question = str((data or {}).get("question") or "").strip()
            if not question or len(question) > MAX_QUESTION:
                self._json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": "invalid_question", "max_length": MAX_QUESTION},
                )
                return
            try:
                answer, result, latency = engine.ask(question)
            except Exception as exc:
                request_id = uuid.uuid4().hex
                log.error(request_id, question, f"{type(exc).__name__}: {exc}")
                self._json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"error": "internal_error", "request_id": request_id},
                )
                return
            log.request(request_record(answer, result, latency, engine.info))
            self._json(HTTPStatus.OK, {"answer": answer.to_dict(), "latency_ms": latency})

        def _feedback(self) -> None:
            data = self._body() or {}
            request_id = str(data.get("request_id") or "")
            rating = str(data.get("rating") or "")
            note = str(data.get("note") or "").strip()[:MAX_NOTE]
            if not _REQUEST_ID.match(request_id) or rating not in ("up", "down"):
                self._json(HTTPStatus.BAD_REQUEST, {"error": "invalid_feedback"})
                return
            log.feedback(request_id, rating, note)
            self._json(HTTPStatus.OK, {"status": "recorded"})

    return Handler


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=["hybrid", "lexical"], default="hybrid")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8019)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--allow-unpinned", action="store_true")
    args = parser.parse_args(argv)
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
