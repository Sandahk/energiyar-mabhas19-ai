"""Phase 2B Gemini benchmark logic, tested offline with a fake vision client."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pytest

from mabhas19.benchmark import phase2b as pb
from mabhas19.benchmark.references import (
    VERIFIED,
    ReferenceItem,
    load_benchmark_pages,
    load_references,
    parse_reference,
)
from mabhas19.core.paths import CONFIGS_DIR, EVAL_DIR
from mabhas19.providers.vision import NUMERIC_RULE, GeminiVision, VisionResult

PAGES = load_benchmark_pages(CONFIGS_DIR / "extraction-benchmark.yaml")[1]
REFS = load_references(EVAL_DIR / "extraction" / "reference")


class FakeVision:
    model = "fake-flash"

    def __init__(self, outcomes: list[Exception | str]) -> None:
        self.outcomes = list(outcomes)
        self.calls = 0

    def ask(self, image: Path, prompt: str, json_mode: bool = False) -> VisionResult:
        self.calls += 1
        out = self.outcomes.pop(0)
        if isinstance(out, Exception):
            raise out
        return VisionResult(out, 0.5, "fake-flash-001", {"prompt_tokens": 10})


def caller(vision: FakeVision, tmp_path: Path) -> tuple[pb.Caller, list[float]]:
    sleeps: list[float] = []
    clock = iter(float(i) * 100 for i in range(1000))
    c = pb.Caller(vision, tmp_path, sleep=sleeps.append, clock=lambda: next(clock))
    return c, sleeps


REQ = pb.Request("x.page", "app07_page", "x", "d", 1, "prompt", True)


def test_requests_cover_only_configured_benchmark_pages() -> None:
    reqs = pb.build_requests(PAGES, REFS)
    assert {r.page_id for r in reqs} <= {p.id for p in PAGES}
    kinds = [r.kind for r in reqs]
    assert kinds.count("text") == len(PAGES)
    assert kinds.count("app07_page") == 3
    assert kinds.count("app07_crop") == 56  # every material row of the three app07 pages
    assert kinds[:3] == ["app07_page"] * 3  # appendix 7 first (priority under quota)


def test_every_extraction_prompt_carries_the_numeric_rule() -> None:
    for req in pb.build_requests(PAGES, REFS):
        assert NUMERIC_RULE in req.prompt


@pytest.mark.parametrize(
    ("message", "kind", "delay"),
    [
        ("429 RESOURCE_EXHAUSTED ... 'retryDelay': '37s'", "rate_limit", 37.0),
        (
            "429 RESOURCE_EXHAUSTED quotaId GenerateRequestsPerDayPerProjectPerModel",
            "quota_daily",
            None,
        ),
        ("503 UNAVAILABLE high demand", "server", None),
        ("400 INVALID_ARGUMENT", "client", None),
    ],
)
def test_error_classification(message: str, kind: str, delay: float | None) -> None:
    assert pb.classify_error(RuntimeError(message)) == (kind, delay)


class ReadTimeout(Exception):
    """Stand-in for an HTTP client's timeout exception."""


@pytest.mark.parametrize(
    "exc", [TimeoutError("x"), ReadTimeout("x"), RuntimeError("The read operation timed out")]
)
def test_timeouts_are_retryable_server_failures(exc: Exception) -> None:
    assert pb.classify_error(exc) == ("server", None)


def test_server_errors_are_retried_with_backoff(tmp_path: Path) -> None:
    fake = FakeVision([RuntimeError("503 UNAVAILABLE"), '{"rows": []}'])
    c, sleeps = caller(fake, tmp_path)
    resp = c.call(REQ, tmp_path / "img.png")
    assert resp.status == "ok"
    assert [a["error"] for a in resp.attempts] == ["server", None]
    assert pb.SERVER_BACKOFF_S[0] in sleeps


def test_rate_limit_honours_retry_delay(tmp_path: Path) -> None:
    fake = FakeVision([RuntimeError("429 RESOURCE_EXHAUSTED 'retryDelay': '12s'"), "{}"])
    c, sleeps = caller(fake, tmp_path)
    assert c.call(REQ, tmp_path / "img.png").status == "ok"
    assert 14.0 in sleeps  # retryDelay + 2 s margin


def test_daily_quota_stops_all_further_requests(tmp_path: Path) -> None:
    fake = FakeVision([RuntimeError("429 RESOURCE_EXHAUSTED PerDay quota")])
    c, _ = caller(fake, tmp_path)
    first = c.call(REQ, tmp_path / "img.png")
    second = c.call(pb.Request("y.page", "app07_page", "y", "d", 1, "p", True), tmp_path / "i")
    assert first.status == "error"
    assert c.quota_exhausted
    assert second.status == "skipped_quota"
    assert fake.calls == 1


def test_successful_responses_are_cached_and_reused(tmp_path: Path) -> None:
    fake = FakeVision(['{"rows": []}'])
    c, _ = caller(fake, tmp_path)
    c.call(REQ, tmp_path / "img.png")
    again = c.call(REQ, tmp_path / "img.png")
    assert fake.calls == 1
    assert again.model_version == "fake-flash-001"
    assert again.usage == {"prompt_tokens": 10}


def test_api_key_is_scrubbed_from_stored_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "SECRET-KEY-123")
    fake = FakeVision([RuntimeError("400 bad request key=SECRET-KEY-123")])
    c, _ = caller(fake, tmp_path)
    c.call(REQ, tmp_path / "img.png")
    stored = (tmp_path / "x.page.json").read_text(encoding="utf-8")
    assert "SECRET-KEY-123" not in stored
    assert "[REDACTED]" in stored


def test_gemini_adapter_requires_explicit_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    ok, why = GeminiVision().availability()
    assert not ok
    assert "not authorized" in why


def ref_row(item_id: str, density_text: str, conductivity: str, dmin: str | None,
            dmax: str | None) -> ReferenceItem:  # fmt: skip
    return ReferenceItem(item_id, "d", 1, "material_row", density_text, (0, 0, 1, 1),
                         {"conductivity": conductivity, "density_min": dmin, "density_max": dmax},
                         status=VERIFIED, verified_by="o",
                         verified_on=date(2026, 10, 6))  # fmt: skip


REF_ROWS = [
    ref_row("r1", "بیش از ۲۰۰۰", "1.80", "2000", None),
    ref_row("r2", "۱۸۰۰ تا ۲۰۰۰", "1.30", "1800", "2000"),
    ref_row("r3", "۱۶۰۰ تا ۱۸۰۰", "1.00", "1600", "1800"),
]


def test_alignment_tolerates_skipped_and_extra_rows() -> None:
    rows = [
        pb.ExtractedRow("بیش از ۲۰۰۰", "۱٫۸۰"),
        pb.ExtractedRow(None, None, "heading only"),  # extra row
        pb.ExtractedRow("۱۶۰۰ تا ۱۸۰۰", "۱٫۰۰"),  # r2 skipped by the model
    ]
    found = pb.align_rows(REF_ROWS, rows)
    assert set(found) == {"r1", "r3"}
    assert found["r3"].conductivity_raw == "۱٫۰۰"


@pytest.mark.parametrize(
    ("expected", "raw", "ok"),
    [("1.30", "۱٫۳۰", True), ("1.30", "۱/۳۰", True), ("1.30", "۱۳۰", False),
     ("0.05", "۰٫۵", False), ("0.05", "۰٫۰۵", True), ("72", "۷۲", None), ("1.30", None, False)],
)  # fmt: skip
def test_decimal_separator_check(expected: str, raw: str | None, ok: bool | None) -> None:
    assert pb.decimal_separator_ok(expected, raw) is ok


def test_app07_scoring_reparses_raw_values_and_lists_errors() -> None:
    found = {
        "r1": pb.ExtractedRow("بیش از ۲۰۰۰", "۱٫۸۰", conductivity_parsed="1.8"),
        "r2": pb.ExtractedRow("۱۸۰۰ تا ۲۰۰۰", "۱۳۰", conductivity_parsed="1.30"),  # separator lost
    }
    rep = pb.score_app07("m", REF_ROWS, found)
    assert rep.total == 8  # r1: k + min (no max); r2 and r3: k + min + max
    assert rep.exact == 4  # r1 k, r1 min, r2 min, r2 max
    wrong = {(e["item"], e["field"]) for e in rep.errors}
    assert ("r2", "conductivity") in wrong
    assert ("r3", "conductivity") in wrong
    r2 = next(e for e in rep.errors if e["item"] == "r2")
    assert (r2["source"], r2["extracted"]) == (
        "1.30",
        "130",
    )  # raw re-parsed, not the model's guess
    assert rep.rows_complete == 1


def test_text_scores_split_verified_and_draft_items() -> None:
    raw: dict[str, Any] = {"page_id": "p", "doc_id": "d", "pdf_page": 1,
                           "status": "draft_pending_human_review", "items": [
        {"id": "a", "doc_id": "d", "pdf_page": 1, "type": "prose", "expected_text": "متن یک",
         "status": VERIFIED, "verified_by": "o", "verified_on": date(2026, 10, 6)},
        {"id": "b", "doc_id": "d", "pdf_page": 1, "type": "prose", "expected_text": "متن دو"},
    ]}  # fmt: skip
    resp = pb.Response("p.text", "text", "p", None, "m", "ok", text="متن یک")
    scores = pb.score_text_requests(resp, parse_reference(raw))
    assert scores["verified"]["items"] == 1
    assert scores["verified"]["exact_normalized"] == 1
    assert scores["draft"]["items"] == 1


@pytest.mark.parametrize(
    ("kind", "data", "ok"),
    [("app07_crop", {"density_raw": "x", "conductivity_raw": "y"}, True),
     ("app07_crop", {"density": "x"}, False),
     ("app07_page", {"rows": [{"density_raw": "x", "conductivity_raw": "y"}]}, True),
     ("app02", {"rows": [{"city_raw": "x"}]}, False),
     ("app03", ["not", "a", "dict"], False)],
)  # fmt: skip
def test_schema_check(kind: str, data: Any, ok: bool) -> None:
    assert pb.schema_ok(kind, data) is ok


def test_json_parsing_tolerates_code_fences() -> None:
    assert pb.parse_json('```json\n{"rows": []}\n```') == {"rows": []}
    assert pb.parse_json("not json") is None
