"""Phase 2B: experimental Gemini vision benchmark (benchmark pages only).

Usage:
    uv run --group vision-gemini python -m mabhas19.cli.gemini_benchmark --allow-external
        [--model gemini-3.8-flash] [--max-requests N] [--score-only]

Requires GEMINI_API_KEY in the environment. Sends only the pages/regions defined in
configs/extraction-benchmark.yaml. Raw responses (no keys, no headers) are cached under
data/interim/gemini-phase2b/<model>/ and reused on re-runs.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import pymupdf

from mabhas19.benchmark import phase2b as pb
from mabhas19.benchmark.references import (
    BenchmarkPage,
    PageReference,
    load_benchmark_pages,
    load_references,
)
from mabhas19.core.env import load_dotenv
from mabhas19.core.paths import CONFIGS_DIR, DATA_DIR, EVAL_DIR, SOURCES_DIR
from mabhas19.providers.ocr import TesseractCli
from mabhas19.providers.vision import GeminiVision

APP07_METHODS = ("tesseract_page", "tesseract_cell", "gemini_page", "gemini_crop")
COUNTERS = ("total", "exact", "model_parsed_total", "model_parsed_exact", "decimal_total",
            "decimal_ok", "rows_total", "rows_complete")  # fmt: skip


def app07_comparison(
    pages: list[BenchmarkPage],
    refs: dict[str, PageReference],
    responses: dict[str, pb.Response],
    workdir: Path,
) -> dict[str, Any]:
    """Appendix 7 numeric accuracy of the four methods on human-verified rows only."""
    tess = TesseractCli(tessdata_dir=DATA_DIR / "models" / "tessdata")
    tess_ok = tess.availability()[0]
    reports = {m: pb.NumericReport(m) for m in APP07_METHODS}
    for p in [p for p in pages if p.category == "table_materials"]:
        ref = refs[p.id]
        rows = [i for i in ref.items if i.type == "material_row" and pb.is_verified(i, ref)]
        found: dict[str, dict[str, pb.ExtractedRow]] = {m: {} for m in APP07_METHODS}
        page_resp = responses.get(f"{p.id}.page")
        if page_resp and page_resp.status == "ok":
            extracted = pb.rows_from_json(pb.parse_json(page_resp.text))
            found["gemini_page"] = pb.align_rows(rows, extracted)
        for item in rows:
            r = responses.get(f"{item.id}.crop")
            data = pb.parse_json(r.text) if r and r.status == "ok" else None
            if isinstance(data, dict):
                found["gemini_crop"][item.id] = pb.rows_from_json({"rows": [data]})[0]
        if tess_ok:
            path = SOURCES_DIR / "mabhas19" / "ed5-1404" / "appendices" / f"{p.doc_id}-v1.pdf"
            with pymupdf.open(path) as pdf:
                page = pdf[p.pdf_page - 1]
                found["tesseract_page"] = pb.tesseract_page_rows(page, tess, workdir, rows)
                found["tesseract_cell"] = pb.tesseract_cell_rows(page, tess, p.doc_id, rows)
        for method, rep in reports.items():
            part = pb.score_app07(method, rows, found[method])
            for k in COUNTERS:
                setattr(rep, k, getattr(rep, k) + getattr(part, k))
            rep.errors += part.errors
    return {m: {**r.summary(), "errors": r.errors} for m, r in reports.items()}


def _mean(values: list[float]) -> float | None:
    return round(statistics.mean(values), 2) if values else None


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--allow-external", action="store_true",
                        help="required: authorizes sending benchmark pages to Gemini")  # fmt: skip
    parser.add_argument("--model", default="gemini-3.8-flash")
    parser.add_argument("--max-requests", type=int, default=None)
    parser.add_argument("--score-only", action="store_true", help="use cached responses only")
    parser.add_argument(
        "--only",
        action="append",
        choices=["text", "app07_page", "app07_crop", "app02", "app03"],
        help="send only these request kinds (repeatable); others use cache or are skipped",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    load_dotenv()  # a project key in .env overrides any OS-level GEMINI_API_KEY
    vision = GeminiVision(model=args.model, allow_external=args.allow_external)
    ok, why = vision.availability()
    if not ok and not args.score_only:
        print(f"Gemini not available: {why}", file=sys.stderr)
        return 1

    _, pages = load_benchmark_pages(CONFIGS_DIR / "extraction-benchmark.yaml")
    refs = load_references(EVAL_DIR / "extraction" / "reference")
    run_dir = DATA_DIR / "interim" / "gemini-phase2b" / args.model
    caller = pb.Caller(vision, run_dir / "responses")
    requests = [r for r in pb.build_requests(pages, refs) if not args.only or r.kind in args.only]
    responses: dict[str, pb.Response] = {}
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for n, req in enumerate(requests, 1):
            cached = caller.cached(req)
            if cached is not None:
                responses[req.id] = cached
                continue
            if args.score_only:
                continue
            if args.max_requests is not None and caller.requests_sent >= args.max_requests:
                break
            resp = caller.call(req, pb.render_request(req, SOURCES_DIR, tmp))
            responses[req.id] = resp
            last = resp.attempts[-1].get("error") if resp.attempts else ""
            print(f"[{n}/{len(requests)}] {req.id}: {resp.status} {resp.latency_s}s {last or ''}",
                  flush=True)  # fmt: skip
        comparison = app07_comparison(pages, refs, responses, tmp)

    ok_resps = [r for r in responses.values() if r.status == "ok"]
    json_resps = [r for r in ok_resps if r.kind != "text"]
    schema_ok = sum(pb.schema_ok(r.kind, pb.parse_json(r.text)) for r in json_resps)
    text = {r.page_id: pb.score_text_requests(r, refs[r.page_id])
            for r in ok_resps if r.kind == "text"}  # fmt: skip
    structured: dict[str, Any] = {}
    for r in ok_resps:
        if r.kind == "app02":
            structured["app02"] = pb.score_app02(pb.parse_json(r.text), refs[r.page_id])
        if r.kind == "app03":
            structured["app03"] = pb.score_app03(pb.parse_json(r.text), refs[r.page_id])
    errors = Counter(a["error"] for r in responses.values() for a in r.attempts if a.get("error"))
    usage_keys = ("prompt_tokens", "output_tokens", "thinking_tokens", "total_tokens")
    kinds = sorted({r.kind for r in ok_resps})
    report = {
        "model_requested": args.model,
        "models_reported": sorted({r.model_version for r in ok_resps if r.model_version}),
        "requests_planned": len(requests),
        "responses_ok": len(ok_resps),
        "responses_by_status": dict(Counter(r.status for r in responses.values())),
        "api_attempts_total": sum(len(r.attempts) for r in responses.values()),
        "attempt_errors": dict(errors),
        "daily_quota_exhausted": caller.quota_exhausted,
        "mean_latency_s": {
            k: _mean([r.latency_s for r in ok_resps if r.kind == k and r.latency_s]) for k in kinds
        },
        "usage_totals": {k: sum((r.usage.get(k) or 0) for r in ok_resps) for k in usage_keys},
        "json_schema_success": f"{schema_ok}/{len(json_resps)}",
        "text": text,
        "structured": structured,
        "app07_comparison": comparison,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=1),
                                          encoding="utf-8")  # fmt: skip
    brief = {k: v for k, v in report.items() if k not in ("text", "app07_comparison")}
    print(json.dumps(brief, ensure_ascii=False, indent=1))
    print(json.dumps({m: {k: v for k, v in s.items() if k != "errors"}
                      for m, s in comparison.items()}, ensure_ascii=False, indent=1))  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
