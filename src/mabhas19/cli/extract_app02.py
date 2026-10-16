"""Phase 2D: appendix 2 (city -> climate class) candidate extraction.

Usage:
    uv run --group vision-gemini python -m mabhas19.cli.extract_app02 --allow-external
    uv run python -m mabhas19.cli.extract_app02            # offline: cached answers only

Row numbers and climate codes come from the PDF text layer; city names from a Gemini
whole-page read (cached under data/interim/gemini-phase2b/<model>/responses/, shared with
Phase 2B) with Tesseract as a second read. Before any call the cost is estimated from the
measured cost of cached appendix 7 pages and the run refuses to start when the worst case
reaches --max-cost. Outputs: data/processed/app02/<model>/. Nothing is written to curated/.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

from mabhas19.benchmark import phase2b as pb
from mabhas19.cli.extract_app07 import _Offline, page_request, response_cost
from mabhas19.core.env import load_dotenv
from mabhas19.core.paths import DATA_DIR, SOURCES_DIR
from mabhas19.providers.ocr import TesseractCli
from mabhas19.tables import app02_build as b

DEFAULT_BUDGET_USD = 1.0
MEASURE_PAGES = (3, 6, 13)  # appendix 7 page answers with measured usage


def estimate(caller: pb.Caller) -> dict[str, Any]:
    missing = [p for p in b.TABLE_PAGES if caller.cached(b.request_for(p)) is None]
    measured = [r for p in MEASURE_PAGES if (r := caller.cached(page_request(p))) is not None]
    costs = [response_cost(r) for r in measured]
    worst = max(costs) if costs else None
    mean = sum(costs) / len(costs) if costs else None
    return {"pages": list(b.TABLE_PAGES), "new_calls": missing,
            "measured_cost_per_page_usd": [round(c, 4) for c in costs],
            "estimate_mean_usd": None if mean is None else round(mean * len(missing), 4),
            "estimate_worst_usd": None if worst is None else round(worst * len(missing), 4),
            }  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="gemini-3.8-flash")
    parser.add_argument("--allow-external", action="store_true")
    parser.add_argument("--max-cost", type=float, default=DEFAULT_BUDGET_USD)
    parser.add_argument("--no-ocr", action="store_true", help="skip the Tesseract second read")
    args = parser.parse_args(argv)
    responses = b.default_responses(args.model)
    vision: pb.AskingVision
    if args.allow_external:
        from mabhas19.providers.vision import GeminiVision  # noqa: PLC0415 - optional SDK

        load_dotenv()
        gemini = GeminiVision(model=args.model, allow_external=True)
        ok, why = gemini.availability()
        if not ok:
            print(f"Gemini not available: {why}", file=sys.stderr)
            return 1
        vision = gemini
    else:
        vision = _Offline(args.model)
    caller = pb.Caller(vision, responses)
    plan = estimate(caller)
    print(json.dumps(plan))
    spent = 0.0
    if plan["new_calls"]:
        worst = plan["estimate_worst_usd"]
        if worst is None or worst >= args.max_cost:
            print(f"estimate {worst} USD not under {args.max_cost} USD: stopping for approval",
                  file=sys.stderr)  # fmt: skip
            return 2
        if not args.allow_external:
            print("pages without a cached answer; rerun with --allow-external", file=sys.stderr)
            return 2
        with tempfile.TemporaryDirectory() as td:
            for p in plan["new_calls"]:
                req = b.request_for(p)
                resp = caller.call(req, pb.render_request(req, SOURCES_DIR, Path(td)))
                print(f"{req.id}: {resp.status} {resp.latency_s}s", flush=True)
                spent += response_cost(resp) if resp.status == "ok" else 0.0
    print(f"API requests sent: {caller.requests_sent}; new cost: {spent:.4f} USD")
    engine = None if args.no_ocr else TesseractCli(tessdata_dir=DATA_DIR / "models" / "tessdata")
    out = DATA_DIR / "processed" / "app02" / args.model
    summary = b.build(args.model, responses, out, engine)
    summary["cost_plan"] = plan
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n",
                                      encoding="utf-8", newline="\n")  # fmt: skip
    print(json.dumps({k: v for k, v in summary.items() if k != "pages"}, ensure_ascii=False,
                     indent=1))  # fmt: skip
    print(f"outputs: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
