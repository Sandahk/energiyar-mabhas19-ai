"""Phase 2E: appendix 3 (occupancy x climate x rating -> max EUI) candidate extraction.

Usage:
    uv run --group vision-gemini python -m mabhas19.cli.extract_app03 --allow-external
    uv run python -m mabhas19.cli.extract_app03            # offline: cached answers only

EUI values, climate-column labels and the energy rating come from the PDF text layer;
occupancy labels from a Gemini whole-page read (cached under
data/interim/gemini-phase2b/<model>/responses/) with Tesseract as a corroboration signal.
Before any call the cost is estimated and the run refuses to start when the worst case
reaches --max-cost. Outputs: data/processed/app03/<model>/. Nothing is written to curated/.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

from mabhas19.benchmark import app07_scoring as sc
from mabhas19.benchmark import phase2b as pb
from mabhas19.cli.extract_app07 import _Offline, response_cost
from mabhas19.core.env import load_dotenv
from mabhas19.core.paths import DATA_DIR, SOURCES_DIR
from mabhas19.providers.ocr import TesseractCli
from mabhas19.tables import app03_build as b

DEFAULT_BUDGET_USD = 1.0
# A 34 x 9 table answer is larger than an appendix 7 page: budget per call from token counts.
EST_INPUT_TOKENS, EST_OUTPUT_TOKENS, EST_THINKING_WORST = 1_400, 6_000, 20_000


def estimate(caller: pb.Caller) -> dict[str, Any]:
    missing = [p for p in b.TABLE_PAGES if caller.cached(b.request_for(p)) is None]
    per_call = sc.request_cost(EST_INPUT_TOKENS, EST_OUTPUT_TOKENS, EST_THINKING_WORST)
    typical = sc.request_cost(EST_INPUT_TOKENS, EST_OUTPUT_TOKENS, 3_000)
    return {"pages": list(b.TABLE_PAGES), "new_calls": missing,
            "assumed_tokens_per_call": {"input": EST_INPUT_TOKENS, "output": EST_OUTPUT_TOKENS,
                                        "thinking_worst": EST_THINKING_WORST},
            "estimate_typical_usd": round(typical * len(missing), 4),
            "estimate_worst_usd": round(per_call * len(missing), 4)}  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="gemini-3.8-flash")
    parser.add_argument("--allow-external", action="store_true")
    parser.add_argument("--max-cost", type=float, default=DEFAULT_BUDGET_USD)
    parser.add_argument("--no-ocr", action="store_true")
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
    out = DATA_DIR / "processed" / "app03" / args.model
    spent = 0.0
    if plan["new_calls"]:
        if plan["estimate_worst_usd"] >= args.max_cost:
            print("estimate not under budget: stopping for approval", file=sys.stderr)
            return 2
        if not args.allow_external:
            print("pages without a cached answer; rerun with --allow-external", file=sys.stderr)
            return 2
        out.mkdir(parents=True, exist_ok=True)
        plan_path = out / "cost_plan_before_fetch.json"
        plan_path.write_text(json.dumps(plan, indent=1) + "\n", encoding="utf-8", newline="\n")
        with tempfile.TemporaryDirectory() as td:
            for p in plan["new_calls"]:
                req = b.request_for(p)
                resp = caller.call(req, pb.render_request(req, SOURCES_DIR, Path(td)))
                print(f"{req.id}: {resp.status} {resp.latency_s}s", flush=True)
                spent += response_cost(resp) if resp.status == "ok" else 0.0
    print(f"API requests sent: {caller.requests_sent}; new cost: {spent:.4f} USD")
    engine = None if args.no_ocr else TesseractCli(tessdata_dir=DATA_DIR / "models" / "tessdata")
    summary = b.build(args.model, responses, out, engine)
    print(json.dumps({k: v for k, v in summary.items() if k != "pages"}, ensure_ascii=False,
                     indent=1))  # fmt: skip
    print(f"outputs: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
