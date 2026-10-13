"""Phase 2C: appendix 7 candidate extraction with the whole-page Gemini method.

Usage:
    uv run --group vision-gemini python -m mabhas19.cli.extract_app07 --allow-external
    uv run python -m mabhas19.cli.extract_app07            # offline: cached responses only

Every table page (PDF pages 3-13) is sent once with the Phase 2B page prompt; successful
answers are cached under data/interim/gemini-phase2b/<model>/responses/ (the same cache as
Phase 2B, so pages 3, 6 and 13 are reused, never re-requested). Before any call the cost of
the missing pages is estimated from measured usage and the run refuses to start when the
estimate reaches the --max-cost budget. Outputs (raw layer, candidate layer, flags,
cross-check and a review page) go to data/processed/app07/<model>/; nothing is written to
curated/.
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
from mabhas19.core.env import load_dotenv
from mabhas19.core.paths import DATA_DIR, SOURCES_DIR

DOC_ID = "m19-ed5-1404-app07"
TABLE_PAGES = tuple(range(3, 14))  # p1 cover, p2 introductory prose (printed page «۱»)
DEFAULT_BUDGET_USD = 1.0


def page_request(pdf_page: int) -> pb.Request:
    page_id = f"app07-p{pdf_page:03d}"
    return pb.Request(f"{page_id}.page", "app07_page", page_id, DOC_ID, pdf_page,
                      pb.APP07_PAGE_PROMPT, True)  # fmt: skip


def response_cost(resp: pb.Response) -> float:
    u = resp.usage
    return sc.request_cost(u.get("prompt_tokens") or 0, u.get("output_tokens") or 0,
                           u.get("thinking_tokens") or 0)  # fmt: skip


def estimate(caller: pb.Caller, pages: tuple[int, ...]) -> dict[str, Any]:
    """Cost of the pages without a cached answer, from the measured cost of cached pages."""
    cached = {p: caller.cached(page_request(p)) for p in pages}
    costs = [response_cost(r) for r in cached.values() if r is not None]
    missing = [p for p, r in cached.items() if r is None]
    mean = sum(costs) / len(costs) if costs else None
    worst = max(costs) if costs else None
    return {
        "pages": list(pages),
        "cached": [p for p, r in cached.items() if r is not None],
        "new_calls": missing,
        "measured_cost_per_page_usd": [round(c, 4) for c in costs],
        "estimate_mean_usd": None if mean is None else round(mean * len(missing), 4),
        "estimate_worst_usd": None if worst is None else round(worst * len(missing), 4),
    }


def within_budget(plan: dict[str, Any], budget_usd: float) -> bool:
    """Calls may start only when the worst-case estimate is known and under budget."""
    worst = plan["estimate_worst_usd"]
    return not plan["new_calls"] or (worst is not None and worst < budget_usd)


def fetch(caller: pb.Caller, pages: list[int]) -> list[pb.Response]:
    out = []
    with tempfile.TemporaryDirectory() as td:
        for p in pages:
            req = page_request(p)
            resp = caller.call(req, pb.render_request(req, SOURCES_DIR, Path(td)))
            last = resp.attempts[-1].get("error") if resp.attempts else None
            print(f"{req.id}: {resp.status} {resp.latency_s}s {last or ''}", flush=True)
            out.append(resp)
    return out


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="gemini-3.8-flash")
    parser.add_argument(
        "--allow-external", action="store_true",
        help="permit Gemini calls for pages without a cached answer")  # fmt: skip
    parser.add_argument(
        "--max-cost", type=float, default=DEFAULT_BUDGET_USD,
        help="refuse to call when the worst-case estimate reaches this (USD)")  # fmt: skip
    parser.add_argument("--fetch-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    run_dir = DATA_DIR / "interim" / "gemini-phase2b" / args.model
    out = DATA_DIR / "processed" / "app07" / args.model
    vision: pb.AskingVision
    if args.allow_external:
        from mabhas19.providers.vision import GeminiVision  # noqa: PLC0415 - optional SDK

        load_dotenv()  # a project key in .env overrides any OS-level GEMINI_API_KEY
        gemini = GeminiVision(model=args.model, allow_external=True)
        ok, why = gemini.availability()
        if not ok:
            print(f"Gemini not available: {why}", file=sys.stderr)
            return 1
        vision = gemini
    else:
        vision = _Offline(args.model)
    caller = pb.Caller(vision, run_dir / "responses")
    plan = estimate(caller, TABLE_PAGES)
    print(json.dumps(plan, ensure_ascii=False))
    calls: list[pb.Response] = []
    if plan["new_calls"]:
        if not within_budget(plan, args.max_cost):
            worst = plan["estimate_worst_usd"]
            print(f"estimate {worst} USD is not under the {args.max_cost} USD budget: "
                  "stopping for approval", file=sys.stderr)  # fmt: skip
            return 2
        if not args.allow_external:
            print("pages without a cached answer; rerun with --allow-external", file=sys.stderr)
            return 2
        out.mkdir(parents=True, exist_ok=True)
        (out / "cost_plan_before_fetch.json").write_text(
            json.dumps(plan, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
        )
        calls = fetch(caller, plan["new_calls"])
    spent = sum(response_cost(r) for r in calls if r.status == "ok")
    print(f"API requests sent: {caller.requests_sent}; new cost: {spent:.4f} USD")
    if args.fetch_only:
        return 0
    from mabhas19.tables.app07_build import build  # noqa: PLC0415

    build(plan, model=args.model, responses=run_dir / "responses", out=out)
    return 0


class _Offline:
    """Placeholder vision for offline runs: any attempt to call it is a programming error."""

    def __init__(self, model: str) -> None:
        self.model = model

    def ask(self, image: Path, prompt: str, json_mode: bool = False) -> Any:
        raise RuntimeError("offline run: no external API calls permitted")


if __name__ == "__main__":
    raise SystemExit(main())
