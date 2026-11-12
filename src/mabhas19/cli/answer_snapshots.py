"""Answer-composer snapshots over the existing development sets (no new benchmark).

``python -m mabhas19.cli.answer_snapshots [--mode hybrid|lexical] [--per-bucket 4]``

Questions are taken in set order from ``dev``, ``heldout``, ``final``, ``final2`` and ``final3``
(all already used for development) and bucketed by what the frozen engine returned - numeric
values, ranges, climate, materials, energy-use limits, main tables, symbols, conflicts and source
issues, refused calculations, unknown entities, missing constraints, ambiguity, unresolved
wording, enumerations, prose, mixed and out of scope. The first ``--per-bucket`` of each bucket
are rendered to ``data/answers/m1_snapshots.{jsonl,md}`` for manual review.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import yaml

from mabhas19.core.paths import DATA_DIR, EVAL_DIR
from mabhas19.generation.contract import Answer
from mabhas19.retrieval.models import RetrievalResult
from mabhas19.retrieval.schema import APP02, APP03, APP07, MAIN

SETS = ("dev", "heldout", "final", "final2", "final3")
OUT = DATA_DIR / "answers"


def buckets(result: RetrievalResult, answer: Answer) -> list[str]:  # noqa: PLR0912
    out = []
    outcomes = {n.outcome for n in result.no_match}
    datasets = {h.dataset for h in result.structured}
    if answer.response_type == "structured_value":
        if len(result.structured) > 1:
            out.append("multiple_rows")
        if result.enumerations:
            out.append("enumeration")
        if datasets & {APP02, f"{MAIN}19-3-2"}:
            out.append("climate")
        if APP07 in datasets:
            out.append("material")
        if APP03 in datasets:
            out.append("energy_use_limit")
        if any(h.output == "printed_symbol" for h in result.structured):
            out.append("symbol")
        if any(
            h.output in ("criterion", "r_range") or (APP07 in datasets) for h in result.structured
        ):
            out.append("range")
        if any(d.startswith(f"{MAIN}19-5") for d in datasets):
            out.append("main_table_value")
    if result.conflicts:
        out.append("conflict")
    if any(h.source_issues for h in result.structured) or any(
        p.source_issues for p in result.prose[:3]
    ):
        out.append("source_issue")
    for o in (
        "unsupported_calculation",
        "unknown_entity",
        "missing_required_dimension",
        "ambiguous_entity",
        "unresolved_constraint",
        "invalid_constraint",
        "no_printed_category",
        "unsupported_output",
    ):
        if o in outcomes:
            out.append(o)
    if answer.response_type == "prose":
        out.append("prose")
    if result.route.route == "mixed":
        out.append("mixed")
    if answer.response_type == "out_of_scope":
        out.append("out_of_scope")
    return out


def render_md(item: dict[str, Any]) -> str:
    a = item["answer"]
    lines = [
        f"### {item['set']}/{item['id']} [{', '.join(item['buckets'])}]",
        "",
        f"**Q:** {item['query']}",
        "",
        f"- type: `{a['response_type']}`",
        f"- headline: {a['headline']}",
    ]
    if a["interpretation"]:
        echo = " | ".join(
            f"{i['label']} = {i['value']}"
            + ("" if i["status"] in ("resolved", "requested") else f" ({i['status']})")
            for i in a["interpretation"]
        )
        lines.append(f"- echo: {echo}")
    for v in a["values"]:
        conds = "; ".join(f"{c['label']}: {c['value']}" for c in v["conditions"])
        src = v["source"] or {}
        lines.append(
            f"- value: «{v['row_label']}» {v['output_label']} = {v['value']} {v['unit'] or ''}"
            f" [{conds}] — {src.get('label')} p.{','.join(src.get('printed_pages') or [])}"
        )
    if a["hidden_values"]:
        lines.append(f"- hidden rows: {a['hidden_values']}")
    for r in a["refusals"]:
        extra = (
            (f" options={r['options']}" if r["options"] else "")
            + (f" missing={r['missing']}" if r["missing"] else "")
            + (f" hint={r['hint']}" if r["hint"] else "")
        )
        lines.append(f"- refusal [{r['outcome']}/{r['reason']}]: {r['message']}{extra}")
    for e in a["excerpts"]:
        lines.append(
            f"- excerpt ({e['source']['label']} «{e['source']['title']}» "
            f"p.{','.join(e['source']['printed_pages'])}): {e['text'][:300]}"
        )
    for n in a["notices"]:
        lines.append(f"- notice [{n['kind']}]: {n['text']}")
    if a["safety"].get("provenance_violations"):
        lines.append(f"- VIOLATIONS: {a['safety']['provenance_violations']}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=["hybrid", "lexical"], default="hybrid")
    parser.add_argument("--per-bucket", type=int, default=4)
    args = parser.parse_args(argv)
    from mabhas19.api.engine import load_engine  # noqa: PLC0415

    engine = load_engine(args.mode)
    chosen: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for name in SETS:
        cases = yaml.safe_load(
            (EVAL_DIR / "retrieval" / f"{name}.yaml").read_text(encoding="utf-8")
        )
        for c in cases["cases"]:
            answer, result, latency = engine.ask(c["query"])
            tags = buckets(result, answer)
            fresh = [t for t in tags if counts.get(t, 0) < args.per_bucket]
            if not fresh:
                continue
            for t in tags:
                counts[t] = counts.get(t, 0) + 1
            chosen.append(
                {
                    "set": name,
                    "id": c["id"],
                    "query": c["query"],
                    "buckets": tags,
                    "latency_ms": latency,
                    "answer": answer.to_dict(),
                }
            )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "m1_snapshots.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in chosen), encoding="utf-8"
    )
    md = [f"# M1 answer snapshots ({args.mode}, {len(chosen)} answers)\n"]
    md += [render_md(x) for x in chosen]
    (OUT / "m1_snapshots.md").write_text("\n".join(md), encoding="utf-8")
    print(
        json.dumps(
            {"answers": len(chosen), "buckets": counts, "engine": engine.info}, ensure_ascii=False
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
