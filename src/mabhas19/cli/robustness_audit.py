"""Robustness audit: generated collisions, question sweeps and the quantity/table survey.

``python -m mabhas19.cli.robustness_audit collisions [--write-baseline]``
``python -m mabhas19.cli.robustness_audit sweep [--mode lexical|hybrid]``
``python -m mabhas19.cli.robustness_audit consistency [--mode lexical|hybrid]``

* ``collisions`` - the namespace collisions of the curated catalogue
  (:func:`mabhas19.api.invariants.collisions`) against the reviewed baseline
  ``eval/robustness/collisions_baseline.yaml``; ``--write-baseline`` rewrites it, keeping every
  existing review and marking new entries ``review: pending``.
* ``sweep`` - generated questions (every city x climate-dependent quantity, every family word x
  quantity, row -> city -> value, city written -> row -> value, near-miss row names), each checked
  against the invariants; the report goes to ``data/audit/sweep.json``.
* ``consistency`` - quantity preservation over the development sets (dev, heldout, final,
  final2, final3); the report goes to ``data/audit/consistency.json``.

Audit only: nothing here changes an answer.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from typing import TYPE_CHECKING, Any

import yaml

from mabhas19.api import invariants as I
from mabhas19.core.paths import CURATED_DIR, DATA_DIR, EVAL_DIR
from mabhas19.retrieval.structured import StructuredStore

if TYPE_CHECKING:
    from mabhas19.api.engine import Engine

OUT = DATA_DIR / "audit"
SETS = ("dev", "heldout", "final", "final2", "final3")


def collision_entries(store: StructuredStore) -> list[dict[str, Any]]:
    return [
        {
            "key": c.key,
            "kind": c.kind,
            "namespaces": list(c.namespaces),
            **({"longer_namespaces": list(c.longer_namespaces)} if c.kind == "prefix" else {}),
            "examples": list(c.examples),
        }
        for c in I.collisions(store)
    ]


def cmd_collisions(write: bool) -> int:
    store = StructuredStore(CURATED_DIR)
    found = collision_entries(store)
    path = I.ROBUSTNESS_DIR / "collisions_baseline.yaml"
    old = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
    reviews = {e["key"]: e for e in (old or {}).get("collisions") or []}
    new = [e["key"] for e in found if e["key"] not in reviews]
    gone = [k for k in reviews if k not in {e["key"] for e in found}]
    kinds = Counter(e["kind"] for e in found)
    print(f"collisions: {len(found)} ({dict(kinds)}); new {len(new)}; no longer found {len(gone)}")
    for k in new:
        print("  new:", k)
    for k in gone:
        print("  gone:", k)
    if write:
        entries = []
        for e in found:
            prev = reviews.get(e["key"]) or {}
            entries.append(
                {**e, "review": prev.get("review", "pending"), "note": prev.get("note", "")}
            )
        data = {
            **{k: v for k, v in (old or {}).items() if k != "collisions"},
            "collisions": entries,
        }
        path.write_text(
            yaml.safe_dump(data, allow_unicode=True, sort_keys=False, width=100), encoding="utf-8"
        )
        print("written:", path)
    return 1 if (new or gone) and not write else 0


def _engine(mode: str) -> Engine:
    from mabhas19.api.engine import load_engine  # noqa: PLC0415

    return load_engine("lexical" if mode == "lexical" else "hybrid")


def check(question: str, answer: Any, result: Any, store: Any = None) -> list[str]:
    return I.quantity_violations(question, answer, result) + I.unwritten_entity_violations(
        question, answer, store
    )


def cmd_sweep(mode: str) -> int:
    engine = _engine(mode)
    rows = []
    for case in I.sweep_questions(engine.store):
        answer, result, _ = engine.ask(case["question"])
        problems = check(case["question"], answer, result, engine.store)
        if case["label"]:
            problems += I.substitution_violations(case["label"], answer)
        rows.append({**case, "response_type": answer.response_type, "problems": problems})
    bad = [r for r in rows if r["problems"]]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sweep.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), "utf-8")
    print(f"sweep: {len(rows)} questions, {len(bad)} with problems")
    for r in bad:
        print("  ", r["family"], r["question"], r["problems"])
    return 1 if bad else 0


def cmd_consistency(mode: str) -> int:
    engine = _engine(mode)
    rows = []
    for name in SETS:
        cases = yaml.safe_load((EVAL_DIR / "retrieval" / f"{name}.yaml").read_text("utf-8"))
        for c in cases["cases"] if isinstance(cases, dict) else cases:
            answer, result, _ = engine.ask(c["query"])
            problems = check(c["query"], answer, result, engine.store)
            rows.append({"set": name, "id": c.get("id"), "query": c["query"], "problems": problems})
    bad = [r for r in rows if r["problems"]]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "consistency.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), "utf-8")
    print(f"consistency: {len(rows)} questions, {len(bad)} with problems")
    for r in bad:
        print("  ", r["set"], r["id"], r["problems"], r["query"][:70])
    return 1 if bad else 0


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collisions")
    c.add_argument("--write-baseline", action="store_true")
    for name in ("sweep", "consistency"):
        p = sub.add_parser(name)
        p.add_argument("--mode", choices=["lexical", "hybrid"], default="lexical")
    args = parser.parse_args(argv)
    if args.cmd == "collisions":
        return cmd_collisions(args.write_baseline)
    if args.cmd == "sweep":
        return cmd_sweep(args.mode)
    return cmd_consistency(args.mode)


if __name__ == "__main__":
    sys.exit(main())
