"""B2: the final offline iteration (explicit quantity phrasings + top-3 display + guards).

``python -m mabhas19.semantic.evaluate2 tune``  - tune on development questions only (the spent
                                                  fresh1 set is never used) and freeze the manifest
``python -m mabhas19.semantic.evaluate2 run``   - A and B2 once on the frozen fresh2 set
``python -m mabhas19.semantic.evaluate2 score`` - metrics, gain attribution and the frozen GO rule

No provider and no spending. The B1 rule (frozen ``semantic1.yaml`` settings) is computed on the
same rankings only to attribute B2's gains (quantity phrasing vs top-3 display).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import product
from typing import Any

import yaml

from mabhas19.api import quantities as Q
from mabhas19.core.paths import DATA_DIR, EVAL_DIR
from mabhas19.semantic import catalog as C
from mabhas19.semantic import quantity as SQ
from mabhas19.semantic import verify as V
from mabhas19.semantic.evaluate import _embedder, _sha, _store, _usable_a, dev_cases
from mabhas19.semantic.retrieve import Retriever, Settings

FRESH = EVAL_DIR / "semantic" / "fresh2.yaml"
MANIFEST = EVAL_DIR / "semantic" / "semantic2.yaml"
B1_MANIFEST = EVAL_DIR / "semantic" / "semantic1.yaml"
OUT = DATA_DIR / "semantic"
GRID = {
    "lex_weight": (0.15, 0.3),
    "floor": (0.88, 0.9, 0.92, 0.94),
    "band": (0.02, 0.03, 0.05),
    "weak_floor": tuple(round(0.78 + 0.01 * i, 2) for i in range(16)),
    "tie_band": (0.005, 0.01, 0.02),
}
GO_RULE = {
    "wrong_quantity": 0,
    "wrong_values": 0,
    "trap_violations": 0,
    "wrong_row_offerings_max": 1,
    "usable_b2_min_share_of_eligible": 0.4,
    "usable_gain_over_a_min": 6,
    "top3_recall_engaged_min": 0.65,
    "top3_recall_first_run": "10/20 = 0.50",
    "unsupported_decline_min_share": 0.9,
    "eligible": "row questions with no digit (numbers stay excluded from semantic mapping)",
    "go": "every item holds; otherwise the semantic line is closed (no third iteration)",
}


def _offered_labels(p: Any) -> list[str]:
    return [r.label for r in p.offered]


def tune() -> int:
    store = _store()
    cat = C.build(store)
    emb = _embedder()
    base = Retriever(cat, store, emb, Settings(version="b2"), analyse=SQ.analyse)
    cases = dev_cases(cat)
    # rank once per case (scores do not depend on the decision settings except lex_weight)
    feats: list[tuple[dict[str, Any], Any, list[Any]]] = []
    for c in cases:
        p = base.propose(c["question"])
        if not p.ranked:
            feats.append((c, p, []))
            continue
        text = " ".join(p.words)
        qv = emb.embed_query(text)
        from mabhas19.semantic.retrieve import _cos, containment  # noqa: PLC0415

        rows = [
            (row, _cos(qv, base.vectors[row.id]), containment(text, row.passage()))
            for row in cat.for_quantity(p.quantity)
        ]
        feats.append((c, p, rows))
    best_key: tuple[float, ...] | None = None
    best = Settings(version="b2")
    best_counts = (0, 0)
    for lw, fl, bd, wf, tb in product(*GRID.values()):
        s = Settings(lw, fl, bd, "b2", wf, tb)
        r = Retriever.__new__(Retriever)
        r.settings = s
        good = wrong = 0
        for c, p, rows in feats:
            if not rows:
                continue
            from mabhas19.semantic.retrieve import Proposal, Scored  # noqa: PLC0415

            scored = sorted(
                (Scored(row, co, lx, co + lw * lx) for row, co, lx in rows),
                key=lambda x: -x.score,
            )
            q = Proposal(p.quantity, p.words)
            q.ranked = scored[:5]
            r._decide_b2(q, scored, p.words)
            offered = _offered_labels(q)
            if not offered:
                continue
            if c["quantity"] == "none" or p.quantity != c["quantity"]:
                wrong += 1
            elif set(offered) & set(c["rows"]):
                good += 1
            else:
                wrong += 1
        key = (good - 3 * wrong, -wrong, fl, wf, -tb)
        if best_key is None or key > best_key:
            best_key, best, best_counts = key, s, (good, wrong)
    manifest = {
        "version": "semantic-2",
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": "B2: explicit quantity phrasings + top-3 display + sibling / cross-table guards",
        "model": "intfloat/multilingual-e5-small (frozen candidate)",
        "catalogue": {"version": cat.version, "rows": len(cat.rows), "sha256": cat.sha256()},
        "settings": best.to_dict(),
        "tuning": {
            "dev": "eval/normalizer gold + eval/clarify/natural1.yaml (fresh1 is spent: not used)",
            "grid": {k: list(v) for k, v in GRID.items()},
            "dev_cases": len(cases),
            "dev_row_cases": sum(1 for c in cases if c["quantity"] != "none"),
            "dev_good": best_counts[0],
            "dev_wrong": best_counts[1],
            "objective": "good - 3*wrong offerings (ties: fewer wrong, higher floors, tighter tie)",
        },
        "fresh": {"path": FRESH.name, "sha256": _sha(FRESH)},
        "go_rule": GO_RULE,
    }
    text = yaml.safe_dump(manifest, allow_unicode=True, sort_keys=False)
    MANIFEST.write_text(
        "# frozen before the fresh2 run; do not edit\n" + text, encoding="utf-8", newline="\n"
    )
    print(json.dumps(manifest["settings"]), best_counts)
    return 0


def run() -> int:
    from mabhas19.api.engine import load_engine  # noqa: PLC0415

    m = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    if m["fresh"]["sha256"] != _sha(FRESH):
        print("the fresh set differs from the frozen manifest")
        return 2
    engine = load_engine("hybrid")
    store = engine.store
    cat = C.build(store)
    if cat.sha256() != m["catalogue"]["sha256"]:
        print("the catalogue differs from the frozen manifest")
        return 2
    emb = _embedder()
    b2 = Retriever(cat, store, emb, Settings(**m["settings"]), analyse=SQ.analyse)
    s1 = yaml.safe_load(B1_MANIFEST.read_text(encoding="utf-8"))["settings"]
    b1 = Retriever(cat, store, emb, Settings(**s1), analyse=SQ.analyse)  # attribution only
    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    for c in yaml.safe_load(FRESH.read_text(encoding="utf-8"))["cases"]:
        gold = set(c["rows"])
        t0 = time.perf_counter()
        a, _, _ = engine.ask(c["question"])
        rec: dict[str, Any] = {
            **c,
            "a_type": a.response_type,
            "a_usable": _usable_a(a, gold) if gold else False,
            "a_values": [v.row_label for v in a.values],
            "a_wrong_value": bool(a.values) and not {v.row_label for v in a.values} <= gold,
            "a_ms": round((time.perf_counter() - t0) * 1000, 1),
        }
        failed = a.response_type in ("prose", "refusal", "out_of_scope")
        failed = failed and not a.values and not a.clarifications
        rec["b_engaged"] = failed
        p = b2.propose(c["question"]) if failed else None
        rec["b"] = p.to_log() if p else None
        rec["b_ranked_labels"] = [x.row.label for x in p.ranked] if p else []
        rec["prod_quantity"] = bool(Q.analyse(c["question"]))
        rec["b1_offered"] = _offered_labels(b1.propose(c["question"])) if failed else []
        rec["b_final"] = ""
        rec["b_final_values"] = []
        if p and p.offered:
            pick = next((x for x in p.offered if x.label in gold), None)
            if pick is not None:
                q = V.canonical(p.quantity, pick, p.code, tuple(p.cities))
                fa, _, _ = engine.ask(q)
                rec["b_canonical"] = q
                rec["b_final"] = V.about(fa, pick)
                rec["b_final_values"] = [(v.row_label, v.value) for v in fa.values]
        records.append(rec)
        print(c["id"], c["group"], rec["a_usable"], (p.decision if p else "-"), rec["b_final"])
    (OUT / "run2.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return 0


def score() -> int:
    m = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    rule = m["go_rule"]
    recs = json.loads((OUT / "run2.json").read_text(encoding="utf-8"))
    digits = set("0123456789۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩")
    rows_ = [x for x in recs if x["group"] == "row"]
    eligible = [x for x in rows_ if not set(x["question"]) & digits]
    traps = [x for x in recs if x["group"] == "trap"]
    unsup = [x for x in recs if x["group"] == "unsupported"]

    def proposed(x: dict[str, Any]) -> bool:
        return bool(x["b"] and x["b"]["offered"])

    def usable(x: dict[str, Any]) -> bool:
        return x["a_usable"] or x["b_final"] in ("value", "condition")

    engaged = [x for x in recs if x["b_engaged"]]
    row_eng = [x for x in eligible if x["b_engaged"]]
    gained = [x for x in eligible if usable(x) and not x["a_usable"]]
    metrics = {
        "eligible_row_questions": len(eligible),
        "usable_a": sum(x["a_usable"] for x in eligible),
        "usable_b2": sum(usable(x) for x in eligible),
        "engaged": len(engaged),
        "quantity_resolved_engaged": sum(1 for x in engaged if x["b"]["quantity"]),
        "wrong_quantity": sum(
            1
            for x in engaged
            if proposed(x) and (x["quantity"] == "none" or x["b"]["quantity"] != x["quantity"])
        ),
        "wrong_row_offerings": sum(
            1 for x in rows_ + unsup if proposed(x) and not set(x["b"]["offered"]) & set(x["rows"])
        ),
        "wrong_values": sum(1 for x in recs if x["a_wrong_value"])
        + sum(1 for x in recs if x["b_final"] == "" and x["b_final_values"]),
        "trap_violations": sum(1 for x in traps if proposed(x)),
        "unsupported_declined": f"{sum(1 for x in unsup if not proposed(x))}/{len(unsup)}",
        "row_top1_engaged": sum(
            1 for x in row_eng if x["b_ranked_labels"][:1] and x["b_ranked_labels"][0] in x["rows"]
        ),
        "row_top3_engaged": sum(
            1 for x in row_eng if set(x["b_ranked_labels"][:3]) & set(x["rows"])
        ),
        "row_engaged": len(row_eng),
        "false_refusals": sum(1 for x in row_eng if not proposed(x)),
        "decisions": {
            d: sum(1 for x in engaged if x["b"]["decision"] == d)
            for d in ("confirm", "choose", "abstain")
        },
        "gain_cases": [x["id"] for x in gained],
        "gain_from_quantity_phrasing": [x["id"] for x in gained if not x["prod_quantity"]],
        "gain_from_top3_display": [
            x["id"]
            for x in gained
            if x["prod_quantity"] and not set(x["b1_offered"]) & set(x["rows"])
        ],
        "latency_ms_semantic_max": max((x["b"]["ms"] for x in engaged if x["b"]), default=0),
    }
    top3 = metrics["row_top3_engaged"] / metrics["row_engaged"] if metrics["row_engaged"] else 0
    unsup_ok = sum(1 for x in unsup if not proposed(x))
    checks = {
        "wrong_quantity": metrics["wrong_quantity"] == rule["wrong_quantity"],
        "wrong_values": metrics["wrong_values"] == rule["wrong_values"],
        "trap_violations": metrics["trap_violations"] == rule["trap_violations"],
        "wrong_row_offerings": metrics["wrong_row_offerings"] <= rule["wrong_row_offerings_max"],
        "usable_share": metrics["usable_b2"]
        >= rule["usable_b2_min_share_of_eligible"] * metrics["eligible_row_questions"],
        "usable_gain": metrics["usable_b2"] - metrics["usable_a"] >= rule["usable_gain_over_a_min"],
        "top3_recall": top3 >= rule["top3_recall_engaged_min"],
        "unsupported_declined": unsup_ok >= rule["unsupported_decline_min_share"] * len(unsup),
    }
    out = {
        "metrics": metrics,
        "top3_recall": round(top3, 3),
        "checks": checks,
        "decision": "GO" if all(checks.values()) else "NO-GO",
    }
    (OUT / "run2_score.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    ap = argparse.ArgumentParser(description="B2 offline evaluation")
    ap.add_argument("command", choices=["tune", "run", "score"])
    args = ap.parse_args(argv)
    return {"tune": tune, "run": run, "score": score}[args.command]()


if __name__ == "__main__":
    raise SystemExit(main())
