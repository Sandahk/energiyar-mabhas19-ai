"""Offline evaluation of option B (no provider, no spending).

``python -m mabhas19.semantic.evaluate tune``   - grid-search the three settings on development
                                                   questions only and write the frozen manifest
``python -m mabhas19.semantic.evaluate run``    - A (current chatbot) and B (+ semantic rows) on the
                                                   frozen fresh set, once; logs to data/semantic/
``python -m mabhas19.semantic.evaluate score``  - metrics and the frozen GO rule

Development questions: the normalizer experiment's gold (``eval/normalizer``) and the frozen
natural set (``eval/clarify/natural1.yaml``). The fresh set (``eval/semantic/fresh1.yaml``) is never
used for tuning; ``run`` refuses to start unless the manifest is frozen and its hashes match.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from itertools import product
from typing import Any

import yaml

from mabhas19.core.paths import DATA_DIR, EVAL_DIR
from mabhas19.semantic import catalog as C
from mabhas19.semantic import verify as V
from mabhas19.semantic.retrieve import MAX_CHOICES, Retriever, Settings

FRESH = EVAL_DIR / "semantic" / "fresh1.yaml"
MANIFEST = EVAL_DIR / "semantic" / "semantic1.yaml"
OUT = DATA_DIR / "semantic"
FAMILY = {
    "energy_limit": "eui",
    "opaque_r": "r_value",
    "hot_water": "hot_water",
    "equipment_central": "efficiency",
    "equipment_unit": "efficiency",
    "daylight": "daylight",
    "light_uniformity": "uniformity",
}
GRID = {
    "lex_weight": (0.0, 0.15, 0.3, 0.5),
    "floor": tuple(round(0.80 + 0.01 * i, 2) for i in range(16)),
    "band": (0.01, 0.02, 0.03, 0.05),
}
GO_RULE = {
    "wrong_quantity": "0 (a B proposal of a quantity other than the gold; on a trap, any proposal)",
    "wrong_values": "0 (a final engine value of a row other than a gold row)",
    "wrong_row_offerings_max": 1,
    "usable_gain_min": 6,
    "usable_b_min_share_of_row_cases": 0.5,
    "unsupported_abstain_min": 9,
    "traps_with_no_proposal_min": 8,
    "go": "every item holds; otherwise option B is not promoted (no tuning on the fresh set)",
}


def _sha(path: Any) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _store() -> Any:
    from mabhas19.core.paths import CURATED_DIR  # noqa: PLC0415
    from mabhas19.retrieval.structured import StructuredStore  # noqa: PLC0415

    return StructuredStore(CURATED_DIR)


def _embedder() -> Any:
    from mabhas19.cli.embedding_experiment import CANDIDATE  # noqa: PLC0415
    from mabhas19.providers.local_embeddings import SentenceTransformerEmbedder  # noqa: PLC0415

    model = yaml.safe_load(CANDIDATE.read_text(encoding="utf-8"))["embedding"]["model_id"]
    return SentenceTransformerEmbedder(model, threads=4)


def dev_cases(cat: C.Catalogue) -> list[dict[str, Any]]:
    """Development questions with gold quantity and rows (row) or none (abstain)."""
    out: list[dict[str, Any]] = []
    norm = EVAL_DIR / "normalizer"
    qs = {
        a["id"]: a["question"]
        for a in yaml.safe_load((norm / "authored1.yaml").read_text(encoding="utf-8"))
    }
    cases = yaml.safe_load((norm / "cases1.yaml").read_text(encoding="utf-8"))
    for c in cases["failure"]:
        g = c["gold"]
        fam = g["family"]
        if fam == "window":
            q = "shgc" if any("SHGC" in e for e in g["entities"]) else "u_value"
        elif fam == "material":
            q = "lambda" if g["output"] == "thermal_conductivity" else "density"
        else:
            q = FAMILY.get(fam, "")
        if q:
            out.append(
                {"id": c["id"], "question": qs[c["id"]], "quantity": q, "rows": g["entities"]}
            )
    for c in cases["abstain_expected"]:
        out.append({"id": c["id"], "question": qs[c["id"]], "quantity": "none", "rows": []})
    natural = yaml.safe_load((EVAL_DIR / "clarify" / "natural1.yaml").read_text(encoding="utf-8"))
    by_label: dict[str, set[str]] = {}
    for r in cat.rows:
        by_label.setdefault(r.label, set()).update(r.quantities)
    for c in natural["cases"]:
        if c["expect"] == "no_answer":
            out.append({"id": c["id"], "question": c["question"], "quantity": "none", "rows": []})
        elif c.get("rows") and all(r in by_label for r in c["rows"]):
            qs_ = set.intersection(*(by_label[r] for r in c["rows"]))
            if len(qs_) == 1:
                out.append(
                    {
                        "id": c["id"],
                        "question": c["question"],
                        "quantity": qs_.pop(),
                        "rows": c["rows"],
                    }
                )
    return out


def _decide(ranked: list[tuple[str, float]], s: Settings) -> list[str]:
    """The offered labels for (label, score) pairs sorted best first (the retriever's rule)."""
    if not ranked:
        return []
    best = ranked[0][1]
    close = [lab for lab, sc in ranked[:5] if sc >= s.floor and best - sc <= s.band]
    return close if 1 <= len(close) <= MAX_CHOICES else []


def tune() -> int:
    store = _store()
    cat = C.build(store)
    emb = _embedder()
    r = Retriever(cat, store, emb, Settings())
    cases = dev_cases(cat)
    feats = []  # per case: (gold quantity, gold rows, resolved quantity, [(label, cos, lex)])
    for c in cases:
        p = r.propose(c["question"])  # resolves the quantity and words once
        rows = []
        if p.ranked:  # the retriever reached ranking (quantity, row words, no number)
            text = " ".join(p.words)
            qv = emb.embed_query(text)
            from mabhas19.semantic.retrieve import _cos, containment  # noqa: PLC0415

            for row in cat.for_quantity(p.quantity):
                rows.append(
                    (row.label, _cos(qv, r.vectors[row.id]), containment(text, row.passage()))
                )
        feats.append((c, p.quantity, rows))
    best: tuple[float, ...] | None = None
    best_s = Settings()
    for lw, fl, bd in product(GRID["lex_weight"], GRID["floor"], GRID["band"]):
        s = Settings(lw, fl, bd)
        good = wrong = 0
        for c, qid, rows in feats:
            ranked = sorted(((lab, co + lw * lx) for lab, co, lx in rows), key=lambda x: -x[1])
            offered = _decide(ranked, s)
            if not offered:
                continue
            if c["quantity"] == "none" or qid != c["quantity"] or not set(offered) & set(c["rows"]):
                wrong += 1
            else:
                good += 1
        key = (good - 3 * wrong, -wrong, fl, -bd)  # ties: fewer wrong, higher floor, tighter band
        if best is None or key > best:
            best, best_s = key, s
    n_row = sum(1 for c in cases if c["quantity"] != "none")
    manifest = {
        "version": "semantic-1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": "option B: deterministic quantity + offline E5/lexical rows; no provider",
        "model": "intfloat/multilingual-e5-small (frozen candidate)",
        "catalogue": {"version": cat.version, "rows": len(cat.rows), "sha256": cat.sha256()},
        "settings": best_s.to_dict(),
        "tuning": {
            "grid": {k: list(v) for k, v in GRID.items()},
            "dev_cases": len(cases),
            "dev_row_cases": n_row,
            "objective": "good - 3*wrong offerings (ties: fewer wrong, higher floor, tighter band)",
            "dev_good": int(best[0] + 3 * -best[1]) if best else 0,
            "dev_wrong": int(-best[1]) if best else 0,
        },
        "fresh": {"path": FRESH.name, "sha256": _sha(FRESH)},
        "go_rule": GO_RULE,
    }
    text = yaml.safe_dump(manifest, allow_unicode=True, sort_keys=False)
    MANIFEST.write_text(
        "# frozen before the fresh run; do not edit\n" + text, encoding="utf-8", newline="\n"
    )
    print(json.dumps(manifest["settings"]), json.dumps(manifest["tuning"], ensure_ascii=False))
    return 0


def _usable_a(answer: Any, rows: set[str]) -> bool:
    if answer.response_type == "structured_value" and answer.values:
        return {v.row_label for v in answer.values} <= rows
    if answer.response_type == "clarification" and answer.clarifications:
        c = answer.clarifications[0]
        if {o.label for o in c.options} & rows:
            return True
        return c.field in V.CONDITIONS and any(
            any(r in i.value for r in rows) for i in answer.interpretation
        )
    return False


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
    s = Settings(**m["settings"])
    r = Retriever(cat, store, engine.embedder if hasattr(engine, "embedder") else _embedder(), s)
    fresh = yaml.safe_load(FRESH.read_text(encoding="utf-8"))["cases"]
    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    for c in fresh:
        gold = set(c["rows"])
        t0 = time.perf_counter()
        a, _, _ = engine.ask(c["question"])
        a_ms = (time.perf_counter() - t0) * 1000
        rec: dict[str, Any] = {
            **c,
            "a_type": a.response_type,
            "a_values": [v.row_label for v in a.values],
            "a_options": [o.label for x in a.clarifications for o in x.options],
            "a_usable": _usable_a(a, gold) if gold else False,
            "a_wrong_value": bool(a.values) and not {v.row_label for v in a.values} <= gold,
            "a_ms": round(a_ms, 1),
        }
        failed = a.response_type in ("prose", "refusal", "out_of_scope") and not a.values
        failed = failed and not a.clarifications
        rec["b_engaged"] = failed
        p = r.propose(c["question"]) if failed else None
        rec["b"] = p.to_log() if p else None
        rec["b_ranked_labels"] = [x.row.label for x in p.ranked] if p else []
        rec["b_final"] = ""
        rec["b_final_values"] = []
        if p and p.offered:
            pick = next((x for x in p.offered if x.label in gold), None)
            if pick is not None:
                q = V.canonical(p.quantity, pick, p.code, tuple(p.cities))
                t1 = time.perf_counter()
                fa, _, _ = engine.ask(q)
                rec["b_canonical"] = q
                rec["b_final"] = V.about(fa, pick)
                rec["b_final_values"] = [(v.row_label, v.value) for v in fa.values]
                rec["b_final_ms"] = round((time.perf_counter() - t1) * 1000, 1)
        records.append(rec)
        print(c["id"], c["group"], rec["a_usable"], (p.decision if p else "-"), rec["b_final"])
    (OUT / "run1.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return 0


def score() -> int:
    m = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    recs = json.loads((OUT / "run1.json").read_text(encoding="utf-8"))
    rows_ = [x for x in recs if x["group"] == "row"]
    traps = [x for x in recs if x["group"] == "trap"]
    unsup = [x for x in recs if x["group"] == "unsupported"]

    def proposed(x: dict[str, Any]) -> bool:
        return bool(x["b"] and x["b"]["offered"])

    def b_usable(x: dict[str, Any]) -> bool:
        return x["a_usable"] or x["b_final"] in ("value", "condition")

    engaged = [x for x in recs if x["b_engaged"]]
    resolved = [x for x in engaged if x["b"]["quantity"]]
    wrong_q = sum(
        1
        for x in engaged
        if proposed(x) and (x["quantity"] == "none" or x["b"]["quantity"] != x["quantity"])
    )
    wrong_rows = sum(
        1 for x in rows_ + unsup if proposed(x) and not set(x["b"]["offered"]) & set(x["rows"])
    )
    wrong_values = sum(1 for x in recs if x["a_wrong_value"])
    row_engaged = [x for x in rows_ if x["b_engaged"]]
    metrics = {
        "row_cases": len(rows_),
        "usable_a": sum(x["a_usable"] for x in rows_),
        "usable_b": sum(b_usable(x) for x in rows_),
        "b_engaged": len(engaged),
        "quantity_resolved_when_engaged": f"{len(resolved)}/{len(engaged)}",
        "quantity_correct_when_resolved": sum(
            1 for x in resolved if x["b"]["quantity"] == x["quantity"]
        ),
        "table_correct_when_proposed": sum(
            1 for x in row_engaged if proposed(x) and x["b"]["quantity"] == x["quantity"]
        ),
        "row_top1_engaged": sum(
            1
            for x in row_engaged
            if x["b_ranked_labels"][:1] and x["b_ranked_labels"][0] in x["rows"]
        ),
        "row_top3_engaged": sum(
            1 for x in row_engaged if set(x["b_ranked_labels"][:3]) & set(x["rows"])
        ),
        "row_cases_engaged": len(row_engaged),
        "decisions_engaged": {
            d: sum(1 for x in engaged if x["b"]["decision"] == d)
            for d in ("confirm", "choose", "abstain")
        },
        "wrong_quantity": wrong_q,
        "wrong_row_offerings": wrong_rows,
        "wrong_values": wrong_values
        + sum(1 for x in recs if x["b_final"] == "" and x["b_final_values"]),
        "traps_no_proposal": sum(1 for x in traps if not proposed(x)),
        "unsupported_abstain": sum(1 for x in unsup if not proposed(x)),
        "false_refusals": sum(1 for x in row_engaged if not proposed(x)),
        "latency_ms_semantic": sorted(x["b"]["ms"] for x in engaged if x["b"]),
        "latency_ms_engine_a": sorted(x["a_ms"] for x in recs),
    }
    rule = m["go_rule"]
    checks = {
        "wrong_quantity": metrics["wrong_quantity"] == 0,
        "wrong_values": metrics["wrong_values"] == 0,
        "wrong_row_offerings": metrics["wrong_row_offerings"] <= rule["wrong_row_offerings_max"],
        "usable_gain": metrics["usable_b"] - metrics["usable_a"] >= rule["usable_gain_min"],
        "usable_b_share": metrics["usable_b"]
        >= rule["usable_b_min_share_of_row_cases"] * metrics["row_cases"],
        "unsupported_abstain": metrics["unsupported_abstain"] >= rule["unsupported_abstain_min"],
        "traps": metrics["traps_no_proposal"] >= rule["traps_with_no_proposal_min"],
    }
    decision = "GO" if all(checks.values()) else "NO-GO"
    out = {"metrics": metrics, "checks": checks, "decision": decision}
    (OUT / "run1_score.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    ap.add_argument("command", choices=["tune", "run", "score"])
    args = ap.parse_args(argv)
    return {"tune": tune, "run": run, "score": score}[args.command]()


if __name__ == "__main__":
    raise SystemExit(main())
