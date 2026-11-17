"""Normalizer shadow experiment 1: freeze, dry run, one run, score.

``python -m mabhas19.normalizer.experiment freeze``
    validates the cases and gold against the printed catalog, records the deterministic engine's
    response to every case (the trigger snapshot) and writes the frozen manifest
    ``eval/normalizer/normalizer1.yaml`` (+ ``.sha256``). No network.
``python -m mabhas19.normalizer.experiment dry-run``
    renders every request that would be sent, with a conservative token bound and worst-case
    cost. No network.
``python -m mabhas19.normalizer.experiment run --allow-external --budget-usd 5``
    the one run: refuses unless the manifest verifies, ``GEMINI_API_KEY`` is set and a budget is
    given. Shadow only: the live answer is never changed. One call per trigger case; a stored
    record is never re-run. Records: ``data/normalizer/run1/``.
``python -m mabhas19.normalizer.experiment score``
    the frozen decision rule over the stored records (no network).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from mabhas19.core.paths import DATA_DIR, EVAL_DIR, REPO_ROOT
from mabhas19.normalizer import catalog as C
from mabhas19.normalizer import prompt as P
from mabhas19.normalizer.verify import Verified, parse, verify

MANIFEST = EVAL_DIR / "normalizer" / "normalizer1.yaml"
CASES = EVAL_DIR / "normalizer" / "cases1.yaml"
AUTHORED = EVAL_DIR / "normalizer" / "authored1.yaml"
RUN_DIR = DATA_DIR / "normalizer" / "run1"
TRIGGER = ("prose", "refusal", "out_of_scope")  # the engine failed closed
SETTINGS: dict[str, Any] = {
    "provider": "gemini",
    "model_id": "gemini-4-argon",
    "effort": "medium",
    "max_output_tokens": 16000,
    "timeout_s": 300.0,
    "max_retries": 1,
}
DECISION_RULE: dict[str, Any] = {
    "useful_rate_min": 0.60,  # useful / mappable failure cases
    "wrong_rate_max": 0.10,  # wrong / all failure + abstain-expected cases
    "value_leaks_max": 0,  # canonical-question numbers with no source (question or printed label)
    "latency_p90_s_max": 20.0,  # over completed calls (the suggestion is shown asynchronously)
    "cost_per_call_usd_max": 0.05,  # mean reported cost per call
    "failed_calls_max": 3,  # more: the run is void (reported; no re-run without approval)
    "controls": "the trigger never fires on a control and every control answer is unchanged",
    "definitions": {
        "valid": "after deterministic verification a family, its output and >=1 printed row "
        "(or a city found by the exact city lookup) survive",
        "correct": "valid; family and output equal gold; every verified row is in the gold "
        "accept set; a verified city / climate / number is absent or equal to gold",
        "useful": "correct and at least one canonical question gets a certified value or a "
        "clarification from the gold table (the unchanged engine)",
        "wrong": "valid and not correct (a wrong confirmation question would be shown); on an "
        "abstain-expected case every valid proposal is wrong",
    },
    "go": "all thresholds hold; otherwise stop the normalizer line (no prompt tuning, no "
    "second model, no new benchmark)",
}
SOURCES = (
    "normalizer/__init__.py",
    "normalizer/catalog.py",
    "normalizer/prompt.py",
    "normalizer/verify.py",
    "normalizer/experiment.py",
    "api/orchestration.py",
)


def _text_bytes(path: Path) -> bytes:
    """File bytes with line endings normalized (a Windows checkout writes CRLF, git stores LF)."""
    return path.read_bytes().replace(b"\r\n", b"\n")


def _sha(path: Path) -> str:
    return hashlib.sha256(_text_bytes(path)).hexdigest()


def _engine() -> Any:
    from mabhas19.api.engine import load_engine  # noqa: PLC0415

    return load_engine("hybrid")


def _questions() -> dict[str, str]:
    authored = yaml.safe_load(AUTHORED.read_text(encoding="utf-8"))
    return {str(q["id"]): str(q["question"]) for q in authored}


def _cases() -> dict[str, Any]:
    cases: dict[str, Any] = yaml.safe_load(CASES.read_text(encoding="utf-8"))
    return cases


def _answer_dump(answer: Any) -> dict[str, Any]:
    d: dict[str, Any] = answer.to_dict()
    d.pop("request_id", None)
    return d


def _check_gold(store: Any, cases: dict[str, Any]) -> list[str]:
    problems = []
    for c in cases["failure"]:
        g = c["gold"]
        fam = C.BY_NAME.get(g["family"])
        if fam is None:
            problems.append(f"{c['id']}: family {g['family']}")
            continue
        printed = C.candidates(store, fam)
        problems += [
            f"{c['id']}: gold «{e}» not printed" for e in g["entities"] if e not in printed
        ]
        if g["output"] not in fam.outputs:
            problems.append(f"{c['id']}: output {g['output']}")
    return problems


def code_fingerprint() -> str:
    h = hashlib.sha256()
    for rel in SOURCES:
        h.update(rel.encode())
        h.update(_text_bytes(REPO_ROOT / "src" / "mabhas19" / rel))
    return h.hexdigest()


def freeze() -> int:
    engine = _engine()
    store = engine.store
    cases = _cases()
    problems = _check_gold(store, cases)
    if problems:
        print("gold problems:", *problems, sep="\n  ")
        return 2
    qs = _questions()
    cat = C.build(store)
    snapshot = {}
    for kind in ("failure", "abstain_expected"):
        for c in cases[kind]:
            a, _, _ = engine.ask(qs[c["id"]])
            snapshot[c["id"]] = a.response_type
            if a.response_type not in TRIGGER:
                problems.append(f"{c['id']}: engine returned {a.response_type}, not a trigger")
    for cid in cases["control"]:
        a, _, _ = engine.ask(qs[cid])
        snapshot[cid] = a.response_type
        if a.response_type in TRIGGER:
            problems.append(f"{cid}: control triggers ({a.response_type})")
    if problems:
        print("selection problems:", *problems, sep="\n  ")
        return 2
    manifest = {
        "schema": "normalizer-experiment-1",
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "purpose": "LLM suggest-and-confirm normalizer, one finite shadow run (not Final-N)",
        "files": {
            "authored": {"path": AUTHORED.name, "sha256": _sha(AUTHORED)},
            "cases": {"path": CASES.name, "sha256": _sha(CASES)},
        },
        "counts": {
            "failure": len(cases["failure"]),
            "abstain_expected": len(cases["abstain_expected"]),
            "control": len(cases["control"]),
            "excluded": len(cases["excluded"]),
        },
        "pins": {
            "catalog_version": C.CATALOG_VERSION,
            "catalog_sha256": C.sha256(cat),
            "prompt_version": P.PROMPT_VERSION,
            "prompt_sha256": P.prompt_sha256(cat),
            "schema_sha256": P.schema_sha256(cat),
            "code_sha256": code_fingerprint(),
            "engine_composer": engine.info.get("composer"),
            "engine_pins": "final8+prepilot",
        },
        "settings": SETTINGS,
        "price_usd_per_mtok": {
            "input": 2.0,
            "output_incl_thinking": 10.0,
            "valid_until": "2026-11-30",
        },
        "trigger": list(TRIGGER),
        "trigger_snapshot": snapshot,
        "decision_rule": DECISION_RULE,
    }
    text = yaml.safe_dump(manifest, allow_unicode=True, sort_keys=False)
    MANIFEST.write_text("# frozen; do not edit\n" + text, encoding="utf-8", newline="\n")
    sha_line = f"{_sha(MANIFEST)}  {MANIFEST.name}\n"
    MANIFEST.with_suffix(".sha256").write_text(sha_line, encoding="utf-8", newline="\n")
    print(f"frozen {MANIFEST.name} sha256 {_sha(MANIFEST)}")
    return 0


def verify_frozen(store: Any) -> list[str]:
    """Differences between the frozen manifest and the current files / code / catalog."""
    out = []
    recorded = MANIFEST.with_suffix(".sha256").read_text().split()[0]
    if recorded != _sha(MANIFEST):
        out.append("manifest sha256 differs")
    m = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    for f in m["files"].values():
        if _sha(MANIFEST.parent / f["path"]) != f["sha256"]:
            out.append(f"{f['path']} changed")
    cat = C.build(store)
    pins = m["pins"]
    for key, now in (
        ("catalog_sha256", C.sha256(cat)),
        ("prompt_sha256", P.prompt_sha256(cat)),
        ("schema_sha256", P.schema_sha256(cat)),
        ("code_sha256", code_fingerprint()),
    ):
        if pins[key] != now:
            out.append(f"{key} changed")
    if m["settings"] != SETTINGS:
        out.append("settings changed")
    return out


def _trigger_ids(m: dict[str, Any]) -> list[str]:
    cases = _cases()
    return [c["id"] for c in cases["failure"]] + [c["id"] for c in cases["abstain_expected"]]


def dry_run() -> int:
    from mabhas19.core.paths import CURATED_DIR  # noqa: PLC0415
    from mabhas19.retrieval.structured import StructuredStore  # noqa: PLC0415

    store = StructuredStore(CURATED_DIR)
    m = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    print("freeze:", verify_frozen(store) or "verified")
    cat = C.build(store)
    qs = _questions()
    ids = _trigger_ids(m)
    bodies = [
        P.request_body(cat, qs[i], effort=SETTINGS["effort"], max_output_tokens=16000) for i in ids
    ]
    chars = [len(json.dumps(b, ensure_ascii=False)) for b in bodies]
    worst = sum(c * 2.0 + 16000 * 10.0 for c in chars) / 1e6
    print(f"calls: {len(ids)}; request chars (token upper bound) max {max(chars)}")
    print(f"worst case (1 token per char + all 16000 output tokens): ${worst:.2f}")
    print(
        f"system text chars: {len(P.system_text(cat))}; "
        f"catalog labels (plain strings in the schema): {len(C.all_labels(cat))}"
    )
    return 0


def _proposer(budget_usd: float, allow_external: bool) -> Any:
    from mabhas19.shadow.provider import Budget, GeminiProposer  # noqa: PLC0415
    from mabhas19.shadow.replay import GEMINI, ProposerSettings  # noqa: PLC0415

    settings = ProposerSettings(
        provider=GEMINI,
        model_id=SETTINGS["model_id"],
        effort=SETTINGS["effort"],
        max_output_tokens=SETTINGS["max_output_tokens"],
        timeout_s=SETTINGS["timeout_s"],
        max_retries=SETTINGS["max_retries"],
    )

    class NormalizerProposer(GeminiProposer):
        """The audited Gemini adapter with this experiment's request body."""

        def __init__(self, cat: dict[str, Any], **kw: Any) -> None:
            super().__init__(settings, **kw)
            self._cat = cat

        def request(self, ctx: Any) -> dict[str, Any]:
            return P.request_body(
                self._cat,
                str(ctx),
                effort=settings.effort,
                max_output_tokens=settings.max_output_tokens,
            )

    def make(cat: dict[str, Any]) -> Any:
        return NormalizerProposer(
            cat,
            allow_external=allow_external,
            manifest_frozen=True,
            budget=Budget(budget_usd, 2_000_000),
        )

    return make


def run(budget_usd: float, allow_external: bool) -> int:
    engine = _engine()
    store = engine.store
    problems = verify_frozen(store)
    if problems:
        print("manifest does not verify:", *problems, sep="\n  ")
        return 2
    m = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    cat = C.build(store)
    qs = _questions()
    cases = _cases()
    proposer = _proposer(budget_usd, allow_external)(cat)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    before = {cid: _answer_dump(engine.ask(qs[cid])[0]) for cid in cases["control"]}
    failed = 0
    for cid in _trigger_ids(m):
        path = RUN_DIR / f"{cid}.json"
        if path.exists():  # never re-run a stored record
            continue
        question = qs[cid]
        live, _, _ = engine.ask(question)
        rec: dict[str, Any] = {"id": cid, "question": question, "live_type": live.response_type}
        if live.response_type not in TRIGGER:
            rec["triggered"] = False
        else:
            rec["triggered"] = True
            resp = proposer.propose(question)
            rec["response"] = {
                "raw_text": resp.raw_text,
                "model": resp.response_model,
                "stop": resp.stop_reason,
                "usage": resp.usage,
                "valid_call": resp.experiment_valid,
                "invalid_reason": resp.invalid_reason,
            }
            failed += 0 if resp.raw_text is not None else 1
            doc, err = parse(resp.raw_text)
            v = verify(doc, question, store, engine._run, err)
            rec["verified"] = v.to_dict()
            rec["canonical_answers"] = _canonical_answers(engine, v)
            rec["live_unchanged"] = _answer_dump(engine.ask(question)[0]) == _answer_dump(live)
        rec["spent_usd"] = round(proposer.meter.spent_usd, 4)
        path.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
        print(cid, rec.get("verified", {}).get("valid"), rec["spent_usd"])
        if failed > DECISION_RULE["failed_calls_max"]:
            print("too many failed calls: run stopped (void)")
            break
    after = {cid: _answer_dump(engine.ask(qs[cid])[0]) for cid in cases["control"]}
    (RUN_DIR / "_controls.json").write_text(
        json.dumps(
            {
                cid: {
                    "type": before[cid]["response_type"],
                    "triggered": before[cid]["response_type"] in TRIGGER,
                    "unchanged": _strip(before[cid]) == _strip(after[cid]),
                }
                for cid in before
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    print(f"spent ${proposer.meter.spent_usd:.4f} over {proposer.meter.calls} calls")
    return 0


def _strip(d: dict[str, Any]) -> dict[str, Any]:
    """An answer dump without per-call identifiers of its parts."""
    s = dict(d.get("safety") or {})
    comp = s.get("composite")
    if isinstance(comp, dict):
        comp = {k: v for k, v in comp.items() if k != "parts"}
        s["composite"] = comp
    return {**d, "safety": s}


def _canonical_answers(engine: Any, v: Verified) -> list[dict[str, Any]]:
    out = []
    for q in v.canonical:
        a, _, _ = engine.ask(q)
        out.append(
            {
                "question": q,
                "type": a.response_type,
                "datasets": sorted(
                    {x.dataset for x in a.values} | {c.dataset for c in a.clarifications}
                ),
                "values": [(x.record_id, x.value) for x in a.values],
                "clarify": [c.field for c in a.clarifications],
                "violations": a.safety.get("provenance_violations"),
            }
        )
    return out


def _correct(v: dict[str, Any], gold: dict[str, Any]) -> bool:
    if not v["valid"] or v["family"] != gold["family"] or v["output"] != gold["output"]:
        return False
    if gold["family"] == C.CITY_FAMILY:
        return bool(v["city"] == gold.get("city"))
    if not v["entities"] or not set(v["entities"]) <= set(gold["entities"]):
        return False
    if v["city"] and v["city"] != gold.get("city"):
        return False
    if v["climate"] and v["climate"] != gold.get("climate"):
        return False
    return all(text in (gold.get("numbers") or []) for _, text in v["numbers"])


def score() -> int:
    cases = _cases()
    rows, latency, costs, failed, leaks = [], [], [], 0, 0
    useful = correct = wrong = abstained = invalid = 0
    gold_tables = {f.name: f.dataset for f in C.FAMILIES}
    for kind in ("failure", "abstain_expected"):
        for c in cases[kind]:
            path = RUN_DIR / f"{c['id']}.json"
            if not path.exists():
                rows.append({"id": c["id"], "kind": kind, "status": "not run"})
                continue
            rec = json.loads(path.read_text(encoding="utf-8"))
            resp = rec.get("response") or {}
            usage = resp.get("usage") or {}
            if resp.get("raw_text") is None:
                failed += 1
            elif "latency_ms" in usage:
                latency.append(usage["latency_ms"] / 1000)
                costs.append(
                    (usage.get("input_tokens", 0) * 2.0 + usage.get("output_tokens", 0) * 10.0)
                    / 1e6
                )
            v = rec.get("verified") or {}
            leaks += len(v.get("leak") or [])
            row = {
                "id": c["id"],
                "kind": kind,
                "valid": v.get("valid"),
                "family": v.get("family"),
                "entities": v.get("entities"),
                "climate": v.get("climate"),
                "city": v.get("city"),
                "discarded": v.get("discarded"),
                "confirmation": v.get("confirmation"),
                "canonical": rec.get("canonical_answers"),
                "live_unchanged": rec.get("live_unchanged"),
            }
            if kind == "failure":
                ok = _correct(v, c["gold"]) if v else False
                ds = gold_tables[c["gold"]["family"]]
                reach = any(
                    a["type"] in ("structured_value", "clarification")
                    and (
                        ds in a["datasets"]
                        or (c["gold"]["family"] == C.CITY_FAMILY and a["values"])
                    )
                    for a in rec.get("canonical_answers") or []
                )
                row["status"] = (
                    "useful"
                    if ok and reach
                    else "correct_unreachable"
                    if ok
                    else (
                        "wrong"
                        if v.get("valid")
                        else "abstain"
                        if v.get("abstained")
                        else "invalid"
                    )
                )
                correct += ok
                useful += ok and reach
                wrong += bool(v.get("valid")) and not ok
            else:
                row["status"] = "wrong" if v.get("valid") else "abstain_ok"
                wrong += bool(v.get("valid"))
            abstained += row["status"].startswith("abstain")
            invalid += row["status"] == "invalid"
            rows.append(row)
    n_fail = len(cases["failure"])
    n_all = n_fail + len(cases["abstain_expected"])
    p90 = sorted(latency)[max(0, math.ceil(0.9 * len(latency)) - 1)] if latency else None
    controls_path = RUN_DIR / "_controls.json"
    controls = (
        json.loads(controls_path.read_text(encoding="utf-8")) if controls_path.exists() else {}
    )
    controls_ok = bool(controls) and all(
        not x["triggered"] and x["unchanged"] for x in controls.values()
    )
    r = DECISION_RULE
    checks = {
        "useful_rate": (useful / n_fail, useful / n_fail >= r["useful_rate_min"]),
        "wrong_rate": (wrong / n_all, wrong / n_all <= r["wrong_rate_max"]),
        "value_leaks": (leaks, leaks <= r["value_leaks_max"]),
        "latency_p90_s": (p90, p90 is not None and p90 <= r["latency_p90_s_max"]),
        "cost_per_call_usd": (
            sum(costs) / len(costs) if costs else None,
            bool(costs) and sum(costs) / len(costs) <= r["cost_per_call_usd_max"],
        ),
        "failed_calls": (failed, failed <= r["failed_calls_max"]),
        "controls": (controls, controls_ok),
    }
    summary = {
        "counts": {
            "failure": n_fail,
            "all": n_all,
            "useful": useful,
            "correct": correct,
            "wrong": wrong,
            "abstained": abstained,
            "invalid": invalid,
        },
        "checks": {k: {"value": val, "pass": ok} for k, (val, ok) in checks.items()},
        "decision": "GO" if all(ok for _, ok in checks.values()) else "NO-GO",
        "total_cost_usd": round(sum(costs), 4),
        "rows": rows,
    }
    out = DATA_DIR / "normalizer" / "run1_analysis.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(
        json.dumps(
            {k: summary[k] for k in ("counts", "decision", "total_cost_usd")}, ensure_ascii=False
        )
    )
    for k, d in summary["checks"].items():
        if k != "controls":
            print(f"  {k}: {d['value']} -> {'pass' if d['pass'] else 'FAIL'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("freeze")
    sub.add_parser("dry-run")
    rp = sub.add_parser("run")
    rp.add_argument("--allow-external", action="store_true")
    rp.add_argument("--budget-usd", type=float, required=True)
    sub.add_parser("score")
    sub.add_parser("verify-freeze")
    args = parser.parse_args(argv)
    if args.command == "freeze":
        return freeze()
    if args.command == "dry-run":
        return dry_run()
    if args.command == "run":
        return run(args.budget_usd, args.allow_external)
    if args.command == "verify-freeze":
        from mabhas19.core.paths import CURATED_DIR  # noqa: PLC0415
        from mabhas19.retrieval.structured import StructuredStore  # noqa: PLC0415

        problems = verify_frozen(StructuredStore(CURATED_DIR))
        print(problems or "verified")
        return 1 if problems else 0
    return score()


if __name__ == "__main__":
    raise SystemExit(main())
