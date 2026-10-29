"""SH0 pilot: frozen manifest, local preflight, guarded provider run, replay-only report.

``python -m mabhas19.shadow.pilot <command>``:

* ``verify`` - the frozen manifest against the live engine (file fingerprint, engine
  fingerprint, prompt hash, canonical / provider schema hashes, every case's candidate-context
  hash). Any difference refuses every later command;
* ``preflight`` - local only: renders every request and checks it (no regulatory value outside
  the query, no source text, no credential, candidate IDs/names from the deterministic context,
  redaction applied) and measures its size;
* ``count-tokens`` - measured input tokens per request (the provider's token-counting endpoint:
  an external call under every guard);
* ``run`` - one proposal per case, recorded once in the replay store (never regenerated); needs
  ``--allow-external``, credentials, the verified frozen manifest and ``--max-usd`` /
  ``--max-tokens``; stops when invalid outputs exceed 10% after the first 20 calls or when the
  budget could be exceeded;
* ``report`` - replay only: re-verifies every record (determinism) and computes the S1-V /
  S1-R metrics against the frozen gold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from functools import cache
from pathlib import Path
from typing import Any

import yaml

from mabhas19.core.paths import CURATED_DIR, DATA_DIR, EVAL_DIR
from mabhas19.shadow.context import (
    CONTEXT_VERSION,
    CandidateContext,
    DeterministicFacts,
    build,
    canonical_json,
    redactions,
)
from mabhas19.shadow.metrics import Gold, aggregate, case_metrics
from mabhas19.shadow.proposal import PROPOSAL_SCHEMA, canonical_schema_sha256
from mabhas19.shadow.provider import (
    CREDENTIAL_VARS,
    AnthropicProposer,
    Budget,
    BudgetExceededError,
    ExternalCallRefusedError,
    input_token_bound,
)
from mabhas19.shadow.replay import (
    INSTRUCTIONS,
    ProposerSettings,
    ReplayKey,
    ReplayStore,
    engine_fingerprint,
    prompt_sha256,
    provider_schema,
    render_request,
)
from mabhas19.shadow.run import shadow
from mabhas19.shadow.schema_projection import ANTHROPIC_PROJECTION
from mabhas19.shadow.schema_projection import sha256 as projection_sha256

MANIFEST = EVAL_DIR / "shadow" / "pilot1.yaml"
STORE = DATA_DIR / "shadow" / "replay"
OUT = DATA_DIR / "shadow"
MANIFEST_SCHEMA = "sh0-pilot-manifest-1"
STRATA = {
    "homograph_possessive": 12,
    "background_claim_request": 12,
    "meaningful_modifier": 10,
    "semantic_colloquial_fail_closed": 14,
    "routing_intent": 12,
    "negative_control": 20,
}
CONTROL_GROUPS = (
    "true_identity",
    "verification_claim",
    "background_constraint",
    "two_results",
    "colloquial_entity",
)
INVALID_ABORT_AFTER, INVALID_ABORT_RATE = 20, 0.10


# ---------------------------------------------------------------- manifest


def pins() -> dict[str, str]:
    """Everything a frozen manifest pins about the live engine and the request contract."""
    return {
        "engine_fingerprint": engine_fingerprint(),
        "prompt_sha256": prompt_sha256(),
        "canonical_schema": PROPOSAL_SCHEMA,
        "canonical_schema_sha256": canonical_schema_sha256(),
        "provider_schema_sha256": projection_sha256(provider_schema()),
        "projection_version": ANTHROPIC_PROJECTION,
        "context_version": CONTEXT_VERSION,
    }


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha_path(manifest: Path) -> Path:
    return manifest.with_suffix(".sha256")


def freeze(doc: dict[str, Any], path: Path) -> str:
    """Write the manifest and its fingerprint file (refuses to overwrite a frozen one)."""
    if path.exists() or sha_path(path).exists():
        raise FileExistsError(f"{path.name} is already frozen")
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "# SH0 pilot manifest (frozen; do not edit).\n" + yaml.safe_dump(
        doc, allow_unicode=True, sort_keys=False, width=110
    )
    path.write_text(text, encoding="utf-8", newline="\n")
    digest = file_sha256(path)
    sha_path(path).write_text(f"{digest}  {path.name}\n", encoding="utf-8", newline="\n")
    return digest


def load(path: Path) -> dict[str, Any]:
    doc: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    return doc


def verify(path: Path, retriever: Any | None = None) -> list[str]:
    """Differences between the frozen manifest and the live engine (empty = frozen)."""
    if not path.is_file() or not sha_path(path).is_file():
        return [f"{path.name} or its fingerprint is missing"]
    pinned = sha_path(path).read_text(encoding="utf-8").split()[0]
    if file_sha256(path) != pinned:
        return [f"{path.name} changed since it was frozen"]
    doc = load(path)
    problems = []
    if doc.get("schema") != MANIFEST_SCHEMA:
        problems.append(f"manifest schema {doc.get('schema')!r}")
    for key, value in pins().items():
        if doc.get("pins", {}).get(key) != value:
            problems.append(
                f"pins.{key}: frozen {doc.get('pins', {}).get(key)!r} != live {value!r}"
            )
    counts: dict[str, int] = {}
    for c in doc.get("cases", []):
        counts[c["stratum"]] = counts.get(c["stratum"], 0) + 1
    if counts != STRATA:
        problems.append(f"strata {counts} != {STRATA}")
    if retriever is not None and not problems:
        for c in doc["cases"]:
            _, ctx, _ = _context(retriever, c["query"])
            if ctx.sha256() != c["context_sha256"]:
                problems.append(f"{c['id']}: candidate context changed")
    return problems


# ---------------------------------------------------------------- preflight


_ID_KEYS = frozenset({"id", "table", "row", "unit", "canonical", "version", "model"})
_TEXT_KEYS = frozenset({"query", "text"})


def request_problems(  # noqa: PLR0912
    body: Mapping[str, Any],
    ctx: CandidateContext,
    retriever: Any,
    env: Mapping[str, str] | None = None,
) -> list[str]:
    """Why a rendered request must not be sent (empty = it may)."""
    problems: list[str] = []
    serialized = canonical_json(body)

    # 1. digits only in query slices, identifiers and printed unit names
    def walk(x: Any, key: str = "") -> None:
        if isinstance(x, dict):
            for k, v in x.items():
                walk(v, k)
        elif isinstance(x, list | tuple):
            for v in x:
                walk(v, key)
        elif isinstance(x, str) and re.search(r"\d", x):
            if key in _TEXT_KEYS and x not in ctx.query:
                problems.append(f"{key}: {x[:40]!r} is not query text")
            elif key == "value" and not (
                x.removeprefix("elapsed:").lower() in ctx.query or x.isdigit()
            ):
                problems.append(f"value {x!r} is not from the query")
            elif key not in _TEXT_KEYS | _ID_KEYS | {"value"}:
                problems.append(f"{key}: {x[:40]!r} carries a number")

    walk(ctx.to_dict())
    if re.search(r"\d", INSTRUCTIONS.replace("Mabhas 19", "")):
        problems.append("the instructions carry a number")
    # 2. no source text
    for chunk in _source_lines():
        if chunk in serialized:
            problems.append(f"source text in the request: {chunk[:40]!r}")
            break
    # 3. no credential
    e = os.environ if env is None else env
    for var in (*CREDENTIAL_VARS, "GEMINI_API_KEY"):
        value = e.get(var)
        if value and value in serialized:
            problems.append(f"the value of {var} is in the request")
    if re.search(r"sk-ant-|AIza[0-9A-Za-z_-]{20}", serialized):
        problems.append("a credential-shaped string is in the request")
    # 4. candidate IDs / names from the deterministic context
    store = retriever.store
    for e_span in ctx.entities:
        for cand in e_span.candidates:
            idx = retriever.planner.identity.get(cand.table)
            if idx is None or cand.row not in idx.by_row:
                problems.append(f"candidate {cand.id} is not an identity-index row")
            elif cand.label != _label(retriever, cand.table, cand.row):
                problems.append(f"candidate {cand.id} label is not its printed label")
    for t in ctx.tables:
        if t.id not in store.schemas:
            problems.append(f"table {t.id} is not a curated table")
    # 5. redaction applied
    if redactions(ctx.query):
        problems.append("personal data left in the query")
    return problems


def _label(retriever: Any, table: str, row: str) -> str:
    from mabhas19.shadow.context import _label as label_of  # noqa: PLC0415

    return label_of(retriever, table, row)


@cache
def _source_lines() -> tuple[str, ...]:
    """Every curated source line long enough to be recognisable (40+ characters)."""
    from mabhas19.text.normalize import normalize_for_search  # noqa: PLC0415

    path = CURATED_DIR / "text" / "m19-ed5-1404-main" / "blocks.jsonl"
    out = set()
    for x in path.read_text(encoding="utf-8").splitlines():
        if x.strip():
            for line in str(json.loads(x).get("text") or "").splitlines():
                norm = normalize_for_search(line)
                if len(norm) >= 40:
                    out.add(norm)
    return tuple(sorted(out))


def preflight(doc: Mapping[str, Any], retriever: Any) -> dict[str, Any]:
    settings = ProposerSettings(**doc["settings"])
    rows = []
    for c in doc["cases"]:
        _, ctx, _ = _context(retriever, c["query"])
        body = render_request(ctx, settings)
        rows.append(
            {
                "id": c["id"],
                "problems": request_problems(body, ctx, retriever),
                "bytes": len(canonical_json(body).encode("utf-8")),
                "chars": len(canonical_json(body)),
                "user_chars": len(body["messages"][0]["content"]),
                "redactions": len(ctx.redactions),
                "input_token_bound": input_token_bound(body),
            }
        )
    return {
        "cases": len(rows),
        "problems": {r["id"]: r["problems"] for r in rows if r["problems"]},
        "bytes_total": sum(r["bytes"] for r in rows),
        "chars_total": sum(r["chars"] for r in rows),
        "max_bytes": max(r["bytes"] for r in rows),
        "input_token_bound_total": sum(r["input_token_bound"] for r in rows),
        "rows": rows,
    }


# ---------------------------------------------------------------- run / report


def _context(retriever: Any, query: str) -> tuple[Any, CandidateContext, DeterministicFacts]:
    prod = retriever.retrieve(query)
    ctx, facts = build(retriever, prod)
    return prod, ctx, facts


def gold_of(c: Mapping[str, Any]) -> Gold:
    g = c.get("gold") or {}
    return Gold(
        expected_rows=frozenset(c.get("expected", {}).get("rows") or []),
        acceptable_rows=frozenset(c.get("expected", {}).get("acceptable_rows") or []),
        expected_outcomes=frozenset(c.get("expected", {}).get("outcomes") or []),
        currently_correct=bool(c.get("currently_correct")),
        stratum=str(c["stratum"]),
        clauses={int(k): str(v) for k, v in (g.get("clauses") or {}).items()},
        entities={int(k): str(v) for k, v in (g.get("entities") or {}).items()},
        values={int(k): str(v) for k, v in (g.get("values") or {}).items()},
    )


def run_pilot(  # noqa: PLR0913
    manifest: Path,
    retriever: Any,
    *,
    allow_external: bool,
    budget: Budget | None,
    store: ReplayStore,
    env: Mapping[str, str] | None = None,
    client: Any = None,
) -> dict[str, Any]:
    """Record one proposal per case (existing records are kept, never regenerated)."""
    problems = verify(manifest, retriever)
    doc = load(manifest) if not problems else {}
    proposer = AnthropicProposer(
        ProposerSettings(**doc["settings"]) if doc else ProposerSettings(),
        allow_external=allow_external,
        manifest_frozen=not problems,
        budget=budget,
        env=env,
        client=client,
    )
    engine = doc["pins"]["engine_fingerprint"]
    calls = invalid = 0
    stopped = ""
    for c in doc["cases"]:
        prod, ctx, _ = _context(retriever, c["query"])
        key = ReplayKey.of(c["query"], ctx, proposer.settings, engine)
        if store.get(key) is not None:
            continue
        try:
            resp = proposer.propose(ctx)
        except BudgetExceededError as exc:
            stopped = f"budget: {exc}"
            break
        calls += 1
        sh = shadow(retriever, c["query"], resp.raw_text, production=prod)
        record = sh.record(key, resp.raw_text, _now(), resp.usage)
        record.response_model = resp.response_model
        record.experiment_valid, record.invalid_reason = resp.experiment_valid, resp.invalid_reason
        store.put(record)
        invalid += int(sh.proposal is None or not resp.experiment_valid)
        if calls >= INVALID_ABORT_AFTER and invalid / calls > INVALID_ABORT_RATE:
            stopped = f"invalid outputs {invalid}/{calls} exceed {INVALID_ABORT_RATE:.0%}"
            break
    return {
        "calls": calls,
        "invalid": invalid,
        "stopped": stopped,
        "input_tokens": proposer.meter.input_tokens,
        "output_tokens": proposer.meter.output_tokens,
        "usd": round(proposer.meter.spent_usd, 4),
    }


def report(manifest: Path, retriever: Any, store: ReplayStore) -> dict[str, Any]:
    """Replay only: every record re-verified (determinism) and scored against the gold."""
    doc = load(manifest)
    settings = ProposerSettings(**doc["settings"])
    rows, missing, nondeterministic = [], [], []
    for c in doc["cases"]:
        prod, ctx, _ = _context(retriever, c["query"])
        key = ReplayKey.of(c["query"], ctx, settings, doc["pins"]["engine_fingerprint"])
        rec = store.get(key)
        if rec is None:
            missing.append(c["id"])
            continue
        sh = shadow(retriever, c["query"], rec.raw_response, production=prod)
        again = sh.record(key, rec.raw_response, rec.timestamp, rec.usage)
        if (again.verifier, again.translated_plan, again.counterfactual) != (
            rec.verifier,
            rec.translated_plan,
            rec.counterfactual,
        ):
            nondeterministic.append(c["id"])
        if rec.experiment_valid:
            rows.append((case_metrics(sh, gold_of(c)), gold_of(c)))
    out = aggregate(rows)
    out["missing_records"] = missing
    total = len(doc["cases"]) - len(missing)
    out["replay_determinism"] = 1 - len(nondeterministic) / total if total else 0.0
    out["nondeterministic"] = nondeterministic
    return out


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


# ---------------------------------------------------------------- CLI


def _retriever() -> Any:
    from mabhas19.indexing.chunking import build_chunks  # noqa: PLC0415
    from mabhas19.retrieval.hybrid import Retriever  # noqa: PLC0415
    from mabhas19.retrieval.structured import StructuredStore  # noqa: PLC0415

    path = CURATED_DIR / "text" / "m19-ed5-1404-main" / "blocks.jsonl"
    blocks = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    return Retriever(build_chunks(blocks, "m19-ed5-1404-main"), StructuredStore(CURATED_DIR))


def main(argv: list[str] | None = None) -> int:  # noqa: PLR0911
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["verify", "preflight", "count-tokens", "run", "report"])
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--store", type=Path, default=STORE)
    parser.add_argument("--allow-external", action="store_true")
    parser.add_argument("--max-usd", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=0)
    args = parser.parse_args(argv)
    r = _retriever()
    problems = verify(args.manifest, r)
    if problems:
        print("manifest not frozen / verified:", *problems, sep="\n  ")
        return 2
    if args.command == "verify":
        print("manifest verified")
        return 0
    doc = load(args.manifest)
    OUT.mkdir(parents=True, exist_ok=True)
    stem = args.manifest.stem
    if args.command == "preflight":
        result = preflight(doc, r)
        (OUT / f"{stem}_preflight.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        print(json.dumps({k: v for k, v in result.items() if k != "rows"}, ensure_ascii=False))
        return 1 if result["problems"] else 0
    budget = Budget(args.max_usd, args.max_tokens) if args.max_usd or args.max_tokens else None
    try:
        if args.command == "count-tokens":
            proposer = AnthropicProposer(
                ProposerSettings(**doc["settings"]),
                allow_external=args.allow_external,
                manifest_frozen=True,
                budget=budget,
            )
            counts = {
                c["id"]: proposer.count_tokens(_context(r, c["query"])[1]) for c in doc["cases"]
            }
            (OUT / f"{stem}_token_counts.json").write_text(json.dumps(counts, indent=1))
            print(json.dumps({"cases": len(counts), "input_tokens_total": sum(counts.values())}))
            return 0
        if args.command == "run":
            summary = run_pilot(
                args.manifest,
                r,
                allow_external=args.allow_external,
                budget=budget,
                store=ReplayStore(args.store),
            )
            (OUT / f"{stem}_run.json").write_text(json.dumps(summary, indent=1))
            print(json.dumps(summary))
            return 0 if not summary["stopped"] else 3
    except ExternalCallRefusedError as exc:
        print(f"refused: {exc}")
        return 2
    result = report(args.manifest, r, ReplayStore(args.store))
    (OUT / f"{stem}_report.json").write_text(json.dumps(result, ensure_ascii=False, indent=1))
    print(json.dumps(result, ensure_ascii=False)[:2000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
