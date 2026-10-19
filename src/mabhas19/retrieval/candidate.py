"""Frozen retrieval candidate manifests for the final benchmarks (reproducibility guard).

``eval/retrieval/<set>_candidate.yaml`` (Final-2 and Final-3: historical, Final-4: current) pins
everything the benchmark tests: the candidate commit and the code (git tree of ``src/``,
per-module hashes), the curated corpus and tables, the structured schemas, the scope map, the
external-domain taxonomy and the policy-ambiguity record, the chunk fingerprint, BM25/RRF
parameters, the E5 model pins (from ``hybrid_candidate.yaml``, whose live weights the experiment
runner verifies on load), the router / query-plan / numeric-parser / entity-resolver /
constraint-parser / evaluator versions and the safety invariants. :func:`check_manifest`
recomputes all of it; any difference refuses the run. After the candidate commit only the frozen
test set itself may change. Nothing here changes retrieval behaviour.
"""

from __future__ import annotations

import hashlib
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from mabhas19.core.paths import CURATED_DIR, EVAL_DIR, REPO_ROOT
from mabhas19.retrieval.hybrid import RRF_K
from mabhas19.retrieval.lexical import K1, B

MANIFESTS = {
    s: EVAL_DIR / "retrieval" / f"{s}_candidate.yaml" for s in ("final2", "final3", "final4")
}
CURRENT = "final4"  # the only manifest that may be (re)written; earlier ones are historical
MANIFEST_PATH = MANIFESTS["final2"]  # historical: the candidate Final-2 was run against
HYBRID_CANDIDATE = EVAL_DIR / "retrieval" / "hybrid_candidate.yaml"
SCOPE_MAP = EVAL_DIR / "retrieval" / "scope_map.yaml"
EXTERNAL = EVAL_DIR / "retrieval" / "external_domains.yaml"
POLICY_AMBIGUITIES = EVAL_DIR / "retrieval" / "policy_ambiguities.yaml"
STRUCTURED_SCHEMAS = EVAL_DIR / "retrieval" / "structured_schemas.yaml"
# v4: plan-driven detection, decomposition by plan coverage with shared context, schema-based
# table references, planner-owned unknown entities, in-scope evidence over external domains
ROUTER_VERSION = "v4"
# typed StructuredQueryPlan + fail-closed validation gate + independent executor re-check
QUERY_PLAN_VERSION = "plan-1"
NUMERIC_PARSER_VERSION = "numeric-1"  # Decimal intervals, closed/open/unspecified bounds, ratios
# record-level, acceptable routes, typed no-match, scope classes, wrong-value / extra-row /
# dropped-constraint safety counts + optional typed-outcome, calculation and enumeration scoring
EVALUATOR_VERSION = "corrected-3"
CONSTRAINT_PARSER_VERSION = "ledger-1"  # one state per extracted constraint (constraints.py)
# meaning-bearing modifiers (negation, count, degree) kept on their word, fully named rows,
# adjectival keys, no ordinals as numbers
ENTITY_RESOLVER_VERSION = "modifiers-2"
SAFETY = {
    "wrong_regulatory_values_expected": 0,
    "wrong_extra_rows_expected": 0,
    "dropped_constraints_expected": 0,
}
# files that may change after the candidate commit (the frozen test set and its fingerprint)
SET_FILES = ("eval/retrieval/final4.yaml", "eval/retrieval/final4.sha256")
MODULES = (
    "src/mabhas19/indexing/chunking.py",
    "src/mabhas19/retrieval/normalization.py",
    "src/mabhas19/retrieval/lexical.py",
    "src/mabhas19/retrieval/vector.py",
    "src/mabhas19/retrieval/hybrid.py",
    "src/mabhas19/retrieval/router.py",
    "src/mabhas19/retrieval/scope.py",
    "src/mabhas19/retrieval/answerability.py",
    "src/mabhas19/retrieval/entities.py",
    "src/mabhas19/retrieval/dimensions.py",
    "src/mabhas19/retrieval/structured.py",
    "src/mabhas19/retrieval/constraints.py",
    "src/mabhas19/retrieval/numeric.py",
    "src/mabhas19/retrieval/schema.py",
    "src/mabhas19/retrieval/plan.py",
    "src/mabhas19/retrieval/planner.py",
    "src/mabhas19/retrieval/executor.py",
    "src/mabhas19/retrieval/models.py",
    "src/mabhas19/providers/local_embeddings.py",
    "src/mabhas19/retrieval/eval.py",
    "src/mabhas19/retrieval/candidate.py",
    "src/mabhas19/cli/eval_retrieval.py",
    "src/mabhas19/cli/embedding_experiment.py",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for p in sorted(x for x in root.rglob("*") if x.is_file()):
        digest.update(p.relative_to(root).as_posix().encode() + b"\0" + _sha(p).encode() + b"\n")
    return digest.hexdigest()


def live_candidate() -> dict[str, Any]:
    """The current candidate, computed from the working tree (no model is loaded)."""
    from mabhas19.cli.eval_retrieval import DOC_ID, _jsonl  # noqa: PLC0415
    from mabhas19.indexing.chunking import build_chunks  # noqa: PLC0415
    from mabhas19.retrieval.vector import DEPTH, chunks_fingerprint  # noqa: PLC0415

    chunks = build_chunks(_jsonl(CURATED_DIR / "text" / DOC_ID / "blocks.jsonl"), DOC_ID)
    hybrid = yaml.safe_load(HYBRID_CANDIDATE.read_text(encoding="utf-8"))
    emb = hybrid["embedding"]
    schemas = {
        p.parent.relative_to(CURATED_DIR).as_posix(): yaml.safe_load(
            p.read_text(encoding="utf-8")
        ).get("schema_version")
        for p in sorted((CURATED_DIR / "tables").rglob("table.yaml"))
    }
    from mabhas19.retrieval.plan import OUTCOMES  # noqa: PLC0415

    return {
        "code": {
            "src_tree": _git("rev-parse", "HEAD:src"),
            "src_clean": _git("status", "--porcelain", "--", "src") == "",
            "modules_sha256": {m: _sha(REPO_ROOT / m) for m in MODULES},
            "router_version": ROUTER_VERSION,
            "query_plan_version": QUERY_PLAN_VERSION,
            "plan_outcomes": list(OUTCOMES),
            "numeric_parser_version": NUMERIC_PARSER_VERSION,
            "constraint_parser_version": CONSTRAINT_PARSER_VERSION,
            "entity_resolver_version": ENTITY_RESOLVER_VERSION,
        },
        "curated": {
            "last_commit": _git("log", "-1", "--format=%h", "--", "curated"),
            "tree_sha256": _tree_digest(CURATED_DIR),
            "structured_schema_versions": schemas,
            "structured_schemas_sha256": _sha(STRUCTURED_SCHEMAS),
        },
        "scope": {
            "scope_map_sha256": _sha(SCOPE_MAP),
            "external_domains_sha256": _sha(EXTERNAL),
            "policy_ambiguities_sha256": _sha(POLICY_AMBIGUITIES),
        },
        "chunks": {"count": len(chunks), "fingerprint": chunks_fingerprint(chunks)},
        "bm25": {"k1": K1, "b": B},
        "fusion": {"method": "rrf", "rrf_k": RRF_K, "depth": DEPTH},
        "embedding": {
            "hybrid_candidate_sha256": _sha(HYBRID_CANDIDATE),
            "model_id": emb["model_id"],
            "revision": emb["revision"],
            "weights_file": emb["weights_file"],
            "weights_sha256": emb["weights_sha256"],
            "dimensions": emb["dimensions"],
            "max_seq_length": emb["max_seq_length"],
            "query_prefix": emb["query_prefix"],
            "passage_prefix": emb["passage_prefix"],
        },
        "evaluator": {
            "version": EVALUATOR_VERSION,
            "eval_py_sha256": _sha(REPO_ROOT / "src/mabhas19/retrieval/eval.py"),
        },
        "safety": dict(SAFETY),
    }


def write_manifest(path: Path = MANIFESTS[CURRENT], label: str = "Final-4") -> dict[str, Any]:
    live = live_candidate()
    if not live["code"]["src_clean"]:
        raise RuntimeError("src/ has uncommitted changes; commit before freezing the candidate")
    if _git("status", "--porcelain", "--", "eval", "curated"):
        raise RuntimeError("eval/ or curated/ has uncommitted changes; commit them first")
    doc = {
        "schema_version": 1,
        "purpose": f"frozen retrieval candidate tested by {label}; a run refuses on any difference",
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "base_commit": _git("rev-parse", "--short", "HEAD"),
        "candidate_commit": _git("rev-parse", "HEAD"),
        "allowed_changes_after": list(SET_FILES),
        "candidate": live,
    }
    path.write_text(
        f"# Generated by python -m mabhas19.cli.freeze_candidate --set {path.stem.split('_')[0]}"
        " --write (do not edit by hand).\n"
        + yaml.safe_dump(doc, allow_unicode=True, sort_keys=False, width=100),
        encoding="utf-8",
        newline="\n",
    )
    return doc


def _commit_drift(commit: str, allowed: list[str]) -> list[str]:
    """The candidate commit must be an ancestor of HEAD, and nothing but the frozen test set may
    have changed since (committed or not; ``data/`` is never versioned)."""
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"], cwd=REPO_ROOT, check=False
    )
    if ancestor.returncode != 0:
        return [f"candidate commit {commit[:10]} is not an ancestor of HEAD"]
    changed = set(_git("diff", "--name-only", commit, "HEAD").splitlines())
    status = _git("status", "--porcelain", "--", "src", "eval", "curated")
    changed |= {line[3:] for line in status.splitlines()}
    extra = sorted(x for x in changed if x and x not in allowed)
    return [f"changed since candidate commit {commit[:10]}: {extra}"] if extra else []


def check_manifest(path: Path = MANIFEST_PATH) -> list[str]:
    """Differences between the frozen manifest and the working tree (empty = frozen)."""
    if not path.is_file():
        return [f"{path.name} is missing"]
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    frozen = doc["candidate"]
    live = live_candidate()
    problems = []
    if not live["code"]["src_clean"]:
        problems.append("src/ has uncommitted changes")
    if doc.get("candidate_commit"):
        problems += _commit_drift(doc["candidate_commit"], doc.get("allowed_changes_after", []))
    for section, values in frozen.items():
        for key, value in values.items():
            if key == "src_clean":
                continue
            got = live.get(section, {}).get(key)
            if got != value:
                problems.append(f"{section}.{key}: frozen {value!r} != live {got!r}")
    return problems
