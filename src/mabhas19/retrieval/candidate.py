"""Frozen retrieval candidate manifests for the final benchmarks (reproducibility guard).

``eval/retrieval/<set>_candidate.yaml`` (Final-2 to Final-6: historical; Final-7: current)
pins
everything the benchmark tests: the candidate commit and the code (git tree of ``src/``,
per-module hashes), the curated corpus and tables, the structured schemas, the scope map, the
external-domain taxonomy and the policy-ambiguity record, the chunk fingerprint, BM25/RRF
parameters, the E5 model pins (from ``hybrid_candidate.yaml``, whose live weights the experiment
runner verifies on load), the router / query-plan / numeric-parser / entity-resolver /
constraint-parser / unit-model / polarity / coordination / evaluator versions, the table-scope
declarations with their provenance and the safety invariants. :func:`check_manifest`
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
    s: EVAL_DIR / "retrieval" / f"{s}_candidate.yaml"
    for s in ("final2", "final3", "final4", "final5", "final6", "final7")
}
CURRENT = "final7"  # the only manifest that may be (re)written; earlier ones are historical
LABELS = {
    "final2": "Final-2",
    "final3": "Final-3",
    "final4": "Final-4",
    "final5": "Final-5",
    "final6": "Final-6",
    "final7": "Final-7",
}
MANIFEST_PATH = MANIFESTS["final2"]  # historical: the candidate Final-2 was run against
HYBRID_CANDIDATE = EVAL_DIR / "retrieval" / "hybrid_candidate.yaml"
SCOPE_MAP = EVAL_DIR / "retrieval" / "scope_map.yaml"
EXTERNAL = EVAL_DIR / "retrieval" / "external_domains.yaml"
POLICY_AMBIGUITIES = EVAL_DIR / "retrieval" / "policy_ambiguities.yaml"
STRUCTURED_SCHEMAS = EVAL_DIR / "retrieval" / "structured_schemas.yaml"
IDENTITY_SNAPSHOT = REPO_ROOT / "tests" / "fixtures" / "identity_snapshot.json"
# v4.1: v4 + no caption match after a printed table ID, operation words generic, broken-plural
# caption/reference keys
ROUTER_VERSION = "v4.1"
# typed StructuredQueryPlan + validation gate + plan admission (grounded spans, anchors,
# coverage, unprinted modifiers block) + query-level plan coverage in five contracts (I1-I8,
# I10: one role per span, concept disjointness, complete requested quantity, unit expressions,
# identity-span accounting, value ownership with corrections, table ownership and scope, generic
# coordinated child plans) + independent result certification (exact output, unit, identity
# family, rejected values, scope: I9) (plan-4); stated-value completeness (I11), grammatical
# homographs never establish identity alone, calculation wording (difference / gap / explicit
# calculate / subtraction) (plan-5)
QUERY_PLAN_VERSION = "plan-5"
# Decimal intervals with typed bounds, ratios, coordinated complementary bounds, printed range
# exceptions negated
NUMERIC_PARSER_VERSION = "numeric-2"
# corrected-4 + wrong-row attribution from curated records only: foreign-table rows, refusal
# bypasses, dimension leaks, identity substitutions (corrected-5); refusal kinds refined by the
# case category, member loss, fail-closed-but-answerable, ledger counts per contract, per-stratum
# headlines (corrected-6); I11 in the span-roles ledger contract (corrected-7)
EVALUATOR_VERSION = "corrected-7"
# source-attested rating vocabulary, coordinated grades/groups/bounds, nearest-cue ownership,
# checked claims, constraint provenance (M2)
CONSTRAINT_PARSER_VERSION = "attested-1"
# source-exact identity (M1): parenthetical/list/container/linking structure, guarded
# morphology, required printed conditions, query-position pruning
ENTITY_RESOLVER_VERSION = "source-structure-1"
# compositional unit expressions: atoms with dimension vectors, attached / standalone
# expressions, ground / conversion / other-quantity verdicts (units.py, I4)
UNIT_MODEL_VERSION = "units-compositional-1"
# exclusions and printed negative forms + corrections («A نه، B», «نه A، B», «A نه بلکه B»,
# «A نیست، B است»): the rejected value is owned and never applied (polarity.py, I6)
POLARITY_VERSION = "polarity-corrections-1"
# generic coordinated child plans: owned clauses, a label word per member, member rows union
# equals the whole-query rows (coverage.split_coordinated, I8)
COORDINATION_VERSION = "generic-coordination-1"
SAFETY = {
    "wrong_regulatory_values_expected": 0,
    "wrong_extra_rows_expected": 0,
    "dropped_constraints_expected": 0,
    "failed_certifications_expected": 0,
    "foreign_table_rows_expected": 0,
    "dimension_leaks_expected": 0,
    "identity_substitutions_expected": 0,
    "refusal_bypassed_expected": 0,
}
# files that may change after the candidate commit: the manifest itself, the frozen test set and
# its fingerprint
SET_FILES = (
    "eval/retrieval/final7_candidate.yaml",
    "eval/retrieval/final7.yaml",
    "eval/retrieval/final7.sha256",
)
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
    "src/mabhas19/retrieval/admission.py",
    "src/mabhas19/retrieval/certification.py",
    "src/mabhas19/retrieval/coverage.py",
    "src/mabhas19/retrieval/units.py",
    "src/mabhas19/retrieval/table_scope.py",
    "src/mabhas19/retrieval/safety_metrics.py",
    "src/mabhas19/retrieval/identity.py",
    "src/mabhas19/retrieval/identity_snapshot.py",
    "src/mabhas19/retrieval/polarity.py",
    "src/mabhas19/retrieval/operations.py",
    "src/mabhas19/retrieval/ratings.py",
    "src/mabhas19/retrieval/final_validation.py",
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


def table_scope_pins() -> dict[str, dict[str, Any]]:
    """Per table and regulatory dimension: printed, universal, fixed (values and provenance) or
    undeclared (which fails closed and must not occur)."""
    from mabhas19.retrieval import table_scope  # noqa: PLC0415
    from mabhas19.retrieval.structured import StructuredStore  # noqa: PLC0415

    pins: dict[str, dict[str, Any]] = {}
    for ds, schema in sorted(StructuredStore(CURATED_DIR).schemas.items()):
        row: dict[str, Any] = {}
        for dim in table_scope.REGULATORY:
            decl = schema.scope_of(dim)
            if schema.dim(dim) is not None:
                row[dim] = "printed"
            elif decl is None:
                row[dim] = "undeclared"
            elif decl.kind == "universal":
                row[dim] = "universal"
            else:
                row[dim] = {"fixed": list(decl.values), "provenance": list(decl.provenance)}
        pins[ds] = row
    return pins


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
    from mabhas19.retrieval import final_validation  # noqa: PLC0415
    from mabhas19.retrieval.coverage import (  # noqa: PLC0415
        COVERAGE_VERSION,
        INVARIANTS,
        LEDGER_KEYS,
        LEDGER_SCHEMA,
    )
    from mabhas19.retrieval.plan import OUTCOMES  # noqa: PLC0415

    return {
        "code": {
            "src_tree": _git("rev-parse", "HEAD:src"),
            "src_clean": _git("status", "--porcelain", "--", "src") == "",
            "modules_sha256": {m: _sha(REPO_ROOT / m) for m in MODULES},
            "router_version": ROUTER_VERSION,
            "query_plan_version": QUERY_PLAN_VERSION,
            "coverage_version": COVERAGE_VERSION,
            "coverage_ledger_schema": LEDGER_SCHEMA,
            "coverage_ledger_keys": list(LEDGER_KEYS),
            "coverage_invariants": list(INVARIANTS),
            "plan_outcomes": list(OUTCOMES),
            "numeric_parser_version": NUMERIC_PARSER_VERSION,
            "constraint_parser_version": CONSTRAINT_PARSER_VERSION,
            "entity_resolver_version": ENTITY_RESOLVER_VERSION,
            "unit_model_version": UNIT_MODEL_VERSION,
            "polarity_version": POLARITY_VERSION,
            "coordination_version": COORDINATION_VERSION,
            "identity_snapshot_sha256": _sha(IDENTITY_SNAPSHOT),
        },
        "curated": {
            "last_commit": _git("log", "-1", "--format=%h", "--", "curated"),
            "tree_sha256": _tree_digest(CURATED_DIR),
            "structured_schema_versions": schemas,
            "structured_schemas_sha256": _sha(STRUCTURED_SCHEMAS),
            "table_scope": table_scope_pins(),
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
            "final_validation_version": final_validation.VERSION,
            "final_validated_sets": list(final_validation.VALIDATED_SETS),
        },
        "safety": dict(SAFETY),
    }


def write_manifest(path: Path = MANIFESTS[CURRENT], label: str = LABELS[CURRENT]) -> dict[str, Any]:
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
