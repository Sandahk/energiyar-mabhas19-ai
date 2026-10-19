"""Frozen retrieval candidate manifest for Final-2 (reproducibility guard).

``eval/retrieval/final2_candidate.yaml`` pins everything the Final-2 benchmark tests: the code
(git tree of ``src/``, per-module hashes), the curated corpus and tables, the scope map and the
external-domain taxonomy, the chunk fingerprint, BM25/RRF parameters, the E5 model pins (from
``hybrid_candidate.yaml``, whose live weights the experiment runner verifies on load), the
structured schema versions and the evaluator. :func:`check_manifest` recomputes all of it; any
difference refuses the run. Nothing here changes retrieval behaviour.
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

MANIFEST_PATH = EVAL_DIR / "retrieval" / "final2_candidate.yaml"
HYBRID_CANDIDATE = EVAL_DIR / "retrieval" / "hybrid_candidate.yaml"
SCOPE_MAP = EVAL_DIR / "retrieval" / "scope_map.yaml"
EXTERNAL = EVAL_DIR / "retrieval" / "external_domains.yaml"
ROUTER_VERSION = "v3.1"  # source-derived routing + structured retrieval v3, scope classes v3.1
EVALUATOR_VERSION = "corrected-1"  # record-level, acceptable routes, typed no-match, scope classes
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
    "src/mabhas19/retrieval/models.py",
    "src/mabhas19/providers/local_embeddings.py",
    "src/mabhas19/retrieval/eval.py",
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
    return {
        "code": {
            "src_tree": _git("rev-parse", "HEAD:src"),
            "src_clean": _git("status", "--porcelain", "--", "src") == "",
            "modules_sha256": {m: _sha(REPO_ROOT / m) for m in MODULES},
            "router_version": ROUTER_VERSION,
        },
        "curated": {
            "last_commit": _git("log", "-1", "--format=%h", "--", "curated"),
            "tree_sha256": _tree_digest(CURATED_DIR),
            "structured_schema_versions": schemas,
        },
        "scope": {
            "scope_map_sha256": _sha(SCOPE_MAP),
            "external_domains_sha256": _sha(EXTERNAL),
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
    }


def write_manifest(path: Path = MANIFEST_PATH) -> dict[str, Any]:
    live = live_candidate()
    if not live["code"]["src_clean"]:
        raise RuntimeError("src/ has uncommitted changes; commit before freezing the candidate")
    doc = {
        "schema_version": 1,
        "purpose": "frozen retrieval candidate tested by Final-2; a run refuses on any difference",
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "base_commit": _git("rev-parse", "--short", "HEAD"),
        "candidate": live,
    }
    path.write_text(
        "# Generated by python -m mabhas19.cli.freeze_candidate --write (do not edit by hand).\n"
        + yaml.safe_dump(doc, allow_unicode=True, sort_keys=False, width=100),
        encoding="utf-8",
        newline="\n",
    )
    return doc


def check_manifest(path: Path = MANIFEST_PATH) -> list[str]:
    """Differences between the frozen manifest and the working tree (empty = frozen)."""
    if not path.is_file():
        return [f"{path.name} is missing"]
    frozen = yaml.safe_load(path.read_text(encoding="utf-8"))["candidate"]
    live = live_candidate()
    problems = []
    if not live["code"]["src_clean"]:
        problems.append("src/ has uncommitted changes")
    for section, values in frozen.items():
        for key, value in values.items():
            if key == "src_clean":
                continue
            got = live.get(section, {}).get(key)
            if got != value:
                problems.append(f"{section}.{key}: frozen {value!r} != live {got!r}")
    return problems
