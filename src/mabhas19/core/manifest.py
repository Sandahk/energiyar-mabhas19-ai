"""Loading and integrity verification of the source manifest (sources/manifest.yaml).

The manifest is the single registry of official source PDFs. Each document has one or
more immutable versions; a better original is added as a new version rather than
replacing the bytes of an existing one.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

SUPPORTED_SCHEMA_VERSIONS = {1}
_HASH_CHUNK_SIZE = 1024 * 1024


class ManifestError(ValueError):
    """The manifest is malformed or internally inconsistent."""


@dataclass(frozen=True)
class SourceVersion:
    version: str
    path: str  # relative to the sources/ directory, POSIX separators
    canonical_filename: str
    original_filename: str
    sha256: str
    size_bytes: int
    page_count: int
    added_on: str
    supersedes: str | None
    provenance: str


@dataclass(frozen=True)
class SourceDocument:
    doc_id: str
    kind: str  # "main" | "appendix"
    appendix_number: int | None
    current_version: str
    versions: tuple[SourceVersion, ...]

    @property
    def current(self) -> SourceVersion:
        for v in self.versions:
            if v.version == self.current_version:
                return v
        raise ManifestError(f"{self.doc_id}: current_version {self.current_version!r} not found")


@dataclass(frozen=True)
class Manifest:
    schema_version: int
    corpus_id: str
    documents: tuple[SourceDocument, ...]

    def get(self, doc_id: str) -> SourceDocument:
        for d in self.documents:
            if d.doc_id == doc_id:
                return d
        raise KeyError(doc_id)

    def all_versions(self) -> list[tuple[SourceDocument, SourceVersion]]:
        return [(d, v) for d in self.documents for v in d.versions]


def _require(mapping: dict[str, Any], key: str, where: str) -> Any:
    if key not in mapping:
        raise ManifestError(f"{where}: missing required field {key!r}")
    return mapping[key]


def _parse_version(raw: dict[str, Any], where: str) -> SourceVersion:
    return SourceVersion(
        version=str(_require(raw, "version", where)),
        path=str(_require(raw, "path", where)),
        canonical_filename=str(_require(raw, "canonical_filename", where)),
        original_filename=str(_require(raw, "original_filename", where)),
        sha256=str(_require(raw, "sha256", where)).lower(),
        size_bytes=int(_require(raw, "size_bytes", where)),
        page_count=int(_require(raw, "page_count", where)),
        added_on=str(_require(raw, "added_on", where)),
        supersedes=raw.get("supersedes"),
        provenance=str(_require(raw, "provenance", where)),
    )


def parse_manifest(raw: dict[str, Any]) -> Manifest:
    """Build a Manifest from parsed YAML and check its internal consistency."""
    schema_version = int(_require(raw, "schema_version", "manifest"))
    if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ManifestError(f"unsupported schema_version {schema_version}")
    corpus = _require(raw, "corpus", "manifest")

    documents = []
    for i, d in enumerate(_require(raw, "documents", "manifest")):
        doc_id = str(_require(d, "doc_id", f"documents[{i}]"))
        versions = tuple(
            _parse_version(v, f"{doc_id}.versions[{j}]")
            for j, v in enumerate(_require(d, "versions", doc_id))
        )
        appendix_number = d.get("appendix_number")
        documents.append(
            SourceDocument(
                doc_id=doc_id,
                kind=str(_require(d, "kind", doc_id)),
                appendix_number=None if appendix_number is None else int(appendix_number),
                current_version=str(_require(d, "current_version", doc_id)),
                versions=versions,
            )
        )

    manifest = Manifest(
        schema_version=schema_version,
        corpus_id=str(_require(corpus, "corpus_id", "corpus")),
        documents=tuple(documents),
    )
    _check_consistency(manifest)
    return manifest


def _check_consistency(manifest: Manifest) -> None:
    doc_ids = [d.doc_id for d in manifest.documents]
    if len(doc_ids) != len(set(doc_ids)):
        raise ManifestError("duplicate doc_id")

    paths = [v.path for _, v in manifest.all_versions()]
    if len(paths) != len(set(paths)):
        raise ManifestError("two versions point at the same path")

    for doc in manifest.documents:
        if doc.kind not in {"main", "appendix"}:
            raise ManifestError(f"{doc.doc_id}: unknown kind {doc.kind!r}")
        if (doc.kind == "appendix") != (doc.appendix_number is not None):
            raise ManifestError(f"{doc.doc_id}: appendix_number must be set only for appendices")
        version_ids = [v.version for v in doc.versions]
        if len(version_ids) != len(set(version_ids)):
            raise ManifestError(f"{doc.doc_id}: duplicate version id")
        _ = doc.current  # raises if current_version is missing
        for v in doc.versions:
            where = f"{doc.doc_id}@{v.version}"
            if Path(v.path).name != v.canonical_filename:
                raise ManifestError(f"{where}: canonical_filename does not match path")
            if not v.canonical_filename.isascii():
                raise ManifestError(f"{where}: canonical_filename must be ASCII")
            if not v.original_filename:
                raise ManifestError(f"{where}: original_filename is empty")
            if len(v.sha256) != 64 or any(c not in "0123456789abcdef" for c in v.sha256):
                raise ManifestError(f"{where}: sha256 is not a 64-char hex digest")
            if v.supersedes is not None and v.supersedes not in version_ids:
                raise ManifestError(f"{where}: supersedes unknown version {v.supersedes!r}")


def load_manifest(path: Path) -> Manifest:
    with path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ManifestError(f"{path}: expected a mapping at top level")
    return parse_manifest(raw)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(_HASH_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sources(manifest: Manifest, sources_dir: Path) -> list[str]:
    """Return a list of integrity problems; an empty list means every source checks out.

    Checks that each listed file exists with the recorded size and SHA-256, and that no
    PDF under ``sources_dir`` is missing from the manifest.
    """
    problems: list[str] = []
    listed: set[Path] = set()
    for doc, v in manifest.all_versions():
        where = f"{doc.doc_id}@{v.version}"
        file = sources_dir / v.path
        listed.add(file.resolve())
        if not file.is_file():
            problems.append(f"{where}: file not found: {v.path}")
            continue
        size = file.stat().st_size
        if size != v.size_bytes:
            problems.append(f"{where}: size {size} != manifest {v.size_bytes}")
        digest = sha256_file(file)
        if digest != v.sha256:
            problems.append(f"{where}: sha256 {digest} != manifest {v.sha256}")

    for pdf in sorted(sources_dir.rglob("*")):
        if pdf.is_file() and pdf.suffix.lower() == ".pdf" and pdf.resolve() not in listed:
            problems.append(f"unlisted source file: {pdf.relative_to(sources_dir).as_posix()}")
    return problems
