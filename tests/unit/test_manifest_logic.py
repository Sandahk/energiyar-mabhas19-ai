"""Behaviour of manifest parsing and verification on synthetic fixtures."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any

import pytest

from mabhas19.core.manifest import ManifestError, parse_manifest, verify_sources

PDF_BYTES = b"%PDF-1.4\n% synthetic fixture\n"


def _raw_manifest(sha256: str, size: int) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "corpus": {"corpus_id": "test-corpus"},
        "documents": [
            {
                "doc_id": "doc-a",
                "kind": "appendix",
                "appendix_number": 1,
                "current_version": "v1",
                "versions": [
                    {
                        "version": "v1",
                        "path": "set/doc-a-v1.pdf",
                        "canonical_filename": "doc-a-v1.pdf",
                        "original_filename": "پیوست1- نمونه.pdf",
                        "sha256": sha256,
                        "size_bytes": size,
                        "page_count": 1,
                        "added_on": "2026-10-04",
                        "supersedes": None,
                        "provenance": "test fixture",
                    }
                ],
            }
        ],
    }


@pytest.fixture
def sources(tmp_path: Path) -> Path:
    (tmp_path / "set").mkdir()
    (tmp_path / "set" / "doc-a-v1.pdf").write_bytes(PDF_BYTES)
    return tmp_path


@pytest.fixture
def raw() -> dict[str, Any]:
    return _raw_manifest(hashlib.sha256(PDF_BYTES).hexdigest(), len(PDF_BYTES))


def test_valid_sources_pass(sources: Path, raw: dict[str, Any]) -> None:
    manifest = parse_manifest(raw)
    assert verify_sources(manifest, sources) == []
    assert manifest.get("doc-a").current.original_filename == "پیوست1- نمونه.pdf"


def test_modified_bytes_are_detected(sources: Path, raw: dict[str, Any]) -> None:
    (sources / "set" / "doc-a-v1.pdf").write_bytes(PDF_BYTES.replace(b"synthetic", b"SYNTHETIC"))
    problems = verify_sources(parse_manifest(raw), sources)
    assert len(problems) == 1
    assert "sha256" in problems[0]


def test_missing_file_is_detected(sources: Path, raw: dict[str, Any]) -> None:
    (sources / "set" / "doc-a-v1.pdf").unlink()
    problems = verify_sources(parse_manifest(raw), sources)
    assert problems == ["doc-a@v1: file not found: set/doc-a-v1.pdf"]


def test_unlisted_pdf_is_detected(sources: Path, raw: dict[str, Any]) -> None:
    (sources / "set" / "stray.pdf").write_bytes(PDF_BYTES)
    assert verify_sources(parse_manifest(raw), sources) == ["unlisted source file: set/stray.pdf"]


def test_new_version_supersedes_previous(sources: Path, raw: dict[str, Any]) -> None:
    better = b"%PDF-1.7\n% better original\n"
    (sources / "set" / "doc-a-v2.pdf").write_bytes(better)
    doc = raw["documents"][0]
    v2 = copy.deepcopy(doc["versions"][0]) | {
        "version": "v2",
        "path": "set/doc-a-v2.pdf",
        "canonical_filename": "doc-a-v2.pdf",
        "sha256": hashlib.sha256(better).hexdigest(),
        "size_bytes": len(better),
        "supersedes": "v1",
    }
    doc["versions"].append(v2)
    doc["current_version"] = "v2"

    manifest = parse_manifest(raw)
    assert manifest.get("doc-a").current.version == "v2"
    assert verify_sources(manifest, sources) == []


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("canonical_filename", "other.pdf", "does not match path"),
        ("sha256", "abc", "64-char hex"),
        ("supersedes", "v0", "supersedes unknown version"),
        ("original_filename", "", "original_filename is empty"),
    ],
)
def test_inconsistent_version_is_rejected(
    raw: dict[str, Any], field: str, value: str, message: str
) -> None:
    raw["documents"][0]["versions"][0][field] = value
    with pytest.raises(ManifestError, match=message):
        parse_manifest(raw)


def test_non_ascii_canonical_filename_is_rejected(raw: dict[str, Any]) -> None:
    version = raw["documents"][0]["versions"][0]
    version["path"] = version["canonical_filename"] = "پیوست.pdf"
    with pytest.raises(ManifestError, match="ASCII"):
        parse_manifest(raw)


def test_missing_current_version_is_rejected(raw: dict[str, Any]) -> None:
    raw["documents"][0]["current_version"] = "v9"
    with pytest.raises(ManifestError, match="current_version"):
        parse_manifest(raw)
