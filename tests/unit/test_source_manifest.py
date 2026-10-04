"""Integrity of the committed official source PDFs against sources/manifest.yaml."""

from __future__ import annotations

import pymupdf
import pytest

from mabhas19.core.manifest import (
    SourceDocument,
    SourceVersion,
    load_manifest,
    sha256_file,
    verify_sources,
)
from mabhas19.core.paths import MANIFEST_PATH, SOURCES_DIR

MANIFEST = load_manifest(MANIFEST_PATH)
VERSIONS = MANIFEST.all_versions()


def test_corpus_is_main_document_plus_nine_appendices() -> None:
    assert MANIFEST.corpus_id == "mabhas19-ed5-1404"
    kinds = sorted((d.kind, d.appendix_number or 0) for d in MANIFEST.documents)
    assert kinds == [("appendix", n) for n in range(1, 10)] + [("main", 0)]


@pytest.mark.parametrize(
    ("doc", "version"), VERSIONS, ids=[f"{d.doc_id}@{v.version}" for d, v in VERSIONS]
)
def test_source_file_matches_manifest(doc: SourceDocument, version: SourceVersion) -> None:
    file = SOURCES_DIR / version.path
    assert file.is_file(), f"{doc.doc_id}: missing source file {version.path}"
    assert file.stat().st_size == version.size_bytes
    assert sha256_file(file) == version.sha256


@pytest.mark.parametrize(
    ("doc", "version"), VERSIONS, ids=[f"{d.doc_id}@{v.version}" for d, v in VERSIONS]
)
def test_page_count_matches_manifest(doc: SourceDocument, version: SourceVersion) -> None:
    with pymupdf.open(SOURCES_DIR / version.path) as pdf:
        assert pdf.page_count == version.page_count, doc.doc_id


def test_no_unlisted_or_mismatched_sources() -> None:
    assert verify_sources(MANIFEST, SOURCES_DIR) == []
