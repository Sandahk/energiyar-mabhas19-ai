"""End-to-end audit of a small synthetic PDF: output files and their record schema."""

from __future__ import annotations

import hashlib
import json
from dataclasses import fields
from pathlib import Path
from typing import Any

import pymupdf
import pytest

from mabhas19.core.manifest import SourceDocument, SourceVersion
from mabhas19.ingest.audit.models import (
    Confidence,
    HeadingCandidate,
    PageAudit,
    PageMapCandidate,
    TableInventoryEntry,
    TextQuality,
)
from mabhas19.ingest.audit.runner import (
    DOCUMENT_FILE,
    HEADINGS_FILE,
    PAGE_MAP_FILE,
    PAGES_FILE,
    SAMPLES_DIR,
    TABLES_FILE,
    AuditOptions,
    SamplePage,
    audit_version,
)

WIDTH, HEIGHT = 462, 663


def _make_pdf(path: Path) -> None:
    doc = pymupdf.open()
    for n in range(1, 4):
        page = doc.new_page(width=WIDTH, height=HEIGHT)
        if n == 1:
            continue  # blank cover page
        page.insert_text((60, 100), "Energy performance of buildings. " * 3, fontsize=10)
        page.insert_text((WIDTH / 2 - 3, 625), str(n - 1), fontsize=10)  # printed page
        if n == 3:  # a 3x3 ruled table with text in every cell
            for i in range(4):
                page.draw_line((60, 200 + 30 * i), (360, 200 + 30 * i))
                page.draw_line((60 + 100 * i, 200), (60 + 100 * i, 290))
            for r in range(3):
                for c in range(3):
                    page.insert_text((70 + 100 * c, 220 + 30 * r), f"v{r}{c}", fontsize=9)
    doc.save(path)


@pytest.fixture(scope="module")
def audit(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[str, Any]]:
    root = tmp_path_factory.mktemp("audit")
    sources, out = root / "sources", root / "interim"
    (sources / "set").mkdir(parents=True)
    pdf_path = sources / "set" / "synthetic-main-v1.pdf"
    _make_pdf(pdf_path)
    sha = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    version = SourceVersion(
        version="v1",
        path="set/synthetic-main-v1.pdf",
        canonical_filename="synthetic-main-v1.pdf",
        original_filename="synthetic.pdf",
        sha256=sha,
        size_bytes=pdf_path.stat().st_size,
        page_count=3,
        added_on="2026-10-04",
        supersedes=None,
        provenance="test",
    )
    doc = SourceDocument("synthetic-main", "main", None, "v1", (version,))
    options = AuditOptions(samples=(SamplePage("synthetic-main", 3, "table"),), dpi=40)
    summary = audit_version(doc, version, sources, out, options)
    return out / "synthetic-main" / sha[:8], summary


def _records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _field_names(cls: type) -> set[str]:
    return {f.name for f in fields(cls)}


def test_output_files_are_written(audit: tuple[Path, dict[str, Any]]) -> None:
    out_dir, _ = audit
    for name in (PAGES_FILE, PAGE_MAP_FILE, HEADINGS_FILE, TABLES_FILE, DOCUMENT_FILE):
        assert (out_dir / name).is_file(), name
    assert (out_dir / SAMPLES_DIR / "p003.png").is_file()


def test_page_records_schema(audit: tuple[Path, dict[str, Any]]) -> None:
    out_dir, _ = audit
    pages = _records(out_dir / PAGES_FILE)
    assert [p["pdf_page"] for p in pages] == [1, 2, 3]
    for p in pages:
        assert set(p) == _field_names(PageAudit)
        assert p["doc_id"] == "synthetic-main"
        assert p["likely_text_layer_quality"] in {q.value for q in TextQuality}
        for key in (
            "text_char_count",
            "persian_char_count",
            "latin_char_count",
            "digit_count",
            "image_count",
            "content_image_count",
            "vector_drawing_count",
        ):
            assert isinstance(p[key], int), key
            assert p[key] >= 0, key
        assert (p["page_width"], p["page_height"]) == (WIDTH, HEIGHT)
        assert isinstance(p["extracted_text_sample"], str)
    assert pages[0]["likely_text_layer_quality"] == TextQuality.NO_TEXT
    assert pages[1]["likely_text_layer_quality"] == TextQuality.LATIN_DIGITS_ONLY
    assert pages[1]["persian_char_count"] == 0


def test_page_map_candidates_schema(audit: tuple[Path, dict[str, Any]]) -> None:
    out_dir, _ = audit
    cands = _records(out_dir / PAGE_MAP_FILE)
    for c in cands:
        assert set(c) == _field_names(PageMapCandidate)
        assert c["verified"] is False
        assert c["confidence"] in {x.value for x in Confidence}
    assert [c["printed_page_candidate"] for c in cands] == [None, "1", "2"]
    assert cands[1]["confidence"] == Confidence.HIGH


def test_table_inventory_schema(audit: tuple[Path, dict[str, Any]]) -> None:
    out_dir, _ = audit
    tables = _records(out_dir / TABLES_FILE)
    for t in tables:
        assert set(t) == _field_names(TableInventoryEntry)
    assert [(t["pdf_page"], t["approximate_table_count"]) for t in tables] == [(3, 1)]
    assert tables[0]["detected_shapes"] == [[3, 3]]


def test_heading_file_is_valid_jsonl(audit: tuple[Path, dict[str, Any]]) -> None:
    out_dir, _ = audit
    for h in _records(out_dir / HEADINGS_FILE):
        assert set(h) == _field_names(HeadingCandidate)


def test_document_summary(audit: tuple[Path, dict[str, Any]]) -> None:
    out_dir, summary = audit
    assert json.loads((out_dir / DOCUMENT_FILE).read_text(encoding="utf-8")) == summary
    assert summary["page_count"] == 3
    assert summary["page_count_matches_manifest"] is True
    assert summary["pdf_bookmarks"] == []
    assert summary["has_pdf_page_labels"] is False
    assert sum(summary["quality_counts"].values()) == 3
    assert summary["table_pages_auto"] == [3]
