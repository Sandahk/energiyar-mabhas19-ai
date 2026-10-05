"""Repair on the real main PDF: three separate representations, evidence-based glyphs."""

from __future__ import annotations

import pymupdf
import pytest

from mabhas19.core.manifest import load_manifest
from mabhas19.core.paths import MANIFEST_PATH, SOURCES_DIR
from mabhas19.ingest.repair.glyphs import detect_kashida_glyphs, kashida_keys
from mabhas19.ingest.repair.pipeline import RepairedPage, repair_page

ZWNJ = "\u200c"


@pytest.fixture(scope="module")
def doc() -> pymupdf.Document:
    version = load_manifest(MANIFEST_PATH).get("m19-ed5-1404-main").current
    return pymupdf.open(SOURCES_DIR / version.path)


@pytest.fixture(scope="module")
def kashida(doc: pymupdf.Document) -> frozenset[tuple[str, int]]:
    return kashida_keys(detect_kashida_glyphs(doc, list(range(18, 30))))


@pytest.fixture(scope="module")
def page62(doc: pymupdf.Document, kashida: frozenset[tuple[str, int]]) -> RepairedPage:
    return repair_page(doc[61], kashida)


def test_kashida_glyph_identified_by_ink_not_spelling(doc: pymupdf.Document) -> None:
    evidence = {(e.font, e.gid): e for e in detect_kashida_glyphs(doc, list(range(18, 30)))}
    assert evidence[("BNazanin", 47)].is_kashida
    assert not evidence[("BNazanin", 184)].is_kashida  # a real lam


def test_raw_text_is_untouched(doc: pymupdf.Document, page62: RepairedPage) -> None:
    assert page62.raw_text == doc[61].get_text()
    assert "1-4-2-19" in page62.raw_text  # the visual-order artefact is preserved in raw


def test_display_text_is_repaired(page62: RepairedPage) -> None:
    display = page62.repaired_display_text
    assert "۱۹-۲-۴-۱ بازدهی انرژی ساختمان" in display
    assert "جمهوری اسلامی ایران" in display  # lam-alef ligature repaired
    assert "(kWh/m2.yr)" in display


def test_search_text_is_normalized_separately(page62: RepairedPage) -> None:
    search = page62.normalized_search_text
    assert "19-2-4-1" in search
    assert ZWNJ not in search
    assert search != page62.repaired_display_text
