"""Glyph clusters from the PDF content stream, plus glyph-level evidence.

Evidence found in the main PDF (B Nazanin, Microsoft Word export):

* the space glyph (gid 3) is mapped to U+200C, so every extracted "ZWNJ" with a
  positive advance width is really a space;
* kashida (tatweel) glyphs are mapped to U+0644 «ل», which turns «شـــدت» into «شللدت»;
  they are identified per (font, glyph id) by their rendered ink, not by spelling;
* a lam-alef ligature is one glyph that MuPDF splits into «ل» plus a zero-width «ا» with
  glyph id -1; keeping the two characters together as one cluster preserves «لا».
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import pymupdf

from mabhas19.text.charclass import ZWNJ

LAM = "ل"
KASHIDA_MAX_INK_RATIO = 0.3  # share of the glyph box height that carries ink
KASHIDA_SAMPLES = 6
INK_DARKNESS = 140  # grey level below which a pixel counts as ink
INK_DPI = 600


@dataclass(frozen=True)
class Cluster:
    """A glyph and the characters it produced, in content-stream order."""

    text: str
    gid: int
    font: str
    x0: float
    x1: float
    top: float
    bottom: float
    baseline: float
    size: float

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def key(self) -> tuple[str, int]:
        return (self.font, self.gid)


def _font_name(raw: str) -> str:
    return raw.split("+", 1)[-1]


def page_clusters(page: pymupdf.Page) -> list[Cluster]:
    """All visible glyph clusters of a page (invisible text render mode is skipped)."""
    clusters: list[Cluster] = []
    for span in page.get_texttrace():
        if span.get("type") == 3 or span.get("opacity", 1) == 0:
            continue
        font = _font_name(span["font"])
        size = float(span["size"])
        for unicode, gid, origin, bbox in span["chars"]:
            ch = chr(unicode)
            if gid == -1 and clusters and clusters[-1].font == font:
                prev = clusters[-1]
                clusters[-1] = Cluster(
                    prev.text + ch,
                    prev.gid,
                    font,
                    prev.x0,
                    prev.x1,
                    prev.top,
                    prev.bottom,
                    prev.baseline,
                    prev.size,
                )
                continue
            clusters.append(
                Cluster(
                    ch,
                    int(gid),
                    font,
                    float(bbox[0]),
                    float(bbox[2]),
                    float(bbox[1]),
                    float(bbox[3]),
                    float(origin[1]),
                    size,
                )
            )
    return clusters


def is_space_glyph(c: Cluster) -> bool:
    """A U+200C/space character that occupies horizontal space is a word space."""
    return c.text in (ZWNJ, " ") and c.width > 0.5


def ink_ratio(page: pymupdf.Page, c: Cluster) -> float:
    """Share of the glyph box rows that contain ink when rendered."""
    clip = pymupdf.Rect(c.x0 + 0.3, c.top, c.x1 - 0.3, c.bottom)
    if clip.is_empty:
        return 0.0
    pix = page.get_pixmap(dpi=INK_DPI, clip=clip, colorspace=pymupdf.csGRAY)
    rows = 0
    for r in range(pix.height):
        row = pix.samples[r * pix.stride : r * pix.stride + pix.width]
        if any(b < INK_DARKNESS for b in row):
            rows += 1
    return rows / pix.height if pix.height else 0.0


@dataclass(frozen=True)
class KashidaEvidence:
    font: str
    gid: int
    median_ink_ratio: float
    samples: int
    is_kashida: bool


def detect_kashida_glyphs(
    doc: pymupdf.Document, pages: list[int] | None = None
) -> list[KashidaEvidence]:
    """Classify every (font, gid) extracted as «ل» by how much vertical ink it draws.

    A real lam has a tall stem (ink over most of the glyph box); a kashida is a thin
    baseline stroke. ``pages`` are 0-based page numbers to sample from (default: all).
    """
    samples: dict[tuple[str, int], list[float]] = defaultdict(list)
    for pno in pages if pages is not None else range(doc.page_count):
        page = doc[pno]
        for c in page_clusters(page):
            if c.text == LAM and len(samples[c.key]) < KASHIDA_SAMPLES:
                samples[c.key].append(ink_ratio(page, c))
    evidence = []
    for (font, gid), ratios in sorted(samples.items()):
        median = statistics.median(ratios)
        evidence.append(
            KashidaEvidence(
                font, gid, round(median, 3), len(ratios), median < KASHIDA_MAX_INK_RATIO
            )
        )
    return evidence


def kashida_keys(evidence: list[KashidaEvidence]) -> frozenset[tuple[str, int]]:
    return frozenset((e.font, e.gid) for e in evidence if e.is_kashida)


def evidence_dict(evidence: list[KashidaEvidence]) -> list[dict[str, Any]]:
    return [vars(e) for e in evidence]
