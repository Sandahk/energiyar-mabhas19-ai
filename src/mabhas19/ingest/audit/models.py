"""Record types written by the source audit (all machine suggestions, never verified)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

AUDIT_SCHEMA_VERSION = 1


class TextQuality(StrEnum):
    GOOD = "GOOD"
    DAMAGED = "DAMAGED"
    LATIN_DIGITS_ONLY = "LATIN_DIGITS_ONLY"
    NO_TEXT = "NO_TEXT"


class Confidence(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"


class _Record:
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)  # type: ignore[call-overload,no-any-return]


@dataclass(frozen=True)
class PageAudit(_Record):
    doc_id: str
    pdf_page: int  # 1-based
    text_char_count: int
    persian_char_count: int
    latin_char_count: int
    digit_count: int
    image_count: int  # all raster images placed on the page
    content_image_count: int  # excluding images repeated on most pages (watermarks)
    vector_drawing_count: int
    page_width: float  # PDF points
    page_height: float
    extracted_text_sample: str
    likely_text_layer_quality: TextQuality
    quality_reasons: list[str]
    quality_signals: dict[str, float]
    has_full_page_raster: bool


@dataclass(frozen=True)
class PageMapCandidate(_Record):
    doc_id: str
    pdf_page: int
    printed_page_candidate: str | None  # numeric value as text, or a front-matter letter
    marker_kind: str  # "number" | "letter" | "absent"
    raw_marker: str | None  # marker exactly as found in the text layer
    confidence: Confidence
    detection_method: str
    verified: bool = False  # machine suggestion; verification happens in curated/
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class HeadingCandidate(_Record):
    doc_id: str
    pdf_page: int
    raw_heading: str  # line text in extraction order, unmodified
    raw_fragments: list[str]
    normalized_candidate_heading: str
    title_raw: str
    kind: str  # "heading" (bold) | "clause" (regular weight) | "toc_entry"
    hyphen_codepoints: list[str]
    toc_printed_page: str | None
    confidence: Confidence
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class TableInventoryEntry(_Record):
    doc_id: str
    pdf_page: int
    probable_table: bool
    approximate_table_count: int
    source: str  # "auto" | "manual_visual_review"
    notes: list[str] = field(default_factory=list)
    detected_shapes: list[list[int]] = field(default_factory=list)  # [rows, cols]
    caption_candidates: list[str] = field(default_factory=list)
