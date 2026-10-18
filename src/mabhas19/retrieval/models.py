"""Retrieval data models: evidence objects with full provenance, routes and results.

Evidence is never paraphrased: a prose hit points to curated block IDs, a structured hit to a
curated record and its source cells. Unresolved source issues travel with the evidence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Route = Literal["structured", "prose", "mixed", "unsupported"]


@dataclass(frozen=True)
class SectionRef:
    number: str | None
    title: str


@dataclass(frozen=True)
class Chunk:
    """A prose retrieval unit built from consecutive curated blocks of one section."""

    chunk_id: str
    doc_id: str
    kind: str  # section | footnotes | caption
    block_ids: tuple[str, ...]
    block_kinds: tuple[str, ...]
    section_number: str | None
    section_path: tuple[SectionRef, ...]
    heading_text: str | None  # heading of the section (context; may be outside block_ids)
    text: str  # final corrected curated text of the blocks, in source order
    pdf_pages: tuple[int, ...]
    printed_pages: tuple[str | None, ...]
    block_pages: tuple[tuple[str, int], ...]  # (block_id, pdf_page): page boundaries
    words: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ProseHit:
    chunk_id: str
    score: float
    rank: int
    block_ids: tuple[str, ...]
    section_path: tuple[SectionRef, ...]
    pdf_pages: tuple[int, ...]
    printed_pages: tuple[str | None, ...]
    matched_terms: list[str]
    text: str


@dataclass
class StructuredHit:
    """One curated record, returned exactly as stored (raw and parsed values, unit)."""

    dataset: str  # curated table id, e.g. m19-ed5-1404-app02.city_climate_classes
    doc_id: str
    record_id: str
    pdf_page: int | None
    printed_page: str | None
    source_cells: list[str]
    values: dict[str, Any]  # field -> {raw/printed, value, unit, ...} as curated
    verification: dict[str, Any]
    source_issues: list[str]
    match: str  # why this record matched (auditable)
    score: float = 1.0


@dataclass
class TableEvidence:
    """Table-level provenance (caption, pages); never a value record."""

    dataset: str
    doc_id: str
    title: str  # printed caption (main tables) or curated table title (appendices)
    caption_blocks: list[str]
    pdf_pages: list[int]
    printed_pages: list[str]
    records: int
    why: str


@dataclass
class StructuredNoMatch:
    """A structured intent that yields no record, with the reason (never a silent drop).

    reason codes: ``climate_class_not_in_table``, ``no_printed_condition`` (19-3-1),
    ``no_applicable_category`` (19-3-3), ``no_matching_row``, ``unknown_place``,
    ``unknown_material``, ``unknown_occupancy``.
    """

    dataset: str
    reason: str
    detail: str
    constraints: dict[str, Any]
    table: TableEvidence | None
    source_issues: list[str] = field(default_factory=list)


@dataclass
class Conflict:
    """Two curated sources disagree; both hits are returned with the issue ID."""

    issue_id: str | None
    field: str
    hits: list[str]  # record IDs
    values: list[str]
    note: str


@dataclass
class RouteDecision:
    route: Route
    reasons: list[str]
    signals: dict[str, Any] = field(default_factory=dict)
    tables: list[str] = field(default_factory=list)  # candidate structured datasets
    structured_query: str | None = None  # decomposed parts (mixed queries), normalized
    prose_query: str | None = None


@dataclass
class RetrievalResult:
    query: str
    normalized_query: str
    route: RouteDecision
    prose: list[ProseHit] = field(default_factory=list)
    structured: list[StructuredHit] = field(default_factory=list)
    conflicts: list[Conflict] = field(default_factory=list)
    no_match: list[StructuredNoMatch] = field(default_factory=list)
    tables: list[TableEvidence] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)  # e.g. "no 5C column (SI-0002)"
