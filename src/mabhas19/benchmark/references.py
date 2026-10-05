"""Benchmark configuration and reference transcriptions (eval/extraction/reference/)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ITEM_REQUIRED = ("id", "doc_id", "pdf_page", "type", "expected_text")
ITEM_TYPES = frozenset({
    "prose", "heading", "clause", "term", "term_pair", "list_entry", "footnote", "caption",
    "date", "table_row", "table_text", "material_row",
})  # fmt: skip
REFERENCE_STATES = frozenset({"draft_pending_human_review", "human_verified"})
PAGE_CATEGORIES = frozenset({
    "prose", "definitions", "clauses", "table", "scanned", "terminology", "table_city_climate",
    "table_energy_limits", "table_text", "table_materials", "standards_fa", "standards_en",
})  # fmt: skip


@dataclass(frozen=True)
class BenchmarkPage:
    id: str
    doc_id: str
    pdf_page: int
    category: str
    reason: str


@dataclass(frozen=True)
class ReferenceItem:
    id: str
    doc_id: str
    pdf_page: int
    type: str
    expected_text: str
    bbox: tuple[float, float, float, float] | None = None
    expected_fields: dict[str, Any] = field(default_factory=dict)
    notes: str | None = None


@dataclass(frozen=True)
class PageReference:
    page_id: str
    doc_id: str
    pdf_page: int
    status: str
    items: tuple[ReferenceItem, ...]
    extra: dict[str, Any]


class ReferenceError(ValueError):
    pass


def load_benchmark_pages(path: Path) -> tuple[int, list[BenchmarkPage]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    pages = [BenchmarkPage(**p) for p in raw["pages"]]
    ids = [p.id for p in pages]
    if len(ids) != len(set(ids)):
        raise ReferenceError("duplicate benchmark page id")
    for p in pages:
        if p.category not in PAGE_CATEGORIES:
            raise ReferenceError(f"{p.id}: unknown category {p.category!r}")
    return int(raw.get("render_dpi", 300)), pages


def parse_reference(raw: dict[str, Any]) -> PageReference:
    for key in ("page_id", "doc_id", "pdf_page", "status", "items"):
        if key not in raw:
            raise ReferenceError(f"reference missing {key!r}")
    if raw["status"] not in REFERENCE_STATES:
        raise ReferenceError(f"{raw['page_id']}: unknown status {raw['status']!r}")
    items = []
    seen: set[str] = set()
    for i, it in enumerate(raw["items"]):
        where = f"{raw['page_id']}.items[{i}]"
        missing = [k for k in ITEM_REQUIRED if k not in it]
        if missing:
            raise ReferenceError(f"{where}: missing {missing}")
        if it["type"] not in ITEM_TYPES:
            raise ReferenceError(f"{where}: unknown type {it['type']!r}")
        if it["id"] in seen:
            raise ReferenceError(f"{where}: duplicate id {it['id']}")
        seen.add(it["id"])
        if (it["doc_id"], it["pdf_page"]) != (raw["doc_id"], raw["pdf_page"]):
            raise ReferenceError(f"{where}: doc_id/pdf_page differ from the page")
        bbox = it.get("bbox")
        if bbox is not None and len(bbox) != 4:
            raise ReferenceError(f"{where}: bbox must have 4 numbers")
        items.append(ReferenceItem(
            id=it["id"], doc_id=it["doc_id"], pdf_page=int(it["pdf_page"]), type=it["type"],
            expected_text=str(it["expected_text"]),
            bbox=tuple(float(v) for v in bbox) if bbox else None,  # type: ignore[arg-type]
            expected_fields=dict(it.get("expected_fields") or {}), notes=it.get("notes"),
        ))  # fmt: skip
    extra = {k: v for k, v in raw.items() if k not in {"items"}}
    return PageReference(raw["page_id"], raw["doc_id"], int(raw["pdf_page"]), raw["status"],
                         tuple(items), extra)  # fmt: skip


def load_references(directory: Path) -> dict[str, PageReference]:
    refs = {}
    for path in sorted(directory.glob("*.yaml")):
        ref = parse_reference(yaml.safe_load(path.read_text(encoding="utf-8")))
        if ref.page_id != path.stem:
            raise ReferenceError(f"{path.name}: page_id {ref.page_id!r} != file name")
        refs[ref.page_id] = ref
    return refs
