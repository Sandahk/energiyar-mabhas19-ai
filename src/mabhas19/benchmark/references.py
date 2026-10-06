"""Benchmark configuration and reference transcriptions (eval/extraction/reference/)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import yaml

ITEM_REQUIRED = ("id", "doc_id", "pdf_page", "type", "expected_text")
ITEM_KNOWN = frozenset({
    *ITEM_REQUIRED, "bbox", "expected_fields", "notes", "status", "verified_by", "verified_on",
})  # fmt: skip
ITEM_TYPES = frozenset({
    "prose", "heading", "clause", "term", "term_pair", "list_entry", "footnote", "caption",
    "date", "table_row", "table_text", "material_row",
})  # fmt: skip
DRAFT, VERIFIED = "draft_pending_human_review", "human_verified"
REFERENCE_STATES = frozenset({DRAFT, VERIFIED})
# expected_fields entries with a fixed type (others are dataset-specific and left as is).
STRING_FIELDS = ("subgroup",)
FOOTNOTE_FIELDS = ("footnote_refs", "category_footnote_refs", "subgroup_footnote_refs")
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
    status: str | None = None  # None: inherits the (draft) page status
    verified_by: str | None = None
    verified_on: date | None = None
    extra: dict[str, Any] = field(default_factory=dict)  # unknown keys, preserved verbatim


@dataclass(frozen=True)
class PageReference:
    page_id: str
    doc_id: str
    pdf_page: int
    status: str
    items: tuple[ReferenceItem, ...]
    extra: dict[str, Any]
    verified_by: str | None = None
    verified_on: date | None = None


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


def _iso_date(value: Any, where: str) -> date:
    """YAML gives an unquoted 2026-10-06 as a date; a quoted one must be ISO formatted."""
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value))
    except ValueError as exc:
        raise ReferenceError(f"{where}: verified_on {value!r} is not an ISO date") from exc


def _verification(raw: dict[str, Any], where: str) -> tuple[str | None, str | None, date | None]:
    """Validated (status, verified_by, verified_on) of a page or an item."""
    status = raw.get("status")
    if status is not None and status not in REFERENCE_STATES:
        raise ReferenceError(
            f"{where}: unknown status {status!r} (allowed: {sorted(REFERENCE_STATES)})"
        )
    by, on = raw.get("verified_by"), raw.get("verified_on")
    if status == VERIFIED:
        if not isinstance(by, str) or not by.strip():
            raise ReferenceError(f"{where}: human_verified requires verified_by")
        if on is None:
            raise ReferenceError(f"{where}: human_verified requires verified_on")
    elif by is not None or on is not None:
        raise ReferenceError(f"{where}: verified_by/verified_on given but status is {status!r}")
    return status, by, (_iso_date(on, where) if on is not None else None)


def _check_expected_fields(fields: dict[str, Any], where: str) -> None:
    for key in STRING_FIELDS:
        if key in fields and fields[key] is not None and not isinstance(fields[key], str):
            raise ReferenceError(f"{where}: {key} must be a string or null")
    for key in FOOTNOTE_FIELDS:
        if key in fields and not (
            isinstance(fields[key], list) and all(isinstance(v, str) for v in fields[key])
        ):
            raise ReferenceError(f"{where}: {key} must be a list of strings")


def _parse_item(it: dict[str, Any], raw: dict[str, Any], where: str) -> ReferenceItem:
    missing = [k for k in ITEM_REQUIRED if k not in it]
    if missing:
        raise ReferenceError(f"{where}: missing {missing}")
    if it["type"] not in ITEM_TYPES:
        raise ReferenceError(f"{where}: unknown type {it['type']!r}")
    if (it["doc_id"], it["pdf_page"]) != (raw["doc_id"], raw["pdf_page"]):
        raise ReferenceError(f"{where}: doc_id/pdf_page differ from the page")
    bbox = it.get("bbox")
    if bbox is not None and len(bbox) != 4:
        raise ReferenceError(f"{where}: bbox must have 4 numbers")
    fields = dict(it.get("expected_fields") or {})
    _check_expected_fields(fields, where)
    status, by, on = _verification(it, where)
    return ReferenceItem(
        id=it["id"], doc_id=it["doc_id"], pdf_page=int(it["pdf_page"]), type=it["type"],
        expected_text=str(it["expected_text"]),
        bbox=tuple(float(v) for v in bbox) if bbox else None,  # type: ignore[arg-type]
        expected_fields=fields, notes=it.get("notes"), status=status, verified_by=by,
        verified_on=on, extra={k: v for k, v in it.items() if k not in ITEM_KNOWN},
    )  # fmt: skip


def parse_reference(raw: dict[str, Any]) -> PageReference:
    for key in ("page_id", "doc_id", "pdf_page", "status", "items"):
        if key not in raw:
            raise ReferenceError(f"reference missing {key!r}")
    status, by, on = _verification(raw, str(raw["page_id"]))
    items = []
    seen: set[str] = set()
    for i, it in enumerate(raw["items"]):
        item = _parse_item(it, raw, f"{raw['page_id']}.items[{i}]")
        if item.id in seen:
            raise ReferenceError(f"{raw['page_id']}.items[{i}]: duplicate id {item.id}")
        seen.add(item.id)
        items.append(item)
    if status == VERIFIED:
        unverified = [it.id for it in items if it.status != VERIFIED]
        if unverified:
            raise ReferenceError(
                f"{raw['page_id']}: page is human_verified but items are not: {unverified}"
            )
    extra = {k: v for k, v in raw.items() if k not in {"items"}}
    return PageReference(raw["page_id"], raw["doc_id"], int(raw["pdf_page"]), str(status),
                         tuple(items), extra, by, on)  # fmt: skip


def load_references(directory: Path) -> dict[str, PageReference]:
    refs = {}
    for path in sorted(directory.glob("*.yaml")):
        ref = parse_reference(yaml.safe_load(path.read_text(encoding="utf-8")))
        if ref.page_id != path.stem:
            raise ReferenceError(f"{path.name}: page_id {ref.page_id!r} != file name")
        refs[ref.page_id] = ref
    return refs
