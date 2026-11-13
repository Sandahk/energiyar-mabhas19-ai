"""Printed option lists for clarification questions, read from the curated store only.

A clarification may offer, as clickable choices, only values the source prints for that field of
that table: the domain of a category dimension (climate classes, energy ratings, building groups,
years), the printed row labels of a main table, the printed Appendix 3 occupancies or the printed
Appendix 7 material names. Both the composer (to build options) and the provenance check (to
verify every displayed option independently) use this module.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from mabhas19.generation import labels as L
from mabhas19.retrieval.schema import APP03, APP07
from mabhas19.retrieval.structured import LABEL_FIELDS
from mabhas19.text.normalize import normalize_for_search

if TYPE_CHECKING:
    from mabhas19.retrieval.structured import StructuredStore

LABEL_KINDS = ("label", "element", "component", "equipment", "place_of_use", "use", "space")


def _unique(items: list[str]) -> list[str]:
    out: list[str] = []
    for item in items:
        s = item.strip()
        if s and s not in out:
            out.append(s)
    return out


def main_row_label(dataset: str, record: dict[str, Any]) -> str:
    """The printed label of a main-table record (its label fields joined by « – »)."""
    parts: list[str] = []
    for path, _ in LABEL_FIELDS.get(dataset.rsplit(".", 1)[-1][1:], ()):
        cur: Any = record
        for part in path.split("."):
            cur = cur.get(part) if isinstance(cur, dict) else None
        if isinstance(cur, str) and cur.strip() and cur.strip() not in parts:
            parts.append(cur.strip())
    return " – ".join(parts)


def label_dimension(store: StructuredStore, dataset: str) -> str | None:
    """The name of the row-label dimension of a main table (``element``, ``component``…)."""
    schema = store.schemas.get(dataset)
    if schema is None:
        return None
    return next((d.name for d in schema.dimensions if d.kind == "label" and d.required), None)


def printed_options(store: StructuredStore, dataset: str, field: str) -> list[str]:
    """Every printed value of ``field`` in ``dataset`` (empty when the source prints no finite
    list for it)."""
    schema = store.schemas.get(dataset)
    if schema is None:
        return []
    dim = schema.dim(field)
    if dim is not None and dim.kind == "category" and dim.domain:
        return [str(v) for v in dim.domain if str(v).strip() not in ("", "-")]
    if dataset == APP03 and field == "occupancy":
        return _unique([str(o.get("label") or "") for o in store.occupancies.values()])
    if dataset == APP07 and field == "material":
        return _unique([str(r.get("display_name") or "") for r in store.app07])
    if dataset in store.main and (field in LABEL_KINDS or field == label_dimension(store, dataset)):
        return _unique([main_row_label(dataset, r) for r in store.main[dataset].records])
    return []


def key(text: str) -> str:
    """Comparison words of a text: normalized, punctuation, parentheses and ZWNJ dropped."""
    norm = normalize_for_search(text.replace("‌", " "))
    return " ".join("".join(ch if ch.isalnum() else " " for ch in norm).split())


def label_key(text: str) -> str:
    """Comparison key of a whole label, insensitive to spacing and ZWNJ («نیمه‌باز» ==
    «نیمهباز» == «نیمه باز»)."""
    return key(text).replace(" ", "")


def printed_label(store: StructuredStore, dataset: str, field: str, engine_text: str) -> str:
    """The printed option an engine label text names (``""`` when none matches exactly)."""
    if dataset == APP03 and field == "occupancy":
        occ = store.occupancies.get(engine_text.strip())
        if occ:
            return str(occ.get("label") or "")
    want = label_key(engine_text)
    matches = [o for o in printed_options(store, dataset, field) if label_key(o) == want]
    return matches[0] if len(matches) == 1 else ""


def plain(label: str) -> str:
    """An output label without its parenthesized symbol (the words the engine reads)."""
    return re.sub(r"\s*\([^)]*\)\s*$", "", label).strip()


def allowed(store: StructuredStore, dataset: str, field: str) -> tuple[set[str], set[str]]:
    """(chip labels, appended texts) a clarification of ``field`` in ``dataset`` may show."""
    if field == "output":
        outputs = list(L.OUTPUT_LABELS.values())
        return {L.CLARIFY_SHOW.format(output=o) for o in outputs}, {plain(o) for o in outputs}
    printed = printed_options(store, dataset, field)
    pairs = [option_text(field, v) for v in printed]
    return set(printed) | {a for a, _ in pairs}, set(printed) | {b for _, b in pairs}


def option_text(field: str, value: str) -> tuple[str, str]:
    """(chip label, words appended to the question) for one printed category value."""
    if field == "year":
        ordinal = L.YEAR_ORDINALS.get(value, value)
        return ordinal, ordinal
    prefix = L.APPEND_PREFIX.get(field, "")
    return value, f"{prefix} {value}".strip()
