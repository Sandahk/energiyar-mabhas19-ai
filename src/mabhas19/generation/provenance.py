"""Provenance checks of a composed answer against the retrieval result it was built from.

Run on every answer before it is shown (and by the tests). Each check returns readable
violations; an empty list means the answer may be displayed.

* **numeric** - every number in a value item (value, unit, conditions) comes from the question
  or that item's own certified row; every other number shown comes from the question, a
  certified row, the engine's plans / typed outcomes / notes, or the source metadata (table
  captions, pages, section numbers, source-issue records), plus the corpus name («مبحث ۱۹»);
* **row** - every value item is a certified row of the result, and its value, unit and
  conditions are printed in that row;
* **citation** - every cited page, table and section exists in the result or source metadata;
* **refusal** - no answer shows a value unless the engine returned certified rows, and every
  certified row is accounted for;
* **interpretation** - every echoed value is rendered from strings the engine produced.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from decimal import Decimal, InvalidOperation
from typing import TYPE_CHECKING, Any

from mabhas19.generation.labels import CORPUS_NUMBERS, SYMBOLS, YEAR_ORDINALS
from mabhas19.text.normalize import normalize_for_search

if TYPE_CHECKING:
    from mabhas19.generation.contract import Answer
    from mabhas19.retrieval.models import RetrievalResult, StructuredHit
    from mabhas19.retrieval.structured import StructuredStore

_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")
_DECIMAL = re.compile(r"(?<=\d)[٫/.](?=\d)")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def numbers(text: str) -> set[str]:
    """Canonical numbers in a text (Persian/Arabic digits, «٫» or «/» decimals)."""
    t = _DECIMAL.sub(".", text.translate(_DIGITS))
    out = set()
    for m in _NUMBER.findall(t):
        try:
            out.add(format(Decimal(m).normalize(), "f"))
        except InvalidOperation:  # pragma: no cover - the pattern only matches numbers
            out.add(m)
    return out


def _strings(x: Any) -> list[str]:
    if isinstance(x, dict):
        return [s for k, v in x.items() for s in (str(k), *_strings(v))]
    if isinstance(x, list | tuple):
        return [s for v in x for s in _strings(v)]
    return [] if x is None else [str(x)]


def _hit_text(hit: StructuredHit) -> list[str]:
    return [*_strings(hit.values), str(hit.unit or ""), *hit.source_issues]


def _derivation_records(result: RetrievalResult, store: StructuredStore) -> list[str]:
    ids = {
        i for p in result.plans for d in p.get("derivations") or [] for i in d.get("inputs") or []
    }
    if not ids:
        return []
    rows = [
        *store.app02,
        *store.app03,
        *store.app07,
        *(r for t in store.main.values() for r in t.records),
    ]
    return [s for r in rows if str(r.get("id") or r.get("record_id")) in ids for s in _strings(r)]


def _metadata(result: RetrievalResult, store: StructuredStore) -> list[str]:
    out: list[str] = []
    for ds in {h.dataset for h in result.structured} | {n.dataset for n in result.no_match}:
        ev = store.table_evidence(ds, "provenance")
        out += [ds, ev.title, *map(str, ev.pdf_pages), *ev.printed_pages]
    for h in result.structured:
        out += [h.dataset, h.record_id, str(h.printed_page or ""), str(h.pdf_page or "")]
    for p in result.prose:
        out += [p.chunk_id, *map(str, p.pdf_pages), *(x or "" for x in p.printed_pages)]
        out += [f"{s.number or ''} {s.title}" for s in p.section_path]
    for plan in result.plans:  # printed labels of the occupancies a plan names by ID
        for e in plan.get("entities") or []:
            for name in str(e.get("text") or "").split(", "):
                out.append(str((store.occupancies.get(name.strip()) or {}).get("label") or ""))
    for sid, issue in store.issues.items():
        out += [sid, str(issue.get("title") or "")]
    return out


def _engine_text(result: RetrievalResult) -> list[str]:
    return [
        *_strings(result.plans),
        *_strings([asdict(n) for n in result.no_match]),
        *result.notes,
        *_strings([asdict(c) for c in result.conflicts]),
        *_strings(result.route.signals.get("scope") or {}),
        *_strings(result.certification_failures),
    ]


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", normalize_for_search(text.translate(_DIGITS))).strip()


def _printed_in(text: str, fields: list[str]) -> bool:
    """``text`` is one printed field, or a printed field followed by a printed unit field."""
    if any(_norm(text) in f for f in fields):
        return True
    head, _, tail = text.rpartition(" ")
    return (
        bool(head)
        and any(_norm(head) in f for f in fields)
        and any(_norm(tail) in f for f in fields)
    )


def check(  # noqa: PLR0912
    answer: Answer, result: RetrievalResult, store: StructuredStore
) -> list[str]:
    hits = {h.record_id: h for h in result.structured}
    problems: list[str] = []
    asked = numbers(result.query)

    # refusal preservation: values only from certified rows, every row accounted for
    if not result.structured and (answer.values or answer.response_type == "structured_value"):
        problems.append("refusal: a value is shown although the engine certified no row")
    shown = [v.record_id for v in answer.values]
    if len(shown) + answer.hidden_values != len(result.structured) and answer.values:
        problems.append("refusal: certified rows are not all accounted for")

    # row and numeric provenance of each value item
    for v in answer.values:
        hit = hits.get(v.record_id)
        if hit is None or hit.dataset != v.dataset:
            problems.append(f"row: {v.record_id} is not a certified row of this result")
            continue
        own = _hit_text(hit)
        own_norm = [_norm(s) for s in own]
        symbols = {label for key, label in SYMBOLS.items() if key in own}
        for text in (v.value, *v.row_label.split(" – ")):  # a label joins printed fields
            if text and text not in symbols and not _printed_in(text, own_norm):
                problems.append(f"row: «{text}» is not printed in {v.record_id}")
        if (v.unit or None) != (hit.unit or None):
            problems.append(f"row: unit {v.unit!r} is not the certified unit of {v.record_id}")
        allowed = asked | {n for s in own for n in numbers(s)}
        for c in v.conditions:
            ordinal = c.value in YEAR_ORDINALS.values()
            if not ordinal and c.value not in symbols and not _printed_in(c.value, own_norm):
                problems.append(f"row: condition «{c.value}» is not printed in {v.record_id}")
        item_text = [v.value, v.unit or "", v.row_label, *(c.value for c in v.conditions)]
        for n in sorted({n for t in item_text for n in numbers(t)} - allowed):
            problems.append(f"numeric: {n} in {v.record_id} is neither asked nor printed")

    # numeric provenance of everything else shown
    certified = [s for h in result.structured for s in _hit_text(h)]
    meta = _metadata(result, store)
    engine = _engine_text(result)
    allowed_all = asked | set(CORPUS_NUMBERS)
    prose = [p.text for p in result.prose]  # excerpts are checked below to be verbatim slices
    for s in [*certified, *meta, *engine, *prose, *_derivation_records(result, store)]:
        allowed_all |= numbers(s)
    for text in answer.texts():
        for n in sorted(numbers(text) - allowed_all):
            problems.append(f"numeric: {n} in «{text[:60]}» has no source")

    # excerpts are verbatim slices of retrieved chunks
    chunks = {p.chunk_id: _norm(p.text) for p in result.prose}
    for e in answer.excerpts:
        body = chunks.get(e.chunk_id)
        pieces = [x for x in (_norm(p) for p in e.text.split("…")) if x]
        if body is None or not all(p in body for p in pieces):
            problems.append(f"excerpt: {e.chunk_id} is not a verbatim slice of a retrieved chunk")

    # citations
    pages = set(meta)
    for src in answer.sources:
        for p in src.printed_pages:
            if p not in pages:
                problems.append(f"citation: page {p} of {src.label} is not in the result")
        for i in src.ids:
            if i not in hits and i not in chunks:
                problems.append(f"citation: {i} is not a record or chunk of this result")
        if src.dataset and src.dataset not in {h.dataset for h in result.structured}:
            problems.append(f"citation: {src.dataset} returned no certified row")

    # interpretation echo
    engine_dump = _norm(" ".join([*engine, *_strings([asdict(h) for h in result.structured])]))
    for echo in answer.interpretation:
        if not echo.source_text:
            problems.append(f"interpretation: {echo.label} has no engine source")
        for t in echo.source_text:
            if _norm(t) not in engine_dump:
                problems.append(f"interpretation: «{t}» is not in the engine result")
    return problems


def dump(answer: Answer) -> str:
    return json.dumps(answer.to_dict(), ensure_ascii=False, sort_keys=True)
