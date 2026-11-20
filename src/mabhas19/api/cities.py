"""The certified city selector: the printed city names a climate question may be answered with.

The list is derived, never maintained: it is read from the same curated records the certified
city lookup (:func:`mabhas19.api.orchestration.city_lookup`) answers from - Appendix 2 and table
19-3-2 - with the same station rule (a printed station in parentheses, «تهران (مهرآباد)», is the
same printed city) and the same comparison key (:func:`mabhas19.generation.options.label_key`).
Two printed spellings with one key («خرم آباد» / «خرم‌آباد») are one entry; the first printed
spelling (Appendix 2 order, then table 19-3-2) is shown.

Search is filtering only: the query is normalized with the same key (Arabic ي/ك as Persian ی/ک,
آ as ا, spaces and ZWNJ ignored) and every entry whose key contains it is listed. There is no
fuzzy matching, no edit distance, no transliteration and no nearest city. Nothing is resolved
here: a selected entry still goes through the certified lookup, which keeps disagreeing sources
(Isfahan) as they are printed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from mabhas19.generation import options as O
from mabhas19.retrieval.schema import APP02, MAIN

if TYPE_CHECKING:
    from mabhas19.retrieval.structured import StructuredStore

T19_3_2 = f"{MAIN}19-3-2"
SOURCES = (APP02, T19_3_2)
STATION = re.compile(r"\s*\([^)]*\)\s*$")  # «تهران (مهرآباد)»: a printed station of the city
# Persian alphabetical order (Unicode order puts پ چ ژ گ after the Arabic letters)
_ORDER = {ch: i for i, ch in enumerate("ابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهی")}


@dataclass(frozen=True)
class CityEntry:
    """One selectable printed city: the name shown and sent, its search key, and the curated
    datasets that print it."""

    name: str
    key: str
    datasets: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "key": self.key}


def city_key(text: str) -> str:
    """The comparison and search key of a name (the certified lookup's own key)."""
    return O.label_key(text)


def printed_name(text: str) -> str:
    """A printed city cell without its station («تهران (مهرآباد)» -> «تهران»)."""
    return " ".join(STATION.sub("", text).split())


def _sort_key(entry: CityEntry) -> tuple[int, ...]:
    return tuple(_ORDER.get(ch, len(_ORDER) + ord(ch)) for ch in entry.key)


def catalogue(store: StructuredStore) -> tuple[CityEntry, ...]:
    """Every printed city of Appendix 2 and table 19-3-2, one entry per key, in Persian order."""
    cells = [(APP02, str((r.get("city") or {}).get("text") or "")) for r in store.app02]
    cells += [
        (T19_3_2, str((r.get("city") or {}).get("text") or "")) for r in store.main[T19_3_2].records
    ]
    names: dict[str, str] = {}
    datasets: dict[str, list[str]] = {}
    for dataset, cell in cells:
        name = printed_name(cell)
        key = city_key(name)
        if not key:
            continue
        names.setdefault(key, name)
        if dataset not in datasets.setdefault(key, []):
            datasets[key].append(dataset)
    entries = [CityEntry(names[k], k, tuple(datasets[k])) for k in names]
    return tuple(sorted(entries, key=_sort_key))


def search(entries: tuple[CityEntry, ...], query: str) -> list[CityEntry]:
    """The entries whose key contains the normalized query (all of them for an empty query)."""
    want = city_key(query)
    return [e for e in entries if want in e.key]


def selected(entries: tuple[CityEntry, ...], text: str) -> CityEntry | None:
    """The entry whose shown name is exactly ``text`` (a selection, never a typed guess)."""
    return next((e for e in entries if e.name == text), None)
