"""The semantic catalogue: the printed rows each regulatory quantity applies to.

Quantities are the production list :data:`mabhas19.api.quantities.QUANTITIES` (printed /
table-definition names; quantity is never inferred semantically). A row is a printed row label of
one of the quantity's tables; its search text adds only printed context (the parent part of a
main-table label, the Appendix 7 category and subgroup) and the reviewed concept words of
``api.wording`` whose printed target is that row. Nothing here is a value.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

from mabhas19.api import quantities as Q
from mabhas19.api import wording as W
from mabhas19.generation import options as O
from mabhas19.retrieval.schema import APP07

if TYPE_CHECKING:
    from mabhas19.retrieval.structured import StructuredStore

CATALOG_VERSION = "semantic-catalog-1"
_NUMBERING = re.compile(r"^[0-9۰-۹]+\s*[.\-)]\s*")


@dataclass(frozen=True)
class Row:
    id: str
    dataset: str
    field: str
    label: str  # the printed row label (shown to the user, and the words the engine is asked)
    quantities: tuple[str, ...]
    context: tuple[str, ...] = ()  # printed context only
    variants: tuple[str, ...] = ()  # reviewed concept words (``api.wording``)

    def passage(self) -> str:
        return " | ".join([self.label, *self.context, *self.variants])


@dataclass
class Catalogue:
    version: str
    rows: list[Row]
    quantities: list[str]
    sources: dict[str, str] = field(default_factory=dict)

    def for_quantity(self, qid: str) -> list[Row]:
        return [r for r in self.rows if qid in r.quantities]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def sha256(self) -> str:
        doc = json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(doc.encode("utf-8")).hexdigest()


def _context(store: StructuredStore, dataset: str, label: str) -> tuple[str, ...]:
    out: list[str] = []
    if dataset == APP07:
        for r in store.app07:
            if str(r.get("display_name") or "").strip() != label:
                continue
            for k in ("category", "subgroup"):
                v = r.get(k)
                text = v.get("text") if isinstance(v, dict) else v
                if not isinstance(text, str):
                    continue
                text = _NUMBERING.sub("", text).strip()  # «۵. چوب …» -> «چوب …»
                if text and text != label and text not in out:
                    out.append(text)
            break
    elif " – " in label:
        out.append(label.split(" – ", 1)[0])
    return tuple(out)


def _variants(label: str) -> tuple[str, ...]:
    shown = [r.shown for r in W.CONCEPT_ROWS if r.to and label.startswith(r.to)]
    shown += [f.shown for f in W.FAMILIES if label in f.rows]
    return tuple(dict.fromkeys(shown))


def build(store: StructuredStore) -> Catalogue:
    rows: dict[tuple[str, str], Row] = {}
    for q in Q.QUANTITIES:
        for dataset, fld in q.tables:
            for label in O.printed_options(store, dataset, fld):
                if q.row_filter and q.row_filter not in label:
                    continue
                if q.id == "u_value" and dataset == Q.T19_5_2 and "U-Value" not in label:
                    continue  # 19-5-2 prints U and SHGC rows; U is asked of the U rows only
                key = (dataset, label)
                prev = rows.get(key)
                rows[key] = Row(
                    id=f"{dataset.rsplit('.', 1)[-1]}:{fld}:{label}",
                    dataset=dataset,
                    field=fld,
                    label=label,
                    quantities=(*prev.quantities, q.id) if prev else (q.id,),
                    context=_context(store, dataset, label),
                    variants=_variants(label),
                )
    return Catalogue(
        version=CATALOG_VERSION,
        rows=list(rows.values()),
        quantities=[q.id for q in Q.QUANTITIES],
        sources={"wording": W.WORDING_VERSION},
    )
