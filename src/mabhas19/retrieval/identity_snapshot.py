"""Deterministic, reviewable snapshot of the derived identity of every structured row.

One entry per printed label (rows printing the same label - the requirement columns of a main
table, the density rows of one Appendix 7 material - share an entry and list their records):
the exact printed label hierarchy, the identity forms the index accepts (identity words per
level, per alternative), their heads, the printed condition a row requires, and every
parenthetical and list item of the printed text with whether it is used as a name.

The committed fixture (``tests/fixtures/identity_snapshot.json``) is compared with a fresh build
by a unit test, so any identity-rule change produces an explicit source-wide diff. Regenerate it
with ``python -m mabhas19.retrieval.identity_snapshot <path>`` after reviewing the diff.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mabhas19.retrieval.identity import (
    IdentityIndex,
    build_indexes,
    parenthetical_class,
    query_tokens,
)
from mabhas19.retrieval.schema import APP03, APP07, MAIN

if TYPE_CHECKING:
    from mabhas19.retrieval.structured import StructuredStore

_PAREN = re.compile(r"\(([^()]*)\)")
_APP07_FIELDS = ("category", "subgroup", "material", "condition")


def _printed_fields(store: StructuredStore, ds: str, record: dict[str, Any]) -> dict[str, str]:
    """The printed label fields of a record, as curated (not normalized)."""
    if ds == APP03:
        return {"occupancy": str(record.get("label") or "")}
    if ds == APP07:
        out = {}
        for k in _APP07_FIELDS:
            text = (record.get(k) or {}).get("text")
            if text:
                out[k] = str(text)
        return out
    from mabhas19.retrieval.structured import LABEL_FIELDS, _field  # noqa: PLC0415

    fields = LABEL_FIELDS.get(ds.removeprefix(MAIN), ())
    return {path: x for path, _ in fields if (x := _field(record, path))}


def _list_items(text: str) -> list[str]:
    from mabhas19.retrieval.identity import _list_items as items  # noqa: PLC0415

    return items(text)


def _structure(
    idx: IdentityIndex, row: str, fields: dict[str, str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    parens, items = [], []
    for name, text in fields.items():
        for m in _PAREN.finditer(text):
            inner = m[1].strip()
            if inner:
                parens.append(
                    {
                        "field": name,
                        "text": inner,
                        "class": parenthetical_class(inner),
                        "name_form": _alone(idx, row, inner, text),
                    }
                )
        pieces = _list_items(text)
        if len(pieces) > 1:
            for p in pieces:
                items.append({"field": name, "text": p, "name_form": _alone(idx, row, p, text)})
    return parens, items


def _alone(idx: IdentityIndex, row: str, part: str, whole: str) -> bool:
    """The part is accepted as a name on its own: an alternative derived from the printed
    structure has a level drawn from the part's words only (never the whole printed field)."""
    if part.strip() == whole.strip():
        return False
    words = set(query_tokens(part))
    return any(
        lv and set(lv) <= words
        for a in idx.by_row.get(row, [])
        if a.rule != "printed"
        for lv in a.levels
    )


def _entry(  # noqa: PLR0913
    idx: IdentityIndex,
    ds: str,
    records: list[str],
    row: str,
    fields: dict[str, str],
    *,
    levels: list[str] | None = None,
) -> dict[str, Any]:
    alts = idx.by_row.get(row, [])
    parens, items = _structure(idx, row, fields)
    entry: dict[str, Any] = {
        "dataset": ds.rsplit(".", maxsplit=1)[-1],
        "identity_row": row,
        "records": records,
        "printed": fields,
    }
    if levels is not None:
        entry["printed_levels"] = levels
    entry["forms"] = [
        {"rule": a.rule, "transforms": a.transforms, "levels": a.levels} for a in alts
    ]
    entry["heads"] = sorted({h for a in alts for h in a.heads})
    entry["condition_required"] = idx.conditions.get(row)
    entry["parentheticals"] = parens
    entry["list_items"] = items
    return entry


def build(store: StructuredStore) -> list[dict[str, Any]]:
    """The snapshot entries, in a fixed order."""
    indexes = build_indexes(store)
    out: list[dict[str, Any]] = []
    for tid, t in sorted(store.main.items()):
        if t.labels is None or t.label_rows is None or tid not in indexes:
            continue
        idx = indexes[tid]
        for lab in t.labels.labels:
            records = sorted(t.records[pos]["record_id"] for pos in t.label_rows[lab.id])
            if not records:
                continue
            first = t.records[t.label_rows[lab.id][0]]
            by_level: dict[int, list[str]] = {}
            for w, lv in zip(lab.tokens, lab.levels, strict=True):
                by_level.setdefault(lv, []).append(w)
            levels = [" ".join(by_level[k]) for k in sorted(by_level)]
            fields = _printed_fields(store, tid, first)
            out.append(_entry(idx, tid, records, records[0], fields, levels=levels))
    occ = indexes[APP03]
    counts: dict[str, int] = {}
    for r in store.app03:
        counts[r["occupancy_id"]] = counts.get(r["occupancy_id"], 0) + 1
    for o in sorted(store.occupancy_index, key=lambda x: str(x["id"])):
        oid = str(o["id"])
        e = _entry(
            occ, APP03, [f"{counts.get(oid, 0)} records"], oid, _printed_fields(store, APP03, o)
        )
        out.append(e)
    mat = indexes[APP07]
    groups: dict[tuple[tuple[str, str], ...], list[str]] = {}
    for r in store.app07:
        key = tuple(sorted(_printed_fields(store, APP07, r).items()))
        groups.setdefault(key, []).append(str(r["id"]))
    for key, rids in groups.items():
        out.append(_entry(mat, APP07, rids, rids[0], dict(key)))
    return out


def dumps(entries: list[dict[str, Any]]) -> str:
    return json.dumps(entries, ensure_ascii=False, indent=1) + "\n"


def main(argv: list[str]) -> int:
    from mabhas19.core.paths import CURATED_DIR  # noqa: PLC0415
    from mabhas19.retrieval.structured import StructuredStore  # noqa: PLC0415

    path = Path(argv[1]) if len(argv) > 1 else None
    text = dumps(build(StructuredStore(CURATED_DIR)))
    if path is None:
        sys.stdout.write(text)
    else:
        path.write_text(text, encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
