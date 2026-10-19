"""Deterministic lookup over the curated structured tables (never over prose chunks).

Datasets (read-only, from curated/):

* Appendix 2 city climate classes, Appendix 3 maximum EUI (+ occupancy types),
  Appendix 7 material thermal properties;
* the 11 structured main-document tables (19-2-1 … 19-5-7).

Every hit is a curated record returned as stored (raw/printed and parsed values, unit,
verification, source cells, source issues). Each structured intent ends in one of four
outcomes, never a silent drop:

* **match** - curated records (:class:`StructuredHit`);
* **known table, no applicable printed value** - :class:`StructuredNoMatch` with table
  provenance (a climate class absent from the table's columns, a degree-day value outside every
  printed condition of 19-3-1, an R outside every printed range of 19-3-3);
* **unknown entity** - :class:`StructuredNoMatch` (``unknown_material``, ``unknown_occupancy``;
  unknown places are detected by the router, which needs the corpus vocabulary);
* **unresolved source issue / conflict** - both records and a :class:`Conflict`
  (SI-0009 for Isfahan); 5C has no EUI column (SI-0002).

Nothing is computed beyond applying printed operators to a given number: no interpolation,
no inferred boundary, p64 symbols stay ``check_mark``/``cross_mark``, standard numbers stay as
displayed, and a density equal to a printed bound returns every row touching that bound.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import yaml

from mabhas19.retrieval.dimensions import (
    decimal_text,
)
from mabhas19.retrieval.entities import (
    EntityResolver,
    Label,
    LabelIndex,
    Resolution,
    etokens,
)
from mabhas19.retrieval.lexical import BM25
from mabhas19.retrieval.models import StructuredHit, StructuredNoMatch, TableEvidence
from mabhas19.retrieval.normalization import index_terms
from mabhas19.retrieval.schema import build_schemas
from mabhas19.text.normalize import normalize_for_search

APP02 = "m19-ed5-1404-app02.city_climate_classes"
APP03 = "m19-ed5-1404-app03.max_energy_use_intensity"
APP07 = "m19-ed5-1404-app07.material_thermal_properties"
MAIN = "m19-ed5-1404-main.t"
MAIN_3_2 = MAIN + "19-3-2"
T_DEGREE_DAYS = MAIN + "19-3-1"
T_R_RATING = MAIN + "19-3-3"
NO_5C_NOTE = (
    "Appendix 3 has no 5C column: no maximum EUI exists for 5C and none is "
    "derived from another class (SI-0002)."
)
_RECORD_META = frozenset(
    {"record_id", "doc_id", "table", "source_cells", "anchor_cell", "source_issue_refs"}
)
# words that describe the asked property, not the material
_PROPERTY_WORDS = frozenset(
    {
        "ضریب",
        "هدایت",
        "حرارتی",
        "چگالی",
        "جرم",
        "مخصوص",
        "مصالح",
        "کیلوگرم",
        "متر",
        "مکعب",
        "وات",
        "کلوین",
        "مقدار",
        "جدول",
        "پیوست",
        "kg",
        "m3",
        "w",
        "mk",
    }
)
# words that name a class of materials, never one curated material row
GENERIC_MATERIAL_WORDS = frozenset(
    {
        "عایق",
        "عایقهای",
        "عایق‌های",
        "ماده",
        "مواد",
        "مصالح",
        "ساختمانی",
        "جنس",
        "نوع",
        "انواع",
        "محصول",
        "برند",
        "مارک",
        "سنگ",
        "چوب",
        "بتن",
        "پلیمر",
        "فلز",
    }
)
_QUESTION_WORDS = frozenset({"چقدر", "چند", "چیست", "کدام", "است", "چه", "مقدار"})
# words that never name an occupancy
_OCCUPANCY_IGNORE = frozenset({"ساختمان", "ساختمانی", "متر", "مربع", "مترمربع", "مساحت"})
_EUI_WORDS = frozenset(
    {
        "شدت",
        "مصرف",
        "انرژی",
        "حداکثر",
        "مجاز",
        "اقلیم",
        "رده",
        "کاربری",
        "ساختمان",
        "حد",
        "kwh",
        "eui",
    }
)
_DEGREE_DAY_VARS = {"hdd": "HDD18°C", "cdd": "CDD10°C"}
# row-label fields per main table (path, level): the printed row/column labels a query names
LABEL_FIELDS: dict[str, tuple[tuple[str, int], ...]] = {
    "19-2-1": (("requirement.topic.text", 0), ("requirement.text", 1)),
    "19-3-1": (("climate_names.text", 0),),
    "19-3-2": (("province.text", 0), ("city.text", 1)),
    "19-3-3": (),
    "19-5-1": (("element.text", 0),),
    "19-5-2": (("section.text", 0), ("row.text", 1)),
    "19-5-3": (("equipment.text", 0), ("specification.text", 1)),
    "19-5-4": (("equipment.text", 0), ("specification.text", 1)),
    "19-5-5": (("place_of_use.text", 0),),
    "19-5-6": (("use.text", 0),),
    "19-5-7": (("space.text", 0),),
}
# words of «show me the table» requests; never a row constraint
REQUEST_WORDS = frozenset(
    {
        "جدول",
        "جداول",
        "محتوا",
        "محتوای",
        "مقدار",
        "مقادیر",
        "عدد",
        "اعداد",
        "ردیف",
        "ردیفهای",
        "ستون",
        "کامل",
        "همه",
        "تمام",
        "تمامی",
        "لیست",
        "فهرست",
        "بگو",
        "بده",
        "نشان",
        "نمایش",
        "شماره",
        "حداقل",
        "حداکثر",
        "مجاز",
        "میزان",
        "اقلیم",
        "اقلیمی",
        "رده",
        "گروه",
        "سال",
        "ساختمان",
        "ساختمانی",
        "ساختمانهای",
        "فضا",
        "فضای",
        "فضاهای",
        "دقیقا",
        "لازم",
        "باشد",
        "باشه",
        "میشه",
        "است",
        "وضعیت",
        "وضعیتی",
        "علامت",
        "علامتی",
        "الزام",
        "الزامات",
        "اجرایی",
        "زمان",
        "بندی",
    }
)


def _field(record: dict[str, Any], path: str) -> str | None:
    cur: Any = record
    for part in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur if isinstance(cur, str) else None


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _yaml(path: Path) -> dict[str, Any]:
    doc: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    return doc


def _strings(obj: Any, keys: frozenset[str]) -> Iterable[str]:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in keys and isinstance(v, str):
                yield v
            elif k == "applies_to" and isinstance(v, list):
                yield from (str(x) for x in v)
            else:
                yield from _strings(v, keys)
    elif isinstance(obj, list):
        for v in obj:
            yield from _strings(v, keys)


_LABEL_KEYS = frozenset({"text", "code", "displayed", "raw"})


@dataclass
class MainTable:
    table_id: str
    meta: dict[str, Any]
    records: list[dict[str, Any]]
    index: BM25
    climate_domain: frozenset[str]  # empty: the table has no climate dimension
    labels: LabelIndex | None = None
    label_rows: dict[str, list[int]] | None = None  # label id -> record positions
    caption_words: frozenset[str] = frozenset()


# ---------------------------------------------------------------------------- query parsing


def climate_codes(norm: str) -> list[str]:
    """Requested climate classes as written («3B», «6A»): validated later per table."""
    found = re.findall(r"(?<![a-z0-9])(\d)\s?([a-z])(?![a-z0-9/])", norm)
    return sorted({f"{a}{b.upper()}" for a, b in found})


def ratings(norm: str) -> list[str]:
    found = re.findall(r"(?:رده|رتبه|کلاس)\s*(?:انرژی\s*|بازدهی\s*)?([a-d])(?![a-z+])", norm)
    return sorted(set(x.upper() for x in found))


def r_value_in(norm: str) -> Decimal | None:
    """The R ratio stated in the query («نسبت R ۵», «R برابر ۲٫۵»), if any."""
    norm = decimal_text(norm)
    m = re.search(r"(?<![a-z])r(?![a-z\-\d])\s*(?:\S+\s*){0,3}?(\d+(?:\.\d+)?)(?![\d\-a-z])", norm)
    return _decimal(m[1]) if m else None


def degree_days_in(norm: str) -> tuple[str, Decimal] | None:
    """(variable, value) for «روز درجه گرمایش ۵۰۰۰», «HDD 5000», «روز درجه سرمایش ...»."""
    norm = decimal_text(norm)
    patterns = (("hdd", r"(?:hdd\w*|روز درجه گرمایش)"), ("cdd", r"(?:cdd\w*|روز درجه سرمایش)"))
    for var, pat in patterns:
        m = re.search(rf"{pat}\s*(?:\S+\s*){{0,3}}?(\d+(?:\.\d+)?)(?![\d\-a-z])", norm)
        if m:
            value = _decimal(m[1])
            if value is not None:
                return _DEGREE_DAY_VARS[var], value
    return None


def _decimal(text: str | None) -> Decimal | None:
    try:
        return Decimal(text) if text is not None else None
    except InvalidOperation:
        return None


# ---------------------------------------------------------------------------- store


class StructuredStore:
    def __init__(self, curated: Path) -> None:
        t = curated / "tables"
        self.app02 = _jsonl(t / "m19-ed5-1404-app02" / "city_climate_classes.jsonl")
        self.app03 = _jsonl(t / "m19-ed5-1404-app03" / "max_eui.jsonl")
        self.occupancies = {
            o["id"]: o for o in _jsonl(t / "m19-ed5-1404-app03" / "occupancy_types.jsonl")
        }
        self.app07 = _jsonl(t / "m19-ed5-1404-app07" / "material_thermal_properties.jsonl")
        self.metas = {
            APP02: _yaml(t / "m19-ed5-1404-app02" / "table.yaml"),
            APP03: _yaml(t / "m19-ed5-1404-app03" / "table.yaml"),
            APP07: _yaml(t / "m19-ed5-1404-app07" / "table.yaml"),
        }
        self.main: dict[str, MainTable] = {}
        for d in sorted((t / "m19-ed5-1404-main").glob("table-*")):
            meta = _yaml(d / "table.yaml")
            recs = _jsonl(d / "records.jsonl")
            labels = [index_terms(" ".join(_strings(r, _LABEL_KEYS))) for r in recs]
            domain = frozenset(c for r in recs for g in _groups(r) for c in g)
            table = MainTable(meta["table_id"], meta, recs, BM25(labels), domain)
            _index_labels(table)
            self.main[meta["table_id"]] = table
        self.app03_domain = frozenset(r["climate_class"] for r in self.app03)
        self.material_terms = [set(index_terms(_material_name(r))) for r in self.app07]
        self.occupancy_index = list(self.occupancies.values())
        self.places = self._places()
        self.resolver = EntityResolver(
            self.app07,
            self.occupancy_index,
            self.places,
            material_ignore=_PROPERTY_WORDS | _QUESTION_WORDS,
            generic_material_words=GENERIC_MATERIAL_WORDS,
            occupancy_ignore=_PROPERTY_WORDS | _EUI_WORDS | _OCCUPANCY_IGNORE,
        )
        self._app07_pos = {r["id"]: i for i, r in enumerate(self.app07)}
        self.issues = {i["id"]: i for i in _yaml(curated / "source_issues.yaml").get("issues", [])}
        self.issue_links = self._quoted_issue_links()
        # main-text blocks linked to source issues (curated provenance, e.g. SI-0004)
        self.block_issues: dict[str, list[str]] = {}
        blocks_path = curated / "text" / "m19-ed5-1404-main" / "blocks.jsonl"
        if blocks_path.is_file():
            for b in _jsonl(blocks_path):
                if b.get("source_issue_refs"):
                    self.block_issues[b["block_id"]] = list(b["source_issue_refs"])
        self.schemas = build_schemas(
            {
                APP02: (str(self.metas[APP02].get("title", APP02)), self.app02),
                APP03: (str(self.metas[APP03].get("title", APP03)), self.app03),
                APP07: (str(self.metas[APP07].get("title", APP07)), self.app07),
                **{
                    tid: (str(t.meta["caption"]["text"]), t.records) for tid, t in self.main.items()
                },
            }
        )
        missing = self.metas[APP03].get("missing_climate_classes") or {}
        # issues the curated Appendix 3 metadata attaches to each missing climate class
        self.app03_missing_issues = {
            str(c): list(v.get("source_issue_refs") or []) for c, v in missing.items()
        }

    # ------------------------------------------------------------------ provenance
    def table_evidence(self, ds: str, why: str) -> TableEvidence:
        if ds in self.main:
            meta = self.main[ds].meta
            cap = meta["caption"]
            return TableEvidence(
                ds,
                meta["doc_id"],
                cap["text"],
                list(cap["blocks"]),
                [meta["pdf_page"]],
                [str(meta["printed_page"])],
                len(self.main[ds].records),
                why,
            )
        meta = self.metas[ds]
        src = meta.get("source", {})
        records = {APP02: self.app02, APP03: self.app03, APP07: self.app07}[ds]
        return TableEvidence(
            ds,
            meta["doc_id"],
            str(meta.get("title", ds)),
            [],
            list(src.get("pdf_pages", [])),
            [str(p) for p in src.get("printed_pages", [])],
            len(records),
            why,
        )

    def _no_match(
        self,
        ds: str,
        reason: str,
        detail: str,
        constraints: dict[str, Any],
        issues: list[str] | None = None,
    ) -> StructuredNoMatch:
        return StructuredNoMatch(
            ds,
            reason,
            detail,
            constraints,
            self.table_evidence(ds, f"no match: {reason}"),
            issues or [],
        )

    # ------------------------------------------------------------------ provenance
    def _quoted_issue_links(self) -> dict[str, list[str]]:
        """Record -> issues whose registry title quotes text printed in that record, within the
        issue's own documents and pages (e.g. a misprinted group title -> the rows printed under
        it). Derived from curated/source_issues.yaml; nothing is attached globally."""
        records: list[tuple[str, dict[str, Any], int | None]] = []
        for ds, rows in ((APP02, self.app02), (APP03, self.app03), (APP07, self.app07)):
            records += [(ds, r, (r.get("source") or {}).get("pdf_page")) for r in rows]
        for tid, t in self.main.items():
            records += [(tid, r, t.meta.get("pdf_page")) for r in t.records]
        links: dict[str, list[str]] = {}
        for issue in self.issues.values():
            quotes = [
                normalize_for_search(q)
                for q in re.findall(r"«([^»]+)»", str(issue.get("title", "")))
                if len(q) >= 4
            ]
            if not quotes:
                continue
            locations = issue.get("locations") or []
            docs = {loc.get("doc_id") for loc in locations}
            pages = {p for loc in locations for p in loc.get("pdf_pages", [])}
            for _, r, page in records:
                if r.get("doc_id") not in docs or (pages and page not in pages):
                    continue
                printed = normalize_for_search(
                    json.dumps(
                        {k: v for k, v in r.items() if k not in _PROVENANCE_KEYS},
                        ensure_ascii=False,
                    )
                )
                if any(q in printed for q in quotes):
                    rid = str(r.get("id") or r.get("record_id"))
                    links.setdefault(rid, []).append(issue["id"])
        return links

    def _issues_of(self, record_id: str, *refs: Iterable[str]) -> list[str]:
        out: list[str] = []
        for group in (*refs, self.issue_links.get(record_id, [])):
            out += [i for i in group if i not in out]
        return out

    # ------------------------------------------------------------------ places
    def _places(self) -> dict[str, list[tuple[str, dict[str, Any]]]]:
        out: dict[str, list[tuple[str, dict[str, Any]]]] = {}

        def add(name: str, ds: str, rec: dict[str, Any]) -> None:
            for alias in _aliases(name):
                out.setdefault(alias, []).append((ds, rec))

        for r in self.app02:
            add(r["city"]["text"], APP02, r)
        for r in self.main[MAIN_3_2].records:
            add(r["city"]["text"], MAIN_3_2, r)
        return out

    def find_places(self, norm: str) -> list[str]:
        """Curated place aliases in the query (longest names; spacing/joined forms match)."""
        return self.resolver.places(norm)

    def ambiguous_places(self, places: list[str]) -> dict[str, list[str]]:
        """Places that resolve to more than one Appendix 2 station (e.g. two Tehran stations)."""
        out = {}
        for p in places:
            stations = sorted({r["city"]["text"] for ds, r in self.places[p] if ds == APP02})
            if len(stations) > 1:
                out[p] = stations
        return out

    def occupancies_in(self, norm: str) -> list[str]:
        """Curated occupancy ids named in the text (entity resolution only)."""
        return list(self.resolver.occupancies(norm).ids)

    # ------------------------------------------------------------------ appendix 7
    def material_terms_of(self, norm: str) -> list[str]:
        """Query words that could name a material (property, generic and function words
        removed)."""
        return self.resolver.material_anchor_words(norm)

    def material_resolution(self, norm: str) -> Resolution:
        return self.resolver.materials(norm)

    def resolve_material(self, norm: str) -> list[int] | None:
        """Rows of the most specific curated material label found in the query; ``None`` when
        the query names no specific material (only generic words such as «عایق»), ``[]`` when it
        names an unknown or conflicting one."""
        res = self.resolver.materials(norm)
        if res.status == "none":
            return None
        return [self._app07_pos[i] for i in res.ids] if res.status == "resolved" else []

    # ------------------------------------------------------------------ hits
    def _hit(self, ds: str, r: dict[str, Any], why: str) -> StructuredHit:
        if ds == APP02:
            return StructuredHit(
                ds,
                r["doc_id"],
                r["id"],
                r["source"]["pdf_page"],
                r["source"]["printed_page"],
                [],
                {
                    "city": r["city"],
                    "climate_class": r["climate_class"],
                    "row_number": r["row_number"],
                },
                r["verification"],
                self._issues_of(r["id"], r["source_issue_refs"]),
                why,
            )
        if ds == APP03:
            occ = self.occupancies[r["occupancy_id"]]
            return StructuredHit(
                ds,
                r["doc_id"],
                r["id"],
                r["source"]["pdf_page"],
                r["source"]["printed_page"],
                [],
                {
                    "occupancy": {"id": occ["id"], "label": occ["label"]},
                    "climate_class": r["climate_class"],
                    "energy_rating": r["energy_rating"],
                    "max_eui": r["max_eui"],
                },
                r["verification"],
                self._issues_of(r["id"], occ.get("source_issue_refs", [])),
                why,
            )
        if ds == APP07:
            return StructuredHit(
                ds,
                r["doc_id"],
                r["id"],
                r["source"]["pdf_page"],
                r["source"]["printed_page"],
                [],
                {
                    "display_name": r["display_name"],
                    "category": r["category"],
                    "density": r["density"],
                    "thermal_conductivity": r["thermal_conductivity"],
                },
                r["verification"],
                self._issues_of(r["id"]),
                why,
            )
        meta = self.main[ds].meta
        cells = list(r["source_cells"])
        return StructuredHit(
            ds,
            r["doc_id"],
            r["record_id"],
            meta["pdf_page"],
            str(meta["printed_page"]),
            cells,
            {k: v for k, v in r.items() if k not in _RECORD_META},
            {"status": "curated", "table_status": meta.get("status")},
            self._issues_of(r["record_id"], r["source_issue_refs"]),
            why,
        )


def _index_labels(t: MainTable) -> None:
    """Row-label index of a main table (unique printed labels -> record positions)."""
    fields = LABEL_FIELDS.get(t.meta["table_id"].removeprefix(MAIN), ())
    if not fields:
        return
    by_label: dict[tuple[str, ...], list[int]] = {}
    texts: dict[tuple[str, ...], list[tuple[str, int]]] = {}
    for i, r in enumerate(t.records):
        parts = [(_field(r, path) or "", level) for path, level in fields]
        key = tuple(p for p, _ in parts)
        by_label.setdefault(key, []).append(i)
        texts[key] = parts
    labels = []
    rows: dict[str, list[int]] = {}
    for n, (key, positions) in enumerate(by_label.items()):
        toks: list[str] = []
        levels: list[int] = []
        for text, level in texts[key]:
            words = [w for w in etokens(text) if w not in "≥≤<>"]
            toks += words
            levels += [level] * len(words)
        if toks:
            lid = f"L{n}"
            labels.append(Label(lid, toks, levels))
            rows[lid] = positions
    caption = set(etokens(t.meta["caption"]["text"]))
    vocab = {w for lab in labels for w in lab.tokens}
    t.caption_words = frozenset(caption)
    # caption words that are not label words never anchor a row
    # request words («حداقل», «فضای») never anchor a row on their own
    t.labels = LabelIndex(labels, ignore=(caption - vocab) | _COMPARISON_WORDS | REQUEST_WORDS)
    t.label_rows = rows


_PROVENANCE_KEYS = frozenset(
    {"verification", "source", "provenance", "source_cells", "anchor_cell", "mask_digest"}
)
_COMPARISON_WORDS = frozenset({"کوچک", "بزرگ", "تر", "کمتر", "بیشتر", "متر", "مربع", "مترمربع"})


def fully_named(m: Any) -> bool:
    """A row label whose most specific printed level is entirely named in the query and whose
    every printed level is at least partly named (equipment and specification alike)."""
    # specificity is the per-level share of named words (by word, so a word printed at two
    # levels names both), deepest level first
    return bool(m.specificity) and m.specificity[0] >= 1.0 and min(m.specificity) > 0


def rating_code(record: dict[str, Any]) -> str | None:
    """The printed energy rating of a record (``code`` in 19-3-3, ``printed`` category in the
    equipment tables); None when the record prints no rating (dash)."""
    rating = record.get("energy_rating")
    if isinstance(rating, str):
        return rating
    if not isinstance(rating, dict):
        return None
    if rating.get("code"):
        return str(rating["code"])
    if rating.get("value_class") == "category" and rating.get("printed"):
        return str(rating["printed"])
    return None


def _aliases(name: str) -> set[str]:
    norm = normalize_for_search(name)
    base = norm.split("(")[0].strip()
    out = {norm, base}
    if base.startswith("بندر "):
        out.add(base.removeprefix("بندر "))
    return {a for a in out if a}


def _material_name(r: dict[str, Any]) -> str:
    parts = [r.get("display_name") or ""]
    for k in ("category", "subgroup", "material", "condition"):
        v = r.get(k)
        if isinstance(v, dict):
            parts.append(v.get("text") or "")
    return " ".join(parts)


def _groups(r: dict[str, Any]) -> list[list[str]]:
    return [v["applies_to"] for v in r.values() if isinstance(v, dict) and "applies_to" in v]
