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
from rapidfuzz.distance import Levenshtein

from mabhas19.retrieval.dimensions import (
    GROUP_LETTERS,
    Interval,
    area_requested,
    decimal_text,
    density_requested,
    groups_requested,
    label_operator_interval,
    printed_interval,
    ratings_requested,
    years_requested,
)
from mabhas19.retrieval.entities import (
    EntityResolver,
    Label,
    LabelIndex,
    Resolution,
    best_matches,
    etokens,
    keys,
)
from mabhas19.retrieval.lexical import BM25
from mabhas19.retrieval.models import Conflict, StructuredHit, StructuredNoMatch, TableEvidence
from mabhas19.retrieval.normalization import index_terms
from mabhas19.text.normalize import normalize_for_search

APP02 = "m19-ed5-1404-app02.city_climate_classes"
APP03 = "m19-ed5-1404-app03.max_energy_use_intensity"
APP07 = "m19-ed5-1404-app07.material_thermal_properties"
MAIN = "m19-ed5-1404-main.t"
MAIN_3_2 = MAIN + "19-3-2"
T_DEGREE_DAYS = MAIN + "19-3-1"
T_R_RATING = MAIN + "19-3-3"
MAX_HITS = 60
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
# words that end the modifier phrase of a matched row label
_PHRASE_BREAK = frozenset({"در", "برای", "طبق", "مطابق", "که", "و", "یا", "با", "از"})
# words that never name a different row (generic building words)
_GENERIC_MODIFIERS = frozenset({"ساختمان", "ساختمانی", "ساختمانها", "ساختمانهای", "فضا", "فضای"})


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


def density_in(norm: str) -> float | None:
    """The number stated as a density («چگالی ۴۰۰», «جرم مخصوص ۱۸۰۰ کیلوگرم»), if any."""
    norm = decimal_text(norm)
    m = re.search(r"(?:چگالی|جرم مخصوص)\s*(?:[^\d\s]+\s*){0,2}?(\d+(?:\.\d+)?)", norm)
    return float(m[1]) if m else None


def numbers(norm: str) -> list[str]:
    return re.findall(r"(?<![\d-])\d+(?:\.\d+)?(?![\d-])", norm)


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


def satisfies(cond: dict[str, Any], x: Decimal) -> bool:
    """Apply the printed operators of a condition («a < X ≤ b») to a given number."""
    lo, hi = cond.get("lower"), cond.get("upper")
    if lo:
        v = Decimal(lo["value"])
        if not (x > v if lo["operator"] == "<" else x >= v):
            return False
    if hi:
        v = Decimal(hi["value"])
        if not (x < v if hi["operator"] == "<" else x <= v):
            return False
    return True


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

    def climate(self, norm: str) -> tuple[list[StructuredHit], list[Conflict]]:
        hits: list[StructuredHit] = []
        conflicts: list[Conflict] = []
        for place in self.find_places(norm):
            group = [self._hit(ds, r, f"place «{place}»") for ds, r in self.places[place]]
            hits += group
            codes = {self._code(h) for h in group}
            if len(codes) > 1:
                issues = sorted({i for h in group for i in h.source_issues})
                conflicts.append(
                    Conflict(
                        issue_id=issues[0] if issues else None,
                        field="climate_class",
                        hits=[h.record_id for h in group],
                        values=sorted(codes),
                        note=f"curated sources disagree for «{place}»; "
                        "both are returned, neither is preferred",
                    )
                )
        return _dedupe(hits), conflicts

    @staticmethod
    def _code(h: StructuredHit) -> str:
        return str(h.values["climate_class"]["code"])

    def wmo(self, norm: str) -> list[StructuredHit]:
        codes = set(re.findall(r"(?<!\d)\d{6}(?!\d)", norm))
        return [
            self._hit(MAIN_3_2, r, f"WMO code {r['wmo_code']['ascii']}")
            for r in self.main[MAIN_3_2].records
            if r["wmo_code"]["ascii"] in codes
        ]

    # ------------------------------------------------------------------ appendix 3
    def eui(self, norm: str) -> tuple[list[StructuredHit], list[str], list[StructuredNoMatch]]:
        occ, notes = self._occupancy_resolution(norm)
        codes, rates = climate_codes(norm), ratings_requested(norm)
        no_match: list[StructuredNoMatch] = []
        constraints: dict[str, Any] = {
            "occupancies": occ,
            "climate_classes": codes,
            "ratings": rates,
        }
        # issues of the resolved occupancy rows travel with every outcome of the lookup
        derived_issues: list[str] = list(
            dict.fromkeys(
                i for o in occ for i in self.occupancies[o].get("source_issue_refs") or []
            )
        )
        if not codes and occ:
            codes, derivation = self.derive_climate(norm)
            if derivation:
                constraints["climate_derivation"] = derivation
                notes.append(
                    "climate class derived from the named place: "
                    + "; ".join(
                        f"{d['place']} -> {d['climate_class']} ({d['record']})" for d in derivation
                    )
                )
                derived_issues = list(
                    dict.fromkeys(
                        [*derived_issues, *(i for d in derivation for i in d["source_issues"])]
                    )
                )
            constraints["climate_classes"] = codes
        # a requested class absent from Appendix 3 is reported whatever else is resolved
        for c in [c for c in codes if c not in self.app03_domain]:
            issues = list(dict.fromkeys(self.app03_missing_issues.get(c, []) + derived_issues))
            if c == "5C":
                notes.append(NO_5C_NOTE)
            no_match.append(
                self._no_match(
                    APP03,
                    "climate_class_not_in_table",
                    f"climate class {c} is not a column of Appendix 3 "
                    f"(columns: {', '.join(sorted(self.app03_domain))})",
                    constraints,
                    issues,
                )
            )
        if not occ:
            no_match.append(
                self._no_match(
                    APP03,
                    "unknown_occupancy",
                    "no curated occupancy type matches the query",
                    constraints,
                )
            )
            return [], notes, no_match
        if not codes:
            no_match.append(
                self._no_match(
                    APP03,
                    "missing_dimension",
                    "Appendix 3 values depend on the climate class, which the query does not "
                    "state or imply; no class is assumed",
                    constraints,
                    derived_issues,
                )
            )
            return [], notes, no_match
        valid = [c for c in codes if c in self.app03_domain]
        rating_domain = {r["energy_rating"] for r in self.app03}
        bad_rates = [r for r in rates if r not in rating_domain]
        if bad_rates:
            no_match.append(
                self._no_match(
                    APP03,
                    "rating_not_in_table",
                    f"rating {', '.join(bad_rates)} is not printed in Appendix 3 "
                    f"(ratings: {', '.join(sorted(rating_domain))})",
                    constraints,
                    derived_issues,
                )
            )
        good_rates = [r for r in rates if r in rating_domain]
        if not rates and _RATING_MENTION.search(norm):
            no_match.append(
                self._no_match(
                    APP03,
                    "missing_dimension",
                    "a rating is referred to but not stated; no rating is assumed",
                    constraints,
                    derived_issues,
                )
            )
            return [], notes, no_match
        if not valid or (rates and not good_rates):
            return [], notes, no_match
        hits = [
            self._derived(
                self._hit(
                    APP03,
                    r,
                    f"occupancy {r['occupancy_id']}, {r['climate_class']}, "
                    f"rating {r['energy_rating']}",
                ),
                derived_issues,
            )
            for r in self.app03
            if r["occupancy_id"] in occ
            and r["climate_class"] in valid
            and (not good_rates or r["energy_rating"] in good_rates)
        ]
        return hits[:MAX_HITS], notes, no_match

    @staticmethod
    def _derived(hit: StructuredHit, issues: list[str]) -> StructuredHit:
        """A hit found through a derived constraint inherits the derivation's issues."""
        hit.source_issues = list(dict.fromkeys([*hit.source_issues, *issues]))
        return hit

    def derive_climate(self, norm: str) -> tuple[list[str], list[dict[str, Any]]]:
        """Climate classes of the places named in the query (Appendix 2 and table 19-3-2), with
        the records used, so the derivation and its source issues stay traceable."""
        codes: list[str] = []
        used: list[dict[str, Any]] = []
        for place in self.find_places(norm):
            for ds, r in self.places[place]:
                code = r["climate_class"]["code"]
                used.append(
                    {
                        "place": place,
                        "dataset": ds,
                        "record": r.get("id") or r.get("record_id"),
                        "climate_class": code,
                        "source_issues": list(r.get("source_issue_refs") or []),
                    }
                )
                if code not in codes:
                    codes.append(code)
        return codes, used

    def occupancies_in(self, norm: str) -> list[str]:
        return self._occupancies(norm)

    def _occupancies(self, norm: str) -> list[str]:
        return self._occupancy_resolution(norm)[0]

    def _occupancy_resolution(self, norm: str) -> tuple[list[str], list[str]]:
        """(occupancy ids, notes). A stated size («۳۰۰۰ متر مربع و بیشتر», «≥ ۳۰۰۰») is
        compared with each label's printed size condition."""
        res = self.resolver.occupancies(norm)
        stated = area_requested(norm) or label_operator_interval(norm)
        sized = {
            oid: label_operator_interval(normalize_for_search(self.occupancies[oid]["label"]))
            for oid in res.ids
        }
        if stated is None:
            notes = (
                ["size not stated: every size row of the occupancy is returned"]
                if sum(1 for v in sized.values() if v) > 1
                else []
            )
            return res.ids, notes
        inside = [o for o, iv in sized.items() if iv is None or stated.within(iv)]
        if inside:
            return inside, []
        overlap = [o for o, iv in sized.items() if iv is not None and stated.overlaps(iv)]
        note = f"stated size {stated.describe()} spans several printed size conditions"
        return overlap, [note] if overlap else []

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

    def materials(
        self, norm: str
    ) -> tuple[list[StructuredHit], list[str], list[StructuredNoMatch]]:
        dens = density_requested(norm)
        res = self.resolver.materials(norm)
        constraints = {
            "material_terms": self.material_terms_of(norm),
            "density": dens.describe() if dens else None,
            "resolution": res.status,
            "leftover": res.leftover,
        }
        if res.status != "resolved":
            reason = {
                "none": "no_material_named",
                "ambiguous": "ambiguous_material",
            }.get(res.status, "unknown_material")
            return (
                [],
                list(res.notes),
                [
                    self._no_match(
                        APP07,
                        reason,
                        "no curated material label matches the named material",
                        constraints,
                    )
                ],
            )
        rows = [self.app07[self._app07_pos[i]] for i in res.ids]
        notes: list[str] = list(res.notes)
        if dens is not None:
            rows, density_notes = _density_match(rows, dens)
            notes += density_notes
            if not rows:
                return (
                    [],
                    notes,
                    [
                        self._no_match(
                            APP07,
                            "no_matching_row",
                            f"no printed density range of the material covers {dens.describe()}",
                            constraints,
                        )
                    ],
                )
        elif len(rows) > 1:
            notes.append(
                "density not stated: several printed rows of the named material are returned, "
                "each with its printed condition; none is chosen"
            )
        return [self._hit(APP07, r, f"material «{r['display_name']}»") for r in rows], notes, []

    # ------------------------------------------------------------------ main tables
    def main_table(  # noqa: PLR0911, PLR0912
        self, table_id: str, norm: str, *, whole_table: bool = False
    ) -> tuple[list[StructuredHit], list[StructuredNoMatch], list[str]]:
        t = self.main[table_id]
        codes = climate_codes(norm)
        notes: list[str] = []
        constraints: dict[str, Any] = {"climate_classes": codes, "ratings": ratings_requested(norm)}
        if table_id == T_DEGREE_DAYS and degree_days_in(norm):
            return self._degree_days(t, norm, constraints)
        if table_id == T_R_RATING and r_value_in(norm) is not None:
            return self._r_rating(t, norm, constraints)
        no_match: list[StructuredNoMatch] = []

        def fail(reason: str, detail: str) -> StructuredNoMatch:
            return self._no_match(table_id, reason, detail, constraints)

        # climate classes
        valid = codes
        if codes and t.climate_domain:
            invalid = [c for c in codes if c not in t.climate_domain]
            valid = [c for c in codes if c in t.climate_domain]
            if invalid:
                no_match.append(
                    fail(
                        "climate_class_not_in_table",
                        f"climate class {', '.join(invalid)} is not a column of table "
                        f"{table_id.removeprefix(MAIN)} (columns: "
                        f"{', '.join(sorted(t.climate_domain))})",
                    )
                )
                if not valid:
                    return [], no_match, notes
        elif codes and not t.climate_domain:
            notes.append(
                f"table {table_id.removeprefix(MAIN)} has no climate dimension; the "
                f"requested class {', '.join(codes)} does not filter it"
            )
            valid = []
        # other printed dimensions
        filters, problems = self._dimension_filters(t, norm, constraints, notes)
        if problems:
            return [], no_match + [fail(r, d) for r, d in problems], notes
        # row labels
        rows, label_note, label_problem = self._match_rows(t, norm, whole_table)
        if label_note:
            notes.append(label_note)
        if label_problem:
            return [], [*no_match, fail("no_matching_row", label_problem)], notes
        if rows is None:
            if t.labels is not None and not whole_table:
                # the table's rows are labelled and none is named: never list a whole column
                detail = "no printed row of the table is named; no column is listed"
                return [], [*no_match, fail("no_matching_row", detail)], notes
            if whole_table or valid or filters:
                rows = list(range(len(t.records)))
            else:
                detail = "no row label of the table matches the query"
                return [], [*no_match, fail("no_matching_row", detail)], notes
        recs = [t.records[i] for i in rows]
        recs = [r for r in recs if _climate_ok(r, valid) and all(f(r) for f in filters)]
        if not recs:
            detail = "no printed row satisfies every stated constraint"
            if constraints.get("area"):
                detail = (
                    f"no printed condition covers area {constraints['area']}; nothing is inferred"
                )
            return [], [*no_match, fail("no_matching_row", detail)], notes
        why = "table reference" if whole_table and rows is not None else "label match"
        return [self._hit(t.table_id, r, why) for r in recs[:MAX_HITS]], no_match, notes

    def _dimension_filters(
        self, t: MainTable, norm: str, constraints: dict[str, Any], notes: list[str]
    ) -> tuple[list[Any], list[tuple[str, str]]]:
        """Filters for the printed group/year/rating/area dimensions; invalid values are
        returned as (reason, detail) problems, values of absent dimensions as notes."""
        filters: list[Any] = []
        problems: list[tuple[str, str]] = []
        rec0 = t.records[0]
        groups, years = groups_requested(norm), years_requested(norm)
        if groups or years:
            constraints |= {"groups": groups, "years": years}
        if "building_group" in rec0:
            domain_g = {r["building_group"]["text"] for r in t.records}
            domain_y = {r["year"]["index"] for r in t.records}
            if [g for g in groups if g not in domain_g]:
                problems.append(("group_not_in_table", f"group {groups} not in {GROUP_LETTERS}"))
            if [y for y in years if y not in domain_y]:
                problems.append(("year_not_in_table", f"year {years} not in {sorted(domain_y)}"))
            if groups:
                filters.append(lambda r, g=set(groups): r["building_group"]["text"] in g)
            if years:
                filters.append(lambda r, y=set(years): r["year"]["index"] in y)
        rates = constraints["ratings"]
        if "energy_rating" in rec0 and rates:
            domain_r = {c for r in t.records if (c := rating_code(r)) is not None}
            bad = [x for x in rates if x not in domain_r]
            if bad:
                problems.append(
                    ("rating_not_in_table", f"rating {bad} not printed ({sorted(domain_r)})")
                )
            filters.append(lambda r, x=set(rates): rating_code(r) in x)
        elif rates:
            notes.append(f"the table has no rating dimension; rating {rates} does not filter it")
        if "use" in rec0:
            stated = area_requested(norm)
            if stated is not None:
                constraints["area"] = stated.describe()
                filters.append(lambda r, a=stated: _area_ok(r, a))
        return filters, problems

    def _match_rows(
        self, t: MainTable, norm: str, whole_table: bool
    ) -> tuple[list[int] | None, str | None, str | None]:
        """(record positions or None when no row label is named, note, no-match detail)."""
        if t.labels is None or t.label_rows is None:
            return None, None, None
        q = t.labels.query_tokens(norm)
        matches = t.labels.match(q)
        # a row whose own printed label (its most specific level) is entirely named in the query
        # is the row asked about; other partial namings do not compete with it
        named = [m for m in matches if fully_named(m)]
        best = best_matches(named or matches)
        if not best:
            residual = [
                w
                for w in q
                if w not in t.labels.ignore
                and w not in REQUEST_WORDS
                and not w.replace(".", "").isdigit()
                and not _near(w, t.caption_words)
                and not re.fullmatch(r"\d[a-z]", w)
                and w not in GROUP_LETTERS
            ]
            if residual and whole_table:
                return None, None, f"no row of the table matches {residual}"
            return None, None, None
        # an unmatched content word right after the matched label words names another row
        used = set().union(*(m.query_idx for m in best))
        end = max(used) + 1
        if max(m.coverage for m in best) < 0.5 and end < len(q):
            nxt = q[end]
            if (
                nxt not in _PHRASE_BREAK
                and nxt not in t.labels.ignore
                and nxt not in _GENERIC_MODIFIERS
                and nxt not in REQUEST_WORDS
                and not nxt.replace(".", "").isdigit()
            ):
                phrase = " ".join(q[min(used) : end + 1])
                return None, None, f"«{phrase}» does not match a printed row label"
        rows = sorted({i for m in best for i in t.label_rows[m.label.id]})
        note = None
        if len({m.label.id for m in best}) > 1:
            note = "several printed rows fit the named label: " + "; ".join(
                " ".join(m.label.tokens) for m in best
            )
        return rows, note, None

    def _degree_days(
        self, t: MainTable, norm: str, constraints: dict[str, Any]
    ) -> tuple[list[StructuredHit], list[StructuredNoMatch], list[str]]:
        found = degree_days_in(norm)
        assert found is not None
        var, x = found
        constraints |= {"variable": var, "value": str(x)}
        hits: list[StructuredHit] = []
        notes: list[str] = []
        for r in t.records:
            conds = [c for c in r["criterion"]["conditions"] if c["variable"] == var]
            if conds and all(satisfies(c, x) for c in conds):
                others = [
                    c["variable"] for c in r["criterion"]["conditions"] if c["variable"] != var
                ]
                why = f"printed condition on {var} satisfied by {x}"
                if others:
                    why += f"; other condition(s) on {', '.join(others)} not given in the query"
                hits.append(self._hit(t.table_id, r, why))
        if not hits:
            return (
                [],
                [
                    self._no_match(
                        t.table_id,
                        "no_printed_condition",
                        f"no printed condition of the table covers {var} = {x}; "
                        "no class is inferred",
                        constraints,
                    )
                ],
                notes,
            )
        return hits, [], notes

    def _r_rating(
        self, t: MainTable, norm: str, constraints: dict[str, Any]
    ) -> tuple[list[StructuredHit], list[StructuredNoMatch], list[str]]:
        x = r_value_in(norm)
        assert x is not None
        constraints |= {"R": str(x)}
        hits = [
            self._hit(t.table_id, r, f"printed range {r['r_range']['printed']} contains R = {x}")
            for r in t.records
            if satisfies(r["r_range"], x)
        ]
        if not hits:
            return (
                [],
                [
                    self._no_match(
                        t.table_id,
                        "no_applicable_category",
                        f"no printed R range of the table contains R = {x}; no rating is inferred",
                        constraints,
                    )
                ],
                [],
            )
        return hits, [], []

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


# a rating referred to («آن رده», «به چه رده») - not a climate class («رده اقلیمی»)
_RATING_MENTION = re.compile(r"(?:رده|رتبه)(?!\s*(?:اقلیم|بازدهی انرژی\s*[a-z]))(?!\s*[a-z])")
_PROVENANCE_KEYS = frozenset(
    {"verification", "source", "provenance", "source_cells", "anchor_cell", "mask_digest"}
)
_COMPARISON_WORDS = frozenset({"کوچک", "بزرگ", "تر", "کمتر", "بیشتر", "متر", "مربع", "مترمربع"})


def _near(word: str, vocab: frozenset[str]) -> bool:
    """Word equal to a vocabulary word up to morphology or one edit (long words only)."""
    if keys(word) & {k for v in vocab for k in keys(v)}:
        return True
    return len(word) >= 5 and any(len(v) >= 5 and Levenshtein.distance(word, v) <= 1 for v in vocab)


def _area_ok(r: dict[str, Any], stated: Interval) -> bool:
    cond = (r.get("use") or {}).get("area_condition")
    if not cond:
        return True  # the row prints no size condition
    printed = printed_interval(cond["comparison_printed"], cond["threshold"]["value"])
    return printed is not None and stated.within(printed)


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


def _printed_density(r: dict[str, Any]) -> tuple[Decimal | None, Decimal | None]:
    d = r["density"]
    lo = Decimal(d["min"]) if d.get("min") else None
    hi = Decimal(d["max"]) if d.get("max") else None
    return lo, hi


def _density_match(
    rows: list[dict[str, Any]], stated: Interval
) -> tuple[list[dict[str, Any]], list[str]]:
    """Rows whose printed density condition fits the stated density.

    A single value keeps the point semantics (inside a printed range, or equal to a printed
    bound -> every row touching that bound). A stated range [a, b] matches the row printing
    exactly that range, else rows whose printed range contains it, else rows it genuinely
    overlaps (more than a shared end point), with a note; nothing is interpolated and the range
    is never reduced to a midpoint or one bound."""
    if stated.lo is not None and stated.lo == stated.hi:
        return _density_filter(rows, float(stated.lo))
    a, b = stated.lo, stated.hi
    exact = [r for r in rows if _printed_density(r) == (a, b)]
    if exact:
        return exact, []
    inside = []
    overlap = []
    for r in rows:
        lo, hi = _printed_density(r)
        if (lo is None or (a is not None and a >= lo)) and (
            hi is None or (b is not None and b <= hi)
        ):
            inside.append(r)
        elif (lo is None or b is None or b > lo) and (hi is None or a is None or a < hi):
            overlap.append(r)
    if inside:
        return inside, []
    if overlap:
        return overlap, [
            f"stated density range {a}–{b} spans several printed ranges; each overlapping "
            "printed row is returned and none is interpolated"
        ]
    return [], []


def _density_filter(rows: list[dict[str, Any]], x: float) -> tuple[list[dict[str, Any]], list[str]]:
    keep, boundary = [], False
    for r in rows:
        d = r["density"]
        lo = float(d["min"]) if d.get("min") else None
        hi = float(d["max"]) if d.get("max") else None
        if x in (lo, hi):
            keep.append(r)
            boundary = True
        elif (lo is None or x > lo) and (hi is None or x < hi):
            keep.append(r)
    notes = (
        [
            "density equals a printed bound: every row touching that bound is returned "
            "(the source does not state inclusivity)"
        ]
        if boundary
        else []
    )
    return keep, notes


def _groups(r: dict[str, Any]) -> list[list[str]]:
    return [v["applies_to"] for v in r.values() if isinstance(v, dict) and "applies_to" in v]


def _climate_ok(r: dict[str, Any], codes: list[str]) -> bool:
    groups = _groups(r)
    return not codes or not groups or any(c in g for g in groups for c in codes)


def _rating_ok(r: dict[str, Any], rates: list[str]) -> bool:
    rating = r.get("energy_rating")
    if not rates or not isinstance(rating, dict) or "code" not in rating:
        return True
    return bool(rating["code"] in rates)


def _dedupe(hits: list[StructuredHit]) -> list[StructuredHit]:
    seen: set[str] = set()
    out = []
    for h in hits:
        if h.record_id not in seen:
            seen.add(h.record_id)
            out.append(h)
    return out
