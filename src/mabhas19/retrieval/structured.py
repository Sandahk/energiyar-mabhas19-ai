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

from mabhas19.retrieval.lexical import BM25
from mabhas19.retrieval.models import Conflict, StructuredHit, StructuredNoMatch, TableEvidence
from mabhas19.retrieval.normalization import index_terms, tokens
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
    m = re.search(r"(?:چگالی|جرم مخصوص)\s*(?:[^\d\s]+\s*){0,2}?(\d+(?:\.\d+)?)", norm)
    return float(m[1]) if m else None


def numbers(norm: str) -> list[str]:
    return re.findall(r"(?<![\d-])\d+(?:\.\d+)?(?![\d-])", norm)


def r_value_in(norm: str) -> Decimal | None:
    """The R ratio stated in the query («نسبت R ۵», «R برابر ۲.۵»), if any."""
    m = re.search(r"(?<![a-z])r(?![a-z\-])\s*(?:\S+\s*){0,3}?(\d+(?:\.\d+)?)(?![\d\-a-z])", norm)
    return _decimal(m[1]) if m else None


def degree_days_in(norm: str) -> tuple[str, Decimal] | None:
    """(variable, value) for «روز درجه گرمایش ۵۰۰۰», «HDD 5000», «روز درجه سرمایش ...»."""
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
            self.main[meta["table_id"]] = MainTable(
                meta["table_id"], meta, recs, BM25(labels), domain
            )
        self.app03_domain = frozenset(r["climate_class"] for r in self.app03)
        self.material_terms = [set(index_terms(_material_name(r))) for r in self.app07]
        self.materials_index = BM25([index_terms(_material_name(r)) for r in self.app07])
        self.occupancy_index = list(self.occupancies.values())
        self.places = self._places()

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
        padded = f" {norm} "
        hits = [p for p in self.places if f" {p} " in padded]
        return [p for p in hits if not any(p != q and p in q for q in hits)]  # longest names

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
        occ = self._occupancies(norm)
        codes, rates = climate_codes(norm), ratings(norm)
        notes: list[str] = []
        no_match: list[StructuredNoMatch] = []
        constraints = {"occupancies": occ, "climate_classes": codes, "ratings": rates}
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
        invalid = [c for c in codes if c not in self.app03_domain]
        for c in invalid:
            issues = ["SI-0002"] if c == "5C" else []
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
        valid = [c for c in codes if c in self.app03_domain]
        if codes and not valid:
            return [], notes, no_match
        hits = [
            self._hit(
                APP03,
                r,
                f"occupancy {r['occupancy_id']}, {r['climate_class']}, rating {r['energy_rating']}",
            )
            for r in self.app03
            if r["occupancy_id"] in occ
            and (not valid or r["climate_class"] in valid)
            and (not rates or r["energy_rating"] in rates)
        ]
        return hits[:MAX_HITS], notes, no_match

    def occupancies_in(self, norm: str) -> list[str]:
        return self._occupancies(norm)

    def _occupancies(self, norm: str) -> list[str]:
        q = set(tokens(norm)) - _PROPERTY_WORDS - _EUI_WORDS
        scored = []
        for o in self.occupancy_index:
            label = set(tokens(o["label"]))
            overlap = len(q & label)
            ops = [c for c in "≥≤<>" if c in o["label"]]
            if overlap and (not any(c in norm for c in "≥≤<>") or any(c in norm for c in ops)):
                scored.append((overlap, o["id"]))
        if not scored:
            return []
        top = max(s for s, _ in scored)
        return [i for s, i in scored if s == top]

    # ------------------------------------------------------------------ appendix 7
    def material_terms_of(self, norm: str) -> list[str]:
        """Query terms that name a specific material (property and generic words removed)."""
        return [
            t
            for t in tokens(norm)
            if t not in _PROPERTY_WORDS
            and t not in GENERIC_MATERIAL_WORDS
            and t not in _QUESTION_WORDS
            and not t.replace(".", "").isdigit()
        ]

    def resolve_material(self, norm: str) -> list[int] | None:
        """Rows whose curated name contains every specific query term; ``None`` when the query
        names no specific material (only generic words such as «عایق»), ``[]`` when it names an
        unknown one."""
        specific = self.material_terms_of(norm)
        if not specific:
            return None
        return [i for i, names in enumerate(self.material_terms) if set(specific) <= names]

    def materials(
        self, norm: str
    ) -> tuple[list[StructuredHit], list[str], list[StructuredNoMatch]]:
        rows_idx = self.resolve_material(norm)
        constraints = {"material_terms": self.material_terms_of(norm), "density": density_in(norm)}
        if not rows_idx:
            reason = "unknown_material" if rows_idx == [] else "no_material_named"
            return (
                [],
                [],
                [
                    self._no_match(
                        APP07,
                        reason,
                        "no curated material row matches the named material",
                        constraints,
                    )
                ],
            )
        q = [t for t in index_terms(norm) if t not in _PROPERTY_WORDS and not t.isdigit()]
        allowed = set(rows_idx)
        ranked = [(i, s) for i, s, _ in self.materials_index.scores(q) if i in allowed]
        top = ranked[0][1] if ranked else 0.0
        rows = [self.app07[i] for i, s in ranked if s >= top * 0.999] or [
            self.app07[i] for i in rows_idx
        ]
        notes: list[str] = []
        dens = density_in(norm)
        if dens is not None:
            rows, notes = _density_filter(rows, dens)
            if not rows:
                return (
                    [],
                    notes,
                    [
                        self._no_match(
                            APP07,
                            "no_matching_row",
                            f"no printed density range of the material covers {dens:g}",
                            constraints,
                        )
                    ],
                )
        return [self._hit(APP07, r, f"material «{r['display_name']}»") for r in rows], notes, []

    # ------------------------------------------------------------------ main tables
    def main_table(
        self, table_id: str, norm: str, *, whole_table: bool = False
    ) -> tuple[list[StructuredHit], list[StructuredNoMatch], list[str]]:
        t = self.main[table_id]
        codes, rates = climate_codes(norm), ratings(norm)
        notes: list[str] = []
        constraints: dict[str, Any] = {"climate_classes": codes, "ratings": rates}
        if table_id == T_DEGREE_DAYS and degree_days_in(norm):
            return self._degree_days(t, norm, constraints)
        if table_id == T_R_RATING and r_value_in(norm) is not None:
            return self._r_rating(t, norm, constraints)
        valid = codes
        if codes and t.climate_domain:
            invalid = [c for c in codes if c not in t.climate_domain]
            valid = [c for c in codes if c in t.climate_domain]
            if invalid:
                nm = self._no_match(
                    table_id,
                    "climate_class_not_in_table",
                    f"climate class {', '.join(invalid)} is not a column of table "
                    f"{table_id.removeprefix(MAIN)} (columns: "
                    f"{', '.join(sorted(t.climate_domain))})",
                    constraints,
                )
                if not valid:
                    return [], [nm], notes
                return self._label_lookup(t, norm, valid, rates, whole_table), [nm], notes
        elif codes and not t.climate_domain:
            notes.append(
                f"table {table_id.removeprefix(MAIN)} has no climate dimension; the "
                f"requested class {', '.join(codes)} does not filter it"
            )
            valid = []
        hits = self._label_lookup(t, norm, valid, rates, whole_table)
        if not hits and not whole_table:
            return (
                [],
                [
                    self._no_match(
                        table_id,
                        "no_matching_row",
                        "no row label of the table matches the query",
                        constraints,
                    )
                ],
                notes,
            )
        return hits, [], notes

    def _label_lookup(
        self, t: MainTable, norm: str, codes: list[str], rates: list[str], whole_table: bool
    ) -> list[StructuredHit]:
        ranked = t.index.scores(index_terms(norm))
        candidates = [(i, s) for i, s, _ in ranked]
        if not candidates and whole_table:
            candidates = [(i, 0.0) for i in range(len(t.records))]
        if candidates and candidates[0][1] > 0:
            top = candidates[0][1]
            candidates = [(i, s) for i, s in candidates if s >= top * 0.999]
        recs = [t.records[i] for i, _ in candidates]
        recs = [r for r in recs if _climate_ok(r, codes) and _rating_ok(r, rates)]
        why = "table reference" if whole_table and not ranked else "label match"
        return [self._hit(t.table_id, r, why) for r in recs[:MAX_HITS]]

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
                list(r["source_issue_refs"]),
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
                list(occ.get("source_issue_refs", [])),
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
                [],
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
            list(r["source_issue_refs"]),
            why,
        )


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
