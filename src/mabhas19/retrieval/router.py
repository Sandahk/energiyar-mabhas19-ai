"""Deterministic, auditable query router (v2): structured | prose | mixed | unsupported.

No LLM. Every decision lists the signals it saw and the rule that fired.

Order of rules (first match wins):

1. **answerability gate** (:mod:`mabhas19.retrieval.answerability`): live information,
   commercial, service, education and outside-Mabhas-19 intents -> unsupported (unless the
   query names a section or table explicitly);
2. **decomposition**: the query is split into clauses at «و» / punctuation; clauses with
   explanatory wording form the *prose part*, clauses with structured signals the *structured
   part* (a clause can be both; a clause with neither joins its neighbour);
3. **structured signals with entity validation** on the structured part:
   * climate lookup only with climate intent AND a curated place (unknown place -> recorded);
   * Appendix 7 only for a resolvable curated material (generic words such as «عایق» never
     count; unknown material -> recorded);
   * Appendix 3 only for a resolvable occupancy (unknown occupancy -> recorded);
   * main tables by explicit reference or table vocabulary;
4. an unresolved entity with no other structured signal -> unsupported (unknown entity);
5. structured + prose parts -> mixed; structured only -> structured;
6. section/clause reference -> prose;
7. fallback: little of the query in the corpus vocabulary and no domain signal ->
   unsupported; otherwise prose.
"""

from __future__ import annotations

import re
from typing import Any

from mabhas19.retrieval.answerability import gate, has_words
from mabhas19.retrieval.models import RouteDecision
from mabhas19.retrieval.normalization import content_tokens, index_terms, normalize_query
from mabhas19.retrieval.structured import (
    APP02,
    APP03,
    APP07,
    MAIN,
    MAIN_3_2,
    StructuredStore,
    climate_codes,
)

# topic vocabulary per structured dataset
TABLE_TOPICS: dict[str, tuple[str, ...]] = {
    MAIN + "19-2-1": (
        "زمان بندی",
        "سال اول",
        "سال دوم",
        "سال سوم",
        "سال چهارم",
        "سال پنجم",
        "گروه ساختمانی",
        "گروه ساختمان",
    ),
    MAIN + "19-3-1": ("روز درجه", "cdd", "hdd"),
    MAIN + "19-3-3": ("نسبت r", "رده بازدهی انرژی ساختمان بر اساس"),
    MAIN + "19-5-1": (
        "حداقل مقاومت حرارتی",
        "r-value",
        "مقاومت حرارتی دیوار",
        "مقاومت حرارتی سقف",
        "مقاومت حرارتی کف",
        "مقاومت حرارتی در",
    ),
    MAIN + "19-5-2": (
        "u-value",
        "ضریب انتقال حرارت",
        "shgc",
        "ضریب بهره خورشیدی",
        "نورگیر سقفی",
        "سایه گیر",
        "پنجره ثابت",
        "پنجره متحرک",
    ),
    MAIN + "19-5-3": (
        "بویلر",
        "چیلر",
        "دیگ بخار",
        "الکتروموتور",
        "پمپ های گریز از مرکز",
        "برج خنک کن",
        "برج های خنک کن",
        "هواساز",
        "فن کویل",
        "تولید همزمان",
    ),
    MAIN + "19-5-4": ("آبگرمکن", "پکیج", "بخاری", "کولر آبی", "کولر گازی", "دمنده"),
    MAIN + "19-5-5": ("دمای آبگرم", "دمای آب گرم", "دوش", "روشویی", "ظرفشویی"),
    MAIN + "19-5-6": ("نور روز", "dla", "مساحت مستقل نور طبیعی", "کفایت نور"),
    MAIN + "19-5-7": ("یکنواختی",),
    APP03: ("شدت مصرف انرژی", "eui", "kwh"),
    APP07: ("ضریب هدایت", "هدایت حرارتی", "چگالی", "جرم مخصوص"),
}
CLIMATE_INTENT = ("اقلیم", "رده اقلیمی", "دسته بندی اقلیمی", "ایستگاه", "wmo", "منطقه اقلیمی")
EXPLANATORY = (
    "تعریف",
    "منظور",
    "چرا",
    "چگونه",
    "توضیح",
    "هدف",
    "الزامات",
    "شرایط",
    "روش",
    "مفهوم",
    "یعنی",
    "معنا",
    "معنی",
    "کاربرد",
    "مسئولیت",
    "وظیفه",
    "محاسبه",
)
MIN_COVERAGE = 0.6
_TABLE_REF = re.compile(r"جدول\s*(?:شماره\s*)?(19-\d+-\d+|پ\s?\d+-\d+)")
_SECTION_REF = re.compile(r"(?<![\d-])(19(?:-\d+){1,4})(?![\d-])")
_CLAUSE_SPLIT = re.compile(r"\s+و\s+|\s*[،,؛;?؟!]\s*")
APPENDIX_TABLES = {"پ2": APP02, "پ3": APP03, "پ7": APP07}


class Router:
    """Deterministic router; see the module docstring for the rules."""

    def __init__(self, store: StructuredStore, vocabulary: set[str]) -> None:
        self.store = store
        self.vocabulary = vocabulary
        labels = [*store.places, *(o["label"] for o in store.occupancy_index)]
        self.structured_vocabulary = (
            {t for label in labels for t in index_terms(label)}
            | {t for names in store.material_terms for t in names}
            | {t for table in store.main.values() for t in table.index.vocabulary()}
        )

    # ------------------------------------------------------------------ public
    def route(self, query: str) -> RouteDecision:
        norm = normalize_query(query)
        refs = _TABLE_REF.findall(norm)
        section_refs = [s for s in _SECTION_REF.findall(norm) if s not in set(refs)]
        signals: dict[str, Any] = {"table_ref": refs, "section_ref": section_refs}
        blocked = gate(norm)
        if blocked and not refs and not section_refs:
            signals["gate"] = {"category": blocked.category, "matched": blocked.matched}
            return RouteDecision(
                "unsupported",
                [f"answerability gate: {blocked.reason} ({blocked.matched})"],
                signals,
            )
        structured_part, prose_part = self.decompose(norm)
        tables, reasons, unknown = self._structured_signals(structured_part, refs)
        content = content_tokens(norm)
        known = [t for t in content if t in self.vocabulary]
        coverage = len(known) / len(content) if content else 0.0
        signals |= {
            "tables": tables,
            "unknown_entities": unknown,
            "coverage": round(coverage, 2),
            "structured_part": structured_part,
            "prose_part": prose_part,
        }
        # an unresolved entity blocks a lookup request only; a query that also asks for an
        # explanation still goes to prose retrieval
        if unknown and not tables and not refs and not section_refs and not prose_part:
            why = "; ".join(f"unknown {u['type']} {u['tokens']}" for u in unknown)
            return RouteDecision("unsupported", [f"unresolved entity: {why}"], signals)
        if tables and prose_part:
            return RouteDecision(
                "mixed",
                [*reasons, f"prose part «{prose_part}»"],
                signals,
                tables,
                structured_part,
                prose_part,
            )
        if tables:
            return RouteDecision("structured", reasons, signals, tables, structured_part, None)
        return _prose_or_fallback(norm, section_refs, coverage, signals)

    def decompose(self, norm: str) -> tuple[str, str]:
        """(structured part, prose part) of a normalized query; the prose part is empty when no
        clause asks for an explanation."""
        clauses = [c for c in _CLAUSE_SPLIT.split(norm) if c.strip()]
        kinds: list[str | None] = []
        for c in clauses:
            asks = bool(has_words(c, EXPLANATORY))
            looks_up = self._has_structured_signal(c)
            kinds.append(
                "both"
                if asks and looks_up
                else "prose"
                if asks
                else "structured"
                if looks_up
                else None
            )
        filled = list(zip(clauses, _fill(kinds), strict=True))
        # purely structured clauses carry the lookup; clauses that also ask for an explanation
        # are used for the lookup only when no purely structured clause exists
        lookup = " و ".join(c for c, k in filled if k == "structured") or " و ".join(
            c for c, k in filled if k == "both"
        )
        prose = " و ".join(c for c, k in filled if k in ("prose", "both"))
        if not prose:
            return norm, ""
        return lookup or norm, prose

    # ------------------------------------------------------------------ signals
    def _has_structured_signal(self, text: str) -> bool:
        if _TABLE_REF.search(text) or climate_codes(text) or re.search(r"(?<!\d)\d{6}(?!\d)", text):
            return True
        if self.store.find_places(text) or self.store.occupancies_in(text):
            return True
        if self.store.resolve_material(text):
            return True
        return any(has_words(text, words) for words in TABLE_TOPICS.values())

    def _structured_signals(
        self, part: str, refs: list[str]
    ) -> tuple[list[str], list[str], list[dict[str, Any]]]:
        tables: list[str] = []
        reasons: list[str] = []
        unknown: list[dict[str, Any]] = []
        for raw in refs:
            ref = raw.replace(" ", "")
            if ref.startswith("پ"):
                tables.append(APPENDIX_TABLES.get(ref.split("-")[0], ""))
            elif MAIN + ref in self.store.main:
                tables.append(MAIN + ref)
        tables = [t for t in tables if t]
        if refs:
            reasons.append(f"table reference {refs}")
        if re.findall(r"(?<!\d)\d{6}(?!\d)", part):
            tables.append(MAIN_3_2)
            reasons.append("6-digit station code")
        self._climate(part, tables, reasons, unknown)
        for table, words in TABLE_TOPICS.items():
            hit = has_words(part, words)
            if hit:
                tables.append(table)
                reasons.append(f"{table.split('.')[-1]} vocabulary {hit}")
        tables = self._require_entities(part, list(dict.fromkeys(tables)), reasons, unknown)
        return tables, reasons, unknown

    def _climate(
        self, part: str, tables: list[str], reasons: list[str], unknown: list[dict[str, Any]]
    ) -> None:
        intent = has_words(part, CLIMATE_INTENT)
        places = self.store.find_places(part)
        if places and intent:
            tables += [APP02, MAIN_3_2]
            reasons.append(f"known place {places} with climate intent {intent}")
            ambiguous = self.store.ambiguous_places(places)
            if ambiguous:
                reasons.append(f"place resolves to several stations {ambiguous}")
        elif places:
            reasons.append(f"place {places} without climate intent: no climate lookup")
        elif intent and not climate_codes(part):
            residual = self._unknown_tokens(part)
            if residual:
                unknown.append({"type": "place", "tokens": residual, "dataset": APP02})

    def _require_entities(
        self, part: str, tables: list[str], reasons: list[str], unknown: list[dict[str, Any]]
    ) -> list[str]:
        """Appendix 3 needs a curated occupancy, Appendix 7 a curated material."""
        out = []
        for t in tables:
            if t == APP03 and not self.store.occupancies_in(part):
                residual = self._unknown_tokens(part)
                if residual:
                    unknown.append({"type": "occupancy", "tokens": residual, "dataset": APP03})
                reasons.append("Appendix 3 vocabulary without a curated occupancy: not a lookup")
            elif t == APP07:
                rows = self.store.resolve_material(part)
                if rows:
                    out.append(t)
                    continue
                if rows == []:
                    unknown.append(
                        {
                            "type": "material",
                            "tokens": self.store.material_terms_of(part),
                            "dataset": APP07,
                        }
                    )
                reasons.append("Appendix 7 vocabulary without a curated material: not a lookup")
            else:
                out.append(t)
        return out

    def _unknown_tokens(self, part: str) -> list[str]:
        """Content words found neither in the corpus nor in any structured label."""
        return [
            t
            for t in content_tokens(part)
            if t not in self.vocabulary
            and t not in self.structured_vocabulary
            and not t.replace(".", "").isdigit()
            and not has_words(t, EXPLANATORY)
        ]


def _prose_or_fallback(
    norm: str, section_refs: list[str], coverage: float, signals: dict[str, Any]
) -> RouteDecision:
    """Rules 6-7: section reference -> prose; low corpus coverage -> unsupported; else prose."""
    if section_refs:
        return RouteDecision(
            "prose", [f"section reference {section_refs}"], signals, prose_query=norm
        )
    if coverage < MIN_COVERAGE:
        why = f"fallback: corpus coverage {coverage:.2f} < {MIN_COVERAGE}, no domain signal"
        return RouteDecision("unsupported", [why], signals)
    why = f"no structured signal; corpus vocabulary covered ({coverage:.2f})"
    return RouteDecision("prose", [why], signals, prose_query=norm)


def _fill(kinds: list[str | None]) -> list[str | None]:
    """Clauses without a signal join the previous clause (or the next one at the start)."""
    out = list(kinds)
    for i, k in enumerate(out):
        if k is None and i > 0:
            out[i] = out[i - 1]
    for i in range(len(out) - 2, -1, -1):
        if out[i] is None:
            out[i] = out[i + 1]
    return out
