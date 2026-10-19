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
7. otherwise the scope map decides (:mod:`mabhas19.retrieval.scope`): an external domain ->
   unsupported; supported or uncertain -> prose.

Explicit section/table/appendix references are validated against the scope map first; a query
whose only references do not exist is ``unsupported`` (``invalid_reference``). Every decision
records its scope class and reason in ``signals["scope"]``.
"""

from __future__ import annotations

import re
from typing import Any

from mabhas19.retrieval.answerability import gate, has_words
from mabhas19.retrieval.dimensions import (
    area_requested,
    density_requested,
    groups_requested,
    ratings_requested,
    years_requested,
)
from mabhas19.retrieval.entities import FUNCTION_WORDS, best_matches, etokens, keys
from mabhas19.retrieval.models import RouteDecision
from mabhas19.retrieval.normalization import (
    QUESTION_WORDS,
    content_tokens,
    index_terms,
    normalize_query,
)
from mabhas19.retrieval.scope import ScopeMap
from mabhas19.retrieval.structured import (
    APP02,
    APP03,
    APP07,
    MAIN,
    MAIN_3_2,
    REQUEST_WORDS,
    T_DEGREE_DAYS,
    T_R_RATING,
    MainTable,
    StructuredStore,
    climate_codes,
    degree_days_in,
    fully_named,
    r_value_in,
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
    APP03: ("شدت مصرف انرژی", "شدت مصرف", "eui", "kwh"),
    APP07: ("ضریب هدایت", "هدایت حرارتی", "چگالی", "جرم مخصوص"),
}
CLIMATE_INTENT = ("اقلیم", "رده اقلیمی", "دسته بندی اقلیمی", "ایستگاه", "wmo", "منطقه اقلیمی")
EXPLANATORY = (
    "تعریف",
    "منظور",
    "چرا",
    "چگونه",
    "چطور",
    "چطوری",
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
# words asking for a printed value (a lookup needs one unless a strong dimension is stated)
VALUE_REQUEST = (
    "چقدر",
    "چند",
    "حداقل",
    "حداکثر",
    "مقدار",
    "میزان",
    "درصد",
    "عدد",
    "سقف",
    "حد",
    "کد",
    "وضعیت",
    "علامت",
    "رتبه",
    "بازده",
    "بازدهی",
    "دما",
    "دمای",
    "مقاومت",
    "ضریب",
    "نسبت",
)
# units and measure words: part of any value request, never prose content of their own
UNIT_WORDS = frozenset(
    {
        "متر",
        "مربع",
        "مترمربع",
        "مکعب",
        "سالانه",
        "ازای",
        "وات",
        "کیلووات",
        "ساعت",
        "درجه",
        "سانتیگراد",
        "درصد",
        "کیلوگرم",
        "واحد",
    }
)
# requests to compute a value the source does not print (interpolation, extrapolation,
# averages of printed values): part of the lookup, always answered «not printed»
DERIVATION_REQUEST = (
    "درون یابی",
    "درونیابی",
    "برون یابی",
    "برونیابی",
    "میانگین",
    "interpolate",
    "extrapolate",
    "average",
)
# requests for an attribute printed only in table 19-3-2
STATION_CODE_REQUEST = ("کد ایستگاه", "wmo", "شماره ایستگاه")
# physical quantities a question may ask for; a table that prints none of the quantities named
# in the query is not the table asked about (e.g. a temperature difference is not an
# efficiency table's value)
QUANTITY_WORDS = frozenset(
    {
        "دما",
        "دمای",
        "درجه",
        "فشار",
        "قطر",
        "ضخامت",
        "سرعت",
        "دبی",
        "ابعاد",
        "طول",
        "ارتفاع",
        "ظرفیت",
        "تعداد",
        "وزن",
        "قیمت",
        "هزینه",
    }
)
T_19_2_1 = MAIN + "19-2-1"
T_19_5_6 = MAIN + "19-5-6"
_TABLE_REF = re.compile(r"جدول\s*(?:شماره\s*)?(19-\d+-\d+|پ\s?\d+-\d+)")
_SECTION_REF = re.compile(r"(?<![\d-])(19(?:-\d+){1,4})(?![\d-])")
_APPENDIX_REF = re.compile(r"پیوست\s*(?:شماره\s*)?(\d+)(?![\d-])")
_CLAUSE_SPLIT = re.compile(r"\s+و\s+|\s*[،,؛;?؟!]\s*")
APPENDIX_TABLES = {"پ2": APP02, "پ3": APP03, "پ7": APP07}


class Router:
    """Deterministic router; see the module docstring for the rules."""

    def __init__(
        self, store: StructuredStore, vocabulary: set[str], scope: ScopeMap | None = None
    ) -> None:
        self.store = store
        self.vocabulary = vocabulary
        self.scope = scope or ScopeMap.load()  # identifiers, concepts, external domains
        store.resolver.vocabulary = vocabulary  # lets the resolver tell unknown words apart
        labels = [*store.places, *(o["label"] for o in store.occupancy_index)]
        self.structured_vocabulary = (
            {t for label in labels for t in index_terms(label)}
            | {t for names in store.material_terms for t in names}
            | {t for table in store.main.values() for t in table.index.vocabulary()}
        )

    # ------------------------------------------------------------------ public
    def route(self, query: str) -> RouteDecision:  # noqa: PLR0911
        norm = normalize_query(query)
        refs = _TABLE_REF.findall(norm)
        section_refs = [s for s in _SECTION_REF.findall(norm) if s not in set(refs)]
        # «پیوست ۳» names the appendix table itself when it is a curated dataset
        refs += [f"پ{n}" for n in _APPENDIX_REF.findall(norm) if f"پ{n}" in APPENDIX_TABLES]
        checks = self.scope.check_references(norm)
        invalid = [c for c in checks if c.status != "valid"]
        bad = {c.ref.replace(" ", "") for c in invalid}
        refs = [r for r in refs if r.replace(" ", "") not in bad]
        section_refs = [r for r in section_refs if r not in bad]
        signals: dict[str, Any] = {
            "table_ref": refs,
            "section_ref": section_refs,
            "references": [
                {"ref": c.ref, "kind": c.kind, "status": c.status, "detail": c.detail}
                for c in checks
            ],
        }
        if invalid and not any(c.status == "valid" for c in checks):
            why = "; ".join(f"{c.kind} {c.ref}: {c.status}" for c in invalid)
            signals["scope"] = {"class": "invalid_reference", "reason": why}
            return RouteDecision("unsupported", [f"invalid reference: {why}"], signals)
        blocked = gate(norm)
        if blocked and not refs and not section_refs:
            signals["gate"] = {"category": blocked.category, "matched": blocked.matched}
            signals["scope"] = {
                "class": "external_domain",
                "reason": f"answerability gate: {blocked.category}",
            }
            return RouteDecision(
                "unsupported",
                [f"answerability gate: {blocked.reason} ({blocked.matched})"],
                signals,
            )
        # pass 1: tables whose printed dimensions the whole query states; pass 2: the lookup
        # part after clause interpretation
        candidates, reasons, unknown = self._structured_signals(norm, refs, norm, section_refs)
        structured_part, prose_part = self.decompose(norm, candidates)
        tables = candidates
        if prose_part:
            # the lookup keeps the selected tables whose dimensions it states
            tables = [t for t in candidates if self._clause_dims(structured_part, [t], norm)]
            unknown = self._structured_signals(structured_part, refs, norm, section_refs)[2]
        named = self._referenced_datasets(refs)
        if named:
            # an explicitly named table scopes the lookup: other datasets contribute only when
            # they are named too (a derivation such as city -> climate class stays an input of
            # the named table's lookup, never an extra answer)
            dropped = [t for t in tables if t not in named]
            tables = [t for t in tables if t in named]
            unknown = [u for u in unknown if u["dataset"] in named]
            if dropped:
                reasons = [*reasons, f"explicit table reference; not queried: {dropped}"]
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
            signals["scope"] = {"class": "unknown_entity", "reason": why}
            return RouteDecision("unsupported", [f"unresolved entity: {why}"], signals)
        if tables and prose_part:
            signals["scope"] = {"class": "supported_mixed", "reason": "lookup and prose parts"}
            return RouteDecision(
                "mixed",
                [*reasons, f"prose part «{prose_part}»"],
                signals,
                tables,
                structured_part,
                prose_part,
            )
        if tables:
            signals["scope"] = {"class": "supported_structured", "reason": "; ".join(reasons)}
            return RouteDecision("structured", reasons, signals, tables, structured_part, None)
        if section_refs:
            signals["scope"] = {"class": "supported", "reason": f"section reference {section_refs}"}
        else:
            d = self.scope.classify_text(norm, self.vocabulary)
            signals["scope"] = {"class": d.scope, "reason": d.reason, "evidence": d.evidence}
            if d.scope == "external_domain":
                return RouteDecision("unsupported", [f"scope: {d.reason}"], signals)
        return _prose_or_fallback(norm, section_refs, coverage, signals)

    def _referenced_datasets(self, refs: list[str]) -> list[str]:
        """Curated datasets named by explicit table/appendix references."""
        out = []
        for raw in refs:
            ref = raw.replace(" ", "")
            ds = APPENDIX_TABLES.get(ref.split("-")[0]) if ref.startswith("پ") else MAIN + ref
            if ds and (ds in self.store.main or ds in self.store.metas):
                out.append(ds)
        return list(dict.fromkeys(out))

    def decompose(self, norm: str, tables: list[str] | None = None) -> tuple[str, str]:
        """(structured part, prose part) of a normalized query.

        Clauses are interpreted by table dimensions, not trigger words: a clause stating
        dimensions of a selected table is lookup-oriented; a clause with meaningful content
        that those dimensions do not explain (or an explanation request) is prose-oriented;
        a clause with neither joins its neighbour, so a lookup's constraints stay together.
        The prose part is empty when no clause is prose-oriented."""
        if tables is None:
            tables = self._structured_signals(norm, [], norm, [])[0]
        if not tables:
            return norm, ""
        explained = self._explained_keys(tables)
        table_words = self._explained_keys(tables, generic=False)
        clauses = _coordinated(_CLAUSE_SPLIT.split(norm))
        kinds: list[str | None] = []
        for c in clauses:
            dims = self._clause_dims(c, tables, norm, label_only_ok=len(clauses) == 1)
            strong = self._strong_clause_dims(c, tables)
            residual = [
                t
                for t in etokens(c)
                if not keys(t) & explained
                and t in self.vocabulary
                and not t.replace(".", "").isdigit()
                and not re.fullmatch(r"\d[a-z]", t)
            ]
            # explanation words printed in the selected tables («الزامات») are not requests
            asks = [
                w for w in has_words(c, EXPLANATORY) if not keys(normalize_query(w)) & table_words
            ]
            asks_prose = bool(asks and len(etokens(c)) > 1)
            if has_words(c, DERIVATION_REQUEST):
                kinds.append("structured")  # asks to derive a value: refused by the lookup
            elif dims and not strong and len(clauses) > 1 and len(residual) >= 3:
                # a clause tied to the table only by a topic word, label or entity, with its
                # own substantial content, asks about something else: prose
                kinds.append("prose")
            elif dims:
                kinds.append("both" if asks_prose else "structured")
            else:
                kinds.append("prose" if len(residual) >= 2 or asks_prose else None)
        filled = list(zip(clauses, _fill(kinds), strict=True))
        # purely structured clauses carry the lookup; clauses that also need prose are used
        # for the lookup only when no purely structured clause exists
        lookup = " و ".join(c for c, k in filled if k == "structured") or " و ".join(
            c for c, k in filled if k == "both"
        )
        prose_part = " و ".join(c for c, k in filled if k in ("prose", "both"))
        if not prose_part:
            return norm, ""
        return lookup or norm, prose_part

    def _explained_keys(self, tables: list[str], *, generic: bool = True) -> frozenset[str]:
        """Matching keys of words that a lookup on these tables explains: printed labels,
        captions, topic words and (generic) request/value/dimension words."""
        words: set[str] = set()
        if generic:
            words |= set(FUNCTION_WORDS) | set(QUESTION_WORDS) | set(REQUEST_WORDS)
            words |= {normalize_query(w) for w in VALUE_REQUEST} | UNIT_WORDS
            for phrase in CLIMATE_INTENT:
                words |= set(etokens(phrase))
        for t in tables:
            for phrase in TABLE_TOPICS.get(t, ()):
                words |= set(etokens(phrase))
            main = self.store.main.get(t)
            if main is not None:
                words |= set(main.caption_words)
                if main.labels is not None:
                    words |= main.labels.vocabulary
            if t == APP03:
                words |= self.store.resolver.occupancy_index.vocabulary
            if t == APP07:
                words |= self.store.resolver.material_index.vocabulary
            if t in (APP02, MAIN_3_2):
                words |= {w for p in self.store.places for w in p.split()}
        return frozenset(k for w in words for k in keys(w))

    def _strong_clause_dims(self, clause: str, tables: list[str]) -> bool:
        """A clause states a value-defining dimension (not just a topic word, label or entity):
        a table reference, station code, climate class, rating, group/year, degree days, R,
        area or density."""
        if _TABLE_REF.search(clause) or re.search(r"(?<!\d)\d{6}(?!\d)", clause):
            return True
        if any(APPENDIX_TABLES.get(f"پ{n}") in tables for n in _APPENDIX_REF.findall(clause)):
            return True
        if climate_codes(clause) or ratings_requested(clause):
            return True
        if {APP02, MAIN_3_2} & set(tables) and self.store.find_places(clause):
            return True
        if any(self._special_dims(t, clause) for t in tables):
            return True
        return APP07 in tables and density_requested(clause) is not None

    def _clause_dims(  # noqa: PLR0911
        self, clause: str, tables: list[str], norm: str, *, label_only_ok: bool = True
    ) -> bool:
        """True when the clause states a printed dimension of one of the selected tables.

        In a multi-clause query a clause whose only dimension is a mentioned row label and
        that carries its own content is not a lookup (``label_only_ok=False``): the label is
        what it talks about, not a value it asks for."""
        if _TABLE_REF.search(clause) or re.search(r"(?<!\d)\d{6}(?!\d)", clause):
            return True
        if any(APPENDIX_TABLES.get(f"پ{n}") in tables for n in _APPENDIX_REF.findall(clause)):
            return True
        codes = climate_codes(clause)
        for t in tables:
            main = self.store.main.get(t)
            if t in (APP02, MAIN_3_2) and self.store.find_places(clause):
                return True
            if t == APP03 and (
                codes
                or ratings_requested(clause)
                or (label_only_ok and self.store.occupancies_in(clause))
            ):
                return True
            # a material named in a clause is a lookup dimension when the clause asks for an
            # Appendix 7 property (or is the whole query), like a mentioned row label
            if (
                t == APP07
                and (label_only_ok or has_words(clause, TABLE_TOPICS[APP07]))
                and self.store.resolve_material(clause)
            ):
                return True
            if main is None:
                continue
            if codes and main.climate_domain:
                return True
            if self._special_dims(t, clause):
                return True
            topic = has_words(clause, TABLE_TOPICS.get(t, ()))
            value = has_words(clause, VALUE_REQUEST)
            if self._label_signal(main, clause)[0] and (label_only_ok or value):
                return True
            if label_only_ok and topic and value and main.labels is not None:
                return True  # a value of the table for a row it may not print
        return False

    # ------------------------------------------------------------------ signals
    def _special_dims(self, table: str, text: str) -> list[str]:
        """Dimensions that only one table prints (group/year, degree days, R, area)."""
        out = []
        if table == T_19_2_1 and (groups_requested(text) or years_requested(text)):
            out.append("group/year")
        if table == T_DEGREE_DAYS and degree_days_in(text):
            out.append("degree-day value")
        if table == T_R_RATING and r_value_in(text) is not None:
            out.append("R value")
        if table == T_R_RATING and ratings_requested(text):
            out.append("rating")
        if table == T_19_5_6 and area_requested(text):
            out.append("area")
        return out

    @staticmethod
    def _caption_quantity(table: MainTable) -> set[str]:
        """Content words of a table's caption (its measured quantity, e.g. «مقاومت حرارتی»,
        «r», «value»), without the request/generic words."""
        return {
            w
            for w in table.caption_words
            if w not in REQUEST_WORDS and w not in FUNCTION_WORDS and not w.isdigit() and len(w) > 0
        } - {"جدول", "19", "value"}

    @staticmethod
    def _row_fully_named(table: MainTable, part: str) -> bool:
        if table.labels is None:
            return False
        return any(fully_named(m) for m in table.labels.match(table.labels.query_tokens(part)))

    def _other_quantity(self, table: MainTable, part: str) -> bool:
        """True when the query names a physical quantity the table does not print for what is
        asked: not in its caption or topic words, nor in a row label the query names (a word in
        another row's condition, e.g. «دمای» of a CHP row, is not the asked row's quantity)."""
        asked = {t for t in etokens(part) if t in QUANTITY_WORDS}
        if not asked:
            return False
        printed = set(table.caption_words)
        if table.labels:
            best = best_matches(table.labels.match(table.labels.query_tokens(part)))
            printed |= {w for m in best for w in m.label.tokens}
        printed |= {w for phrase in TABLE_TOPICS.get(table.table_id, ()) for w in etokens(phrase)}
        printed_keys = {k for w in printed for k in keys(w)}
        return not any(keys(a) & printed_keys for a in asked)

    @staticmethod
    def _label_signal(table: MainTable, text: str) -> tuple[bool, bool]:
        """(a printed row label is named, it is named strongly: >= 2 of its words with a fully
        matched printed level)."""
        if table.labels is None:
            return False, False
        best = best_matches(table.labels.match(table.labels.query_tokens(text)))
        if not best:
            return False, False
        strong = any(
            (
                (n := len({m.label.tokens[i] for i in m.label_idx})) >= 2
                and max(m.specificity) >= 0.75
            )
            or (n >= 4 and max(m.specificity) >= 0.5)
            for m in best
        )
        return True, strong

    def _main_tables(self, part: str, section_refs: list[str]) -> tuple[list[str], list[str]]:
        """Main tables nominated by topic words, a strong row label or a table-specific
        dimension, and selected only when one of their dimensions is stated and a value is
        asked for (or a strong dimension is stated). An explicit section reference wins over
        topic vocabulary."""
        codes = climate_codes(part)
        value = bool(has_words(part, VALUE_REQUEST))
        out: list[str] = []
        reasons: list[str] = []
        for tid, t in self.store.main.items():
            topic = has_words(part, TABLE_TOPICS.get(tid, ()))
            label, strong_label = self._label_signal(t, part)
            special = self._special_dims(tid, part)
            climate = bool(codes and t.climate_domain)
            # only a value printed in one table alone nominates it (group/year, degree days,
            # R); area and rating are dimensions but other tables print them too
            nominating = [d for d in special if d not in ("area", "rating")]
            # a weakly named row plus a climate class plus the table's own measured quantity
            # (from its caption, e.g. «R-Value», «مقاومت حرارتی») names the table too
            quantity = bool(set(etokens(part)) & self._caption_quantity(t))
            weak_named = label and climate and quantity and self._row_fully_named(t, part)
            if not (topic or strong_label or nominating or weak_named):
                continue
            if self._other_quantity(t, part):
                continue  # asks for a quantity this table does not print
            if tid == T_19_2_1 and not special:
                continue  # its values are per building group and year
            strong = climate or bool(special)
            # a one-clause definition question about a table concept is prose
            asks = [
                w
                for w in has_words(part, EXPLANATORY)
                if not keys(normalize_query(w)) & self._explained_keys([tid], generic=False)
            ]
            if asks and not strong and len(_CLAUSE_SPLIT.split(part.strip(" ؟?"))) == 1:
                continue
            row_request = topic and value and t.labels is not None  # a row the table may lack
            if not (label or climate or special or row_request):
                continue
            if not (value or strong):
                continue
            if section_refs and not strong:
                continue
            out.append(tid)
            why = [f"topic {topic}"] if topic else []
            why += ["row label"] if label else []
            why += ["climate class"] if climate else []
            why += special
            reasons.append(f"{tid.split('.')[-1]}: {', '.join(why)}")
        return out, reasons

    def _structured_signals(
        self,
        part: str,
        refs: list[str],
        query: str | None = None,
        section_refs: list[str] | None = None,
    ) -> tuple[list[str], list[str], list[dict[str, Any]]]:
        query = query if query is not None else part
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
        self._climate(part, tables, reasons, unknown, query)
        main, main_reasons = self._main_tables(part, section_refs or [])
        tables += main
        reasons += main_reasons
        for table in (APP03, APP07):
            hit = has_words(part, TABLE_TOPICS[table])
            if (
                table == APP03
                and not hit
                and ratings_requested(part)
                and self.store.occupancies_in(part)
            ):
                hit = ["occupancy + rating"]
            if hit:
                tables.append(table)
                reasons.append(f"{table.split('.')[-1]} {hit}")
        tables = self._require_entities(part, list(dict.fromkeys(tables)), reasons, unknown)
        return tables, reasons, unknown

    def _climate(
        self,
        part: str,
        tables: list[str],
        reasons: list[str],
        unknown: list[dict[str, Any]],
        query: str | None = None,
    ) -> None:
        # climate intent may sit in another clause of the same query («… در قوچان؛ اقلیمش چیه»)
        intent = has_words(query if query is not None else part, CLIMATE_INTENT)
        places = self.store.find_places(part)
        if places and intent and has_words(part, STATION_CODE_REQUEST):
            # the station (WMO) code is printed only in table 19-3-2, never in Appendix 2
            tables.append(MAIN_3_2)
            reasons.append(f"known place {places}; station code is printed in 19-3-2 only")
        elif places and intent:
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
            if t == APP03 and not self.store.occupancies_in(part) and climate_codes(part):
                out.append(t)
                reasons.append("Appendix 3 with a stated climate class: occupancy is validated")
            elif t == APP03 and not self.store.occupancies_in(part):
                residual = self._unknown_tokens(part)
                if residual:
                    unknown.append({"type": "occupancy", "tokens": residual, "dataset": APP03})
                reasons.append("Appendix 3 vocabulary without a curated occupancy: not a lookup")
            elif t == APP07:
                rows = self._clause_material(part)
                if rows:
                    out.append(t)
                    continue
                if rows == [] and "پ7" in self._appendix_refs(part):
                    # explicitly asks Appendix 7 for a material it does not print: a lookup
                    # that ends in a typed no-match, never a prose guess
                    out.append(t)
                    reasons.append("Appendix 7 named explicitly: unprinted material is a no-match")
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

    @staticmethod
    def _appendix_refs(part: str) -> set[str]:
        return {f"پ{n}" for n in _APPENDIX_REF.findall(part)}

    def _clause_material(self, part: str) -> list[int] | None:
        """Material rows named in the clause that asks for an Appendix 7 property (the whole
        part when no single clause does)."""
        whole = self.store.resolve_material(part)
        if whole:
            return whole
        for c in _CLAUSE_SPLIT.split(part):
            if has_words(c, TABLE_TOPICS[APP07]):
                rows = self.store.resolve_material(c)
                if rows:
                    return rows
        return whole

    def _unknown_tokens(self, part: str) -> list[str]:
        """Content words found neither in the corpus nor in any structured label."""
        known = self.vocabulary | self.structured_vocabulary
        return [
            t
            for t in etokens(part)
            if t not in FUNCTION_WORDS
            and t not in QUESTION_WORDS
            and not keys(t) & known
            and not t.replace(".", "").isdigit()
            and not has_words(t, EXPLANATORY)
        ]


def _prose_or_fallback(
    norm: str, section_refs: list[str], coverage: float, signals: dict[str, Any]
) -> RouteDecision:
    """Section reference -> prose; otherwise prose. Corpus coverage is recorded as diagnostic
    metadata only: unsupported needs positive evidence (gate category, unresolved entity,
    unknown reference), never low lexical coverage."""
    if section_refs:
        return RouteDecision(
            "prose", [f"section reference {section_refs}"], signals, prose_query=norm
        )
    note = " (low corpus coverage, diagnostic only)" if coverage < MIN_COVERAGE else ""
    why = f"no structured signal; corpus vocabulary covered ({coverage:.2f}){note}"
    return RouteDecision("prose", [why], signals, prose_query=norm)


_VALUE_TAIL = re.compile(r"(?:\d[a-z]?|[a-z]\+*|متر\s?مربع|مترمربع|متر)$")
_VALUE_HEAD = re.compile(
    r"^(?:\d|[a-z]\b|بیشتر|کمتر|بالاتر|پایین\s?تر|بزرگ\s?تر|کوچک\s?تر|به\s?بالا|به\s?پایین)"
)


def _coordinated(parts: list[str]) -> list[str]:
    """Clauses after splitting at «و»/punctuation, re-joining coordinated values: a clause that
    ends with a value (climate class, number, rating) and one that starts with a value were one
    constraint («اقلیم‌های 2a و 2b»)."""
    out: list[str] = []
    for p in (x.strip() for x in parts):
        if not p:
            continue
        if out and _VALUE_TAIL.search(out[-1]) and _VALUE_HEAD.search(p):
            out[-1] = f"{out[-1]} و {p}"
        else:
            out.append(p)
    return out


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
