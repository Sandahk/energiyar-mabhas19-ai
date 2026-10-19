"""Deterministic, auditable query router (v4): structured | prose | mixed | unsupported.

No LLM. Every decision lists the signals it saw and the rule that fired. Structured detection
is *plan-driven*: the router never decides from topic trigger words. It dry-runs the query
planner (:mod:`mabhas19.retrieval.planner`) against the plausible candidate tables and reads the
typed plan and its validation outcome (:func:`mabhas19.retrieval.plan.validate`). The router
never executes a plan and never returns records; only the executor does.

Order of rules (first match wins):

1. **references** (scope map): a query whose only references do not exist is ``unsupported``
   (``invalid_reference``). Table references are resolved from the canonical schema: a printed
   table ID anywhere in a sentence that names a table («جدول … ۱۹-۵-۳»), or a caption/title that
   identifies exactly one table («جدول زمان‌بندی»); an ambiguous caption is recorded as a typed
   ambiguity and never guessed;
2. **intent gate** (:mod:`mabhas19.retrieval.answerability`): live information, commercial,
   service and education requests -> unsupported (unless a section or table is named);
3. **plan-driven structured detection**, clause by clause: candidate tables are those sharing
   vocabulary (printed labels, captions, outputs, units) with the clause; each is dry-run
   planned. A table *qualifies* when its plan has a concrete subject (a resolved entity or an
   applied/checked/invalid constraint), explains at least one content word or declared unit,
   asks for the table's output (named output or value request), and is valid or ends in a typed
   outcome (missing dimension, invalid constraint, ambiguity, unsupported calculation…). Strong
   anchors (an explicit reference; a named output with a printed place or a stated constraint;
   a valid plan with stated constraints and a value request or resolved entity) qualify alone;
   any other anchor needs the plan to explain most of the clause, so a word shared by chance
   never makes a lookup. A table whose output is not named loses to one whose output is; a
   table whose explained words are a strict subset of another's is not what is asked; a clause
   asking for a quantity the table does not print, or for one of its *input* dimensions («چه
   رده‌ای» of Appendix 3), does not anchor it;
4. **decomposition by plan coverage**: anchored clauses form the structured part (and also the
   prose part when they carry an explanation form - یعنی، منظور، چرا، چطور، جزو … حساب…; an
   anchored definition question without a stated constraint is prose only). A clause that no
   plan anchors is prose when it has a question form (an interrogative, or it closes a «؟»
   sentence) with content no selected table explains, a conceptual word, or three unexplained
   content words; purpose remarks («برای انتخاب عایق می‌خوام»), calculation requests and context
   clauses join their neighbour, context clauses stating constraints join the lookup, and
   coordinated values stay together. An explicitly referenced table scopes its own sentence;
5. **unknown entities** are decided by the planner: a plan whose only failure is an entity that
   no curated record prints is a typed outcome; with a stated constraint it is a structured
   lookup ending in ``unknown_entity``; without one, a request for a named table output is
   ``unsupported`` (``unknown_entity``, typed per candidate table) only when the unresolved
   words are not Mabhas 19 vocabulary (also in spoken form) - a familiar word in a conceptual
   question goes to prose;
6. structured + prose parts -> mixed; structured only -> structured; section reference -> prose;
7. otherwise the scope map decides (:mod:`mabhas19.retrieval.scope`), with **in-scope evidence
   outranking external-domain evidence**: a domain-gate or external-taxonomy match is
   ``unsupported`` only when it outweighs the strong Mabhas 19 evidence of the query (defined
   terms - also written joined - and concept phrases of titles, captions and table rows);
   otherwise prose. Low corpus coverage is diagnostic only, never a reason for unsupported.

Every decision records its scope class and reason in ``signals["scope"]``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from mabhas19.retrieval.answerability import gate, has_words
from mabhas19.retrieval.entities import FUNCTION_WORDS, NEGATORS, etokens, keys
from mabhas19.retrieval.models import RouteDecision
from mabhas19.retrieval.normalization import QUESTION_WORDS, content_tokens, normalize_query
from mabhas19.retrieval.numeric import UNITS
from mabhas19.retrieval.plan import PlanOutcome, StructuredQueryPlan, validate
from mabhas19.retrieval.planner import QUANTITY_WORDS as PLANNER_QUANTITIES
from mabhas19.retrieval.planner import Planner, features
from mabhas19.retrieval.schema import TableSchema
from mabhas19.retrieval.scope import ScopeMap
from mabhas19.retrieval.structured import (
    APP02,
    APP03,
    APP07,
    MAIN,
    MAIN_3_2,
    REQUEST_WORDS,
    StructuredStore,
)

MIN_COVERAGE = 0.6
# explanation forms: a clause carrying one asks for an explanation, whatever else it states
EXPLANATION = (
    "یعنی",
    "منظور",
    "چرا",
    "چطور",
    "چطوری",
    "چگونه",
    "توضیح",
    "مفهوم",
    "معنا",
    "معنی",
    "تعریف",
    "جزو",
)
# «با چه … تعریف شده/کرده»: how a table defines a value, not a request for a definition
_DEFINED_AS = re.compile(r"تعریف\s*(?:شده|کرده|کرد|میشه|می\s*شود|میشود|شدن|شدند)")
# weaker conceptual wording: makes an unanchored clause prose, never splits an anchored one
CONCEPTUAL = (
    "هدف",
    "الزامات",
    "شرایط",
    "روش",
    "کاربرد",
    "مسئولیت",
    "وظیفه",
    "محاسبه",
    "لازمه",
    "اجباریه",
)
# interrogatives that open a question of their own
INTERROGATIVES = frozenset(
    {
        "کجا",
        "کجاها",
        "چرا",
        "چطور",
        "چطوری",
        "چگونه",
        "آیا",
        "ایا",
        "چی",
        "چه",
        "چند",
        "چقدر",
        "چقدره",
        "چنده",
        "کدام",
        "کدوم",
        "کی",
    }
)
_INTERROGATIVE_KEYS = frozenset(k for w in INTERROGATIVES for k in keys(w)) | {"کجای"}
# words asking for a printed value
VALUE_REQUEST = (
    "چقدر",
    "چقدره",
    "چند",
    "چنده",
    "حداقل",
    "حداکثر",
    "مقدار",
    "میزان",
    "سقف",
    "حد",
    "کد",
    "وضعیت",
    "علامت",
    "رتبه",
    "کدام",
    "کدوم",
)
# measured quantities a question may ask for; a table must print the one asked
QUANTITY_WORDS = PLANNER_QUANTITIES | {"ظرفیت"}
# a first-person purpose remark («برای انتخاب عایق می‌خوام بدونم») is context, not a request
_PURPOSE = re.compile(r"(?:^|\s)برای\s.*(?:می\s?خوام|می\s?خواهم|نیاز\s?دارم|لازم\s?دارم)|^برای\s")
# «جزو کدام رده» asks a value; «جزو … است/حساب میشه» asks a classification
_WHICH_CLASS = re.compile(r"جزو\s*(?:کدام|کدوم|چه)")
_TABLE_WORD = re.compile(r"(?<![ؠ-ۿ])(?:جدول|جداول)")
_TABLE_ID = re.compile(r"(?<![\d-])(19\s?-\s?\d+\s?-\s?\d+|پ\s?\d+\s?-\s?\d+)(?![\d-])")
_TABLE_REF = re.compile(r"جدول\s*(?:شماره\s*)?(19-\d+-\d+|پ\s?\d+-\d+)")
_SECTION_REF = re.compile(r"(?<![\d-])(19(?:-\d+){1,4})(?![\d-])")
_APPENDIX_REF = re.compile(r"پیوست\s*(?:شماره\s*)?(\d+)(?![\d-])")
_CLAUSE_SPLIT = re.compile(r"\s+و\s+|\s*[،,]\s*")
# spans the planner reads as constraints (never content words of their own)
_CONSTRAINT_SPAN = re.compile(
    r"سال\s*(?:اول|یکم|دوم|سوم|چهارم|پنجم|ششم|هفتم|\d+)(?![ؠ-ۿ\d])"
    r"|(?<![a-z0-9])\d\s?[a-z](?![a-z0-9])"
    r"|(?:رده|رتبه|کلاس)\s*(?:بازدهی\s*)?(?:انرژی\s*)?[a-z]\+{0,3}(?![a-z0-9+])"
    r"|گروه\s*(?:های\s*)?(?:ساختمانی\s*)?[«\"'(]?\s*(?:الف|ب|پ|ج|د|ه|و|ی)\s*[»\"')]?(?![ؠ-ۿ])"
    r"|\d+(?:[.٫/]\d+)?"
)
APPENDIX_TABLES = {"پ2": APP02, "پ3": APP03, "پ7": APP07}
_APPENDIX_NUMBERS = {APP02: 2, APP03: 3, APP07: 7}
# auxiliaries, colloquial verbs and fillers: never content of their own
_FILLERS = frozenset(
    {
        "شدن",
        "شده",
        "شد",
        "شدند",
        "میشه",
        "میشن",
        "بشه",
        "بشن",
        "میشود",
        "شود",
        "کرد",
        "کرده",
        "کردن",
        "کنه",
        "کنن",
        "کنم",
        "کنیم",
        "باشه",
        "باشن",
        "باشد",
        "باشند",
        "داره",
        "دارد",
        "دارن",
        "دارند",
        "داشته",
        "داشت",
        "داریم",
        "میتونه",
        "تونه",
        "بتونه",
        "گرفت",
        "گرفته",
        "میگیره",
        "گیره",
        "اومده",
        "امده",
        "گذاشته",
        "گوید",
        "میگوید",
        "میگه",
        "گفته",
        "بیان",
        "انرژی",
        "جزو",
        "بند",
        "فصل",
        "بخش",
        "ماده",
        "تبصره",
        "پیوست",
        "yr",
        "قرار",
        "گیرد",
        "میگیرد",
        "ساله",
        "سالانه",
        "بگیره",
        "نباید",
        "خب",
        "حالا",
        "اصلا",
        "اصلاً",
        "دقیقا",
        "مثلا",
        "واقعا",
        "لطفا",
        "خیلی",
        "نظر",
        "یه",
        "یک",
        "تو",
        "توی",
        "واسه",
        "همون",
        "همین",
        "این",
        "اون",
        "آن",
        "چنین",
        "هست",
        "هستش",
        "است",
        "ست",
        "نیست",
        "بود",
        "هستند",
        "بگو",
        "بده",
        "بدونم",
        "میخوام",
        "خوام",
        "مبحث",
        "مبحث19",
        "نوزده",
        "نوزدهم",
        "مقررات",
        "ملی",
        "طبق",
        "مطابق",
        "اساس",
        "بر",
        "لازم",
        "باید",
        "مجاز",
    }
)
# comparison words: never what a table lookup is about, but part of a question of its own
_COMPARISON = ("کم", "زیاد", "بیشتر", "کمتر", "بیش", "بالا", "بالاتر", "پایین", "پایینتر")
# words that never name what a table lookup is about
GENERIC_KEYS = frozenset(
    k
    for w in (
        *(t for forms in UNITS.values() for f in forms for t in etokens(f)),
        *FUNCTION_WORDS,
        *QUESTION_WORDS,
        *REQUEST_WORDS,
        *VALUE_REQUEST,
        *INTERROGATIVES,
        *_FILLERS,
        *_COMPARISON,
    )
    for k in keys(w)
)
# units declared by a table schema (a stated unit points to the tables that print it)
_DECLARED_UNITS = ("kwh", "m2", "m2k/w", "w/mk", "kg/m3", "celsius")
# words that never count as unexplained content of a clause
PROSE_GENERIC_KEYS = frozenset(
    k
    for w in (
        *(t for forms in UNITS.values() for f in forms for t in etokens(f)),
        *FUNCTION_WORDS,
        *QUESTION_WORDS,
        *INTERROGATIVES,
        *_FILLERS,
    )
    for k in keys(w)
)


@dataclass
class Candidate:
    """One candidate table's dry-run plan and the evidence that anchors it to a clause."""

    dataset: str
    plan: StructuredQueryPlan
    outcome: PlanOutcome | None
    explicit: bool = False
    named: bool = False  # an output of the table is named (not a stated input)
    value: bool = False  # a value is requested
    subject: bool = False  # resolved entity or applied/checked/invalid constraint
    entity: bool = False  # the subject includes a resolved or ambiguous entity
    concrete: bool = False  # a stated constraint besides the entity (class, rating, number…)
    covered: bool = False  # every content word is explained by the table
    foreign: list[str] = field(default_factory=list)  # asked quantities the table lacks
    asks_input: list[str] = field(default_factory=list)  # «چه رده‌ای» of an input dimension
    explained: frozenset[int] = frozenset()
    items: int = 0  # content words and stated constraints of the text
    resolved: bool = False  # a uniquely resolved entity
    place: bool = False  # a printed place named exactly
    words_explained: bool = False  # the table explains a content word (not only constraints)
    majority: bool = False
    unknown_words: list[str] = field(default_factory=list)
    unfamiliar: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return self.outcome is None

    @property
    def unknown_entity(self) -> bool:
        return self.outcome is not None and self.outcome.code == "unknown_entity"

    def summary(self) -> dict[str, Any]:
        return {
            "table": self.dataset,
            "outcome": self.outcome.code if self.outcome else "valid",
            "evidence": [
                n
                for n, v in (
                    ("explicit", self.explicit),
                    ("output named", self.named),
                    ("value request", self.value),
                    ("entity", self.entity),
                    ("constraint", self.concrete),
                    ("all words explained", self.covered),
                    ("most words explained", self.majority),
                )
                if v
            ],
        }


class Router:
    """Deterministic router; see the module docstring for the rules."""

    def __init__(
        self, store: StructuredStore, vocabulary: set[str], scope: ScopeMap | None = None
    ) -> None:
        self.store = store
        self.vocabulary = vocabulary
        self.scope = scope or ScopeMap.load()  # identifiers, concepts, external domains
        store.resolver.vocabulary = vocabulary  # lets the planner tell unknown words apart
        self.planner = Planner(store, store.schemas)
        self.schemas = store.schemas
        self._cache: dict[tuple[Any, ...], list[Candidate]] = {}
        self._summaries: list[dict[str, Any]] = []
        self._section_scope = False
        self._selected: list[str] = []
        self._single_clause = True
        self.vocab_keys = frozenset(k for w in vocabulary for k in keys(w))
        place_words = {w for p in store.places for w in etokens(p)}
        province_words = {w for p in self.planner.provinces for w in etokens(p)}
        # printed climate class names («فوق‌العاده گرم و کم باران») name the climate dimension
        climate_names = {
            w
            for r in store.main[MAIN + "19-3-1"].records
            for x in (r.get("climate_names") or {}).get("by_class") or []
            for w in etokens(x["name"])
        }
        # words a lookup on each table explains: its schema vocabulary, caption/title, printed
        # row labels and entity names (also written joined), units and climate class names
        self.table_keys: dict[str, frozenset[str]] = {}
        for ds, schema in self.schemas.items():
            texts = [self._title(ds), *self._printed_texts(ds)]
            texts += [n for d in schema.dimensions for n in d.nouns]
            texts += [form for u in _units(schema) for form in UNITS.get(u, ())]
            if schema.family == "place" or ds == APP03:
                texts += sorted(place_words | province_words)
                texts += ["شهر شهرستان استان جزیره بندر روستا ایستگاه"]
            if schema.dim("climate_class") is not None or schema.family == "place":
                texts += sorted(climate_names)
            words = set(self.planner.own_words[ds]) | _text_keys(texts)
            self.table_keys[ds] = frozenset(words)
        self.output_names = {
            ds: [(o.name, (*o.names, *o.cues)) for o in s.outputs] for ds, s in self.schemas.items()
        }
        # output -> the dimension printed in the same field (a stated value of it is an input)
        self.output_dims = {
            ds: {o.name: {d.name for d in s.dimensions if d.path == o.path} for o in s.outputs}
            for ds, s in self.schemas.items()
        }
        self.table_units = {ds: _units(s) for ds, s in self.schemas.items()}
        self.output_keys = {
            ds: frozenset(
                k
                for o in s.outputs
                for n in (*o.names, *o.cues)
                for w in etokens(n)
                for k in keys(w)
            )
            for ds, s in self.schemas.items()
        }
        # caption/title words per table, for caption references («جدول زمان‌بندی»)
        self.captions: dict[str, frozenset[str]] = {}
        for ds in self.schemas:
            self.captions[ds] = frozenset(
                k
                for w in etokens(normalize_query(self._title(ds)))
                if w not in FUNCTION_WORDS
                for k in keys(w)
                if not k.isdigit()
            )
        sup = self.scope.data.get("supported", {})
        # printed English equivalents of defined terms («Climate» -> «اقلیم»)
        self.bilingual = [
            (tuple(etokens(d["english"])), normalize_query(d["term"]))
            for d in sup.get("defined_terms", [])
            if d.get("english") and etokens(d["english"])
        ]
        # defined terms written joined or spaced («درایوولتاژمتغیر», «میلگرد»)
        self.joined_terms = {
            normalize_query(d["term"]).replace(" ", ""): normalize_query(d["term"])
            for d in sup.get("defined_terms", [])
            if len(etokens(d["term"])) > 1
        }

    # ------------------------------------------------------------------ public
    def route(self, query: str) -> RouteDecision:  # noqa: PLR0911, PLR0915
        norm = normalize_query(query)
        refs, ambiguous = self.table_references(norm)
        section_refs = [s for s in _SECTION_REF.findall(norm) if s not in set(refs)]
        # «پیوست ۳» names the appendix table itself when it is a curated dataset
        refs += [f"پ{n}" for n in _APPENDIX_REF.findall(norm) if f"پ{n}" in APPENDIX_TABLES]
        refs = list(dict.fromkeys(refs))
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
        if ambiguous:
            signals["ambiguous_table_ref"] = ambiguous
        if invalid and not any(c.status == "valid" for c in checks):
            why = "; ".join(f"{c.kind} {c.ref}: {c.status}" for c in invalid)
            signals["scope"] = {"class": "invalid_reference", "reason": why}
            return RouteDecision("unsupported", [f"invalid reference: {why}"], signals)
        blocked = gate(norm)
        if blocked and blocked.category != "outside_mabhas19" and not refs and not section_refs:
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
        named = self.referenced_datasets(refs)
        content = content_tokens(norm)
        known = [t for t in content if t in self.vocabulary]
        coverage = len(known) / len(content) if content else 0.0
        signals["coverage"] = round(coverage, 2)
        structured_part, prose_part, tables, unknown, kinds = self.decompose(
            norm, named, section_scope=bool(section_refs) and not refs
        )
        signals |= {
            "tables": tables,
            "unknown_entities": unknown,
            "structured_part": structured_part,
            "prose_part": prose_part,
            "clauses": kinds,
        }
        reasons = [
            f"{c['table'].split('.')[-1]}: dry-run plan {c['outcome']} ({', '.join(c['evidence'])})"
            for c in self._summaries
        ]
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
        if unknown and not prose_part and not section_refs:
            why = "; ".join(f"unknown {u['type']} {u['tokens']}" for u in unknown)
            signals["scope"] = {"class": "unknown_entity", "reason": why}
            return RouteDecision("unsupported", [f"planner: unresolved entity: {why}"], signals)
        signals["unknown_entities"] = []
        if section_refs:
            signals["scope"] = {"class": "supported", "reason": f"section reference {section_refs}"}
            return _prose_or_fallback(norm, section_refs, coverage, signals)
        d = self.scope.classify_text(norm, self.vocabulary)
        strong = self.strong_evidence(norm, d.evidence.get("strong_anchors", []))
        signals["scope"] = {
            "class": d.scope,
            "reason": d.reason,
            "evidence": {**d.evidence, "strong_in_scope": strong},
        }
        if blocked and not refs:
            # a domain outside Mabhas 19 named by the gate: unsupported unless stronger
            # Mabhas 19 evidence is present
            gate_words = {w for m in blocked.matched for w in etokens(m)}
            if len(gate_words) > len(strong):
                signals["gate"] = {"category": blocked.category, "matched": blocked.matched}
                signals["scope"] = {
                    "class": "external_domain",
                    "reason": f"answerability gate: {blocked.category}",
                    "evidence": {"strong_in_scope": strong},
                }
                return RouteDecision(
                    "unsupported",
                    [f"answerability gate: {blocked.reason} ({blocked.matched})"],
                    signals,
                )
            signals["gate_overridden"] = {"matched": blocked.matched, "by": strong}
        if d.scope == "external_domain":
            ext = d.evidence.get("external", {})
            ext_words = {w for terms in ext.values() for t in terms for w in t.split()}
            if len(strong) >= len(ext_words) and len(strong) > len(d.evidence["strong_anchors"]):
                signals["scope"] = {
                    "class": "uncertain",
                    "reason": "in-scope evidence outranks the external-domain match",
                    "evidence": {**d.evidence, "strong_in_scope": strong},
                }
            else:
                return RouteDecision("unsupported", [f"scope: {d.reason}"], signals)
        return _prose_or_fallback(norm, section_refs, coverage, signals)

    # ------------------------------------------------------------------ references
    def table_references(self, norm: str) -> tuple[list[str], list[dict[str, Any]]]:
        """Explicit table references resolved from the canonical schema: printed IDs anywhere in
        a sentence that names a table, and caption/title words after «جدول» that identify exactly
        one table. Ambiguous captions are returned separately (never guessed)."""
        refs: list[str] = []
        ambiguous: list[dict[str, Any]] = []
        for m in _TABLE_REF.finditer(norm):
            refs.append(m[1])
        for sentence, _ in _questions(norm):
            if not _TABLE_WORD.search(sentence):
                continue
            for m in _TABLE_ID.finditer(sentence):
                ref = re.sub(r"\s", "", m[1])
                if ref.startswith("پ") or MAIN + ref in self.store.main:
                    refs.append(ref)
            for m in _TABLE_WORD.finditer(sentence):
                tail = sentence[m.end() :]
                tables, words = self._caption_match(tail)
                if len(tables) == 1:
                    refs.append(_ref_of(tables[0]))
                elif len(tables) > 1:
                    ambiguous.append({"words": words, "tables": tables})
        return list(dict.fromkeys(refs)), ambiguous

    def _caption_match(self, tail: str) -> tuple[list[str], list[str]]:
        """Tables whose caption/title contains the run of words after «جدول» (function words,
        numbers and IDs skipped). A run of one generic word («جدول اقلیم») names no table."""
        best: list[str] = []
        used: list[str] = []
        for t in etokens(tail)[:12]:
            if t in FUNCTION_WORDS or t in ("شماره", "های", "ها") or re.match(r"\d|پ\d", t):
                continue
            hits = [ds for ds, cap in self.captions.items() if keys(t) & cap]
            narrowed = [ds for ds in hits if not used or ds in best]
            if not narrowed:
                break
            best = narrowed
            used.append(t)
        if len(used) == 1 and keys(used[0]) & GENERIC_KEYS:
            return [], []
        return best, used

    def referenced_datasets(self, refs: list[str]) -> list[str]:
        """Curated datasets named by explicit table/appendix references."""
        out = []
        for raw in refs:
            ref = raw.replace(" ", "")
            ds = APPENDIX_TABLES.get(ref.split("-")[0]) if ref.startswith("پ") else MAIN + ref
            if ds and (ds in self.store.main or ds in self.store.metas):
                out.append(ds)
        return list(dict.fromkeys(out))

    # ------------------------------------------------------------------ detection
    def candidates(self, text: str, named: list[str]) -> list[str]:
        """Plausible tables: named explicitly, sharing a content word with the table's
        vocabulary, or naming a printed place."""
        words = self._words(text)
        units = {x.removeprefix("unit:") for x in self._stated(text) if x.startswith("unit:")}
        out = list(named)
        for ds, tkeys in self.table_keys.items():
            if ds in out:
                continue
            if (
                any(keys(w) & tkeys for w in words)
                or any(has_words(text, names) for _, names in self.output_names[ds])
                or units & self.table_units[ds]
            ):
                out.append(ds)
        if (
            self.store.find_places(text)
            or self.planner.named_provinces(text)
            or re.search(r"(?<!\d)\d{6}(?!\d)", text)
        ):
            out += [d for d in (APP02, MAIN_3_2) if d not in out]
        return out

    def detect(self, text: str, named: list[str], full: str | None = None) -> list[Candidate]:
        """Dry-run plans (never executed) of every plausible table for one text."""
        key = (text, tuple(named), full)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        out = [self._candidate(ds, text, ds in named, full) for ds in self.candidates(text, named)]
        self._cache[key] = out
        return out

    def _candidate(self, ds: str, text: str, explicit: bool, full: str | None) -> Candidate:
        schema = self.schemas[ds]
        plan = self.planner.plan(ds, text, explicit_reference=explicit, full_norm=full)
        outcome = validate(plan, schema)
        c = Candidate(ds, plan, outcome, explicit=explicit)
        probe = self._bilingual(text)
        stated = {
            x.dimension for x in plan.constraints if x.state in ("applied", "checked", "invalid")
        }
        stated |= set(plan.enumerated)
        c.named = any(
            name not in stated
            and not self.output_dims[ds][name] & stated
            and has_words(probe, names)
            for name, names in self.output_names[ds]
        )
        # a value request, or a stated value to look up or to check («… ۰٫۳۵ کدومه؟»)
        c.value = bool(has_words(text, VALUE_REQUEST)) or plan.operation in (
            "reverse",
            "existence",
        )
        resolved = [e for e in plan.entities if e.status in ("resolved", "ambiguous")]
        words = self._words(text)
        explained_words = _explained(words, self.table_keys[ds])
        # an entity named only by the table's output words («حداکثر ضریب انتقال حرارتی» is
        # also a 19-5-2 section label) names no row of its own
        own = any(not keys(words[i]) & self.output_keys[ds] for i in explained_words)
        c.entity = bool(resolved) and own
        c.resolved = c.entity and any(e.status == "resolved" for e in resolved)
        c.place = any(e.kind == "place" and e.status == "resolved" for e in resolved)
        c.concrete = any(
            x.state in ("applied", "checked", "invalid") for x in plan.constraints
        ) or bool(plan.computation and plan.constraints)
        c.subject = c.entity or c.concrete
        explained = set(explained_words)
        # stated constraints (class, rating, group, year, station) are explained only by a
        # table that has the dimension
        items = self._stated(text)
        used = set(stated) | {f"unit:{u}" for u in self.table_units[ds]}
        if any(d.kind == "climate_from_place" for d in plan.derivations):
            used.add("climate_class")
        explained |= {len(words) + j for j, dim in enumerate(items) if dim in used}
        # the table explains a content word or a unit it prints, not only stated constraints
        c.words_explained = bool(explained_words) or any(
            x.startswith("unit:") and x in used for x in items
        )
        c.explained = frozenset(explained)
        total = len(words) + len(items)
        c.items = total
        c.covered = bool(total) and len(c.explained) == total
        # the table explains most of what the text states (a lone shared word is not a lookup)
        c.majority = 2 * len(c.explained) > total
        # an asked quantity the table prints no output for (a quantity word inside the named
        # row's own label - «تست نشت» - is part of the entity, not the question)
        entity_keys = {k for e in resolved for w in etokens(e.text) for k in keys(w)}
        c.foreign = [
            w
            for w in etokens(text)
            if w in QUANTITY_WORDS
            and not keys(w) & self.output_keys[ds]
            and not keys(w) & entity_keys
        ]
        if plan.operation not in ("reverse", "existence") and not plan.enumerations:
            c.asks_input = self._asks_input(ds, text)
        if c.unknown_entity:
            texts = [e.text for e in plan.entities if e.status == "unknown" and e.text]
            unknown = [w for t in texts for w in etokens(t)]
            if not unknown:
                unknown = [w for i, w in enumerate(words) if i not in c.explained]
            c.unknown_words = unknown
            c.unfamiliar = [
                w
                for w in unknown
                if not self._familiar(w)
                and not keys(w) & GENERIC_KEYS
                and not w.replace(".", "").isdigit()
            ]
        return c

    def _familiar(self, word: str) -> bool:
        """A Mabhas 19 word, also in its colloquial spoken form («ساختمونا» ~ «ساختمانها»)."""
        spoken = word.replace("ون", "ان")
        forms = {word, spoken, spoken.removesuffix("ا")}
        return any(keys(f) & self.vocab_keys or keys(f) & GENERIC_KEYS for f in forms)

    def _asks_input(self, ds: str, text: str) -> list[str]:
        """Input dimensions the text asks for («چه رده‌ای», «کدام اقلیم») that the table does not
        print as an output: the question is about another source, not this table's value."""
        out_names = " ".join(n for _, names in self.output_names[ds] for n in names)
        asked = []
        for d in self.schemas[ds].dimensions:
            for noun in d.nouns:
                n = re.escape(normalize_query(noun))
                if re.search(rf"(?:چه|کدام|کدوم)\s*{n}(?!\s*ها)", text) and (
                    normalize_query(noun) not in out_names
                ):
                    asked.append(d.name)
        return asked

    def strong(self, c: Candidate) -> bool:
        """Anchors that need no coverage majority: an explicit reference, a named output with an
        exactly printed place or a stated constraint, or a valid plan whose stated constraints
        come with a value request or a resolved entity and that explains a content word."""
        return (
            c.explicit
            or (c.named and (c.place or c.concrete))
            or (c.valid and c.concrete and (c.value or c.resolved) and c.words_explained)
        )

    def qualifies(self, c: Candidate) -> bool:  # noqa: PLR0911
        """The plan anchors the text to this table (valid or typed); see the module docstring."""
        if c.explicit:
            return True
        if c.foreign or c.asks_input or not c.words_explained:
            return False
        if c.outcome is not None and c.outcome.code == "unsupported_output":
            return False
        if self._section_scope and not c.concrete:
            return False  # a named section scopes the question to its text
        if c.unknown_entity:
            # a real structured request about an entity no record prints (a stated constraint
            # and a request): typed outcome; its unprinted words never count against it
            return c.concrete and (c.named or c.value)
        strong = self.strong(c)
        if not strong and not c.majority:
            return False  # a shared word or two is not a lookup
        if not c.subject:
            return False
        if c.valid:
            return (
                strong
                or c.named
                or c.value
                or c.plan.operation in ("existence", "reverse")
                or (c.covered and c.items >= 2 and self._single_clause)  # a bare row name
            )
        if c.plan.computation:
            # a calculation over this table's values: its operands are named
            return c.majority and c.entity and (c.named or c.value)
        return (c.named or c.concrete) and (c.named or c.value)

    def unknown_request(self, c: Candidate) -> bool:
        """A request for a named table output whose plan fails only on an entity that is not
        Mabhas 19 vocabulary (no stated constraint: «رده اقلیمی لندن»)."""
        return (
            c.unknown_entity
            and not c.concrete
            and not c.foreign
            and not c.asks_input
            and bool(c.unfamiliar)
            and c.named
        )

    def select(self, cands: list[Candidate]) -> list[Candidate]:
        """Qualifying candidates, without those whose explained words are a strict subset of
        another qualifying candidate's (that table is not what is asked)."""
        q = [c for c in cands if self.qualifies(c)]
        if any(c.named for c in q):
            # the question names the output it asks for: a table whose output is not named is
            # not what is asked
            q = [c for c in q if c.named or c.explicit]
        return [
            c
            for c in q
            if c.explicit or not any(c.explained < o.explained for o in q if o.dataset != c.dataset)
        ]

    # ------------------------------------------------------------------ decomposition
    def decompose(
        self, norm: str, named: list[str] | None = None, *, section_scope: bool = False
    ) -> tuple[str, str, list[str], list[dict[str, Any]], list[dict[str, Any]]]:
        """(structured part, prose part, tables, unknown entities, clause kinds).

        Clauses are interpreted by dry-run plans: a clause a plan anchors is lookup-oriented
        (and also prose-oriented when it carries an explanation form); a clause no plan anchors
        is prose-oriented when it asks something of its own; anything else joins its neighbour.
        The prose part is empty when no clause is prose-oriented."""
        named = named or []
        self._cache = {}
        self._summaries = []
        self._section_scope = section_scope
        units = _clauses(norm, bool(named))
        self._single_clause = len(units) <= 1
        whole = self.detect(norm, named)
        whole_sel = self.select(whole)
        self._selected = [c.dataset for c in whole_sel]
        table_keys = frozenset().union(*(self.table_keys[c.dataset] for c in whole_sel))
        kinds: list[str | None] = []
        kind: str | None
        report: list[dict[str, Any]] = []
        clause_tables: list[str] = []
        chosen: list[Candidate] = []
        for clause, scoped, question in units:
            sel = self.select(self.detect(clause, []))
            explains = self._explanation(clause, sel)
            if sel and explains and not any(c.concrete for c in sel):
                kind = "prose"  # a definition question about a table concept
            elif sel:
                clause_tables += [c.dataset for c in sel]
                chosen += sel
                kind = "both" if explains else "structured"
            elif scoped:
                kind = self._scoped_kind(clause, named, explains)
            elif not explains and any(self.unknown_request(c) for c in self.detect(clause, [])):
                kind = None  # asks a table value for an entity no record prints
            elif features(clause).computation:
                kind = None  # a calculation request belongs to the lookup it refers to
            else:
                kind = self._prose_kind(clause, table_keys, question=question)
            kinds.append(kind)
            report.append({"clause": clause, "kind": kind, "tables": [c.dataset for c in sel]})
        if "structured" in kinds or "both" in kinds:
            # a context clause stating constraints belongs to the lookup
            kinds = [
                "structured" if k is None and self._stated(u[0]) else k
                for u, k in zip(units, kinds, strict=True)
            ]
        filled = list(zip([u[0] for u in units], _fill(kinds), strict=True))
        lookup = " و ".join(c for c, k in filled if k == "structured") or " و ".join(
            c for c, k in filled if k == "both"
        )
        prose_part = " و ".join(c for c, k in filled if k in ("prose", "both"))
        if clause_tables:
            # each anchored clause contributes the tables its own plan anchors
            structured = (lookup or norm) if prose_part else norm
            tables = list(
                dict.fromkeys(clause_tables + [c.dataset for c in whole_sel if c.explicit])
            )
            if prose_part and lookup:
                structured = self._shared_context(filled, tables, report)
        elif whole_sel and (not prose_part or any(self.strong(c) for c in whole_sel)):
            # the lookup's evidence is spread over clauses: the whole query is the lookup
            structured, chosen = norm, list(whole_sel)
            tables = [c.dataset for c in chosen]
        else:
            unknown = [self._unknown_signal(c) for c in whole if self.unknown_request(c)]
            return norm, prose_part, [], unknown, report
        self._summaries = [c.summary() for c in chosen]
        return structured, prose_part, self._scope_named(tables, named), [], report

    def _shared_context(
        self,
        filled: list[tuple[str, str | None]],
        tables: list[str],
        report: list[dict[str, Any]],
    ) -> str:
        """The structured part, with *shared context*: a prose clause that names the entity a
        lookup needs («… جزو ردیف مسکونی بزرگ حساب می‌شود؟» before «سقف شدت مصرف رده C آن
        چند است؟») also belongs to the lookup when the lookup's own clauses name no entity of
        that table. The clause stays in the prose part as well, so tokens need not be
        partitioned exclusively; constraints are read from the whole query either way."""
        kind = "structured" if any(k == "structured" for _, k in filled) else "both"
        own = [c for c, k in filled if k == kind]
        lookup = " و ".join(own)
        shared: set[str] = set()
        for ds in tables:
            plan = next((c.plan for c in self.detect(lookup, []) if c.dataset == ds), None)
            if plan is not None and any(
                e.status in ("resolved", "ambiguous") for e in plan.entities
            ):
                continue  # the lookup already names its entity
            for clause, k in filled:
                if k != "prose" or clause in shared:
                    continue
                if any(c.dataset == ds and c.entity for c in self.detect(clause, [])):
                    shared.add(clause)
        if not shared:
            return lookup
        for r in report:
            if r["clause"] in shared:
                r["shared"] = True
        return " و ".join(c for c, k in filled if k == kind or c in shared)

    @staticmethod
    def _scope_named(tables: list[str], named: list[str]) -> list[str]:
        """An explicitly named table scopes the lookup: other datasets contribute only when they
        are named too (a derivation such as city -> climate class stays an input of the named
        table's lookup, never an extra answer)."""
        if not named:
            return list(dict.fromkeys(tables))
        kept = [t for t in tables if t in named]
        return list(dict.fromkeys(kept or tables))

    def _explanation(self, clause: str, sel: list[Candidate]) -> bool:
        """The clause carries an explanation form that is not a word of the selected tables
        («الزامات» printed in a caption is not a request)."""
        text = _WHICH_CLASS.sub(" ", _DEFINED_AS.sub(" ", clause))
        own = frozenset().union(*(self.table_keys[c.dataset] for c in sel)) if sel else frozenset()
        asks = [w for w in has_words(text, EXPLANATION) if not keys(normalize_query(w)) & own]
        return bool(asks) and len(etokens(clause)) > 1

    def _scoped_kind(self, clause: str, named: list[str], explains: bool) -> str | None:
        """A clause in a sentence that names a table is about that table, unless it asks for an
        explanation or carries a question of its own (three or more words the table does not
        explain)."""
        if explains:
            return "prose"
        tkeys = frozenset().union(*(self.table_keys[ds] for ds in named))
        words = self._words(clause)
        explained = _explained(words, tkeys)
        residual = [
            w for i, w in enumerate(words) if i not in explained and keys(w) & self.vocab_keys
        ]
        return "prose" if len(residual) >= 3 else None

    def _prose_kind(
        self, clause: str, table_keys: frozenset[str], *, question: bool = False
    ) -> str | None:
        """An unanchored clause: prose when it asks something of its own (an explanation form,
        an interrogative with unexplained content, or two unexplained content words); a purpose
        remark or a clause without content joins its neighbour."""
        if (
            _PURPOSE.search(clause)
            and not set(etokens(clause)) & INTERROGATIVES
            and not has_words(clause, VALUE_REQUEST)
        ):
            return None
        words = self._words(clause, PROSE_GENERIC_KEYS)
        explained = _explained(words, table_keys)
        residual = [
            w for i, w in enumerate(words) if i not in explained and keys(w) & self.vocab_keys
        ]
        text = _WHICH_CLASS.sub(" ", _DEFINED_AS.sub(" ", clause))
        if has_words(text, EXPLANATION) and len(etokens(clause)) > 1:
            return "prose"
        conceptual = bool(has_words(clause, CONCEPTUAL))
        if len(residual) < 2 and not conceptual and self._names_selected_output(clause):
            return None  # asks the lookup's own output («کد اقلیم … رو چی بگیرم»)
        # a question form: an interrogative, or the clause that closes a «؟» sentence
        asks = question or any(keys(w) & _INTERROGATIVE_KEYS for w in etokens(clause))
        if (asks or conceptual) and (len(residual) + int(conceptual) >= 2 or residual):
            return "prose"
        # a statement without a question form is context («یه پروژه اداری تو مراغه داریم»)
        # unless it carries a topic of its own
        return "prose" if len(residual) >= 3 else None

    def _names_selected_output(self, clause: str) -> bool:
        """The clause names an output of a table the whole query selects."""
        return any(
            has_words(self._bilingual(clause), names)
            for ds in self._selected
            for _, names in self.output_names[ds]
        )

    def _title(self, ds: str) -> str:
        """The printed caption of a main table or the printed title of an appendix."""
        if ds in _APPENDIX_NUMBERS:
            return str(self.scope.appendices.get(_APPENDIX_NUMBERS[ds], {}).get("title", ""))
        return self.schemas[ds].title

    def _printed_texts(self, ds: str) -> list[str]:
        """Printed row labels / entity names of a dataset."""
        s = self.store
        if ds == APP03:
            return [str(o["label"]) for o in s.occupancies.values()]
        if ds == APP07:
            return [
                f"{r.get('display_name', '')} {(r.get('condition') or {}).get('text') or ''}"
                for r in s.app07
            ]
        t = s.main.get(ds)
        if t is None or t.labels is None:
            return []
        return [" ".join(lab.tokens) for lab in t.labels.labels]

    def _unknown_signal(self, c: Candidate) -> dict[str, Any]:
        kind = next((e.kind for e in c.plan.entities if e.status == "unknown"), "entity")
        return {"type": kind, "tokens": c.unfamiliar, "dataset": c.dataset}

    # ------------------------------------------------------------------ helpers
    def _words(self, text: str, generic: frozenset[str] = GENERIC_KEYS) -> list[str]:
        """Content words of a text: constraint spans (classes, ratings, groups, numbers) and
        generic, function and question words removed."""
        stripped = _CONSTRAINT_SPAN.sub(" ", text)
        return [
            w
            for w in etokens(stripped)
            if not keys(w) & generic
            and not any(k.replace(".", "").isdigit() for k in keys(w))
            and not re.fullmatch(r"[a-g]\+{0,3}", w)
        ]

    @staticmethod
    def _stated(text: str) -> list[str]:
        """Dimensions the text states as constraints, and the declared units it names (one
        item each)."""
        f = features(text)
        dims = [
            dim
            for dim, present in (
                ("climate_class", f.climates),
                ("rating", f.ratings),
                ("group", f.groups),
                ("year", f.years),
                ("wmo_code", f.wmo),
            )
            if present
        ]
        units = sorted(
            u for u in _DECLARED_UNITS if any(has_words(text, (form,)) for form in UNITS[u])
        )
        return dims + [f"unit:{u}" for u in units]

    def _bilingual(self, text: str) -> str:
        """The text with the Persian defined term appended for each printed English equivalent
        it uses («climate class» -> «اقلیم»): source-derived, never an alias."""
        toks = etokens(text)
        extra = []
        for eng, term in self.bilingual:
            n = len(eng)
            if any(tuple(toks[i : i + n]) == eng for i in range(len(toks) - n + 1)):
                extra.append(term)
        return f"{text} {' '.join(extra)}" if extra else text

    def strong_evidence(self, norm: str, anchors: list[str]) -> list[str]:
        """Strong in-scope evidence words: the scope map's strong anchors (single-word defined
        terms, concept phrases of titles, captions and table rows) plus the words of multiword
        defined terms written joined («درایوولتاژمتغیر»), which token matching cannot see."""
        out = list(anchors)
        toks = etokens(norm)
        for i in range(len(toks)):
            for n in (1, 2, 3):
                if i + n > len(toks):
                    break
                term = self.joined_terms.get("".join(toks[i : i + n]))
                if term is not None and n < len(etokens(term)):
                    out += [w for w in etokens(term) if w not in out]
        return out


def _units(schema: TableSchema) -> set[str]:
    """Unit keys a table declares for its outputs and dimensions."""
    return {u for o in schema.outputs for u in o.units} | {
        u for d in schema.dimensions for u in d.units
    }


def _text_keys(texts: list[str]) -> set[str]:
    """Matching keys of the words of printed texts, plus their joined bigrams/trigrams
    («هایپر مارکت» ~ «هایپرمارکت») and negator-joined words split («بدونبستری» ~ «بستری»)."""
    out: set[str] = set()
    for text in texts:
        toks = etokens(text)
        for i, w in enumerate(toks):
            out |= keys(w)
            for neg in NEGATORS:
                if w.startswith(neg) and len(w) > len(neg) + 2:
                    out |= keys(w[len(neg) :])
            for n in (2, 3):
                if i + n <= len(toks):
                    out.add("".join(toks[i : i + n]))
    return out


def _explained(words: list[str], table_keys: frozenset[str]) -> set[int]:
    """Positions of words a table explains: by key, or as part of a joined bigram/trigram."""
    out = {i for i, w in enumerate(words) if keys(w) & table_keys}
    for n in (2, 3):
        for i in range(len(words) - n + 1):
            joined = "".join(words[i : i + n])
            if keys(joined) & table_keys:
                out |= set(range(i, i + n))
    return out


def _ref_of(ds: str) -> str:
    for ref, d in APPENDIX_TABLES.items():
        if d == ds:
            return ref
    return ds.removeprefix(MAIN)


def _clauses(norm: str, named: bool) -> list[tuple[str, bool, bool]]:
    """(clause, its sentence names a table explicitly, the clause closes a question)."""
    units: list[tuple[str, bool, bool]] = []
    for sentence, question in _questions(norm):
        scoped = named and (
            bool(_TABLE_WORD.search(sentence)) or bool(_APPENDIX_REF.search(sentence))
        )
        parts = _coordinated(_CLAUSE_SPLIT.split(sentence))
        units += [(c, scoped, question and i == len(parts) - 1) for i, c in enumerate(parts)]
    return units


def _questions(norm: str) -> list[tuple[str, bool]]:
    """(sentence, it ends with a question mark). A full stop ends a sentence only before
    whitespace or the end («kwh/m2.yr», «0.57» stay whole)."""
    out = []
    for m in re.finditer(r"(?:[^?؟!؛;.]|\.(?!\s|$))+(?:[?؟]+|[!؛;]|\.(?=\s|$)|$)", norm):
        text = m[0].strip()
        question = text.endswith(("?", "؟"))
        text = re.sub(r"^و\s+", "", text.rstrip("?؟!؛;. ").strip())
        if text:
            out.append((text, question))
    return out


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
    why = f"no anchored structured plan; corpus vocabulary covered ({coverage:.2f}){note}"
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
