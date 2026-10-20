"""Plan admission: before a structured plan may execute, prove the query supports it.

The planner turns text into a typed :class:`~mabhas19.retrieval.plan.StructuredQueryPlan`; the
validation gate checks the plan against the schema. Admission is a separate, earlier proof that
the *query* genuinely asks this: every admitted component points to the exact query span that
grounded it (query span -> interpreted feature -> schema dimension/output), and nothing exists
only because a table happens to print a similar word.

Admission certifies

* the **target table** (explicitly referenced tables have absolute scope priority: no other
  table is admitted while the question names one);
* a **structured anchor**: an explicit table reference, a stated constraint (climate class,
  rating, group, year, station code, number), a requested enumeration, or an exactly named
  printed row/entity together with a requested output or value — topic words alone never;
* the **entity / row identity**, named source-exactly (:mod:`mabhas19.retrieval.identity`);
* every **constraint** (dimension value, number/operator) with its span;
* every **qualifier** and **exclusion** (:mod:`mabhas19.retrieval.polarity`);
* the **lookup direction** and the **calculation status** (:mod:`mabhas19.retrieval.operations`;
  no calculation is ever admitted);
* **coverage**: every meaning-bearing span is classified (table target, entity, output,
  constraint, qualifier, exclusion, calculation, context, unresolved). An unresolved word that
  could alter regulatory meaning - a printed word of another row, or an unprinted word, standing
  next to the named entity («پمپ حرارتی **زمین‌گرمایی**») - blocks execution; harmless context
  stays unresolved.

A refused plan ends in a typed outcome and never reaches the executor.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from mabhas19.retrieval import polarity
from mabhas19.retrieval.answerability import has_words
from mabhas19.retrieval.entities import FUNCTION_WORDS, keys
from mabhas19.retrieval.identity import query_tokens
from mabhas19.retrieval.normalization import QUESTION_WORDS
from mabhas19.retrieval.numeric import UNITS
from mabhas19.retrieval.operations import operations
from mabhas19.retrieval.plan import Constraint, PlanOutcome, StructuredQueryPlan
from mabhas19.retrieval.schema import APP03, APP07, TableSchema

if TYPE_CHECKING:
    from mabhas19.retrieval.planner import Planner

COMPONENT_CLASSES = (
    "table_target",
    "entity",
    "output",
    "constraint",
    "qualifier",
    "exclusion",
    "calculation",
    "context",
    "unresolved",
)
# explicit structured lookup wording (a value is asked)
LOOKUP_WORDS = (
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
    "چیست",
    "چیه",
)
_TOKEN = re.compile(r"[a-z]*\d+(?:\.\d+)?[a-z]*|[a-z]+|[ؠ-يٮ-ۓە-ۿ]+|[≥≤<>λ]")
_REFERENCE = re.compile(
    r"(?:جدول|جداول|پیوست|فصل|بند|بخش|ماده|تبصره)\s*(?:شماره\s*)?(?:پ\s?)?\d+(?:\s?[-–]\s?\d+)*"
)
_CLIMATE = re.compile(r"(?<![a-z0-9])\d\s?[a-z](?![a-z0-9/])")
_RATING = re.compile(r"(?:رده|رتبه|کلاس)\s*(?:بازدهی\s*)?(?:انرژی\s*)?[a-z]\+{0,3}(?![a-z0-9+])")
_GROUP = re.compile(
    r"گروه\s*(?:های\s*)?(?:ساختمانی\s*|ساختمان\s*)?[«\"'(]?\s*(?:الف|ب|پ|ج|د|ه|و|ی)"
)
_YEAR = re.compile(r"سال\s*(?:اول|یکم|دوم|سوم|چهارم|پنجم|ششم|هفتم|\d+)")
_NUMBER = re.compile(r"\d+(?:[.٫/]\d+)?")
_UNIT_WORDS = frozenset(w for forms in UNITS.values() for f in forms for w in _TOKEN.findall(f))


@dataclass
class Grounding:
    component: str  # table_target | output | entity | constraint | qualifier | exclusion | …
    value: str  # the interpreted feature (dimension = value, output name, row label…)
    span: tuple[int, int] | None  # character span of the grounding text in the query
    basis: str  # explicit | exact | morphology | alias | printed | derived | default


@dataclass
class PlanAdmission:
    """The typed admission certificate of one plan (or its refusal)."""

    target: str
    admitted: bool
    outcome: PlanOutcome | None = None
    groundings: list[Grounding] = field(default_factory=list)
    entity_words: list[str] = field(default_factory=list)  # label words the query names
    entity_rows: list[str] = field(default_factory=list)
    row_words: list[str] = field(default_factory=list)  # every printed word of the admitted rows
    exclusions: list[list[str]] = field(default_factory=list)  # merged tokens per phrase
    constraints: list[Constraint] = field(default_factory=list)
    operation: str = "direct"
    requested_field: str | None = None
    calculation: str | None = None
    explicit_targets: list[str] = field(default_factory=list)
    coverage: list[tuple[str, str]] = field(default_factory=list)  # (token, class)

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "admitted": self.admitted,
            "outcome": self.outcome.code if self.outcome else None,
            "reason": self.outcome.reason if self.outcome else None,
            "groundings": [
                {"component": g.component, "value": g.value, "span": g.span, "basis": g.basis}
                for g in self.groundings
            ],
            "operation": self.operation,
            "calculation": self.calculation,
            "explicit": list(self.explicit_targets),
            "unresolved": [t for t, c in self.coverage if c == "unresolved"],
        }


def _span(text: str, phrase: str) -> tuple[int, int] | None:
    i = text.find(phrase) if phrase else -1
    return (i, i + len(phrase)) if i >= 0 else None


def _refuse(a: PlanAdmission, code: str, reason: str, detail: str) -> PlanAdmission:
    a.admitted = False
    a.outcome = PlanOutcome(code, detail, reason)
    return a


def admit(  # noqa: PLR0911, PLR0912, PLR0913, PLR0915
    planner: Planner,
    plan: StructuredQueryPlan,
    schema: TableSchema,
    text: str,
    full: str | None = None,
    *,
    explicit_targets: list[str] | None = None,
    output_text: str | None = None,
) -> PlanAdmission:
    """Admit or refuse one plan. ``text`` is the structured part of the normalized query,
    ``full`` the whole query (constraints and calculations may sit in another clause)."""
    full = full or text
    explicit = list(explicit_targets or [])
    a = PlanAdmission(
        plan.target,
        True,
        operation=plan.operation,
        requested_field=plan.requested_field,
        explicit_targets=explicit,
    )
    # 1. explicit table reference has absolute scope priority
    if explicit and plan.target not in explicit:
        return _refuse(
            a,
            "unresolved_constraint",
            "explicit_target_scope",
            f"the question names {explicit}; {plan.target} may not provide regulatory values",
        )
    if plan.explicit_reference:
        ref = _REFERENCE.search(full)
        a.groundings.append(
            Grounding("table_target", plan.target, ref.span() if ref else None, "explicit")
        )
    # 2. calculation status: no operation over printed values is ever admitted
    ops = operations(full)
    if plan.computation or ops:
        op = ops[0] if ops else None
        a.calculation = plan.computation or (op.kind if op else None)
        a.groundings.append(
            Grounding(
                "calculation",
                str(a.calculation),
                (op.start, op.end) if op else None,
                "printed",
            )
        )
        return _refuse(
            a,
            "unsupported_calculation",
            "unsupported_calculation",
            f"the question asks for a computed value ({a.calculation}); only printed values exist",
        )
    # 3. qualifiers and exclusions, read before any flattening
    pol = polarity.parse(text)
    excluded = polarity.parse(full).exclusions  # an exclusion anywhere in the question applies
    a.exclusions = [query_tokens(x.text) for x in excluded]
    for x in excluded:
        a.groundings.append(Grounding("exclusion", x.text, (x.start, x.end), "printed"))
    index = planner.identity.get(plan.target)
    for q, scope in zip(pol.negative_forms, pol.negated, strict=True):
        if index is None or index.prints(query_tokens(q.text)[0]):
            a.groundings.append(Grounding("qualifier", q.text, (q.start, q.end), "printed"))
        else:  # a negative form no label prints negates its phrase (an exclusion)
            a.exclusions.append(query_tokens(scope.text))
            a.groundings.append(
                Grounding(
                    "exclusion",
                    pol.positive[q.start : scope.end],
                    (q.start, scope.end),
                    "printed",
                )
            )
    # 4. constraints: each one grounded in its own span
    for c in plan.constraints:
        if c.state not in ("applied", "checked"):
            continue
        span = _constraint_span(full, c)
        if span is None and c.source == "" and _derived(plan, c):
            a.groundings.append(Grounding("constraint", c.describe(), None, "derived"))
            a.constraints.append(c)
            continue
        if span is None:
            return _refuse(
                a,
                "unresolved_constraint",
                "ungrounded_constraint",
                f"{c.describe()} is not grounded in a span of the question",
            )
        a.groundings.append(Grounding("constraint", c.describe(), span, "printed"))
        a.constraints.append(c)
    for en in plan.enumerations:
        a.groundings.append(
            Grounding(
                "constraint", f"enumeration {en.dimension}", _span(full, en.request), "printed"
            )
        )
    # 5. output: named, or the table's default output of a value request
    out = schema.output(plan.requested_field or "") if plan.requested_field else None
    named_output = None
    for o in schema.outputs:
        # printed English equivalents of defined terms count («climate class» ~ «اقلیم»)
        hit = has_words(output_text or full, (*o.names, *o.cues))
        if hit:
            named_output = (o.name, hit[0])
            break
    if named_output:
        a.groundings.append(
            Grounding("output", named_output[0], _span(full, named_output[1]), "printed")
        )
    elif out is not None:
        a.groundings.append(Grounding("output", out.name, None, "default"))
    # 6. entity identity: source-exact
    resolved = [e for e in plan.entities if e.status == "resolved"]
    if resolved and schema.family != "place" and plan.target in planner.identity:
        rows = _entity_rows(plan, planner)
        exact, head, _ = planner.identify(plan.target, text, full)
        # an explicitly requested enumeration lists every printed row under the named head
        enumerated = any(
            e.dimension in ("occupancy", "material", "label") for e in plan.enumerations
        )
        named_rows = set(exact) | (set(head) if enumerated else set())
        if rows and not set(rows) <= named_rows:
            return _refuse(
                a,
                "ambiguous_entity",
                "entity_not_named_exactly",
                "the named entity is not a printed row named exactly (normalization, morphology "
                "or a reviewed alias); no nearest row is chosen",
            )
        a.entity_rows = rows
        idx = planner.identity[plan.target]
        toks = query_tokens(pol.positive)
        named = {w for r in rows for w in idx.named(r, toks).words}
        a.entity_words = sorted(named)
        a.row_words = sorted(
            {w for r in rows for alt in idx.by_row.get(r, []) for w in alt.words}
            | {w for r in rows for w in idx.printed.get(r, [])}
        )
        for w in a.entity_words:
            a.groundings.append(Grounding("entity", w, _word_span(text, w), "exact"))
    elif resolved and schema.family == "place":
        for ent in resolved:
            for place in ent.ids:
                span = _span(full.replace(" ", ""), place.replace(" ", ""))
                if span is None:
                    return _refuse(
                        a,
                        "unknown_entity",
                        "place_not_named",
                        f"«{place}» is not named in the question",
                    )
                a.groundings.append(Grounding("entity", place, span, "exact"))
                a.entity_words.append(place)
                a.row_words.append(place)
    # 7. a real structured anchor is required
    # a row named only by words that are also prepositions («روی»: zinc / on) is named only
    # together with its output: «… روی خاک رسی چقدر …» asks nothing of zinc
    homograph = bool(a.entity_words) and all(w in _PREPOSITIONS for w in a.entity_words)
    lookup = bool(named_output) or (bool(has_words(full, LOOKUP_WORDS)) and not homograph)
    anchored = (
        plan.explicit_reference
        or bool(a.constraints)
        or bool(plan.enumerations)
        or (bool(a.entity_words) and lookup)
        or (bool(a.entity_words) and plan.operation == "existence")
    )
    if not anchored:
        return _refuse(
            a,
            "unresolved_constraint",
            "no_structured_anchor",
            "no explicit reference, stated constraint, or exactly named row with a requested "
            "value: topic words alone are not a structured lookup",
        )
    # 8. coverage: an unresolved word next to the named entity that could change its meaning
    a.coverage = _coverage(planner, plan, text, a)
    blocking = _blocking(planner, plan, text, a)
    if blocking:
        return _refuse(
            a,
            "unresolved_constraint",
            "unresolved_modifier",
            f"unresolved words {blocking} qualify the named entity and could change which printed "
            "row answers; nothing is executed",
        )
    return a


def _derived(plan: StructuredQueryPlan, c: Constraint) -> bool:
    return any(d.kind == "climate_from_place" for d in plan.derivations) and (
        c.dimension == "climate_class"
    )


def _constraint_span(full: str, c: Constraint) -> tuple[int, int] | None:  # noqa: PLR0911, PLR0912
    """The query span a constraint was read from."""
    dim = c.dimension
    if dim == "climate_class":
        if c.source and c.source in full:
            return _span(full, c.source)
        m = _CLIMATE.search(full)
        return m.span() if m else None
    if dim == "rating":
        m = _RATING.search(full)
        if m:
            return m.span()
        for v in c.values:
            sp = _span(full, v.lower())
            if sp:
                return sp
        return None
    if dim == "group":
        m = _GROUP.search(full)
        return m.span() if m else None
    if dim == "year":
        m = _YEAR.search(full)
        return m.span() if m else None
    if dim == "province":
        return _span(full, "استان")
    if c.source:
        sp = _span(full, c.source)
        if sp:
            return sp
    for v in c.values:
        sp = _span(full, str(v).lower())
        if sp:
            return sp
    m = _NUMBER.search(full)
    return m.span() if m and c.interval is not None else None


def _entity_rows(plan: StructuredQueryPlan, planner: Planner) -> list[str]:
    if plan.target in (APP03, APP07):
        kind = "occupancy" if plan.target == APP03 else "material"
        return [
            i for e in plan.entities if e.kind == kind and e.status == "resolved" for i in e.ids
        ]
    return list(plan.row_ids or [])


def _word_span(text: str, word: str) -> tuple[int, int] | None:
    for m in _TOKEN.finditer(text):
        if keys(m[0]) & keys(word):
            return m.span()
    return _span(text.replace(" ", ""), word)


# prepositions that are also printed names («روی» zinc)
_PREPOSITIONS = frozenset({"روی", "زیر", "کنار", "پشت", "بالای", "داخل", "بیرون", "میان"})

# verb participles of a relative clause («… که با ماسه رودخانه‌ای ساخته شده») never name a row
_PARTICIPLES = frozenset({"ساخته", "شده", "گرفته", "داشته", "کرده", "قرار"})


def _generic(planner: Planner) -> frozenset[str]:
    from mabhas19.retrieval.planner import (  # noqa: PLC0415
        _COLLOQUIAL,
        _STRUCTURAL_WORDS,
        _WHOLE_WORDS,
    )
    from mabhas19.retrieval.structured import REQUEST_WORDS  # noqa: PLC0415

    words = (
        set(FUNCTION_WORDS)
        | set(QUESTION_WORDS)
        | set(REQUEST_WORDS)
        | set(_COLLOQUIAL)
        | set(_WHOLE_WORDS)
        | set(LOOKUP_WORDS)
        | _UNIT_WORDS
        | _PARTICIPLES
    )
    return frozenset({k for w in words for k in keys(w)} | set(_STRUCTURAL_WORDS))


def _joined_keys(planner: Planner, target: str, rows: list[str]) -> set[str]:
    """Keys of joined spellings of consecutive printed label words («پشمشیشه», «ایزوبوتیلن»)."""
    index = planner.identity.get(target)
    out: set[str] = set()
    for row in rows if index is not None else []:
        ws = index.printed.get(row, []) if index is not None else []
        for n in (2, 3):
            out |= {k for i in range(len(ws) - n + 1) for k in keys("".join(ws[i : i + n]))}
    return out


def _coverage(
    planner: Planner, plan: StructuredQueryPlan, text: str, a: PlanAdmission
) -> list[tuple[str, str]]:
    """Every token of the structured text with the component class that explains it."""
    by_comp: dict[str, list[tuple[int, int]]] = {}  # component -> grounded spans
    for g in a.groundings:
        if g.span is not None:
            by_comp.setdefault(g.component, []).append(g.span)
    generic = _generic(planner)
    entity_keys = {k for w in (*a.entity_words, *a.row_words) for k in keys(w)}
    entity_keys |= _joined_keys(planner, plan.target, a.entity_rows)
    # a parenthesised gloss («(زینک)», «(PTFE)») is context, an exclusion inside parentheses not
    gloss = [m.span() for m in re.finditer(r"\([^()]*\)", text)]
    out_names = {
        k
        for o in planner.schemas[plan.target].outputs
        for n in (*o.names, *o.cues)
        for w in _TOKEN.findall(n)
        for k in keys(w)
    }
    ref_spans = [m.span() for m in _REFERENCE.finditer(text)]
    out: list[tuple[str, str]] = []
    tokens = list(_TOKEN.finditer(text))
    for i, m in enumerate(tokens):
        tok, (s, e) = m[0], m.span()
        # a word that forms a printed row word with its neighbour («نیمه باز» ~ «نیمهباز»)
        joined = [tokens[j][0] + tokens[j + 1][0] for j in (i - 1, i) if 0 <= j < len(tokens) - 1]

        def inside(spans: list[tuple[int, int]], s: int = s, e: int = e) -> bool:
            return any(a0 <= s and e <= b0 for a0, b0 in spans)

        if inside(ref_spans):
            cls = "table_target"
        elif inside(by_comp.get("calculation", [])):
            cls = "calculation"
        elif inside(by_comp.get("exclusion", [])):
            cls = "exclusion"
        elif inside(by_comp.get("qualifier", [])):
            cls = "qualifier"
        elif inside(by_comp.get("constraint", [])) or _NUMBER.fullmatch(tok):
            cls = "constraint"
        elif keys(tok) & entity_keys or any(keys(x) & entity_keys for x in joined):
            cls = "entity"
        elif keys(tok) & out_names:
            cls = "output"
        elif keys(tok) & generic or tok in generic or inside(gloss):
            cls = "context"
        else:
            cls = "unresolved"
        out.append((tok, cls))
    return out


def _blocking(
    planner: Planner, plan: StructuredQueryPlan, text: str, a: PlanAdmission
) -> list[str]:
    """Unresolved words that could change which printed row answers.

    * a word a printed sibling of the admitted row carries (a label sharing one of its named
      words): it may select that sibling instead;
    * a word outside Mabhas 19 vocabulary in modifier position, right after the named entity
      (Persian attributes follow their noun): an unprinted variant or product («… کوانتومی»).

    A word before the entity («خونه ویلایی», «شعبه بانک») or after a clause word («برای …»)
    is context and never blocks."""
    if not a.entity_words or plan.target not in planner.identity:
        return []
    idx = planner.identity[plan.target]
    own = {k for w in (*a.entity_words, *a.row_words) for k in keys(w)}
    rows = set(a.entity_rows)
    siblings = {
        k
        for w in a.entity_words
        for g in idx.group(w)
        if g.row not in rows
        for x in g.words
        for k in keys(x)
    } - own
    vocab = planner.vocab_keys()
    cov = a.coverage
    out = []
    for i, (tok, cls) in enumerate(cov):
        if cls != "unresolved":
            continue
        modifier = i > 0 and cov[i - 1][1] == "entity"
        if keys(tok) & siblings or (modifier and vocab and not keys(tok) & vocab):
            out.append(tok)
    return out
