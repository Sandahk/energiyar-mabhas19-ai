"""Plan coverage: the query-level stage between per-plan admission and execution.

Admission proves, one table at a time, that the query supports a plan. Some unsafe answers are
only visible across plans and clauses: a span read as one table's stated constraint and as
another table's requested output, a rating copied from an explanatory clause, a second table
admitted on a value another table owns. This stage sees every candidate plan of the query
together (refused ones included), records a typed **coverage ledger** - each grounded span with
its offsets, clause owner, semantic role, candidate table, grounding basis and schema reference -
and decides which admitted plans may execute. It is a function of the query, the clause
partition, the candidates' typed spans, the schemas and the curated identity index; it never
reads parser internals beyond the typed plan and admission certificate, so the same checks can
later verify a proposal made by another parser.

Invariants (each violation is recorded in the ledger; a violated plan never executes):

* **I1 one role per span** - a span grounding a stated constraint (with its cue words) never
  also establishes a requested output, in the same or another candidate plan;
* **I2 concept disjointness** - a direct lookup never asks for the concept it states as a
  constraint («رده C» stated and the rating asked; «چگالی اسمی ۵۰۰» stated and the density
  asked);
* **I3 complete requested quantity** - the quantity a question asks for is one complete span,
  wherever it stands: a printed output phrase of any table (longest first) with the words that
  follow it («حداکثر مصرف **آب** سالانه»), the phrase a trailing «چند / چقدر» asks about
  («**مقاومت فشاری**‌اش چند …»), the phrase after «چه», and the output slot before the named
  entity («**ظرفیت گرمایی ویژه** بتن …»). Every meaning-bearing word of the span must be attested
  by the target table's printed vocabulary (its output names, cues and printed headers, printed
  column headers, caption and dimension words). A span is a noun phrase: entity words (of this
  plan, of another candidate, of a row any table names exactly in the query, a place marker),
  constraints, units, function words and pronouns, comparison and modality words and the
  predicate (a verb, and the noun of a light verb: «حساب میشه», «تعیین کرده») end it and never
  block; a possessive clitic closes it («اقلیمش …»), and a noun right before this plan's entity
  is that entity's head noun when the section holding the table prints it («ضریب انتقال حرارت
  **پنجره** ثابت» of 19-5-2; «حداکثر مصرف **آب** مطب …» stays a quantity word: Appendix 3
  prints none). Spans in the router's lookup clauses decide; when those
  hold no request, the rest of the plan's sentence does («…سبکه، مقاومت فشاری‌اش چند …»). A span
  with an unattested word, or naming another quantity while the table's own output is not named,
  is ``unsupported_output``; an empty request keeps the schema default. No synonym is equated
  («رسانایی گرمایی» is not «هدایت حرارتی»);
* **I4 unit expression** (:mod:`mabhas19.retrieval.units`) - every unit expression of the
  lookup's clauses, before or after the entity, with or without «بر حسب», Latin or Persian,
  compound or with a temporal basis, is read as dimensional atoms: the printed unit (or an
  elliptical form of it) grounds the request; the same physical quantity in other units or on
  another basis is ``unsupported_calculation`` (``unit_conversion``: no conversion exists);
  another physical quantity is ``unsupported_output``. A unit attached to a stated number is
  that number's unit (only a unit the source never prints refuses it);
* **I5 identity-span accounting** - every source-attested identity-bearing span is accounted
  for by the admitted rows: head-only evidence (an unresolved word that is the printed head of
  other rows) shares a printed level with them (``identity_family_conflict``); a printed
  condition phrase the question uses positively - negative forms included («بدون دودکش») - is
  printed by the admitted rows whenever another row of the table prints it
  (``qualifier_family_conflict``; a negated other family, «نه بخاری», is an exclusion and stays
  allowed); one subject described as two printed identities («هم مهدکودکه هم پیش‌دبستانی») is
  ``ambiguous_entity`` (``one_subject_two_identities``), never split. Joined or copula forms
  («مهدکودکه») are accounting evidence only, never a naming rule. The printed levels and the
  required phrases are handed to certification, which re-checks them on the row;
* **I6 value ownership** - a dimension's values come from the clauses owning the lookup; values
  of the same dimension stated only in an explanatory clause are ignored when the lookup states
  the dimension, inherited when they are one unconflicted value, and unresolved otherwise; a
  value stated only inside the rejected span of a correction («4B نه، 5B», «نه 4B، 5B», «4B نه
  بلکه 5B», «4B نیست، 5B است»; :func:`mabhas19.retrieval.polarity.corrections`) is never an
  active constraint (dropped; a constraint left without values is unresolved);
* **I7 table ownership** - a plan executes only with an anchor of its own: an entity, an explicit
  reference, an established output, or a constraint span no candidate with a stronger anchor
  grounds. An unowned plan is dropped (it is not what the question asked; no no-match);
* **I8 coordinated entities** (:func:`split_coordinated`) - coordinated entities of any
  entity-identifying dimension (places, occupancies, materials, equipment and other printed
  rows) become one child plan per member, each admitted, executed and certified against its own
  member. Members are the conjuncts of the router's lookup clauses that the identity index names
  exactly by a printed label word (a category value or a number names no member; a conjunction
  inside one printed name - «سرمایش و گرمایش ۴ لوله», a quoted name - is never cut); together
  they name exactly the rows the whole question names, the plan's own rows among them, and a
  single subject described twice («هم X هم Y») is never split (I5). A stated qualifier value
  (density, area, condition) belongs to the nearest member before it; a categorical value
  (rating, climate class, group, year) belongs to its member when every member's segment states
  its own («مطب … 2B … رده D و کتابخانه … 1B … رده C»), else it is shared and stays OR-valued
  inside each child. Parent certification checks that no row answers two members.
* **I10 table scope** (:mod:`mabhas19.retrieval.table_scope`) - a stated value of a regulatory
  dimension the table does not print must fall inside the table's declared scope: ``universal``
  dimensions never block, a value outside a ``fixed`` scope is ``invalid_constraint``, an
  undeclared dimension fails closed.

I9 (output certification) lives in :mod:`mabhas19.retrieval.certification`.

The invariants are grouped into five contracts plus the row contract (``CONTRACTS``; every ledger
violation names its contract): span roles and ownership (I1, I6), output (I2, I3, I4), table (I7,
I10), identity (I5), coordination (I8) and row certification (I9).
"""

from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import dataclass, field, replace
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from mabhas19.retrieval import polarity, table_scope, units
from mabhas19.retrieval.entities import keys
from mabhas19.retrieval.plan import (
    Constraint,
    EntityRef,
    PlanOutcome,
    StructuredQueryPlan,
    concept,
    validate,
)
from mabhas19.text.normalize import normalize_for_search

if TYPE_CHECKING:
    from mabhas19.retrieval.admission import PlanAdmission
    from mabhas19.retrieval.planner import Planner
    from mabhas19.retrieval.schema import Output, TableSchema
    from mabhas19.retrieval.structured import StructuredStore

COVERAGE_VERSION = "coverage-2"
# the coverage-ledger schema (``Ledger.to_dict``): pinned by the candidate manifest, so a run's
# ledgers stay comparable and a future proposal can be verified against the same layout
LEDGER_SCHEMA = "coverage-ledger-2"
LEDGER_KEYS = ("schema", "version", "clauses", "corrections", "spans", "unit_intents",
               "violations", "adjustments", "verdicts")  # fmt: skip
INVARIANTS = ("I1", "I2", "I3", "I4", "I5", "I6", "I7", "I8", "I9", "I10")
# the contract each invariant belongs to (stable ledger codes inside named contracts)
CONTRACTS = {
    "I1": "span_roles",
    "I6": "span_roles",
    "I2": "output",
    "I3": "output",
    "I4": "output",
    "I7": "table",
    "I10": "table",
    "I5": "identity",
    "I8": "coordination",
    "I9": "row",
}
# roles a span may not carry at once (across every candidate plan of a query)
INCOMPATIBLE = frozenset(
    {
        frozenset({"output", "constraint"}),
        frozenset({"output", "exclusion"}),
        frozenset({"output", "qualifier"}),
    }
)
# the outcome reported for a refused plan, by first violated invariant in this order
_PRECEDENCE = ("I4", "I3", "I10", "I2", "I6", "I5", "I8")
_TOKEN = re.compile(r"[a-z]*\d+(?:\.\d+)?[a-z]*|[a-z]+|[ؠ-يٮ-ۓە-ۿ]+|[≥≤<>λ]")
_BETWEEN = re.compile(r"^[\s‌]*$")  # only spacing between two adjacent words
# numeric comparison words that may stand between a dimension cue and its value
_OPERATOR_WORDS = frozenset(
    {
        "کمتر",
        "بیشتر",
        "بیش",
        "از",
        "بین",
        "تا",
        "برابر",
        "مساوی",
        "بالای",
        "زیر",
        "کوچکتر",
        "بزرگتر",
        "تر",
        "≥",
        "≤",
        "<",
        ">",
    }
)
_CATEGORY_DIMS = ("rating", "climate_class", "group", "year")
# an explicit request for a unit («بر حسب X»): X may be any word, read or refused
_UNIT_REQUEST = re.compile(r"(?:بر\s?حسب|به\s?واحد|با\s?واحد|در\s?واحد|به\s?یکای)\s+")


@lru_cache(maxsize=8192)
def _identify(
    planner: Planner, ds: str, text: str, full: str | None = None
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """(exactly named rows, head-only rows) of a text: the planner's identity resolution, cached
    (a pure function of the curated identity index and the text)."""
    exact, head, _ = planner.identify(ds, text, full)
    return tuple(exact), tuple(head)


def _concept(name: str) -> str:
    """The canonical concept of a dimension or output name (rating = printed energy rating)."""
    return {"energy_rating": "rating", "climate_classes": "climate_class"}.get(name, concept(name))


# ------------------------------------------------------------------ ledger types


@dataclass(frozen=True)
class Clause:
    index: int
    start: int  # -1 when the clause could not be located in the query
    end: int
    text: str
    kind: str | None  # the router's clause kind (structured | both | prose | None)
    owner: bool  # the clause belongs to the structured lookup (not an explanation question)


@dataclass
class Span:
    start: int
    end: int
    text: str
    role: str  # table_target | entity | output | unit | constraint | qualifier | exclusion | slot
    table: str
    basis: str  # explicit | exact | printed | derived | default | unit | unresolved
    ref: str | None = None  # schema dimension / output / unit
    clause: int | None = None
    structural: bool = True  # False: harmless context (never decides a row)
    state: str = "grounded"  # grounded | discarded (role conflict) | blocking

    def overlaps(self, other: Span) -> bool:
        return self.start < other.end and other.start < self.end

    def to_dict(self) -> dict[str, Any]:
        return {
            "span": [self.start, self.end],
            "text": self.text,
            "role": self.role,
            "table": self.table,
            "basis": self.basis,
            "ref": self.ref,
            "clause": self.clause,
            "structural": self.structural,
            "state": self.state,
        }


@dataclass
class Violation:
    table: str
    invariant: str
    reason: str
    detail: str
    member: str | None = None
    code: str | None = None  # the plan outcome it refuses with (default: by invariant)

    @property
    def contract(self) -> str:
        return CONTRACTS[self.invariant]

    def to_dict(self) -> dict[str, Any]:
        return {
            "table": self.table,
            "member": self.member,
            "contract": self.contract,
            "invariant": self.invariant,
            "reason": self.reason,
            "detail": self.detail,
        }


@dataclass
class UnitIntent:
    start: int
    end: int
    text: str
    unit: units.Unit | None  # None: not a unit expression this parser reads (never converted)
    attached: bool = False  # follows a stated number (the number's unit)


@dataclass
class Candidate:
    """One candidate plan of a query (admitted or refused) as the coverage stage sees it."""

    plan: StructuredQueryPlan
    admission: PlanAdmission
    gate: PlanOutcome | None  # the validation gate's outcome (None: the plan is valid)
    text: str  # the structured text admission read
    member: str | None = None  # the coordinated member of a child plan (I8)
    outcome: PlanOutcome | None = None  # set when coverage refuses an admitted plan
    owned: bool = True
    violations: list[Violation] = field(default_factory=list)

    @property
    def target(self) -> str:
        return self.plan.target

    @property
    def admitted(self) -> bool:
        return self.admission.admitted and self.gate is None

    @property
    def executable(self) -> bool:
        return self.admitted and self.outcome is None and self.owned


@dataclass
class Ledger:
    query: str
    clauses: list[Clause]
    corrections: list[polarity.Correction] = field(default_factory=list)
    spans: list[Span] = field(default_factory=list)
    unit_intents: list[UnitIntent] = field(default_factory=list)
    violations: list[Violation] = field(default_factory=list)
    # I6 values dropped or inherited without a refusal (inspectable, not violations)
    adjustments: list[dict[str, str]] = field(default_factory=list)
    verdicts: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": LEDGER_SCHEMA,
            "version": COVERAGE_VERSION,
            "clauses": [
                {"index": c.index, "span": [c.start, c.end], "kind": c.kind, "owner": c.owner}
                for c in self.clauses
            ],
            "corrections": [
                {
                    "pattern": c.pattern,
                    "rejected": [c.rejected.start, c.rejected.end],
                    "accepted": [c.accepted.start, c.accepted.end],
                    "text": [c.rejected.text, c.accepted.text],
                }
                for c in self.corrections
            ],
            "spans": [s.to_dict() for s in self.spans],
            "unit_intents": [
                {
                    "span": [u.start, u.end],
                    "text": u.text,
                    "unit": u.unit.describe() if u.unit else None,
                    "attached": u.attached,
                }
                for u in self.unit_intents
            ],
            "violations": [v.to_dict() for v in self.violations],
            "adjustments": self.adjustments,
            "verdicts": self.verdicts,
        }


# ------------------------------------------------------------------ clauses


def clauses_of(full: str, report: list[dict[str, Any]]) -> list[Clause]:
    """The router's clauses located in the query; an explanation-question clause (kind
    ``prose``) does not own the lookup. A single clause owns everything."""
    if len(report) < 2:
        return [Clause(0, 0, len(full), full, "structured", True)]
    out, pos = [], 0
    for i, c in enumerate(report):
        text = str(c.get("clause") or "")
        at = full.find(text, pos) if text else -1
        if at < 0 and text:
            at = full.find(text)
        start, end = (at, at + len(text)) if at >= 0 else (-1, -1)
        if at >= 0:
            pos = end
        kind = c.get("kind")
        out.append(Clause(i, start, end, text, kind, kind != "prose"))
    return out


_SENTENCE = re.compile(r"(?:[^?؟!؛;.]|\.(?!\s|$))+(?:[?؟]+|[!؛;]|\.(?=\s|$)|$)")


def sentences(full: str) -> list[tuple[int, int, bool]]:
    """(start, end, it is a question) of each sentence («؟ ! ؛ .» end one; «0.57» does not)."""
    return [
        (m.start(), m.end(), m[0].rstrip().endswith(("?", "؟")))
        for m in _SENTENCE.finditer(full)
        if m[0].strip()
    ]


def foreign_questions(full: str, anchors: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Separate question sentences: a question sentence holding none of the plan's own anchors
    (entity, output, explicit reference) asks something of its own."""
    if not anchors:
        return []
    return [
        (s, e)
        for s, e, question in sentences(full)
        if question and not any(s <= a < e for a, _ in anchors)
    ]


def requested_here(foreign: list[tuple[int, int]], start: int, end: int) -> bool:
    """A request (quantity or unit) at this span belongs to the plan: it lies in no separate
    question sentence. The router's clause kinds do not decide this - the clause holding
    «… چند مگاپاسکاله؟» is the question even when the router files it as prose."""
    return not any(s <= start and end <= e for s, e in foreign)


def owned_by(clauses: list[Clause], full: str, anchors: list[tuple[int, int]]) -> list[Clause]:
    """The clauses one plan owns: the router's lookup clauses, minus those in a *separate
    question* - a question sentence holding none of the plan's own anchors (entity, output,
    explicit reference) asks something of its own («فرق رده‌های A و B چیه؟ …»). A statement
    sentence stays shared context («ساختمان ما در اقلیم 3A است. …»)."""
    foreign = foreign_questions(full, anchors)
    if not foreign:
        return clauses
    return [
        replace(c, owner=False)
        if c.owner and c.start >= 0 and any(s <= c.start and c.end <= e for s, e in foreign)
        else c
        for c in clauses
    ]


def clause_at(clauses: list[Clause], start: int, end: int) -> Clause | None:
    return next((c for c in clauses if c.start <= start and end <= c.end and c.start >= 0), None)


def _owned(clauses: list[Clause], start: int, end: int) -> bool:
    c = clause_at(clauses, start, end)
    return c is None or c.owner  # an unlocated span is never treated as foreign


# ------------------------------------------------------------------ spans


def _tokens(text: str) -> list[re.Match[str]]:
    return list(_TOKEN.finditer(text))


def cue_keys(schemas: dict[str, TableSchema], target: str, dimension: str) -> set[str]:
    """Words that attach a stated value to a dimension: its own cues, the words of a named
    condition, and every schema output/dimension name of the same concept."""
    words: list[str] = []
    d = schemas[target].dim(dimension)
    if d is not None:
        words += d.cues
    if dimension.startswith("condition:"):
        words.append(dimension.split(":", 1)[1])
    want = _concept(dimension)
    for sc in schemas.values():
        for o in sc.outputs:
            if _concept(o.name) == want:
                words += [*o.names, *o.cues]
        for dd in sc.dimensions:
            if _concept(dd.name) == want:
                words += dd.cues
    return {k for w in words for t in _TOKEN.findall(w) for k in keys(t)}


def extend_with_cue(full: str, start: int, end: int, cues: set[str]) -> tuple[int, int]:
    """A constraint span widened to the cue words right before its value («چگالی اسمی ۵۰۰»,
    «اقلیم 4b»): the cue is part of the stated constraint, never a separate role."""
    toks = _tokens(full)
    i = next((j for j, m in enumerate(toks) if m.start() >= start), len(toks))
    new, steps, j = start, 0, i - 1
    while j >= 0 and steps < 4:
        m = toks[j]
        if not _BETWEEN.match(full[m.end() : new]):
            break
        if keys(m[0]) & cues:
            new = m.start()
        elif m[0] not in _OPERATOR_WORDS:
            break
        j -= 1
        steps += 1
    return new, end


def _occurrences(full: str, phrase: str) -> list[tuple[int, int]]:
    if not phrase.strip():
        return []
    pattern = rf"(?<![ؠ-ۿa-z0-9]){re.escape(phrase)}(?![ؠ-ۿa-z0-9])"
    return [m.span() for m in re.finditer(pattern, full)]


def full_span(
    cand: Candidate, component: str, value: str, span: tuple[int, int], full: str
) -> tuple[int, int] | None:
    """A grounding's span in the whole query. Admission records entity and qualifier spans in the
    structured text (place names in its space-free form); constraint, output, reference,
    calculation and exclusion spans are already whole-query spans."""
    if component not in ("entity", "qualifier"):
        return span
    if cand.target and cand.plan.entities and cand.plan.entities[0].kind == "place":
        name = value.replace(" ", "")
        m = re.search(r"\s?".join(map(re.escape, name)), full) if name else None
        return m.span() if m else None
    offset = full.find(cand.text) if cand.text else -1
    if offset >= 0:
        return span[0] + offset, span[1] + offset
    region = cand.text[span[0] : span[1]]
    for m in _TOKEN.finditer(full):
        if keys(m[0]) & keys(region or value):
            return m.span()
    return None


def anchors_of(cand: Candidate, full: str) -> list[tuple[int, int]]:
    """Whole-query spans of the plan's own anchors (entity, explicit reference, named output)."""
    out = []
    for g in cand.admission.groundings:
        if g.span is None or not (
            g.component in ("entity", "table_target")
            or (g.component == "output" and g.basis != "default")
        ):
            continue
        sp = full_span(cand, g.component, g.value, g.span, full)
        if sp is not None:
            out.append(sp)
    return out


def candidate_spans(
    cand: Candidate, full: str, clauses: list[Clause], schemas: dict[str, TableSchema]
) -> list[Span]:
    """The candidate's grounded spans (constraints widened to their cues; each occurrence of a
    named output phrase)."""
    out: list[Span] = []
    for g in cand.admission.groundings:
        if g.span is None:
            continue
        mapped = full_span(cand, g.component, g.value, g.span, full)
        if mapped is None:
            continue
        start, end = mapped
        ref = None
        if g.component == "constraint":
            dim = next(
                (c.dimension for c in cand.admission.constraints if c.describe() == g.value),
                g.value.split(" ")[0],
            )
            ref = dim
            start, end = extend_with_cue(full, start, end, cue_keys(schemas, cand.target, dim))
        if g.component == "output":
            ref = g.value
            phrase = full[start:end]
            for s, e in _occurrences(full, phrase) or [(start, end)]:
                c = clause_at(clauses, s, e)
                out.append(
                    Span(
                        s, e, full[s:e], "output", cand.target, g.basis, ref, c.index if c else None
                    )
                )
            continue
        c = clause_at(clauses, start, end)
        out.append(
            Span(
                start,
                end,
                full[start:end],
                g.component,
                cand.target,
                g.basis,
                ref or (g.value if g.component in ("entity", "table_target") else None),
                c.index if c else None,
            )
        )
    return out


# ------------------------------------------------------------------ I1


def role_conflicts(spans: list[Span]) -> list[tuple[Span, Span]]:
    """Pairs of overlapping spans whose roles are incompatible (I1), across all plans."""
    out = []
    for i, a in enumerate(spans):
        for b in spans[i + 1 :]:
            if a.overlaps(b) and frozenset({a.role, b.role}) in INCOMPATIBLE:
                out.append((a, b))
    return out


def established_outputs(spans: list[Span], table: str, clauses: list[Clause]) -> set[str]:
    """Outputs of a table named by at least one occurrence that carries no other role and lies in
    a clause owning the lookup."""
    return {
        s.ref
        for s in spans
        if s.table == table
        and s.role == "output"
        and s.state == "grounded"
        and s.ref
        and _owned(clauses, s.start, s.end)
    }


# ------------------------------------------------------------------ I2


def concept_conflict(plan: StructuredQueryPlan) -> str | None:
    """The concept a direct lookup both states and asks for (I2), if any."""
    if plan.operation != "direct" or not plan.requested_field:
        return None
    asked = _concept(plan.requested_field)
    stated = {_concept(c.dimension) for c in plan.constraints if c.state == "applied"}
    return asked if asked in stated else None


# ------------------------------------------------------------------ I3


def output_slot(admission: PlanAdmission, text: str) -> list[str]:
    """The unresolved words right before the first named-entity word, with nothing but spacing
    between them: the place of the requested output in «X-e entity» (I3)."""
    toks = _tokens(text)
    cov = admission.coverage
    if len(toks) != len(cov):
        return []
    first = next((i for i, (_, cls) in enumerate(cov) if cls == "entity"), None)
    if first is None:
        return []
    slot: list[str] = []
    j, nxt = first - 1, toks[first].start()
    while j >= 0 and cov[j][1] == "unresolved" and _BETWEEN.match(text[toks[j].end() : nxt]):
        if not re.fullmatch(r"\d+(?:\.\d+)?", cov[j][0]):
            slot.insert(0, cov[j][0])
        nxt = toks[j].start()
        j -= 1
    return slot


# a trailing question word asks for the phrase before it; «چه» for the phrase after it
_ASK_AFTER = frozenset({"چند", "چنده", "چقدر", "چقدره", "چه قدر"})
_ASK_BEFORE = frozenset({"چه"})
# detached clitics («فشاری اش» after normalization) belong to the previous word
_CLITICS = frozenset({"اش", "ش", "ها", "های", "هایش", "ای", "ی", "یی", "ه", "شان", "مان", "تان"})
_CLITIC_END = re.compile(r"(?:اش|ش|ای|یی|ی|ه)$")
# the predicate ends a noun phrase: verbs and auxiliaries, and the light verbs whose preceding
# noun is part of the verb («حساب میشه», «تعیین کرده», «مصرف کند»)
_VERBS = frozenset(
    {"است", "هست", "نیست", "هستند", "بود", "باشد", "باشه", "بشه", "داره", "دارد", "دارند",
     "داشته", "آمده", "امده", "اومده", "گذشته", "رفته", "بگیرد", "بگیره", "میگیرد", "دیدم"}
)  # fmt: skip
_LIGHT_VERBS = frozenset(
    {"می", "کند", "کنه", "کنند", "کرد", "کرده", "کردن", "میکند", "میکنه", "شده", "شود",
     "میشه", "میشود", "شدن", "گرفته", "داده", "دادن", "دادند", "خورده", "خورد", "خوردن",
     "خوردند", "زده", "تواند", "تونه"}
)  # fmt: skip
# the modality of a request, never part of the quantity («مجاز», «قابل قبول», «الزامی»)
_MODALITY = frozenset({"مجاز", "قابل", "قبول", "الزامی", "لازم", "مورد", "نیاز", "تعیین"})
# pronouns and deictics («رده اقلیمی آنجا», «اقلیم ما»)
_PRONOUNS = frozenset(
    {"آنجا", "انجا", "اینجا", "اونجا", "آن", "ان", "این", "اون", "ما", "شما", "آنها", "انها",
     "اونها", "همان", "همین", "هر", "خود", "خودش"}
)  # fmt: skip
_CLOSED = _VERBS | _LIGHT_VERBS | _MODALITY | _OPERATOR_WORDS | _PRONOUNS


def named_rows_words(planner: Planner, full: str) -> frozenset[str]:
    """Keys of the words of every printed row some table's identity index names exactly in the
    query (another entity of the question: «COP کولر گازی … و چیلر …»), and the place markers."""
    from mabhas19.retrieval.identity import query_tokens  # noqa: PLC0415
    from mabhas19.retrieval.planner import PLACE_MARKERS  # noqa: PLC0415

    # a second entity is named with its printed family head (the head of its first level:
    # «کولر گازی»; «برقی» alone does not name «آبگرمکن برقی»), never through printed output
    # vocabulary (the quantity phrase itself)
    output_vocab = {
        k
        for sc in planner.schemas.values()
        for o in sc.outputs
        for n in (*o.names, *o.cues)
        for w in _TOKEN.findall(normalize_for_search(n).lower())
        for k in keys(w)
    }
    words: set[str] = set(PLACE_MARKERS)
    toks = query_tokens(polarity.parse(full).positive)
    for ds, idx in planner.identity.items():
        exact, _ = _identify(planner, ds, full, full)
        for row in exact:
            named = idx.named(row, toks).words
            heads = {
                k for alt in idx.by_row.get(row, []) if alt.levels and alt.levels[0]
                for k in keys(alt.levels[0][0])
            }  # fmt: skip
            if any(keys(w) & heads for w in named) and not any(
                keys(w) & output_vocab for w in named
            ):
                words |= set(named)
    return frozenset(k for w in words for k in keys(w))


_ATTEST_CACHE: dict[int, dict[str, frozenset[str]]] = {}


def _word_keys(texts: list[str]) -> frozenset[str]:
    return frozenset(
        k for t in texts for w in _TOKEN.findall(normalize_for_search(t).lower()) for k in keys(w)
    )


def attested_words(store: StructuredStore, ds: str) -> frozenset[str]:
    """Keys of the words a table's printed vocabulary attests for a requested quantity: output
    names, cues and printed definition headers, the printed column headers, the caption,
    dimension names, cues and nouns."""
    cache = _ATTEST_CACHE.setdefault(id(store), {})
    if ds not in cache:
        sc = store.schemas[ds]
        texts = [store.table_evidence(ds, "").title, *store.headers.get(ds, [])]
        for o in sc.outputs:
            texts += [*o.names, *o.cues, *o.header]
        for d in sc.dimensions:
            texts += [d.name.replace("_", " "), *d.cues, *d.nouns]
        cache[ds] = _word_keys(texts)
    return cache[ds]


def output_phrases(
    schemas: dict[str, TableSchema],
) -> list[tuple[tuple[str, ...], frozenset[str]]]:
    """Every printed output name and cue as (word tuple, the tables printing it), longest first
    (ties in a fixed order: a phrase several tables print is each of theirs)."""
    out: dict[tuple[str, ...], set[str]] = {}
    for ds, sc in schemas.items():
        for o in sc.outputs:
            for n in (*o.names, *o.cues):
                words = tuple(_TOKEN.findall(normalize_for_search(n).lower()))
                if words:
                    out.setdefault(words, set()).add(ds)
    return [(w, frozenset(t)) for w, t in sorted(out.items(), key=lambda x: (-len(x[0]), x[0]))]


@dataclass
class QuantitySpan:
    start: int
    end: int
    words: list[str]  # the meaning-bearing words the target must attest
    kind: str  # head | ask_after | ask_before
    own: bool  # opened by a printed output phrase of the target itself


def _token_classes(  # noqa: PLR0913
    cand: Candidate,
    full: str,
    toks: list[re.Match[str]],
    *,
    generic: frozenset[str],
    identity_keys: frozenset[str],
    bounds: list[tuple[int, int]],
    entities: list[tuple[int, int]],
) -> list[str]:
    """word | entity | bound for every token of the query (admission's own classes inside the
    structured text, the same rules outside it)."""
    by_pos: dict[int, str] = {}
    offset = full.find(cand.text) if cand.text else -1
    ctoks = _tokens(cand.text) if offset >= 0 else []
    if offset >= 0 and len(ctoks) == len(cand.admission.coverage):
        for m, (_, kind) in zip(ctoks, cand.admission.coverage, strict=True):
            by_pos[offset + m.start()] = kind
    out = []
    for i, m in enumerate(toks):
        tok = m[0]
        cls = by_pos.get(m.start())
        nxt = toks[i + 1][0] if i + 1 < len(toks) else ""
        stem = _CLITIC_END.sub("", tok)
        if (
            any(a <= m.start() and m.end() <= b for a, b in bounds)
            or re.fullmatch(r"[\d.]+[a-z]?|[a-z]", tok)
            or tok in _CLOSED
            or (len(stem) >= 2 and stem in _CLOSED)
            or nxt in _LIGHT_VERBS
        ):
            out.append("bound")
        elif (
            cls == "entity"
            or keys(tok) & identity_keys
            or any(a <= m.start() and m.end() <= b for a, b in entities)
        ):
            out.append("entity")
        elif cls in ("unresolved", "output") or (
            cls is None and not (keys(tok) & generic or tok in generic)
        ):
            out.append("word")
        else:
            out.append("bound")
    return out


def quantity_spans(  # noqa: PLR0913
    cand: Candidate,
    full: str,
    classes: list[str],
    toks: list[re.Match[str]],
    *,
    phrases: list[tuple[tuple[str, ...], frozenset[str]]],
    entity_heads: frozenset[str] = frozenset(),
) -> list[QuantitySpan]:
    """The requested-quantity spans of the query for one candidate (I3)."""

    def adjacent(i: int, j: int) -> bool:
        a, b = sorted((i, j))
        return bool(_BETWEEN.match(full[toks[a].end() : toks[b].start()]))

    def run(i: int, step: int) -> list[int]:
        out, j = [], i + step
        while 0 <= j < len(toks) and classes[j] == "word" and adjacent(j - step, j):
            out.append(j)
            if step > 0 and words[j] == "اش":
                break  # a possessive clitic closes the phrase
            j += step
        # the head noun of the entity right after it («… پنجره ثابت»)
        nxt = out[-1] + 1 if step > 0 and out else -1
        head = 0 < nxt < len(toks) and classes[nxt] == "entity" and adjacent(out[-1], nxt)
        if head and keys(words[out[-1]]) & entity_heads:
            out.pop()
        return sorted(out)

    def possessive(tok: str, phrase_word: str) -> bool:
        return len(tok) > 3 and tok.endswith("ش") and bool(keys(tok[:-1]) & keys(phrase_word))

    spans: list[QuantitySpan] = []
    taken: set[int] = set()
    words = [m[0] for m in toks]
    for phrase, tables in phrases:
        n = len(phrase)
        for i in range(len(toks) - n + 1):
            if any(j in taken for j in range(i, i + n)):
                continue
            if not all(keys(words[i + k]) & keys(phrase[k]) for k in range(n)):
                continue
            if not all(adjacent(i + k, i + k + 1) for k in range(n - 1)):
                continue
            taken |= set(range(i, i + n))
            own = cand.target in tables
            last = i + n - 1
            mods = [] if possessive(words[last], phrase[-1]) else run(last, 1)
            idx = mods if own else [*range(i, i + n), *mods]
            spans.append(
                QuantitySpan(
                    toks[i].start(),
                    toks[(mods or [i + n - 1])[-1]].end(),
                    [words[j] for j in idx if classes[j] == "word"],
                    "head",
                    own,
                )
            )
    for i, w in enumerate(words):
        if w in _ASK_AFTER:
            idx = run(i, -1)
        elif w in _ASK_BEFORE:
            idx = run(i, 1)
        else:
            continue
        idx = [j for j in idx if j not in taken]
        if idx:
            spans.append(
                QuantitySpan(
                    toks[idx[0]].start(),
                    toks[idx[-1]].end(),
                    [words[j] for j in idx],
                    "ask_after" if w in _ASK_AFTER else "ask_before",
                    False,
                )
            )
    return spans


def quantity_violation(  # noqa: PLR0913
    cand: Candidate,
    store: StructuredStore,
    planner: Planner,
    full: str,
    *,
    clauses: list[Clause],
    foreign: list[tuple[int, int]],
    spans: list[Span],
    intents: list[UnitIntent],
    named_elsewhere: frozenset[str] = frozenset(),
) -> Violation | None:
    """I3: the first requested-quantity span the target does not attest completely."""
    from mabhas19.retrieval.admission import _generic  # noqa: PLC0415

    # entity words: the admitted rows' own words, the printed heads of the table's rows (identity
    # evidence, I5) and words the table prints both in its rows and in its own section («پنجره»
    # of 19-5-2). Another row's other word is not («حداکثر مصرف برق …»: «برق» of another
    # occupancy label; Appendix 3 has no section of its own)
    idx = planner.identity.get(cand.target)
    section = {k for w in store.section_words.get(cand.target, ()) for k in keys(w)}
    table_words: set[str] = set()
    if idx is not None:
        table_words = {k for ws in idx.printed.values() for w in ws for k in keys(w)} & section
        # a printed head of the table's rows is identity evidence (accounted for by I5)
        table_words |= {
            k for alts in idx.by_row.values() for a in alts for h in a.heads for k in keys(h)
        }
    identity_keys = (
        frozenset(
            k for w in (*cand.admission.entity_words, *cand.admission.row_words) for k in keys(w)
        )
        | table_words
        | named_elsewhere
    )
    bounds = [
        (s.start, s.end)
        for s in spans
        if s.table == cand.target
        and s.role in ("constraint", "qualifier", "exclusion", "table_target")
    ] + [(u.start, u.end) for u in intents]
    # entity spans of every candidate of the query: another plan's entity is never a quantity
    entities = [(s.start, s.end) for s in spans if s.role == "entity"]
    toks = _tokens(full)
    classes = _token_classes(
        cand,
        full,
        toks,
        generic=_generic(planner),
        identity_keys=identity_keys,
        bounds=bounds,
        entities=entities,
    )
    heads = frozenset(k for w in store.section_words.get(cand.target, ()) for k in keys(w))
    found = [
        q
        for q in quantity_spans(
            cand, full, classes, toks, phrases=output_phrases(store.schemas), entity_heads=heads
        )
        if requested_here(foreign, q.start, q.end)
    ]
    # the table's own output named anywhere in the plan's request makes a phrase of another
    # table's quantity that table's request; the router's lookup clauses decide which spans are
    # checked, the rest of the sentence only when they hold none
    own_named = any(q.own for q in found)
    lookup = [q for q in found if _owned(clauses, q.start, q.end)]
    found = lookup or found
    attested = attested_words(store, cand.target)

    def joined(q: QuantitySpan) -> set[str]:
        """Span words forming a printed joined word with their neighbour («آب گرم» ~ «آبگرم»)."""
        out = set()
        for a, b in zip(q.words, q.words[1:], strict=False):
            if keys(a + b) & attested:
                out |= {a, b}
        return out

    def ok(w: str) -> bool:
        if len(w) < 2 or w in _CLITICS or keys(w) & attested:
            return True
        stem = _CLITIC_END.sub("", w)
        if len(stem) >= 2 and keys(stem) & attested:
            return True
        # a part of a printed joined word («آب» of «آبگرم»)
        return len(w) >= 2 and any(a.startswith(w) and len(a) - len(w) >= 3 for a in attested)

    for q in found:
        pairs = joined(q)
        bad = [w for w in q.words if w not in pairs and not ok(w)]
        if not bad:
            continue
        if not q.own and own_named and (q.kind == "head" or len(bad) == len(q.words)):
            continue  # another table's quantity in a question that also names this table's own
        return Violation(
            cand.target,
            "I3",
            "unattested_quantity",
            f"the requested quantity «{full[q.start : q.end]}» carries {bad}, which "
            f"{cand.target.split('.')[-1]} does not print",
            cand.member,
            "unsupported_output",
        )
    return None


def default_output(schema: TableSchema) -> str | None:
    d = schema.direction("direct")
    return d.output if d is not None else (schema.outputs[0].name if schema.outputs else None)


# ------------------------------------------------------------------ I4


def unit_intents(
    full: str, clauses: list[Clause], vocabulary: frozenset[str] = frozenset()
) -> list[UnitIntent]:
    """Unit expressions of the clauses owning the lookup (requests) and units attached to stated
    numbers: explicit requests («بر حسب …») first, then every other unit expression."""
    out: list[UnitIntent] = list(_requested_units(full, vocabulary))
    for e in units.find(full):
        if any(o.start < e.end and e.start < o.end for o in out):
            continue
        if e.attached or _owned(clauses, e.start, e.end):
            out.append(UnitIntent(e.start, e.end, e.text, e.unit, e.attached))
    return sorted(
        (u for u in out if u.attached or _owned(clauses, u.start, u.end)), key=lambda u: u.start
    )


def _requested_units(full: str, vocabulary: frozenset[str]) -> list[UnitIntent]:
    """After «بر حسب / به واحد …» the longest following phrase the unit parser reads is the
    requested unit; a following word that is neither a readable unit nor Mabhas 19 vocabulary
    («بر حسب فوت مربع») is an unknown unit (refused, never converted), while a vocabulary word
    («بر حسب کاربری», «به واحد مسکونی») is no unit request at all."""
    out = []
    for m in _UNIT_REQUEST.finditer(full):
        rest = full[m.end() :]
        words = list(re.finditer(r"\S+", rest))[:6]
        best: tuple[int, units.Unit | None] = (0, None)
        for n in range(len(words), 0, -1):
            parsed = units.parse(rest[: words[n - 1].end()])
            if parsed is not None:
                best = (words[n - 1].end(), parsed)
                break
        if best[1] is None:
            first = words[0][0].strip(".،,؟?!؛") if words else ""
            if not first or keys(first) & vocabulary or not re.search(r"[ؠ-ۿ]", first):
                continue
            best = (words[0].end(), None)
        s, e = m.start(), m.end() + best[0]
        out.append(UnitIntent(s, e, full[s:e], best[1]))
    return out


def unit_named_output(intents: list[UnitIntent], schema: TableSchema) -> str | None:
    """The one output of the table whose printed unit a requested unit grounds («… چند کیلوگرم
    بر متر مکعب» names the density), if exactly one."""
    named = set()
    for u in intents:
        if u.attached or u.unit is None or not u.unit.num:
            continue
        for o in schema.outputs:
            printed = units.parse(o.unit) if o.unit else None
            if printed is not None and units.verdict(u.unit, printed) == units.GROUND:
                named.add(o.name)
    return next(iter(named)) if len(named) == 1 else None


def unit_verdict(
    intents: list[UnitIntent], output: Output | None, numeric: bool
) -> tuple[Violation | None, bool]:
    """(the I4 violation or None, a requested unit names this output). The violation's table is
    filled in by the caller."""

    def refuse(reason: str, detail: str, code: str) -> tuple[Violation, bool]:
        return Violation("", "I4", reason, detail, None, code), False

    printed = units.parse(output.unit) if output is not None and output.unit else None
    shown = output.unit if output is not None else ""
    named = False
    for u in intents:
        if u.attached:
            # the unit of a stated number: only a unit the source never prints is a conversion
            if numeric and u.unit is not None and u.unit.foreign:
                return refuse(
                    "unit_conversion",
                    f"a stated number is in «{u.text}», not a printed unit; no conversion is "
                    "performed",
                    "unsupported_calculation",
                )
            continue
        if u.unit is None:
            return refuse(
                "unit_conversion",
                f"the requested unit «{u.text}» is not a unit the source prints",
                "unsupported_calculation",
            )
        if printed is None:
            if u.unit.foreign:
                return refuse(
                    "unit_conversion",
                    f"the requested unit «{u.text}» is not printed for this table",
                    "unsupported_calculation",
                )
            continue
        v = units.verdict(u.unit, printed)
        if v == units.CONVERSION:
            return refuse(
                "unit_conversion",
                f"«{u.text}» asks the printed quantity in other units or on another basis than "
                f"the printed «{shown}»; no conversion is performed",
                "unsupported_calculation",
            )
        if v == units.OTHER_QUANTITY:
            return refuse(
                "other_quantity_unit",
                f"«{u.text}» is a unit of another physical quantity than the printed «{shown}»",
                "unsupported_output",
            )
        named = named or bool(u.unit.num)
    return None, named


# ------------------------------------------------------------------ I5


def family_conflict(planner: Planner, cand: Candidate, full: str) -> tuple[list[str], list[str]]:
    """(rows another identity span of the query names that share no printed level with the
    admitted rows, rows of that evidence that do share one).

    The other identity spans are words the admission left unresolved that are the printed head of
    other rows («**تخته** سیمانی …» heads «صفحات تخته چندلا»): head-only identity evidence. Rows
    the same words name exactly are the admitted name's siblings, told apart by stated
    constraints («مسکونی ≥/< ۳۰۰۰»), never a second identity. Rows without identity words (bands
    identified by numbers) carry no family."""
    idx = planner.identity.get(cand.target)
    rows = list(cand.admission.entity_rows)
    if idx is None or not rows:
        return [], []
    _, head = _identify(planner, cand.target, cand.text, full)
    unresolved = {k for t, c in cand.admission.coverage if c == "unresolved" for k in keys(t)}
    evidence = [
        r
        for r in head
        if r not in rows
        and any(keys(w) & unresolved for alt in idx.by_row.get(r, []) for w in alt.heads)
    ]

    def levels(row: str) -> set[tuple[str, ...]]:
        return {tuple(lv) for alt in idx.by_row.get(row, []) for lv in alt.levels if lv}

    mine = set().union(*(levels(r) for r in rows))
    evidence = [r for r in evidence if levels(r)]
    if not evidence or not mine:
        return [], []
    conflict = [r for r in evidence if not levels(r) & mine]
    consistent = [r for r in evidence if levels(r) & mine]
    return conflict, consistent


def phrase_accounting(
    planner: Planner, store: StructuredStore, cand: Candidate
) -> tuple[list[str], list[list[str]]]:
    """(printed qualifier phrases only other rows print, phrases every admitted row prints): a
    positive printed condition phrase of the question must be printed by an admitted row (I5).
    A phrase of the table's own caption («غیرنورگذر» of 19-5-1) is the whole table's family."""
    from mabhas19.retrieval.identity import query_tokens  # noqa: PLC0415

    idx = planner.identity.get(cand.target)
    rows = list(cand.admission.entity_rows)
    if idx is None or not rows:
        return [], []
    caption = _word_keys([store.table_evidence(cand.target, "").title])
    conflicts: list[str] = []
    required: list[list[str]] = []
    for g in cand.admission.groundings:
        if g.component != "qualifier":
            continue
        phrase = query_tokens(g.value)
        if not phrase or all(keys(w) & caption for w in phrase):
            continue
        mine = [r for r in rows if idx.excluded(r, [phrase])]
        if len(mine) == len(rows):
            required.append(phrase)
        elif not mine and any(idx.excluded(r, [phrase]) for r in idx.printed if r not in rows):
            conflicts.append(g.value)
    return conflicts, required


# «هم X هم Y», «هم X و هم Y»: one subject described as two things
_BOTH = re.compile(
    r"(?<![ؠ-ۿ])هم\s+((?:[^\s()،,؛;؟?.!]+\s*){1,4}?)\s*(?:و\s+)?هم\s+((?:[^\s()،,؛;؟?.!]+\s*){1,4})"
)
_COPULA_END = re.compile(r"(?:ه|است|هست|ست)$")


def _identified(planner: Planner, ds: str, text: str) -> set[str]:
    """Rows the identity index names (exactly or by head) in a phrase; a trailing copula
    («مهدکودکه») is read off only as accounting evidence."""
    words = text.split()
    for cand_words in (words, [*words[:-1], _COPULA_END.sub("", words[-1])] if words else []):
        phrase = " ".join(w for w in cand_words if w)
        if not phrase:
            continue
        exact, head = _identify(planner, ds, phrase)
        if exact or head:
            return set(exact) | set(head)
    return set()


def one_subject_two_identities(planner: Planner, cand: Candidate, full: str) -> list[str]:
    """The two descriptions of «هم X هم Y» when they name disjoint printed rows of the table."""
    if cand.target not in planner.identity:
        return []
    for m in _BOTH.finditer(full):
        x, y = m[1].strip(), m[2].strip()
        rx, ry = _identified(planner, cand.target, x), _identified(planner, cand.target, y)
        # only the words up to the first row-naming prefix of Y belong to it
        if rx and ry and not rx & ry:
            return [x, y]
    return []


# ------------------------------------------------------------------ I6


def _clause_values(text: str) -> dict[str, set[str]]:
    from mabhas19.retrieval.planner import features  # noqa: PLC0415

    f = features(text)
    return {
        "rating": set(f.ratings),
        "climate_class": set(f.climates),
        "group": set(f.groups),
        "year": {str(y) for y in f.years},
    }


def clause_precedence(  # noqa: PLR0912
    plan: StructuredQueryPlan, clauses: list[Clause]
) -> tuple[dict[int, Constraint], list[str]]:
    """Replacement constraints (by index in ``plan.constraints``) and notes (I6)."""
    external = [c for c in clauses if not c.owner and c.start >= 0]
    owners = [c for c in clauses if c.owner]
    if not external or not owners:
        return {}, []
    inside: dict[str, set[str]] = {d: set() for d in _CATEGORY_DIMS}
    outside: dict[str, set[str]] = {d: set() for d in _CATEGORY_DIMS}
    for c in owners:
        for d, vals in _clause_values(c.text).items():
            inside[d] |= vals
    for c in external:
        for d, vals in _clause_values(c.text).items():
            outside[d] |= vals
    out: dict[int, Constraint] = {}
    notes: list[str] = []
    for i, con in enumerate(plan.constraints):
        if con.dimension not in _CATEGORY_DIMS or con.state not in ("applied", "checked"):
            continue
        values = [str(v) for v in con.values]
        foreign = [
            v for v in values if v not in inside[con.dimension] and v in outside[con.dimension]
        ]
        if not foreign:
            continue
        if inside[con.dimension]:
            keep = [v for v in values if v not in foreign]
            if keep:
                out[i] = replace(con, values=keep)
                notes.append(
                    f"{con.dimension} {foreign} stated only in an explanatory clause; the lookup's "
                    f"own clause states {sorted(inside[con.dimension])}"
                )
            else:
                out[i] = replace(
                    con, state="unresolved", detail=f"{con.dimension} only in an explanatory clause"
                )
        elif len(values) == 1:
            notes.append(f"{con.dimension} {values} inherited from a single explanatory mention")
        else:
            out[i] = replace(
                con,
                state="unresolved",
                detail=f"{con.dimension} {foreign}: several values, all outside the lookup",
            )
            notes.append(f"{con.dimension} {foreign}: several values outside the lookup")
    return out, notes


def rejected_values(full: str, corrections: list[polarity.Correction]) -> dict[str, set[str]]:
    """Category values stated only inside rejected spans of corrections (I6 value ownership)."""
    if not corrections:
        return {}
    inside: dict[str, set[str]] = {d: set() for d in _CATEGORY_DIMS}
    rest = list(full)
    for c in corrections:
        for d, vals in _clause_values(c.rejected.text).items():
            inside[d] |= vals
        rest[c.rejected.start : c.rejected.end] = " " * (c.rejected.end - c.rejected.start)
    outside = _clause_values("".join(rest))
    return {d: inside[d] - outside[d] for d in _CATEGORY_DIMS if inside[d] - outside[d]}


def correction_ownership(
    plan: StructuredQueryPlan, rejected: dict[str, set[str]]
) -> tuple[dict[int, Constraint], list[str]]:
    """Replacement constraints without rejected values (I6); a constraint left without values is
    unresolved."""
    out: dict[int, Constraint] = {}
    notes: list[str] = []
    for i, con in enumerate(plan.constraints):
        bad = {v.upper() for v in rejected.get(con.dimension, set())}
        if not bad or con.state not in ("applied", "checked", "unsupported_by_table"):
            continue
        drop = [str(v) for v in con.values if str(v).upper() in bad]
        if not drop:
            continue
        keep = [str(v) for v in con.values if str(v).upper() not in bad]
        notes.append(f"{con.dimension} {drop} rejected by a correction; not a stated value")
        out[i] = (
            replace(con, values=keep)
            if keep
            else replace(
                con,
                state="unresolved" if con.state != "unsupported_by_table" else con.state,
                values=[],
                detail=f"{con.dimension}: only rejected values {drop}",
            )
        )
    return out, notes


def owned_values(clauses: list[Clause], full: str) -> dict[str, set[str]]:
    """Category values stated in the clauses owning the lookup (the whole query when the router
    found one clause)."""
    owners = [c for c in clauses if c.owner and c.start >= 0]
    if len(clauses) < 2 or not owners:
        return _clause_values(full)
    out: dict[str, set[str]] = {d: set() for d in _CATEGORY_DIMS}
    for c in owners:
        for d, vals in _clause_values(c.text).items():
            out[d] |= vals
    return out


def apply_constraints(cand: Candidate, new: dict[int, Constraint]) -> None:
    """Swap replaced constraints into the plan and the admission certificate."""
    if not new:
        return
    old = {id(cand.plan.constraints[i]): c for i, c in new.items()}
    cand.plan.constraints = [new.get(i, c) for i, c in enumerate(cand.plan.constraints)]
    cand.admission.constraints = [
        old.get(id(c), c)
        for c in cand.admission.constraints
        if old.get(id(c), c).state in ("applied", "checked")
    ]


# ------------------------------------------------------------------ I10


def scope_violations(
    cand: Candidate, schema: TableSchema, owned: dict[str, set[str]], rejected: dict[str, set[str]]
) -> list[Violation]:
    """I10: stated values of regulatory dimensions the table does not print, owned by the lookup
    and not rejected, checked against the declared scope (and recorded for certification)."""
    out = []
    for con in cand.plan.constraints:
        if con.dimension not in table_scope.REGULATORY or con.state != "unsupported_by_table":
            continue
        mine = {v.upper() for v in owned.get(con.dimension, set())}
        bad = {v.upper() for v in rejected.get(con.dimension, set())}
        values = [
            str(v) for v in con.values if str(v).upper() in mine and str(v).upper() not in bad
        ]
        if not values:
            continue
        v = table_scope.verdict(schema, con.dimension, values)
        if v.state == table_scope.IN_SCOPE:
            cand.admission.scope_values[con.dimension] = list(values)
        if v.blocks:
            out.append(
                Violation(
                    cand.target,
                    "I10",
                    "out_of_scope_value"
                    if v.state == table_scope.OUT_OF_SCOPE
                    else "undeclared_scope",
                    v.detail,
                    cand.member,
                    "invalid_constraint"
                    if v.state == table_scope.OUT_OF_SCOPE
                    else "unresolved_constraint",
                )
            )
    return out


# ------------------------------------------------------------------ I7


def strength(cand: Candidate, spans: list[Span], clauses: list[Clause]) -> int:
    """2: an entity or explicit reference of its own; 1: an established output; 0: constraints."""
    mine = [s for s in spans if s.table == cand.target and s.state == "grounded"]
    if any(s.role in ("entity", "table_target") for s in mine):
        return 2
    if established_outputs(spans, cand.target, clauses):
        return 1
    return 0


def unowned(
    cand: Candidate, others: list[Candidate], spans: list[Span], clauses: list[Clause]
) -> bool:
    """I7: the plan's anchors are all spans a candidate of another table with a stronger anchor
    grounds (refused candidates included, so a value cannot fall through to a weaker table)."""
    mine = strength(cand, spans, clauses)
    if mine == 2:
        return False
    stronger = [o for o in others if o.target != cand.target and strength(o, spans, clauses) > mine]
    if not stronger:
        return False
    their = [
        s for s in spans if any(s.table == o.target for o in stronger) and s.state == "grounded"
    ]
    if mine == 1:
        anchors = [
            s
            for s in spans
            if s.table == cand.target and s.role == "output" and s.state == "grounded"
        ]
    else:
        anchors = [s for s in spans if s.table == cand.target and s.role == "constraint"]
    return bool(anchors) and all(any(a.overlaps(t) for t in their) for a in anchors)


# ------------------------------------------------------------------ I8


_CONJUNCTION = re.compile(r"\s+(?:و|یا)\s+|\s*[،,]\s*")
_QUOTED = re.compile(r"«[^»]*»|\"[^\"]*\"")


def _inside_name(planner: Planner, ds: str, full: str, cut: tuple[int, int]) -> bool:
    """The conjunction lies inside one printed name: the words around it name a row exactly that
    neither side names alone («سرمایش و گرمایش ۴ لوله»)."""
    left = " ".join(full[: cut[0]].split()[-3:])
    right = " ".join(full[cut[1] :].split()[:3])
    if not left or not right:
        return False
    joint = set(_identify(planner, ds, f"{left} و {right}")[0])
    alone = set(_identify(planner, ds, left)[0]) | set(_identify(planner, ds, right)[0])
    return bool(joint - alone)


_CATEGORY_WORD = re.compile(r"\d+(?:\.\d+)?|\d[a-z]|[a-z]|الف|ب|پ|ج|د|ه")


def _named_by_label(planner: Planner, ds: str, text: str, rows: set[str]) -> bool:
    """Some row is named by a printed label word of the text, not by a category value or a
    number only («5B چقدر است» names no member)."""
    from mabhas19.retrieval.identity import query_tokens  # noqa: PLC0415

    idx = planner.identity[ds]
    toks = query_tokens(polarity.parse(text).positive)
    return any(not _CATEGORY_WORD.fullmatch(w) for r in rows for w in idx.named(r, toks).words)


def coordinated_members(
    planner: Planner,
    ds: str,
    full: str,
    rows: list[str],
    clauses: list[Clause] | None = None,
) -> list[tuple[str, list[str], int]]:
    """(member text, its rows, start) of the coordinated entities of a query: the conjuncts
    (outside quotes and printed names, inside the lookup clauses) the identity index names
    exactly by a printed label word, adjacent conjuncts naming the same rows merged; at least two
    members with disjoint rows that together are the rows the whole question names, the plan's
    own rows among them - else none."""
    clauses = clauses or [Clause(0, 0, len(full), full, "structured", True)]
    quoted = [m.span() for m in _QUOTED.finditer(full)]
    cuts = [
        m.span()
        for m in _CONJUNCTION.finditer(full)
        if not any(a < m.start() < b for a, b in quoted)
        and not _inside_name(planner, ds, full, m.span())
    ]
    parts: list[tuple[int, int]] = []
    pos = 0
    for a, b in cuts:
        parts.append((pos, a))
        pos = b
    parts.append((pos, len(full)))
    named: list[tuple[str, set[str], int]] = []
    for a, b in parts:
        text = full[a:b].strip()
        exact = set(_identify(planner, ds, text)[0]) if text else set()
        if exact and not (_owned(clauses, a, b) and _named_by_label(planner, ds, text, exact)):
            exact = set()
        if named and exact and exact & named[-1][1]:
            prev = named[-1]
            named[-1] = (full[prev[2] : b].strip(), prev[1] | exact, prev[2])
        elif exact:
            named.append((text, exact, a))
    union = set().union(*(m[1] for m in named)) if named else set()
    whole = set(_identify(planner, ds, full, full)[0])
    if len(named) < 2 or not set(rows) <= union or union != whole:
        return []
    for i, x in enumerate(named):
        if any(x[1] & y[1] for y in named[i + 1 :]):
            return []
    return [(t, sorted(r), start) for t, r, start in named]


def split_coordinated(
    plan: StructuredQueryPlan,
    schema: TableSchema,
    planner: Planner | None = None,
    full: str = "",
    clauses: list[Clause] | None = None,
) -> list[tuple[StructuredQueryPlan, str | None]]:
    """Coordinated entities as one child plan per member (I8): places («کهنوج و آقاجاری») by
    their resolved names, every other entity dimension by :func:`coordinated_members`. A
    qualifier value goes to the nearest member before it; other constraints are shared. Other
    plans are returned unchanged."""
    if schema.family != "place":
        if planner is None:
            return [(plan, None)]
        return _split_rows(plan, schema, planner, full, clauses)
    ent = next((e for e in plan.entities if e.kind == "place" and e.status == "resolved"), None)
    if ent is None or len(ent.ids) < 2:
        return [(plan, None)]
    out: list[tuple[StructuredQueryPlan, str | None]] = []
    for member in ent.ids:
        child = deepcopy(plan)
        child.entities = [
            EntityRef("place", member, "resolved", [member])
            if e is not None and e.kind == "place" and e.status == "resolved"
            else e
            for e in child.entities
        ]
        child.notes.append(f"coordinated member «{member}» of «{ent.text}»")
        out.append((child, member))
    return out


def _split_rows(
    plan: StructuredQueryPlan,
    schema: TableSchema,
    planner: Planner,
    full: str,
    clauses: list[Clause] | None,
) -> list[tuple[StructuredQueryPlan, str | None]]:
    from mabhas19.retrieval.admission import _constraint_span  # noqa: PLC0415

    if plan.target not in planner.identity or plan.operation != "direct" or plan.computation:
        return [(plan, None)]  # a refused calculation stays one refusal
    ent = next(
        (e for e in plan.entities if e.status in ("resolved", "ambiguous") and e.kind != "place"),
        None,
    )
    if ent is None or _BOTH.search(full):
        return [(plan, None)]  # «هم X هم Y» is one subject (I5), never coordination
    rows = list(plan.row_ids or []) if plan.row_ids else list(ent.ids)
    members = coordinated_members(planner, plan.target, full, rows, clauses)
    if not members:
        return [(plan, None)]
    starts = [start for _, _, start in members]
    ends = [*starts[1:], len(full)]
    segments = [_clause_values(full[a:b]) for a, b in zip(starts, ends, strict=True)]

    def owner(c: Constraint) -> int | None:
        """The member a qualifier value belongs to (the nearest one before it); None: shared."""
        d = schema.dim(c.dimension)
        qualifier = (d is not None and d.qualifier) or c.dimension.startswith("condition:")
        span = _constraint_span(full, c) if qualifier else None
        if span is None:
            return None
        before = [i for i, s in enumerate(starts) if s <= span[0]]
        return before[-1] if before else 0

    def values_for(c: Constraint, i: int) -> list[str] | None:
        """A categorical constraint's values for member i (None: shared, unchanged)."""
        if c.dimension not in _CATEGORY_DIMS or not all(seg[c.dimension] for seg in segments):
            return None
        mine = {v.upper() for v in segments[i][c.dimension]}
        return [v for v in c.values if str(v).upper() in mine]

    out: list[tuple[StructuredQueryPlan, str | None]] = []
    for i, (text, member_rows, _) in enumerate(members):
        child = deepcopy(plan)
        child.row_ids = list(member_rows)
        child.entities = [
            replace(e, ids=list(member_rows), status="resolved")
            if e is ent or (e.kind == ent.kind and e.text == ent.text)
            else e
            for e in child.entities
        ]
        kept = []
        for c in child.constraints:
            if owner(c) not in (None, i):
                continue
            vals = values_for(c, i)
            kept.append(c if vals is None else replace(c, values=vals))
        child.constraints = [c for c in kept if c.values or c.interval is not None or c.ratio]
        child.notes.append(f"coordinated member «{text}» ({len(member_rows)} printed rows)")
        out.append((child, text))
    return out


# ------------------------------------------------------------------ the stage


def evaluate(  # noqa: PLR0912, PLR0915
    planner: Planner,
    store: StructuredStore,
    cands: list[Candidate],
    full: str,
    clauses: list[Clause],
) -> Ledger:
    """Check every candidate together; refuse or drop the admitted ones that violate I1-I7, fix
    each executable plan's requested output (named or the schema default) and record the
    admitted unit and identity-family evidence for certification."""
    from mabhas19.retrieval.certification import printed_levels  # noqa: PLC0415

    schemas = store.schemas
    ledger = Ledger(full, clauses, polarity.corrections(full))
    rejected = rejected_values(full, ledger.corrections)
    # I6 first: it decides which stated values each plan keeps (corrections, then clauses)
    for cand in cands:
        if not cand.admitted:
            continue
        cand.admission.rejected_values = {d: sorted(v) for d, v in rejected.items()}
        fixed, dropped = correction_ownership(cand.plan, rejected)
        apply_constraints(cand, fixed)
        cand.plan.notes += dropped
        ledger.adjustments += [
            {"table": cand.target, "invariant": "I6", "detail": n} for n in dropped
        ]
        if any(c.state == "unresolved" for c in fixed.values()):
            gate = validate(cand.plan, schemas[cand.target])
            cand.violations.append(
                Violation(cand.target, "I6", "rejected_value", "; ".join(dropped), cand.member)
            )
            if gate is not None:
                cand.outcome = PlanOutcome(gate.code, gate.detail, "rejected_value")
            continue
        new, notes = clause_precedence(cand.plan, owned_by(clauses, full, anchors_of(cand, full)))
        apply_constraints(cand, new)
        cand.plan.notes += notes
        ledger.adjustments += [
            {"table": cand.target, "invariant": "I6", "detail": n} for n in notes
        ]
        if any(c.state == "unresolved" for c in new.values()):
            gate = validate(cand.plan, schemas[cand.target])
            cand.violations.append(
                Violation(cand.target, "I6", "clause_precedence", "; ".join(notes), cand.member)
            )
            if gate is not None:
                cand.outcome = PlanOutcome(gate.code, gate.detail, "clause_precedence")
    spans = [s for cand in cands for s in candidate_spans(cand, full, clauses, schemas)]
    # I1: an output occurrence overlapping a constraint (or qualifier/exclusion) is discarded
    for a, b in role_conflicts(spans):
        out = a if a.role == "output" else b
        other = b if out is a else a
        if out.state == "grounded":
            out.state = "discarded"
            ledger.violations.append(
                Violation(
                    out.table,
                    "I1",
                    "role_conflict",
                    f"«{out.text}» grounds {other.role} {other.ref} of "
                    f"{other.table.split('.')[-1]}; it does not establish the output",
                )
            )
    ledger.spans = spans
    whole = [Clause(0, 0, len(full), full, "structured", True)]
    all_intents = unit_intents(full, whole, frozenset(planner.vocab_keys()))
    ledger.unit_intents = all_intents
    elsewhere = named_rows_words(planner, full)
    for cand in cands:
        if not cand.admitted or cand.outcome is not None:
            continue
        schema = schemas[cand.target]
        plan = cand.plan
        foreign = foreign_questions(full, anchors_of(cand, full))
        intents = [u for u in all_intents if u.attached or requested_here(foreign, u.start, u.end)]
        default = default_output(schema)
        named = established_outputs(spans, cand.target, clauses) | {
            g.value
            for g in cand.admission.groundings
            if g.component == "output" and g.basis != "default" and g.span is None
        }
        requested = plan.requested_field
        if plan.operation == "direct":
            # I2 on the plan as proposed
            stated = concept_conflict(plan)
            if stated:
                cand.violations.append(
                    Violation(
                        cand.target,
                        "I2",
                        "output_is_stated_constraint",
                        f"the lookup states {stated} and asks for it",
                        cand.member,
                    )
                )
            effective = requested if requested in named else default
            if effective not in named:
                effective = unit_named_output(intents, schema) or effective
            output = schema.output(effective or "")
            numeric = any(c.interval is not None for c in plan.constraints if c.state == "applied")
            unit_violation, unit_named = unit_verdict(intents, output, numeric)
            if unit_violation is not None:
                cand.violations.append(
                    replace(unit_violation, table=cand.target, member=cand.member)
                )
            q = quantity_violation(
                cand,
                store,
                planner,
                full,
                clauses=clauses,
                foreign=foreign,
                spans=spans,
                intents=intents,
                named_elsewhere=elsewhere,
            )
            if q is not None:
                cand.violations.append(q)
            quantity = output is not None and output.kind in ("number", "interval")
            if quantity and effective not in named and not unit_named:
                slot = output_slot(cand.admission, cand.text)
                if slot:
                    cand.violations.append(
                        Violation(
                            cand.target,
                            "I3",
                            "unsupported_output_slot",
                            f"«{' '.join(slot)}» asks for a quantity the table does not print "
                            f"(its output is {effective})",
                            cand.member,
                        )
                    )
            if effective != requested and effective is not None:
                plan.notes.append(
                    f"requested output «{requested}» is not established by the question; the "
                    f"table's own output {effective} applies"
                )
                plan.requested_field = effective
            cand.admission.output = effective
            cand.admission.unit = output.unit if output is not None else None
            printed_unit = units.parse(output.unit) if output is not None and output.unit else None
            cand.admission.requested_unit = next(
                (
                    u.text
                    for u in intents
                    if not u.attached
                    and u.unit is not None
                    and u.unit.num
                    and printed_unit is not None
                    and units.verdict(u.unit, printed_unit) == units.GROUND
                ),
                None,
            )
        elif plan.operation in ("whole_table", "existence"):
            cand.admission.output = requested or default
            output = schema.output(cand.admission.output or "")
            cand.admission.unit = output.unit if output is not None else None
        # I10
        owned = owned_values(owned_by(clauses, full, anchors_of(cand, full)), full)
        cand.violations += scope_violations(cand, schema, owned, rejected)
        # I5
        foreign_phrases, required = phrase_accounting(planner, store, cand)
        cand.admission.required_phrases = required
        if foreign_phrases:
            cand.violations.append(
                Violation(
                    cand.target,
                    "I5",
                    "qualifier_family_conflict",
                    f"{foreign_phrases} is printed for other rows of the table, not for the "
                    f"admitted {cand.admission.entity_rows[:3]}",
                    cand.member,
                    "ambiguous_entity",
                )
            )
        both = one_subject_two_identities(planner, cand, full)
        if both and cand.member is None:
            cand.violations.append(
                Violation(
                    cand.target,
                    "I5",
                    "one_subject_two_identities",
                    f"one subject is described as «{both[0]}» and as «{both[1]}», two printed "
                    "rows; which one governs is not stated",
                    cand.member,
                    "ambiguous_entity",
                )
            )
        conflict, consistent = family_conflict(planner, cand, full)
        if conflict:
            cand.violations.append(
                Violation(
                    cand.target,
                    "I5",
                    "identity_family_conflict",
                    f"other identity words name {conflict[:3]}, which share no printed level with "
                    f"the admitted {cand.admission.entity_rows[:3]}",
                    cand.member,
                )
            )
        elif consistent:
            rec = _records(store, cand.target)
            cand.admission.family_levels = sorted(
                {
                    lv
                    for r in consistent
                    if r in rec
                    for lv in printed_levels(store, cand.target, rec[r])
                }
            )
        for v in cand.violations:
            ledger.violations.append(v)
        first = next(
            (v for inv in _PRECEDENCE for v in cand.violations if v.invariant == inv), None
        )
        if first is not None and cand.outcome is None:
            cand.outcome = _outcome(first)
    # I7: ownership over every candidate (refused ones included, so that a refused table's
    # values cannot fall through to a weaker one; an unowned plan is not what was asked)
    for cand in cands:
        if not cand.admitted:
            continue
        if unowned(cand, cands, spans, clauses):
            cand.owned = False
            v = Violation(
                cand.target,
                "I7",
                "unowned_table",
                "its only anchors are spans a candidate table with a stronger anchor grounds",
                cand.member,
            )
            cand.violations.append(v)
            ledger.violations.append(v)
    ledger.verdicts = [
        {
            "table": c.target,
            "member": c.member,
            "admitted": c.admitted,
            "executable": c.executable,
            "owned": c.owned,
            "outcome": c.outcome.code if c.outcome else (c.gate.code if c.gate else None),
            "reason": c.outcome.reason
            if c.outcome
            else (c.admission.outcome.reason if c.admission.outcome else None),
            "output": c.admission.output,
            "unit": c.admission.unit,
            "requested_unit": c.admission.requested_unit,
            "family_levels": c.admission.family_levels,
            "invariants": [v.invariant for v in c.violations],
        }
        for c in cands
    ]
    return ledger


def _outcome(v: Violation) -> PlanOutcome:
    code = (
        v.code
        or {
            "I2": "unresolved_constraint",
            "I3": "unsupported_output",
            "I4": "unsupported_calculation",
            "I5": "ambiguous_entity",
            "I6": "unresolved_constraint",
            "I8": "ambiguous_entity",
            "I10": "invalid_constraint",
        }[v.invariant]
    )
    return PlanOutcome(code, v.detail, v.reason)


def _records(store: StructuredStore, ds: str) -> dict[str, dict[str, Any]]:
    from mabhas19.retrieval.schema import APP03, APP07  # noqa: PLC0415

    if ds == APP07:
        return {str(r["id"]): r for r in store.app07}
    if ds == APP03:
        # Appendix 3 identity rows are occupancies
        return {str(o["id"]): o for o in store.occupancy_index}
    t = store.main.get(ds)
    return {str(r["record_id"]): r for r in t.records} if t is not None else {}


__all__ = [
    "CONTRACTS",
    "COVERAGE_VERSION",
    "INVARIANTS",
    "LEDGER_KEYS",
    "LEDGER_SCHEMA",
    "Candidate",
    "Clause",
    "Ledger",
    "Span",
    "UnitIntent",
    "Violation",
    "clauses_of",
    "evaluate",
    "split_coordinated",
]
