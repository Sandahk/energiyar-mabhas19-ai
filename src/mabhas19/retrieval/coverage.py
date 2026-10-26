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
* **I3 output slot** - for a table producing a quantity, the requested output is either named
  (an established output phrase or a requested unit equal to the printed unit), or the output
  slot - the unresolved words right before the named entity («**ظرفیت گرمایی ویژه** بتن …») -
  is empty and the schema default applies; an occupied slot is ``unsupported_output``;
* **I4 unit intent** - a requested unit («بر حسب X», «به واحد X», a standalone compound unit)
  or a foreign unit attached to a stated number must be the source's printed unit;
  otherwise ``unsupported_calculation`` (``unit_conversion``): no conversion exists;
* **I5 identity family consistency** - every identity-bearing span (rows named exactly but not
  admitted, rows whose printed head an unresolved word names) shares a printed level with the
  admitted rows; otherwise ``ambiguous_entity`` (``identity_family_conflict``). The printed
  levels of that evidence are handed to certification, which re-checks them on the row;
* **I6 clause precedence** - a dimension's values come from the clauses owning the lookup; values
  of the same dimension stated only in an explanatory clause are ignored when the lookup states
  the dimension, inherited when they are one unconflicted value, and unresolved otherwise;
* **I7 table ownership** - a plan executes only with an anchor of its own: an entity, an explicit
  reference, an established output, or a constraint span no candidate with a stronger anchor
  grounds. An unowned plan is dropped (it is not what the question asked; no no-match);
* **I8 coordinated entities** (:func:`split_coordinated`) - coordinated printed places become one
  child plan per member, each admitted, executed and certified against its own member.

I9 (output certification) lives in :mod:`mabhas19.retrieval.certification`.
"""

from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from mabhas19.retrieval import units
from mabhas19.retrieval.entities import keys
from mabhas19.retrieval.plan import (
    Constraint,
    EntityRef,
    PlanOutcome,
    StructuredQueryPlan,
    concept,
    validate,
)

if TYPE_CHECKING:
    from mabhas19.retrieval.admission import PlanAdmission
    from mabhas19.retrieval.planner import Planner
    from mabhas19.retrieval.schema import Output, TableSchema
    from mabhas19.retrieval.structured import StructuredStore

COVERAGE_VERSION = "coverage-1"
INVARIANTS = ("I1", "I2", "I3", "I4", "I5", "I6", "I7", "I8", "I9")
# roles a span may not carry at once (across every candidate plan of a query)
INCOMPATIBLE = frozenset(
    {
        frozenset({"output", "constraint"}),
        frozenset({"output", "exclusion"}),
        frozenset({"output", "qualifier"}),
    }
)
# the outcome reported for a refused plan, by first violated invariant in this order
_PRECEDENCE = ("I4", "I3", "I2", "I6", "I5")
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
# unit intent: an explicit request for a unit, or a compound unit not attached to a number
_UNIT_REQUEST = re.compile(r"(?:بر\s?حسب|به\s?واحد|با\s?واحد|در\s?واحد|به\s?یکای)\s+")
_LATIN_UNIT = re.compile(
    r"(?<![a-z0-9/])([a-z°%]+[23²³]?(?:\s?[/·.]\s?[a-z°]+[23²³]?)+)(?![a-z0-9/])"
)
_ATTACHED = re.compile(r"\d\s*$")
_PER = re.compile(r"(?<![ؠ-ۿ])بر(?![ؠ-ۿ])")


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

    def to_dict(self) -> dict[str, Any]:
        return {
            "table": self.table,
            "member": self.member,
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
    spans: list[Span] = field(default_factory=list)
    unit_intents: list[UnitIntent] = field(default_factory=list)
    violations: list[Violation] = field(default_factory=list)
    # I6 values dropped or inherited without a refusal (inspectable, not violations)
    adjustments: list[dict[str, str]] = field(default_factory=list)
    verdicts: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": COVERAGE_VERSION,
            "clauses": [
                {"index": c.index, "span": [c.start, c.end], "kind": c.kind, "owner": c.owner}
                for c in self.clauses
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


def owned_by(clauses: list[Clause], full: str, anchors: list[tuple[int, int]]) -> list[Clause]:
    """The clauses one plan owns: the router's lookup clauses, minus those in a *separate
    question* - a question sentence holding none of the plan's own anchors (entity, output,
    explicit reference) asks something of its own («فرق رده‌های A و B چیه؟ …»). A statement
    sentence stays shared context («ساختمان ما در اقلیم 3A است. …»)."""
    if not anchors:
        return clauses
    foreign = [
        (s, e)
        for s, e, question in sentences(full)
        if question and not any(s <= a < e for a, _ in anchors)
    ]
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


def default_output(schema: TableSchema) -> str | None:
    d = schema.direction("direct")
    return d.output if d is not None else (schema.outputs[0].name if schema.outputs else None)


# ------------------------------------------------------------------ I4


def unit_intents(
    full: str, clauses: list[Clause], vocabulary: frozenset[str] = frozenset()
) -> list[UnitIntent]:
    """Requested units in the clauses owning the lookup, and Latin units attached to numbers:
    explicit requests first, then Latin unit expressions, then Persian compounds."""
    out: list[UnitIntent] = []
    for found in (
        _requested_units(full, vocabulary),
        _latin_units(full),
        _compound_units(full),
    ):
        for u in found:
            if any(o.start < u.end and u.start < o.end for o in out):
                continue
            if u.attached or _owned(clauses, u.start, u.end):
                out.append(u)
    return out


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


def _latin_units(full: str) -> list[UnitIntent]:
    out = []
    for m in _LATIN_UNIT.finditer(full):
        parsed = units.parse(m[1])
        if parsed is not None:
            s, e = m.span(1)
            out.append(UnitIntent(s, e, m[1], parsed, bool(_ATTACHED.search(full[:s]))))
    return out


def _compound_units(full: str) -> list[UnitIntent]:
    """«… kWh بر متر مربع در سال»: the widest readable compound around each «بر» (a compound
    right after a number is that number's unit and no request)."""
    out = []
    for m in _PER.finditer(full):
        left = list(re.finditer(r"\S+", full[: m.start()]))[-3:]
        right = list(re.finditer(r"\S+", full[m.end() :]))[:5]
        spans = [
            (left[-nl].start(), m.end() + right[nr - 1].end())
            for nl in range(len(left), 0, -1)
            for nr in range(len(right), 0, -1)
        ]
        for s, e in spans:
            parsed = units.parse(full[s:e])
            if parsed is not None and parsed.num and parsed.den and not parsed.foreign:
                if not _ATTACHED.search(full[:s]):
                    out.append(UnitIntent(s, e, full[s:e].strip(), parsed))
                break
    return out


def unit_verdict(
    intents: list[UnitIntent], output: Output | None, numeric: bool
) -> tuple[str | None, bool]:
    """(violation detail or None, the requested unit names this output)."""
    printed = units.parse(output.unit) if output is not None and output.unit else None
    named = False
    for u in intents:
        if u.attached:
            # the unit of a stated number: only a foreign unit is a conversion request
            if numeric and u.unit is not None and u.unit.foreign:
                return f"a stated number is in «{u.text}», not a printed unit", False
            continue
        if u.unit is None:
            return f"the requested unit «{u.text}» is not a unit the source prints", False
        if printed is None:
            if u.unit.foreign:
                return f"the requested unit «{u.text}» is not printed for this table", False
            continue
        if not units.compatible(u.unit, printed):
            shown = output.unit if output is not None else ""
            return (
                f"the requested unit «{u.text}» is not the printed «{shown}»; no conversion is "
                "performed",
                False,
            )
        named = True
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
    _, head, _ = planner.identify(cand.target, cand.text, full)
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


def split_coordinated(
    plan: StructuredQueryPlan, schema: TableSchema
) -> list[tuple[StructuredQueryPlan, str | None]]:
    """Coordinated printed places («کهنوج و آقاجاری») as one child plan per member; every
    member is a separately resolved printed name, so a conjunction inside one printed name is
    never split. Other plans are returned unchanged."""
    if schema.family != "place":
        return [(plan, None)]
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
    ledger = Ledger(full, clauses)
    # I6 first: it decides which stated values each plan keeps
    for cand in cands:
        if not cand.admitted:
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
    intents = unit_intents(full, clauses, frozenset(planner.vocab_keys()))
    ledger.unit_intents = intents
    for cand in cands:
        if not cand.admitted or cand.outcome is not None:
            continue
        schema = schemas[cand.target]
        plan = cand.plan
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
            output = schema.output(effective or "")
            numeric = any(c.interval is not None for c in plan.constraints if c.state == "applied")
            detail, unit_named = unit_verdict(intents, output, numeric)
            if detail:
                cand.violations.append(
                    Violation(cand.target, "I4", "unit_conversion", detail, cand.member)
                )
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
            cand.admission.requested_unit = next(
                (u.text for u in intents if not u.attached and u.unit is not None), None
            )
        elif plan.operation in ("whole_table", "existence"):
            cand.admission.output = requested or default
            output = schema.output(cand.admission.output or "")
            cand.admission.unit = output.unit if output is not None else None
        # I5
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
    code = {
        "I2": "unresolved_constraint",
        "I3": "unsupported_output",
        "I4": "unsupported_calculation",
        "I5": "ambiguous_entity",
        "I6": "unresolved_constraint",
    }[v.invariant]
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
    "COVERAGE_VERSION",
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
