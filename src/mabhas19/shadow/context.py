"""The deterministic candidate context (sent to the model) and facts (kept by the verifier).

Both are read from one production run (:meth:`Retriever.retrieve`) and the deterministic
parsers; nothing is computed twice differently and nothing is inferred:

* **clauses** - the router's clause partition located in the normalized query
  (:func:`coverage.clauses_of`); one whole-query clause when it has one clause or a clause
  cannot be located;
* **entity spans** - every query span that names a printed row of a candidate table in the
  identity index (``exact`` / ``head_only``) or names it only with grammatical homographs
  (``homograph_only``), with the candidate rows' IDs and printed labels;
* **tables** - the router's candidate tables (selected, anchored in a clause, explicitly
  referenced, or naming an unknown entity) with their schema outputs and printed units;
* **values** - the independent stated-value scanner of I11 (ratings, climate classes, groups,
  years, elapsed years) and every stated number (:func:`numeric.find_quantities`);
* **units**, **operations**, **polarity** (exclusions, printed negative forms, rejected
  readings of corrections) and **coordination** members of the production plans.

The context carries no regulatory value: labels and titles have their digits masked, units are
the schema's printed unit names, and every other text is a slice of the query itself.
:class:`DeterministicFacts` (never sent) holds what the production run actually did per table:
plan operations, requested outputs, resolved entity IDs, grounded anchor / entity / constraint
spans, checked claims, returned rows and typed outcomes.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field, replace
from functools import cache
from typing import TYPE_CHECKING, Any

from mabhas19.retrieval import coverage, numeric, operations, polarity, units
from mabhas19.retrieval.entities import _ETOKEN, GRAMMATICAL_HOMOGRAPHS, MEANING_PREFIXES
from mabhas19.retrieval.identity import query_tokens, row_label
from mabhas19.retrieval.schema import APP03
from mabhas19.text.normalize import normalize_for_search

if TYPE_CHECKING:
    from mabhas19.retrieval.hybrid import Retriever
    from mabhas19.retrieval.identity import IdentityIndex
    from mabhas19.retrieval.models import RetrievalResult
    from mabhas19.retrieval.structured import StructuredStore

CONTEXT_VERSION = "sh0-context-2"  # + same-length personal-data redaction
MAX_CANDIDATES = 12  # candidate rows listed per entity span (exact ones first)
VALUE_DIMENSIONS = ("rating", "climate_class", "group", "year", "number")
_DIGITS = re.compile(r"\d+(?:[.٫/]\d+)*")
_ANCHOR_ROLES = frozenset({"entity", "output", "table_target"})

Pair = tuple[int, int]


@dataclass(frozen=True)
class ClauseItem:
    index: int
    start: int
    end: int
    text: str
    router_kind: str | None  # structured | both | prose | None


@dataclass(frozen=True)
class EntityCandidate:
    id: str  # «<table>|<row>»
    table: str
    row: str
    label: str  # printed label, digits masked
    status: str  # exact | head_only | homograph_only


@dataclass(frozen=True)
class EntitySpan:
    index: int
    start: int
    end: int
    text: str
    homograph: bool  # every word is a grammatical homograph
    candidates: tuple[EntityCandidate, ...]
    truncated: bool = False


@dataclass(frozen=True)
class OutputItem:
    name: str
    unit: str | None  # the schema's printed unit name


@dataclass(frozen=True)
class TableItem:
    id: str
    title: str  # digits masked
    family: str
    selected: bool  # in the router's decision
    outputs: tuple[OutputItem, ...]


@dataclass(frozen=True)
class ValueItem:
    index: int
    start: int
    end: int
    text: str
    dimension: str  # rating | climate_class | group | year | number
    value: str  # comparable form («C», «3B», «1», «elapsed:3», the number as written)


@dataclass(frozen=True)
class UnitItem:
    index: int
    start: int
    end: int
    text: str
    canonical: str  # atoms (units.Unit.describe)
    attached: bool  # the unit of a stated number, not a request


@dataclass(frozen=True)
class OperationItem:
    index: int
    start: int
    end: int
    text: str
    kind: str


@dataclass(frozen=True)
class PolarityItem:
    index: int
    start: int
    end: int
    text: str
    kind: str  # exclusion | negative_form | correction_rejected


@dataclass(frozen=True)
class MemberItem:
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class CoordinationItem:
    table: str
    members: tuple[MemberItem, ...]


@dataclass(frozen=True)
class Redaction:
    start: int
    end: int
    kind: str  # email | url | long_number


@dataclass(frozen=True)
class CandidateContext:
    """Everything the proposer may refer to (by index or ID); sent to the model."""

    version: str
    query: str  # the normalized query: every offset refers to it
    clauses: tuple[ClauseItem, ...]
    tables: tuple[TableItem, ...]
    entities: tuple[EntitySpan, ...]
    values: tuple[ValueItem, ...]
    units: tuple[UnitItem, ...]
    operations: tuple[OperationItem, ...]
    polarity: tuple[PolarityItem, ...]
    coordination: tuple[CoordinationItem, ...]
    aligned: bool  # query tokens were aligned to offsets (else no entity span is offered)
    redactions: tuple[Redaction, ...] = ()  # personal data masked in every text (same length)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def table(self, table_id: str) -> TableItem | None:
        return next((t for t in self.tables if t.id == table_id), None)

    def candidate_ids(self) -> frozenset[str]:
        return frozenset(c.id for e in self.entities for c in e.candidates)

    def clause_of(self, start: int, end: int) -> int | None:
        """The clause holding the span (None when it lies between clauses)."""
        return next((c.index for c in self.clauses if c.start <= start and end <= c.end), None)


@dataclass(frozen=True)
class TableFacts:
    """What the production run did on one table (never sent to the model)."""

    table: str
    operations: tuple[str, ...]  # plan operations (one per plan or coordinated child)
    requested: tuple[str | None, ...]  # requested outputs of those plans
    entity_ids: frozenset[str]  # resolved entity / row IDs
    entity_states: tuple[str, ...]
    anchors: tuple[Pair, ...]  # grounded entity / output / table-target spans
    entity_spans: tuple[Pair, ...]  # grounded entity spans
    applied: tuple[Pair, ...]  # grounded constraint spans the plan applies
    checked: tuple[Pair, ...]  # grounded constraint spans the plan checks as a claim
    rows: tuple[str, ...]
    outcomes: tuple[tuple[str, str], ...]  # (outcome, reason) of typed no-matches
    members: tuple[str, ...]
    # (dimension, value) of the category values the plans apply / check (the span of a
    # corrected constraint may point at the rejected reading; its values do not)
    applied_values: frozenset[tuple[str, str]] = frozenset()
    checked_values: frozenset[tuple[str, str]] = frozenset()

    def applies(self, item: ValueItem) -> bool:
        if item.dimension == "number":
            return any(s <= item.start and item.end <= e for s, e in self.applied)
        return (item.dimension, item.value) in self.applied_values

    def checks(self, item: ValueItem) -> bool:
        if item.dimension == "number":
            return any(s <= item.start and item.end <= e for s, e in self.checked)
        return (item.dimension, item.value) in self.checked_values


@dataclass(frozen=True)
class DeterministicFacts:
    route: str
    tables: tuple[str, ...]  # the router's decision
    per_table: dict[str, TableFacts] = field(default_factory=dict)
    query: str = ""  # the unredacted normalized query (never sent; offsets equal the context's)

    def rows(self) -> frozenset[str]:
        return frozenset(r for f in self.per_table.values() for r in f.rows)


def canonical_json(obj: Any) -> str:
    """One byte-stable serialization (sorted keys, no ASCII escaping, compact)."""
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def mask_digits(text: str) -> str:
    """A printed label or title without its numbers (conditions and thresholds stay out of the
    model's input; identifiers keep theirs)."""
    return _DIGITS.sub("<n>", normalize_for_search(text))


# ---------------------------------------------------------------- builder


def build(
    retriever: Retriever, production: RetrievalResult
) -> tuple[CandidateContext, DeterministicFacts]:
    """The candidate context and the deterministic facts of one production run."""
    full = production.normalized_query
    facts = _facts(production)
    table_ids = _candidate_tables(production)
    tokens = _aligned_tokens(full)
    entities = _entity_spans(retriever, table_ids, full, tokens) if tokens is not None else ()
    ctx = CandidateContext(
        version=CONTEXT_VERSION,
        query=full,
        clauses=_clauses(production),
        tables=tuple(_table_item(retriever, t, t in production.route.tables) for t in table_ids),
        entities=entities,
        values=value_items(full),
        units=tuple(
            UnitItem(i, e.start, e.end, full[e.start : e.end], e.unit.describe(), e.attached)
            for i, e in enumerate(units.find(full))
        ),
        operations=tuple(
            OperationItem(i, o.start, o.end, full[o.start : o.end], o.kind)
            for i, o in enumerate(operations.operations(full))
        ),
        polarity=polarity_items(full),
        coordination=_coordination(full, facts),
        aligned=tokens is not None,
    )
    return redact_context(ctx), facts


# ---------------------------------------------------------------- redaction

# personal data never sent: e-mail addresses, URLs and digit runs of 8 or more (phone, national
# ID, postal code, account numbers; no Mabhas 19 value or code is that long - WMO codes have 6)
_PERSONAL = (
    ("email", re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")),
    ("url", re.compile(r"(?:https?://|www\.)\S+")),
    ("long_number", re.compile(r"\+?\d(?:[\d\s-]{6,}\d)")),
)


def redactions(text: str) -> tuple[Redaction, ...]:
    found: list[Redaction] = []
    for kind, pattern in _PERSONAL:
        for m in pattern.finditer(text):
            if kind == "long_number" and sum(ch.isdigit() for ch in m[0]) < 8:
                continue
            if not any(r.start < m.end() and m.start() < r.end for r in found):
                found.append(Redaction(m.start(), m.end(), kind))
    return tuple(sorted(found, key=lambda r: r.start))


def redact(text: str) -> str:
    """The text with personal data masked character by character (offsets are kept)."""
    chars = list(text)
    for r in redactions(text):
        for i in range(r.start, r.end):
            if not chars[i].isspace():
                chars[i] = "#"
    return "".join(chars)


def redact_context(ctx: CandidateContext) -> CandidateContext:
    """Every query slice of the context taken from the redacted query."""
    found = redactions(ctx.query)
    if not found:
        return ctx
    q = redact(ctx.query)

    def sl(item: Any) -> Any:
        return replace(item, text=q[item.start : item.end])

    return replace(
        ctx,
        query=q,
        clauses=tuple(sl(c) for c in ctx.clauses),
        entities=tuple(sl(e) for e in ctx.entities),
        values=tuple(
            replace(sl(v), value=q[v.start : v.end] if v.dimension == "number" else v.value)
            for v in ctx.values
        ),
        units=tuple(sl(u) for u in ctx.units),
        operations=tuple(sl(o) for o in ctx.operations),
        polarity=tuple(sl(x) for x in ctx.polarity),
        coordination=tuple(
            replace(c, members=tuple(sl(m) for m in c.members)) for c in ctx.coordination
        ),
        redactions=found,
    )


def _candidate_tables(production: RetrievalResult) -> list[str]:
    signals = production.route.signals
    out = list(production.route.tables)
    for c in signals.get("clauses") or []:
        out += list(c.get("tables") or [])
    out += [str(u["dataset"]) for u in signals.get("unknown_entities") or []]
    out += [n.dataset for n in production.no_match]
    out += [t.dataset for t in production.tables]
    return list(dict.fromkeys(out))


def _clauses(production: RetrievalResult) -> tuple[ClauseItem, ...]:
    full = production.normalized_query
    report = list(production.route.signals.get("clauses") or [])
    found = coverage.clauses_of(full, report)
    if len(found) < 2 or any(c.start < 0 for c in found):
        return (ClauseItem(0, 0, len(full), full, found[0].kind if len(found) == 1 else None),)
    return tuple(ClauseItem(c.index, c.start, c.end, c.text, c.kind) for c in found)


def _table_item(retriever: Retriever, table_id: str, selected: bool) -> TableItem:
    schema = retriever.store.schemas[table_id]
    return TableItem(
        table_id,
        mask_digits(schema.title),
        schema.family,
        selected,
        tuple(OutputItem(o.name, o.unit) for o in schema.outputs),
    )


# ---------------------------------------------------------------- entity spans


def _aligned_tokens(full: str) -> list[tuple[str, int, int]] | None:
    """The identity index's query tokens (``identity.query_tokens``) with their offsets in the
    query: meaning-bearing prefixes merged onto the next word. None when they cannot be aligned
    exactly (no entity span is then offered: fail safe)."""
    if normalize_for_search(full) != full:
        return None
    raw = [(m[0], m.start(), m.end()) for m in _ETOKEN.finditer(full)]
    out: list[tuple[str, int, int]] = []
    i = 0
    while i < len(raw):
        t, s, e = raw[i]
        if i + 1 < len(raw) and t in MEANING_PREFIXES:
            out.append((t + raw[i + 1][0], s, raw[i + 1][2]))
            i += 2
            continue
        out.append((t, s, e))
        i += 1
    return out if [t for t, _, _ in out] == query_tokens(full) else None


def _runs(positions: Iterable[int]) -> list[tuple[int, int]]:
    """Contiguous runs of token positions as (first, last)."""
    out: list[tuple[int, int]] = []
    for p in sorted(set(positions)):
        if out and p == out[-1][1] + 1:
            out[-1] = (out[-1][0], p)
        else:
            out.append((p, p))
    return out


def _entity_spans(
    retriever: Retriever,
    table_ids: Sequence[str],
    full: str,
    tokens: list[tuple[str, int, int]],
) -> tuple[EntitySpan, ...]:
    toks = [t for t, _, _ in tokens]
    found: dict[Pair, list[EntityCandidate]] = {}
    for table_id in table_ids:
        idx = retriever.planner.identity.get(table_id)
        if idx is None:
            continue
        for row in idx.by_row:
            status = _status(idx, row, toks)
            if status is None:
                continue
            label = _label(retriever, table_id, row)
            cand = EntityCandidate(f"{table_id}|{row}", table_id, row, label, status)
            for first, last in _runs(idx.named_positions(row, toks)):
                span = (tokens[first][1], tokens[last][2])
                if cand not in found.setdefault(span, []):
                    found[span].append(cand)
    order = {"exact": 0, "head_only": 1, "homograph_only": 2}
    out = []
    for i, (span, cands) in enumerate(sorted(found.items())):
        cands.sort(key=lambda c: (order[c.status], c.id))
        text = full[span[0] : span[1]]
        words = query_tokens(text)
        out.append(
            EntitySpan(
                i,
                span[0],
                span[1],
                text,
                bool(words) and all(w in GRAMMATICAL_HOMOGRAPHS for w in words),
                tuple(cands[:MAX_CANDIDATES]),
                len(cands) > MAX_CANDIDATES,
            )
        )
    return tuple(out)


def _status(idx: IdentityIndex, row: str, toks: list[str]) -> str | None:
    named = idx.named(row, toks)
    if named.status in ("exact", "head_only"):
        return named.status
    words = idx.named_words(row, toks)
    if words and all(w in GRAMMATICAL_HOMOGRAPHS for w in words):
        return "homograph_only"
    return None


@cache
def _record_map(store: StructuredStore) -> dict[str, dict[str, Any]]:
    rows = [*store.app02, *store.app07, *(r for t in store.main.values() for r in t.records)]
    return {str(r.get("id") or r.get("record_id")): r for r in rows}


def _label(retriever: Retriever, table_id: str, row: str) -> str:
    store = retriever.store
    if table_id == APP03:  # identity rows of Appendix 3 are occupancy types
        return mask_digits(str(store.occupancies.get(row, {}).get("label", "")))
    record = _record_map(store).get(row)
    return mask_digits(row_label(store, table_id, record)) if record is not None else ""


# ---------------------------------------------------------------- values, polarity


def value_items(full: str) -> tuple[ValueItem, ...]:
    """Stated values with their spans: the I11 scanner (``coverage.value_mentions``) read with
    offsets, plus every stated number."""
    found: list[tuple[int, int, str, str]] = []
    for m in coverage._LETTER.finditer(full):
        found.append((m.start(1), m.end(1), "rating", m[1].upper()))
    for m in coverage._CLIMATE_CODE.finditer(full):
        found.append((m.start(), m.end(), "climate_class", f"{m[1]}{m[2].upper()}"))
    for m in coverage._GROUP.finditer(full):
        found.append((m.start(1), m.end(1), "group", m[1]))
        pos = m.end()
        while g := coverage._GROUP_MORE.match(full, pos):
            found.append((g.start(1), g.end(1), "group", g[1]))
            pos = g.end()
    for m in coverage._YEAR.finditer(full):
        found.append((m.start(1), m.end(1), "year", coverage._ORD.get(m[1], m[1])))
        pos = m.end()
        while g := coverage._YEAR_MORE.match(full, pos):
            found.append((g.start(1), g.end(1), "year", coverage._ORD.get(g[1], g[1])))
            pos = g.end()
    for m in coverage._YEAR_ADJ.finditer(full):
        found.append((m.start(1), m.end(1), "year", coverage._ORD_ADJ[m[1]]))
    for m in coverage._ELAPSED.finditer(full):
        found.append((m.start(), m.end(), "year", f"elapsed:{coverage._CARD.get(m[1], m[1])}"))
    for q in numeric.find_quantities(full):
        found.append((q.start, q.end, "number", full[q.start : q.end]))
    items = sorted(set(found))
    return tuple(ValueItem(i, s, e, full[s:e], d, v) for i, (s, e, d, v) in enumerate(items))


def polarity_items(full: str) -> tuple[PolarityItem, ...]:
    """Exclusions, printed negative forms and rejected readings of corrections, located in the
    query (the polarity parser is authoritative: the proposal may only keep them)."""
    pol = polarity.parse(full)
    found: list[tuple[int, int, str]] = [
        (s.start, s.end, "exclusion") for s in pol.exclusions if 0 <= s.start < s.end
    ]
    pos = 0
    for form in pol.negative_forms:
        at = full.find(form.text, pos)
        if at >= 0:
            found.append((at, at + len(form.text), "negative_form"))
            pos = at + len(form.text)
    found += [(c.rejected.start, c.rejected.end, "correction_rejected") for c in pol.corrections]
    items = sorted(set(found))
    return tuple(PolarityItem(i, s, e, full[s:e], k) for i, (s, e, k) in enumerate(items))


# ---------------------------------------------------------------- facts


def _pairs(spans: Iterable[dict[str, Any]]) -> tuple[Pair, ...]:
    return tuple(sorted({(int(s["span"][0]), int(s["span"][1])) for s in spans}))


def _facts(production: RetrievalResult) -> DeterministicFacts:
    spans = [s for s in production.coverage.get("spans") or [] if s.get("state") == "grounded"]
    verdicts = list(production.coverage.get("verdicts") or [])
    per: dict[str, TableFacts] = {}
    tables = list(
        dict.fromkeys(
            [p["target"] for p in production.plans]
            + [h.dataset for h in production.structured]
            + [n.dataset for n in production.no_match]
        )
    )
    for t in tables:
        plans = [p for p in production.plans if p["target"] == t]
        mine = [s for s in spans if s["table"] == t]
        checked_dims = {
            c["dimension"] for p in plans for c in p["constraints"] if c["state"] == "checked"
        }
        applied_dims = {
            c["dimension"] for p in plans for c in p["constraints"] if c["state"] == "applied"
        }
        constraint = [s for s in mine if s["role"] == "constraint"]
        per[t] = TableFacts(
            table=t,
            operations=tuple(str(p["operation"]) for p in plans),
            requested=tuple(p.get("requested_field") for p in plans),
            entity_ids=frozenset(
                str(i)
                for p in plans
                for e in p["entities"]
                if e["status"] == "resolved"
                for i in e["ids"]
            ),
            entity_states=tuple(str(e["status"]) for p in plans for e in p["entities"]),
            anchors=_pairs(s for s in mine if s["role"] in _ANCHOR_ROLES),
            entity_spans=_pairs(s for s in mine if s["role"] == "entity"),
            applied=_pairs(s for s in constraint if s.get("ref") in applied_dims),
            checked=_pairs(s for s in constraint if s.get("ref") in checked_dims),
            rows=tuple(h.record_id for h in production.structured if h.dataset == t),
            outcomes=tuple(
                (str(n.outcome or ""), n.reason) for n in production.no_match if n.dataset == t
            ),
            members=tuple(
                str(v["member"]) for v in verdicts if v.get("table") == t and v.get("member")
            ),
            applied_values=_values_of(plans, "applied"),
            checked_values=_values_of(plans, "checked"),
        )
    return DeterministicFacts(
        production.route.route, tuple(production.route.tables), per, production.normalized_query
    )


def _values_of(plans: list[dict[str, Any]], state: str) -> frozenset[tuple[str, str]]:
    return frozenset(
        (str(c["dimension"]), coverage._value_key(v))
        for p in plans
        for c in p["constraints"]
        if c["state"] == state and c["dimension"] in VALUE_DIMENSIONS
        for v in c["values"] or []
    )


def _coordination(full: str, facts: DeterministicFacts) -> tuple[CoordinationItem, ...]:
    out = []
    for t, f in facts.per_table.items():
        members = []
        pos = 0
        for text in f.members:
            at = full.find(text, pos)
            if at < 0:
                at = full.find(text)
            if at >= 0:
                members.append(MemberItem(at, at + len(text), text))
                pos = at + len(text)
        if members:
            out.append(CoordinationItem(t, tuple(members)))
    return tuple(out)
