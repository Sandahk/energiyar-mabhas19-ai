"""Typed structured query plans and the fail-closed validation gate.

No structured record is returned from partially interpreted text: the planner first turns the
structured part of a query into a :class:`StructuredQueryPlan` for one target dataset, the gate
(:func:`validate`) checks it against the dataset's :class:`~mabhas19.retrieval.schema.TableSchema`,
and only a valid plan is executed. Every meaningful component of the query ends in a typed
state (constraint/qualifier/entity states, ``unresolved`` features); anything that cannot be
represented safely fails closed with one :class:`PlanOutcome` - the lookup is never broadened.

Outcome codes (``StructuredNoMatch.outcome``):

* ``missing_required_dimension`` - a dimension the value depends on is not stated;
* ``unresolved_constraint`` - a stated number, qualifier or comparison could not be interpreted;
* ``invalid_constraint`` - a stated value outside the table's printed domain (climate class,
  rating, group, year…);
* ``no_printed_category`` - interpreted constraints that no printed row/range satisfies;
* ``unsupported_reverse_lookup`` - the table defines no inversion for the asked direction;
* ``unsupported_output`` - the question asks for a quantity the table does not print;
* ``unsupported_calculation`` - totals, averages, ratios, interpolation, area products…;
* ``unknown_entity`` / ``ambiguous_entity`` - entity resolution failed or is not unique;
* ``contradiction`` - an executed row contradicted an applied constraint (internal safety net;
  zero rows are returned);
* ``certification_failed`` - a returned row failed independent result certification
  (:mod:`mabhas19.retrieval.certification`; zero rows are returned).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from mabhas19.retrieval.numeric import NumInterval
from mabhas19.retrieval.schema import TableSchema

OPERATIONS = ("direct", "reverse", "existence", "whole_table", "unsupported_calculation")
# every ledger entry ends in exactly one of these states
LEDGER_STATES = frozenset(
    {
        "applied",  # interpreted and used to select rows
        "checked",  # a claimed value compared with the identified row, not a filter
        "invalid",  # outside the printed domain
        "unresolved",  # not safely interpretable
        "contradicted",  # a qualifier that contradicts every candidate row
        "unsupported_by_table",  # the table has no such dimension (recorded, not a filter)
        "unsupported_calculation",  # part of a refused calculation
        "enumerated",  # an explicitly requested enumeration of printed alternatives
    }
)
OUTCOMES = (
    "missing_required_dimension",
    "unresolved_constraint",
    "invalid_constraint",
    "no_printed_category",
    "unsupported_reverse_lookup",
    "unsupported_output",
    "unsupported_calculation",
    "unknown_entity",
    "ambiguous_entity",
    "contradiction",
    "certification_failed",
)


@dataclass
class Constraint:
    """One stated constraint on a dimension (or, in a reverse lookup, on an output value)."""

    dimension: str
    values: list[str] = field(default_factory=list)  # category / identity values
    interval: NumInterval | None = None  # numeric constraints
    ratio: tuple[str, str] | None = None
    source: str = ""  # the query text it was read from
    state: str = "applied"
    detail: str = ""
    reason: str = ""  # specific typed reason when invalid (e.g. ``rating_not_in_table``)
    # provenance: the parser rule that read it («cue», «coordinated_bound», «printed_condition»,
    # «claim», «label_condition», «rating_letter», «rating_description», «coordinated_values»)
    # and the curated source blocks attesting a descriptive wording
    rule: str = ""
    attested_in: list[str] = field(default_factory=list)

    def describe(self) -> str:
        if self.interval is not None:
            return f"{self.dimension} {self.interval.describe()}"
        if self.ratio is not None:
            return f"{self.dimension} = {self.ratio[0]}:{self.ratio[1]}"
        return f"{self.dimension} ∈ {self.values}"


@dataclass
class Qualifier:
    """A meaning-bearing word that distinguishes rows («غیر چگالشی», «بدون الیاف», «تک اثره»)."""

    word: str
    polarity: str  # positive | negative
    state: str = "applied"  # applied | contradicted | unresolved
    detail: str = ""


@dataclass
class EntityRef:
    kind: str  # place | occupancy | material | label
    text: str
    status: str  # resolved | unknown | ambiguous | none
    ids: list[str] = field(default_factory=list)
    detail: str = ""


@dataclass
class Derivation:
    """A constraint obtained from another curated record (e.g. a climate class from a city)."""

    kind: str
    inputs: list[str]  # record IDs used
    output: str
    issues: list[str] = field(default_factory=list)


@dataclass
class Enumeration:
    """An explicitly requested enumeration of printed alternatives: every printed member of an
    enumerable dimension within a resolved, source-defined scope, each result carrying its own
    value of that dimension. Allowed only when the schema marks the dimension enumerable, the
    question asks for the alternatives (never merely because the dimension is absent) and no
    stated constraint selects one member."""

    dimension: str
    basis: str  # explicit_request
    request: str  # the words that ask for the alternatives
    parent: str = ""  # the resolved scope (entity / row label) the members belong to
    members: list[str] = field(default_factory=list)  # printed values within that scope


@dataclass
class StructuredQueryPlan:
    target: str
    operation: str = "direct"
    requested_field: str | None = None
    entities: list[EntityRef] = field(default_factory=list)
    constraints: list[Constraint] = field(default_factory=list)
    qualifiers: list[Qualifier] = field(default_factory=list)
    derivations: list[Derivation] = field(default_factory=list)
    computation: str | None = None  # requested computation kind (never executed)
    unresolved: list[str] = field(default_factory=list)  # query features not interpreted
    enumerations: list[Enumeration] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)  # value-determining dimensions not stated
    row_ids: list[str] | None = None  # rows identified by entity/label resolution
    notes: list[str] = field(default_factory=list)
    explicit_reference: bool = False

    @property
    def enumerated(self) -> list[str]:
        return [e.dimension for e in self.enumerations]

    def constraint(self, dimension: str) -> Constraint | None:
        return next((c for c in self.constraints if c.dimension == dimension), None)

    def ledger(self) -> list[dict[str, Any]]:
        """One entry per stated constraint / qualifier / unresolved feature, with its state."""
        out: list[dict[str, Any]] = []
        refused = self.computation is not None
        for c in self.constraints:
            name = concept(c.dimension)
            value: Any = (
                c.interval.describe()
                if c.interval is not None
                else (f"{c.ratio[0]}:{c.ratio[1]}" if c.ratio else c.values)
            )
            out.append(
                {
                    "table": self.target,
                    "constraint": name,
                    "dimension": c.dimension,
                    "value": value,
                    "state": "unsupported_calculation" if refused else c.state,
                }
            )
        for q in self.qualifiers:
            out.append(
                {
                    "table": self.target,
                    "constraint": f"qualifier:{q.word}",
                    "value": f"{'-' if q.polarity == 'negative' else '+'}{q.word}",
                    "state": q.state,
                }
            )
        out += [
            {
                "table": self.target,
                "constraint": f"unresolved:{u}",
                "value": u,
                "state": "unresolved",
            }
            for u in self.unresolved
        ]
        out += [
            {
                "table": self.target,
                "constraint": f"enumeration:{e.dimension}",
                "value": e.members,
                "state": "enumerated",
            }
            for e in self.enumerations
        ]
        if self.computation:
            out.append(
                {
                    "table": self.target,
                    "constraint": "computation",
                    "value": self.computation,
                    "state": "unsupported_calculation",
                }
            )
        return out

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for c, src in zip(d["constraints"], self.constraints, strict=True):
            c["interval"] = src.interval.describe() if src.interval is not None else None
        return d


def concept(dimension: str) -> str:
    """The canonical constraint concept of a dimension (shared by every table's ledger)."""
    if dimension in ("min_r_value", "r_ratio"):
        return "r_value"
    if dimension in {"label:m2", "area"}:
        return "area"
    if dimension.startswith("degree_days"):
        return "degree_days"
    if dimension.startswith("condition:") and "چگالی" in dimension:
        return "density"
    return dimension


@dataclass
class PlanOutcome:
    code: str
    detail: str
    reason: str  # specific reason (kept stable for consumers, e.g. ``rating_not_in_table``)


def validate(plan: StructuredQueryPlan, schema: TableSchema) -> PlanOutcome | None:  # noqa: PLR0911, PLR0912
    """The gate between planning and execution: ``None`` = executable, else the typed outcome.
    Order matters only for reporting; any failing check blocks execution."""
    if plan.computation:
        return PlanOutcome(
            "unsupported_calculation",
            f"the question asks for a computed value ({plan.computation}); only printed values "
            "exist and no calculation is performed",
            "unsupported_calculation",
        )
    for e in plan.entities:
        if e.status == "unknown":
            return PlanOutcome(
                "unknown_entity", e.detail or f"no curated {e.kind} matches", f"unknown_{e.kind}"
            )
        if e.status == "ambiguous":
            return PlanOutcome(
                "ambiguous_entity",
                e.detail or f"several curated {e.kind}s fit",
                f"ambiguous_{e.kind}",
            )
    if plan.operation == "reverse":
        d = schema.direction("reverse")
        if d is None or not d.supported:
            note = d.note if d else "the table defines no reverse lookup"
            return PlanOutcome("unsupported_reverse_lookup", note, "unsupported_reverse_lookup")
    if plan.requested_field is not None and schema.output(plan.requested_field) is None:
        return PlanOutcome(
            "unsupported_output",
            f"the table prints no «{plan.requested_field}»",
            "unsupported_output",
        )
    for c in plan.constraints:
        if c.state == "invalid":
            return PlanOutcome(
                "invalid_constraint", c.detail or c.describe(), c.reason or "invalid_constraint"
            )
    unresolved = [c.describe() for c in plan.constraints if c.state == "unresolved"]
    unresolved += [q.word for q in plan.qualifiers if q.state in ("unresolved", "contradicted")]
    unresolved += plan.unresolved
    if unresolved:
        return PlanOutcome(
            "unresolved_constraint",
            f"not safely interpretable for this table: {unresolved}; nothing is broadened",
            "unresolved_constraint",
        )
    if plan.missing:
        return PlanOutcome(
            "missing_required_dimension",
            f"the value depends on {plan.missing}, which the question does not state, and the "
            "question does not ask for the printed alternatives; nothing is assumed",
            "missing_dimension",
        )
    if plan.operation in ("direct", "existence"):
        stated = {c.dimension for c in plan.constraints if c.state in ("applied", "checked")}
        stated |= set(plan.enumerated)
        resolved = {e.kind for e in plan.entities if e.status == "resolved"}
        missing = [
            d.name
            for d in schema.dimensions
            if d.required
            and d.name not in stated
            and d.name not in resolved
            and not (d.kind == "label" and "label" in resolved)
            and not (d.kind == "identity" and d.name in stated)
            and not (d.name == "place" and {"wmo_code", "province"} & stated)
        ]
        if missing:
            return PlanOutcome(
                "missing_required_dimension",
                f"the value depends on {missing}, which the question does not state; nothing "
                "is assumed",
                "missing_dimension",
            )
    return None
