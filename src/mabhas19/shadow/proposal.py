"""The typed semantic proposal (``sh0-proposal-1``) and its strict, total parser.

A proposal only *labels and chooses* among the items of a
:class:`~mabhas19.shadow.context.CandidateContext`. Every item refers to a context item by
index and echoes its offsets (and text, where the item is a query slice); identifiers are
strings the context supplied. There is no field for a number, a record, a printed value, a
canonical name or a citation; ``notes`` is free text that is logged and never parsed.

* ``clauses`` - one role per deterministic clause: ``lookup`` (asks for a printed value),
  ``verify_request`` (asks to check a stated claim), ``background`` (context only),
  ``explanation`` (asks for an explanation, answered from prose) or ``unsupported``;
* ``entities`` - per entity span: ``use`` (``identity`` names a printed row; ``grammatical``
  is a function word / homograph; ``contextual`` is context, not the identity; ``ambiguous``)
  and ``candidate`` (a candidate ID of that span, ``none`` or ``ambiguous``);
* ``values`` - per stated value: ``role`` (``lookup_constraint``, ``verify_request``,
  ``background``, ``excluded``, ``corrected_away``) and its owning ``clause``;
* ``polarity`` - per exclusion / negative form / rejected reading: ``kept`` and ``clause``;
* ``operation`` - ``none``, a calculation kind (``operations.py``) or ``ambiguous``;
* ``requests`` - the requested tables with ``intent`` (``lookup``, ``reverse_lookup``,
  ``verify``, ``ambiguous``), ``output`` (a schema output, ``none`` or ``ambiguous``) and
  ``unit`` (a unit index, ``none`` or ``ambiguous``); a table not listed is not requested;
* ``coordination`` - per coordinated table, its member spans.

:func:`parse` never raises: malformed input yields ``(None, errors)``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

PROPOSAL_SCHEMA = "sh0-proposal-1"
CLAUSE_ROLES = ("lookup", "verify_request", "background", "explanation", "unsupported")
REQUESTED_CLAUSES = frozenset({"lookup", "verify_request"})
ENTITY_USES = ("identity", "grammatical", "contextual", "ambiguous")
VALUE_ROLES = ("lookup_constraint", "verify_request", "background", "excluded", "corrected_away")
CALCULATIONS = (
    "average",
    "interpolation",
    "extrapolation",
    "sum",
    "ratio",
    "percentage_change",
    "difference",
    "conversion",
    "multiplication",
    "calculation",
)
OPERATIONS = ("none", *CALCULATIONS, "ambiguous")
INTENTS = ("lookup", "reverse_lookup", "verify", "ambiguous")
NONE, AMBIGUOUS = "none", "ambiguous"
_TOP = ("schema", "clauses", "entities", "values", "polarity", "operation", "requests",
        "coordination")  # fmt: skip


@dataclass(frozen=True)
class ClauseLabel:
    index: int
    start: int
    end: int
    role: str


@dataclass(frozen=True)
class EntityLabel:
    index: int
    start: int
    end: int
    text: str
    use: str
    candidate: str  # a candidate ID | none | ambiguous


@dataclass(frozen=True)
class ValueLabel:
    index: int
    start: int
    end: int
    text: str
    role: str
    clause: int | None


@dataclass(frozen=True)
class PolarityLabel:
    index: int
    start: int
    end: int
    kept: bool
    clause: int | None


@dataclass(frozen=True)
class OperationLabel:
    kind: str
    refs: tuple[int, ...]


@dataclass(frozen=True)
class Request:
    table: str
    intent: str
    output: str  # a schema output | none | ambiguous
    unit: int | str  # a unit index | none | ambiguous


@dataclass(frozen=True)
class Member:
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class CoordinationLabel:
    table: str
    members: tuple[Member, ...]


@dataclass(frozen=True)
class Proposal:
    clauses: tuple[ClauseLabel, ...]
    entities: tuple[EntityLabel, ...]
    values: tuple[ValueLabel, ...]
    polarity: tuple[PolarityLabel, ...]
    operation: OperationLabel
    requests: tuple[Request, ...]
    coordination: tuple[CoordinationLabel, ...]
    notes: str = ""  # logged, never parsed or trusted

    def to_dict(self) -> dict[str, Any]:
        """The proposal as plain JSON types (it parses back to an equal proposal)."""
        d: dict[str, Any] = json.loads(json.dumps(asdict(self), ensure_ascii=False))
        d["schema"] = PROPOSAL_SCHEMA
        return d


class _Invalid(ValueError):
    pass


def _obj(
    x: Any, where: str, keys: tuple[str, ...], optional: tuple[str, ...] = ()
) -> dict[str, Any]:
    if not isinstance(x, dict):
        raise _Invalid(f"{where}: object expected")
    missing = [k for k in keys if k not in x]
    extra = [k for k in x if k not in keys and k not in optional]
    if missing or extra:
        raise _Invalid(f"{where}: missing {missing} / unknown {extra}")
    return x


def _int(x: Any, where: str) -> int:
    if not isinstance(x, int) or isinstance(x, bool) or x < 0:
        raise _Invalid(f"{where}: non-negative integer expected")
    return x


def _opt_int(x: Any, where: str) -> int | None:
    return None if x is None else _int(x, where)


def _str(x: Any, where: str, allowed: tuple[str, ...] | None = None) -> str:
    if not isinstance(x, str):
        raise _Invalid(f"{where}: string expected")
    if allowed is not None and x not in allowed:
        raise _Invalid(f"{where}: {x!r} not in {list(allowed)}")
    return x


def _list(x: Any, where: str) -> list[Any]:
    if not isinstance(x, list):
        raise _Invalid(f"{where}: array expected")
    return x


def _span(d: dict[str, Any], where: str) -> tuple[int, int]:
    start, end = _int(d["start"], f"{where}.start"), _int(d["end"], f"{where}.end")
    if end <= start:
        raise _Invalid(f"{where}: empty or inverted span")
    return start, end


def parse(raw: str | bytes | dict[str, Any]) -> tuple[Proposal | None, list[str]]:
    """A strictly typed proposal, or ``(None, errors)``: unknown or missing keys, wrong types,
    values outside the closed vocabularies and empty spans are schema errors."""
    try:
        doc = json.loads(raw) if isinstance(raw, str | bytes) else raw
        return _parse(doc), []
    except json.JSONDecodeError as exc:
        return None, [f"not JSON: {exc.msg}"]
    except _Invalid as exc:
        return None, [str(exc)]
    except RecursionError:
        return None, ["nesting too deep"]


def _parse(doc: Any) -> Proposal:
    d = _obj(doc, "proposal", _TOP, ("notes",))
    if d["schema"] != PROPOSAL_SCHEMA:
        raise _Invalid(f"schema: {d['schema']!r} is not {PROPOSAL_SCHEMA}")
    clauses = []
    for i, x in enumerate(_list(d["clauses"], "clauses")):
        c = _obj(x, f"clauses[{i}]", ("index", "start", "end", "role"))
        s, e = _span(c, f"clauses[{i}]")
        clauses.append(
            ClauseLabel(_int(c["index"], "index"), s, e, _str(c["role"], "role", CLAUSE_ROLES))
        )
    entities = []
    for i, x in enumerate(_list(d["entities"], "entities")):
        c = _obj(x, f"entities[{i}]", ("index", "start", "end", "text", "use", "candidate"))
        s, e = _span(c, f"entities[{i}]")
        entities.append(
            EntityLabel(
                _int(c["index"], "index"),
                s,
                e,
                _str(c["text"], "text"),
                _str(c["use"], "use", ENTITY_USES),
                _str(c["candidate"], "candidate"),
            )
        )
    values = []
    for i, x in enumerate(_list(d["values"], "values")):
        c = _obj(x, f"values[{i}]", ("index", "start", "end", "text", "role", "clause"))
        s, e = _span(c, f"values[{i}]")
        values.append(
            ValueLabel(
                _int(c["index"], "index"),
                s,
                e,
                _str(c["text"], "text"),
                _str(c["role"], "role", VALUE_ROLES),
                _opt_int(c["clause"], "clause"),
            )
        )
    pol = []
    for i, x in enumerate(_list(d["polarity"], "polarity")):
        c = _obj(x, f"polarity[{i}]", ("index", "start", "end", "kept", "clause"))
        s, e = _span(c, f"polarity[{i}]")
        if not isinstance(c["kept"], bool):
            raise _Invalid(f"polarity[{i}].kept: boolean expected")
        pol.append(
            PolarityLabel(
                _int(c["index"], "index"), s, e, c["kept"], _opt_int(c["clause"], "clause")
            )
        )
    o = _obj(d["operation"], "operation", ("kind", "refs"))
    op = OperationLabel(
        _str(o["kind"], "operation.kind", OPERATIONS),
        tuple(_int(r, "operation.refs") for r in _list(o["refs"], "operation.refs")),
    )
    requests = []
    for i, x in enumerate(_list(d["requests"], "requests")):
        c = _obj(x, f"requests[{i}]", ("table", "intent", "output", "unit"))
        unit = c["unit"]
        if not (
            isinstance(unit, int) and not isinstance(unit, bool) and unit >= 0
        ) and unit not in (
            NONE,
            AMBIGUOUS,
        ):
            raise _Invalid(f"requests[{i}].unit: unit index, 'none' or 'ambiguous' expected")
        requests.append(
            Request(
                _str(c["table"], "table"),
                _str(c["intent"], "intent", INTENTS),
                _str(c["output"], "output"),
                unit,
            )
        )
    coordination = []
    for i, x in enumerate(_list(d["coordination"], "coordination")):
        c = _obj(x, f"coordination[{i}]", ("table", "members"))
        members = []
        for j, y in enumerate(_list(c["members"], "members")):
            m = _obj(y, f"coordination[{i}].members[{j}]", ("start", "end", "text"))
            s, e = _span(m, "member")
            members.append(Member(s, e, _str(m["text"], "text")))
        coordination.append(CoordinationLabel(_str(c["table"], "table"), tuple(members)))
    notes = d.get("notes", "")
    if not isinstance(notes, str):
        raise _Invalid("notes: string expected")
    return Proposal(
        tuple(clauses),
        tuple(entities),
        tuple(values),
        tuple(pol),
        op,
        tuple(requests),
        tuple(coordination),
        notes,
    )


# ---------------------------------------------------------------- JSON Schema (for providers)


def _span_props(*extra: str) -> dict[str, Any]:
    props: dict[str, Any] = {
        "start": {"type": "integer", "minimum": 0},
        "end": {"type": "integer", "minimum": 1},
    }
    for k in extra:
        props[k] = {"type": "string"}
    return props


def _item(props: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(props),
        "properties": props,
    }


def json_schema() -> dict[str, Any]:
    """The proposal's JSON Schema (structured output for any provider). Candidate membership
    and offsets are per query and checked by the verifier, never by the schema alone."""
    index = {"type": "integer", "minimum": 0}
    clause_ref = {"type": ["integer", "null"], "minimum": 0}
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": PROPOSAL_SCHEMA,
        "type": "object",
        "additionalProperties": False,
        "required": list(_TOP),
        "properties": {
            "schema": {"const": PROPOSAL_SCHEMA},
            "clauses": {
                "type": "array",
                "items": _item(
                    {"index": index, **_span_props(), "role": {"enum": list(CLAUSE_ROLES)}}
                ),
            },
            "entities": {
                "type": "array",
                "items": _item(
                    {
                        "index": index,
                        **_span_props("text"),
                        "use": {"enum": list(ENTITY_USES)},
                        "candidate": {"type": "string"},
                    }
                ),
            },
            "values": {
                "type": "array",
                "items": _item(
                    {
                        "index": index,
                        **_span_props("text"),
                        "role": {"enum": list(VALUE_ROLES)},
                        "clause": clause_ref,
                    }
                ),
            },
            "polarity": {
                "type": "array",
                "items": _item(
                    {
                        "index": index,
                        **_span_props(),
                        "kept": {"type": "boolean"},
                        "clause": clause_ref,
                    }
                ),
            },
            "operation": _item(
                {"kind": {"enum": list(OPERATIONS)}, "refs": {"type": "array", "items": index}}
            ),
            "requests": {
                "type": "array",
                "items": _item(
                    {
                        "table": {"type": "string"},
                        "intent": {"enum": list(INTENTS)},
                        "output": {"type": "string"},
                        "unit": {"oneOf": [index, {"enum": [NONE, AMBIGUOUS]}]},
                    }
                ),
            },
            "coordination": {
                "type": "array",
                "items": _item(
                    {
                        "table": {"type": "string"},
                        "members": {"type": "array", "items": _item(_span_props("text"))},
                    }
                ),
            },
            "notes": {"type": "string", "maxLength": 2000},
        },
    }
