"""Deterministic projection of the canonical proposal schema onto a provider's schema subset.

The canonical ``sh0-proposal-1`` schema (:func:`proposal.canonical_schema`) is the contract;
a provider's constrained decoding only accepts a subset of JSON Schema, so it receives a
*projection*: the same object shapes, property names, required keys, enums and constants, with
the constraints the provider cannot express removed. Constrained decoding is an additional
reliability layer; every response is still validated by the canonical parser and the verifier.

Anthropic structured outputs (``output_config.format``, ``type: json_schema``) accept object,
array, string, integer, number, boolean and null types, ``enum``, ``const``, ``anyOf``,
``allOf``, ``$ref``/``$defs`` and ``additionalProperties: false`` (required on every object).
They do not accept numeric bounds (``minimum``, ``maximum``, ``exclusiveMinimum``,
``exclusiveMaximum``, ``multipleOf``), string length bounds (``minLength``, ``maxLength``),
complex array constraints (``minItems``, ``maxItems``, ``uniqueItems``), recursive schemas or
any other ``additionalProperties``. The projection therefore:

* removes those constraint keywords (the canonical parser still enforces them);
* rewrites ``oneOf`` as ``anyOf`` (the branches here are disjoint by type, so both admit the
  same documents);
* rewrites a type list (``["integer", "null"]``) as ``anyOf`` of single types;
* drops the ``$schema`` / ``title`` annotations;
* refuses (raises) on anything it does not know how to project, so a canonical change can never
  pass through silently.

Gemini structured outputs (``generationConfig.responseJsonSchema``) document object, array,
string, number, integer, boolean and null types (a type list such as ``["integer", "null"]``
included), ``enum``, ``anyOf``, ``$ref``, ``additionalProperties``, ``minimum`` / ``maximum``
and ``minItems`` / ``maxItems`` (enforced); ``minLength`` / ``maxLength`` are accepted but
ignored; ``oneOf``, ``const`` and ``exclusiveMinimum`` / ``exclusiveMaximum`` / ``multipleOf``
are not documented. The Gemini projection therefore keeps the numeric and array bounds and type
lists, removes the rest of those keywords, rewrites ``oneOf`` as ``anyOf`` and ``const`` as a
one-value ``enum``, and gives an untyped ``enum`` the type of its values (the same documents are
admitted). ``additionalProperties: false`` is kept on every object for both providers.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

ANTHROPIC_PROJECTION = "anthropic-projection-1"
GEMINI_PROJECTION = "gemini-projection-1"
# keywords removed by the Anthropic projection (enforced by the canonical parser only)
REMOVED = frozenset(
    {
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "minLength",
        "maxLength",
        "minItems",
        "maxItems",
        "uniqueItems",
    }
)
GEMINI_REMOVED = frozenset(
    {"exclusiveMinimum", "exclusiveMaximum", "multipleOf", "minLength", "maxLength", "uniqueItems"}
)
_ANNOTATIONS = frozenset({"$schema", "title", "description"})
_KEPT = frozenset(
    {"type", "properties", "required", "additionalProperties", "items", "enum", "const", "anyOf"}
)
_JSON_TYPES = {str: "string", bool: "boolean", int: "integer", float: "number"}


@dataclass(frozen=True)
class Projection:
    version: str
    removed: frozenset[str]
    kept: frozenset[str]
    type_list_as_any_of: bool  # ["integer", "null"] -> anyOf of single types
    const_as_enum: bool  # const c -> enum [c]
    typed_enums: bool  # an untyped enum gets the type of its values


ANTHROPIC = Projection(ANTHROPIC_PROJECTION, REMOVED, _KEPT, True, False, False)
GEMINI = Projection(
    GEMINI_PROJECTION,
    GEMINI_REMOVED,
    (_KEPT - {"const"}) | {"minimum", "maximum", "minItems", "maxItems"},
    False,
    True,
    True,
)
PROJECTIONS = {"anthropic": ANTHROPIC, "gemini": GEMINI}


class ProjectionError(ValueError):
    pass


def project_anthropic(schema: dict[str, Any]) -> dict[str, Any]:
    """The Anthropic projection of a canonical schema (a new object; the input is untouched)."""
    return _node(schema, "$", ANTHROPIC)


def project_gemini(schema: dict[str, Any]) -> dict[str, Any]:
    """The Gemini projection of a canonical schema (a new object; the input is untouched)."""
    return _node(schema, "$", GEMINI)


def _node(node: Any, path: str, p: Projection) -> dict[str, Any]:  # noqa: PLR0912
    if not isinstance(node, dict):
        raise ProjectionError(f"{path}: schema object expected")
    out: dict[str, Any] = {}
    for key in sorted(node):
        value = node[key]
        if key in p.removed or key in _ANNOTATIONS:
            continue
        if key == "oneOf":
            out["anyOf"] = [_node(b, f"{path}.oneOf[{i}]", p) for i, b in enumerate(value)]
        elif key == "anyOf":
            out["anyOf"] = [_node(b, f"{path}.anyOf[{i}]", p) for i, b in enumerate(value)]
        elif key == "properties":
            out["properties"] = {k: _node(v, f"{path}.{k}", p) for k, v in value.items()}
        elif key == "items":
            out["items"] = _node(value, f"{path}[]", p)
        elif key == "additionalProperties":
            if value is not False:
                raise ProjectionError(f"{path}: additionalProperties must be false")
            out[key] = False
        elif key == "const" and p.const_as_enum:
            out["enum"] = [value]
        elif key in p.kept:
            out[key] = value
        else:
            raise ProjectionError(f"{path}: no projection for {key!r}")
    if p.typed_enums and "enum" in out and "type" not in out:
        types = {_JSON_TYPES.get(type(v)) for v in out["enum"]}
        if len(types) != 1 or None in types:
            raise ProjectionError(f"{path}: an untyped enum needs values of one JSON type")
        out["type"] = types.pop()
    if p.type_list_as_any_of and isinstance(out.get("type"), list):
        types = out.pop("type")
        out = {"anyOf": [{"type": t} for t in types], **out}
    if out.get("type") == "object" and out.get("additionalProperties") is not False:
        raise ProjectionError(f"{path}: every object needs additionalProperties: false")
    return out


def removed_constraints(
    schema: dict[str, Any], path: str = "$", projection: Projection = ANTHROPIC
) -> list[str]:
    """Every constraint the projection removes, as ``path:keyword`` (what the canonical parser
    alone enforces)."""
    out: list[str] = []
    if isinstance(schema, dict):
        for key in sorted(schema):
            if key in projection.removed:
                out.append(f"{path}:{key}")
            value = schema[key]
            if key == "properties":
                for k in sorted(value):
                    out += removed_constraints(value[k], f"{path}.{k}", projection)
            elif key == "items":
                out += removed_constraints(value, f"{path}[]", projection)
            elif key in ("oneOf", "anyOf"):
                for i, b in enumerate(value):
                    out += removed_constraints(b, f"{path}.{key}[{i}]", projection)
    return out


def structure(schema: Any) -> Any:
    """The semantic skeleton shared by canonical and projected schemas: property names,
    required keys, enums, constants and the type set of every node (``oneOf``/``anyOf`` and type
    lists normalized). Equal skeletons mean the projection changed no proposal semantics beyond
    the removed constraints."""
    if not isinstance(schema, dict):
        return schema
    types: set[str] = set()
    t = schema.get("type")
    if isinstance(t, str):
        types.add(t)
    elif isinstance(t, list):
        types |= set(t)
    # a constant is a one-value enum; an untyped enum has the type of its values
    enum = [schema["const"]] if "const" in schema else schema.get("enum")
    if enum and not types:
        types |= {_JSON_TYPES.get(type(v), "?") for v in enum}
    branches = [structure(b) for b in [*schema.get("oneOf", []), *schema.get("anyOf", [])]]
    for b in branches:
        types |= set(b["types"])
    informative = [b for b in branches if b["enum"] or b["properties"]]
    return {
        "types": sorted(types),
        "branches": sorted(json.dumps(b, sort_keys=True) for b in informative),
        "enum": enum,
        "required": sorted(schema.get("required", [])),
        "properties": {k: structure(v) for k, v in sorted(schema.get("properties", {}).items())},
        "items": structure(schema["items"]) if "items" in schema else None,
    }


def sha256(schema: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(schema, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
