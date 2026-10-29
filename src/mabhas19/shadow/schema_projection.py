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
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

ANTHROPIC_PROJECTION = "anthropic-projection-1"
# keywords removed by the projection (enforced by the canonical parser only)
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
_ANNOTATIONS = frozenset({"$schema", "title", "description"})
_KEPT = frozenset(
    {"type", "properties", "required", "additionalProperties", "items", "enum", "const", "anyOf"}
)


class ProjectionError(ValueError):
    pass


def project_anthropic(schema: dict[str, Any]) -> dict[str, Any]:
    """The Anthropic projection of a canonical schema (a new object; the input is untouched)."""
    return _node(schema, "$")


def _node(node: Any, path: str) -> dict[str, Any]:  # noqa: PLR0912
    if not isinstance(node, dict):
        raise ProjectionError(f"{path}: schema object expected")
    out: dict[str, Any] = {}
    for key in sorted(node):
        value = node[key]
        if key in REMOVED or key in _ANNOTATIONS:
            continue
        if key == "oneOf":
            out["anyOf"] = [_node(b, f"{path}.oneOf[{i}]") for i, b in enumerate(value)]
        elif key == "anyOf":
            out["anyOf"] = [_node(b, f"{path}.anyOf[{i}]") for i, b in enumerate(value)]
        elif key == "properties":
            out["properties"] = {k: _node(v, f"{path}.{k}") for k, v in value.items()}
        elif key == "items":
            out["items"] = _node(value, f"{path}[]")
        elif key == "additionalProperties":
            if value is not False:
                raise ProjectionError(f"{path}: additionalProperties must be false")
            out[key] = False
        elif key in _KEPT:
            out[key] = value
        else:
            raise ProjectionError(f"{path}: no projection for {key!r}")
    if isinstance(out.get("type"), list):
        types = out.pop("type")
        out = {"anyOf": [{"type": t} for t in types], **out}
    if out.get("type") == "object" and out.get("additionalProperties") is not False:
        raise ProjectionError(f"{path}: every object needs additionalProperties: false")
    return out


def removed_constraints(schema: dict[str, Any], path: str = "$") -> list[str]:
    """Every constraint the projection removes, as ``path:keyword`` (what the canonical parser
    alone enforces)."""
    out: list[str] = []
    if isinstance(schema, dict):
        for key in sorted(schema):
            if key in REMOVED:
                out.append(f"{path}:{key}")
            value = schema[key]
            if key == "properties":
                for k in sorted(value):
                    out += removed_constraints(value[k], f"{path}.{k}")
            elif key == "items":
                out += removed_constraints(value, f"{path}[]")
            elif key in ("oneOf", "anyOf"):
                for i, b in enumerate(value):
                    out += removed_constraints(b, f"{path}.{key}[{i}]")
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
    branches = [structure(b) for b in [*schema.get("oneOf", []), *schema.get("anyOf", [])]]
    for b in branches:
        types |= set(b["types"])
    informative = [b for b in branches if b["enum"] or b["const"] is not None or b["properties"]]
    return {
        "types": sorted(types),
        "branches": sorted(json.dumps(b, sort_keys=True) for b in informative),
        "enum": schema.get("enum"),
        "const": schema.get("const"),
        "required": sorted(schema.get("required", [])),
        "properties": {k: structure(v) for k, v in sorted(schema.get("properties", {}).items())},
        "items": structure(schema["items"]) if "items" in schema else None,
    }


def sha256(schema: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(schema, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
