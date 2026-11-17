"""The normalizer request: fixed instructions, the closed catalog, the user's question, and a
response schema whose short choices are enums.

The row labels (217 printed strings, up to 153 characters) are plain strings in the schema:
Gemini refuses a schema with that many long enum values ("too many states for serving"). The
closed set is enforced by :mod:`mabhas19.normalizer.verify`, which keeps a label only when it
equals a printed row of the proposed family exactly and discards anything else, unrepaired.

The model only *reads* the question and *chooses*: a table family, up to three printed row
labels, a printed output, a printed climate code, an energy rating, a city name (verified later
against the printed city lists) and verbatim number spans with their character offsets. It never
writes a value, a calculation or an alias.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from mabhas19.normalizer import catalog as C

PROMPT_VERSION = "normalizer-prompt-1"
MAX_ENTITIES = 3
MAX_NUMBERS = 2

INSTRUCTIONS = """\
You map one Persian question about the Iranian building energy code (Mabhas 19) onto a closed \
catalog of printed table rows. The catalog follows these instructions. You do not answer the \
question and you never state any regulatory value.

Return one JSON object:
- decision: "propose" if the question clearly asks for a quantity that one catalog family \
prints for a row you can identify; otherwise "abstain".
- family: the catalog family, or "none" when abstaining.
- entities: the printed row label(s) of that family the person means, copied exactly from the \
family's candidates. Give more than one (at most 3) only when the question genuinely fits each \
of them equally (for example two printed variants of the same thing); otherwise one. Empty when \
abstaining, and for the city family.
- output: the printed output of that family the person asks for, or "none".
- climate_code: only a printed climate code of that family that the question itself states \
(like 3B). Never derive a code from a city, a description («گرم», «سرد») or your own knowledge; \
use "" when the question does not state a printed code.
- city: the city name exactly as written in the question, or "".
- rating: an energy class A-D only if the question states it, else "".
- numbers: only numbers the question itself contains that set the family's number dimension \
(area for energy_limit, density for material): copy the digits exactly as written in the \
question ("text") and give "start", the 0-based character offset of that text in the question. \
Never convert, round or add numbers.

Rules:
- Map everyday or colloquial Persian to the printed label that means the same thing (for example \
a colloquial word for a part of the building, a space or a device). If no printed label means \
what the person asks, or the quantity asked is not one the family prints, abstain: a near \
neighbour with a different meaning is wrong.
- Do not guess missing details. A missing climate, rating, area or density stays empty; the \
system asks the person afterwards.
- Never write a number, value, unit or name that is not either in the question or in the catalog.
"""


def render_catalog(cat: dict[str, Any]) -> str:
    """The catalog as the model sees it (byte-stable)."""
    lines = ["CATALOG", f"version: {cat['version']}"]
    for f in cat["families"]:
        lines.append("")
        lines.append(f"family: {f['family']}  ({f['description']})")
        outs = ", ".join(f"{o['name']} = «{o['label']}»" for o in f["outputs"])
        lines.append(f"  outputs: {outs}")
        if f["climate_codes"]:
            lines.append(f"  climate codes: {', '.join(f['climate_codes'])}")
        if f["rating"]:
            lines.append(f"  energy ratings: {', '.join(f['rating'])}")
        if f["number_dimensions"]:
            lines.append(f"  number dimensions: {', '.join(f['number_dimensions'])}")
        if f["candidates"]:
            lines.append("  candidates:")
            lines.extend(f"    - {c}" for c in f["candidates"])
        elif f["family"] == C.CITY_FAMILY:
            lines.append("  candidates: none (give the city as written; it is checked later)")
    return "\n".join(lines)


def system_text(cat: dict[str, Any]) -> str:
    return f"{INSTRUCTIONS}\n{render_catalog(cat)}\n"


def schema(cat: dict[str, Any]) -> dict[str, Any]:
    """The response schema (Gemini ``responseJsonSchema``): enums for the short closed choices;
    the row labels are plain strings, checked by exact match in the verifier."""
    families = [f["family"] for f in cat["families"]]
    return {
        "type": "object",
        "properties": {
            "decision": {"type": "string", "enum": ["propose", "abstain"]},
            "family": {"type": "string", "enum": [*families, "none"]},
            "entities": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": MAX_ENTITIES,
            },
            "output": {"type": "string", "enum": [*C.all_outputs(cat), "none"]},
            "climate_code": {"type": "string", "enum": ["", *C.all_codes(cat)]},
            "city": {"type": "string"},
            "rating": {"type": "string", "enum": ["", *C.RATINGS]},
            "numbers": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "dimension": {"type": "string", "enum": ["area", "density"]},
                        "text": {"type": "string"},
                        "start": {"type": "integer", "minimum": 0},
                    },
                    "required": ["dimension", "text", "start"],
                    "additionalProperties": False,
                },
                "maxItems": MAX_NUMBERS,
            },
        },
        "required": [
            "decision",
            "family",
            "entities",
            "output",
            "climate_code",
            "city",
            "rating",
            "numbers",
        ],
        "additionalProperties": False,
    }


def user_text(question: str) -> str:
    return json.dumps({"question": question}, ensure_ascii=False)


def request_body(
    cat: dict[str, Any], question: str, *, effort: str, max_output_tokens: int
) -> dict[str, Any]:
    """The Gemini ``generateContent`` body (no temperature/topP/seed, no safety override)."""
    return {
        "systemInstruction": {"parts": [{"text": system_text(cat)}]},
        "contents": [{"role": "user", "parts": [{"text": user_text(question)}]}],
        "generationConfig": {
            "maxOutputTokens": max_output_tokens,
            "thinkingConfig": {"thinkingLevel": effort.upper()},
            "responseMimeType": "application/json",
            "responseJsonSchema": schema(cat),
        },
    }


def prompt_sha256(cat: dict[str, Any]) -> str:
    doc = {"version": PROMPT_VERSION, "system": system_text(cat), "user_format": user_text("")}
    return hashlib.sha256(json.dumps(doc, ensure_ascii=False).encode("utf-8")).hexdigest()


def schema_sha256(cat: dict[str, Any]) -> str:
    return C.sha256(schema(cat))
