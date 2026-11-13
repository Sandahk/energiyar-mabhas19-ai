"""Deterministic verification of a normalizer proposal, and what it would lead to.

Every field is checked independently against printed data; a field that fails is discarded (and
recorded), never repaired. A proposal is *valid* when a family, its output and at least one
printed row (or, for the city family, a city found by the exact certified city lookup) survive.
A valid proposal yields canonical questions built only from printed labels, a printed climate
code (stated, or taken deterministically from a verified city with one printed class), a stated
rating and verbatim number spans, plus the confirmation question the user would see. The answer
would come only from the unchanged engine running a canonical question.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

from mabhas19.api.orchestration import city_lookup
from mabhas19.generation import labels as L
from mabhas19.generation import options as O
from mabhas19.generation.provenance import numbers
from mabhas19.normalizer import catalog as C
from mabhas19.normalizer.prompt import MAX_ENTITIES, MAX_NUMBERS

if TYPE_CHECKING:
    from mabhas19.generation.contract import Answer
    from mabhas19.retrieval.models import RetrievalResult
    from mabhas19.retrieval.structured import StructuredStore

Run = Callable[[str, str], "tuple[Answer, RetrievalResult]"]
KEYS = ("decision", "family", "entities", "output", "climate_code", "city", "rating", "numbers")


@dataclass
class Verified:
    valid: bool = False
    abstained: bool = False
    parse_error: str = ""
    family: str = ""
    entities: list[str] = field(default_factory=list)
    output: str = ""
    climate: str = ""  # a printed code of the family's table (stated, or from a verified city)
    climate_source: str = ""  # "question" or "city"
    city: str = ""
    city_codes: list[str] = field(default_factory=list)
    rating: str = ""
    numbers: list[tuple[str, str]] = field(default_factory=list)  # (dimension, verbatim text)
    discarded: list[str] = field(default_factory=list)
    canonical: list[str] = field(default_factory=list)
    confirmation: str = ""
    leak: list[str] = field(default_factory=list)  # numbers in a canonical question with no source

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse(raw: str | None) -> tuple[dict[str, Any] | None, str]:
    """The proposal object, or ``(None, reason)``."""
    if not raw:
        return None, "no text"
    try:
        doc = json.loads(raw)
    except ValueError:
        return None, "not JSON"
    if not isinstance(doc, dict) or set(doc) != set(KEYS):
        return None, "wrong keys"
    if not isinstance(doc["entities"], list) or not isinstance(doc["numbers"], list):
        return None, "wrong types"
    if not all(isinstance(doc[k], str) for k in KEYS if k not in ("entities", "numbers")):
        return None, "wrong types"
    return doc, ""


def _numbers(
    doc: dict[str, Any], question: str, family: C.Family, out: Verified
) -> list[tuple[str, str]]:
    kept: list[tuple[str, str]] = []
    for n in doc["numbers"][:MAX_NUMBERS]:
        if not isinstance(n, dict):
            out.discarded.append("number: not an object")
            continue
        dim, text, start = n.get("dimension"), n.get("text"), n.get("start")
        if dim not in family.numbers:
            out.discarded.append(f"number: dimension {dim!r} not used by {family.name}")
            continue
        if not (isinstance(text, str) and isinstance(start, int) and start >= 0):
            out.discarded.append("number: malformed span")
            continue
        if question[start : start + len(text)] != text or not numbers(text):
            out.discarded.append(f"number: «{text}» is not a verbatim span at {start}")
            continue
        kept.append((str(dim), text))
    return kept


def verify(  # noqa: PLR0912, PLR0915
    doc: dict[str, Any] | None,
    question: str,
    store: StructuredStore,
    run: Run,
    parse_error: str = "",
) -> Verified:
    out = Verified(parse_error=parse_error)
    if doc is None:
        return out
    if doc["decision"] != "propose":
        out.abstained = True
        return out
    family = C.BY_NAME.get(doc["family"])
    if family is None:
        out.discarded.append(f"family: {doc['family']!r}")
        return out
    out.family = family.name
    if doc["output"] not in family.outputs:
        out.discarded.append(f"output: {doc['output']!r} not printed by {family.name}")
        return out
    out.output = doc["output"]
    printed = C.candidates(store, family)
    for e in doc["entities"][:MAX_ENTITIES]:
        if isinstance(e, str) and e in printed and e not in out.entities:
            out.entities.append(e)
        else:
            out.discarded.append(f"entity: «{e}» is not a printed row of {family.name}")
    if doc["city"].strip():
        city = city_lookup(run, doc["city"].strip())
        if city is None:
            out.discarded.append(f"city: «{doc['city']}» not found by the exact city lookup")
        elif doc["city"].strip() not in question:
            out.discarded.append(f"city: «{doc['city']}» is not written in the question")
        else:
            out.city, out.city_codes = doc["city"].strip(), city.codes
    codes = C.climate_codes(store, family)
    stated = doc["climate_code"]
    if stated:
        if stated not in codes:
            out.discarded.append(f"climate: «{stated}» is not printed in {family.name}")
        elif stated not in question:
            out.discarded.append(f"climate: «{stated}» is not written in the question")
        else:
            out.climate, out.climate_source = stated, "question"
    if family.climate and out.city and len(out.city_codes) == 1:
        if out.climate and out.climate != out.city_codes[0]:
            out.discarded.append("climate: the stated code differs from the city's printed class")
            out.climate = out.climate_source = ""
        elif not out.climate and out.city_codes[0] in codes:
            out.climate, out.climate_source = out.city_codes[0], "city"
    if doc["rating"]:
        if not family.rating or doc["rating"] not in C.RATINGS:
            out.discarded.append(f"rating: «{doc['rating']}» not used by {family.name}")
        else:
            out.rating = doc["rating"]
    out.numbers = _numbers(doc, question, family, out)
    if family.name == C.CITY_FAMILY:
        out.valid = bool(out.city)
    else:
        out.valid = bool(out.entities)
    if out.valid:
        _canonical(out, family, question)
    return out


def _canonical(out: Verified, family: C.Family, question: str) -> None:
    if family.name == C.CITY_FAMILY:
        out.canonical = [L.CITY_LOOKUP.format(city=out.city)]
        out.confirmation = f"منظور شما رده اقلیمی شهر «{out.city}» است؟"
    else:
        head = O.plain(L.OUTPUT_LABELS[out.output])
        tail = ""
        if out.climate:
            tail += f" در اقلیم {out.climate}"
        if out.rating:
            tail += f" رده {out.rating}"
        for dim, text in out.numbers:
            tail += f" ({L.APPEND_PREFIX.get(dim, dim)} {text})"
        out.canonical = [f"{head} {e}{tail}" for e in out.entities]
        shown = " یا ".join(f"«{e}»" for e in out.entities)
        out.confirmation = f"منظور شما «{L.OUTPUT_LABELS[out.output]}» برای {shown} است؟"
    allowed = numbers(question)
    allowed |= {n for e in out.entities for n in numbers(e)}
    allowed |= numbers(out.climate) | numbers(L.OUTPUT_LABELS.get(out.output, ""))
    for q in out.canonical:
        out.leak += sorted(numbers(q) - allowed)
