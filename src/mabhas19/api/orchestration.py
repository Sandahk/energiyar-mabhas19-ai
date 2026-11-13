"""Follow-up orchestration over the frozen engine: city -> climate, and side-by-side values.

Everything here is built from ordinary engine calls (retrieve -> compose -> provenance check,
one question each); nothing is retrieved, ranked, matched or computed here. Two conveniences:

* **City for climate.** The climate clarification also accepts a city name. The city is looked up
  through the engine exactly like «رده اقلیمی شهر …» (Appendix 2 and table 19-3-2 only); the
  lookup is accepted only when the engine resolved that exact printed name (after its normal
  spelling normalization) and nothing else. One printed class: the original question continues
  with that class added, and the city rows are shown with the answer. Different printed classes
  (Isfahan): no class is chosen; the rows are shown and the user picks one. An unknown city: the
  climate question is asked again. The printed class is passed on unchanged (5C stays 5C; each
  table accepts or refuses it).
* **Side by side.** When the only missing item of a lookup is a small printed column (Appendix 3
  energy rating, or the residential area band), the question is asked once per printed value,
  each a fully specified engine call, and the certified values are shown together instead of
  another question. Any difference other than that one column (row, quantity, unit, source
  issues, notices, conflicts) keeps the clarification.

:func:`composite_problems` checks a combined answer against the checked answers it was built
from: every value and context row is identical to a value of one of them, every number shown
appears in one of them, every notice, source and echo item comes from one of them (or is a fixed
label), and clarification options are a subset of the original printed options. A combined
answer that fails is never shown; the plain engine answer is shown instead.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from mabhas19.generation import labels as L
from mabhas19.generation import options as O
from mabhas19.generation.composer import COMPOSER_VERSION
from mabhas19.generation.composer import _missing_dims as _composer_missing
from mabhas19.generation.contract import (
    Answer,
    Clarification,
    ClarifyOption,
    Interpretation,
    Notice,
    Source,
    ValueItem,
)
from mabhas19.generation.provenance import numbers
from mabhas19.retrieval.schema import APP02, APP03, MAIN

if TYPE_CHECKING:
    from mabhas19.retrieval.models import RetrievalResult
    from mabhas19.retrieval.structured import StructuredStore

Run = Callable[[str, str], tuple[Answer, "RetrievalResult"]]

CLIMATE = "climate_class"
CITY_DATASETS = frozenset({APP02, f"{MAIN}19-3-2"})
# (table, missing field) pairs whose printed values may be shown side by side
SIDE_BY_SIDE = frozenset({(APP03, "rating"), (APP03, "area")})
MAX_SIDE = 4
MAX_CITY = 40
_STATION = re.compile(r"\s*\([^)]*\)\s*$")  # «تهران (مهرآباد)»: a printed station of the city
_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")


@dataclass(frozen=True)
class Part:
    """One engine call: the question asked, its checked answer and the result behind it."""

    question: str
    answer: Answer
    result: RetrievalResult


@dataclass
class Reply:
    answer: Answer
    result: RetrievalResult  # the main engine call (logged with the answer)
    parts: list[Part] = field(default_factory=list)


def _new_id() -> str:
    return uuid.uuid4().hex


def _call(run: Run, question: str, request_id: str | None = None) -> Part:
    answer, result = run(question, request_id or _new_id())
    return Part(question, answer, result)


def _clean(part: Part) -> bool:
    return not part.answer.safety.get("provenance_violations")


# ---------------------------------------------------------------- composite check


def composite_problems(  # noqa: PLR0912
    answer: Answer,
    parts: list[Part],
    fixed_notices: tuple[str, ...] = (),
    summary_keys: tuple[str, ...] = (),
) -> list[str]:
    """Problems of a combined answer against the checked answers it was built from."""
    problems = [f"part: {p.question[:40]} failed its own check" for p in parts if not _clean(p)]
    answers = [p.answer for p in parts]
    part_values = [v for a in answers for v in a.values]
    for v in answer.values:
        if v not in part_values:
            problems.append(f"value: {v.record_id} is not a value of a checked part")
    for v in answer.context:
        if v not in part_values:
            problems.append(f"context: {v.record_id} is not a value of a checked part")
    part_rows = {h.record_id for p in parts for h in p.result.structured}
    for rid in answer.safety.get("certified_row_ids") or []:
        if rid not in part_rows:
            problems.append(f"row: {rid} was certified by no part")
    part_notices = [n for a in answers for n in a.notices]
    for n in answer.notices:
        if n not in part_notices and n.text not in fixed_notices:
            problems.append(f"notice: «{n.text[:40]}» comes from no part")
    part_sources = [s for a in answers for s in a.sources]
    for s in answer.sources:
        if s not in part_sources:
            problems.append(f"source: {s.label} comes from no part")
    part_echo = [i for a in answers for i in a.interpretation]
    for i in answer.interpretation:
        if i in part_echo:
            continue
        seen = {e.value for e in part_echo if e.key == i.key}
        if i.key not in summary_keys or not set(i.value.split(" / ")) <= seen:
            problems.append(f"interpretation: «{i.value}» comes from no part")
    for c in answer.clarifications:
        if answer.values:
            problems.append("clarification: shown together with certified values")
        key = (c.dataset, c.field)
        same = [x for a in answers for x in a.clarifications if (x.dataset, x.field) == key]
        if not same or not set(c.options) <= set(same[0].options):
            problems.append(f"clarification: options of {c.field} are not the printed options")
    allowed = {n for a in answers for t in a.texts() for n in numbers(t)}
    for text in answer.texts():
        for num in sorted(numbers(text) - allowed):
            problems.append(f"numeric: {num} in «{text[:60]}» appears in no part")
    return problems


def _accept(composite: Answer, fallback: Reply, parts: list[Part], **kw: Any) -> Reply:
    problems = composite_problems(composite, parts, **kw)
    if problems:  # fail closed: the plain engine answer, with the reason recorded
        safety = {**fallback.answer.safety, "orchestration_rejected": problems}
        return Reply(replace(fallback.answer, safety=safety), fallback.result, fallback.parts)
    composite.safety["provenance_violations"] = []
    return Reply(composite, fallback.result, parts)


def _unique_notices(notices: list[Notice]) -> list[Notice]:
    out: list[Notice] = []
    for n in notices:
        if n not in out:
            out.append(n)
    return out


def _unique_sources(sources: list[Source]) -> list[Source]:
    out: list[Source] = []
    for s in sources:
        if s not in out:
            out.append(s)
    return out


# ---------------------------------------------------------------- side by side


def _missing_dims(nm: Any, plan: dict[str, Any]) -> list[str]:
    """The items the engine reported missing, read exactly as the composer reads them."""
    return [str(m) for m in _composer_missing(nm, plan)]


def _candidates(part: Part, store: StructuredStore) -> list[str] | None:  # noqa: PLR0911
    """The words to add for each printed value of the only missing column, or ``None``."""
    a, r = part.answer, part.result
    if a.response_type != "clarification" or not a.clarifications:
        return None
    c = a.clarifications[0]
    if c.outcome != "missing_required_dimension" or (c.dataset, c.field) not in SIDE_BY_SIDE:
        return None
    if len(r.plans) != 1 or str(r.plans[0].get("target")) != c.dataset:
        return None
    plan = r.plans[0]
    refused = [n for n in r.no_match if n.dataset == c.dataset]
    if len(refused) != 1 or _missing_dims(refused[0], plan) != [c.field]:
        return None  # more than one item is missing (or another refusal): ask
    if c.field == "area":  # the residential rows printed per area band
        entities = plan.get("entities") or []
        if len(entities) != 1 or entities[0].get("status") != "resolved":
            return None
        ids = [str(i) for i in entities[0].get("ids") or []]
        texts = [str((store.occupancies.get(i) or {}).get("label") or "") for i in ids]
        if not all(texts):
            return None
    else:
        texts = [o.text for o in c.options]
    return texts if 2 <= len(texts) <= MAX_SIDE else None


def _consistent(field_name: str, cands: list[Part]) -> bool:  # noqa: PLR0911
    """The candidates differ only in the one missing column: one certified value each, same
    table, quantity, unit, source issues and notices, no conflict and nothing unresolved."""
    for p in cands:
        a = p.answer
        if a.response_type != "structured_value" or len(a.values) != 1 or a.hidden_values:
            return False
        if a.clarifications or a.refusals or p.result.conflicts or not _clean(p):
            return False
        if any(i.status in ("ambiguous", "unresolved") for i in a.interpretation):
            return False
    vs = [p.answer.values[0] for p in cands]
    if len({(v.dataset, v.output, v.output_label, v.unit) for v in vs}) != 1:
        return False
    if len({tuple(sorted(v.source_issues)) for v in vs}) != 1:
        return False
    if len({tuple(p.answer.notices) for p in cands}) != 1:
        return False
    if len({v.record_id for v in vs}) != len(vs):
        return False
    if field_name == "area":  # one printed row per band; every other condition the same
        return len({v.row_label for v in vs}) == len(vs) and len({v.conditions for v in vs}) == 1
    # one row; exactly one printed condition differs (the column itself)
    if len({v.row_label for v in vs}) != 1:
        return False
    conds = [dict((c.label, c.value) for c in v.conditions) for v in vs]
    if len({tuple(sorted(c)) for c in conds}) != 1:
        return False
    differing = {k for k in conds[0] if len({c[k] for c in conds}) > 1}
    return len(differing) == 1


def _side_answer(base: Part, cands: list[Part], request_id: str) -> Answer:
    a = base.answer
    c = a.clarifications[0]
    values = [p.answer.values[0] for p in cands]
    kept = ("requested", "resolved")
    echo = [i for i in a.interpretation if i.key != c.field and i.status in kept]
    if c.field != "area":
        shown = [
            next((i.value for i in p.answer.interpretation if i.key == c.field), "") for p in cands
        ]
        if all(shown):
            label = L.FIELD_LABELS.get(c.field, c.field_label)
            echo.append(Interpretation(c.field, label, " / ".join(shown), "resolved", tuple(shown)))
    return Answer(
        request_id=request_id,
        question=a.question,
        response_type="structured_value",
        headline=L.SIDE_BY_SIDE_HEADLINE[c.field],
        interpretation=echo,
        values=values,
        notices=[Notice("note", L.SIDE_BY_SIDE_NOTE), *cands[0].answer.notices],
        sources=_unique_sources([s for p in cands for s in p.answer.sources]),
        safety={
            "route": a.safety.get("route"),
            "status": "certified_rows",
            "certified_row_ids": [h.record_id for p in cands for h in p.result.structured],
            "prose_chunk_ids": [],
            "composer": COMPOSER_VERSION,
            "composite": {
                "kind": "side_by_side",
                "field": c.field,
                "dataset": c.dataset,
                "parts": [
                    {
                        "question": p.question,
                        "request_id": p.answer.request_id,
                        "rows": [h.record_id for h in p.result.structured],
                    }
                    for p in cands
                ],
            },
        },
    )


def side_by_side(run: Run, store: StructuredStore, part: Part, request_id: str) -> Reply:
    plain = Reply(part.answer, part.result, [part])
    texts = _candidates(part, store)
    if texts is None:
        return plain
    cands = [_call(run, f"{part.question} ({t})") for t in texts]
    field_name = part.answer.clarifications[0].field
    if not _consistent(field_name, cands):
        return plain
    composite = _side_answer(part, cands, request_id)
    return _accept(
        composite,
        plain,
        [part, *cands],
        fixed_notices=(L.SIDE_BY_SIDE_NOTE,),
        summary_keys=(field_name,),
    )


# ---------------------------------------------------------------- city -> climate


def _typed_city(text: str) -> str:
    words = text.split()
    if words and words[0] == L.CITY_PREFIX:
        words = words[1:]
    return " ".join(words)


def _printed_code(store: StructuredStore, dataset: str, typed: str) -> str:
    """The printed climate class the user typed instead of a city (``""`` if none)."""
    want = typed.translate(_DIGITS).replace(" ", "").upper()
    return next((c for c in O.printed_options(store, dataset, CLIMATE) if c.upper() == want), "")


@dataclass(frozen=True)
class City:
    """A city lookup: the engine call and the certified rows printed with exactly that name
    (a station in parentheses, «تهران (مهرآباد)», is the same printed city)."""

    part: Part
    rows: tuple[ValueItem, ...]

    @property
    def codes(self) -> list[str]:
        out: list[str] = []
        for v in self.rows:
            if v.value not in out:
                out.append(v.value)
        return out

    def notices(self) -> list[Notice]:
        """The lookup's notices that concern the kept rows (a conflict only when they differ)."""
        issues = {i for v in self.rows for i in v.source_issues}
        out = []
        for n in self.part.answer.notices:
            issue = n.kind == "source_issue" and set(n.ids) <= issues
            if issue or (n.kind == "conflict" and len(self.codes) > 1):
                out.append(n)
        return out


def city_lookup(run: Run, name: str) -> City | None:  # noqa: PLR0911
    """The engine's certified climate rows of exactly the printed city ``name``, or ``None``."""
    if not name or len(name) > MAX_CITY or numbers(name):
        return None
    part = _call(run, L.CITY_LOOKUP.format(city=name))
    a, r = part.answer, part.result
    if a.response_type != "structured_value" or a.clarifications or not a.values:
        return None
    if not _clean(part):
        return None
    if any(v.dataset not in CITY_DATASETS or v.output != CLIMATE for v in a.values):
        return None
    if not r.plans or any(str(p.get("target")) not in CITY_DATASETS for p in r.plans):
        return None
    want = O.label_key(name)
    for p in r.plans:
        entities = p.get("entities") or []
        if len(entities) != 1 or entities[0].get("kind") != "place":
            return None
        ids = [str(i) for i in entities[0].get("ids") or []]
        if entities[0].get("status") != "resolved" or not ids:
            return None
        if any(O.label_key(i) != want for i in ids):
            return None  # the engine read another (or a wider) name
    # only rows printed with exactly this name (the engine may also return «بندر بوشهر» for
    # «بوشهر»: such a row is never used)
    rows = tuple(v for v in a.values if O.label_key(_STATION.sub("", v.row_label)) == want)
    return City(part, rows) if rows else None


def _climate_question(part: Part) -> Clarification | None:
    a = part.answer
    if a.response_type == "clarification" and a.clarifications:
        c = a.clarifications[0]
        if c.field == CLIMATE and c.outcome == "missing_required_dimension":
            return c
    return None


def city_flow(run: Run, store: StructuredStore, base: str, typed: str, request_id: str) -> Reply:
    """The climate clarification of ``base`` answered with ``typed`` (a city or a printed code)."""
    base_part = _call(run, base, request_id)
    plain = Reply(base_part.answer, base_part.result, [base_part])
    clar = _climate_question(base_part)
    if clar is None:  # the question no longer asks for a climate: show what it does
        return plain
    name = _typed_city(typed)
    code = _printed_code(store, clar.dataset, name)
    if code:
        return respond(run, store, f"{base} ({O.option_text(CLIMATE, code)[1]})", request_id)
    city = city_lookup(run, name)
    if city is None:  # never guessed: the climate question again, saying why
        shown = name and not numbers(name) and len(name) <= MAX_CITY
        note = L.CITY_UNKNOWN.format(city=name) if shown else L.CITY_UNKNOWN_PLAIN
        again = replace(base_part.answer, clarifications=[replace(clar, note=note)])
        return _accept(again, plain, [base_part])
    codes = city.codes
    context = list(city.rows)
    city_notices = city.notices()
    if len(codes) == 1:
        cont = respond(run, store, f"{base} ({O.option_text(CLIMATE, codes[0])[1]})", request_id)
        composite = replace(
            cont.answer,
            context=context,
            notices=_unique_notices(
                [Notice("note", L.NOTICE_CITY_CLIMATE), *city_notices, *cont.answer.notices]
            ),
            safety={
                **cont.answer.safety,
                "composite": {
                    **(cont.answer.safety.get("composite") or {}),
                    "city": _city_record(city, codes),
                },
            },
        )
        fallback = Reply(base_part.answer, base_part.result, [base_part])
        return _accept(
            composite,
            fallback,
            [city.part, *cont.parts],
            fixed_notices=(L.NOTICE_CITY_CLIMATE, L.SIDE_BY_SIDE_NOTE),
            summary_keys=tuple(
                k for k in ("rating", "area") if any(i.key == k for i in cont.answer.interpretation)
            ),
        )
    # the printed sources disagree: show both, choose none
    printed = O.printed_options(store, clar.dataset, CLIMATE)
    offered = tuple(ClarifyOption(*O.option_text(CLIMATE, c)) for c in printed if c in codes)
    if len(offered) != len(codes):
        offered = clar.options  # a class this table does not print: the full printed list
    choose = replace(
        clar,
        options=offered,
        note=L.CITY_CONFLICT.format(city=name),
        free_text=False,
        free_text_prefix="",
        placeholder="",
    )
    composite = replace(
        base_part.answer,
        clarifications=[choose],
        context=context,
        notices=_unique_notices([*city_notices, *base_part.answer.notices]),
        safety={
            **base_part.answer.safety,
            "composite": {"kind": "city_conflict", "city": _city_record(city, codes)},
        },
    )
    return _accept(composite, plain, [base_part, city.part])


def _city_record(city: City, codes: list[str]) -> dict[str, Any]:
    return {
        "question": city.part.question,
        "request_id": city.part.answer.request_id,
        "rows": [v.record_id for v in city.rows],
        "codes": codes,
    }


# ---------------------------------------------------------------- entry point


def respond(  # noqa: PLR0913
    run: Run,
    store: StructuredStore,
    question: str,
    request_id: str,
    *,
    follow_up: dict[str, Any] | None = None,
    base: str | None = None,
) -> Reply:
    """The answer to one request: a typed city (or code) for a climate question is looked up
    first; otherwise one engine call, shown side by side when its only missing column is small."""
    if follow_up and base and follow_up.get("field") == CLIMATE and follow_up.get("kind") == "free":
        return city_flow(run, store, base, str(follow_up.get("text") or ""), request_id)
    return side_by_side(run, store, _call(run, question, request_id), request_id)
