"""Follow-up orchestration over the frozen engine: city -> climate, and side-by-side values.

Everything here is built from ordinary engine calls (retrieve -> compose -> provenance check,
one question each); nothing is retrieved, ranked, matched or computed here. Two conveniences:

* **City for climate.** The climate clarification also accepts a city chosen from the certified
  city list (:mod:`mabhas19.api.cities`; never typed text). The city is looked up
  through the engine exactly like «رده اقلیمی شهر …» (Appendix 2 and table 19-3-2 only); the
  lookup is accepted only when the engine resolved that exact printed name (after its normal
  spelling normalization) and nothing else. One printed class: the original question continues
  with that class added, and the city rows are shown with the answer. Different printed classes
  (Isfahan): no class is chosen; the rows are shown and the user picks one. An unknown city: the
  climate question is asked again. The printed class is passed on unchanged (5C stays 5C; each
  table accepts or refuses it).
* **Wording and row groups** (:mod:`mabhas19.api.wording`). Reviewed spelling / morphology
  variants are rewritten before the engine reads the question, with a visible note. When the
  engine still cannot answer, an everyday word for several printed rows («بام», «پنجره»), or a
  table's quantity asked with no row at all, offers those printed rows to choose from; rows
  that differ only in a printed band are each looked up and shown side by side. Nothing is
  chosen for the user, and every value still comes from an ordinary engine call.
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
from typing import TYPE_CHECKING, Any, Literal

from mabhas19.api import quantities as Q
from mabhas19.api import wording as W
from mabhas19.api.cities import STATION as _STATION
from mabhas19.api.cities import catalogue
from mabhas19.generation import labels as L
from mabhas19.generation import options as O
from mabhas19.generation.composer import COMPOSER_VERSION, MAX_OPTIONS, _Sources
from mabhas19.generation.composer import _missing_dims as _composer_missing
from mabhas19.generation.contract import (
    Answer,
    Clarification,
    ClarifyOption,
    Interpretation,
    Notice,
    Refusal,
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


def _side_answer(  # noqa: PLR0913, PLR0917
    base: Part,
    cands: list[Part],
    request_id: str,
    field_name: str | None = None,
    dataset: str | None = None,
    notices: tuple[str, ...] = (),
) -> Answer:
    a = base.answer
    c = a.clarifications[0] if field_name is None else None
    fname = c.field if c is not None else str(field_name)
    dset = c.dataset if c is not None else str(dataset)
    values = [p.answer.values[0] for p in cands]
    kept = ("requested", "resolved")
    if c is None:  # a row group: the echo items every lookup shares
        echo = [i for i in cands[0].answer.interpretation if i.status in kept]
        echo = [i for i in echo if all(i in p.answer.interpretation for p in cands)]
    else:
        echo = [i for i in a.interpretation if i.key != fname and i.status in kept]
    if fname != "area":
        shown = [
            next((i.value for i in p.answer.interpretation if i.key == fname), "") for p in cands
        ]
        if all(shown) and not any(i.key == fname for i in echo):
            label = L.FIELD_LABELS.get(fname, c.field_label if c is not None else "")
            echo.append(Interpretation(fname, label, " / ".join(shown), "resolved", tuple(shown)))
    return Answer(
        request_id=request_id,
        question=a.question,
        response_type="structured_value",
        headline=L.SIDE_BY_SIDE_HEADLINE[fname],
        interpretation=echo,
        values=values,
        notices=[
            *(Notice("note", n) for n in notices),
            Notice("note", L.SIDE_BY_SIDE_NOTE),
            *cands[0].answer.notices,
        ],
        sources=_unique_sources([s for p in cands for s in p.answer.sources]),
        safety={
            "route": a.safety.get("route"),
            "status": "certified_rows",
            "certified_row_ids": [h.record_id for p in cands for h in p.result.structured],
            "prose_chunk_ids": [],
            "composer": COMPOSER_VERSION,
            "composite": {
                "kind": "side_by_side",
                "field": fname,
                "dataset": dset,
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


def city_flow(run: Run, store: StructuredStore, base: str, name: str, request_id: str) -> Reply:
    """The climate clarification of ``base`` answered with the city ``name`` chosen from the
    certified city list (:mod:`mabhas19.api.cities`; the server accepts listed names only)."""
    base_part = _call(run, base, request_id)
    plain = Reply(base_part.answer, base_part.result, [base_part])
    clar = _climate_question(base_part)
    if clar is None:  # the question no longer asks for a climate: show what it does
        return plain
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
        prompt=L.CITY_CONFLICT_PROMPT,
        options=offered,
        note=L.CITY_CONFLICT.format(city=name),
        free_text=False,
        free_text_prefix="",
        placeholder="",
        city_select=False,  # only the printed classes of this city
    )
    composite = replace(
        base_part.answer,
        headline=L.CITY_CONFLICT_PROMPT,
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


# ---------------------------------------------------------------- row groups


FAILED = frozenset({"prose", "refusal", "out_of_scope"})
DAYLIGHT = f"{MAIN}19-5-6"


def _choice_problems(answer: Answer, store: StructuredStore, parts: list[Part]) -> list[str]:
    """Problems of a row-group clarification: printed options only, no value, and no number
    that is not in the question, an option, the table citation or a checked part."""
    problems = [f"part: {p.question[:40]} failed its own check" for p in parts if not _clean(p)]
    if answer.values or answer.context:
        problems.append("values: a clarification shows no value")
    allowed = numbers(answer.question)
    allowed |= {n for a in (p.answer for p in parts) for t in a.texts() for n in numbers(t)}
    for c in answer.clarifications:
        printed = set(O.printed_options(store, c.dataset, c.field))
        for o in c.options:
            if o.label not in printed or o.text != o.label:
                problems.append(f"option: «{o.label[:40]}» is not a printed row")
            allowed |= numbers(o.label)
        allowed |= numbers(c.table)
    for text in answer.texts():
        for num in sorted(numbers(text) - allowed):
            problems.append(f"numeric: {num} in «{text[:60]}» appears in no source")
    return problems


def _choice(  # noqa: PLR0913, PLR0917
    store: StructuredStore,
    part: Part,
    dataset: str,
    field_name: str,
    rows: list[str],
    prompt: str,
    note: str,
    kind: str,
    asked: Q.Quantity | None = None,
) -> Reply:
    """Ask the user to choose one of ``rows`` (printed labels of ``dataset``); choose none.
    ``asked``: the engine read another entity (a misread city): nothing of that reading is
    shown, only the asked quantity."""
    clar = Clarification(
        field=field_name,
        field_label=L.FIELD_LABELS.get(field_name, ""),
        prompt=prompt,
        outcome=kind,
        dataset=dataset,
        table=_Sources(store).table(dataset)[0],
        options=tuple(ClarifyOption(r, r) for r in rows),
    )
    answer = Answer(
        request_id=part.answer.request_id,
        question=part.question,
        response_type="clarification",
        headline=prompt,
        interpretation=(
            [Interpretation("output", L.REQUESTED, asked.label, "requested")]
            if asked
            else [i for i in part.answer.interpretation if i.status == "requested"]
        ),
        # the engine's checked excerpts stay (a question may also ask something the text answers)
        excerpts=[] if asked else part.answer.excerpts,
        refusals=[] if asked else part.answer.refusals,
        notices=[Notice("note", note)] if note else [],
        clarifications=[clar],
        sources=[] if asked else part.answer.sources,
        safety={
            "route": part.answer.safety.get("route"),
            "status": "needs_clarification",
            "certified_row_ids": [],
            "prose_chunk_ids": part.answer.safety.get("prose_chunk_ids") or [],
            "composer": COMPOSER_VERSION,
            "composite": {"kind": kind, "dataset": dataset, "field": field_name},
        },
    )
    problems = _choice_problems(answer, store, [part])
    if problems:  # fail closed: the plain engine answer, with the reason recorded
        safety = {**part.answer.safety, "orchestration_rejected": problems}
        return Reply(replace(part.answer, safety=safety), part.result, [part])
    answer.safety["provenance_violations"] = []
    return Reply(answer, part.result, [part])


def _band(  # noqa: PLR0913, PLR0917
    run: Run,
    part: Part,
    rows: list[str],
    field_name: str,
    dataset: str,
    notes: tuple[str, ...],
    request_id: str,
) -> Reply | None:
    """Printed rows of one group that differ only in a printed band, each looked up separately
    and shown side by side; ``None`` when they are not all certified alike."""
    if not 2 <= len(rows) <= MAX_SIDE:
        return None
    cands = [_call(run, f"{part.question} ({r})") for r in rows]
    if not _consistent("area", cands):  # one printed row each, every other condition the same
        return None
    if [p.answer.values[0].row_label for p in cands] != rows:
        return None
    composite = _side_answer(part, cands, request_id, field_name, dataset, notes)
    plain = Reply(part.answer, part.result, [part])
    return _accept(
        composite,
        plain,
        [part, *cands],
        fixed_notices=(*notes, L.SIDE_BY_SIDE_NOTE),
        summary_keys=(field_name,),
    )


def _printed(store: StructuredStore, dataset: str, field_name: str, rows: tuple[str, ...]) -> bool:
    printed = O.printed_options(store, dataset, field_name)
    return bool(rows) and all(r in printed for r in rows)


def _band_of_refusal(part: Part, store: StructuredStore) -> list[str] | None:
    """Table 19-5-6 refused for a missing row condition while reading the use as exactly two
    printed rows of the same use that differ only in their printed area band."""
    a = part.answer
    if a.response_type != "refusal" or [r.dataset for r in a.refusals] != [DAYLIGHT]:
        return None
    if a.refusals[0].outcome != "missing_required_dimension":
        return None
    uses = [i for i in a.interpretation if " / " in i.value and i.key != "output"]
    if len(uses) != 1:
        return None
    rows = [x.strip() for x in uses[0].value.split(" / ")]
    if len(rows) != 2 or not _printed(store, DAYLIGHT, "use", tuple(rows)):
        return None
    return rows if len({r.split()[0] for r in rows}) == 1 else None


def widen(
    run: Run, store: StructuredStore, reply: Reply, request_id: str, *, misread: bool = False
) -> Reply:
    """When the engine could not answer (or read a city as another entity, ``misread``): a row
    group the question names, a band of printed rows, or the printed rows of a quantity asked
    with no row; otherwise the reply unchanged."""
    if not reply.parts or (reply.answer.response_type not in FAILED and not misread):
        return reply
    part = reply.parts[0]
    request = Q.analyse(part.question) if misread else None
    asked = request.quantity if request else None
    fam = W.family(part.question)
    if fam is not None and _printed(store, fam.dataset, fam.field, fam.rows):
        rows = list(fam.rows)
        if fam.band:
            band = _band(run, part, rows, fam.field, fam.dataset, (fam.note,), request_id)
            if band is not None:
                return band
        return _choice(
            store, part, fam.dataset, fam.field, rows, fam.prompt, fam.note, "row_group", asked
        )
    pair = _band_of_refusal(part, store)
    if pair is not None:
        band = _band(run, part, pair, "use", DAYLIGHT, (), request_id)
        if band is not None:
            return band
    missing = W.missing_row(part.question)
    if missing is not None:
        printed = O.printed_options(store, missing.dataset, missing.field)
        if printed:
            prompt = L.CLARIFY_PROMPTS.get(missing.field, L.CLARIFY_GENERIC)
            return _choice(
                store,
                part,
                missing.dataset,
                missing.field,
                printed,
                prompt,
                "",
                "missing_row",
                asked,
            )
    return reply


def _noted(reply: Reply, original: str, rules: list[W.Rule]) -> Reply:
    """The reply with a note for each wording rule applied (fixed texts, no numbers)."""
    if not rules:
        return reply
    a = reply.answer
    safety = {
        **a.safety,
        "wording": {
            "version": W.WORDING_VERSION,
            "original": original,
            "rules": [r.name for r in rules],
        },
    }
    notes = [Notice("note", r.note) for r in rules]
    answer = replace(a, notices=_unique_notices([*notes, *a.notices]), safety=safety)
    return Reply(answer, reply.result, reply.parts)


# ---------------------------------------------------------------- structured rescue


def _rescue_problems(answer: Answer, store: StructuredStore, parts: list[Part]) -> list[str]:
    """Problems of a rescue answer: no value, city rows only from the checked city lookup,
    printed options only, and no number without a source."""
    problems = [f"part: {p.question[:40]} failed its own check" for p in parts if not _clean(p)]
    if answer.values:
        problems.append("values: a rescue answer shows no value")
    part_values = [v for p in parts for v in p.answer.values]
    for v in answer.context:
        if v not in part_values:
            problems.append(f"context: {v.record_id} is not a value of a checked part")
    allowed = numbers(answer.question)
    allowed |= {n for p in parts for t in p.answer.texts() for n in numbers(t)}
    for c in answer.clarifications:
        printed = set(O.printed_options(store, c.dataset, c.field))
        for o in c.options:
            if o.label not in printed or o.text != o.label:
                problems.append(f"option: «{o.label[:40]}» is not a printed row")
            allowed |= numbers(o.label)
        allowed |= numbers(c.table)
    for r in answer.refusals:
        allowed |= numbers(r.table)
    allowed |= {n for s in answer.sources for n in numbers(s.label)}
    for text in answer.texts():
        for num in sorted(numbers(text) - allowed):
            problems.append(f"numeric: {num} in «{text[:60]}» appears in no source")
    return problems


def _rescued(  # noqa: PLR0913
    part: Part,
    store: StructuredStore,
    quantity: Q.Quantity,
    city: City | None,
    parts: list[Part],
    *,
    question: str,
    refusal: Refusal | None = None,
    clarification: Clarification | None = None,
) -> Reply:
    context = list(city.rows) if city else []
    notices = [Notice("note", L.NOTICE_CITY_CLIMATE), *city.notices()] if city else []
    sources = _unique_sources([v.source for v in context if v.source]) if city else []
    rtype: Literal["clarification", "refusal"] = "clarification" if clarification else "refusal"
    head = clarification.prompt if clarification else (refusal.message if refusal else "")
    answer = Answer(
        request_id=part.answer.request_id,
        question=question,
        response_type=rtype,
        headline=head,
        interpretation=[Interpretation("output", L.REQUESTED, quantity.label, "requested")],
        refusals=[refusal] if refusal else [],
        notices=_unique_notices(notices),
        clarifications=[clarification] if clarification else [],
        sources=sources,
        context=context,
        safety={
            "route": part.answer.safety.get("route"),
            "status": "needs_clarification" if clarification else "refused",
            "certified_row_ids": [],
            "prose_chunk_ids": [],
            "composer": COMPOSER_VERSION,
            "composite": {
                "kind": "rescue",
                "quantity": quantity.id,
                "replaced": part.answer.response_type,
                **({"city": _city_record(city, city.codes)} if city else {}),
            },
        },
    )
    problems = _rescue_problems(answer, store, parts)
    if problems:  # fail closed: the engine's own answer, with the reason recorded
        safety = {**part.answer.safety, "orchestration_rejected": problems}
        return Reply(replace(part.answer, safety=safety), part.result, [part])
    answer.safety["provenance_violations"] = []
    return Reply(answer, part.result, parts)


def _city_read_as_row(part: Part, request: Q.Request) -> bool:
    """The engine refused only because it read the words left besides the quantity as an
    unknown row of the quantity's climate-dependent table (whether those words are a printed
    city is decided afterwards by the exact certified city lookup, never here)."""
    a, r = part.answer, part.result
    tables = {d for d, _ in request.quantity.tables}
    if a.response_type != "refusal" or a.values or a.clarifications or not request.leftover:
        return False
    if not request.quantity.climate_table or request.code:
        return False
    if len(a.refusals) != 1 or a.refusals[0].outcome != "unknown_entity":
        return False
    if not r.no_match or any(n.outcome != "unknown_entity" for n in r.no_match):
        return False
    if any(n.dataset not in tables for n in r.no_match) or r.structured:
        return False
    unknown = r.route.signals.get("unknown_entities") or []
    tokens = [str(t) for u in unknown for t in (u.get("tokens") or [])]
    datasets = {str(u.get("dataset")) for u in unknown}
    return datasets <= tables and W.letters(" ".join(tokens)) == " ".join(request.leftover)


def _rescue_city(
    run: Run, store: StructuredStore, part: Part, request: Q.Request, *, misread: bool
) -> tuple[bool, City | None, str]:
    """(whether the rescue may go on, the certified city, the city as written). Only a printed
    city may be left besides the quantity, and only where climate applies; after a misread
    (:func:`city_misread`) only the one printed city written is used, whatever else is left."""
    if not request.leftover:
        return True, None, ""
    if misread:
        written = _printed_cities(store, part.question)
        if request.quantity.climate_table and not request.code and len(written) == 1:
            city = city_lookup(run, written[0])
            return True, city, written[0] if city else ""
        return True, None, ""
    if not request.quantity.climate_table or len(request.leftover) > Q.MAX_PLACE_WORDS:
        return False, None, ""
    if request.code:
        return False, None, ""
    name = " ".join(request.leftover)
    city = city_lookup(run, name)
    return city is not None, city, name


def rescue(  # noqa: PLR0911
    run: Run, store: StructuredStore, reply: Reply, *, misread: bool = False
) -> Reply:
    """A value request for exactly one table quantity that the engine answered with prose or
    out of scope (nothing else named but filler, a printed climate code or a printed city) gets
    a typed answer instead: the climate's absence from the table, the table's printed rows to
    choose from, or a refusal asking for the row. One refusal is also replaced: an unknown row
    that is exactly the words left besides the quantity when those words are a printed city
    (the certified city lookup decides). The engine's reading of a written city as another
    entity (``misread``, :func:`city_misread`) is replaced too, whatever else the question says:
    only the written printed city is looked up. Anything else keeps the engine's answer."""
    if not reply.parts:
        return reply
    part = reply.parts[0]
    request = Q.analyse(part.question)
    if request is None:
        return reply
    misread = misread and reply.answer is part.answer  # not when a row group already answers
    as_row = _city_read_as_row(part, request)
    if reply.answer.response_type not in ("prose", "out_of_scope") and not (as_row or misread):
        return reply
    quantity = request.quantity
    sources = _Sources(store)
    go_on, city, place_name = _rescue_city(run, store, part, request, misread=misread)
    if not go_on:
        return reply
    codes = city.codes if city else ([request.code] if request.code else [])
    parts = [part, city.part] if city else [part]
    if not quantity.climate_table and codes:
        return reply
    question = part.question
    if quantity.climate_table and codes:
        table = sources.table(quantity.climate_table)[0]
        printed = O.printed_options(store, quantity.climate_table, CLIMATE)
        if len(codes) == 1 and codes[0] not in printed:
            place = f" («{place_name}»)" if city else ""
            message = L.RESCUE_CLIMATE.format(code=codes[0], place=place, table=table)
            refusal = Refusal(
                "climate_not_printed",
                "climate_class_not_in_table",
                message,
                dataset=quantity.climate_table,
                table=table,
            )
            return _rescued(part, store, quantity, city, parts, question=question, refusal=refusal)
        if len(codes) == 1 and city:  # the next step keeps the city's printed class
            question = f"{question} ({O.option_text(CLIMATE, codes[0])[1]})"
    rows = quantity.rows(store)
    if quantity.choose and len(quantity.tables) == 1 and 0 < len(rows) <= MAX_OPTIONS:
        dataset, field_name = quantity.tables[0]
        clar = Clarification(
            field=field_name,
            field_label=L.FIELD_LABELS.get(field_name, ""),
            prompt=L.CLARIFY_PROMPTS.get(field_name, L.CLARIFY_GENERIC),
            outcome="missing_row",
            dataset=dataset,
            table=sources.table(dataset)[0],
            options=tuple(ClarifyOption(r, r) for r in rows),
        )
        return _rescued(part, store, quantity, city, parts, question=question, clarification=clar)
    dataset, field_name = quantity.tables[-1]
    table = sources.table(dataset)[0]
    if quantity.id == "u_value":
        message = L.RESCUE_U.format(
            transparent=sources.table(Q.T19_5_2)[0], opaque=sources.table(Q.T19_5_1)[0]
        )
    else:
        label = L.RESCUE_FIELD.get(field_name, L.FIELD_LABELS.get(field_name, ""))
        tables = " و ".join(dict.fromkeys(sources.table(d)[0] for d, _ in quantity.tables))
        message = L.RESCUE_ROW.format(table=tables, field=label)
        if quantity.climate_table and not codes:
            message += L.RESCUE_ROW_CLIMATE
    refusal = Refusal(
        "missing_row",
        "row_not_named",
        message,
        dataset=dataset,
        missing=(L.RESCUE_FIELD.get(field_name, L.FIELD_LABELS.get(field_name, "")),),
        table=table,
    )
    return _rescued(part, store, quantity, city, parts, question=question, refusal=refusal)


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
    """The answer to one request: a city chosen for a climate question is looked up first;
    otherwise one engine call, shown side by side when its only missing column is small."""
    if follow_up and base and follow_up.get("field") == CLIMATE and follow_up.get("kind") == "city":
        return city_flow(run, store, base, str(follow_up.get("text") or ""), request_id)
    chosen = row_choice(run, store, follow_up, base) or family_choice(store, follow_up, base)
    if chosen is not None:  # the chosen printed row is the question's only name for the thing
        question = chosen
    text, rules = W.rewrite(question)
    part = _call(run, text, request_id)
    misread = city_misread(store, part)
    reply = side_by_side(run, store, part, request_id)
    reply = widen(run, store, reply, request_id, misread=misread)
    reply = rescue(run, store, reply, misread=misread)
    if chosen is not None:
        carried = carry_city(run, store, chosen, reply, request_id)
        if carried is not None:
            reply = carried
    return _noted(reply, question, rules)


def _word_keys(text: str) -> list[str]:
    return [
        O.label_key(w) for w in re.findall(r"[^\W\d_]+", W.letters(text).replace("\u200c", " "))
    ]


def city_misread(store: StructuredStore, part: Part) -> bool:
    """A - the engine read an exact printed city written in the question as another entity: no
    value, one table quantity asked (:func:`mabhas19.api.quantities.analyse`), no explicit table
    or section reference, and a plan outside that quantity's tables rests on a non-place entity
    one of whose printed label words merely begins with the written city («تبریز» ->
    «چوب‌های طبیعی تبریزی») while that label word itself is not written («چوب تبریزی» is)."""
    a, r = part.answer, part.result
    if a.values or a.response_type == "structured_value":
        return False
    request = Q.analyse(part.question)
    if request is None:
        return False
    if r.route.signals.get("table_ref") or r.route.signals.get("section_ref"):
        return False
    cities = {O.label_key(c) for c in _printed_cities(store, part.question)}
    if not cities:
        return False
    asked = {d for d, _ in request.quantity.tables}
    written = set(_word_keys(part.question))
    for plan in r.plans:
        if str(plan.get("target")) in asked:
            continue
        for entity in plan.get("entities") or []:
            if entity.get("kind") == "place":
                continue
            for w in _word_keys(str(entity.get("text") or "")):
                if w not in written and any(w != c and w.startswith(c) for c in cities):
                    return True
    return False


def carry_city(
    run: Run, store: StructuredStore, chosen: str, reply: Reply, request_id: str
) -> Reply | None:
    """C - after a row choice, a question that names exactly one printed city and is answered
    with the climate question continues through the certified city lookup (:func:`city_flow`;
    a conflict still stops and asks). ``None`` otherwise."""
    if reply.answer.response_type != "clarification" or not reply.parts:
        return None
    if reply.answer is not reply.parts[0].answer or _climate_question(reply.parts[0]) is None:
        return None
    written = _printed_cities(store, chosen)
    if len(written) != 1:
        return None
    return city_flow(run, store, chosen, written[0], request_id)


ROW_FIELDS = frozenset(
    {"element", "component", "equipment", "place_of_use", "use", "space", "occupancy", "material"}
)
_RATING = re.compile(r"رده(?:[\s\u200c]+انرژی)?[\s\u200c]+([A-D])(?![A-Za-z+])")
_ANY_DIGIT = re.compile(r"[0-9]")


def _printed_cities(store: StructuredStore, text: str) -> list[str]:
    """Printed city names written in ``text`` (exact, as written; one to three words), from the
    certified city list (:func:`mabhas19.api.cities.catalogue`)."""
    names = {e.key for e in catalogue(store)}
    words = re.findall(r"[^\W\d_]+", text.replace("\u200c", " "))
    out: list[str] = []
    for n in (3, 2, 1):
        for i in range(len(words) - n + 1):
            phrase = " ".join(words[i : i + n])
            if O.label_key(phrase) in names and not any(phrase in c for c in out):
                out.append(phrase)
    return out


def row_choice(
    run: Run, store: StructuredStore, follow_up: dict[str, Any] | None, base: str | None
) -> str | None:
    """A printed row chosen from a row clarification becomes the follow-up's only name for the
    thing: the quantity's printed name + the printed row + only the conditions the user wrote (a
    printed climate code, «رده X», a printed city). The user's other words are dropped, so an
    extra word («دیوار بیرونی») cannot stop the lookup once every condition is known. A question
    with any other number keeps the plain follow-up (its numbers are conditions only the engine
    may read)."""
    if not follow_up or not base or follow_up.get("kind") != "option":
        return None
    field_name = str(follow_up.get("field") or "")
    if field_name not in ROW_FIELDS:
        return None
    text = W.letters(base).translate(Q._DIGITS)
    codes = sorted({f"{a}{b.upper()}" for a, b in Q._CODE.findall(text)})
    if len(codes) > 1 or _ANY_DIGIT.search(Q._CODE.sub(" ", text)):
        return None
    request = Q.analyse(base)
    if request is not None:
        head = Q.CANONICAL[request.quantity.id]
        tables = [(d, f) for d, f in request.quantity.tables if f == field_name]
    else:  # the engine's own row question: its plan names the table and the asked output
        part = _call(run, base)
        plans = [p for p in part.result.plans if p.get("requested_field")]
        if len(plans) != 1:
            return None
        head = O.plain(L.OUTPUT_LABELS.get(str(plans[0]["requested_field"]), ""))
        tables = [(str(plans[0].get("target")), field_name)]
    want = O.label_key(str(follow_up.get("text") or ""))  # the page drops parentheses
    rows = list(
        dict.fromkeys(
            r for d, f in tables for r in O.printed_options(store, d, f) if O.label_key(r) == want
        )
    )
    if not head or len(rows) != 1:
        return None
    tail = f" در اقلیم {codes[0]}" if codes else ""
    if rating := _RATING.search(text):
        tail += f" رده {rating.group(1)}"
    if not codes:
        tail += "".join(f" در {c}" for c in _printed_cities(store, text))
    return f"{head} {rows[0]}{tail}"


def family_choice(
    store: StructuredStore, follow_up: dict[str, Any] | None, base: str | None
) -> str | None:
    """``base`` with its family word replaced by the printed row the user chose from that
    family's clarification, or ``None`` when this follow-up is not such a choice."""
    if not follow_up or not base or follow_up.get("kind") != "option":
        return None
    fam = W.family(base)
    if fam is None or follow_up.get("field") != fam.field:
        return None
    if not _printed(store, fam.dataset, fam.field, fam.rows):
        return None
    want = O.label_key(str(follow_up.get("text") or ""))  # the page drops parentheses
    rows = [r for r in fam.rows if O.label_key(r) == want]
    return W.with_row(base, fam, rows[0]) if len(rows) == 1 else None
