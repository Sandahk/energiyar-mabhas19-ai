"""Generated robustness sweeps (frozen engine, lexical prose ranking; the structured path is the
same). Every question is generated from the curated catalogue (:mod:`mabhas19.api.invariants`),
not hand-picked, and every answer is checked against the invariants:

* quantity preservation (reviewed exceptions X1-X8, ``eval/robustness/exceptions.yaml``);
* no unwritten word (a resolved entity rests on a written word);
* no nearest-concept substitution (near-miss names never get another row's value);
* any path gives the direct printed question's value (row -> city, city written -> row);
* a city whose sources disagree (Isfahan) always stops and asks.
"""

from __future__ import annotations

from collections import defaultdict
from functools import cache

import pytest
import yaml
from tests.unit.test_natural1 import _ask, _engine

from mabhas19.api import invariants as I
from mabhas19.api import quantities as Q
from mabhas19.api.cities import catalogue, city_key, printed_name
from mabhas19.api.server import added_question, clarify_request
from mabhas19.core.paths import EVAL_DIR
from mabhas19.generation import options as O
from mabhas19.generation.contract import Answer
from mabhas19.retrieval.schema import APP02, MAIN

T19_5_1, T19_5_2, T19_3_2 = f"{MAIN}19-5-1", f"{MAIN}19-5-2", f"{MAIN}19-3-2"
CONFLICT = "اصفهان"


@cache
def _sweep() -> list[dict[str, str]]:
    return I.sweep_questions(_engine().store)


def _problems(question: str, label: str = "") -> list[str]:
    answer, result, _ = _engine().ask(question)
    out = I.quantity_violations(question, answer, result)
    out += I.unwritten_entity_violations(question, answer, _engine().store)
    if label:
        out += I.substitution_violations(label, answer)
    return out


# Open findings of the audit (2026-11-23), reported for review before any runtime guard (Phase 3):
# for a multi-word listed city the frozen engine plans the city's climate lookup instead of the
# asked quantity (an energy-use question gets the city's climate class; R / SHGC get a city-table
# refusal). No wrong regulatory value is shown. A new finding fails; a fixed one must be removed.
_EUI = "حد مجاز مصرف انرژی در {} چقدره؟"
_MOGHAN = "پارس آباد مغان"
KNOWN_OPEN = {
    "city_quantity": {
        *(_EUI.format(c) for c in ("بندر بوشهر", "بندر لنگه", "بندر ماهشهر", _MOGHAN)),
        *(_EUI.format(c) for c in ("تربت حیدریه", "خرم آباد")),
        f"برای سقف یه ساختمان تو {_MOGHAN} حداقل مقاومت حرارتی چقدره؟",
        f"حداقل مقاومت حرارتی {_MOGHAN} چقدره؟",
        f"حداقل R دیوار در {_MOGHAN} چقدره؟",
        f"حداکثر SHGC پنجره در {_MOGHAN} چقدره؟",
    },
}


@pytest.mark.parametrize(
    "family", ["city_quantity", "family_quantity", "near_miss_added", "near_miss_dropped"]
)
def test_generated_questions_keep_every_invariant(family: str) -> None:
    cases = [c for c in _sweep() if c["family"] == family]
    assert cases
    bad = {c["question"]: p for c in cases if (p := _problems(c["question"], c["label"]))}
    assert set(bad) == KNOWN_OPEN.get(family, set()), bad


def test_the_sweep_covers_every_listed_city_and_family() -> None:
    cities = {e.name for e in catalogue(_engine().store)}
    questions = [c["question"] for c in _sweep() if c["family"] == "city_quantity"]
    assert len(questions) == len(cities) * len(I.CITY_TEMPLATES)
    assert all(any(f" {c} " in q for q in questions) for c in cities)


# ---------------------------------------------------------------- the invariants catch the bug


def test_the_invariants_catch_the_original_tabriz_misreading() -> None:
    """The frozen engine's own answer (before the orchestration guard) breaks both invariants."""
    question = "برای سقف یه ساختمان تو تبریز حداقل مقاومت حرارتی چقدره؟"
    answer, result = _engine()._run(question, "x" * 32)
    assert I.quantity_violations(question, answer, result)
    assert I.unwritten_entity_violations(question, answer, _engine().store)
    assert _problems(question) == []  # the deployed answer keeps them


@pytest.mark.parametrize(
    ("set_name", "case_id"),
    [
        ("final2", "G077"),
        ("final", "F078"),
        ("final2", "G106"),
        ("final3", "K119"),
        ("final3", "K123"),
    ],
)
def test_the_reviewed_exceptions_are_recognized(set_name: str, case_id: str) -> None:
    cases = yaml.safe_load((EVAL_DIR / "retrieval" / f"{set_name}.yaml").read_text("utf-8"))
    question = next(c["query"] for c in cases["cases"] if c["id"] == case_id)
    assert _problems(question) == []


def test_written_tabrizi_wood_is_exception_x1_not_a_misreading() -> None:
    assert _problems("مقاومت حرارتی چوب تبریزی چقدره؟") == []
    assert _problems("ضریب هدایت حرارتی چوب تبریزی چقدره؟") == []


# ---------------------------------------------------------------- multi-step flows


@cache
def _names() -> frozenset[str]:
    return frozenset(e.name for e in catalogue(_engine().store))


@cache
def _city_codes() -> dict[str, set[str]]:
    """The printed classes of every listed city (Appendix 2 and 19-3-2 records)."""
    store = _engine().store
    codes: dict[str, set[str]] = defaultdict(set)
    records = [*store.app02, *store.main[T19_3_2].records]
    keys = {e.key: e.name for e in catalogue(store)}
    for r in records:
        name = keys[city_key(printed_name(str(r["city"]["text"])))]
        codes[name].add(str(r["climate_class"]["code"]))
    return codes


def _city_per_class(dataset: str) -> list[tuple[str, str]]:
    """One listed city per printed class of ``dataset`` (the first in list order with one class),
    for the classes 3B, 4B and 5B (three climates the tables print)."""
    printed = O.printed_options(_engine().store, dataset, "climate_class")
    out = []
    for code in ("3B", "4B", "5B"):
        if code not in printed:
            continue
        city = next((c for c, cs in _city_codes().items() if cs == {code}), None)
        if city is not None:
            out.append((city, code))
    return out


def _follow(a: Answer, kind: str, text: str) -> Answer:
    c = a.clarifications[0]
    follow = clarify_request(
        {"parent": "0" * 32, "field": c.field, "kind": kind, "text": text}, _names()
    )
    assert follow is not None, text
    return _ask(added_question(a.question, follow), follow_up=follow, base=a.question)


def _option(a: Answer, label: str) -> Answer:
    assert a.response_type == "clarification", a.headline
    return _follow(
        a, "option", next(o.text for o in a.clarifications[0].options if o.label == label)
    )


def _direct(head: str, row: str, code: str) -> list[tuple[str, str]]:
    a = _ask(f"{head} {row} در اقلیم {code}")
    return [(v.record_id, v.value) for v in a.values]


def _check_final(final: Answer, head: str, row: str, code: str) -> list[str]:
    out = []
    if final.response_type != "structured_value":
        out.append(f"{row} / {code}: {final.response_type} «{final.headline[:50]}»")
    elif [(v.record_id, v.value) for v in final.values] != _direct(head, row, code):
        out.append(f"{row} / {code}: value differs from the direct question")
    if final.safety.get("provenance_violations") or final.safety.get("orchestration_rejected"):
        out.append(f"{row} / {code}: safety")
    return out


# (start question, quantity head, table, row field) for the row -> city flows
ROW_FLOWS = [
    ("حداقل مقاومت حرارتی چقدره؟", Q.CANONICAL["r_value"], T19_5_1),
    ("حداکثر ضریب انتقال حرارتی پنجره چقدر است؟", "", T19_5_2),
    ("حداکثر SHGC مجاز چقدره؟", "", T19_5_2),
]


@pytest.mark.parametrize(("start", "head", "dataset"), ROW_FLOWS)
def test_row_then_selected_city_gives_the_direct_value(start: str, head: str, dataset: str) -> None:
    first = _ask(start)
    assert first.response_type == "clarification", first.headline
    rows = [o.label for o in first.clarifications[0].options]
    assert rows
    bad = []
    for row in rows:
        chosen = _option(first, row)
        for city, code in _city_per_class(dataset):
            final = _follow(chosen, "city", city)
            bad += _check_final(final, head, row, code)
            bad += I.quantity_violations(final.question, final)
    assert bad == []


# (family/row word, quantity phrase, quantity head) for city written -> row flows
CITY_FIRST = [
    ("دیوار", "حداقل مقاومت حرارتی", Q.CANONICAL["r_value"], T19_5_1),
    ("سقف", "حداقل مقاومت حرارتی", Q.CANONICAL["r_value"], T19_5_1),
    ("پنجره", "حداکثر ضریب انتقال حرارتی", "", T19_5_2),
]


@pytest.mark.parametrize(("word", "phrase", "head", "dataset"), CITY_FIRST)
def test_city_written_then_row_gives_the_direct_value(
    word: str, phrase: str, head: str, dataset: str
) -> None:
    bad = []
    for city, code in _city_per_class(dataset):
        first = _ask(f"برای {word} در {city} {phrase} چقدره؟")
        assert first.response_type == "clarification", first.headline
        for row in [o.label for o in first.clarifications[0].options]:
            final = _option(first, row)
            bad += _check_final(final, head, row, code)
            bad += [f"{row}: context {v.value}" for v in final.context if v.value != code]
    assert bad == []


@pytest.mark.parametrize("path", ["selected", "written"])
def test_a_city_whose_sources_disagree_always_stops(path: str) -> None:
    assert _city_codes()[CONFLICT] == {"3B", "4B"}
    if path == "selected":
        out = _follow(
            _option(_ask("حداقل مقاومت حرارتی چقدره؟"), "سقف مجاور فضای آزاد"), "city", CONFLICT
        )
    else:
        out = _option(
            _ask(f"برای سقف در {CONFLICT} حداقل مقاومت حرارتی چقدره؟"), "سقف مجاور فضای آزاد"
        )
    assert out.response_type == "clarification"
    assert not out.values
    assert [o.label for o in out.clarifications[0].options] == ["3B", "4B"]
    assert {(v.dataset, v.value) for v in out.context} == {(APP02, "4B"), (T19_3_2, "3B")}
