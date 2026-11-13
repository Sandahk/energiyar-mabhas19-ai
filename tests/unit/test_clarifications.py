"""Clarifications before refusal: a refused lookup for one missing or ambiguous item asks for it.

The retrieval engine is unchanged; the composer reads its typed outcomes. Options are printed
values only, a choice re-asks the original question with the choice added, and nothing is
answered from the choice itself.
"""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from functools import cache

import pytest

from mabhas19.api.engine import Engine
from mabhas19.core.paths import CURATED_DIR
from mabhas19.generation import options as O
from mabhas19.generation.composer import compose
from mabhas19.generation.contract import Answer, ClarifyOption
from mabhas19.generation.labels import CLARIFY_PLACEHOLDER, CLARIFY_PROMPTS
from mabhas19.generation.provenance import check, numbers
from mabhas19.indexing.chunking import build_chunks
from mabhas19.retrieval.hybrid import Retriever
from mabhas19.retrieval.models import RetrievalResult
from mabhas19.retrieval.structured import StructuredStore


@cache
def _store() -> StructuredStore:
    return StructuredStore(CURATED_DIR)


@cache
def _retriever() -> Retriever:  # the frozen router, planner and structured lookup (BM25 prose)
    path = CURATED_DIR / "text" / "m19-ed5-1404-main" / "blocks.jsonl"
    blocks = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    return Retriever(build_chunks(blocks, "m19-ed5-1404-main"), _store())


@cache
def _cached(query: str) -> RetrievalResult:
    return _retriever().retrieve(query)


WALL_U = "ضریب انتقال حرارت مجاز دیوار خارجی چقدر است؟"
EUI_PROSE = "برای یک ساختمان مسکونی در اقلیم 3B، حد مجاز مصرف انرژی چقدر است؟"
POLYCARBONATE_R = "مقاومت حرارتی پلی‌کربنات چقدر است؟"
WALL_R = "حداقل مقاومت حرارتی دیوار خارجی چقدر است؟"
EUI_NO_AREA = "شدت مصرف انرژی مجاز ساختمان مسکونی در اقلیم 1B رده A"
HOT_WATER = "جدول ۱۹-۵-۵"


def _result(query: str) -> RetrievalResult:
    return copy.deepcopy(_cached(query))


@cache
def _answer(query: str) -> Answer:
    return compose(_result(query), _store(), "c1")


def test_u_value_of_a_wall_is_not_offered_window_rows() -> None:
    """The engine matched table 19-5-2 (transparent parts) by the quantity's name only: no
    option names a wall, so none is offered and the refusal says so."""
    a = _answer(WALL_U)
    assert a.response_type == "refusal"
    assert not a.clarifications
    r = a.refusals[0]
    assert r.reason == "ungrounded_ambiguity"
    assert not r.options
    assert "جدول ۱۹-۵-۲" in r.message
    assert all(i.status != "ambiguous" for i in a.interpretation)
    assert all("u value" not in t for t in a.texts())


def test_energy_limit_question_routed_to_prose_is_not_given_a_made_up_clarification() -> None:
    a = _answer(EUI_PROSE)
    assert a.response_type == "prose"
    assert not a.clarifications


def test_r_of_a_material_explains_lambda_and_offers_only_the_printed_lambda() -> None:
    a = _answer(POLYCARBONATE_R)
    assert a.response_type == "clarification"
    assert not a.values
    c = a.clarifications[0]
    assert c.field == "output"
    for word in ("λ", "(R)", "ضخامت"):
        assert word in c.note
    assert [o.text for o in c.options] == ["ضریب هدایت حرارتی"]
    assert not numbers(c.note + c.prompt) - numbers(a.question) - {"7"}
    follow = compose(_result(f"{POLYCARBONATE_R} ({c.options[0].text})"), _store(), "c2")
    assert follow.response_type == "structured_value"
    assert follow.values[0].output == "thermal_conductivity"


def test_ambiguous_wall_asks_which_wall_then_which_climate_then_answers() -> None:
    a = _answer(WALL_R)
    assert a.response_type == "clarification"
    c = a.clarifications[0]
    assert c.field == "element"
    assert len(c.options) == 3
    assert all(o.label.startswith("دیوار خارجی") for o in c.options)
    q2 = f"{WALL_R} ({c.options[1].text})"
    second = compose(_result(q2), _store(), "c3")
    assert second.response_type == "clarification"
    c2 = second.clarifications[0]
    assert c2.field == "climate_class"
    assert [o.label for o in c2.options] == O.printed_options(_store(), c2.dataset, "climate_class")
    assert c2.hint
    third = compose(_result(f"{q2} ({c2.options[5].text})"), _store(), "c4")
    assert third.response_type == "structured_value"
    assert not third.safety["provenance_violations"]


def test_missing_area_asks_for_free_text_without_an_example_number() -> None:
    a = _answer(EUI_NO_AREA)
    assert a.response_type == "clarification"
    c = a.clarifications[0]
    assert c.field == "area"
    assert c.free_text
    assert not c.options
    assert c.free_text_prefix == "مساحت"
    assert not numbers(c.placeholder + c.prompt)
    typed = compose(_result(f"{EUI_NO_AREA} (مساحت 300 مترمربع)"), _store(), "c5")
    assert typed.response_type == "structured_value"


def test_missing_row_of_a_main_table_offers_its_printed_rows() -> None:
    a = _answer(HOT_WATER)
    c = a.clarifications[0]
    assert c.field == "place_of_use"
    assert [o.label for o in c.options] == O.printed_options(_store(), c.dataset, "place_of_use")
    for o in c.options:
        follow = compose(_result(f"{HOT_WATER} ({o.text})"), _store(), "c6")
        assert follow.response_type == "structured_value"


def test_a_forged_option_is_a_provenance_violation() -> None:
    result = _result(WALL_R)
    a = _answer(WALL_R)
    forged = replace(
        a.clarifications[0],
        options=(*a.clarifications[0].options, ClarifyOption("دیوار خارجی ۹۹", "دیوار خارجی ۹۹")),
    )
    bad = replace(a, clarifications=[forged])
    problems = check(bad, result, _store())
    assert any("not printed" in p for p in problems)
    assert any("numeric: 99" in p for p in problems)


def test_a_clarification_never_rides_on_certified_values() -> None:
    ok = compose(_result("ضریب هدایت حرارتی پلی‌کربنات چقدر است؟"), _store(), "c7")
    assert ok.response_type == "structured_value"
    c = _answer(WALL_R).clarifications[0]
    bad = replace(ok, clarifications=[c])
    problems = check(bad, _result("ضریب هدایت حرارتی پلی‌کربنات چقدر است؟"), _store())
    assert any("together with certified values" in p for p in problems)


@pytest.mark.parametrize("field", sorted(CLARIFY_PROMPTS))
def test_prompts_and_placeholders_carry_no_number(field: str) -> None:
    assert not numbers(CLARIFY_PROMPTS[field])
    assert not numbers(CLARIFY_PLACEHOLDER.get(field, ""))


PARTY_WALL = "حداقل مقاومت حرارتی دیوار بین دو واحد گرمایش‌شده در اقلیم 3B طبق جدول ۱۹-۵-۱"


def _engine() -> Engine:
    return Engine(_retriever(), _store(), "lexical")


def test_choices_that_lead_nowhere_are_dropped_and_the_refusal_stays() -> None:
    """A party wall matches the three exterior-wall rows by «دیوار», but each of them is
    refused once chosen («بین» stays unexplained): no dead-end choice is offered."""
    assert _answer(PARTY_WALL).response_type == "clarification"  # composer alone would ask
    answer, _, _ = _engine().ask(PARTY_WALL)
    assert answer.response_type == "refusal"
    assert not answer.clarifications
    assert answer.refusals[0].outcome == "ambiguous_entity"


def test_choices_that_lead_on_are_kept() -> None:
    answer, _, _ = _engine().ask(WALL_R)
    assert answer.response_type == "clarification"
    assert len(answer.clarifications[0].options) == 3


def test_printed_label_matching_ignores_spacing_and_zwnj() -> None:
    store = _store()
    ds = "m19-ed5-1404-main.t19-5-1"
    printed = O.printed_label(store, ds, "element", "دیوار خارجی مجاور فضای نیمهباز کنترل نشده")
    assert printed in O.printed_options(store, ds, "element")
    assert O.printed_label(store, ds, "element", "دیوار ناموجود") == ""
