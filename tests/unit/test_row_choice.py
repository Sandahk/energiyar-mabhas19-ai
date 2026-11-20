"""A printed row chosen from a row clarification is the follow-up's only name for the thing
(``orchestration.row_choice``): «quantity -> row choice -> climate / city -> value» flows reach the
certified value even when the original wording had an extra word («دیوار بیرونی»)."""

from __future__ import annotations

from functools import cache
from typing import Any

import pytest
from tests.unit.test_natural1 import _ask, _engine

from mabhas19.api.orchestration import row_choice
from mabhas19.api.server import clarify_request
from mabhas19.core.paths import CURATED_DIR
from mabhas19.generation.contract import Answer
from mabhas19.retrieval.structured import StructuredStore

WALL = "دیوار خارجی مجاور فضای باز"
U_FIXED = "حداکثر ضریب انتقال حرارتی U-Value (W/m2.k) – ثابت"


@cache
def _store() -> StructuredStore:
    return StructuredStore(CURATED_DIR)


def _page(a: Answer, kind: str, text: str) -> Answer:
    c = a.clarifications[0]
    follow = clarify_request({"parent": "0" * 32, "field": c.field, "kind": kind, "text": text})
    assert follow is not None
    return _ask(f"{a.question} ({follow['text']})", follow_up=follow, base=a.question)


def _pick(a: Answer, label: str) -> Answer:
    assert a.response_type == "clarification", a.headline
    option = next(o for o in a.clarifications[0].options if o.label == label)
    return _page(a, "option", option.text)


def _same_as(final: Answer, direct: str) -> None:
    assert final.response_type == "structured_value", final.headline
    assert not final.safety.get("provenance_violations")
    expected = _ask(direct)
    assert expected.values
    assert [(v.record_id, v.value) for v in final.values] == [
        (v.record_id, v.value) for v in expected.values
    ]


def test_the_reported_flow_wall_r_then_row_then_tehran_gives_the_value() -> None:
    chosen = _pick(_ask("برای دیوار بیرونی حداقل R چقدره؟"), WALL)
    assert "بیرونی" not in chosen.question  # the user's extra word is not carried on
    assert WALL in chosen.question
    final = _page(chosen, "free", "شهر تهران")
    _same_as(final, f"حداقل مقاومت حرارتی {WALL} در اقلیم 3B")
    assert {v.value for v in final.context} == {"3B"}  # the certified Tehran rows


@pytest.mark.parametrize(
    ("question", "row", "direct"),
    [
        ("حداقل مقاومت حرارتی دیوار چقدره؟", WALL, f"حداقل مقاومت حرارتی {WALL} در اقلیم 4B"),
        (
            "برای بام حداقل مقاومت حرارتی چقدره؟",
            "سقف مجاور فضای نیمه باز کنترل نشده",
            "حداقل مقاومت حرارتی سقف مجاور فضای نیمه باز کنترل نشده در اقلیم 4B",
        ),
        ("حداکثر ضریب انتقال حرارت پنجره دوجداره چقدره؟", U_FIXED, f"{U_FIXED} در اقلیم 4B"),
        ("حداکثر ضریب انتقال حرارتی پنجره چقدر است؟", U_FIXED, f"{U_FIXED} در اقلیم 4B"),
        (
            "حداکثر SHGC مجاز چقدره؟",
            "حد اکثر ضریب بهره خورشیدی SHGC – بین %۲۰ تا %۵۰",
            "حد اکثر ضریب بهره خورشیدی SHGC – بین %۲۰ تا %۵۰ در اقلیم 4B",
        ),
    ],
)
def test_row_then_city_reaches_the_certified_value(question: str, row: str, direct: str) -> None:
    final = _page(_pick(_ask(question), row), "free", "شهر تبریز")  # Tabriz: printed 4B
    _same_as(final, direct)
    assert {v.value for v in final.context} == {"4B"}


@pytest.mark.parametrize("code", ["3B", "5B"])
def test_row_then_climate_code_reaches_the_certified_value(code: str) -> None:
    chosen = _pick(_ask("برای دیوار بیرونی حداقل R چقدره؟"), WALL)
    final = _pick(chosen, code)
    _same_as(final, f"حداقل مقاومت حرارتی {WALL} در اقلیم {code}")


def test_appendix_3_use_then_city_gives_the_printed_ratings() -> None:
    final = _page(
        _pick(_ask("شدت مصرف انرژی مجاز ساختمان اداری چقدره؟"), "اداری خصوصی"), "free", "شهر تهران"
    )
    direct = _ask("حداکثر شدت مصرف انرژی مجاز اداری خصوصی در اقلیم 3B")
    assert [(v.record_id, v.value) for v in final.values] == [
        (v.record_id, v.value) for v in direct.values
    ]


def _follow(field: str, text: str, kind: str = "option") -> dict[str, Any]:
    return {"field": field, "kind": kind, "text": text}


def test_the_canonical_question_keeps_only_the_users_printed_conditions() -> None:
    run = _engine()._run
    q = row_choice(
        run,
        _store(),
        _follow("occupancy", "اداری خصوصی"),
        "مصرف انرژی مجاز اداره رده B در اقلیم 3B",
    )
    assert q == "حداکثر شدت مصرف انرژی مجاز اداری خصوصی در اقلیم 3B رده B"
    q = row_choice(run, _store(), _follow("element", WALL), "حداقل R دیوار بیرونی تو تهران چقدره؟")
    assert q == f"حداقل مقاومت حرارتی {WALL} در تهران"


@pytest.mark.parametrize(
    ("follow", "base"),
    [
        (_follow("climate_class", "اقلیم 3B"), "برای دیوار بیرونی حداقل R چقدره؟"),  # not a row
        (_follow("element", WALL, kind="free"), "برای دیوار بیرونی حداقل R چقدره؟"),  # typed
        (_follow("element", "دیوار پارتیشن"), "برای دیوار بیرونی حداقل R چقدره؟"),  # not printed
        (_follow("occupancy", "اداری خصوصی"), "شدت مصرف انرژی مجاز اداری ۱۵۰۰ متری چقدره؟"),
    ],
)
def test_no_canonical_question_otherwise(follow: dict[str, Any], base: str) -> None:
    assert row_choice(_engine()._run, _store(), follow, base) is None
