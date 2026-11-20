"""End-to-end family-choice flows (``api.wording`` families through ``api.orchestration``).

Each flow is driven exactly as the chat page and server drive it (the answer's question plus the
chosen option, parentheses dropped by :func:`mabhas19.api.server.clarify_request`) until a final
answer, which must be the certified value the direct printed-row question gets.
"""

from __future__ import annotations

from functools import cache
from typing import Any

import pytest
from tests.unit.test_natural1 import _ask

from mabhas19.api import wording as W
from mabhas19.api.cities import catalogue
from mabhas19.api.orchestration import family_choice
from mabhas19.api.server import added_question, clarify_request
from mabhas19.core.paths import CURATED_DIR
from mabhas19.generation.contract import Answer
from mabhas19.retrieval.structured import StructuredStore

ROOF = "برای بام حداقل مقاومت حرارتی چقدره؟"
ROOF_ROWS = ("سقف مجاور فضای آزاد", "سقف مجاور فضای نیمه باز کنترل نشده")
U_ROWS = (
    "حداکثر ضریب انتقال حرارتی U-Value (W/m2.k) – ثابت",
    "حداکثر ضریب انتقال حرارتی U-Value (W/m2.k) – متحرک",
)
SHGC_ROWS = (
    "حد اکثر ضریب بهره خورشیدی SHGC – کمتر از %۲۰",
    "حد اکثر ضریب بهره خورشیدی SHGC – بین %۲۰ تا %۵۰",
    "حد اکثر ضریب بهره خورشیدی SHGC – بین %۵۰ تا %۷۰",
    "حد اکثر ضریب بهره خورشیدی SHGC – بیشتر از %۷۰",
)


@cache
def _cities() -> frozenset[str]:
    """The certified city list (contract m2-answer-4: a city is chosen, never typed)."""
    return frozenset(e.name for e in catalogue(StructuredStore(CURATED_DIR)))


def _page(a: Answer, kind: str, text: str) -> Answer:
    """One follow-up exactly as the page sends it and the server rebuilds it."""
    c = a.clarifications[0]
    raw = {"parent": "0" * 32, "field": c.field, "kind": kind, "text": text}
    follow = clarify_request(raw, _cities())
    assert follow is not None
    return _ask(added_question(a.question, follow), follow_up=follow, base=a.question)


def _option(a: Answer, label: str) -> Answer:
    assert a.response_type == "clarification"
    option = next(o for o in a.clarifications[0].options if o.label == label)
    return _page(a, "option", option.text)


def _climate(a: Answer, code: str) -> Answer:
    assert a.clarifications[0].field == "climate_class"
    option = next(o for o in a.clarifications[0].options if o.label == code)
    return _page(a, "option", option.text)


def _certified(a: Answer, row: str, direct: str) -> None:
    """``a`` is exactly the certified value of ``row`` that the direct question gets."""
    assert a.response_type == "structured_value"
    assert not a.safety.get("provenance_violations")
    assert not a.safety.get("orchestration_rejected")
    assert [v.row_label for v in a.values] == [row]
    expected = _ask(direct)
    assert expected.response_type == "structured_value"
    assert [(v.record_id, v.value) for v in a.values] == [
        (v.record_id, v.value) for v in expected.values
    ]


@pytest.mark.parametrize("row", ROOF_ROWS)
@pytest.mark.parametrize("code", ["3B", "0B", "4A"])
def test_roof_row_then_climate_code_gives_the_certified_value(row: str, code: str) -> None:
    chosen = _option(_ask(ROOF), row)
    assert "بام" not in chosen.question  # the printed row replaced the family word
    assert row in chosen.question
    final = _climate(chosen, code)
    _certified(final, row, f"حداقل مقاومت حرارتی {row} در اقلیم {code}")


@pytest.mark.parametrize(
    ("row", "city", "code"), [(ROOF_ROWS[0], "مشهد", "3B"), (ROOF_ROWS[1], "تبریز", "4B")]
)
def test_roof_row_then_city_gives_the_certified_climate_and_value(
    row: str, city: str, code: str
) -> None:
    final = _page(_option(_ask(ROOF), row), "city", city)
    _certified(final, row, f"حداقل مقاومت حرارتی {row} در اقلیم {code}")
    assert {v.value for v in final.context} == {code}
    assert all(city in v.row_label for v in final.context)


@pytest.mark.parametrize("row", ROOF_ROWS)
def test_roof_with_the_climate_in_the_question_needs_one_choice(row: str) -> None:
    final = _option(_ask("حداقل مقاومت حرارتی بام در اقلیم 3B چقدره؟"), row)
    _certified(final, row, f"حداقل مقاومت حرارتی {row} در اقلیم 3B")


@pytest.mark.parametrize("row", U_ROWS)
def test_window_u_choice_gives_the_certified_value(row: str) -> None:
    final = _option(_ask("حداکثر ضریب انتقال حرارتی پنجره در اقلیم 3B چقدر است؟"), row)
    _certified(final, row, f"{row} در اقلیم 3B")


def test_window_u_choice_then_climate_gives_the_certified_value() -> None:
    chosen = _option(_ask("حداکثر ضریب انتقال حرارتی پنجره چقدر است؟"), U_ROWS[0])
    assert "پنجره" not in chosen.question
    _certified(_climate(chosen, "2A"), U_ROWS[0], f"{U_ROWS[0]} در اقلیم 2A")


@pytest.mark.parametrize("row", SHGC_ROWS)
def test_shgc_choice_then_climate_gives_the_certified_value(row: str) -> None:
    a = _ask("حداکثر ضریب بهره خورشیدی پنجره چقدره؟")  # no climate: the bands are chosen
    assert a.response_type == "clarification"
    assert [o.label for o in a.clarifications[0].options] == list(SHGC_ROWS)
    chosen = _option(a, row)
    assert "پنجره" not in chosen.question
    _certified(_climate(chosen, "2B"), row, f"{row} در اقلیم 2B")


@pytest.mark.parametrize(
    ("base", "field", "text", "expected"),
    [
        (ROOF, "element", ROOF_ROWS[0], "برای سقف مجاور فضای آزاد حداقل مقاومت حرارتی چقدره؟"),
        (
            "حداقل مقاومت حرارتی پشت بام در اقلیم 3B",
            "element",
            ROOF_ROWS[1],
            "حداقل مقاومت حرارتی سقف مجاور فضای نیمه باز کنترل نشده در اقلیم 3B",
        ),
        (
            "حداکثر ضریب انتقال حرارتی پنجره در اقلیم 3B",
            "component",
            "حداکثر ضریب انتقال حرارتی U-Value  W/m2.k  – متحرک",  # as the server passes it
            f"حداکثر ضریب انتقال حرارتی {U_ROWS[1]} در اقلیم 3B",
        ),
    ],
)
def test_the_family_word_is_replaced_by_the_chosen_row_not_kept(
    base: str, field: str, text: str, expected: str
) -> None:
    store = StructuredStore(CURATED_DIR)
    follow: dict[str, Any] = {"field": field, "kind": "option", "text": text}
    assert family_choice(store, follow, base) == expected


@pytest.mark.parametrize(
    "follow",
    [
        {"field": "element", "kind": "option", "text": "دیوار خارجی مجاور خاک"},  # not the family's
        {"field": "climate_class", "kind": "option", "text": "اقلیم 3B"},  # another field
        {"field": "element", "kind": "free", "text": ROOF_ROWS[0]},  # typed, not chosen
    ],
)
def test_only_a_chosen_row_of_the_family_replaces_its_word(follow: dict[str, Any]) -> None:
    assert family_choice(StructuredStore(CURATED_DIR), follow, ROOF) is None


def test_with_row_replaces_only_the_family_word() -> None:
    fam = W.family(ROOF)
    assert fam is not None
    assert W.with_row(ROOF, fam, ROOF_ROWS[0]).count(ROOF_ROWS[0]) == 1
