"""Follow-up orchestration: a city for the climate question, and side-by-side certified values.

Every case runs the frozen engine (lexical prose ranking; the structured path is the same). The
values asserted are the printed curated cells (checked by record ID).
"""

from __future__ import annotations

import json
from dataclasses import replace
from functools import cache
from typing import Any

from mabhas19.api import orchestration as OR
from mabhas19.api.engine import Engine
from mabhas19.core.paths import CURATED_DIR
from mabhas19.generation import labels as L
from mabhas19.generation import options as O
from mabhas19.generation.contract import Answer
from mabhas19.generation.provenance import numbers
from mabhas19.indexing.chunking import build_chunks
from mabhas19.retrieval.hybrid import Retriever
from mabhas19.retrieval.models import RetrievalResult
from mabhas19.retrieval.schema import APP02, APP03, MAIN
from mabhas19.retrieval.structured import StructuredStore

WALL = "حداقل مقاومت حرارتی دیوار خارجی مجاور فضای باز چقدر است؟"
Q2 = "برای ساختمان اداری سقف مصرف انرژی چقدره؟"
OFFICE = f"{Q2} (اداری خصوصی)"
OFFICE_3B = f"{OFFICE} (اقلیم 3B)"
RESIDENTIAL_3B_C = "حداکثر شدت مصرف انرژی مسکونی در اقلیم 3B رده C"
T19_3_2 = f"{MAIN}19-3-2"


@cache
def _store() -> StructuredStore:
    return StructuredStore(CURATED_DIR)


@cache
def _engine() -> Engine:
    path = CURATED_DIR / "text" / "m19-ed5-1404-main" / "blocks.jsonl"
    blocks = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    retriever = Retriever(build_chunks(blocks, "m19-ed5-1404-main"), _store())
    return Engine(retriever, _store(), "lexical")


@cache
def _ask(question: str) -> Answer:
    return _engine().ask(question)[0]


@cache
def _city(question: str, typed: str) -> Answer:
    """The climate question of ``question`` answered by typing ``typed`` in its text box."""
    first = _ask(question)
    c = first.clarifications[0]
    assert c.field == "climate_class"
    assert c.free_text
    text = f"{c.free_text_prefix} {typed}"
    follow = {"parent_request_id": first.request_id, "field": c.field, "kind": "free", "text": text}
    question_with_city = f"{first.question} ({text})"
    answer, _, _ = _engine().ask(question_with_city, follow_up=follow, base=first.question)
    return answer


def _ids(items: Any) -> set[str]:
    return {v.record_id for v in items}


# ---------------------------------------------------------------- the climate question


def test_the_climate_question_offers_printed_codes_and_a_city_box() -> None:
    a = _ask(WALL)
    c = a.clarifications[0]
    assert c.field == "climate_class"
    assert c.prompt == L.CLARIFY_PROMPTS["climate_class"]
    assert [o.label for o in c.options] == O.printed_options(_store(), c.dataset, "climate_class")
    assert c.free_text
    assert c.free_text_prefix == L.CITY_PREFIX
    assert not numbers(c.prompt + c.placeholder + c.hint)


def test_fixed_orchestration_texts_carry_no_number() -> None:
    texts = [
        L.CITY_UNKNOWN,
        L.CITY_UNKNOWN_PLAIN,
        L.CITY_CONFLICT,
        L.NOTICE_CITY_CLIMATE,
        L.SIDE_BY_SIDE_NOTE,
        L.CLARIFY_CLIMATE_HINT,
        *L.SIDE_BY_SIDE_HEADLINE.values(),
    ]
    assert all(not numbers(t) for t in texts)


# ---------------------------------------------------------------- city -> climate


def test_tehran_gives_3b_from_its_printed_rows_and_continues() -> None:
    a = _city(WALL, "تهران")
    assert a.response_type == "structured_value"
    assert [(v.record_id, v.value) for v in a.values] == [("main-t19-5-1-r02c03", "۱/۴۳")]
    assert {v.dataset for v in a.context} == {APP02, T19_3_2}
    assert {v.value for v in a.context} == {"3B"}
    assert "main-t19-3-2-r08" in _ids(a.context)
    assert any(n.text == L.NOTICE_CITY_CLIMATE for n in a.notices)
    assert any(i.key == "climate_class" and i.value == "3B" for i in a.interpretation)
    assert a.safety["provenance_violations"] == []
    assert a.safety["composite"]["city"]["codes"] == ["3B"]


def test_a_city_printed_only_in_appendix_2() -> None:
    a = _city(WALL, "آباده")
    assert [(v.record_id, v.value) for v in a.values] == [("main-t19-5-1-r02c04", "۱/۶۹")]
    assert [(v.dataset, v.row_label, v.value) for v in a.context] == [(APP02, "آباده", "4B")]


def test_an_unknown_city_asks_for_the_climate_again() -> None:
    a = _city(WALL, "شهرک ناموجود")
    assert a.response_type == "clarification"
    c = a.clarifications[0]
    assert c.field == "climate_class"
    assert [o.label for o in c.options] == O.printed_options(_store(), c.dataset, "climate_class")
    assert "شهرک ناموجود" in c.note
    assert not a.values
    assert not a.context


def test_no_partial_or_wider_city_name_is_accepted() -> None:
    for typed in ("تهرا", "تهران مهرآباد", "اصفهان شمالی"):
        assert OR.city_lookup(_engine()._run, typed) is None, typed


def test_only_rows_printed_with_exactly_that_name_are_used() -> None:
    """The engine also returns Appendix 2's «بندر بوشهر» for «بوشهر»; it is never used."""
    city = OR.city_lookup(_engine()._run, "بوشهر")
    assert city is not None
    assert [(v.dataset, v.row_label) for v in city.rows] == [(T19_3_2, "بوشهر")]
    assert any(v.row_label == "بندر بوشهر" for v in city.part.answer.values)


def test_isfahan_shows_both_printed_classes_and_chooses_none() -> None:
    a = _city(WALL, "اصفهان")
    assert a.response_type == "clarification"
    assert not a.values
    c = a.clarifications[0]
    assert [o.label for o in c.options] == ["3B", "4B"]
    assert not c.free_text
    assert "اصفهان" in c.note
    assert {(v.dataset, v.value) for v in a.context} == {(APP02, "4B"), (T19_3_2, "3B")}
    assert any(n.kind == "conflict" for n in a.notices)
    assert any("SI-0009" in n.ids for n in a.notices)
    assert a.safety["provenance_violations"] == []


def test_ardabil_5c_uses_the_printed_5b_5c_column_of_19_5_1() -> None:
    a = _city(WALL, "اردبیل")
    assert [(v.record_id, v.value) for v in a.values] == [("main-t19-5-1-r02c05", "۱/۹۶")]
    assert {v.value for v in a.context} == {"5C"}
    assert any(i.key == "climate_class" and i.value == "5C" for i in a.interpretation)


def test_ardabil_5c_is_still_refused_by_appendix_3() -> None:
    a = _city(OFFICE, "اردبیل")
    assert a.response_type == "refusal"
    assert not a.values
    assert a.refusals[0].reason == "climate_class_not_in_table"
    assert {v.value for v in a.context} == {"5C"}
    assert any("SI-0002" in n.ids for n in a.notices)


def test_a_typed_printed_code_is_taken_as_the_code() -> None:
    a = _city(WALL, "3b")
    assert [(v.record_id, v.value) for v in a.values] == [("main-t19-5-1-r02c03", "۱/۴۳")]
    assert not a.context


# ---------------------------------------------------------------- side by side


def _single(question: str) -> Answer:
    return _engine()._run(question, "s" * 32)[0]


def test_question_2_shows_all_four_ratings_each_certified_on_its_own() -> None:
    a = _ask(OFFICE_3B)
    assert a.response_type == "structured_value"
    assert not a.clarifications
    assert [(v.record_id, v.value) for v in a.values] == [
        ("app03-A-r05-3B", "33"),
        ("app03-B-r05-3B", "67"),
        ("app03-C-r05-3B", "100"),
        ("app03-D-r05-3B", "133"),
    ]
    for v, rating in zip(a.values, "ABCD", strict=True):  # identical to the one-rating answer
        assert _single(f"{OFFICE_3B} (رده انرژی {rating})").values == [v]
    assert any(i.key == "rating" and i.value == "A / B / C / D" for i in a.interpretation)
    assert not numbers(a.headline)
    assert a.safety["composite"]["kind"] == "side_by_side"
    assert a.safety["provenance_violations"] == []


def test_city_then_side_by_side_for_question_2() -> None:
    a = _city(OFFICE, "تهران")
    assert [v.value for v in a.values] == ["33", "67", "100", "133"]
    assert {v.value for v in a.context} == {"3B"}
    assert a.safety["provenance_violations"] == []


def test_residential_area_bands_side_by_side() -> None:
    a = _ask(RESIDENTIAL_3B_C)
    assert a.response_type == "structured_value"
    assert {(v.row_label, v.value) for v in a.values} == {
        ("مسکونی ≥ ۳۰۰۰ مترمربع", "131"),
        ("مسکونی < ۳۰۰۰ مترمربع", "90"),
    }
    assert {v.dataset for v in a.values} == {APP03}


def test_two_missing_items_are_still_asked() -> None:
    a = _ask("حداکثر شدت مصرف انرژی مسکونی در اقلیم 3B")
    assert a.response_type == "clarification"
    assert a.clarifications[0].field == "area"


def test_row_choices_are_still_asked_never_shown_side_by_side() -> None:
    occupancy = _ask(Q2)
    assert occupancy.response_type == "clarification"
    assert occupancy.clarifications[0].field == "occupancy"
    wall = _ask("حداقل مقاومت حرارتی دیوار چقدر است؟")
    assert wall.response_type == "clarification"
    assert wall.clarifications[0].field != "climate_class"
    assert not wall.values


def _tampered(change: str) -> OR.Run:
    """The engine, with the rating-B candidate's answer altered (as if its row differed)."""

    def run(question: str, request_id: str) -> tuple[Answer, RetrievalResult]:
        a, r = _engine()._run(question, request_id)
        if question.endswith("(رده انرژی B)") and a.values:
            v = a.values[0]
            changed = {
                "issue": replace(v, source_issues=("SI-0002",)),
                "unit": replace(v, unit="W/m2"),
                "row": replace(v, row_label="اداری عمومی"),
            }[change]
            a = replace(a, values=[changed])
        return a, r

    return run


def test_any_difference_besides_the_column_keeps_the_question() -> None:
    for change in ("issue", "unit", "row"):
        run = _tampered(change)
        base = OR.Part(OFFICE_3B, *run(OFFICE_3B, "b" * 32))
        reply = OR.side_by_side(run, _store(), base, "b" * 32)
        assert reply.answer.response_type == "clarification", change
        assert reply.answer.clarifications[0].field == "rating"


def test_the_composite_check_accepts_the_real_answer_and_rejects_a_forgery() -> None:
    run = _engine()._run
    base = OR.Part(OFFICE_3B, *run(OFFICE_3B, "c" * 32))
    reply = OR.side_by_side(run, _store(), base, "c" * 32)
    kw: dict[str, Any] = {"fixed_notices": (L.SIDE_BY_SIDE_NOTE,), "summary_keys": ("rating",)}
    assert OR.composite_problems(reply.answer, reply.parts, **kw) == []
    for p in reply.parts[1:]:  # every value certified by its own engine call
        assert p.answer.safety["provenance_violations"] == []
        assert _ids(p.answer.values) <= {h.record_id for h in p.result.structured}
    forged = replace(reply.answer, values=[replace(reply.answer.values[0], value="999")])
    problems = OR.composite_problems(forged, reply.parts, **kw)
    assert any(p.startswith("value:") for p in problems)
    assert any("numeric: 999" in p for p in problems)
    merged = replace(reply.answer, values=reply.answer.values[:1], notices=[])
    merged.notices.append(replace(reply.answer.notices[0], text="یک مقدار برای همه رده‌ها"))
    assert any(p.startswith("notice:") for p in OR.composite_problems(merged, reply.parts, **kw))
