"""Phase 3: the general fail-closed quantity guard (frozen before the code, 2026-11-23).

Once ``quantities.analyse`` identifies exactly one table quantity, the PRIMARY engine result
(values, else the first clarification, else the first refusal) must be compatible with it -
its own tables and the outputs their schemas print - unless a scoped quantity-transformation
exception applies (X1, X2, X4, X5, X6; ``eval/robustness/exceptions.yaml``). X3 city rows are
supporting context only; X7/X8 are audit-only; X9 is lexical only. After a violation the engine's
reading is suppressed and the asked quantity's existing structured path answers; the guard never
produces a value, a row or a climate.
"""

from __future__ import annotations

from functools import cache

import pytest
import yaml
from tests.unit.test_natural1 import _ask, _engine

from mabhas19.api import guard as G
from mabhas19.api import invariants as I
from mabhas19.api.cities import catalogue
from mabhas19.api.server import added_question, clarify_request
from mabhas19.core.paths import EVAL_DIR
from mabhas19.generation.contract import Answer
from mabhas19.retrieval.schema import APP02, APP03, APP07, MAIN

T19_5_1, T19_5_2, T19_3_2 = f"{MAIN}19-5-1", f"{MAIN}19-5-2", f"{MAIN}19-3-2"
CITY_TABLES = {APP02, T19_3_2}
MOGHAN = "پارس آباد مغان"
# the six multi-word cities of the audit's open findings and their certified classes
MULTI = {
    "بندر بوشهر": "1B",
    "بندر لنگه": "0B",
    "بندر ماهشهر": "1B",
    MOGHAN: "3B",
    "تربت حیدریه": "4B",
    "خرم آباد": "3A",
}
EUI = "حد مجاز مصرف انرژی در {} چقدره؟"
# the 10 open findings of the audit (test_robustness_sweeps, 2026-11-23), unchanged
FINDINGS = [
    *(EUI.format(c) for c in ("بندر بوشهر", "بندر لنگه", "بندر ماهشهر", MOGHAN)),
    *(EUI.format(c) for c in ("تربت حیدریه", "خرم آباد")),
    f"برای سقف یه ساختمان تو {MOGHAN} حداقل مقاومت حرارتی چقدره؟",
    f"حداقل مقاومت حرارتی {MOGHAN} چقدره؟",
    f"حداقل R دیوار در {MOGHAN} چقدره؟",
    f"حداکثر SHGC پنجره در {MOGHAN} چقدره؟",
]
ROOF = ("سقف مجاور فضای آزاد", "سقف مجاور فضای نیمه باز کنترل نشده")
SHGC_20_50 = "حد اکثر ضریب بهره خورشیدی SHGC – بین %۲۰ تا %۵۰"


def _full(question: str) -> tuple[Answer, object]:
    answer, result, _ = _engine().ask(question)
    return answer, result


def _raw(question: str) -> tuple[Answer, object]:
    return _engine()._run(question, "x" * 32)


def _no_city_primary(a: Answer) -> None:
    primary = {v.dataset for v in a.values}
    primary |= {c.dataset for c in a.clarifications[:1]}
    if not a.values and not a.clarifications:
        primary |= {r.dataset for r in a.refusals[:1]}
    assert not primary & CITY_TABLES, a.headline


# ---------------------------------------------------------------- the 10 frozen findings


@pytest.mark.parametrize("question", FINDINGS)
def test_the_engine_reading_of_each_finding_is_a_violation(question: str) -> None:
    answer, result = _raw(question)
    assert G.violation(_engine().store, question, answer, result)


@pytest.mark.parametrize("question", FINDINGS)
def test_each_finding_keeps_the_asked_quantity(question: str) -> None:
    a, result = _full(question)
    _no_city_primary(a)
    assert I.quantity_violations(question, a, result) == []
    assert G.violation(_engine().store, question, a, result) is None
    assert not a.values  # the guard never produces a value
    assert not a.safety.get("provenance_violations")
    assert a.safety.get("quantity_guard")  # the suppressed reading is recorded


@pytest.mark.parametrize(("city", "code"), MULTI.items())
def test_energy_limit_asks_for_the_use_with_the_city_class_as_context(city: str, code: str) -> None:
    a, _ = _full(EUI.format(city))
    assert a.response_type == "refusal"
    assert [r.dataset for r in a.refusals] == [APP03]
    assert {v.value for v in a.context} == {code}
    assert {v.dataset for v in a.context} <= CITY_TABLES
    assert [i.value for i in a.interpretation if i.key == "output"] == [
        "حداکثر شدت مصرف انرژی مجاز"
    ]


def test_r_and_shgc_findings_offer_the_asked_quantitys_rows() -> None:
    roof, _ = _full(FINDINGS[6])
    assert [o.label for o in roof.clarifications[0].options] == list(ROOF)
    for q in FINDINGS[7:9]:
        a, _ = _full(q)
        c = a.clarifications[0]
        assert (c.dataset, c.field) == (T19_5_1, "element")
        assert set(ROOF) <= {o.label for o in c.options}
        assert {v.value for v in a.context} == {"3B"}
    shgc, _ = _full(FINDINGS[9])
    c = shgc.clarifications[0]
    assert (c.dataset, c.field) == (T19_5_2, "component")
    assert all("SHGC" in o.label for o in c.options)


# ---------------------------------------------------------------- what must not change


@pytest.mark.parametrize(("city", "code"), MULTI.items())
@pytest.mark.parametrize("template", ["رده اقلیمی {} چیست؟", "اقلیم {} چیه؟"])
def test_direct_climate_questions_still_answer_the_climate(
    city: str, code: str, template: str
) -> None:
    a, _ = _full(template.format(city))
    assert a.response_type == "structured_value"
    assert {v.value for v in a.values} == {code}
    assert {v.dataset for v in a.values} <= CITY_TABLES
    assert not a.safety.get("quantity_guard")


@pytest.mark.parametrize("city", MULTI)
@pytest.mark.parametrize(
    "template",
    [
        "حد مجاز مصرف انرژی در {} چقدره؟",
        "حداقل مقاومت حرارتی {} چقدره؟",
        "حداکثر ضریب انتقال حرارتی پنجره در {} چقدره؟",
        "حداکثر SHGC پنجره در {} چقدره؟",
    ],
)
def test_multi_word_city_templates_never_finish_with_another_quantity(
    city: str, template: str
) -> None:
    q = template.format(city)
    a, result = _full(q)
    _no_city_primary(a)
    assert I.quantity_violations(q, a, result) == []


def test_the_original_tabriz_misreading_is_a_violation_and_is_suppressed() -> None:
    q = "برای سقف یه ساختمان تو تبریز حداقل مقاومت حرارتی چقدره؟"
    raw, result = _raw(q)
    assert G.violation(_engine().store, q, raw, result)
    a, _ = _full(q)
    assert APP07 not in {c.dataset for c in a.clarifications}
    assert [o.label for o in a.clarifications[0].options] == list(ROOF)


def test_tabrizi_wood_written_keeps_its_appendix_7_path() -> None:
    for q in ("مقاومت حرارتی چوب تبریزی چقدره؟", "ضریب هدایت حرارتی چوب تبریزی چقدره؟"):
        raw, result = _raw(q)
        assert G.violation(_engine().store, q, raw, result) is None, q
        a, _ = _full(q)
        assert not a.safety.get("quantity_guard")
    a, _ = _full("ضریب هدایت حرارتی چوب تبریزی چقدره؟")
    assert [(v.dataset, v.value) for v in a.values] == [(APP07, "۰٫۱۲")]


def _case(set_name: str, case_id: str) -> str:
    cases = yaml.safe_load((EVAL_DIR / "retrieval" / f"{set_name}.yaml").read_text("utf-8"))
    return str(next(c["query"] for c in cases["cases"] if c["id"] == case_id))


@pytest.mark.parametrize(
    ("xid", "question"),
    [
        ("X1", "مقاومت حرارتی چوب تبریزی چقدره؟"),
        ("X2", "حداکثر ضریب انتقال حرارتی دیوار خارجی مجاور فضای باز در اقلیم 3B چقدره؟"),
        ("X3", "حداقل مقاومت حرارتی برای اردبیل چقدره؟"),
        ("X4", _case("final3", "K119")),
        ("X4/X5", _case("final", "F078")),
        ("X5", _case("final2", "G077")),
        ("X6", _case("final2", "G106")),
        ("X7", _case("final3", "K123")),
        ("X8", "هدف مبحث ۱۹ چیست؟"),
        ("X9", _case("final3", "K150")),
    ],
)
def test_the_reviewed_exceptions_are_not_intercepted(xid: str, question: str) -> None:
    raw, result = _raw(question)
    assert G.violation(_engine().store, question, raw, result) is None, xid
    a, _ = _full(question)
    assert not a.safety.get("quantity_guard"), xid


# ---------------------------------------------------------------- multi-step equivalence


@cache
def _names() -> frozenset[str]:
    return frozenset(e.name for e in catalogue(_engine().store))


def _option(a: Answer, label: str) -> Answer:
    c = a.clarifications[0]
    text = next(o.text for o in c.options if o.label == label)
    raw = {"parent": "0" * 32, "field": c.field, "kind": "option", "text": text}
    follow = clarify_request(raw, _names())
    assert follow is not None
    return _ask(added_question(a.question, follow), follow_up=follow, base=a.question)


def _same_as_direct(final: Answer, direct: str) -> None:
    assert final.response_type == "structured_value", final.headline
    assert not final.safety.get("provenance_violations")
    expected = _ask(direct)
    assert expected.values
    assert [(v.record_id, v.value) for v in final.values] == [
        (v.record_id, v.value) for v in expected.values
    ]


@pytest.mark.parametrize("question", [FINDINGS[6], FINDINGS[7], FINDINGS[8]])
def test_moghan_r_then_roof_row_gives_the_direct_3b_value(question: str) -> None:
    first, _ = _full(question)
    final = _option(first, ROOF[0])
    _same_as_direct(final, f"حداقل مقاومت حرارتی {ROOF[0]} در اقلیم 3B")


def test_moghan_shgc_then_band_gives_the_direct_3b_value() -> None:
    first, _ = _full(FINDINGS[9])
    final = _option(first, SHGC_20_50)
    _same_as_direct(final, f"{SHGC_20_50} در اقلیم 3B")


def test_isfahan_written_still_stops_at_the_conflict() -> None:
    first, _ = _full("برای سقف ساختمان در اصفهان حداقل مقاومت حرارتی چقدره؟")
    out = _option(first, ROOF[0])
    assert out.response_type == "clarification"
    assert not out.values
    assert [o.label for o in out.clarifications[0].options] == ["3B", "4B"]


# ---------------------------------------------------------------- generated: every listed city


def test_every_listed_city_in_every_quantity_template_keeps_the_quantity() -> None:
    """Each generated case ends with the asked quantity, a clarification or refusal for it, or a
    reviewed scoped exception - never silently with another quantity."""
    store = _engine().store
    bad = []
    for e in catalogue(store):
        for t in I.CITY_TEMPLATES:
            q = t.format(city=e.name)
            a, result = _full(q)
            if G.violation(store, q, a, result) or I.quantity_violations(q, a, result):
                bad.append(q)
    assert bad == []
