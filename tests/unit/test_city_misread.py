"""A printed city read as another entity, «سقف» for the roof rows, and a written city carried
through a row choice (frozen before the code, 2026-11-20).

* **A - city/material collision.** «تبریز» was read by the frozen engine as the Appendix 7
  material «چوب‌های طبیعی تبریزی، اکومه» and an R / U question became a λ question. When the user
  writes an exact listed city and one table quantity is identified, an engine reading that rests
  on a different entity whose printed label merely begins with that city name (and is not itself
  written) is not shown; the asked quantity's printed rows are offered instead. The material
  written in the question («چوب تبریزی») keeps its Appendix 7 path.
* **B - «سقف» with R** offers the two printed 19-5-1 roof rows, exactly like «بام».
* **C - a city written in the question** is carried through a row choice: when the engine then
  asks for the climate, the same certified city lookup continues (a conflict still stops).
"""

from __future__ import annotations

from functools import cache

import pytest
import yaml
from tests.unit.test_natural1 import _ask, _engine

from mabhas19.api.cities import catalogue
from mabhas19.api.server import added_question, clarify_request
from mabhas19.core.paths import CURATED_DIR, EVAL_DIR
from mabhas19.generation.contract import Answer
from mabhas19.retrieval.schema import APP02, APP03, APP07, MAIN
from mabhas19.retrieval.structured import StructuredStore

T19_5_1 = f"{MAIN}19-5-1"
T19_3_2 = f"{MAIN}19-3-2"
ROOF = ("سقف مجاور فضای آزاد", "سقف مجاور فضای نیمه باز کنترل نشده")
WALL = "دیوار خارجی مجاور فضای باز"
U_FIXED = "حداکثر ضریب انتقال حرارتی U-Value (W/m2.k) – ثابت"
LAMBDA = "ضریب هدایت حرارتی (λ)"


@cache
def _cities() -> frozenset[str]:
    return frozenset(e.name for e in catalogue(StructuredStore(CURATED_DIR)))


def _follow(a: Answer, kind: str, text: str) -> Answer:
    c = a.clarifications[0]
    raw = {"parent": "0" * 32, "field": c.field, "kind": kind, "text": text}
    follow = clarify_request(raw, _cities())
    assert follow is not None, text
    return _ask(added_question(a.question, follow), follow_up=follow, base=a.question)


def _row(a: Answer, label: str) -> Answer:
    assert a.response_type == "clarification", a.headline
    option = next(o for o in a.clarifications[0].options if o.label == label)
    return _follow(a, "option", option.text)


def _no_appendix_7(a: Answer) -> None:
    """Nothing of the answer comes from, or asks about, Appendix 7 (λ)."""
    datasets = {v.dataset for v in [*a.values, *a.context]}
    datasets |= {c.dataset for c in a.clarifications} | {r.dataset for r in a.refusals}
    datasets |= {s.dataset for s in a.sources}
    assert APP07 not in datasets, a.headline
    assert not any(i.value == LAMBDA or i.key == "material" for i in a.interpretation)
    assert LAMBDA not in a.headline


def _same_as_direct(final: Answer, direct: str) -> None:
    """The multi-step answer is the direct printed question's certified value, row and source."""
    assert final.response_type == "structured_value", final.headline
    assert not final.safety.get("provenance_violations")
    assert not final.safety.get("orchestration_rejected")
    expected = _ask(direct)
    assert expected.values
    assert [(v.record_id, v.value, v.unit) for v in final.values] == [
        (v.record_id, v.value, v.unit) for v in expected.values
    ]
    assert [v.source for v in final.values] == [v.source for v in expected.values]


# ---------------------------------------------------------------- the diagnosed phrasings


@pytest.mark.parametrize(
    ("question", "city", "code"),
    [
        ("برای سقف یه ساختمان تو تبریز حداقل مقاومت حرارتی چقدره؟", "تبریز", "4B"),
        ("برای سقف ساختمان در تبریز حداقل R چقدره؟", "تبریز", "4B"),
        ("برای سقف تو تبریز R حداقلش چنده؟", "تبریز", "4B"),
        ("برای سقف ساختمان در تهران حداقل مقاومت حرارتی چقدره؟", "تهران", "3B"),
        ("برای بام ساختمان در تبریز حداقل مقاومت حرارتی چقدره؟", "تبریز", "4B"),
    ],
)
@pytest.mark.parametrize("row", ROOF)
def test_roof_with_a_written_city_offers_the_roof_rows_then_gives_the_value(
    question: str, city: str, code: str, row: str
) -> None:
    first = _ask(question)
    _no_appendix_7(first)
    assert first.response_type == "clarification", first.headline
    c = first.clarifications[0]
    assert (c.dataset, c.field) == (T19_5_1, "element")
    assert [o.label for o in c.options] == list(ROOF)
    final = _row(first, row)  # the written city is carried: no second city choice
    _no_appendix_7(final)
    _same_as_direct(final, f"حداقل مقاومت حرارتی {row} در اقلیم {code}")
    assert {v.value for v in final.context} == {code}
    assert {v.dataset for v in final.context} <= {APP02, T19_3_2}
    assert city in final.safety["composite"]["city"]["question"]


def test_r_with_only_the_city_offers_the_element_rows_with_the_certified_class() -> None:
    first = _ask("حداقل مقاومت حرارتی تبریز چقدره؟")
    _no_appendix_7(first)
    assert first.response_type == "clarification", first.headline
    c = first.clarifications[0]
    assert (c.dataset, c.field) == (T19_5_1, "element")
    assert set(ROOF) <= {o.label for o in c.options}
    assert {v.value for v in first.context} == {"4B"}
    final = _row(first, ROOF[0])
    _same_as_direct(final, f"حداقل مقاومت حرارتی {ROOF[0]} در اقلیم 4B")


def test_window_u_in_tabriz_is_a_u_question_and_carries_the_city() -> None:
    first = _ask("حداکثر ضریب انتقال حرارتی پنجره در تبریز چقدره؟")
    _no_appendix_7(first)
    assert first.response_type == "clarification", first.headline
    assert first.clarifications[0].field == "component"
    final = _row(first, U_FIXED)
    _no_appendix_7(final)
    _same_as_direct(final, f"{U_FIXED} در اقلیم 4B")
    assert {v.value for v in final.context} == {"4B"}


# ---------------------------------------------------------------- what must not change


def test_tabrizi_wood_written_in_the_question_keeps_its_appendix_7_path() -> None:
    a = _ask("مقاومت حرارتی چوب تبریزی چقدره؟")
    assert a.response_type == "clarification"
    c = a.clarifications[0]
    assert (c.field, c.dataset) == ("output", APP07)
    assert [i.value for i in a.interpretation if i.key == "output"] == [LAMBDA]
    b = _ask("ضریب هدایت حرارتی چوب تبریزی چقدره؟")
    assert b.response_type == "structured_value"
    assert [(v.dataset, v.record_id, v.value) for v in b.values] == [
        (APP07, "app07-p008-r06", "۰٫۱۲")
    ]


def test_k119_explicit_table_question_is_unchanged() -> None:
    cases = yaml.safe_load((EVAL_DIR / "retrieval" / "final3.yaml").read_text(encoding="utf-8"))
    k119 = next(c for c in cases["cases"] if c["id"] == "K119")
    a = _ask(k119["query"])
    assert a.response_type == "refusal"
    assert [i.value for i in a.interpretation if i.key == "output"] == ["بازه نسبت R"]
    assert not a.safety.get("composite")


@pytest.mark.parametrize(
    ("question", "field"),
    [
        ("برای ساختمان اداری سقف مصرف انرژی چقدره؟", "occupancy"),
        ("سقف مصرف انرژی ساختمان مسکونی در تبریز چقدره؟", "area"),
    ],
)
def test_saqf_as_an_energy_cap_is_unaffected(question: str, field: str) -> None:
    a = _ask(question)
    assert a.response_type == "clarification"
    assert [(c.dataset, c.field) for c in a.clarifications] == [(APP03, field)]
    assert not a.safety.get("composite")


def test_bam_without_a_city_is_unchanged_and_saqf_behaves_like_it() -> None:
    for word in ("بام", "سقف"):
        a = _ask(f"برای {word} حداقل مقاومت حرارتی چقدره؟")
        assert a.response_type == "clarification", word
        assert [o.label for o in a.clarifications[0].options] == list(ROOF)
        chosen = _row(a, ROOF[1])
        assert chosen.clarifications[0].field == "climate_class"  # no city written: asked
        assert chosen.clarifications[0].city_select


def test_a_named_roof_row_is_not_a_family() -> None:
    a = _ask(f"حداقل مقاومت حرارتی {ROOF[0]} در اقلیم 3B چقدره؟")
    _same_as_direct(a, f"حداقل مقاومت حرارتی {ROOF[0]} در اقلیم 3B")


def test_tehran_wall_control_still_carries_the_city() -> None:
    first = _ask("برای دیوار در تهران حداقل مقاومت حرارتی چقدره؟")
    final = _row(first, WALL)
    _same_as_direct(final, f"حداقل مقاومت حرارتی {WALL} در اقلیم 3B")
    assert {v.value for v in final.context} == {"3B"}


# ---------------------------------------------------------------- the conflict is preserved


def test_isfahan_written_in_the_question_stops_at_the_conflict() -> None:
    first = _ask("برای سقف ساختمان در اصفهان حداقل مقاومت حرارتی چقدره؟")
    assert [o.label for o in first.clarifications[0].options] == list(ROOF)
    out = _row(first, ROOF[0])
    assert out.response_type == "clarification"
    assert not out.values
    c = out.clarifications[0]
    assert c.field == "climate_class"
    assert [o.label for o in c.options] == ["3B", "4B"]
    assert not c.city_select
    assert {(v.dataset, v.value) for v in out.context} == {(APP02, "4B"), (T19_3_2, "3B")}
    final = _follow(out, "option", c.options[0].text)
    _same_as_direct(final, f"حداقل مقاومت حرارتی {ROOF[0]} در اقلیم 3B")


# ---------------------------------------------------------------- every city


TEMPLATES = (
    "برای سقف یه ساختمان تو {c} حداقل مقاومت حرارتی چقدره؟",
    "حداقل مقاومت حرارتی {c} چقدره؟",
    "حداکثر ضریب انتقال حرارتی پنجره در {c} چقدره؟",
    "حد مجاز مصرف انرژی در {c} چقدره؟",
)


def test_no_listed_city_turns_a_table_quantity_into_an_appendix_7_question() -> None:
    bad = []
    for e in catalogue(_engine().store):
        for t in TEMPLATES:
            a = _ask(t.format(c=e.name))
            lam = any(i.value == LAMBDA or i.key == "material" for i in a.interpretation)
            app07 = APP07 in {c.dataset for c in a.clarifications} | {v.dataset for v in a.values}
            if lam or app07:
                bad.append((e.name, t))
    assert bad == []
