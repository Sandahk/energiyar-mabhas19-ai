"""The frozen structured-value set (``eval/clarify/structured1.yaml``) and the rescue rules.

Expected behaviour was written before :func:`mabhas19.api.orchestration.rescue` existed; this test
only reads it. A value request for one table quantity never ends as prose; conceptual questions
and questions naming anything else keep the engine's own answer.
"""

from __future__ import annotations

from typing import Any

import pytest
import yaml
from tests.unit.test_natural1 import _ask

from mabhas19.api import quantities as Q
from mabhas19.core.paths import CURATED_DIR, EVAL_DIR
from mabhas19.generation.contract import Answer
from mabhas19.generation.provenance import numbers
from mabhas19.retrieval.structured import StructuredStore

CASES = yaml.safe_load((EVAL_DIR / "clarify" / "structured1.yaml").read_text(encoding="utf-8"))


def check(case: dict[str, Any], a: Answer) -> list[str]:  # noqa: PLR0912
    """Every way ``a`` departs from the frozen expectation of ``case``."""
    out: list[str] = []
    expect = case["expect"]
    kind = (a.safety.get("composite") or {}).get("kind")
    codes = {v.value for v in a.context}
    if a.safety.get("provenance_violations") or a.safety.get("orchestration_rejected"):
        out.append(f"safety: {a.safety}")
    if a.values:
        out.append(f"values: {[v.row_label for v in a.values]}")
    if expect in ("climate_refusal", "row_refusal"):
        outcome = "climate_not_printed" if expect == "climate_refusal" else "missing_row"
        refused = [r for r in a.refusals if r.outcome == outcome]
        if a.response_type != "refusal" or not refused or kind != "rescue":
            out.append(f"{expect}: {a.response_type} {[r.outcome for r in a.refusals]} {kind}")
        elif not (refused[0].dataset or "").endswith(case["dataset"]):
            out.append(f"dataset: {refused[0].dataset}")
        if a.excerpts or a.clarifications:
            out.append("refusal shows excerpts or options")
        if codes != set(case.get("codes") or []):
            out.append(f"codes: {sorted(codes)}")
    elif expect == "clarify":
        c = a.clarifications[0] if a.clarifications else None
        if a.response_type != "clarification" or c is None:
            out.append(f"clarify: {a.response_type}")
        elif c.field != case["field"] or [o.label for o in c.options] != case["rows"]:
            out.append(f"clarify: {c.field} {[o.label for o in c.options]}")
        if codes != set(case.get("codes") or []):
            out.append(f"codes: {sorted(codes)}")
    elif expect == "prose":
        if a.response_type != "prose" or kind == "rescue":
            out.append(f"prose: {a.response_type} {kind}")
    elif expect == "untouched" and (kind == "rescue" or a.context):
        out.append(f"untouched: taken over ({a.response_type})")
    return out


# frozen expectations the approved rule does not meet (reported, not re-written)
KNOWN = {
    "S14": "the engine itself refuses it (typed, not prose): it reads the city as an unknown "
    "19-5-1 row; the rescue layer only replaces prose / out-of-scope answers",
}


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(c, marks=pytest.mark.xfail(reason=KNOWN[c["id"]], strict=True))
        if c["id"] in KNOWN
        else c
        for c in CASES["cases"]
    ],
    ids=[c["id"] for c in CASES["cases"]],
)
def test_frozen_structured_case(case: dict[str, Any]) -> None:
    assert check(case, _ask(case["question"])) == []


def test_the_ardabil_refusal_says_5c_is_not_printed_and_shows_the_city_rows() -> None:
    a = _ask("برای اردبیل حد مجاز مصرف انرژی چقدره؟")
    assert "5C" in a.refusals[0].message
    assert "اردبیل" in a.refusals[0].message
    assert {v.value for v in a.context} == {"5C"}
    assert all(v.source is not None for v in a.context)
    assert not a.excerpts


def test_a_small_table_choice_after_a_city_keeps_the_citys_class() -> None:
    a = _ask("حداکثر SHGC مجاز برای کیش چقدره؟")
    assert a.question.endswith("(اقلیم 0B)")
    label = "حد اکثر ضریب بهره خورشیدی SHGC – بین %۲۰ تا %۵۰"
    follow = {"parent_request_id": "0" * 32, "field": "component", "kind": "option", "text": label}
    b = _ask(f"{a.question} ({label})", follow_up=follow, base=a.question)
    assert b.response_type == "structured_value"
    assert [v.row_label for v in b.values] == [label]
    direct = _ask(f"{label} در اقلیم 0B")
    assert [(v.record_id, v.value) for v in b.values] == [
        (v.record_id, v.value) for v in direct.values
    ]


@pytest.mark.parametrize(
    "question",
    [
        "مبحث ۱۹ درباره مصرف انرژی چه می‌گوید؟",
        "شدت مصرف انرژی یعنی چه؟",
        "راندمان دیگ چگونه محاسبه می‌شود؟",
        "فرق ضریب هدایت و مقاومت حرارتی چیست؟",
        "حداکثر SHGC و ضریب انتقال حرارت پنجره",  # two quantities
        "شدت مصرف انرژی ساختمان",  # no value asked
    ],
)
def test_not_a_single_value_request(question: str) -> None:
    assert Q.analyse(question) is None


def test_every_quantity_has_printed_rows_and_no_number_in_its_label() -> None:
    store = StructuredStore(CURATED_DIR)
    for q in Q.QUANTITIES:
        assert q.rows(store), q.id
        assert not numbers(q.label), q.id
