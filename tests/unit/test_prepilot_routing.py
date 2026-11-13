"""Pre-pilot routing patch (2026-11-13): two query families only.

A  an energy-consumption limit names the Appendix 3 output (missing fields are still asked).
B  a U-value of an opaque element is about table 19-5-1, whose requirement is a minimum R: the
   engine still refuses the U slot; the answer explains and offers the printed R explicitly.
"""

from __future__ import annotations

import copy
import json
from functools import cache

import pytest
import yaml

from mabhas19.api.engine import Engine, prepilot_pins
from mabhas19.core.paths import CURATED_DIR
from mabhas19.generation.composer import compose
from mabhas19.generation.labels import CLARIFY_U_AS_R, NOTICE_U_AS_R
from mabhas19.generation.provenance import numbers
from mabhas19.indexing.chunking import build_chunks
from mabhas19.retrieval.candidate import CURRENT, MANIFESTS
from mabhas19.retrieval.hybrid import Retriever
from mabhas19.retrieval.models import RetrievalResult
from mabhas19.retrieval.router import ENERGY_LIMIT, U_WORDING
from mabhas19.retrieval.structured import StructuredStore


@cache
def _store() -> StructuredStore:
    return StructuredStore(CURATED_DIR)


@cache
def _retriever() -> Retriever:  # the router, planner and structured lookup (BM25 prose)
    path = CURATED_DIR / "text" / "m19-ed5-1404-main" / "blocks.jsonl"
    blocks = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    return Retriever(build_chunks(blocks, "m19-ed5-1404-main"), _store())


def _result(query: str) -> RetrievalResult:
    return copy.deepcopy(_retriever().retrieve(query))


def _engine() -> Engine:
    return Engine(_retriever(), _store(), "lexical")


APP03 = "m19-ed5-1404-app03.max_energy_use_intensity"
T1 = "m19-ed5-1404-main.t19-5-1"
T2 = "m19-ed5-1404-main.t19-5-2"


@pytest.mark.parametrize(
    "query",
    [
        "برای یک ساختمان مسکونی در اقلیم 3B، حد مجاز مصرف انرژی چقدر است؟",
        "حد مجاز مصرف انرژی ساختمان‌های مسکونی طبق مبحث ۱۹ چیه؟",
        "بیشینه مصرف انرژی مجاز هتل در اقلیم 0B",
    ],
)
def test_energy_limit_wording_reaches_appendix_3_and_asks_what_is_missing(query: str) -> None:
    r = _result(query)
    assert r.route.tables == [APP03]
    a = compose(r, _store(), "p1")
    assert a.response_type == "clarification"
    assert a.clarifications[0].dataset == APP03
    assert not a.values


def test_energy_limit_fully_stated_answers_the_printed_row() -> None:
    a = compose(_result("حد مجاز مصرف انرژی خوابگاه در اقلیم 4B رده A"), _store(), "p2")
    assert a.response_type == "structured_value"
    assert [v.record_id for v in a.values] == ["app03-A-r22-4B"]


@pytest.mark.parametrize(
    ("text", "hit"),
    [
        ("حد مجاز مصرف انرژی", True),
        ("سقف مصرف انرژی", True),
        ("مصرف مجاز انرژی", True),
        ("مصرف انرژی سالانه مجاز", True),
        ("مصرف انرژی", False),
        ("حداکثر ضریب انتقال حرارت", False),
    ],
)
def test_energy_limit_pattern(text: str, *, hit: bool) -> None:
    assert bool(ENERGY_LIMIT.search(text)) is hit


@pytest.mark.parametrize(
    ("text", "hit"),
    [
        ("ضریب انتقال حرارت", True),
        ("ضریب انتقال حرارتی", True),
        ("u-value دیوار", True),
        ("u value", True),
        ("حداکثر u کف", True),
        ("ضریب هدایت حرارتی", False),
        ("bus", False),
    ],
)
def test_u_wording_pattern(text: str, *, hit: bool) -> None:
    assert bool(U_WORDING.search(text)) is hit


def test_u_of_an_exterior_wall_asks_which_wall_of_table_19_5_1_with_the_r_note() -> None:
    answer, result, _ = _engine().ask("ضریب انتقال حرارت مجاز دیوار خارجی چقدر است؟")
    assert result.route.tables == [T1]
    assert result.route.signals["u_as_r"] is True
    assert answer.response_type == "clarification"
    c = answer.clarifications[0]
    assert c.field == "element"
    assert c.dataset == T1
    assert len(c.options) == 3
    assert "حداقل مقاومت حرارتی (R)" in c.note
    assert numbers(c.note) <= {"19", "5", "1"}  # «مبحث ۱۹», «جدول ۱۹-۵-۱» only


def test_u_of_one_element_is_refused_then_r_is_offered_and_answered_with_a_notice() -> None:
    q = "حداکثر U کف مجاور خاک در اقلیم 2A"
    answer, _, _ = _engine().ask(q)
    assert answer.response_type == "clarification"
    assert answer.refusals[0].outcome == "unsupported_output"  # the engine never answers U
    c = answer.clarifications[0]
    assert c.field == "output"
    assert [o.text for o in c.options] == ["حداقل مقاومت حرارتی"]
    assert c.note == CLARIFY_U_AS_R.format(table=c.table)
    follow, _, _ = _engine().ask(f"{q} ({c.options[0].text})")
    assert follow.response_type == "structured_value"
    assert [(v.dataset, v.output, v.value) for v in follow.values] == [(T1, "min_r_value", "۰/۲۷")]
    assert any(n.text == NOTICE_U_AS_R for n in follow.notices)
    assert not follow.safety["provenance_violations"]


def test_window_u_still_belongs_to_table_19_5_2() -> None:
    r = _result("ضریب انتقال حرارت مجاز پنجره در اقلیم 3B")
    assert r.route.tables == [T2]
    assert "u_as_r" not in r.route.signals


def test_r_wording_is_unchanged_and_carries_no_u_notice() -> None:
    a = compose(_result("حداقل مقاومت حرارتی سقف مجاور فضای آزاد در اقلیم 3B"), _store(), "p3")
    assert a.response_type == "structured_value"
    assert all(n.text != NOTICE_U_AS_R for n in a.notices)


def test_prepilot_pins_change_only_listed_modules_from_their_final8_hash() -> None:
    final8 = yaml.safe_load(MANIFESTS[CURRENT].read_text(encoding="utf-8"))["candidate"]
    pinned, bad = prepilot_pins(final8)
    assert not bad
    before = final8["code"]["modules_sha256"]
    after = pinned["code"]["modules_sha256"]
    changed = {k for k in before if before[k] != after[k]}
    assert changed == {"src/mabhas19/retrieval/router.py"}
    assert {k: v for k, v in pinned.items() if k != "code"} == {
        k: v for k, v in final8.items() if k != "code"
    }
    tampered = copy.deepcopy(final8)
    tampered["code"]["modules_sha256"]["src/mabhas19/retrieval/router.py"] = "0" * 64
    _, bad = prepilot_pins(tampered)
    assert bad
