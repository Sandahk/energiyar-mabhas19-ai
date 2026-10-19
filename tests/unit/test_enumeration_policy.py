"""Multi-row structured output only as an explicit, typed enumeration of printed alternatives.

An enumeration is allowed only when the schema marks the dimension enumerable, the question
explicitly asks for the alternatives, no stated constraint selects one member, every row lies in
the resolved source-defined scope and carries its own value of the dimension, and the result
metadata says it is an enumeration. A merely absent value-determining dimension fails closed.
"""

from __future__ import annotations

import json
from functools import cache
from typing import Any

import pytest

from mabhas19.core.paths import CURATED_DIR, EVAL_DIR
from mabhas19.indexing.chunking import build_chunks
from mabhas19.retrieval.eval import load_cases
from mabhas19.retrieval.executor import enumeration_value
from mabhas19.retrieval.hybrid import Retriever
from mabhas19.retrieval.models import RetrievalResult
from mabhas19.retrieval.planner import enumeration_request
from mabhas19.retrieval.structured import StructuredStore
from mabhas19.text.normalize import normalize_for_search as norm

APP03 = "m19-ed5-1404-app03.max_energy_use_intensity"
APP07 = "m19-ed5-1404-app07.material_thermal_properties"
MAIN = "m19-ed5-1404-main.t"


@cache
def _r() -> Retriever:
    path = CURATED_DIR / "text" / "m19-ed5-1404-main" / "blocks.jsonl"
    blocks = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
    return Retriever(build_chunks(blocks, "m19"), StructuredStore(CURATED_DIR))


def _lookup(ds: str, query: str) -> tuple[list[Any], list[Any], list[str], Any]:
    return _r().lookup(ds, query)


def _record(ds: str, rid: str) -> dict[str, Any]:
    s = _r().store
    rows = s.app03 if ds == APP03 else s.app07 if ds == APP07 else s.main[ds].records
    return next(r for r in rows if (r.get("id") or r.get("record_id")) == rid)


# (dataset, question without the dimension, the same question asking for the alternatives,
#  enumerated dimension, expected number of alternatives)
CASES = [
    (
        MAIN + "19-5-1",
        "حداقل مقاومت حرارتی دیوار خارجی مجاور خاک",
        "حداقل مقاومت حرارتی دیوار خارجی مجاور خاک در همه اقلیم‌ها",
        "climate_class",
        6,
    ),
    (
        APP03,
        "حداکثر شدت مصرف انرژی اداری عمومی در اقلیم 3B",
        "حداکثر شدت مصرف انرژی اداری عمومی در اقلیم 3B برای همه رده‌ها",
        "rating",
        4,
    ),
    (
        MAIN + "19-2-1",
        "بازرسی طراحی برای گروه الف چه علامتی دارد",
        "بازرسی طراحی برای گروه الف در همه سال‌ها چه علامتی دارد",
        "year",
        5,
    ),
    (
        MAIN + "19-5-2",
        "حداکثر SHGC پنجره در اقلیم 3B",
        "حداکثر SHGC پنجره در اقلیم 3B برای PF های مختلف",
        "row_condition",
        4,
    ),
    (
        APP07,
        "ضریب هدایت حرارتی چوب بلوط",
        "ضریب هدایت حرارتی چوب بلوط در همه چگالی‌ها",
        "density",
        2,
    ),
    (
        APP03,
        "حداکثر شدت مصرف انرژی ساختمان اداری در اقلیم 3B برای رده C",
        "حداکثر شدت مصرف انرژی ساختمان اداری در اقلیم 3B برای رده C به تفکیک کاربری",
        "occupancy",
        2,
    ),
]


@pytest.mark.parametrize("case", CASES, ids=lambda c: f"{c[0].split('.')[-1]}:{c[3]}")
def test_absent_dimension_fails_closed(case: tuple[str, str, str, str, int]) -> None:
    ds, absent, _, _, _ = case
    hits, no_match, _, plan = _lookup(ds, absent)
    assert hits == []
    assert plan.enumerations == []
    assert {n.outcome for n in no_match} <= {"missing_required_dimension", "ambiguous_entity"}
    assert no_match


@pytest.mark.parametrize("case", CASES, ids=lambda c: f"{c[0].split('.')[-1]}:{c[3]}")
def test_requested_alternatives_are_a_typed_enumeration(
    case: tuple[str, str, str, str, int],
) -> None:
    ds, _, asked, dim, n = case
    hits, no_match, notes, plan = _lookup(ds, asked)
    assert no_match == []
    assert [e.dimension for e in plan.enumerations] == [dim]
    e = plan.enumerations[0]
    assert e.basis == "explicit_request"
    assert e.request
    assert e.members
    assert len(hits) == n
    values = [enumeration_value(_r().store, ds, _record(ds, h.record_id), dim) for h in hits]
    assert all(values)  # every result carries its distinguishing label
    assert len(set(values)) == n  # one row per printed alternative
    assert set(values) <= set(e.members)  # all within the resolved scope
    assert any(x.startswith("enumeration of printed alternatives") for x in notes)


def test_a_stated_member_still_selects_within_an_enumeration() -> None:
    hits, _, _, plan = _lookup(
        MAIN + "19-5-1", "حداقل مقاومت حرارتی دیوار خارجی مجاور خاک در اقلیم 3A در همه حالت‌ها"
    )
    assert "climate_class" not in plan.enumerated
    assert [h.record_id for h in hits] == ["main-t19-5-1-r06c03"]


def test_enumeration_metadata_reaches_the_result() -> None:
    r = _r().retrieve("حداقل مقاومت حرارتی دیوار خارجی مجاور خاک در همه اقلیم‌ها چقدر است؟")
    assert len(r.structured) == 6
    assert [(e["table"], e["dimension"], e["basis"]) for e in r.enumerations] == [
        (MAIN + "19-5-1", "climate_class", "explicit_request")
    ]
    assert any(c["state"] == "enumerated" for c in r.constraints)


@pytest.mark.parametrize(
    ("text", "requested"),
    [
        ("در همه اقلیم‌ها", True),
        ("برای اقلیم‌های مختلف", True),
        ("کدام اقلیم‌ها", True),
        ("به تفکیک", True),
        ("در اقلیم 3A", False),
        ("حداقل R دیوار", False),
        ("چقدر است", False),
    ],
)
def test_enumeration_requests_are_explicit(text: str, requested: bool) -> None:
    assert (enumeration_request(norm(text), ("اقلیم",)) is not None) is requested


@cache
def _set_results(name: str) -> list[RetrievalResult]:
    return [_r().retrieve(c["query"]) for c in load_cases(EVAL_DIR / "retrieval" / f"{name}.yaml")]


@pytest.mark.parametrize("name", ["dev", "heldout", "final", "final2", "final3"])
def test_every_development_enumeration_satisfies_the_policy(name: str) -> None:
    for r in _set_results(name):
        for e in r.enumerations:
            assert e["basis"] == "explicit_request", r.query
            hits = [h for h in r.structured if h.dataset == e["table"]]
            assert hits, r.query
            for h in hits:
                value = enumeration_value(
                    _r().store, e["table"], _record(e["table"], h.record_id), e["dimension"]
                )
                assert value, (r.query, h.record_id)
                assert value in e["members"], (r.query, h.record_id, value)
