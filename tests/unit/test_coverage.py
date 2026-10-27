"""The plan-coverage stage: pure functions and the Final-5 and Final-6 unsafe mechanisms
(development data).

Each Final-5 mechanism is pinned to the invariant that now stops it: M079 (I5 identity family),
M102 (I1 + I2 + I3 output slot), M195 (I4 unit intent), M011 (I1 + I2 + I7 table ownership),
M075 (I6 clause precedence) and M082 (I8 coordinated child plans). The Final-6 mechanisms (Safety
Closure 2): N201 and N207 (I3 complete requested quantity), N222 and N223 (I4 unit expressions),
N254 (I10 table scope), N069 (I6 correction ownership), N300 (I8 generic coordination), N328 and
N334 (I5 identity-span accounting).
"""

from __future__ import annotations

import json
from functools import cache
from typing import Any

import pytest

from mabhas19.core.paths import CURATED_DIR
from mabhas19.indexing.chunking import build_chunks
from mabhas19.retrieval import units
from mabhas19.retrieval.coverage import (
    Clause,
    Span,
    clauses_of,
    concept_conflict,
    owned_by,
    role_conflicts,
    sentences,
    split_coordinated,
    unit_intents,
)
from mabhas19.retrieval.hybrid import Retriever
from mabhas19.retrieval.models import RetrievalResult
from mabhas19.retrieval.plan import Constraint, EntityRef, StructuredQueryPlan
from mabhas19.retrieval.structured import StructuredStore

APP02 = "m19-ed5-1404-app02.city_climate_classes"
APP03 = "m19-ed5-1404-app03.max_energy_use_intensity"
APP07 = "m19-ed5-1404-app07.material_thermal_properties"


@cache
def _r() -> Retriever:
    path = CURATED_DIR / "text" / "m19-ed5-1404-main" / "blocks.jsonl"
    blocks = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
    return Retriever(build_chunks(blocks, "m19"), StructuredStore(CURATED_DIR))


def _violations(res: RetrievalResult) -> set[tuple[str, str]]:
    return {(v["table"], v["invariant"]) for v in res.coverage["violations"]}


# ---------------------------------------------------------------- units (I4, I9)


@pytest.mark.parametrize(
    ("requested", "printed", "ok"),
    [
        ("kWh/m2.yr", "kWh/m2.yr", True),
        ("kWh", "kWh/m2.yr", True),
        ("کیلووات ساعت بر متر مربع در سال", "kWh/m2.yr", True),
        ("kbtu/ft2·yr", "kWh/m2.yr", False),
        ("kw", "kWh/m2.yr", False),  # power is not energy: another quantity
        ("وات بر متر کلوین", "W/m.K", True),
        ("lb/ft3", "kg/m3", False),
    ],
)
def test_unit_compatibility_never_converts(requested: str, printed: str, ok: bool) -> None:
    a, b = units.parse(requested), units.parse(printed)
    assert a is not None
    assert b is not None
    assert units.compatible(a, b) is ok


def test_unit_intents_read_requests_and_attached_units_only() -> None:
    whole = [Clause(0, 0, 999, "", "structured", True)]
    vocab = frozenset(_r().planner.vocab_keys())
    got = unit_intents("سقف رده a را بر حسب kbtu/ft2·yr اعلام کنید", whole, vocab)
    assert [u.unit.describe() for u in got if u.unit] == ["kbtu/ft2·yr"]
    # «بر حسب کاربری» and «به واحد مسکونی» are no unit requests
    assert unit_intents("شدت مصرف بر حسب کاربری چگونه تعیین می شود", whole, vocab) == []
    assert unit_intents("به واحد مسکونی ما چه رده ای می رسد", whole, vocab) == []
    attached = unit_intents("چگالی 50 lb/ft3", whole, vocab)
    assert attached
    assert attached[0].attached
    assert attached[0].unit is not None
    assert attached[0].unit.foreign


# ---------------------------------------------------------------- clauses (I6)


def test_a_separate_question_without_the_plans_anchors_is_not_owned() -> None:
    full = "فرق رده های a و b چیه؟ برای مسجد در اقلیم 0b سقف مصرف چنده؟"
    report: list[dict[str, Any]] = [
        {"clause": "فرق رده های a و b چیه", "kind": None},
        {"clause": "برای مسجد در اقلیم 0b سقف مصرف چنده", "kind": "structured"},
    ]
    clauses = clauses_of(full, report)
    entity = full.index("مسجد")
    owned = owned_by(clauses, full, [(entity, entity + 4)])
    assert [c.owner for c in owned] == [False, True]
    assert [q for _, _, q in sentences(full)] == [True, True]


def test_a_statement_sentence_stays_shared_context() -> None:
    full = "ساختمان ما در اقلیم 3a است. سقف مصرف رده c برای هتل چنده؟"
    report: list[dict[str, Any]] = [
        {"clause": "ساختمان ما در اقلیم 3a است", "kind": None},
        {"clause": "سقف مصرف رده c برای هتل چنده", "kind": "structured"},
    ]
    hotel = full.index("هتل")
    owned = owned_by(clauses_of(full, report), full, [(hotel, hotel + 3)])
    assert all(c.owner for c in owned)


# ---------------------------------------------------------------- roles and concepts (I1, I2)


def test_role_conflict_is_output_against_constraint_only() -> None:
    a = Span(38, 44, "رده c", "constraint", APP03, "printed", "rating")
    b = Span(38, 41, "رده", "output", "t", "printed", "rating")
    c = Span(0, 4, "سوله", "entity", APP03, "exact")
    d = Span(0, 4, "سوله", "entity", "t", "exact")
    assert role_conflicts([a, b, c, d]) == [(a, b)]


def test_a_direct_lookup_may_not_ask_for_what_it_states() -> None:
    plan = StructuredQueryPlan("t", requested_field="rating")
    plan.constraints = [Constraint("rating", ["C"])]
    assert concept_conflict(plan) == "rating"
    plan.requested_field = "r_range"
    assert concept_conflict(plan) is None
    density = StructuredQueryPlan(APP07, requested_field="density")
    density.constraints = [Constraint("condition:چگالی اسمی", [])]
    assert concept_conflict(density) == "density"


# ---------------------------------------------------------------- coordination (I8)


def test_coordinated_places_become_one_child_plan_per_member() -> None:
    plan = StructuredQueryPlan(APP02)
    plan.entities = [EntityRef("place", "کهنوج, اقاجاری", "resolved", ["کهنوج", "اقاجاری"])]
    children = split_coordinated(plan, _r().store.schemas[APP02])
    assert [m for _, m in children] == ["کهنوج", "اقاجاری"]
    assert [c.entities[0].ids for c, _ in children] == [["کهنوج"], ["اقاجاری"]]
    other = StructuredQueryPlan(APP03)
    assert split_coordinated(other, _r().store.schemas[APP03]) == [(other, None)]


# ---------------------------------------------------------------- Final-5 mechanisms


def test_m079_head_noun_of_another_family_blocks_the_row() -> None:
    res = _r().retrieve(
        "تخته سیمانی با تراشه چوب که چگالیش کمتر از ۱٬۲۰۰ کیلوگرم بر متر مکعبه، "
        "ضریب هدایت حرارتیش چنده؟"
    )
    assert res.structured == []
    assert (APP07, "I5") in _violations(res)
    assert any(n.reason == "identity_family_conflict" for n in res.no_match)


def test_m102_unprinted_output_is_unsupported_not_density_or_lambda() -> None:
    res = _r().retrieve(
        "ظرفیت گرمایی ویژه بتن هوادار اتوکلاو شده با چگالی اسمی ۵۰۰ در پیوست ۷ چقدر است؟"
    )
    assert res.structured == []
    assert {(APP07, "I1"), (APP07, "I2"), (APP07, "I3")} <= _violations(res)
    assert [(n.outcome, n.reason) for n in res.no_match if n.dataset == APP07] == [
        ("unsupported_output", "unsupported_output_slot")
    ]


def test_m195_foreign_requested_unit_is_a_refused_conversion() -> None:
    res = _r().retrieve(
        "حداکثر شدت مصرف مجاز کتابخانه در اقلیم 0B برای رده A را بر حسب kBtu/ft²·yr اعلام کنید."
    )
    assert res.structured == []
    assert any(
        n.dataset == APP03
        and n.outcome == "unsupported_calculation"
        and n.reason == "unit_conversion"
        for n in res.no_match
    )


def test_m011_rating_never_admits_the_rating_table_beside_appendix_3() -> None:
    res = _r().retrieve("سوله صنعتی که تهویه داره تو اقلیم 4B، رده C حداکثر چقدر مصرف مجازه؟")
    assert [h.record_id for h in res.structured] == ["app03-C-r33-4B"]
    t = "m19-ed5-1404-main.t19-3-3"
    assert {(t, "I1"), (t, "I2"), (t, "I7")} <= _violations(res)
    assert all(n.dataset != t for n in res.no_match)  # not what was asked: no no-match either


def test_m075_ratings_of_the_explanatory_clause_are_ignored() -> None:
    res = _r().retrieve(
        "جدول‌های رده‌های C و B و A کجا اومده؟ برای یه خوابگاه تو اقلیم 5B، سقف رده C چنده؟"
    )
    assert [h.record_id for h in res.structured] == ["app03-C-r22-5B"]
    assert any(a["invariant"] == "I6" for a in res.coverage["adjustments"])


def test_m082_coordinated_places_are_certified_per_member() -> None:
    res = _r().retrieve("رده اقلیمی کهنوج و آقاجاری در پیوست ۲ چیست؟")
    assert sorted(h.record_id for h in res.structured) == ["app02-p002-r03", "app02-p004-r07"]
    assert res.certification_failures == []


def test_a_printed_unit_request_names_the_output() -> None:
    """«حداکثر چند kWh بر متر مربع در سال» names the EUI by its printed unit (no output slot)."""
    res = _r().retrieve(
        "خونه ویلایی تو اقلیم 3B برای رده B حداکثر چند kWh بر متر مربع در سال باید مصرف کنه؟"
    )
    assert [h.record_id for h in res.structured] == ["app03-B-r03-3B"]
    assert res.structured[0].output == "max_eui"
    assert res.structured[0].unit == "kWh/m2.yr"


# ---------------------------------------------------------------- Final-6 mechanisms


def _refusal(res: RetrievalResult, ds: str) -> list[tuple[str | None, str]]:
    return [(n.outcome, n.reason) for n in res.no_match if n.dataset == ds]


def test_n201_an_unprinted_quantity_after_the_entity_is_refused() -> None:
    res = _r().retrieve(
        "بتن سبک‌دانه با رس منبسط که عیار سیمانش بالای ۳۰۰ و ماسه‌اش فقط سبکه، مقاومت فشاری‌اش "
        "چند مگاپاسکاله؟"
    )
    assert res.structured == []
    assert {(APP07, "I3"), (APP07, "I4")} <= _violations(res)
    assert all(o == "unsupported_output" for o, _ in _refusal(res, APP07))


def test_n207_a_meaning_changing_noun_inside_the_output_phrase_is_refused() -> None:
    res = _r().retrieve(
        "در پیوست ۳، برای ساختمان اداری عمومی در اقلیم 2A و رده D، حداکثر مصرف آب سالانه به "
        "ازای هر متر مربع چقدر است؟"
    )
    assert res.structured == []
    assert _refusal(res, APP03) == [("unsupported_output", "unattested_quantity")]


@pytest.mark.parametrize(
    "query",
    [
        "برای مقایسه با قبض‌های ماهانه، سقف شدت مصرف اداری عمومی در اقلیم 2A و رده C را به kWh "
        "بر متر مربع در ماه لازم دارم.",
        "تعمیرگاه خودرو تو اقلیم 1B، رده C؛ سقف مصرفش رو به متر مکعب گاز در سال برای هر متر مربع "
        "بگید.",
    ],
)
def test_n222_n223_a_unit_of_another_basis_or_fuel_is_a_refused_conversion(query: str) -> None:
    res = _r().retrieve(query)
    assert res.structured == []
    assert _refusal(res, APP03) == [("unsupported_calculation", "unit_conversion")]


def test_n254_a_rating_outside_the_tables_fixed_scope_is_invalid() -> None:
    res = _r().retrieve(
        "برای کسب رده بازدهی B، حداقل مقاومت حرارتی کف مجاور هوای آزاد در اقلیم 2A چقدر باید باشد؟"
    )
    t = "m19-ed5-1404-main.t19-5-1"
    assert res.structured == []
    assert _refusal(res, t) == [("invalid_constraint", "out_of_scope_value")]


def test_n069_a_corrected_climate_never_leaks() -> None:
    res = _r().retrieve(
        "مهمانسرایی در اقلیم 4B نه، در 5B بررسی می‌کنم؛ رده B برایش چه سقف مصرفی دارد؟"
    )
    assert [h.record_id for h in res.structured] == ["app03-B-r21-5B"]
    assert [c["pattern"] for c in res.coverage["corrections"]] == ["A نه، B"]


def test_n300_coordinated_materials_each_keep_their_own_density() -> None:
    res = _r().retrieve(
        "ضریب هدایت حرارتی سیلیکون خالص و پلی‌استایرن منبسط با چگالی ۱۴ کیلوگرم بر متر مکعب رو "
        "می‌خوام."
    )
    assert sorted(h.record_id for h in res.structured) == ["app07-p006-r15", "app07-p011-r11"]
    assert res.certification_failures == []


def test_n328_one_subject_with_two_printed_identities_is_ambiguous() -> None:
    res = _r().retrieve(
        "یه ساختمون داریم که هم مهدکودکه هم پیش‌دبستانی، تو اقلیم 3B. برای رده D چقدر می‌تونه "
        "مصرف کنه؟"
    )
    assert res.structured == []
    assert _refusal(res, APP03) == [("ambiguous_entity", "one_subject_two_identities")]


def test_n334_another_familys_printed_qualifier_is_ambiguous() -> None:
    res = _r().retrieve(
        "پکیج گازی بدون دودکش که تو آشپزخونه نصب می‌کنن، حداقل چه بازدهی باید داشته باشه؟"
    )
    t = "m19-ed5-1404-main.t19-5-4"
    assert res.structured == []
    assert _refusal(res, t) == [("ambiguous_entity", "qualifier_family_conflict")]


def test_rows_carry_the_certified_output_and_its_printed_unit() -> None:
    res = _r().retrieve("ضریب هدایت حرارتی بتن هوادار اتوکلاو شده با چگالی اسمی ۵۰۰ چقدر است؟")
    assert res.structured
    assert all((h.output, h.unit) == ("thermal_conductivity", "W/m.K") for h in res.structured)
