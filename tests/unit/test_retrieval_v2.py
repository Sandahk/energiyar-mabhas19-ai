"""Regression tests for retrieval v2: answerability, entity resolution, typed no-match outcomes
and mixed-query decomposition (curated data is read-only)."""

from __future__ import annotations

import json
from functools import cache

from mabhas19.core.paths import CURATED_DIR
from mabhas19.indexing.chunking import build_chunks
from mabhas19.retrieval.answerability import gate, has_words
from mabhas19.retrieval.final_test import QUOTAS, sample_anchors
from mabhas19.retrieval.hybrid import Retriever
from mabhas19.retrieval.models import RetrievalResult
from mabhas19.retrieval.structured import MAIN, StructuredStore, degree_days_in, r_value_in
from mabhas19.text.normalize import normalize_for_search as norm


@cache
def _retriever() -> Retriever:
    path = CURATED_DIR / "text" / "m19-ed5-1404-main" / "blocks.jsonl"
    blocks = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
    return Retriever(build_chunks(blocks, "m19"), StructuredStore(CURATED_DIR))


def _run(query: str) -> RetrievalResult:
    return _retriever().retrieve(query)


def _reasons(r: RetrievalResult) -> list[tuple[str, str]]:
    return [(n.dataset, n.reason) for n in r.no_match]


# ---------------------------------------------------------------- known table, no applicable value


def test_invalid_climate_class_6a_is_reported_not_ignored() -> None:
    r = _run("حداکثر ضریب انتقال حرارت پنجره ثابت در اقلیم 6A")
    assert r.structured == []  # never rows for all climates
    assert _reasons(r) == [(MAIN + "19-5-2", "climate_class_not_in_table")]
    table = r.no_match[0].table
    assert table is not None
    assert table.caption_blocks == ["main-p090-b005"]


def test_hdd_5000_has_no_printed_condition() -> None:
    assert degree_days_in(norm("روز درجه گرمایش ۵۰۰۰")) is not None
    assert degree_days_in(norm("شرط روز درجه سرمایش برای اقلیم 0B")) is None  # «0B» is a class
    r = _run("اقلیمی با روز درجه گرمایش ۵۰۰۰ در کدام رده اقلیمی قرار می‌گیرد")
    assert r.structured == []
    assert _reasons(r) == [(MAIN + "19-3-1", "no_printed_condition")]


def test_r_5_has_no_applicable_category() -> None:
    assert r_value_in(norm("رده C بر اساس نسبت R")) is None
    r = _run("ساختمانی که نسبت R آن ۵ است در کدام رده بازدهی قرار می‌گیرد")
    assert r.structured == []
    assert _reasons(r) == [(MAIN + "19-3-3", "no_applicable_category")]
    one = _run("ساختمانی با نسبت R برابر 1.5 در کدام رده است")
    assert [h.record_id for h in one.structured] == ["main-t19-3-3-r02"]


# ---------------------------------------------------------------- entities


def test_unknown_city_is_not_a_wildcard_lookup() -> None:
    r = _run("رده اقلیمی شهر لندن")
    assert r.route.route == "unsupported"
    assert r.structured == []
    assert _reasons(r) == [("m19-ed5-1404-app02.city_climate_classes", "unknown_place")]


def test_known_city_without_climate_intent_is_not_a_climate_lookup() -> None:
    r = _run("شهرداری تبریز")
    assert "m19-ed5-1404-app02.city_climate_classes" not in r.route.tables
    assert r.structured == []


def test_unknown_material_or_brand_returns_no_rows() -> None:
    store = _retriever().store
    assert store.resolve_material(norm("ضریب هدایت حرارتی نانوتک")) == []
    hits, _, no_match = store.materials(norm("ضریب هدایت حرارتی نانوتک"))
    assert hits == []
    assert [n.reason for n in no_match] == ["unknown_material"]
    assert _run("ضریب هدایت حرارتی عایق برند نانوتک").route.route == "unsupported"


def test_generic_insulation_word_is_not_a_material() -> None:
    assert _retriever().store.resolve_material(norm("ضریب هدایت حرارتی عایق")) is None
    r = _run("ضریب هدایت حرارتی عایق")
    assert r.structured == []
    assert r.route.route != "structured"


def test_eui_without_occupancy_is_not_a_lookup() -> None:
    concept = _run("شدت مصرف انرژی در تعیین رده بازدهی ساختمان‌ها چه نقشی دارد")
    assert concept.route.route == "prose"
    assert concept.structured == []
    unknown = _run("حداکثر شدت مصرف انرژی مجاز برای آسمان‌خراش ۸۰ طبقه")
    assert unknown.route.route == "unsupported"


# ---------------------------------------------------------------- answerability gate


def test_gate_categories() -> None:
    weather = gate(norm("آب و هوای امروز رشت چطور است"))
    zoning = gate(norm("حداکثر تعداد طبقات مجاز در طرح تفصیلی تهران"))
    hours = gate(norm("ساعت کاری دفتر مقررات ملی ساختمان"))
    assert weather is not None
    assert weather.category == "live_information"
    assert zoning is not None
    assert zoning.category == "outside_mabhas19"
    assert hours is not None
    assert hours.category == "service_information"
    assert gate(norm("الزامات هوای تازه در ساختمان")) is None  # domain «هوا» without a time word
    assert gate(norm("شدت مصرف انرژی مرکز خرید در اقلیم 2B")) is None


def test_gated_queries_route_unsupported() -> None:
    for q in (
        "آب و هوای امروز رشت چطور است",
        "حداکثر تعداد طبقات مجاز در طرح تفصیلی تهران",
        "ساعت کاری دفتر مقررات ملی ساختمان",
    ):
        assert _run(q).route.route == "unsupported"


def test_token_matching_is_bounded() -> None:
    assert has_words(norm("برنامه دوشنبه"), ["دوش"]) == []
    assert has_words(norm("دسته‌بندی اقلیمی"), ["اقلیم"]) == ["اقلیم"]
    assert has_words(norm("به چه معناست"), ["معنا"]) == ["معنا"]


# ---------------------------------------------------------------- decomposition


def test_mixed_query_decomposition_is_recorded() -> None:
    r = _run("شدت مصرف انرژی مجاز هتل در اقلیم 1B رده B و تعریف شدت مصرف انرژی ساختمان")
    assert r.route.route == "mixed"
    assert r.route.structured_query == norm("شدت مصرف انرژی مجاز هتل در اقلیم 1B رده B")
    assert r.route.prose_query == norm("تعریف شدت مصرف انرژی ساختمان")
    assert [h.record_id for h in r.structured] == ["app03-B-r20-1B"]
    t = _run("جدول ۱۹-۲-۱ گروه ب سال دوم پایش و تعریف سامانه پایش")
    assert t.route.structured_query == norm("جدول ۱۹-۲-۱ گروه ب سال دوم پایش")
    assert t.route.prose_query == norm("تعریف سامانه پایش")
    assert [h.record_id for h in t.structured] == ["main-t19-2-1-r08c22"]


def test_p64_symbols_remain_uninterpreted_with_table_evidence() -> None:
    r = _run("علامت تیک و ضربدر در جدول ۱۹-۲-۱ دقیقاً به چه معناست")
    assert [t.dataset for t in r.tables] == [MAIN + "19-2-1"]
    symbols = {h.values["printed_symbol"]["symbol"] for h in r.structured}
    assert symbols <= {"check_mark", "cross_mark"}
    assert all(not isinstance(v, bool) for h in r.structured for v in h.values.values())


def test_final_test_anchor_sampler_is_seeded_and_respects_exclusions() -> None:
    a = sample_anchors(CURATED_DIR, seed=7, exclude={"19-1-1", "app02-p002-r01"})
    b = sample_anchors(CURATED_DIR, seed=7, exclude={"19-1-1", "app02-p002-r01"})
    assert a == b
    assert {k: len(v) for k, v in a["anchors"].items()} == QUOTAS
    assert all(x["section"] != "19-1-1" for x in a["anchors"]["section"])
    assert all(x["record"] != "app02-p002-r01" for x in a["anchors"]["climate_city"])
