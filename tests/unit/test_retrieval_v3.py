"""Routing/structured retrieval v3: source-derived entity resolution, dimension validation,
provenance propagation, mixed intent and answerability (curated data is read-only).

Tests state generic properties against curated labels; they do not encode benchmark wording.
"""

from __future__ import annotations

import json
from decimal import Decimal
from functools import cache

import pytest

from mabhas19.core.paths import CURATED_DIR
from mabhas19.indexing.chunking import build_chunks
from mabhas19.retrieval.dimensions import (
    Interval,
    area_requested,
    density_requested,
    groups_requested,
    printed_interval,
    ratings_requested,
    years_requested,
)
from mabhas19.retrieval.entities import etokens, keys, split_number_prefix
from mabhas19.retrieval.hybrid import Retriever
from mabhas19.retrieval.models import ProseHit, RetrievalResult
from mabhas19.retrieval.structured import StructuredStore, _density_match
from mabhas19.text.normalize import normalize_for_search as norm


@cache
def _store() -> StructuredStore:
    return StructuredStore(CURATED_DIR)


@cache
def _retriever() -> Retriever:
    path = CURATED_DIR / "text" / "m19-ed5-1404-main" / "blocks.jsonl"
    blocks = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
    return Retriever(build_chunks(blocks, "m19"), _store())


def _run(query: str) -> RetrievalResult:
    return _retriever().retrieve(query)


def _material(query: str) -> tuple[str, list[str]]:
    res = _store().material_resolution(norm(query))
    return res.status, res.ids


# ---------------------------------------------------------------- stage 1: normalization


def test_entity_tokens_never_carry_punctuation() -> None:
    assert etokens("لار، چقدره؟ «گچ»") == ["لار", "چقدره", "گچ"]


def test_reversible_morphology_keys() -> None:
    assert "صفحه" in keys("صفحات")  # Arabic plural
    assert "سنگ" in keys("سنگها")
    assert "پرلیت" in keys("پرلیتی")  # adjectival
    assert "اقلیم" in keys("اقلیمش")  # possessive
    assert "4" in keys("چهارم")  # ordinal
    assert keys("روش") == {"روش"}  # stems shorter than 3 letters are never produced


def test_number_prefix_split_needs_a_label_word() -> None:
    assert split_number_prefix("چهارلوله", {"لوله"}) == ["چهار", "لوله"]
    assert split_number_prefix("دوش", {"ش"}) == ["دوش"]
    assert split_number_prefix("چهارلوله", set()) == ["چهارلوله"]


# ---------------------------------------------------------------- stage 1: materials


def test_material_resolution_is_label_anchored() -> None:
    # colloquial question words, attached punctuation and function words never block
    assert _material("ضریب هدایت حرارتی سنگ آهکی نیمه‌سخت چقدره؟") == (
        "resolved",
        ["app07-p010-r10"],
    )
    # the deepest printed level decides between same-family rows
    assert _material("ضریب هدایت حرارتی آلومینیوم") == ("resolved", ["app07-p013-r07"])


def test_material_morphology_and_spacing() -> None:
    status, ids = _material("ضریب هدایت حرارتی صفحه سیمانی الیافی")  # source: «صفحات»
    assert status == "resolved"
    assert set(ids) == {"app07-p011-r05", "app07-p011-r06"}
    assert _material("پلی‌ایزوبوتیلن")[1] == ["app07-p007-r04"]  # source: «پلی‌ایزو بوتیلن»


def test_unprinted_words_are_notes_not_blockers() -> None:
    res = _store().material_resolution(norm("بتن معمولی متراکم با سنگدانه سیلیسی"))
    assert res.status == "resolved"
    assert "معمولی" in res.leftover
    assert res.notes


def test_unknown_and_generic_materials() -> None:
    assert _material("ضریب هدایت حرارتی نانوتک")[0] == "unknown"
    assert _material("ضریب هدایت حرارتی عایق")[0] == "none"  # generic class word only


def test_no_alias_from_wording() -> None:
    # «XPS» is printed nowhere in curated/: it is reported as a leftover, never an alias
    res = _store().material_resolution(norm("عایق XPS با CFC و پوسته سطحی"))
    assert "xps" in res.leftover
    assert _material("ضریب هدایت حرارتی XPS")[0] != "resolved"


# ---------------------------------------------------------------- stage 1: places, occupancies


def test_places_match_joined_and_spaced_forms() -> None:
    store = _store()
    assert store.find_places(norm("رده اقلیمی قائم‌شهر")) == ["قائمشهر"]
    assert store.find_places(norm("اقلیم بندرعباس")) == ["بندرعباس"]
    assert store.find_places(norm("در شهر لار، کد اقلیم")) == ["لار"]


def test_occupancies_match_joined_forms_and_prefer_exact_words() -> None:
    store = _store()
    assert store.occupancies_in(norm("هایپرمارکت در اقلیم 1B")) == ["app03-occ-23"]
    assert store.occupancies_in(norm("مال در اقلیم 0B")) == ["app03-occ-27"]  # not «مالی»


# ---------------------------------------------------------------- stage 2: dimensions


@cache
def _final_cases() -> dict[str, dict[str, object]]:
    from mabhas19.core.paths import EVAL_DIR  # noqa: PLC0415
    from mabhas19.retrieval.eval import load_cases  # noqa: PLC0415

    return {c["id"]: c for c in load_cases(EVAL_DIR / "retrieval" / "final.yaml")}


def _evidence(case_id: str) -> tuple[str, dict[str, list[str]]]:
    case = _final_cases()[case_id]
    return str(case["query"]), case["evidence"]  # type: ignore[return-value]


@pytest.mark.parametrize("case_id", ["F043", "F061", "F065", "F078", "F093", "F113"])
def test_wrong_row_regressions_return_exactly_the_printed_row(case_id: str) -> None:
    query, ev = _evidence(case_id)
    got = {h.record_id for h in _run(query).structured}
    assert set(ev["records"]) <= got
    assert got <= set(ev["records"]) | set(ev.get("acceptable_records", []))


@pytest.mark.parametrize(
    ("case_id", "reason"),
    [
        ("F116", "rating_not_in_table"),
        ("F118", "no_matching_row"),
        ("F119", "no_matching_row"),
        ("F122", "no_matching_row"),
        ("F125", "climate_class_not_in_table"),
    ],
)
def test_no_match_regressions_are_typed(case_id: str, reason: str) -> None:
    query, ev = _evidence(case_id)
    r = _run(query)
    assert r.structured == []
    assert (ev["tables"][0], reason) in {(n.dataset, n.reason) for n in r.no_match}


def test_area_boundary_gap_is_not_inferred() -> None:
    stated = area_requested(norm("دقیقاً ۲۰۰۰ متر مربع"))
    smaller = printed_interval("کوچک‌تر", "2000")
    larger = printed_interval("بزرگتر", "2000")
    assert stated is not None
    assert smaller is not None
    assert larger is not None
    assert not stated.within(smaller)
    assert not stated.within(larger)
    above = area_requested(norm("۳۰۰۰ متر مربع و بیشتر"))
    assert above is not None
    assert above.within(Interval.from_op("≥", Decimal(3000)))


def test_constraint_parsing() -> None:
    assert ratings_requested(norm("برای رده E")) == ["E"]  # validated per table later
    assert groups_requested(norm("ساختمان‌های گروه ب")) == ["ب"]
    assert years_requested(norm("در سال چهارم و سال 3")) == [3, 4]


def test_appendix3_needs_a_climate_class_and_never_broadens() -> None:
    r = _run("حداکثر شدت مصرف انرژی هتل برای رده A")
    assert r.structured == []
    assert ("m19-ed5-1404-app03.max_energy_use_intensity", "missing_dimension") in {
        (n.dataset, n.reason) for n in r.no_match
    }


# ---------------------------------------------------------------- stage 3: provenance


def test_quoted_issue_links_are_scoped_to_the_printed_rows() -> None:
    links = _store().issue_links
    group14 = {rid for rid, issues in links.items() if "SI-0006" in issues}
    rows = {r["id"] for r in _store().app07 if r["category"]["text"].startswith("۱۴.")}
    assert group14 == rows  # every row printed under the misprinted title, nothing else


def test_appendix7_hit_inherits_the_group_issue_and_others_do_not() -> None:
    alu = _run("ضریب هدایت حرارتی آلومینیوم")
    assert alu.structured
    assert all("SI-0006" in h.source_issues for h in alu.structured)
    gypsum = _run("ضریب هدایت حرارتی گچ اندود با پرلیت")
    assert gypsum.structured
    assert all("SI-0006" not in h.source_issues for h in gypsum.structured)


def test_derived_climate_class_carries_its_source_issues() -> None:
    r = _run("حداکثر شدت مصرف انرژی مجاز مسکونی در اردبیل برای رده D")
    assert r.structured == []
    five_c = [n for n in r.no_match if n.reason == "climate_class_not_in_table"]
    assert five_c
    assert {"SI-0001", "SI-0002"} <= set(five_c[0].source_issues)


def test_missing_climate_class_issues_come_from_curated_metadata() -> None:
    store = _store()
    assert set(store.app03_missing_issues["5C"]) == {"SI-0001", "SI-0002"}


# ---------------------------------------------------------------- stage 4: mixed intent


@pytest.mark.parametrize(
    "case_id", ["F087", "F088", "F090", "F092", "F094", "F095", "F096", "F097", "F098"]
)
def test_second_question_without_trigger_words_is_prose(case_id: str) -> None:
    query, _ = _evidence(case_id)
    d = _retriever().router.route(query)
    assert d.route == "mixed"
    assert d.prose_query
    assert d.structured_query


def test_lookup_constraints_stay_together() -> None:
    query, ev = _evidence("F074")  # the column constraint sits after a comma
    r = _run(query)
    assert r.route.route == "structured"
    assert [h.record_id for h in r.structured] == ev["records"]


def test_topic_word_without_dimensions_is_prose() -> None:
    router = _retriever().router
    assert router.route("ضریب انتقال حرارت (U-Value) یعنی چی؟").route == "prose"
    assert router.route("ضریب عملکرد COP یعنی چه").route == "prose"


def test_explicit_section_reference_wins_over_topic_words() -> None:
    d = _retriever().router.route("طبق بند ۱۹-۵-۲-۲-۴ سنسور دما روی کدام لوله چیلر نصب شود")
    assert d.route == "prose"


def test_caption_words_are_not_explanation_requests() -> None:
    query, _ = _evidence("F084")  # «الزامات» is part of the quoted caption
    assert _retriever().router.route(query).route == "structured"


# ---------------------------------------------------------------- stage 5: answerability


def test_low_corpus_coverage_never_decides_unsupported() -> None:
    d = _retriever().router.route("اون تابلوی فلزی رده انرژی دم در رو کی صادر می‌کنه؟")
    assert d.route != "unsupported"
    assert "coverage" in d.signals  # still recorded as diagnostic metadata


def test_references_are_validated_against_the_curated_index() -> None:
    router = _retriever().router
    missing = router.route("بند ۱۹-۹-۳ مبحث نوزدهم چه الزامی دارد")
    assert missing.route == "unsupported"
    assert missing.signals["unknown_references"] == ["19-9-3"]
    assert router.route("بند ۱۹-۲-۱-۹ درباره پیوست‌ها").route == "prose"
    assert router.route("محتوای جدول ۱۹-۵-۹").route == "unsupported"


def test_appendix_reference_names_the_appendix_table() -> None:
    d = _retriever().router.route("رده اقلیمی شهر طبق پیوست ۲")
    assert "پ2" in d.signals["table_ref"]


def test_unknown_occupancy_with_a_stated_class_is_a_typed_lookup() -> None:
    r = _run("سقف شدت مصرف انرژی برای ترمینال فرودگاه در اقلیم 2B رده D")
    assert r.route.route == "structured"
    assert r.structured == []
    assert "unknown_occupancy" in {n.reason for n in r.no_match}


def test_5c_rule_exposes_its_issue_without_an_occupancy() -> None:
    query, _ = _evidence("F131")
    r = _run(query)
    assert r.structured == []
    exposed = {i for n in r.no_match for i in n.source_issues}
    assert "SI-0002" in exposed


# ---------------------------------------------------------------- density intervals


def test_density_ranges_parse_as_intervals() -> None:
    expected = Interval(Decimal(2300), False, Decimal(2600), False)
    for text in (
        "چگالی ۲۳۰۰ تا ۲۶۰۰",
        "چگالی 2300-2600 کیلوگرم بر متر مکعب",
        "جرم مخصوص ۲۳۰۰–۲۶۰۰ kg/m3",
        "چگالی ۲۳۰۰ کیلوگرم بر متر مکعب تا ۲۶۰۰ کیلوگرم بر متر مکعب",
    ):
        assert density_requested(norm(text)) == expected, text
    point = density_requested(norm("چگالی ۵۰۰"))
    assert point == Interval(Decimal(500), False, Decimal(500), False)


def _rows(*ranges: tuple[str, str]) -> list[dict[str, object]]:
    return [{"id": f"r{i}", "density": {"min": lo, "max": hi}} for i, (lo, hi) in enumerate(ranges)]


def test_density_range_matching_never_collapses_to_a_bound_or_midpoint() -> None:
    rows = _rows(("1800", "2000"), ("2000", "2300"), ("2300", "2600"))

    def ids(lo: int, hi: int) -> list[str]:
        got, _ = _density_match(rows, Interval(Decimal(lo), False, Decimal(hi), False))
        return [str(r["id"]) for r in got]

    assert ids(2000, 2300) == ["r1"]  # the printed range itself
    assert ids(2100, 2200) == ["r1"]  # inside one printed range
    assert ids(1900, 2100) == ["r0", "r1"]  # spans two: both returned, with a note
    assert ids(100, 200) == []  # outside every printed range: no-match, nothing inferred
    got, notes = _density_match(rows, Interval(Decimal(2000), False, Decimal(2000), False))
    assert [r["id"] for r in got] == ["r0", "r1"]  # a point on a shared bound
    assert notes


def test_material_lookup_uses_the_stated_density_range() -> None:
    r = _run(
        "ضریب هدایت حرارتی بتن متراکم با سنگدانه سیلیسی با جرم مخصوص ۲۳۰۰-۲۶۰۰ کیلوگرم بر متر مکعب"
    )
    assert [h.record_id for h in r.structured] == ["app07-p003-r09"]


# ---------------------------------------------------------------- explicit table scope


def test_explicit_table_reference_scopes_the_lookup() -> None:
    r = _run("شهر کرمان در جدول ۱۹-۳-۲ چه کدی دارد")
    assert r.structured
    assert {h.dataset for h in r.structured} == {"m19-ed5-1404-main.t19-3-2"}


def test_several_explicit_references_are_all_queried() -> None:
    r = _run("رده اقلیمی کرمان طبق جدول ۱۹-۳-۲ و پیوست ۲")
    assert {h.dataset for h in r.structured} == {
        "m19-ed5-1404-main.t19-3-2",
        "m19-ed5-1404-app02.city_climate_classes",
    }


def test_derivation_for_a_named_table_is_input_not_answer() -> None:
    r = _run("طبق پیوست ۳ حداکثر شدت مصرف انرژی هتل در شیراز برای رده A")
    assert r.structured
    assert {h.dataset for h in r.structured} == {"m19-ed5-1404-app03.max_energy_use_intensity"}
    assert any("derived from the named place" in n for n in r.notes)


def test_interpolation_requests_are_refused_by_the_lookup() -> None:
    r = _run("ضریب هدایت حرارتی پشم سنگ با چگالی ۵ چقدر است؟ اگر نیست درون‌یابی کن")
    assert r.route.route == "structured"
    assert r.structured == []
    assert any("never performed" in n for n in r.notes)


# ---------------------------------------------------------------- block issues in prose evidence


def _chunk_with(block_id: str) -> str:
    return next(c.chunk_id for c in _retriever().lexical.chunks if block_id in c.block_ids)


def _hit_for(chunk_id: str) -> ProseHit:
    c = next(c for c in _retriever().lexical.chunks if c.chunk_id == chunk_id)
    hit = ProseHit(
        c.chunk_id, 1.0, 1, c.block_ids, c.section_path, c.pdf_pages, c.printed_pages, [], c.text
    )
    _retriever()._attach_block_issues([hit])
    return hit


def test_chunk_with_a_linked_block_exposes_its_issue() -> None:
    hit = _hit_for(_chunk_with("main-p118-b008"))
    assert hit.source_issues == ["SI-0004"]
    assert hit.issue_origins == {"SI-0004": ["main-p118-b008"]}


def test_unrelated_chunk_inherits_no_issue() -> None:
    hit = _hit_for(_chunk_with("main-p083-b011"))  # Appendix 7 reference, not SI-0004
    assert hit.source_issues == []
    assert hit.issue_origins == {}


def test_same_issue_from_several_blocks_is_one_id_with_traceable_origins() -> None:
    chunk = _chunk_with("main-p146-b009")
    hit = _hit_for(chunk)
    assert hit.source_issues == ["SI-0004"]
    linked = [b for b in hit.block_ids if b in _store().block_issues]
    assert len(linked) >= 2
    assert hit.issue_origins["SI-0004"] == linked


def test_issue_propagation_does_not_change_rankings() -> None:
    r = _retriever()
    query = "حداکثر توان مجاز سامانه روشنایی مصنوعی در پیوست ۸"
    result = r.retrieve(query)
    plain = r.rank_prose(result.route.prose_query or query, 10)
    assert [(h.chunk_id, h.score, h.rank) for h in result.prose] == [
        (h.chunk_id, h.score, h.rank) for h in plain
    ]
    assert any("SI-0004" in h.source_issues for h in result.prose)
