"""Structured safety: no wrong regulatory value, every constraint accounted for.

Tests state generic properties against curated labels with their own wording; they do not
encode benchmark questions. The last tests run every development set (dev, heldout, final,
Final-2) and hold the safety headline at zero wrong regulatory values.
"""

from __future__ import annotations

import json
from functools import cache
from typing import Any

import pytest

from mabhas19.core.paths import CURATED_DIR, EVAL_DIR
from mabhas19.indexing.chunking import build_chunks
from mabhas19.retrieval.constraints import STATES, violates
from mabhas19.retrieval.dimensions import decimal_text, ratings_requested, years_requested
from mabhas19.retrieval.entities import merge_modifiers, negated_word
from mabhas19.retrieval.eval import (
    dropped_constraints,
    evaluate,
    load_cases,
    metrics,
    wrong_extra,
    wrong_values,
)
from mabhas19.retrieval.hybrid import Retriever
from mabhas19.retrieval.models import RetrievalResult, StructuredHit
from mabhas19.retrieval.structured import StructuredStore, r_value_in
from mabhas19.text.normalize import normalize_for_search as norm

SETS = ("dev", "heldout", "final", "final2", "final3")


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


def _ids(query: str) -> list[str]:
    return [h.record_id for h in _run(query).structured]


def _reasons(query: str) -> list[str]:
    return [n.reason for n in _run(query).no_match]


def _states(r: RetrievalResult) -> dict[str, str]:
    return {c["constraint"]: c["state"] for c in r.constraints}


# ------------------------------------------------- stage 1: polarity and distinguishing words


def test_negators_and_count_prefixes_are_meaning_bearing() -> None:
    assert merge_modifiers(["بدون", "الیاف"], None) == ["بدونالیاف"]
    assert merge_modifiers(["غیر", "چگالشی"], None) == ["غیرچگالشی"]
    assert negated_word("بدونالیاف") == "الیاف"
    assert negated_word("غیرچگالشی") == "چگالشی"
    assert negated_word("باالیاف") is None
    # a count prefix merges only into a printed label word
    assert merge_modifiers(["تک", "سرعته"], {"تکسرعته"}) == ["تکسرعته"]
    assert merge_modifiers(["تک", "نفره"], {"تکسرعته"}) == ["تک", "نفره"]


@pytest.mark.parametrize(
    ("positive", "negative"),
    [
        (
            "حداقل بازدهی پکیج چگالشی کمتر از ۷۰ کیلووات چقدر است؟",
            "حداقل بازدهی پکیج گازی غیر چگالشی کمتر از ۷۰ کیلووات چقدر است؟",
        ),
        (
            "کلاس بازدهی الکتروموتور تکفاز تکسرعته با توان بیش از ۱۲۰ وات چیست؟",
            "کلاس بازدهی الکتروموتور تکفاز چندسرعته با توان بیش از ۱۲۰ وات چیست؟",
        ),
    ],
)
def test_polar_counterparts_return_disjoint_single_rows(positive: str, negative: str) -> None:
    pos, neg = _ids(positive), _ids(negative)
    assert len(pos) == 1
    assert len(neg) == 1
    assert pos != neg


def test_with_and_without_qualifier_resolve_to_different_materials() -> None:
    with_ = _store().material_resolution(norm("پلی پروپیلن با الیاف شیشه"))
    without = _store().material_resolution(norm("پلی پروپیلن بدون الیاف شیشه"))
    assert with_.status == without.status == "resolved"
    assert len(with_.ids) == len(without.ids) == 1
    assert with_.ids != without.ids


def test_floor_and_roof_are_hard_constraints() -> None:
    tail = "مجاور فضای نیمه باز کنترل نشده در اقلیم 4A چقدر است؟"
    roof = _ids("حداقل مقاومت حرارتی سقف " + tail)
    floor = _ids("حداقل مقاومت حرارتی کف " + tail)
    assert len(roof) == len(floor) == 1
    assert roof != floor


# ------------------------------------------------- stage 1: ordinals, decimals, ratings


def test_ordinal_year_is_a_year_not_a_count() -> None:
    assert years_requested(norm("در سال چهارم")) == [4]
    r = _run("وضعیت انرژی تجدیدپذیر برای گروه د در سال چهارم طبق جدول ۱۹-۲-۱ چیست؟")
    assert len(r.structured) == 1
    assert r.structured[0].values["year"]["index"] == 4
    assert _states(r) == {"group": "applied", "year": "applied"}


@pytest.mark.parametrize("value", ["4.5", "۴٫۵", "٤٫٥", "4/5"])
def test_decimals_are_never_truncated(value: str) -> None:
    assert "4.5" in decimal_text(norm(f"نسبت {value}"))
    r = _run(f"اگر نسبت R برابر {value} باشد طبق جدول ۱۹-۳-۳ چه رده ای دارد؟")
    assert r.structured == []
    assert [(n.reason, n.outcome) for n in r.no_match] == [
        ("no_applicable_category", "no_printed_category")
    ]
    assert _states(r)["r_value"] == "applied"  # interpreted exactly; no printed range holds it


def test_product_marking_is_not_an_r_value() -> None:
    assert r_value_in(norm("مقدار R100 روی برچسب")) is None
    assert r_value_in(norm("مقدار R برابر 2.5")) is not None


@pytest.mark.parametrize("rating", ["A+", "A++", "A+++"])
def test_unprinted_rating_is_a_typed_no_match(rating: str) -> None:
    assert ratings_requested(norm(f"رده {rating}")) == [rating]
    r = _run(f"سقف شدت مصرف انرژی رده {rating} برای ساختمان اداری عمومی در اقلیم 2B چقدر است؟")
    assert r.structured == []  # never dropped and broadened to every rating
    assert [n.reason for n in r.no_match] == ["rating_not_in_table"]
    assert _states(r)["rating"] == "invalid"


def test_rating_referred_to_but_not_stated_returns_no_rows() -> None:
    r = _run("سقف شدت مصرف انرژی آن رده برای ساختمان اداری عمومی در اقلیم 2B چقدر است؟")
    assert r.structured == []
    assert "unresolved_constraint" in [n.outcome for n in r.no_match]


# ------------------------------------------------- stage 1: computed values and row naming


@pytest.mark.parametrize("word", ["میانگین", "درون یابی"])
def test_computed_value_is_unsupported_calculation(word: str) -> None:
    r = _run(f"{word} R دیوار خارجی مجاور فضای باز در اقلیم 2A و 2B طبق جدول ۱۹-۵-۱ چقدر است؟")
    assert r.structured == []
    assert {n.reason for n in r.no_match} == {"unsupported_calculation"}
    assert set(_states(r).values()) == {"unsupported_calculation"}


def test_unnamed_row_never_lists_a_column() -> None:
    r = _run("حداقل مقاومت حرارتی در اقلیم 3A چقدر است؟")
    assert r.structured == []
    assert "missing_required_dimension" in [n.outcome for n in r.no_match]


def test_material_rows_without_stated_density_fail_closed() -> None:
    # oak is printed for two densities with different λ: one answer is implied, none is assumed
    r = _run("ضریب هدایت حرارتی چوب بلوط در پیوست ۷ چقدر است؟")
    assert r.structured == []
    assert "missing_required_dimension" in [n.outcome for n in r.no_match]


# ------------------------------------------------- stage 2: unverified entities


def test_unprinted_product_name_stays_unresolved() -> None:
    res = _store().material_resolution(norm("بلوک هبلکس"))
    assert res.status == "unknown"
    assert res.ids == []


# ------------------------------------------------- stage 3: references and decomposition


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("فصل نهم مبحث نوزدهم درباره چیست؟", [("chapter", "nonexistent")]),
        ("فصل ۹ مبحث نوزدهم درباره چیست؟", [("chapter", "nonexistent")]),
        ("فصل سوم مبحث نوزدهم درباره چیست؟", [("chapter", "valid")]),
        ("در فصل سرد دمای داخل چند درجه باشد؟", []),
    ],
)
def test_chapter_references_are_validated(query: str, expected: list[tuple[str, str]]) -> None:
    d = _retriever().router.route(query)
    assert [(r["kind"], r["status"]) for r in d.signals["references"]] == expected
    assert (d.route == "unsupported") == (expected == [("chapter", "nonexistent")])


def test_coordinated_climate_classes_stay_one_constraint() -> None:
    r = _run("جدول ۱۹-۳-۱ برای اقلیم های 3A و 3B چه روز-درجه سرمایشی دارد؟")
    assert r.route.route == "structured"
    assert r.structured
    assert _states(r) == {"climate_class": "applied"}


def test_unprinted_material_with_explicit_appendix_is_a_typed_no_match() -> None:
    r = _run("ضریب هدایت حرارتی چوب بامبو در پیوست ۷ چقدر است؟")
    assert r.route.route == "structured"
    assert r.structured == []
    assert r.no_match


def test_topic_noun_alone_with_another_quantity_is_not_a_lookup() -> None:
    # the table prints efficiencies; a temperature question about the same equipment is prose
    d = _retriever().router.route("اختلاف دمای آب رفت و برگشت چیلر حداقل چند درجه است؟")
    assert d.route == "prose"


# ------------------------------------------------- stage 4: two-step answerability


@pytest.mark.parametrize(
    "query",
    [
        "طرز تهیه خورش قیمه چطوریه؟",
        "برای سرماخوردگی بچه چه دارویی خوبه؟",
        "لاستیک ماشین رو هر چند کیلومتر عوض کنیم؟",
        "مستاجر ودیعه رو پس نمیده، شکایت کنم؟",
        "موش تو انباری داریم، چه سمی بزنیم؟",
        "مودم رو کجای خونه بذاریم که آنتن بده؟",
    ],
)
def test_everyday_categories_are_external(query: str) -> None:
    d = _retriever().router.route(query)
    assert d.route == "unsupported"
    assert d.signals["scope"]["class"] == "external_domain"


@pytest.mark.parametrize(
    "query",
    [
        "چند درصد جای پارک پارکینگ مسکونی باید شارژر خودرو برقی داشته باشد؟",
        "نشت هوا از درز پنجره ها چگونه باید کنترل شود؟",
        "مجری در صورت بروز خسارت ناشی از عملکرد خود چه تعهدی دارد؟",
        "صدای تجهیزات تهویه مطبوع در مبحث ۱۹ محدود شده است؟",
    ],
)
def test_technical_use_of_everyday_words_stays_in_scope(query: str) -> None:
    assert _retriever().router.route(query).route != "unsupported"


# ------------------------------------------------- stage 5: appendix-level source issues


def test_explicit_uncurated_appendix_reference_carries_its_source_issues() -> None:
    r = _run("عنوان پیوست ۶ مبحث نوزدهم چیست؟")
    issues = {n.split()[0] for n in r.notes if n.startswith("SI-")}
    assert {"SI-0004", "SI-0007"} <= issues


def test_curated_appendix_issues_stay_record_level() -> None:
    r = _run("پیوست ۳ مبحث نوزدهم درباره چیست؟")
    assert not [n for n in r.notes if "(appendix 3)" in n]


# ------------------------------------------------- evaluator and global invariants


def _hit(record_id: str, dataset: str = "ds") -> StructuredHit:
    return StructuredHit(dataset, "doc", record_id, None, None, [], {}, {}, [], "test")


def test_wrong_values_counts_every_unrequested_record() -> None:
    r = RetrievalResult(query="q", normalized_query="q", route=None)  # type: ignore[arg-type]
    r.structured = [_hit("a"), _hit("b"), _hit("c")]
    assert wrong_values(r, {"records": ["a"], "acceptable_records": ["b"]}) == ["c"]
    assert wrong_values(r, {"no_records": True}) == ["a", "b", "c"]
    assert wrong_values(r, {"blocks": ["x"]}) == ["a", "b", "c"]
    r.structured = [_hit("a", "t1"), _hit("b", "t2")]
    assert wrong_values(r, {"tables": ["t1"]}) == ["b"]


def test_wrong_extra_rows_are_only_those_beside_a_complete_answer() -> None:
    r = RetrievalResult(query="q", normalized_query="q", route=None)  # type: ignore[arg-type]
    r.structured = [_hit("a"), _hit("c")]
    ev = {"records": ["a"]}
    assert wrong_extra(r, ev, wrong_values(r, ev)) == ["c"]  # beside the answer: extra row
    ev = {"records": ["b"]}
    assert wrong_extra(r, ev, wrong_values(r, ev)) == []  # stands in for it: wrong value


def test_constraint_lost_in_decomposition_is_a_dropped_constraint() -> None:
    r = _run("حداقل مقاومت حرارتی سقف مجاور فضای نیمه باز کنترل نشده در اقلیم 4A چقدر است؟")
    assert dropped_constraints(r) == []
    r.constraints = []  # as if the climate class never reached the lookup
    assert dropped_constraints(r) == ["climate_class: not in the lookup"]


@cache
def _set_results(name: str) -> tuple[list[Any], dict[str, Any]]:
    return evaluate(_retriever(), load_cases(EVAL_DIR / "retrieval" / f"{name}.yaml"))


@pytest.mark.parametrize("name", SETS)
def test_no_wrong_regulatory_value_on_any_development_set(name: str) -> None:
    results, _ = _set_results(name)
    m = metrics(results)
    assert m["wrong_value_records"] == 0, m["wrong_value_cases"]
    assert m["dropped_constraints"] == 0, m["dropped_constraint_cases"]


@pytest.mark.parametrize("name", SETS)
def test_every_constraint_ends_in_exactly_one_state(name: str) -> None:
    for c in load_cases(EVAL_DIR / "retrieval" / f"{name}.yaml"):
        r = _run(c["query"])
        seen = set()
        for entry in r.constraints:
            assert entry["state"] in STATES, (c["id"], entry)
            key = (entry["table"], entry["constraint"])
            assert key not in seen, (c["id"], key)  # one state per (table, constraint)
            seen.add(key)
            if entry["state"] != "applied":
                continue
            for h in r.structured:
                if h.dataset == entry["table"]:
                    record = {"climate_class": h.values.get("climate_class"), **h.values}
                    assert not violates(record, entry["constraint"], entry["value"]), (
                        c["id"],
                        h.record_id,
                        entry,
                    )


def test_structured_schemas_file_is_a_fresh_build() -> None:
    import yaml  # noqa: PLC0415

    from mabhas19.retrieval.schema import SCHEMA_PATH, schemas_document  # noqa: PLC0415

    on_disk = yaml.safe_load(SCHEMA_PATH.read_text(encoding="utf-8"))
    fresh = yaml.safe_load(yaml.safe_dump(schemas_document(_store().schemas), allow_unicode=True))
    assert on_disk == fresh


def test_every_dataset_declares_its_lookup_directions() -> None:
    for ds, schema in _store().schemas.items():
        ops = {d.operation for d in schema.directions}
        assert {"direct", "reverse", "whole_table"} <= ops, ds
        assert schema.computations == [], ds  # no calculation is permitted anywhere
        assert all(
            d.kind in ("category", "interval", "label", "entity", "identity")
            for d in schema.dimensions
        )
