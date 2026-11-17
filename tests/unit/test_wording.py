"""Deterministic wording rules and row families (``mabhas19.api.wording``): every rule's
variants, its limits, and the meaning-changing mappings that must stay absent."""

from __future__ import annotations

import pytest

from mabhas19.api import wording as W
from mabhas19.core.paths import CURATED_DIR
from mabhas19.generation import options as O
from mabhas19.generation.provenance import numbers
from mabhas19.retrieval.schema import MAIN
from mabhas19.retrieval.structured import StructuredStore

ZWNJ = "‌"


def _names(question: str) -> list[str]:
    return [r.name for r in W.rewrite(question)[1]]


@pytest.mark.parametrize(
    ("question", "rule", "expected"),
    [
        ("حداکثر یو ولیو پنجره ثابت", "u_value", "حداکثر U-Value پنجره ثابت"),
        ("حداکثر یوولیو پنجره ثابت", "u_value", "حداکثر U-Value پنجره ثابت"),
        ("حداکثر یو-ولیو پنجره ثابت", "u_value", "حداکثر U-Value پنجره ثابت"),
        (f"حداکثر یو{ZWNJ}ولیو پنجره", "u_value", "حداکثر U-Value پنجره"),
        ("حداکثر اس اچ جی سی پنجره", "shgc", "حداکثر SHGC پنجره"),
        ("حداکثر اس‌اچ‌جی‌سی پنجره", "shgc", "حداکثر SHGC پنجره"),
        ("یکنواختی روشنایی پارکینگ", "uniformity", "یکنواختی نور پارکینگ"),
        ("راندمان آبگرمکن گازی", "gas_water_heater", "راندمان آبگرمکن گازسوز"),
        ("راندمان آب گرمکن گازی", "gas_water_heater", "راندمان آبگرمکن گازسوز"),
        ("راندمان ابگرمکن گازی", "gas_water_heater", "راندمان آبگرمکن گازسوز"),
        ("راندمان بخاری گازی", "gas_heater", "راندمان بخاری گازسوز"),
        ("دمای سینک ظرفشویی آشپزخانه", "sink", "دمای ظرفشویی آشپزخانه"),
        ("دمای سینک ظرف شویی", "sink", "دمای ظرفشویی"),
        ("دمای آب گرم دوش حمام", "shower", "دمای آب گرم دوش"),
        ("مصرف انرژی اداره خصوصی", "office", "مصرف انرژی اداری خصوصی"),
        (f"مصرف انرژی اداره{ZWNJ}ی خصوصی", "office", "مصرف انرژی اداری خصوصی"),
        ("مصرف انرژی ادارهٔ خصوصی", "office", "مصرف انرژی اداری خصوصی"),
        ("بازده اسپلیت", "split", "بازده کولر گازی"),
        ("رده انرژی کولر اسپلیت", "split", "رده انرژی کولر گازی"),
        ("یکنواختی نور دفتر کار", "office_space", "یکنواختی نور اداری و فضاهای کار"),
        ("یکنواختی نور سالن ورزشی", "sports_hall", "یکنواختی نور فضاهای ورزشی"),
        ("بازده پکیج دیواری", "wall_package", "بازده پکیج"),
    ],
)
def test_each_rule_rewrites_its_variants(question: str, rule: str, expected: str) -> None:
    text, rules = W.rewrite(question)
    assert text == expected
    assert rule in [r.name for r in rules]


def test_arabic_letters_are_matched_and_written_as_persian() -> None:
    text, rules = W.rewrite("راندمان بخاري گازي")  # Arabic yeh
    assert [r.name for r in rules] == ["gas_heater"]
    assert text == "راندمان بخاری گازسوز"


def test_a_question_no_rule_matches_is_passed_on_exactly_as_written() -> None:
    q = "حداكثر دماي آب گرم دوش چقدره؟"  # Arabic letters, no rule: the engine normalizes them
    assert W.rewrite(q) == (q, [])


@pytest.mark.parametrize(
    "question",
    [
        "حداقل SEER کولر گازی چقدره؟",  # printed: never touched
        "حداقل بازده پکیج گازی چقدر است؟",  # printed: never touched
        "سیستم گازی چه بازدهی دارد؟",  # «گازی» alone
        "دوش و وان",
        "ادارات دولتی",  # only the word «اداره» itself
    ],
)
def test_printed_and_unrelated_wording_is_left_alone(question: str) -> None:
    assert W.rewrite(question) == (question, [])


@pytest.mark.parametrize(
    "question",
    [
        "دفتر کار من کجاست؟",  # no uniformity asked
        "اسپلیت چند کیلووات است؟",  # no efficiency asked
        "پکیج دیواری چه ابعادی دارد؟",  # no efficiency asked
        "سالن ورزشی چه دمایی دارد؟",
        "وقتی کسی تو اداره نیست شوفاژ رو روی چند درجه بذاریم؟",  # «اداره»: no table lookup
    ],
)
def test_concept_rows_apply_only_with_their_tables_quantity(question: str) -> None:
    assert W.rewrite(question) == (question, [])


@pytest.mark.parametrize(
    "question",
    [
        "کفایت نور روز مدرسه",
        "کفایت نور روز بیمارستان",
        "کفایت نور روز فروشگاه",
        "دمای آب گرم حمام",
        "دمای آب دستشویی عمومی",
        "حداکثر ضریب سایه پنجره",
        "ضریب انتقال حرارت پنجره بازشو",
    ],
)
def test_meaning_changing_mappings_are_never_made(question: str) -> None:
    assert W.rewrite(question) == (question, [])


def test_every_note_is_a_fixed_text_without_numbers() -> None:
    for rule in (*W.WORDING, *W.CONCEPT_ROWS):
        assert not numbers(rule.note), rule.name
        assert rule.shown in rule.note
    for fam in W.FAMILIES:
        assert not numbers(fam.note), fam.name
        assert not numbers(fam.prompt), fam.name


def test_every_concept_row_and_family_row_is_printed() -> None:
    store = StructuredStore(CURATED_DIR)
    for fam in W.FAMILIES:
        printed = O.printed_options(store, fam.dataset, fam.field)
        assert all(r in printed for r in fam.rows), fam.name
    rows = {
        "split": (f"{MAIN}19-5-4", "equipment"),
        "office_space": (f"{MAIN}19-5-7", "space"),
        "sports_hall": (f"{MAIN}19-5-7", "space"),
    }
    for rule in W.CONCEPT_ROWS:
        if rule.name in rows:
            printed = O.printed_options(store, *rows[rule.name])
            assert any(rule.to in p for p in printed), rule.name


@pytest.mark.parametrize(
    ("question", "name"),
    [
        ("برای بام حداقل مقاومت حرارتی چقدره؟", "roof"),
        ("حداقل مقاومت حرارتی پشت بام در اقلیم 3B", "roof"),
        ("حداکثر ضریب انتقال حرارتی پنجره در اقلیم 3B", "window_u"),
        ("حداکثر U-Value پنجره‌ها", "window_u"),
        ("حداکثر ضریب بهره خورشیدی پنجره در اقلیم 2B", "window_shgc"),
        ("حداکثر SHGC پنجره", "window_shgc"),
    ],
)
def test_a_family_is_named_by_its_word_and_its_quantity(question: str, name: str) -> None:
    fam = W.family(question)
    assert fam is not None
    assert fam.name == name


@pytest.mark.parametrize(
    "question",
    [
        "بام سبز چیست؟",  # no quantity
        "حداقل مقاومت حرارتی دیوار و بام",  # another element named too
        "حداکثر ضریب انتقال حرارتی پنجره ثابت",  # the row is already named
        "حداکثر ضریب بهره خورشیدی پنجره با نسبت ۴۰ درصد",  # a band is stated: never guessed
        "حداکثر ضریب انتقال حرارتی و SHGC پنجره",  # two quantities
        "حداکثر SHGC نورگیر سقفی",  # a printed row of its own
    ],
)
def test_no_family_when_the_question_names_more_or_less(question: str) -> None:
    assert W.family(question) is None


@pytest.mark.parametrize(
    ("question", "dataset"),
    [
        ("حداکثر دمای آب گرم مصرفی چقدر باید باشه؟", "19-5-5"),
        ("حداقل نسبت مساحت کفایت نور روز چقدر است؟", "19-5-6"),
        ("حداقل یکنواختی نور مصنوعی چقدر است؟", "19-5-7"),
    ],
)
def test_a_quantity_with_no_row_names_its_table(question: str, dataset: str) -> None:
    m = W.missing_row(question)
    assert m is not None
    assert m.dataset.endswith(dataset)


@pytest.mark.parametrize(
    "question",
    [
        "حداکثر دمای آب گرم استخر چقدر است؟",
        "دمای آب گرم حمام حداکثر چند درجه باشه؟",
        "کفایت نور روز مدرسه چقدره؟",
        "حداقل یکنواختی نور استخر چقدر است؟",
    ],
)
def test_any_other_content_word_keeps_the_printed_rows_unoffered(question: str) -> None:
    assert W.missing_row(question) is None
