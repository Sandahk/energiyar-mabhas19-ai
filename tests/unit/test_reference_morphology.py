"""Caption/reference morphology: a broken plural in a table reference («جدول مراکز استان») names
the same printed caption as its singular; a derived form never matches anything unprinted and an
ambiguous caption still fails closed."""

from __future__ import annotations

import json
from functools import cache

import pytest

from mabhas19.core.paths import CURATED_DIR
from mabhas19.indexing.chunking import build_chunks
from mabhas19.retrieval.entities import keys
from mabhas19.retrieval.hybrid import Retriever
from mabhas19.retrieval.models import RouteDecision
from mabhas19.retrieval.router import caption_keys
from mabhas19.retrieval.structured import StructuredStore

APP02 = "m19-ed5-1404-app02.city_climate_classes"
APP07 = "m19-ed5-1404-app07.material_thermal_properties"
MAIN = "m19-ed5-1404-main.t"


@cache
def _retriever() -> Retriever:
    path = CURATED_DIR / "text" / "m19-ed5-1404-main" / "blocks.jsonl"
    blocks = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
    return Retriever(build_chunks(blocks, "m19"), StructuredStore(CURATED_DIR))


def _route(query: str) -> RouteDecision:
    return _retriever().router.route(query)


@pytest.mark.parametrize(
    ("plural", "singular"),
    [("مراکز", "مرکز"), ("منابع", "منبع"), ("جداول", "جدول"), ("مصارف", "مصرف")],
)
def test_broken_plural_shares_a_key_with_its_singular(plural: str, singular: str) -> None:
    assert caption_keys(plural) & caption_keys(singular)


@pytest.mark.parametrize("word", ["استان", "اقلیم", "مرکز", "تجهیزات", "ایران"])
def test_words_outside_the_pattern_keep_their_keys(word: str) -> None:
    assert caption_keys(word) == keys(word)


def test_a_derived_form_matches_only_captions_printing_the_word_itself() -> None:
    # «حرارت» fits the pattern shape but is no plural: its derived form «حررت» is no word, so
    # it matches exactly the captions that print «حرارت» (or a suffixed form of it)
    router = _retriever().router
    derived = caption_keys("حرارت") - keys("حرارت")
    assert derived
    for ds, cap in router.captions.items():
        title = router._title(ds)
        assert bool(derived & cap) == ("حرارت" in title), ds  # «حرارت», «حرارتی»


@pytest.mark.parametrize(
    ("query", "ref", "table"),
    [
        ("جدول مراکز استان چه شهرهایی دارد؟", "19-3-2", MAIN + "19-3-2"),
        ("جدول مرکز استان چه شهرهایی دارد؟", "19-3-2", MAIN + "19-3-2"),
        ("در جدول ضرایب انتقال حرارت مصالح، چگالی آلومینیوم چقدر است؟", "پ7", APP07),
        ("حداقل بازدهی کولر گازی تو جدول تجهیزات غیرمرکزی چقدره؟", "19-5-4", MAIN + "19-5-4"),
    ],
)
def test_plural_and_singular_references_resolve_alike(query: str, ref: str, table: str) -> None:
    d = _route(query)
    assert d.signals["table_ref"] == [ref]
    assert d.tables == [table]


def test_an_ambiguous_plural_caption_still_fails_closed() -> None:
    # «مرکز» is printed in the captions of 19-3-2 and 19-5-3: no table is chosen
    d = _route("جدول مراکز چه می‌گوید؟")
    assert d.signals["table_ref"] == []
    (amb,) = d.signals["ambiguous_table_ref"]
    assert {MAIN + "19-3-2", MAIN + "19-5-3"} <= set(amb["tables"])
    assert d.tables == []


def test_a_conflict_is_surfaced_when_both_referenced_tables_are_in_scope() -> None:
    res = _retriever().retrieve(
        "رده اقلیمی اصفهان در مبحث ۱۹ چیست؟ جدول مراکز استان و پیوست ۲ یک عدد را می‌گویند؟"
    )
    records = {h.record_id for h in res.structured}
    assert {"main-t19-3-2-r04", "app02-p002-r07"} <= records
    assert all("SI-0009" in h.source_issues for h in res.structured)
    assert res.conflicts
    assert {h.dataset for h in res.structured} <= {APP02, MAIN + "19-3-2"}
