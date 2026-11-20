"""Offline semantic row-retrieval prototype (``mabhas19.semantic``): catalogue, the retrieval
decision rule and the canonical question. A fake embedder keeps these tests offline and fast."""

from __future__ import annotations

from functools import cache
from typing import Any

import pytest

from mabhas19.core.paths import CURATED_DIR
from mabhas19.retrieval.structured import StructuredStore
from mabhas19.semantic import catalog as C
from mabhas19.semantic import verify as V
from mabhas19.semantic.retrieve import Retriever, Settings, containment


class FakeEmbedder:
    """Bag-of-characters vectors: deterministic, offline, good enough to exercise the rules."""

    def _v(self, text: str) -> list[float]:
        v = [0.0] * 64
        for ch in text:
            v[ord(ch) % 64] += 1.0
        return v

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._v(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._v(text)


@cache
def _store() -> StructuredStore:
    return StructuredStore(CURATED_DIR)


@cache
def _cat() -> C.Catalogue:
    return C.build(_store())


def _retriever(**kw: Any) -> Retriever:
    return Retriever(_cat(), _store(), FakeEmbedder(), Settings(**kw))


def test_the_catalogue_holds_only_printed_rows_per_quantity() -> None:
    cat = _cat()
    assert len(cat.rows) == 217
    assert {r.label for r in cat.for_quantity("hot_water")} == {
        "دوش و وان",
        "روشویی عمومی",
        "روشویی خصوصی",
        "ظرفشویی آشپزخانه",
    }
    assert all("SHGC" in r.label for r in cat.for_quantity("shgc"))
    assert not any("SHGC" in r.label for r in cat.for_quantity("u_value"))
    assert cat.sha256() == C.build(_store()).sha256()  # reproducible


@pytest.mark.parametrize(
    "question",
    [
        "شدت مصرف انرژی یعنی چه؟",  # conceptual
        "حداکثر SHGC و ضریب انتقال حرارت پنجره",  # two quantities
        "نسبت مساحت پنجره به کف چقدر باید باشه؟",  # N58: no printed quantity is named
        "ضریب سایه پنجره چقدره؟",  # not SHGC
    ],
)
def test_no_quantity_no_proposal(question: str) -> None:
    p = _retriever().propose(question)
    assert p.quantity == ""
    assert p.decision == "abstain"
    assert not p.offered


def test_a_number_in_the_question_stops_retrieval() -> None:
    p = _retriever(floor=0.0, band=10.0).propose(
        "حداقل کفایت نور روز ساختمان اداری ۲۰۰۰ متر مربعی؟"
    )
    assert p.decision == "abstain"
    assert p.reason == "the question states a number"


def test_rows_come_only_from_the_resolved_quantity() -> None:
    p = _retriever(floor=0.0, band=10.0).propose("حداکثر دمای آب گرم آشپزخونه چقدره؟")
    assert p.quantity == "hot_water"
    assert {s.row.label for s in p.ranked} <= {r.label for r in _cat().for_quantity("hot_water")}


def test_the_decision_rule_confirms_chooses_or_abstains() -> None:
    q = "حداکثر دمای آب گرم آشپزخونه چقدره؟"
    assert _retriever(floor=99.0).propose(q).decision == "abstain"  # nothing above the floor
    many = _retriever(floor=0.0, band=10.0).propose(q)  # all four close
    assert many.decision == "abstain"
    assert many.reason.endswith("equally close")
    one = _retriever(floor=0.0, band=0.0).propose(q)
    assert one.decision == "confirm"
    assert len(one.offered) == 1


def test_containment_is_the_share_of_user_trigrams_found() -> None:
    assert containment("آشپزخانه", "ظرفشویی آشپزخانه") == 1.0
    assert containment("استخر", "ظرفشویی آشپزخانه") == 0.0


def test_the_canonical_question_uses_printed_words_only() -> None:
    row = next(r for r in _cat().for_quantity("r_value") if r.label == "سقف مجاور فضای آزاد")
    assert (
        V.canonical("r_value", row, "3B") == "حداقل مقاومت حرارتی سقف مجاور فضای آزاد در اقلیم 3B"
    )
    with pytest.raises(AssertionError):
        V.canonical("not_a_quantity", row)


# ---- B2: explicit quantity phrasings, top-3 display and guards

from mabhas19.semantic import quantity as SQ  # noqa: E402
from mabhas19.semantic.retrieve import Proposal, Scored  # noqa: E402


@pytest.mark.parametrize(
    ("question", "quantity"),
    [
        ("کف که زیرش باز است، R حداقلش چنده؟", "r_value"),  # bare R
        ("تخته چوبی چه ضریب هدایت حرارتی‌ای داره؟", "lambda"),  # «چه … داره»
        ("لاندای بلوک رو برای محاسبه دیوار لازم دارم", "lambda"),  # purpose, not method
        ("آب گرم شیر روشویی عمومی حداکثر چند درجه باید باشه؟", "hot_water"),  # «آب گرم … درجه»
    ],
)
def test_b2_phrasings_resolve_the_quantity_deterministically(question: str, quantity: str) -> None:
    r = SQ.analyse(question)
    assert r is not None
    assert r.quantity.id == quantity
    # (at B2's freeze these resolved only through the prototype; since 2026-11-20 the narrow
    # forms A-C are production rules in api/quantities.py, so «extended only» no longer holds)


@pytest.mark.parametrize(
    "question",
    [
        "راندمان دیگ چگونه محاسبه می‌شود؟",  # a method question stays conceptual
        "آب گرم مصرفی ساختمان از کجا تامین میشه؟",  # «آب گرم» without degrees / temperature
        "راندمان آبگرمکن چه رده‌ای داره و دمای آب گرم چند درجه است؟",  # two quantities
        "نسبت مساحت پنجره به کف چه حدی داره؟",  # no printed quantity named
    ],
)
def test_b2_phrasings_never_create_a_quantity_otherwise(question: str) -> None:
    assert SQ.analyse(question) is None


def _b2(**kw: Any) -> Retriever:
    base = {"floor": 0.9, "band": 0.03, "version": "b2", "weak_floor": 0.8, "tie_band": 0.01}
    return _retriever(**(base | kw))


def _decide(r: Retriever, words: list[str], ranked: list[tuple[str, str, float]]) -> Proposal:
    rows = {(x.dataset.rsplit(".", 1)[-1], x.label): x for x in _cat().rows}
    scored = [Scored(rows[(d, lab)], sc, 0.0, sc) for d, lab, sc in ranked]
    p = Proposal("x", words)
    p.ranked = scored[:5]
    r._decide_b2(p, scored, words)
    return p


def test_b2_never_confirms_a_weak_top_row() -> None:
    p = _decide(
        _b2(),
        ["گاراژ"],
        [
            ("t19-5-7", "پارکینگ", 0.85),
            ("t19-5-7", "فضاهای ورزشی", 0.82),
            ("t19-5-7", "کلاس‌های درس", 0.81),
        ],
    )
    assert p.decision == "choose"
    assert len(p.offered) == 3


def test_b2_abstains_below_the_weak_floor_and_on_ties() -> None:
    weak = _decide(
        _b2(), ["استخر"], [("t19-5-5", "دوش و وان", 0.7), ("t19-5-5", "روشویی عمومی", 0.69)]
    )
    assert weak.decision == "abstain"
    tied = _decide(
        _b2(),
        ["پنجره"],
        [
            ("t19-5-2", "حد اکثر ضریب بهره خورشیدی SHGC – کمتر از %۲۰", 0.92),
            ("t19-5-2", "حد اکثر ضریب بهره خورشیدی SHGC – بین %۲۰ تا %۵۰", 0.919),
            ("t19-5-2", "حد اکثر ضریب بهره خورشیدی SHGC – بین %۵۰ تا %۷۰", 0.918),
            ("t19-5-2", "حد اکثر ضریب بهره خورشیدی SHGC – بیشتر از %۷۰", 0.917),
        ],
    )
    assert tied.decision == "abstain"


def test_b2_sibling_guard_keeps_both_coolers_when_the_words_do_not_tell_them_apart() -> None:
    ranked = [
        ("t19-5-4", "کولر گازی – بر اساس نسبت بازدهی انرژی فصلی SEER", 1.0),
        ("t19-5-4", "کولر آبی – خنک کنندگی بر مبنای ضریب بازدهی انرژی EEI", 0.9),
        ("t19-5-4", "فن – دمنده و مکنده", 0.85),
    ]
    vague = _decide(_b2(), ["کولر", "سلولزی"], ranked)
    assert vague.decision == "choose"
    assert {r.label for r in vague.offered} >= {ranked[0][1], ranked[1][1]}
    named = _decide(_b2(), ["کولر", "گازی"], ranked)  # «گازی» is in the top row only
    assert named.decision == "confirm"


def test_b2_cross_table_guard_brings_each_tables_row_for_a_shared_word() -> None:
    ranked = [
        ("t19-5-1", "سقف مجاور فضای آزاد", 0.95),
        ("t19-5-1", "سقف مجاور فضای نیمه باز کنترل نشده", 0.9),
        ("t19-5-1", "دیوار خارجی مجاور فضای باز", 0.85),
        ("t19-5-2", "نورگیر سقفی – حداکثر ضریب انتقال حرارتی U-Value (W/m2.k)", 0.8),
    ]
    p = _decide(_b2(), ["اسکای", "لایت", "سقف"], ranked)
    assert p.decision == "choose"
    assert "نورگیر سقفی – حداکثر ضریب انتقال حرارتی U-Value (W/m2.k)" in {
        r.label for r in p.offered
    }
    assert len(p.offered) <= 3
