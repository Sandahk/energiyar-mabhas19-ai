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
