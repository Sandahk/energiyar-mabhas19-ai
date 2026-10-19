"""Answerability v3.1: the Mabhas 19 scope map and scope classification (read-only)."""

from __future__ import annotations

import json
from functools import cache

import yaml

from mabhas19.core.paths import CURATED_DIR
from mabhas19.indexing.chunking import build_chunks
from mabhas19.retrieval.hybrid import Retriever
from mabhas19.retrieval.models import RouteDecision
from mabhas19.retrieval.scope import SCOPE_MAP_PATH, ScopeMap, build_scope_map
from mabhas19.retrieval.structured import StructuredStore


@cache
def _retriever() -> Retriever:
    path = CURATED_DIR / "text" / "m19-ed5-1404-main" / "blocks.jsonl"
    blocks = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
    return Retriever(build_chunks(blocks, "m19"), StructuredStore(CURATED_DIR))


def _route(query: str) -> RouteDecision:
    return _retriever().router.route(query)


def _scope(query: str) -> str:
    return str(_route(query).signals["scope"]["class"])


def _refs(query: str) -> list[tuple[str, str]]:
    return [(r["kind"], r["status"]) for r in _route(query).signals["references"]]


def test_scope_map_file_is_a_fresh_build() -> None:
    on_disk = yaml.safe_load(SCOPE_MAP_PATH.read_text(encoding="utf-8"))
    assert on_disk == build_scope_map()


def test_scope_map_identifiers_come_from_curated() -> None:
    m = ScopeMap.load()
    assert "19-5-1-1" in m.sections
    assert "19-2-1-9" in m.sections  # numbered clause
    assert "19-5-7" in m.tables
    assert sorted(m.appendices) == list(range(1, 10))


def test_clearly_external_domain_is_unsupported_with_its_domain() -> None:
    d = _route("ضریب رفتار سازه و ضوابط طراحی لرزه‌ای قاب فولادی چیست؟")
    assert d.route == "unsupported"
    assert d.signals["scope"]["class"] == "external_domain"
    assert "structural_seismic" in d.signals["scope"]["evidence"]["external"]


def test_valid_conceptual_question_with_low_lexical_overlap_is_answered() -> None:
    d = _route("این سیستم مدیریت هوشمند ساختمون دقیقاً چه کاری برامون می‌کنه؟")
    assert d.route == "prose"
    assert d.signals["scope"]["class"] in ("supported", "uncertain")


def test_nonexistent_section_is_invalid_and_not_sent_to_prose() -> None:
    d = _route("بند ۱۹-۹-۳ درباره ممیزی چه می‌گوید")
    assert d.route == "unsupported"
    assert d.prose_query is None
    assert _refs("بند ۱۹-۹-۳ درباره ممیزی چه می‌گوید") == [("section", "nonexistent")]


def test_valid_parent_but_nonexistent_child_section() -> None:
    assert _refs("الزامات بند ۱۹-۵-۱-۹ چیست") == [("section", "nonexistent_child")]
    assert _scope("الزامات بند ۱۹-۵-۱-۹ چیست") == "invalid_reference"


def test_nonexistent_table_and_appendix() -> None:
    assert _refs("مقادیر جدول ۱۹-۵-۹") == [("table", "nonexistent")]
    assert _scope("مقادیر جدول ۱۹-۵-۹") == "invalid_reference"
    assert _refs("طبق پیوست ۱۲ مبحث") == [("appendix", "nonexistent")]
    assert _scope("طبق پیوست ۱۲ مبحث") == "invalid_reference"


def test_malformed_reference() -> None:
    assert ("table", "malformed") in _refs("جدول ۱۹-۵ چه می‌گوید")


def test_valid_references_still_route_normally() -> None:
    assert _route("بند ۱۹-۲-۱-۹ درباره پیوست‌ها").route == "prose"
    assert _refs("طبق پیوست ۸ مبحث") == [("appendix", "valid")]


def test_unknown_structured_entity_keeps_its_typed_outcome() -> None:
    d = _route("رده اقلیمی شهر لندن")
    assert d.route == "unsupported"
    assert d.signals["scope"]["class"] == "unknown_entity"
    r = _retriever().retrieve("رده اقلیمی شهر لندن")
    assert [n.reason for n in r.no_match] == ["unknown_place"]


def test_uncertain_question_falls_through_to_prose() -> None:
    d = _route("تعرفه برق در ساعات اوج چگونه تعیین می‌شود")
    assert d.signals["scope"]["class"] == "uncertain"
    assert d.route == "prose"
    assert d.prose_query


def test_external_hand_off_from_mabhas19_is_traceable() -> None:
    d = _route("اسپرینکلر و سیستم اطفای حریق در ساختمان اداری چطور طراحی می‌شود")
    scope = d.signals["scope"]
    assert scope["class"] == "external_domain"
    assert "مبحث سوم" in scope["reason"]
    assert scope["evidence"]["handed_off_by"]  # Mabhas 19 blocks that refer to Mabhas 3


def test_every_route_records_a_scope_reason() -> None:
    for query in (
        "رده اقلیمی تبریز",
        "رده اقلیمی تبریز چیست و دسته‌بندی اقلیم چگونه انجام می‌شود",
        "تعریف ساختمان کم مصرف",
        "قیمت آجر در بازار",
        "جدول ۱۹-۹-۹",
    ):
        scope = _route(query).signals["scope"]
        assert scope["class"]
        assert scope["reason"]
