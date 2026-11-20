"""The certified city selector (contract ``m2-answer-4``, composer ``m2-composer-6``).

A climate clarification offers two paths: a printed climate class, or a city chosen from the list
derived from the certified city records (Appendix 2 and table 19-3-2). The search box only
filters that list; a typed city is never accepted for a climate question. A chosen city goes
through the certified lookup and the original question continues with its printed class, or,
where the sources disagree (Isfahan), both printed classes are offered and none is chosen.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator
from functools import cache
from pathlib import Path
from typing import Any

import pytest
from tests.unit.test_natural1 import _ask, _engine

from mabhas19.api import cities as C
from mabhas19.api.chatlog import ChatLog
from mabhas19.api.orchestration import city_lookup
from mabhas19.api.server import added_question, clarify_request, make_handler
from mabhas19.core.paths import CURATED_DIR, REPO_ROOT
from mabhas19.generation import options as O
from mabhas19.generation.composer import COMPOSER_VERSION
from mabhas19.generation.contract import CONTRACT_VERSION, Answer
from mabhas19.retrieval.models import RetrievalResult, RouteDecision
from mabhas19.retrieval.schema import APP02, MAIN
from mabhas19.retrieval.structured import StructuredStore

T19_3_2 = f"{MAIN}19-3-2"
PAGE = REPO_ROOT / "src" / "mabhas19" / "api" / "static" / "index.html"
WALL = "دیوار خارجی مجاور فضای باز"
ROOF = "سقف مجاور فضای نیمه باز کنترل نشده"
U_FIXED = "حداکثر ضریب انتقال حرارتی U-Value (W/m2.k) – ثابت"
SHGC = "حد اکثر ضریب بهره خورشیدی SHGC – بین %۲۰ تا %۵۰"


@cache
def _store() -> StructuredStore:
    return StructuredStore(CURATED_DIR)


@cache
def _entries() -> tuple[C.CityEntry, ...]:
    return C.catalogue(_store())


def _names() -> list[str]:
    return [e.name for e in _entries()]


# ---------------------------------------------------------------- versions and the composer


def test_the_contract_and_composer_versions() -> None:
    assert CONTRACT_VERSION == "m2-answer-4"
    assert COMPOSER_VERSION == "m2-composer-6"


def test_a_climate_question_offers_codes_and_the_city_list_but_no_text_box() -> None:
    a = _ask("حداقل مقاومت حرارتی دیوار خارجی مجاور فضای باز چقدر است؟")
    c = a.clarifications[0]
    assert c.field == "climate_class"
    assert [o.label for o in c.options] == O.printed_options(_store(), c.dataset, "climate_class")
    assert c.city_select
    assert not c.free_text
    assert not c.free_text_prefix
    assert not c.placeholder


def test_other_clarifications_never_offer_the_city_list() -> None:
    a = _ask("حداقل مقاومت حرارتی دیوار چقدره؟")
    assert a.response_type == "clarification"
    assert a.clarifications[0].field != "climate_class"
    assert not a.clarifications[0].city_select


# ---------------------------------------------------------------- the catalogue


def test_the_catalogue_has_exactly_the_77_printed_cities() -> None:
    names = _names()
    assert len(names) == 77
    assert len(set(names)) == 77
    assert len({e.key for e in _entries()}) == 77
    assert names.count("تهران") == 1
    assert names.count("اصفهان") == 1
    assert not any("(" in n or ")" in n for n in names)  # stations are the same printed city


def test_the_catalogue_is_derived_from_the_certified_city_records_only() -> None:
    app02 = {C.city_key(C.printed_name(str(r["city"]["text"]))) for r in _store().app02}
    t19 = {
        C.city_key(C.printed_name(str(r["city"]["text"]))) for r in _store().main[T19_3_2].records
    }
    assert {e.key for e in _entries()} == app02 | t19
    for e in _entries():
        assert set(e.datasets) <= {APP02, T19_3_2}
        assert (APP02 in e.datasets) == (e.key in app02)
        assert (T19_3_2 in e.datasets) == (e.key in t19)
    tehran = next(e for e in _entries() if e.name == "تهران")
    assert set(tehran.datasets) == {APP02, T19_3_2}


def test_no_second_city_list_is_kept_in_the_code() -> None:
    src = (REPO_ROOT / "src" / "mabhas19" / "api" / "cities.py").read_text(encoding="utf-8")
    page = PAGE.read_text(encoding="utf-8")
    for name in ("تبریز", "کرمانشاه", "یاسوج", "زاهدان"):
        assert name not in src
        assert name not in page


def test_every_catalogue_entry_is_accepted_by_the_certified_lookup() -> None:
    run = _engine()._run
    refused = [e.name for e in _entries() if city_lookup(run, e.name) is None]
    assert refused == []


# ---------------------------------------------------------------- filtering


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("تهر", ["تهران"]),
        ("تب", ["تبریز"]),
        ("Tehran", []),
        ("tabriz", []),
        ("zzz", []),
        ("شهر ناموجود", []),
        ("كرمان", ["کرمان", "کرمانشاه"]),  # Arabic kaf
        ("اباده", ["آباده"]),  # آ as ا
        ("خرم‌آباد", ["خرم آباد"]),  # ZWNJ and space ignored
        ("بندرعباس", ["بندرعباس"]),
    ],
)
def test_search_is_normalized_substring_filtering_only(query: str, expected: list[str]) -> None:
    assert [e.name for e in C.search(_entries(), query)] == expected


def test_an_empty_query_lists_every_city_and_nothing_is_guessed() -> None:
    assert [e.name for e in C.search(_entries(), "")] == _names()
    assert [e.name for e in C.search(_entries(), "   ")] == _names()
    assert C.selected(_entries(), "تهرا") is None  # a prefix is not a selection
    assert C.selected(_entries(), "Tehran") is None
    assert C.selected(_entries(), "تهران (مهرآباد)") is None  # only the listed name
    entry = C.selected(_entries(), "تهران")
    assert entry is not None
    assert entry.name == "تهران"


# ---------------------------------------------------------------- server validation


def _city_follow(text: str, field: str = "climate_class", kind: str = "city") -> Any:
    raw = {"parent": "b" * 32, "field": field, "kind": kind, "text": text}
    return clarify_request(raw, frozenset(_names()))


def test_a_listed_city_is_a_valid_follow_up_and_reaches_the_question_with_its_prefix() -> None:
    follow = _city_follow("تهران")
    assert follow == {
        "parent_request_id": "b" * 32,
        "field": "climate_class",
        "kind": "city",
        "text": "تهران",
    }
    assert added_question("سؤال", follow) == "سؤال (شهر تهران)"


@pytest.mark.parametrize(
    "follow",
    [
        ("تهرا", "climate_class", "city"),  # not listed
        ("Tehran", "climate_class", "city"),
        ("شهر تهران", "climate_class", "city"),
        ("تهران", "climate_class", "free"),  # typed text for a climate question
        ("شهر تهران", "climate_class", "free"),
        ("تهران", "element", "city"),  # a city only answers a climate question
        ("تهران", "area", "city"),
    ],
)
def test_unknown_cities_and_typed_climate_text_are_rejected(follow: tuple[str, str, str]) -> None:
    assert _city_follow(*follow) is None


def test_other_follow_ups_are_unchanged() -> None:
    names = frozenset(_names())
    option = {"parent": "b" * 32, "field": "climate_class", "kind": "option", "text": "اقلیم 3B"}
    area = {"parent": "b" * 32, "field": "area", "kind": "free", "text": "مساحت 300"}
    assert clarify_request(option, names) is not None
    assert clarify_request(area, names) is not None


class _Stub:
    def __init__(self) -> None:
        self.info: dict[str, Any] = {"mode": "stub"}
        self.cities = _entries()
        self.calls: list[tuple[str, dict[str, Any] | None, str | None]] = []

    def ask(
        self, question: str, *, follow_up: dict[str, Any] | None = None, base: str | None = None
    ) -> tuple[Answer, RetrievalResult, float]:
        self.calls.append((question, follow_up, base))
        result = RetrievalResult(question, question, RouteDecision("unsupported", ["gate"]))
        answer = Answer("a" * 32, question, "out_of_scope", "x", safety={})
        return answer, result, 1.0


@pytest.fixture
def server(tmp_path: Path) -> Iterator[tuple[str, _Stub]]:
    from http.server import ThreadingHTTPServer  # noqa: PLC0415

    stub = _Stub()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(stub, ChatLog(tmp_path)))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", stub
    httpd.shutdown()
    httpd.server_close()


def _http(url: str, payload: dict[str, Any] | None = None) -> tuple[int, Any]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8"))


def test_get_cities_returns_only_the_selectable_entries(server: tuple[str, _Stub]) -> None:
    url, _ = server
    status, body = _http(url + "/api/cities")
    assert status == 200
    assert [c["name"] for c in body["cities"]] == _names()
    assert [c["key"] for c in body["cities"]] == [e.key for e in _entries()]
    assert all(set(c) == {"name", "key"} for c in body["cities"])


def test_the_server_accepts_a_listed_city_and_rejects_anything_else(
    server: tuple[str, _Stub],
) -> None:
    url, stub = server
    ask = {"question": "حداقل مقاومت حرارتی دیوار", "participant": "t1"}
    ok = {"parent": "b" * 32, "field": "climate_class", "kind": "city", "text": "تبریز"}
    status, _ = _http(url + "/api/ask", {**ask, "clarify": ok})
    assert status == 200
    question, follow, base = stub.calls[-1]
    assert question == "حداقل مقاومت حرارتی دیوار (شهر تبریز)"
    assert follow is not None
    assert follow["kind"] == "city"
    assert base == "حداقل مقاومت حرارتی دیوار"
    for bad in (
        {**ok, "text": "شهرک ناموجود"},
        {**ok, "text": "Tabriz"},
        {**ok, "kind": "free", "text": "شهر تبریز"},
        {**ok, "field": "element"},
    ):
        status, body = _http(url + "/api/ask", {**ask, "clarify": bad})
        assert status == 400, bad
        assert body["error"] == "invalid_clarification"
    assert len(stub.calls) == 1


def test_the_page_offers_the_two_paths_and_never_sends_typed_climate_text() -> None:
    page = PAGE.read_text(encoding="utf-8")
    assert "انتخاب اقلیم" in page
    assert "انتخاب شهر" in page
    assert "/api/cities" in page
    assert '"city"' in page


# ---------------------------------------------------------------- end-to-end flows


def _choose(a: Answer, kind: str, text: str, cities: frozenset[str] = frozenset()) -> Answer:
    c = a.clarifications[0]
    follow = clarify_request(
        {"parent": "0" * 32, "field": c.field, "kind": kind, "text": text}, cities
    )
    assert follow is not None, text
    return _ask(added_question(a.question, follow), follow_up=follow, base=a.question)


def _row(a: Answer, label: str) -> Answer:
    assert a.response_type == "clarification", a.headline
    option = next(o for o in a.clarifications[0].options if o.label == label)
    return _choose(a, "option", option.text)


def _city(a: Answer, name: str) -> Answer:
    assert a.response_type == "clarification", a.headline
    assert a.clarifications[0].field == "climate_class"
    assert a.clarifications[0].city_select
    return _choose(a, "city", name, frozenset(_names()))


def _same_values(final: Answer, direct: str) -> None:
    assert final.response_type == "structured_value", final.headline
    assert not final.safety.get("provenance_violations")
    assert not final.safety.get("orchestration_rejected")
    expected = _ask(direct)
    assert expected.values
    assert [(v.record_id, v.value) for v in final.values] == [
        (v.record_id, v.value) for v in expected.values
    ]


@pytest.mark.parametrize(
    ("question", "row", "city", "code", "direct_head"),
    [
        # 19-5-1 R, wall, with the informal word «بیرونی»
        ("برای دیوار بیرونی حداقل R چقدره؟", WALL, "تهران", "3B", f"حداقل مقاومت حرارتی {WALL}"),
        ("حداقل مقاومت حرارتی دیوار چقدره؟", WALL, "تبریز", "4B", f"حداقل مقاومت حرارتی {WALL}"),
        # 19-5-1 R, roof
        ("برای بام حداقل مقاومت حرارتی چقدره؟", ROOF, "تبریز", "4B", f"حداقل مقاومت حرارتی {ROOF}"),
        # 19-5-2 U, with the informal word «دوجداره»
        ("حداکثر ضریب انتقال حرارت پنجره دوجداره چقدره؟", U_FIXED, "تهران", "3B", U_FIXED),
        ("حداکثر ضریب انتقال حرارتی پنجره چقدر است؟", U_FIXED, "تبریز", "4B", U_FIXED),
        # 19-5-2 SHGC
        ("حداکثر SHGC مجاز چقدره؟", SHGC, "تبریز", "4B", SHGC),
    ],
)
def test_row_then_selected_city_reaches_the_certified_value(
    question: str, row: str, city: str, code: str, direct_head: str
) -> None:
    chosen = _row(_ask(question), row)
    assert row in chosen.question
    final = _city(chosen, city)
    _same_values(final, f"{direct_head} در اقلیم {code}")
    assert {v.value for v in final.context} == {code}  # the certified city rows, shown
    assert all(v.dataset in (APP02, T19_3_2) for v in final.context)
    assert final.safety["composite"]["city"]["codes"] == [code]


def test_appendix_3_use_then_selected_city_gives_the_printed_ratings() -> None:
    chosen = _row(_ask("شدت مصرف انرژی مجاز ساختمان اداری چقدره؟"), "اداری خصوصی")
    final = _city(chosen, "تهران")
    direct = _ask("حداکثر شدت مصرف انرژی مجاز اداری خصوصی در اقلیم 3B")
    assert direct.values
    assert [(v.record_id, v.value) for v in final.values] == [
        (v.record_id, v.value) for v in direct.values
    ]
    assert not final.safety.get("provenance_violations")
    assert {v.value for v in final.context} == {"3B"}


def test_a_city_printed_only_in_appendix_2_is_selectable() -> None:
    a = _ask(f"حداقل مقاومت حرارتی {WALL} چقدر است؟")
    final = _city(a, "آباده")
    _same_values(final, f"حداقل مقاومت حرارتی {WALL} در اقلیم 4B")
    assert [(v.dataset, v.row_label, v.value) for v in final.context] == [(APP02, "آباده", "4B")]


# ---------------------------------------------------------------- the source conflict


def test_isfahan_shows_both_printed_classes_and_chooses_none() -> None:
    a = _ask(f"حداقل مقاومت حرارتی {WALL} چقدر است؟")
    out = _city(a, "اصفهان")
    assert out.response_type == "clarification"
    assert not out.values
    c = out.clarifications[0]
    assert c.field == "climate_class"
    assert [o.label for o in c.options] == ["3B", "4B"]
    assert not c.city_select  # only the printed conflicting classes are offered
    assert not c.free_text
    assert {v.value for v in out.context} == {"3B", "4B"}
    assert {v.dataset for v in out.context} == {APP02, T19_3_2}
    assert out.safety["composite"]["kind"] == "city_conflict"
    assert not out.safety.get("orchestration_rejected")
    # choosing one of the printed classes then continues with exactly that class
    final = _choose(out, "option", c.options[1].text)
    _same_values(final, f"حداقل مقاومت حرارتی {WALL} در اقلیم 4B")


def test_direct_city_wording_in_the_question_is_unchanged() -> None:
    """A printed city written in the question itself is still read by the structured rescue."""
    a = _ask("حداقل مقاومت حرارتی برای اردبیل چقدره؟")
    assert a.response_type == "clarification", a.headline
    assert a.safety["composite"]["kind"] == "rescue"
    assert a.clarifications[0].field == "element"
    assert {v.value for v in a.context} == {"5C"}
