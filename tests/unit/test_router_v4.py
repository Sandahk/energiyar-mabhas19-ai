"""Routing v4: plan-driven structured detection, decomposition by plan coverage, schema-based
table references, planner-owned unknown entities and in-scope evidence over external domains."""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path

import pytest
import yaml

from mabhas19.core.paths import CURATED_DIR, EVAL_DIR
from mabhas19.indexing.chunking import build_chunks
from mabhas19.retrieval.entities import Label, LabelIndex, etokens, keys
from mabhas19.retrieval.executor import Executor
from mabhas19.retrieval.hybrid import Retriever
from mabhas19.retrieval.models import RetrievalResult, RouteDecision
from mabhas19.retrieval.structured import StructuredStore
from mabhas19.text.normalize import normalize_for_search

APP02 = "m19-ed5-1404-app02.city_climate_classes"
APP03 = "m19-ed5-1404-app03.max_energy_use_intensity"
APP07 = "m19-ed5-1404-app07.material_thermal_properties"
MAIN = "m19-ed5-1404-main.t"


@cache
def _retriever() -> Retriever:
    path = CURATED_DIR / "text" / "m19-ed5-1404-main" / "blocks.jsonl"
    blocks = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
    return Retriever(build_chunks(blocks, "m19"), StructuredStore(CURATED_DIR))


def _route(query: str) -> RouteDecision:
    return _retriever().router.route(query)


def _run(query: str) -> RetrievalResult:
    return _retriever().retrieve(query)


# ---------------------------------------------------------------- plan-driven detection


def test_router_dry_runs_plans_and_never_executes(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*_: object, **__: object) -> None:
        raise AssertionError("the router executed a plan")

    monkeypatch.setattr(Executor, "run", refuse)
    d = _route("لاندای ماسه سنگ آهکی چنده؟")
    assert (d.route, d.tables) == ("structured", [APP07])


@pytest.mark.parametrize(
    ("query", "tables"),
    [
        ("لاندای ماسه سنگ آهکی چنده؟", [APP07]),  # an output cue, no topic trigger word
        ("climate class تبریز چیه؟", [APP02, MAIN + "19-3-2"]),  # printed English equivalent
        ("سقف شدت مصرف انرژی ویلا در اقلیم 2A برای رده B چقدره؟", [APP03]),  # ویلا ~ ویلایی
    ],
)
def test_valid_plans_route_structured(query: str, tables: list[str]) -> None:
    d = _route(query)
    assert d.route == "structured"
    assert d.tables == tables
    assert all("dry-run plan" in r for r in d.reasons)


def test_a_typed_plan_outcome_routes_structured_without_records() -> None:
    r = _run("جمع حداقل R دیوار خارجی مجاور خاک و کف مجاور خاک در اقلیم 3B چقدر میشه؟")
    assert r.route.route == "structured"
    assert r.structured == []
    assert [n.outcome for n in r.no_match] == ["unsupported_calculation"]


def test_a_shared_word_does_not_make_a_lookup() -> None:
    # «باز» of «بازی» or «روی» (zinc) never anchors a table on their own
    assert _route("بازی فوتبال امشب ساعت چند از تلویزیون پخش میشه؟").tables == []
    assert _route("برای نصب پنل خورشیدی روی پشت بوم چه شرایطی هست؟").tables == []


def test_a_table_that_explains_only_a_stated_constraint_is_not_asked() -> None:
    # 19-3-3 prints rating ranges, 19-3-1 degree-day criteria: neither is the asked value
    d = _route("هتل تو اقلیم 2B، برای رده C حد مجاز kWh/m2.yr چقدره؟")
    assert d.tables == [APP03]


# ---------------------------------------------------------------- decomposition


def test_explanation_form_keeps_its_prose_half() -> None:
    d = _route("حداقل R دیوار خارجی مجاور خاک در اقلیم 3B چنده؟ این دیوار یعنی چه؟")
    assert d.route == "mixed"
    assert d.prose_query == normalize_for_search("این دیوار یعنی چه")
    assert d.tables == [MAIN + "19-5-1"]


def test_purpose_remark_does_not_create_prose() -> None:
    d = _route("رده اقلیمی تبریز چیه؟ برای انتخاب پنجره می‌خوام بدونم")
    assert d.route == "structured"
    assert d.prose_query is None


def test_definition_question_about_a_table_concept_is_prose() -> None:
    assert _route("ضریب عملکرد COP یعنی چه").route == "prose"


# ---------------------------------------------------------------- explicit references


@pytest.mark.parametrize(
    ("query", "ref"),
    [
        ("تو جدول زمان‌بندی برای گروه ب در سال دوم پمپ دورمتغیر چه علامتی داره؟", "19-2-1"),
        ("حداقل بازدهی کولر گازی تو جدول تجهیزات غیرمرکزی ۱۹-۵-۴ چقدره؟", "19-5-4"),
    ],
)
def test_table_references_resolve_from_the_schema(query: str, ref: str) -> None:
    d = _route(query)
    assert ref in d.signals["table_ref"]
    assert d.tables == [MAIN + ref]


def test_an_ambiguous_caption_is_typed_never_guessed() -> None:
    d = _route("جدول دسته‌بندی اقلیمی چه می‌گوید؟")
    assert d.signals["table_ref"] == []
    (amb,) = d.signals["ambiguous_table_ref"]
    assert {MAIN + "19-3-1", MAIN + "19-3-2"} <= set(amb["tables"])
    assert d.tables == []


# ---------------------------------------------------------------- unknown entities


def test_familiar_word_in_a_conceptual_question_is_prose() -> None:
    d = _route("چگالی توان روشنایی یک ساختمان اداری چگونه محاسبه می‌شود؟")
    assert d.route == "prose"
    assert d.signals["unknown_entities"] == []


def test_unknown_entity_with_a_stated_constraint_is_a_typed_lookup() -> None:
    r = _run("حداقل مقاومت حرارتی بام در اقلیم 3B چقدر است؟")
    assert r.route.route == "structured"
    assert r.structured == []
    assert [n.outcome for n in r.no_match] == ["unknown_entity"]


def test_unknown_entity_outcomes_come_from_the_planner() -> None:
    r = _run("رده اقلیمی شهر لندن")
    assert r.route.route == "unsupported"
    assert {(n.outcome, n.reason) for n in r.no_match} == {("unknown_entity", "unknown_place")}


# ---------------------------------------------------------------- scope precedence


@pytest.mark.parametrize(
    "query",
    [
        "پله‌برقی‌ها باید درایوسرعت‌متغیر داشته باشند؟",  # defined term written joined
        "بتن مسلح با سنگدانه معمولی که بیش از ۲ درصد میلگرد داره، ضریب هدایت حرارتیش؟",
        "کاربری‌هایی مثل آموزش و درمان در مبحث چطور دسته‌بندی میشن؟",  # curated row word
    ],
)
def test_in_scope_evidence_outranks_external_domains(query: str) -> None:
    assert _route(query).route != "unsupported"


@pytest.mark.parametrize(
    ("query", "domain"),
    [
        ("ارتفاع نرده پله ساختمان مسکونی چقدر باید باشد؟", "railings_guards"),
        ("فاصله دتکتور دود در راهرو چقدر باشد؟", "fire_life_safety"),
        ("شیب رمپ ورودی ساختمان اداری چند درصد باشد؟", "accessibility"),
    ],
)
def test_reviewed_external_domains(query: str, domain: str) -> None:
    d = _route(query)
    assert d.route == "unsupported"
    assert domain in d.signals["scope"]["evidence"]["external"]


def test_taxonomy_v3_terms_are_absent_from_the_curated_corpus() -> None:
    texts = [
        normalize_for_search(p.read_text(encoding="utf-8"))
        for p in Path(CURATED_DIR).rglob("*")
        if p.suffix in (".jsonl", ".yaml")
    ]
    corpus = " ".join(texts)
    vocab = {k for w in set(etokens(corpus)) for k in keys(w)}
    doc = yaml.safe_load((EVAL_DIR / "retrieval" / "external_domains.yaml").read_text("utf-8"))
    assert doc["version"] == 3
    added = [
        "عمق پی",
        "خاک رسی",
        "دتکتور",
        "آشکارساز دود",
        "نفره",
        "رمپ",
        "ویلچر",
        "نرده",
        "جان پناه",
        "درد",
        "گرمازدگی",
        "گلدون",
        "عکاسی",
        "5g",
    ]
    terms = {normalize_for_search(t) for d in doc["domains"] for t in d["terms"]}
    for term in added:
        t = normalize_for_search(term)
        assert t in terms
        toks = etokens(t)
        if len(toks) == 1:
            assert not keys(toks[0]) & vocab, term
        else:
            assert t not in corpus, term
            assert t.replace(" ", "") not in corpus.replace(" ", ""), term


# ---------------------------------------------------------------- morphology


def test_adjectival_morphology_is_reversible() -> None:
    assert "ویلا" in keys("ویلایی")
    assert "هوا" in keys("هوایی")


def test_degree_prefix_is_part_of_the_printed_compound() -> None:
    # «باز» (open) does not name «نیمه باز» (semi-open): the row stays unresolved
    hits, no_match, _, _ = _retriever().lookup(MAIN + "19-5-1", "حداقل R کف باز در اقلیم 3B")
    assert hits == []
    assert [n.outcome for n in no_match] == ["ambiguous_entity"]
    hits, _, _, _ = _retriever().lookup(
        MAIN + "19-5-1", "حداقل R کف مجاور فضای نیمه‌باز کنترل‌نشده در اقلیم 5B"
    )
    assert [h.record_id for h in hits] == ["main-t19-5-1-r14c05"]


def _matched_words(labels: list[list[str]], query: str) -> dict[str, set[str]]:
    idx = LabelIndex([Label(str(i), t, [0] * len(t)) for i, t in enumerate(labels)])
    return {
        " ".join(m.label.tokens): {m.label.tokens[i] for i in m.label_idx}
        for m in idx.match(idx.query_tokens(query))
    }


@pytest.mark.parametrize(
    ("positive", "qualified"),
    [("باز", "نیمه"), ("چگالشی", "غیر"), ("لولا", "بدون")],
)
def test_meaning_bearing_modifiers_never_match_by_substring(positive: str, qualified: str) -> None:
    # the same rule for degree («نیمه»), negation («غیر», «بدون»): the bare positive term
    # does not name the qualified printed word, and the full qualified phrase does
    label = ["کف", "مجاور", qualified, positive]
    merged = qualified + positive
    assert merged not in _matched_words([label], f"کف {positive}").get(f"کف مجاور {merged}", set())
    assert merged in _matched_words([label], f"کف {qualified} {positive}")[f"کف مجاور {merged}"]


def test_count_qualifiers_follow_the_same_rule() -> None:
    label = ["چیلر", "جذبی", "تک", "اثره"]
    assert "تکاثره" in _matched_words([label], "چیلر جذبی تک اثره")["چیلر جذبی تکاثره"]
    assert "تکاثره" not in _matched_words([label], "چیلر جذبی اثره")["چیلر جذبی تکاثره"]


# ---------------------------------------------------------------- shared structured context


@pytest.mark.parametrize(
    ("query", "records", "prose"),
    [
        (
            "یه آپارتمان ۵۰۰ متری تو اقلیم 3B داریم؛ این جزو مسکونی کوچک حساب میشه؟ "
            "سقف شدت مصرف رده B اش چنده؟",
            ["app03-B-r02-3B"],
            "این جزو مسکونی کوچک حساب میشه",
        ),
        (
            "دیوار خارجی مجاور خاک جزو پوسته غیرنورگذر حساب میشه؟ حداقل R اش در اقلیم 3B چنده؟",
            ["main-t19-5-1-r06c03"],
            "دیوار خارجی مجاور خاک جزو پوسته غیرنورگذر حساب میشه",
        ),
    ],
)
def test_explained_entity_is_shared_with_the_lookup(
    query: str, records: list[str], prose: str
) -> None:
    r = _run(query)
    assert r.route.route == "mixed"
    assert r.route.prose_query == normalize_for_search(prose)  # the explanation stays whole
    assert normalize_for_search(prose) in (r.route.structured_query or "")  # and is shared
    assert [h.record_id for h in r.structured] == records
    assert any(c.get("shared") for c in r.route.signals["clauses"])


def test_context_is_not_shared_when_the_lookup_names_its_entity() -> None:
    d = _route(
        "بیمارستان تو اقلیم 5B برای رسیدن به رده A حداکثر چقدر می‌تونه مصرف کنه؟ "
        "رده A تو مبحث چه عنوانی داره؟"
    )
    assert d.route == "mixed"
    assert "عنوانی" not in (d.structured_query or "")
    assert not any(c.get("shared") for c in d.signals["clauses"])


def test_shared_context_adds_no_rows_beyond_the_identified_entity() -> None:
    # the shared clause names the size class, so exactly one Appendix 3 row is returned
    r = _run(
        "آپارتمانی دقیقاً ۳۰۰۰ متر مربعی در اقلیم 4A دارم؛ در پیوست ۳ جزو ردیف مسکونی بزرگ "
        "حساب می‌شود؟ سقف شدت مصرف رده C آن چند است؟"
    )
    assert [h.record_id for h in r.structured] == ["app03-C-r01-4A"]
    assert r.structured[0].source_issues == ["SI-0003"]
    assert r.no_match == []


# ---------------------------------------------------------------- adjudicated policies


@pytest.mark.parametrize(
    ("query", "records", "outcomes"),
    [
        ("استان اصفهان در کدام دسته اقلیمی قرار می‌گیرد", ["main-t19-3-2-r04"], []),
        (
            "شدت مصرف انرژی مجاز مسکونی را برای اقلیمی بین 2B و 3A میانیابی کن",
            [],
            ["unsupported_calculation"],
        ),
        (
            "حداکثر شدت مصرف انرژی مجاز ساختمان اداری در اقلیم 3B برای رده C",
            [],
            ["ambiguous_entity"],
        ),
        (
            "شدت مصرف انرژی مجاز ساختمان مسکونی در اقلیم 1B رده A",
            [],
            ["missing_required_dimension"],
        ),
    ],
    ids=["H093", "H115", "R027", "R032"],
)
def test_adjudications_hold(query: str, records: list[str], outcomes: list[str]) -> None:
    r = _run(query)
    assert [h.record_id for h in r.structured] == records
    assert [n.outcome for n in r.no_match] == outcomes
