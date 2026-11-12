"""Answer composer (M1): response types, refusal preservation, provenance guards, isolation."""

from __future__ import annotations

import ast
import copy
import json
import subprocess
import sys
from dataclasses import replace
from functools import cache

import pytest

from mabhas19.core.paths import CURATED_DIR, REPO_ROOT
from mabhas19.generation import composer
from mabhas19.generation.composer import compose, excerpt_text, printed
from mabhas19.generation.contract import Answer, Condition, ValueItem
from mabhas19.generation.provenance import check, numbers
from mabhas19.indexing.chunking import build_chunks
from mabhas19.retrieval.hybrid import Retriever
from mabhas19.retrieval.models import RetrievalResult, RouteDecision
from mabhas19.retrieval.structured import StructuredStore

PACKAGES = ("generation", "api")


@cache
def _store() -> StructuredStore:
    return StructuredStore(CURATED_DIR)


@cache
def _retriever() -> Retriever:
    path = CURATED_DIR / "text" / "m19-ed5-1404-main" / "blocks.jsonl"
    blocks = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    return Retriever(build_chunks(blocks, "m19-ed5-1404-main"), _store())


def _result(query: str) -> RetrievalResult:
    return copy.deepcopy(_cached(query))


@cache
def _cached(query: str) -> RetrievalResult:
    return _retriever().retrieve(query)


MATERIAL = "ضریب هدایت حرارتی بتن هوادار اتوکلاو شده با چگالی ۴۰۰"
AMBIGUOUS = "حداکثر شدت مصرف انرژی مجاز ساختمان اداری در اقلیم 3B برای رده C"
EUI = "شدت مصرف انرژی مجاز دانشگاه و آموزش عالی در اقلیم 0B رده A"
CONFLICT = "رده اقلیمی اصفهان"
CALCULATION = (
    "حداقل R دیوار مجاور فضای باز برای اقلیمی بین 3B و 4A، مثلاً میانگین دو ستون جدول "
    "۱۹-۵-۱، چقدر می‌شود؟"
)
PRICE = "قیمت آجر در بازار تهران چقدر است"
CONCEPT = "آسایش حرارتی چیست؟"


def test_printed_reads_the_printed_form() -> None:
    assert printed({"raw": "۳۷۵ تا ۴۲۵", "min": "375"}) == "۳۷۵ تا ۴۲۵"
    assert printed({"code": "4B", "cell": "x"}) == "4B"
    assert printed({"symbol": "check_mark"}).startswith("✓")
    assert printed(None) == ""
    assert printed(True) == ""


def test_numbers_are_canonical_across_scripts() -> None:
    assert numbers("۰٫۱۵ و ۱/۴۳ و 0.40 و ٪۹۲") == {"0.15", "1.43", "0.4", "92"}


def test_structured_value_answer_is_built_from_the_certified_row() -> None:
    result = _result(MATERIAL)
    answer = compose(result, _store(), "r1")
    assert answer.response_type == "structured_value"
    assert [v.record_id for v in answer.values] == [h.record_id for h in result.structured]
    v = answer.values[0]
    assert (v.value, v.unit, v.row_label) == ("۰٫۱۵", "W/m.K", "بتن هوادار اتوکلاو شده")
    assert any(c.label == "چگالی" and c.value.startswith("۳۷۵ تا ۴۲۵") for c in v.conditions)
    assert v.source is not None
    assert v.source.label == "پیوست ۷"
    assert v.source.printed_pages == ("۵",)
    assert answer.headline.startswith("ضریب هدایت حرارتی: ۰٫۱۵")
    echo = {i.key: i.value for i in answer.interpretation}
    assert echo["density"] == "400"
    assert echo["output"] == "ضریب هدایت حرارتی"
    assert answer.safety["provenance_violations"] == []


def test_eui_answer_echoes_occupancy_climate_and_rating() -> None:
    answer = compose(_result(EUI), _store(), "r2")
    echo = {i.key: i.value for i in answer.interpretation}
    assert echo["occupancy"] == "دانشگاه و آموزش عالی"  # the printed label, never the ID
    assert echo["climate_class"] == "0B"
    assert echo["rating"] == "A"
    assert answer.values[0].value == "50"


def test_ambiguous_entity_stays_a_refusal_with_the_printed_options() -> None:
    result = _result(AMBIGUOUS)
    assert not result.structured
    answer = compose(result, _store(), "r3")
    assert answer.response_type == "refusal"
    assert not answer.values
    r = answer.refusals[0]
    assert r.outcome == "ambiguous_entity"
    assert len(r.options) == 2
    assert r.hint
    assert any(i.status == "ambiguous" for i in answer.interpretation)


def test_conflicting_sources_show_both_values_and_prefer_none() -> None:
    answer = compose(_result(CONFLICT), _store(), "r4")
    assert answer.response_type == "structured_value"
    assert {v.value for v in answer.values} == {"3B", "4B"}
    assert any(n.kind == "conflict" for n in answer.notices)
    assert "هر دو" in answer.headline


def test_calculation_is_refused_and_never_computed() -> None:
    answer = compose(_result(CALCULATION), _store(), "r5")
    assert answer.response_type == "refusal"
    assert not answer.values
    assert answer.refusals[0].outcome == "unsupported_calculation"


def test_out_of_scope_and_prose_types() -> None:
    oos = compose(_result(PRICE), _store(), "r6")
    assert oos.response_type == "out_of_scope"
    assert not oos.values
    assert not oos.excerpts
    prose = compose(_result(CONCEPT), _store(), "r7")
    assert prose.response_type == "prose"
    assert 1 <= len(prose.excerpts) <= 3
    assert all(e.source.label for e in prose.excerpts)
    assert not prose.interpretation


# ---------------------------------------------------------------- refusal preservation


@pytest.mark.parametrize("query", [AMBIGUOUS, CALCULATION, PRICE, CONCEPT])
def test_a_result_without_certified_rows_never_yields_a_value(query: str) -> None:
    result = _result(query)
    assert not result.structured
    answer = compose(result, _store(), "r8")
    assert not answer.values
    assert answer.response_type != "structured_value"


def test_values_cannot_be_forged_or_carried_onto_a_refusal() -> None:
    """A value item naming a row the engine did not return, or an answer with values checked
    against a result that certified nothing, fails the provenance check."""
    result = _result(MATERIAL)
    answer = composer._compose(result, _store(), "r9")
    forged = copy.deepcopy(answer)
    forged.values = [replace(forged.values[0], record_id="app07-p006-r03")]
    assert any(p.startswith("row:") for p in check(forged, result, _store()))
    refused = _result(AMBIGUOUS)
    assert any(p.startswith("refusal:") for p in check(answer, refused, _store()))


# ---------------------------------------------------------------- provenance guards


def _material_answer() -> tuple[Answer, RetrievalResult]:
    result = _result(MATERIAL)
    return composer._compose(result, _store(), "r10"), result


def test_numeric_provenance_rejects_an_invented_number() -> None:
    answer, result = _material_answer()
    bad = copy.deepcopy(answer)
    bad.headline = "ضریب هدایت حرارتی: ۰٫۲۵"
    assert any(p.startswith("numeric:") for p in check(bad, result, _store()))
    bad = copy.deepcopy(answer)
    bad.values = [replace(bad.values[0], conditions=(Condition("چگالی", "۹۹۹"),))]
    assert any("999" in p for p in check(bad, result, _store()))


def test_row_provenance_rejects_a_value_from_another_row_or_unit() -> None:
    answer, result = _material_answer()
    v: ValueItem = answer.values[0]
    for forged in (replace(v, value="۰٫۱۸"), replace(v, unit="W/m2.K")):
        bad = copy.deepcopy(answer)
        bad.values = [forged]
        assert any(p.startswith(("row:", "numeric:")) for p in check(bad, result, _store()))


def test_citation_provenance_rejects_an_unknown_page() -> None:
    answer, result = _material_answer()
    bad = copy.deepcopy(answer)
    src = bad.values[0].source
    assert src is not None
    bad.sources = [replace(src, printed_pages=("۴۴",))]
    assert any(p.startswith("citation:") for p in check(bad, result, _store()))


def test_interpretation_must_come_from_the_engine() -> None:
    answer, result = _material_answer()
    bad = copy.deepcopy(answer)
    bad.interpretation = [replace(bad.interpretation[0], value="x", source_text=("ساختگی",))]
    assert any(p.startswith("interpretation:") for p in check(bad, result, _store()))


def test_excerpts_must_be_verbatim() -> None:
    result = _result(CONCEPT)
    answer = composer._compose(result, _store(), "r11")
    assert check(answer, result, _store()) == []
    bad = copy.deepcopy(answer)
    bad.excerpts = [replace(bad.excerpts[0], text="آسایش حرارتی یعنی دمای ثابت ۲۲ درجه")]
    problems = check(bad, result, _store())
    assert any(p.startswith("excerpt:") for p in problems)


def test_compose_replaces_a_violating_answer_with_a_refusal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = composer._compose

    def forged(result: RetrievalResult, store: StructuredStore, rid: str) -> Answer:
        a = real(result, store, rid)
        a.headline = "مقدار: ۱۲۳۴۵"
        return a

    monkeypatch.setattr(composer, "_compose", forged)
    answer = compose(_result(MATERIAL), _store(), "r12")
    assert answer.response_type == "refusal"
    assert not answer.values
    assert not answer.interpretation
    assert answer.refusals[0].outcome == "display_check_failed"
    assert answer.safety["provenance_violations"]


def test_excerpt_text_is_a_verbatim_window() -> None:
    text = (
        "جمله اول درباره دیوار است. " * 10 + "جمله کلیدی درباره پنجره ثابت است. " + "پایان. " * 30
    )
    out = excerpt_text(text, ["پنجره"])
    assert "پنجره" in out
    assert len(out) < len(text)
    assert all(piece.strip() in text for piece in out.split("…") if piece.strip())


REVERSE = "حداقل R درهای غیرنورگذر بدون لولا تو همه اقلیم‌ها ۰.۵۷ ـه؟"
WHOLE_TABLE = "جدول ۱۹-۵-۱"
RATING_E = "حداکثر شدت مصرف انرژی ساختمان اداری عمومی در اقلیم 3B برای رده E چقدره؟"
SEVERAL_WALLS = "حداقل مقاومت حرارتی دیوار بین دو واحد گرمایش‌شده در اقلیم 3B طبق جدول ۱۹-۵-۱"


def test_reverse_lookup_rows_show_the_printed_value_they_were_found_by() -> None:
    answer = compose(_result(REVERSE), _store(), "r14")
    assert answer.response_type == "structured_value"
    assert answer.values
    for v in answer.values:
        first = v.conditions[0]
        assert first.label == "حداقل مقاومت حرارتی (R)"
        assert numbers(first.value.split()[0]) == {"0.57"}  # the value, then its printed unit
    assert answer.headline.startswith("ردیف‌های چاپ‌شده‌ای که مقدار")


def test_whole_table_request_names_what_is_missing() -> None:
    answer = compose(_result(WHOLE_TABLE), _store(), "r15")
    r = answer.refusals[0]
    assert r.outcome == "missing_required_dimension"
    assert set(r.missing) == {"جزء پوسته ساختمان", "اقلیم"}
    assert r.hint
    assert r.table == "جدول ۱۹-۵-۱"


def test_refusal_quotes_whole_question_words_only() -> None:
    answer = compose(_result(RATING_E), _store(), "r16")
    assert answer.refusals[0].message.startswith("«E»:")


def test_ambiguous_options_are_listed_one_by_one() -> None:
    answer = compose(_result(SEVERAL_WALLS), _store(), "r17")
    options = answer.refusals[0].options
    assert len(options) == 3
    assert all(";" not in o for o in options)


def test_bare_captions_are_passed_over_while_longer_excerpts_exist() -> None:
    answer = compose(_result(CONCEPT), _store(), "r18")
    assert all(len(e.text) >= composer.MIN_EXCERPT for e in answer.excerpts)


def test_empty_result_is_out_of_scope_not_an_answer() -> None:
    empty = RetrievalResult("x", "x", RouteDecision("prose", []))
    answer = compose(empty, _store(), "r13")
    assert answer.response_type == "out_of_scope"
    assert not answer.values


# ---------------------------------------------------------------- isolation


def test_answer_layer_does_not_import_the_shadow_package() -> None:
    for pkg in PACKAGES:
        for path in (REPO_ROOT / "src" / "mabhas19" / pkg).glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                assert not any(n.startswith("mabhas19.shadow") for n in names), path.name


def test_retrieval_engine_matches_the_frozen_final8_pins() -> None:
    from mabhas19.api.engine import pin_drift  # noqa: PLC0415

    assert pin_drift() == []


def test_loading_the_answer_layer_loads_no_shadow_module() -> None:
    code = (
        "import sys, mabhas19.api.server, mabhas19.api.engine, mabhas19.generation.composer;"
        "print(any(m.startswith('mabhas19.shadow') for m in sys.modules))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_templates_carry_no_regulatory_number() -> None:
    from mabhas19.generation import labels  # noqa: PLC0415

    texts: list[str] = []
    for name in dir(labels):
        if name.isupper() and name != "NOTE_PREFIXES":  # notes quote the engine note they render
            x = getattr(labels, name)
            texts += [v for v in (x.values() if isinstance(x, dict) else [x]) if isinstance(v, str)]
    found = {n for t in texts for n in numbers(t)}
    assert found <= set(labels.CORPUS_NUMBERS), found
