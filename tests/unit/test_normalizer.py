"""Normalizer shadow experiment: catalog, schema, verifier and guards (no network)."""

from __future__ import annotations

import json
from functools import cache
from typing import Any

import pytest

from mabhas19.api.engine import Engine
from mabhas19.core.paths import CURATED_DIR
from mabhas19.generation import options as O
from mabhas19.generation.provenance import numbers
from mabhas19.indexing.chunking import build_chunks
from mabhas19.normalizer import catalog as C
from mabhas19.normalizer import experiment as X
from mabhas19.normalizer import prompt as P
from mabhas19.normalizer.verify import parse, verify
from mabhas19.retrieval.hybrid import Retriever
from mabhas19.retrieval.structured import StructuredStore
from mabhas19.shadow.provider import ExternalCallRefusedError


@cache
def _store() -> StructuredStore:
    return StructuredStore(CURATED_DIR)


@cache
def _engine() -> Engine:
    path = CURATED_DIR / "text" / "m19-ed5-1404-main" / "blocks.jsonl"
    blocks = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    return Engine(
        Retriever(build_chunks(blocks, "m19-ed5-1404-main"), _store()), _store(), "lexical"
    )


def _proposal(**kw: Any) -> str:
    doc = {
        "decision": "propose",
        "family": "none",
        "entities": [],
        "output": "none",
        "climate_code": "",
        "city": "",
        "rating": "",
        "numbers": [],
    }
    doc.update(kw)
    return json.dumps(doc, ensure_ascii=False)


def _verify(raw: str, question: str) -> Any:
    doc, err = parse(raw)
    return verify(doc, question, _store(), _engine()._run, err)


ROOF_KERMAN = "بام تخت تو کرمان چقدر عایق لازمه؟ R حداقلش رو میخوام"
ROOF = "سقف مجاور فضای آزاد"


def test_every_candidate_is_a_printed_option_and_the_schema_is_closed() -> None:
    cat = C.build(_store())
    for f, fam in zip(cat["families"], C.FAMILIES, strict=True):
        assert f["candidates"] == C.candidates(_store(), fam)
        if fam.field:
            assert f["candidates"] == O.printed_options(_store(), fam.dataset, fam.field)
    sch = P.schema(cat)
    # the 217 long row labels are plain strings (Gemini refuses that many long enum values); the
    # verifier enforces them; the remaining enums are short closed choices
    assert sch["properties"]["entities"]["items"] == {"type": "string"}
    assert sch["properties"]["entities"]["maxItems"] == P.MAX_ENTITIES
    assert sch["additionalProperties"] is False
    enums = _enums(sch)
    assert all(len(x) <= 20 for e in enums for x in e)
    assert sum(len(e) for e in enums) < 50
    assert not set(C.all_labels(cat)) & {x for e in enums for x in e}
    # no regulatory number: only «Mabhas 19», «0-based», «at most 3» and the example «3B»
    assert numbers(P.INSTRUCTIONS) == {"0", "3", "19"}


def _enums(node: Any) -> list[list[str]]:
    if isinstance(node, dict):
        found = [node["enum"]] if "enum" in node else []
        return found + [e for v in node.values() for e in _enums(v)]
    if isinstance(node, list):
        return [e for v in node for e in _enums(v)]
    return []


def test_a_proposal_becomes_a_printed_canonical_question_and_a_confirmation() -> None:
    raw = _proposal(family="opaque_r", entities=[ROOF], output="min_r_value", city="کرمان")
    v = _verify(raw, ROOF_KERMAN)
    assert v.valid
    assert (v.climate, v.climate_source) == ("3B", "city")
    assert v.canonical == [f"حداقل مقاومت حرارتی {ROOF} در اقلیم 3B"]
    assert ROOF in v.confirmation
    assert not v.leak


@pytest.mark.parametrize(
    ("raw", "reason"),
    [
        (_proposal(family="opaque_r", entities=["بام"], output="min_r_value"), "entity"),
        (_proposal(family="opaque_r", entities=[ROOF], output="max_eui"), "output"),
        (_proposal(family="nope", entities=[ROOF], output="min_r_value"), "family"),
    ],
)
def test_unprinted_rows_outputs_or_families_are_discarded(raw: str, reason: str) -> None:
    v = _verify(raw, ROOF_KERMAN)
    assert not v.valid
    assert any(d.startswith(reason) for d in v.discarded)


def test_a_climate_must_be_printed_and_written_in_the_question() -> None:
    q = "حداقل R برای بام ساختمان در اقلیم 5A چقدر است؟"
    for code in ("5A", "5B"):  # 5A is not printed in 19-5-1; 5B is not in the question
        v = _verify(
            _proposal(family="opaque_r", entities=[ROOF], output="min_r_value", climate_code=code),
            q,
        )
        assert v.valid
        assert v.climate == ""
        assert any(d.startswith("climate") for d in v.discarded)


def test_numbers_must_be_verbatim_spans_of_the_question() -> None:
    q = "ضریب هدایت حرارتی بتن معمولی با چگالی ۲۴۰۰ کیلوگرم بر متر مکعب چقدر است؟"
    concrete = "بتن‌های با سنگدانه متداول (سیلیسی، سیلیسی-آهکی و آهکی) متراکم"
    good = {"dimension": "density", "text": "۲۴۰۰", "start": q.index("۲۴۰۰")}
    for spans, kept in (
        ([good], [("density", "۲۴۰۰")]),
        ([{**good, "text": "2400"}], []),  # converted digits: not verbatim
        ([{**good, "start": 0}], []),  # wrong offset
        ([{**good, "dimension": "area"}], []),  # not a material dimension
    ):
        raw = _proposal(
            family="material", entities=[concrete], output="thermal_conductivity", numbers=spans
        )
        assert _verify(raw, q).numbers == kept


def test_a_city_must_pass_the_exact_lookup_and_be_written_in_the_question() -> None:
    raw = _proposal(family="opaque_r", entities=[ROOF], output="min_r_value", city="کرما")
    v = _verify(raw, ROOF_KERMAN)
    assert v.city == ""
    raw = _proposal(family="opaque_r", entities=[ROOF], output="min_r_value", city="شیراز")
    assert _verify(raw, ROOF_KERMAN).city == ""  # printed, but not in the question


def test_abstention_and_garbage_are_never_valid() -> None:
    assert _verify(_proposal(decision="abstain"), ROOF_KERMAN).abstained
    for raw in (None, "", "not json", json.dumps({"decision": "propose"})):
        v = verify(*parse(raw)[:1], ROOF_KERMAN, _store(), _engine()._run)
        assert not v.valid


def test_scoring_definitions() -> None:
    gold = {
        "family": "opaque_r",
        "entities": [ROOF],
        "output": "min_r_value",
        "city": "کرمان",
        "climate": "3B",
    }
    base = {
        "valid": True,
        "family": "opaque_r",
        "entities": [ROOF],
        "output": "min_r_value",
        "city": "",
        "climate": "",
        "numbers": [],
    }
    assert X._correct(base, gold)  # omitting the stated city is allowed (the engine asks)
    assert X._correct({**base, "city": "کرمان", "climate": "3B"}, gold)
    assert not X._correct({**base, "climate": "2B"}, gold)  # a different climate is wrong
    assert not X._correct({**base, "entities": [ROOF, "کف مجاور خاک"]}, gold)  # a wrong extra row
    assert not X._correct({**base, "valid": False}, gold)


def test_no_external_call_without_every_guard() -> None:
    make = X._proposer(5.0, allow_external=False)
    with pytest.raises(ExternalCallRefusedError):
        make(C.build(_store()))


def test_the_gold_names_only_printed_rows() -> None:
    assert X._check_gold(_store(), X._cases()) == []
    cases = X._cases()
    ids = [c["id"] for c in cases["failure"]] + [c["id"] for c in cases["abstain_expected"]]
    ids += list(cases["control"]) + list(cases["excluded"])
    assert sorted(ids) == sorted(X._questions())  # every authored question is accounted for


# --- free-text row labels: the verifier is the only closed-set enforcement --------------------


def _roof(**kw: Any) -> str:
    return _proposal(**{"family": "opaque_r", "output": "min_r_value", **kw})


@pytest.mark.parametrize(
    "label",
    [
        "سقف رو به هوای آزاد",  # invented label with the same meaning
        "بام تخت",  # the user's own words
        "Roof exposed to outdoor air",  # another language
        "",
    ],
)
def test_an_invented_row_label_is_rejected(label: str) -> None:
    v = _verify(_roof(entities=[label]), ROOF_KERMAN)
    assert not v.valid
    assert v.entities == []
    assert v.canonical == []
    assert v.confirmation == ""
    assert any(d.startswith("entity") for d in v.discarded)


@pytest.mark.parametrize(
    "label",
    [
        ROOF + " ",  # trailing space
        " " + ROOF,  # leading space
        ROOF + ".",  # punctuation
        ROOF.replace("ی", "ي"),  # Arabic yeh
        ROOF.replace(" ", "‌", 1),  # zero-width non-joiner instead of a space
        ROOF[:-1],  # one character short
        "سقف مجاور فضای باز",  # one word different
    ],
)
def test_a_close_but_not_exact_label_is_rejected_and_not_repaired(label: str) -> None:
    assert label != ROOF
    v = _verify(_roof(entities=[label]), ROOF_KERMAN)
    assert not v.valid
    assert v.entities == []  # never mapped to the nearest printed label
    assert v.discarded == [f"entity: «{label}» is not a printed row of opaque_r"]


def test_a_valid_printed_label_is_accepted_exactly_as_printed() -> None:
    printed = C.candidates(_store(), C.BY_NAME["opaque_r"])
    assert ROOF in printed
    v = _verify(_roof(entities=[ROOF]), ROOF_KERMAN)
    assert v.valid
    assert v.entities == [ROOF]
    assert v.discarded == []


def test_a_printed_label_of_another_family_or_a_non_string_is_rejected() -> None:
    other = C.candidates(_store(), C.BY_NAME["material"])[0]
    assert other not in C.candidates(_store(), C.BY_NAME["opaque_r"])
    for label in (other, 3, None, [ROOF], {"label": ROOF}):
        v = _verify(_roof(entities=[label]), ROOF_KERMAN)
        assert not v.valid
        assert v.entities == []


@pytest.mark.parametrize("output", ["max_eui", "min_r", "Min_R_Value", "", "none", "R"])
def test_an_output_the_family_does_not_print_is_rejected(output: str) -> None:
    v = _verify(_roof(entities=[ROOF], output=output), ROOF_KERMAN)
    assert not v.valid
    assert v.output == ""
    assert v.canonical == []
    assert any(d.startswith("output") for d in v.discarded)


@pytest.mark.parametrize("code", ["5A", "3b", "3B ", "7B", "گرم و خشک"])
def test_a_climate_not_printed_for_the_table_or_not_exact_is_rejected(code: str) -> None:
    q = f"حداقل R برای بام ساختمان در اقلیم {code} چقدر است؟"
    v = _verify(_roof(entities=[ROOF], climate_code=code), q)
    assert v.climate == ""
    assert any(d.startswith("climate") for d in v.discarded)
    assert all(code.strip() not in c for c in v.canonical)


def test_an_invented_number_is_rejected() -> None:
    q = "ضریب هدایت حرارتی بتن معمولی چقدر است؟"  # the question states no density
    concrete = "بتن‌های با سنگدانه متداول (سیلیسی، سیلیسی-آهکی و آهکی) متراکم"
    raw = _proposal(
        family="material",
        entities=[concrete],
        output="thermal_conductivity",
        numbers=[{"dimension": "density", "text": "2400", "start": 0}],
    )
    v = _verify(raw, q)
    assert v.numbers == []
    assert all("2400" not in c for c in v.canonical)
    assert not v.leak


def test_canonical_questions_use_only_verified_fields() -> None:
    raw = _roof(
        entities=[ROOF, "سقف رو به هوای آزاد", ROOF + " "],  # one printed, two not
        climate_code="5A",  # not written in the question
        city="شیراز",  # printed city, not in the question
        rating="A",  # opaque_r takes no rating
        numbers=[{"dimension": "area", "text": "120", "start": 0}],  # not a dimension here
    )
    v = _verify(raw, ROOF_KERMAN)
    assert v.valid
    assert v.entities == [ROOF]
    assert (v.city, v.climate, v.rating, v.numbers) == ("", "", "", [])
    assert v.canonical == [f"حداقل مقاومت حرارتی {ROOF}"]
    assert len(v.discarded) == 6
    assert not v.leak
