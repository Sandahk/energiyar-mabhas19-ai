"""SH0 shadow proposer: schema, candidate context, verifier, translation, replay (no provider).

* **context** - offsets are exact, the value scanner agrees with I11's, and nothing but query
  text, identifiers and printed unit names carries a digit (no regulatory value is sent);
* **echo** - the production reading written as a proposal verifies and its counterfactual is
  identical to production (the translation does not perturb the pipeline);
* **fixtures** - the two Final-8 semantic classes (a homograph / possessive read as identity; a
  stated background fact returned as its own record), written generically;
* **malicious proposals** - every adversarial class is rejected or narrows: no counterfactual
  row is ever a row production does not return (``_safe``), and certification never fails;
* **metamorphic** - random mutations of valid proposals keep that property; malformed input
  never raises;
* **partial policy** - a failing optional field becomes ``ambiguous`` (fail closed on what it
  concerns), a failing required field rejects the plan, the raw value is never used;
* **replay** - deterministic keys, digests and records; immutable store;
* **production invariance** - production results are byte-identical with and without the
  shadow machinery, no production module imports it, the frozen Final-8 module hashes hold,
  and S1 eligibility never covers a certified answer.
"""

from __future__ import annotations

import copy
import json
import random
import re
from collections.abc import Callable
from dataclasses import asdict
from functools import cache
from pathlib import Path
from typing import Any

import pytest
import yaml

from mabhas19.core.paths import CURATED_DIR, REPO_ROOT
from mabhas19.indexing.chunking import build_chunks
from mabhas19.retrieval import candidate, coverage
from mabhas19.retrieval.hybrid import Retriever
from mabhas19.retrieval.models import RetrievalResult
from mabhas19.retrieval.structured import StructuredStore
from mabhas19.shadow import proposal as P
from mabhas19.shadow.context import CandidateContext, DeterministicFacts, build, canonical_json
from mabhas19.shadow.replay import (
    INSTRUCTIONS,
    ProposerSettings,
    ReplayConflictError,
    ReplayKey,
    ReplayRecord,
    ReplayStore,
    engine_fingerprint,
    provider_schema,
    render_request,
)
from mabhas19.shadow.run import ShadowRun, echo_proposal, shadow
from mabhas19.shadow.translate import fallback_eligible, mask

APP02 = "m19-ed5-1404-app02.city_climate_classes"
APP03 = "m19-ed5-1404-app03.max_energy_use_intensity"
APP07 = "m19-ed5-1404-app07.material_thermal_properties"
T_3_2 = "m19-ed5-1404-main.t19-3-2"

# background fact + another lookup (Final-8 Q264 class, written generically)
BACKGROUND = "ما تو شیراز هستیم که اقلیمش 3B هست؛ حداکثر شدت مصرف انرژی هتل با رده C چقدره؟"
CLAIM = "اقلیم شیراز 3B هست؟"
# grammatical homographs of the closed lexicon («روی» also prints zinc in Appendix 7)
HOMOGRAPH = "ضریب هدایت حرارتی آلومینیوم روی دیوار چقدر است؟"
HOMOGRAPH_ALONE = "ضریب هدایت حرارتی روی چقدر است؟"
# a whole printed name used as a possessive (Final-8 Q099 class, written generically)
POSSESSIVE_WITH_ENTITY = "این زمین مال ماست؛ حداکثر شدت مصرف انرژی هتل تو اقلیم 3B با رده C چقدره؟"
POSSESSIVE_ONLY = "این پروژه مال ماست، تو اقلیم 3B با رده C حداکثر شدت مصرف انرژی چقدره؟"
NAME_AS_IDENTITY = "حداکثر شدت مصرف انرژی مال در اقلیم 3B با رده C چقدره؟"
COORDINATED = "حداکثر شدت مصرف انرژی هتل و بیمارستان در اقلیم 3B با رده C چقدره؟"
CORRECTION = "حداکثر شدت مصرف انرژی هتل در اقلیم 3B با رده C، نه رده D، چقدره؟"
CALCULATION = "اختلاف حداکثر شدت مصرف انرژی هتل در اقلیم 3B بین رده C و رده D چقدره؟"
FOREIGN_UNIT = "حداکثر شدت مصرف انرژی هتل در اقلیم 3B با رده C بر حسب btu چقدره؟"
SOURCE_UNIT = (
    "حداکثر شدت مصرف انرژی هتل در اقلیم 3B با رده C بر حسب کیلووات ساعت بر متر مربع در سال چقدره؟"
)
PROSE = "هدف از تدوین مبحث نوزدهم مقررات ملی ساختمان چیست؟"
FIXTURES = (
    BACKGROUND,
    CLAIM,
    HOMOGRAPH,
    HOMOGRAPH_ALONE,
    POSSESSIVE_WITH_ENTITY,
    POSSESSIVE_ONLY,
    NAME_AS_IDENTITY,
    COORDINATED,
    CORRECTION,
    CALCULATION,
    FOREIGN_UNIT,
    SOURCE_UNIT,
)
SEED = 20261028


def _new_retriever() -> Retriever:
    path = CURATED_DIR / "text" / "m19-ed5-1404-main" / "blocks.jsonl"
    blocks = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
    return Retriever(build_chunks(blocks, "m19"), StructuredStore(CURATED_DIR))


@cache
def _r() -> Retriever:
    return _new_retriever()


@cache
def _base(query: str) -> tuple[RetrievalResult, CandidateContext, DeterministicFacts]:
    prod = _r().retrieve(query)
    ctx, facts = build(_r(), prod)
    return prod, ctx, facts


def _echo(query: str) -> dict[str, Any]:
    _, ctx, facts = _base(query)
    return echo_proposal(ctx, facts)


def _run(query: str, doc: Any) -> ShadowRun:
    run = shadow(_r(), query, doc, production=_base(query)[0])
    _safe(run)
    return run


def _rows(res: RetrievalResult) -> set[str]:
    return {h.record_id for h in res.structured}


def _safe(run: ShadowRun) -> None:
    """The global SH0 property: a counterfactual never returns a row production does not."""
    cf = run.counterfactual
    if cf is None:
        assert run.plan is None
        return
    assert set(cf.rows) <= _rows(run.production), (run.context.query, cf.rows)
    assert cf.diff["adds_rows"] == bool(cf.added_rows)
    if cf.added_rows:
        assert cf.guard == "restriction_added_rows"
        assert cf.rows == []


def _cf_rows(run: ShadowRun) -> list[str]:
    assert run.counterfactual is not None, run.verification and run.verification.to_dict()
    return run.counterfactual.rows


def _index(items: list[dict[str, Any]], text: str) -> int:
    return next(i for i, x in enumerate(items) if x["text"] == text)


def _status(run: ShadowRun) -> str:
    return run.verification.status if run.verification else "unparsed"


def _checks(run: ShadowRun, outcome: str) -> set[str]:
    assert run.verification is not None
    return {d.check for d in run.verification.decisions if d.outcome == outcome}


# ---------------------------------------------------------------- proposal schema


def test_parser_round_trips_a_valid_proposal() -> None:
    doc = _echo(BACKGROUND)
    parsed, errors = P.parse(json.dumps(doc, ensure_ascii=False))
    assert errors == []
    assert parsed is not None
    assert parsed.to_dict()["schema"] == P.PROPOSAL_SCHEMA
    again, _ = P.parse(parsed.to_dict())
    assert again == parsed


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda d: d.update(extra=1), "unknown"),
        (lambda d: d.pop("values"), "missing"),
        (lambda d: d.update(schema="sh0-proposal-0"), "schema"),
        (lambda d: d["clauses"][0].update(role="question"), "role"),
        (lambda d: d["values"][0].update(start=True), "integer"),
        (lambda d: d["values"][0].update(end=d["values"][0]["start"]), "empty"),
        (lambda d: d["values"][0].update(value="3B"), "unknown"),  # no field for a value
        (lambda d: d["operation"].update(kind="lookup_value"), "kind"),
        (lambda d: d["requests"][0].update(unit=-1), "unit"),
        (
            lambda d: d["polarity"].append(
                {"index": 0, "start": 0, "end": 1, "kept": "yes", "clause": 0}
            ),
            "boolean",
        ),
        (lambda d: d.update(notes=3), "notes"),
    ],
)
def test_parser_rejects_malformed_proposals(
    edit: Callable[[dict[str, Any]], Any], message: str
) -> None:
    doc = copy.deepcopy(_echo(BACKGROUND))
    edit(doc)
    parsed, errors = P.parse(doc)
    assert parsed is None
    assert errors
    assert message in errors[0], errors


def test_parser_never_raises_on_garbage() -> None:
    text = json.dumps(_echo(COORDINATED), ensure_ascii=False)
    rng = random.Random(SEED)
    for _ in range(300):
        chars = list(text)
        for _ in range(rng.randint(1, 6)):
            i = rng.randrange(len(chars))
            chars[i] = rng.choice(['"', "{", "]", "0", "-", ",", "x", "", "null"])
        parsed, errors = P.parse("".join(chars))
        assert (parsed is None) == bool(errors)
    for raw in ("", "null", "[]", "1", '{"schema": "sh0-proposal-1"}', "[" * 5000):
        parsed, errors = P.parse(raw)
        assert parsed is None
        assert errors


def test_json_schema_matches_the_parser() -> None:
    schema = P.canonical_schema()
    props = schema["properties"]
    assert schema["required"] == list(P._TOP)
    assert schema["additionalProperties"] is False
    assert props["clauses"]["items"]["properties"]["role"]["enum"] == list(P.CLAUSE_ROLES)
    assert props["entities"]["items"]["properties"]["use"]["enum"] == list(P.ENTITY_USES)
    assert props["values"]["items"]["properties"]["role"]["enum"] == list(P.VALUE_ROLES)
    assert props["operation"]["properties"]["kind"]["enum"] == list(P.OPERATIONS)
    assert props["requests"]["items"]["properties"]["intent"]["enum"] == list(P.INTENTS)
    # no field can carry a value, record, citation or name
    flat = json.dumps(schema)
    for word in ('value"', "record", "citation", "quote", "source"):
        assert word not in flat.replace("values", ""), word


# ---------------------------------------------------------------- candidate context


_ID_KEYS = frozenset({"id", "table", "row", "unit", "canonical", "version"})
_QUERY_KEYS = frozenset({"query", "text"})


def _no_values(ctx: CandidateContext) -> None:
    """Digits occur only in query slices, identifiers and printed unit names."""

    def walk(x: Any, key: str = "") -> None:
        if isinstance(x, dict):
            for k, v in x.items():
                walk(v, k)
        elif isinstance(x, list | tuple):
            for v in x:
                walk(v, key)
        elif isinstance(x, str) and re.search(r"\d", x):
            if key in _QUERY_KEYS:
                assert x in ctx.query, (key, x)
            elif key == "value":  # the comparable form of a stated value
                assert x.removeprefix("elapsed:").lower() in ctx.query or x.isdigit(), x
            else:
                assert key in _ID_KEYS, (key, x)

    walk(ctx.to_dict())


def _offsets(ctx: CandidateContext) -> None:
    q = ctx.query
    for group in (ctx.clauses, ctx.entities, ctx.values, ctx.units, ctx.operations, ctx.polarity):
        for item in group:
            assert 0 <= item.start < item.end <= len(q)
            assert q[item.start : item.end] == item.text, item
    for c in ctx.coordination:
        for m in c.members:
            assert q[m.start : m.end] == m.text
    starts = [c.start for c in ctx.clauses]
    assert starts == sorted(starts)


def _dev_queries(n: int, sets: tuple[str, ...] = ("dev", "final7", "final8")) -> list[str]:
    qs: list[str] = []
    for s in sets:
        doc = yaml.safe_load((REPO_ROOT / "eval" / "retrieval" / f"{s}.yaml").read_text("utf-8"))
        qs += [str(c["query"]) for c in doc["cases"]]
    random.Random(SEED).shuffle(qs)
    return qs[:n]


@pytest.mark.parametrize("query", [*FIXTURES, *_dev_queries(25)])
def test_context_is_exact_and_carries_no_regulatory_value(query: str) -> None:
    _, ctx, _ = _base(query)
    _offsets(ctx)
    _no_values(ctx)
    found: dict[str, set[str]] = {d: set() for d in ("rating", "climate_class", "group", "year")}
    for v in ctx.values:
        if v.dimension != "number":
            found[v.dimension].add(v.value)
    assert found == coverage.value_mentions(ctx.query)  # the I11 scanner, read with offsets
    assert ctx.aligned
    assert len({c.id for e in ctx.entities for c in e.candidates}) == len(ctx.candidate_ids())
    again, _ = build(_r(), _base(query)[0])
    assert again.sha256() == ctx.sha256()


def test_context_offers_the_identity_candidates_and_homographs() -> None:
    _, ctx, _ = _base(HOMOGRAPH)
    by_text = {e.text: e for e in ctx.entities}
    assert by_text["روی"].homograph
    assert {c.status for c in by_text["روی"].candidates} == {"homograph_only"}
    assert any(c.status == "exact" for c in by_text["الومینیوم"].candidates)
    _, ctx, _ = _base(BACKGROUND)
    assert [c.text for c in ctx.clauses] == [
        "ما تو شیراز هستیم که اقلیمش 3b هست",
        "حداکثر شدت مصرف انرژی هتل با رده c چقدره",
    ]
    assert [(v.dimension, v.value) for v in ctx.values] == [
        ("climate_class", "3B"),
        ("rating", "C"),
    ]
    assert {t.id for t in ctx.tables} == {APP02, T_3_2, APP03}


def test_request_sent_to_a_provider_holds_no_value_or_corpus_text() -> None:
    _, ctx, _ = _base(SOURCE_UNIT)
    request = render_request(ctx)
    # model, limits, the pinned instructions, the redacted context and the projected schema:
    # nothing else (no document, no corpus text, no credential, no sampling parameter)
    assert set(request) == {"model", "max_tokens", "system", "messages", "output_config"}
    assert request["model"] == "claude-opus-5-5"
    assert request["system"] == INSTRUCTIONS
    assert request["output_config"] == {
        "effort": "medium",
        "format": {"type": "json_schema", "schema": provider_schema()},
    }
    [message] = request["messages"]
    assert message["role"] == "user"
    assert json.loads(message["content"])["context"] == json.loads(json.dumps(ctx.to_dict()))
    assert not re.search(r"\d", INSTRUCTIONS.replace("Mabhas 19", ""))
    _no_values(ctx)
    texts = {line for c in _r().lexical.chunks[:200] for line in c.text.splitlines()}
    body = canonical_json(request)
    assert not any(len(t) > 40 and t in body for t in texts)


# ---------------------------------------------------------------- echo (negative control)


@pytest.mark.parametrize("query", [*FIXTURES, PROSE, *_dev_queries(25, ("heldout", "final6"))])
def test_the_production_reading_as_a_proposal_changes_nothing(query: str) -> None:
    run = _run(query, _echo(query))
    assert _status(run) == "verified", run.verification and run.verification.to_dict()
    assert run.counterfactual is not None
    assert run.counterfactual.diff["identical"], run.counterfactual.to_dict()


# ---------------------------------------------------------------- Final-8 classes


def test_background_fact_never_becomes_its_own_record() -> None:
    prod, ctx, _ = _base(BACKGROUND)
    assert {h.dataset for h in prod.structured} == {APP02, APP03}  # production: the Q264 class
    doc = _echo(BACKGROUND)
    doc["clauses"][0]["role"] = "background"
    doc["values"][_index(doc["values"], "3b")]["role"] = "lookup_constraint"
    run = _run(BACKGROUND, doc)
    assert _status(run) == "verified"
    assert run.plan is not None
    assert run.plan.dropped == {APP02: "anchored_outside_requested_clauses",
                                T_3_2: "anchored_outside_requested_clauses"}  # fmt: skip
    # the stated fact still constrains the other lookup; its own row is gone
    assert _cf_rows(run) == [h.record_id for h in prod.structured if h.dataset == APP03]
    assert run.counterfactual
    assert run.counterfactual.diff["removes_rows"]
    # the model listing the background tables as requests changes nothing
    doc2 = copy.deepcopy(doc)
    assert {r["table"] for r in doc2["requests"]} == {APP02, T_3_2, APP03}
    assert _cf_rows(_run(BACKGROUND, doc2)) == _cf_rows(run)
    del ctx


def test_a_stated_fact_read_as_a_lookup_constraint_is_not_a_claim() -> None:
    doc = _echo(BACKGROUND)
    doc["values"][_index(doc["values"], "3b")]["role"] = "lookup_constraint"
    run = _run(BACKGROUND, doc)  # clause 0 still a lookup clause
    assert run.plan is not None
    assert run.plan.dropped.get(APP02) == "claim_not_requested"
    assert all(not r.startswith("app02") for r in _cf_rows(run))


def test_a_verified_claim_request_keeps_the_deterministic_claim_result() -> None:
    prod, _, _ = _base(CLAIM)
    doc = _echo(CLAIM)
    doc["clauses"][0]["role"] = "verify_request"
    assert doc["values"][0]["role"] == "verify_request"
    run = _run(CLAIM, doc)
    assert sorted(_cf_rows(run)) == sorted(_rows(prod))
    assert _rows(prod)


def test_a_background_claim_produces_no_row() -> None:
    doc = _echo(CLAIM)
    doc["values"][0]["role"] = "background"
    run = _run(CLAIM, doc)
    assert run.plan is not None
    assert set(run.plan.dropped.values()) == {"claim_not_requested"}
    assert _cf_rows(run) == []


def test_a_background_value_the_lookup_applies_refuses_that_lookup() -> None:
    doc = _echo(BACKGROUND)
    doc["clauses"][0]["role"] = "background"
    doc["values"][_index(doc["values"], "3b")]["role"] = "background"
    run = _run(BACKGROUND, doc)
    assert run.counterfactual is not None
    assert run.counterfactual.refused.get(APP03) == "value_role_conflict"
    assert _cf_rows(run) == []


def test_a_lexicon_homograph_forced_to_identity_is_rejected() -> None:
    for query in (HOMOGRAPH, HOMOGRAPH_ALONE):
        doc = _echo(query)
        e = doc["entities"][_index(doc["entities"], "روی")]
        _, ctx, _ = _base(query)
        e["use"], e["candidate"] = "identity", ctx.entities[e["index"]].candidates[0].id
        run = _run(query, doc)
        assert _status(run) == "rejected"
        assert "homograph" in _checks(run, "rejected")
        assert run.counterfactual is None


def test_a_homograph_marked_grammatical_names_nothing_unless_another_entity_does() -> None:
    for query, expected in ((HOMOGRAPH_ALONE, []), (HOMOGRAPH, ["app07-p013-r07"])):
        doc = _echo(query)
        doc["entities"][_index(doc["entities"], "روی")]["use"] = "grammatical"
        run = _run(query, doc)
        assert _status(run) == "verified"
        assert _cf_rows(run) == expected  # aluminium is still grounded by its own word


def test_a_possessive_name_marked_grammatical_withdraws_the_identity() -> None:
    # alone: no occupancy is left, the lookup fails closed
    doc = _echo(POSSESSIVE_ONLY)
    doc["entities"][_index(doc["entities"], "مال")]["use"] = "grammatical"
    run = _run(POSSESSIVE_ONLY, doc)
    assert run.plan is not None
    assert [m["text"] for m in run.plan.masked] == ["مال"]
    assert run.counterfactual is not None
    assert "مال" not in run.counterfactual.query
    assert _cf_rows(run) == []
    # beside another grounded entity: the pipeline now names the hotel - a row production did
    # not return, so SH0 logs it and refuses it (restrictions never add rows)
    prod, _, _ = _base(POSSESSIVE_WITH_ENTITY)
    assert not prod.structured
    doc = _echo(POSSESSIVE_WITH_ENTITY)
    doc["entities"][_index(doc["entities"], "مال")]["use"] = "grammatical"
    run = _run(POSSESSIVE_WITH_ENTITY, doc)
    cf = run.counterfactual
    assert cf is not None
    assert cf.added_rows == ["app03-C-r20-3B"]
    assert cf.guard == "restriction_added_rows"
    assert cf.rows == []
    assert cf.diff["adds_rows"]
    assert run.fallback_eligible  # production failed closed: what a future S1 would review


def test_a_printed_name_outside_the_lexicon_follows_the_production_reading() -> None:
    prod, _, _ = _base(NAME_AS_IDENTITY)
    assert _rows(prod) == {"app03-C-r27-3B"}
    assert _cf_rows(_run(NAME_AS_IDENTITY, _echo(NAME_AS_IDENTITY))) == ["app03-C-r27-3B"]
    doc = _echo(NAME_AS_IDENTITY)
    doc["entities"][_index(doc["entities"], "مال")]["use"] = "grammatical"
    run = _run(NAME_AS_IDENTITY, doc)  # a wrong «grammatical» only narrows
    assert _cf_rows(run) == []
    assert run.counterfactual
    assert run.counterfactual.diff["removes_rows"]


# ---------------------------------------------------------------- malicious proposals


def _mutated(query: str, edit: Callable[[dict[str, Any], CandidateContext], Any]) -> ShadowRun:
    doc = copy.deepcopy(_echo(query))
    edit(doc, _base(query)[1])
    return _run(query, doc)


def _foreign_candidate(d: dict[str, Any], _: CandidateContext) -> None:
    d["entities"][0]["candidate"] = f"{APP07}|app07-p999-r99"


def _other_span_candidate(d: dict[str, Any], ctx: CandidateContext) -> None:
    d["entities"][0]["candidate"] = ctx.entities[1].candidates[0].id


def _other_span_offsets(d: dict[str, Any], ctx: CandidateContext) -> None:
    d["entities"][0].update(start=ctx.entities[1].start, end=ctx.entities[1].end,
                            text=ctx.entities[1].text)  # fmt: skip


def _shift(d: dict[str, Any], _: CandidateContext) -> None:
    d["values"][0]["start"] += 1


def _swap(d: dict[str, Any], ctx: CandidateContext) -> None:
    d["entities"][0]["candidate"] = ctx.entities[1].candidates[0].id
    d["entities"][1]["candidate"] = ctx.entities[0].candidates[0].id


def _invented_output(d: dict[str, Any], _: CandidateContext) -> None:
    d["requests"][0]["output"] = "max_eui_kwh"


def _foreign_unit(d: dict[str, Any], ctx: CandidateContext) -> None:
    d["requests"][0].update(output="max_eui", unit=ctx.units[0].index)


def _omit_value(d: dict[str, Any], _: CandidateContext) -> None:
    d["values"].pop()


def _drop_exclusion(d: dict[str, Any], _: CandidateContext) -> None:
    d["polarity"][0]["kept"] = False


def _exclusion_as_constraint(d: dict[str, Any], _: CandidateContext) -> None:
    d["values"][_index(d["values"], "d")]["role"] = "lookup_constraint"


def _drop_calculation(d: dict[str, Any], _: CandidateContext) -> None:
    d["operation"] = {"kind": "none", "refs": []}


def _background_as_lookup(d: dict[str, Any], _: CandidateContext) -> None:
    d["clauses"][0]["role"] = "lookup"
    d["values"][_index(d["values"], "3b")]["role"] = "verify_request"


def _lookup_as_background(d: dict[str, Any], _: CandidateContext) -> None:
    d["clauses"][1]["role"] = "background"


def _homograph_identity(d: dict[str, Any], ctx: CandidateContext) -> None:
    e = d["entities"][_index(d["entities"], "روی")]
    e.update(use="identity", candidate=ctx.entities[e["index"]].candidates[0].id)


def _drop_member(d: dict[str, Any], _: CandidateContext) -> None:
    d["coordination"][0]["members"].pop()


def _arbitrary_table(d: dict[str, Any], _: CandidateContext) -> None:
    d["requests"].append({"table": APP07, "intent": "lookup", "output": "density", "unit": "none"})


def _invented_table(d: dict[str, Any], _: CandidateContext) -> None:
    d["requests"].append({"table": "m19-ed5-1404-main.t19-9-9", "intent": "lookup",
                          "output": "none", "unit": "none"})  # fmt: skip


def _value_points_elsewhere(d: dict[str, Any], ctx: CandidateContext) -> None:
    d["values"][0].update(start=ctx.values[1].start, end=ctx.values[1].end)


MALICIOUS: list[tuple[str, str, Callable[[dict[str, Any], CandidateContext], Any], set[str]]] = [
    ("candidate ID outside the list", HOMOGRAPH, _foreign_candidate, {"invalid"}),
    ("correct ID, wrong span", COORDINATED, _other_span_candidate, {"partial"}),
    ("correct ID, another span's offsets", COORDINATED, _other_span_offsets, {"invalid"}),
    ("invented offsets", BACKGROUND, _shift, {"invalid"}),
    ("swapped entities", COORDINATED, _swap, {"partial"}),
    ("invented output", SOURCE_UNIT, _invented_output, {"invalid"}),
    ("foreign unit marked as the source unit", FOREIGN_UNIT, _foreign_unit, {"partial"}),
    ("omitted stated value", BACKGROUND, _omit_value, {"rejected"}),
    ("removed exclusion", CORRECTION, _drop_exclusion, {"rejected"}),
    ("excluded value used as a constraint", CORRECTION, _exclusion_as_constraint, {"rejected"}),
    ("removed calculation intent", CALCULATION, _drop_calculation, {"rejected"}),
    ("background clause relabelled as lookup", BACKGROUND, _background_as_lookup, {"verified"}),
    ("lookup clause relabelled as background", BACKGROUND, _lookup_as_background, {"verified"}),
    ("homograph forced to identity", HOMOGRAPH, _homograph_identity, {"rejected"}),
    ("one coordination member dropped", COORDINATED, _drop_member, {"rejected"}),
    ("arbitrary extra table (not supplied)", BACKGROUND, _arbitrary_table, {"invalid"}),
    ("invented table", BACKGROUND, _invented_table, {"invalid"}),
    ("value reference pointing to another span", BACKGROUND, _value_points_elsewhere, {"invalid"}),
]


@pytest.mark.parametrize(
    ("name", "query", "edit", "statuses"), MALICIOUS, ids=[m[0] for m in MALICIOUS]
)
def test_malicious_proposals_are_rejected_or_narrow(
    name: str,
    query: str,
    edit: Callable[[dict[str, Any], CandidateContext], Any],
    statuses: set[str],
) -> None:
    run = _mutated(query, edit)
    assert _status(run) in statuses, (name, run.verification and run.verification.to_dict())
    if run.counterfactual is not None:  # verified or partial: narrows (``_safe`` ran)
        assert set(run.counterfactual.rows) <= _rows(run.production)
        assert run.counterfactual.certification_failures == 0
        assert not run.counterfactual.added_rows


def test_malicious_relabelling_cannot_restore_more_than_production() -> None:
    prod, _, _ = _base(BACKGROUND)
    run = _mutated(BACKGROUND, _background_as_lookup)
    assert set(_cf_rows(run)) == _rows(prod)  # the Q264 row: what production already returns
    run = _mutated(BACKGROUND, _lookup_as_background)
    assert {r for r in _cf_rows(run)} < _rows(prod)


# ---------------------------------------------------------------- metamorphic


def _random_edit(rng: random.Random, d: dict[str, Any], ctx: CandidateContext) -> None:  # noqa: PLR0912
    choice = rng.randrange(12)
    if choice == 0 and d["clauses"]:
        rng.choice(d["clauses"])["role"] = rng.choice(P.CLAUSE_ROLES)
    elif choice == 1 and d["entities"]:
        rng.choice(d["entities"])["use"] = rng.choice(P.ENTITY_USES)
    elif choice == 2 and d["entities"]:
        ids = [*sorted(ctx.candidate_ids()), P.NONE, P.AMBIGUOUS, f"{APP07}|fake"]
        rng.choice(d["entities"])["candidate"] = rng.choice(ids)
    elif choice == 3 and d["values"]:
        rng.choice(d["values"])["role"] = rng.choice(P.VALUE_ROLES)
    elif choice == 4 and d["values"]:
        v = rng.choice(d["values"])
        v["clause"] = rng.choice([None, *range(len(ctx.clauses) + 1)])
    elif choice == 5:
        d["operation"] = {"kind": rng.choice(P.OPERATIONS), "refs": []}
    elif choice == 6 and d["requests"]:
        r = rng.choice(d["requests"])
        outputs = [o.name for t in ctx.tables for o in t.outputs]
        r["output"] = rng.choice([*outputs, P.NONE, P.AMBIGUOUS])
        r["intent"] = rng.choice(P.INTENTS)
    elif choice == 7 and d["requests"]:
        d["requests"][rng.randrange(len(d["requests"]))]["unit"] = rng.choice(
            [P.NONE, P.AMBIGUOUS, *range(len(ctx.units) + 1)]
        )
    elif choice == 8 and d["requests"]:
        d["requests"].pop(rng.randrange(len(d["requests"])))
    elif choice == 9:
        t = rng.choice([*(x.id for x in ctx.tables), APP07, APP02])
        d["requests"].append({"table": t, "intent": "lookup", "output": "none", "unit": "none"})
    elif choice == 10 and d["polarity"]:
        rng.choice(d["polarity"])["kept"] = rng.random() < 0.5
    elif choice == 11:
        group = rng.choice(["clauses", "entities", "values"])
        if d[group]:
            item = rng.choice(d[group])
            item["start"] = max(0, item["start"] + rng.choice([-2, -1, 1, 2]))


def test_random_mutations_never_add_a_row() -> None:
    rng = random.Random(SEED)
    seen: dict[str, int] = {}
    for _ in range(60):
        query = rng.choice(FIXTURES)
        doc = copy.deepcopy(_echo(query))
        ctx = _base(query)[1]
        for _ in range(rng.randint(1, 3)):
            _random_edit(rng, doc, ctx)
        run = _run(query, doc)  # ``_safe``: never a row production does not return
        seen[_status(run)] = seen.get(_status(run), 0) + 1
        if run.counterfactual is not None:
            assert run.counterfactual.certification_failures == 0
    assert set(seen) >= {"invalid", "rejected", "verified"}, seen


def test_adversarial_restrictions_only_remove_rows_on_every_fixture() -> None:
    """Every single-field restriction a model could make (a clause as background, a value as
    background, an entity as grammatical or ambiguous, a request dropped or ambiguous) keeps
    or removes production rows."""
    for query in FIXTURES:
        base = _echo(query)
        edits: list[tuple[str, int, dict[str, str] | None]] = [
            *(("clauses", i, {"role": "background"}) for i in range(len(base["clauses"]))),
            *(("values", i, {"role": "background"}) for i in range(len(base["values"]))),
            *(
                ("entities", i, upd)
                for i in range(len(base["entities"]))
                for upd in ({"use": "grammatical", "candidate": "none"}, {"candidate": "ambiguous"})
            ),
            *(("requests", i, {"intent": "ambiguous"}) for i in range(len(base["requests"]))),
            *(("requests", i, None) for i in range(len(base["requests"]))),  # not requested
        ]
        for group, i, update in edits:
            doc = copy.deepcopy(base)
            if update is None:
                doc[group].pop(i)
            else:
                doc[group][i].update(update)
            _run(query, doc)


# ---------------------------------------------------------------- partial-proposal policy


def test_a_failing_optional_field_becomes_ambiguous_and_fails_closed() -> None:
    prod, ctx, _ = _base(COORDINATED)
    assert len(_rows(prod)) == 2
    run = _mutated(COORDINATED, _other_span_candidate)
    assert run.verification is not None
    assert run.verification.proposal is not None
    assert _status(run) == "partial"
    replaced = [d for d in run.verification.decisions if d.outcome == "replaced"]
    assert [(d.field, d.check) for d in replaced] == [("entities", "span_candidates")]
    # the raw model value is never used: the verified proposal says «ambiguous»
    assert run.verification.proposal.entities[0].candidate == P.AMBIGUOUS
    raw_id = ctx.entities[1].candidates[0].id
    assert raw_id not in json.dumps(run.plan.to_dict() if run.plan else {})
    assert run.counterfactual is not None
    assert run.counterfactual.refused == {APP03: "entity_ambiguous"}
    assert _cf_rows(run) == []


def test_a_failing_required_field_rejects_the_whole_plan() -> None:
    for query, edit in ((BACKGROUND, _omit_value), (CALCULATION, _drop_calculation),
                        (COORDINATED, _drop_member), (CORRECTION, _drop_exclusion)):  # fmt: skip
        run = _mutated(query, edit)
        assert _status(run) == "rejected"
        assert run.verification is not None
        assert run.verification.proposal is None
        assert run.plan is None
        assert run.counterfactual is None


def test_replacing_any_optional_field_by_ambiguous_only_narrows() -> None:
    for query in FIXTURES:
        base = _echo(query)
        rows = set(_cf_rows(_run(query, base)))
        for i in range(len(base["requests"])):
            for key in ("output", "unit", "intent"):
                doc = copy.deepcopy(base)
                doc["requests"][i][key] = P.AMBIGUOUS
                assert set(_cf_rows(_run(query, doc))) <= rows


def test_unit_requests_are_checked_by_the_unit_model() -> None:
    doc = _echo(SOURCE_UNIT)
    doc["requests"][0].update(output="max_eui", unit=0)
    run = _run(SOURCE_UNIT, doc)
    assert _status(run) == "verified"
    assert run.counterfactual
    assert run.counterfactual.diff["identical"]
    run = _mutated(FOREIGN_UNIT, _foreign_unit)
    assert "I4_unit" in _checks(run, "replaced")
    assert _cf_rows(run) == []


def test_an_added_calculation_refuses_and_a_detected_one_stays() -> None:
    doc = _echo(BACKGROUND)
    doc["operation"] = {"kind": "difference", "refs": []}
    run = _run(BACKGROUND, doc)
    assert run.verification is not None
    assert run.verification.calculation_added
    assert _cf_rows(run) == []
    assert set(run.counterfactual.refused.values()) == {"unsupported_calculation"}  # type: ignore[union-attr]
    doc = _echo(CALCULATION)
    assert doc["operation"]["kind"] == "difference"
    assert _status(_run(CALCULATION, doc)) == "verified"


# ---------------------------------------------------------------- masking


def test_mask_keeps_offsets_back_to_the_query() -> None:
    q = "این زمین مال ماست؛ حداکثر"
    text, index = mask(q, [{"start": 9, "end": 12}]) or ("", [])
    assert text == "این زمین ماست؛ حداکثر"
    assert all(q[i] == ch for ch, i in zip(text, index, strict=True))
    assert mask(q, []) == (q, list(range(len(q))))


# ---------------------------------------------------------------- replay


def _key(query: str = BACKGROUND, **settings: Any) -> ReplayKey:
    s = ProposerSettings(**settings)
    return ReplayKey.of(query, _base(query)[1], s, "engine-fixture")


def test_replay_keys_are_deterministic_and_complete() -> None:
    a, b = _key(), _key()
    assert a == b
    assert a.digest() == b.digest()
    assert len(a.digest()) == 64
    assert a.model_id == "claude-opus-5-5"
    assert a.settings == (("effort", "medium"), ("max_output_tokens", 16000), ("max_retries", 1),
                          ("structured_output", True), ("timeout_s", 120.0))  # fmt: skip
    assert a.canonical_schema_sha256 == P.canonical_schema_sha256()
    assert a.provider_schema_sha256 != a.canonical_schema_sha256
    variants = [
        _key(CLAIM),
        _key(effort="high"),
        _key(max_output_tokens=8000),
        _key(model_id="claude-sonnet-5-5"),
        ReplayKey(**{**asdict(a), "settings": a.settings, "prompt_sha256": "0" * 64}),
        ReplayKey(**{**asdict(a), "settings": a.settings, "engine": "other"}),
        ReplayKey(**{**asdict(a), "settings": a.settings, "context_sha256": "1" * 64}),
        ReplayKey(**{**asdict(a), "settings": a.settings, "provider_schema_sha256": "2" * 64}),
        ReplayKey(**{**asdict(a), "settings": a.settings, "canonical_schema_sha256": "3" * 64}),
    ]
    assert len({a.digest(), *(v.digest() for v in variants)}) == len(variants) + 1
    assert ReplayKey.from_dict(json.loads(json.dumps(a.to_dict()))) == a
    assert "temperature" not in json.dumps(a.to_dict())


def test_replay_records_round_trip_byte_identically(tmp_path: Path) -> None:
    run = _run(BACKGROUND, _echo(BACKGROUND))
    raw = json.dumps(_echo(BACKGROUND), ensure_ascii=False)
    usage = {"input_tokens": 1, "output_tokens": 2, "latency_ms": 3, "provider_model_version": "x"}
    rec = run.record(_key(), raw, "2026-10-28T00:00:00Z", usage)
    text = rec.to_json()
    assert ReplayRecord.from_json(text).to_json() == text
    assert run.record(_key(), raw, "2026-10-28T00:00:00Z", usage).to_json() == text
    tampered = json.loads(text)
    tampered["key"]["model_id"] = "claude-other"
    with pytest.raises(ValueError, match="digest"):
        ReplayRecord.from_json(json.dumps(tampered))
    store = ReplayStore(tmp_path)
    assert store.get(rec.key) is None
    assert store.missing([rec.key]) == [rec.key]
    path = store.put(rec)
    assert path.read_text(encoding="utf-8") == text
    assert store.put(rec) == path
    assert store.missing([rec.key]) == []
    stored = store.get(rec.key)
    assert stored is not None
    assert stored.to_json() == text
    changed = run.record(_key(), raw, "2026-10-29T00:00:00Z", usage)
    with pytest.raises(ReplayConflictError):
        store.put(changed)


def test_the_engine_fingerprint_is_stable() -> None:
    assert engine_fingerprint() == engine_fingerprint()
    assert len(engine_fingerprint()) == 64


# ---------------------------------------------------------------- production invariance


def _bytes(res: RetrievalResult) -> str:
    return json.dumps(asdict(res), ensure_ascii=False, sort_keys=True, default=str)


def test_production_results_are_byte_identical_with_and_without_shadow() -> None:
    queries = [*FIXTURES, PROSE, *_dev_queries(20, ("final4", "final5"))]
    clean = _new_retriever()  # never touched by the shadow machinery
    used = _new_retriever()
    rng = random.Random(SEED)
    for q in queries:
        before = used.retrieve(q)
        frozen = _bytes(before)
        ctx, facts = build(used, before)
        doc = echo_proposal(ctx, facts)
        for _ in range(2):
            _random_edit(rng, doc, ctx)
        run = shadow(used, q, doc, production=before)
        assert run.production is before
        assert _bytes(before) == frozen
        assert _bytes(used.retrieve(q)) == _bytes(clean.retrieve(q)) == frozen


def test_no_production_module_imports_the_shadow_package() -> None:
    for path in (REPO_ROOT / "src" / "mabhas19").rglob("*.py"):
        if "shadow" in path.parts:
            continue
        assert "mabhas19.shadow" not in path.read_text(encoding="utf-8"), path


def test_the_frozen_final8_engine_modules_are_unchanged() -> None:
    doc = yaml.safe_load((REPO_ROOT / "eval/retrieval/final8_candidate.yaml").read_text("utf-8"))
    frozen = doc["candidate"]["code"]["modules_sha256"]
    assert frozen == {m: candidate._sha(REPO_ROOT / m) for m in candidate.MODULES}


def test_s1_may_never_act_on_a_production_answer() -> None:
    for q in [*FIXTURES, PROSE, *_dev_queries(30, ("final2", "final3"))]:
        prod = _r().retrieve(q)
        eligible = fallback_eligible(prod)
        if prod.structured:
            assert not eligible, q
        if eligible:
            assert not prod.structured
            assert any(n.outcome for n in prod.no_match)
    assert not fallback_eligible(_base(BACKGROUND)[0])  # certified rows: never eligible
    assert fallback_eligible(_base(FOREIGN_UNIT)[0])  # typed refusal: eligible
    assert not fallback_eligible(_r().retrieve(PROSE))  # a prose route is not a failed lookup
