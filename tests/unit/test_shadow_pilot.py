"""SH0 pilot infrastructure: schema projection, redaction, the guarded Anthropic proposer,
S1-V / S1-R metrics and the frozen pilot manifest. No test touches the network: the proposer
only ever talks to an in-process fake client, and every guard is exercised.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from functools import cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from mabhas19.core.paths import CURATED_DIR
from mabhas19.indexing.chunking import build_chunks
from mabhas19.retrieval.hybrid import Retriever
from mabhas19.retrieval.models import RetrievalResult
from mabhas19.retrieval.structured import StructuredStore
from mabhas19.shadow import pilot, replay
from mabhas19.shadow import proposal as P
from mabhas19.shadow import schema_projection as SP
from mabhas19.shadow.context import CandidateContext, DeterministicFacts, build, redact, redactions
from mabhas19.shadow.metrics import Gold, aggregate, case_metrics, s1v_check
from mabhas19.shadow.provider import (
    AnthropicProposer,
    Budget,
    BudgetExceededError,
    BudgetMeter,
    ExternalCallRefusedError,
    authorize,
)
from mabhas19.shadow.replay import (
    ProposerSettings,
    ReplayKey,
    ReplayStore,
    provider_schema,
    render_request,
    render_user,
)
from mabhas19.shadow.run import ShadowRun, echo_proposal, shadow

ENV = {"ANTHROPIC_API_KEY": "test-placeholder-not-a-key"}
BUDGET = Budget(30.0, 1_500_000)
BACKGROUND = "ما تو شیراز هستیم که اقلیمش 3B هست؛ حداکثر شدت مصرف انرژی هتل با رده C چقدره؟"
NAME_AS_IDENTITY = "حداکثر شدت مصرف انرژی مال در اقلیم 3B با رده C چقدره؟"
POSSESSIVE_WITH_ENTITY = "این زمین مال ماست؛ حداکثر شدت مصرف انرژی هتل تو اقلیم 3B با رده C چقدره؟"
FOREIGN_UNIT = "حداکثر شدت مصرف انرژی هتل در اقلیم 3B با رده C بر حسب btu چقدره؟"
PERSONAL = (
    "شماره من 09121234567 و ایمیلم a.b@example.com است؛ "
    "حداکثر شدت مصرف انرژی هتل در اقلیم 3B با رده C چقدره؟"
)
HOTEL_3B_C, MALL_3B_C = "app03-C-r20-3B", "app03-C-r27-3B"


@cache
def _r() -> Retriever:
    path = CURATED_DIR / "text" / "m19-ed5-1404-main" / "blocks.jsonl"
    blocks = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
    return Retriever(build_chunks(blocks, "m19"), StructuredStore(CURATED_DIR))


@cache
def _base(query: str) -> tuple[RetrievalResult, CandidateContext, DeterministicFacts]:
    prod = _r().retrieve(query)
    ctx, facts = build(_r(), prod)
    return prod, ctx, facts


def _echo(query: str) -> dict[str, Any]:
    return echo_proposal(*_base(query)[1:])


def _run(query: str, doc: Any) -> ShadowRun:
    return shadow(_r(), query, doc, production=_base(query)[0])


def _index(items: list[dict[str, Any]], text: str) -> int:
    return next(i for i, x in enumerate(items) if x["text"] == text)


# ---------------------------------------------------------------- schema projection


def _walk(node: Any, visit: Callable[[dict[str, Any]], None]) -> None:
    if isinstance(node, dict):
        visit(node)
        for v in node.values():
            _walk(v, visit)
    elif isinstance(node, list):
        for v in node:
            _walk(v, visit)


def test_the_provider_projection_is_deterministic_and_supported() -> None:
    canonical = P.canonical_schema()
    projected = SP.project_anthropic(canonical)
    assert projected == SP.project_anthropic(P.canonical_schema())
    assert SP.sha256(projected) == SP.sha256(provider_schema())
    assert SP.project_anthropic(projected) == projected  # idempotent
    assert canonical == P.canonical_schema()  # the canonical schema is untouched

    def supported(node: dict[str, Any]) -> None:
        assert not set(node) & SP.REMOVED, node
        assert "oneOf" not in node
        assert not isinstance(node.get("type"), list), node
        if node.get("type") == "object":
            assert node.get("additionalProperties") is False, node

    _walk(projected, supported)


def test_the_projection_keeps_the_proposal_semantics() -> None:
    canonical = P.canonical_schema()
    projected = SP.project_anthropic(canonical)
    assert SP.structure(projected) == SP.structure(canonical)
    removed = SP.removed_constraints(canonical)
    assert "$.notes:maxLength" in removed
    assert "$.values[].index:minimum" in removed
    assert all(r.split(":")[1] in SP.REMOVED for r in removed)
    # the skeleton detects a projection that would change semantics (an enum value dropped)
    broken = copy.deepcopy(projected)
    broken["properties"]["clauses"]["items"]["properties"]["role"]["enum"].pop()
    assert SP.structure(broken) != SP.structure(canonical)


def test_the_projection_refuses_what_it_cannot_express() -> None:
    with pytest.raises(SP.ProjectionError, match="patternProperties"):
        SP.project_anthropic(
            {"type": "object", "additionalProperties": False, "patternProperties": {}}
        )
    with pytest.raises(SP.ProjectionError, match="additionalProperties"):
        SP.project_anthropic({"type": "object", "properties": {}})


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda d: d["values"][0].update(index=-1), "non-negative"),  # minimum: projected away
        (lambda d: d["clauses"][0].update(start=-3), "non-negative"),
        (lambda d: d.update(notes="x" * (P.MAX_NOTES + 1)), "longer than"),  # maxLength
        (lambda d: d["values"][0].update(clause=-1), "non-negative"),  # integer|null minimum
    ],
)
def test_canonical_constraints_reject_what_the_provider_schema_permits(
    edit: Callable[[dict[str, Any]], Any], message: str
) -> None:
    doc = copy.deepcopy(_echo(BACKGROUND))
    edit(doc)
    parsed, errors = P.parse(doc)
    assert parsed is None
    assert message in errors[0], errors


def test_a_provider_schema_change_cannot_change_the_canonical_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = _base(BACKGROUND)[1]
    before = ReplayKey.of(BACKGROUND, ctx, ProposerSettings(), "e")
    monkeypatch.setattr(replay, "provider_schema", lambda provider="anthropic": {"type": "object"})
    after = ReplayKey.of(BACKGROUND, ctx, ProposerSettings(), "e")
    # recorded separately: a new provider schema makes new keys, the contract hash stays
    assert after.provider_schema_sha256 != before.provider_schema_sha256
    assert after.canonical_schema_sha256 == before.canonical_schema_sha256
    assert after.digest() != before.digest()
    # and the parser still enforces the canonical contract
    doc = copy.deepcopy(_echo(BACKGROUND))
    doc["values"][0]["role"] = "maybe"
    assert P.parse(doc)[0] is None


# ---------------------------------------------------------------- redaction


def test_personal_data_is_masked_in_place() -> None:
    text = "تماس 0912 123 4567، ایمیل x.y@mail.ir، سایت https://ex.am/p، ایستگاه 408480، سال 1404"
    found = {r.kind for r in redactions(text)}
    assert found == {"long_number", "email", "url"}
    masked = redact(text)
    assert len(masked) == len(text)
    assert "0912" not in masked
    assert "x.y@mail.ir" not in masked
    assert "408480" in masked  # a six-digit WMO code is no personal data
    assert "1404" in masked


def test_the_context_is_redacted_but_the_pipeline_is_not() -> None:
    prod, ctx, facts = _base(PERSONAL)
    assert {r.kind for r in ctx.redactions} == {"long_number", "email"}
    assert "09121234567" not in ctx.canonical_json()
    assert "a.b@example.com" not in ctx.canonical_json()
    assert facts.query == prod.normalized_query  # the verifier keeps the real query
    assert len(ctx.query) == len(facts.query)
    run = _run(PERSONAL, _echo(PERSONAL))
    assert run.verification is not None
    assert run.verification.status == "verified"
    assert run.counterfactual is not None
    assert run.counterfactual.diff["identical"]  # same result as the unredacted production
    assert run.counterfactual.query == prod.normalized_query


# ---------------------------------------------------------------- external-call guards


class _Fake:
    """An in-process stand-in for the SDK client (records every request; no network)."""

    def __init__(self, reply: Callable[[dict[str, Any]], str], model: str = "claude-opus-5-5",
                 fail: bool = False, stop: str = "end_turn") -> None:  # fmt: skip
        self.messages = self
        self.calls: list[dict[str, Any]] = []
        self.reply, self.model, self.fail, self.stop = reply, model, fail, stop

    def create(self, **kw: Any) -> Any:
        self.calls.append(kw)
        if self.fail:
            raise ConnectionError("simulated")
        return SimpleNamespace(
            content=[
                SimpleNamespace(type="thinking", thinking=""),
                SimpleNamespace(type="text", text=self.reply(kw)),
            ],
            model=self.model,
            stop_reason=self.stop,
            usage=SimpleNamespace(input_tokens=3000, output_tokens=900),
            _request_id="req_fixture",
        )

    def count_tokens(self, **kw: Any) -> Any:
        self.calls.append({"count_tokens": kw})
        return SimpleNamespace(input_tokens=2500)


def _echo_reply(kw: dict[str, Any]) -> str:
    content = kw["messages"][0]["content"]
    for q in (BACKGROUND, NAME_AS_IDENTITY, FOREIGN_UNIT):
        _, ctx, facts = _base(q)
        if render_user(ctx) == content:
            return json.dumps(echo_proposal(ctx, facts), ensure_ascii=False)
    return "{}"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"allow_external": False, "manifest_frozen": True, "budget": BUDGET, "env": ENV},
            "allow-external",
        ),
        (
            {"allow_external": True, "manifest_frozen": True, "budget": BUDGET, "env": {}},
            "credentials",
        ),
        (
            {"allow_external": True, "manifest_frozen": False, "budget": BUDGET, "env": ENV},
            "manifest",
        ),
        ({"allow_external": True, "manifest_frozen": True, "budget": None, "env": ENV}, "budget"),
        (
            {"allow_external": True, "manifest_frozen": True, "budget": Budget(0, 10), "env": ENV},
            "budget",
        ),
    ],
)
def test_no_external_call_without_every_guard(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(ExternalCallRefusedError, match=message):
        authorize(**kwargs)
    fake = _Fake(_echo_reply)
    with pytest.raises(ExternalCallRefusedError, match=message):
        AnthropicProposer(client=fake, **kwargs)
    assert fake.calls == []


def test_only_the_approved_model_can_be_requested() -> None:
    with pytest.raises(ExternalCallRefusedError, match="approved pilot model"):
        AnthropicProposer(
            ProposerSettings(model_id="claude-sonnet-5-5"),
            allow_external=True,
            manifest_frozen=True,
            budget=BUDGET,
            env=ENV,
            client=_Fake(_echo_reply),
        )


def _proposer(fake: _Fake, budget: Budget = BUDGET) -> AnthropicProposer:
    return AnthropicProposer(
        allow_external=True, manifest_frozen=True, budget=budget, env=ENV, client=fake
    )


def test_the_request_is_text_only_structured_and_without_sampling_or_fallback() -> None:
    fake = _Fake(_echo_reply)
    resp = _proposer(fake).propose(_base(BACKGROUND)[1])
    [kw] = fake.calls
    assert set(kw) == {"model", "max_tokens", "system", "messages", "output_config"}
    assert kw["model"] == "claude-opus-5-5"
    assert kw["output_config"]["effort"] == "medium"
    assert kw["output_config"]["format"]["schema"] == provider_schema()
    assert all(isinstance(m["content"], str) for m in kw["messages"])  # no image / document
    for forbidden in ("temperature", "top_p", "fallbacks", "betas", "thinking", "tools"):
        assert forbidden not in kw
    assert resp.experiment_valid
    assert resp.response_model == "claude-opus-5-5"
    assert resp.usage["request_id"] == "req_fixture"
    parsed, errors = P.parse(resp.raw_text or "")
    assert parsed is not None, errors


def test_another_served_model_invalidates_the_record() -> None:
    resp = _proposer(_Fake(_echo_reply, model="claude-opus-5")).propose(_base(BACKGROUND)[1])
    assert not resp.experiment_valid
    assert "claude-opus-5" in resp.invalid_reason


def test_failures_and_refusals_are_recorded_never_regenerated() -> None:
    fake = _Fake(_echo_reply, fail=True)
    resp = _proposer(fake).propose(_base(BACKGROUND)[1])
    assert len(fake.calls) == 1  # the SDK's own transport retry is the only retry
    assert resp.raw_text is None
    assert not resp.experiment_valid
    refused = _proposer(_Fake(_echo_reply, stop="refusal")).propose(_base(BACKGROUND)[1])
    assert refused.raw_text is None
    assert refused.stop_reason == "refusal"
    # a schema-invalid answer is a result, not a retry
    bad = _Fake(lambda kw: '{"schema": "sh0-proposal-1"}')
    resp = _proposer(bad).propose(_base(BACKGROUND)[1])
    assert len(bad.calls) == 1
    assert P.parse(resp.raw_text or "")[0] is None


def test_the_budget_meter_refuses_the_call_that_could_exceed_it() -> None:
    meter = BudgetMeter(Budget(1.0, 100_000))
    meter.admit(5000, 16000)  # worst case $0.34
    meter.add(5000, 16000)
    meter.add(5000, 16000)
    with pytest.raises(BudgetExceededError):
        meter.admit(5000, 16000)  # would reach $1.02
    fake = _Fake(_echo_reply)
    proposer = _proposer(fake, Budget(0.05, 1_000_000))
    with pytest.raises(BudgetExceededError):
        proposer.propose(_base(BACKGROUND)[1])  # 16k output tokens alone cost $0.32
    assert fake.calls == []


def test_token_counting_uses_the_exact_request() -> None:
    fake = _Fake(_echo_reply)
    assert _proposer(fake).count_tokens(_base(BACKGROUND)[1]) == 2500
    [call] = fake.calls
    sent = call["count_tokens"]
    body = _proposer(_Fake(_echo_reply)).request(_base(BACKGROUND)[1])
    assert sent == {k: body[k] for k in ("model", "system", "messages", "output_config")}


# ---------------------------------------------------------------- metrics


def _gold(rows: set[str], stratum: str, *, correct: bool = False, **labels: Any) -> Gold:
    return Gold(frozenset(rows), currently_correct=correct, stratum=stratum, **labels)


def test_a_veto_removing_an_unasked_record_is_correct() -> None:
    doc = _echo(BACKGROUND)
    doc["clauses"][0]["role"] = "background"
    m = case_metrics(_run(BACKGROUND, doc), _gold({HOTEL_3B_C}, "background_claim_request"))
    assert m["veto"] == "correct"
    assert m["would_drop_unasked_claim"]
    assert m["would_remove_rows"] == ["app02-p003-r25"]
    assert m["unsafe_new_rows"] == []


def test_a_veto_removing_an_expected_record_is_harmful() -> None:
    doc = _echo(NAME_AS_IDENTITY)
    doc["entities"][_index(doc["entities"], "مال")]["use"] = "grammatical"
    gold = _gold({MALL_3B_C}, "negative_control", correct=True, entities={0: "identity"})
    m = case_metrics(_run(NAME_AS_IDENTITY, doc), gold)
    assert m["veto"] == "harmful"
    assert m["would_mask_false_identity"]
    assert m["would_force_fail_closed"]
    assert m["preservation"]["expected_rows_removed"]
    assert m["labels"]["entity:grammatical"]["overused"] == 1
    echo = case_metrics(_run(NAME_AS_IDENTITY, _echo(NAME_AS_IDENTITY)), gold)
    assert echo["veto"] == "none"
    assert echo["preservation"]["unchanged"]
    assert echo["labels"]["entity:identity"] == {"proposed": 1, "gold": 1, "agreed": 1}


def test_a_veto_on_a_table_without_rows_is_neutral() -> None:
    doc = _echo(FOREIGN_UNIT)
    doc["requests"][0]["output"] = "ambiguous"
    m = case_metrics(_run(FOREIGN_UNIT, doc), _gold(set(), "meaningful_modifier"))
    assert m["veto"] == "neutral"
    assert m["would_remove_rows"] == []


def test_recovery_is_measured_and_attributed_but_blocked() -> None:
    doc = _echo(POSSESSIVE_WITH_ENTITY)
    doc["entities"][_index(doc["entities"], "مال")]["use"] = "grammatical"
    run = _run(POSSESSIVE_WITH_ENTITY, doc)
    m = case_metrics(run, _gold({HOTEL_3B_C}, "homograph_possessive"))
    assert m["would_add_rows"] == [HOTEL_3B_C]
    assert m["adds_rows_after_veto"]
    assert not m["adds_rows_from_positive_llm_choice"]
    assert m["counterfactual_record_exact"]
    assert run.counterfactual is not None
    assert run.counterfactual.rows == []  # blocked in SH0
    assert m["unsafe_new_rows"] == []


def test_the_first_s1v_criteria_allow_no_harmful_veto() -> None:
    doc = _echo(NAME_AS_IDENTITY)
    doc["entities"][_index(doc["entities"], "مال")]["use"] = "grammatical"
    gold = _gold({MALL_3B_C}, "negative_control", correct=True)
    report = aggregate([(case_metrics(_run(NAME_AS_IDENTITY, doc), gold), gold)])
    assert report["harmful_vetoes_on_currently_correct"] == 1
    check = s1v_check(report, {"malicious_verifier_escapes": 0, "replay_determinism": 1.0,
                               "harmful_vetoes_unseen": 0})  # fmt: skip
    assert not check["harmful_vetoes_dev"]["pass"]
    assert not s1v_check(report, {})["replay_determinism"]["pass"]  # unmeasured never passes


# ---------------------------------------------------------------- pilot manifest


def _manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(
        pilot,
        "STRATA",
        {"background_claim_request": 1, "negative_control": 1, "meaningful_modifier": 1},
    )
    cases = []
    for i, (q, stratum, rows, correct) in enumerate(
        [
            (BACKGROUND, "background_claim_request", [HOTEL_3B_C], False),
            (NAME_AS_IDENTITY, "negative_control", [MALL_3B_C], True),
            (FOREIGN_UNIT, "meaningful_modifier", [], False),
        ]
    ):
        _, ctx, _ = _base(q)
        cases.append(
            {
                "id": f"T{i}",
                "stratum": stratum,
                "query": q,
                "normalized_query": _base(q)[0].normalized_query,
                "context_sha256": ctx.sha256(),
                "expected": {"rows": rows},
                "currently_correct": correct,
                "gold": {"clauses": {}, "entities": {}, "values": {}},
            }
        )
    doc = {
        "schema": pilot.MANIFEST_SCHEMA,
        "pins": pilot.pins(),
        "settings": ProposerSettings().to_dict(),
        "cases": cases,
    }
    path = tmp_path / "pilot.yaml"
    pilot.freeze(doc, path)
    return path


def test_a_frozen_manifest_verifies_and_detects_any_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _manifest(tmp_path, monkeypatch)
    assert pilot.verify(path, _r()) == []
    with pytest.raises(FileExistsError):
        pilot.freeze(pilot.load(path), path)
    doc = pilot.load(path)
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace("T0", "T9"), encoding="utf-8")
    assert pilot.verify(path, _r()) == [f"{path.name} changed since it was frozen"]
    for change in ("engine_fingerprint", "prompt_sha256", "provider_schema_sha256"):
        d = copy.deepcopy(doc)
        d["pins"][change] = "0" * 64
        other = tmp_path / f"{change}.yaml"
        pilot.freeze(d, other)
        assert any(change in p for p in pilot.verify(other, _r()))
    d = copy.deepcopy(doc)
    d["cases"][0]["context_sha256"] = "0" * 64
    other = tmp_path / "context.yaml"
    pilot.freeze(d, other)
    assert pilot.verify(other, _r()) == ["T0: candidate context changed"]


def test_preflight_validates_every_request_locally(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _manifest(tmp_path, monkeypatch)
    result = pilot.preflight(pilot.load(path), _r())
    assert result["problems"] == {}
    assert result["cases"] == 3
    assert result["max_bytes"] > 0
    # each check fires on a request that breaks it
    _, ctx, _ = _base(BACKGROUND)
    body = render_request(ctx)
    secret = {"ANTHROPIC_API_KEY": ctx.query[:12]}
    assert any("ANTHROPIC_API_KEY" in p for p in pilot.request_problems(body, ctx, _r(), secret))
    leaked = copy.deepcopy(body)
    leaked["system"] += pilot._source_lines()[0]
    assert any("source text" in p for p in pilot.request_problems(leaked, ctx, _r(), {}))
    forged = ctx.__class__(**{**ctx.__dict__, "tables": ()})
    bad = forged.__class__(**{**forged.__dict__, "entities": tuple(
        e.__class__(**{**e.__dict__, "candidates": tuple(
            c.__class__(**{**c.__dict__, "label": c.label + " 3000"}) for c in e.candidates
        )}) for e in ctx.entities
    )})  # fmt: skip
    problems = pilot.request_problems(render_request(bad), bad, _r(), {})
    assert any("carries a number" in p for p in problems)
    assert any("not its printed label" in p for p in problems)


def test_the_pilot_run_records_once_and_replays_deterministically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _manifest(tmp_path, monkeypatch)
    store = ReplayStore(tmp_path / "replay")
    fake = _Fake(_echo_reply)
    summary = pilot.run_pilot(
        path, _r(), allow_external=True, budget=BUDGET, store=store, env=ENV, client=fake
    )
    assert summary["calls"] == 3
    assert summary["stopped"] == ""
    again = pilot.run_pilot(
        path, _r(), allow_external=True, budget=BUDGET, store=store, env=ENV, client=fake
    )
    assert again["calls"] == 0  # recorded once, never regenerated
    assert len(fake.calls) == 3
    report = pilot.report(path, _r(), store)
    assert report["missing_records"] == []
    assert report["replay_determinism"] == 1.0
    assert report["unsafe_new_rows"] == 0
    assert report["harmful_vetoes_on_currently_correct"] == 0


def test_the_pilot_run_refuses_without_guards_and_stops_on_invalid_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _manifest(tmp_path, monkeypatch)
    store = ReplayStore(tmp_path / "replay")
    fake = _Fake(_echo_reply)
    for kwargs in (
        {"allow_external": False, "budget": BUDGET, "env": ENV},
        {"allow_external": True, "budget": None, "env": ENV},
        {"allow_external": True, "budget": BUDGET, "env": {}},
    ):
        with pytest.raises(ExternalCallRefusedError):
            pilot.run_pilot(path, _r(), store=store, client=fake, **kwargs)
    unfrozen = tmp_path / "missing.yaml"
    with pytest.raises(ExternalCallRefusedError, match="manifest"):
        pilot.run_pilot(unfrozen, _r(), allow_external=True, budget=BUDGET, store=store,
                        env=ENV, client=fake)  # fmt: skip
    assert fake.calls == []
    monkeypatch.setattr(pilot, "INVALID_ABORT_AFTER", 2)
    bad = _Fake(lambda kw: "not json")
    summary = pilot.run_pilot(
        path, _r(), allow_external=True, budget=BUDGET, store=store, env=ENV, client=bad
    )
    assert summary["calls"] == 2
    assert "invalid outputs" in summary["stopped"]


def test_the_cli_refuses_without_a_frozen_manifest(tmp_path: Path) -> None:
    assert pilot.main(["run", "--manifest", str(tmp_path / "none.yaml"), "--allow-external",
                       "--max-usd", "30", "--max-tokens", "1000"]) == 2  # fmt: skip
