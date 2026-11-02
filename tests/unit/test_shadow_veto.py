"""Candidate S1-V policy ``s1v-grammatical-veto-1``: its single lever (a verifier-accepted
``grammatical`` label on a grounded span), every non-authoritative field, the subset guarantee,
the veto-benchmark metrics and the veto manifest in the pilot harness. No network: provider
calls go to an in-process fake.
"""

from __future__ import annotations

import copy
import json
from datetime import date
from functools import cache
from json import dumps
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from mabhas19.core.paths import CURATED_DIR, EVAL_DIR
from mabhas19.indexing.chunking import build_chunks
from mabhas19.retrieval.hybrid import Retriever
from mabhas19.retrieval.models import RetrievalResult
from mabhas19.retrieval.structured import StructuredStore
from mabhas19.shadow import pilot
from mabhas19.shadow import provider as PV
from mabhas19.shadow import veto as V
from mabhas19.shadow.context import CandidateContext, DeterministicFacts, build
from mabhas19.shadow.provider import Budget
from mabhas19.shadow.replay import ProposerSettings, ReplayStore, user_turn
from mabhas19.shadow.run import ShadowRun, echo_proposal, shadow

NAME_AS_IDENTITY = "حداکثر شدت مصرف انرژی مال در اقلیم 3B با رده C چقدره؟"
NESTED = "حداکثر شدت مصرف انرژی کلینیک تشخیصی در اقلیم 3B با رده C چقدره؟"
POSSESSIVE = "این زمین مال ماست؛ حداکثر شدت مصرف انرژی هتل تو اقلیم 3B با رده C چقدره؟"
BACKGROUND = "ما تو شیراز هستیم که اقلیمش 3B هست؛ حداکثر شدت مصرف انرژی هتل با رده C چقدره؟"
MALL_3B_C = "app03-C-r27-3B"
GENV = {"GEMINI_API_KEY": "test-placeholder-not-a-key"}


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


def _proposal(query: str) -> dict[str, Any]:
    return copy.deepcopy(echo_proposal(*_base(query)[1:]))


def _run(query: str, doc: dict[str, Any]) -> ShadowRun:
    return shadow(_r(), query, doc, production=_base(query)[0])


def _entity(doc: dict[str, Any], text: str) -> dict[str, Any]:
    return next(e for e in doc["entities"] if e["text"] == text)


def _prod_rows(query: str) -> list[str]:
    return [h.record_id for h in _base(query)[0].structured]


# ---------------------------------------------------------------- the single lever


def test_an_accepted_grammatical_label_on_a_grounded_span_masks_it() -> None:
    assert _prod_rows(NAME_AS_IDENTITY) == [MALL_3B_C]
    doc = _proposal(NAME_AS_IDENTITY)
    _entity(doc, "مال")["use"], _entity(doc, "مال")["candidate"] = "grammatical", "none"
    d = V.decide(_r(), _run(NAME_AS_IDENTITY, doc))
    assert (d.policy, d.reason, d.applied) == ("s1v-grammatical-veto-1", "applied", True)
    assert [m["text"] for m in d.masked] == ["مال"]
    assert d.removed == [MALL_3B_C]
    assert set(d.rows) <= set(_prod_rows(NAME_AS_IDENTITY))
    assert d.counterfactual is not None
    assert d.counterfactual.certification_failures == 0


def test_the_echo_proposal_never_vetoes() -> None:
    for q in (NAME_AS_IDENTITY, NESTED, BACKGROUND):
        d = V.decide(_r(), _run(q, _proposal(q)))
        assert (d.applied, d.removed, d.rows) == (False, [], _prod_rows(q))


@pytest.mark.parametrize(
    ("change", "what"),
    [
        (lambda d: _entity(d, "مال").update(use="contextual", candidate="none"), "contextual"),
        (lambda d: _entity(d, "مال").update(use="ambiguous", candidate="none"), "ambiguous"),
        (lambda d: _entity(d, "مال").update(candidate="ambiguous"), "ambiguous candidate"),
        (lambda d: _entity(d, "مال").update(candidate="none"), "identity, no candidate"),
        (lambda d: [r.update(intent="verify") for r in d["requests"]], "request intent"),
        (lambda d: [r.update(output="ambiguous") for r in d["requests"]], "output"),
        (lambda d: [r.update(unit="ambiguous") for r in d["requests"]], "unit"),
        (lambda d: [c.update(role="background") for c in d["clauses"]], "clause background"),
        (lambda d: [v.update(role="background") for v in d["values"]], "value background"),
        (lambda d: d["operation"].update(kind="ambiguous"), "operation"),
    ],
)  # fmt: skip
def test_no_other_field_has_veto_authority(change: Any, what: str) -> None:
    doc = _proposal(NAME_AS_IDENTITY)
    change(doc)
    run = _run(NAME_AS_IDENTITY, doc)
    d = V.decide(_r(), run)
    assert not d.applied, what
    assert d.removed == [], what
    assert d.rows == _prod_rows(NAME_AS_IDENTITY), what


def test_a_grammatical_label_off_the_grounding_does_not_act() -> None:
    _, ctx, facts = _base(POSSESSIVE)
    grounded = set(V.grounded_entities(ctx, facts))
    off = [e for e in ctx.entities if e.index not in grounded]
    assert off, "the fixture needs an ungrounded entity span"
    doc = _proposal(POSSESSIVE)
    for e in doc["entities"]:
        if e["index"] == off[0].index:
            e["use"], e["candidate"] = "grammatical", "none"
    d = V.decide(_r(), _run(POSSESSIVE, doc))
    assert (d.reason, d.removed) == ("not_eligible", [])


def test_a_self_contradicting_nested_label_does_not_act() -> None:
    doc = _proposal(NESTED)
    inner = [e for e in doc["entities"] if e["text"] in ("کلینیک", "تشخیصی")]
    assert inner
    assert any(e["use"] == "identity" for e in doc["entities"])
    for e in inner:
        e["use"], e["candidate"] = "grammatical", "none"
    d = V.decide(_r(), _run(NESTED, doc))
    assert (d.reason, d.removed) == ("not_eligible", [])


def test_a_fabricating_proposal_is_not_used_at_all() -> None:
    doc = _proposal(NAME_AS_IDENTITY)
    e = _entity(doc, "مال")
    e["use"], e["candidate"], e["end"] = "grammatical", "none", e["end"] + 1  # invented offsets
    d = V.decide(_r(), _run(NAME_AS_IDENTITY, doc))
    assert (d.reason, d.applied, d.removed) == ("integrity", False, [])


def test_no_proposal_means_production_stands() -> None:
    d = V.decide(_r(), _run(NAME_AS_IDENTITY, None))  # type: ignore[arg-type]
    assert (d.reason, d.rows) == ("no_proposal", _prod_rows(NAME_AS_IDENTITY))


# ---------------------------------------------------------------- metrics


def _gold(group: str, rows: dict[str, str], opportunity: bool = False) -> V.VetoGold:
    return V.VetoGold(group=group, rows=rows, spans={}, opportunity=opportunity)


def test_vetoes_are_classified_by_the_rows_they_remove() -> None:
    g = _gold("A_grammatical_misread", {"r1": "wrong", "r2": "correct", "r3": "uncertain"})
    assert V.classify([], False, g) == "none"
    assert V.classify([], True, g) == "neutral"
    assert V.classify(["r1"], True, g) == "correct"
    assert V.classify(["r1", "r2"], True, g) == "harmful"
    assert V.classify(["r3"], True, g) == "uncertain"
    assert V.classify(["r1", "r3"], True, g) == "uncertain"
    assert V.classify(["unlabelled"], True, g) == "uncertain"


def test_the_aggregate_separates_recall_precision_and_harm() -> None:
    base = {"schema_valid": True, "status": "verified", "rows_outside_production": [],
            "guard_refused": False, "certification_failures": 0}  # fmt: skip
    cases = [
        {**base, "group": "A_grammatical_misread", "opportunity": True, "veto": "correct",
         "production_correct": False, "spans": [("grammatical", "grammatical")]},
        {**base, "group": "A_grammatical_misread", "opportunity": True, "veto": "none",
         "production_correct": False, "spans": [("grammatical", "identity")]},
        {**base, "group": "B_identity_lookalike", "opportunity": False, "veto": "harmful",
         "production_correct": True, "spans": [("identity", "grammatical")]},
        {**base, "group": "C_plain_identity", "opportunity": False, "veto": "none",
         "production_correct": True, "spans": [("identity", "identity")]},
        {**base, "group": "D_ambiguous", "opportunity": False, "veto": "uncertain",
         "production_correct": False, "spans": [("ambiguous", "grammatical")]},
    ]  # fmt: skip
    out = V.aggregate(cases)
    assert (out["opportunities"], out["opportunity_correct"], out["opportunity_missed"]) == (
        2,
        1,
        1,
    )
    assert out["opportunity_recall"] == 0.5
    assert out["acting_vetoes"] == 3  # correct + harmful + uncertain (neutral does not act)
    assert out["veto_precision"] == pytest.approx(1 / 3)
    assert (out["harmful_on_controls"], out["controls"]) == (1, 2)
    assert (out["harmful_on_correct_production"], out["correct_production_cases"]) == (1, 2)
    assert out["span_accuracy"]["grammatical"]["accuracy"] == 0.5
    assert out["span_accuracy"]["identity"]["grammatical"] == 1
    check = V.criteria(out, {"replay_determinism": 1.0})
    assert not check["harmful_vetoes"]["pass"]
    assert not check["veto_precision"]["pass"]
    assert check["opportunity_recall"]["pass"]


def test_gold_derived_span_effects_and_oracle_rows() -> None:
    prod, ctx, facts = _base(NAME_AS_IDENTITY)
    mall = next(e.index for e in ctx.entities if e.text == "مال")
    effect, removed = V.span_effect(_r(), ctx, facts, prod, mall, {MALL_3B_C: "correct"})
    assert (effect, removed) == ("removes_correct", [MALL_3B_C])
    effect, _ = V.span_effect(_r(), ctx, facts, prod, mall, {MALL_3B_C: "wrong"})
    assert effect == "removes_wrong"
    assert V.oracle_rows(_r(), ctx, facts, prod, {mall: "identity"}) == [MALL_3B_C]
    assert V.oracle_rows(_r(), ctx, facts, prod, {mall: "grammatical"}) == []


# ---------------------------------------------------------------- veto manifest in the harness


def _veto_manifest(tmp_path: Path) -> tuple[Path, Path]:
    cases = []
    for i, (q, group, rows) in enumerate(
        [
            (NAME_AS_IDENTITY, "A_grammatical_misread", {MALL_3B_C: "wrong"}),
            (NESTED, "C_plain_identity", dict.fromkeys(_prod_rows(NESTED), "correct")),
        ]
    ):
        prod, ctx, _ = _base(q)
        spans = {e.index: "identity" for e in ctx.entities}
        if group.startswith("A"):
            spans = {next(e.index for e in ctx.entities if e.text == "مال"): "grammatical"}
        cases.append({
            "id": f"V{i}", "stratum": group, "query": q,
            "normalized_query": prod.normalized_query, "context_sha256": ctx.sha256(),
            "production_rows": [h.record_id for h in prod.structured],
            "gold": {"rows": rows, "spans": spans, "opportunity": group.startswith("A")},
        })  # fmt: skip
    base = tmp_path / "veto.yaml"
    pilot.freeze(
        {
            "schema": pilot.VETO_MANIFEST_SCHEMA,
            "pins": pilot.pins(),
            "settings": ProposerSettings().to_dict(),
            "strata": {"A_grammatical_misread": 1, "C_plain_identity": 1},
            "cases": cases,
        },
        base,
    )
    path = tmp_path / "veto_gemini.yaml"
    pilot.freeze(pilot.provider_manifest(base, ProposerSettings.gemini()), path)
    return base, path


def test_a_veto_manifest_verifies_with_its_own_groups_and_production_rows(
    tmp_path: Path,
) -> None:
    base, path = _veto_manifest(tmp_path)
    assert pilot.verify(base, _r()) == []
    assert pilot.verify(path, _r()) == []
    doc = pilot.load(path)
    assert doc["base_schema"] == pilot.VETO_MANIFEST_SCHEMA
    assert doc["strata"] == {"A_grammatical_misread": 1, "C_plain_identity": 1}
    raw = yaml.safe_load(base.read_text(encoding="utf-8"))
    raw["cases"][0]["production_rows"] = ["app03-C-r20-3B"]
    pilot.freeze(raw, tmp_path / "moved.yaml")
    assert pilot.verify(tmp_path / "moved.yaml", _r()) == ["V0: production rows changed"]


class _Fake:
    """Answers every generateContent with the deterministic reading, but «مال» grammatical."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def post(self, url: str, *, json: dict[str, Any], headers: dict[str, str],
             timeout: float) -> Any:  # fmt: skip
        self.calls.append(url)
        user = user_turn(json)
        query = next(q for q in (NAME_AS_IDENTITY, NESTED) if _base(q)[1].query in user)
        doc = _proposal(query)
        for e in doc["entities"]:
            if e["text"] == "مال":
                e["use"], e["candidate"] = "grammatical", "none"
        payload = {
            "candidates": [{"content": {"parts": [{"text": dumps(doc)}]},
                            "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 1000, "candidatesTokenCount": 300,
                              "thoughtsTokenCount": 200},
            "modelVersion": "gemini-3.8-flash",
        }  # fmt: skip
        return SimpleNamespace(status_code=200, json=lambda: payload)


def test_the_veto_report_scores_v1_from_replay_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(PV.PRICE_VALID_UNTIL, "gemini-3.8-flash", date(9999, 1, 1))
    _, path = _veto_manifest(tmp_path)
    store = ReplayStore(tmp_path / "replay")
    fake = _Fake()
    summary = pilot.run_pilot(path, _r(), allow_external=True, budget=Budget(3.0, 1_200_000),
                              store=store, env=GENV, client=fake)  # fmt: skip
    assert summary["calls"] == 2
    out = pilot.report(path, _r(), store)
    assert len(fake.calls) == 2  # the report made no call
    assert out["policy"] == "s1v-grammatical-veto-1"
    assert out["replay_determinism"] == 1.0
    assert (out["opportunities"], out["opportunity_correct"]) == (1, 1)
    assert out["harmful_vetoes"] == 0
    assert out["rows_outside_production"] == 0
    assert out["veto_by_group"]["A_grammatical_misread"] == {"correct": 1}


# ---------------------------------------------------------------- the frozen benchmark


def test_the_argon_manifest_differs_from_flash_only_in_the_model_contract() -> None:
    flash_path = EVAL_DIR / "shadow" / "veto1_gemini.yaml"
    assert pilot.file_sha256(flash_path) == pilot.sha_path(flash_path).read_text().split()[0]
    flash = yaml.safe_load(flash_path.read_text(encoding="utf-8"))
    argon = yaml.safe_load(pilot.MANIFEST.read_text(encoding="utf-8"))
    assert argon["base"] == flash["base"]
    same_pins = set(flash["pins"]) - {"engine_fingerprint"}
    assert {k: argon["pins"][k] for k in same_pins} == {k: flash["pins"][k] for k in same_pins}
    changed = {k for k, v in flash["settings"].items() if argon["settings"][k] != v}
    assert changed == {"model_id", "timeout_s"}
    assert argon["settings"]["timeout_s"] == 1200.0


def test_the_frozen_veto_benchmark_is_unseen_and_structurally_sound() -> None:
    base = EVAL_DIR / "shadow" / "veto1.yaml"
    assert pilot.file_sha256(base) == pilot.sha_path(base).read_text().split()[0]
    doc = pilot.load(pilot.MANIFEST)
    assert doc["base"]["sha256"] == pilot.file_sha256(base)
    assert doc["base_schema"] == pilot.VETO_MANIFEST_SCHEMA
    assert doc["settings"]["model_id"] == "gemini-4-argon"
    assert doc["settings"]["effort"] == "medium"
    cases = doc["cases"]
    assert len(cases) >= 120
    assert len({c["id"] for c in cases}) == len(cases)
    assert len({c["normalized_query"] for c in cases}) == len(cases)
    seen = {c["normalized_query"] for c in pilot.load(pilot.BASE_MANIFEST)["cases"]}
    assert not seen & {c["normalized_query"] for c in cases}
    for c in cases:
        g = V.gold_of(c)
        assert g.group in V.GROUPS
        assert set(g.rows) == set(c["production_rows"])
        assert set(c["expected_v1_rows"]) <= set(c["production_rows"])
        assert set(g.spans.values()) <= set(V.SPAN_USES)
        assert set(g.rows.values()) <= set(V.ROW_LABELS)
        if g.group == "A_grammatical_misread":
            assert g.opportunity
            assert "grammatical" in g.spans.values()
            assert c["expected_v1_rows"] != c["production_rows"]
        elif g.group in V.CONTROLS:
            assert g.production_correct
            assert set(g.spans.values()) == {"identity"}
            assert c["expected_v1_rows"] == c["production_rows"]
        else:
            assert "ambiguous" in g.spans.values()
