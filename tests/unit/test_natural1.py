"""The frozen natural-language set (``eval/clarify/natural1.yaml``) against the chat engine.

Expected behaviour was written before the wording / row-group code existed; this test only
reads it. Hybrid mode (the live configuration) is checked by the replay script; the lexical
engine used here gives the same outcomes for these questions.
"""

from __future__ import annotations

import json
from functools import cache
from typing import Any

import pytest
import yaml

from mabhas19.api import wording as W
from mabhas19.api.engine import Engine
from mabhas19.core.paths import CURATED_DIR, EVAL_DIR
from mabhas19.generation.contract import Answer
from mabhas19.indexing.chunking import build_chunks
from mabhas19.retrieval.hybrid import Retriever
from mabhas19.retrieval.structured import StructuredStore

CASES = yaml.safe_load((EVAL_DIR / "clarify" / "natural1.yaml").read_text(encoding="utf-8"))
NOTES = {r.note for r in (*W.WORDING, *W.CONCEPT_ROWS)} | {f.note for f in W.FAMILIES}


@cache
def _engine() -> Engine:
    store = StructuredStore(CURATED_DIR)
    path = CURATED_DIR / "text" / "m19-ed5-1404-main" / "blocks.jsonl"
    blocks = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    return Engine(Retriever(build_chunks(blocks, "m19-ed5-1404-main"), store), store, "lexical")


def _ask(question: str, **kw: Any) -> Answer:
    return _engine().ask(question, **kw)[0]


def check(case: dict[str, Any], a: Answer) -> list[str]:
    """Every way ``a`` departs from the frozen expectation of ``case``."""
    out: list[str] = []
    expect = case["expect"]
    rows = set(case.get("rows") or [])
    shown = {v.row_label for v in a.values}
    if a.safety.get("provenance_violations") or a.safety.get("orchestration_rejected"):
        out.append(f"safety: {a.safety.get('provenance_violations')}")
    if expect == "value":
        if a.response_type != "structured_value" or shown != rows or a.clarifications:
            out.append(f"value: {a.response_type} {sorted(shown)}")
    elif expect == "side_by_side":
        if a.response_type != "structured_value" or shown != rows or len(a.values) < 2:
            out.append(f"side_by_side: {a.response_type} {sorted(shown)} ({len(a.values)})")
    elif expect == "clarify":
        c = a.clarifications[0] if a.clarifications else None
        if a.response_type != "clarification" or c is None:
            out.append(f"clarify: {a.response_type}")
        elif c.field != case["field"] or {o.label for o in c.options} != rows:
            out.append(f"clarify: {c.field} {sorted(o.label for o in c.options)}")
    elif expect == "no_answer" and (a.values or a.clarifications):
        out.append(f"no_answer: {a.response_type} {sorted(shown)}")
    if "notice" in case:
        noted = any(n.text in NOTES for n in a.notices)
        if noted != case["notice"]:
            out.append(f"notice: {[n.text for n in a.notices]}")
    return out


@pytest.mark.parametrize("case", CASES["cases"], ids=[c["id"] for c in CASES["cases"]])
def test_frozen_natural_case(case: dict[str, Any]) -> None:
    assert check(case, _ask(case["question"])) == []


def _choose(a: Answer, label: str, field: str) -> Answer:
    """The page's follow-up: the answer's question with the chosen printed option."""
    follow = {"parent_request_id": a.request_id, "field": field, "kind": "option", "text": label}
    return _ask(f"{a.question} ({label})", follow_up=follow, base=a.question)


def test_choosing_a_roof_row_continues_to_the_climate_question() -> None:
    a = _ask("برای بام حداقل مقاومت حرارتی چقدره؟")
    b = _choose(a, "سقف مجاور فضای آزاد", "element")
    assert b.response_type == "clarification"
    assert b.clarifications[0].field == "climate_class"


def test_choosing_a_window_row_gives_its_certified_value() -> None:
    a = _ask("حداکثر ضریب انتقال حرارتی پنجره در اقلیم 3B چقدر است؟")
    label = "حداکثر ضریب انتقال حرارتی U-Value (W/m2.k) – متحرک"
    b = _choose(a, label.replace("(", " ").replace(")", " "), "component")
    assert b.response_type == "structured_value"
    assert [v.row_label for v in b.values] == [label]


@pytest.mark.parametrize(
    ("question", "label", "field"),
    [
        ("حداکثر دمای آب گرم مصرفی چقدر باید باشه؟", "ظرفشویی آشپزخانه", "place_of_use"),
        ("حداقل نسبت مساحت کفایت نور روز چقدر است؟", "آموزشی", "use"),
        ("حداقل یکنواختی روشنایی مصنوعی چقدر است؟", "کلاس‌های درس", "space"),
        ("حداقل راندمان آبگرمکن گازی چقدره؟", "آبگرمکن – گازسوز مخزن‌دار", "equipment"),
    ],
)
def test_choosing_an_offered_row_gives_that_rows_certified_value(
    question: str, label: str, field: str
) -> None:
    b = _choose(_ask(question), label, field)
    assert b.response_type == "structured_value"
    assert [v.row_label for v in b.values] == [label]


def test_a_side_by_side_band_shows_each_row_once_and_chooses_none() -> None:
    a = _ask("حداکثر ضریب بهره خورشیدی پنجره در اقلیم 2B چقدره؟")
    assert a.safety["composite"]["kind"] == "side_by_side"
    assert len({v.record_id for v in a.values}) == len(a.values) == 4
    assert len({v.row_label for v in a.values}) == 4
