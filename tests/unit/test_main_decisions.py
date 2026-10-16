"""Main-text approved rules and review decisions: guarded, recorded, raw text untouched."""

from __future__ import annotations

import hashlib
from typing import Any

import pytest

from mabhas19.ingest import main_blocks as mb
from mabhas19.ingest import main_decisions as md


def blk(bid: str, text: str, kind: str = "paragraph", **kw: Any) -> dict[str, Any]:
    base: dict[str, Any] = {"block_id": bid, "doc_id": "d", "pdf_page": 20, "printed_page": "۱",
                            "printed_page_confidence": "high", "page_role": "body", "kind": kind,
                            "section_path": [{"number": "19-2", "title": "x"}], "text": text,
                            "text_raw": "raw", "text_repair_v1": text, "flags": [],
                            "bbox": [0, 0, 10, 10], "applied_rules": []}  # fmt: skip
    return {**base, **kw}


def test_rules_m1_b1_c1_k1() -> None:
    b = mb.apply_text_rules(blk("a", "الف( متن )R-Value( ادامه", "list_item"))
    assert b["text"] == "الف) متن (R-Value) ادامه"
    assert {r["rule"] for r in b["applied_rules"]} == {"M1", "B1"}
    e = mb.apply_text_rules(blk("b", "سیم( و یک پروتکل", "list_item"))
    assert (e["kind"], e["text"]) == ("paragraph", "سیم( و یک پروتکل")
    assert e["continues_previous_block"] is True
    c = mb.apply_text_rules(blk("c", "جدول ۲ -۱۹-۵: الزامات", "table_caption"))
    assert c["text"] == "جدول ۱۹-۵-۲: الزامات"
    k = mb.apply_text_rules(blk("d", "ر( تجهیزات و .....، هستند", "toc_entry"), "body")
    assert k["kind"] == "list_item"
    assert any(r["rule"] == "K1" for r in k["applied_rules"])


def _sha(t: str) -> str:
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


def test_replace_is_guarded_and_keeps_layers() -> None:
    b = blk("x", "این اس ت که")
    d = {"id": "S-x", "block": "x", "decision": "replace", "expect_text_sha256": _sha(b["text"]),
         "replacements": [{"from": "اس ت", "to": "است", "count": 1}]}  # fmt: skip
    [out] = md.apply_decisions([b], {"decisions": [d]})
    assert out["text"] == "این است که"
    assert (out["text_raw"], out["text_before_review"]) == ("raw", "این اس ت که")
    assert out["verification"]["status"] == "assistant_reviewed"
    with pytest.raises(md.DecisionError, match="stale"):
        md.apply_decisions([blk("x", "changed")], {"decisions": [d]})
    bad = {**d, "replacements": [{"from": "اس ت", "to": "است", "count": 2}]}
    with pytest.raises(md.DecisionError, match="expected 2"):
        md.apply_decisions([b], {"decisions": [bad]})


def test_exclusion_merge_and_classification() -> None:
    ocr = blk("o", "ocr text", "ocr_text")
    parts = [blk("f1", "𝑅 =", "formula"), blk("f2", "E1", "formula"), blk("f3", "E2", "formula")]
    bold = blk("h", "فرزانه صادق", "heading_unnumbered", flags=["unnumbered_bold_line"])
    doc = {"decisions": [
        {"id": "O", "block": "o", "decision": "exclude_from_retrieval", "reason": "notice"},
        {"id": "F", "decision": "merge_formula", "blocks": ["f1", "f2", "f3"], "new_block": "fm",
         "formula": {"plain": "R = E1 / E2", "latex": "R"},
         "expect_texts": {"f1": "𝑅 =", "f2": "E1", "f3": "E2"}},
        {"id": "U", "decision": "classify", "classes": {"h": "signature_name"}},
    ]}  # fmt: skip
    out = {b["block_id"]: b for b in md.apply_decisions([ocr, *parts, bold], doc)}
    assert out["o"]["retrieval"] == {"include": False, "reason": "notice"}
    assert out["fm"]["text"] == "R = E1 / E2"
    assert out["fm"]["fragments"] == ["f1", "f2", "f3"]
    assert out["f1"]["kind"] == "formula_fragment"
    assert out["f1"]["retrieval"]["include"] is False
    assert out["h"]["structural_class"] == "signature_name"
    assert out["h"]["open_flags"] == []
    assert out["h"]["resolved_flags"] == ["unnumbered_bold_line"]


def test_unknown_block_fails() -> None:
    d = {"id": "U", "decision": "classify", "classes": {"nope": "x"}}
    with pytest.raises(md.DecisionError, match="unknown"):
        md.apply_decisions([blk("a", "x")], {"decisions": [d]})
