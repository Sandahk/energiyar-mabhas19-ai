from __future__ import annotations

import hashlib
from collections import Counter
from typing import Any

import pytest

from mabhas19.ingest.main_corpus import refresh_section_titles
from mabhas19.ingest.main_decisions import DecisionError, _apply_errata, _apply_reclassification
from mabhas19.ingest.main_errata import Finding, errata_decisions, replacements, space_compounds


def _sha(t: str) -> str:
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


def _entry(text: str, reps: list[dict[str, Any]], result: str) -> dict[str, Any]:
    return {
        "id": "ER-x",
        "block": "b",
        "expect_text_sha256": _sha(text),
        "result_text_sha256": _sha(result),
        "replacements": reps,
    }


def test_brackets_are_applied_before_word_joins() -> None:
    text = "متن )تعری ف فوق( است"
    bracket = Finding("mirrored_brackets", ")تعری ف فوق(", "(تعری ف فوق)", "", rule="B2")
    found = [Finding("split_word", "تعری ف", "تعریف", ""), bracket]
    out, reps = replacements(text, found)
    assert out == "متن (تعریف فوق) است"
    assert [r["rule"] for r in reps] == ["B2", None]


def test_apply_errata_keeps_provenance_and_status_inputs() -> None:
    text = "کاملا ً )دبی( و ع ملیات"
    result = "کاملاً (دبی) و عملیات"
    reps = [
        {"from": ")دبی(", "to": "(دبی)", "count": 1, "rule": "B2", "candidates": ["E1"]},
        {"from": "کاملا ً", "to": "کاملاً", "count": 1, "rule": "T1", "candidates": ["E2"]},
        {"from": "ع ملیات", "to": "عملیات", "count": 1, "rule": None, "candidates": ["E3"]},
    ]
    block: dict[str, Any] = {"block_id": "b", "text": text, "applied_rules": []}
    _apply_errata(block, _entry(text, reps, result))
    assert (block["text_before_errata"], block["text"]) == (text, result)
    assert [r["rule"] for r in block["applied_rules"]] == ["B2", "T1"]
    assert [d["id"] for d in block["decisions"]] == ["ER-x"]  # per-block case is reviewed
    assert block["errata"] == {"id": "ER-x", "candidates": ["E1", "E2", "E3"]}


def test_rule_only_errata_add_no_review_decision() -> None:
    text = "متن )دبی("
    reps = [{"from": ")دبی(", "to": "(دبی)", "count": 1, "rule": "B2", "candidates": ["E1"]}]
    block: dict[str, Any] = {"block_id": "b", "text": text}
    _apply_errata(block, _entry(text, reps, "متن (دبی)"))
    assert "decisions" not in block


def test_apply_errata_is_guarded() -> None:
    reps = [{"from": "ع ملیات", "to": "عملیات", "count": 1, "rule": None, "candidates": []}]
    with pytest.raises(DecisionError):
        _apply_errata({"block_id": "b", "text": "تغییر کرده"}, _entry("ع ملیات", reps, "عملیات"))


def test_reclassify_links_the_caption() -> None:
    text = "اقلیم‌های مختلف (m2.k/W)"
    entry = _entry(text, [], text) | {
        "reclassify": {
            "kind": "caption_continuation",
            "caption_block": "main-p084-b003",
            "candidates": ["E9"],
        }
    }
    block: dict[str, Any] = {"block_id": "b", "text": text, "kind": "heading_unnumbered"}
    _apply_errata(block, entry)
    assert (block["kind"], block["caption_block"], block["text"]) == (
        "caption_continuation",
        "main-p084-b003",
        text,
    )
    assert "text_before_errata" not in block


def test_errata_decisions_pin_text_and_result() -> None:
    blocks = {"main-b": {"text": "بر اس اس و اس اس"}}
    cands: list[dict[str, Any]] = [
        {
            "verdict": "confirmed",
            "block_id": "main-b",
            "defect": "split_word",
            "found": "اس اس",
            "proposed": "اساس",
            "rule": None,
            "candidate_id": "E1",
            "extra": {},
        }
    ]
    (d,) = errata_decisions(cands, blocks)
    assert d["replacements"][0]["count"] == 2
    assert d["result_text_sha256"] == _sha("بر اساس و اساس")


def test_section_titles_follow_the_corrected_heading() -> None:
    path = [{"number": "19-1-94", "title": "قدیمی )x("}]
    heading = {
        "kind": "heading",
        "section_number": "19-1-94",
        "text": "۱۹-۱-۹۴ جدید (x)",
        "section_path": path,
    }
    body = {
        "kind": "paragraph",
        "section_number": None,
        "text": "متن",
        "section_path": [dict(path[0])],
    }
    blocks: list[dict[str, Any]] = [heading, body]
    refresh_section_titles(blocks)
    assert [b["section_path"][0]["title"] for b in blocks] == ["جدید (x)", "جدید (x)"]
    assert blocks[1]["section_path"][0]["number"] == "19-1-94"


def test_space_compounds_lists_printed_spacing() -> None:
    freq = Counter({"شبیه": 5, "سازی": 5, "شبیه\u200cسازی": 9})
    assert space_compounds("شبیه سازی", freq) == Counter({"شبیه سازی": 1})


def _relabel(text: str, **kw: Any) -> dict[str, Any]:
    return {
        "id": "DB-x",
        "block": "b",
        "from_kind": "footnote",
        "to_kind": "paragraph",
        "structural_class": "definition_body",
        "basis": "assistant_reviewed_structure",
        "expect_text_sha256": _sha(text),
        **kw,
    }


def test_reclassification_changes_kind_only() -> None:
    text = "Thermal Comfort\nوضعیتی است"
    block: dict[str, Any] = {
        "block_id": "b",
        "kind": "footnote",
        "text": text,
        "verification": {"status": "machine_extracted", "decisions": []},
    }
    _apply_reclassification(block, _relabel(text))
    assert (block["kind"], block["structural_class"], block["text"]) == (
        "paragraph",
        "definition_body",
        text,
    )
    assert block["verification"] == {"status": "machine_extracted", "decisions": []}
    assert block["reclassification"][0]["from_kind"] == "footnote"


def test_reclassification_is_guarded() -> None:
    with pytest.raises(DecisionError):
        _apply_reclassification({"block_id": "b", "kind": "paragraph", "text": "x"}, _relabel("x"))
    with pytest.raises(DecisionError):
        _apply_reclassification({"block_id": "b", "kind": "footnote", "text": "y"}, _relabel("x"))
