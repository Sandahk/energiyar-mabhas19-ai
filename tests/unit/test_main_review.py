"""Main-corpus review preparation: grouping and proposals (nothing is changed)."""

from __future__ import annotations

from typing import Any

from mabhas19.ingest import main_review as mr


def block(bid: str, text: str, **kw: Any) -> dict[str, Any]:
    base: dict[str, Any] = {"block_id": bid, "pdf_page": 20, "printed_page": "۱",
                            "kind": "list_item",
                            "text": text, "text_raw": text, "flags": [], "section_path": [
                                {"number": "19-2", "title": "x"}], "page_role": "body",
                            "font_size": 12.0, "bbox": [0, 0, 10, 10]}  # fmt: skip
    return {**base, **kw}


def test_mirrored_markers_split_real_markers_from_wrapped_parentheticals() -> None:
    flag = ["list_marker_paren_mirrored"]
    blocks = [block("a", "الف( متن", flags=flag), block("b", "ب( متن", flags=flag),
              block("c", "سیم( و یک پروتکل", flags=flag)]  # fmt: skip
    g = mr.mirrored_markers(blocks)
    assert {i.block["block_id"] for i in g.items if i.title == "not a list marker"} == {"c"}  # type: ignore[index]
    assert g.decisions == 2


def test_caption_numbers_proposal_keeps_19_first() -> None:
    blocks = [block("t1", "جدول ۱-۲ -۱۹: زمان‌بندی", kind="table_caption"),
              block("t2", "جدول ۲ -۱۹-۵: الزامات", kind="table_caption"),
              block("t3", "جدول ۱۹-۵-۱: حداقل", kind="table_caption")]  # fmt: skip
    g = mr.caption_numbers(blocks)
    proposals = {i.block["block_id"]: i.proposal for i in g.items}  # type: ignore[index]
    assert set(proposals) == {"t1", "t2"}
    assert "«۱۹-۲-۱»" in proposals["t1"]
    assert "«۱۹-۵-۲»" in proposals["t2"]


def test_bold_line_classes() -> None:
    assert mr.bold_class(block("a", "Life Cycle Assessment"), None) == ("english_gloss", False)
    assert mr.bold_class(block("b", "پیوست ۲: دسته‌بندی", pdf_page=146), None)[0] == "appendix_list"
    cap = block("c", "جدول ۱۹-۵-۱: حداقل", kind="table_caption")
    assert mr.bold_class(block("d", "اقلیم‌های مختلف (m2.k/W)"), cap) == (
        "caption_continuation",
        False,
    )
    assert mr.bold_class(block("e", "فرزانه صادق", page_role="front_matter"), None) == (
        "signature",
        True,
    )


def test_bracket_rule_proposal() -> None:
    g = mr.bracket_group([block("x", "حرارتی )R-Value( یک", flags=["bracket_mirroring_suspect"])])
    assert "«(R-Value)»" in g.items[0].proposal
    assert g.decisions == 1
