"""Main-text block rules: section numbers, kinds, paragraph grouping, hierarchy, flags."""

from __future__ import annotations

import pytest

from mabhas19.ingest import main_blocks as mb


def line(text: str, *, top: float, **attrs: float | str) -> mb.Line:
    """A repaired line; attrs: x0, size, bold, latin, math (numbers) and raw (text)."""
    num = {k: float(v) for k, v in attrs.items() if k != "raw"}
    raw = str(attrs.get("raw", text))
    x0 = num.get("x0", 48.0)
    return mb.Line(
        20,
        (x0, top, 411.0, top + 14),
        raw,
        text,
        num.get("size", 12.0),
        num.get("bold", 0.0),
        num.get("math", 0.0),
        num.get("latin", 0.0),
    )


LAYOUT = mb.Layout(48.0, 411.0, 19.5)


@pytest.mark.parametrize(
    ("text", "number"),
    [("۱۹-۲-۴ رده‌بندی", "19-2-4"), ("۱۹-۵-۲-۲-۱- عایق‌کاری", "19-5-2-2-1"), ("۱۹‑۱ تعاریف", "19-1"),
     ("۲۰-۱ چیزی", None), ("۱۹ متر", None), ("بند ۱۹-۲ باشد", None)],
)  # fmt: skip
def test_section_numbers_as_printed(text: str, number: str | None) -> None:
    assert mb.section_number(text) == number


def test_kinds() -> None:
    assert mb.line_kind(line("۱۹-۲-۴ رده‌بندی", top=0, bold=0.9)) == "heading"
    assert mb.line_kind(line("۱۹-۲-۴-۱ بازدهی انرژی", top=0)) == "clause"
    assert mb.line_kind(line("1 Site Energy", top=0, size=8, latin=1)) == "footnote"
    assert mb.line_kind(line("Life Cycle Assessment (LCA)", top=0, size=10, bold=1, latin=1)) == (
        "footnote")  # fmt: skip
    assert mb.line_kind(line("جدول ۱۹-۵-۱: الزامات", top=0, size=10, bold=1)) == "table_caption"
    assert mb.line_kind(line("جدول ۱۹-۵-۱ باشد .", top=0)) == "paragraph"  # wrapped reference
    assert mb.line_kind(line("x = y", top=0, math=0.8)) == "formula"
    assert mb.line_kind(line("۱۹-۱ تعاریف ........ ۱", top=0)) == "toc_entry"
    assert mb.line_kind(line("تبصره ۱ - متن", top=0)) == "note"


def test_paragraphs_join_and_wrapped_cross_reference_stays_inside() -> None:
    lines = [line("متن اول که ادامه دارد مطابق بند", top=100),
             line("۱۹-۵-۲ باشد .", top=119.5, x0=352),  # short last line, no new clause
             line("۱۹-۵-۳ بند جدید است.", top=139)]  # fmt: skip
    blocks = mb.group_blocks(lines, LAYOUT)
    assert [b["kind"] for b in blocks] == ["paragraph", "clause"]
    assert len(blocks[0]["lines"]) == 2


def test_new_clause_after_a_sentence_end_even_on_a_full_line() -> None:
    lines = [line("۱۹-۲-۳-۱ متن کامل که با نقطه تمام می‌شود.", top=100),
             line("۱۹-۲-۳-۲ بند بعدی", top=119.5)]  # fmt: skip
    assert [b["kind"] for b in mb.group_blocks(lines, LAYOUT)] == ["clause", "clause"]


def test_quality_flags_never_fix() -> None:
    assert "spacing_suspect" in mb.quality_flags("تعاری ف ب ی ر ت", "x")
    assert mb.quality_flags("", "\u200c") == ["repair_empty"]
    assert "list_marker_paren_mirrored" in mb.quality_flags("الف( متن", "x")
    assert "list_marker_letter_missing" in mb.quality_flags("( متن", "x")
    assert "bracket_mirroring_suspect" in mb.quality_flags("مقاومت )R-Value( یک", "x")
    assert mb.quality_flags("مقاومت (R-Value) یک", "x") == []
    block = mb.finish_block({"kind": "paragraph", "lines": [line("متن", top=0, raw="نتم")]})
    assert (block["text"], block["text_raw"]) == ("متن", "نتم")  # both kept, never mixed


def test_hierarchy_path_and_sequence_problems() -> None:
    h = mb.Hierarchy()
    assert h.enter("19-2", "کلیات", "b1") == []
    assert h.enter("19-2-4", "رده‌بندی", "b2") == []
    assert [p["number"] for p in h.path()] == ["19-2", "19-2-4"]
    assert h.enter("19-3", "اقلیم", "b3") == []
    assert [p["number"] for p in h.path()] == ["19-3"]
    assert h.enter("19-2-9", "x", "b4")  # out of order: reported, not reordered


def test_toc_entry_parsing() -> None:
    assert mb.toc_entry("۱۹-۱ تعاریف ............ ۱") == {
        "section_number": "19-1",
        "title": "تعاریف",
        "toc_printed_page": "۱",
    }
