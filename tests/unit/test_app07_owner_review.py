"""Owner decision package: block construction and the no-preselection guarantee."""

from __future__ import annotations

from typing import Any

from mabhas19.tables import app07 as a7
from mabhas19.tables import app07_owner_review as ow


def candidates() -> list[dict[str, Any]]:
    def row(mat: str | None, y: float) -> dict[str, Any]:
        return {
            "category_raw": "۱. مصالح",
            "subgroup_raw": None,
            "material_raw": mat,
            "condition_raw": None,
            "density_raw": "۱۰۰ تا ۲۰۰",
            "conductivity_raw": "۰٫۵۰",
        }

    page = a7.PageInput(
        "d",
        8,
        "۷",
        "app07-p008.page",
        "m",
        "t",
        "test",
        [row("پشم سنگ", 0), row(None, 0), row(None, 0), row("پشم شیشه", 0)],
        expected_rows=4,
        row_ys=[200.0, 216.0, 232.0, 248.0],
    )
    return a7.normalize_page(page, a7.Context())


def test_material_runs_become_one_block_with_source_and_boundary() -> None:
    [block] = ow.material_blocks(candidates())
    assert block.rows == [
        ("app07-p008-r01", "source"),
        ("app07-p008-r02", "inherits"),
        ("app07-p008-r03", "inherits"),
        ("app07-p008-r04", "context"),
    ]
    assert block.questions[0].options == ("YES", "NO", "PARTIAL", "UNCLEAR")


def test_crop_starts_above_the_first_row_and_ends_below_the_last() -> None:
    cands = candidates()
    [block] = ow.material_blocks(cands)
    [(page, top, bottom)] = ow.crop_spans(block, cands)
    assert page == 8
    assert top == ow.TABLE_TOP  # first row on the page: headings above it are included
    assert bottom > 248.0


def test_suspicious_blocks_ask_no_numeric_questions() -> None:
    for block in ow.suspicious_blocks():
        for q in block.questions:
            assert "conductivity" not in q.text.lower()


def test_html_has_no_preselected_answer() -> None:
    cands = candidates()
    blocks = ow.material_blocks(cands)
    no_crops: list[ow.Crop] = []
    items = [(b, no_crops, [ow.row_facts(cands[0], "source")]) for b in blocks]
    page = ow.review_html(items)
    assert "checked" not in page.replace("input:checked", "")
    assert "localStorage" not in page
