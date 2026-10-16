from __future__ import annotations

import pymupdf
import pytest

from mabhas19.ingest.repair.glyphs import Cluster
from mabhas19.tables.main_audit import (
    TABLES,
    cell_flags,
    classify_value,
    grid_cells,
    image_placements,
    parse_value,
    superscripts,
)


def _c(text: str, x0: float, size: float = 10.0, baseline: float = 100.0) -> Cluster:
    return Cluster(text, 1, "BNazanin", x0, x0 + 5, baseline - size, baseline + 2, baseline, size)


def test_table_list() -> None:
    assert len(TABLES) == 11
    assert len({t.pdf_page for t in TABLES}) == 11


def test_grid_cells_spans_right_to_left() -> None:
    # two columns, header spanning both; column 0 is the rightmost one
    cells = grid_cells([(0, 0, 20, 10), (10, 10, 20, 20), (0, 10, 10, 20), (0, 20, 20, 40)])
    by_pos = {(c["row"], c["col"]): (c["rowspan"], c["colspan"]) for c in cells}
    assert by_pos == {(0, 0): (1, 2), (1, 0): (1, 1), (1, 1): (1, 1), (2, 0): (1, 2)}
    assert next(c for c in cells if c["row"] == 1 and c["col"] == 0)["bbox"][0] == 10


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("", "empty"),
        ("-", "dash"),
        ("۱/۹۶", "number"),
        ("۴۰۷۰۶۰", "number"),
        ("٪۹۲", "percent"),
        ("۷۰%", "percent"),
        ("2A , 2B", "climate_classes"),
        ("5B,5C", "climate_classes"),
        ("۳۷۷۲-۳۰-۱-۱ A2", "identifier"),
        ("۲-۷۸۱۷", "identifier"),
        ("IE4", "category"),
        ("A+++", "category"),
        ("1 < R ≤ 2", "expression"),
        ("بویلر چگالشی", "label"),
    ],
)
def test_classify_value(text: str, kind: str) -> None:
    assert classify_value(text) == kind


def test_parse_value_is_exact_and_keeps_the_token() -> None:
    p = parse_value("۰/۰۹")
    assert p is not None
    assert (p["token"], p["value"], p["decimal_separator"], p["unit"]) == (
        "۰/۰۹",
        "0.09",
        "/",
        None,
    )
    q = parse_value("٪۸۵")
    assert q is not None
    assert (q["value"], q["unit"], q["percent_sign"]) == ("85", "%", "٪")
    assert parse_value("IE4") is None
    assert parse_value("-") is None


def test_superscripts_need_small_and_raised_glyphs() -> None:
    line = [_c("ر", 50), _c("د", 44), _c("ه", 38)]
    assert superscripts([*line, _c("۱", 30, size=6, baseline=95)]) == "۱"
    assert superscripts([*line, _c("/", 30, size=6, baseline=100)]) == ""  # small, not raised


def test_cell_flags_mirrored_brackets_and_multi_line() -> None:
    assert "mirrored_brackets" in cell_flags(")فوری(", ")فوری(", [], 1)
    assert "mirrored_brackets" not in cell_flags("گرم (2A)، کم باران (2B)", "x", [], 1)
    assert "multi_line" in cell_flags("a\nb", "a\nb", [], 2)


def test_image_placements_follow_the_ctm() -> None:
    doc = pymupdf.open()
    page = doc.new_page(width=200, height=100)
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 2, 2), False)
    target = pymupdf.Rect(20, 30, 60, 50)
    page.insert_image(target, pixmap=pix)
    placed = image_placements(page)
    assert len(placed) == 1
    assert all(abs(a - b) < 0.01 for a, b in zip(placed[0][1], target, strict=True))
