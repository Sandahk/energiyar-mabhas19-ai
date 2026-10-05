"""Main-PDF text repair: glyph evidence, visual->logical order, representation separation.

Synthetic lines mimic the main PDF: glyph clusters in visual (left-to-right) positions,
space glyphs mapped to U+200C, kashida glyphs mapped to «ل», lam-alef as one cluster.
"""

from __future__ import annotations

from collections import Counter

from mabhas19.ingest.repair.bidi import VisualItem, visual_to_logical
from mabhas19.ingest.repair.glyphs import Cluster, is_space_glyph
from mabhas19.ingest.repair.pipeline import FinalFormStats, RepairOptions, repair_line

ZWNJ = "\u200c"
FONT = "BNazanin"
KASHIDA_GID = 47
KASHIDA = frozenset({(FONT, KASHIDA_GID)})


def rtl_line(
    tokens: list[tuple[str, int] | str], font: str = FONT, start: float = 400.0, width: float = 5.0
) -> list[Cluster]:
    """Clusters for a right-to-left line: tokens in logical order, placed leftwards."""
    out = []
    x = start
    for tok in tokens:
        text, gid = (tok, 100) if isinstance(tok, str) else tok
        w = 3.0 if text == ZWNJ else width
        out.append(Cluster(text, gid, font, x - w, x, 90.0, 102.0, 100.0, 12.0))
        x -= w
    return out


def ltr_run(text: str, x0: float, font: str = FONT) -> list[Cluster]:
    return [Cluster(c, 50, font, x0 + 5 * i, x0 + 5 * i + 5, 90, 102, 100, 12)
            for i, c in enumerate(text)]  # fmt: skip


def repair(line: list[Cluster], **opts: bool) -> str:
    return repair_line(line, KASHIDA, None, RepairOptions(infer_zwnj=False, **opts))


def test_space_glyph_mapped_to_zwnj_becomes_space() -> None:
    line = rtl_line(["ا", "ب", (ZWNJ, 3), "پ"])
    assert is_space_glyph(line[2])
    assert repair(line) == "اب پ"
    assert repair(line, space_glyphs=False) == f"اب{ZWNJ}پ"


def test_kashida_glyph_is_dropped_without_creating_a_space() -> None:
    # «شـــدت» stored as ش + kashida + kashida + د + ت
    line = rtl_line(["ش", ("ل", KASHIDA_GID), ("ل", KASHIDA_GID), "د", "ت"])
    assert repair(line) == "شدت"
    assert repair(line, drop_kashida=False) == "شللدت"


def test_genuine_lam_is_kept() -> None:
    line = rtl_line(["ا", ("ل", 184), "گ", "و"])  # «الگو»: a real «ال»
    assert repair(line) == "الگو"


def test_lam_alef_ligature_cluster_keeps_logical_order() -> None:
    line = rtl_line(["ک", ("لا", 212), "ی"])  # «کلای»
    assert repair(line) == "کلای"
    assert repair(line, keep_ligature_clusters=False) == "کالی"  # the raw-extraction bug


def test_large_gap_inserts_a_word_space() -> None:
    a = rtl_line(["م", "ن"], start=400)
    b = rtl_line(["ت", "و"], start=380)  # 10pt gap > 0.3 * 12pt
    assert repair(a + b) == "من تو"
    assert repair(a + b, gap_spaces=False) == "منتو"


def test_overprinted_duplicate_glyphs_are_dropped() -> None:
    line = rtl_line(["ب", "د"])
    assert repair([*line, line[0]]) == "بد"


def test_visually_reversed_section_number_is_reconstructed() -> None:
    # «۱۹-۲-۴-۱ متن»: the number is stored left to right as 1-4-2-19.
    number = ltr_run("1-4-2-19", 300)
    text = rtl_line(["م", "ت", "ن"], start=290)
    assert repair(text + number) == "۱۹-۲-۴-۱ متن"
    assert repair(text + number, section_numbers=False) == "۱-۴-۲-۱۹ متن"


def test_digits_follow_the_font_that_draws_them() -> None:
    persian = ltr_run("55", 300, font="BNazanin")
    latin = ltr_run("2002", 200, font="TimesNewRomanPSMT")
    assert repair(persian) == "۵۵"
    assert repair(latin) == "2002"


def test_brackets_around_latin_text_in_rtl_line() -> None:
    items = [VisualItem(c, 10.0 * i, 10.0 * i + 9) for i, c in enumerate("ب (MJ) ا")]
    assert visual_to_logical(items) == "ا (MJ) ب"


def test_purely_latin_line_keeps_visual_order() -> None:
    items = [VisualItem(c, 10.0 * i, 10.0 * i + 9) for i, c in enumerate("Site (Final)")]
    assert visual_to_logical(items) == "Site (Final)"


def test_zwnj_inferred_after_final_form_glyph() -> None:
    # Glyph 189 («ن» final form) is almost always followed by a word break; when it is
    # directly followed by a letter, the source must have had a ZWNJ (ساختمان‌ها).
    stats = FinalFormStats()
    for _ in range(10):
        stats.end[(FONT, 189)] += 1
        stats.total[(FONT, 189)] += 1
    line = rtl_line(["م", "ا", ("ن", 189), "ه", "ا"])
    assert repair_line(line, KASHIDA, stats, RepairOptions()) == f"مان{ZWNJ}ها"
    assert repair_line(line, KASHIDA, None, RepairOptions()) == "مانها"


def test_operations_are_counted() -> None:
    ops: Counter[str] = Counter()
    line = rtl_line(["ش", ("ل", KASHIDA_GID), "د", (ZWNJ, 3), "ت"])
    repair_line(line, KASHIDA, None, RepairOptions(), ops)
    assert ops["kashida_dropped"] == 1
    assert ops["space_glyph_restored"] == 1
