"""Repair pipeline for one page: raw text -> repaired display text -> search text."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from itertools import pairwise

import pymupdf

from mabhas19.ingest.repair.bidi import logical_order
from mabhas19.ingest.repair.glyphs import Cluster, is_space_glyph, page_clusters
from mabhas19.text.charclass import HYPHEN_VARIANTS, ZWNJ, is_digit, is_persian_letter
from mabhas19.text.normalize import normalize_for_search, to_persian_digits

# Persian fonts used by the source (B Nazanin, B Lotus, B Mitra, B Titr ...) draw ASCII
# digit codes with Persian digit glyphs; Latin fonts (Times, Arial) draw Latin digits.
_PERSIAN_FONT = re.compile(r"^B[A-Z]")
# Letters that never connect to the following letter (so need no ZWNJ after them).
RIGHT_JOINING = frozenset("اآأإٱدذرزژوؤةۀ")
GAP_SPACE_RATIO = 0.3  # gap (in font sizes) that implies a missing space
LINE_TOLERANCE_RATIO = 0.4  # baseline distance (in font sizes) within one visual line
FINAL_FORM_MIN_RATIO = 0.75
FINAL_FORM_MIN_COUNT = 5

_DIGIT = "0-9٠-٩۰-۹"
_HY = "".join(re.escape(h) for h in sorted(HYPHEN_VARIANTS))
_VISUAL_SECTION = re.compile(
    rf"(?<![{_DIGIT}])[{_DIGIT}]{{1,3}}(?:[{_HY}][{_DIGIT}]{{1,3}}){{1,5}}(?![{_DIGIT}])"
)
_PART_19 = {"19", "۱۹", "١٩"}


@dataclass(frozen=True)
class RepairOptions:
    """Each repair operation can be switched off for ablation."""

    space_glyphs: bool = True
    drop_kashida: bool = True
    keep_ligature_clusters: bool = True
    gap_spaces: bool = True
    geometric_order: bool = True
    mirror_brackets: bool = True
    section_numbers: bool = True
    persian_digits: bool = True
    infer_zwnj: bool = True


@dataclass
class RepairedPage:
    raw_text: str
    repaired_display_text: str
    normalized_search_text: str
    operations: Counter[str] = field(default_factory=Counter)


@dataclass
class _Item:
    text: str
    x0: float
    x1: float
    cluster: Cluster | None  # None for inserted spaces


def _lines(clusters: list[Cluster]) -> list[list[Cluster]]:
    lines: list[list[Cluster]] = []
    for c in sorted(clusters, key=lambda c: c.baseline):
        if lines:
            ref = max(lines[-1], key=lambda x: x.size)
            if abs(c.baseline - ref.baseline) <= LINE_TOLERANCE_RATIO * max(ref.size, c.size):
                lines[-1].append(c)
                continue
        lines.append([c])
    return lines


def _fix_sections(text: str, ops: Counter[str]) -> str:
    def repl(m: re.Match[str]) -> str:
        groups = re.split(f"[{_HY}]", m.group(0))
        if groups[-1] in _PART_19 and groups[0] not in _PART_19:
            ops["section_number_reversed"] += 1
            return "-".join(reversed(groups))
        return m.group(0)

    return _VISUAL_SECTION.sub(repl, text)


class FinalFormStats:
    """Which (font, gid) letter glyphs are final/isolated forms, learned from a document.

    A glyph that is almost always followed by a word break is a final or isolated form;
    if such a glyph of a dual-joining letter is followed directly by another letter, the
    source must have had a ZWNJ there (otherwise the letter would have been drawn joined).
    """

    def __init__(self) -> None:
        self.end: Counter[tuple[str, int]] = Counter()
        self.total: Counter[tuple[str, int]] = Counter()

    @staticmethod
    def family_key(c: Cluster) -> tuple[str, int]:
        """Regular and bold cuts of the source fonts share glyph ids (e.g. lam 181-184)."""
        return (c.font.replace("Bold", ""), c.gid)

    def observe(self, items: list[_Item]) -> None:
        for cur, nxt in zip(items, [*items[1:], None], strict=True):
            c = cur.cluster
            if c is None or len(c.text) != 1 or not is_persian_letter(c.text):
                continue
            key = self.family_key(c)
            self.total[key] += 1
            nxt_c = nxt.cluster if nxt else None
            if nxt_c is None or not is_persian_letter(nxt_c.text[0]):
                self.end[key] += 1

    def is_final_form(self, key: tuple[str, int]) -> bool:
        total = self.total[key]
        return total >= FINAL_FORM_MIN_COUNT and self.end[key] / total >= FINAL_FORM_MIN_RATIO


def _gap_spaces(items: list[_Item], ops: Counter[str]) -> list[_Item]:
    """Spaces for visible gaps between glyphs that have no space glyph between them."""
    extra = []
    for a, b in pairwise(sorted(items, key=lambda it: it.x0)):
        size = max(a.cluster.size if a.cluster else 0, b.cluster.size if b.cluster else 0)
        if b.x0 - a.x1 > GAP_SPACE_RATIO * size and " " not in (a.text, b.text):
            extra.append(_Item(" ", a.x1, b.x0, None))
            ops["gap_space_inserted"] += 1
    return extra


def _logical_items(
    line: list[Cluster], kashida: frozenset[tuple[str, int]], opt: RepairOptions, ops: Counter[str]
) -> list[_Item]:
    items: list[_Item] = []
    seen: set[tuple[str, int, str, float, float]] = set()
    for c in line:
        text = c.text
        # Word simulates bold by drawing some glyphs twice at the same position.
        mark = (c.text, c.gid, c.font, round(c.x0, 1), round(c.baseline, 1))
        if mark in seen:
            ops["overprint_duplicate_dropped"] += 1
            continue
        seen.add(mark)
        if opt.drop_kashida and c.key in kashida:
            # Keep an empty placeholder so the gap it fills is not mistaken for a space.
            ops["kashida_dropped"] += 1
            items.append(_Item("", c.x0, c.x1, None))
            continue
        if opt.space_glyphs and is_space_glyph(c):
            if text != " ":
                ops["space_glyph_restored"] += 1
            text = " "
        if opt.persian_digits and _PERSIAN_FONT.match(c.font) and any(is_digit(ch) for ch in text):
            text = to_persian_digits(text)
        if opt.keep_ligature_clusters or len(text) == 1:
            if len(text) > 1:
                ops["ligature_cluster_kept"] += 1
            items.append(_Item(text, c.x0, c.x1, c))
        else:
            for ch in text:
                items.append(_Item(ch, c.x0, c.x1, c))
    if opt.gap_spaces:
        items += _gap_spaces(items, ops)
    if not opt.geometric_order:
        return [it for it in items if it.text]
    order = logical_order(items, lambda it: it.text, lambda it: (it.x0 + it.x1) / 2,
                          opt.mirror_brackets)  # fmt: skip
    return [_Item(text, it.x0, it.x1, it.cluster) for it, text in order if text]


def _insert_zwnj(items: list[_Item], stats: FinalFormStats, ops: Counter[str]) -> str:
    parts: list[str] = []
    for cur, nxt in zip(items, [*items[1:], None], strict=True):
        parts.append(cur.text)
        c, n = cur.cluster, (nxt.cluster if nxt else None)
        if (
            c is not None
            and n is not None
            and len(c.text) == 1
            and is_persian_letter(c.text)
            and c.text not in RIGHT_JOINING
            and is_persian_letter(n.text[0])
            and stats.is_final_form(stats.family_key(c))
        ):
            parts.append(ZWNJ)
            ops["zwnj_inferred"] += 1
    return "".join(parts)


def repair_page(
    page: pymupdf.Page,
    kashida: frozenset[tuple[str, int]],
    stats: FinalFormStats | None = None,
    options: RepairOptions | None = None,
    clip: pymupdf.Rect | None = None,
) -> RepairedPage:
    """Repair a page (or only the glyphs whose centre lies inside ``clip``)."""
    opt = options or RepairOptions()
    ops: Counter[str] = Counter()
    raw = page.get_text(clip=clip) if clip is not None else page.get_text()
    clusters = page_clusters(page)
    if clip is not None:
        clusters = [
            c for c in clusters
            if clip.contains(pymupdf.Point((c.x0 + c.x1) / 2, (c.top + c.bottom) / 2))
        ]  # fmt: skip
    texts = [repair_line(line, kashida, stats, opt, ops) for line in _lines(clusters)]
    display = "\n".join(t for t in texts if t)
    return RepairedPage(raw, display, normalize_for_search(display), ops)


def repair_line(
    line: list[Cluster],
    kashida: frozenset[tuple[str, int]],
    stats: FinalFormStats | None = None,
    options: RepairOptions | None = None,
    ops: Counter[str] | None = None,
) -> str:
    """Repaired display text of one visual line of glyph clusters."""
    opt = options or RepairOptions()
    ops = ops if ops is not None else Counter()
    items = _logical_items(line, kashida, opt, ops)
    if opt.infer_zwnj and stats is not None:
        text = _insert_zwnj(items, stats, ops)
    else:
        text = "".join(it.text for it in items)
    if opt.section_numbers:
        text = _fix_sections(text, ops)
    return " ".join(text.split())


def learn_final_forms(doc: pymupdf.Document, kashida: frozenset[tuple[str, int]]) -> FinalFormStats:
    """Collect final-form statistics over the whole document (no ZWNJ inference yet)."""
    stats = FinalFormStats()
    opt = RepairOptions(infer_zwnj=False)
    for page in doc:
        for line in _lines(page_clusters(page)):
            stats.observe(_logical_items(line, kashida, opt, Counter()))
    return stats
