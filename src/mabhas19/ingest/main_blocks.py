"""Main Mabhas 19 text blocks: line model, block classification, paragraphs, hierarchy.

Pure logic over repaired visual lines (no PDF access here). A line carries its raw text
(the text-layer characters in content-stream order, i.e. the damaged extraction) and its
repaired display text (``repair_v1``); the two are never mixed.

Block kinds: ``heading`` (numbered, bold), ``clause`` (numbered, regular weight), ``paragraph``,
``list_item``, ``note`` («تبصره», «یادآوری», «توضیح»), ``formula`` (mostly math glyphs),
``footnote``, ``table_caption``, ``figure_caption``, ``toc_entry``; furniture (``running_header``,
``page_number``) is kept but excluded from the text corpus. Tables and figures are separate
region blocks built by the caller.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Any

from mabhas19.text.normalize import normalize_for_search, to_ascii_digits

STATUS = "machine_extracted_pending_review"
LATIN_GLOSS = 0.6  # share of Latin letters in an English term gloss
_SENTENCE_END = re.compile(r"[.:؛!؟»\")]\s*$")
BODY_SIZE_MIN = 11.0  # body text is BNazanin 12; footnotes 8, table cells 6-10
FOOTNOTE_SIZE_MAX = 9.0
_D = "0-9۰-۹٠-٩"
SECTION = re.compile(
    rf"^\s*([{_D}]{{1,3}}(?:\s?[-‐‑–]\s?[{_D}]{{1,3}}){{1,7}})[-‐‑–]?(?=\s|$|[:.)])"
)
_LIST = re.compile(rf"^\s*(?:[-–•▪●]|[{_D}]{{1,2}}[-.)]|[آ-ی]{{1,3}}[)(]|\()\s*")
_MARKER_MIRRORED = re.compile(r"^\s*[آ-ی]{1,3}\(")  # «الف(» where «الف)» is printed
_MIRRORED_LATIN_BRACKETS = re.compile(r"\)[A-Za-z0-9][^()]{0,40}\(")
_B1 = re.compile(r"\)([A-Za-z0-9][^()]{0,40})\(")  # capture the Latin run
_MARKER_MISSING = re.compile(r"^\s*\(\s")  # a list item starting with a bare «(»
_NOTE = re.compile(r"^\s*(تبصره|یادآوری|توضیح|نکته)\b")
_TABLE_CAPTION = re.compile(r"^\s*جدول\s")
_FIGURE_CAPTION = re.compile(r"^\s*(شکل|نمودار)\s")
_TOC_LEADER = re.compile(r"\.{5,}|…{2,}")
_SINGLE_LETTER = re.compile(r"(?<!\S)[آ-ی](?!\S)")
_BAD_CHARS = re.compile("[�-]")


@dataclass
class Line:
    pdf_page: int
    bbox: tuple[float, float, float, float]  # x0, top, x1, bottom (PDF points)
    raw: str  # text-layer characters, content-stream order (damaged)
    text: str  # repaired display text (repair_v1)
    size: float  # dominant glyph size
    bold: float  # share of bold glyphs
    math: float  # share of math-font glyphs
    latin: float  # share of Latin letters among letters
    ops: dict[str, int] = field(default_factory=dict)
    machine: str | None = None  # repair_v1 text before the approved glyph rules (if different)
    glyph_rules: dict[str, int] = field(default_factory=dict)  # e.g. {"G1": 1}


def section_number(text: str) -> str | None:
    """Printed section number at the start of a repaired line, ASCII digits, '-' separated."""
    m = SECTION.match(text)
    if not m:
        return None
    parts = re.split(r"\s?[-‐‑–]\s?", to_ascii_digits(m[1]))
    if parts[0] != "19":
        return None
    return "-".join(parts)


def section_level(number: str) -> int:
    return number.count("-")


def _is_footnote(line: Line) -> bool:
    """Small type: footnotes, incl. the English term glosses («Life Cycle Assessment (LCA)»)."""
    small = line.size < BODY_SIZE_MIN
    return line.size <= FOOTNOTE_SIZE_MAX or (
        small and (_footnote_start(line.text) or line.latin >= LATIN_GLOSS)
    )


def _captionlike(line: Line) -> bool:
    return line.bold >= 0.5 or line.size < BODY_SIZE_MIN  # captions are set apart from body


_RULES: tuple[tuple[str, Any], ...] = (
    ("toc_entry", lambda ln: bool(_TOC_LEADER.search(ln.text) or _TOC_LEADER.search(ln.raw))),
    ("formula", lambda ln: ln.math >= 0.3),
    ("footnote", _is_footnote),
    ("table_caption", lambda ln: _captionlike(ln) and bool(_TABLE_CAPTION.match(ln.text))),
    ("figure_caption", lambda ln: _captionlike(ln) and bool(_FIGURE_CAPTION.match(ln.text))),
    ("heading", lambda ln: bool(section_number(ln.text)) and ln.bold >= 0.5),
    ("clause", lambda ln: bool(section_number(ln.text))),
    ("heading_unnumbered", lambda ln: ln.bold >= 0.6),
    ("note", lambda ln: bool(_NOTE.match(ln.text))),
    ("list_item", lambda ln: bool(_LIST.match(ln.text))),
)


def line_kind(line: Line) -> str:
    """Kind of the block a line would start (paragraph continuation is decided later).

    The rules are tried in order; the first match wins.
    """
    return next((kind for kind, rule in _RULES if rule(line)), "paragraph")


def quality_flags(text: str, raw: str) -> list[str]:
    """Signs that the repair left damage behind (flags, never fixes)."""
    flags = []
    if not text.strip() and raw.strip():
        flags.append("repair_empty")
    if _BAD_CHARS.search(text):
        flags.append("unmapped_glyphs")
    singles = [t for t in _SINGLE_LETTER.findall(text) if t != "و"]
    if len(singles) >= 3:
        flags.append("spacing_suspect")
    words = re.split(r"\s+|\.{3,}|…+", text)  # leader dots are legitimate, not words
    if any(len(tok) > 28 and not tok.isascii() for tok in words):
        flags.append("missing_spaces_suspect")
    if _MARKER_MIRRORED.match(text):
        flags.append("list_marker_paren_mirrored")
    if _MARKER_MISSING.match(text):
        flags.append("list_marker_letter_missing")
    if _MIRRORED_LATIN_BRACKETS.search(text):
        flags.append("bracket_mirroring_suspect")  # «)R-Value(» around a Latin run
    return flags


@dataclass
class Layout:
    body_left: float  # left edge of full-width body lines
    body_right: float
    pitch: float  # typical baseline-to-baseline distance of body lines


def page_layout(lines: list[Line]) -> Layout:
    body = [ln for ln in lines if ln.size >= BODY_SIZE_MIN]
    if not body:
        return Layout(0.0, 0.0, 20.0)
    lefts = [round(ln.bbox[0]) for ln in body]
    rights = [round(ln.bbox[2]) for ln in body]
    tops = sorted(ln.bbox[1] for ln in body)
    gaps = [b - a for a, b in pairwise(tops) if 0 < b - a < 40]
    return Layout(float(statistics.mode(lefts)), float(statistics.mode(rights)),
                  float(statistics.median(gaps)) if gaps else 20.0)  # fmt: skip


def _ends_paragraph(line: Line, layout: Layout) -> bool:
    """RTL justified text: a line that stops short of the left margin ends its paragraph."""
    return line.bbox[0] > layout.body_left + 15


def group_blocks(lines: list[Line], layout: Layout) -> list[dict[str, Any]]:
    """Consecutive lines merged into blocks; each block keeps its lines' raw and repaired text."""
    blocks: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    prev: Line | None = None
    for line in lines:
        kind = line_kind(line)
        if (kind == "clause" and current is not None and not current["closed"] and prev is not None
                and not _SENTENCE_END.search(prev.text)):  # fmt: skip
            kind = "paragraph"  # a wrapped cross-reference («... بند ۱۹-۵-۲ باشد»), not a clause
        starts = kind not in ("paragraph",)
        if current is not None and prev is not None and not starts:
            gap = line.bbox[1] - prev.bbox[1]
            joinable = current["kind"] in ("paragraph", "clause", "note", "list_item", "footnote")
            if joinable and not current["closed"] and gap <= layout.pitch * 1.5:
                _append(current, line)
                current["closed"] = _ends_paragraph(line, layout)
                prev = line
                continue
        if kind == "footnote" and current is not None and current["kind"] == "footnote" and not (
                _footnote_start(line.text)):  # fmt: skip
            _append(current, line)
            prev = line
            continue
        current = {"kind": kind, "lines": [line], "closed": _ends_paragraph(line, layout)}
        if kind in ("heading", "heading_unnumbered", "table_caption", "figure_caption",
                    "toc_entry", "formula"):  # fmt: skip
            current["closed"] = True
        blocks.append(current)
        prev = line
    return blocks


_FOOTNOTE_START = re.compile(rf"^\s*[{_D}]{{1,2}}\s")


def _footnote_start(text: str) -> bool:
    return bool(_FOOTNOTE_START.match(text))


def _append(block: dict[str, Any], line: Line) -> None:
    block["lines"].append(line)


def finish_block(block: dict[str, Any]) -> dict[str, Any]:
    lines: list[Line] = block["lines"]
    text = "\n".join(ln.text for ln in lines)
    raw = "\n".join(ln.raw for ln in lines)
    x0 = min(ln.bbox[0] for ln in lines)
    y0 = min(ln.bbox[1] for ln in lines)
    x1 = max(ln.bbox[2] for ln in lines)
    y1 = max(ln.bbox[3] for ln in lines)
    number = section_number(lines[0].text) if block["kind"] in ("heading", "clause") else None
    ops: dict[str, int] = {}
    for ln in lines:
        for k, v in ln.ops.items():
            ops[k] = ops.get(k, 0) + v
    machine = "\n".join(ln.machine if ln.machine is not None else ln.text for ln in lines)
    glyph_rules: dict[str, int] = {}
    for ln in lines:
        for k, v in ln.glyph_rules.items():
            glyph_rules[k] = glyph_rules.get(k, 0) + v
    flags = quality_flags(text, raw)
    if block["kind"] == "heading_unnumbered":
        flags.append("unnumbered_bold_line")  # heading, term gloss or emphasis: review
    return {
        "kind": block["kind"], "section_number": number,
        "text_raw": raw, "text_repair_v1": machine, "text": text,
        "text_normalized": normalize_for_search(text),
        "applied_rules": [{"rule": k, "count": v} for k, v in sorted(glyph_rules.items())],
        "bbox": [round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)],
        "line_count": len(lines), "font_size": round(statistics.median(ln.size for ln in lines), 1),
        "repair_operations": ops, "flags": flags,
    }  # fmt: skip


def toc_entry(text: str) -> dict[str, Any]:
    """«۱۹-۱ تعاریف ....... ۱» -> number, title, printed page (as printed, no lookup)."""
    number = section_number(text)
    parts = _TOC_LEADER.split(text)
    title = parts[0]
    if number:
        title = SECTION.sub("", title, count=1)
    page = parts[-1].strip() if len(parts) > 1 else None
    return {"section_number": number, "title": " ".join(title.split()) or None,
            "toc_printed_page": page or None}  # fmt: skip


class Hierarchy:
    """Section stack built from printed heading numbers, in reading order."""

    def __init__(self) -> None:
        self.stack: list[dict[str, Any]] = []
        self.last: tuple[int, ...] | None = None

    def enter(self, number: str, title: str, block_id: str) -> list[str]:
        """Register a heading; returns sequence problems (never reorders anything)."""
        key = tuple(int(x) for x in number.split("-"))
        problems = []
        if self.last is not None and key <= self.last:
            problems.append(f"heading {number} does not follow {'-'.join(map(str, self.last))}")
        level = section_level(number)
        self.stack = [h for h in self.stack if h["level"] < level and number.startswith(
            h["number"] + "-")]  # fmt: skip
        if level > 1 and not self.stack:
            problems.append(f"heading {number} has no parent heading")
        self.stack.append({"number": number, "title": title, "level": level, "block_id": block_id})
        self.last = key
        return problems

    def note_clause(self, number: str) -> None:
        self.last = max(self.last or (), tuple(int(x) for x in number.split("-")))

    def path(self) -> list[dict[str, Any]]:
        return [{"number": h["number"], "title": h["title"]} for h in self.stack]


# --------------------------------------------------------------------------- approved rules
# M1, B1 and C1 were approved in the assistant review of 2026-10-16 (eval/extraction/review/
# main-assistant-review.yaml); they act on repaired text only, raw text is never touched.

ALPHABET_MARKERS = frozenset(
    [
        "الف",
        "ب",
        "پ",
        "ت",
        "ث",
        "ج",
        "چ",
        "ح",
        "خ",
        "د",
        "ذ",
        "ر",
        "ز",
        "ژ",
        "س",
        "ش",
        "ص",
        "ض",
        "ط",
        "ظ",
        "ع",
        "غ",
        "ف",
        "ق",
        "ک",
        "گ",
        "ل",
        "م",
        "ن",
        "و",
        "ه",
        "ی",
    ]
)
_LEADING_MARKER = re.compile(r"^(\s*)([آ-ی]{1,3})\(")
_CAPTION_NO = re.compile(r"(جدول\s+)([۰-۹]+(?:\s*-\s*[۰-۹]+)+)(\s*:)")


def rule_m1(text: str) -> tuple[str, bool]:
    """«الف( ...» -> «الف) ...» for a genuine leading alphabetic list marker."""
    m = _LEADING_MARKER.match(text)
    if not m or m[2] not in ALPHABET_MARKERS:
        return text, False
    return f"{m[1]}{m[2]})" + text[m.end() :], True


def is_false_marker(text: str) -> bool:
    """A list item whose «marker» is a word (wrapped parenthetical, e.g. «سیم(», «شود(.»)."""
    m = _LEADING_MARKER.match(text)
    return bool(m) and m[2] not in ALPHABET_MARKERS  # type: ignore[index]


def rule_b1(text: str) -> tuple[str, int]:
    """«)Latin(» -> «(Latin)» (mirrored brackets around a Latin run)."""
    return _B1.subn(r"(\1)", text)


def rule_c1(text: str) -> tuple[str, str | None]:
    """Table-caption number printed out of order («۱-۲ -۱۹», «۲ -۱۹-۵») -> «۱۹-…»."""
    m = _CAPTION_NO.search(text)
    if not m:
        return text, None
    parts = re.split(r"\s*-\s*", m[2])
    if parts[0] == "۱۹" or "۱۹" not in parts:
        return text, None
    if parts[-1] == "۱۹":
        fixed = list(reversed(parts))
    else:
        i = parts.index("۱۹")
        fixed = parts[i:] + parts[:i]
    number = "-".join(fixed)
    return text[: m.start(2)] + number + text[m.end(2) :], number


def apply_text_rules(block: dict[str, Any], page_role: str = "body") -> dict[str, Any]:
    """Approved M1/B1/C1 on one block's repaired text; every application is recorded.

    Also corrects one classification error: a «toc_entry» outside the TOC pages (an ellipsis
    «.....» in body text looks like leader dots) is reclassified by its form.
    """
    text = block.get("text")
    if not text:
        return block
    rules = block.setdefault("applied_rules", [])
    if block["kind"] == "toc_entry" and page_role != "toc":
        block["kind"] = "list_item" if _LIST.match(text) else "paragraph"
        rules.append({"rule": "K1", "count": 1, "note": "toc_entry outside the TOC pages "
                      f"reclassified as {block['kind']}"})  # fmt: skip
    exception = block["kind"] == "list_item" and is_false_marker(text)
    if exception:
        block["kind"] = "paragraph"
        block["continues_previous_block"] = True
        rules.append({"rule": "M1-exception", "count": 1, "note": "wrapped parenthetical, "
                      "reclassified from list_item; M1 not applied"})  # fmt: skip
    else:
        text, done = rule_m1(text)  # any block starting with an alphabetic list marker
        if done:
            rules.append({"rule": "M1", "count": 1})
    text, n = rule_b1(text)
    if n:
        rules.append({"rule": "B1", "count": n})
    if block["kind"] == "table_caption":
        text, number = rule_c1(text)
        if number:
            rules.append({"rule": "C1", "count": 1, "number": number})
    block["text"] = text
    block["text_normalized"] = normalize_for_search(text)
    flags = quality_flags(text, block["text_raw"])
    if exception:  # the approved exception: not a list marker, so no marker flag
        flags = [f for f in flags if f != "list_marker_paren_mirrored"]
    block["flags"] = flags + [f for f in block["flags"] if f == "unnumbered_bold_line"]
    return block
