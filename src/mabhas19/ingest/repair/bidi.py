"""Visual-order to logical-order conversion for one right-to-left text line.

The main PDF stores glyphs in *visual* (left-to-right) order. For a right-to-left
paragraph, logical order is obtained by reversing the visual sequence, except that runs
of left-to-right material (Latin words, numbers and the neutrals sandwiched between
them) keep their internal order. Brackets resolved to the right-to-left direction were
drawn mirrored, so they are mirrored back.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from mabhas19.text.charclass import is_digit, is_latin_letter, is_persian_letter

# Only brackets are mirrored back: in the main PDF «(MJ/m2)» extracts with mirrored
# bracket glyph codes, whereas guillemets («۳») already extract as logical characters.
MIRRORED = {"(": ")", ")": "(", "[": "]", "]": "[", "{": "}", "}": "{"}


@dataclass(frozen=True)
class VisualItem:
    """One glyph cluster (possibly several characters, e.g. a lam-alef ligature)."""

    text: str
    x0: float
    x1: float


def strong_type(text: str) -> str:
    """'L' (Latin letter or digit), 'R' (Arabic script) or 'N' (neutral)."""
    for ch in text:
        if is_latin_letter(ch) or is_digit(ch):
            return "L"
        if is_persian_letter(ch) or "\u0600" <= ch <= "ۿ":
            return "R"
    return "N"


def logical_order[T](
    items: Sequence[T],
    text_of: Callable[[T], str],
    center_of: Callable[[T], float],
    mirror_brackets: bool = True,
) -> list[tuple[T, str]]:
    """Items of one RTL line in logical order, each paired with its (mirrored) text."""
    ordered = sorted(items, key=center_of)
    types = [strong_type(text_of(it)) for it in ordered]
    if "R" not in types:  # a purely left-to-right line (e.g. an English footnote)
        return [(it, text_of(it)) for it in ordered]
    resolved = list(types)
    for i, t in enumerate(types):
        if t != "N":
            continue
        # A neutral joins a left-to-right run only with strong L material on both sides.
        left = next((types[j] for j in range(i - 1, -1, -1) if types[j] != "N"), None)
        right = next((types[j] for j in range(i + 1, len(types)) if types[j] != "N"), None)
        resolved[i] = "L" if left == "L" and right == "L" else "R"

    runs: list[tuple[str, list[tuple[T, str]]]] = []
    for it, direction in zip(ordered, resolved, strict=True):
        text = text_of(it)
        if direction == "R" and mirror_brackets:
            text = "".join(MIRRORED.get(c, c) for c in text)
        if runs and runs[-1][0] == direction:
            runs[-1][1].append((it, text))
        else:
            runs.append((direction, [(it, text)]))

    out: list[tuple[T, str]] = []
    for direction, members in reversed(runs):
        out.extend(members if direction == "L" else reversed(members))
    return out


def visual_to_logical(items: list[VisualItem], mirror_brackets: bool = True) -> str:
    """Logical text of a right-to-left line given its glyph clusters in any order."""
    pairs = logical_order(items, lambda it: it.text, lambda it: (it.x0 + it.x1) / 2,
                          mirror_brackets)  # fmt: skip
    return "".join(text for _, text in pairs)
