"""Typed calculation intent: which operation over printed values a question asks for.

No calculation is ever performed: every structured schema permits none, so any requested
operation ends in ``unsupported_calculation`` with zero regulatory values (the operands are not
the answer). This module only *recognises* the request and records the span that grounded it.

Operations: ``sum`` (مجموع، جمع کل، روی هم)، ``difference`` (اختلاف، تفاضل، فرق، چقدر بیشتر/کمتر
از)، ``average`` (میانگین، معدل، به طور متوسط)، ``ratio`` (نسبت X به Y، چند برابر، تقسیم)،
``interpolation`` (درون‌یابی، میان‌یابی)، ``extrapolation`` (برون‌یابی)، ``multiplication``
(ضرب در، × an area or count)، ``percentage_change`` (چند درصد بیشتر/کمتر، درصد کاهش/افزایش/تغییر)
and ``conversion`` (تبدیل به، به واحد).

Words that also name printed quantities need operands: «اختلاف» or «فرق» is a difference only
when two operands of the same kind are stated (two ratings, classes, entities or numbers joined
by «و»/«با»/«تا»); «اختلاف دمای آب رفت و برگشت» names a printed quantity, not an operation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from mabhas19.text.normalize import normalize_for_search


@dataclass(frozen=True)
class Operation:
    kind: str
    trigger: str
    start: int
    end: int


def _n(words: tuple[str, ...]) -> list[str]:
    return [normalize_for_search(w) for w in words]


# operations recognised from their word alone
_DIRECT: tuple[tuple[str, list[str]], ...] = (
    ("average", _n(("میانگین", "معدل", "به طور متوسط", "بطور متوسط", "average"))),
    (
        "interpolation",
        _n(("درون یابی", "درونیابی", "میان یابی", "میانیابی", "بین یابی", "interpolate")),
    ),
    ("extrapolation", _n(("برون یابی", "برونیابی", "extrapolate"))),
    ("sum", _n(("مجموع", "جمع کل", "حاصل جمع", "روی هم رفته", "رویهم رفته", "رویهم"))),
    ("ratio", _n(("چند برابر", "چندبرابر", "تقسیم بر"))),
    (
        "percentage_change",
        _n(("درصد کاهش", "درصد افزایش", "درصد تغییر", "چند درصد بیشتر", "چند درصد کمتر")),
    ),
    ("difference", _n(("تفاضل", "چقدر بیشتر از", "چقدر کمتر از", "چه قدر بیشتر از"))),
    ("conversion", _n(("تبدیل کن", "تبدیل به", "به واحد"))),
    ("multiplication", _n(("ضرب در", "ضربدر عدد", "×"))),
)
# «جمع» as an operation, not «جمع آوری» (collection) or a plural
_SUM_WORD = re.compile(r"(?<![ؠ-ۿ])جمع(?!\s?اوری)(?![ؠ-ۿ])(?=.*(?:\sو\s|،|\d))")
# words that are operations only with two operands
_NEEDS_OPERANDS = re.compile(r"(?<![ؠ-ۿ])(اختلاف|فرق)(?![ؠ-ۿ])")
_RATIO_OF = re.compile(r"نسبت\s+(?!r\b|مصرف\s*واقعی)[^\d،؟?]{2,40}?\s+به\s+[^\d\s]")
# two operands of the same kind: classes/ratings/numbers joined by «و», «با», «تا», «،»
_OPERANDS = re.compile(
    r"(?:\d\s?[a-z]|(?:رده|کلاس)\s*[a-z]\+*|\d+(?:[.٫/]\d+)?)\s*(?:و|با|تا|،|,)\s*"
    r"(?:(?:رده|کلاس|اقلیم)\s*)?(?:\d\s?[a-z]|[a-z]\+*(?![a-z])|\d+(?:[.٫/]\d+)?)"
    r"|بین\s+\S+(?:\s+\S+){0,3}\s+و\s+\S+"
)


def operations(norm: str) -> list[Operation]:
    """Requested operations in a normalized query, in text order."""
    out: list[Operation] = []
    for kind, words in _DIRECT:
        for w in words:
            for m in re.finditer(rf"(?<![ؠ-ۿ]){re.escape(w)}", norm):
                out.append(Operation(kind, w, m.start(), m.end()))
    for m in _SUM_WORD.finditer(norm):
        out.append(Operation("sum", "جمع", m.start(), m.end()))
    for m in _RATIO_OF.finditer(norm):
        out.append(Operation("ratio", "نسبت … به", m.start(), m.end()))
    if _OPERANDS.search(norm):
        for m in _NEEDS_OPERANDS.finditer(norm):
            out.append(Operation("difference", m[1], m.start(), m.end()))
    return sorted(set(out), key=lambda o: o.start)


def requested(norm: str) -> str | None:
    """The first requested operation kind, or None."""
    ops = operations(norm)
    return ops[0].kind if ops else None
