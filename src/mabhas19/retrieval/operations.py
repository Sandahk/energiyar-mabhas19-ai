"""Typed calculation intent: which operation over printed values a question asks for.

No calculation is ever performed: every structured schema permits none, so any requested
operation ends in ``unsupported_calculation`` with zero regulatory values (the operands are not
the answer). This module only *recognises* the request and records the span that grounded it.

Operations: ``sum`` (مجموع، جمع کل، روی هم)، ``difference`` (اختلاف، تفاضل، فرق، تفریق، چقدر
بیشتر/کمتر از، «عدد منهای عدد»)، ``average`` (میانگین، معدل، به طور متوسط)، ``ratio`` (نسبت X به Y،
چند برابر، تقسیم)، ``interpolation`` (درون‌یابی، میان‌یابی)، ``extrapolation`` (برون‌یابی)،
``multiplication`` (ضرب در، × an area or count)، ``percentage_change`` (چند درصد بیشتر/کمتر، درصد
کاهش/افزایش/تغییر)، ``conversion`` (تبدیل به، به واحد) and ``calculation`` - an explicit request
to calculate (the imperative «حساب کنید / محاسبه کن …»; «چطور حساب کنیم» asks for a method and
«حساب می‌شود» means "counts as", so neither is one).

Words that also name printed quantities or everyday actions need operands: «اختلاف», «فرق»,
«فاصله» (a gap between two values; «فاصله اسپرینکلرها» is a distance) and «کم/کسر کنید» (to
subtract; «هزینه رو کم کنه» is to reduce) are a difference only when two operands of the same
kind are stated: two ratings, classes or numbers joined by «و»/«با»/«تا»/«،», a «از X به/تا Y»
pair, or «این/هر دو + a value noun» («فاصله این دو حد مجاز»). «اختلاف دمای آب رفت و برگشت» names
a printed quantity, not an operation. The operands are never the answer.
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
    ("difference", _n(("تفاضل", "تفریق", "چقدر بیشتر از", "چقدر کمتر از", "چه قدر بیشتر از"))),
    ("conversion", _n(("تبدیل کن", "تبدیل به", "به واحد"))),
    ("multiplication", _n(("ضرب در", "ضربدر عدد", "×"))),
)
# «جمع» as an operation, not «جمع آوری» (collection) or a plural
_SUM_WORD = re.compile(r"(?<![ؠ-ۿ])جمع(?!\s?اوری)(?![ؠ-ۿ])(?=.*(?:\sو\s|،|\d))")
# words that are operations only with two operands
_NEEDS_OPERANDS = re.compile(
    r"(?<![ؠ-ۿ])(اختلاف|فرق|فاصله|(?:کم|کسر)\s?(?:کن|کنید|کنین|بکن|بکنید|بکنین))(?![ؠ-ۿ])"
)
# an explicit imperative request to calculate (not «حساب کنیم» / «حساب می شود»)
_CALCULATE = re.compile(
    r"(?<![ؠ-ۿ])(?:حساب|محاسبه)\s?(?:کن|کنید|کنین|بکن|بکنید|بکنین|نمایید|بفرمایید)(?![ؠ-ۿ])"
)
# two distinct regulatory values anywhere in the question are operands too («حد رده B را از حد
# رده A کم کنید»): rating letters standing alone, climate codes
_RATING_VALUE = re.compile(r"(?<![a-z0-9+°./\-])(?<!\d\s)(?<!درجه\s)([a-d])\+{0,3}(?![a-z0-9+/\-])")
_CLIMATE_VALUE = re.compile(r"(?<![a-z0-9])([0-5]\s?[a-c])(?![a-z0-9/])")
# «250 منهای 200»: a subtraction (a bare «منهای X» is an exclusion, see ``polarity``)
_MINUS = re.compile(r"\d+(?:[.٫/]\d+)?\s*منهای\s*\d")
_RATIO_OF = re.compile(r"نسبت\s+(?!r\b|مصرف\s*واقعی)[^\d،؟?]{2,40}?\s+به\s+[^\d\s]")
# two operands of the same kind: classes/ratings/numbers joined by «و», «با», «تا», «،»
_OPERANDS = re.compile(
    r"(?:\d\s?[a-z]|(?:رده|کلاس)\s*[a-z]\+*|\d+(?:[.٫/]\d+)?)\s*(?:و|با|تا|،|,)\s*"
    r"(?:(?:رده|کلاس|اقلیم)\s*)?(?:\d\s?[a-z]|[a-z]\+*(?![a-z])|\d+(?:[.٫/]\d+)?)"
    r"|بین\s+\S+(?:\s+\S+){0,3}\s+و\s+\S+"
    # «از رده d به c», «از 2a تا 3b», «از 200 تا 250»
    r"|(?<![ؠ-ۿ])از\s+(?:(?:رده|کلاس|اقلیم)\s*)?(?:\d\s?[a-z]|[a-z]\+*(?![a-z])|\d+(?:[.٫/]\d+)?)\s*"
    r"(?:به|تا)\s+(?:(?:رده|کلاس|اقلیم)\s*)?(?:\d\s?[a-z]|[a-z]\+*(?![a-z])|\d+(?:[.٫/]\d+)?)"
    # «این دو حد مجاز», «هر دو عدد»
    r"|(?<![ؠ-ۿ])(?:این|اون|ان|هر)\s+(?:دو|2)\s*(?:تا\s*)?(?:حد|عدد|مقدار|رقم|سقف|رده|ردیف|کلاس)"
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
    for m in _CALCULATE.finditer(norm):
        out.append(Operation("calculation", m[0], m.start(), m.end()))
    for m in _MINUS.finditer(norm):
        out.append(Operation("difference", "منهای", m.start(), m.end()))
    for m in _RATIO_OF.finditer(norm):
        out.append(Operation("ratio", "نسبت … به", m.start(), m.end()))
    if (
        _OPERANDS.search(norm)
        or len(set(_RATING_VALUE.findall(norm))) > 1
        or len({c.replace(" ", "") for c in _CLIMATE_VALUE.findall(norm)}) > 1
    ):
        for m in _NEEDS_OPERANDS.finditer(norm):
            out.append(Operation("difference", m[1], m.start(), m.end()))
    return sorted(set(out), key=lambda o: o.start)


def requested(norm: str) -> str | None:
    """The first requested operation kind, or None."""
    ops = operations(norm)
    return ops[0].kind if ops else None
