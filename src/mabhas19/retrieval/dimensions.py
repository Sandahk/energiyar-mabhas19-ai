"""Structured table dimensions and constraint parsing/validation (routing v3).

Every constraint extracted from a query ends in exactly one state (never silently dropped):

* ``applied`` - valid for the table and used as a filter;
* ``invalid`` - outside the table's printed domain -> typed no-match;
* ``ambiguous`` - several printed rows/labels fit (all are returned, with a note);
* ``unsupported`` - the table has no such dimension (reported as a note, not a filter).

Dimensions per table (printed in curated/):

* 19-2-1: building group, year, requirement (topic -> requirement);
* 19-3-1: degree-day condition / climate class; 19-3-2: city / WMO code;
* 19-3-3: R interval, energy rating;
* 19-5-1: element, climate group; 19-5-2: section -> row, climate group;
* 19-5-3, 19-5-4: equipment -> specification; 19-5-5: place of use;
* 19-5-6: use with its printed area condition; 19-5-7: space;
* Appendix 3: occupancy (+ printed size condition), climate class, rating;
* Appendix 7: material hierarchy, density.

Numeric comparisons are parsed generically («بیشتر از ۲۰۰۰», «۳۰۰۰ و بیشتر», «≥»,
«حداقل»...) and compared as intervals with the printed operators; an exact boundary that no
printed condition covers stays uncovered (nothing is interpolated or inferred).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal

from mabhas19.retrieval.entities import ORDINALS

GROUP_LETTERS = ("الف", "ب", "پ", "ج", "د")
_RATING = re.compile(r"(?:رده|رتبه|کلاس)\s*(?:بازدهی\s*)?(?:انرژی\s*)?([a-z])(?![a-z0-9+])")
_GROUP = re.compile(
    r"گروه\s*(?:های\s*)?(?:ساختمانی\s*|ساختمان\s*(?:های\s*)?)?(الف|ب|پ|ج|د)(?![ؠ-ۿ])"
)
_YEAR = re.compile(r"سال\s*(اول|یکم|دوم|سوم|چهارم|پنجم|ششم|\d+)(?![ؠ-ۿ\d])")
_NUM = r"(\d+(?:\.\d+)?)"
_UNIT = r"\s*(?:متر\s*مربع|مترمربع|متر)(?:ی)?"
_GT = r"(?:بزرگ\s?تر|بیشتر|بالاتر)"
_LT = r"(?:کوچک\s?تر|کمتر|پایین\s?تر)"
_AREA_PATTERNS: tuple[tuple[str, str], ...] = (
    (rf"{_GT}\s*از\s*{_NUM}{_UNIT}", ">"),
    (rf"بالای\s*{_NUM}{_UNIT}", ">"),
    (rf"{_LT}\s*از\s*{_NUM}{_UNIT}", "<"),
    (rf"زیر\s*{_NUM}{_UNIT}", "<"),
    (rf"حداقل\s*{_NUM}{_UNIT}", "≥"),
    (rf"حداکثر\s*{_NUM}{_UNIT}", "≤"),
    (rf"{_NUM}{_UNIT}\s*(?:و|یا)\s*(?:{_GT}|به\s*بالا)", "≥"),
    (rf"{_NUM}{_UNIT}\s*(?:و|یا)\s*(?:{_LT}|به\s*پایین)", "≤"),
    (rf"{_NUM}{_UNIT}", "="),
)
_PRINTED_OPS = {"≥": "≥", "≤": "≤", "<": "<", ">": ">"}


@dataclass(frozen=True)
class Interval:
    """A set of numbers: [lo, hi] with open/closed ends; None = unbounded."""

    lo: Decimal | None
    lo_open: bool
    hi: Decimal | None
    hi_open: bool

    @classmethod
    def from_op(cls, op: str, value: Decimal) -> Interval:
        return {
            "=": cls(value, False, value, False),
            ">": cls(value, True, None, False),
            "≥": cls(value, False, None, False),
            "<": cls(None, False, value, True),
            "≤": cls(None, False, value, False),
        }[op]

    def within(self, other: Interval) -> bool:
        """True when every number of self belongs to other."""
        if other.lo is not None:
            if self.lo is None or self.lo < other.lo:
                return False
            if self.lo == other.lo and other.lo_open and not self.lo_open:
                return False
        if other.hi is not None:
            if self.hi is None or self.hi > other.hi:
                return False
            if self.hi == other.hi and other.hi_open and not self.hi_open:
                return False
        return True

    def overlaps(self, other: Interval) -> bool:
        lo = _max_lo(self, other)
        hi = _min_hi(self, other)
        if lo[0] is None or hi[0] is None:
            return True
        return lo[0] < hi[0] or (lo[0] == hi[0] and not lo[1] and not hi[1])

    def describe(self) -> str:
        if self.lo is not None and self.lo == self.hi:
            return f"= {self.lo}"
        parts = []
        if self.lo is not None:
            parts.append(f"{'>' if self.lo_open else '≥'} {self.lo}")
        if self.hi is not None:
            parts.append(f"{'<' if self.hi_open else '≤'} {self.hi}")
        return " and ".join(parts)


def _max_lo(a: Interval, b: Interval) -> tuple[Decimal | None, bool]:
    if a.lo is None:
        return b.lo, b.lo_open
    if b.lo is None or a.lo > b.lo:
        return a.lo, a.lo_open
    if b.lo > a.lo:
        return b.lo, b.lo_open
    return a.lo, a.lo_open or b.lo_open


def _min_hi(a: Interval, b: Interval) -> tuple[Decimal | None, bool]:
    if a.hi is None:
        return b.hi, b.hi_open
    if b.hi is None or a.hi < b.hi:
        return a.hi, a.hi_open
    if b.hi < a.hi:
        return b.hi, b.hi_open
    return a.hi, a.hi_open or b.hi_open


# ------------------------------------------------------------------ parsing


def ratings_requested(norm: str) -> list[str]:
    """Every rating letter the query asks for (validated against a table later)."""
    return sorted({m.upper() for m in _RATING.findall(norm)})


def groups_requested(norm: str) -> list[str]:
    return sorted(set(_GROUP.findall(norm)))


def years_requested(norm: str) -> list[int]:
    out = set()
    for y in _YEAR.findall(norm):
        out.add(int(ORDINALS.get(y, y)))
    return sorted(out)


def area_requested(norm: str) -> Interval | None:
    """The area condition stated in the query («بزرگ‌تر از ۲۰۰۰ متر مربع», «۳۰۰۰ متر مربع و
    بیشتر», «دقیقاً ۲۰۰۰ متر مربعی»)."""
    for pattern, op in _AREA_PATTERNS:
        m = re.search(pattern, norm)
        if m:
            return Interval.from_op(op, Decimal(m[1]))
    return None


_DENSITY_UNIT = r"(?:\s*(?:کیلوگرم\s*(?:بر|در)\s*متر\s*مکعب|kg\s*/?\s*m3))?"
_DENSITY = re.compile(
    rf"(?:چگالی|جرم\s*مخصوص)\s*(?:[^\d\s]+\s*){{0,2}}?{_NUM}{_DENSITY_UNIT}"
    rf"(?:\s*(?:تا|-|–|—)\s*{_NUM}{_DENSITY_UNIT})?"
)


def density_requested(norm: str) -> Interval | None:
    """The density stated in the query as an interval: a single value («چگالی ۴۰۰») is a point,
    a range («چگالی ۲۰۰۰ تا ۲۳۰۰», «۲۰۰۰-۲۳۰۰ کیلوگرم بر متر مکعب») is the closed interval
    between its printed ends; never a midpoint or one boundary."""
    m = _DENSITY.search(norm)
    if not m:
        return None
    a = Decimal(m[1])
    if m[2] is None:
        return Interval(a, False, a, False)
    b = Decimal(m[2])
    lo, hi = min(a, b), max(a, b)
    return Interval(lo, False, hi, False)


def printed_interval(op_printed: str, value: str) -> Interval | None:
    """Interval of a printed condition: «≥ ۳۰۰۰», «کوچک‌تر از ۲۰۰۰» (strict as printed)."""
    op = _PRINTED_OPS.get(op_printed.strip())
    if op is None:
        word = op_printed.replace("‌", "").replace(" ", "")
        op = (
            ">"
            if word.startswith(("بزرگ", "بیش"))
            else "<"
            if word.startswith(("کوچک", "کم"))
            else None
        )
    if op is None:
        return None
    return Interval.from_op(op, Decimal(value))


def label_operator_interval(label: str) -> Interval | None:
    """Printed size condition inside a label («مسکونی ≥ ۳۰۰۰ مترمربع»)."""
    m = re.search(r"([≥≤<>])\s*(\d+(?:\.\d+)?)", label)
    return Interval.from_op(m[1], Decimal(m[2])) if m else None
