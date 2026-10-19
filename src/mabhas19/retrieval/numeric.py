"""One numeric model for every structured table: exact numerals, typed bounds, intervals.

Numbers are parsed from search-normalized text (digits are already ASCII) and stored as
:class:`~decimal.Decimal`, never ``float``. Numerals:

* decimal separators «.», «٫» and «/» between digits (the sources print «۱/۹۶» for 1.96, so a
  slash between digits is a decimal separator, never a division);
* thousands separators «٬» always, and «,»/«،» when exactly three digits follow and the integer
  part has one to three digits («1,450» = 1450); otherwise «,» is a decimal comma («0,5»).

A bound is ``closed`` (≤, ≥, «حداقل», «حداکثر», «یا مساوی»), ``open`` (<, >, «کمتر از», «بیش
از», «کوچک‌تر», «بزرگ‌تر») or ``unspecified`` - a printed range «۶۰۰ تا ۸۰۰» / «بین ۲۰ تا ۵۰»
whose end inclusivity the source does not state. Every table comparison is an interval
operation; a value equal to an unspecified bound is reported as ``boundary`` (it may belong to
the row; never silently decided either way).

Query intervals: a plain or «دقیقاً» value is a point; «X تا Y», «بین X و Y» are the closed
interval the user states; comparison words and symbols give half-lines.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Final, Literal

BoundKind = Literal["closed", "open", "unspecified"]
CLOSED: Final = "closed"
OPEN: Final = "open"
UNSPECIFIED: Final = "unspecified"
Membership = Literal["yes", "no", "boundary"]
Relation = Literal["within", "boundary", "overlap", "disjoint"]

# a numeral: digits with decimal / thousands separators (never glued to Latin letters, so «m2»,
# «co2», «ie3» and climate codes «3b» are not numbers)
NUMERAL = r"(?<![a-z\d.٫/,،٬])\d+(?:[.٫/,،٬]\d+)*(?![\d]|[a-z](?![a-z]))"
_NUMERAL = re.compile(NUMERAL)


def parse_numeral(raw: str) -> Decimal | None:
    """Decimal value of one numeral («4٫5», «1,450», «0/35», «1٬450»); None if malformed."""
    s = raw.strip()
    s = s.replace("٬", "")
    m = re.fullmatch(r"(\d{1,3})[,،](\d{3})", s)
    if m:
        s = m[1] + m[2]
    s = re.sub(r"[٫/,،]", ".", s)
    if s.count(".") > 1:
        return None
    try:
        return Decimal(s)
    except InvalidOperation:
        return None


@dataclass(frozen=True)
class Bound:
    value: Decimal
    kind: BoundKind


@dataclass(frozen=True)
class NumInterval:
    """[lo, hi] with typed ends; ``None`` = unbounded."""

    lo: Bound | None
    hi: Bound | None

    @classmethod
    def point(cls, x: Decimal) -> NumInterval:
        return cls(Bound(x, CLOSED), Bound(x, CLOSED))

    @classmethod
    def from_op(cls, op: str, x: Decimal) -> NumInterval:
        return {
            "=": cls.point(x),
            ">": cls(Bound(x, OPEN), None),
            "≥": cls(Bound(x, CLOSED), None),
            "<": cls(None, Bound(x, OPEN)),
            "≤": cls(None, Bound(x, CLOSED)),
        }[op]

    @property
    def is_point(self) -> bool:
        return (
            self.lo is not None
            and self.hi is not None
            and self.lo.value == self.hi.value
            and self.lo.kind == self.hi.kind == CLOSED
        )

    def contains(self, x: Decimal) -> Membership:
        """Membership of a number; ``boundary`` when x equals an unspecified bound."""
        result: Membership = "yes"
        for b, below in ((self.lo, True), (self.hi, False)):
            if b is None:
                continue
            if x == b.value:
                if b.kind == OPEN:
                    return "no"
                if b.kind == UNSPECIFIED:
                    result = "boundary"
            elif (x < b.value) if below else (x > b.value):
                return "no"
        return result

    def relation(self, q: NumInterval) -> Relation:
        """How a query interval q relates to this (printed) interval: entirely inside,
        touching only at an unspecified bound, partly overlapping, or disjoint."""
        if q.is_point:
            m = self.contains(q.lo.value)  # type: ignore[union-attr]
            relation: dict[Membership, Relation] = {
                "yes": "within",
                "boundary": "boundary",
                "no": "disjoint",
            }
            return relation[m]
        if not _overlaps(self, q):
            return "disjoint"
        if _covers(self, q):
            return "within"
        return "overlap"

    def describe(self) -> str:
        if self.is_point:
            return f"= {self.lo.value}"  # type: ignore[union-attr]
        parts = []
        sym = {
            (True, CLOSED): "≥",
            (True, OPEN): ">",
            (True, UNSPECIFIED): "≥?",
            (False, CLOSED): "≤",
            (False, OPEN): "<",
            (False, UNSPECIFIED): "≤?",
        }
        if self.lo is not None:
            parts.append(f"{sym[(True, self.lo.kind)]} {self.lo.value}")
        if self.hi is not None:
            parts.append(f"{sym[(False, self.hi.kind)]} {self.hi.value}")
        return " and ".join(parts) or "any"


def _overlaps(a: NumInterval, b: NumInterval) -> bool:
    lo = _tighter(a.lo, b.lo, lower=True)
    hi = _tighter(a.hi, b.hi, lower=False)
    if lo is None or hi is None:
        return True
    if lo.value < hi.value:
        return True
    return lo.value == hi.value and OPEN not in (lo.kind, hi.kind)


def _tighter(a: Bound | None, b: Bound | None, *, lower: bool) -> Bound | None:
    if a is None:
        return b
    if b is None:
        return a
    if a.value != b.value:
        return (a if a.value > b.value else b) if lower else (a if a.value < b.value else b)
    return a if a.kind == OPEN else b


def _covers(outer: NumInterval, inner: NumInterval) -> bool:
    """Every number of inner is in outer (an unspecified outer end counts as covering)."""
    for o, i, lower in ((outer.lo, inner.lo, True), (outer.hi, inner.hi, False)):
        if o is None:
            continue
        if i is None:
            return False
        if i.value != o.value:
            if (i.value < o.value) if lower else (i.value > o.value):
                return False
        elif o.kind == OPEN and i.kind != OPEN:
            return False
    return True


# ------------------------------------------------------------------ units

# unit key -> printed/asked spellings (normalized). Keys are compared, never converted.
UNITS: dict[str, tuple[str, ...]] = {
    "percent": ("درصد", "در صد", "%", "٪"),
    "m2k/w": ("m2.k/w", "m2k/w", "m².k/w", "متر مربع کلوین بر وات"),
    "w/mk": ("w/m.k", "w/mk", "وات بر متر کلوین"),
    "w/m2k": ("w/m2.k", "w/m2k", "وات بر متر مربع کلوین"),
    "m2": ("متر مربع", "مترمربع", "متر مربعی", "مترمربعی", "متری", "m2"),
    "kg/m3": ("کیلوگرم بر متر مکعب", "کیلوگرم در متر مکعب", "kg/m3", "kg/m³"),
    "kw": ("کیلووات", "kw"),
    "w": ("وات", "w"),
    "kwh": ("کیلووات ساعت", "kwh"),
    "celsius": ("درجه سانتیگراد", "درجه سانتی گراد", "درجه", "°c"),
    "mm": ("میلی متر", "میلیمتر", "mm"),
    "cm": ("سانتی متر", "سانتیمتر", "سانت", "سانتی", "cm"),
}
_UNIT_ALT = sorted(((s, k) for k, forms in UNITS.items() for s in forms), key=lambda x: -len(x[0]))
_UNIT_RE = re.compile(
    r"\s*(" + "|".join(re.escape(s) for s, _ in _UNIT_ALT) + r")(?:ه|ی|اش|ش)?(?![ؠ-ۿa-z])"
)
_UNIT_BEFORE = re.compile(r"(%|٪)\s*$")
_EXCLUDED = re.compile(r"(?:به\s?غیر\s?از|بغیر\s?از|به\s?جز|بجز|غیر\s?از|جز)\s*\(?\s*(?:بین)?\s*$")


def unit_after(text: str, end: int) -> tuple[str | None, int]:
    m = _UNIT_RE.match(text, end)
    if not m:
        return None, end
    form = m[1]
    return next(k for s, k in _UNIT_ALT if s == form), m.end()


# ------------------------------------------------------------------ quantities in text


@dataclass(frozen=True)
class Quantity:
    """A numeric statement in text: its interval, unit key and character span."""

    interval: NumInterval
    unit: str | None
    start: int
    end: int
    raw: str
    numbers: tuple[Decimal, ...]
    ratio: tuple[Decimal, Decimal] | None = None  # «۱ به ۶»: compared as a pair, never divided
    negated: bool = False  # «به غیر از ۷۵ تا ۲۰۰»: an excluded interval


_GT_OPEN = r"(?:بیش\s?از|بیشتر\s?از|بالای|بالاتر\s?از|بزرگ\s?تر\s?از|افزون\s?بر|>)"
_LT_OPEN = r"(?:کمتر\s?از|کم\s?تر\s?از|زیر|پایین\s?تر\s?از|کوچک\s?تر\s?از|<)"
_GE = r"(?:حداقل|حد\s?اقل|دست\s?کم|بیشتر\s?یا\s?مساوی|بزرگ\s?تر\s?یا\s?مساوی|≥|>=)"
_LE = r"(?:حداکثر|حد\s?اکثر|کمتر\s?یا\s?مساوی|کوچک\s?تر\s?یا\s?مساوی|حداکثر\s?تا|≤|<=)"
_EQ = r"(?:دقیقا|برابر\s?با|برابر|معادل|=)"
_SUFFIX_GE = r"(?:\s*(?:و|یا)\s*(?:بیشتر|بالاتر|بزرگ\s?تر)|\s*به\s*بالا)"
_SUFFIX_LE = r"(?:\s*(?:و|یا)\s*(?:کمتر|پایین\s?تر|کوچک\s?تر)|\s*به\s*پایین)"
_N = NUMERAL
_PREFIX_UNIT = r"(?:[%٪]\s*)?"
_RANGE = re.compile(
    rf"(?:بین|مابین|از)?\s*{_PREFIX_UNIT}({_N})\s*(?:[%٪])?\s*(?:تا|الی|و|-|–|—)\s*{_PREFIX_UNIT}({_N})"
)
_OPS = (
    (re.compile(rf"{_GE}\s*{_PREFIX_UNIT}({_N})"), "≥"),
    (re.compile(rf"{_LE}\s*{_PREFIX_UNIT}({_N})"), "≤"),
    (re.compile(rf"{_GT_OPEN}\s*{_PREFIX_UNIT}({_N})"), ">"),
    (re.compile(rf"{_LT_OPEN}\s*{_PREFIX_UNIT}({_N})"), "<"),
    (re.compile(rf"{_EQ}\s*{_PREFIX_UNIT}({_N})"), "="),
)
_PLAIN = re.compile(rf"{_PREFIX_UNIT}({_N})")
_RATIO = re.compile(rf"({_N})\s*(?:به|:)\s*({_N})(?!\s*(?:درصد|%))")


def find_quantities(text: str, *, printed: bool = False) -> list[Quantity]:  # noqa: PLR0912, PLR0915
    """Every numeric statement in normalized text, left to right, non-overlapping.

    ``printed`` reads source wording: an «X تا Y» range has unspecified ends (the source does
    not state inclusivity). In a query a stated range is the closed set the user names."""
    taken: list[tuple[int, int]] = []
    out: list[Quantity] = []

    def free(a: int, b: int) -> bool:
        return all(b <= x or a >= y for x, y in taken)

    def add(  # noqa: PLR0913
        iv: NumInterval,
        a: int,
        b: int,
        raw: str,
        nums: tuple[Decimal, ...],
        *,
        unit: str | None = None,
        ratio: tuple[Decimal, Decimal] | None = None,
    ) -> None:
        end = b
        if unit is None:
            unit, end = unit_after(text, b)
        if unit is None and (re.search(r"[%٪]", raw) or _UNIT_BEFORE.search(text[:a])):
            unit = "percent"
        taken.append((a, end))
        negated = bool(_EXCLUDED.search(text[max(0, a - 16) : a]))
        out.append(Quantity(iv, unit, a, end, raw, nums, ratio, negated))

    range_kind: BoundKind = "unspecified" if printed else "closed"
    for m in _RATIO.finditer(text):
        a, b = parse_numeral(m[1]), parse_numeral(m[2])
        if a is not None and b is not None:
            add(
                NumInterval.point(a), m.start(1), m.end(2), m[0], (a, b), unit="ratio", ratio=(a, b)
            )
    for m in _RANGE.finditer(text):
        if not free(m.start(), m.end()):
            continue
        a, b = parse_numeral(m[1]), parse_numeral(m[2])
        word = text[m.start() : m.end()]
        if a is None or b is None or a >= b:
            continue
        # «۱ به ۶» is a ratio, «سال ۱۴۰۰ و ۱۴۰۱» two values: only explicit range words count
        if not re.search(r"تا|الی|-|–|—|بین|مابین", word):
            continue
        iv = NumInterval(Bound(a, range_kind), Bound(b, range_kind))
        add(iv, m.start(), m.end(), word, (a, b))
    for pattern, op in _OPS:
        for m in pattern.finditer(text):
            if not free(m.start(), m.end()):
                continue
            x = parse_numeral(m[1])
            if x is None:
                continue
            iv = NumInterval.from_op(op, x)
            end = m.end()
            add(iv, m.start(), end, text[m.start() : end], (x,))
    for m in _PLAIN.finditer(text):
        if not free(m.start(1), m.end(1)):
            continue
        x = parse_numeral(m[1])
        if x is None:
            continue
        tail = text[m.end() :]
        unit, uend = unit_after(text, m.end())
        rest = text[uend:]
        ge, le = re.match(_SUFFIX_GE, rest), re.match(_SUFFIX_LE, rest)
        if ge:
            iv, stop = NumInterval(Bound(x, CLOSED), None), uend + ge.end()
        elif le:
            iv, stop = NumInterval(None, Bound(x, CLOSED)), uend + le.end()
        else:
            iv, stop = NumInterval.point(x), m.end()
        del tail
        add(iv, m.start(1), stop, text[m.start(1) : stop], (x,), unit=unit if (ge or le) else None)
    out.sort(key=lambda q: q.start)
    return out


def printed_interval(text: str) -> NumInterval | None:
    """The interval a printed condition states («بیش از ۴۰», «۶۰۰ تا ۸۰۰», «۱۲۰۰»,
    «کمتر از %۲۰», «کوچک‌تر از ۲۰۰۰ متر مربع»), when the text is exactly one condition."""
    qs = find_quantities(text, printed=True)
    return qs[0].interval if len(qs) == 1 else None


def interval_from_operators(cond: dict[str, object]) -> NumInterval:
    """A curated condition with printed operators ({"lower": {"value", "operator"}, ...})."""

    def bound(side: object, lower: bool) -> Bound | None:
        if not isinstance(side, dict):
            return None
        value = Decimal(str(side["value"]))
        op = str(side.get("operator", ""))
        kind: BoundKind = "open" if op in ("<", ">") else "closed"
        del lower
        return Bound(value, kind)

    return NumInterval(bound(cond.get("lower"), True), bound(cond.get("upper"), False))
