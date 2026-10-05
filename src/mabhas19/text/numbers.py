"""Safe parsing of numeric values read from regulatory sources.

Rules:

* the raw detected string is always kept;
* a value is never "corrected" by guessing - anything ambiguous is reported with low
  confidence and ``requires_verification``;
* digits are compared by value across scripts (ASCII, Persian, Arabic-Indic), but mixing
  scripts inside one number is flagged because it is a typical OCR confusion;
* ``/`` is accepted as a decimal separator only when the caller says the context is a
  numeric cell (the main PDF stores «۱٫۰۹» as ``1/09``); elsewhere it may be a date.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation

from mabhas19.text.charclass import ZWNJ, digit_value

DECIMAL_SEPARATORS = frozenset({".", "٫"})  # full stop, ARABIC DECIMAL SEPARATOR
SLASH = "/"
AMBIGUOUS_SEPARATORS = frozenset({",", "،"})  # comma, ARABIC COMMA
THOUSANDS_SEPARATORS = frozenset({"٬", "\u2009", "\u202f"})  # Arabic thousands, thin spaces
MINUS_SIGNS = frozenset({"-", "\u2212", "\u2010", "\u2011", "\u2012", "\u2013", "\ufe63", "\uff0d"})
_IGNORED = frozenset({" ", ZWNJ, "\u200e", "\u200f", "\u061c"})

HIGH, MEDIUM, LOW = "high", "medium", "low"
_RANK = {HIGH: 2, MEDIUM: 1, LOW: 0}


def _script(ch: str) -> str:
    cp = ord(ch)
    if 0x0660 <= cp <= 0x0669:
        return "arabic-indic"
    if 0x06F0 <= cp <= 0x06F9:
        return "persian"
    return "ascii"


@dataclass(frozen=True)
class NumericValue:
    raw_detected_value: str
    normalized_value: str | None  # ASCII digits, "." decimal point, digits kept as printed
    confidence: str  # "high" | "medium" | "low"
    issues: tuple[str, ...] = field(default_factory=tuple)

    @property
    def requires_verification(self) -> bool:
        return self.confidence != HIGH or self.normalized_value is None

    def as_decimal(self) -> Decimal | None:
        if self.normalized_value is None:
            return None
        try:
            return Decimal(self.normalized_value)
        except InvalidOperation:
            return None


def _lowest(*levels: str) -> str:
    return min(levels, key=lambda lvl: _RANK[lvl])


@dataclass
class _Scan:
    digits: list[str] = field(default_factory=list)
    scripts: set[str] = field(default_factory=set)
    separators: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    confidence: str = HIGH
    error: str | None = None


def _scan(body: str, slash_is_decimal: bool, scan: _Scan) -> _Scan:
    """Classify every character of a numeric token; stop at the first invalid one."""
    for ch in body:
        value = digit_value(ch)
        if value is not None:
            scan.digits.append(str(value))
            scan.scripts.add(_script(ch))
        elif ch in DECIMAL_SEPARATORS or (ch == SLASH and slash_is_decimal):
            scan.separators.append(ch)
            scan.digits.append(".")
        elif ch in THOUSANDS_SEPARATORS:
            scan.issues.append(f"thousands separator U+{ord(ch):04X} removed")
            scan.confidence = _lowest(scan.confidence, MEDIUM)
        elif ch in AMBIGUOUS_SEPARATORS:
            scan.error = f"ambiguous separator {ch!r}"
        elif ch == SLASH:
            scan.error = "'/' outside a numeric-cell context (date?)"
        else:
            scan.error = f"unexpected character {ch!r}"
        if scan.error:
            break
    return scan


def _structure_error(text: str, separators: list[str]) -> str | None:
    if len(separators) > 1:
        return "more than one decimal separator"
    if not any(c.isdigit() for c in text):
        return "no digits"
    return None


def parse_numeric(
    raw: str, *, slash_is_decimal: bool = False, source_confidence: str = HIGH
) -> NumericValue:
    """Parse one numeric token without guessing. See module docstring for the rules."""
    scan = _Scan(confidence=source_confidence)
    body = "".join(c for c in raw if c not in _IGNORED)
    marks = [c for c in body if unicodedata.category(c) == "Mn"]
    if marks:
        # Combining marks cannot belong to a number: OCR noise. Removing them does not
        # change any digit, but the value is no longer fully trustworthy.
        body = "".join(c for c in body if unicodedata.category(c) != "Mn")
        scan.issues.append(f"{len(marks)} stray combining mark(s) removed")
        scan.confidence = _lowest(scan.confidence, MEDIUM)
    if not body:
        return NumericValue(raw, None, LOW, ("empty",))

    sign = ""
    if body[0] in MINUS_SIGNS:
        sign = "-"
        if body[0] != "-":
            scan.issues.append(f"minus written as U+{ord(body[0]):04X}")
        body = body[1:]

    _scan(body, slash_is_decimal, scan)
    text = "".join(scan.digits)
    error = scan.error or _structure_error(text, scan.separators)
    if error:
        return NumericValue(raw, None, LOW, (*scan.issues, error))
    if text.startswith(".") or text.endswith("."):
        side = "before" if text.startswith(".") else "after"
        scan.issues.append(f"no digit {side} the decimal separator")
        scan.confidence = LOW
    if len(scan.scripts) > 1:
        scan.issues.append(f"mixed digit scripts: {sorted(scan.scripts)}")
        scan.confidence = _lowest(scan.confidence, MEDIUM)
    if scan.separators and scan.separators[0] == SLASH:
        scan.issues.append("'/' read as decimal separator (numeric-cell context)")
    return NumericValue(raw, sign + text, scan.confidence, tuple(scan.issues))


def same_value(a: str | None, b: str | None) -> bool:
    """Numeric equality of two normalized values (1.30 == 1.3); False if either is None."""
    if a is None or b is None:
        return False
    try:
        return Decimal(a) == Decimal(b)
    except InvalidOperation:
        return False
