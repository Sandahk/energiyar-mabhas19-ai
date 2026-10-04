"""Detection of Mabhas 19 section numbers such as «19-5-1-2» in raw extracted text.

Section numbers in the source PDFs may use any hyphen variant (notably U+2011) and,
because of right-to-left layout, may be extracted in *visual* order («2-1-5-19»). The
canonical form always starts with the chapter number 19 and uses ASCII hyphens.
The input text is never modified.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from mabhas19.text.charclass import HYPHEN_VARIANTS, digits_to_int

PART_NUMBER = 19
MAX_COMPONENT_DIGITS = 3
MAX_DEPTH = 6

_DIGIT_CLASS = "0-9\u0660-\u0669\u06f0-\u06f9"
_HYPHEN_CLASS = "".join(re.escape(h) for h in sorted(HYPHEN_VARIANTS))
# Digit groups joined by hyphen variants, optionally padded with spaces or ZWNJ.
_SEPARATOR = rf"[ \u200c]*[{_HYPHEN_CLASS}][ \u200c]*"
_GROUP = rf"[{_DIGIT_CLASS}]{{1,{MAX_COMPONENT_DIGITS}}}"
_NUMBER_PATTERN = re.compile(
    rf"(?<![{_DIGIT_CLASS}]){_GROUP}"
    rf"(?:{_SEPARATOR}{_GROUP}){{1,{MAX_DEPTH - 1}}}"
    rf"(?![{_DIGIT_CLASS}])"
)
_SPLIT = re.compile(_SEPARATOR)


def canonical_from_components(components: Sequence[int]) -> str | None:
    """Canonical «19-a-b» form from numeric components in reading order or visual order.

    Returns None unless exactly one end of the sequence is the part number 19 and the
    depth is plausible.
    """
    parts = list(components)
    if not 2 <= len(parts) <= MAX_DEPTH:
        return None
    if parts[0] == PART_NUMBER:
        ordered = parts
    elif parts[-1] == PART_NUMBER:
        ordered = parts[::-1]  # extracted in visual (left-to-right) order
    else:
        return None
    if any(p < 0 or p > 10**MAX_COMPONENT_DIGITS - 1 for p in ordered):
        return None
    return "-".join(str(p) for p in ordered)


def parse_section_number(raw: str) -> str | None:
    """Canonical section number for a raw token like «19\u20115\u20111\u20112» or «2-1-5-19»."""
    pieces = _SPLIT.split(raw.strip())
    values = [digits_to_int(p) for p in pieces]
    if any(v is None for v in values):
        return None
    return canonical_from_components([v for v in values if v is not None])


def find_section_numbers(text: str) -> list[tuple[str, str]]:
    """All (raw match, canonical number) pairs found in text, in order of appearance."""
    found: list[tuple[str, str]] = []
    for m in _NUMBER_PATTERN.finditer(text):
        canonical = parse_section_number(m.group(0))
        if canonical is not None:
            found.append((m.group(0), canonical))
    return found
