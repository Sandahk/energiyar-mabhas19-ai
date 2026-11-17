"""B2: explicit extra quantity phrasings for the prototype (``api.quantities`` is unchanged).

Each addition is a written phrase rule, tested, and still deterministic; the quantity is never
inferred semantically:

* a bare «R» names the minimum thermal resistance (19-5-1 prints R);
* «آب گرم» names the hot-water temperature only when the question also asks degrees / temperature
  («چند درجه», «دما»);
* «چه … داره / دارد / داشته باشه» is a value request («تخته … چه ضریب هدایتی داره؟»);
* «برای محاسبه …» states a purpose, not a method question, so it does not make the question
  conceptual (method questions - «چگونه / چطور محاسبه» - still are).
"""

from __future__ import annotations

import re

from mabhas19.api import quantities as Q
from mabhas19.api.wording import letters

_S = r"[\s‌]*"
_S1 = r"[\s‌]+"
_B = r"(?<![\w‌])"
_E = r"(?![\w])"

# quantity id -> (extra cue, a phrase that must also be in the question, or "")
EXTRA: dict[str, tuple[tuple[str, str], ...]] = {
    "r_value": ((rf"{_B}R{_E}", ""),),
    "hot_water": ((rf"آب{_S}گرم(?!{_S}کن)", rf"درجه|{_B}دما"),),
}
VALUE_EXTRA = re.compile(rf"{_B}چه{_E}[^؟?.!]{{0,60}}?(?:داره|دارد|داشته{_S1}باش)")
PURPOSE = re.compile(rf"برای{_S1}محاسبه\w*")


def _quantity_matches(text: str) -> list[tuple[Q.Quantity, list[re.Match[str]]]]:
    out = []
    for q in Q.QUANTITIES:
        ms = list(q.pattern().finditer(text))
        for cue, needs in EXTRA.get(q.id, ()):
            if not needs or re.search(needs, text):
                ms += list(re.finditer(cue, text, re.IGNORECASE))
        if ms:
            out.append((q, ms))
    return out


def analyse(question: str) -> Q.Request | None:
    """``api.quantities.analyse`` with the explicit B2 phrasings above."""
    text = letters(question)
    if Q.CONCEPT.search(PURPOSE.sub(" ", text)):
        return None
    if not (Q.VALUE.search(text) or VALUE_EXTRA.search(text)):
        return None
    found = _quantity_matches(text)
    if len(found) != 1:
        return None
    quantity, matches = found[0]
    rest = text
    merged: list[list[int]] = []
    for start, end in sorted({m.span() for m in matches}):  # overlapping cues: one span
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    for start, end in reversed(merged):
        rest = rest[:start] + " " + rest[end:]
    rest = rest.translate(Q._DIGITS)
    codes = {f"{a}{b.upper()}" for a, b in Q._CODE.findall(rest)}
    if len(codes) > 1:
        return None
    rest = Q._CODE.sub(" ", rest)
    words = [w for w in Q._WORD.findall(rest.replace("‌", " ")) if w not in Q.FILLERS]
    return Q.Request(quantity, next(iter(codes), ""), tuple(words))


def extended_only(question: str) -> bool:
    """The quantity was resolved only thanks to a B2 phrasing (for the gain attribution)."""
    return Q.analyse(question) is None and analyse(question) is not None
