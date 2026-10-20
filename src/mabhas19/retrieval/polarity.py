"""Typed negation and contrast parsing for structured queries (read before any flattening).

Two different things are written with negative words in Persian and must never be confused:

* a **negative-form qualifier** names a printed row: «بدون تهویه», «غیر چگالشی», «بدون مخزن»
  select the rows printing that form. It is *required*, like any positive qualifier;
* an **exclusion** removes a reading: «نه X», «به جز X», «جز X», «غیر از X», «بغیر از X»,
  «منهای X», «(نه X)», and the contrast «X، نه Y» / «X نه Y». The excluded phrase must not be
  satisfied by any returned row.

So «سوله با تهویه (نه بدون تهویه)» parses to required «با تهویه» and excluded «بدون تهویه».
Parentheses are interpreted *before* they are removed: an exclusion inside parentheses ends at
the closing parenthesis. The parser returns the positive text (exclusions removed) on which
entities are resolved, and the typed spans for admission and certification.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# exclusion markers (longest first); «غیر از» excludes, «غیر X» (no «از») is a qualifier form
_EXCLUDE = r"(?:به\s?غیر\s?از|بغیر\s?از|غیر\s?از|به\s?جز|بجز|منهای|جز|نه)"
# outside parentheses an exclusion runs to the next clause break, at most six words
_BARE = re.compile(
    rf"(?<![ؠ-ۿ]){_EXCLUDE}\s+((?:[^\s()،,؛;؟?.!]+\s*){{1,6}}?)(?=[()،,؛;؟?.!]|$|\s+و\s)"
)
_STOP = re.compile(r"\s+(?:در|برای|تو|توی|با|به|که|از|چه|چند|چقدر|حداقل|حداکثر)\s.*$")
# negative-form qualifiers that name printed rows
NEGATIVE_FORMS = ("بدون", "غیر", "فاقد")


@dataclass(frozen=True)
class Span:
    text: str
    start: int
    end: int


@dataclass
class Polarity:
    """Positive text (exclusions removed) and the typed exclusion / qualifier spans."""

    positive: str
    exclusions: list[Span] = field(default_factory=list)
    negative_forms: list[Span] = field(default_factory=list)  # «بدون X», «غیر X» (required)
    negated: list[Span] = field(default_factory=list)  # the whole X of each negative form


def _trim(phrase: str) -> str:
    """An unparenthesised exclusion stops before a preposition or question word that starts the
    rest of the sentence («نه بدون تهویه در اقلیم …» excludes «بدون تهویه»)."""
    return _STOP.sub("", phrase).strip()


_PAREN_OPEN = re.compile(rf"\(\s*{_EXCLUDE}\s+")


def _paren_exclusions(text: str) -> list[tuple[int, int, int]]:
    """(paren start, phrase start, paren end) of every «( نه … )», nested parentheses balanced."""
    out = []
    for m in _PAREN_OPEN.finditer(text):
        depth, i = 1, m.end()
        while i < len(text) and depth:
            depth += {"(": 1, ")": -1}.get(text[i], 0)
            i += 1
        if depth == 0:
            out.append((m.start(), m.end(), i))
    # an exclusion nested inside another is part of the outer one
    return [x for x in out if not any(o[0] < x[0] and x[2] <= o[2] for o in out)]


def parse(text: str) -> Polarity:
    exclusions: list[Span] = []
    parts = _paren_exclusions(text)
    for _start, phrase, end in parts:
        exclusions.append(Span(text[phrase : end - 1].strip(), phrase, end - 1))
    out = text
    for start, _phrase, end in reversed(parts):
        out = out[:start] + " " + out[end:]
    for m in _BARE.finditer(out):
        bare = _trim(m[1])
        if bare:
            at = text.find(bare)
            exclusions.append(Span(bare, at, at + len(bare)))
    positive = _BARE.sub(lambda m: " " + m[1][len(_trim(m[1])) :] + " ", out)
    positive = re.sub(r"\s+", " ", positive).strip()
    forms, negated = [], []
    for m in _FORM.finditer(positive):
        forms.append(Span(m[0], m.start(), m.end()))
        scope = _SCOPE.match(positive, m.end() - len(m[1]))
        what = _trim(scope[1]) if scope else m[1]
        at = m.end() - len(m[1])
        negated.append(Span(what, at, at + len(what)))
    return Polarity(positive, exclusions, forms, negated)


_FORM = re.compile(r"(?<![ؠ-ۿ])(?:بدون|غیر|فاقد)\s?(?!از)([ؠ-ۿ]+)")
# what a negative form negates runs to the next clause break or parenthesis, at most six words
_SCOPE = re.compile(r"((?:[^\s()،,؛;؟?.!]+\s*){1,6}?)(?=[()،,؛;؟?.!]|$|\s+و\s)")
