"""Typed negation and contrast parsing for structured queries (read before any flattening).

Two different things are written with negative words in Persian and must never be confused:

* a **negative-form qualifier** names a printed row: «بدون تهویه», «غیر چگالشی», «بدون مخزن»
  select the rows printing that form. It is *required*, like any positive qualifier;
* an **exclusion** removes a reading: «نه X», «به جز X», «جز X», «غیر از X», «بغیر از X»,
  «منهای X», «(نه X)», and the contrast «X، نه Y» / «X نه Y». The excluded phrase must not be
  satisfied by any returned row.

A third construct replaces one reading with another: a **correction** (self-correction or
contrast) - «A نه، B», «نه A، B», «A نه بلکه B», «A نیست، B است» - has a *rejected* span A and an
*accepted* replacement B. It is typed only (the positive text and the exclusions are unchanged);
the plan-coverage stage owns the values: a value stated only inside a rejected span never
becomes an active constraint. Four surface patterns, no discourse interpretation.

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
# outside parentheses an exclusion runs to the next clause break - a dash between spaces («—»,
# normalized to « - ») included, never the hyphen of «19-5-1» or «1-2» - at most six words
_BARE = re.compile(
    rf"(?<![ؠ-ۿ]){_EXCLUDE}\s+((?:[^\s()،,؛;؟?.!—–]+\s*){{1,6}}?)(?=[()،,؛;؟?.!—–]|$|\s+و\s|\s*-\s)"
)
_STOP = re.compile(r"\s+(?:در|برای|تو|توی|با|به|که|از|چه|چند|چقدر|حداقل|حداکثر)\s.*$")
# negative-form qualifiers that name printed rows
NEGATIVE_FORMS = ("بدون", "غیر", "فاقد")


@dataclass(frozen=True)
class Span:
    text: str
    start: int
    end: int


@dataclass(frozen=True)
class Correction:
    """A rejected reading and its accepted replacement (spans in the parsed text)."""

    pattern: str  # «A نه، B» | «نه A، B» | «A نه بلکه B» | «A نیست، B است»
    rejected: Span
    accepted: Span


@dataclass
class Polarity:
    """Positive text (exclusions removed) and the typed exclusion / qualifier spans."""

    positive: str
    exclusions: list[Span] = field(default_factory=list)
    negative_forms: list[Span] = field(default_factory=list)  # «بدون X», «غیر X» (required)
    negated: list[Span] = field(default_factory=list)  # the whole X of each negative form
    corrections: list[Correction] = field(default_factory=list)


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
    return Polarity(positive, exclusions, forms, negated, corrections(text))


_FORM = re.compile(r"(?<![ؠ-ۿ])(?:بدون|غیر|فاقد)\s?(?!از)([ؠ-ۿ]+)")
# what a negative form negates runs to the next clause break or parenthesis, at most six words
_SCOPE = re.compile(r"((?:[^\s()،,؛;؟?.!—–]+\s*){1,6}?)(?=[()،,؛;؟?.!—–]|$|\s+و\s|\s*-\s)")


# ------------------------------------------------------------------ corrections

_BREAK = "()،,؛;؟?.!—–"
_WORD = re.compile(rf"[^\s{_BREAK}]+")
# a rejected phrase starts after the last preposition before the marker («در اقلیم 4B نه، …»)
_PREPOSITION = frozenset({"در", "برای", "تو", "توی", "با", "به", "از", "که"})
# «A نه، B» and «A نه بلکه B» (the marker follows A)
_AFTER_A = re.compile(r"(?<![ؠ-ۿ])نه\s*(?:[،,]\s*(?:بلکه\s+)?|[،,]?\s*بلکه\s+)")
# «نه A، B» and «نه A بلکه B» (the marker precedes A; A is at most four words)
_BEFORE_A = re.compile(
    rf"(?<![ؠ-ۿ])نه\s+((?:[^\s{_BREAK}]+\s*){{1,4}}?)\s*(?:[،,]\s*(?:بلکه\s+)?|بلکه\s+)"
)
# «A نیست، B است» (B is closed by the copula)
_NOT_A = re.compile(r"(?<![ؠ-ۿ])نیست\s*[،,]?\s*(?:بلکه\s+)?")
_COPULA = re.compile(r"\s+(?:است|هست)(?![ؠ-ۿ])")


def _before(text: str, end: int) -> Span | None:
    """The rejected phrase right before a marker: at most four words back to a clause break,
    after the last preposition; none after the alternative «یا» («… یا نه، …» asks a yes/no)."""
    start = max((text.rfind(c, 0, end) for c in _BREAK), default=-1) + 1
    words = list(_WORD.finditer(text, start, end))[-4:]
    if not words or words[-1][0] == "یا":
        return None
    cut = [i for i, w in enumerate(words) if w[0] in _PREPOSITION]
    words = words[cut[-1] + 1 :] if cut else words
    if not words:
        return None
    return Span(text[words[0].start() : words[-1].end()], words[0].start(), words[-1].end())


def _after(text: str, start: int, end: int | None = None) -> Span | None:
    """The accepted phrase after a marker: up to the next clause break (or ``end``), at most six
    words, cut before a later preposition that starts the rest of the sentence."""
    stop = next((i for i, c in enumerate(text[start:], start) if c in _BREAK), len(text))
    stop = min(stop, end) if end is not None else stop
    words = [w for w in _WORD.finditer(text, start, stop)][:6]
    cut = [i for i, w in enumerate(words) if i and w[0] in _PREPOSITION]
    words = words[: cut[0]] if cut else words
    if not words:
        return None
    return Span(text[words[0].start() : words[-1].end()], words[0].start(), words[-1].end())


def corrections(text: str) -> list[Correction]:
    """Every correction construct of the text (see the module docstring)."""
    out: list[Correction] = []
    for m in _AFTER_A.finditer(text):
        a, b = _before(text, m.start()), _after(text, m.end())
        if a and b:
            pattern = "A نه بلکه B" if "بلکه" in m[0] else "A نه، B"
            out.append(Correction(pattern, a, b))
    for m in _BEFORE_A.finditer(text):
        a_text = m[1].strip()
        b = _after(text, m.end())
        if a_text and b and not any(c.rejected.start <= m.start(1) < c.rejected.end for c in out):
            at = m.start(1)
            out.append(Correction("نه A، B", Span(a_text, at, at + len(a_text)), b))
    for m in _NOT_A.finditer(text):
        a = _before(text, m.start())
        cop = _COPULA.search(text, m.end())
        b = _after(text, m.end(), cop.start()) if cop is not None else None
        # B is closed by its copula within the same clause
        if (
            a
            and b
            and cop is not None
            and not any(c in _BREAK for c in text[m.end() : cop.start()])
        ):
            out.append(Correction("A نیست، B است", a, b))
    return out
