"""Energy-rating expressions: one deterministic parser (M2).

A rating is stated by

* an explicit grade letter after «رده/رتبه/کلاس [بازدهی] [انرژی]», bare or in parentheses
  («رده C», «رده (D)», «A+», «A++»), possibly coordinated («رده‌های A و B»);
* a descriptive wording the source itself attaches to a printed grade. The vocabulary is read
  from the curated Mabhas 19 text, never written by hand: the definitions «رده C : ساختمان کم
  مصرف» and the enumeration «… در چهار رده D (…)، C (…)، B (…)، A (…)», and the printed phrase
  «حداقل رده بازدهی انرژی D». A description maps to a grade only through such a printed pair;
  a description printed for two grades maps to none.

When both a description and a letter are stated they must agree («رده کم‌مصرف (C)»); a
disagreement («رده کم‌مصرف (B)», «حداقل رده C») is a typed conflict and the constraint is invalid
(nothing is returned). «حداقل رده» names a grade only together with its letter.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import cache

from mabhas19.text.normalize import normalize_for_search

_DOC = "m19-ed5-1404-main"
_LETTER = r"[a-z]\+{0,3}"
_HEAD = re.compile(r"(?<![ؠ-ۿ])(?:رده|رتبه|کلاس)(?:\s*های)?\s*(?:بازدهی\s*)?(?:انرژی\s*)?")
_GRADE = re.compile(rf"\s*\(?\s*({_LETTER})\s*\)?(?![a-z0-9+])")
_MORE = re.compile(rf"\s*(?:و|،|,|یا)\s*\(?\s*({_LETTER})\s*\)?(?![a-z0-9+])")
_DEFINITION = re.compile(rf"^رده\s*({_LETTER})\s*:\s*ساختمان\s+(.+?)\s*$")
_ENUMERATED = re.compile(rf"(?<![a-z])({_LETTER})\s*\(\s*([^()]+?)\s*\)")
_MINIMUM = re.compile(rf"حداقل\s+رده\s+(?:بازدهی\s*)?(?:انرژی\s*)?({_LETTER})(?![a-z0-9+])")
MINIMUM = "حداقل رده"


@dataclass(frozen=True)
class Attestation:
    letter: str
    block: str  # curated block the printed pair is read from


@dataclass
class RatingMention:
    """One stated rating: its grades, query span, basis and any conflict."""

    letters: list[str]
    start: int
    end: int
    text: str
    basis: str  # letter | description | letter+description
    description: str | None = None
    attested_in: list[str] = field(default_factory=list)
    conflict: str | None = None


@cache
def vocabulary() -> dict[str, tuple[Attestation, ...]]:
    """Source-attested descriptive wording -> the printed grade(s) it is paired with."""
    from mabhas19.core.paths import CURATED_DIR  # noqa: PLC0415

    path = CURATED_DIR / "text" / _DOC / "blocks.jsonl"
    pairs: dict[str, set[Attestation]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        b = json.loads(line)
        if not (b.get("retrieval") or {}).get("include", True):
            continue
        text = normalize_for_search(str(b.get("text") or "")).strip()
        found: list[tuple[str, str]] = []
        if m := _DEFINITION.match(text):
            found.append((m[2], m[1]))
        if "چهار رده" in text:
            found += [(m[2], m[1]) for m in _ENUMERATED.finditer(text)]
        found += [(MINIMUM, m[1]) for m in _MINIMUM.finditer(text)]
        for printed, letter in found:
            desc = re.sub(r"\s+", " ", printed).strip()
            pairs.setdefault(desc, set()).add(Attestation(letter.upper(), str(b["block_id"])))
    return {d: tuple(sorted(a, key=lambda x: x.block)) for d, a in sorted(pairs.items())}


def _grade_of(desc: str) -> tuple[str | None, list[str]]:
    """The single printed grade a description is paired with (None if several), blocks."""
    att = vocabulary().get(desc, ())
    letters = {a.letter for a in att}
    return (next(iter(letters)) if len(letters) == 1 else None), [a.block for a in att]


def parse(norm: str) -> list[RatingMention]:
    """Every rating stated in a normalized query."""
    out: list[RatingMention] = []
    descriptions = sorted((d for d in vocabulary() if d != MINIMUM), key=len, reverse=True)
    for m in _HEAD.finditer(norm):
        pos = m.end()
        desc = None
        for d in descriptions:
            dm = re.match(rf"(?:ساختمان\s*)?{re.escape(d)}(?![ؠ-ۿ])", norm[pos:])
            if dm:
                desc, pos = d, pos + dm.end()
                break
        letters: list[str] = []
        if gm := _GRADE.match(norm, pos):
            letters.append(gm[1].upper())
            pos = gm.end()
            while mm := _MORE.match(norm, pos):
                letters.append(mm[1].upper())
                pos = mm.end()
        minimum = norm[: m.start()].rstrip().endswith("حداقل")
        if not letters and not desc:
            continue
        start = norm.rfind("حداقل", 0, m.start()) if minimum else m.start()
        mention = RatingMention(letters, start, pos, norm[start:pos].strip(), "letter")
        for named, active in ((desc, bool(desc)), (MINIMUM, minimum and bool(letters))):
            if not active or named is None:
                continue
            grade, blocks = _grade_of(named)
            mention.description = named if named != MINIMUM else (desc or MINIMUM)
            mention.attested_in += blocks
            if grade is None:
                mention.conflict = f"«{named}» is printed for no single grade"
            elif letters and set(letters) != {grade}:
                mention.conflict = f"«{named}» is printed for grade {grade}, not {letters}"
            elif not letters:
                mention.letters = [grade]
            mention.basis = "letter+description" if letters else "description"
        out.append(mention)
    return out


def letters(norm: str) -> list[str]:
    """The grades stated (conflicting mentions contribute none)."""
    return sorted({x for m in parse(norm) if m.conflict is None for x in m.letters})


def conflicts(norm: str) -> list[str]:
    return [f"{m.text}: {m.conflict}" for m in parse(norm) if m.conflict]
