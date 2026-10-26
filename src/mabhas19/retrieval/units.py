"""Unit expressions: canonical form and compatibility (no conversion is ever performed).

A unit expression is read as numerator and denominator *atoms*: «kWh/m2.yr» -> ({kwh}, {m2,
yr}), «کیلووات ساعت بر متر مربع در سال» -> the same, «W/m.K» -> ({w}, {m, k}). Atoms are the
unit keys of :data:`mabhas19.retrieval.numeric.UNITS` (their Persian and Latin printed forms),
their SI parts (kg, m, m3, k), and a year marker; anything else is a *foreign* atom (kBtu, ft²,
lb…) - read only so it can be refused, never converted.

A requested unit is *compatible* with a printed unit when every requested atom sits on the same
side of the printed unit («kWh» or «kWh/m²» for a printed «kWh/m2.yr»); a foreign atom, an atom
on the other side or another quantity («kW» for «kWh») is not.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from mabhas19.retrieval.numeric import UNITS
from mabhas19.text.normalize import normalize_for_search

# SI parts of compound unit keys and spelled variants -> atom
_ATOMS: dict[str, str] = {
    "kwh": "kwh",
    "kw": "kw",
    "w": "w",
    "m2": "m2",
    "m²": "m2",
    "m3": "m3",
    "m³": "m3",
    "m": "m",
    "k": "k",
    "kg": "kg",
    "mm": "mm",
    "cm": "cm",
    "yr": "yr",
    "year": "yr",
    "a": "yr",
    "celsius": "celsius",
    "°c": "celsius",
    "c": "celsius",
    "percent": "percent",
    "%": "percent",
}
# joined atoms of the compound unit keys («w/mk», «m2k/w»)
_JOINED = {"mk": ("m", "k"), "m2k": ("m2", "k")}
KNOWN_ATOMS = frozenset(_ATOMS.values())
# Persian printed unit forms (longest first) -> their key; «بر» divides, «در سال» is per year
_PERSIAN = sorted(
    (
        (normalize_for_search(f), k)
        for k, forms in UNITS.items()
        for f in forms
        if re.search(r"[ؠ-ۿ]", f)
    ),
    key=lambda x: -len(x[0]),
)
_PER_YEAR = re.compile(r"\s*(?:در\s*سال|سالانه|سالیانه|بر\s*سال)\s*$")


@dataclass(frozen=True)
class Unit:
    num: frozenset[str]
    den: frozenset[str]
    text: str

    @property
    def foreign(self) -> frozenset[str]:
        return (self.num | self.den) - KNOWN_ATOMS

    def describe(self) -> str:
        return "/".join(
            ["·".join(sorted(self.num)) or "1", *(["·".join(sorted(self.den))] if self.den else [])]
        )


def _atoms(part: str) -> list[str]:
    out = []
    for raw in re.split(r"[.\s·*×\-]+", part.strip()):
        if not raw:
            continue
        out += list(_JOINED.get(raw, (_ATOMS.get(raw, raw),)))
    return out


def parse(text: str) -> Unit | None:
    """The canonical unit of a unit expression, or None when it is not one."""
    t = normalize_for_search(text).lower().replace("²", "2").replace("³", "3").strip(" .،,؟?")
    if not t:
        return None
    per_year = bool(_PER_YEAR.search(t))
    t = _PER_YEAR.sub("", t)
    for form, key in _PERSIAN:
        t = re.sub(rf"(?<![ؠ-ۿ]){re.escape(form)}(?:ی)?(?![ؠ-ۿ])", f" {key} ", t)
    t = re.sub(r"(?<![ؠ-ۿ])بر(?![ؠ-ۿ])", "/", t)
    if re.search(r"[ؠ-ۿ]", t):
        return None  # an unknown Persian word: not a unit expression this parser can read
    num_part, _, den_part = t.partition("/")
    num, den = _atoms(num_part), _atoms(den_part.replace("/", "."))
    # a compound key read as one atom («kg/m3», «w/m.k») is split into its parts
    expanded_num: list[str] = []
    expanded_den: list[str] = []
    for a in num:
        if "/" in a:
            n, d = a.split("/", 1)
            expanded_num += _atoms(n)
            expanded_den += _atoms(d)
        else:
            expanded_num.append(a)
    expanded_den += den
    if per_year:
        expanded_den.append("yr")
    if not expanded_num and not expanded_den:
        return None
    return Unit(frozenset(expanded_num), frozenset(expanded_den), text.strip())


def compatible(requested: Unit, printed: Unit) -> bool:
    """Every requested atom is a printed atom on the same side (no conversion exists)."""
    if requested.foreign or not requested.num:
        return False
    return requested.num <= printed.num and requested.den <= printed.den


def same(a: Unit, b: Unit) -> bool:
    return a.num == b.num and a.den == b.den


__all__ = ["KNOWN_ATOMS", "Unit", "compatible", "parse", "same"]
