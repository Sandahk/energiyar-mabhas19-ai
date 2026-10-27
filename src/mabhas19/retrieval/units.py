"""Unit expressions: dimensional atoms, basis terms and a three-way verdict (never a conversion).

A unit expression is a product of *atoms* over a product of *basis* atoms: «kWh/m2.yr»,
«کیلووات ساعت بر متر مربع در سال» and «kWh بر متر مربع در سال» are all ({kwh}, {m2, yr});
«W/m.K» is ({w}, {m, k}); «متر مکعب گاز در سال برای هر متر مربع» is ({m3_gas}, {yr, m2}).
Every atom has a physical dimension - an exponent vector over mass, length, time, temperature,
percent and currency - so energy, power, length, area, volume, mass, time, temperature,
pressure, percent and currency are told apart (a volume of a named fuel measures energy through a
heating value). Connectors are «/», «بر», «per», «در», «به ازای (هر)», «برای هر» and «هر»;
temporal bases are year, month, day and hour in Latin and Persian forms, including the adverbs
«سالانه», «ماهانه», «روزانه». The lexicon is a closed list of physical units, not regulatory
vocabulary.

A requested unit compared with a printed unit (:func:`verdict`):

* **ground** - every requested atom is a printed atom on the same side («kWh» or «kWh/m²» for a
  printed «kWh/m2.yr»): the request names the printed unit;
* **conversion** - the same physical quantity in other units or on another basis (MJ, kBtu/ft²,
  kWh per month, m³ of gas per year): ``unsupported_calculation`` (``unit_conversion``);
* **other_quantity** - another physical dimension (MPa for a conductivity, kW for an energy):
  ``unsupported_output``.

Atoms the source never prints (Btu, ft, lb, MJ…) are *foreign*: they are read only so they can be
refused.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from mabhas19.retrieval.numeric import UNITS
from mabhas19.text.normalize import normalize_for_search

# dimension vectors (mass, length, time, temperature, percent, currency)
_E = (1, 2, -2, 0, 0, 0)  # energy
_P = (1, 2, -3, 0, 0, 0)  # power
_L = (0, 1, 0, 0, 0, 0)
_A = (0, 2, 0, 0, 0, 0)
_V = (0, 3, 0, 0, 0, 0)
_M = (1, 0, 0, 0, 0, 0)
_T = (0, 0, 1, 0, 0, 0)
_K = (0, 0, 0, 1, 0, 0)
_PA = (1, -1, -2, 0, 0, 0)  # pressure
_PCT = (0, 0, 0, 0, 1, 0)
_CUR = (0, 0, 0, 0, 0, 1)
KINDS: dict[tuple[int, ...], str] = {
    _E: "energy",
    _P: "power",
    _L: "length",
    _A: "area",
    _V: "volume",
    _M: "mass",
    _T: "time",
    _K: "temperature",
    _PA: "pressure",
    _PCT: "percent",
    _CUR: "currency",
}

# atom -> (dimension, Latin forms, Persian forms)
LEXICON: dict[str, tuple[tuple[int, ...], tuple[str, ...], tuple[str, ...]]] = {
    "kwh": (_E, ("kwh",), ("کیلووات ساعت", "کیلو وات ساعت")),
    "wh": (_E, ("wh",), ("وات ساعت",)),
    "mwh": (_E, ("mwh",), ("مگاوات ساعت", "مگا وات ساعت")),
    "j": (_E, (), ("ژول",)),
    "kj": (_E, ("kj",), ("کیلوژول", "کیلو ژول")),
    "mj": (_E, ("mj",), ("مگاژول", "مگا ژول")),
    "gj": (_E, ("gj",), ("گیگاژول", "گیگا ژول")),
    "btu": (_E, ("btu",), ("بی تی یو",)),
    "kbtu": (_E, ("kbtu",), ()),
    "therm": (_E, ("therm", "therms"), ()),
    "kcal": (_E, ("kcal",), ("کیلوکالری", "کیلو کالری")),
    "cal": (_E, ("cal",), ("کالری",)),
    # a volume or mass of a named fuel measures energy (through its heating value)
    "m3_gas": (_E, (), ("متر مکعب گاز", "مترمکعب گاز")),
    "l_fuel": (_E, (), ("لیتر گازوئیل", "لیتر نفت", "لیتر سوخت", "لیتر بنزین")),
    "kw": (_P, ("kw",), ("کیلووات", "کیلو وات")),
    "w": (_P, ("w",), ("وات",)),
    "mw": (_P, ("mw",), ("مگاوات", "مگا وات")),
    "hp": (_P, ("hp",), ("اسب بخار",)),
    "m": (_L, ("m",), ("متر",)),
    "cm": (_L, ("cm",), ("سانتیمتر", "سانتی متر")),
    "mm": (_L, ("mm",), ("میلیمتر", "میلی متر")),
    "km": (_L, ("km",), ("کیلومتر",)),
    "ft": (_L, ("ft",), ("فوت",)),
    "in": (_L, ("in", "inch"), ("اینچ",)),
    "m2": (_A, ("m2", "sqm"), ("متر مربع", "مترمربع")),
    "cm2": (_A, ("cm2",), ("سانتیمتر مربع",)),
    "ft2": (_A, ("ft2", "sqft"), ("فوت مربع",)),
    "m3": (_V, ("m3",), ("متر مکعب", "مترمکعب")),
    "l": (_V, ("l", "lit"), ("لیتر",)),
    "ft3": (_V, ("ft3", "cf"), ("فوت مکعب",)),
    "kg": (_M, ("kg",), ("کیلوگرم", "کیلو گرم", "کیلو")),
    "g": (_M, ("g", "gr"), ("گرم",)),
    "lb": (_M, ("lb", "lbs"), ("پوند",)),
    "yr": (_T, ("yr", "year", "a"), ("سال",)),
    "month": (_T, ("month", "mo"), ("ماه",)),
    "day": (_T, ("day", "d"), ("روز",)),
    "h": (_T, ("h", "hr"), ("ساعت",)),
    "k": (_K, ("k",), ("کلوین",)),
    "celsius": (
        _K,
        ("°c", "celsius", "c"),
        ("درجه سانتیگراد", "درجه سانتی گراد", "درجه سلسیوس", "درجه"),
    ),
    "fahrenheit": (_K, ("°f", "f"), ("فارنهایت", "درجه فارنهایت")),
    "pa": (_PA, ("pa",), ("پاسکال",)),
    "kpa": (_PA, ("kpa",), ("کیلوپاسکال", "کیلو پاسکال")),
    "mpa": (_PA, ("mpa",), ("مگاپاسکال", "مگا پاسکال")),
    "psi": (_PA, ("psi",), ()),
    "bar": (_PA, ("bar",), ()),
    "percent": (_PCT, ("%", "٪", "percent"), ("درصد", "در صد")),
    "rial": (_CUR, (), ("ریال",)),
    "toman": (_CUR, (), ("تومان", "تومن")),
    "usd": (_CUR, ("$", "usd"), ("دلار",)),
    "eur": (_CUR, ("€", "eur"), ("یورو",)),
}
# joined Latin keys of compound units («w/mk», «m2k/w») and their atoms
_JOINED = {"mk": ("m", "k"), "m2k": ("m2", "k")}
# a Persian form that alone is no unit request (a year, a month, «آب گرم», «درجه روز»…): read
# only inside a compound, after a connector or attached to a number
_AMBIGUOUS = frozenset(
    {"سال", "ماه", "روز", "ساعت", "متر", "گرم", "درجه", "وات", "بار", "تن", "کالری", "لیتر"}
)
# per-time adverbs (a basis of their own)
_ADVERBS = {
    "سالانه": "yr",
    "سالیانه": "yr",
    "ماهانه": "month",
    "ماهیانه": "month",
    "روزانه": "day",
}
_LATIN_ATOMS = {f: a for a, (_, latin, _) in LEXICON.items() for f in latin}
_PERSIAN = sorted(
    ((normalize_for_search(f), a) for a, (_, _, fa) in LEXICON.items() for f in fa),
    key=lambda x: -len(x[0]),
)
# Persian unit words take a colloquial copula or ezafe («مگاپاسکاله», «متر مربعی»)
_SUFFIX = r"(?:ه|ی|اش|ش|های|ها)?"
_PERSIAN_RE = re.compile(
    r"(?<![ؠ-ۿ])(" + "|".join(re.escape(f).replace(r"\ ", r"[\s‌]?") for f, _ in _PERSIAN) + r")"
    + _SUFFIX + r"(?![ؠ-ۿ])"
)  # fmt: skip
_LATIN_RE = re.compile(
    r"(?<![a-z0-9])([a-z°%٪$€]+[23²³]?(?:\s?[/·.*×]\s?[a-z°]+[23²³]?)*)(?![a-z0-9])"
)
_CONNECTOR = re.compile(
    r"\s*(?:/|(?<![ؠ-ۿ])(?:به\s?ازای\s?(?:هر\s?)?|برای\s?هر|بر\s?(?:هر\s?)?|در\s?(?:هر\s?)?|هر|per)(?![ؠ-ۿa-z]))\s*"
)
_ADVERB_RE = re.compile(r"\s*(" + "|".join(_ADVERBS) + r")(?![ؠ-ۿ])")
_ORDINAL = re.compile(r"\s*(?:اول|یکم|دوم|سوم|چهارم|پنجم|ششم|هفتم|\d)")
_NUMBER_BEFORE = re.compile(r"(?:\d|[%٪])\s*$")


def _canon(atom: str) -> tuple[int, ...]:
    return LEXICON[atom][0]


@dataclass(frozen=True)
class Unit:
    num: frozenset[str]
    den: frozenset[str]
    text: str

    @property
    def foreign(self) -> frozenset[str]:
        """Atoms the source never prints (read only to be refused)."""
        return (self.num | self.den) - SOURCE_ATOMS

    @property
    def dimension(self) -> tuple[int, ...]:
        vec = [0] * 6
        for a in self.num:
            vec = [x + y for x, y in zip(vec, _canon(a), strict=True)]
        for a in self.den:
            vec = [x - y for x, y in zip(vec, _canon(a), strict=True)]
        return tuple(vec)

    def kinds(self, side: frozenset[str]) -> Counter[str]:
        return Counter(KINDS.get(_canon(a), "other") for a in side)

    def describe(self) -> str:
        return "/".join(
            ["·".join(sorted(self.num)) or "1", *(["·".join(sorted(self.den))] if self.den else [])]
        )


@dataclass(frozen=True)
class Expression:
    """A unit expression found in text."""

    unit: Unit
    start: int
    end: int
    text: str
    attached: bool  # follows a number: the number's unit, not a request
    standalone: bool  # one ambiguous Persian word, no compound (never a request on its own)


def _latin_atoms(part: str) -> list[str] | None:
    out: list[str] = []
    for piece in re.split(r"[.\s·*×\-]+", part.strip()):
        if not piece:
            continue
        raw = piece.replace("²", "2").replace("³", "3")
        if raw in _JOINED:
            out += list(_JOINED[raw])
        elif raw in _LATIN_ATOMS:
            out.append(_LATIN_ATOMS[raw])
        else:
            return None
    return out


def _latin(token: str) -> tuple[list[str], list[str]] | None:
    """(numerator, basis) atoms of a Latin compound («kwh/m2.yr», «m2.k/w»), None if not one."""
    num_part, _, den_part = token.partition("/")
    num, den = _latin_atoms(num_part), _latin_atoms(den_part.replace("/", "."))
    if num is None or den is None or not (num or den):
        return None
    return num, den


def _atom_at(text: str, i: int) -> tuple[list[str], list[str], int, bool] | None:
    """(numerator atoms, basis atoms, end, ambiguous single word) of the unit token at ``i``."""
    m = _PERSIAN_RE.match(text, i)
    if m:
        form = re.sub(r"[\s‌]", " ", m[1])
        atom = next(a for f, a in _PERSIAN if f.replace(" ", "") == form.replace(" ", ""))
        return [atom], [], m.end(), form in _AMBIGUOUS
    m = _LATIN_RE.match(text, i)
    if m:
        parts = _latin(m[1])
        if parts is not None:
            num, den = parts
            single = len(num) + len(den) == 1 and not den and len(m[1]) < 2
            return num, den, m.end(), single or m[1] in ("a", "d", "h", "in", "l", "mo")
    return None


def _expression(text: str, i: int) -> tuple[Unit, int, bool] | None:
    """The longest unit expression starting at ``i``."""
    first = _atom_at(text, i)
    if first is None:
        return None
    num, den, end, ambiguous = first
    num, den = list(num), list(den)
    in_den = bool(den)
    parts = 1
    all_ambiguous = ambiguous
    while True:
        adv = _ADVERB_RE.match(text, end)
        if adv:
            den.append(_ADVERBS[adv[1]])
            end, parts = adv.end(), parts + 1
            continue
        conn = _CONNECTOR.match(text, end)
        nxt = _atom_at(text, conn.end()) if conn and conn.end() > end else None
        if conn and nxt is not None and not _ORDINAL.match(text, nxt[2]):
            den += nxt[0] + nxt[1]
            end, in_den, parts = nxt[2], True, parts + 1
            all_ambiguous = all_ambiguous and nxt[3]
            continue
        # a product of atoms («متر مربع کلوین»): only spacing between them
        prod = _atom_at(text, end + len(text[end:]) - len(text[end:].lstrip()))
        if prod is not None and text[end:].startswith(" ") and not _ORDINAL.match(text, prod[2]):
            (den if in_den else num).extend(prod[0])
            den.extend(prod[1])
            end, parts = prod[2], parts + 1
            all_ambiguous = all_ambiguous and prod[3]
            continue
        break
    if _ORDINAL.match(text, end) and not den and len(num) == 1:
        return None  # «سال اول», «ماه سوم»: a year or month, not a unit
    unit = Unit(frozenset(num), frozenset(den), text[i:end].strip())
    # a product of ambiguous words only («روز درجه»: degree days) is no unit request either
    return unit, end, all_ambiguous


_BASIS_BEFORE = re.compile(
    r"(?:(?<![ؠ-ۿ])(?:به\s?ازای\s?(?:هر)?|برای\s?هر|هر)|(?<![a-z])per|/)\s*$"
)


def _basis(text: str, i: int) -> tuple[Unit, int] | None:
    """A basis without a numerator: «سالانه به ازای هر متر مربع», «برای هر متر مربع»."""
    den: list[str] = []
    end = i
    adv = _ADVERB_RE.match(text, i)
    if adv:
        den.append(_ADVERBS[adv[1]])
        end = adv.end()
        conn = _CONNECTOR.match(text, end)
        if conn is None:
            return None  # a lone adverb («مصرف سالانه») is no unit
        end = conn.end()
    elif not _BASIS_BEFORE.search(text[:i]):
        return None
    found = _expression(text, end)
    if found is None:
        return None
    unit, stop, _ = found
    return Unit(frozenset(), frozenset(den) | unit.num | unit.den, text[i:stop].strip()), stop


def find(text: str) -> list[Expression]:
    """Every unit expression of a (normalized) text, left to right, each the longest one at its
    start; a single ambiguous word («سال», «گرم», «c») counts only right after a number."""
    t = text.lower()
    out: list[Expression] = []
    i = 0
    while i < len(t):
        if i and re.match(r"[ؠ-ۿa-z0-9]", t[i - 1]):
            i += 1
            continue
        attached = bool(_NUMBER_BEFORE.search(t[:i]))
        basis = None if attached else _basis(t, i)
        if basis is not None:
            unit, end = basis
            out.append(Expression(unit, i, end, t[i:end].strip(), False, False))
            i = end
            continue
        found = _expression(t, i)
        if found is None:
            i += 1
            continue
        unit, end, ambiguous = found
        if not ambiguous or attached:
            out.append(Expression(unit, i, end, t[i:end].strip(), attached, ambiguous))
        i = end
    return out


def parse(text: str) -> Unit | None:
    """The unit of a text that is exactly one unit expression, or None."""
    t = normalize_for_search(text).lower().replace("²", "2").replace("³", "3").strip(" .،,؟?")
    if not t:
        return None
    found = _expression(t, 0)
    if found is None or t[found[1] :].strip():
        return None
    return found[0]


def compatible(requested: Unit, printed: Unit) -> bool:
    """Every requested atom is a printed atom on the same side (the printed unit, possibly
    elliptical: «kWh» for «kWh/m2.yr»)."""
    if requested.foreign or not requested.num:
        return False
    return requested.num <= printed.num and requested.den <= printed.den


def same(a: Unit, b: Unit) -> bool:
    return a.num == b.num and a.den == b.den


GROUND = "ground"
CONVERSION = "conversion"
OTHER_QUANTITY = "other_quantity"


def verdict(requested: Unit, printed: Unit) -> str:
    """ground | conversion (same quantity, other units or basis) | other_quantity."""
    if not requested.foreign and requested.num <= printed.num and requested.den <= printed.den:
        return GROUND
    if requested.dimension == printed.dimension:
        return CONVERSION
    # an elliptical request: the quantity of each side named by the printed one
    if requested.kinds(requested.num) and all(
        requested.kinds(side) <= printed.kinds(full) for side, full in (
            (requested.num, printed.num), (requested.den, printed.den)
        )
    ):  # fmt: skip
        return CONVERSION
    return OTHER_QUANTITY


def _source_atoms() -> frozenset[str]:
    out: set[str] = set()
    for key, forms in UNITS.items():
        for f in (key, *forms):
            u = parse(f)
            if u is not None:
                out |= u.num | u.den
    return frozenset(out)


SOURCE_ATOMS: frozenset[str] = frozenset()
SOURCE_ATOMS = _source_atoms() | {"yr"}  # «در سال» of the printed EUI unit
KNOWN_ATOMS = SOURCE_ATOMS  # backwards-compatible name

__all__ = [
    "CONVERSION",
    "GROUND",
    "KNOWN_ATOMS",
    "LEXICON",
    "OTHER_QUANTITY",
    "SOURCE_ATOMS",
    "Expression",
    "Unit",
    "compatible",
    "find",
    "parse",
    "same",
    "verdict",
]
