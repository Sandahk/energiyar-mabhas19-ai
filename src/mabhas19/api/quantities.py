"""Regulatory table quantities as the user names them, and the value request around them.

Used by :func:`mabhas19.api.orchestration.rescue` only after the frozen engine answered a
question with prose or out-of-scope: when the question names exactly one table quantity, asks
for its value, and contains nothing else but filler, a printed climate code or a city, it must
not end as document excerpts. Each quantity's cue phrases are its printed / table-definition
names (``retrieval.schema`` ``Output.names``) without words that also mean other things («رده»,
«کلاس», «درصد», «دما»); nothing here is a value, and nothing decides a row.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from mabhas19.api.wording import letters
from mabhas19.generation import labels as L
from mabhas19.generation import options as O
from mabhas19.retrieval.schema import APP03, APP07, MAIN

if TYPE_CHECKING:
    from mabhas19.retrieval.structured import StructuredStore

_S = r"[\s‌]*"
_S1 = r"[\s‌]+"
_B = r"(?<![\w‌])"
_E = r"(?![\w])"
T19_5_1, T19_5_2 = f"{MAIN}19-5-1", f"{MAIN}19-5-2"


@dataclass(frozen=True)
class Quantity:
    id: str
    label: str  # the composer's Persian name (shown; never a value)
    cues: tuple[str, ...]  # regex sources on Persian letter forms
    tables: tuple[tuple[str, str], ...]  # (dataset, row field)
    climate_table: str = ""  # the table whose printed climate columns apply ("" if none)
    row_filter: str = ""  # a printed word every row of this quantity carries (19-5-2)
    choose: bool = True  # offer the printed rows when there are few; False: always ask
    # narrow extra phrasings: (cue, a phrase that must also be present or "", excluded contexts)
    guarded: tuple[tuple[str, str, str], ...] = ()

    def pattern(self) -> re.Pattern[str]:
        return re.compile("|".join(f"(?:{c})" for c in self.cues), re.IGNORECASE)

    def spans(self, text: str) -> list[tuple[int, int]]:
        """Where the question names this quantity: printed names, then the guarded phrasings
        (with the words they require, «چند درجه», which are part of the same request)."""
        out = [m.span() for m in self.pattern().finditer(text)]
        for cue, needs, excluded in self.guarded:
            if needs and not re.search(needs, text, re.IGNORECASE):
                continue
            if excluded and re.search(excluded, text, re.IGNORECASE):
                continue
            found = [m.span() for m in re.finditer(cue, text, re.IGNORECASE)]
            if found and needs:
                found += [m.span() for m in re.finditer(needs, text, re.IGNORECASE)]
            out += found
        return out

    def che(self) -> re.Pattern[str]:
        """C: «چه» + this quantity's printed name (+ at most two words) + «داره / دارد / داشته
        باش»: a value request («چه ضریب هدایت حرارتی‌ای داره؟»); «چه نقشی دارد» is not."""
        names = "|".join(f"(?:{c})" for c in self.cues)
        return re.compile(
            rf"{_B}چه{_S1}(?:{names})(?:{_S}ا?ی)?(?:{_S1}[^\s؟?.!،]+){{0,2}}?"
            rf"{_S1}(?:باید{_S1})?(?:داره|دارد|داشته{_S1}باش)",
            re.IGNORECASE,
        )

    def rows(self, store: StructuredStore) -> list[str]:
        out: list[str] = []
        for dataset, field in self.tables:
            for label in O.printed_options(store, dataset, field):
                if self.row_filter and self.row_filter not in label:
                    continue
                out.append(label)
        return out


QUANTITIES: tuple[Quantity, ...] = (
    Quantity(
        "eui",
        L.OUTPUT_LABELS["max_eui"],
        (
            rf"شدت{_S1}مصرف",
            rf"سقف{_S1}مصرف",
            rf"حداکثر{_S1}مصرف",
            rf"مصرف{_S1}(?:مجاز{_S1})?انرژی",
            rf"{_B}EUI{_E}",
        ),
        ((APP03, "occupancy"),),
        climate_table=APP03,
    ),
    Quantity(
        "r_value",
        L.OUTPUT_LABELS["min_r_value"],
        (rf"مقاومت{_S1}حرارتی", rf"{_B}R{_S}-?{_S}value{_E}"),
        ((T19_5_1, "element"),),
        climate_table=T19_5_1,
        # B: «حداقل R» / «R حداقل(ش)» only; never the R ratio (19-3-3) or a seismic R factor
        guarded=(
            (
                rf"حداقل{_S1}R{_E}|{_B}R{_S1}حداقل(?:ش)?{_E}",
                "",
                rf"نسبت{_S1}R|{_B}R{_S1}نسبت|رفتار{_S1}R",
            ),
        ),
    ),
    Quantity(
        "u_value",
        "حداکثر ضریب انتقال حرارت (U)",
        (rf"ضریب{_S1}انتقال{_S1}حرارت", rf"{_B}U{_S}-?{_S}Value{_E}"),
        # opaque parts print a minimum R (19-5-1), transparent parts a U (19-5-2): always ask
        ((T19_5_1, "element"), (T19_5_2, "component")),
        climate_table=T19_5_2,
        row_filter="",
        choose=False,
    ),
    Quantity(
        "shgc",
        "حداکثر ضریب بهره خورشیدی (SHGC)",
        (r"SHGC", rf"ضریب{_S1}بهره{_S1}خورشیدی"),
        ((T19_5_2, "component"),),
        climate_table=T19_5_2,
        row_filter="SHGC",
    ),
    Quantity(
        "lambda",
        L.OUTPUT_LABELS["thermal_conductivity"],
        (rf"(?:ضریب{_S1})?هدایت{_S1}حرارتی", rf"ضریب{_S1}هدایت", r"λ", r"لامبدا", r"لاندا"),
        ((APP07, "material"),),
    ),
    Quantity(
        "density",
        L.OUTPUT_LABELS["density"],
        (r"چگالی", rf"جرم{_S1}(?:مخصوص|حجمی)", rf"وزن{_S1}مخصوص"),
        ((APP07, "material"),),
    ),
    Quantity(
        "efficiency",
        "حداقل بازدهی",
        (
            r"بازده",
            r"راندمان",
            rf"{_B}COP{_E}",
            rf"ضریب{_S1}عملکرد",
            rf"{_B}EEI{_E}",
            rf"{_B}SEER{_E}",
        ),
        ((f"{MAIN}19-5-3", "equipment"), (f"{MAIN}19-5-4", "equipment")),
    ),
    Quantity(
        "hot_water",
        L.OUTPUT_LABELS["max_temperature"],
        (rf"دمای?(?:{_S1}مجاز)?{_S1}آب{_S}گرم", rf"آب{_S}گرم{_S1}مصرفی"),
        ((f"{MAIN}19-5-5", "place_of_use"),),
        # A: «آب گرم» with «چند درجه» / «دما»; never «آبگرمکن» (the appliance) or a circuit,
        # tank, boiler or heating temperature
        guarded=(
            (
                rf"آب{_S}گرم(?!{_S}کن)",
                rf"چند{_S1}درجه|{_B}دما",
                rf"برگشت|{_B}رفت{_E}|{_B}خط{_E}|مخزن|{_B}دیگ|بویلر|لوله|گرمایش|شوفاژ",
            ),
        ),
    ),
    Quantity(
        "daylight",
        L.OUTPUT_LABELS["min_daylight_area"],
        (
            rf"(?:نسبت{_S1})?(?:مساحت{_S1})?کفایت{_S1}نور(?:{_S1}روز)?",
            rf"(?:نسبت{_S1})?(?:مساحت{_S1})?نور{_S1}روز",
        ),
        ((f"{MAIN}19-5-6", "use"),),
    ),
    Quantity(
        "uniformity",
        L.OUTPUT_LABELS["min_uniformity"],
        (rf"یکنواختی(?:{_S1}(?:نور|روشنایی))?(?:{_S1}مصنوعی)?",),
        ((f"{MAIN}19-5-7", "space"),),
    ),
)
BY_ID = {q.id: q for q in QUANTITIES}
# the printed words a canonical question starts with (each quantity's printed name)
CANONICAL = {
    "eui": "حداکثر شدت مصرف انرژی مجاز",
    "r_value": "حداقل مقاومت حرارتی",
    "u_value": "حداکثر ضریب انتقال حرارتی",
    "shgc": "حداکثر ضریب بهره خورشیدی",
    "lambda": "ضریب هدایت حرارتی",
    "density": "چگالی",
    "efficiency": "حداقل بازدهی",
    "hot_water": "حداکثر دمای آب گرم مصرفی",
    "daylight": "حداقل نسبت مساحت کفایت نور روز",
    "uniformity": "حداقل یکنواختی نور مصنوعی",
}

# asking for a value (not for an explanation)
VALUE = re.compile(
    rf"حداقل|حداکثر|حد{_S1}مجاز|مقدار{_S1}مجاز|مجاز|چقدر|چند|{_B}باید{_E}|{_B}لازم|کمینه|بیشینه"
)
# explanation, definition, method or document questions stay with the engine
CONCEPT = re.compile(
    rf"چیست|{_B}یعنی{_E}|چه{_S}می{_S}گوید|چی{_S}میگه|درباره|دربارهٔ|چگونه|چطور|{_B}تعریف|"
    rf"{_B}فرق{_E}|تفاوت|منظور|محاسبه|توضیح|{_B}چرا{_E}|{_B}کجا|{_B}واحد"
)
_CODE = re.compile(r"(?<![A-Za-z0-9])([0-5])\s*([ABCabc])(?![A-Za-z])")
_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")
_WORD = re.compile(r"[^\W\d_]+")
# value-request, generic and connecting words; anything else may name a row (or something the
# tables do not print), so the engine's own answer is kept
FILLERS = frozenset(
    {
        "حد",
        "حداقل",
        "حداکثر",
        "مجاز",
        "مجازه",
        "مقدار",
        "میزان",
        "کمینه",
        "بیشینه",
        "چقدر",
        "چقدره",
        "چند",
        "چنده",
        "چیه",
        "است",
        "هست",
        "باید",
        "باشد",
        "باشه",
        "لازم",
        "لازمه",
        "برای",
        "واسه",
        "در",
        "تو",
        "توی",
        "از",
        "به",
        "که",
        "و",
        "را",
        "رو",
        "اقلیم",
        "شهر",
        "ساختمان",
        "ساختمانها",
        "سالانه",
        "انرژی",
        "طبق",
        "مبحث",
    }
)
MAX_PLACE_WORDS = 3
# C: the words of «چه <quantity> … داره» itself, dropped only when that phrase was found
CHE_WORDS = frozenset({"چه", "ای", "ی", "داره", "دارد", "داشته", "باش"})


@dataclass(frozen=True)
class Request:
    """A value request for one table quantity: what is left besides it (a city, possibly)."""

    quantity: Quantity
    code: str  # a climate code written in the question ("" if none)
    leftover: tuple[str, ...]  # remaining words (a place name, or something else)


def analyse(question: str) -> Request | None:
    """The question as a value request for exactly one table quantity, or ``None``."""
    text = letters(question)
    if CONCEPT.search(text):
        return None
    if not (VALUE.search(text) or any(q.che().search(text) for q in QUANTITIES)):
        return None
    found = [(q, q.spans(text)) for q in QUANTITIES]
    found = [(q, sp) for q, sp in found if sp]
    if len(found) != 1:
        return None
    quantity, spans = found[0]
    merged: list[list[int]] = []
    for start, end in sorted(set(spans)):  # overlapping phrasings: one span
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    rest = text
    for start, end in reversed(merged):
        rest = rest[:start] + " " + rest[end:]
    rest = rest.translate(_DIGITS)
    codes = {f"{a}{b.upper()}" for a, b in _CODE.findall(rest)}
    if len(codes) > 1:
        return None
    rest = _CODE.sub(" ", rest)
    drop = FILLERS | (CHE_WORDS if quantity.che().search(text) else frozenset())
    words = [w for w in _WORD.findall(rest.replace("‌", " ")) if w not in drop]
    return Request(quantity, next(iter(codes), ""), tuple(words))
