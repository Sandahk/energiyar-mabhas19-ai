"""Deterministic wording rules and concept families (reviewed lists; no model).

Three kinds, in increasing caution:

* **Wording** (:data:`WORDING`): a spelling, transliteration or morphology variant of printed
  wording («یو ولیو» -> «U-Value», «اداره» -> «اداری»). Rewritten before the engine reads the
  question, and shown to the user as a short note.
* **Concept rows** (:data:`CONCEPT_ROWS`): an everyday name of exactly one printed row (or of
  one printed row group the engine itself then clarifies), rewritten only when the question
  also names that table's quantity («اسپلیت» -> «کولر گازی» when an efficiency is asked).
  Shown as a note.
* **Families** (:data:`FAMILIES`): an everyday word for several printed rows («بام» -> the
  roof rows of table 19-5-1). A family never picks a row: when the engine cannot answer, the
  user chooses one of the printed rows, or (for a condition band of at most four rows) the
  certified value of each row is shown side by side.

A table whose quantity is named with no row at all (:data:`MISSING_ROW`) offers its printed rows
when no other content word is left in the question.

Meaning-changing mappings (school -> «آموزشی», hospital -> «درمانی», shop -> «تجاری», bathroom
-> «دوش و وان», «دستشویی» -> «روشویی», shading coefficient -> SHGC) are deliberately absent:
such questions keep the engine's refusal. Arabic letters, ZWNJ and plural/singular forms are
normalized by the engine itself; here they only make the patterns match.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from mabhas19.retrieval.schema import MAIN

WORDING_VERSION = "wording-1"

_ARABIC = str.maketrans({"ي": "ی", "ى": "ی", "ك": "ک", "ە": "ه"})
_S = r"[\s‌]*"  # optional space or ZWNJ
_S1 = r"[\s‌]+"  # a space or ZWNJ
_B = r"(?<![\w‌])"  # word start
_E = r"(?![\w])"  # word end


def letters(text: str) -> str:
    """Persian letter forms only (Arabic yeh / kaf -> Persian); nothing else changes."""
    return text.translate(_ARABIC)


@dataclass(frozen=True)
class Rule:
    """``pattern`` (on Persian letter forms) -> ``to``; ``shown`` is the user's wording."""

    name: str
    shown: str
    pattern: re.Pattern[str]
    to: str
    note: str
    cue: re.Pattern[str] | None = None  # the rule applies only when this is in the question

    def applies(self, text: str) -> bool:
        return bool(self.pattern.search(text)) and (self.cue is None or bool(self.cue.search(text)))


def _rule(  # noqa: PLR0913, PLR0917
    name: str, shown: str, pattern: str, to: str, note: str, cue: str | None = None
) -> Rule:
    return Rule(
        name,
        shown,
        re.compile(pattern, re.IGNORECASE),
        to,
        note.format(shown=shown, to=to),
        re.compile(cue, re.IGNORECASE) if cue else None,
    )


_NOTE_WORDING = "«{shown}» به‌عنوان «{to}» در نظر گرفته شد."
_NOTE_ROW = "«{shown}» به‌عنوان ردیف چاپ‌شدهٔ «{to}» در نظر گرفته شد."
_NOTE_GROUP = "«{shown}» به ردیف‌های «{to}» جدول مربوط است؛ ردیف را خودتان انتخاب کنید."

WORDING: tuple[Rule, ...] = (
    _rule("u_value", "یو ولیو", rf"{_B}یو{_S}-?{_S}ولیو{_E}", "U-Value", _NOTE_WORDING),
    _rule("shgc", "اس اچ جی سی", rf"{_B}اس{_S}اچ{_S}جی{_S}سی{_E}", "SHGC", _NOTE_WORDING),
    _rule(
        "uniformity", "یکنواختی روشنایی", rf"یکنواختی{_S1}روشنایی", "یکنواختی نور", _NOTE_WORDING
    ),
    _rule(
        "gas_water_heater",
        "آبگرمکن گازی",
        rf"{_B}[آا]ب{_S}گرمکن{_S1}گازی{_E}",
        "آبگرمکن گازسوز",
        _NOTE_WORDING,
    ),
    _rule("gas_heater", "بخاری گازی", rf"{_B}بخاری{_S1}گازی{_E}", "بخاری گازسوز", _NOTE_WORDING),
    _rule("sink", "سینک ظرفشویی", rf"{_B}سینک{_S1}ظرف{_S}شویی", "ظرفشویی", _NOTE_WORDING),
    _rule("shower", "دوش حمام", rf"{_B}دوش{_S1}حمام{_E}", "دوش", _NOTE_WORDING),
    # only where a building use is looked up in a table (energy-use limit, daylight, lighting
    # uniformity); elsewhere «اداره» stays as written (the prose search reads it as it is)
    _rule(
        "office",
        "اداره",
        rf"{_B}اداره(?:‌ی|ٔ)?{_E}",
        "اداری",
        _NOTE_WORDING,
        cue=rf"مصرف{_S1}(?:مجاز{_S1})?انرژی|شدت{_S1}مصرف|کفایت{_S1}نور|یکنواختی",
    ),
)

_EFFICIENCY = r"بازده|راندمان|SEER|رده{_S1}انرژی|برچسب".replace("{_S1}", _S1)
_UNIFORMITY = rf"یکنواختی{_S1}(?:نور|روشنایی)"

CONCEPT_ROWS: tuple[Rule, ...] = (
    _rule(
        "split",
        "اسپلیت",
        rf"{_B}(?:کولر{_S1})?اسپلیت{_E}",
        "کولر گازی",
        _NOTE_ROW,
        cue=_EFFICIENCY,
    ),
    _rule(
        "office_space",
        "دفتر کار",
        rf"{_B}دفتر{_S1}کار{_E}",
        "اداری و فضاهای کار",
        _NOTE_ROW,
        cue=_UNIFORMITY,
    ),
    _rule(
        "sports_hall",
        "سالن ورزشی",
        rf"{_B}سالن(?:{_S}های)?{_S1}ورزشی{_E}",
        "فضاهای ورزشی",
        _NOTE_ROW,
        cue=_UNIFORMITY,
    ),
    _rule(
        "wall_package",
        "پکیج دیواری",
        rf"{_B}پکیج{_S1}دیواری{_E}",
        "پکیج",
        _NOTE_GROUP,
        cue=_EFFICIENCY,
    ),
)


def rewrite(question: str) -> tuple[str, list[Rule]]:
    """The question with every applicable rule applied, and the rules; unchanged (and no
    rules) when none applies."""
    text = letters(question)
    applied: list[Rule] = []
    for rule in (*WORDING, *CONCEPT_ROWS):
        if rule.applies(text):
            text = rule.pattern.sub(rule.to, text)
            applied.append(rule)
    return (text, applied) if applied else (question, [])


# ---------------------------------------------------------------- families


@dataclass(frozen=True)
class Family:
    """An everyday word for several printed rows of one table, for one quantity."""

    name: str
    shown: str
    word: re.Pattern[str]
    cue: re.Pattern[str]  # the table's quantity, as asked
    dataset: str
    field: str
    rows: tuple[str, ...]  # printed row labels (checked against the store by the tests)
    prompt: str
    band: bool = False  # the rows differ only in a printed condition band: side by side
    excluded: re.Pattern[str] | None = None  # words that already name a row: not this family

    @property
    def note(self) -> str:
        return _NOTE_FAMILY.format(shown=self.shown)

    def matches(self, text: str) -> bool:
        t = letters(text)
        if not (self.word.search(t) and self.cue.search(t)):
            return False
        return self.excluded is None or not self.excluded.search(t)


_NOTE_FAMILY = "«{shown}» به چند ردیف چاپ‌شدهٔ جدول مربوط است؛ هیچ ردیفی به‌جای شما انتخاب نشده است."
_U = rf"U{_S}-?{_S}Value|{_B}U{_E}|ضریب{_S1}انتقال{_S1}حرارت"
_SHGC = rf"SHGC|ضریب{_S1}بهره{_S1}خورشیدی"
_WINDOW = rf"{_B}پنجره(?:{_S}ها|{_S}ی)?{_E}"
_U_ROWS = (
    "حداکثر ضریب انتقال حرارتی U-Value (W/m2.k) – ثابت",
    "حداکثر ضریب انتقال حرارتی U-Value (W/m2.k) – متحرک",
)
_SHGC_ROWS = (
    "حد اکثر ضریب بهره خورشیدی SHGC – کمتر از %۲۰",
    "حد اکثر ضریب بهره خورشیدی SHGC – بین %۲۰ تا %۵۰",
    "حد اکثر ضریب بهره خورشیدی SHGC – بین %۵۰ تا %۷۰",
    "حد اکثر ضریب بهره خورشیدی SHGC – بیشتر از %۷۰",
)

FAMILIES: tuple[Family, ...] = (
    Family(
        "roof",
        "بام",
        re.compile(rf"{_B}(?:پشت{_S})?بام(?:{_S}ها)?{_E}"),
        re.compile(rf"مقاومت{_S1}حرارتی|{_B}R{_E}", re.IGNORECASE),
        f"{MAIN}19-5-1",
        "element",
        ("سقف مجاور فضای آزاد", "سقف مجاور فضای نیمه باز کنترل نشده"),
        "نوع سقف را انتخاب کنید:",
        excluded=re.compile(rf"دیوار|{_B}کف{_E}|{_B}در(?:ها|های){_E}|مجاور"),
    ),
    Family(
        "window_u",
        "پنجره",
        re.compile(_WINDOW),
        re.compile(_U, re.IGNORECASE),
        f"{MAIN}19-5-2",
        "component",
        _U_ROWS,
        "نوع پنجره را انتخاب کنید:",
        excluded=re.compile(rf"ثابت|متحرک|نورگیر|{_B}در{_S1}ورودی|{_SHGC}", re.IGNORECASE),
    ),
    Family(
        "window_shgc",
        "پنجره",
        re.compile(_WINDOW),
        re.compile(_SHGC, re.IGNORECASE),
        f"{MAIN}19-5-2",
        "component",
        _SHGC_ROWS,
        "نسبت سطح پنجره را انتخاب کنید:",
        band=True,
        excluded=re.compile(rf"٪|%|درصد|نسبت|نورگیر|{_U}", re.IGNORECASE),
    ),
)


def family(text: str) -> Family | None:
    """The one family the question names, or ``None`` (none, or more than one)."""
    found = [f for f in FAMILIES if f.matches(text)]
    return found[0] if len(found) == 1 else None


def with_row(text: str, fam: Family, row: str) -> str:
    """The question with the family's word replaced by the chosen printed row (the row is then
    the question's only name for the thing; the broad word is not kept beside it)."""
    return fam.word.sub(lambda _: row, letters(text), count=1)


# ---------------------------------------------------------------- a quantity with no row


@dataclass(frozen=True)
class MissingRow:
    dataset: str
    field: str
    cue: re.Pattern[str]
    words: frozenset[str]  # the quantity's own words


MISSING_ROW: tuple[MissingRow, ...] = (
    MissingRow(
        f"{MAIN}19-5-5",
        "place_of_use",
        re.compile(rf"دمای?{_S1}آب{_S}گرم"),
        frozenset({"دمای", "دما", "آب", "گرم", "آبگرم", "مصرفی", "مصرف"}),
    ),
    MissingRow(
        f"{MAIN}19-5-6",
        "use",
        re.compile(rf"کفایت{_S1}نور{_S1}روز"),
        frozenset({"نسبت", "مساحت", "کفایت", "نور", "روز"}),
    ),
    MissingRow(
        f"{MAIN}19-5-7",
        "space",
        re.compile(rf"یکنواختی{_S1}نور"),
        frozenset({"یکنواختی", "نور", "مصنوعی"}),
    ),
)

# words that ask, qualify the asking or name the building in general; anything else left in the
# question may name a row the table does not print, so no row is offered then
_FILLERS = frozenset(
    {
        "حداکثر",
        "حداقل",
        "حد",
        "اکثر",
        "اقل",
        "مجاز",
        "چقدر",
        "چقدره",
        "چند",
        "چنده",
        "است",
        "هست",
        "باید",
        "باشد",
        "باشه",
        "درجه",
        "چه",
        "میزان",
        "مقدار",
        "برای",
        "در",
        "از",
        "به",
        "که",
        "و",
        "را",
        "رو",
        "لازم",
        "لازمه",
        "ساختمان",
    }
)
_WORD = re.compile(r"[^\W\d_]+")


def missing_row(text: str) -> MissingRow | None:
    """The table whose quantity the question names with no other content word, or ``None``."""
    t = letters(text)
    found = [m for m in MISSING_ROW if m.cue.search(t)]
    if len(found) != 1:
        return None
    m = found[0]
    words = set(_WORD.findall(t.replace("‌", " ")))
    return m if words <= (m.words | _FILLERS) else None
