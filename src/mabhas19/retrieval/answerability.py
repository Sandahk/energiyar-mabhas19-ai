"""Deterministic answerability gate: intents that Mabhas 19 does not answer.

The gate recognises *kinds of request*, not individual questions. Each category is a reusable
intent with the reason it is outside the regulation; matches are reported with the category
and the triggering words so every decision is auditable. Domain overlap alone never makes a
question answerable, and missing exact wording alone never makes a conceptual Mabhas question
unanswerable: the gate only fires on positive evidence of an out-of-scope intent.

Categories:

* ``live_information`` - current/forecast conditions (weather today, tomorrow, now);
* ``commercial`` - prices, purchase, brands, product recommendations;
* ``service_information`` - office hours, contact details, appointments, registration;
* ``education_services`` - courses, exam preparation, class schedules;
* ``outside_mabhas19`` - topics of other regulations or fields (zoning plans, fire egress,
  structural design, finance, unrelated subjects).
"""

from __future__ import annotations

import re
from collections.abc import Collection
from dataclasses import dataclass

from mabhas19.text.normalize import normalize_for_search

_LIVE_TIME = ("امروز", "فردا", "الان", "اکنون", "این هفته", "هفته آینده", "امشب")
_LIVE_TOPIC = ("هوا", "دما", "باران", "برف", "آفتابی", "ابری", "بارش")
CATEGORIES: dict[str, tuple[str, ...]] = {
    "commercial": (
        "قیمت",
        "فروش",
        "بازار",
        "تخفیف",
        "نمایندگی",
        "برند",
        "مارک",
        "بهترین برند",
        "ارزان",
        "گران ترین",
    ),
    "service_information": (
        "ساعت کاری",
        "ساعت کار",
        "شماره تماس",
        "تلفن",
        "آدرس دفتر",
        "وقت ملاقات",
        "نوبت دهی",
        "ایمیل",
    ),
    "education_services": (
        "کنکور",
        "برنامه کلاس",
        "دانشجو",
        "ترم",
        "ثبت نام",
        "آزمون ورودی",
        "پکیج آموزشی",
        "دوره آموزشی",
    ),
    "outside_mabhas19": (
        "طرح تفصیلی",
        "طرح جامع",
        "تراکم ساختمانی",
        "کاربری اراضی",
        "پله فرار",
        "راه فرار",
        "خروج اضطراری",
        "آرماتور",
        "میلگرد",
        "طراحی سازه",
        "محاسبات سازه",
        "تیر و ستون",
        "زلزله",
        "وام",
        "مالیات",
        "اجاره",
        "بورس",
        "فوتبال",
        "دستور پخت",
        "بلیت",
        "کیلومتر",
        "جمعیت",
    ),
}
# «ست»: copula («معناست»); «ش/اش»: possessive («اقلیمش»)
_SUFFIXES = ("", "ی", "ها", "های", "ای", "ان", "ات", "ست", "ش", "اش")


@dataclass(frozen=True)
class GateResult:
    category: str
    matched: list[str]
    reason: str


def has_words(norm: str, words: Collection[str]) -> list[str]:
    """Words present in a normalized query: whole tokens (a few attached suffixes allowed),
    multi-word phrases spacing/ZWNJ-insensitively. «دوش» does not match «دوشنبه»."""
    flat = norm.replace(" ", "")
    out = []
    for w in words:
        wn = normalize_for_search(w)
        if " " in wn:
            if wn.replace(" ", "") in flat:
                out.append(w)
            continue
        suffix = "|".join(re.escape(s) for s in _SUFFIXES)
        # punctuation is a word boundary («اقلیم؟»), letters and digits are not
        if re.search(rf"(?<!\w){re.escape(wn)}(?:{suffix})(?!\w)", norm):
            out.append(w)
    return out


def gate(norm: str) -> GateResult | None:
    """First out-of-scope intent found in a normalized query, or ``None``."""
    times, topics = has_words(norm, _LIVE_TIME), has_words(norm, _LIVE_TOPIC)
    if times and topics:
        return GateResult(
            "live_information",
            times + topics,
            "current or forecast conditions are not regulation content",
        )
    for category, words in CATEGORIES.items():
        hit = has_words(norm, words)
        if hit:
            return GateResult(
                category,
                hit,
                f"{category.replace('_', ' ')} requests are not answered by Mabhas 19",
            )
    return None
