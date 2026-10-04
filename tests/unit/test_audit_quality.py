"""Text-layer quality classification."""

from __future__ import annotations

from mabhas19.ingest.audit.models import TextQuality
from mabhas19.ingest.audit.quality import classify_text_quality
from mabhas19.text.charclass import char_profile

CLEAN_PERSIAN = (
    "مبحث نوزدهم مقررات ملی ساختمان به موضوع صرفه\u200cجویی در مصرف انرژی در ساختمان\u200cها "
    "می\u200cپردازد و الزامات طراحی، اجرا و بهره\u200cبرداری را "
    "برای کاهش مصرف انرژی بیان می\u200cکند. "
    "بالا بردن بازدهی تجهیزات لازم است."
)
# Mimics the main PDF: ZWNJ instead of spaces, reversed lam-alef, split words.
DAMAGED_PERSIAN = (
    "\u200cالزم\u200cاالجرا\u200cشدن\u200cاین\u200cپیوست\u200c"
    "به\u200cمرور\u200cو\u200cطبق\u200cاعالم\u200cنهاد\u200c"
    "قانونی\u200cمسئول\u200cانجام\u200cخواه\u200cد\u200cشد\u200c"
    "بهره\u200cبرداری\u200cانرژ ی\u200cاالجرا"
)


def classify(text: str) -> TextQuality:
    return classify_text_quality(char_profile(text))[0]


def test_empty_and_near_empty_pages_have_no_text() -> None:
    assert classify("") == TextQuality.NO_TEXT
    assert classify("\u200c\u200c 3 \u200c") == TextQuality.NO_TEXT
    assert classify("- - - - - - - - - -") == TextQuality.NO_TEXT


def test_latin_and_digit_only_page() -> None:
    text = "Domestic Hot Water (DHW) Weather Physical Contact 0B 1B 2A 113 121 128"
    assert classify(text) == TextQuality.LATIN_DIGITS_ONLY


def test_digits_only_table_is_latin_digits_only() -> None:
    assert classify("113 121 128 128 144 131 175 146 167 77 83 87") == (
        TextQuality.LATIN_DIGITS_ONLY
    )


def test_clean_persian_is_good() -> None:
    quality, reasons = classify_text_quality(char_profile(CLEAN_PERSIAN))
    assert quality == TextQuality.GOOD, reasons


def test_damaged_persian_is_damaged_with_reasons() -> None:
    quality, reasons = classify_text_quality(char_profile(DAMAGED_PERSIAN))
    assert quality == TextQuality.DAMAGED
    assert any("ZWNJ" in r for r in reasons)
    assert any("lam-alef" in r for r in reasons)


def test_persian_with_some_english_terms_is_still_persian() -> None:
    assert classify(CLEAN_PERSIAN + " Life Cycle Assessment (LCA)") == TextQuality.GOOD
