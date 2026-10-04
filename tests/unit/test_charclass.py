"""Persian / Latin / digit character classification."""

from __future__ import annotations

import pytest

from mabhas19.text.charclass import (
    char_profile,
    digit_value,
    digits_to_int,
    is_latin_letter,
    is_persian_letter,
)


@pytest.mark.parametrize("ch", ["ا", "ب", "پ", "چ", "ژ", "گ", "ک", "ی", "ي", "ك", "ﻻ", "ﺏ"])
def test_persian_and_arabic_letters_including_presentation_forms(ch: str) -> None:
    assert is_persian_letter(ch)


@pytest.mark.parametrize("ch", ["۱", "١", "1", "\u200c", "\u064b", "،", "؛", "a", " "])
def test_non_letters_are_not_persian_letters(ch: str) -> None:
    assert not is_persian_letter(ch)


@pytest.mark.parametrize("ch", ["a", "Z", "é", "Ä"])
def test_latin_letters(ch: str) -> None:
    assert is_latin_letter(ch)


@pytest.mark.parametrize("ch", ["ا", "1", "۱", "-", "Ω"])
def test_non_latin(ch: str) -> None:
    assert not is_latin_letter(ch)


@pytest.mark.parametrize(
    ("ch", "value"), [("7", 7), ("۷", 7), ("٧", 7), ("0", 0), ("۰", 0), ("٠", 0)]
)
def test_digit_values_for_all_three_digit_sets(ch: str, value: int) -> None:
    assert digit_value(ch) == value


@pytest.mark.parametrize(
    ("text", "value"), [("126", 126), ("۱۲۶", 126), ("١٢٦", 126), ("1۲٦", 126), ("", None)]
)
def test_digits_to_int(text: str, value: int | None) -> None:
    assert digits_to_int(text) == value


@pytest.mark.parametrize("text", ["12a", "۱-۲", "ب"])
def test_digits_to_int_rejects_non_digits(text: str) -> None:
    assert digits_to_int(text) is None


def test_profile_counts_each_class_separately() -> None:
    p = char_profile("ساختمان Building 19 ۱۹ ١٩")
    assert p.persian_letters == 7
    assert p.latin_letters == 8
    assert (p.ascii_digits, p.persian_digits, p.arabic_indic_digits) == (2, 2, 2)
    assert p.digits == 6
    assert p.total_chars == 7 + 8 + 6


def test_profile_counts_zwnj_but_not_as_a_character() -> None:
    p = char_profile("می\u200cشود")
    assert p.zwnj == 1
    assert p.total_chars == 5
    assert p.persian_letters == 5


def test_profile_detects_extraction_damage_markers() -> None:
    p = char_profile("الزم\u200cاالجرا انرژ ی ﻻ")
    assert p.reversed_lam_alef == 1
    assert p.single_letter_words == 2  # «ی» and the presentation-form «ﻻ»
    assert p.presentation_forms == 1


def test_single_letter_waw_is_not_counted_as_damage() -> None:
    assert char_profile("آب و هوا").single_letter_words == 0
