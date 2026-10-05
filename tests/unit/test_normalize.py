"""Persian normalization for search/comparison (never applied to display text)."""

from __future__ import annotations

import pytest

from mabhas19.text.normalize import (
    fold_hyphens,
    normalize_for_comparison,
    normalize_for_search,
    remove_marks,
    to_ascii_digits,
    to_persian_digits,
)

ZWNJ = "\u200c"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("كيلو", "کیلو"),  # Arabic kaf/yeh -> Persian
        ("مسكوني", "مسکونی"),
        ("۱۹-۵-۱", "19-5-1"),  # Persian digits -> ASCII
        ("١٩", "19"),  # Arabic-Indic digits -> ASCII
        ("19\u20115\u20111", "19-5-1"),  # non-breaking hyphens folded
        (f"می{ZWNJ}شود", "می شود"),  # ZWNJ is a word break for search
        ("شـــدت", "شدت"),  # tatweel removed
        ("م\u064fدیریت", "مدیریت"),  # diacritics removed
        ("ﻻ", "لا"),  # presentation-form lam-alef (NFKC)
        ("m²", "m2"),  # superscript (NFKC)
        ("\u200fمتن\u200e", "متن"),  # bidi controls removed
        ("  Energy   Meter ", "energy meter"),
        ("تأمین", "تامین"),  # hamza mark dropped for matching
    ],
)
def test_normalize_for_search(raw: str, expected: str) -> None:
    assert normalize_for_search(raw) == expected


def test_comparison_folds_punctuation_and_spacing() -> None:
    assert normalize_for_comparison("رده D : منطبق") == normalize_for_comparison("رده D: منطبق")
    assert normalize_for_comparison("«مدیریت»،") == normalize_for_comparison('"مدیریت",')
    assert normalize_for_comparison("( MJ/m2 )") == "(mj/m2)"


def test_digit_conversions_round_trip() -> None:
    assert to_persian_digits("1404/04/19") == "۱۴۰۴/۰۴/۱۹"
    assert to_ascii_digits("۱۴۰۴/۰۴/۱۹") == "1404/04/19"
    assert to_persian_digits("٥") == "۵"  # Arabic-Indic five -> Persian five


def test_fold_hyphens_and_remove_marks_are_targeted() -> None:
    assert fold_hyphens("a\u2013b\u2212c") == "a-b-c"
    assert remove_marks("ب\u0650سم") == "بسم"


def test_search_normalization_does_not_alter_its_input() -> None:
    display = f"ساختمان{ZWNJ}ها ۱۹-۲-۴"
    before = display
    normalize_for_search(display)
    assert display == before
