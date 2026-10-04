"""Section-number detection with Unicode hyphen variants and RTL visual order."""

from __future__ import annotations

import pytest

from mabhas19.text.section_numbers import (
    canonical_from_components,
    find_section_numbers,
    parse_section_number,
)

HYPHENS = ["-", "\u2010", "\u2011", "\u2012", "\u2013", "\u2014", "\u2212", "\ufe63", "\uff0d"]


@pytest.mark.parametrize("hyphen", HYPHENS, ids=[f"U+{ord(h):04X}" for h in HYPHENS])
def test_every_hyphen_variant_is_accepted(hyphen: str) -> None:
    assert parse_section_number(f"19{hyphen}5{hyphen}1{hyphen}2") == "19-5-1-2"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("19-1", "19-1"),
        ("19\u20111\u20111", "19-1-1"),  # U+2011 as in the main PDF
        ("2-1-5-19", "19-5-1-2"),  # visual (left-to-right) extraction order
        ("1-4-2-19", "19-2-4-1"),
        ("۱۹-۳-۲", "19-3-2"),  # Persian digits
        ("١٩-٣-٢", "19-3-2"),  # Arabic-Indic digits
        ("19 \u2011 7 \u2011 1 \u2011 3", "19-7-1-3"),  # padded with spaces
        ("19\u200c-\u200c10", "19-10"),  # padded with ZWNJ
        ("19-1-10", "19-1-10"),
    ],
)
def test_parse_section_number(raw: str, expected: str) -> None:
    assert parse_section_number(raw) == expected


@pytest.mark.parametrize("raw", ["19", "13-2-1", "1-2-3", "19-", "a-19", "19-1-2-3-4-5-6"])
def test_non_section_numbers_are_rejected(raw: str) -> None:
    assert parse_section_number(raw) is None


def test_reading_order_wins_when_both_ends_are_19() -> None:
    assert canonical_from_components([19, 2, 19]) == "19-2-19"


def test_find_section_numbers_in_running_text() -> None:
    text = "مطابق جدول 19\u20115\u20111 و بند ۱۹-۳-۲ و نه 1404-07-15 یا 2-4-2-19\u200cبازدهی"
    assert find_section_numbers(text) == [
        ("19\u20115\u20111", "19-5-1"),
        ("۱۹-۳-۲", "19-3-2"),
        ("2-4-2-19", "19-2-4-2"),
    ]


def test_find_does_not_split_longer_numbers() -> None:
    assert find_section_numbers("119-5-1") == []
