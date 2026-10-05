"""Numeric safety: no guessing, decimals preserved, OCR confusions surfaced."""

from __future__ import annotations

from decimal import Decimal

import pytest

from mabhas19.text.numbers import parse_numeric, same_value


@pytest.mark.parametrize(
    "raw",
    ["0", "۰", "٠"],  # ASCII, Persian and Arabic-Indic zero all have value 0
)
def test_zero_variants(raw: str) -> None:
    v = parse_numeric(raw)
    assert v.normalized_value == "0"
    assert v.confidence == "high"


@pytest.mark.parametrize(
    ("raw", "value"),
    [("1", "1"), ("۱", "1"), ("١", "1"), ("5", "5"), ("۵", "5"), ("٥", "5"),
     ("6", "6"), ("۶", "6"), ("٦", "6")],
)  # fmt: skip
def test_lookalike_digits_map_by_value(raw: str, value: str) -> None:
    assert parse_numeric(raw).normalized_value == value


def test_mixed_digit_scripts_are_flagged_not_trusted() -> None:
    v = parse_numeric("1۵0")  # typical OCR mix: Latin 1 and 0 around a Persian five
    assert v.normalized_value == "150"
    assert v.confidence == "medium"
    assert v.requires_verification
    assert any("mixed digit scripts" in i for i in v.issues)


@pytest.mark.parametrize(
    ("raw", "value"),
    [("0.035", "0.035"), ("۰٫۰۳۵", "0.035"), ("۱٫۳۰", "1.30"), ("1.30", "1.30")],
)
def test_decimals_are_preserved_exactly(raw: str, value: str) -> None:
    v = parse_numeric(raw)
    assert v.normalized_value == value  # string, not float: trailing zeros kept
    assert v.as_decimal() == Decimal(value)


def test_dropped_decimal_separator_is_a_different_value() -> None:
    assert not same_value("0.035", parse_numeric("0035").normalized_value)
    assert not same_value("0.035", parse_numeric("0.35").normalized_value)
    assert not same_value("1.30", parse_numeric("۱۳۰").normalized_value)
    assert same_value("1.30", "1.3")


def test_slash_is_decimal_only_in_numeric_cell_context() -> None:
    cell = parse_numeric("۱/۰۹", slash_is_decimal=True)
    assert cell.normalized_value == "1.09"
    assert any("'/' read as decimal" in i for i in cell.issues)
    date = parse_numeric("1404/04/19")
    assert date.normalized_value is None
    assert date.requires_verification


def test_persian_thousands_separator_is_removed_and_noted() -> None:
    v = parse_numeric("۱۱٬۳۴۰")
    assert v.normalized_value == "11340"
    assert v.confidence == "medium"


@pytest.mark.parametrize("raw", ["1,5", "۱،۵"])
def test_ambiguous_comma_is_not_resolved(raw: str) -> None:
    v = parse_numeric(raw)
    assert v.normalized_value is None
    assert v.confidence == "low"


@pytest.mark.parametrize("minus", ["-", "\u2212", "\u2013"])
def test_minus_variants(minus: str) -> None:
    v = parse_numeric(f"{minus}2.5")
    assert v.normalized_value == "-2.5"


@pytest.mark.parametrize(
    ("raw", "issue"),
    [(".35", "no digit before"), ("1.", "no digit after"), ("1.2.3", "more than one"),
     ("Vee", "unexpected character"), ("", "empty")],
)  # fmt: skip
def test_malformed_values_require_verification(raw: str, issue: str) -> None:
    v = parse_numeric(raw)
    assert v.requires_verification
    assert any(issue in i for i in v.issues)


def test_never_guesses_a_missing_leading_zero() -> None:
    v = parse_numeric(".35")
    assert v.normalized_value == ".35"  # reported as read, not "0.35"
    assert v.confidence == "low"


def test_stray_combining_marks_are_noted() -> None:
    v = parse_numeric("\u064d ۱/۸۰", slash_is_decimal=True)
    assert v.normalized_value == "1.80"
    assert v.confidence == "medium"


def test_source_confidence_caps_result() -> None:
    v = parse_numeric("625", source_confidence="medium")
    assert v.requires_verification
    assert v.raw_detected_value == "625"
