"""Character classification for Persian/Latin/digit content.

These helpers only *count and classify* characters; they never modify text. Persian
normalization is a separate concern (Phase 2).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

ZWNJ = "\u200c"

# Hyphen-like characters seen (or plausible) in section numbers such as «19-5-1-2»
# (the main PDF writes them with U+2011).
HYPHEN_VARIANTS = frozenset(
    "-"  # U+002D hyphen-minus
    "\u2010"  # hyphen
    "\u2011"  # non-breaking hyphen (used by the main Mabhas 19 PDF)
    "\u2012"  # figure dash
    "\u2013"  # en dash
    "\u2014"  # em dash
    "\u2212"  # minus sign
    "\ufe63"  # small hyphen-minus
    "\uff0d"  # fullwidth hyphen-minus
)

_ASCII_DIGITS = "0123456789"
_ARABIC_INDIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"  # U+0660..U+0669
_PERSIAN_DIGITS = "۰۱۲۳۴۵۶۷۸۹"  # U+06F0..U+06F9
_DIGIT_VALUES = {
    **{c: i for i, c in enumerate(_ASCII_DIGITS)},
    **{c: i for i, c in enumerate(_ARABIC_INDIC_DIGITS)},
    **{c: i for i, c in enumerate(_PERSIAN_DIGITS)},
}

# Arabic-script letter blocks: Arabic, Arabic Supplement, Arabic Extended-A and the two
# presentation-form blocks (which indicate unnormalized glyph-level extraction).
_ARABIC_SCRIPT_RANGES = (
    (0x0600, 0x06FF),
    (0x0750, 0x077F),
    (0x08A0, 0x08FF),
    (0xFB50, 0xFDFF),
    (0xFE70, 0xFEFF),
)
_PRESENTATION_FORM_RANGES = ((0xFB50, 0xFDFF), (0xFE70, 0xFEFF))
_LATIN_RANGES = ((0x0041, 0x005A), (0x0061, 0x007A), (0x00C0, 0x024F))

# Extraction artefact of the Word→PDF export: lam-alef ligatures come out reversed as
# alef-lam, producing sequences such as «االجرا» (for «الاجرا») that do not occur in
# well-formed Persian.
_REVERSED_LAM_ALEF = re.compile("اال")
_ARABIC_SCRIPT_WORD = re.compile("[\u0600-\u06ff\u0750-\u077f\ufb50-\ufdff\ufe70-\ufeff]+")


def _in_ranges(cp: int, ranges: tuple[tuple[int, int], ...]) -> bool:
    return any(lo <= cp <= hi for lo, hi in ranges)


def is_persian_letter(ch: str) -> bool:
    """True for an Arabic-script *letter* (Persian/Arabic), excluding digits and marks."""
    return _in_ranges(ord(ch), _ARABIC_SCRIPT_RANGES) and unicodedata.category(ch).startswith("L")


def is_latin_letter(ch: str) -> bool:
    return _in_ranges(ord(ch), _LATIN_RANGES) and unicodedata.category(ch).startswith("L")


def digit_value(ch: str) -> int | None:
    """Value of an ASCII, Arabic-Indic or Persian (Extended Arabic-Indic) digit."""
    return _DIGIT_VALUES.get(ch)


def is_digit(ch: str) -> bool:
    return ch in _DIGIT_VALUES


def digits_to_int(text: str) -> int | None:
    """Parse a run made only of (possibly mixed-script) digits; None otherwise."""
    if not text:
        return None
    value = 0
    for ch in text:
        d = digit_value(ch)
        if d is None:
            return None
        value = value * 10 + d
    return value


@dataclass(frozen=True)
class CharProfile:
    """Counts of character classes in a piece of extracted text."""

    total_chars: int  # non-whitespace characters, excluding ZWNJ and other format chars
    persian_letters: int
    latin_letters: int
    ascii_digits: int
    arabic_indic_digits: int
    persian_digits: int
    zwnj: int
    presentation_forms: int
    reversed_lam_alef: int
    arabic_script_words: int
    single_letter_words: int

    @property
    def digits(self) -> int:
        return self.ascii_digits + self.arabic_indic_digits + self.persian_digits


def char_profile(text: str) -> CharProfile:
    total = persian = latin = ascii_d = arabic_d = persian_d = zwnj = pres = 0
    for ch in text:
        if ch == ZWNJ:
            zwnj += 1
            continue
        if ch.isspace() or unicodedata.category(ch) == "Cf":
            continue
        total += 1
        cp = ord(ch)
        if ch in _ASCII_DIGITS:
            ascii_d += 1
        elif ch in _ARABIC_INDIC_DIGITS:
            arabic_d += 1
        elif ch in _PERSIAN_DIGITS:
            persian_d += 1
        elif is_persian_letter(ch):
            persian += 1
            if _in_ranges(cp, _PRESENTATION_FORM_RANGES):
                pres += 1
        elif is_latin_letter(ch):
            latin += 1

    words = _ARABIC_SCRIPT_WORD.findall(text)
    # «و» ("and") is a legitimate one-letter word; other isolated letters indicate words
    # broken apart by extraction.
    single = sum(1 for w in words if len(w) == 1 and w != "و")
    return CharProfile(
        total_chars=total,
        persian_letters=persian,
        latin_letters=latin,
        ascii_digits=ascii_d,
        arabic_indic_digits=arabic_d,
        persian_digits=persian_d,
        zwnj=zwnj,
        presentation_forms=pres,
        reversed_lam_alef=len(_REVERSED_LAM_ALEF.findall(text)),
        arabic_script_words=len(words),
        single_letter_words=single,
    )
