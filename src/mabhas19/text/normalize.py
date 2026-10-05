"""Persian text normalization for search and for evaluation.

Three representations are kept separate throughout the project:

* ``raw_text`` - exactly what an extractor produced (never modified);
* ``repaired_display_text`` - faithful to the printed page, with extraction artefacts fixed;
* ``normalized_search_text`` - produced here, for indexing and matching only.

Nothing in this module is applied to display text.
"""

from __future__ import annotations

import re
import unicodedata

from mabhas19.text.charclass import HYPHEN_VARIANTS, ZWNJ

TATWEEL = "ـ"
# Arabic-script code points folded to their Persian equivalents for search.
_LETTER_FOLDS = {
    "ك": "ک",  # ARABIC KAF -> KEHEH (Persian kaf)
    "ي": "ی",  # ARABIC YEH -> FARSI YEH
    "ى": "ی",  # ALEF MAKSURA -> FARSI YEH
    "ۀ": "ه",  # HEH WITH YEH ABOVE -> HEH (the yeh mark is dropped below)
    "ة": "ه",  # TEH MARBUTA -> HEH
    "أ": "ا",  # ALEF WITH HAMZA ABOVE -> ALEF
    "إ": "ا",  # ALEF WITH HAMZA BELOW -> ALEF
    "آ": "ا",  # ALEF WITH MADDA -> ALEF
    "ؤ": "و",  # WAW WITH HAMZA -> WAW
}
_DIGIT_FOLDS = {
    **{chr(0x0660 + i): str(i) for i in range(10)},
    **{chr(0x06F0 + i): str(i) for i in range(10)},
}
_PERSIAN_DIGITS = {str(i): chr(0x06F0 + i) for i in range(10)}
_PUNCT_FOLDS = {
    "،": ",",  # ARABIC COMMA
    "؛": ";",  # ARABIC SEMICOLON
    "؟": "?",  # ARABIC QUESTION MARK
    "٫": ".",  # ARABIC DECIMAL SEPARATOR
    "٬": ",",  # ARABIC THOUSANDS SEPARATOR
    "«": '"',
    "»": '"',
    "“": '"',
    "”": '"',
    "‘": "'",
    "’": "'",
}
_BIDI_CONTROLS = re.compile("[\u200e\u200f\u202a-\u202e\u2066-\u2069]")
_WHITESPACE = re.compile(r"\s+")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:!?)\]])")
_SPACE_AFTER_OPEN = re.compile(r"([(\[])\s+")


def to_ascii_digits(text: str) -> str:
    return "".join(_DIGIT_FOLDS.get(c, c) for c in text)


def to_persian_digits(text: str) -> str:
    """ASCII and Arabic-Indic digits to Persian (extended Arabic-Indic) digits."""
    return "".join(_PERSIAN_DIGITS.get(_DIGIT_FOLDS.get(c, c), c) for c in text)


def fold_hyphens(text: str) -> str:
    return "".join("-" if c in HYPHEN_VARIANTS else c for c in text)


def remove_marks(text: str) -> str:
    """Drop tatweel and combining marks (harakat, superscript alef, hamza marks)."""
    return "".join(c for c in text if c != TATWEEL and unicodedata.category(c) != "Mn")


def normalize_for_search(text: str) -> str:
    """Normalized representation for indexing and matching (never for display).

    NFKC (folds presentation forms, superscripts), Persian letter forms, ASCII digits,
    ASCII hyphens, no tatweel/diacritics/bidi controls, ZWNJ treated as a word break,
    lower-cased Latin, single spaces.
    """
    text = unicodedata.normalize("NFKC", text)
    text = _BIDI_CONTROLS.sub("", text)
    text = "".join(_LETTER_FOLDS.get(c, c) for c in text)
    text = remove_marks(text)
    text = fold_hyphens(to_ascii_digits(text))
    text = text.replace(ZWNJ, " ").lower()
    return _WHITESPACE.sub(" ", text).strip()


def normalize_for_comparison(text: str) -> str:
    """Search normalization plus punctuation folding, for scoring extractions.

    Ignores superficial differences (Arabic vs Latin punctuation, quote styles, spaces
    around punctuation) that do not change meaning.
    """
    text = "".join(_PUNCT_FOLDS.get(c, c) for c in text)
    text = normalize_for_search(text)
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
    return _SPACE_AFTER_OPEN.sub(r"\1", text)
