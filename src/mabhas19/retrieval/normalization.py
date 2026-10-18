"""Persian search normalization for queries and index text (never for source-facing text).

Builds on :func:`mabhas19.text.normalize.normalize_for_search` (NFKC, Arabic/Persian letter
variants, ASCII digits, ZWNJ as a word break, no kashida/diacritics, single spaces) and adds
search tokenization:

* punctuation folded away, section numbers kept whole («19-5-1»);
* stopwords dropped for scoring;
* adjacent-token concatenations added (``شبیه سازی`` also yields ``شبیهسازی``), so ZWNJ,
  space and no-space spellings of a compound match each other. Source text keeps its
  printed spacing; only the search representation is spacing-insensitive.
"""

from __future__ import annotations

import re
from itertools import pairwise

from mabhas19.text.normalize import normalize_for_search

# Section/table numbers («19-5-1», «پ3-1»), decimals, Latin words, Persian words
_TOKEN = re.compile(r"[a-z]*\d+(?:[-./]\d+)*[a-z]*|[a-z]+|[؀-ۿ]+")
STOPWORDS = frozenset(
    {
        "و",
        "در",
        "به",
        "از",
        "که",
        "این",
        "را",
        "با",
        "است",
        "برای",
        "یا",
        "آن",
        "تا",
        "بر",
        "هر",
        "می",
        "شود",
        "شده",
        "باشد",
        "بوده",
        "نیز",
        "یک",
        "های",
        "ها",
        "ای",
        "هم",
        "اگر",
        "باید",
        "آنها",
        "آن‌ها",
        "چه",
        "چیست",
        "کدام",
        "چقدر",
        "چند",
        "چگونه",
        "است؟",
        "ی",
        "دارد",
        "دارند",
        "کند",
        "کنند",
        "the",
        "of",
        "and",
    }
)
QUESTION_WORDS = frozenset(
    {
        "چیست",
        "چه",
        "کدام",
        "چقدر",
        "چند",
        "چگونه",
        "چرا",
        "آیا",
        "کجا",
        "منظور",
    }
)


def normalize_query(text: str) -> str:
    return normalize_for_search(text)


def tokens(text: str, *, keep_stopwords: bool = False) -> list[str]:
    """Search tokens of an already normalized or raw text."""
    norm = normalize_for_search(text)
    out = _TOKEN.findall(norm)
    return out if keep_stopwords else [t for t in out if t not in STOPWORDS]


def index_terms(text: str) -> list[str]:
    """Tokens plus adjacent concatenations (spacing/ZWNJ-insensitive compounds)."""
    base = tokens(text, keep_stopwords=True)
    terms = [t for t in base if t not in STOPWORDS]
    terms += [a + b for a, b in pairwise(base) if _joinable(a, b)]
    return terms


def _joinable(a: str, b: str) -> bool:
    persian = re.compile(r"^[؀-ۿ]+$")
    return bool(persian.match(a) and persian.match(b)) and len(a) + len(b) >= 4


def content_tokens(text: str) -> list[str]:
    """Tokens without stopwords and question words (for coverage/route decisions)."""
    return [t for t in tokens(text) if t not in QUESTION_WORDS]
