"""Heuristic classification of a page's text-layer quality."""

from __future__ import annotations

from mabhas19.ingest.audit.models import TextQuality
from mabhas19.text.charclass import CharProfile

# Fewer meaningful characters than this means the page has no usable text layer.
MIN_TEXT_CHARS = 20
# Below this many Persian letters (and share of letters) the layer is Latin/digits only.
MIN_PERSIAN_LETTERS = 10
MIN_PERSIAN_SHARE = 0.05
# Damage thresholds. Well-formed Persian has roughly 0.01-0.03 ZWNJ per Persian letter;
# the damaged Word export of the main document shows 0.08-0.31.
MAX_ZWNJ_RATIO = 0.06
MAX_SINGLE_LETTER_WORD_RATIO = 0.15
MAX_PRESENTATION_FORM_RATIO = 0.05
MAX_REVERSED_LAM_ALEF = 1


def quality_signals(p: CharProfile) -> dict[str, float]:
    persian = max(p.persian_letters, 1)
    return {
        "zwnj_per_persian_letter": round(p.zwnj / persian, 4),
        "single_letter_word_ratio": round(p.single_letter_words / max(p.arabic_script_words, 1), 4),
        "presentation_form_ratio": round(p.presentation_forms / persian, 4),
        "reversed_lam_alef_count": float(p.reversed_lam_alef),
    }


def classify_text_quality(p: CharProfile) -> tuple[TextQuality, list[str]]:
    """Return the quality class and the human-readable reasons for it."""
    letters_and_digits = p.persian_letters + p.latin_letters + p.digits
    if letters_and_digits < MIN_TEXT_CHARS:
        return TextQuality.NO_TEXT, [f"only {letters_and_digits} letters/digits extracted"]

    letters = p.persian_letters + p.latin_letters
    persian_share = p.persian_letters / letters if letters else 0.0
    if p.persian_letters < MIN_PERSIAN_LETTERS or persian_share < MIN_PERSIAN_SHARE:
        return TextQuality.LATIN_DIGITS_ONLY, [
            f"{p.persian_letters} Persian letters vs {p.latin_letters} Latin letters "
            f"and {p.digits} digits"
        ]

    s = quality_signals(p)
    reasons = []
    if s["zwnj_per_persian_letter"] > MAX_ZWNJ_RATIO:
        reasons.append(f"ZWNJ ratio {s['zwnj_per_persian_letter']:.3f} > {MAX_ZWNJ_RATIO}")
    if s["single_letter_word_ratio"] > MAX_SINGLE_LETTER_WORD_RATIO:
        reasons.append(
            f"isolated-letter word ratio {s['single_letter_word_ratio']:.3f} "
            f"> {MAX_SINGLE_LETTER_WORD_RATIO}"
        )
    if s["presentation_form_ratio"] > MAX_PRESENTATION_FORM_RATIO:
        reasons.append(f"presentation-form ratio {s['presentation_form_ratio']:.3f}")
    if p.reversed_lam_alef > MAX_REVERSED_LAM_ALEF:
        reasons.append(f"{p.reversed_lam_alef} reversed lam-alef sequences («اال»)")
    if reasons:
        return TextQuality.DAMAGED, reasons
    return TextQuality.GOOD, ["no damage signals above thresholds"]
