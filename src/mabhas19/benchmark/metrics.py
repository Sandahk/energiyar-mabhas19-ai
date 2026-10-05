"""Extraction metrics: CER/WER of a reference item located inside a page's output."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

from mabhas19.text.normalize import normalize_for_comparison

WINDOW_SLACK = 0.25  # extra context searched around the fuzzy match, relative to item length


@dataclass(frozen=True)
class TextScore:
    cer: float  # raw characters (whitespace collapsed)
    wer: float
    cer_normalized: float  # after normalize_for_comparison
    wer_normalized: float
    matched_text: str  # best-matching span of the hypothesis (normalized form)


def _collapse(text: str) -> str:
    return " ".join(text.split())


def _semi_global(ref: Sequence[str], hyp: Sequence[str]) -> tuple[int, int, int]:
    """Edit distance of ``ref`` against its best-matching contiguous span of ``hyp``.

    Returns (distance, span_start, span_end). Free gaps at both ends of ``hyp``.
    """
    n, m = len(ref), len(hyp)
    prev = [0] * (m + 1)  # row 0: starting anywhere in hyp is free
    starts = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        cur_starts = [0] * (m + 1)
        for j in range(1, m + 1):
            sub = prev[j - 1] + (ref[i - 1] != hyp[j - 1])
            dele = prev[j] + 1
            ins = cur[j - 1] + 1
            best = min(sub, dele, ins)
            cur[j] = best
            cur_starts[j] = (
                starts[j - 1] if best == sub else starts[j] if best == dele else cur_starts[j - 1]
            )
        prev, starts = cur, cur_starts
    end = min(range(m + 1), key=lambda j: prev[j])
    return prev[end], starts[end], end


def _windowed(ref: str, hyp: str) -> str:
    """Narrow ``hyp`` to the region around the best fuzzy match of ``ref``."""
    if not ref or not hyp:
        return hyp
    align = fuzz.partial_ratio_alignment(ref, hyp)
    if align is None:
        return hyp
    slack = int(len(ref) * WINDOW_SLACK) + 5
    return hyp[max(0, align.dest_start - slack) : align.dest_end + slack]


def _rates(ref: str, hyp: str) -> tuple[float, float, str]:
    if not ref:
        return 0.0, 0.0, ""
    window = _windowed(ref, hyp)
    dist, s, e = _semi_global(ref, window)
    cer = dist / len(ref)
    ref_words = ref.split()
    win_words = _windowed(ref, hyp).split()
    wdist, _, _ = _semi_global(ref_words, win_words) if ref_words else (0, 0, 0)
    wer = wdist / max(len(ref_words), 1)
    return min(cer, 1.0), min(wer, 1.0), window[s:e]


def score_text(reference: str, hypothesis: str) -> TextScore:
    """Score one reference item against a whole page (or region) of extracted text."""
    cer, wer, _ = _rates(_collapse(reference), _collapse(hypothesis))
    cer_n, wer_n, matched = _rates(
        normalize_for_comparison(reference), normalize_for_comparison(hypothesis)
    )
    return TextScore(round(cer, 4), round(wer, 4), round(cer_n, 4), round(wer_n, 4), matched)


def label_match(expected: str | None, predicted: str | None) -> bool:
    """Exact match after comparison normalization (superficial Unicode differences ignored)."""
    if expected is None or predicted is None:
        return expected is None and predicted is None
    return normalize_for_comparison(expected) == normalize_for_comparison(predicted)


def char_error_rate(reference: str, hypothesis: str) -> float:
    """Plain (non-searching) CER between two strings."""
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return min(Levenshtein.distance(reference, hypothesis) / len(reference), 1.0)
