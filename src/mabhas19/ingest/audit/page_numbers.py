"""Candidate printed-page markers (machine suggestions only; never verified here)."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from mabhas19.ingest.audit.models import Confidence, PageMapCandidate
from mabhas19.text.charclass import ZWNJ, digits_to_int

# Front matter of the main document is lettered in Arabic alphabetical (hijāʾī) order,
# observed as «أ ب ت ث ج ح خ …», i.e. without the Persian-only letters پ چ ژ گ.
HIJAI_ORDER = "ابتثجحخدذرزسشصضطظعغفقکلمنوهی"
_LETTER_ALIASES = {"أ": "ا", "إ": "ا", "آ": "ا", "ك": "ک", "ي": "ی", "ى": "ی", "ة": "ه"}

METHOD_TEXT_FOOTER = "text_layer_footer_band"
METHOD_VECTOR_GLYPH = "vector_glyph_presence_in_footer_band+sequence_inference"
METHOD_NONE = "none"


@dataclass(frozen=True)
class FooterObservation:
    """What was found in the footer band of one page."""

    pdf_page: int
    text_tokens: tuple[str, ...]  # visible words in the footer band, extraction order
    vector_glyph_count: int  # drawn glyph clusters centred in the footer band
    hidden_marker_tokens: tuple[str, ...] = ()  # marker-like text that does not render


def letter_index(letter: str) -> int | None:
    """1-based position of a front-matter letter in hijāʾī order."""
    letter = _LETTER_ALIASES.get(letter, letter)
    pos = HIJAI_ORDER.find(letter)
    return pos + 1 if pos >= 0 else None


def classify_marker(raw: str) -> tuple[str, str] | None:
    """('number', '12') / ('letter', 'ب') for a plausible page marker, else None."""
    token = raw.replace(ZWNJ, "").strip()
    if not token:
        return None
    number = digits_to_int(token)
    if number is not None:
        return "number", str(number)
    if len(token) == 1 and letter_index(token) is not None:
        return "letter", token
    return None


def _marker_from_tokens(tokens: tuple[str, ...]) -> tuple[str, str, str] | None:
    hits = [(raw, c) for raw in tokens if (c := classify_marker(raw)) is not None]
    if len(hits) != 1:
        return None
    raw, (kind, value) = hits[0]
    return raw, kind, value


def _offset(kind: str, value: str, pdf_page: int) -> int | None:
    if kind == "number":
        return int(value) - pdf_page
    index = letter_index(value)
    return None if index is None else index - pdf_page


def build_page_map_candidates(
    doc_id: str, observations: list[FooterObservation]
) -> list[PageMapCandidate]:
    """Turn footer observations into candidates, scoring them by sequence consistency.

    A text-layer marker is ``high`` confidence when an adjacent page carries a marker of
    the same kind with the same offset (printed - pdf_page), otherwise ``medium``.
    Pages whose marker exists only as vector glyphs get a value *inferred* from their
    ordinal among marked pages, at ``low`` confidence.
    """
    text_hits: dict[int, tuple[str, str, str]] = {}
    for obs in observations:
        hit = _marker_from_tokens(obs.text_tokens)
        if hit is not None:
            text_hits[obs.pdf_page] = hit

    offsets = {
        page: (kind, _offset(kind, value, page)) for page, (_, kind, value) in text_hits.items()
    }

    candidates: list[PageMapCandidate] = []
    vector_ordinal = 0
    for obs in observations:
        page = obs.pdf_page
        if page in text_hits:
            raw, kind, value = text_hits[page]
            mine = offsets[page]
            neighbours = [offsets.get(page - 1), offsets.get(page + 1)]
            consistent = any(n == mine for n in neighbours)
            notes = [] if consistent else ["offset differs from both neighbouring pages"]
            candidates.append(
                PageMapCandidate(
                    doc_id=doc_id,
                    pdf_page=page,
                    printed_page_candidate=value,
                    marker_kind=kind,
                    raw_marker=raw,
                    confidence=Confidence.HIGH if consistent else Confidence.MEDIUM,
                    detection_method=METHOD_TEXT_FOOTER,
                    notes=notes,
                )
            )
        elif obs.vector_glyph_count > 0 and not text_hits:
            vector_ordinal += 1
            inferred = str(vector_ordinal)
            notes = [
                "marker is drawn as vector outlines (no text); value inferred from the "
                "page's ordinal among marked pages - requires OCR/visual verification"
            ]
            if obs.vector_glyph_count != len(inferred):
                notes.append(
                    f"{obs.vector_glyph_count} glyph cluster(s) drawn but inferred value "
                    f"has {len(inferred)} digit(s)"
                )
            candidates.append(
                PageMapCandidate(
                    doc_id=doc_id,
                    pdf_page=page,
                    printed_page_candidate=inferred,
                    marker_kind="number",
                    raw_marker=None,
                    confidence=Confidence.LOW,
                    detection_method=METHOD_VECTOR_GLYPH,
                    notes=notes,
                )
            )
        else:
            notes = ["no visible page marker found in the footer band"]
            if obs.hidden_marker_tokens:
                notes.append(
                    "text layer contains marker-like text "
                    f"{list(obs.hidden_marker_tokens)} that does not render visibly; ignored"
                )
            candidates.append(
                PageMapCandidate(
                    doc_id=doc_id,
                    pdf_page=page,
                    printed_page_candidate=None,
                    marker_kind="absent",
                    raw_marker=None,
                    confidence=Confidence.NONE,
                    detection_method=METHOD_NONE,
                    notes=notes,
                )
            )
    return candidates


@dataclass
class OffsetRun:
    """Consecutive pages sharing one marker kind and one (printed - pdf_page) offset."""

    marker_kind: str
    offset_printed_minus_pdf: int | None
    first_pdf_page: int
    last_pdf_page: int


def offset_runs(candidates: list[PageMapCandidate]) -> list[OffsetRun]:
    runs: list[OffsetRun] = []
    for c in candidates:
        offset = (
            None
            if c.printed_page_candidate is None
            else _offset(c.marker_kind, c.printed_page_candidate, c.pdf_page)
        )
        last = runs[-1] if runs else None
        if (
            last is not None
            and (last.marker_kind, last.offset_printed_minus_pdf) == (c.marker_kind, offset)
            and last.last_pdf_page == c.pdf_page - 1
        ):
            last.last_pdf_page = c.pdf_page
        else:
            runs.append(OffsetRun(c.marker_kind, offset, c.pdf_page, c.pdf_page))
    return runs


def confidence_counts(candidates: list[PageMapCandidate]) -> dict[str, int]:
    counts = Counter(c.confidence.value for c in candidates)
    return {level.value: counts.get(level.value, 0) for level in Confidence}
