from __future__ import annotations

from collections import Counter

from mabhas19.ingest.main_errata import (
    ZWNJ,
    Finding,
    apply_proposals,
    bracket_findings,
    caption_continuations,
    mirrored_spans,
    split_words,
)


def _freq() -> Counter[str]:
    return Counter({"است": 50, "مصنوعی": 30, "عملیات": 9, "شبیه": 40, "سازی": 60,
                    "شبیه" + ZWNJ + "سازی": 30, "حد": 5, "اکثر": 5, "حداکثر": 40,
                    "که": 300})  # fmt: skip


def test_split_word_joins_fragments_but_not_printed_compounds() -> None:
    found, compounds = split_words("این اس ت که شبیه سازی مصنوع ی و ع ملیات", _freq())
    pairs = {f.found: f.proposed for f in found}
    assert pairs["اس ت"] == "است"
    assert pairs["مصنوع ی"] == "مصنوعی"
    assert pairs["ع ملیات"] == "عملیات"
    assert "شبیه سازی" not in pairs  # ZWNJ compound: printed spacing, kept
    assert compounds == 1


def test_source_faithful_forms_are_never_proposed() -> None:
    found, _ = split_words("حد اکثر", _freq())
    assert found == []


def test_detached_tanwin_is_a_deterministic_join() -> None:
    found, _ = split_words("کاملا \u064b مستقل", Counter())
    assert [(f.kind, f.proposed, f.rule) for f in found] == [("detached_mark", "کاملا\u064b", "T1")]


def test_mirrored_spans_ignore_list_markers_and_normal_pairs() -> None:
    assert mirrored_spans("الف) گرم (2A)، کم باران (2B)") == ([], 0)
    text = "توزیع نور )نسبت روشنایی( باید"
    spans, unmatched = mirrored_spans(text)
    assert [text[a:b] for a, b in spans] == [")نسبت روشنایی("]
    assert unmatched == 0
    f = bracket_findings(text)[0]
    assert (f.proposed, f.rule) == ("(نسبت روشنایی)", "B2")


def test_cross_block_bracket_is_reported_as_malformed() -> None:
    f = bracket_findings("شبکه‌های ارتباطی )از جمله اینترنت")
    assert [x.kind for x in f] == ["malformed_brackets"]


def test_caption_continuation_is_a_short_centred_line_below_a_caption() -> None:
    blocks = [
        {"block_id": "c", "pdf_page": 1, "kind": "table_caption", "bbox": [50, 60, 400, 72],
         "text": "جدول ۱"},
        {"block_id": "u", "pdf_page": 1, "kind": "heading_unnumbered", "bbox": [180, 80, 270, 92],
         "text": "اقلیم‌های مختلف (m2.k/W)"},
        {"block_id": "p", "pdf_page": 1, "kind": "paragraph", "bbox": [50, 300, 400, 320],
         "text": "متن"},
    ]  # fmt: skip
    found = caption_continuations(blocks)
    assert list(found) == ["u"]
    assert found["u"].extra["caption_block"] == "c"


def test_apply_proposals_replaces_every_occurrence() -> None:
    f = Finding("split_word", "اس اس", "اساس", "")
    assert apply_proposals("بر اس اس و بر اس اس", [f]) == "بر اساس و بر اساس"
