from __future__ import annotations

import json
from functools import cache
from typing import Any

from mabhas19.core.paths import CURATED_DIR
from mabhas19.indexing.chunking import build_chunks
from mabhas19.retrieval.eval import (
    CaseResult,
    _structured,
    check_evidence_ids,
    route_matrix,
    unsupported_scores,
    unsupported_scores_strict,
)
from mabhas19.retrieval.hybrid import Retriever, numbered_lines, rrf
from mabhas19.retrieval.lexical import BM25
from mabhas19.retrieval.models import (
    RetrievalResult,
    RouteDecision,
    StructuredHit,
    StructuredNoMatch,
)
from mabhas19.retrieval.normalization import index_terms, tokens
from mabhas19.retrieval.router import Router
from mabhas19.retrieval.structured import NO_5C_NOTE, StructuredStore, density_in
from mabhas19.text.normalize import normalize_for_search as norm


@cache
def _store() -> StructuredStore:
    return StructuredStore(CURATED_DIR)


def _block(i: int, kind: str, text: str, **kw: Any) -> dict[str, Any]:
    return {
        "block_id": f"b{i:02d}",
        "kind": kind,
        "text": text,
        "pdf_page": kw.get("page", 1),
        "printed_page": "1",
        "retrieval": {"include": kw.get("include", True)},
        "font_size": kw.get("font", 12.0),
        "section_path": kw.get("path", [{"number": "19-1", "title": "تعاریف"}]),
    }


# ---------------------------------------------------------------- normalization


def test_search_normalization_is_spacing_and_letter_variant_insensitive() -> None:
    a, b, c = index_terms("شبیه‌سازی"), index_terms("شبیه سازی"), index_terms("شبیهسازی")
    assert "شبیهسازی" in a
    assert "شبیهسازی" in b
    assert c == ["شبیهسازی"]
    assert tokens("حداكثر دمای گرمايش") == tokens("حداکثر دمای گرمایش")
    assert tokens("بند ۱۹-۲-۱-۹") == ["بند", "19-2-1-9"]


# ---------------------------------------------------------------- chunking


def test_chunks_follow_sections_and_keep_provenance() -> None:
    blocks = [
        _block(1, "heading", "۱۹-۱-۱ تعریف یک"),
        _block(2, "footnote", "Term\nمتن تعریف که در فونت بدنه است " * 2, font=12.0),
        _block(3, "footnote", "۱ پانویس کوچک", font=7.0),
        _block(4, "heading", "۱۹-۱-۲ تعریف دو"),
        _block(5, "paragraph", "متن " * 20),
        _block(6, "table_caption", "جدول ۱۹-۵-۱: عنوان"),
        _block(7, "caption_continuation", "(m2.k/W)"),
        _block(8, "running_header", "سرصفحه", include=False),
    ]
    chunks = build_chunks(blocks, "doc")
    kinds = {c.kind: c.block_ids for c in chunks}
    ids = [i for c in chunks for i in c.block_ids]
    assert sorted(ids) == [f"b{i:02d}" for i in range(1, 8)]  # each eligible block exactly once
    assert ("b01", "b02") in [c.block_ids[:2] for c in chunks]  # 12 pt definition stays in flow
    assert kinds["footnotes"] == ("b03",)
    assert kinds["caption"] == ("b06", "b07")


def test_numbered_lines_find_clauses() -> None:
    chunks = build_chunks([_block(1, "clause", "۱۹-۲-۱-۹ اطلاعات تفصیلی " + "متن " * 10)], "doc")
    assert numbered_lines(chunks) == {"19-2-1-9": [chunks[0].chunk_id]}


# ---------------------------------------------------------------- ranking


def test_bm25_and_rrf() -> None:
    bm = BM25([["عایق", "حرارتی"], ["پنجره"], ["عایق"]])
    assert next(i for i, _, _ in bm.scores(["عایق", "حرارتی"])) == 0
    fused = rrf([["a", "b"], ["b", "c"]])
    assert fused[0][0] == "b"


# ---------------------------------------------------------------- structured safety


def test_isfahan_returns_both_sources_with_the_conflict() -> None:
    hits, conflicts = _store().climate(norm("رده اقلیمی اصفهان"))
    codes = {h.record_id: h.values["climate_class"]["code"] for h in hits}
    assert codes == {"app02-p002-r07": "4B", "main-t19-3-2-r04": "3B"}
    assert [(c.issue_id, c.values) for c in conflicts] == [("SI-0009", ["3B", "4B"])]


def test_kerman_station_code_stays_as_printed() -> None:
    (hit,) = _store().wmo(norm("ایستگاه ۴۸۸۴۱۰"))
    assert hit.values["wmo_code"]["printed"] == "۴۸۸۴۱۰"
    assert hit.source_issues == ["SI-0010"]


def test_no_eui_is_derived_for_5c() -> None:
    hits, notes, no_match = _store().eui(
        norm("شدت مصرف انرژی مسکونی ≥ ۳۰۰۰ مترمربع اقلیم 5C رده C")
    )
    assert hits == []
    assert notes == [NO_5C_NOTE]
    # v3: issues come from the curated Appendix 3 metadata for 5C (SI-0002, SI-0001) and from
    # the resolved occupancy row used by the lookup («مسکونی ≥ ۳۰۰۰»: SI-0003)
    assert [(n.reason, sorted(n.source_issues)) for n in no_match] == [
        ("climate_class_not_in_table", ["SI-0001", "SI-0002", "SI-0003"])
    ]


def test_density_on_a_printed_bound_returns_both_rows() -> None:
    hits, notes, _ = _store().materials(
        norm("ضریب هدایت حرارتی اندود و ملات آهکی یا سیمانی چگالی ۱۸۰۰")
    )
    assert {h.record_id for h in hits} == {"app07-p003-r02", "app07-p003-r03"}
    assert notes
    assert density_in(norm("پشم سنگ با چگالی ۳۰")) == 30.0


def test_p64_symbols_stay_uninterpreted() -> None:
    hits, no_match, _ = _store().main_table(
        "m19-ed5-1404-main.t19-2-1", norm("گروه الف سال اول عایق‌کاری پوسته خارجی")
    )
    assert no_match == []
    assert [h.record_id for h in hits] == ["main-t19-2-1-r02c05"]
    symbol = hits[0].values["printed_symbol"]
    assert symbol["symbol"] == "check_mark"
    assert not any(isinstance(v, bool) for v in hits[0].values.values())


# ---------------------------------------------------------------- router


def _router() -> Router:
    return Router(_store(), {"شدت", "مصرف", "انرژی", "دانشگاه", "اموزش", "عالی", "اقلیم"})


def test_router_is_token_bounded_and_needs_entities() -> None:
    r = _router()
    assert (
        r.route("شدت مصرف انرژی مجاز دانشگاه و آموزش عالی در اقلیم 0B رده A").route == "structured"
    )
    concept = r.route("شدت مصرف انرژی ساختمان ایده‌آل")
    assert concept.route != "structured"  # vocabulary without occupancy/climate class
    assert r.route("قیمت آجر در بازار").route == "unsupported"
    mixed = r.route("رده اقلیمی اصفهان چیست و دسته‌بندی اقلیم چگونه انجام می‌شود")
    assert mixed.route == "mixed"
    assert mixed.reasons


# ---------------------------------------------------------------- evaluation semantics


def _result(ids: list[str], no_match: list[str] | None = None) -> RetrievalResult:
    hits = [StructuredHit("t", "d", i, 1, "1", [], {}, {}, [], "x") for i in ids]
    nm = [StructuredNoMatch(ds, "unknown_material", "", {}, None) for ds in no_match or []]
    return RetrievalResult("q", "q", RouteDecision("structured", []), structured=hits, no_match=nm)


def _case() -> CaseResult:
    return CaseResult("R", "c", "q", "structured", "structured", [], True)


def test_record_exact_match_semantics() -> None:
    res = _case()
    _structured(_result(["a", "b"]), {"records": ["a"], "acceptable_records": ["b"]}, res)
    assert res.record_exact
    res = _case()
    _structured(_result(["a", "x"]), {"records": ["a"]}, res)
    assert res.record_exact is False
    assert res.structured_extra == ["x"]
    res = _case()
    _structured(_result(["b"]), {"records": ["a"], "acceptable_records": ["b"]}, res)
    assert res.record_exact is False
    assert res.structured_missing == ["a"]


def test_records_are_scored_at_record_level_even_with_tables() -> None:
    ev = {"records": ["a"], "tables": ["t"]}
    wrong = _case()
    _structured(_result(["z"]), ev, wrong)  # wrong row of the right table
    assert wrong.record_exact is False
    assert wrong.structured_exact_original  # the historical scorer accepted it
    typed = _case()
    _structured(_result([], no_match=["t"]), ev, typed)  # typed no-match for an expected table
    assert typed.record_exact is False
    assert typed.structured_exact_original
    assert typed.table_match is None
    assert typed.typed_no_match is None


def test_typed_no_match_and_table_match() -> None:
    ok = _case()
    _structured(_result([], no_match=["t"]), {"no_records": True, "tables": ["t"]}, ok)
    assert ok.typed_no_match
    rows = _case()
    _structured(_result(["a"]), {"no_records": True, "tables": ["t"]}, rows)
    assert rows.typed_no_match is False
    silent = _case()
    _structured(_result([]), {"no_records": True, "tables": ["t"]}, silent)
    assert silent.typed_no_match is False  # absence must be typed
    table = _case()
    _structured(_result(["a"]), {"tables": ["t"]}, table)
    assert table.table_match
    assert table.record_exact is None


def test_unknown_evidence_ids_are_reported() -> None:
    cases = [{"id": "R1", "evidence": {"blocks": ["nope"], "records": ["a"]}}]
    assert check_evidence_ids(cases, set(), set(), {"a"}, set()) == ["R1: unknown blocks ['nope']"]


def test_retriever_end_to_end_on_curated() -> None:
    path = CURATED_DIR / "text" / "m19-ed5-1404-main" / "blocks.jsonl"
    blocks = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
    retriever = Retriever(build_chunks(blocks, "m19"), _store())
    res = retriever.retrieve("بند ۱۹-۲-۱-۹ درباره پیوست‌ها")
    assert res.route.route == "prose"
    assert "main-p057-b006" in res.prose[0].block_ids


def test_route_matrix_and_unsupported_scores() -> None:
    rs = [
        CaseResult("a", "c", "q", "unsupported", "unsupported", [], True),
        CaseResult("b", "c", "q", "unsupported", "prose", [], False),
        CaseResult("c", "c", "q", "prose", "unsupported", [], False),
    ]
    m = route_matrix(rs)
    assert m["unsupported"]["prose"] == 1
    assert m["prose"]["unsupported"] == 1
    assert unsupported_scores(rs) == {
        "tp": 1,
        "fp": 1,
        "fn": 1,
        "allowed_other": 0,
        "precision": 0.5,
        "recall": 0.5,
    }


def test_unsupported_scores_honor_acceptable_routes() -> None:
    rs = [
        # expected unsupported, prose explicitly acceptable -> neither tp nor fn
        CaseResult("a", "c", "q", "unsupported", "prose", [], True, ["prose"]),
        # expected prose, unsupported explicitly acceptable -> tp, not fp
        CaseResult("b", "c", "q", "prose", "unsupported", [], True, ["unsupported"]),
        CaseResult("c", "c", "q", "structured", "unsupported", [], False),
    ]
    assert unsupported_scores(rs) == {
        "tp": 1,
        "fp": 1,
        "fn": 0,
        "allowed_other": 1,
        "precision": 0.5,
        "recall": 1.0,
    }
    strict = unsupported_scores_strict(rs)
    assert (strict["tp"], strict["fp"], strict["fn"]) == (0, 2, 1)
