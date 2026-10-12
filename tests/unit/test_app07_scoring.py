"""Offline appendix 7 scoring (Phase 2B): numeric categories, labels, cache, cost."""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from mabhas19.benchmark import app07_scoring as sc
from mabhas19.benchmark import phase2b as pb
from mabhas19.benchmark.references import VERIFIED, ReferenceItem

Z = "‌"


def row(item_id: str, text: str, fields: dict[str, Any]) -> ReferenceItem:
    base = {"density_qualifier": None, "density_min": None, "density_max": None}
    return ReferenceItem(
        item_id, "d", 1, "material_row", text, (0, 0, 1, 1), {**base, **fields},
        status=VERIFIED, verified_by="o", verified_on=date(2026, 10, 6),
    )  # fmt: skip


REFS = [
    row("r1", "بیش از ۲۰۰۰", {"density_qualifier": "بیش از", "density_min": "2000",
                              "conductivity": "1.80", "category": "۱. اندود"}),
    row("r2", "۱۸۰۰ تا ۲۰۰۰", {"density_min": "1800", "density_max": "2000",
                               "conductivity": "1.30", "category": "۱. اندود"}),
    row("r3", "سرب", {"density_min": "11340", "density_max": "11340", "conductivity": "35",
                      "category": "۱۴. فلزت", "material": "سرب"}),
    row("r4", "کمتر از ۱۰۰۰", {"density_qualifier": "کمتر از", "density_max": "1000",
                               "conductivity": "0.40", "category": "۱۴. فلزت", "material": "سرب"}),
]  # fmt: skip
PERFECT = {
    "r1": pb.ExtractedRow("بیش از ۲۰۰۰", "۱٫۸۰"),
    "r2": pb.ExtractedRow("۱۸۰۰ تا ۲۰۰۰", "۱٫۳۰"),
    "r3": pb.ExtractedRow("۱۱۳۴۰", "۳۵"),
    "r4": pb.ExtractedRow("کمتر از ۱۰۰۰", "۰٫۴۰"),
}


def test_perfect_extraction_scores_every_category() -> None:
    s = sc.score_numeric("m", REFS, PERFECT).summary()
    assert s == {
        "numeric_exact": "10/10",  # 4 conductivities + 6 density bounds
        "rows_fully_numeric": "4/4",
        "decimal_preserved": "3/3",
        "integer_conductivity": "1/1",
        "five_digit_density": "2/2",
        "ranges": "1/1",
        "qualifiers": "2/2",
    }


@pytest.mark.parametrize(
    ("item", "bad", "category"),
    [
        ("r2", pb.ExtractedRow("۱۸۰۰ تا ۲۰۰۰", "۱۳۰"), "decimal_preserved"),  # separator lost
        ("r3", pb.ExtractedRow("۱۱۳۴", "۳۵"), "five_digit_density"),  # digit dropped
        ("r3", pb.ExtractedRow("۱۱۳۴۰", "۳۸"), "integer_conductivity"),
        ("r2", pb.ExtractedRow("۲۰۰۰ تا ۱۸۰۰", "۱٫۳۰"), "ranges"),  # bounds swapped
        ("r1", pb.ExtractedRow("۲۰۰۰", "۱٫۸۰"), "qualifiers"),  # «بیش از» lost
    ],
)
def test_each_error_class_is_counted(item: str, bad: pb.ExtractedRow, category: str) -> None:
    found = {**PERFECT, item: bad}
    s = sc.score_numeric("m", REFS, found)
    assert s.summary()[category] != sc.score_numeric("m", REFS, PERFECT).summary()[category]
    assert s.summary()["rows_fully_numeric"] == "3/4"
    assert any(e["item"] == item for e in s.errors)


def test_values_come_from_raw_text_not_model_parsed_fields() -> None:
    lying = {**PERFECT, "r2": pb.ExtractedRow("۱۸۰۰ تا ۲۰۰۰", "۱۳۰", conductivity_parsed="1.30")}
    s = sc.score_numeric("m", REFS, lying)
    err = next(e for e in s.errors if e["item"] == "r2")
    assert (err["source"], err["extracted"]) == ("1.30", "130")


def test_missing_row_counts_as_numeric_failure() -> None:
    found = {k: v for k, v in PERFECT.items() if k != "r3"}
    s = sc.score_numeric("m", REFS, found).summary()
    assert s["numeric_exact"] == "7/10"  # r3 loses k, min and max
    assert s["rows_fully_numeric"] == "3/4"


def test_lenient_label_ignores_dash_colon_and_footnote_markers() -> None:
    assert sc.lenient_label("- متراکم") == sc.lenient_label("متراکم")
    assert sc.lenient_label("بتن هوادار اتوکلاو شده ۱:") == sc.lenient_label(
        "بتن هوادار اتوکلاو شده"
    )
    assert sc.lenient_label("- مسلح:۱") == sc.lenient_label("مسلح")
    assert sc.lenient_label("گرماشکن ۲") != sc.lenient_label("گرماشکنی")  # real text loss
    assert sc.lenient_label(None) is None


def test_label_scoring_strict_lenient_inherited_and_wrong_field() -> None:
    raw = {
        "r1": {"category_raw": "۱. اندود", "material_raw": None, "subgroup_raw": None,
               "condition_raw": None},
        "r2": {"category_raw": "۱. اندود", "material_raw": "- x", "subgroup_raw": None,
               "condition_raw": None},  # material where the reference has none
        "r3": {"category_raw": "۱۴. فلزت", "material_raw": "- سرب", "subgroup_raw": None,
               "condition_raw": None},  # dash only: lenient match
        "r4": {"category_raw": "۱۴. فلزت", "material_raw": None, "subgroup_raw": None,
               "condition_raw": None},  # inherited «سرب» correctly not repeated
    }  # fmt: skip
    s = sc.score_labels(REFS, raw)
    summary = s.summary()
    assert summary["material"] == {"strict": "1/3", "lenient": "2/3"}  # r4 inherited
    assert summary["context"] == {"inherited_not_repeated": 1}
    assert [m["item"] for m in s.mismatches] == ["r2"]


def test_fields_not_printed_on_the_page_are_scored_as_context() -> None:
    ref = row("app07-p006-r05", "x", {"density_min": "1", "density_max": "2",
                                       "conductivity": "0.1", "category": "۲. بتن"})  # fmt: skip
    empty = {
        "category_raw": None,
        "material_raw": None,
        "subgroup_raw": None,
        "condition_raw": None,
    }
    s = sc.score_labels([ref], {ref.id: empty})
    assert s.context["absent_correctly"] == 1
    assert s.counts["category_total"] == 0


def write(tmp: Path, rid: str, **data: Any) -> None:
    (tmp / f"{rid}.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def test_collect_cached_gemini_results_offline(tmp_path: Path) -> None:
    page_rows = [{"density_raw": "بیش از ۲۰۰۰", "conductivity_raw": "۱٫۸۰",
                  "category_raw": "۱. اندود"},
                 {"density_raw": None, "conductivity_raw": None, "material_raw": "heading"},
                 {"density_raw": "۱۸۰۰ تا ۲۰۰۰", "conductivity_raw": "۱٫۳۰"}]  # fmt: skip
    write(tmp_path, "p.page", status="ok", text=json.dumps({"rows": page_rows}))
    write(
        tmp_path,
        "r1.crop",
        status="ok",
        text=json.dumps({"density_raw": "x", "conductivity_raw": "y"}),
    )
    write(tmp_path, "r2.crop", status="error", text=None)
    g = sc.collect_gemini(tmp_path, "p", REFS[:2])
    assert g.detection == {"expected": 2, "returned": 3, "matched": 2, "missing": [], "extra": [2]}
    assert g.page_raw["r1"]["category_raw"] == "۱. اندود"
    assert set(g.crop) == {"r1"}


def test_usage_and_cost_summary(tmp_path: Path) -> None:
    usage = {"prompt_tokens": 1353, "output_tokens": 1948, "thinking_tokens": 9640}
    write(tmp_path, "a.page", status="ok", latency_s=76.64, usage=usage,
          attempts=[{"attempt": 1, "error": None}])  # fmt: skip
    write(tmp_path, "b.crop", status="error", attempts=[{"attempt": 1, "error": "server"}])
    s = sc.usage_summary(tmp_path, ["a.page", "b.crop", "c.crop"])
    assert s["status"] == {"ok": 1, "error": 1, "not_attempted": 1}
    assert s["attempt_errors"] == {"server": 1}
    assert s["by_kind"]["page"]["cost_usd"] == pytest.approx(0.0445, abs=1e-4)
    assert sc.request_cost(1_000_000, 0, 0) == pytest.approx(0.75)
    assert sc.request_cost(0, 500_000, 500_000) == pytest.approx(3.75)


def test_scoring_modules_never_load_an_api_client() -> None:
    import mabhas19.cli.score_app07  # noqa: F401, PLC0415

    assert "google.genai" not in sys.modules or "mabhas19.providers.vision" not in sys.modules
    assert "anthropic" not in sys.modules
