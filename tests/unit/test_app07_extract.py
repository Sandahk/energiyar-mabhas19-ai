"""Phase 2C appendix 7 post-processing: label rules, numeric safety, inheritance, flags."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from mabhas19.benchmark import phase2b as pb
from mabhas19.cli import extract_app07 as cli
from mabhas19.tables import app07 as a7

CAT2 = "۲. بتن و فرآورده‌های بتنی"


def row(*labels: str | None, dens: str | None = "۱۰۰ تا ۲۰۰", k: str | None = "۰٫۵۰"
        ) -> dict[str, Any]:  # fmt: skip
    """A model row; labels are category, subgroup, material, condition (in that order)."""
    cat, sub, mat, cond = (*labels, None, None, None, None)[:4]
    return {"category_raw": cat, "subgroup_raw": sub, "material_raw": mat,
            "condition_raw": cond, "density_raw": dens, "conductivity_raw": k}  # fmt: skip


def page(rows: list[dict[str, Any]], pdf_page: int = 4, expected: int | None = None
         ) -> a7.PageInput:  # fmt: skip
    return a7.PageInput("d", pdf_page, "x", f"app07-p{pdf_page:03d}.page", "m", "t", "test",
                        rows, expected_rows=expected)  # fmt: skip


def codes(rec: dict[str, Any]) -> set[str]:
    return {f["code"] for f in rec["suspicious"]}


# --------------------------------------------------------------------------- labels


@pytest.mark.parametrize(
    ("raw", "heading", "text", "footnotes", "ambiguous"),
    [
        ("- متراکم", False, "متراکم", [], False),
        ("بتن با خرده چوب:", True, "بتن با خرده چوب", [], False),
        ("- مسلح:۱", False, "مسلح", ["1"], False),  # digit right after the colon
        ("بتن هوادار اتوکلاو شده ۱:", True, "بتن هوادار اتوکلاو شده", ["1"], False),
        ("بتن هوادار اتوکلاو شده¹:", True, "بتن هوادار اتوکلاو شده", ["1"], False),
        ("- با خاکستر (حدود ۶۵۰) ۱", False, "با خاکستر (حدود ۶۵۰)", ["1"], False),
        ("۳. مواد آب بندی و گرماشکن ۲", True, "۳. مواد آب بندی و گرماشکن", ["2"], True),
        ("- عیار سیمان بیش از ۳۰۰ ۱:", False, "عیار سیمان بیش از ۳۰۰", ["1"], True),
    ],
)
def test_clean_label(raw: str, heading: bool, text: str, footnotes: list[str],
                     ambiguous: bool) -> None:  # fmt: skip
    lbl = a7.clean_label(raw, heading=heading)
    assert (lbl.text, lbl.footnotes, lbl.footnote_ambiguous) == (text, footnotes, ambiguous)
    assert lbl.raw == raw  # the raw text is always kept


@pytest.mark.parametrize(
    "raw", ["- نسبت: ۱ به ۶", "- نسبت: ۳", "پی وی سی با ۴۰ درصد روان ساز", "مواد ۲"]
)
def test_values_inside_labels_are_not_footnotes(raw: str) -> None:
    assert a7.clean_label(raw, heading=False).footnotes == []


def test_logical_lines_join_wraps_and_split_at_dashes_and_colons() -> None:
    raw = "بتن با سنگدانه بسیار سبک:\n- متشکل از پرلیت (از ۳ تا ۶\nمیلیمتر) اجرای درجا:"
    raw += "\n- نسبت: ۱ به ۳"
    assert a7.logical_lines(raw) == [
        "بتن با سنگدانه بسیار سبک:",
        "- متشکل از پرلیت (از ۳ تا ۶ میلیمتر) اجرای درجا:",
        "- نسبت: ۱ به ۳",
    ]


@pytest.mark.parametrize(
    ("field", "raw", "kind"),
    [
        ("category_raw", "۱. اندود", a7.CATEGORY),
        ("category_raw", "بتن سبک دانه:", a7.SUBGROUP),  # unnumbered heading in category field
        ("subgroup_raw", "- متراکم:", a7.HEADING),  # dashed heading given as a subgroup
        ("subgroup_raw", "الیافی", a7.ITEM),  # item label given as a subgroup
        ("material_raw", "- چگالی اسمی: ۸۰۰", a7.KV),
        ("material_raw", "- دو حجم پرلیت برای یک حجم گچ", a7.KV),
        ("material_raw", "موزاییک", a7.ITEM),
        ("condition_raw", "- عیار سیمان بیش از ۳۰۰ ۱:", a7.HEADING),
    ],
)
def test_lines_are_classified_by_printed_form(field: str, raw: str, kind: str) -> None:
    assert a7.classify(field, a7.clean_label(raw, heading=False)) == kind


def test_mixed_direction_detection() -> None:
    assert a7.mixed_direction("چگالی طبیعی ۵۰۰kg/m3 تا ۶۰۰")
    assert a7.mixed_direction("HCFC -\n:CFC -")
    assert not a7.mixed_direction("پلی اتیلن چگالی زیاد (HD)")


# --------------------------------------------------------------------------- numbers


@pytest.mark.parametrize(
    ("raw", "kind", "q", "lo", "hi"),
    [
        ("۱۸۰۰ تا ۲۰۰۰", "range", None, "1800", "2000"),
        ("بیش از ۲۴۰۰", "more_than", ">", "2400", None),
        ("کمتر از ۶۰۰", "less_than", "<", None, "600"),
        ("۱۱۳۴۰", "single", None, "11340", "11340"),
    ],
)
def test_density_forms(raw: str, kind: str, q: str | None, lo: str | None,
                       hi: str | None) -> None:  # fmt: skip
    d = a7.parse_density(raw)
    assert (d.kind, d.qualifier, d.min, d.max, d.error) == (kind, q, lo, hi, None)


@pytest.mark.parametrize(
    "raw",
    ["۲۰۰۰ تا ۱۸۰۰", "۱۸۰۰ تا", "حدود ۲۰۰۰", "۱۲۰۰،۵", "1۸00", "۰", "۱۸۰۰ - ۲۰۰۰"],
)
def test_malformed_density_is_rejected_not_repaired(raw: str) -> None:
    d = a7.parse_density(raw)
    assert (d.kind, d.min, d.max) == (None, None, None)
    assert d.error


@pytest.mark.parametrize(
    ("raw", "value"),
    [
        ("۰٫۵۰", "0.50"),
        ("۱٫۸۰", "1.80"),
        ("۲٫۰", "2.0"),
        ("۳۵", "35"),
        ("۰٫۰۵۴", "0.054"),
    ],
)
def test_conductivity_keeps_printed_precision(raw: str, value: str) -> None:
    assert a7.parse_conductivity(raw).value == value


@pytest.mark.parametrize("raw", ["۰/۴۶", "۰،۴۶", "۰٫", "۱٫۲٫۳", "۱ تا ۲", None])
def test_ambiguous_conductivity_is_rejected(raw: str | None) -> None:
    assert a7.parse_conductivity(raw).value is None


# --------------------------------------------------------------------------- records


def test_record_fields_and_status() -> None:
    [rec] = a7.normalize_page(page([row(CAT2, None, "- متراکم")], expected=1), a7.Context())
    for key in ("doc_id", "pdf_page", "printed_page", "source_row_order", "raw_category",
                "raw_subgroup", "raw_material", "raw_condition", "raw_density_text",
                "raw_conductivity_text", "parsed_density_min", "parsed_density_max",
                "parsed_density_qualifier", "parsed_conductivity", "units", "footnote_refs",
                "normalization_notes", "extraction_model", "extraction_timestamp",
                "verification_status"):  # fmt: skip
        assert key in rec
    assert rec["verification_status"] == "machine_extracted_pending_review"
    assert rec["raw_material"] == "- متراکم"
    assert rec["material"] == "متراکم"
    assert rec["units"] == {"density": "kg/m3", "thermal_conductivity": "W/m.K"}


def test_heading_copied_into_material_is_dropped_and_flagged() -> None:
    recs = a7.normalize_page(page([row("۱. اندود", None, "۱. اندود"), row("۱. اندود")]),
                             a7.Context())  # fmt: skip
    assert [r["material"] for r in recs] == [None, None]
    assert "duplicate_or_merged_label" in codes(recs[0])


def test_material_and_footnote_inherited_by_continuation_rows() -> None:
    recs = a7.normalize_page(page([
        row(CAT2, "بتن متداول:", "- مسلح:۱", "درصد میلگرد: بین ۱ تا ۲ درصد"),
        row(CAT2, "بتن متداول:", None, "درصد میلگرد: بیش از ۲ درصد"),
    ]), a7.Context())  # fmt: skip
    assert [r["material"] for r in recs] == ["مسلح", "مسلح"]
    assert [r["footnote_refs"] for r in recs] == [["1"], ["1"]]
    assert recs[1]["material_source"] == "inherited"
    assert recs[1]["condition"] == "درصد میلگرد: بیش از ۲ درصد"


def test_cross_page_continuation_and_condition_in_material_field() -> None:
    ctx = a7.Context()
    a7.normalize_page(page([row(None, None, "بتن بسیار سبک:\n- متشکل از پرلیت:\n- نسبت: ۱ به ۳")],
                           pdf_page=5), ctx)  # fmt: skip
    first, second, aac = a7.normalize_page(page([
        row(None, None, "- نسبت: ۱ به ۶"),
        row(None, None, "- لایه های بتن"),
        row(None, "بتن هوادار ۱:", "- چگالی اسمی: ۸۰۰"),
    ], pdf_page=6), ctx)  # fmt: skip
    assert (first["subgroup"], first["material"], first["condition"]) == (
        "بتن بسیار سبک",
        "متشکل از پرلیت",
        "نسبت: ۱ به ۶",
    )
    assert first["subgroup_source"] == "inherited_previous_page"
    assert (second["material"], second["condition"]) == ("لایه های بتن", None)
    assert (aac["subgroup"], aac["subgroup_footnote_refs"], aac["material"]) == (
        "بتن هوادار",
        ["1"],
        None,
    )  # new subgroup: material is not carried over
    assert aac["condition"] == "چگالی اسمی: ۸۰۰"


def test_subgroup_ends_at_an_undashed_material() -> None:
    recs = a7.normalize_page(page([
        row(None, "بتن با خرده چوب:", "- ساخته شده با سیمان"),
        row(None, None, "موزاییک"),
        row(None, None, None),
    ]), a7.Context(category=a7.Slot(CAT2)))  # fmt: skip
    assert [r["subgroup"] for r in recs] == ["بتن با خرده چوب", None, None]
    assert [r["material"] for r in recs] == ["ساخته شده با سیمان", "موزاییک", "موزاییک"]


def test_dashed_heading_is_material_and_item_below_is_condition() -> None:
    recs = a7.normalize_page(page([
        row("بتن سبک دانه:", "- با پوکه طبیعی ۱:", "- با ذرات ریز"),
        row("بتن سبک دانه:", "- با پوکه طبیعی ۱:", None),  # value row continuing the label
        row("بتن سبک دانه:", "- با پوکه طبیعی ۱:", "- بدون ذرات ریز"),
    ]), a7.Context(category=a7.Slot(CAT2)))  # fmt: skip
    assert [r["subgroup"] for r in recs] == ["بتن سبک دانه"] * 3
    assert [r["material"] for r in recs] == ["با پوکه طبیعی"] * 3
    assert [r["condition"] for r in recs] == ["با ذرات ریز", "با ذرات ریز", "بدون ذرات ریز"]
    assert recs[1]["condition_parts"][0]["source"] == "inherited"
    assert all(r["footnote_refs"] == ["1"] for r in recs)


def test_sub_headings_are_not_inherited_but_flagged() -> None:
    recs = a7.normalize_page(page([
        row("۱۱. عایق", "- اکسترود شده با حفره های پر از:", "- هوا:", "- ضخامت تا ۶۰"),
        row("۱۱. عایق", "- اکسترود شده با حفره های پر از:", None, "- ضخامت بیش از ۶۰"),
    ]), a7.Context())  # fmt: skip
    assert recs[0]["condition"] == "هوا؛ ضخامت تا ۶۰"
    assert recs[1]["condition"] == "ضخامت بیش از ۶۰"
    assert "heading_scope_ambiguous" in codes(recs[1])


def test_merged_label_flags_the_row_and_the_next() -> None:
    recs = a7.normalize_page(page([row("۱۵. گچ", None, "گچ با روکش\nگچ با الیاف"),
                                   row("۱۵. گچ", None, "گچ ضدآتش")]), a7.Context())  # fmt: skip
    assert "duplicate_or_merged_label" in codes(recs[0])
    assert "duplicate_or_merged_label" in codes(recs[1])
    assert recs[0]["raw_material"] == "گچ با روکش\nگچ با الیاف"


def test_numeric_flags_and_row_count() -> None:
    recs = a7.normalize_page(page([row("۱. اندود", dens=None), row("۱. اندود", k="۰/۴۶")],
                                  expected=3), a7.Context())  # fmt: skip
    assert {"missing_numeric", "unexpected_row_count"} <= codes(recs[0])
    assert "unparseable_numeric" in codes(recs[1])
    assert recs[1]["parsed_conductivity"] is None
    assert recs[1]["raw_conductivity_text"] == "۰/۴۶"
    assert all(r["row_y"] is None for r in recs)


def test_model_parsed_values_are_never_used() -> None:
    lying = {**row("۱. اندود", k="۱۳۰"), "conductivity_parsed": "1.30"}
    [rec] = a7.normalize_page(page([lying]), a7.Context())
    assert rec["parsed_conductivity"] == "130"
    assert any("model's own parse" in n for n in rec["normalization_notes"])


def test_category_number_jump_is_flagged() -> None:
    ctx = a7.Context()
    a7.normalize_page(page([row("۲. بتن")]), ctx)
    [rec] = a7.normalize_page(page([row("۴. پلیمر")]), ctx)
    assert "structure_inconsistent" in codes(rec)


# --------------------------------------------------------------------------- cost guard


def _cached(tmp: Path, pdf_page: int, usage: dict[str, int | None]) -> None:
    req = cli.page_request(pdf_page)
    data = {"request_id": req.id, "kind": req.kind, "page_id": req.page_id, "item_id": None,
            "model_requested": "m", "status": "ok", "usage": usage}  # fmt: skip
    (tmp / f"{req.id}.json").write_text(json.dumps(data), encoding="utf-8")


def test_estimate_uses_measured_cost_and_handles_missing_thinking(tmp_path: Path) -> None:
    _cached(tmp_path, 3, {"prompt_tokens": 1353, "output_tokens": 1948, "thinking_tokens": 9640})
    _cached(tmp_path, 13, {"prompt_tokens": 1353, "output_tokens": 2719, "thinking_tokens": None})
    caller = pb.Caller(cli._Offline("m"), tmp_path)
    plan = cli.estimate(caller, (3, 4, 5, 13))
    assert plan["cached"] == [3, 13]
    assert plan["new_calls"] == [4, 5]
    assert plan["estimate_worst_usd"] == pytest.approx(2 * 0.0445, abs=1e-3)
    assert cli.within_budget(plan, 1.0)
    assert not cli.within_budget(plan, 0.05)
    assert not cli.within_budget({**plan, "estimate_worst_usd": None}, 1.0)
    assert cli.within_budget({**plan, "new_calls": []}, 0.0)
