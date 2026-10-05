"""Runs every candidate method on every benchmark page and scores it against references."""

from __future__ import annotations

import json
import statistics
import time
import traceback
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

import pymupdf

from mabhas19.benchmark import extractors as ex
from mabhas19.benchmark.metrics import label_match, score_text
from mabhas19.benchmark.references import BenchmarkPage, PageReference, ReferenceItem
from mabhas19.ingest.repair.glyphs import (
    detect_kashida_glyphs,
    evidence_dict,
    kashida_keys,
)
from mabhas19.ingest.repair.pipeline import learn_final_forms
from mabhas19.providers.ocr import OcrEngine
from mabhas19.providers.vision import VisionExtractor
from mabhas19.tables.schemas import NumericField, SourceRef, to_dict, validate
from mabhas19.text.numbers import parse_numeric, same_value

RUN, NOT_RUN, NOT_APPLICABLE, ERROR = "RUN", "NOT_RUN", "NOT_APPLICABLE", "ERROR"
MAIN_TABLE_COLUMNS = ["5B,5C", "4A, 4B", "3A, 3B", "2A, 2B", "1B", "0B"]  # left to right
STRUCTURED_CATEGORIES = {"table", "table_city_climate", "table_energy_limits", "table_materials"}


@dataclass
class MethodResult:
    page_id: str
    method: str
    kind: str  # "text" | "structured"
    status: str
    reason: str = ""
    seconds: float = 0.0
    item_scores: dict[str, dict[str, Any]] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)
    numeric_errors: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class Context:
    sources_dir: Path
    out_dir: Path
    ocr: dict[str, OcrEngine]
    vision: dict[str, VisionExtractor]
    kashida: frozenset[tuple[str, int]] = frozenset()
    stats: Any = None
    evidence: list[dict[str, Any]] = field(default_factory=list)


def _source_path(sources_dir: Path, doc_id: str) -> Path:
    kind = "main" if doc_id.endswith("-main") else "appendices"
    return sources_dir / "mabhas19" / "ed5-1404" / kind / f"{doc_id}-v1.pdf"


def _timed(fn: Callable[[], Any]) -> tuple[Any, float]:
    start = time.perf_counter()
    value = fn()
    return value, time.perf_counter() - start


# ------------------------------------------------------------------------ text scoring


def _score_text_items(text: str, items: tuple[ReferenceItem, ...]) -> dict[str, dict[str, Any]]:
    return {it.id: asdict(score_text(it.expected_text, text)) for it in items}


def _text_summary(scores: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if not scores:
        return {}
    keys = ("cer", "wer", "cer_normalized", "wer_normalized")
    return {f"mean_{k}": round(statistics.mean(s[k] for s in scores.values()), 4) for k in keys}


# ------------------------------------------------------------------ structured scoring


def _numeric_check(result: MethodResult, item: str, fld: str, expected: str | None,
                   got: NumericField | None) -> bool:  # fmt: skip
    ok = got is not None and same_value(expected, got.normalized_value)
    s = result.summary
    s["numeric_total"] = s.get("numeric_total", 0) + 1
    if ok:
        s["numeric_exact"] = s.get("numeric_exact", 0) + 1
        return True
    entry = {"item": item, "field": fld, "expected": expected,
             "raw_detected_value": got.raw_detected_value if got else None,
             "normalized_value": got.normalized_value if got else None,
             "confidence": got.confidence if got else None,
             "flagged_for_verification": got.requires_verification if got else True}  # fmt: skip
    if got is not None and got.normalized_value is not None and not got.requires_verification:
        s["numeric_wrong_unflagged"] = s.get("numeric_wrong_unflagged", 0) + 1
    if got is None or got.normalized_value is None:
        s["numeric_missing"] = s.get("numeric_missing", 0) + 1
    result.numeric_errors.append(entry)
    return False


def _count(result: MethodResult, key: str, ok: bool) -> None:
    s = result.summary
    s[f"{key}_total"] = s.get(f"{key}_total", 0) + 1
    if ok:
        s[f"{key}_exact"] = s.get(f"{key}_exact", 0) + 1


def _finish_rates(result: MethodResult) -> None:
    s = result.summary
    for key in ("numeric", "label", "climate", "rows"):
        total = s.get(f"{key}_total", 0)
        if total:
            s[f"{key}_accuracy"] = round(s.get(f"{key}_exact", 0) / total, 4)


def _nearest(y: float | None, candidates: list[Any], y_of: Callable[[Any], float]) -> Any:
    if y is None or not candidates:
        return None
    best = min(candidates, key=lambda c: abs(y_of(c) - y))
    return best if abs(y_of(best) - y) <= 10 else None


def score_main_table(result: MethodResult, rows: list[tuple[float, list[ex.Cell]]],
                     ref: PageReference) -> None:  # fmt: skip
    for item in ref.items:
        if item.type != "table_row" or item.bbox is None:
            continue
        y = (item.bbox[1] + item.bbox[3]) / 2
        row = _nearest(y, rows, lambda r: r[0])
        cells = row[1] if row else []
        all_ok = len(cells) == len(MAIN_TABLE_COLUMNS)
        for col, cell in zip(MAIN_TABLE_COLUMNS, cells + [None] * 6, strict=False):
            got = None
            if cell is not None:
                v = parse_numeric(
                    cell.text, slash_is_decimal=True, source_confidence=cell.confidence
                )
                got = NumericField.from_value(v)
            all_ok &= _numeric_check(result, item.id, col, item.expected_fields["values"][col], got)
        _count(result, "rows", all_ok)


def score_city_climate(result: MethodResult, records: list[Any], ref: PageReference) -> None:
    by_number = {r.row_number: r for r in records if r.row_number is not None}
    for k, item in enumerate(i for i in ref.items if i.type == "table_row"):
        f = item.expected_fields
        rec = by_number.get(f["row_number"]) or (records[k] if k < len(records) else None)
        city_ok = rec is not None and label_match(f["city"], rec.city_raw)
        climate_ok = rec is not None and rec.climate_class == f["climate_class"]
        _count(result, "label", city_ok)
        _count(result, "climate", climate_ok)
        _count(result, "rows", city_ok and climate_ok)


def score_energy_limits(result: MethodResult, records: list[Any], ref: PageReference) -> None:
    cells = {(r.row_number, r.climate_class): r for r in records}
    for item in ref.items:
        if item.type != "table_row":
            continue
        f = item.expected_fields
        all_ok = True
        label_rec = None
        for climate, expected in f["values"].items():
            rec = cells.get((f["row_number"], climate))
            label_rec = label_rec or rec
            got = rec.max_eui_kwh_m2_year if rec else None
            all_ok &= _numeric_check(result, item.id, climate, expected, got)
        if f.get("label_check", True):
            ok = label_rec is not None and label_match(
                f["occupancy_type"], label_rec.occupancy_type_raw
            )
            _count(result, "label", ok)
            all_ok &= ok
        _count(result, "rows", all_ok)


def score_materials(result: MethodResult, records: list[Any], ref: PageReference) -> None:
    for item in ref.items:
        if item.type != "material_row" or item.bbox is None:
            continue
        f = item.expected_fields
        y = (item.bbox[1] + item.bbox[3]) / 2
        rec = _nearest(y, records, lambda r: r.source.row_y)
        dens = rec.density if rec else None
        ok = _numeric_check(result, item.id, "thermal_conductivity", f["conductivity"],
                            rec.thermal_conductivity if rec else None)  # fmt: skip
        for bound in ("min", "max"):
            expected = f[f"density_{bound}"]
            got = getattr(dens, bound) if dens else None
            if expected is None:
                # e.g. «بیش از ۲۴۰۰» has no upper bound: a predicted one is an error.
                ok &= got is None or got.normalized_value is None
                continue
            ok &= _numeric_check(result, item.id, f"density_{bound}", expected, got)
        _count(result, "rows", ok)


# ------------------------------------------------------------------------- the run


def _not_run(page: BenchmarkPage, method: str, kind: str, status: str, reason: str) -> MethodResult:
    return MethodResult(page.id, method, kind, status, reason)


def _vision_text(vision: VisionExtractor, image: Path) -> str:
    return vision.extract_text(image).text


def run_page(page: BenchmarkPage, ref: PageReference, ctx: Context) -> list[MethodResult]:
    results: list[MethodResult] = []
    page_dir = ctx.out_dir / page.id
    page_dir.mkdir(parents=True, exist_ok=True)
    pdf = pymupdf.open(_source_path(ctx.sources_dir, page.doc_id))
    pg = pdf[page.pdf_page - 1]
    is_main = page.doc_id.endswith("-main")

    def text_method(name: str, fn: Callable[[], str]) -> None:
        try:
            text, secs = _timed(fn)
        except Exception as exc:  # recorded as an ERROR result
            results.append(MethodResult(page.id, name, "text", ERROR, f"{exc!r}"))
            (page_dir / f"{name}.error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            return
        (page_dir / f"{name}.txt").write_text(text, encoding="utf-8")
        scores = _score_text_items(text, ref.items)
        results.append(MethodResult(page.id, name, "text", RUN, "", round(secs, 2), scores,
                                    _text_summary(scores)))  # fmt: skip

    text_method("text_layer_raw", lambda: ex.text_layer_raw(pg))
    if is_main:
        text_method("repair_v1", lambda: ex.repair_v1(pg, ctx.kashida, ctx.stats))
    else:
        results.append(_not_run(page, "repair_v1", "text", NOT_APPLICABLE,
                                "appendix text is vector outlines: nothing to repair"))  # fmt: skip
    for name, engine in ctx.ocr.items():
        ok, why = engine.availability()
        if ok:
            text_method(f"{name}_page", partial(ex.tesseract_page, pg, engine, page_dir))
        else:
            results.append(_not_run(page, f"{name}_page", "text", NOT_RUN, why))
    for name, vis in ctx.vision.items():
        ok, why = vis.availability()
        if ok:
            image = ex.render_clean(pg, page_dir / f"vision_{name}.png")
            text_method(f"vision_{name}_text", partial(_vision_text, vis, image))
        else:
            results.append(_not_run(page, f"vision_{name}_text", "text", NOT_RUN, why))

    if page.category in STRUCTURED_CATEGORIES:
        results += run_structured(page, ref, pg, ctx, page_dir)
    pdf.close()
    return results


def run_structured(page: BenchmarkPage, ref: PageReference, pg: pymupdf.Page, ctx: Context,
                   page_dir: Path) -> list[MethodResult]:  # fmt: skip
    out: list[MethodResult] = []
    tess = ctx.ocr.get("tesseract")
    tess_ok = tess is not None and tess.availability()[0]

    def record(name: str, fn: Callable[[MethodResult], list[Any]]) -> None:
        res = MethodResult(page.id, name, "structured", RUN)
        try:
            records, secs = _timed(lambda: fn(res))
        except Exception as exc:  # recorded as an ERROR result
            out.append(MethodResult(page.id, name, "structured", ERROR, f"{exc!r}"))
            (page_dir / f"{name}.error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            return
        res.seconds = round(secs, 2)
        problems = [p for r in records if not isinstance(r, tuple) for p in validate(r)]
        res.summary["schema_problems"] = len(problems)
        _finish_rates(res)
        dump = [to_dict(r) if not isinstance(r, tuple) else
                {"y": r[0], "cells": [asdict(c) for c in r[1]]} for r in records]  # fmt: skip
        (page_dir / f"{name}.json").write_text(json.dumps(dump, ensure_ascii=False, indent=1),
                                               encoding="utf-8")  # fmt: skip
        out.append(res)

    modes = [("grid_textlayer", "textlayer"), ("grid_ocr", "ocr")]
    if page.category == "table_materials":
        if tess_ok:

            def materials(res: MethodResult) -> list[Any]:
                recs = ex.app07_rows_ocr(pg, tess, page.doc_id)  # type: ignore[arg-type]
                score_materials(res, recs, ref)
                return recs

            record("app07_rows_ocr", materials)
        else:
            out.append(_not_run(page, "app07_rows_ocr", "structured", NOT_RUN, "no OCR engine"))
    else:
        for name, mode in modes:
            if mode == "ocr" and not tess_ok:
                out.append(_not_run(page, name, "structured", NOT_RUN, "no OCR engine"))
                continue

            def grid(res: MethodResult, mode: str = mode, name: str = name) -> list[Any]:
                rows = ex.grid_rows(pg, mode, tess if tess_ok else None, ctx.kashida)
                src = SourceRef(page.doc_id, page.pdf_page, name)
                if page.category == "table":
                    vals = ex.value_rows(rows)
                    score_main_table(res, vals, ref)
                    return vals
                if page.category == "table_city_climate":
                    recs = ex.city_climate_rows(rows, src)
                    score_city_climate(res, recs, ref)
                    return recs
                columns = list(reversed(ref.extra["columns"]))  # cells run left to right
                recs2 = ex.energy_limit_cells(rows, src, ref.extra["energy_rating"], columns)
                score_energy_limits(res, recs2, ref)
                return recs2

            record(name, grid)
    for vname, vis in ctx.vision.items():
        ok, why = vis.availability()
        status = NOT_RUN if not ok else NOT_APPLICABLE
        reason = why or "structured vision run not implemented"
        out.append(_not_run(page, f"vision_{vname}_json", "structured", status, reason))
    return out


def prepare_main_evidence(ctx: Context) -> None:
    """Learn glyph evidence for the main PDF once (kashida glyphs, final-form glyphs)."""
    pdf = pymupdf.open(_source_path(ctx.sources_dir, "m19-ed5-1404-main"))
    evidence = detect_kashida_glyphs(pdf, list(range(18, pdf.page_count)))
    ctx.kashida = kashida_keys(evidence)
    ctx.evidence = evidence_dict(evidence)
    ctx.stats = learn_final_forms(pdf, ctx.kashida)
    pdf.close()


def run_benchmark(pages: list[BenchmarkPage], refs: dict[str, PageReference],
                  ctx: Context) -> dict[str, Any]:  # fmt: skip
    prepare_main_evidence(ctx)
    results: list[MethodResult] = []
    for page in pages:
        print(f"  {page.id} ...", flush=True)
        results += run_page(page, refs[page.id], ctx)
    report = {
        "kashida_evidence": ctx.evidence,
        "engines": {n: e.availability() for n, e in ctx.ocr.items()},
        "vision": {n: v.availability() for n, v in ctx.vision.items()},
        "results": [asdict(r) for r in results],
    }
    (ctx.out_dir / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=1),
                                              encoding="utf-8")  # fmt: skip
    return report
