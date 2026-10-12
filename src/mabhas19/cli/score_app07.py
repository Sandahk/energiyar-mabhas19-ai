"""Score cached appendix 7 extractions offline against the human-verified references.

Usage:
    uv run python -m mabhas19.cli.score_app07 [--model gemini-3.8-flash] [--with-tesseract]

Reads only cached responses under data/interim/gemini-phase2b/<model>/responses/ and
never calls an external API. --with-tesseract also runs the local Tesseract comparison.
Writes data/interim/gemini-phase2b/<model>/app07-scores.json.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import pymupdf

from mabhas19.benchmark import app07_scoring as sc
from mabhas19.benchmark import phase2b as pb
from mabhas19.benchmark.references import load_references
from mabhas19.core.paths import DATA_DIR, EVAL_DIR, SOURCES_DIR
from mabhas19.providers.ocr import TesseractCli

APP07_PDF = SOURCES_DIR / "mabhas19" / "ed5-1404" / "appendices" / "m19-ed5-1404-app07-v1.pdf"


def score(model: str, with_tesseract: bool) -> dict[str, Any]:
    responses = DATA_DIR / "interim" / "gemini-phase2b" / model / "responses"
    refs = load_references(EVAL_DIR / "extraction" / "reference")
    found: dict[str, dict[str, pb.ExtractedRow]] = {
        m: {} for m in ("gemini_page", "gemini_crop", "tesseract_page", "tesseract_cell")
    }
    per_page: dict[str, Any] = {}
    all_rows = []
    labels = sc.LabelScore()
    request_ids: list[str] = []
    tess = TesseractCli(tessdata_dir=DATA_DIR / "models" / "tessdata") if with_tesseract else None
    with tempfile.TemporaryDirectory() as td:
        for page_id in sc.APP07_PAGE_IDS:
            rows = sc.verified_rows(refs[page_id])
            all_rows += rows
            request_ids += [f"{page_id}.page"] + [f"{r.id}.crop" for r in rows]
            gem = sc.collect_gemini(responses, page_id, rows)
            page_found = {"gemini_page": gem.page, "gemini_crop": gem.crop}
            if tess is not None:
                pno = int(page_id.rsplit("p", 1)[1])
                with pymupdf.open(APP07_PDF) as pdf:
                    page = pdf[pno - 1]
                    page_found["tesseract_page"] = pb.tesseract_page_rows(
                        page, tess, Path(td), rows
                    )
                    page_found["tesseract_cell"] = pb.tesseract_cell_rows(
                        page, tess, "m19-ed5-1404-app07", rows
                    )
            for m, f in page_found.items():
                found[m].update(f)
            page_labels = sc.score_labels(rows, gem.page_raw)
            labels.counts.update(page_labels.counts)
            labels.context.update(page_labels.context)
            labels.mismatches += page_labels.mismatches
            per_page[page_id] = {
                "row_detection": gem.detection,
                **{m: sc.score_numeric(m, rows, f).summary() for m, f in page_found.items()},
            }
    combined = {
        m: sc.score_numeric(m, all_rows, f) for m, f in found.items() if f or m.startswith("gemini")
    }
    return {
        "model": model,
        "verified_rows": len(all_rows),
        "per_page": per_page,
        "combined": {m: s.summary() for m, s in combined.items()},
        "numeric_errors": {m: s.errors for m, s in combined.items()},
        "labels_gemini_page": labels.summary(),
        "label_mismatches": labels.mismatches,
        "requests": sc.usage_summary(responses, request_ids),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="gemini-3.8-flash")
    parser.add_argument("--with-tesseract", action="store_true")
    args = parser.parse_args(argv)
    report = score(args.model, args.with_tesseract)
    out = DATA_DIR / "interim" / "gemini-phase2b" / args.model / "app07-scores.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    brief = {k: report[k] for k in ("verified_rows", "combined", "labels_gemini_page", "requests")}
    print(json.dumps(brief, ensure_ascii=False, indent=1))
    print(
        f"numeric errors per method: { {m: len(e) for m, e in report['numeric_errors'].items()} }"
    )
    print(f"full report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
