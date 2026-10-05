"""Run the Phase 2A extraction benchmark.

Usage:
    uv run python -m mabhas19.cli.extraction_benchmark [--page PAGE_ID ...]
        [--allow-external-vision anthropic|gemini ...]

External vision APIs are only called for providers named with --allow-external-vision
(and only when their credentials and SDK are present). Outputs: eval/runs/extraction/<run>/.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from mabhas19.benchmark.references import load_benchmark_pages, load_references
from mabhas19.benchmark.runner import Context, run_benchmark
from mabhas19.core.paths import CONFIGS_DIR, DATA_DIR, EVAL_DIR, SOURCES_DIR
from mabhas19.providers.ocr import EasyOcrEngine, OcrEngine, TesseractCli
from mabhas19.providers.vision import AnthropicVision, GeminiVision, VisionExtractor


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--page", action="append", help="limit to these benchmark page ids")
    parser.add_argument("--allow-external-vision", action="append", default=[],
                        choices=["anthropic", "gemini"])  # fmt: skip
    args = parser.parse_args(argv)

    _, pages = load_benchmark_pages(CONFIGS_DIR / "extraction-benchmark.yaml")
    refs = load_references(EVAL_DIR / "extraction" / "reference")
    pages = [p for p in pages if not args.page or p.id in args.page]
    run_dir = EVAL_DIR / "runs" / "extraction" / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True)
    ocr: dict[str, OcrEngine] = {
        "tesseract": TesseractCli(tessdata_dir=DATA_DIR / "models" / "tessdata"),
        "easyocr": EasyOcrEngine(DATA_DIR / "models" / "easyocr"),
    }
    allowed = set(args.allow_external_vision)
    vision: dict[str, VisionExtractor] = {
        "anthropic": AnthropicVision(allow_external="anthropic" in allowed),
        "gemini": GeminiVision(allow_external="gemini" in allowed),
    }
    print(f"benchmark run -> {run_dir}")
    run_benchmark(pages, refs, Context(SOURCES_DIR, run_dir, ocr, vision))
    print(f"results: {Path(run_dir) / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
