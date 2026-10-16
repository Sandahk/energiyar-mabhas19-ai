"""Build the candidate text corpus of the main Mabhas 19 document (offline).

Usage:
    uv run python -m mabhas19.cli.extract_main [--no-ocr]

Needs the Phase 1 audit output (python -m mabhas19.cli.audit_sources). Text-layer pages are
repaired with repair_v1; the scanned page is read with Tesseract into separate OCR blocks.
Outputs: data/processed/main/m19-ed5-1404-main/. No external API, nothing written to curated/.
"""

from __future__ import annotations

import argparse
import json

from mabhas19.core.paths import DATA_DIR
from mabhas19.ingest.main_corpus import DOC_ID, build
from mabhas19.providers.ocr import TesseractCli


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-ocr", action="store_true")
    args = parser.parse_args(argv)
    engine = None if args.no_ocr else TesseractCli(tessdata_dir=DATA_DIR / "models" / "tessdata")
    summary = build(DATA_DIR / "processed" / "main" / DOC_ID, engine)
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
