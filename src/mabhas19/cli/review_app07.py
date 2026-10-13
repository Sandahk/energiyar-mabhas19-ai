"""Prepare human review material for the suspicious appendix 7 candidate rows (offline).

Usage:
    uv run python -m mabhas19.cli.review_app07 [--model gemini-3.8-flash]

Reads data/processed/app07/<model>/ and writes review_suspicious/ inside it. No external
API is called, nothing is verified and nothing is written to curated/.
"""

from __future__ import annotations

import argparse
import json

from mabhas19.core.paths import DATA_DIR
from mabhas19.tables.app07_review import write_review


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="gemini-3.8-flash")
    args = parser.parse_args(argv)
    summary = write_review(DATA_DIR / "processed" / "app07" / args.model)
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
