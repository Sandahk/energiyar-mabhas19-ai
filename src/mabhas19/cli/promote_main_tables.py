"""Promote the 11 main-document tables to curated/ (preview by default).

Usage:
    uv run python -m mabhas19.cli.promote_main_tables                  # preview under data/
    uv run python -m mabhas19.cli.promote_main_tables --write-curated  # write curated/

Reads data/processed/main_tables/candidate/ (run extract_main_tables first), validates the
whole set and refuses to write anything if a problem is found. No external API is called.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from mabhas19.core.paths import CURATED_DIR, DATA_DIR
from mabhas19.tables.main_extract import DOC_ID
from mabhas19.tables.main_tables_promote import (
    APP02_META,
    APP02_RECORDS,
    app02_with_backlink,
    issues_with_new,
    load_candidate,
    totals,
    validate_isfahan_backlink,
    validate_promotion,
    write_preview,
)

BASE = DATA_DIR / "processed" / "main_tables"


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write-curated", action="store_true")
    args = parser.parse_args(argv)
    built = load_candidate(BASE / "candidate")
    corpus = _jsonl(CURATED_DIR / "text" / DOC_ID / "blocks.jsonl")
    issues = issues_with_new((CURATED_DIR / "source_issues.yaml").read_text(encoding="utf-8"))
    app02_before = (CURATED_DIR / APP02_RECORDS).read_text(encoding="utf-8")
    app02 = app02_with_backlink(
        app02_before, (CURATED_DIR / APP02_META).read_text(encoding="utf-8")
    )
    problems = validate_promotion(built, corpus, issues)
    problems += validate_isfahan_backlink(built, app02_before, app02[0], issues)
    if problems:
        print("promotion blocked:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 2
    root = CURATED_DIR if args.write_curated else BASE / "promotion_preview" / "curated"
    for p in write_preview(built, issues, root, app02):
        print(f"wrote {p}")
    print(json.dumps(totals(built), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
