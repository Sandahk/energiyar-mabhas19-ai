"""Promote the reviewed appendix 3 table to curated/ (preview by default).

Usage:
    uv run python -m mabhas19.cli.promote_app03                  # preview under data/
    uv run python -m mabhas19.cli.promote_app03 --write-curated  # write curated/tables/

Reads data/processed/app03/<model>/ (run extract_app03 first), validates the table and
refuses to write anything if a problem is found. No external API is called.
"""

from __future__ import annotations

import argparse
import json
import sys

from mabhas19.core.paths import CURATED_DIR, DATA_DIR
from mabhas19.tables.app03_promote import DOC_ID, build_table, write_table


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="gemini-3.8-flash")
    parser.add_argument("--write-curated", action="store_true")
    args = parser.parse_args(argv)
    cand_dir = DATA_DIR / "processed" / "app03" / args.model
    table, problems = build_table(cand_dir, args.model)
    if problems:
        print("promotion blocked:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 2
    base = CURATED_DIR if args.write_curated else cand_dir / "promotion_preview" / "curated"
    paths = write_table(table, base / "tables" / DOC_ID)
    print(json.dumps(table["meta"]["counts"], ensure_ascii=False, indent=1))
    for p in paths:
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
