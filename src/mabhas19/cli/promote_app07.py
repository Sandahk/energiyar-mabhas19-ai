"""Promote the reviewed appendix 7 table to curated/ (preview by default).

Usage:
    uv run python -m mabhas19.cli.promote_app07                  # preview under data/
    uv run python -m mabhas19.cli.promote_app07 --write-curated  # write curated/tables/

Reads data/processed/app07/<model>/candidates.jsonl (run extract_app07 first), validates the
table and refuses to write anything if a problem is found. No external API is called.
"""

from __future__ import annotations

import argparse
import json
import sys

from mabhas19.core.paths import CURATED_DIR, DATA_DIR
from mabhas19.tables.app07_promote import DOC_ID, build_table, validate, write_table


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="gemini-3.8-flash")
    parser.add_argument("--write-curated", action="store_true")
    args = parser.parse_args(argv)
    cand_dir = DATA_DIR / "processed" / "app07" / args.model
    records, meta = build_table(cand_dir, args.model)
    problems = validate(records)
    if problems:
        print("promotion blocked:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 2
    base = CURATED_DIR if args.write_curated else cand_dir / "promotion_preview" / "curated"
    paths = write_table(records, meta, base / "tables" / DOC_ID)
    print(json.dumps(meta["counts"], ensure_ascii=False, indent=1))
    for p in paths:
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
