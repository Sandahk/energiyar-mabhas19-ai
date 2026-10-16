"""Promote the reviewed main-text corpus to curated/ (preview by default).

Usage:
    uv run python -m mabhas19.cli.promote_main                  # preview under data/
    uv run python -m mabhas19.cli.promote_main --write-curated  # write curated/

Reads data/processed/main/m19-ed5-1404-main/ (run extract_main first), validates it and
refuses to write anything if a problem is found. No external API is called.
"""

from __future__ import annotations

import argparse
import json
import sys

from mabhas19.core.paths import CURATED_DIR, DATA_DIR
from mabhas19.ingest.main_promote import DOC_ID, build_preview, write_files


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write-curated", action="store_true")
    args = parser.parse_args(argv)
    corpus_dir = DATA_DIR / "processed" / "main" / DOC_ID
    files, meta, problems = build_preview(corpus_dir)
    if problems:
        print("promotion blocked:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 2
    root = CURATED_DIR if args.write_curated else corpus_dir / "promotion_preview" / "curated"
    for p in write_files(files, meta, root):
        print(f"wrote {p}")
    print(json.dumps(meta["counts"], ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
