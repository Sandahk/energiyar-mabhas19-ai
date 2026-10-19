"""Freeze or check the retrieval candidate manifest for Final-2.

Usage:
    uv run python -m mabhas19.cli.freeze_candidate          # check (exit 2 on any drift)
    uv run python -m mabhas19.cli.freeze_candidate --write  # write final2_candidate.yaml

Writing requires a clean src/ tree. The Final-2 run (embedding_experiment --set final2) checks
the manifest and refuses to start on any difference.
"""

from __future__ import annotations

import argparse
import sys

from mabhas19.retrieval.candidate import MANIFEST_PATH, check_manifest, write_manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)
    if args.write:
        doc = write_manifest()
        print(f"wrote {MANIFEST_PATH} (base {doc['base_commit']})")
        return 0
    problems = check_manifest()
    if problems:
        print("candidate drift:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 2
    print(f"candidate matches {MANIFEST_PATH.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
