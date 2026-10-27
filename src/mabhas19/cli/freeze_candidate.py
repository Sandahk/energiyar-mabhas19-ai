"""Freeze or check the retrieval candidate manifest of a final benchmark.

Usage:
    uv run python -m mabhas19.cli.freeze_candidate --set final7          # check (exit 2 on drift)
    uv run python -m mabhas19.cli.freeze_candidate --set final7 --write  # write the manifest

Writing requires clean src/, eval/ and curated/ trees. A final run (embedding_experiment --set
final7) checks the manifest and refuses to start on any difference. final2_candidate.yaml to
final6_candidate.yaml are historical: they record the candidates those sets were run against and
are never rewritten.
"""

from __future__ import annotations

import argparse
import sys

from mabhas19.retrieval.candidate import (
    CURRENT,
    LABELS,
    MANIFESTS,
    check_manifest,
    write_manifest,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--set", dest="set_name", choices=sorted(MANIFESTS), default=CURRENT)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)
    path = MANIFESTS[args.set_name]
    if args.write:
        if args.set_name != CURRENT:
            print(f"{path.name} is historical and is never rewritten", file=sys.stderr)
            return 2
        doc = write_manifest(path, LABELS[args.set_name])
        print(f"wrote {path} (base {doc['base_commit']})")
        return 0
    problems = check_manifest(path)
    if problems:
        print("candidate drift:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 2
    print(f"candidate matches {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
