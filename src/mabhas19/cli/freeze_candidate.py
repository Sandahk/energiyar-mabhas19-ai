"""Freeze or check the retrieval candidate manifest of a final benchmark.

Usage:
    uv run python -m mabhas19.cli.freeze_candidate --set final3          # check (exit 2 on drift)
    uv run python -m mabhas19.cli.freeze_candidate --set final3 --write  # write the manifest

Writing requires a clean src/ tree. A final run (embedding_experiment --set final3) checks the
manifest and refuses to start on any difference. final2_candidate.yaml is historical: it records
the candidate Final-2 was run against and is never rewritten.
"""

from __future__ import annotations

import argparse
import sys

from mabhas19.retrieval.candidate import MANIFESTS, check_manifest, write_manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--set", dest="set_name", choices=sorted(MANIFESTS), default="final3")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)
    path = MANIFESTS[args.set_name]
    if args.write:
        if args.set_name != "final3":
            print(f"{path.name} is historical and is never rewritten", file=sys.stderr)
            return 2
        doc = write_manifest(path, "Final-3")
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
