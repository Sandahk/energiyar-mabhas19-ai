"""Build the Mabhas 19 scope map from curated/ and the reviewed external-domain taxonomy.

Usage:
    uv run python -m mabhas19.cli.build_scope_map

Writes eval/retrieval/scope_map.yaml (derived; a test keeps it equal to a fresh build).
"""

from __future__ import annotations

import argparse

from mabhas19.retrieval.scope import SCOPE_MAP_PATH, write_scope_map


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args(argv)
    data = write_scope_map()
    ids = data["identifiers"]
    print(
        f"wrote {SCOPE_MAP_PATH}: {len(ids['sections'])} sections, {len(ids['clauses'])} clauses, "
        f"{len(ids['tables'])} tables, {len(ids['appendices'])} appendices, "
        f"{len(data['supported']['defined_terms'])} defined terms, {len(data['handoffs'])} "
        f"hand-offs, {len(data['external_domains'])} external domains"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
