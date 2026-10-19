"""Write the machine-readable structured dataset schemas.

Usage:
    uv run python -m mabhas19.cli.build_structured_schemas

Writes eval/retrieval/structured_schemas.yaml (derived from curated/; a test keeps it equal to a
fresh build).
"""

from __future__ import annotations

import argparse

from mabhas19.core.paths import CURATED_DIR
from mabhas19.retrieval.schema import SCHEMA_PATH, write_schemas
from mabhas19.retrieval.structured import StructuredStore


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args(argv)
    store = StructuredStore(CURATED_DIR)
    write_schemas(store.schemas)
    print(f"wrote {SCHEMA_PATH}: {len(store.schemas)} datasets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
