"""Minimal loader for the project's git-ignored ``.env`` file (no external dependency)."""

from __future__ import annotations

import os
from pathlib import Path

from mabhas19.core.paths import REPO_ROOT

DOTENV_PATH = REPO_ROOT / ".env"


def parse_dotenv(text: str) -> dict[str, str]:
    """``KEY=VALUE`` lines; blank lines and ``#`` comments ignored; optional quotes stripped."""
    values: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key and value:
            values[key] = value
    return values


def load_dotenv(path: Path = DOTENV_PATH) -> list[str]:
    """Load ``.env`` into the process environment and return the names that were set.

    Values in ``.env`` take precedence over variables inherited from the OS, so a project
    key in ``.env`` replaces e.g. an older user-level ``GEMINI_API_KEY``. Values are never
    printed or logged.
    """
    if not path.is_file():
        return []
    values = parse_dotenv(path.read_text(encoding="utf-8"))
    os.environ.update(values)
    return sorted(values)
