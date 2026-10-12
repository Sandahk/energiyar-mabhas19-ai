"""The minimal .env loader."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from mabhas19.core.env import load_dotenv, parse_dotenv


def test_parse_dotenv_handles_comments_quotes_and_export() -> None:
    text = "# comment\n\nA=1\nexport B = two \nC=\"quoted value\"\nD='x'\nEMPTY=\nnot a pair\n"
    assert parse_dotenv(text) == {"A": "1", "B": "two", "C": "quoted value", "D": "x"}


def test_dotenv_overrides_os_variable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "old-user-level-key")
    env = tmp_path / ".env"
    env.write_text("GEMINI_API_KEY=new-project-key\n", encoding="utf-8")
    assert load_dotenv(env) == ["GEMINI_API_KEY"]
    assert os.environ["GEMINI_API_KEY"] == "new-project-key"


def test_missing_dotenv_is_a_no_op(tmp_path: Path) -> None:
    assert load_dotenv(tmp_path / "absent.env") == []
