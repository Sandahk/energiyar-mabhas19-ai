"""Well-known repository paths."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

SOURCES_DIR = REPO_ROOT / "sources"
MANIFEST_PATH = SOURCES_DIR / "manifest.yaml"
CURATED_DIR = REPO_ROOT / "curated"
DATA_DIR = REPO_ROOT / "data"
EVAL_DIR = REPO_ROOT / "eval"
CONFIGS_DIR = REPO_ROOT / "configs"
