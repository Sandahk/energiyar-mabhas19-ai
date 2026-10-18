"""Seeded anchor sampler for authoring the final, untouched retrieval test set.

The sampler draws evidence *anchors* (sections, structured records, tables) uniformly at
random from curated/, with a recorded seed, so that query authors write questions about
evidence chosen independently of any retrieval run. It never reads retrieval code, indexes,
benchmark results or the development/diagnostic sets. See eval/retrieval/FINAL_TEST_PROCEDURE.md.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import yaml

# anchors per category (the procedure document explains each category)
QUOTAS = {
    "section": 30,
    "climate_city": 10,
    "eui": 10,
    "material": 10,
    "main_table": 20,
    "table_reference": 5,
}


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def sample_anchors(curated: Path, seed: int, exclude: set[str] | None = None) -> dict[str, Any]:
    """Random anchors per category; ``exclude`` holds evidence IDs already used by earlier sets
    (sections, records), so the final set does not reuse them."""
    rng = random.Random(seed)
    exclude = exclude or set()
    sections = [
        s
        for s in _jsonl(curated / "structure" / "m19-ed5-1404-main" / "sections.jsonl")
        if s["section_number"] not in exclude
    ]
    t = curated / "tables"
    pools: dict[str, list[dict[str, Any]]] = {
        "section": [
            {"section": s["section_number"], "title": s["title"], "pdf_page": s["first_pdf_page"]}
            for s in sections
        ],
        "climate_city": [
            {"record": r["id"], "city": r["city"]["text"]}
            for r in _jsonl(t / "m19-ed5-1404-app02" / "city_climate_classes.jsonl")
            if r["id"] not in exclude
        ],
        "eui": [
            {
                "record": r["id"],
                "occupancy": r["occupancy_label"],
                "climate_class": r["climate_class"],
                "rating": r["energy_rating"],
            }
            for r in _jsonl(t / "m19-ed5-1404-app03" / "max_eui.jsonl")
            if r["id"] not in exclude
        ],
        "material": [
            {"record": r["id"], "material": r["display_name"]}
            for r in _jsonl(t / "m19-ed5-1404-app07" / "material_thermal_properties.jsonl")
            if r["id"] not in exclude
        ],
        "main_table": [
            {"record": r["record_id"], "table": r["table"]}
            for d in sorted((t / "m19-ed5-1404-main").glob("table-*"))
            for r in _jsonl(d / "records.jsonl")
            if r["record_id"] not in exclude
        ],
        "table_reference": [
            {"table": yaml.safe_load((d / "table.yaml").read_text(encoding="utf-8"))["table_id"]}
            for d in sorted((t / "m19-ed5-1404-main").glob("table-*"))
        ],
    }
    anchors = {k: rng.sample(pool, min(QUOTAS[k], len(pool))) for k, pool in pools.items()}
    return {"seed": seed, "quotas": QUOTAS, "excluded": len(exclude), "anchors": anchors}
