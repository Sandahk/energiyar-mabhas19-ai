"""Closed candidate sets for the normalizer, read from the curated store only.

A *family* is one printed table the normalizer may point to: its row field, the outputs it
prints, and whether a climate class, an energy rating or a verbatim number (area, density) can
constrain it. Every candidate is a printed option (:func:`options.printed_options`), so the
model can only ever choose a string the source prints.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from mabhas19.generation import labels as L
from mabhas19.generation import options as O
from mabhas19.retrieval.schema import APP02, APP03, APP07, MAIN

if TYPE_CHECKING:
    from mabhas19.retrieval.structured import StructuredStore

CATALOG_VERSION = "normalizer-catalog-1"
CITY_FAMILY = "city_climate"
RATINGS = ("A", "B", "C", "D")


@dataclass(frozen=True)
class Family:
    name: str
    dataset: str
    field: str  # the row field ("" for the city family)
    outputs: tuple[str, ...]
    climate: bool = False
    rating: bool = False
    numbers: tuple[str, ...] = ()  # constraint dimensions a verbatim number may fill
    description: str = ""


FAMILIES: tuple[Family, ...] = (
    Family(
        "energy_limit",
        APP03,
        "occupancy",
        ("max_eui",),
        True,
        True,
        ("area",),
        "maximum annual energy-use intensity of a building, by building use",
    ),
    Family(
        "opaque_r",
        f"{MAIN}19-5-1",
        "element",
        ("min_r_value",),
        True,
        description="minimum thermal resistance (R) of opaque envelope parts",
    ),
    Family(
        "window",
        f"{MAIN}19-5-2",
        "component",
        ("max_value",),
        True,
        description="maximum U-value or SHGC of transparent envelope parts",
    ),
    Family(
        "material",
        APP07,
        "material",
        ("thermal_conductivity", "density"),
        numbers=("density",),
        description="thermal conductivity (lambda) or density of a building material",
    ),
    Family(
        "hot_water",
        f"{MAIN}19-5-5",
        "place_of_use",
        ("max_temperature",),
        description="maximum hot-water temperature at a point of use",
    ),
    Family(
        "equipment_central",
        f"{MAIN}19-5-3",
        "equipment",
        ("threshold",),
        description="minimum efficiency of central heating/cooling equipment",
    ),
    Family(
        "equipment_unit",
        f"{MAIN}19-5-4",
        "equipment",
        ("threshold",),
        description="minimum efficiency of non-central (unit) heating/cooling equipment",
    ),
    Family(
        "daylight",
        f"{MAIN}19-5-6",
        "use",
        ("min_daylight_area",),
        description="minimum daylight-sufficiency area ratio, by use",
    ),
    Family(
        "light_uniformity",
        f"{MAIN}19-5-7",
        "space",
        ("min_uniformity",),
        description="minimum uniformity of artificial light, by space",
    ),
    Family(
        CITY_FAMILY,
        APP02,
        "",
        ("climate_class",),
        description="climate class of a city (the city is checked against the printed lists)",
    ),
)
BY_NAME = {f.name: f for f in FAMILIES}


def candidates(store: StructuredStore, family: Family) -> list[str]:
    return O.printed_options(store, family.dataset, family.field) if family.field else []


def climate_codes(store: StructuredStore, family: Family) -> list[str]:
    return O.printed_options(store, family.dataset, "climate_class") if family.climate else []


def build(store: StructuredStore) -> dict[str, Any]:
    """The whole catalog: per family its printed candidates, outputs and constraint slots."""
    out: dict[str, Any] = {"version": CATALOG_VERSION, "families": []}
    for f in FAMILIES:
        out["families"].append(
            {
                "family": f.name,
                "table": f.dataset,
                "description": f.description,
                "outputs": [{"name": o, "label": L.OUTPUT_LABELS.get(o, o)} for o in f.outputs],
                "candidates": candidates(store, f),
                "climate_codes": climate_codes(store, f),
                "rating": list(RATINGS) if f.rating else [],
                "number_dimensions": list(f.numbers),
                "takes_city": True,
            }
        )
    return out


def all_labels(catalog: dict[str, Any]) -> list[str]:
    return sorted({c for f in catalog["families"] for c in f["candidates"]})


def all_outputs(catalog: dict[str, Any]) -> list[str]:
    return sorted({o["name"] for f in catalog["families"] for o in f["outputs"]})


def all_codes(catalog: dict[str, Any]) -> list[str]:
    return sorted({c for f in catalog["families"] for c in f["climate_codes"]})


def sha256(doc: Any) -> str:
    text = json.dumps(doc, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
