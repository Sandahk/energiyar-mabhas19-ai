"""Candidate record schemas for the three critical regulatory datasets (Phase 2A).

Every record keeps the source label exactly as extracted (``*_raw``) next to a normalized
form, carries its source page, and stores numbers as ``NumericField`` (raw detected
value + normalized value + confidence). Nothing here is verified; ``verification_status``
starts as "unverified" and only curation (Phase 4) may change it.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from mabhas19.text.normalize import normalize_for_search
from mabhas19.text.numbers import NumericValue

CLIMATE_CLASS = re.compile(r"^[0-9][A-Z]$")
ENERGY_RATINGS = frozenset({"A", "B", "C", "D"})
VERIFICATION_STATES = frozenset({"unverified", "verified", "rejected"})
DENSITY_KINDS = frozenset({"range", "more_than", "less_than", "single"})


@dataclass(frozen=True)
class NumericField:
    raw_detected_value: str
    normalized_value: str | None
    confidence: str
    requires_verification: bool
    issues: list[str] = field(default_factory=list)

    @classmethod
    def from_value(cls, v: NumericValue) -> NumericField:
        return cls(v.raw_detected_value, v.normalized_value, v.confidence,
                   v.requires_verification, list(v.issues))  # fmt: skip


@dataclass(frozen=True)
class SourceRef:
    doc_id: str
    pdf_page: int
    method: str  # extraction method that produced the record
    row_y: float | None = None  # vertical centre of the row in PDF points


@dataclass(frozen=True)
class CityClimateRow:
    """Appendix 2: city -> climate class."""

    source: SourceRef
    row_number: int | None
    city_raw: str | None
    city_normalized: str | None
    climate_class_raw: str | None
    climate_class: str | None  # as printed, upper-cased; never remapped (e.g. 5C stays 5C)
    verification_status: str = "unverified"


@dataclass(frozen=True)
class EnergyLimitCell:
    """Appendix 3: maximum energy-use intensity for one use type x climate x rating."""

    source: SourceRef
    table_id: str | None
    row_number: int | None
    occupancy_type_raw: str | None
    occupancy_type_normalized: str | None
    energy_rating: str
    climate_class: str
    max_eui_kwh_m2_year: NumericField
    verification_status: str = "unverified"


@dataclass(frozen=True)
class DensitySpec:
    kind: str  # range | more_than | less_than | single
    raw: str
    min: NumericField | None
    max: NumericField | None


@dataclass(frozen=True)
class MaterialPropertyRow:
    """Appendix 7: one value row of the material thermal-property table.

    Proposed after inspecting pages 3, 6 and 13: rows are grouped under numbered
    categories (sometimes continued from the previous page), may carry a sub-group
    heading, a condition line (nominal density, rebar percentage, mix ratio), and are
    often continuation rows that inherit the material label of the row above.
    """

    source: SourceRef
    category_raw: str | None
    material_raw: str | None
    material_inherited: bool
    condition_raw: str | None
    density: DensitySpec | None
    thermal_conductivity: NumericField | None
    units: dict[str, str]
    footnote_refs: list[str] = field(default_factory=list)
    verification_status: str = "unverified"


def to_dict(record: Any) -> dict[str, Any]:
    return asdict(record)


def normalized_label(raw: str | None) -> str | None:
    return None if raw is None else normalize_for_search(raw)


def _check_numeric(name: str, f: NumericField | None, problems: list[str]) -> None:
    if f is None:
        return
    if f.normalized_value is None and not f.requires_verification:
        problems.append(f"{name}: unparsed value must require verification")
    if f.confidence not in {"high", "medium", "low"}:
        problems.append(f"{name}: unknown confidence {f.confidence!r}")


def validate(record: Any) -> list[str]:
    """Schema-level problems of a record (empty list = valid)."""
    problems: list[str] = []
    if getattr(record, "verification_status", "unverified") not in VERIFICATION_STATES:
        problems.append("verification_status invalid")
    src = getattr(record, "source", None)
    if not isinstance(src, SourceRef) or src.pdf_page < 1:
        problems.append("source missing or invalid")
    if isinstance(record, CityClimateRow):
        if record.climate_class is not None and not CLIMATE_CLASS.match(record.climate_class):
            problems.append(f"climate_class not a climate code: {record.climate_class!r}")
    elif isinstance(record, EnergyLimitCell):
        if record.energy_rating not in ENERGY_RATINGS:
            problems.append(f"energy_rating invalid: {record.energy_rating!r}")
        if not CLIMATE_CLASS.match(record.climate_class):
            problems.append(f"climate_class invalid: {record.climate_class!r}")
        _check_numeric("max_eui_kwh_m2_year", record.max_eui_kwh_m2_year, problems)
    elif isinstance(record, MaterialPropertyRow):
        if record.density is not None and record.density.kind not in DENSITY_KINDS:
            problems.append(f"density.kind invalid: {record.density.kind!r}")
        if record.density is not None:
            _check_numeric("density.min", record.density.min, problems)
            _check_numeric("density.max", record.density.max, problems)
        _check_numeric("thermal_conductivity", record.thermal_conductivity, problems)
        if set(record.units) != {"density", "thermal_conductivity"}:
            problems.append("units must name density and thermal_conductivity")
    else:
        problems.append(f"unknown record type {type(record).__name__}")
    return problems
