"""Candidate table record schemas (appendices 2, 3 and 7)."""

from __future__ import annotations

from mabhas19.benchmark.extractors import parse_density
from mabhas19.tables.schemas import (
    CityClimateRow,
    DensitySpec,
    EnergyLimitCell,
    MaterialPropertyRow,
    NumericField,
    SourceRef,
    to_dict,
    validate,
)
from mabhas19.text.numbers import parse_numeric

SRC = SourceRef("m19-ed5-1404-app02", 2, "test", 160.0)


def num(raw: str) -> NumericField:
    return NumericField.from_value(parse_numeric(raw))


def test_city_row_keeps_5c_as_printed() -> None:
    row = CityClimateRow(SRC, 5, "اردبیل", "اردبیل", "5C", "5C")
    assert validate(row) == []
    assert to_dict(row)["climate_class"] == "5C"


def test_city_row_rejects_non_climate_code() -> None:
    row = CityClimateRow(SRC, 5, "اردبیل", "اردبیل", "SC", "SC")
    assert any("climate" in p for p in validate(row))


def test_energy_limit_cell_schema() -> None:
    cell = EnergyLimitCell(SourceRef("m19-ed5-1404-app03", 2, "test"), "P3-1", 1,
                           "ویلایی", "ویلایی", "D", "0B", num("90"))  # fmt: skip
    assert validate(cell) == []
    bad = EnergyLimitCell(SourceRef("m19-ed5-1404-app03", 2, "test"), None, 1, None, None,
                          "E", "0B", num("90"))  # fmt: skip
    assert validate(bad) == ["energy_rating invalid: 'E'"]


def test_material_row_schema_and_units() -> None:
    row = MaterialPropertyRow(
        SourceRef("m19-ed5-1404-app07", 13, "test", 388.0), "۱۴. فلزت و آلیاژها", "سرب", False,
        None, DensitySpec("single", "۱۱۳۴۰", num("۱۱۳۴۰"), num("۱۱۳۴۰")), num("35"),
        {"density": "kg/m3", "thermal_conductivity": "W/m.K"},
    )  # fmt: skip
    assert validate(row) == []
    d = to_dict(row)
    assert d["thermal_conductivity"]["raw_detected_value"] == "35"
    assert d["density"]["min"]["normalized_value"] == "11340"


def test_unparsed_numbers_must_require_verification() -> None:
    field = NumericField("Vee", None, "low", requires_verification=False)
    row = MaterialPropertyRow(SourceRef("d", 1, "t"), None, None, True, None, None, field,
                              {"density": "kg/m3", "thermal_conductivity": "W/m.K"})  # fmt: skip
    assert any("must require verification" in p for p in validate(row))


def test_density_text_parsing() -> None:
    rng = parse_density("۱۸۰۰ تا ۲۰۰۰", "medium")
    assert rng is not None
    assert rng.kind == "range"
    assert (rng.min and rng.min.normalized_value, rng.max and rng.max.normalized_value) == (
        "1800",
        "2000",
    )
    more = parse_density("بیش از ۲۴۰۰", "medium")
    assert more is not None
    assert more.kind == "more_than"
    assert more.max is None
    less = parse_density("کمتر از ۲۱۰۰", "medium")
    assert less is not None
    assert less.kind == "less_than"
    assert less.min is None
    single = parse_density("۷۰", "medium")
    assert single is not None
    assert single.kind == "single"
