"""Machine-readable schemas of the structured datasets (validated by the query planner).

A schema states, for one curated dataset: its dimensions (type, domain, required, whether an
absent dimension may be *enumerated* - every printed value returned, each record carrying its
own value - or is a *qualifier*), its output fields (unit, numeric or category), the interval
semantics of its printed conditions, the supported lookup directions (direct, reverse,
whole-table) and the permitted computations (none). Structure is declared once here; domains,
units and printed intervals are read from ``curated/``. ``eval/retrieval/structured_schemas.yaml``
is the derived, versioned dump (a test keeps it equal to a fresh build).

Lookup-direction policy (audited per table, never assumed):

* **direct** - from the table's input dimensions to its printed output;
* **reverse** - from a printed output value back to the rows that print it; supported only
  where the source defines a meaningful inversion (a range table, a value grid, a class list);
  several rows may legitimately match (``one_to_many``);
* **whole_table** - an explicit table/appendix reference with no row named;
* everything else (e.g. an energy-use limit back to an occupancy) is
  ``unsupported_reverse_lookup``.

Table scope (I10, :mod:`mabhas19.retrieval.table_scope`): for every *regulatory* dimension
(climate class, rating, group, year) a table does not print, the schema declares whether it is
``universal`` (the dimension does not affect the table's values: a material's λ for any rating)
or ``fixed`` to the values the source establishes, with the curated blocks that establish them
(the prescriptive 19-5-x requirements are those for rating D). A stated value outside a fixed
scope is an ``invalid_constraint``; an undeclared dimension fails closed. Applicability is never
inferred from the data.

Printed output headers (``Output.header``) are printed definition phrases naming an output beyond
its query names, each attested by the curated blocks listed with it; they only attest the words
of a requested quantity (I3) and never name an output in a question.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from mabhas19.core.paths import EVAL_DIR

SCHEMA_PATH = EVAL_DIR / "retrieval" / "structured_schemas.yaml"
SCHEMA_VERSION = 1

APP02 = "m19-ed5-1404-app02.city_climate_classes"
APP03 = "m19-ed5-1404-app03.max_energy_use_intensity"
APP07 = "m19-ed5-1404-app07.material_thermal_properties"
MAIN = "m19-ed5-1404-main.t"


@dataclass
class Dimension:
    name: str
    kind: str  # category | interval | label | entity | identity
    path: str  # record field (dotted) the dimension reads
    required: bool = False
    # may be enumerated - every printed member returned, each self-describing - but only when
    # the question explicitly asks for the alternatives (see retrieval.plan.Enumeration)
    enumerable: bool = False
    qualifier: bool = False  # distinguishes rows that share an entity (with/without, single/multi)
    cues: tuple[str, ...] = ()  # words that attach a stated number to this dimension
    units: tuple[str, ...] = ()  # unit keys (retrieval.numeric.UNITS) of stated numbers
    weak_cues: tuple[str, ...] = ()  # accepted only with an explicit table reference
    domain: list[str] = field(default_factory=list)  # printed values (category/identity)
    nouns: tuple[str, ...] = ()  # words naming the dimension in an enumeration request
    interval_semantics: str | None = None  # how printed bounds are read
    # how a stated value is compared with the printed one: ``exact`` (Decimal equality),
    # ``interval`` (containment in printed bounds) or ``categorical`` (equality of labels)
    value_semantics: str = ""

    def __post_init__(self) -> None:
        if not self.value_semantics:
            self.value_semantics = "interval" if self.kind == "interval" else "categorical"


@dataclass
class Output:
    name: str
    path: str
    kind: str  # number | category | symbol | interval | text
    unit: str | None = None
    names: tuple[str, ...] = ()  # words naming this output in a question
    cues: tuple[str, ...] = ()  # words that attach a stated number to this output (reverse)
    units: tuple[str, ...] = ()
    weak_cues: tuple[str, ...] = ()
    # outputs a row may print *instead of* this one for the same requirement (a 19-5-3 row
    # prints a minimum efficiency or a required energy rating); certification accepts only these,
    # and the returned field names the one printed - never another output silently
    alternatives: tuple[str, ...] = ()
    # printed definition phrases of this output (attestation of a requested quantity's words,
    # never a query name) and the curated blocks printing them
    header: tuple[str, ...] = ()
    header_blocks: tuple[str, ...] = ()
    # ``exact`` printed scalars (λ, EUI, R, U…) are matched by Decimal equality only - a reverse
    # lookup never returns a nearby value; ``interval`` outputs (printed ranges) by containment;
    # ``categorical`` (classes, ratings, symbols) by equality
    value_semantics: str = ""

    def __post_init__(self) -> None:
        if not self.value_semantics:
            self.value_semantics = {"number": "exact", "interval": "interval"}.get(
                self.kind, "categorical"
            )


@dataclass
class ScopeDecl:
    """The scope of a regulatory dimension the table does not print."""

    dimension: str
    kind: str  # universal | fixed
    values: tuple[str, ...] = ()  # fixed: the applicable values
    provenance: tuple[str, ...] = ()  # curated blocks establishing a fixed scope
    basis: str = ""


@dataclass
class Direction:
    operation: str  # direct | reverse | whole_table
    inputs: tuple[str, ...]
    output: str
    supported: bool = True
    one_to_many: bool = False
    note: str = ""


@dataclass
class TableSchema:
    dataset: str
    family: str  # place | eui | material | label | criterion | rating_range
    title: str
    dimensions: list[Dimension]
    outputs: list[Output]
    directions: list[Direction]
    computations: list[str] = field(default_factory=list)  # permitted: none
    reverse_policy: str = ""
    records: int = 0
    scope: list[ScopeDecl] = field(default_factory=list)  # regulatory dimensions not printed

    def dim(self, name: str) -> Dimension | None:
        return next((d for d in self.dimensions if d.name == name), None)

    def scope_of(self, dimension: str) -> ScopeDecl | None:
        return next((x for x in self.scope if x.dimension == dimension), None)

    def output(self, name: str) -> Output | None:
        return next((o for o in self.outputs if o.name == name), None)

    def direction(self, operation: str, output: str | None = None) -> Direction | None:
        return next(
            (
                d
                for d in self.directions
                if d.operation == operation and (output is None or d.output == output)
            ),
            None,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_CLIMATE = Dimension(
    "climate_class",
    "category",
    "climate_group.applies_to",
    required=True,
    enumerable=True,
    nouns=("اقلیم",),
)
_ROW_BANDS = Dimension(
    "row_condition",
    "label",
    "row",
    enumerable=True,
    nouns=("pf", "سایه گیر", "توان", "حالت"),
)
_R_CUES = ("مقاومت حرارتی", "r value", "r-value", "rvalue", "مقدار r", "حداقل r", "r")
_LAMBDA_CUES = ("ضریب هدایت حرارتی", "ضریب هدایت", "هدایت حرارتی", "λ", "لامبدا", "لاندا")
_DENSITY_CUES = ("چگالی", "جرم مخصوص", "جرم حجمی", "وزن مخصوص")
_AREA_CUES = ("مساحت", "زیربنا", "زیر بنا", "متراژ", "متر مربع", "مترمربع")


def _main(table: str) -> str:
    return MAIN + table


def _universal(*dims: str, basis: str) -> list[ScopeDecl]:
    return [ScopeDecl(d, "universal", basis=basis) for d in dims]


# the prescriptive requirements of 19-5 are set for energy rating D (19-4-1-1 الف)
_PRESCRIPTIVE_D = ("main-p077-b007", "main-p077-b008")
_NO_SCHEDULE = (
    "no group or year column: table 19-2-1 schedules when a requirement applies, not its value"
)
_ALL_CLIMATES = "the requirement is printed without a climate column (every climate)"
_CLASSIFICATION = "a climate classification depends on no rating, group or year"


def _rating_d(*blocks: str) -> ScopeDecl:
    return ScopeDecl(
        "rating",
        "fixed",
        ("D",),
        (*blocks, *_PRESCRIPTIVE_D),
        "the printed values are the requirements for energy rating D",
    )


# regulatory dimensions a table does not print, per dataset (I10)
SCOPES: dict[str, list[ScopeDecl]] = {
    APP02: _universal("rating", "group", "year", basis=_CLASSIFICATION),
    _main("19-3-2"): _universal("rating", "group", "year", basis=_CLASSIFICATION),
    APP03: _universal(
        "group", "year", basis="limits printed per occupancy, climate class and rating only"
    ),
    APP07: _universal(
        "climate_class",
        "rating",
        "group",
        "year",
        basis="printed physical properties of materials (no rating, climate or schedule)",
    ),
    _main("19-2-1"): [
        *_universal("climate_class", basis="the schedule applies in every part of the country"),
        ScopeDecl(
            "rating",
            "fixed",
            ("D",),
            ("main-p063-b007",),
            "the schedule is for meeting the requirements of energy rating D (19-2-5)",
        ),
    ],
    _main("19-3-1"): _universal("rating", "group", "year", basis=_CLASSIFICATION),
    _main("19-3-3"): _universal(
        "climate_class", "group", "year", basis="ratings by the R ratio, the same in every climate"
    ),
    _main("19-5-1"): [
        _rating_d("main-p083-b010"),
        *_universal("group", "year", basis=_NO_SCHEDULE),
    ],
    _main("19-5-2"): [
        _rating_d("main-p091-b004"),
        *_universal("group", "year", basis=_NO_SCHEDULE),
    ],
    _main("19-5-3"): [
        *_universal("climate_class", basis=_ALL_CLIMATES),
        *_universal("group", "year", basis=_NO_SCHEDULE),
    ],
    _main("19-5-4"): [
        *_universal("climate_class", basis=_ALL_CLIMATES),
        *_universal("group", "year", basis=_NO_SCHEDULE),
    ],
    _main("19-5-5"): [
        *_universal("climate_class", basis=_ALL_CLIMATES),
        _rating_d("main-p013-b005"),
        *_universal("group", "year", basis=_NO_SCHEDULE),
    ],
    _main("19-5-6"): [
        *_universal("climate_class", basis=_ALL_CLIMATES),
        _rating_d(),
        *_universal("group", "year", basis=_NO_SCHEDULE),
    ],
    _main("19-5-7"): [
        *_universal("climate_class", basis=_ALL_CLIMATES),
        _rating_d("main-p013-b005"),
        *_universal("group", "year", basis=_NO_SCHEDULE),
    ],
}


def _specs() -> dict[str, dict[str, Any]]:
    """Declared structure per dataset (domains are filled from curated data)."""
    return {
        APP02: {
            "family": "place",
            "dimensions": [
                Dimension("place", "entity", "city.text", required=True),
                Dimension("climate_class", "category", "climate_class.code"),
            ],
            "outputs": [
                Output(
                    "climate_class", "climate_class.code", "category", names=("اقلیم", "رده اقلیمی")
                ),
            ],
            "directions": [
                Direction("direct", ("place",), "climate_class"),
                Direction(
                    "reverse",
                    ("climate_class",),
                    "place",
                    one_to_many=True,
                    note="a printed class lists every city printed with it",
                ),
                Direction("whole_table", (), "climate_class"),
            ],
            "reverse_policy": "class -> cities supported (one-to-many); no geographic inference",
        },
        _main("19-3-2"): {
            "family": "place",
            "dimensions": [
                Dimension("place", "entity", "city.text", required=True),
                Dimension("province", "entity", "province.text", qualifier=True),
                Dimension("wmo_code", "identity", "wmo_code.ascii"),
                Dimension("climate_class", "category", "climate_class.code"),
            ],
            "outputs": [
                Output(
                    "climate_class", "climate_class.code", "category", names=("اقلیم", "رده اقلیمی")
                ),
                Output(
                    "wmo_code",
                    "wmo_code.ascii",
                    "category",
                    names=("کد ایستگاه", "wmo", "شماره ایستگاه"),
                ),
            ],
            "directions": [
                Direction("direct", ("place",), "climate_class"),
                Direction("direct", ("wmo_code",), "climate_class"),
                Direction("reverse", ("climate_class",), "place", one_to_many=True),
                Direction("whole_table", (), "climate_class"),
            ],
            "reverse_policy": "class -> province capitals supported (one-to-many)",
        },
        APP03: {
            "family": "eui",
            "dimensions": [
                Dimension(
                    "occupancy",
                    "entity",
                    "occupancy_id",
                    required=True,
                    enumerable=True,
                    nouns=("کاربری",),
                ),
                Dimension(
                    "climate_class",
                    "category",
                    "climate_class",
                    required=True,
                    enumerable=True,
                    nouns=("اقلیم",),
                ),
                Dimension(
                    "rating",
                    "category",
                    "energy_rating",
                    required=True,
                    enumerable=True,
                    nouns=("رده", "رتبه"),
                ),
                Dimension(
                    "area",
                    "interval",
                    "occupancy_label",
                    enumerable=True,
                    nouns=("اندازه", "مساحت", "متراژ", "زیربنا"),
                    qualifier=True,
                    cues=_AREA_CUES,
                    units=("m2",),
                    interval_semantics="printed operator in the label (≥, <)",
                ),
            ],
            "outputs": [
                Output(
                    "max_eui",
                    "max_eui.value",
                    "number",
                    "kWh/m2.yr",
                    names=("شدت مصرف", "eui", "سقف مصرف", "حداکثر مصرف"),
                    cues=("شدت مصرف", "eui"),
                    units=("kwh",),
                    header=(
                        "حداکثر میزان مجاز شدت مصرف انرژی",
                        "انرژی مصرفی سالانه به‌ازای هر متر مربع فضای کنترل‌شده",
                    ),
                    header_blocks=("main-p072-b009", "main-p070-b003"),
                ),
            ],
            "directions": [
                Direction("direct", ("occupancy", "climate_class", "rating"), "max_eui"),
                Direction(
                    "reverse",
                    ("max_eui",),
                    "occupancy",
                    supported=False,
                    note="an energy-use limit does not identify an occupancy, class and "
                    "rating; the source defines no inversion",
                ),
                Direction("whole_table", (), "max_eui"),
            ],
            "reverse_policy": "prohibited (EUI value -> occupancy/class/rating is not defined)",
        },
        APP07: {
            "family": "material",
            "dimensions": [
                Dimension(
                    "material",
                    "entity",
                    "display_name",
                    required=True,
                    enumerable=True,
                    nouns=("نوع", "حالت"),
                ),
                Dimension(
                    "density",
                    "interval",
                    "density",
                    qualifier=True,
                    enumerable=True,
                    nouns=("چگالی", "جرم مخصوص"),
                    cues=_DENSITY_CUES,
                    units=("kg/m3",),
                    interval_semantics="«a تا b» ends unspecified; «بیش از»/«کمتر از» open",
                ),
                Dimension(
                    "condition",
                    "label",
                    "condition.text",
                    qualifier=True,
                    interval_semantics="numbers inside printed conditions compared as "
                    "intervals; ratios compared as pairs",
                ),
            ],
            "outputs": [
                Output(
                    "thermal_conductivity",
                    "thermal_conductivity.value",
                    "number",
                    "W/m.K",
                    names=_LAMBDA_CUES,
                    cues=_LAMBDA_CUES,
                ),
                Output("density", "density", "interval", "kg/m3", names=_DENSITY_CUES),
            ],
            "directions": [
                Direction("direct", ("material", "density", "condition"), "thermal_conductivity"),
                Direction(
                    "reverse",
                    ("thermal_conductivity", "density"),
                    "material",
                    one_to_many=True,
                    note="a printed λ (and density) -> the rows printing it",
                ),
                Direction("whole_table", (), "thermal_conductivity"),
            ],
            "reverse_policy": "λ/density -> material rows supported (one-to-many)",
        },
        _main("19-2-1"): {
            "family": "label",
            "dimensions": [
                Dimension("requirement", "label", "requirement", required=True),
                Dimension(
                    "group",
                    "category",
                    "building_group.text",
                    required=True,
                    enumerable=True,
                    nouns=("گروه",),
                ),
                Dimension(
                    "year", "category", "year.index", required=True, enumerable=True, nouns=("سال",)
                ),
            ],
            "outputs": [
                Output(
                    "printed_symbol",
                    "printed_symbol.symbol",
                    "symbol",
                    names=("علامت", "تیک", "ضربدر"),
                ),
            ],
            "directions": [
                Direction("direct", ("requirement", "group", "year"), "printed_symbol"),
                Direction(
                    "reverse",
                    ("printed_symbol",),
                    "requirement",
                    supported=False,
                    note="symbols are not defined by the source; no inversion",
                ),
                Direction("whole_table", (), "printed_symbol"),
            ],
            "reverse_policy": "prohibited",
        },
        _main("19-3-1"): {
            "family": "criterion",
            "dimensions": [
                Dimension(
                    "climate_class",
                    "category",
                    "climate_classes.applies_to",
                    enumerable=True,
                    nouns=("اقلیم",),
                ),
                Dimension(
                    "degree_days",
                    "interval",
                    "criterion.conditions",
                    cues=("hdd", "cdd", "روز درجه", "درجه روز"),
                    interval_semantics="printed operators (<, ≤)",
                ),
            ],
            "outputs": [
                Output(
                    "criterion",
                    "criterion.printed",
                    "interval",
                    names=("روز درجه", "hdd", "cdd", "معیار"),
                ),
                Output(
                    "climate_class",
                    "climate_classes.applies_to",
                    "category",
                    names=("اقلیم", "رده"),
                ),
            ],
            "directions": [
                Direction("direct", ("climate_class",), "criterion"),
                Direction(
                    "reverse",
                    ("degree_days",),
                    "climate_class",
                    one_to_many=True,
                    note="a degree-day value -> the class whose printed condition holds",
                ),
                Direction("whole_table", (), "criterion"),
            ],
            "reverse_policy": "degree days -> class supported",
        },
        _main("19-3-3"): {
            "family": "rating_range",
            "dimensions": [
                Dimension(
                    "rating", "category", "energy_rating.code", enumerable=True, nouns=("رده",)
                ),
                Dimension(
                    "r_ratio",
                    "interval",
                    "r_range",
                    cues=("نسبت r", "نسبت مصرف", "r ratio", "نسبت مصرف واقعی"),
                    weak_cues=("r",),
                    interval_semantics="printed operators (<, ≤)",
                ),
            ],
            "outputs": [
                Output(
                    "r_range", "r_range.printed", "interval", names=("بازه", "محدوده", "نسبت r")
                ),
                Output("rating", "energy_rating.code", "category", names=("رده", "رتبه")),
            ],
            "directions": [
                Direction("direct", ("rating",), "r_range"),
                Direction(
                    "reverse",
                    ("r_ratio",),
                    "rating",
                    note="an R ratio -> the rating whose printed range contains it",
                ),
                Direction("whole_table", (), "r_range"),
            ],
            "reverse_policy": "R ratio -> rating supported (exactly one printed range or none)",
        },
        _main("19-5-1"): {
            "family": "label",
            "dimensions": [
                Dimension("element", "label", "element", required=True),
                _CLIMATE,
            ],
            "outputs": [
                Output(
                    "min_r_value",
                    "min_r_value.value",
                    "number",
                    "m2.K/W",
                    names=("مقاومت حرارتی", "r value", "r-value", "r"),
                    cues=_R_CUES,
                    units=("m2k/w",),
                ),
            ],
            "directions": [
                Direction("direct", ("element", "climate_class"), "min_r_value"),
                Direction(
                    "reverse",
                    ("min_r_value",),
                    "element",
                    one_to_many=True,
                    note="a printed R -> the element/climate cells printing it",
                ),
                Direction("whole_table", (), "min_r_value"),
            ],
            "reverse_policy": "value -> cells supported (one-to-many)",
        },
        _main("19-5-2"): {
            "family": "label",
            "dimensions": [
                Dimension("component", "label", "section", required=True),
                _CLIMATE,
                _ROW_BANDS,
            ],
            "outputs": [
                Output(
                    "max_value",
                    "max_value.value",
                    "number",
                    names=("u-value", "u value", "ضریب انتقال حرارت", "shgc", "ضریب بهره خورشیدی"),
                    cues=(
                        "u-value",
                        "u value",
                        "ضریب انتقال حرارت",
                        "ضریب انتقال حرارتی",
                        "shgc",
                        "ضریب بهره خورشیدی",
                    ),
                ),
            ],
            "directions": [
                Direction("direct", ("component", "climate_class"), "max_value"),
                Direction("reverse", ("max_value",), "component", one_to_many=True),
                Direction("whole_table", (), "max_value"),
            ],
            "reverse_policy": "value -> cells supported (one-to-many)",
        },
        _main("19-5-3"): {
            "family": "label",
            "dimensions": [
                Dimension("equipment", "label", "equipment", required=True),
                Dimension("rating", "category", "energy_rating.printed"),
                _ROW_BANDS,
            ],
            "outputs": [
                Output(
                    "threshold",
                    "threshold.value",
                    "number",
                    names=("بازده", "بازدهی", "راندمان", "cop", "ضریب عملکرد", "کلاس", "ie"),
                    cues=("بازده", "بازدهی", "راندمان", "cop", "ضریب عملکرد"),
                    alternatives=("energy_rating",),
                ),
                Output(
                    "energy_rating",
                    "energy_rating.printed",
                    "category",
                    names=("رده انرژی", "رده", "برچسب"),
                ),
                Output("standard", "standard.displayed", "text", names=("استاندارد",)),
            ],
            "directions": [
                Direction("direct", ("equipment",), "threshold"),
                Direction("reverse", ("energy_rating",), "equipment", one_to_many=True),
                Direction("whole_table", (), "threshold"),
            ],
            "reverse_policy": "rating -> equipment supported (one-to-many); "
            "threshold -> equipment prohibited",
        },
        _main("19-5-4"): {
            "family": "label",
            "dimensions": [
                Dimension("equipment", "label", "equipment", required=True),
                Dimension("rating", "category", "energy_rating.printed"),
                _ROW_BANDS,
            ],
            "outputs": [
                Output(
                    "threshold",
                    "threshold.value",
                    "number",
                    names=("بازده", "بازدهی", "راندمان", "eei", "seer"),
                    cues=("بازده", "بازدهی", "راندمان", "eei", "seer"),
                    alternatives=("energy_rating",),
                ),
                Output(
                    "energy_rating",
                    "energy_rating.printed",
                    "category",
                    names=("رده انرژی", "رده", "برچسب"),
                ),
                Output("standard", "standard.displayed", "text", names=("استاندارد",)),
            ],
            "directions": [
                Direction("direct", ("equipment",), "threshold"),
                Direction("reverse", ("energy_rating",), "equipment", one_to_many=True),
                Direction("whole_table", (), "threshold"),
            ],
            "reverse_policy": "rating -> equipment supported (one-to-many); "
            "threshold -> equipment prohibited",
        },
        _main("19-5-5"): {
            "family": "label",
            "dimensions": [Dimension("place_of_use", "label", "place_of_use", required=True)],
            "outputs": [
                Output(
                    "max_temperature",
                    "max_temperature.value",
                    "number",
                    "°C",
                    names=("دما", "دمای", "درجه"),
                    cues=("دما", "دمای"),
                    units=("celsius",),
                ),
            ],
            "directions": [
                Direction("direct", ("place_of_use",), "max_temperature"),
                Direction(
                    "reverse",
                    ("max_temperature",),
                    "place_of_use",
                    supported=False,
                    note="a temperature does not identify a place of use",
                ),
                Direction("whole_table", (), "max_temperature"),
            ],
            "reverse_policy": "prohibited",
        },
        _main("19-5-6"): {
            "family": "label",
            "dimensions": [
                Dimension("use", "label", "use", required=True),
                Dimension(
                    "area",
                    "interval",
                    "use.area_condition",
                    qualifier=True,
                    cues=_AREA_CUES,
                    units=("m2",),
                    interval_semantics="printed «کوچک‌تر/بزرگ‌تر از» open",
                ),
            ],
            "outputs": [
                Output(
                    "min_daylight_area",
                    "min_daylight_area.value",
                    "number",
                    "%",
                    names=("نور روز", "کفایت نور", "درصد"),
                    cues=("کفایت نور روز",),
                ),
            ],
            "directions": [
                Direction("direct", ("use", "area"), "min_daylight_area"),
                Direction("reverse", ("min_daylight_area",), "use", supported=False),
                Direction("whole_table", (), "min_daylight_area"),
            ],
            "reverse_policy": "prohibited",
        },
        _main("19-5-7"): {
            "family": "label",
            "dimensions": [Dimension("space", "label", "space", required=True)],
            "outputs": [
                Output(
                    "min_uniformity",
                    "min_uniformity.value",
                    "number",
                    names=("یکنواختی",),
                    cues=("یکنواختی",),
                ),
            ],
            "directions": [
                Direction("direct", ("space",), "min_uniformity"),
                Direction("reverse", ("min_uniformity",), "space", supported=False),
                Direction("whole_table", (), "min_uniformity"),
            ],
            "reverse_policy": "prohibited",
        },
    }


def _domain(records: list[dict[str, Any]], path: str) -> list[str]:
    out: set[str] = set()
    for r in records:
        cur: Any = r
        for part in path.split("."):
            cur = cur.get(part) if isinstance(cur, dict) else None
        if isinstance(cur, list):
            out |= {str(x) for x in cur}
        elif cur is not None and not isinstance(cur, dict):
            out.add(str(cur))
    return sorted(out)


def build_schemas(datasets: dict[str, tuple[str, list[dict[str, Any]]]]) -> dict[str, TableSchema]:
    """Schemas for every dataset: ``datasets`` maps id -> (title, curated records)."""
    out = {}
    for ds, spec in _specs().items():
        title, records = datasets[ds]
        dims = []
        for declared in spec["dimensions"]:
            d = Dimension(**asdict(declared))
            if d.kind in ("category", "identity"):
                d.domain = _domain(records, d.path)
            dims.append(d)
        out[ds] = TableSchema(
            ds,
            spec["family"],
            title,
            dims,
            [Output(**asdict(o)) for o in spec["outputs"]],
            [Direction(**asdict(x)) for x in spec["directions"]],
            [],
            spec.get("reverse_policy", ""),
            len(records),
            [ScopeDecl(**asdict(x)) for x in SCOPES.get(ds, [])],
        )
    return out


def schemas_document(schemas: dict[str, TableSchema]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "description": "structured dataset schemas (derived from curated/; validated by the "
        "query planner)",
        "tables": {ds: s.to_dict() for ds, s in sorted(schemas.items())},
    }


def write_schemas(schemas: dict[str, TableSchema], path: Path = SCHEMA_PATH) -> None:
    path.write_text(
        "# Generated by python -m mabhas19.cli.build_structured_schemas (do not edit by hand).\n"
        + yaml.safe_dump(schemas_document(schemas), allow_unicode=True, sort_keys=False, width=100),
        encoding="utf-8",
        newline="\n",
    )
