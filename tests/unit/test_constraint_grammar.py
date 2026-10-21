"""Source-attested constraint vocabulary (M2): generated properties of the constraint parser.

Generated from the schemas, the printed rows and the rating vocabulary read from the curated
Mabhas 19 text (deterministic): every comparison form keeps its operator and inclusivity in
Latin, Persian and Arabic-Indic digits; complementary coordinated bounds form one interval of
one dimension, anything else stays separate or fails closed; rating letters, plus grades and
attested descriptions agree or the constraint is invalid; coordinated category values are all
applied; no number is dropped or given to a second dimension.
"""

from __future__ import annotations

import json
import random
from decimal import Decimal
from functools import cache

import pytest

from mabhas19.core.paths import CURATED_DIR
from mabhas19.indexing.chunking import build_chunks
from mabhas19.retrieval import ratings
from mabhas19.retrieval.dimensions import groups_requested
from mabhas19.retrieval.hybrid import Retriever
from mabhas19.retrieval.numeric import CLOSED, OPEN, Bound, NumInterval, find_quantities
from mabhas19.retrieval.plan import StructuredQueryPlan
from mabhas19.retrieval.structured import StructuredStore
from mabhas19.text.normalize import normalize_for_search as norm

APP03 = "m19-ed5-1404-app03.max_energy_use_intensity"
APP07 = "m19-ed5-1404-app07.material_thermal_properties"
MAIN = "m19-ed5-1404-main.t"
SEED = 20261021


@cache
def _r() -> Retriever:
    path = CURATED_DIR / "text" / "m19-ed5-1404-main" / "blocks.jsonl"
    blocks = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
    return Retriever(build_chunks(blocks, "m19"), StructuredStore(CURATED_DIR))


def _plan(ds: str, text: str) -> StructuredQueryPlan:
    q = norm(text)
    return _r().planner.plan(ds, q, full_norm=q)


def _digits(x: str) -> list[str]:
    persian = x.translate(str.maketrans("0123456789.", "۰۱۲۳۴۵۶۷۸۹٫"))
    arabic = x.translate(str.maketrans("0123456789.", "٠١٢٣٤٥٦٧٨٩٫"))
    return [x, persian, arabic]


# ---------------------------------------------------------------- comparison grammar

# (wording, which end, inclusivity)
_FORMS = [
    ("کمتر از", "hi", OPEN),
    ("زیر", "hi", OPEN),
    ("<", "hi", OPEN),
    ("بیشتر از", "lo", OPEN),
    ("بیش از", "lo", OPEN),
    ("بالای", "lo", OPEN),
    (">", "lo", OPEN),
    ("حداقل", "lo", CLOSED),
    ("≥", "lo", CLOSED),
    ("حداکثر", "hi", CLOSED),
    ("≤", "hi", CLOSED),
]
_VALUES = ["20", "150", "1600", "0.035"]


@pytest.mark.parametrize(("word", "end", "kind"), _FORMS)
@pytest.mark.parametrize("value", _VALUES)
def test_each_comparison_keeps_its_operator_and_inclusivity(
    word: str, end: str, kind: str, value: str
) -> None:
    for digits in _digits(value):
        (q,) = find_quantities(norm(f"چگالی {word} {digits}"))
        bound = q.interval.lo if end == "lo" else q.interval.hi
        other = q.interval.hi if end == "lo" else q.interval.lo
        assert bound is not None
        assert bound.value == Decimal(value)
        assert bound.kind == kind
        assert other is None


@pytest.mark.parametrize(
    ("text", "lo", "hi"),
    [
        ("20 یا کمتر", None, (20, CLOSED)),
        ("20 یا بیشتر", (20, CLOSED), None),
        ("بین 20 تا 30", (20, CLOSED), (30, CLOSED)),
        ("برابر 20", (20, CLOSED), (20, CLOSED)),
    ],
)
def test_inclusive_forms_are_never_confused_with_exclusive_ones(
    text: str, lo: tuple[int, str] | None, hi: tuple[int, str] | None
) -> None:
    (q,) = find_quantities(norm(text))
    for bound, want in ((q.interval.lo, lo), (q.interval.hi, hi)):
        if want is None:
            assert bound is None
        else:
            assert bound is not None
            assert (bound.value, bound.kind) == (Decimal(want[0]), want[1])


def test_a_printed_exception_range_is_negated() -> None:
    printed = "با توان بیش از ۱۲۰وات و کمتر از ۱۰۰۰ کیلووات (به غیر از ۷۵ تا ۲۰۰ کیلووات)"
    qs = find_quantities(norm(printed), printed=True)
    assert [q.negated for q in qs] == [False, False, True]
    assert not find_quantities(norm("بین ۷۵ تا ۲۰۰ کیلووات"), printed=True)[0].negated


# ---------------------------------------------------------------- coordinated bounds


def _density_rows() -> list[tuple[str, Decimal, Decimal]]:
    """(material name, printed density lo, hi) of App07 rows printing a closed range."""
    out = []
    for r in _r().store.app07:
        d = r["density"]
        if d.get("kind") == "range" and d.get("min") and d.get("max"):
            out.append((str(r["display_name"]), Decimal(d["min"]), Decimal(d["max"])))
    rng = random.Random(f"{SEED}:density")
    return rng.sample(out, 6)


@pytest.mark.parametrize("coordinator", ["ولی", "و", "اما"])
@pytest.mark.parametrize("row", _density_rows(), ids=lambda r: f"{r[1]}-{r[2]}")
def test_complementary_bounds_form_one_interval(
    coordinator: str, row: tuple[str, Decimal, Decimal]
) -> None:
    name, lo, hi = row
    p = _plan(APP07, f"ضریب هدایت حرارتی {name} با چگالی بالای {lo} {coordinator} زیر {hi}")
    density = [c for c in p.constraints if c.dimension == "density"]
    assert len(density) == 1
    (c,) = density
    assert c.rule == "coordinated_bound"
    assert c.interval == NumInterval(Bound(lo, OPEN), Bound(hi, OPEN))
    assert c.source.startswith("بالای")
    assert c.source.endswith(str(hi))


def test_mixed_inclusivity_bounds_keep_each_end() -> None:
    p = _plan(MAIN + "19-3-1", "روز درجه گرمایش بیشتر از ۳۰۰۰ و حداکثر ۴۰۰۰ چه اقلیمی است؟")
    (c,) = [c for c in p.constraints if c.dimension.startswith("degree_days")]
    assert c.interval is not None
    assert c.interval.lo is not None
    assert c.interval.hi is not None
    assert (c.interval.lo.value, c.interval.lo.kind) == (Decimal(3000), OPEN)
    assert (c.interval.hi.value, c.interval.hi.kind) == (Decimal(4000), CLOSED)


def test_contradictory_bounds_fail_closed() -> None:
    res = _r().retrieve("ضریب هدایت حرارتی پشم سنگ با چگالی بالای ۱۷۵ ولی زیر ۱۵۰ چقدر است؟")
    assert res.structured == []
    assert "contradictory_bounds" in {n.reason for n in res.no_match}


def test_two_bounds_on_the_same_side_are_not_merged_and_fail_closed() -> None:
    res = _r().retrieve("ضریب هدایت حرارتی پشم سنگ با چگالی بالای ۱۵۰ و بالای ۱۷۵ چقدر است؟")
    assert res.structured == []


def test_bounds_of_two_dimensions_stay_separate() -> None:
    p = _plan(APP07, "پلی استایرن اکسترود با چگالی بالای ۳۰ و ضخامت بیش از ۶۰ میلی‌متر")
    density = [c for c in p.constraints if c.dimension == "density"]
    assert density
    assert all(c.rule != "coordinated_bound" for c in density)


# ---------------------------------------------------------------- ratings


def test_the_rating_vocabulary_is_read_from_the_source() -> None:
    vocab = ratings.vocabulary()
    assert vocab
    for attestations in vocab.values():
        assert attestations
        assert all(a.block.startswith("main-p") for a in attestations)


def _rating_domains() -> list[tuple[str, str]]:
    out = []
    for ds, s in _r().planner.schemas.items():
        d = s.dim("rating")
        if d is not None:
            out += [(ds, v) for v in d.domain if v != "-"]
    return out


@pytest.mark.parametrize(("ds", "letter"), _rating_domains())
def test_every_printed_grade_and_plus_grade_parses(ds: str, letter: str) -> None:
    for text in (f"رده {letter}", f"رده انرژی {letter}", f"رده ({letter})", f"کلاس {letter}"):
        assert ratings.letters(norm(text)) == [letter]


def _attested() -> list[tuple[str, str]]:
    out = []
    for desc, att in ratings.vocabulary().items():
        letters = {a.letter for a in att}
        if desc != ratings.MINIMUM and len(letters) == 1:
            out.append((desc, next(iter(letters))))
    return out


@pytest.mark.parametrize(("desc", "letter"), _attested())
def test_a_description_agrees_with_its_printed_grade(desc: str, letter: str) -> None:
    assert ratings.letters(norm(f"رده {desc}")) == [letter]
    assert ratings.letters(norm(f"رده {desc} ({letter})")) == [letter]
    assert ratings.conflicts(norm(f"رده {desc} ({letter})")) == []


@pytest.mark.parametrize(("desc", "letter"), _attested())
def test_a_description_contradicting_the_letter_fails_closed(desc: str, letter: str) -> None:
    other = next(x for x in "ABCD" if x != letter)
    text = f"حداکثر شدت مصرف انرژی کتابخانه در اقلیم 3B برای رده {desc} ({other})"
    assert ratings.conflicts(norm(text))
    res = _r().retrieve(text)
    assert res.structured == []
    assert "rating_conflict" in {n.reason for n in res.no_match}


def test_the_minimum_class_phrase_needs_its_printed_grade() -> None:
    assert ratings.letters(norm("حداقل رده (D)")) == ["D"]
    assert ratings.conflicts(norm("حداقل رده C"))
    assert ratings.letters(norm("حداقل به چه رده انرژی باید رسید")) == []


def test_coordinated_rating_letters_are_all_applied() -> None:
    res = _r().retrieve("رده‌های A و B برای کتابخانه در اقلیم 3B چه سقف مصرفی دارند؟")
    got = {h.record_id.split("-")[1] for h in res.structured}
    assert got == {"A", "B"}


# ---------------------------------------------------------------- groups, ordinals, cardinals


@pytest.mark.parametrize(
    ("text", "groups"),
    [
        ("گروه‌های ج و د", ["ج", "د"]),
        ("گروه الف، ب و پ", ["الف", "ب", "پ"]),
        ("گروه ب", ["ب"]),
    ],
)
def test_coordinated_groups_are_all_read(text: str, groups: list[str]) -> None:
    assert groups_requested(norm(text)) == sorted(groups)


def test_coordinated_groups_return_a_row_per_group() -> None:
    res = _r().retrieve("شارژر خودرو برقی در سال اول برای گروه‌های ج و د چه وضعیتی دارد؟")
    assert len(res.structured) == 2
    p = res.plans[0]
    (g,) = [c for c in p["constraints"] if c["dimension"] == "group"]
    assert sorted(g["values"]) == ["ج", "د"]


def test_an_ordinal_year_and_a_cardinal_count_each_get_one_explanation() -> None:
    p = _plan(MAIN + "19-2-1", "سیستم سرمایش و گرمایش ۴ لوله برای گروه ب در سال سوم")
    years = [c for c in p.constraints if c.dimension == "year"]
    assert [c.values for c in years] == [["3"]]
    assert not [c for c in p.constraints if c.dimension.startswith("label:")]
    assert not p.unresolved


# ---------------------------------------------------------------- reverse lookups, safety


def test_a_reverse_interval_returns_only_the_printed_row_it_equals() -> None:
    ok = _r().retrieve("اقلیمی که روز درجه گرمایش آن بیشتر از ۳۰۰۰ و حداکثر ۴۰۰۰ است چیست؟")
    assert [h.record_id for h in ok.structured] == ["main-t19-3-1-r06"]
    for straddling in ("بیشتر از ۲۵۰۰ و حداکثر ۴۰۰۰", "حداقل ۳۰۰۰ و حداکثر ۴۰۰۰"):
        res = _r().retrieve(f"اقلیمی که روز درجه گرمایش آن {straddling} است چیست؟")
        assert res.structured == []


@pytest.mark.parametrize(("kw", "item"), [("۷۵", "item13"), ("۱۵۰", "item13"), ("۵۰۰", "item12")])
def test_a_stated_power_selects_only_the_band_printing_it(kw: str, item: str) -> None:
    res = _r().retrieve(f"الکتروموتور سه فاز تک‌سرعته با توان {kw} کیلووات چه کلاس بازدهی دارد؟")
    assert [h.record_id for h in res.structured] == [f"main-t19-5-3-{item}"]


def test_a_number_with_an_explicit_cue_never_attaches_to_another_dimension() -> None:
    p = _plan(APP07, "بتن سبکدانه با رس یا شیست منبسط، چگالی 1600 تا 1800 و ماسه رودخانه")
    dims = [c.dimension for c in p.constraints if c.source.endswith("1800")]
    assert dims == ["density"]


@pytest.mark.parametrize(
    "text",
    [
        "ضریب هدایت حرارتی پشم سنگ با چگالی بالای ۱۵۰ ولی زیر ۱۷۵ چقدر است؟",
        "رده‌های A و B برای کتابخانه در اقلیم 3B چه سقف مصرفی دارند؟",
        "شارژر خودرو برقی در سال اول برای گروه‌های ج و د چه وضعیتی دارد؟",
        "انبار یخچال‌دار در اقلیم 4A برای رده نزدیک به مصرف صفر (A) چه شدت مصرف مجازی دارد؟",
    ],
)
def test_every_stated_number_and_value_is_grounded_and_certified(text: str) -> None:
    res = _r().retrieve(text)
    assert res.structured
    assert res.certification_failures == []
    for a in res.admissions:
        assert a["admitted"]
        assert not a.get("unresolved") or all(not w.isdigit() for w in a["unresolved"])


@pytest.mark.parametrize("use", ["اداری", "تجاری"])
def test_a_label_number_never_anchors_a_lookup_of_another_tables_output(use: str) -> None:
    # the area selects a 19-5-6 row, but the question asks Appendix 3's energy use
    res = _r().retrieve(
        f"یه ساختمون {use} ۵۰۰ متری تو اقلیم 4A دارم؛ برای رده C کل مصرف سالانه مجازش چند "
        "کیلووات ساعته؟"
    )
    assert not [h for h in res.structured if h.dataset == MAIN + "19-5-6"]
    named = _r().retrieve(f"ساختمان {use} ۵۰۰ متری حداقل چند درصد کفایت نور روز لازم داره؟")
    assert {h.dataset for h in named.structured} == {MAIN + "19-5-6"}
