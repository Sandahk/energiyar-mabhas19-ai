"""Property-style safety tests of the structured engine, generated from the curated tables.

Queries are built from printed records (deterministic seeds) and then mutated one feature at a
time. The safety properties hold for every generated case:

1. a returned record satisfies every applied constraint (checked by an oracle written here
   from the raw curated fields, independent of the executor);
2. no record contradicting a negated or counted qualifier is returned;
3. invalid / out-of-domain constraints never broaden results (zero records);
4. unresolved constraints, removed required dimensions and unknown entities yield zero records;
5. unsupported calculations, outputs and reverse lookups yield zero records;
6. every returned value carries exact source provenance.

Boundary cases are generated from every printed interval: just below, at and just above each
printed bound, in Latin, Persian and Arabic-Indic numerals.
"""

from __future__ import annotations

import json
import random
from decimal import Decimal
from functools import cache
from typing import Any

import pytest

from mabhas19.core.paths import CURATED_DIR
from mabhas19.indexing.chunking import build_chunks
from mabhas19.retrieval.hybrid import Retriever
from mabhas19.retrieval.models import StructuredHit
from mabhas19.retrieval.numeric import NumInterval, interval_from_operators, printed_interval
from mabhas19.retrieval.plan import StructuredQueryPlan
from mabhas19.retrieval.structured import StructuredStore
from mabhas19.text.normalize import normalize_for_search as norm

APP02 = "m19-ed5-1404-app02.city_climate_classes"
APP03 = "m19-ed5-1404-app03.max_energy_use_intensity"
APP07 = "m19-ed5-1404-app07.material_thermal_properties"
MAIN = "m19-ed5-1404-main.t"
SEED = 20261019
FAIL_CLOSED = {
    "missing_required_dimension",
    "unresolved_constraint",
    "invalid_constraint",
    "no_printed_category",
    "unsupported_reverse_lookup",
    "unsupported_output",
    "unsupported_calculation",
    "unknown_entity",
    "ambiguous_entity",
}


@cache
def _r() -> Retriever:
    path = CURATED_DIR / "text" / "m19-ed5-1404-main" / "blocks.jsonl"
    blocks = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
    return Retriever(build_chunks(blocks, "m19"), StructuredStore(CURATED_DIR))


def _store() -> StructuredStore:
    return _r().store


def _lookup(
    ds: str, query: str, *, explicit: bool = False
) -> tuple[list[StructuredHit], list[Any], StructuredQueryPlan]:
    hits, no_match, _, plan = _r().lookup(ds, query, explicit_reference=explicit)
    _assert_safe(ds, hits, plan)
    return hits, no_match, plan


def _digits(text: str, style: str) -> str:
    if style == "persian":
        return text.translate(str.maketrans("0123456789.", "۰۱۲۳۴۵۶۷۸۹٫"))
    if style == "arabic":
        return text.translate(str.maketrans("0123456789.", "٠١٢٣٤٥٦٧٨٩٫"))
    return text


# ------------------------------------------------------------------ independent oracle


def _climates(ds: str, r: dict[str, Any]) -> set[str] | None:
    if ds in (APP02, MAIN + "19-3-2"):
        return {r["climate_class"]["code"]}
    if ds == APP03:
        return {r["climate_class"]}
    groups = [v["applies_to"] for v in r.values() if isinstance(v, dict) and "applies_to" in v]
    return {c for g in groups for c in g} if groups else None


def _record(ds: str, rid: str) -> dict[str, Any]:
    s = _store()
    if ds == APP02:
        return next(r for r in s.app02 if r["id"] == rid)
    if ds == APP03:
        return next(r for r in s.app03 if r["id"] == rid)
    if ds == APP07:
        return next(r for r in s.app07 if r["id"] == rid)
    return next(r for r in s.main[ds].records if r["record_id"] == rid)


def _oracle(ds: str, r: dict[str, Any], dim: str, c: dict[str, Any]) -> bool | None:
    """Raw-field check of one applied constraint; None when the oracle has no opinion."""
    values = set(c.get("values") or [])
    if dim == "climate_class":
        printed = _climates(ds, r)
        return printed is None or bool(printed & values)
    if dim == "rating":
        rating = r.get("energy_rating")
        code = (
            rating
            if isinstance(rating, str)
            else (rating or {}).get("code") or ((rating or {}).get("printed"))
        )
        return code in values
    if dim == "group":
        return r["building_group"]["text"] in values
    if dim == "year":
        return str(r["year"]["index"]) in values
    return None


def _assert_safe(ds: str, hits: list[StructuredHit], plan: StructuredQueryPlan) -> None:
    applied = [c for c in plan.to_dict()["constraints"] if c["state"] == "applied"]
    for h in hits:
        # 6. exact provenance on every returned value
        assert h.record_id, h
        assert h.doc_id, h.record_id
        assert h.pdf_page is not None, h.record_id
        if ds.startswith(MAIN):
            assert h.source_cells, h.record_id
        r = _record(ds, h.record_id)
        for c in applied:
            ok = _oracle(ds, r, c["dimension"], c)
            assert ok is not False, (ds, h.record_id, c)  # 1. never contradicts a constraint
        # every applied numeric constraint, re-checked by the executor's independent re-check
        for c in plan.constraints:
            if c.state == "applied":
                assert _r().executor.satisfies(ds, r, c), (ds, h.record_id, c.describe())


def _closed(hits: list[StructuredHit], no_match: list[Any]) -> None:
    assert hits == []
    assert no_match
    assert {n.outcome for n in no_match} <= FAIL_CLOSED | {None}


# ------------------------------------------------------------------ appendix 2 / 19-3-2


def _cities(n: int) -> list[dict[str, Any]]:
    rng = random.Random(SEED)
    return rng.sample(_store().app02, n)


@pytest.mark.parametrize("r", _cities(12), ids=lambda r: r["id"])
def test_city_lookup_returns_only_that_city(r: dict[str, Any]) -> None:
    hits, _, _ = _lookup(APP02, f"رده اقلیمی {r['city']['text']}")
    assert r["id"] in {h.record_id for h in hits}
    city = norm(r["city"]["text"]).split("(")[0].strip()
    assert all(city in norm(h.values["city"]["text"]) for h in hits)


@pytest.mark.parametrize("name", ["زرین آباد", "کوهسار نو", "دشتستان کهنه"])
def test_unlisted_place_never_inherits_a_class(name: str) -> None:
    for ds in (APP02, MAIN + "19-3-2"):
        hits, no_match, _ = _lookup(ds, f"رده اقلیمی شهر {name}")
        _closed(hits, no_match)


def test_unlisted_town_with_its_province_is_unknown() -> None:
    for r in _store().main[MAIN + "19-3-2"].records[:8]:
        province = r["province"]["text"]
        hits, no_match, _ = _lookup(APP02, f"رده اقلیمی شهر کوهسارنو استان {province}")
        _closed(hits, no_match)


@pytest.mark.parametrize("code", ["0B", "3A", "4B", "1A", "7A", "6C"])
def test_reverse_class_lookup_is_exact_or_typed(code: str) -> None:
    hits, no_match, _ = _lookup(APP02, f"کدام شهرها رده اقلیمی {code} دارند")
    printed = {r["climate_class"]["code"] for r in _store().app02}
    if code in printed:
        assert hits
        assert {h.values["climate_class"]["code"] for h in hits} == {code}
    else:
        _closed(hits, no_match)


# ------------------------------------------------------------------ appendix 3


def _eui_records(n: int) -> list[dict[str, Any]]:
    rng = random.Random(SEED + 3)
    return rng.sample(_store().app03, n)


def _eui_query(r: dict[str, Any], climate: str | None, rating: str | None, tail: str = "") -> str:
    q = f"حداکثر شدت مصرف انرژی {r['occupancy_label']}"
    if climate:
        q += f" در اقلیم {climate}"
    if rating:
        q += f" برای رده {rating}"
    return q + tail


@pytest.mark.parametrize("r", _eui_records(20), ids=lambda r: r["id"])
def test_eui_exact_query_and_mutations(r: dict[str, Any]) -> None:
    hits, _, _ = _lookup(APP03, _eui_query(r, r["climate_class"], r["energy_rating"]))
    assert r["id"] in {h.record_id for h in hits}
    # 3. invalid rating / class never broaden
    for rating in ("A+", "A++", "E"):
        _closed(*_lookup(APP03, _eui_query(r, r["climate_class"], rating))[:2])
    for climate in ("5C", "7A"):
        _closed(*_lookup(APP03, _eui_query(r, climate, r["energy_rating"]))[:2])
    # 4. a removed required dimension (climate class) is never assumed
    _closed(*_lookup(APP03, _eui_query(r, None, r["energy_rating"]))[:2])
    # 5. calculations and reverse lookups return no operands
    for tail in (" میانگین دو رده چقدر است", " و نسبت آن به رده D چند برابر است"):
        _closed(*_lookup(APP03, _eui_query(r, r["climate_class"], r["energy_rating"], tail))[:2])
    _closed(*_lookup(APP03, "کدام کاربری شدت مصرف انرژی 150 کیلووات ساعت دارد")[:2])
    _closed(
        *_lookup(
            APP03,
            "برای ساختمان 500 متری اداری عمومی در اقلیم 3B رده A چند کیلووات ساعت در سال مجاز است",
        )[:2]
    )


def test_unknown_occupancy_is_unknown() -> None:
    _closed(*_lookup(APP03, "حداکثر شدت مصرف انرژی ایستگاه فضایی در اقلیم 3B رده A")[:2])


# ------------------------------------------------------------------ appendix 7


def _density_bounds() -> list[tuple[dict[str, Any], Decimal]]:
    out = []
    for r in _store().app07:
        iv = printed_interval(norm(r["density"]["raw"]))
        if iv is None:
            continue
        for b in (iv.lo, iv.hi):
            if b is not None:
                out.append((r, b.value))
    rng = random.Random(SEED + 7)
    return rng.sample(out, 30)


@pytest.mark.parametrize("case", _density_bounds(), ids=lambda c: f"{c[0]['id']}@{c[1]}")
def test_density_boundaries_never_return_a_disjoint_row(
    case: tuple[dict[str, Any], Decimal],
) -> None:
    r, bound = case
    for delta, style in ((Decimal(-1), "latin"), (Decimal(0), "persian"), (Decimal(1), "arabic")):
        x = bound + delta
        q = _digits(f"ضریب هدایت حرارتی {r['display_name']} با چگالی {x}", style)
        hits, _, _ = _lookup(APP07, q)
        point = NumInterval.point(x)
        for h in hits:
            iv = printed_interval(norm(_record(APP07, h.record_id)["density"]["raw"]))
            assert iv is not None
            assert iv.relation(point) in ("within", "boundary"), (h.record_id, str(x))


@pytest.mark.parametrize("value", ["0.025", "۰٫۰۲۵", "٠٫٠٢٥", "0/025"])
def test_unprinted_conductivity_bound_finds_nothing(value: str) -> None:
    lowest = min(Decimal(r["thermal_conductivity"]["value"]) for r in _store().app07)
    assert lowest > Decimal("0.025")
    _closed(*_lookup(APP07, f"عایقی با ضریب هدایت حرارتی کمتر از {value} هست؟")[:2])


@pytest.mark.parametrize("word", ["پلاستوفوم نانویی", "عایق برند ایکس", "چوب بامبو"])
def test_unknown_material_is_unknown(word: str) -> None:
    hits, no_match, _ = _lookup(APP07, f"ضریب هدایت حرارتی {word}")
    _closed(hits, no_match)


# ------------------------------------------------------------------ main tables with labels


def _label_cases(table: str, n: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    recs = [r for r in _store().main[MAIN + table].records if r.get("climate_group")]
    return rng.sample(recs, min(n, len(recs)))


@pytest.mark.parametrize("r", _label_cases("19-5-1", 18, SEED + 51), ids=lambda r: r["record_id"])
def test_envelope_exact_query_and_mutations(r: dict[str, Any]) -> None:
    climate = r["climate_group"]["applies_to"][0]
    element = r["element"]["text"]
    hits, _, _ = _lookup(MAIN + "19-5-1", f"حداقل مقاومت حرارتی {element} در اقلیم {climate}")
    assert r["record_id"] in {h.record_id for h in hits}
    assert all(h.values["element"]["text"] == element for h in hits)
    _closed(*_lookup(MAIN + "19-5-1", f"حداقل مقاومت حرارتی {element} در اقلیم 9A")[:2])
    _closed(
        *_lookup(MAIN + "19-5-1", f"میانگین مقاومت حرارتی {element} در اقلیم {climate} و 2A")[:2]
    )
    _closed(
        *_lookup(MAIN + "19-5-1", f"حداقل مقاومت حرارتی {element} در اقلیم {climate} چند دلار است")[
            :2
        ]
    )


def _qualifier_pairs() -> list[tuple[str, str, str]]:
    """(table, positive label text, negated label text) where the tables print both «X» and
    «غیرX» / «تک»…«چند» variants of one head."""
    out = []
    for table in ("19-5-3", "19-5-4"):
        recs = _store().main[MAIN + table].records
        texts = {f"{r['equipment']['text']} {r['specification']['text']}": r for r in recs}
        for a in texts:
            for b in texts:
                if a != b and (
                    "غیر" + a.split()[-1] in b.replace(" ", "")
                    or ("غیرچگالشی" in b.replace(" ", "") and "چگالشی" in a and "غیر" not in a)
                ):
                    out.append((table, a, b))
    return out[:6]


def test_negated_qualifier_rows_are_never_returned() -> None:
    # «بویلر چگالشی» vs «بویلر غیرچگالشی», «پکیج چگالشی» vs «پکیج گازی غیر چگالشی»
    for table, query, banned in (
        ("19-5-3", "حداقل بازده بویلر چگالشی", "غیرچگالشی"),
        ("19-5-3", "حداقل بازده بویلر غیر چگالشی", None),
        ("19-5-4", "حداقل بازده پکیج چگالشی", "غیرچگالشی"),
        ("19-5-4", "حداقل بازده پکیج گازی غیر چگالشی", None),
    ):
        hits, _, _ = _lookup(MAIN + table, query)
        assert hits, query
        for h in hits:
            spec = f"{h.values['equipment']['text']} {h.values['specification']['text']}".replace(
                " ", ""
            )
            if banned:
                assert banned not in spec, (query, h.record_id)
            else:
                assert "غیرچگالشی" in spec, (query, h.record_id)


@pytest.mark.parametrize(
    "query",
    [
        "حداقل ضریب عملکرد چیلر جذبی تک اثره",  # the table prints only «دو اثره»
        "کلاس بازدهی الکتروموتور سه فاز چند سرعته با توان ۵۰ کیلووات",
    ],
)
def test_contradicting_count_qualifier_returns_nothing(query: str) -> None:
    hits, no_match, _ = _lookup(MAIN + "19-5-3", query)
    _closed(hits, no_match)


# ------------------------------------------------------------------ interval tables


def _r_bounds() -> list[Decimal]:
    out = set()
    for r in _store().main[MAIN + "19-3-3"].records:
        iv = interval_from_operators(r["r_range"])
        for b in (iv.lo, iv.hi):
            if b is not None:
                out |= {b.value - Decimal("0.01"), b.value, b.value + Decimal("0.01")}
    return sorted(out)


@pytest.mark.parametrize("x", _r_bounds(), ids=str)
@pytest.mark.parametrize("style", ["latin", "persian", "arabic"])
def test_r_ratio_boundaries_are_exact(x: Decimal, style: str) -> None:
    q = _digits(f"اگر نسبت R برابر {x} باشد طبق جدول ۱۹-۳-۳ چه رده ای دارد", style)
    hits, no_match, _ = _lookup(MAIN + "19-3-3", q, explicit=True)
    containing = [
        r["record_id"]
        for r in _store().main[MAIN + "19-3-3"].records
        if interval_from_operators(r["r_range"]).contains(x) == "yes"
    ]
    assert sorted(h.record_id for h in hits) == sorted(containing)
    if not containing:
        _closed(hits, no_match)


def _dd_bounds() -> list[tuple[str, Decimal]]:
    out = set()
    for r in _store().main[MAIN + "19-3-1"].records:
        for c in r["criterion"]["conditions"]:
            iv = interval_from_operators(c)
            for b in (iv.lo, iv.hi):
                if b is not None:
                    out |= {
                        (c["variable"], b.value + d) for d in (Decimal(-1), Decimal(0), Decimal(1))
                    }
    return sorted(out)


@pytest.mark.parametrize("case", _dd_bounds(), ids=lambda c: f"{c[0]}={c[1]}")
def test_degree_day_boundaries_are_exact(case: tuple[str, Decimal]) -> None:
    var, x = case
    word = "گرمایش" if var.startswith("HDD") else "سرمایش"
    hits, no_match, _ = _lookup(MAIN + "19-3-1", f"روز درجه {word} {x} مربوط به کدام اقلیم است")
    for h in hits:
        conds = [
            c
            for c in _record(MAIN + "19-3-1", h.record_id)["criterion"]["conditions"]
            if c["variable"] == var
        ]
        assert conds
        assert all(interval_from_operators(c).contains(x) == "yes" for c in conds)
    if not hits:
        _closed(hits, no_match)


@pytest.mark.parametrize("pf", ["19", "20", "21", "49", "50", "51", "69", "70", "71"])
def test_pf_band_boundaries(pf: str) -> None:
    hits, _, _ = _lookup(MAIN + "19-5-2", f"حداکثر SHGC پنجره با PF {pf} درصد در اقلیم 3B")
    x = Decimal(pf)
    for h in hits:
        iv = printed_interval(norm(h.values["row"]["text"]))
        assert iv is not None
        assert iv.contains(x) in ("yes", "boundary"), (pf, h.values["row"]["text"])


@pytest.mark.parametrize("area", ["1999", "2000", "2001", "۲۰۰۰", "٢٠٠١"])
def test_daylight_area_boundaries(area: str) -> None:
    hits, no_match, _ = _lookup(
        MAIN + "19-5-6", f"حداقل نور روز برای ساختمان اداری {area} متر مربع"
    )
    x = Decimal(norm(area))
    for h in hits:
        cond = h.values["use"].get("area_condition")
        if cond:
            iv = printed_interval(norm(h.values["use"]["text"]))
            assert iv is not None
            assert iv.contains(x) == "yes", (area, h.values["use"]["text"])
    if x == 2000:
        _closed(hits, no_match)  # «کوچک‌تر از» and «بزرگ‌تر از» 2000 both exclude 2000


# ------------------------------------------------------------------ outputs and reverse lookups


@pytest.mark.parametrize(
    ("ds", "query"),
    [
        (MAIN + "19-5-5", "کدام محل مصرف حداکثر دمای آب گرم 49 درجه دارد"),
        (MAIN + "19-5-7", "کدام فضا حداقل یکنواختی 0.7 دارد"),
        (MAIN + "19-2-1", "کدام الزام برای گروه الف در سال اول تیک دارد و قیمتش چند است"),
    ],
)
def test_unsupported_reverse_or_output_returns_nothing(ds: str, query: str) -> None:
    hits, no_match, _ = _lookup(ds, query)
    _closed(hits, no_match)
