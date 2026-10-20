"""Property tests of plan admission and result certification, generated from every schema.

Queries are generated (deterministic seeds) from the printed rows of every structured dataset
and mutated one feature at a time. Whatever the route, these safety properties must hold:

1. no non-admitted plan executes (every returned row belongs to an admitted plan);
2. no returned row fails independent certification (and no certification failure is ever
   needed for queries built from the source);
3. an exact-valued reverse lookup never returns a nearby value;
4. an explicitly named table never leaks another table's rows;
5. an excluded qualifier never appears in a returned row;
6. an unsupported calculation returns zero regulatory values.

Mutations: a conceptual question with one table word, an unrelated entity sharing one token, an
explicitly named different table; a missing distinguishing modifier, an unprinted extra
modifier, an excluded (negated) sibling, a similar sibling row; exact, nearby, just below/above
values in Latin, Persian and Arabic-Indic digits and with thousands separators; exact, near-miss,
prohibited and multi-row reverse lookups; sums, differences, ratios and interpolation.
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
from mabhas19.retrieval.identity import IdentityIndex, query_tokens, row_label
from mabhas19.retrieval.models import RetrievalResult
from mabhas19.retrieval.polarity import parse
from mabhas19.retrieval.structured import StructuredStore
from mabhas19.text.normalize import normalize_for_search as norm

APP02 = "m19-ed5-1404-app02.city_climate_classes"
APP03 = "m19-ed5-1404-app03.max_energy_use_intensity"
APP07 = "m19-ed5-1404-app07.material_thermal_properties"
MAIN = "m19-ed5-1404-main.t"
SEED = 20261020
N = 4  # generated cases per dataset and mutation


@cache
def _r() -> Retriever:
    path = CURATED_DIR / "text" / "m19-ed5-1404-main" / "blocks.jsonl"
    blocks = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
    return Retriever(build_chunks(blocks, "m19"), StructuredStore(CURATED_DIR))


def _store() -> StructuredStore:
    return _r().store


def _identity() -> dict[str, IdentityIndex]:
    return _r().planner.identity


def _records(ds: str) -> list[dict[str, Any]]:
    s = _store()
    if ds == APP02:
        return list(s.app02)
    if ds == APP03:
        return list(s.app03)
    if ds == APP07:
        return list(s.app07)
    return list(s.main[ds].records)


def _rid(r: dict[str, Any]) -> str:
    return str(r.get("id") or r.get("record_id"))


def _row(ds: str, r: dict[str, Any]) -> str:
    """The identity row of a record (an Appendix 3 record is identified by its occupancy)."""
    return str(r["occupancy_id"]) if ds == APP03 else _rid(r)


def _retrieve(query: str) -> RetrievalResult:
    res = _r().retrieve(query)
    _invariants(res)
    return res


def _invariants(res: RetrievalResult) -> None:
    """Properties 1, 2, 4 and 5 for any retrieval result."""
    admitted = {a["target"] for a in res.admissions if a["admitted"]}
    explicit = {e for a in res.admissions for e in a.get("explicit", [])}
    pol = parse(res.normalized_query)
    excluded = [query_tokens(e.text) for e in pol.exclusions]
    for h in res.structured:
        assert h.dataset in admitted, (res.query, h.record_id)  # 1
        assert not explicit or h.dataset in explicit, (res.query, h.record_id)  # 4
        label = {w for w in query_tokens(row_label(_store(), h.dataset, _r()._record(h)))}
        for ph in excluded:  # 5
            content = [w for w in ph if len(w) > 1]
            assert not (content and set(content) <= label), (res.query, h.record_id, ph)
    assert res.certification_failures == [], (res.query, res.certification_failures)  # 2


DATASETS = sorted(
    [APP02, APP03, APP07, *(t for t in StructuredStore(CURATED_DIR).main)],
)
LABELLED = [APP03, APP07, *(MAIN + t for t in ("19-2-1", "19-5-1", "19-5-3", "19-5-4", "19-5-5",
                                              "19-5-6", "19-5-7"))]  # fmt: skip


def _output_name(ds: str) -> str:
    return str(_store().schemas[ds].outputs[0].names[0])


def _exact_query(ds: str, r: dict[str, Any]) -> str:
    """A question built from the printed row: its label, its stated dimensions, its output."""
    s = _store()
    family = s.schemas[ds].family
    if family == "place":
        return f"رده اقلیمی {r['city']['text']} چیست؟"
    if ds == APP03:
        label = s.occupancies[r["occupancy_id"]]["label"].split("/")[0]
        return f"حداکثر شدت مصرف {label} در اقلیم {r['climate_class']} رده {r['energy_rating']}"
    if ds == APP07:
        return f"ضریب هدایت حرارتی {r['display_name']} چقدر است؟"
    if family == "criterion":
        return f"معیار روز درجه اقلیم {r['climate_classes']['applies_to'][0]} چیست؟"
    if family == "rating_range":
        return f"بازه نسبت R برای رده {r['energy_rating']['code']} چیست؟"
    label = row_label(s, ds, r)
    extra = ""
    cc = (r.get("climate_group") or {}).get("applies_to")
    if cc:
        extra += f" در اقلیم {cc[0]}"
    if ds == MAIN + "19-2-1":
        extra += f" برای گروه {r['building_group']['text']} در سال {r['year']['index']}"
    return f"{_output_name(ds)} {label}{extra} چقدر است؟"


def _sample(ds: str, n: int = N, seed: int = SEED) -> list[dict[str, Any]]:
    rows = _records(ds)
    return random.Random(f"{seed}:{ds}").sample(rows, min(n, len(rows)))


# ---------------------------------------------------------------- exact (positive) queries


@pytest.mark.parametrize("ds", DATASETS, ids=lambda d: d.split(".")[-1])
def test_source_built_queries_satisfy_every_invariant(ds: str) -> None:
    for r in _sample(ds):
        res = _retrieve(_exact_query(ds, r))
        allowed = {ds, APP02, MAIN + "19-3-2"} if ds in (APP02, MAIN + "19-3-2") else {ds}
        assert all(h.dataset in allowed for h in res.structured), res.query


# ---------------------------------------------------------------- candidate-admission negatives


def _identity_words(ds: str, row: str) -> list[str]:
    return [w for a in _identity()[ds].by_row.get(row, []) for w in a.words]


@pytest.mark.parametrize("ds", LABELLED, ids=lambda d: d.split(".")[-1])
def test_conceptual_question_with_one_table_word_is_not_a_lookup(ds: str) -> None:
    for r in _sample(ds):
        for w in _identity_words(ds, _row(ds, r))[:2]:
            res = _retrieve(f"اهمیت {w} در مبحث ۱۹ چیست و چرا باید به آن توجه کرد؟")
            assert res.structured == [], (res.query, [h.record_id for h in res.structured])


@pytest.mark.parametrize("ds", LABELLED, ids=lambda d: d.split(".")[-1])
def test_unrelated_entity_sharing_one_token_returns_no_row(ds: str) -> None:
    idx = _identity()[ds]
    for r in _sample(ds):
        alts = idx.by_row.get(_row(ds, r), [])
        words = [w for a in alts for w in a.words if w not in a.heads]
        for w in words[:1]:
            res = _retrieve(f"{_output_name(ds)} سردخانه میوه {w} چقدر است؟")
            assert _rid(r) not in _returned(ds, res) | {h.record_id for h in res.structured}, (
                res.query
            )


@pytest.mark.parametrize("ds", DATASETS, ids=lambda d: d.split(".")[-1])
def test_explicitly_named_other_table_never_leaks(ds: str) -> None:
    other = "19-5-7" if ds != MAIN + "19-5-7" else "19-5-5"
    for r in _sample(ds, 2):
        res = _retrieve(f"طبق جدول {other}، {_exact_query(ds, r)}")
        assert all(h.dataset == MAIN + other for h in res.structured), res.query


# ---------------------------------------------------------------- entity mutations


def _siblings(ds: str) -> list[tuple[str, str, list[str]]]:
    """(row, sibling row, the row's distinguishing words) for labels that share a head."""
    idx = _identity()[ds]
    out = []
    for a in idx.alternatives:
        name = _name(a)
        if not name:
            continue
        group = [g for g in idx.group(name[0]) if g is not a and g.row != a.row]
        for g in group:
            # words of the row's own name that tell this label from the sibling
            dist = [w for w in name if not any(w == x for x in g.words)]
            if dist and g.words:
                out.append((a.row, g.row, dist))
                break
    rng = random.Random(f"{SEED}:siblings:{ds}")
    return rng.sample(out, min(N, len(out)))


def _name(a: Any) -> list[str]:
    """The most specific printed level of a label: the row's own name (an Appendix 7 material,
    a main-table item), not its category."""
    return next((lv for lv in reversed(a.levels) if lv), [])


def _label(ds: str, row: str) -> str:
    """The printed label of an identity row (an Appendix 3 occupancy, else a record)."""
    if ds == APP03:
        label = str(_store().occupancies[row]["label"]).split("/")[0]
        return f"{label} در اقلیم 3B رده C"
    rec = next(r for r in _records(ds) if _rid(r) == row)
    return row_label(_store(), ds, rec)


def _returned(ds: str, res: RetrievalResult) -> set[str]:
    """Identity rows of the returned records (Appendix 3 records map to their occupancy)."""
    if ds == APP03:
        occ = {r["id"]: r["occupancy_id"] for r in _store().app03}
        return {occ[h.record_id] for h in res.structured if h.record_id in occ}
    return {h.record_id for h in res.structured}


@pytest.mark.parametrize("ds", LABELLED, ids=lambda d: d.split(".")[-1])
def test_missing_distinguishing_modifier_never_names_the_row(ds: str) -> None:
    for row, _sib, dist in _siblings(ds):
        alts = _identity()[ds].by_row[row]
        name = next(_name(a) for a in alts if set(dist) <= set(_name(a)))
        kept = [w for w in name if w not in dist]  # general levels left out too
        if any(_name(a) != name and set(_name(a)) <= set(kept) for a in alts):
            continue  # another printed name of the row («خشت، گل، …») is still stated whole
        res = _retrieve(f"{_output_name(ds)} {' '.join(kept)} چقدر است؟")
        assert row not in _returned(ds, res), res.query


@pytest.mark.parametrize("ds", LABELLED, ids=lambda d: d.split(".")[-1])
def test_extra_unprinted_modifier_returns_no_row(ds: str) -> None:
    for r in _sample(ds):
        words = norm(_label(ds, _row(ds, r))).split(" ")
        # the modifier follows the first word (attributive position), not a printed numbering
        at = next(i for i, w in enumerate(words) if any(c.isalpha() for c in w)) + 1
        res = _retrieve(
            f"{_output_name(ds)} {' '.join(words[:at])} کوانتومی {' '.join(words[at:])} چقدر است؟"
        )
        assert _row(ds, r) not in _returned(ds, res), res.query


@pytest.mark.parametrize("ds", LABELLED, ids=lambda d: d.split(".")[-1])
def test_excluded_sibling_is_never_returned(ds: str) -> None:
    for row, sib, _dist in _siblings(ds):
        res = _retrieve(f"{_output_name(ds)} {_label(ds, sib)} (نه {_label(ds, row)}) چقدر است؟")
        same = _identity_words(ds, row) == _identity_words(ds, sib)
        assert same or row not in _returned(ds, res), res.query


@pytest.mark.parametrize("ds", LABELLED, ids=lambda d: d.split(".")[-1])
def test_similar_sibling_never_returns_the_other_row(ds: str) -> None:
    for row, sib, _dist in _siblings(ds):
        res = _retrieve(f"{_output_name(ds)} {_label(ds, sib)} چقدر است؟")
        same = _identity_words(ds, row) == _identity_words(ds, sib)
        assert same or row not in _returned(ds, res), res.query


# ---------------------------------------------------------------- numeric mutations


def _styles(x: str) -> list[str]:
    latin = x
    persian = x.translate(str.maketrans("0123456789.", "۰۱۲۳۴۵۶۷۸۹٫"))
    arabic = x.translate(str.maketrans("0123456789.", "٠١٢٣٤٥٦٧٨٩٫"))
    return [latin, persian, arabic]


# exact-valued reverse lookups the source defines: (dataset, output path, question template)
_R_Q = "در جدول ۱۹-۵-۱ مقاومت حرارتی {v} مربوط به کدام اقلیم‌ها و جداره‌هاست؟"
_U_Q = "در جدول ۱۹-۵-۲ ضریب انتقال حرارت {v} مربوط به کدام جداره و اقلیم است؟"
REVERSE = [
    (APP07, "thermal_conductivity.value", "کدام مصالح در پیوست ۷ ضریب هدایت حرارتی {v} دارند؟"),
    (MAIN + "19-5-1", "min_r_value.value", _R_Q),
    (MAIN + "19-5-2", "max_value.value", _U_Q),
]


def _value(r: dict[str, Any], path: str) -> str | None:
    cur: Any = r
    for part in path.split("."):
        cur = cur.get(part) if isinstance(cur, dict) else None
    return str(cur) if cur not in (None, "") else None


@pytest.mark.parametrize(("ds", "path", "template"), REVERSE, ids=lambda x: str(x)[-20:])
def test_exact_reverse_lookup_returns_only_rows_printing_the_value(
    ds: str, path: str, template: str
) -> None:
    values = sorted({v for r in _records(ds) if (v := _value(r, path))})
    rng = random.Random(f"{SEED}:reverse:{ds}")
    for v in rng.sample(values, min(N, len(values))):
        printing = {_rid(r) for r in _records(ds) if _value(r, path) == v}
        step = Decimal(1).scaleb(Decimal(v).as_tuple().exponent)  # type: ignore[arg-type]
        for style_v in _styles(v):
            res = _retrieve(template.format(v=style_v))
            got = {h.record_id for h in res.structured}
            assert got <= printing, (res.query, got - printing)  # 3: never a nearby value
        for near in (Decimal(v) + step, Decimal(v) - step):
            res = _retrieve(template.format(v=str(near)))
            got = {h.record_id for h in res.structured}
            exact = {_rid(r) for r in _records(ds) if _value(r, path) == str(near)}
            assert got <= exact, (res.query, got - exact)  # 3: a near miss is not a match


# reverse lookups the source does not define (a value does not identify a row)
PROHIBITED = [
    "کدام کاربری در پیوست ۳ شدت مصرف ۱۲۰ دارد؟",
    "دمای آبگرم ۴۵ درجه در جدول ۱۹-۵-۵ مربوط به کدام نقطه مصرف است؟",
    "کدام کاربری در جدول ۱۹-۵-۶ نسبت کفایت نور روز ۵۰ درصد دارد؟",
    "کدام فضا در جدول ۱۹-۵-۷ یکنواختی ۰٫۴ دارد؟",
]


@pytest.mark.parametrize("query", PROHIBITED)
def test_prohibited_reverse_lookups_return_nothing(query: str) -> None:
    assert _retrieve(query).structured == []


@pytest.mark.parametrize("area", ["3000", "۳۰۰۰", "۳٬۰۰۰", "3,000", "2999", "3001"])
def test_thousands_separators_and_boundaries_select_the_printed_size_row(area: str) -> None:
    res = _retrieve(f"حداکثر شدت مصرف ساختمان مسکونی {area} متر مربعی در اقلیم 3B رده C")
    got = {h.record_id for h in res.structured}
    labels = {
        _store().occupancies[next(r for r in _store().app03 if r["id"] == g)["occupancy_id"]][
            "label"
        ]
        for g in got
    }
    big = area in ("3000", "۳۰۰۰", "۳٬۰۰۰", "3,000", "3001")
    assert len(got) <= 1
    for label in labels:
        assert ("≥" in label) == big, (area, label)


# ---------------------------------------------------------------- calculations

CALCULATIONS = [
    "مجموع حداقل مقاومت حرارتی دیوار خارجی مجاور خاک و کف مجاور خاک در اقلیم 3B چقدر است؟",
    "اختلاف سقف شدت مصرف رده C و رده D برای هتل در اقلیم 3A چقدر است؟",
    "نسبت ضریب هدایت حرارتی پشم سنگ به پشم شیشه چقدر است؟",
    "شدت مصرف مجاز دبستان را برای اقلیمی بین 2B و 3A درون‌یابی کن",
    "میانگین حداقل مقاومت حرارتی سقف مجاور فضای آزاد در همه اقلیم‌ها چقدر است؟",
    "سقف شدت مصرف دفتر پست در اقلیم 4A رده B چند درصد بیشتر از رده A است؟",
]


@pytest.mark.parametrize("query", CALCULATIONS)
def test_unsupported_calculations_return_zero_values(query: str) -> None:
    res = _retrieve(query)
    assert res.structured == [], query  # 6
    assert any(n.outcome == "unsupported_calculation" for n in res.no_match), query
