"""Property tests of the plan-coverage stage (I1-I9), generated from the curated tables.

Every generated query goes through :func:`_retrieve`, which enforces the global invariants:

* **P11** - no two grounded spans of the ledger carry incompatible roles (within or across
  plans); no executed direct lookup asks for a concept it states as a constraint; every
  returned row belongs to an executable plan, prints exactly the admitted output (or a declared
  alternative) and passed certification;
* **P12** (where a query has a base query) - metamorphic narrowing: harmless context leaves the
  rows unchanged; an unresolved meaning-bearing span in a structural slot only keeps or removes
  rows, never adds or changes one.

Mutations (P1-P10): head-family mutations, sibling-family substitutions, an unprinted requested
output, a foreign requested unit, a foreign unit on a stated number, concepts shared by several
tables, ratings in an explanatory clause, two and three coordinated places, mixed
lookup/prose ownership, a second table without an anchor of its own. Seeds are fixed.
"""

from __future__ import annotations

import json
import random
from functools import cache
from typing import Any

import pytest

from mabhas19.core.paths import CURATED_DIR
from mabhas19.indexing.chunking import build_chunks
from mabhas19.retrieval.certification import printed_levels
from mabhas19.retrieval.coverage import INCOMPATIBLE, split_coordinated
from mabhas19.retrieval.hybrid import Retriever
from mabhas19.retrieval.models import RetrievalResult
from mabhas19.retrieval.plan import EntityRef, StructuredQueryPlan, concept
from mabhas19.retrieval.structured import StructuredStore

APP02 = "m19-ed5-1404-app02.city_climate_classes"
APP03 = "m19-ed5-1404-app03.max_energy_use_intensity"
APP07 = "m19-ed5-1404-app07.material_thermal_properties"
MAIN = "m19-ed5-1404-main.t"
SEED = 20261026
N = 6


@cache
def _r() -> Retriever:
    path = CURATED_DIR / "text" / "m19-ed5-1404-main" / "blocks.jsonl"
    blocks = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
    return Retriever(build_chunks(blocks, "m19"), StructuredStore(CURATED_DIR))


def _store() -> StructuredStore:
    return _r().store


def _concept(name: str) -> str:
    return {"energy_rating": "rating"}.get(name, concept(name))


# ---------------------------------------------------------------- global invariants (P11)


def _retrieve(query: str) -> RetrievalResult:
    res = _r().retrieve(query)
    _p11(res)
    return res


def _p11(res: RetrievalResult) -> None:
    spans = [s for s in res.coverage.get("spans", []) if s["state"] == "grounded"]
    for i, a in enumerate(spans):
        for b in spans[i + 1 :]:
            overlap = a["span"][0] < b["span"][1] and b["span"][0] < a["span"][1]
            roles = frozenset({a["role"], b["role"]})
            assert not (overlap and roles in INCOMPATIBLE), (res.query, a, b)
    executable = {
        (a["target"]) for a in res.admissions if (a.get("coverage") or {}).get("executable")
    }
    hit_tables = {h.dataset for h in res.structured}
    assert hit_tables <= executable, (res.query, hit_tables, executable)
    for p in res.plans:
        if p["target"] not in hit_tables or p["operation"] != "direct":
            continue
        stated = {_concept(c["dimension"]) for c in p["constraints"] if c["state"] == "applied"}
        assert _concept(p["requested_field"] or "") not in stated, (res.query, p["target"])
    schemas = _store().schemas
    for h in res.structured:
        verdict: dict[str, Any] = next(
            (a["coverage"] for a in res.admissions if a["target"] == h.dataset), {}
        )
        admitted = verdict.get("output")
        if admitted:
            o = schemas[h.dataset].output(admitted)
            allowed = {admitted, *(o.alternatives if o else ())}
            assert h.output in allowed, (res.query, h.record_id, h.output, admitted)
    assert res.certification_failures == [], (res.query, res.certification_failures)


def _rows(res: RetrievalResult) -> set[str]:
    return {h.record_id for h in res.structured}


def _narrows(base: str, mutated: str, *, equal: bool = False) -> None:
    """P12: the mutated query keeps (or removes) the base query's rows, never adds one."""
    a, b = _rows(_retrieve(base)), _rows(_retrieve(mutated))
    if equal:
        assert a == b, (base, mutated, sorted(a ^ b))
    else:
        assert b <= a, (base, mutated, sorted(b - a))


# ---------------------------------------------------------------- generators


def _sample(rows: list[Any], tag: str, n: int = N) -> list[Any]:
    return random.Random(f"{SEED}:{tag}").sample(rows, min(n, len(rows)))


def _app03_query(r: dict[str, Any], extra: str = "") -> str:
    label = _store().occupancies[r["occupancy_id"]]["label"].split("/")[0].strip()
    return f"حداکثر شدت مصرف {label} در اقلیم {r['climate_class']} رده {r['energy_rating']}{extra}"


def _app07_query(r: dict[str, Any], head: str = "", extra: str = "") -> str:
    return f"ضریب هدایت حرارتی {head + ' ' if head else ''}{r['display_name']}{extra} چقدر است؟"


def _families() -> dict[str, list[str]]:
    """Appendix 7 printed head words -> the rows they head."""
    idx = _r().planner.identity[APP07]
    out: dict[str, list[str]] = {}
    for row, alts in idx.by_row.items():
        for alt in alts:
            for w in alt.heads:
                if len(w) > 2:
                    out.setdefault(w, []).append(row)
    return out


def _app07(rid: str) -> dict[str, Any]:
    return next(r for r in _store().app07 if r["id"] == rid)


def _shares_family(rid: str, rows: list[str]) -> bool:
    mine = set(printed_levels(_store(), APP07, _app07(rid)))
    return any(mine & set(printed_levels(_store(), APP07, _app07(x))) for x in rows)


# ---------------------------------------------------------------- P1, P2 identity family (I5)


def test_p1_head_from_another_family_never_selects_the_row() -> None:
    fams = _families()
    rows = _sample(list(_store().app07), "p1")
    for r in rows:
        others = sorted(h for h, hs in fams.items() if not _shares_family(r["id"], hs))
        for head in _sample(others, f"p1:{r['id']}", 2):
            res = _retrieve(_app07_query(r, head))
            for h in res.structured:
                if h.dataset == APP07:
                    assert _shares_family(h.record_id, fams[head]), (res.query, h.record_id)


def test_p1_head_from_the_rows_own_family_keeps_the_answer() -> None:
    """Recall control: the row's own printed head word changes nothing."""
    idx = _r().planner.identity[APP07]
    for r in _sample(list(_store().app07), "p1own"):
        base = _app07_query(r)
        if r["id"] not in _rows(_retrieve(base)):
            continue
        head = next((w for alt in idx.by_row[r["id"]] for w in alt.heads if len(w) > 2), "")
        assert r["id"] in _rows(_retrieve(_app07_query(r, head))), (base, head)


def test_p2_sibling_family_substitution_never_returns_the_other_family() -> None:
    """Rows of different categories sharing printed words: B's head with A's words never returns
    A alone."""
    idx = _r().planner.identity[APP07]
    fams = _families()
    rows = list(_store().app07)
    pairs = []
    for a in rows:
        wa = {w for alt in idx.by_row.get(a["id"], []) for w in alt.words}
        for b in rows:
            if a["category"]["text"] == b["category"]["text"]:
                continue
            wb = {w for alt in idx.by_row.get(b["id"], []) for w in alt.words}
            if len(wa & wb) >= 2:
                pairs.append((a, b))
    for a, b in _sample(pairs, "p2", 8):
        head = next((w for alt in idx.by_row[b["id"]] for w in alt.heads if w in fams), None)
        if head is None:
            continue
        res = _retrieve(_app07_query(a, head))
        for h in res.structured:
            if h.dataset == APP07:
                assert _shares_family(h.record_id, fams[head]), (res.query, h.record_id)


# ---------------------------------------------------------------- P3 unprinted output (I1-I3)

# test-only quantity phrases no structured table prints (never runtime vocabulary)
UNPRINTED = ("ظرفیت گرمایی ویژه", "ضریب جذب صوت", "مقاومت فشاری", "عمر مفید")


def test_p3_unprinted_output_is_never_answered_with_the_default_output() -> None:
    for r in _sample(list(_store().app07), "p3"):
        for q in UNPRINTED:
            res = _retrieve(f"{q} {r['display_name']} چقدر است؟")
            assert APP07 not in {h.dataset for h in res.structured}, res.query
    for r in _sample(list(_store().app03), "p3b"):
        label = _store().occupancies[r["occupancy_id"]]["label"].split("/")[0].strip()
        for q in UNPRINTED[:2]:
            res = _retrieve(
                f"{q} {label} در اقلیم {r['climate_class']} رده {r['energy_rating']} چقدر است؟"
            )
            assert APP03 not in {h.dataset for h in res.structured}, res.query


def test_p3_a_stated_constraint_never_names_the_output() -> None:
    """M102 form: the condition's own word («چگالی اسمی») does not make density the answer."""
    for r in [
        x for x in _store().app07 if "چگالی اسمی" in str((x.get("condition") or {}).get("text"))
    ][:4]:
        cond = r["condition"]["text"].split(":")[-1].strip()
        res = _retrieve(f"ظرفیت گرمایی ویژه {r['display_name']} با چگالی اسمی {cond} چقدر است؟")
        assert res.structured == [], (res.query, _rows(res))


# ---------------------------------------------------------------- P4, P5 units (I4, I9)

FOREIGN_UNITS = ("kBtu/ft²·yr", "Btu/h·ft·°F", "lb/ft3", "kcal/m2")


def test_p4_foreign_requested_unit_is_refused_as_a_conversion() -> None:
    for r in _sample(list(_store().app03), "p4"):
        for u in FOREIGN_UNITS:
            res = _retrieve(_app03_query(r, f" را بر حسب {u} اعلام کنید"))
            assert res.structured == [], res.query
            assert any(
                n.dataset == APP03 and n.outcome == "unsupported_calculation" for n in res.no_match
            ), (res.query, [(n.dataset, n.outcome) for n in res.no_match])


def test_p4_the_printed_unit_changes_nothing() -> None:
    for r in _sample(list(_store().app03), "p4ok"):
        _narrows(_app03_query(r), _app03_query(r, " را بر حسب kWh/m2.yr اعلام کنید"), equal=True)
    for r in _sample(list(_store().app07), "p4okb"):
        _narrows(
            _app07_query(r),
            f"ضریب هدایت حرارتی {r['display_name']} بر حسب W/m.K چقدر است؟",
            equal=True,
        )
    for r in _sample(list(_store().app07), "p4unit"):
        res = _retrieve(_app07_query(r))
        assert all(h.unit == "W/m.K" for h in res.structured if h.dataset == APP07), res.query


def test_p5_foreign_unit_on_a_stated_number_is_never_read_as_the_printed_unit() -> None:
    rows = [r for r in _store().app07 if r["density"]["kind"] == "range"]
    for r in _sample(rows, "p5"):
        lo, hi = float(r["density"]["min"]), float(r["density"]["max"])
        mid = int((lo + hi) / 2)
        name = r["display_name"]
        base = f"ضریب هدایت حرارتی {name} با چگالی {mid} کیلوگرم بر متر مکعب چقدر است؟"
        res = _retrieve(f"ضریب هدایت حرارتی {name} با چگالی {mid} lb/ft3 چقدر است؟")
        assert res.structured == [], (res.query, _rows(res))
        _narrows(base, f"ضریب هدایت حرارتی {name} با چگالی {mid} kg/m3 چقدر است؟", equal=True)


# ---------------------------------------------------------------- P6, P10 table ownership (I7)


def test_p6_shared_concepts_never_admit_another_table_beside_the_named_entity() -> None:
    for r in _sample(list(_store().app03), "p6", 10):
        res = _retrieve(_app03_query(r) + " حداکثر چقدر مصرف مجازه؟")
        assert {h.dataset for h in res.structured} <= {APP03}, (res.query, _rows(res))


def test_p10_a_stated_value_alone_admits_no_table_whose_output_it_names() -> None:
    for rating in "ABCD":
        for cc in ("1B", "3A", "4B"):
            res = _retrieve(f"رده {rating} در اقلیم {cc} حداکثر چقدر مصرف مجازه؟")
            assert MAIN + "19-3-3" not in {h.dataset for h in res.structured}, res.query
            res = _retrieve(f"ساختمانی در اقلیم {cc} برای رده {rating} چه شرایطی دارد؟")
            assert not {APP02, MAIN + "19-3-3"} & {h.dataset for h in res.structured}, res.query


# ---------------------------------------------------------------- P7 clause precedence (I6)


def test_p7_ratings_of_an_explanatory_clause_never_filter_the_lookup() -> None:
    for r in _sample(list(_store().app03), "p7", 8):
        label = _store().occupancies[r["occupancy_id"]]["label"].split("/")[0].strip()
        others = [x for x in "ABCD" if x != r["energy_rating"]]
        q = (
            f"جدول‌های رده‌های {others[0]} و {others[1]} کجا اومده؟ برای {label} در اقلیم "
            f"{r['climate_class']} سقف رده {r['energy_rating']} چنده؟"
        )
        res = _retrieve(q)
        for h in res.structured:
            if h.dataset == APP03:
                assert h.values["energy_rating"] == r["energy_rating"], (q, h.record_id)


def test_p7_several_explanatory_ratings_and_none_in_the_lookup_fail_closed() -> None:
    for r in _sample(list(_store().app03), "p7b"):
        label = _store().occupancies[r["occupancy_id"]]["label"].split("/")[0].strip()
        q = (
            f"فرق رده‌های A و B و C چیه؟ برای {label} در اقلیم {r['climate_class']} سقف "
            "مصرف مجاز چنده؟"
        )
        res = _retrieve(q)
        assert APP03 not in {h.dataset for h in res.structured}, (q, _rows(res))


# ---------------------------------------------------------------- P8 coordinated entities (I8)


def _cities(k: int, tag: str) -> list[dict[str, Any]]:
    """Unqualified printed city names (a station-qualified name such as «تهران (امام خمینی)»
    resolves to every station of its city: one member, several rows, by the place resolver)."""
    rows = [r for r in _store().app02 if "(" not in r["city"]["text"]]
    return _sample(rows, tag, k)


@pytest.mark.parametrize("k", [2, 3])
def test_p8_coordinated_places_map_one_to_one_and_ignore_order(k: int) -> None:
    for i in range(4):
        rows = _cities(k, f"p8:{k}:{i}")
        names = [r["city"]["text"] for r in rows]
        q1 = f"رده اقلیمی {' و '.join(names)} در پیوست ۲ چیست؟"
        q2 = f"رده اقلیمی {' و '.join(reversed(names))} در پیوست ۲ چیست؟"
        a, b = _retrieve(q1), _retrieve(q2)
        assert _rows(a) == _rows(b), (q1, q2)
        assert _rows(a) <= {r["id"] for r in rows}, (q1, _rows(a))
        for h in a.structured:  # each row answers exactly one member
            city = h.values["city"]["text"]
            assert sum(city == n for n in names) == 1, (q1, h.record_id, city)


def test_p8_a_single_printed_place_is_never_split() -> None:
    plan = StructuredQueryPlan(APP02)
    plan.entities = [EntityRef("place", "بویین و میاندشت", "resolved", ["بویین و میاندشت"])]
    assert split_coordinated(plan, _store().schemas[APP02]) == [(plan, None)]


def test_p8_a_conjunction_inside_a_printed_label_is_not_coordination() -> None:
    res = _retrieve("حداکثر شدت مصرف بانک و موسسه مالی در اقلیم 3A رده C")
    occupancies = {_r()._record(h)["occupancy_id"] for h in res.structured if h.dataset == APP03}
    assert len(occupancies) <= 1, (res.query, occupancies)


# ---------------------------------------------------------------- P9 mixed ownership


def test_p9_an_entity_of_the_explanatory_clause_never_gets_rows() -> None:
    for i in range(4):
        a, b = _cities(2, f"p9:{i}")
        q = (
            f"شهر {a['city']['text']} چه ویژگی‌های اقلیمی دارد که در مبحث آمده؟ "
            f"رده اقلیمی {b['city']['text']} در پیوست ۲ چیست؟"
        )
        res = _retrieve(q)
        assert a["id"] not in _rows(res) or b["id"] in _rows(res), (q, _rows(res))


# ---------------------------------------------------------------- P12 metamorphic narrowing


def test_p12_harmless_context_changes_nothing() -> None:
    for r in _sample(list(_store().app03), "p12a"):
        base = _app03_query(r)
        _narrows(base, f"لطفاً بفرمایید {base} ممنون.", equal=True)
    for r in _sample(list(_store().app07), "p12b"):
        base = _app07_query(r)
        _narrows(base, f"لطفاً بفرمایید {base}", equal=True)


def test_p12_an_unresolved_word_in_a_structural_slot_only_narrows() -> None:
    for r in _sample(list(_store().app07), "p12c", 10):
        base = _app07_query(r)
        _narrows(base, _app07_query(r, "کوانتومی"))  # head slot
        _narrows(base, _app07_query(r, extra=" کوانتومی"))  # modifier slot
    for r in _sample(list(_store().app03), "p12d", 10):
        label = _store().occupancies[r["occupancy_id"]]["label"].split("/")[0].strip()
        base = _app03_query(r)
        mutated = base.replace(label, f"{label} کوانتومی", 1)
        _narrows(base, mutated)
