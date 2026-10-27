"""Property tests of the plan-coverage stage (I1-I10), generated from the curated tables.

Every generated query goes through :func:`_retrieve`, which enforces the global invariants:

* **P11** - no two grounded spans of the ledger carry incompatible roles (within or across
  plans); no executed direct lookup asks for a concept it states as a constraint; every
  returned row belongs to an executable plan, prints exactly the admitted output (or a declared
  alternative) and passed certification;
* **P12** (where a query has a base query) - metamorphic narrowing: harmless context leaves the
  rows unchanged; an unresolved meaning-bearing span in a structural slot only keeps or removes
  rows, never adds or changes one.

* **P13** (Safety Closure 2) - adding a conflicting or unsupported semantic component to a query
  (an unprinted quantity, a foreign unit, a value outside the table's scope, another family's
  printed qualifier, a rejected alternative, another coordinated member) keeps the base rows or
  fails closed: it never returns a row the base query (or, for coordination, one of its members)
  would not return.

Mutations (P1-P10): head-family mutations, sibling-family substitutions, an unprinted requested
output, a foreign requested unit, a foreign unit on a stated number, concepts shared by several
tables, ratings in an explanatory clause, two and three coordinated places, mixed
lookup/prose ownership, a second table without an anchor of its own. Safety Closure 2 (P14-P19):
corrections, table scope, unit expressions, complete requested quantities, identity-span
accounting, generic coordination. Seeds are fixed.
"""

from __future__ import annotations

import itertools
import json
import random
from functools import cache
from typing import Any

import pytest

from mabhas19.core.paths import CURATED_DIR
from mabhas19.indexing.chunking import build_chunks
from mabhas19.retrieval import table_scope, units
from mabhas19.retrieval.certification import printed_levels
from mabhas19.retrieval.coverage import INCOMPATIBLE, coordinated_members, split_coordinated
from mabhas19.retrieval.hybrid import Retriever
from mabhas19.retrieval.identity import row_label
from mabhas19.retrieval.models import RetrievalResult
from mabhas19.retrieval.plan import EntityRef, StructuredQueryPlan, concept
from mabhas19.retrieval.schema import TableSchema
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

# requested unit -> the typed refusal for an energy-use intensity (kWh/m2.yr)
FOREIGN_UNITS = {
    "kBtu/ft²·yr": "unsupported_calculation",  # the same quantity in other units
    "lb/ft3": "unsupported_output",  # a density: another physical quantity
    "kcal/m2": "unsupported_calculation",
    "Btu/h·ft·°F": "unsupported_output",  # a conductivity: another physical quantity
}


def test_p4_foreign_requested_unit_is_refused_as_a_conversion() -> None:
    for r in _sample(list(_store().app03), "p4"):
        for u, code in FOREIGN_UNITS.items():
            res = _retrieve(_app03_query(r, f" را بر حسب {u} اعلام کنید"))
            assert res.structured == [], res.query
            assert any(n.dataset == APP03 and n.outcome == code for n in res.no_match), (
                res.query,
                [(n.dataset, n.outcome) for n in res.no_match],
            )


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


# ================================================================ Safety Closure 2 (P13-P19)


def _p13(base: str, mutated: str) -> RetrievalResult:
    """P13: the mutated query keeps (or removes) the base rows; it never redirects to another
    regulatory row."""
    a = _rows(_retrieve(base))
    res = _retrieve(mutated)
    assert _rows(res) <= a, (base, mutated, sorted(_rows(res) - a))
    return res


def _label(r: dict[str, Any]) -> str:
    return str(_store().occupancies[r["occupancy_id"]]["label"].split("/")[0].strip())


def _outcomes(res: RetrievalResult, ds: str) -> set[str | None]:
    return {n.outcome for n in res.no_match if n.dataset == ds}


# ---------------------------------------------------------------- P14 corrections (I6)

CORRECTIONS = (
    "حداکثر شدت مصرف {label} در اقلیم {a} نه، در اقلیم {b} برای رده {rt} چقدر است؟",
    "حداکثر شدت مصرف {label}، نه اقلیم {a}، اقلیم {b}، رده {rt} چقدر است؟",
    "حداکثر شدت مصرف {label} در اقلیم {a} نه بلکه {b} برای رده {rt} چقدر است؟",
    "اقلیم ساختمان {a} نیست، {b} است. حداکثر شدت مصرف {label} رده {rt} چقدر است؟",
)


def _twin_climate(r: dict[str, Any]) -> str | None:
    """Another printed climate class of the same occupancy and rating."""
    return next(
        (
            x["climate_class"]
            for x in _store().app03
            if x["occupancy_id"] == r["occupancy_id"]
            and x["energy_rating"] == r["energy_rating"]
            and x["climate_class"] != r["climate_class"]
        ),
        None,
    )


def test_p14_a_rejected_value_never_leaks_and_the_replacement_alone_is_active() -> None:
    for r in _sample(list(_store().app03), "p14", 8):
        wrong = _twin_climate(r)
        if wrong is None:
            continue
        base = _app03_query(r)
        for pattern in CORRECTIONS:
            q = pattern.format(
                label=_label(r), a=wrong, b=r["climate_class"], rt=r["energy_rating"]
            )
            res = _p13(base, q)
            got = [h for h in res.structured if h.dataset == APP03]
            assert all(h.values["climate_class"] == r["climate_class"] for h in got), (q, got)


def test_p14_without_a_correction_both_values_stay_stated() -> None:
    """Positive control: two climates without a correction marker are still both stated (the
    lookup is not narrowed by the correction rule)."""
    from mabhas19.retrieval.polarity import corrections  # noqa: PLC0415

    for r in _sample(list(_store().app03), "p14b"):
        wrong = _twin_climate(r)
        if wrong is None:
            continue
        cc, rt = r["climate_class"], r["energy_rating"]
        q = f"حداکثر شدت مصرف {_label(r)} در اقلیم {wrong} و {cc} رده {rt}"
        assert corrections(q.lower()) == [], q
        res = _retrieve(q)
        assert not any("rejected" in str(n) for n in res.notes), q


# ---------------------------------------------------------------- P15 table scope (I10)


def _block_texts() -> dict[str, str]:
    path = CURATED_DIR / "text" / "m19-ed5-1404-main" / "blocks.jsonl"
    return {
        b["block_id"]: str(b.get("text") or "")
        for b in (json.loads(x) for x in path.read_text(encoding="utf-8").splitlines())
    }


def test_p15_every_table_declares_the_scope_of_every_regulatory_dimension() -> None:
    blocks = _block_texts()
    for ds, schema in _store().schemas.items():
        assert table_scope.undeclared(schema) == [], ds
        for decl in schema.scope:
            assert schema.dim(decl.dimension) is None, (ds, decl.dimension)
            assert decl.kind in ("universal", "fixed"), (ds, decl)
            if decl.kind == "fixed":
                assert decl.values, (ds, decl)
                assert decl.provenance, (ds, decl)
                for b in decl.provenance:
                    assert b in blocks, (ds, b)
                # the establishing source states the rating it fixes
                assert any("رده" in blocks[b] and "D" in blocks[b] for b in decl.provenance)


def test_p15_printed_output_headers_are_printed_in_their_blocks() -> None:
    from mabhas19.text.normalize import normalize_for_search  # noqa: PLC0415

    blocks = {k: normalize_for_search(v).replace(" ", "") for k, v in _block_texts().items()}
    for ds, schema in _store().schemas.items():
        for o in schema.outputs:
            for phrase in o.header:
                want = normalize_for_search(phrase).replace(" ", "")
                assert any(want in blocks[b] for b in o.header_blocks), (ds, o.name, phrase)


def test_p15_an_undeclared_dimension_fails_closed() -> None:
    bare = TableSchema("t", "label", "t", [], [], [])
    v = table_scope.verdict(bare, "rating", ["B"])
    assert v.state == table_scope.UNDECLARED
    assert v.blocks


def _r_rows() -> list[dict[str, Any]]:
    t = _store().main[MAIN + "19-5-1"]
    return [r for r in t.records if r.get("min_r_value", {}).get("value")]


def _r_query(r: dict[str, Any], prefix: str = "") -> str:
    label = row_label(_store(), MAIN + "19-5-1", r)
    cc = r["climate_group"]["applies_to"][0]
    return f"{prefix}حداقل مقاومت حرارتی {label} در اقلیم {cc} چقدر است؟"


def test_p15_a_value_outside_a_fixed_scope_is_an_invalid_constraint() -> None:
    for r in _sample(_r_rows(), "p15"):
        base = _r_query(r)
        for rating in "ABC":
            res = _p13(base, _r_query(r, f"برای کسب رده بازدهی {rating}، "))
            assert MAIN + "19-5-1" not in {h.dataset for h in res.structured}, res.query
            assert "invalid_constraint" in _outcomes(res, MAIN + "19-5-1"), res.query


def test_p15_the_fixed_value_and_universal_dimensions_change_nothing() -> None:
    for r in _sample(_r_rows(), "p15d"):
        _narrows(_r_query(r), _r_query(r, "برای کسب رده بازدهی D، "), equal=True)
    for r in _sample(list(_store().app07), "p15u"):
        base = _app07_query(r)
        _narrows(base, base.replace(" چقدر است؟", " برای ساختمان رده B چقدر است؟"), equal=True)


# ---------------------------------------------------------------- P16 unit expressions (I4)

EUI = units.parse("kWh/m2.yr")


@pytest.mark.parametrize(
    "text",
    [
        "kWh/m2.yr",
        "کیلووات ساعت بر متر مربع در سال",
        "kWh بر متر مربع در سال",
        "کیلووات ساعت به ازای هر متر مربع در سال",
        "kwh/m²·yr",
    ],
)
def test_p16_latin_and_persian_forms_are_one_unit(text: str) -> None:
    u = units.parse(text)
    assert u is not None, text
    assert EUI is not None
    assert units.same(u, EUI), (text, u.describe())


@pytest.mark.parametrize(
    ("text", "verdict"),
    [
        ("kWh", units.GROUND),
        ("kWh/m2", units.GROUND),
        ("MJ/m2.yr", units.CONVERSION),
        ("kBtu/ft2.yr", units.CONVERSION),
        ("kWh/m2.month", units.CONVERSION),  # another temporal basis
        ("کیلووات ساعت بر متر مربع در ماه", units.CONVERSION),
        ("متر مکعب گاز در سال برای هر متر مربع", units.CONVERSION),  # a fuel volume is energy
        ("مگاپاسکال", units.OTHER_QUANTITY),
        ("kW", units.OTHER_QUANTITY),
        ("W/m.K", units.OTHER_QUANTITY),
        ("تومان", units.OTHER_QUANTITY),
    ],
)
def test_p16_unit_verdicts_never_convert(text: str, verdict: str) -> None:
    u = units.parse(text)
    assert u is not None, text
    assert EUI is not None
    assert units.verdict(u, EUI) == verdict, (text, u.describe())


def test_p16_units_are_found_without_a_cue_and_attached_units_are_the_numbers() -> None:
    found = units.find("سقف رده c را به kwh بر متر مربع در ماه بگویید")
    assert [e.unit.describe() for e in found] == ["kwh/m2·month"]
    assert units.find("در سال اول برای گروه ج") == []  # a year, not a unit
    assert units.find("دمای آب گرم چند درجه") == []  # «گرم», «درجه» alone are no request
    attached = units.find("چگالی ۵۰ پوند بر فوت مکعب")
    assert attached
    assert attached[0].attached


def test_p16_a_unit_request_anywhere_is_refused_by_quantity() -> None:
    for r in _sample(list(_store().app03), "p16"):
        base = _app03_query(r)
        res = _p13(base, base + " را به kWh بر متر مربع در ماه بگویید")
        assert "unsupported_calculation" in _outcomes(res, APP03), res.query
        res = _p13(base, "بر حسب MJ/m2 " + base)
        assert "unsupported_calculation" in _outcomes(res, APP03), res.query
        _narrows(base, base + " (kWh/m2.yr)", equal=True)
    for r in _sample(list(_store().app07), "p16b"):
        base = _app07_query(r)
        res = _p13(base, f"{r['display_name']} چند مگاپاسکال مقاومت دارد؟")
        assert APP07 not in {h.dataset for h in res.structured}, res.query


def test_p16_a_foreign_unit_on_a_stated_number_is_refused() -> None:
    rows = [r for r in _store().app07 if r["density"]["kind"] == "range"]
    for r in _sample(rows, "p16c"):
        mid = int((float(r["density"]["min"]) + float(r["density"]["max"])) / 2)
        base = f"ضریب هدایت حرارتی {r['display_name']} با چگالی {mid} کیلوگرم بر متر مکعب چقدر است؟"
        res = _p13(base, base.replace("کیلوگرم بر متر مکعب", "پوند بر فوت مکعب"))
        assert APP07 not in {h.dataset for h in res.structured}, res.query


# ---------------------------------------------------------------- P17 requested quantity (I3)


def test_p17_an_unprinted_quantity_after_the_entity_is_refused() -> None:
    for r in _sample(list(_store().app07), "p17"):
        base = _app07_query(r)
        for q in UNPRINTED:
            for form in (
                f"{r['display_name']}، {q}اش چقدر است؟",
                f"{r['display_name']} چه {q}ی دارد؟",
            ):
                res = _p13(base, form)
                assert APP07 not in {h.dataset for h in res.structured}, res.query


def test_p17_an_output_name_with_a_meaning_changing_noun_is_refused() -> None:
    for r in _sample(list(_store().app03), "p17b"):
        base = _app03_query(r)
        for noun in ("آب", "برق"):
            q = base.replace("حداکثر شدت مصرف", f"حداکثر مصرف {noun}", 1)
            res = _p13(base, q)
            assert APP03 not in {h.dataset for h in res.structured}, res.query
            assert "unsupported_output" in _outcomes(res, APP03), res.query


def test_p17_printed_header_words_change_nothing() -> None:
    for r in _sample(list(_store().app03), "p17c"):
        base = _app03_query(r)
        q = base.replace("حداکثر شدت مصرف", "حداکثر شدت مصرف انرژی مجاز سالانه", 1)
        _narrows(base, q, equal=True)


# ---------------------------------------------------------------- P18 identity accounting (I5)


def _qualified_pairs() -> list[tuple[str, str, str, str]]:
    """(table, a row printing a negative-form phrase, that phrase, a row of another head that
    does not print it) for the label tables."""
    out = []
    planner = _r().planner
    for ds in (MAIN + "19-5-3", MAIN + "19-5-4"):
        idx = planner.identity[ds]
        for row, printed in idx.printed.items():
            for w in printed:
                if not w.startswith(("بدون", "غیر")) or len(w) < 7:
                    continue
                phrase = ("بدون " + w[4:]) if w.startswith("بدون") else ("غیر " + w[3:])
                for other, op in idx.printed.items():
                    if other != row and op and printed and op[0] != printed[0] and w not in op:
                        out.append((ds, row, phrase, other))
    return out


def _label_of(ds: str, rid: str) -> str:
    rec = next(r for r in _store().main[ds].records if r["record_id"] == rid)
    return row_label(_store(), ds, rec)


def test_p18_a_sibling_familys_printed_qualifier_never_selects_a_row() -> None:
    for ds, _row, phrase, other in _sample(_qualified_pairs(), "p18", 8):
        base = f"{_label_of(ds, other)} حداقل چه بازدهی باید داشته باشد؟"
        res = _p13(base, f"{_label_of(ds, other)} {phrase} حداقل چه بازدهی باید داشته باشد؟")
        assert other not in _rows(res), (res.query, _rows(res))


def test_p18_the_rows_own_qualifier_and_a_negated_sibling_keep_the_answer() -> None:
    for ds, row, _phrase, other in _sample(_qualified_pairs(), "p18b", 6):
        own = f"{_label_of(ds, row)} حداقل چه بازدهی باید داشته باشد؟"
        if row not in _rows(_retrieve(own)):
            continue
        sibling_head = _label_of(ds, other).split()[0]
        _p13(own, own.replace(" حداقل", f" (نه {sibling_head}) حداقل", 1))


def test_p18_one_subject_with_two_printed_identities_is_ambiguous() -> None:
    rows = _sample([r for r in _store().app03 if r["energy_rating"] == "C"], "p18c", 6)
    for a, b in itertools.pairwise(rows):
        if a["occupancy_id"] == b["occupancy_id"] or a["climate_class"] != b["climate_class"]:
            continue
        q = (
            f"یک ساختمان داریم که هم {_label(a)} است هم {_label(b)}، در اقلیم {a['climate_class']}"
            " برای رده C حداکثر شدت مصرف چقدر است؟"
        )
        res = _retrieve(q)
        assert APP03 not in {h.dataset for h in res.structured}, (q, _rows(res))


# ---------------------------------------------------------------- P19 coordination (I8)


def _app03_pairs(tag: str, k: int = 2) -> list[list[dict[str, Any]]]:
    """Groups of k rows of distinct, single-word-free occupancies sharing climate and rating."""
    rows = [r for r in _store().app03 if " و " not in _label(r)]
    out = []
    for r in _sample(rows, tag, 12):
        mates = [
            x
            for x in rows
            if x["climate_class"] == r["climate_class"]
            and x["energy_rating"] == r["energy_rating"]
            and x["occupancy_id"] != r["occupancy_id"]
        ]
        group = [r, *_sample(mates, f"{tag}:{r['id']}", k - 1)]
        if len({x["occupancy_id"] for x in group}) == k:
            out.append(group)
    return out[:4]


@pytest.mark.parametrize("k", [2, 3])
def test_p19_coordinated_occupancies_map_one_to_one_and_ignore_order(k: int) -> None:
    for group in _app03_pairs(f"p19:{k}", k):
        cc, rt = group[0]["climate_class"], group[0]["energy_rating"]
        names = [_label(x) for x in group]
        q1 = f"حداکثر شدت مصرف {' و '.join(names)} در اقلیم {cc} رده {rt} چقدر است؟"
        q2 = f"حداکثر شدت مصرف {' و '.join(reversed(names))} در اقلیم {cc} رده {rt} چقدر است؟"
        a, b = _retrieve(q1), _retrieve(q2)
        assert _rows(a) == _rows(b), (q1, q2)
        singles = set().union(*(_rows(_retrieve(_app03_query(x))) for x in group))
        assert _rows(a) <= singles, (q1, sorted(_rows(a) - singles))  # P13
        occ = [_r()._record(h)["occupancy_id"] for h in a.structured if h.dataset == APP03]
        assert len(occ) == len(set(occ)), (q1, occ)  # no member answered twice


def test_p19_member_specific_values_stay_with_their_member() -> None:
    for group in _app03_pairs("p19b"):
        a, b = group
        twin = next(
            (
                x
                for x in _store().app03
                if x["occupancy_id"] == b["occupancy_id"]
                and x["climate_class"] != a["climate_class"]
                and x["energy_rating"] != a["energy_rating"]
            ),
            None,
        )
        if twin is None:
            continue
        q = (
            f"حداکثر شدت مصرف {_label(a)} در اقلیم {a['climate_class']} برای رده "
            f"{a['energy_rating']} و {_label(b)} در اقلیم {twin['climate_class']} برای رده "
            f"{twin['energy_rating']} چقدر است؟"
        )
        assert _rows(_retrieve(q)) <= {a["id"], twin["id"]}, q


def test_p19_coordinated_materials_never_answer_outside_their_members() -> None:
    rows = _sample(list(_store().app07), "p19c", 8)
    for a, b in zip(rows[::2], rows[1::2], strict=True):
        q = f"ضریب هدایت حرارتی {a['display_name']} و {b['display_name']} چقدر است؟"
        singles = _rows(_retrieve(_app07_query(a))) | _rows(_retrieve(_app07_query(b)))
        assert _rows(_retrieve(q)) <= singles, q


def test_p19_a_conjunction_inside_a_printed_name_is_never_cut() -> None:
    planner = _r().planner
    ds = MAIN + "19-2-1"
    full = "در سال دوم برای گروه ب، «سرمایش و گرمایش ۴ لوله» چه علامتی دارد"
    rows = list(planner.identify(ds, full, full)[0])
    members = coordinated_members(planner, ds, full, rows)
    assert len(members) < 2, members
    occ = "حداکثر شدت مصرف بانک و موسسه مالی در اقلیم 3A رده C"
    a03 = list(planner.identify(APP03, occ, occ)[0])
    assert len(coordinated_members(planner, APP03, occ, a03)) < 2


def test_p19_coordinated_ratings_stay_one_or_valued_plan() -> None:
    for r in _sample(list(_store().app03), "p19d"):
        q = f"حداکثر شدت مصرف {_label(r)} در اقلیم {r['climate_class']} برای رده B و C"
        res = _retrieve(q)
        assert len([p for p in res.plans if p["target"] == APP03]) == 1, q
        assert {h.values["energy_rating"] for h in res.structured if h.dataset == APP03} <= {
            "B",
            "C",
        }, q


def test_p19_an_unknown_member_never_redirects_the_known_one() -> None:
    for r in _sample(list(_store().app03), "p19e"):
        base = _app03_query(r)
        q = base.replace(_label(r), f"{_label(r)} و کوانتوم‌سرای مریخی", 1)
        _p13(base, q)
