"""Source-structure identity (M1): properties generated from the curated source inventory.

Every printed parenthetical, list, container/linking head and short-stem plural of the
structured tables is enumerated (through the identity snapshot) and each derived identity is
checked both ways: a printed name names its row - and only rows printing it - while a
description, classifier, number, unit, coordinated phrase or homograph never names anything.
No test encodes benchmark wording.
"""

from __future__ import annotations

import json
import re
from functools import cache
from typing import Any

import pytest

from mabhas19.core.paths import CURATED_DIR
from mabhas19.indexing.chunking import build_chunks
from mabhas19.retrieval import identity_snapshot
from mabhas19.retrieval.entities import keys
from mabhas19.retrieval.hybrid import Retriever
from mabhas19.retrieval.identity import (
    IdentityIndex,
    _short_plural,
    coordinated,
    head_structure,
    list_names,
    parenthetical_class,
    parenthetical_names,
    query_forms,
    query_tokens,
)
from mabhas19.retrieval.polarity import parse
from mabhas19.retrieval.structured import StructuredStore
from mabhas19.text.normalize import normalize_for_search as norm

APP03 = "m19-ed5-1404-app03.max_energy_use_intensity"
APP07 = "m19-ed5-1404-app07.material_thermal_properties"
MAIN = "m19-ed5-1404-main.t"
NAME_CLASSES = ("acronym", "noun", "term")


@cache
def _r() -> Retriever:
    path = CURATED_DIR / "text" / "m19-ed5-1404-main" / "blocks.jsonl"
    blocks = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
    return Retriever(build_chunks(blocks, "m19"), StructuredStore(CURATED_DIR))


@cache
def _snapshot() -> tuple[dict[str, Any], ...]:
    return tuple(identity_snapshot.build(_r().store))


def _ds(short: str) -> str:
    return next(d for d in _r().planner.identity if d.endswith(short))


def _index(short: str) -> IdentityIndex:
    return _r().planner.identity[_ds(short)]


def _exact(short: str, text: str) -> list[str]:
    """Rows a query names exactly (as the planner chooses them: pruned)."""
    q = norm(text)
    return _r().planner.identify(_ds(short), q, q)[0]


def _prints(short: str, row: str, words: list[str]) -> bool:
    printed = _index(short).printed.get(row, [])
    return all(any(keys(w) & keys(x) for x in printed) for w in words)


def _derived(rule: str) -> list[tuple[str, str, list[str]]]:
    """(dataset, identity row, derived level words) for every alternative of a rule."""
    out = []
    for e in _snapshot():
        for f in e["forms"]:
            tag = "|".join([f["rule"], *f["transforms"]])
            if rule in tag:
                level: list[str] = next((lv for lv in reversed(f["levels"]) if lv), [])
                if rule.startswith("parenthetical"):
                    level = _name_level(f["levels"], e)
                out.append((e["dataset"], e["identity_row"], level))
    return out


def _name_level(levels: list[list[str]], e: dict[str, Any]) -> list[str]:
    """The level a parenthetical name replaced (the one built from a parenthetical)."""
    names = {w for p in e["parentheticals"] for w in query_tokens(p["text"])}
    return next((lv for lv in levels if lv and set(lv) & names), [])


def _ids(case: tuple[str, str, list[str]]) -> str:
    return f"{case[1]}:{'_'.join(case[2])}"


# ---------------------------------------------------------------- the snapshot fixture


def test_identity_snapshot_matches_the_reviewed_fixture() -> None:
    path = CURATED_DIR.parent / "tests" / "fixtures" / "identity_snapshot.json"
    fresh = identity_snapshot.dumps(list(_snapshot()))
    assert path.read_text(encoding="utf-8") == fresh, (
        "the derived identity changed: review the source-wide diff and regenerate with "
        "`python -m mabhas19.retrieval.identity_snapshot tests/fixtures/identity_snapshot.json`"
    )


# ---------------------------------------------------------------- parenthetical names


@pytest.mark.parametrize("case", _derived("parenthetical"), ids=_ids)
def test_parenthetical_name_names_its_row_and_only_rows_printing_it(
    case: tuple[str, str, list[str]],
) -> None:
    ds, row, level = case
    exact = _exact(ds, " ".join(level))
    assert row in exact
    assert all(_prints(ds, r, level) for r in exact)


def _parentheticals() -> list[tuple[str, str, str, str, str]]:
    out = []
    for e in _snapshot():
        for p in e["parentheticals"]:
            out.append((e["dataset"], e["identity_row"], p["field"], p["text"], p["class"]))
    return out


@pytest.mark.parametrize("p", _parentheticals(), ids=lambda p: f"{p[1]}:{p[3]}")
def test_only_name_classes_in_name_fields_derive_an_alternative(
    p: tuple[str, str, str, str, str],
) -> None:
    ds, row, field, inner, cls = p
    entry = next(e for e in _snapshot() if e["identity_row"] == row and e["dataset"] == ds)
    text = entry["printed"][field]
    names = [n for c, i, n in parenthetical_names(text) if i == inner]
    words = [query_tokens(n) for n in names] or [query_tokens(inner)]
    derived = [
        f["levels"]
        for f in entry["forms"]
        if f["rule"].startswith("parenthetical")
        and any(lv and lv[0] == w[0] and set(lv) <= set(w) for lv in f["levels"] for w in words)
    ]
    eligible = (
        cls in NAME_CLASSES
        and "specification" not in field
        and field != "condition"
        and not coordinated(text)
    )
    assert bool(derived) == eligible, (inner, cls, field)


@pytest.mark.parametrize(
    "p",
    [p for p in _parentheticals() if any(c.isdigit() for c in p[3]) or "/" in p[3]],
    ids=lambda p: f"{p[1]}:{p[3]}",
)
def test_numeric_and_unit_parentheticals_are_never_names(
    p: tuple[str, str, str, str, str],
) -> None:
    assert parenthetical_class(p[3]) in ("numeric", "unit")


@pytest.mark.parametrize(
    "p",
    [p for p in _parentheticals() if p[4] in ("adjective", "description", "list", "short_latin")],
    ids=lambda p: f"{p[1]}:{p[3]}",
)
def test_descriptive_parenthetical_alone_names_nothing_through_it(
    p: tuple[str, str, str, str, str],
) -> None:
    ds, row, _field, inner, _cls = p
    words = query_tokens(inner)
    for alt in _index(ds).by_row.get(row, []):
        if alt.rule != "printed":
            assert not any(lv and set(lv) <= set(words) for lv in alt.levels)


@pytest.mark.parametrize(
    ("inner", "cls"),
    [
        ("PTFE", "acronym"),
        ("HD", "short_latin"),
        ("اصطلاحاً یونولیت یا پلاستوفوم", "term"),
        ("نئوپان", "noun"),
        ("صنعتی", "adjective"),
        ("برگ ریز", "description"),
        ("آلتوگلاس، پلکسی گلاس", "list"),
        ("از ۳ تا ۶ میلیمتر", "numeric"),
        ("W/m2.k", "numeric"),
    ],
)
def test_parenthetical_classes(inner: str, cls: str) -> None:
    assert parenthetical_class(inner) == cls


# ---------------------------------------------------------------- printed lists


@pytest.mark.parametrize("case", _derived("list"), ids=_ids)
def test_printed_list_item_names_its_row_and_only_rows_printing_it(
    case: tuple[str, str, list[str]],
) -> None:
    ds, row, level = case
    exact = _exact(ds, " ".join(level))
    assert row in exact
    assert all(_prints(ds, r, level) for r in exact)


def _list_fields() -> list[tuple[str, str, str]]:
    seen, out = set(), []
    for e in _snapshot():
        for item in e["list_items"]:
            key = (e["dataset"], e["printed"][item["field"]])
            if key not in seen:
                seen.add(key)
                out.append((e["dataset"], e["identity_row"], e["printed"][item["field"]]))
    return out


@pytest.mark.parametrize("f", _list_fields(), ids=lambda f: f"{f[1]}")
def test_a_coordinated_phrase_is_never_split_into_names(f: tuple[str, str, str]) -> None:
    _dataset, _row, text = f
    items = [x for x in re.split(r"[،,]", re.sub(r"\([^()]*\)", " ", norm(text))) if x.strip()]
    if any(" و " in f" {x.strip()} " for x in items):
        assert list_names(text) == []


@pytest.mark.parametrize(
    "text",
    ["گرم و بارانی، گرم و کم باران", "تولید همزمان حرارت و برق، برودت و برق", "تنها یک نام"],
)
def test_descriptive_lists_do_not_split(text: str) -> None:
    assert list_names(text) == []


def test_elliptical_list_item_takes_the_shared_words() -> None:
    assert list_names("مدار اولیه، ثانویه") == ["مدار اولیه", "مدار ثانویه"]


# ---------------------------------------------------------------- container and linking heads


@pytest.mark.parametrize("case", _derived("container"), ids=_ids)
def test_container_remainder_names_its_row_and_only_rows_printing_it(
    case: tuple[str, str, list[str]],
) -> None:
    ds, row, level = case
    # a row whose name is printed twice with different conditions needs its condition too
    entry = next(e for e in _snapshot() if e["identity_row"] == row)
    condition = entry["printed"].get("condition", "") if entry["condition_required"] else ""
    exact = _exact(ds, f"{' '.join(level)} {condition}")
    assert row in exact
    assert all(_prints(ds, r, level) for r in exact)


@pytest.mark.parametrize(
    ("levels", "kept"),
    [
        ([["گچ"], ["گچ", "و", "خاک"]], ["گچ", "و", "خاک"]),  # a conjunct, not a container
        ([["خاک", "و", "خشت"], ["خاک", "تثبیت"]], ["خاک", "تثبیت"]),
        ([["اصفهان"], ["اصفهان"]], ["اصفهان"]),  # nothing would remain
        ([["صفحات", "پایه"], ["صفحات", "تخته", "چندلا"]], ["تخته", "چندلا"]),
    ],
)
def test_container_rule_never_strips_a_conjunct_or_a_whole_name(
    levels: list[list[str]], kept: list[str]
) -> None:
    assert head_structure(levels)[0][-1] == kept


def test_a_conjunct_left_alone_names_no_mixture() -> None:
    # «گچ و خاک» is a mixture: «خاک» alone never names it
    mixture = [e["identity_row"] for e in _snapshot() if "گچ و خاک" in str(e["printed"])]
    assert mixture
    assert not set(mixture) & set(_exact("material_thermal_properties", "خاک"))


@pytest.mark.parametrize("case", _derived("linking"), ids=_ids)
def test_linking_head_names_its_row(case: tuple[str, str, list[str]]) -> None:
    ds, row, level = case
    assert row in _exact(ds, " ".join(level))


@pytest.mark.parametrize("word", ["متشکل", "ساخته", "تشکیل"])
def test_a_linking_word_alone_names_nothing(word: str) -> None:
    assert _exact("material_thermal_properties", word) == []


# ---------------------------------------------------------------- morphology


def _participles() -> list[tuple[str, str, str]]:
    """(dataset, row, «X شده» printed) for every printed participle of a name field."""
    out, seen = [], set()
    for e in _snapshot():
        for name, text in e["printed"].items():
            if "condition" in name or "specification" in name:
                continue
            for m in re.finditer(r"([ؠ-ۿ]{3,})\s+شده", norm(text)):
                if m[1].endswith("ن") or (e["dataset"], m[0]) in seen:
                    continue
                seen.add((e["dataset"], m[0]))
                out.append((e["dataset"], e["identity_row"], m[0]))
    return out


@pytest.mark.parametrize("p", _participles(), ids=lambda p: f"{p[1]}:{p[2]}")
def test_a_participle_written_solid_matches_the_printed_spacing(p: tuple[str, str, str]) -> None:
    _dataset, _row, spaced = p
    joined = spaced.replace(" ", "")
    stem = spaced.split()[0]
    assert keys(stem) <= query_forms([joined])


def test_a_negated_participle_keeps_its_negation() -> None:
    assert "کنترل" not in query_forms(["کنترلنشده"])


def test_relative_ye_is_dropped_only_before_ke() -> None:
    assert "کف" in query_forms(["کفی", "که"])
    assert "کف" not in query_forms(["کفی"])
    assert "کف" not in query_forms(["کفی", "مجاور"])


# ---------------------------------------------------------------- short-stem plural guard


def _plural_labels() -> list[tuple[str, str, str, list[str], list[str]]]:
    """(dataset, row, stem, label words, distinguishing words) of printed short-stem plurals."""
    out = []
    for short in ("t19-5-1", "t19-5-2", "t19-5-3", "t19-5-4", "t19-5-7"):
        idx = _index(short)
        for a in idx.alternatives:
            for lv in a.levels:
                for w in lv:
                    stem = _short_plural(w)
                    if stem is None:
                        continue
                    sibs = [g for g in idx.group(w) if g is not a and g.row != a.row]
                    rest = [x for x in lv if x != w]
                    telling = [x for x in rest if any(x not in g.words for g in sibs)]
                    out.append((short, a.row, stem, rest, telling))
    return out


@pytest.mark.parametrize("p", _plural_labels(), ids=lambda p: f"{p[1]}")
def test_short_stem_names_its_plural_only_with_its_distinguishing_words(
    p: tuple[str, str, str, list[str], list[str]],
) -> None:
    short, row, stem, rest, telling = p
    assert telling
    named = _exact(short, f"حداقل مقاومت {stem} {' '.join(rest)} در اقلیم 3A")
    assert row in named
    # the preposition: «در اقلیم …» / «در» + a shared word / a broken run name no door
    assert row not in _exact(short, f"حداقل مقاومت {stem} اقلیم 3A")
    shared = [x for x in rest if x not in telling]
    if shared:
        assert row not in _exact(short, f"حداقل مقاومت {stem} {' '.join(shared)} اقلیم 3A")
    assert row not in _exact(short, f"حداقل مقاومت {stem} اقلیم 3A {' '.join(telling)}")


# ---------------------------------------------------------------- places, dashes, captions


def _marker_places() -> list[str]:
    markers = ("بندر", "شهر", "شهرستان", "روستا", "بخش", "شهرک")
    places = {norm(r["city"]["text"]) for r in _r().store.app02}
    return sorted(p for p in places if p.split()[0] in markers)


@pytest.mark.parametrize("place", _marker_places())
def test_a_printed_place_beginning_with_a_marker_word_resolves(place: str) -> None:
    asked, unknown = _r().planner.resolve_places(norm(f"اقلیم {place} چیست؟"))
    assert place in asked
    assert unknown is None


def test_an_unknown_place_after_a_marker_is_still_typed() -> None:
    _asked, unknown = _r().planner.resolve_places(norm("اقلیم بندر کوانتومی چیست؟"))
    assert unknown == "کوانتومی"


def test_a_dash_between_spaces_ends_an_exclusion() -> None:
    p = parse(norm("کلینیک الف — نه کلینیک ب — در اقلیم 1B"))
    assert [e.text for e in p.exclusions] == ["کلینیک ب"]


def test_a_hyphen_inside_an_id_or_range_never_ends_an_exclusion() -> None:
    p = parse(norm("نه دیوار 19-5-1 مجاور خاک، در اقلیم 1B"))
    assert [e.text for e in p.exclusions] == ["دیوار 19-5-1 مجاور خاک"]


def test_a_quoted_caption_is_the_table_reference() -> None:
    caption = _r().router._title(MAIN + "19-5-1").split(":", 1)[-1].strip()
    for element in ("کف مجاور خاک", "سقف مجاور فضای آزاد"):
        res = _r().retrieve(f"در جدول «{caption}»، {element} در اقلیم 4B چه مقداری دارد؟")
        assert res.structured
        assert {h.dataset for h in res.structured} == {MAIN + "19-5-1"}


# ---------------------------------------------------------------- missing-modifier mutations


def _multiword_derived() -> list[tuple[str, str, list[str]]]:
    return [
        c
        for rule in ("parenthetical", "list", "container", "linking")
        for c in _derived(rule)
        if len(c[2]) > 1
    ]


@pytest.mark.parametrize("case", _multiword_derived(), ids=_ids)
def test_a_derived_name_missing_a_distinguishing_word_does_not_name_its_row(
    case: tuple[str, str, list[str]],
) -> None:
    ds, row, level = case
    idx = _index(ds)
    same = {e["identity_row"] for e in _snapshot() if row in e["records"]} | {row}
    for k, dropped in enumerate(level):
        cut = [w for i, w in enumerate(level) if i != k]
        # a sibling printing every remaining word but not the dropped one fits the cut as well
        fits = [
            a.row
            for a in idx.alternatives
            if a.row not in same
            and all(any(keys(w) & keys(x) for x in a.words) for w in cut)
            and not any(keys(dropped) & keys(x) for x in a.words)
        ]
        if fits:
            assert row not in _exact(ds, " ".join(cut)), (cut, fits[:3])
