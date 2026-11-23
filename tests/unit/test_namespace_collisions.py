"""Namespace collisions are generated from the curated catalogue and match the reviewed baseline
(``eval/robustness/collisions_baseline.yaml``); the reviewed exceptions are complete.

A new curated name that collides with another namespace fails this test until the baseline is
regenerated (``python -m mabhas19.cli.robustness_audit collisions --write-baseline``) and the new
entry is reviewed.
"""

from __future__ import annotations

from functools import cache

from mabhas19.api import invariants as I
from mabhas19.api import quantities as Q
from mabhas19.core.paths import CURATED_DIR
from mabhas19.retrieval.structured import StructuredStore

CLASSES = {"guarded", "watch", "same_concept", "family_word", "generic_word"}


@cache
def _store() -> StructuredStore:
    return StructuredStore(CURATED_DIR)


@cache
def _found() -> dict[str, I.Collision]:
    return {c.key: c for c in I.collisions(_store())}


def _baseline() -> dict[str, dict[str, object]]:
    return {str(e["key"]): e for e in I.baseline()["collisions"]}


def test_the_generated_collisions_are_exactly_the_reviewed_baseline() -> None:
    found, base = _found(), _baseline()
    assert sorted(set(found) - set(base)) == []  # new collisions: regenerate and review
    assert sorted(set(base) - set(found)) == []  # stale entries
    for key, c in found.items():
        assert list(c.namespaces) == base[key]["namespaces"], key
        if c.kind == "prefix":
            assert list(c.longer_namespaces) == base[key]["longer_namespaces"], key


def test_every_collision_is_reviewed() -> None:
    for key, e in _baseline().items():
        assert e["review"] in CLASSES, key
        assert e["note"], key


def test_a_city_that_begins_another_namespaces_word_is_guarded_or_watched() -> None:
    """The class of the Tabriz bug, found from the data: a listed city name that is the start of
    a word of a material or row (the engine may read the city as that entity)."""
    risky = [
        c
        for c in _found().values()
        if c.kind == "prefix"
        and "city" in c.namespaces
        and any(I.group(n) != "city" for n in c.longer_namespaces)
    ]
    assert risky  # today: «تبریز» -> «چوب‌های طبیعی تبریزی»
    base = _baseline()
    for c in risky:
        assert base[c.key]["review"] == "guarded", c.key


def test_the_near_miss_foreign_word_belongs_to_no_namespace() -> None:
    names = I.namespaces(_store())
    key = I.word_keys(I.FOREIGN)
    assert key
    assert all(key[0] not in words for words in names.values())


def test_the_reviewed_exceptions_are_complete() -> None:
    data = I.exceptions()
    ids = [x["id"] for x in data["exceptions"]]
    assert ids == [f"X{i}" for i in range(1, 10)]
    for x in data["exceptions"]:
        assert x["condition"], x["id"]
        assert x.get("evidence") or x["kind"] in ("not_a_table_answer",), x["id"]
        assert x["scope"] in data["scopes"], x["id"]


def test_every_exception_is_scoped_as_approved() -> None:
    """X1-X9 are not one allow-list: only quantity transformations may change the primary
    answer's quantity; audit exclusions never reach the runtime; X9 only satisfies the
    written-word invariant; X3 only adds supporting context."""
    data = I.exceptions()
    scope = {x["id"]: x["scope"] for x in data["exceptions"]}
    assert scope == {
        "X1": "quantity_transformation",
        "X2": "quantity_transformation",
        "X3": "supporting_context",
        "X4": "quantity_transformation",
        "X5": "quantity_transformation",
        "X6": "quantity_transformation",
        "X7": "audit_exclusion",
        "X8": "audit_exclusion",
        "X9": "lexical_equivalence",
    }
    scopes = data["scopes"]
    assert scopes["quantity_transformation"]["bypasses"] == [
        "quantity_preservation",
        "quantity_table_consistency",
    ]
    assert scopes["supporting_context"]["bypasses"] == []
    assert scopes["audit_exclusion"] == {
        **scopes["audit_exclusion"],
        "bypasses": [],
        "runtime": False,
    }
    assert scopes["lexical_equivalence"]["bypasses"] == ["unwritten_word"]
    # the code applies each exception only within its scope
    assert I.EXCEPTION_USE == {
        "X1": "quantity_preservation",
        "X2": "quantity_preservation",
        "X4": "quantity_preservation",
        "X5": "quantity_preservation",
        "X6": "quantity_preservation",
        "X3": "context",
        "X7": "audit",
        "X8": "audit",
        "X9": "unwritten_word",
    }
    for xid, use in I.EXCEPTION_USE.items():
        bypasses = scopes[scope[xid]]["bypasses"]
        if use in ("quantity_preservation", "unwritten_word"):
            assert use in bypasses, xid
        else:
            assert bypasses == [], xid
    assert set(data["quantities"]) == {q.id for q in Q.QUANTITIES}
