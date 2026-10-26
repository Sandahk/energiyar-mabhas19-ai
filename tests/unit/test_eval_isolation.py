"""The evaluator scores the runtime; it must never share the runtime's logic.

* importing the scorer (``retrieval.eval``) and the wrong-row attribution
  (``retrieval.safety_metrics``) loads no planner, admission, coverage, certification, executor,
  identity or routing module;
* the attribution classes are computed from curated records and case labels only.
"""

import ast
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from mabhas19.retrieval import safety_metrics as sm

RUNTIME = (
    "mabhas19.retrieval.planner",
    "mabhas19.retrieval.admission",
    "mabhas19.retrieval.coverage",
    "mabhas19.retrieval.certification",
    "mabhas19.retrieval.executor",
    "mabhas19.retrieval.identity",
    "mabhas19.retrieval.router",
    "mabhas19.retrieval.hybrid",
)


def test_scorer_imports_no_runtime_module() -> None:
    code = (
        "import sys, json\n"
        "import mabhas19.retrieval.eval, mabhas19.retrieval.safety_metrics\n"
        "print(json.dumps(sorted(m for m in sys.modules if m.startswith('mabhas19'))))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout
    loaded = set(json.loads(out))
    assert not loaded & set(RUNTIME), sorted(loaded & set(RUNTIME))


def test_attribution_module_imports_only_the_standard_library() -> None:
    tree = ast.parse(Path(sm.__file__).read_text(encoding="utf-8"))
    names: list[str] = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            names += [a.name for a in n.names]
        elif isinstance(n, ast.ImportFrom) and n.module:
            names.append(n.module)
    assert not [x for x in names if x and x.startswith("mabhas19")], names


R: dict[str, dict[str, Any]] = {
    # Appendix 3: dormitory 5B, ratings C/B/A (same identity, rating differs)
    "app03-C-r22-5B": {"occupancy_id": "o22", "climate_class": "5B", "energy_rating": "C"},
    "app03-B-r22-5B": {"occupancy_id": "o22", "climate_class": "5B", "energy_rating": "B"},
    "app03-C-r33-4B": {"occupancy_id": "o33", "climate_class": "4B", "energy_rating": "C"},
    # Appendix 7: cement-bonded board vs wood-chip concrete (different identity)
    "app07-p008-r18": {
        "category": {"text": "چوب"},
        "subgroup": {"text": "صفحات پایه چوبی"},
        "material": {"text": "صفحات با تراشه‌های چسبیده با سیمان"},
        "density": {"raw": "کمتر از ۱۲۰۰"},
    },
    "app07-p006-r12": {
        "category": {"text": "بتن"},
        "subgroup": {"text": "بتن با خرده چوب"},
        "material": {"text": "ساخته شده با تراشه‌های چوب و سیمان"},
        "density": {"raw": "۴۵۰ تا ۶۵۰"},
    },
    "main-t19-3-3-r03": {"energy_rating": {"code": "C"}},
}


def test_second_table_row_is_foreign() -> None:  # M011
    a = sm.attribute({"records": ["app03-C-r33-4B"]}, ["main-t19-3-3-r03"], R)
    assert a.foreign_table_rows == ["main-t19-3-3-r03"]


def test_rating_from_another_clause_is_a_dimension_leak() -> None:  # M075
    a = sm.attribute({"records": ["app03-C-r22-5B"]}, ["app03-B-r22-5B"], R)
    assert a.dimension_leaks == {"rating": ["app03-B-r22-5B"]}
    assert not a.identity_substitutions
    assert not a.foreign_table_rows


def test_another_material_is_an_identity_substitution() -> None:  # M079
    a = sm.attribute({"records": ["app07-p008-r18"]}, ["app07-p006-r12"], R)
    assert a.identity_substitutions == ["app07-p006-r12"]


def test_rows_where_none_exist_bypass_the_refusal_by_kind() -> None:  # M102, M195
    table = [sm.APP07]
    out = {"no_records": True, "tables": table, "no_match_outcomes": ["unsupported_output"]}
    assert sm.attribute(out, ["app07-p006-r12"], R).refusal_bypassed == {
        "output": ["app07-p006-r12"]
    }
    calc = {
        "no_records": True,
        "tables": [sm.APP03],
        "no_match_outcomes": ["unsupported_calculation"],
    }
    assert sm.attribute(calc, ["app03-C-r22-5B"], R).refusal_bypassed == {
        "calculation": ["app03-C-r22-5B"]
    }
    unit = {**calc, "refusal_kind": "unit"}
    assert sm.attribute(unit, ["app03-C-r22-5B"], R).refusal_bypassed == {
        "unit": ["app03-C-r22-5B"]
    }


def test_any_row_of_a_prose_case_is_foreign() -> None:
    assert sm.attribute({"blocks": ["main-p001-b001"]}, ["app03-C-r22-5B"], R).foreign_table_rows


def test_every_wrong_row_lands_in_exactly_one_class() -> None:
    ev = {"records": ["app03-C-r22-5B", "app07-p008-r18"]}
    wrong = ["app03-B-r22-5B", "app07-p006-r12", "main-t19-3-3-r03"]
    a = sm.attribute(ev, wrong, R)
    got = (
        a.foreign_table_rows
        + a.identity_substitutions
        + [i for ids in a.dimension_leaks.values() for i in ids]
        + [i for ids in a.refusal_bypassed.values() for i in ids]
    )
    assert sorted(got) == sorted(wrong)


def test_dataset_of_record_prefixes() -> None:
    assert sm.dataset_of("main-t19-2-1-r07c16") == "m19-ed5-1404-main.t19-2-1"
    assert sm.dataset_of("main-t19-5-3-item09") == "m19-ed5-1404-main.t19-5-3"
    assert sm.dataset_of("app02-p002-r22") == sm.APP02
