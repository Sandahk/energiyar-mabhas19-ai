"""The frozen quantity-phrasing set (``eval/clarify/quantity2.yaml``): rules A-C of
``api.quantities`` through the structured rescue. Written before the rules existed."""

from __future__ import annotations

from typing import Any

import pytest
import yaml
from tests.unit.test_natural1 import _ask
from tests.unit.test_structured1 import check as structured_check

from mabhas19.core.paths import EVAL_DIR

CASES = yaml.safe_load((EVAL_DIR / "clarify" / "quantity2.yaml").read_text(encoding="utf-8"))


def check(case: dict[str, Any], question: str) -> list[str]:
    a = _ask(question)
    kind = (a.safety.get("composite") or {}).get("kind")
    if case["expect"] == "untouched":
        out = [f"taken over: {a.response_type}"] if kind == "rescue" else []
        if a.safety.get("provenance_violations"):
            out.append(f"safety: {a.safety}")
        return out
    return structured_check(case, a)


@pytest.mark.parametrize("case", CASES["cases"], ids=[c["id"] for c in CASES["cases"]])
def test_frozen_quantity_phrasing_case(case: dict[str, Any]) -> None:
    assert check(case, case["question"]) == []
