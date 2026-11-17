"""A chosen printed row -> a canonical printed question for the unchanged engine, and the check
that the engine's answer is about that row (a certified value of it, or a question about one of
its printed conditions). The engine certifies; nothing here does."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mabhas19.api import quantities as Q

if TYPE_CHECKING:
    from mabhas19.generation.contract import Answer
    from mabhas19.semantic.catalog import Row

# the words a canonical question starts with: each quantity's printed name
CANONICAL = {
    "eui": "حداکثر شدت مصرف انرژی مجاز",
    "r_value": "حداقل مقاومت حرارتی",
    "u_value": "حداکثر ضریب انتقال حرارتی",
    "shgc": "حداکثر ضریب بهره خورشیدی",
    "lambda": "ضریب هدایت حرارتی",
    "density": "چگالی",
    "efficiency": "حداقل بازدهی",
    "hot_water": "حداکثر دمای آب گرم مصرفی",
    "daylight": "حداقل نسبت مساحت کفایت نور روز",
    "uniformity": "حداقل یکنواختی نور مصنوعی",
}
CONDITIONS = frozenset({"climate_class", "rating", "area", "density", "group", "year"})


def canonical(quantity: str, row: Row, code: str = "", cities: tuple[str, ...] = ()) -> str:
    """Only printed words: the quantity's printed name, the printed row, a printed climate code
    or printed city the user wrote."""
    assert quantity in Q.BY_ID
    tail = f" در اقلیم {code}" if code else ""
    tail += "".join(f" در {c}" for c in cities)
    return f"{CANONICAL[quantity]} {row.label}{tail}"


def about(answer: Answer, row: Row) -> str:
    """``"value"`` (certified values, all of ``row``), ``"condition"`` (a question about a printed
    condition of ``row``), or ``""`` (anything else, including another row)."""
    if answer.response_type == "structured_value" and answer.values:
        return "value" if all(v.row_label == row.label for v in answer.values) else ""
    if answer.response_type == "clarification" and answer.clarifications:
        c = answer.clarifications[0]
        named = any(row.label in i.value for i in answer.interpretation)
        return "condition" if c.field in CONDITIONS and named else ""
    return ""
