"""Table scope (I10): whether a stated regulatory value can apply to a table at all.

A table prints some regulatory dimensions (Appendix 3: climate class and rating) and not others.
For every regulatory dimension it does not print, its schema declares the scope
(:class:`~mabhas19.retrieval.schema.ScopeDecl`): ``universal`` - the dimension does not affect
the table's values, so a stated value neither filters nor blocks - or ``fixed`` to the values the
source establishes, with the curated blocks that establish them (the prescriptive 19-5-x
requirements are those for energy rating D). A stated value outside a fixed scope asks for a
requirement the table does not print (``invalid_constraint``: «برای کسب رده B، حداقل R کف …» of
table 19-5-1, which prints the rating-D values); a dimension with no declaration fails closed.
Applicability is never inferred from the curated data. The plan-coverage stage and result
certification both apply :func:`verdict`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mabhas19.retrieval.schema import TableSchema

REGULATORY = ("climate_class", "rating", "group", "year")
# verdict states
PRINTED = "printed"  # the table prints the dimension (its own domain check applies)
UNIVERSAL = "universal"
IN_SCOPE = "in_scope"
OUT_OF_SCOPE = "out_of_scope"
UNDECLARED = "undeclared"


@dataclass(frozen=True)
class ScopeVerdict:
    dimension: str
    values: tuple[str, ...]
    state: str
    detail: str = ""

    @property
    def blocks(self) -> bool:
        return self.state in (OUT_OF_SCOPE, UNDECLARED)


def _norm(v: str) -> str:
    return str(v).replace(" ", "").upper()


def verdict(schema: TableSchema, dimension: str, values: list[str]) -> ScopeVerdict:
    """The scope verdict of stated values of a regulatory dimension for one table."""
    vals = tuple(str(v) for v in values)
    if schema.dim(dimension) is not None:
        return ScopeVerdict(dimension, vals, PRINTED)
    decl = schema.scope_of(dimension)
    if decl is None:
        return ScopeVerdict(
            dimension,
            vals,
            UNDECLARED,
            f"the table prints no {dimension} and declares no scope for it; nothing is assumed",
        )
    if decl.kind == "universal":
        return ScopeVerdict(dimension, vals, UNIVERSAL)
    allowed = {_norm(x) for x in decl.values}
    outside = [v for v in vals if _norm(v) not in allowed]
    if outside:
        return ScopeVerdict(
            dimension,
            vals,
            OUT_OF_SCOPE,
            f"{dimension} {outside}: the table prints the requirements for {list(decl.values)} "
            f"only ({', '.join(decl.provenance)})",
        )
    return ScopeVerdict(dimension, vals, IN_SCOPE)


def undeclared(schema: TableSchema) -> list[str]:
    """Regulatory dimensions the table neither prints nor declares (must be empty)."""
    return [d for d in REGULATORY if schema.dim(d) is None and schema.scope_of(d) is None]


__all__ = [
    "IN_SCOPE",
    "OUT_OF_SCOPE",
    "PRINTED",
    "REGULATORY",
    "UNDECLARED",
    "UNIVERSAL",
    "ScopeVerdict",
    "undeclared",
    "verdict",
]
