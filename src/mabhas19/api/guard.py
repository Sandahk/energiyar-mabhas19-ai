"""Phase 3 fail-closed quantity guard (runtime; ``eval/robustness/exceptions.yaml``).

Invariant: once :func:`mabhas19.api.quantities.analyse` identifies exactly one table quantity, the
PRIMARY engine result must be that quantity - the answer's values, else its first clarification,
else its first refusal (secondary refusals and unused plans never decide). Compatible means: a
table of the quantity, printing an output the table's schema declares for it (Appendix 7 λ and
density told apart by output; 19-5-2 U and SHGC, which share an output, by the printed row).

Only the quantity-transformation exceptions may permit another primary table: X1 (R for a written
Appendix 7 material -> the λ offer), X2 (U for an opaque element -> 19-5-1 R: one of U's own
tables), X4 (an explicit table or section reference), X5 (the implementation schedule 19-2-1), X6
(another quantity written by name or symbol, including the city's climate classification). City
tables are supporting context only (X3): never the primary answer of a quantity question. The
audit exclusions (X7, X8) and the lexical equivalence (X9) play no part here.

:func:`violation` only reports. The orchestration then suppresses the engine's reading and lets the
asked quantity's existing structured path answer (row groups, structured rescue with the exact
certified city lookup, a typed refusal); nothing here produces a value, a row or a climate.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from mabhas19.api import quantities as Q
from mabhas19.api import wording as W
from mabhas19.generation import options as O
from mabhas19.retrieval.schema import APP02, APP07, MAIN

if TYPE_CHECKING:
    from mabhas19.generation.contract import Answer
    from mabhas19.retrieval.models import RetrievalResult
    from mabhas19.retrieval.structured import StructuredStore

CITY_TABLES = frozenset({APP02, f"{MAIN}19-3-2"})  # X3: supporting context only
SCHEDULE = f"{MAIN}19-2-1"  # X5
# outputs of a table shared by two quantities that belong to one of them only
OWNED_OUTPUTS = {"lambda": frozenset({"thermal_conductivity"}), "density": frozenset({"density"})}
# tables shared by quantities with one output name: told apart by the printed row
ROW_TOLD = frozenset({f"{MAIN}19-5-2"})
# X6: printed symbols of quantities (the deterministic reader does not count them on their own)
SYMBOLS = {
    "u_value": re.compile(r"(?<![A-Za-z])U(?![A-Za-z])"),
    "shgc": re.compile(r"SHGC", re.IGNORECASE),
    "lambda": re.compile(r"λ"),
    "r_value": re.compile(r"(?<![A-Za-z])R(?![A-Za-z])"),
}
_S = r"[\s‌]+"
# X6: the city's climate classification is itself asked («رده اقلیمی …», «اقلیم … چیه؟»)
CLIMATE_ASK = re.compile(
    rf"رده{_S}?(?:ی{_S}?)?اقلیمی|(?:کدام|چه){_S}اقلیم"
    rf"|اقلیم{_S}\S+(?:{_S}\S+){{0,2}}{_S}(?:چیست|چیه|کدومه|کدام{_S}است)"
)
_WORD = re.compile(r"[^\W\d_]+")
MIN_WORD = 3


def _words(text: str) -> set[str]:
    words = _WORD.findall(W.letters(text).replace("‌", " "))
    return {k for k in (O.label_key(w) for w in words) if len(k) >= MIN_WORD}


def allowed_outputs(store: StructuredStore, quantity: Q.Quantity) -> set[str]:
    """The outputs the quantity's own tables print for it (schemas), and ``""`` (a printed-row
    listing names no other quantity)."""
    out = {""}
    for dataset, _ in quantity.tables:
        schema = store.schemas.get(dataset)
        out |= {str(getattr(o, "name", o)) for o in (schema.outputs if schema else [])}
    for other, owned in OWNED_OUTPUTS.items():
        if other != quantity.id:
            out -= owned
    return out


def primary(answer: Answer) -> tuple[str, set[str], set[str]] | None:
    """(kind, tables, outputs) of the answer's primary block; ``None`` for prose or out of scope
    (no table answer, so no substitution)."""
    if answer.values:
        return "value", {v.dataset for v in answer.values}, {v.output for v in answer.values}
    if answer.clarifications:
        c = answer.clarifications[0]
        return "clarification", {c.dataset} if c.dataset else set(), {c.field}
    if answer.refusals:
        r = answer.refusals[0]
        return "refusal", {r.dataset} if r.dataset else set(), set()
    return None


def _material_written(question: str, answer: Answer) -> bool:
    """X1: a word of the Appendix 7 material the answer names is written in the question."""
    names = [i.value for i in answer.interpretation if i.key == "material"]
    return any(_words(n) & _words(question) for n in names)


def _other_quantity_written(question: str, dataset: str, asked: Q.Quantity) -> bool:
    """X6: another quantity whose tables include ``dataset`` is written by name or symbol."""
    text = W.letters(question)
    for q in Q.QUANTITIES:
        if q.id == asked.id or all(d != dataset for d, _ in q.tables):
            continue
        symbol = SYMBOLS.get(q.id)
        if q.pattern().search(text) or (symbol is not None and symbol.search(text)):
            return True
    return False


def violation(  # noqa: PLR0911
    store: StructuredStore, question: str, answer: Answer, result: RetrievalResult | None
) -> str | None:
    """Why the primary result of ``answer`` is not the one table quantity ``question`` asks
    for, or ``None`` (compatible, a scoped exception, or no single quantity asked)."""
    request = Q.analyse(question)
    if request is None:
        return None
    signals = result.route.signals if result is not None else {}
    if signals.get("table_ref") or signals.get("section_ref"):
        return None  # X4
    found = primary(answer)
    if found is None or not found[1]:
        return None
    kind, tables, outputs = found
    asked = request.quantity
    own = {d for d, _ in asked.tables}
    text = W.letters(question)
    for dataset in sorted(tables - own):
        if dataset in CITY_TABLES and CLIMATE_ASK.search(text):
            continue  # X6: the city's climate classification is itself asked
        x1 = dataset == APP07 and asked.id == "r_value" and kind == "clarification"
        if x1 and _material_written(question, answer):
            continue  # X1: R for a written material -> the λ offer
        if dataset == SCHEDULE:
            continue  # X5
        if _other_quantity_written(question, dataset, asked):
            continue  # X6
        return f"{asked.id}: primary {kind} from {dataset}"
    if kind != "value" or not tables <= own:
        return None
    extra = outputs - allowed_outputs(store, asked)
    if extra:
        return f"{asked.id}: primary output {sorted(extra)}"
    for v in answer.values:
        if v.dataset in ROW_TOLD and not asked.pattern().search(W.letters(v.row_label)):
            return f"{asked.id}: primary row «{v.row_label}» of another quantity"
    return None
