"""The typed answer contract between retrieval and every client (API, UI, logs).

An :class:`Answer` is built only by :mod:`mabhas19.generation.composer` from one frozen
:class:`~mabhas19.retrieval.models.RetrievalResult`. A client never reads retrieval internals.

Five response types:

* ``structured_value`` - certified table rows (values, units, printed conditions, the row
  each claim comes from), with the interpretation echo;
* ``prose`` - retrieved source excerpts, quoted verbatim with their section and page;
* ``refusal`` - a typed fail-closed outcome explained in Persian, with what is missing;
* ``out_of_scope`` - the question is not answerable from the loaded Mabhas 19 corpus;
* ``clarification`` - the engine refused because one item is missing or ambiguous, and the
  answer asks for exactly that item (with the printed options, when the source prints a finite
  list). Choosing an option re-asks the original question with the choice added; nothing is
  answered from the UI.

A mixed question may carry several kinds of block at once (values and excerpts, or a refusal
and excerpts); ``response_type`` names the primary one.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

ResponseType = Literal["structured_value", "prose", "refusal", "out_of_scope", "clarification"]
EchoStatus = Literal["resolved", "ambiguous", "unresolved", "requested"]
CONTRACT_VERSION = "m2-answer-3"


@dataclass(frozen=True)
class Source:
    """A displayed reference: a curated table, an appendix or a main-text section."""

    kind: Literal["table", "appendix", "section"]
    label: str  # Persian citation text, e.g. «جدول ۱۹-۵-۱» or «بند ۱۹-۲-۱-۹»
    title: str  # printed caption or section title ("" when the source has none in Persian)
    document: str  # doc_id
    printed_pages: tuple[str, ...] = ()
    pdf_pages: tuple[int, ...] = ()
    dataset: str | None = None
    ids: tuple[str, ...] = ()  # record or chunk IDs behind this reference


@dataclass(frozen=True)
class Interpretation:
    """One field of the interpretation echo, read from the deterministic plan or a certified
    row; never inferred by the composer."""

    key: str  # machine name: a dimension, an entity kind or ``output``
    label: str  # Persian field name
    value: str
    status: EchoStatus = "resolved"
    # the exact engine strings the value was rendered from (plan, no-match or row fields)
    source_text: tuple[str, ...] = ()


@dataclass(frozen=True)
class Condition:
    label: str
    value: str


@dataclass(frozen=True)
class ValueItem:
    """One certified row and the value it prints for the requested output."""

    dataset: str
    record_id: str
    row_label: str
    output: str  # machine name of the certified output
    output_label: str
    value: str  # exactly as printed in the source cell
    unit: str | None  # printed unit (as curated)
    unit_label: str | None  # Persian reading of the unit (no numbers added)
    conditions: tuple[Condition, ...] = ()
    source_issues: tuple[str, ...] = ()
    source: Source | None = None


@dataclass(frozen=True)
class Excerpt:
    chunk_id: str
    text: str  # a verbatim slice of the chunk text («…» marks a cut)
    source: Source
    source_issues: tuple[str, ...] = ()


@dataclass(frozen=True)
class Refusal:
    outcome: str  # the engine's typed fail-closed outcome
    reason: str  # the engine's specific reason code
    message: str  # Persian explanation
    dataset: str | None = None
    missing: tuple[str, ...] = ()  # Persian names of what would make it answerable
    hint: str = ""
    options: tuple[str, ...] = ()  # printed candidates the question did not distinguish
    table: str = ""  # Persian citation of the table the outcome belongs to


@dataclass(frozen=True)
class ClarifyOption:
    """One printed choice: its chip label and the words added to the question when chosen."""

    label: str
    text: str


@dataclass(frozen=True)
class Clarification:
    """A follow-up question for exactly one missing or ambiguous item of a refused lookup."""

    field: str  # machine name: a dimension, an entity kind or ``output``
    field_label: str  # Persian name of the field
    prompt: str  # Persian question shown to the user
    outcome: str  # the engine's typed outcome this clarification answers
    dataset: str = ""
    table: str = ""  # Persian citation of the table the options come from
    options: tuple[ClarifyOption, ...] = ()  # printed values only
    free_text: bool = False  # the user may type the value (no finite printed list)
    free_text_prefix: str = ""  # words put before the typed value (e.g. «مساحت»)
    placeholder: str = ""  # unit-neutral, never an example number
    hint: str = ""
    note: str = ""  # a fixed explanation (e.g. R versus λ); no regulatory value


@dataclass(frozen=True)
class Notice:
    kind: Literal["conflict", "source_issue", "note", "technical"]
    text: str
    ids: tuple[str, ...] = ()


@dataclass
class Answer:
    request_id: str
    question: str
    response_type: ResponseType
    headline: str
    interpretation: list[Interpretation] = field(default_factory=list)
    values: list[ValueItem] = field(default_factory=list)
    excerpts: list[Excerpt] = field(default_factory=list)
    refusals: list[Refusal] = field(default_factory=list)
    notices: list[Notice] = field(default_factory=list)
    clarifications: list[Clarification] = field(default_factory=list)
    sources: list[Source] = field(default_factory=list)
    hidden_values: int = 0  # certified rows beyond the display limit (still in ``row_ids``)
    safety: dict[str, Any] = field(default_factory=dict)
    contract: str = CONTRACT_VERSION
    # certified rows of an earlier lookup this answer depends on (a city's climate class),
    # copied unchanged from that lookup's own checked answer (api.orchestration); never set by
    # the composer
    context: list[ValueItem] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def texts(self) -> list[str]:
        """Every string a client may display (the provenance checks read exactly these)."""
        out = [self.headline]
        out += [f"{i.label} {i.value}" for i in self.interpretation]
        for v in [*self.values, *self.context]:
            out += [v.row_label, v.output_label, v.value, v.unit or "", v.unit_label or ""]
            out += [f"{c.label} {c.value}" for c in v.conditions]
        out += [e.text for e in self.excerpts]
        for r in self.refusals:
            out += [r.message, r.hint, *r.missing, *r.options]
        out += [n.text for n in self.notices]
        for c in self.clarifications:
            out += [c.prompt, c.field_label, c.table, c.placeholder, c.hint, c.note]
            out += [x for o in c.options for x in (o.label, o.text)]
        out += [f"{s.label} {s.title} {' '.join(s.printed_pages)}" for s in self.sources]
        return [t for t in out if t]
