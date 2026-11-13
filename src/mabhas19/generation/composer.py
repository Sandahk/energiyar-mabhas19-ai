"""Deterministic answer composer: one frozen :class:`RetrievalResult` -> one :class:`Answer`.

Read-only over the engine. The composer never retrieves, never ranks, never computes and never
turns a refusal into an answer:

* values come only from ``result.structured`` (rows the engine has certified), each value read
  from the certified output cell of its own row, exactly as printed;
* the interpretation echo is rendered from the engine's plans (or, for a route-level unknown
  entity, from its typed no-match), never inferred;
* excerpts are verbatim slices of retrieved chunks, chosen by the engine's matched terms;
* refusals and out-of-scope answers come from the typed outcomes and the scope decision;
* a refusal for one missing or ambiguous item becomes a clarification: a Persian question for
  exactly that item, offering only the values the source prints for it. Choosing one re-asks
  the original question with the choice added; the engine answers (or refuses) again.

Every composed answer is checked by :func:`mabhas19.generation.provenance.check` before it is
returned; an answer that fails any check is replaced by a refusal that shows nothing.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Literal

from mabhas19.generation import labels as L
from mabhas19.generation import options as O
from mabhas19.generation.contract import (
    Answer,
    Clarification,
    ClarifyOption,
    Condition,
    EchoStatus,
    Excerpt,
    Interpretation,
    Notice,
    Refusal,
    ResponseType,
    Source,
    ValueItem,
)
from mabhas19.generation.provenance import check
from mabhas19.retrieval.schema import APP02, APP03, APP07, MAIN
from mabhas19.retrieval.structured import LABEL_FIELDS
from mabhas19.text.normalize import normalize_for_search

if TYPE_CHECKING:
    from mabhas19.retrieval.models import ProseHit, RetrievalResult, StructuredHit
    from mabhas19.retrieval.structured import StructuredStore

COMPOSER_VERSION = "m2-composer-2"
MAX_VALUES = 12  # certified rows displayed; the rest are counted, never dropped silently
MAX_EXCERPTS = 3
EXCERPT_CHARS = 420
MIN_EXCERPT = 60
MAX_OPTIONS = 16  # more printed choices than this: no clarification (the user retypes)
_FA_DIGITS = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")
_PRINTED_KEYS = ("printed", "displayed", "raw", "text", "code", "value", "label")
_BOOKKEEPING = frozenset(
    {"row_number", "item_number", "sub_row", "metric", "metric_basis", "formula_ref", "notes"}
)


def fa_digits(text: str) -> str:
    return text.translate(_FA_DIGITS)


def printed(cell: Any) -> str:
    """The printed form of a curated cell (symbols read as their printed mark)."""
    if cell is None or isinstance(cell, bool):
        return ""
    if isinstance(cell, str | int | float):
        return str(cell).strip()
    if isinstance(cell, list):
        return "، ".join(p for p in (printed(c) for c in cell) if p)
    if isinstance(cell, dict):
        if cell.get("symbol"):
            return L.SYMBOLS.get(str(cell["symbol"]), str(cell["symbol"]))
        for key in _PRINTED_KEYS:
            value = cell.get(key)
            usable = isinstance(value, str | int | float) and not isinstance(value, bool)
            if usable and str(value).strip():
                return str(value).strip()
    return ""


# ---------------------------------------------------------------- rows


def _suffix(dataset: str) -> str:
    return dataset.rsplit(".", maxsplit=1)[-1]


def _label_keys(dataset: str) -> tuple[str, ...]:
    if dataset == APP03:
        return ("occupancy",)
    if dataset == APP07:
        return ("display_name",)
    if dataset in (APP02, f"{MAIN}19-3-2"):
        return ("city",)
    return tuple(path.split(".")[0] for path, _ in LABEL_FIELDS.get(_suffix(dataset)[1:], ()))


def row_label(dataset: str, values: dict[str, Any]) -> str:
    """The printed label of a certified row, from the row's own fields."""
    if dataset == APP07:
        return printed(values.get("display_name"))
    if dataset == APP03:
        return printed(values.get("occupancy"))
    if dataset in (APP02, f"{MAIN}19-3-2"):
        return printed(values.get("city"))
    parts = []
    for path, _ in LABEL_FIELDS.get(_suffix(dataset)[1:], ()):
        cur: Any = values
        for part in path.split("."):
            cur = cur.get(part) if isinstance(cur, dict) else None
        if isinstance(cur, str) and cur.strip() and cur.strip() not in parts:
            parts.append(cur.strip())
    return " – ".join(parts)


def _conditions(hit: StructuredHit, output_key: str | None) -> tuple[Condition, ...]:
    skip = {*_label_keys(hit.dataset), *_BOOKKEEPING}
    if output_key:
        skip.add(output_key)
    out = []
    for key, cell in hit.values.items():
        if key in skip or key not in L.CONDITION_LABELS:
            continue
        text = printed(cell)
        if text and isinstance(cell, dict) and cell.get("unit"):
            text = f"{text} {cell['unit']}"
        if key == "year" and isinstance(cell, dict) and not cell.get("text"):
            text = L.YEAR_ORDINALS.get(text, text)
        if text:
            out.append(Condition(L.CONDITION_LABELS[key], text))
    return tuple(out)


class _Sources:
    """Citations from the engine's table evidence, row pages and prose hits (cached)."""

    def __init__(self, store: StructuredStore) -> None:
        self.store = store
        self._tables: dict[str, tuple[str, str]] = {}

    def table(self, dataset: str) -> tuple[str, str]:
        if dataset not in self._tables:
            ev = self.store.table_evidence(dataset, "answer citation")
            if dataset.startswith(MAIN):
                label, _, title = ev.title.partition(":")
                self._tables[dataset] = (label.strip(), title.strip())
            else:
                number = int(dataset.split(".", maxsplit=1)[0].rsplit("app", 1)[-1])
                self._tables[dataset] = (f"پیوست {fa_digits(str(number))}", "")
        return self._tables[dataset]

    def row(self, hit: StructuredHit) -> Source:
        label, title = self.table(hit.dataset)
        return Source(
            kind="table" if hit.dataset.startswith(MAIN) else "appendix",
            label=label,
            title=title,
            document=hit.doc_id,
            printed_pages=(hit.printed_page,) if hit.printed_page else (),
            pdf_pages=(hit.pdf_page,) if hit.pdf_page else (),
            dataset=hit.dataset,
            ids=(hit.record_id,),
        )


def _value_item(
    hit: StructuredHit, store: StructuredStore, sources: _Sources, requested: dict[str, str]
) -> ValueItem:
    schema = store.schemas.get(hit.dataset)
    out = schema.output(hit.output) if (schema is not None and hit.output) else None
    key = out.path.split(".")[0] if out is not None else None
    value = printed(hit.values.get(key)) if key else ""
    unit = hit.unit
    conditions = _conditions(hit, key)
    rf = requested.get(hit.dataset)
    asked = schema.output(rf) if (schema is not None and rf and not hit.output) else None
    if asked is not None:  # a reverse lookup: the row prints the asked value as its key
        cell = hit.values.get(asked.path.split(".")[0])
        text = printed(cell)
        if text and isinstance(cell, dict) and cell.get("unit"):
            text = f"{text} {cell['unit']}"
        if text and rf in L.OUTPUT_LABELS:
            conditions = (Condition(L.OUTPUT_LABELS[rf], text), *conditions)
    return ValueItem(
        dataset=hit.dataset,
        record_id=hit.record_id,
        row_label=row_label(hit.dataset, hit.values),
        output=hit.output or "",
        output_label=L.OUTPUT_LABELS.get(hit.output or "", "") if hit.output else "",
        value=value,
        unit=unit,
        unit_label=L.UNIT_LABELS.get(unit) if unit else None,
        conditions=conditions,
        source_issues=tuple(hit.source_issues),
        source=sources.row(hit),
    )


# ---------------------------------------------------------------- interpretation echo


def _interval_text(interval: Any) -> str:
    text = str(interval).strip().replace(" and ", " و ")
    return text[2:].strip() if text.startswith("= ") else text


def _unique(names: Iterable[str]) -> list[str]:
    out: list[str] = []
    for name in names:
        n = name.strip()
        if n and n not in out:
            out.append(n)
    return out


def _entity_name(name: str, store: StructuredStore) -> str:
    """A plan names an Appendix 3 occupancy by its curated ID: show its printed label."""
    label = (store.occupancies.get(name.strip()) or {}).get("label")
    return str(label) if label else name


_ENTITY_STATUS: dict[str, EchoStatus] = {"resolved": "resolved", "ambiguous": "ambiguous"}


def _entity_label(kind: str, dataset: str) -> str:
    if kind == "label":
        return L.LABEL_KIND.get(_suffix(dataset), "")
    return L.FIELD_LABELS.get(kind, "")


def interpretation(  # noqa: PLR0912
    result: RetrievalResult,
    datasets: set[str] | None,
    store: StructuredStore,
    ungrounded: frozenset[str] = frozenset(),
) -> list[Interpretation]:
    """The interpretation echo: the requested output, the named entities and the constraints
    of each structured plan (``datasets`` limits it to the plans that answered)."""
    items: list[Interpretation] = []
    seen: set[tuple[str, str]] = set()
    rows_by_id = {h.record_id: h for h in result.structured}

    def add(item: Interpretation) -> None:
        if item.label and item.value and (item.key, item.value) not in seen:
            seen.add((item.key, item.value))
            items.append(item)

    for p in result.plans:
        ds = str(p.get("target", ""))
        if datasets is not None and ds not in datasets:
            continue
        rf = p.get("requested_field")
        if rf and rf in L.OUTPUT_LABELS:
            add(Interpretation("output", L.REQUESTED, L.OUTPUT_LABELS[rf], "requested", (rf,)))
        for e in p.get("entities") or []:
            text = str(e.get("text") or "")
            status: EchoStatus = _ENTITY_STATUS.get(str(e.get("status")), "unresolved")
            label = _entity_label(str(e.get("kind")), ds)
            rows = [rows_by_id[i] for i in e.get("ids") or [] if i in rows_by_id]
            if status == "ambiguous" and ds in ungrounded:
                continue  # matched only by the quantity's name: not what the user named
            if status == "resolved" and rows:  # the printed label of the rows it resolved to
                names = " / ".join(_unique(row_label(h.dataset, h.values) for h in rows))
            elif str(e.get("kind")) in O.LABEL_KINDS and ds in store.main:
                field = O.label_dimension(store, ds) or "label"
                names = " / ".join(
                    _unique(O.printed_label(store, ds, field, n) or n for n in _split(text))
                )
            else:
                names = " / ".join(_unique(_entity_name(n, store) for n in _split(text)))
            add(Interpretation(str(e.get("kind")), label, names, status, (text,)))
        for c in [*(p.get("constraints") or []), *(p.get("qualifiers") or [])]:
            dim = str(c.get("dimension"))
            values = [str(v) for v in c.get("values") or []]
            if values:
                shown = [L.YEAR_ORDINALS.get(v, v) if dim == "year" else v for v in values]
                value, raw = "، ".join(_unique(shown)), tuple(values)
            elif c.get("interval"):
                value, raw = _interval_text(c["interval"]), (str(c["interval"]),)
            else:
                continue
            state: EchoStatus = "resolved" if c.get("state") == "applied" else "unresolved"
            add(Interpretation(dim, L.FIELD_LABELS.get(dim, ""), value, state, raw))
    if not result.plans:  # a route-level unknown entity carries only its typed no-match
        for nm in result.no_match:
            tokens = [str(t) for t in (nm.constraints or {}).get("tokens") or []]
            kind = nm.reason.removeprefix("unknown_")
            if tokens and nm.reason.startswith("unknown_") and kind != "label":
                label = _entity_label(kind, nm.dataset)
                add(Interpretation(kind, label, " ".join(tokens), "unresolved", tuple(tokens)))
    return items


# ---------------------------------------------------------------- refusals


_QUOTED = re.compile(r"«([^»]+)»|'([^']+)'")


def _words(detail: str, constraints: dict[str, Any], query: str) -> list[str]:
    """Question words a refusal names: the engine's tokens or quoted words that occur in the
    normalized question (internal expressions are never shown)."""
    tokens = [str(t) for t in constraints.get("tokens") or []]
    if not tokens:
        tokens = [a or b for a, b in _QUOTED.findall(detail)]
    norm = f" {normalize_for_search(query)} "
    return _unique(t for t in tokens if t.strip() and f" {normalize_for_search(t)} " in norm)[:4]


_DEPENDS = re.compile(r"depends on \[([^\]]*)\]")


def _split(text: str) -> list[str]:
    return [x for part in text.split(", ") for x in part.split("; ")]


def _refusal(nm: Any, plan: dict[str, Any] | None, query: str, table: str) -> Refusal:
    outcome = nm.outcome or ("unknown_entity" if nm.reason.startswith("unknown_") else "no_result")
    message = L.REASON_MESSAGES.get(nm.reason) or L.OUTCOME_MESSAGES.get(
        outcome, L.OUTCOME_MESSAGES["no_result"]
    )
    words = _words(nm.detail, nm.constraints or {}, query)
    if words:
        message = f"{'، '.join(f'«{w}»' for w in words)}: {message}"
    summary = (nm.constraints or {}).get("plan") or {}
    missing_dims = [str(m) for m in (plan or {}).get("missing") or summary.get("missing") or []]
    if not missing_dims and (m := _DEPENDS.search(nm.detail)):
        missing_dims = [a or b for a, b in _QUOTED.findall(m.group(1))]
    missing = tuple(L.FIELD_LABELS.get(m, "") for m in missing_dims if L.FIELD_LABELS.get(m))
    hint = next((L.MISSING_HINTS[m] for m in missing_dims if m in L.MISSING_HINTS), "")
    options: tuple[str, ...] = ()
    if outcome == "ambiguous_entity" and plan:
        options = tuple(
            _unique(
                name
                for e in plan.get("entities") or []
                if e.get("status") == "ambiguous"
                for name in _split(str(e.get("text") or ""))
            )
        )
        hint = hint or (L.AMBIGUOUS_HINT if options else "")
    return Refusal(outcome, nm.reason, message, nm.dataset, missing, hint, options, table)


def refusals(result: RetrievalResult, sources: _Sources) -> list[Refusal]:
    plans = {str(p.get("target")): p for p in result.plans}
    out: list[Refusal] = []
    unknown = {nm.reason for nm in result.no_match if nm.reason.startswith("unknown_")}
    for nm in result.no_match:
        if nm.reason == "unknown_label" and unknown - {"unknown_label"}:
            continue  # the same words already fail as a place, material or occupancy
        table = sources.table(nm.dataset)[0] if nm.dataset in sources.store.schemas else ""
        r = _refusal(nm, plans.get(nm.dataset), result.query, table)
        if all((x.outcome, x.reason, x.message) != (r.outcome, r.reason, r.message) for x in out):
            out.append(r)
    failed = {f.get("table") for f in result.certification_failures}
    for ds in sorted(x for x in failed if x and x not in {r.dataset for r in out}):
        out.append(
            Refusal(
                "certification_failed",
                "certification_failed",
                L.OUTCOME_MESSAGES["certification_failed"],
                ds,
                table=sources.table(ds)[0] if ds in sources.store.schemas else "",
            )
        )
    return out


def out_of_scope(result: RetrievalResult) -> Refusal:
    scope = result.route.signals.get("scope") or {}
    cls, reason = str(scope.get("class") or ""), str(scope.get("reason") or "")
    if cls == "invalid_reference":
        message = L.OUT_OF_SCOPE_REFERENCE
    elif reason.startswith("answerability gate:"):
        message = L.GATE_MESSAGES.get(reason.split(":", 1)[1].strip(), L.OUT_OF_SCOPE)
    elif "covered by" in reason:
        message = f"{L.OUT_OF_SCOPE} {L.COVERED_BY} {reason.split('covered by', 1)[1].strip()}"
    else:
        message = L.OUT_OF_SCOPE
    return Refusal("out_of_scope", cls or "no_evidence", message)


def _ungrounded(r: Refusal) -> Refusal:
    """An ambiguity matched only by the quantity's name: no options (they would name rows the
    user did not ask about)."""
    message = L.AMBIGUOUS_UNGROUNDED.format(table=r.table or "این جدول")
    return Refusal(r.outcome, "ungrounded_ambiguity", message, r.dataset, table=r.table)


# ---------------------------------------------------------------- clarifications

# words that ask for a quantity rather than name an entity
_QUESTION_WORDS = (
    "حداکثر حداقل مجاز مقدار میزان چقدر چند است چیست چه کدام در برای و از به را با که "
    "طبق مبحث جدول ساختمان"
)


def _output_words(store: StructuredStore) -> set[str]:
    """Words that name a requested quantity (or ask for one) rather than an entity."""
    words = set(O.key(_QUESTION_WORDS).split())
    for schema in store.schemas.values():
        for out in schema.outputs:
            for name in (*out.names, *out.header):
                words |= set(O.key(name).split())
    for label in L.OUTPUT_LABELS.values():
        words |= set(O.key(label).split())
    return words


def _missing_dims(nm: Any, plan: dict[str, Any] | None) -> list[str]:
    summary = (nm.constraints or {}).get("plan") or {}
    dims = [str(m) for m in (plan or {}).get("missing") or summary.get("missing") or []]
    if not dims and (m := _DEPENDS.search(nm.detail)):
        dims = [a or b for a, b in _QUOTED.findall(m.group(1))]
    return dims


def _ask_for(field: str, dataset: str, store: StructuredStore, table: str) -> Clarification | None:
    """A clarification for one missing field, or ``None`` when it cannot be asked safely."""
    schema = store.schemas.get(dataset)
    dim = schema.dim(field) if schema is not None else None
    label = L.FIELD_LABELS.get(field, "")
    prompt = L.CLARIFY_PROMPTS.get(field, "")
    if dim is None or not (label and prompt):
        return None
    common: dict[str, Any] = {
        "field": field,
        "field_label": label,
        "prompt": prompt,
        "outcome": "missing_required_dimension",
        "dataset": dataset,
        "table": table,
    }
    if dim.kind == "interval":
        return Clarification(
            **common,
            free_text=True,
            free_text_prefix=L.APPEND_PREFIX.get(field, ""),
            placeholder=L.CLARIFY_PLACEHOLDER.get(field, ""),
        )
    printed_values = O.printed_options(store, dataset, field)
    if not printed_values:
        return None
    if dim.kind == "category":
        options = tuple(ClarifyOption(*O.option_text(field, v)) for v in printed_values)
        hint = L.CLARIFY_CLIMATE_HINT if field == "climate_class" else ""
        return Clarification(**common, options=options, hint=hint)
    if len(printed_values) <= MAX_OPTIONS:
        return Clarification(**common, options=tuple(ClarifyOption(v, v) for v in printed_values))
    return None


def _clarify_missing(
    nm: Any, plan: dict[str, Any] | None, store: StructuredStore, table: str
) -> Clarification | None:
    schema = store.schemas.get(nm.dataset)
    if schema is None:
        return None

    def what_first(field: str) -> bool:  # ask for the row (what) before its conditions
        dim = schema.dim(field)
        return dim is None or dim.kind not in ("label", "entity")

    for field in sorted(_missing_dims(nm, plan), key=what_first):
        if (c := _ask_for(field, nm.dataset, store, table)) is not None:
            return c
    return None


def _ambiguous_candidates(entity: dict[str, Any]) -> list[str]:
    if entity.get("kind") == "occupancy" and entity.get("ids"):
        return [str(i) for i in entity["ids"]]
    return _split(str(entity.get("text") or ""))


def _clarify_ambiguous(
    nm: Any, plan: dict[str, Any] | None, query: str, store: StructuredStore, table: str
) -> Clarification | Literal["ungrounded"] | None:
    """Printed candidates of an ambiguous entity; ``"ungrounded"`` when the candidates share no
    word with the question beyond the requested quantity's name (the user named something
    else, and offering these rows would answer a different question)."""
    ambiguous = [e for e in (plan or {}).get("entities") or [] if e.get("status") == "ambiguous"]
    if len(ambiguous) != 1:
        return None
    entity = ambiguous[0]
    kind = str(entity.get("kind"))
    field = (O.label_dimension(store, nm.dataset) or "label") if kind == "label" else kind
    names = [O.printed_label(store, nm.dataset, field, c) for c in _ambiguous_candidates(entity)]
    if not names or not all(names):
        return None
    shared = set(O.key(query).split()) & {w for name in names for w in O.key(name).split()}
    if not shared:
        return None  # matched by the engine's own normalization: keep the plain refusal
    if shared <= _output_words(store):
        return "ungrounded"
    return Clarification(
        field=field,
        field_label=L.FIELD_LABELS.get(field, "") or L.LABEL_KIND.get(_suffix(nm.dataset), ""),
        prompt=L.CLARIFY_PROMPTS.get(field, L.CLARIFY_GENERIC),
        outcome="ambiguous_entity",
        dataset=nm.dataset,
        table=table,
        options=tuple(ClarifyOption(n, n) for n in _unique(names)),
    )


def _entity_printed(entity: dict[str, Any], dataset: str, store: StructuredStore) -> str:
    ids = [str(i) for i in entity.get("ids") or []]
    if dataset == APP07:
        rows = [r for r in store.app07 if str(r.get("id")) in ids]
        return str(rows[0].get("display_name") or "") if len(rows) == 1 else ""
    kind = str(entity.get("kind"))
    field = (O.label_dimension(store, dataset) or "label") if kind == "label" else kind
    return O.printed_label(store, dataset, field, str(entity.get("text") or ""))


def _clarify_output(
    nm: Any, plan: dict[str, Any] | None, query: str, store: StructuredStore, table: str
) -> Clarification | None:
    """The engine resolved one entity and the printed quantity but could not anchor every word
    of the question: offer that printed quantity for that entity (never another quantity)."""
    if nm.reason != "no_structured_anchor" or not plan or plan.get("missing"):
        return None
    rf = str(plan.get("requested_field") or "")
    entities = plan.get("entities") or []
    if rf not in L.OUTPUT_LABELS or len(entities) != 1 or entities[0].get("status") != "resolved":
        return None
    name = _entity_printed(entities[0], nm.dataset, store)
    if not name:
        return None
    output = L.OUTPUT_LABELS[rf]
    note = ""
    if nm.dataset == APP07 and rf == "thermal_conductivity" and "مقاومت" in O.key(query):
        note = L.CLARIFY_R_LAMBDA.format(table=table, entity=name)
    return Clarification(
        field="output",
        field_label=L.REQUESTED,
        prompt=L.CLARIFY_OUTPUT.format(output=output, entity=name),
        outcome=str(nm.outcome or ""),
        dataset=nm.dataset,
        table=table,
        options=(ClarifyOption(L.CLARIFY_SHOW.format(output=output), O.plain(output)),),
        note=note,
    )


def clarifications(
    result: RetrievalResult, store: StructuredStore, sources: _Sources
) -> tuple[list[Clarification], frozenset[str]]:
    """The clarification for the first refused lookup that can ask for one item, and the
    datasets whose ambiguity was matched only by the quantity's name."""
    plans = {str(p.get("target")): p for p in result.plans}
    ungrounded: set[str] = set()
    for nm in result.no_match:
        table = sources.table(nm.dataset)[0] if nm.dataset in store.schemas else ""
        plan = plans.get(nm.dataset)
        found: Clarification | Literal["ungrounded"] | None = None
        if nm.outcome == "missing_required_dimension":
            found = _clarify_missing(nm, plan, store, table)
        elif nm.outcome == "ambiguous_entity":
            found = _clarify_ambiguous(nm, plan, result.query, store, table)
        elif nm.outcome == "unresolved_constraint":
            found = _clarify_output(nm, plan, result.query, store, table)
        if found == "ungrounded":
            ungrounded.add(nm.dataset)
        elif isinstance(found, Clarification):
            return [found], frozenset(ungrounded)
    return [], frozenset(ungrounded)


# ---------------------------------------------------------------- excerpts


_SENTENCE = re.compile(r"(?<=[.!?؟؛:])\s+|\n+")


def excerpt_text(text: str, terms: Iterable[str]) -> str:
    """A verbatim slice of ``text`` around the sentences with most matched terms."""
    if len(text) <= EXCERPT_CHARS:
        return text.strip()
    parts = [s for s in _SENTENCE.split(text) if s.strip()]
    keys = [normalize_for_search(t) for t in terms if t and t not in ("vector",)]
    keys = [k for k in keys if k]
    scores = [sum(normalize_for_search(s).count(k) for k in keys) for s in parts]
    best = max(range(len(parts)), key=lambda i: (scores[i], -i)) if parts else 0
    lo = hi = best
    size = len(parts[best]) if parts else 0
    while size < EXCERPT_CHARS and (lo > 0 or hi < len(parts) - 1):
        if hi < len(parts) - 1:
            hi += 1
            size += len(parts[hi]) + 1
        elif lo > 0:
            lo -= 1
            size += len(parts[lo]) + 1
    body = " ".join(p.strip() for p in parts[lo : hi + 1])
    if len(body) > EXCERPT_CHARS * 2:
        body = body[: EXCERPT_CHARS * 2].rsplit(" ", 1)[0]
        return ("… " if lo > 0 else "") + body + " …"
    return ("… " if lo > 0 else "") + body + (" …" if hi < len(parts) - 1 else "")


def _excerpt_hits(hits: list[ProseHit], n: int) -> list[ProseHit]:
    """The first ``n`` hits in the engine's order, passing over bare captions and footnotes
    (shorter than :data:`MIN_EXCERPT` characters) while longer hits remain in the top five."""
    top = hits[:5]
    long = [h for h in top if len(h.text.strip()) >= MIN_EXCERPT]
    return (long or top)[:n]


def _excerpt(hit: ProseHit) -> Excerpt:
    ref = hit.section_path[-1] if hit.section_path else None
    number = ref.number if ref else None
    pages = tuple(_unique(p for p in hit.printed_pages if p))
    source = Source(
        kind="section",
        label=f"بند {fa_digits(number)}" if number else "متن مبحث ۱۹",
        title=ref.title if ref else "",
        document=hit.chunk_id.split(":")[0] if ":" in hit.chunk_id else "m19-ed5-1404-main",
        printed_pages=pages,
        pdf_pages=tuple(dict.fromkeys(hit.pdf_pages)),
        ids=(hit.chunk_id,),
    )
    return Excerpt(
        hit.chunk_id, excerpt_text(hit.text, hit.matched_terms), source, tuple(hit.source_issues)
    )


# ---------------------------------------------------------------- notices


def _notices(
    result: RetrievalResult, issues: Iterable[str], store: StructuredStore
) -> list[Notice]:
    out: list[Notice] = []
    for c in result.conflicts:
        text = f"{L.CONFLICT} ({' / '.join(c.values)})"
        out.append(
            Notice("conflict", text, tuple([*([c.issue_id] if c.issue_id else []), *c.hits]))
        )
    for sid in _unique(issues):
        title = str((store.issues.get(sid) or {}).get("title") or "")
        out.append(Notice("source_issue", f"{L.SOURCE_ISSUE} {sid}: {title}".strip(), (sid,)))
    for note in result.notes:
        persian = next((fa for en, fa in L.NOTE_PREFIXES if note.startswith(en)), None)
        if persian is None and "prints its requirement as" in note:
            persian = L.NOTE_ALTERNATIVE
        if persian is not None:
            if all(n.text != persian for n in out):
                out.append(Notice("note", persian))
        else:
            out.append(Notice("technical", note))
    return out


# ---------------------------------------------------------------- compose


def _headline(
    rtype: ResponseType,
    values: list[ValueItem],
    refused: list[Refusal],
    conflicts: bool,
    reverse: bool = False,
) -> str:
    if rtype == "structured_value":
        if conflicts:
            return L.HEADLINE_DIFFERENT
        distinct = {(v.output_label, v.value, v.unit) for v in values}
        if len(distinct) == 1 and values[0].value and values[0].output_label:
            v = values[0]
            unit = v.unit_label or v.unit or ""
            return f"{v.output_label}: {v.value} {unit}".strip()
        if reverse and not any(v.value for v in values):
            return L.HEADLINE_REVERSE
        return L.HEADLINE_ROWS
    if rtype == "prose":
        return L.HEADLINE_PROSE
    return refused[0].message if refused else L.OUTCOME_MESSAGES["no_result"]


def _compose(
    result: RetrievalResult, store: StructuredStore, request_id: str, clarify_ok: bool = True
) -> Answer:
    sources = _Sources(store)
    route = result.route.route
    requested = {str(p.get("target")): str(p.get("requested_field") or "") for p in result.plans}
    all_items = [_value_item(h, store, sources, requested) for h in result.structured]
    values = all_items[:MAX_VALUES]
    structured_route = route in ("structured", "mixed")
    refused = refusals(result, sources)
    n_excerpts = 2 if values else MAX_EXCERPTS
    shown_hits = _excerpt_hits(result.prose, n_excerpts)
    excerpts = [_excerpt(h) for h in shown_hits]
    rtype: ResponseType
    if values:
        rtype = "structured_value"
    elif route == "unsupported" and not result.no_match:
        rtype, refused = "out_of_scope", [out_of_scope(result)]
    elif refused or (structured_route and not excerpts):
        rtype = "refusal"
        refused = refused or [Refusal("no_result", "no_result", L.OUTCOME_MESSAGES["no_result"])]
    elif excerpts:
        rtype = "prose"
    else:
        rtype, refused = "out_of_scope", [out_of_scope(result)]
    clarify: list[Clarification] = []
    ungrounded: frozenset[str] = frozenset()
    if rtype == "refusal":
        clarify, ungrounded = clarifications(result, store, sources)
        clarify = clarify if clarify_ok else []
        if clarify:
            rtype = "clarification"
        refused = [_ungrounded(r) if r.dataset in ungrounded else r for r in refused]
    answered = {v.dataset for v in all_items}
    echo = []
    if structured_route or result.no_match:
        echo = interpretation(result, answered or None, store, ungrounded)
    issues = [i for v in values for i in v.source_issues]
    issues += [i for e in excerpts for i in e.source_issues]
    cited = [*(v.source for v in values if v.source), *(e.source for e in excerpts)]
    unique_sources: list[Source] = []
    for s in cited:
        if all(
            (s.label, s.printed_pages, s.ids) != (u.label, u.printed_pages, u.ids)
            for u in unique_sources
        ):
            unique_sources.append(s)
    return Answer(
        request_id=request_id,
        question=result.query,
        response_type=rtype,
        headline=clarify[0].prompt
        if clarify
        else _headline(
            rtype,
            values,
            refused,
            bool(result.conflicts),
            any(p.get("operation") == "reverse" for p in result.plans),
        ),
        interpretation=echo,
        values=values,
        excerpts=excerpts,
        refusals=refused,
        notices=_notices(result, issues, store),
        clarifications=clarify,
        sources=unique_sources,
        hidden_values=len(all_items) - len(values),
        safety={
            "route": route,
            "status": {
                "structured_value": "certified_rows",
                "prose": "retrieved_excerpts",
                "refusal": "refused",
                "out_of_scope": "out_of_scope",
                "clarification": "needs_clarification",
            }[rtype],
            "certified_row_ids": [h.record_id for h in result.structured],
            "prose_chunk_ids": [h.chunk_id for h in shown_hits],
            "composer": COMPOSER_VERSION,
        },
    )


def compose(
    result: RetrievalResult, store: StructuredStore, request_id: str, *, clarify: bool = True
) -> Answer:
    """The answer for one retrieval result; any provenance violation yields a refusal.
    ``clarify=False`` keeps a refusal a refusal (no follow-up question)."""
    answer = _compose(result, store, request_id, clarify)
    violations = check(answer, result, store)
    answer.safety["provenance_violations"] = violations
    if not violations:
        return answer
    message = L.OUTCOME_MESSAGES["display_check_failed"]
    return Answer(
        request_id=request_id,
        question=result.query,
        response_type="refusal",
        headline=message,
        refusals=[Refusal("display_check_failed", "provenance", message)],
        safety={**answer.safety, "status": "refused", "certified_row_ids": []},
    )
